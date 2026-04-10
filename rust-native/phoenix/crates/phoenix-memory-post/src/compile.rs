use std::cmp::Ordering;
use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    EntityMemoryCard, EntityMemoryIdentityCard, EntityMemoryStateView, MemoryClaimAtom,
    MemoryClaimStatus, MemoryCompilerSummary, MemoryConflictKind, MemoryConflictRecord,
    MemoryContinuityGapRecord, MemoryDeltaRecord, MemoryEventRecord, MemoryGapKind,
    MemoryStateRecord, RelationshipMemoryLedger, RelationshipMemoryRef,
};
use phoenix_types::{BiTemporalWindow, EntityId};
use serde::{Deserialize, Serialize};

use crate::normalize::{MemoryEntityProfile, MemoryNormalizedBatch, MemoryPendingReview};
use crate::registry::{
    active_scalar_slot_keys, slot_definition_for_relation_family, source_class_priority,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledMemory {
    #[serde(default)]
    pub claims: Vec<MemoryClaimAtom>,
    #[serde(default)]
    pub events: Vec<MemoryEventRecord>,
    #[serde(default)]
    pub states: Vec<MemoryStateRecord>,
    #[serde(default)]
    pub deltas: Vec<MemoryDeltaRecord>,
    #[serde(default)]
    pub conflicts: Vec<MemoryConflictRecord>,
    #[serde(default)]
    pub gaps: Vec<MemoryContinuityGapRecord>,
    #[serde(default)]
    pub entity_cards: Vec<EntityMemoryCard>,
    #[serde(default)]
    pub relationship_ledgers: Vec<RelationshipMemoryLedger>,
    pub summary: MemoryCompilerSummary,
}

pub fn compile_memory(batch: &MemoryNormalizedBatch) -> CompiledMemory {
    let mut claims = batch.claims.clone();
    let mut states = Vec::new();
    let mut deltas = Vec::new();
    let mut conflicts = Vec::new();
    let mut events = Vec::new();
    let mut gaps = Vec::new();

    let claims_by_slot = group_claims_by_slot(&claims);
    let scalar_slots = active_scalar_slot_keys(&batch.slot_definitions);

    for ((entity_key, slot_key), claim_indices) in claims_by_slot {
        if !scalar_slots.iter().any(|value| value == &slot_key) {
            continue;
        }
        let entity_id = EntityId(entity_key.clone());
        let mut positive = claim_indices
            .iter()
            .copied()
            .filter(|index| {
                matches!(
                    claims[*index].status,
                    MemoryClaimStatus::Active
                        | MemoryClaimStatus::Supported
                        | MemoryClaimStatus::Candidate
                )
            })
            .collect::<Vec<_>>();
        let contradictory = claim_indices
            .iter()
            .copied()
            .filter(|index| claims[*index].status == MemoryClaimStatus::Contradicted)
            .collect::<Vec<_>>();

        positive.sort_by(|left, right| compare_claims(&claims[*left], &claims[*right]));

        if positive.is_empty() {
            if !contradictory.is_empty() {
                let conflict_id = format!("conflict:{}:{}", entity_key, slot_key);
                let conflict = MemoryConflictRecord {
                    conflict_id: conflict_id.clone(),
                    entity_id: entity_id.clone(),
                    slot_key: slot_key.clone(),
                    kind: MemoryConflictKind::SupportVsContradiction,
                    preferred_claim_id: None,
                    status: MemoryClaimStatus::Contradicted,
                    temporal: merge_temporal(
                        contradictory
                            .iter()
                            .map(|index| &claims[*index].temporal)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                    claim_ids: contradictory
                        .iter()
                        .map(|index| claims[*index].claim_id.clone())
                        .collect(),
                };
                conflicts.push(conflict.clone());
                events.push(conflict_event("conflict_opened", &conflict));
                gaps.push(MemoryContinuityGapRecord {
                    gap_id: format!("gap:conflict:{}:{}", entity_key, slot_key),
                    entity_id,
                    slot_key,
                    kind: MemoryGapKind::UnresolvedConflict,
                    status: MemoryClaimStatus::Deferred,
                    detail: "contradictory evidence without an active supported value".to_owned(),
                    temporal: conflict.temporal.clone(),
                    claim_ids: conflict.claim_ids.clone(),
                });
            }
            continue;
        }

        let mut winning_index = positive[0];
        let mut history = vec![winning_index];
        let mut unresolved_conflict = None::<MemoryConflictRecord>;
        let mut competitive_conflicts = Vec::<MemoryConflictRecord>::new();

        for candidate_index in positive.iter().copied().skip(1) {
            let winner = &claims[winning_index];
            let candidate = &claims[candidate_index];
            if same_value(winner, candidate) {
                if claims[winning_index].status != MemoryClaimStatus::Supported
                    && candidate.status == MemoryClaimStatus::Supported
                {
                    claims[winning_index].status = MemoryClaimStatus::Supported;
                }
                continue;
            }
            match compare_claims(winner, candidate) {
                Ordering::Less => {
                    let conflict = competitive_current_conflict(
                        winner,
                        candidate,
                        &entity_id,
                        &slot_key,
                        Some(candidate.claim_id.as_str()),
                    );
                    claims[winning_index].status = MemoryClaimStatus::Superseded;
                    if let Some(conflict) = conflict {
                        competitive_conflicts.push(conflict);
                    }
                    winning_index = candidate_index;
                    history.push(candidate_index);
                }
                Ordering::Greater => {
                    let conflict = competitive_current_conflict(
                        winner,
                        candidate,
                        &entity_id,
                        &slot_key,
                        Some(winner.claim_id.as_str()),
                    );
                    claims[candidate_index].status = MemoryClaimStatus::Superseded;
                    if let Some(conflict) = conflict {
                        competitive_conflicts.push(conflict);
                    }
                }
                Ordering::Equal => {
                    let conflict = MemoryConflictRecord {
                        conflict_id: format!(
                            "conflict:{}:{}:{}:{}",
                            entity_key, slot_key, winner.claim_id, candidate.claim_id
                        ),
                        entity_id: entity_id.clone(),
                        slot_key: slot_key.clone(),
                        kind: MemoryConflictKind::MutuallyExclusive,
                        preferred_claim_id: Some(winner.claim_id.clone()),
                        status: MemoryClaimStatus::Deferred,
                        temporal: merge_temporal(&[&winner.temporal, &candidate.temporal]),
                        claim_ids: vec![winner.claim_id.clone(), candidate.claim_id.clone()],
                    };
                    unresolved_conflict = Some(conflict);
                }
            }
        }

        let winner = claims[winning_index].clone();
        let state = MemoryStateRecord {
            state_id: format!("state:{}:{}", entity_key, slot_key),
            entity_id: entity_id.clone(),
            slot_key: slot_key.clone(),
            value: winner.object_value.clone(),
            value_entity_id: winner.object_entity_id.clone(),
            status: if winner.status == MemoryClaimStatus::Supported {
                MemoryClaimStatus::Supported
            } else {
                MemoryClaimStatus::Active
            },
            source_class: winner.source_class.clone(),
            confidence_millis: winner.confidence_millis,
            temporal: winner.temporal.clone(),
            claim_ids: vec![winner.claim_id.clone()],
        };
        states.push(state.clone());
        events.push(state_event("state_started", &state));

        let unique_history = history
            .into_iter()
            .map(|index| claims[index].clone())
            .collect::<Vec<_>>();
        for pair in unique_history.windows(2) {
            let old = &pair[0];
            let new = &pair[1];
            if same_value(old, new) {
                continue;
            }
            let delta = MemoryDeltaRecord {
                delta_id: format!("delta:{}:{}:{}", entity_key, slot_key, new.claim_id),
                entity_id: entity_id.clone(),
                slot_key: slot_key.clone(),
                old_value: Some(old.object_value.clone()),
                old_value_entity_id: old.object_entity_id.clone(),
                new_value: Some(new.object_value.clone()),
                new_value_entity_id: new.object_entity_id.clone(),
                caused_by_event_id: Some(format!(
                    "event:state_changed:{}:{}",
                    entity_key, new.claim_id
                )),
                canonical_caused_by_event_id: None,
                temporal: new.temporal.clone(),
                claim_ids: vec![old.claim_id.clone(), new.claim_id.clone()],
            };
            events.push(delta_event(
                "state_ended",
                entity_id.clone(),
                &slot_key,
                old,
            ));
            events.push(delta_event(
                "state_changed",
                entity_id.clone(),
                &slot_key,
                new,
            ));
            deltas.push(delta);
        }

        if !competitive_conflicts.is_empty() {
            let mut merged_claim_ids = competitive_conflicts
                .iter()
                .flat_map(|conflict| conflict.claim_ids.iter().cloned())
                .collect::<Vec<_>>();
            merged_claim_ids.sort();
            merged_claim_ids.dedup();
            let merged_temporal = merge_temporal(
                &competitive_conflicts
                    .iter()
                    .map(|conflict| &conflict.temporal)
                    .collect::<Vec<_>>(),
            );
            for conflict in &competitive_conflicts {
                conflicts.push(conflict.clone());
                events.push(conflict_event("conflict_opened", conflict));
            }
            gaps.push(MemoryContinuityGapRecord {
                gap_id: format!("gap:competitive:{}:{}", entity_key, slot_key),
                entity_id: entity_id.clone(),
                slot_key: slot_key.clone(),
                kind: MemoryGapKind::UnresolvedConflict,
                status: MemoryClaimStatus::Deferred,
                detail: "multiple overlapping current values remain in contention".to_owned(),
                temporal: merged_temporal,
                claim_ids: merged_claim_ids,
            });
        }

        if let Some(mut conflict) = unresolved_conflict {
            conflict.preferred_claim_id = Some(winner.claim_id.clone());
            conflicts.push(conflict.clone());
            events.push(conflict_event("conflict_opened", &conflict));
            gaps.push(MemoryContinuityGapRecord {
                gap_id: format!("gap:conflict:{}:{}", entity_key, slot_key),
                entity_id: entity_id.clone(),
                slot_key: slot_key.clone(),
                kind: MemoryGapKind::UnresolvedConflict,
                status: MemoryClaimStatus::Deferred,
                detail: "multiple competing current values remain unresolved".to_owned(),
                temporal: conflict.temporal.clone(),
                claim_ids: conflict.claim_ids.clone(),
            });
        }

        if contradictory
            .iter()
            .any(|index| claims[*index].status == MemoryClaimStatus::Contradicted)
        {
            let contradiction_claim_ids = contradictory
                .iter()
                .map(|index| claims[*index].claim_id.clone())
                .collect::<Vec<_>>();
            let conflict = MemoryConflictRecord {
                conflict_id: format!("conflict:contradiction:{}:{}", entity_key, slot_key),
                entity_id: entity_id.clone(),
                slot_key: slot_key.clone(),
                kind: MemoryConflictKind::SupportVsContradiction,
                preferred_claim_id: Some(winner.claim_id.clone()),
                status: MemoryClaimStatus::Deferred,
                temporal: merge_temporal(
                    contradictory
                        .iter()
                        .map(|index| &claims[*index].temporal)
                        .collect::<Vec<_>>()
                        .as_slice(),
                ),
                claim_ids: contradiction_claim_ids.clone(),
            };
            conflicts.push(conflict.clone());
            events.push(conflict_event("conflict_opened", &conflict));
            gaps.push(MemoryContinuityGapRecord {
                gap_id: format!("gap:contradiction:{}:{}", entity_key, slot_key),
                entity_id,
                slot_key,
                kind: MemoryGapKind::UnresolvedConflict,
                status: MemoryClaimStatus::Deferred,
                detail: "contradiction judgment exists against the current state".to_owned(),
                temporal: conflict.temporal.clone(),
                claim_ids: contradiction_claim_ids,
            });
        }
    }

    add_pending_review_gaps(&mut gaps, &batch.pending_reviews);
    add_missing_current_value_gaps(
        &mut gaps,
        &batch.entity_profiles,
        &states,
        &batch.slot_definitions,
    );

    let relationship_ledgers = build_relationship_ledgers(&claims, &batch.slot_definitions);
    let (relationship_conflicts, relationship_gaps, relationship_events) =
        build_relationship_conflicts(&relationship_ledgers);
    conflicts.extend(relationship_conflicts);
    gaps.extend(relationship_gaps);
    events.extend(relationship_events);
    let entity_cards = build_entity_cards(
        &batch.entity_profiles,
        &states,
        &deltas,
        &conflicts,
        &gaps,
        &relationship_ledgers,
    );

    claims.sort_by(|left, right| left.claim_id.cmp(&right.claim_id));
    states.sort_by(|left, right| left.state_id.cmp(&right.state_id));
    deltas.sort_by(|left, right| left.delta_id.cmp(&right.delta_id));
    conflicts.sort_by(|left, right| left.conflict_id.cmp(&right.conflict_id));
    gaps.sort_by(|left, right| left.gap_id.cmp(&right.gap_id));
    events.sort_by(|left, right| left.event_id.cmp(&right.event_id));

    let summary = build_summary(
        &claims,
        &events,
        &states,
        &deltas,
        &conflicts,
        &gaps,
        &entity_cards,
        &relationship_ledgers,
        &batch.source_class_counts,
    );

    CompiledMemory {
        claims,
        events,
        states,
        deltas,
        conflicts,
        gaps,
        entity_cards,
        relationship_ledgers,
        summary,
    }
}

fn group_claims_by_slot(claims: &[MemoryClaimAtom]) -> BTreeMap<(String, String), Vec<usize>> {
    let mut grouped = BTreeMap::<(String, String), Vec<usize>>::new();
    for (index, claim) in claims.iter().enumerate() {
        let Some(entity_id) = claim.source_entity_id.as_ref() else {
            continue;
        };
        grouped
            .entry((entity_id.0.clone(), claim.slot_key.clone()))
            .or_default()
            .push(index);
    }
    grouped
}

fn compare_claims(left: &MemoryClaimAtom, right: &MemoryClaimAtom) -> Ordering {
    compare_temporal(&left.temporal, &right.temporal)
        .then_with(|| {
            source_class_priority(&left.source_class)
                .cmp(&source_class_priority(&right.source_class))
        })
        .then_with(|| left.confidence_millis.cmp(&right.confidence_millis))
}

fn compare_temporal(left: &BiTemporalWindow, right: &BiTemporalWindow) -> Ordering {
    left.valid_from
        .cmp(&right.valid_from)
        .then_with(|| left.recorded_from.cmp(&right.recorded_from))
}

fn same_value(left: &MemoryClaimAtom, right: &MemoryClaimAtom) -> bool {
    left.object_value == right.object_value && left.object_entity_id == right.object_entity_id
}

fn competitive_current_conflict(
    winner: &MemoryClaimAtom,
    candidate: &MemoryClaimAtom,
    entity_id: &EntityId,
    slot_key: &str,
    preferred_claim_id: Option<&str>,
) -> Option<MemoryConflictRecord> {
    if !current_conflict_compatible(winner, candidate) {
        return None;
    }
    let mut claim_ids = vec![winner.claim_id.clone(), candidate.claim_id.clone()];
    claim_ids.sort();
    claim_ids.dedup();
    Some(MemoryConflictRecord {
        conflict_id: format!(
            "conflict:competitive:{}:{}:{}:{}",
            entity_id.0, slot_key, claim_ids[0], claim_ids[1]
        ),
        entity_id: entity_id.clone(),
        slot_key: slot_key.to_owned(),
        kind: MemoryConflictKind::TemporalOverlap,
        preferred_claim_id: preferred_claim_id.map(str::to_owned),
        status: MemoryClaimStatus::Deferred,
        temporal: merge_temporal(&[&winner.temporal, &candidate.temporal]),
        claim_ids,
    })
}

fn current_conflict_compatible(left: &MemoryClaimAtom, right: &MemoryClaimAtom) -> bool {
    !same_value(left, right)
        && current_window(&left.temporal)
        && current_window(&right.temporal)
        && temporal_overlap(&left.temporal, &right.temporal)
}

fn current_window(window: &BiTemporalWindow) -> bool {
    window.valid_to.is_none()
}

fn temporal_overlap(left: &BiTemporalWindow, right: &BiTemporalWindow) -> bool {
    let left_start = left.valid_from.unwrap_or(i64::MIN);
    let left_end = left.valid_to.unwrap_or(i64::MAX);
    let right_start = right.valid_from.unwrap_or(i64::MIN);
    let right_end = right.valid_to.unwrap_or(i64::MAX);
    left_start <= right_end && right_start <= left_end
}

fn merge_temporal(temporals: &[&BiTemporalWindow]) -> BiTemporalWindow {
    let mut merged = BiTemporalWindow::default();
    for temporal in temporals {
        merged.valid_from = min_opt(merged.valid_from, temporal.valid_from);
        merged.valid_to = max_opt(merged.valid_to, temporal.valid_to);
        merged.recorded_from = min_opt(merged.recorded_from, temporal.recorded_from);
        merged.recorded_to = max_opt(merged.recorded_to, temporal.recorded_to);
    }
    merged
}

fn min_opt(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn max_opt(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn state_event(kind: &str, state: &MemoryStateRecord) -> MemoryEventRecord {
    MemoryEventRecord {
        event_id: format!("event:{}:{}", kind, state.state_id),
        canonical_event_id: None,
        document_id: String::new(),
        kind: kind.to_owned(),
        slot_key: state.slot_key.clone(),
        subject_entity_id: Some(state.entity_id.clone()),
        object_entity_id: state.value_entity_id.clone(),
        old_value: None,
        new_value: Some(state.value.clone()),
        conflict_id: None,
        temporal: state.temporal.clone(),
        claim_ids: state.claim_ids.clone(),
        evidence_refs: Vec::new(),
    }
}

fn delta_event(
    kind: &str,
    entity_id: EntityId,
    slot_key: &str,
    claim: &MemoryClaimAtom,
) -> MemoryEventRecord {
    MemoryEventRecord {
        event_id: format!("event:{}:{}:{}", kind, entity_id.0, claim.claim_id),
        canonical_event_id: None,
        document_id: claim.document_id.clone(),
        kind: kind.to_owned(),
        slot_key: slot_key.to_owned(),
        subject_entity_id: Some(entity_id),
        object_entity_id: claim.object_entity_id.clone(),
        old_value: None,
        new_value: Some(claim.object_value.clone()),
        conflict_id: None,
        temporal: claim.temporal.clone(),
        claim_ids: vec![claim.claim_id.clone()],
        evidence_refs: claim.evidence_refs.clone(),
    }
}

fn conflict_event(kind: &str, conflict: &MemoryConflictRecord) -> MemoryEventRecord {
    MemoryEventRecord {
        event_id: format!("event:{}:{}", kind, conflict.conflict_id),
        canonical_event_id: None,
        document_id: String::new(),
        kind: kind.to_owned(),
        slot_key: conflict.slot_key.clone(),
        subject_entity_id: Some(conflict.entity_id.clone()),
        object_entity_id: None,
        old_value: None,
        new_value: None,
        conflict_id: Some(conflict.conflict_id.clone()),
        temporal: conflict.temporal.clone(),
        claim_ids: conflict.claim_ids.clone(),
        evidence_refs: Vec::new(),
    }
}

fn add_pending_review_gaps(
    gaps: &mut Vec<MemoryContinuityGapRecord>,
    pending: &[MemoryPendingReview],
) {
    for review in pending {
        let Some(slot_key) = review.slot_key.clone() else {
            continue;
        };
        gaps.push(MemoryContinuityGapRecord {
            gap_id: format!("gap:pending:{}:{}", review.entity_id.0, review.review_id),
            entity_id: review.entity_id.clone(),
            slot_key,
            kind: MemoryGapKind::BrokenContinuity,
            status: MemoryClaimStatus::Deferred,
            detail: review.detail.clone(),
            temporal: review.temporal.clone(),
            claim_ids: vec![review.review_id.clone()],
        });
    }
}

fn add_missing_current_value_gaps(
    gaps: &mut Vec<MemoryContinuityGapRecord>,
    entity_profiles: &[MemoryEntityProfile],
    states: &[MemoryStateRecord],
    slot_definitions: &[phoenix_semantic_v2::StateSlotDefinitionRecord],
) {
    let state_keys = states
        .iter()
        .map(|state| (state.entity_id.0.clone(), state.slot_key.clone()))
        .collect::<std::collections::BTreeSet<_>>();
    for profile in entity_profiles {
        for slot_key in active_scalar_slot_keys(slot_definitions) {
            if state_keys.contains(&(profile.entity_id.0.clone(), slot_key.clone())) {
                continue;
            }
            gaps.push(MemoryContinuityGapRecord {
                gap_id: format!("gap:missing:{}:{}", profile.entity_id.0, slot_key),
                entity_id: profile.entity_id.clone(),
                slot_key: slot_key.clone(),
                kind: MemoryGapKind::MissingCurrentValue,
                status: MemoryClaimStatus::Deferred,
                detail: "no current compiled value for tracked slot".to_owned(),
                temporal: BiTemporalWindow::default(),
                claim_ids: Vec::new(),
            });
        }
    }
}

fn build_relationship_ledgers(
    claims: &[MemoryClaimAtom],
    slot_definitions: &[phoenix_semantic_v2::StateSlotDefinitionRecord],
) -> Vec<RelationshipMemoryLedger> {
    let mut grouped = BTreeMap::<(String, String, String), Vec<&MemoryClaimAtom>>::new();
    for claim in claims {
        let Some(relation_family) = claim.relation_family.as_deref() else {
            continue;
        };
        let Some(slot) = slot_definition_for_relation_family(relation_family, slot_definitions)
        else {
            continue;
        };
        if !slot.relationship_only {
            continue;
        }
        let (Some(source_entity_id), Some(target_entity_id)) = (
            claim.source_entity_id.as_ref(),
            claim.target_entity_id.as_ref(),
        ) else {
            continue;
        };
        grouped
            .entry((
                relation_family.to_owned(),
                source_entity_id.0.clone(),
                target_entity_id.0.clone(),
            ))
            .or_default()
            .push(claim);
    }

    let mut ledgers = Vec::new();
    for ((relation_family, source_entity_id, target_entity_id), rows) in grouped {
        let supporting = rows
            .iter()
            .filter(|claim| {
                matches!(
                    claim.status,
                    MemoryClaimStatus::Active
                        | MemoryClaimStatus::Supported
                        | MemoryClaimStatus::Candidate
                )
            })
            .map(|claim| claim.claim_id.clone())
            .collect::<Vec<_>>();
        let contradicting = rows
            .iter()
            .filter(|claim| claim.status == MemoryClaimStatus::Contradicted)
            .map(|claim| claim.claim_id.clone())
            .collect::<Vec<_>>();
        let current_status = if !supporting.is_empty() && contradicting.is_empty() {
            MemoryClaimStatus::Active
        } else if !supporting.is_empty() {
            MemoryClaimStatus::Deferred
        } else if !contradicting.is_empty() {
            MemoryClaimStatus::Contradicted
        } else {
            MemoryClaimStatus::Candidate
        };
        ledgers.push(RelationshipMemoryLedger {
            ledger_id: format!(
                "relationship:{}:{}:{}",
                relation_family, source_entity_id, target_entity_id
            ),
            relation_family,
            source_entity_id: EntityId(source_entity_id),
            target_entity_id: EntityId(target_entity_id),
            current_status,
            temporal: merge_temporal(&rows.iter().map(|row| &row.temporal).collect::<Vec<_>>()),
            supporting_claim_ids: supporting,
            contradicting_claim_ids: contradicting,
        });
    }
    ledgers.sort_by(|left, right| left.ledger_id.cmp(&right.ledger_id));
    ledgers
}

fn build_relationship_conflicts(
    ledgers: &[RelationshipMemoryLedger],
) -> (
    Vec<MemoryConflictRecord>,
    Vec<MemoryContinuityGapRecord>,
    Vec<MemoryEventRecord>,
) {
    let mut conflicts = Vec::new();
    let mut gaps = Vec::new();
    let mut events = Vec::new();
    for ledger in ledgers {
        if ledger.supporting_claim_ids.is_empty() || ledger.contradicting_claim_ids.is_empty() {
            continue;
        }
        let mut claim_ids = ledger
            .supporting_claim_ids
            .iter()
            .chain(ledger.contradicting_claim_ids.iter())
            .cloned()
            .collect::<Vec<_>>();
        claim_ids.sort();
        claim_ids.dedup();
        let slot_key = format!("relation.{}", ledger.relation_family);
        let conflict = MemoryConflictRecord {
            conflict_id: format!("conflict:relationship:{}", ledger.ledger_id),
            entity_id: ledger.source_entity_id.clone(),
            slot_key: slot_key.clone(),
            kind: MemoryConflictKind::SupportVsContradiction,
            preferred_claim_id: ledger.supporting_claim_ids.first().cloned(),
            status: MemoryClaimStatus::Deferred,
            temporal: ledger.temporal.clone(),
            claim_ids: claim_ids.clone(),
        };
        let gap = MemoryContinuityGapRecord {
            gap_id: format!("gap:relationship:{}", ledger.ledger_id),
            entity_id: ledger.source_entity_id.clone(),
            slot_key,
            kind: MemoryGapKind::UnresolvedConflict,
            status: MemoryClaimStatus::Deferred,
            detail: "relationship ledger contains supporting and contradicting evidence".to_owned(),
            temporal: ledger.temporal.clone(),
            claim_ids,
        };
        events.push(conflict_event("conflict_opened", &conflict));
        conflicts.push(conflict);
        gaps.push(gap);
    }
    (conflicts, gaps, events)
}

fn build_entity_cards(
    profiles: &[MemoryEntityProfile],
    states: &[MemoryStateRecord],
    deltas: &[MemoryDeltaRecord],
    conflicts: &[MemoryConflictRecord],
    gaps: &[MemoryContinuityGapRecord],
    ledgers: &[RelationshipMemoryLedger],
) -> Vec<EntityMemoryCard> {
    let mut cards = Vec::with_capacity(profiles.len());
    for profile in profiles {
        let mut current_state = states
            .iter()
            .filter(|state| state.entity_id == profile.entity_id)
            .map(|state| EntityMemoryStateView {
                slot_key: state.slot_key.clone(),
                value: state.value.clone(),
                value_entity_id: state.value_entity_id.clone(),
                confidence_millis: state.confidence_millis,
                temporal: state.temporal.clone(),
                claim_ids: state.claim_ids.clone(),
            })
            .collect::<Vec<_>>();
        current_state.sort_by(|left, right| left.slot_key.cmp(&right.slot_key));

        let mut recent_deltas = deltas
            .iter()
            .filter(|delta| delta.entity_id == profile.entity_id)
            .cloned()
            .collect::<Vec<_>>();
        recent_deltas
            .sort_by(|left, right| right.temporal.valid_from.cmp(&left.temporal.valid_from));
        recent_deltas.truncate(8);

        let mut active_relationships = ledgers
            .iter()
            .filter(|ledger| ledger.source_entity_id == profile.entity_id)
            .map(|ledger| RelationshipMemoryRef {
                relation_family: ledger.relation_family.clone(),
                target_entity_id: ledger.target_entity_id.clone(),
                status: ledger.current_status,
                temporal: ledger.temporal.clone(),
                supporting_claim_ids: ledger.supporting_claim_ids.clone(),
                contradicting_claim_ids: ledger.contradicting_claim_ids.clone(),
            })
            .collect::<Vec<_>>();
        active_relationships.sort_by(|left, right| {
            left.relation_family
                .cmp(&right.relation_family)
                .then_with(|| left.target_entity_id.0.cmp(&right.target_entity_id.0))
        });

        let mut active_conflicts = conflicts
            .iter()
            .filter(|conflict| conflict.entity_id == profile.entity_id)
            .cloned()
            .collect::<Vec<_>>();
        active_conflicts.sort_by(|left, right| left.slot_key.cmp(&right.slot_key));

        let mut open_gaps = gaps
            .iter()
            .filter(|gap| gap.entity_id == profile.entity_id)
            .cloned()
            .collect::<Vec<_>>();
        open_gaps.sort_by(|left, right| left.slot_key.cmp(&right.slot_key));

        let mut top_evidence_claim_ids = BTreeMap::<String, ()>::new();
        for state in &current_state {
            for claim_id in &state.claim_ids {
                top_evidence_claim_ids.insert(claim_id.clone(), ());
            }
        }
        for delta in &recent_deltas {
            for claim_id in &delta.claim_ids {
                top_evidence_claim_ids.insert(claim_id.clone(), ());
            }
        }

        cards.push(EntityMemoryCard {
            entity_id: profile.entity_id.clone(),
            identity: EntityMemoryIdentityCard {
                entity_id: profile.entity_id.clone(),
                canonical_name: profile.canonical_name.clone(),
                aliases: profile.aliases.clone(),
                effective_kind: profile.effective_kind.clone(),
                linked_mention_count: profile.linked_mention_count,
                continuity_refs: profile.continuity_refs.clone(),
            },
            current_state,
            recent_deltas,
            active_relationships,
            active_conflicts,
            open_gaps,
            top_evidence_claim_ids: top_evidence_claim_ids.into_keys().take(8).collect(),
        });
    }
    cards.sort_by(|left, right| left.entity_id.0.cmp(&right.entity_id.0));
    cards
}

fn build_summary(
    claims: &[MemoryClaimAtom],
    events: &[MemoryEventRecord],
    states: &[MemoryStateRecord],
    deltas: &[MemoryDeltaRecord],
    conflicts: &[MemoryConflictRecord],
    gaps: &[MemoryContinuityGapRecord],
    entity_cards: &[EntityMemoryCard],
    relationship_ledgers: &[RelationshipMemoryLedger],
    source_class_counts: &BTreeMap<String, usize>,
) -> MemoryCompilerSummary {
    let mut active_slot_counts = BTreeMap::<String, usize>::new();
    for state in states {
        *active_slot_counts
            .entry(state.slot_key.clone())
            .or_default() += 1;
    }
    let mut unresolved_gap_counts = BTreeMap::<String, usize>::new();
    for gap in gaps {
        *unresolved_gap_counts
            .entry(format!("{:?}", gap.kind).to_lowercase())
            .or_default() += 1;
    }
    let mut status_counts = BTreeMap::<String, usize>::new();
    for claim in claims {
        *status_counts
            .entry(format!("{:?}", claim.status).to_lowercase())
            .or_default() += 1;
    }
    MemoryCompilerSummary {
        claim_count: claims.len(),
        event_count: events.len(),
        state_count: states.len(),
        delta_count: deltas.len(),
        conflict_count: conflicts.len(),
        gap_count: gaps.len(),
        entity_card_count: entity_cards.len(),
        relationship_ledger_count: relationship_ledgers.len(),
        active_slot_counts,
        unresolved_gap_counts,
        source_class_counts: source_class_counts.clone(),
        status_counts,
    }
}
