use phoenix_semantic_v2::{
    StateSlotCandidateRecord, StateSlotCardinality, StateSlotDefinitionId,
    StateSlotDefinitionRecord, StateSlotLifecycle, StateSlotPromotionDecisionId,
    StateSlotPromotionDecisionRecord, StateSlotTemporalMode, StateSlotUpdateOperator,
    StateWriteProposal, StateWriteProposalId,
};

use crate::normalize::StateSchemaEvidenceRow;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PromotionOutput {
    pub slot_definitions: Vec<StateSlotDefinitionRecord>,
    pub promotion_decisions: Vec<StateSlotPromotionDecisionRecord>,
    pub write_proposals: Vec<StateWriteProposal>,
}

pub fn promote_slot_definitions(
    seed_definitions: &[StateSlotDefinitionRecord],
    candidates: &[StateSlotCandidateRecord],
    rows: &[StateSchemaEvidenceRow],
    created_at: i64,
) -> PromotionOutput {
    let mut slot_definitions = seed_definitions.to_vec();
    let mut promotion_decisions = Vec::new();

    for candidate in candidates {
        let existing_index = slot_definitions
            .iter()
            .position(|definition| definition.slot_key == candidate.slot_key);

        let mut definition = existing_index
            .map(|index| slot_definitions[index].clone())
            .unwrap_or_else(|| StateSlotDefinitionRecord {
                slot_id: StateSlotDefinitionId(format!("slot:{}", candidate.slot_key)),
                family_id: candidate.family_id.clone(),
                slot_key: candidate.slot_key.clone(),
                slot_name: candidate.normalized_name.clone(),
                owner_type: candidate.owner_type,
                value_type: candidate.value_type,
                cardinality: StateSlotCardinality::Single,
                temporal_mode: StateSlotTemporalMode::DurableUntilChanged,
                update_operator: if candidate.slot_key.starts_with("state.") {
                    StateSlotUpdateOperator::Infer
                } else {
                    StateSlotUpdateOperator::Replace
                },
                evidence_threshold_millis: 760,
                contradiction_policy: "preserve contradictory evidence and defer uncertain writes"
                    .to_owned(),
                salience_millis: 620,
                lifecycle: StateSlotLifecycle::Candidate,
                single_value: true,
                relationship_only: false,
                relation_families: Vec::new(),
                aliases: Vec::new(),
            });

        let previous_lifecycle = definition.lifecycle;
        merge_relation_families(
            &mut definition.relation_families,
            &candidate.relation_families,
        );
        definition.salience_millis = definition
            .salience_millis
            .max(candidate.utility_score_millis);
        definition.evidence_threshold_millis = definition
            .evidence_threshold_millis
            .min(candidate.canonicalization_score_millis.max(600));
        definition.lifecycle = desired_lifecycle(&definition, candidate);

        match existing_index {
            Some(index) => slot_definitions[index] = definition.clone(),
            None => slot_definitions.push(definition.clone()),
        }

        if previous_lifecycle != definition.lifecycle || existing_index.is_none() {
            promotion_decisions.push(StateSlotPromotionDecisionRecord {
                decision_id: StateSlotPromotionDecisionId(format!(
                    "decision:slot:{}:{}",
                    definition.slot_key, created_at
                )),
                slot_id: definition.slot_id.clone(),
                candidate_ids: vec![candidate.candidate_id.clone()],
                previous_lifecycle,
                next_lifecycle: definition.lifecycle,
                rationale: promotion_rationale(candidate, definition.lifecycle),
                support_count: candidate.support_count,
                conflict_count: candidate.conflict_count,
                utility_score_millis: candidate.utility_score_millis,
                created_at,
            });
        }
    }

    slot_definitions.sort_by(|left, right| left.slot_key.cmp(&right.slot_key));
    promotion_decisions.sort_by(|left, right| left.decision_id.0.cmp(&right.decision_id.0));
    let write_proposals = build_write_proposals(&slot_definitions, rows);

    PromotionOutput {
        slot_definitions,
        promotion_decisions,
        write_proposals,
    }
}

fn merge_relation_families(target: &mut Vec<String>, updates: &[String]) {
    for value in updates {
        if !target.iter().any(|existing| existing == value) {
            target.push(value.clone());
        }
    }
    target.sort();
}

fn desired_lifecycle(
    definition: &StateSlotDefinitionRecord,
    candidate: &StateSlotCandidateRecord,
) -> StateSlotLifecycle {
    let family_key = definition
        .family_id
        .0
        .strip_prefix("family:")
        .unwrap_or("discovered");
    if definition.relationship_only {
        return definition.lifecycle.max(StateSlotLifecycle::Active);
    }
    if family_key == "role_preference" {
        return if candidate.support_count >= 2 && candidate.conflict_count == 0 {
            StateSlotLifecycle::Candidate
        } else {
            definition.lifecycle
        };
    }
    if family_key == "discovered" {
        return StateSlotLifecycle::Candidate.max(definition.lifecycle);
    }
    if matches!(
        definition.lifecycle,
        StateSlotLifecycle::Active | StateSlotLifecycle::Stable
    ) {
        return if candidate.support_count >= 2 {
            StateSlotLifecycle::Stable
        } else {
            StateSlotLifecycle::Active
        };
    }
    if candidate.support_count >= 2 && candidate.conflict_count == 0 {
        StateSlotLifecycle::Active
    } else if candidate.support_count >= 1 {
        StateSlotLifecycle::Candidate
    } else {
        definition.lifecycle
    }
}

fn promotion_rationale(
    candidate: &StateSlotCandidateRecord,
    lifecycle: StateSlotLifecycle,
) -> String {
    match lifecycle {
        StateSlotLifecycle::Active => format!(
            "promoted from evidence-backed candidate with support_count={} document_count={}",
            candidate.support_count, candidate.document_count
        ),
        StateSlotLifecycle::Stable => format!(
            "stabilized active slot with repeated support_count={} and low conflict_count={}",
            candidate.support_count, candidate.conflict_count
        ),
        StateSlotLifecycle::Candidate => format!(
            "retained in candidate mode pending stronger corroboration (support_count={}, conflict_count={})",
            candidate.support_count, candidate.conflict_count
        ),
        StateSlotLifecycle::Reserved => "seed slot retained as reserved".to_owned(),
        StateSlotLifecycle::Deprecated => "slot deprecated by governance".to_owned(),
    }
}

fn build_write_proposals(
    definitions: &[StateSlotDefinitionRecord],
    rows: &[StateSchemaEvidenceRow],
) -> Vec<StateWriteProposal> {
    let mut proposals = Vec::new();
    for row in rows {
        if !row.positive {
            continue;
        }
        let Some(definition) = definitions
            .iter()
            .find(|definition| definition.slot_key == row.slot_key)
        else {
            continue;
        };
        if !matches!(
            definition.lifecycle,
            StateSlotLifecycle::Active | StateSlotLifecycle::Stable
        ) || definition.relationship_only
        {
            continue;
        }
        proposals.push(StateWriteProposal {
            proposal_id: StateWriteProposalId(format!(
                "proposal:{}:{}:{}",
                row.slot_key, row.source_document_id, row.relation_family
            )),
            owner_entity_id: row.source_entity_id.clone(),
            owner_type: definition.owner_type,
            slot_key: definition.slot_key.clone(),
            before_value: None,
            after_value: Some(row.target_label.clone()),
            after_value_entity_id: row.target_entity_id.clone(),
            operation: definition.update_operator,
            effective_time: Some(row.created_at),
            source_document_id: row.source_document_id.clone(),
            source_event_id: None,
            evidence_refs: vec![format!("relation_family:{}", row.relation_family)],
        });
    }
    proposals.sort_by(|left, right| left.proposal_id.0.cmp(&right.proposal_id.0));
    proposals.dedup_by(|left, right| left.proposal_id == right.proposal_id);
    proposals
}
