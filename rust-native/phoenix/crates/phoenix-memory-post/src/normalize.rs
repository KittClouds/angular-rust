use std::collections::{BTreeMap, BTreeSet};

use phoenix_scope_analysis::{ScopeAnalysisContext, ScopeEntityProfile};
use phoenix_semantic_v2::{
    scope_storage_key, DocumentArchive, ErScopePatchSidecar, MemoryClaimAtom, MemoryClaimStatus,
    MemoryModality, RelationDecisionOutcome, RelationScopePatchSidecar, ScopeLexSidecar,
    SemanticEntityRecord, SessionArchive, StateSchemaScopeSidecar, StateSlotDefinitionRecord,
};
use phoenix_types::{BiTemporalWindow, EntityId, EntityKind, ScopeKey};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::registry::{
    merged_slot_definitions, normalize_relation_family_key, slot_definition_for_relation_family,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryEntityProfile {
    pub entity_id: EntityId,
    pub scope: ScopeKey,
    pub scope_key: String,
    pub canonical_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub effective_kind: Option<EntityKind>,
    pub mention_count: usize,
    pub linked_mention_count: usize,
    #[serde(default)]
    pub continuity_refs: Vec<String>,
    #[serde(default)]
    pub document_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryPendingReview {
    pub review_id: String,
    pub entity_id: EntityId,
    pub slot_key: Option<String>,
    pub relation_family: Option<String>,
    pub outcome: String,
    pub detail: String,
    pub confidence_millis: u32,
    pub temporal: BiTemporalWindow,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryNormalizedBatch {
    #[serde(default)]
    pub entity_profiles: Vec<MemoryEntityProfile>,
    #[serde(default)]
    pub slot_definitions: Vec<StateSlotDefinitionRecord>,
    #[serde(default)]
    pub claims: Vec<MemoryClaimAtom>,
    #[serde(default)]
    pub pending_reviews: Vec<MemoryPendingReview>,
    #[serde(default)]
    pub source_class_counts: BTreeMap<String, usize>,
}

pub fn build_entity_profiles(
    archives: &[DocumentArchive],
    lexical: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    session: Option<&SessionArchive>,
) -> Vec<MemoryEntityProfile> {
    let mut by_entity = FxHashMap::<String, MemoryEntityProfile>::default();
    for archive in archives {
        for entity in &archive.entities {
            let entry = by_entity
                .entry(entity.entity_id.0.clone())
                .or_insert_with(|| profile_from_record(archive, entity, session));
            merge_profile(entry, archive, entity);
        }
    }

    if let Some(lexical) = lexical {
        for alias in &lexical.alias_entries {
            for posting in &alias.postings {
                if let Some(profile) = by_entity.get_mut(&posting.entity_id) {
                    if !profile
                        .aliases
                        .iter()
                        .any(|value| value == &alias.normalized)
                    {
                        profile.aliases.push(alias.normalized.clone());
                    }
                    let continuity_ref =
                        format!("lexical:{}:{}", alias.normalized, posting.document_id);
                    if !profile
                        .continuity_refs
                        .iter()
                        .any(|value| value == &continuity_ref)
                    {
                        profile.continuity_refs.push(continuity_ref);
                    }
                }
            }
        }
    }

    if let Some(er_sidecar) = er_sidecar {
        for alias in &er_sidecar.alias_additions {
            if let Some(profile) = by_entity.get_mut(&alias.entity_id.0) {
                if !profile
                    .aliases
                    .iter()
                    .any(|value| value == &alias.alias_surface)
                {
                    profile.aliases.push(alias.alias_surface.clone());
                }
                profile
                    .continuity_refs
                    .push(format!("er_alias:{}", alias.case_id));
            }
        }
        for override_row in &er_sidecar.type_overrides {
            if let Some(profile) = by_entity.get_mut(&override_row.entity_id.0) {
                profile.effective_kind = Some(override_row.kind.clone());
                profile
                    .continuity_refs
                    .push(format!("er_type:{}", override_row.case_id));
            }
        }
        for link in &er_sidecar.entity_links {
            if let Some(profile) = by_entity.get_mut(&link.entity_id.0) {
                profile.linked_mention_count += 1;
                profile
                    .continuity_refs
                    .push(format!("er_link:{}", link.case_id));
            }
        }
    }

    let mut rows = by_entity.into_values().collect::<Vec<_>>();
    for row in &mut rows {
        row.aliases.sort();
        row.aliases.dedup();
        row.continuity_refs.sort();
        row.continuity_refs.dedup();
        row.document_ids.sort();
        row.document_ids.dedup();
    }
    rows.sort_by(|left, right| left.entity_id.0.cmp(&right.entity_id.0));
    rows
}

pub fn normalize_memory_inputs(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    lexical: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
    state_schema_sidecar: Option<&StateSchemaScopeSidecar>,
) -> MemoryNormalizedBatch {
    let entity_profiles = build_entity_profiles(archives, lexical, er_sidecar, session);
    let label_by_id = entity_profiles
        .iter()
        .map(|profile| (profile.entity_id.0.clone(), profile.canonical_name.clone()))
        .collect::<FxHashMap<_, _>>();
    let document_created_at = archives
        .iter()
        .map(|archive| {
            (
                archive.manifest.document_id.clone(),
                archive.manifest.created_at,
            )
        })
        .collect::<FxHashMap<_, _>>();

    let mut batch = MemoryNormalizedBatch {
        entity_profiles,
        slot_definitions: merged_slot_definitions(state_schema_sidecar),
        ..Default::default()
    };

    if let Some(relation_sidecar) = relation_sidecar {
        for edge in &relation_sidecar.edge_additions {
            let relation_family = normalize_relation_family_key(&edge.edge_type).into_owned();
            let temporal = temporal_at(edge.created_at);
            let slot =
                slot_definition_for_relation_family(&relation_family, &batch.slot_definitions);
            let claim = MemoryClaimAtom {
                claim_id: format!(
                    "claim:relation-edge:{}:{}:{}",
                    edge.document_id, edge.case_id, edge.edge_type
                ),
                document_id: edge.document_id.clone(),
                source_entity_id: Some(edge.source_entity_id.clone()),
                target_entity_id: Some(edge.target_entity_id.clone()),
                slot_key: slot
                    .map(|slot| slot.slot_key.to_owned())
                    .unwrap_or_else(|| format!("relation.{relation_family}")),
                relation_family: Some(relation_family),
                subject_label: entity_label(&label_by_id, &edge.source_entity_id),
                object_label: entity_label(&label_by_id, &edge.target_entity_id),
                object_entity_id: Some(edge.target_entity_id.clone()),
                object_value: entity_label(&label_by_id, &edge.target_entity_id),
                status: MemoryClaimStatus::Active,
                modality: MemoryModality::Asserted,
                confidence_millis: edge.confidence_millis,
                source_class: "relation_edge_addition".to_owned(),
                provenance_label: edge.window_id.clone(),
                window_id: Some(edge.window_id.clone()),
                source_case_id: Some(edge.case_id.clone()),
                temporal,
                evidence_refs: edge.evidence_refs.clone(),
            };
            push_claim(&mut batch, claim);
        }

        for judgment in &relation_sidecar.support_judgments {
            let relation_family = normalize_relation_family_key(&judgment.edge_type).into_owned();
            let slot =
                slot_definition_for_relation_family(&relation_family, &batch.slot_definitions);
            let claim = MemoryClaimAtom {
                claim_id: format!(
                    "claim:relation-support:{}:{}:{}",
                    judgment.document_id, judgment.case_id, judgment.edge_type
                ),
                document_id: judgment.document_id.clone(),
                source_entity_id: Some(judgment.source_entity_id.clone()),
                target_entity_id: Some(judgment.target_entity_id.clone()),
                slot_key: slot
                    .map(|slot| slot.slot_key.to_owned())
                    .unwrap_or_else(|| format!("relation.{relation_family}")),
                relation_family: Some(relation_family),
                subject_label: entity_label(&label_by_id, &judgment.source_entity_id),
                object_label: entity_label(&label_by_id, &judgment.target_entity_id),
                object_entity_id: Some(judgment.target_entity_id.clone()),
                object_value: entity_label(&label_by_id, &judgment.target_entity_id),
                status: MemoryClaimStatus::Supported,
                modality: MemoryModality::Asserted,
                confidence_millis: judgment.confidence_millis,
                source_class: "relation_support_judgment".to_owned(),
                provenance_label: judgment.window_id.clone(),
                window_id: Some(judgment.window_id.clone()),
                source_case_id: Some(judgment.case_id.clone()),
                temporal: temporal_at(judgment.created_at),
                evidence_refs: judgment.evidence_refs.clone(),
            };
            push_claim(&mut batch, claim);
        }

        for judgment in &relation_sidecar.contradiction_judgments {
            let relation_family = normalize_relation_family_key(&judgment.edge_type).into_owned();
            let slot =
                slot_definition_for_relation_family(&relation_family, &batch.slot_definitions);
            let claim = MemoryClaimAtom {
                claim_id: format!(
                    "claim:relation-contradiction:{}:{}:{}",
                    judgment.document_id, judgment.case_id, judgment.edge_type
                ),
                document_id: judgment.document_id.clone(),
                source_entity_id: Some(judgment.source_entity_id.clone()),
                target_entity_id: Some(judgment.target_entity_id.clone()),
                slot_key: slot
                    .map(|slot| slot.slot_key.to_owned())
                    .unwrap_or_else(|| format!("relation.{relation_family}")),
                relation_family: Some(relation_family),
                subject_label: entity_label(&label_by_id, &judgment.source_entity_id),
                object_label: entity_label(&label_by_id, &judgment.target_entity_id),
                object_entity_id: Some(judgment.target_entity_id.clone()),
                object_value: entity_label(&label_by_id, &judgment.target_entity_id),
                status: MemoryClaimStatus::Contradicted,
                modality: MemoryModality::Asserted,
                confidence_millis: judgment.confidence_millis,
                source_class: "relation_contradiction_judgment".to_owned(),
                provenance_label: judgment.window_id.clone(),
                window_id: Some(judgment.window_id.clone()),
                source_case_id: Some(judgment.case_id.clone()),
                temporal: temporal_at(judgment.created_at),
                evidence_refs: judgment.evidence_refs.clone(),
            };
            push_claim(&mut batch, claim);
        }

        for decision in &relation_sidecar.decisions {
            if !matches!(
                decision.outcome,
                RelationDecisionOutcome::Defer | RelationDecisionOutcome::Reject
            ) {
                continue;
            }
            let slot_key = decision
                .edge_type
                .as_deref()
                .and_then(|edge_type| {
                    slot_definition_for_relation_family(edge_type, &batch.slot_definitions)
                })
                .map(|slot| slot.slot_key.to_owned());
            let Some(entity_id) = decision.source_entity_id.clone() else {
                continue;
            };
            batch.pending_reviews.push(MemoryPendingReview {
                review_id: decision.case_id.clone(),
                entity_id,
                slot_key,
                relation_family: decision
                    .edge_type
                    .as_deref()
                    .map(|edge_type| normalize_relation_family_key(edge_type).into_owned()),
                outcome: format!("{:?}", decision.outcome).to_lowercase(),
                detail: decision.rationale.clone(),
                confidence_millis: decision.score_millis.max(0) as u32,
                temporal: temporal_at(decision.reviewed_at),
            });
        }
    }

    let mut archived_relation_keys = BTreeSet::new();
    for claim in &batch.claims {
        if let (Some(source_entity_id), Some(target_entity_id), Some(relation_family)) = (
            claim.source_entity_id.as_ref(),
            claim.target_entity_id.as_ref(),
            claim.relation_family.as_ref(),
        ) {
            archived_relation_keys.insert((
                claim.document_id.clone(),
                source_entity_id.0.clone(),
                target_entity_id.0.clone(),
                relation_family.clone(),
            ));
        }
    }

    for archive in archives {
        let created_at = document_created_at
            .get(&archive.manifest.document_id)
            .copied()
            .unwrap_or(archive.manifest.created_at);
        for relation in &archive.relations {
            let relation_family = normalize_relation_family_key(&relation.edge_type).into_owned();
            let key = (
                archive.manifest.document_id.clone(),
                relation.source_entity_id.0.clone(),
                relation.target_entity_id.0.clone(),
                relation_family.clone(),
            );
            if archived_relation_keys.contains(&key) {
                continue;
            }
            let slot =
                slot_definition_for_relation_family(&relation_family, &batch.slot_definitions);
            let claim = MemoryClaimAtom {
                claim_id: format!(
                    "claim:archive-relation:{}:{}:{}:{}",
                    archive.manifest.document_id,
                    relation.source_entity_id.0,
                    relation.target_entity_id.0,
                    relation.edge_type
                ),
                document_id: archive.manifest.document_id.clone(),
                source_entity_id: Some(relation.source_entity_id.clone()),
                target_entity_id: Some(relation.target_entity_id.clone()),
                slot_key: slot
                    .map(|slot| slot.slot_key.to_owned())
                    .unwrap_or_else(|| format!("relation.{relation_family}")),
                relation_family: Some(relation_family),
                subject_label: entity_label(&label_by_id, &relation.source_entity_id),
                object_label: entity_label(&label_by_id, &relation.target_entity_id),
                object_entity_id: Some(relation.target_entity_id.clone()),
                object_value: entity_label(&label_by_id, &relation.target_entity_id),
                status: MemoryClaimStatus::Candidate,
                modality: MemoryModality::Asserted,
                confidence_millis: 500,
                source_class: "archive_relation".to_owned(),
                provenance_label: format!(
                    "archive_relation:{}:{}",
                    relation.sentence_index,
                    relation.chunk_id.clone().unwrap_or_default()
                ),
                window_id: None,
                source_case_id: None,
                temporal: temporal_at(created_at),
                evidence_refs: Vec::new(),
            };
            push_claim(&mut batch, claim);
        }
    }

    if let Some(er_sidecar) = er_sidecar {
        for alias in &er_sidecar.alias_additions {
            let label = entity_label(&label_by_id, &alias.entity_id);
            push_claim(
                &mut batch,
                MemoryClaimAtom {
                    claim_id: format!("claim:er-alias:{}:{}", alias.document_id, alias.case_id),
                    document_id: alias.document_id.clone(),
                    source_entity_id: Some(alias.entity_id.clone()),
                    target_entity_id: None,
                    slot_key: "identity.alias".to_owned(),
                    relation_family: None,
                    subject_label: label.clone(),
                    object_label: alias.alias_surface.clone(),
                    object_entity_id: None,
                    object_value: alias.alias_surface.clone(),
                    status: MemoryClaimStatus::Active,
                    modality: MemoryModality::Observed,
                    confidence_millis: alias.confidence_millis,
                    source_class: "er_alias_addition".to_owned(),
                    provenance_label: alias.case_id.clone(),
                    window_id: None,
                    source_case_id: Some(alias.case_id.clone()),
                    temporal: temporal_at(alias.created_at),
                    evidence_refs: vec![format!("alias:{}", alias.normalized)],
                },
            );
        }

        for override_row in &er_sidecar.type_overrides {
            let label = entity_label(&label_by_id, &override_row.entity_id);
            push_claim(
                &mut batch,
                MemoryClaimAtom {
                    claim_id: format!(
                        "claim:er-type:{}:{}",
                        override_row.document_id, override_row.case_id
                    ),
                    document_id: override_row.document_id.clone(),
                    source_entity_id: Some(override_row.entity_id.clone()),
                    target_entity_id: None,
                    slot_key: "identity.kind".to_owned(),
                    relation_family: None,
                    subject_label: label,
                    object_label: format!("{:?}", override_row.kind),
                    object_entity_id: None,
                    object_value: format!("{:?}", override_row.kind),
                    status: MemoryClaimStatus::Active,
                    modality: MemoryModality::Observed,
                    confidence_millis: override_row.confidence_millis,
                    source_class: "er_type_override".to_owned(),
                    provenance_label: override_row.case_id.clone(),
                    window_id: None,
                    source_case_id: Some(override_row.case_id.clone()),
                    temporal: temporal_at(override_row.created_at),
                    evidence_refs: Vec::new(),
                },
            );
        }

        for link in &er_sidecar.entity_links {
            let label = entity_label(&label_by_id, &link.entity_id);
            push_claim(
                &mut batch,
                MemoryClaimAtom {
                    claim_id: format!("claim:er-link:{}:{}", link.document_id, link.case_id),
                    document_id: link.document_id.clone(),
                    source_entity_id: Some(link.entity_id.clone()),
                    target_entity_id: None,
                    slot_key: "identity.link".to_owned(),
                    relation_family: None,
                    subject_label: label,
                    object_label: link.entity_id.0.clone(),
                    object_entity_id: Some(link.entity_id.clone()),
                    object_value: link.entity_id.0.clone(),
                    status: MemoryClaimStatus::Active,
                    modality: MemoryModality::Observed,
                    confidence_millis: link.confidence_millis,
                    source_class: "er_entity_link".to_owned(),
                    provenance_label: link.case_id.clone(),
                    window_id: None,
                    source_case_id: Some(link.case_id.clone()),
                    temporal: temporal_at(link.created_at),
                    evidence_refs: Vec::new(),
                },
            );
        }
    }

    batch
        .claims
        .sort_by(|left, right| left.claim_id.cmp(&right.claim_id));
    batch
        .pending_reviews
        .sort_by(|left, right| left.review_id.cmp(&right.review_id));
    batch
}

pub fn normalize_memory_inputs_from_analysis(
    analysis: &ScopeAnalysisContext,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
    state_schema_sidecar: Option<&StateSchemaScopeSidecar>,
) -> MemoryNormalizedBatch {
    let entity_profiles = memory_entity_profiles_from_scope(analysis);
    let label_by_id = analysis.label_by_entity.as_ref();
    let document_created_at = &analysis.runtime.indices.document_created_at;

    let mut batch = MemoryNormalizedBatch {
        entity_profiles,
        slot_definitions: merged_slot_definitions(state_schema_sidecar),
        ..Default::default()
    };

    if let Some(relation_sidecar) = relation_sidecar {
        for edge in &relation_sidecar.edge_additions {
            let relation_family = normalize_relation_family_key(&edge.edge_type).into_owned();
            let temporal = temporal_at(edge.created_at);
            let slot =
                slot_definition_for_relation_family(&relation_family, &batch.slot_definitions);
            let claim = MemoryClaimAtom {
                claim_id: format!(
                    "claim:relation-edge:{}:{}:{}",
                    edge.document_id, edge.case_id, edge.edge_type
                ),
                document_id: edge.document_id.clone(),
                source_entity_id: Some(edge.source_entity_id.clone()),
                target_entity_id: Some(edge.target_entity_id.clone()),
                slot_key: slot
                    .map(|slot| slot.slot_key.to_owned())
                    .unwrap_or_else(|| format!("relation.{relation_family}")),
                relation_family: Some(relation_family),
                subject_label: entity_label(label_by_id, &edge.source_entity_id),
                object_label: entity_label(label_by_id, &edge.target_entity_id),
                object_entity_id: Some(edge.target_entity_id.clone()),
                object_value: entity_label(label_by_id, &edge.target_entity_id),
                status: MemoryClaimStatus::Active,
                modality: MemoryModality::Asserted,
                confidence_millis: edge.confidence_millis,
                source_class: "relation_edge_addition".to_owned(),
                provenance_label: edge.window_id.clone(),
                window_id: Some(edge.window_id.clone()),
                source_case_id: Some(edge.case_id.clone()),
                temporal,
                evidence_refs: edge.evidence_refs.clone(),
            };
            push_claim(&mut batch, claim);
        }

        for judgment in &relation_sidecar.support_judgments {
            let relation_family = normalize_relation_family_key(&judgment.edge_type).into_owned();
            let slot_key =
                slot_definition_for_relation_family(&relation_family, &batch.slot_definitions)
                    .map(|slot| slot.slot_key.to_owned())
                    .unwrap_or_else(|| format!("relation.{relation_family}"));
            push_claim(
                &mut batch,
                MemoryClaimAtom {
                    claim_id: format!(
                        "claim:relation-support:{}:{}:{}",
                        judgment.document_id, judgment.case_id, judgment.edge_type
                    ),
                    document_id: judgment.document_id.clone(),
                    source_entity_id: Some(judgment.source_entity_id.clone()),
                    target_entity_id: Some(judgment.target_entity_id.clone()),
                    slot_key,
                    relation_family: Some(relation_family),
                    subject_label: entity_label(label_by_id, &judgment.source_entity_id),
                    object_label: entity_label(label_by_id, &judgment.target_entity_id),
                    object_entity_id: Some(judgment.target_entity_id.clone()),
                    object_value: entity_label(label_by_id, &judgment.target_entity_id),
                    status: MemoryClaimStatus::Supported,
                    modality: MemoryModality::Asserted,
                    confidence_millis: judgment.confidence_millis,
                    source_class: "relation_support_judgment".to_owned(),
                    provenance_label: judgment.window_id.clone(),
                    window_id: Some(judgment.window_id.clone()),
                    source_case_id: Some(judgment.case_id.clone()),
                    temporal: temporal_at(judgment.created_at),
                    evidence_refs: judgment.evidence_refs.clone(),
                },
            );
        }

        for judgment in &relation_sidecar.contradiction_judgments {
            let relation_family = normalize_relation_family_key(&judgment.edge_type).into_owned();
            let slot_key =
                slot_definition_for_relation_family(&relation_family, &batch.slot_definitions)
                    .map(|slot| slot.slot_key.to_owned())
                    .unwrap_or_else(|| format!("relation.{relation_family}"));
            push_claim(
                &mut batch,
                MemoryClaimAtom {
                    claim_id: format!(
                        "claim:relation-contradiction:{}:{}:{}",
                        judgment.document_id, judgment.case_id, judgment.edge_type
                    ),
                    document_id: judgment.document_id.clone(),
                    source_entity_id: Some(judgment.source_entity_id.clone()),
                    target_entity_id: Some(judgment.target_entity_id.clone()),
                    slot_key,
                    relation_family: Some(relation_family),
                    subject_label: entity_label(label_by_id, &judgment.source_entity_id),
                    object_label: entity_label(label_by_id, &judgment.target_entity_id),
                    object_entity_id: Some(judgment.target_entity_id.clone()),
                    object_value: entity_label(label_by_id, &judgment.target_entity_id),
                    status: MemoryClaimStatus::Contradicted,
                    modality: MemoryModality::Asserted,
                    confidence_millis: judgment.confidence_millis,
                    source_class: "relation_contradiction_judgment".to_owned(),
                    provenance_label: judgment.window_id.clone(),
                    window_id: Some(judgment.window_id.clone()),
                    source_case_id: Some(judgment.case_id.clone()),
                    temporal: temporal_at(judgment.created_at),
                    evidence_refs: judgment.evidence_refs.clone(),
                },
            );
        }

        for decision in &relation_sidecar.decisions {
            if !matches!(
                decision.outcome,
                RelationDecisionOutcome::Defer | RelationDecisionOutcome::Reject
            ) {
                continue;
            }
            let slot_key = decision
                .edge_type
                .as_deref()
                .and_then(|edge_type| {
                    slot_definition_for_relation_family(edge_type, &batch.slot_definitions)
                })
                .map(|slot| slot.slot_key.to_owned());
            let Some(entity_id) = decision.source_entity_id.clone() else {
                continue;
            };
            batch.pending_reviews.push(MemoryPendingReview {
                review_id: decision.case_id.clone(),
                entity_id,
                slot_key,
                relation_family: decision
                    .edge_type
                    .as_deref()
                    .map(|edge_type| normalize_relation_family_key(edge_type).into_owned()),
                outcome: format!("{:?}", decision.outcome).to_lowercase(),
                detail: decision.rationale.clone(),
                confidence_millis: decision.score_millis.max(0) as u32,
                temporal: temporal_at(decision.reviewed_at),
            });
        }
    }

    let mut archived_relation_keys = BTreeSet::new();
    for claim in &batch.claims {
        if let (Some(source_entity_id), Some(target_entity_id), Some(relation_family)) = (
            claim.source_entity_id.as_ref(),
            claim.target_entity_id.as_ref(),
            claim.relation_family.as_ref(),
        ) {
            archived_relation_keys.insert((
                claim.document_id.clone(),
                source_entity_id.0.clone(),
                target_entity_id.0.clone(),
                relation_family.clone(),
            ));
        }
    }

    for archived in analysis.archived_relations.iter() {
        let relation_family =
            normalize_relation_family_key(&archived.relation.edge_type).into_owned();
        let key = (
            archived.document_id.clone(),
            archived.relation.source_entity_id.0.clone(),
            archived.relation.target_entity_id.0.clone(),
            relation_family.clone(),
        );
        if archived_relation_keys.contains(&key) {
            continue;
        }
        let created_at = document_created_at
            .get(&archived.document_id)
            .copied()
            .unwrap_or(archived.created_at);
        let slot_key =
            slot_definition_for_relation_family(&relation_family, &batch.slot_definitions)
                .map(|slot| slot.slot_key.to_owned())
                .unwrap_or_else(|| format!("relation.{relation_family}"));
        push_claim(
            &mut batch,
            MemoryClaimAtom {
                claim_id: format!(
                    "claim:archive-relation:{}:{}:{}:{}",
                    archived.document_id,
                    archived.relation.source_entity_id.0,
                    archived.relation.target_entity_id.0,
                    archived.relation.edge_type
                ),
                document_id: archived.document_id.clone(),
                source_entity_id: Some(archived.relation.source_entity_id.clone()),
                target_entity_id: Some(archived.relation.target_entity_id.clone()),
                slot_key,
                relation_family: Some(relation_family),
                subject_label: entity_label(label_by_id, &archived.relation.source_entity_id),
                object_label: entity_label(label_by_id, &archived.relation.target_entity_id),
                object_entity_id: Some(archived.relation.target_entity_id.clone()),
                object_value: entity_label(label_by_id, &archived.relation.target_entity_id),
                status: MemoryClaimStatus::Candidate,
                modality: MemoryModality::Asserted,
                confidence_millis: 500,
                source_class: "archive_relation".to_owned(),
                provenance_label: format!(
                    "archive_relation:{}:{}",
                    archived.relation.sentence_index,
                    archived.relation.chunk_id.clone().unwrap_or_default()
                ),
                window_id: None,
                source_case_id: None,
                temporal: temporal_at(created_at),
                evidence_refs: Vec::new(),
            },
        );
    }

    if let Some(er_sidecar) = analysis.runtime.sidecars.er.as_ref() {
        for alias in &er_sidecar.alias_additions {
            let label = entity_label(label_by_id, &alias.entity_id);
            push_claim(
                &mut batch,
                MemoryClaimAtom {
                    claim_id: format!("claim:er-alias:{}:{}", alias.document_id, alias.case_id),
                    document_id: alias.document_id.clone(),
                    source_entity_id: Some(alias.entity_id.clone()),
                    target_entity_id: None,
                    slot_key: "identity.alias".to_owned(),
                    relation_family: None,
                    subject_label: label.clone(),
                    object_label: alias.alias_surface.clone(),
                    object_entity_id: None,
                    object_value: alias.alias_surface.clone(),
                    status: MemoryClaimStatus::Active,
                    modality: MemoryModality::Observed,
                    confidence_millis: alias.confidence_millis,
                    source_class: "er_alias_addition".to_owned(),
                    provenance_label: alias.case_id.clone(),
                    window_id: None,
                    source_case_id: Some(alias.case_id.clone()),
                    temporal: temporal_at(alias.created_at),
                    evidence_refs: vec![format!("alias:{}", alias.normalized)],
                },
            );
        }

        for override_row in &er_sidecar.type_overrides {
            let label = entity_label(label_by_id, &override_row.entity_id);
            push_claim(
                &mut batch,
                MemoryClaimAtom {
                    claim_id: format!(
                        "claim:er-type:{}:{}",
                        override_row.document_id, override_row.case_id
                    ),
                    document_id: override_row.document_id.clone(),
                    source_entity_id: Some(override_row.entity_id.clone()),
                    target_entity_id: None,
                    slot_key: "identity.kind".to_owned(),
                    relation_family: None,
                    subject_label: label,
                    object_label: format!("{:?}", override_row.kind),
                    object_entity_id: None,
                    object_value: format!("{:?}", override_row.kind),
                    status: MemoryClaimStatus::Active,
                    modality: MemoryModality::Observed,
                    confidence_millis: override_row.confidence_millis,
                    source_class: "er_type_override".to_owned(),
                    provenance_label: override_row.case_id.clone(),
                    window_id: None,
                    source_case_id: Some(override_row.case_id.clone()),
                    temporal: temporal_at(override_row.created_at),
                    evidence_refs: Vec::new(),
                },
            );
        }

        for link in &er_sidecar.entity_links {
            let label = entity_label(label_by_id, &link.entity_id);
            push_claim(
                &mut batch,
                MemoryClaimAtom {
                    claim_id: format!("claim:er-link:{}:{}", link.document_id, link.case_id),
                    document_id: link.document_id.clone(),
                    source_entity_id: Some(link.entity_id.clone()),
                    target_entity_id: None,
                    slot_key: "identity.link".to_owned(),
                    relation_family: None,
                    subject_label: label,
                    object_label: link.entity_id.0.clone(),
                    object_entity_id: Some(link.entity_id.clone()),
                    object_value: link.entity_id.0.clone(),
                    status: MemoryClaimStatus::Active,
                    modality: MemoryModality::Observed,
                    confidence_millis: link.confidence_millis,
                    source_class: "er_entity_link".to_owned(),
                    provenance_label: link.case_id.clone(),
                    window_id: None,
                    source_case_id: Some(link.case_id.clone()),
                    temporal: temporal_at(link.created_at),
                    evidence_refs: Vec::new(),
                },
            );
        }
    }

    batch
        .claims
        .sort_by(|left, right| left.claim_id.cmp(&right.claim_id));
    batch
        .pending_reviews
        .sort_by(|left, right| left.review_id.cmp(&right.review_id));
    batch
}

fn memory_entity_profiles_from_scope(analysis: &ScopeAnalysisContext) -> Vec<MemoryEntityProfile> {
    analysis
        .entity_profiles
        .iter()
        .map(|profile| memory_entity_profile_from_scope(analysis, profile))
        .collect()
}

fn memory_entity_profile_from_scope(
    analysis: &ScopeAnalysisContext,
    profile: &ScopeEntityProfile,
) -> MemoryEntityProfile {
    MemoryEntityProfile {
        entity_id: profile.entity_id.clone(),
        scope: analysis.scope.clone(),
        scope_key: analysis.scope_key.clone(),
        canonical_name: profile.canonical_name.clone(),
        aliases: profile.aliases.clone(),
        effective_kind: profile.effective_kind.clone(),
        mention_count: profile.mention_count,
        linked_mention_count: profile.linked_mention_count,
        continuity_refs: profile.continuity_refs.clone(),
        document_ids: profile.document_ids.clone(),
    }
}

fn profile_from_record(
    archive: &DocumentArchive,
    entity: &SemanticEntityRecord,
    session: Option<&SessionArchive>,
) -> MemoryEntityProfile {
    let mut continuity_refs = Vec::new();
    if let Some(session) = session {
        continuity_refs.push(format!("session:{}", session.session_id.0));
    }
    MemoryEntityProfile {
        entity_id: entity.entity_id.clone(),
        scope: archive.manifest.scope.clone(),
        scope_key: scope_storage_key(&archive.manifest.scope),
        canonical_name: entity.canonical_name.clone(),
        aliases: entity.aliases.clone(),
        effective_kind: entity.kind.clone(),
        mention_count: entity.mention_count,
        linked_mention_count: 0,
        continuity_refs,
        document_ids: vec![archive.manifest.document_id.clone()],
    }
}

fn merge_profile(
    profile: &mut MemoryEntityProfile,
    archive: &DocumentArchive,
    entity: &SemanticEntityRecord,
) {
    if profile.canonical_name.is_empty() {
        profile.canonical_name = entity.canonical_name.clone();
    }
    if profile.effective_kind.is_none() {
        profile.effective_kind = entity.kind.clone();
    }
    profile.mention_count += entity.mention_count;
    profile.aliases.extend(entity.aliases.clone());
    profile
        .document_ids
        .push(archive.manifest.document_id.clone());
}

fn entity_label(label_by_id: &FxHashMap<String, String>, entity_id: &EntityId) -> String {
    label_by_id
        .get(&entity_id.0)
        .cloned()
        .unwrap_or_else(|| entity_id.0.clone())
}

fn temporal_at(ts: i64) -> BiTemporalWindow {
    BiTemporalWindow {
        valid_from: Some(ts),
        valid_to: None,
        recorded_from: Some(ts),
        recorded_to: None,
    }
}

fn push_claim(batch: &mut MemoryNormalizedBatch, claim: MemoryClaimAtom) {
    *batch
        .source_class_counts
        .entry(claim.source_class.clone())
        .or_default() += 1;
    batch.claims.push(claim);
}
