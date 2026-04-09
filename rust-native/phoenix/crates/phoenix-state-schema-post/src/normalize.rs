use std::collections::{BTreeMap, BTreeSet};

use phoenix_semantic_v2::{
    default_state_slot_definitions, default_state_slot_families, DocumentArchive,
    RelationScopePatchSidecar, StateSlotDefinitionRecord, StateSlotFamilyRecord,
    StateSlotOwnerType, StateSlotValueType,
};
use phoenix_types::EntityId;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSchemaEvidenceRow {
    pub slot_key: String,
    pub family_key: String,
    pub relation_family: String,
    pub source_document_id: String,
    pub source_entity_id: EntityId,
    pub target_entity_id: Option<EntityId>,
    pub target_label: String,
    pub owner_type: StateSlotOwnerType,
    pub value_type: StateSlotValueType,
    pub confidence_millis: u32,
    pub created_at: i64,
    pub source_class: String,
    pub positive: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSchemaNormalizedInputs {
    #[serde(default)]
    pub slot_families: Vec<StateSlotFamilyRecord>,
    #[serde(default)]
    pub seed_slot_definitions: Vec<StateSlotDefinitionRecord>,
    #[serde(default)]
    pub evidence_rows: Vec<StateSchemaEvidenceRow>,
    #[serde(default)]
    pub diagnostics: BTreeMap<String, usize>,
}

pub fn normalize_state_schema_inputs(
    archives: &[DocumentArchive],
    relation_sidecar: Option<&RelationScopePatchSidecar>,
) -> StateSchemaNormalizedInputs {
    let slot_families = default_state_slot_families();
    let seed_slot_definitions = default_state_slot_definitions();
    let label_by_entity = entity_labels(archives);
    let seed_by_relation = seed_definitions_by_relation(&seed_slot_definitions);
    let mut evidence_rows = Vec::new();
    let mut diagnostics = BTreeMap::<String, usize>::new();

    if let Some(sidecar) = relation_sidecar {
        for edge in &sidecar.edge_additions {
            let spec = classify_relation_family(&edge.edge_type, &seed_by_relation);
            if should_filter_discovered_relation(&edge.edge_type, &spec.family_key) {
                *diagnostics
                    .entry(format!("filtered_relation_family:{}", edge.edge_type))
                    .or_default() += 1;
                continue;
            }
            evidence_rows.push(StateSchemaEvidenceRow {
                slot_key: spec.slot_key,
                family_key: spec.family_key,
                relation_family: edge.edge_type.clone(),
                source_document_id: edge.document_id.clone(),
                source_entity_id: edge.source_entity_id.clone(),
                target_entity_id: Some(edge.target_entity_id.clone()),
                target_label: label_by_entity
                    .get(&edge.target_entity_id.0)
                    .cloned()
                    .unwrap_or_else(|| edge.target_entity_id.0.clone()),
                owner_type: spec.owner_type,
                value_type: spec.value_type,
                confidence_millis: edge.confidence_millis,
                created_at: edge.created_at,
                source_class: "relation_edge_addition".to_owned(),
                positive: true,
            });
            *diagnostics
                .entry(format!("relation_family:{}", edge.edge_type))
                .or_default() += 1;
        }

        for judgment in &sidecar.support_judgments {
            let spec = classify_relation_family(&judgment.edge_type, &seed_by_relation);
            if should_filter_discovered_relation(&judgment.edge_type, &spec.family_key) {
                *diagnostics
                    .entry(format!("filtered_relation_family:{}", judgment.edge_type))
                    .or_default() += 1;
                continue;
            }
            evidence_rows.push(StateSchemaEvidenceRow {
                slot_key: spec.slot_key,
                family_key: spec.family_key,
                relation_family: judgment.edge_type.clone(),
                source_document_id: judgment.document_id.clone(),
                source_entity_id: judgment.source_entity_id.clone(),
                target_entity_id: Some(judgment.target_entity_id.clone()),
                target_label: label_by_entity
                    .get(&judgment.target_entity_id.0)
                    .cloned()
                    .unwrap_or_else(|| judgment.target_entity_id.0.clone()),
                owner_type: spec.owner_type,
                value_type: spec.value_type,
                confidence_millis: judgment.confidence_millis,
                created_at: judgment.created_at,
                source_class: "relation_support_judgment".to_owned(),
                positive: true,
            });
        }

        for judgment in &sidecar.contradiction_judgments {
            let spec = classify_relation_family(&judgment.edge_type, &seed_by_relation);
            if should_filter_discovered_relation(&judgment.edge_type, &spec.family_key) {
                *diagnostics
                    .entry(format!("filtered_relation_family:{}", judgment.edge_type))
                    .or_default() += 1;
                continue;
            }
            evidence_rows.push(StateSchemaEvidenceRow {
                slot_key: spec.slot_key,
                family_key: spec.family_key,
                relation_family: judgment.edge_type.clone(),
                source_document_id: judgment.document_id.clone(),
                source_entity_id: judgment.source_entity_id.clone(),
                target_entity_id: Some(judgment.target_entity_id.clone()),
                target_label: label_by_entity
                    .get(&judgment.target_entity_id.0)
                    .cloned()
                    .unwrap_or_else(|| judgment.target_entity_id.0.clone()),
                owner_type: spec.owner_type,
                value_type: spec.value_type,
                confidence_millis: judgment.confidence_millis,
                created_at: judgment.created_at,
                source_class: "relation_contradiction_judgment".to_owned(),
                positive: false,
            });
        }
    }

    let mut archived_relation_keys = BTreeSet::new();
    for row in &evidence_rows {
        archived_relation_keys.insert((
            row.source_document_id.clone(),
            row.source_entity_id.0.clone(),
            row.target_entity_id
                .as_ref()
                .map(|value| value.0.clone())
                .unwrap_or_default(),
            row.relation_family.clone(),
        ));
    }

    for archive in archives {
        for relation in &archive.relations {
            let key = (
                archive.manifest.document_id.clone(),
                relation.source_entity_id.0.clone(),
                relation.target_entity_id.0.clone(),
                relation.edge_type.clone(),
            );
            if archived_relation_keys.contains(&key) {
                continue;
            }
            let spec = classify_relation_family(&relation.edge_type, &seed_by_relation);
            if should_filter_discovered_relation(&relation.edge_type, &spec.family_key) {
                *diagnostics
                    .entry(format!("filtered_relation_family:{}", relation.edge_type))
                    .or_default() += 1;
                continue;
            }
            evidence_rows.push(StateSchemaEvidenceRow {
                slot_key: spec.slot_key,
                family_key: spec.family_key,
                relation_family: relation.edge_type.clone(),
                source_document_id: archive.manifest.document_id.clone(),
                source_entity_id: relation.source_entity_id.clone(),
                target_entity_id: Some(relation.target_entity_id.clone()),
                target_label: label_by_entity
                    .get(&relation.target_entity_id.0)
                    .cloned()
                    .unwrap_or_else(|| relation.target_entity_id.0.clone()),
                owner_type: spec.owner_type,
                value_type: spec.value_type,
                confidence_millis: 500,
                created_at: archive.manifest.created_at,
                source_class: "archive_relation".to_owned(),
                positive: true,
            });
        }
    }

    StateSchemaNormalizedInputs {
        slot_families,
        seed_slot_definitions,
        evidence_rows,
        diagnostics,
    }
}

#[derive(Clone, Debug)]
struct SlotMappingSpec {
    slot_key: String,
    family_key: String,
    owner_type: StateSlotOwnerType,
    value_type: StateSlotValueType,
}

fn entity_labels(archives: &[DocumentArchive]) -> FxHashMap<String, String> {
    let mut labels = FxHashMap::<String, String>::default();
    for archive in archives {
        for entity in &archive.entities {
            labels
                .entry(entity.entity_id.0.clone())
                .or_insert_with(|| entity.canonical_name.clone());
        }
    }
    labels
}

fn seed_definitions_by_relation(
    definitions: &[StateSlotDefinitionRecord],
) -> FxHashMap<String, StateSlotDefinitionRecord> {
    let mut rows = FxHashMap::<String, StateSlotDefinitionRecord>::default();
    for definition in definitions {
        for family in &definition.relation_families {
            rows.insert(family.to_ascii_lowercase(), definition.clone());
        }
        for alias in &definition.aliases {
            rows.entry(alias.to_ascii_lowercase())
                .or_insert_with(|| definition.clone());
        }
    }
    rows
}

fn classify_relation_family(
    relation_family: &str,
    seed_by_relation: &FxHashMap<String, StateSlotDefinitionRecord>,
) -> SlotMappingSpec {
    let normalized = normalize_relation_family(relation_family);
    if let Some(definition) = seed_by_relation.get(&normalized) {
        return SlotMappingSpec {
            slot_key: definition.slot_key.clone(),
            family_key: definition
                .family_id
                .0
                .strip_prefix("family:")
                .unwrap_or("discovered")
                .to_owned(),
            owner_type: definition.owner_type,
            value_type: definition.value_type,
        };
    }

    if contains_any(&normalized, &["status", "phase"]) {
        return reserved_spec(
            "project.status",
            "lifecycle",
            StateSlotOwnerType::Project,
            StateSlotValueType::Enum,
        );
    }
    if contains_any(&normalized, &["assigned", "owner", "assignee"]) {
        return reserved_spec(
            "task.owner",
            "assignment",
            StateSlotOwnerType::Task,
            StateSlotValueType::EntityRef,
        );
    }
    if contains_any(&normalized, &["deadline", "due", "schedule"]) {
        return reserved_spec(
            "task.due_date",
            "schedule",
            StateSlotOwnerType::Task,
            StateSlotValueType::Date,
        );
    }
    if contains_any(&normalized, &["completion", "completed", "task_state"]) {
        return reserved_spec(
            "task.completion_state",
            "lifecycle",
            StateSlotOwnerType::Task,
            StateSlotValueType::Enum,
        );
    }
    if contains_any(&normalized, &["prefer", "likes", "preference"]) {
        return reserved_spec(
            "entity.preference",
            "role_preference",
            StateSlotOwnerType::Entity,
            StateSlotValueType::RankedChoice,
        );
    }
    if contains_any(&normalized, &["role", "acts_as", "serves_as"]) {
        return reserved_spec(
            "entity.role",
            "role_preference",
            StateSlotOwnerType::Entity,
            StateSlotValueType::String,
        );
    }

    SlotMappingSpec {
        slot_key: format!("state.{normalized}"),
        family_key: "discovered".to_owned(),
        owner_type: StateSlotOwnerType::Unknown,
        value_type: StateSlotValueType::EntityRef,
    }
}

fn reserved_spec(
    slot_key: &str,
    family_key: &str,
    owner_type: StateSlotOwnerType,
    value_type: StateSlotValueType,
) -> SlotMappingSpec {
    SlotMappingSpec {
        slot_key: slot_key.to_owned(),
        family_key: family_key.to_owned(),
        owner_type,
        value_type,
    }
}

fn normalize_relation_family(value: &str) -> String {
    let mut normalized = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    while normalized.contains("__") {
        normalized = normalized.replace("__", "_");
    }
    normalized.trim_matches('_').to_owned()
}

fn contains_any(value: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|pattern| value.contains(pattern))
}

fn should_filter_discovered_relation(relation_family: &str, family_key: &str) -> bool {
    if family_key != "discovered" {
        return false;
    }
    matches!(
        normalize_relation_family(relation_family).as_str(),
        "relates_to" | "meets" | "gives" | "attacks" | "talks_to" | "looks_at"
    )
}
