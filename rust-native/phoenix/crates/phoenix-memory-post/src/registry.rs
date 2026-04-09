use phoenix_semantic_v2::{
    default_state_slot_definitions, StateSchemaScopeSidecar, StateSlotDefinitionRecord,
    StateSlotLifecycle,
};

pub fn merged_slot_definitions(
    state_schema_sidecar: Option<&StateSchemaScopeSidecar>,
) -> Vec<StateSlotDefinitionRecord> {
    let mut definitions = default_state_slot_definitions();
    if let Some(sidecar) = state_schema_sidecar {
        for update in &sidecar.slot_definitions {
            match definitions
                .iter()
                .position(|definition| definition.slot_key == update.slot_key)
            {
                Some(index) => definitions[index] = update.clone(),
                None => definitions.push(update.clone()),
            }
        }
    }
    definitions.sort_by(|left, right| left.slot_key.cmp(&right.slot_key));
    definitions
}

pub fn slot_definition_for_relation_family<'a>(
    relation_family: &str,
    definitions: &'a [StateSlotDefinitionRecord],
) -> Option<&'a StateSlotDefinitionRecord> {
    definitions.iter().find(|definition| {
        definition.lifecycle != StateSlotLifecycle::Deprecated
            && definition
                .relation_families
                .iter()
                .any(|value| value == relation_family)
    })
}

pub fn active_scalar_slot_keys(definitions: &[StateSlotDefinitionRecord]) -> Vec<String> {
    definitions
        .iter()
        .filter(|definition| {
            matches!(
                definition.lifecycle,
                StateSlotLifecycle::Active | StateSlotLifecycle::Stable
            ) && !definition.relationship_only
        })
        .map(|definition| definition.slot_key.clone())
        .collect()
}

pub fn source_class_priority(source_class: &str) -> u8 {
    match source_class {
        "relation_edge_addition" => 5,
        "relation_support_judgment" => 4,
        "archive_relation" => 3,
        "er_type_override" => 2,
        "er_alias_addition" => 2,
        "er_entity_link" => 2,
        "relation_contradiction_judgment" => 1,
        _ => 0,
    }
}
