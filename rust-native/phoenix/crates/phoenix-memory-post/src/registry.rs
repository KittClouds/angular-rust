use std::borrow::Cow;

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

pub fn normalize_relation_family_key(relation_family: &str) -> Cow<'_, str> {
    let base = relation_family
        .rsplit("::")
        .next()
        .unwrap_or(relation_family)
        .trim();
    if !base
        .bytes()
        .any(|byte| byte.is_ascii_uppercase() || byte == b'-' || byte == b' ')
    {
        return Cow::Borrowed(base);
    }

    let mut normalized = String::with_capacity(base.len());
    for byte in base.bytes() {
        match byte {
            b'A'..=b'Z' => normalized.push((byte + 32) as char),
            b'-' | b' ' => normalized.push('_'),
            _ => normalized.push(byte as char),
        }
    }
    Cow::Owned(normalized)
}

pub fn slot_definition_for_relation_family<'a>(
    relation_family: &str,
    definitions: &'a [StateSlotDefinitionRecord],
) -> Option<&'a StateSlotDefinitionRecord> {
    let normalized = normalize_relation_family_key(relation_family);
    definitions.iter().find(|definition| {
        definition.lifecycle != StateSlotLifecycle::Deprecated
            && definition
                .relation_families
                .iter()
                .any(|value| normalize_relation_family_key(value).as_ref() == normalized.as_ref())
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

#[cfg(test)]
mod tests {
    use super::{normalize_relation_family_key, slot_definition_for_relation_family};
    use phoenix_semantic_v2::default_state_slot_definitions;

    #[test]
    fn normalizes_prefixed_relation_family_keys() {
        assert_eq!(
            normalize_relation_family_key("window::located_in").as_ref(),
            "located_in"
        );
        assert_eq!(
            normalize_relation_family_key("Works-For").as_ref(),
            "works_for"
        );
    }

    #[test]
    fn slot_lookup_accepts_prefixed_relation_families() {
        let definitions = default_state_slot_definitions();
        let slot = slot_definition_for_relation_family("window::located_in", &definitions)
            .expect("location slot");
        assert_eq!(slot.slot_key, "entity.location");
    }
}
