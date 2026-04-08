#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrackedSlot {
    pub slot_key: &'static str,
    pub relation_family: &'static str,
    pub single_value: bool,
    pub relationship_only: bool,
    pub active: bool,
}

pub const TRACKED_SLOTS: &[TrackedSlot] = &[
    TrackedSlot {
        slot_key: "entity.location",
        relation_family: "located_in",
        single_value: true,
        relationship_only: false,
        active: true,
    },
    TrackedSlot {
        slot_key: "entity.employer",
        relation_family: "works_for",
        single_value: true,
        relationship_only: false,
        active: true,
    },
    TrackedSlot {
        slot_key: "entity.membership",
        relation_family: "member_of",
        single_value: true,
        relationship_only: false,
        active: true,
    },
    TrackedSlot {
        slot_key: "relationship.commands",
        relation_family: "commands",
        single_value: false,
        relationship_only: true,
        active: true,
    },
    TrackedSlot {
        slot_key: "relationship.protects",
        relation_family: "protects",
        single_value: false,
        relationship_only: true,
        active: true,
    },
    TrackedSlot {
        slot_key: "project.status",
        relation_family: "",
        single_value: true,
        relationship_only: false,
        active: false,
    },
    TrackedSlot {
        slot_key: "task.owner",
        relation_family: "",
        single_value: true,
        relationship_only: false,
        active: false,
    },
    TrackedSlot {
        slot_key: "task.due_date",
        relation_family: "",
        single_value: true,
        relationship_only: false,
        active: false,
    },
    TrackedSlot {
        slot_key: "task.completion_state",
        relation_family: "",
        single_value: true,
        relationship_only: false,
        active: false,
    },
    TrackedSlot {
        slot_key: "entity.preference",
        relation_family: "",
        single_value: true,
        relationship_only: false,
        active: false,
    },
    TrackedSlot {
        slot_key: "entity.role",
        relation_family: "",
        single_value: true,
        relationship_only: false,
        active: false,
    },
];

pub fn slot_for_relation_family(relation_family: &str) -> Option<&'static TrackedSlot> {
    TRACKED_SLOTS
        .iter()
        .find(|slot| slot.active && slot.relation_family == relation_family)
}

pub fn active_scalar_slot_keys() -> Vec<&'static str> {
    TRACKED_SLOTS
        .iter()
        .filter(|slot| slot.active && !slot.relationship_only)
        .map(|slot| slot.slot_key)
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
