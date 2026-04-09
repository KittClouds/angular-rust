use std::collections::BTreeMap;

use phoenix_semantic_v2::{EntityMemoryCard, RelationshipMemoryLedger};

pub fn count_card_slots(cards: &[EntityMemoryCard]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::<String, usize>::new();
    for card in cards {
        for state in &card.current_state {
            *counts.entry(state.slot_key.clone()).or_default() += 1;
        }
    }
    counts
}

pub fn count_relationship_families(
    ledgers: &[RelationshipMemoryLedger],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::<String, usize>::new();
    for ledger in ledgers {
        *counts.entry(ledger.relation_family.clone()).or_default() += 1;
    }
    counts
}
