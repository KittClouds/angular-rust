use phoenix_alex::Lexicon;
use phoenix_types::{EntityKind, ImplicitMatchHit, ScopeKey};

pub fn match_implicit(text: &str, scope: &ScopeKey, lexicon: &Lexicon) -> Vec<ImplicitMatchHit> {
    lexicon
        .scan(text, scope)
        .into_iter()
        .filter_map(|matched| {
            let mut entries = matched.entries;
            if entries.is_empty() {
                return None;
            }
            entries.sort_by_key(|entry| entity_kind_rank(entry.kind.as_ref()));
            let best = entries[0].clone();
            Some(ImplicitMatchHit {
                range: matched.range,
                surface: matched.surface,
                label: best.label,
                kind: best.kind,
                resolved_entity_id: Some(best.entity_id.clone()),
                candidate_entity_ids: entries
                    .iter()
                    .map(|entry| entry.entity_id.clone())
                    .collect(),
                candidate_labels: entries.iter().map(|entry| entry.label.clone()).collect(),
                confidence: matched.confidence,
            })
        })
        .collect()
}

fn entity_kind_rank(kind: Option<&EntityKind>) -> usize {
    match kind {
        Some(EntityKind::Character) => 0,
        Some(EntityKind::Npc) => 1,
        Some(EntityKind::Location) => 2,
        Some(EntityKind::Faction) => 3,
        Some(EntityKind::Organization) => 4,
        Some(EntityKind::Item) => 5,
        Some(EntityKind::Concept) => 6,
        Some(EntityKind::Event) => 7,
        Some(EntityKind::Other) | None => 8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_alex::Lexicon;
    use phoenix_types::{EntityId, GenderHint, LexiconEntry};

    #[test]
    fn implicit_matching_keeps_multiword_offsets() {
        let lexicon = Lexicon::from_entries(&[LexiconEntry {
            entity_id: EntityId("luffy".to_owned()),
            label: "Monkey D. Luffy".to_owned(),
            aliases: vec!["Luffy".to_owned()],
            kind: Some(EntityKind::Character),
            gender: Some(GenderHint::Male),
            number: None,
            scope: ScopeKey::default(),
        }])
        .expect("lexicon");

        let hits = match_implicit("Monkey D. Luffy arrived.", &ScopeKey::default(), &lexicon);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].range.start, 0);
        assert_eq!(hits[0].range.end, 15);
        assert_eq!(hits[0].label, "Monkey D. Luffy");
    }
}
