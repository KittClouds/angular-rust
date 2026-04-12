use compact_str::CompactString;
use phoenix_types::{
    ClaimId, ClaimRecord, ConceptRecord, EventId, EventRecord, Proposition, SemanticOrder,
    SemanticRelation, StateId, StateRecord, TruthStatus, ValueRecord,
};
use rustc_hash::FxHashMap;

#[derive(Clone, Debug, Default)]
pub struct SemanticBundle {
    pub events: Vec<EventRecord>,
    pub claims: Vec<ClaimRecord>,
    pub states: Vec<StateRecord>,
    pub values: Vec<ValueRecord>,
    pub concepts: Vec<ConceptRecord>,
    pub relations: Vec<SemanticRelation>,
}

pub struct SemanticLowerer;

impl SemanticLowerer {
    pub fn lower(propositions: &[Proposition]) -> SemanticBundle {
        let mut bundle = SemanticBundle::default();
        let mut clause_ords = FxHashMap::<usize, u32>::default();
        for (index, proposition) in propositions.iter().enumerate() {
            let clause_ord = clause_ords
                .entry(proposition.sentence_index)
                .and_modify(|value| *value += 1)
                .or_insert(0);
            let order = SemanticOrder {
                doc_ord: index as u32,
                section_ord: 0,
                sentence_ord: proposition.sentence_index as u32,
                clause_ord: *clause_ord,
                local_ord: index as u32,
            };
            if is_state_proposition(proposition) {
                bundle.states.push(StateRecord {
                    state_id: Some(StateId(format!("state:{}", proposition.proposition_id))),
                    label: proposition.predicate.predicate.clone(),
                    proposition_id: proposition.proposition_id.clone(),
                    order,
                });
            } else if is_event_proposition(proposition) {
                bundle.events.push(EventRecord {
                    event_id: Some(EventId(format!("event:{}", proposition.proposition_id))),
                    label: proposition.predicate.predicate.clone(),
                    proposition_id: proposition.proposition_id.clone(),
                    order,
                });
            } else {
                bundle.claims.push(ClaimRecord {
                    claim_id: Some(ClaimId(format!("claim:{}", proposition.proposition_id))),
                    label: proposition.predicate.predicate.clone(),
                    proposition_id: proposition.proposition_id.clone(),
                    order,
                });
            }

            if proposition.predicate.relation_type.contains("value") {
                bundle.values.push(ValueRecord {
                    value_id: None,
                    label: proposition.predicate.predicate.clone(),
                    proposition_id: proposition.proposition_id.clone(),
                });
            }
            if proposition.predicate.relation_type.contains("concept") {
                bundle.concepts.push(ConceptRecord {
                    concept_id: None,
                    label: proposition.predicate.predicate.clone(),
                    proposition_id: proposition.proposition_id.clone(),
                });
            }
            if proposition.arguments.len() >= 2 {
                bundle.relations.push(SemanticRelation {
                    edge_type: CompactString::from("about"),
                    source_id: CompactString::from(format!(
                        "{}:{}",
                        proposition.proposition_id, proposition.arguments[0].role
                    )),
                    target_id: CompactString::from(format!(
                        "{}:{}",
                        proposition.proposition_id, proposition.arguments[1].role
                    )),
                    status: TruthStatus::Asserted,
                });
            }
        }
        bundle
    }
}

fn is_event_proposition(proposition: &Proposition) -> bool {
    let predicate = normalize_semantic_label(proposition.predicate.predicate.as_str());
    let relation = normalize_semantic_label(proposition.predicate.relation_type.as_str());
    relation.contains("event")
        || relation.contains("action")
        || is_event_relation_family(relation.as_str())
        || is_event_predicate(predicate.as_str())
        || is_generic_relates_to_event(relation.as_str(), predicate.as_str())
}

fn is_state_proposition(proposition: &Proposition) -> bool {
    let predicate = normalize_semantic_label(proposition.predicate.predicate.as_str());
    let relation = normalize_semantic_label(proposition.predicate.relation_type.as_str());
    relation.contains("state")
        || is_state_relation_family(relation.as_str())
        || is_state_predicate(predicate.as_str())
}

fn normalize_semantic_label(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    for ch in value.chars() {
        let next = if ch.is_ascii_alphanumeric() {
            ch.to_ascii_lowercase()
        } else {
            '_'
        };
        if next == '_' && normalized.ends_with('_') {
            continue;
        }
        normalized.push(next);
    }
    normalized.trim_matches('_').to_owned()
}

fn is_generic_relates_to_event(relation: &str, predicate: &str) -> bool {
    relation == "relates_to" && !predicate.is_empty() && !is_state_predicate(predicate)
}

fn is_state_predicate(predicate: &str) -> bool {
    matches!(
        predicate,
        "be" | "is"
            | "are"
            | "was"
            | "were"
            | "remain"
            | "remains"
            | "remained"
            | "become"
            | "becomes"
            | "became"
            | "seem"
            | "seems"
            | "seemed"
            | "feel"
            | "feels"
            | "felt"
            | "have"
            | "has"
            | "had"
            | "own"
            | "owns"
            | "owned"
            | "work"
            | "works"
            | "worked"
            | "live"
            | "lives"
            | "lived"
            | "reside"
            | "resides"
            | "resided"
            | "belong"
            | "belongs"
            | "belonged"
            | "serve"
            | "serves"
            | "served"
            | "lead"
            | "leads"
            | "led"
            | "manage"
            | "manages"
            | "managed"
            | "hold"
            | "holds"
            | "held"
    )
}

fn is_event_predicate(predicate: &str) -> bool {
    matches!(
        predicate,
        "attack"
            | "attacked"
            | "attacks"
            | "fight"
            | "fought"
            | "fights"
            | "meet"
            | "met"
            | "meets"
            | "give"
            | "gave"
            | "gives"
            | "write"
            | "wrote"
            | "writes"
            | "map"
            | "mapped"
            | "maps"
            | "say"
            | "said"
            | "tell"
            | "told"
            | "announce"
            | "announced"
            | "report"
            | "reported"
            | "reports"
            | "join"
            | "joined"
            | "joins"
            | "leave"
            | "left"
            | "leaves"
            | "arrive"
            | "arrived"
            | "arrives"
            | "travel"
            | "traveled"
            | "travels"
            | "move"
            | "moved"
            | "moves"
            | "start"
            | "started"
            | "starts"
            | "begin"
            | "began"
            | "begins"
            | "end"
            | "ended"
            | "ends"
            | "build"
            | "built"
            | "builds"
            | "create"
            | "created"
            | "creates"
            | "destroy"
            | "destroyed"
            | "destroys"
    )
}

fn is_state_relation_family(relation: &str) -> bool {
    matches!(
        relation,
        "located_in"
            | "located_at"
            | "based_in"
            | "lives_in"
            | "resides_in"
            | "works_for"
            | "employed_by"
            | "member_of"
            | "belongs_to"
            | "serves_as"
            | "acts_as"
            | "has_role"
            | "holds_role"
            | "owns"
            | "has"
            | "manages"
            | "leads"
            | "heads"
            | "captains"
    ) || relation.starts_with("state_")
}

fn is_event_relation_family(relation: &str) -> bool {
    matches!(
        relation,
        "attacks"
            | "meets"
            | "gives"
            | "writes"
            | "maps"
            | "joins"
            | "leaves"
            | "travels"
            | "moves"
            | "arrives"
            | "starts"
            | "ends"
            | "reports"
            | "announces"
            | "builds"
            | "creates"
            | "destroys"
    ) || relation.starts_with("event_")
}

#[cfg(test)]
mod tests {
    use super::SemanticLowerer;
    use compact_str::CompactString;
    use phoenix_types::{
        Argument, DocumentId, PredicateFrame, Proposition, ProvenanceRef, ScopeOp, SourceRange,
    };

    fn proposition(predicate: &str, relation_type: &str) -> Proposition {
        Proposition {
            proposition_id: CompactString::from("prop:0"),
            sentence_index: 0,
            predicate: PredicateFrame {
                predicate: CompactString::from(predicate),
                trigger_range: SourceRange::new(0, predicate.len() as u32),
                relation_type: CompactString::from(relation_type),
            },
            clause_range: None,
            arguments: vec![
                Argument {
                    role: CompactString::from("subject"),
                    mention_index: None,
                    entity_id: None,
                    range: None,
                },
                Argument {
                    role: CompactString::from("object"),
                    mention_index: None,
                    entity_id: None,
                    range: None,
                },
            ]
            .into(),
            scope_ops: vec![ScopeOp {
                kind: CompactString::from("assertion"),
                polarity: None,
                modality: None,
            }]
            .into(),
            attribution: None,
            conditional: None,
            quote: None,
            evidence: vec![ProvenanceRef {
                document_id: Some(DocumentId("doc".to_owned())),
                note_id: None,
                label: CompactString::from("test"),
                kind: None,
                range: SourceRange::new(0, 4),
            }]
            .into(),
        }
    }

    #[test]
    fn lower_treats_employment_as_state() {
        let bundle = SemanticLowerer::lower(&[proposition("works", "works_for")]);
        assert_eq!(bundle.states.len(), 1);
        assert!(bundle.events.is_empty());
    }

    #[test]
    fn lower_treats_conflict_as_event() {
        let bundle = SemanticLowerer::lower(&[proposition("attacked", "attacks")]);
        assert_eq!(bundle.events.len(), 1);
        assert!(bundle.states.is_empty());
    }

    #[test]
    fn lower_treats_generic_relates_to_verbs_as_events() {
        let bundle = SemanticLowerer::lower(&[proposition("reinforced", "relates_to")]);
        assert_eq!(bundle.events.len(), 1);
        assert!(bundle.claims.is_empty());
    }

    #[test]
    fn lower_keeps_generic_relates_to_work_predicates_stateful() {
        let bundle = SemanticLowerer::lower(&[proposition("worked", "relates_to")]);
        assert_eq!(bundle.states.len(), 1);
        assert!(bundle.events.is_empty());
    }
}
