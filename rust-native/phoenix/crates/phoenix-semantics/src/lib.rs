use compact_str::CompactString;
use phoenix_types::{
    ClaimRecord, ConceptRecord, EventId, EventRecord, Proposition, SemanticOrder, SemanticRelation,
    StateId, StateRecord, TruthStatus, ValueRecord, ClaimId,
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
                        proposition.proposition_id,
                        proposition.arguments[0].role
                    )),
                    target_id: CompactString::from(format!(
                        "{}:{}",
                        proposition.proposition_id,
                        proposition.arguments[1].role
                    )),
                    status: TruthStatus::Asserted,
                });
            }
        }
        bundle
    }
}

fn is_event_proposition(proposition: &Proposition) -> bool {
    proposition.predicate.relation_type.contains("event")
        || proposition.predicate.relation_type.contains("action")
}

fn is_state_proposition(proposition: &Proposition) -> bool {
    matches!(
        proposition.predicate.predicate.as_str(),
        "be" | "is" | "are" | "was" | "were" | "remain" | "remains" | "remained" | "become"
            | "becomes" | "became" | "seem" | "seems" | "seemed" | "feel" | "feels"
            | "felt"
    ) || proposition.predicate.relation_type.contains("state")
}
