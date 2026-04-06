use compact_str::CompactString;
use phoenix_machine::SurfaceCompileArtifacts;
use phoenix_types::{
    Argument, PredicateFrame, Proposition, ProvenanceRef, QuoteFrame, ScopeOp, SourceRange,
};

pub struct PropositionLowerer;

impl PropositionLowerer {
    pub fn lower(artifacts: &SurfaceCompileArtifacts) -> Vec<Proposition> {
        artifacts
            .structure
            .relations
            .iter()
            .enumerate()
            .map(|(index, relation)| Proposition {
                proposition_id: CompactString::from(format!("prop:{index}")),
                sentence_index: relation.sentence_index,
                predicate: PredicateFrame {
                    predicate: CompactString::from(relation.lemma.as_str()),
                    trigger_range: SourceRange::from(relation.verb_range),
                    relation_type: CompactString::from(relation.relation_type.as_str()),
                },
                clause_range: artifacts
                    .structure
                    .sentence_frames
                    .get(relation.sentence_index)
                    .and_then(|frame| frame.clause_ranges.first().copied())
                    .map(SourceRange::from),
                arguments: [
                    relation.subject.as_ref().map(|slot| Argument {
                        role: CompactString::from("subject"),
                        mention_index: None,
                        entity_id: slot.entity_ref.as_ref().and_then(|entity| match entity {
                            phoenix_types::MentionEntityRef::Known(id) => Some(id.clone()),
                            phoenix_types::MentionEntityRef::Speculative(_) => None,
                        }),
                        range: Some(SourceRange::from(slot.range)),
                    }),
                    relation.object.as_ref().map(|slot| Argument {
                        role: CompactString::from("object"),
                        mention_index: None,
                        entity_id: slot.entity_ref.as_ref().and_then(|entity| match entity {
                            phoenix_types::MentionEntityRef::Known(id) => Some(id.clone()),
                            phoenix_types::MentionEntityRef::Speculative(_) => None,
                        }),
                        range: Some(SourceRange::from(slot.range)),
                    }),
                ]
                .into_iter()
                .flatten()
                .collect(),
                scope_ops: [ScopeOp {
                    kind: CompactString::from("assertion"),
                    polarity: None,
                    modality: None,
                }]
                .into_iter()
                .collect(),
                attribution: None,
                conditional: None,
                quote: relation.evidence.first().map(|evidence| QuoteFrame {
                    quote_range: SourceRange::from(evidence.range),
                    speaker_entity_id: None,
                }),
                evidence: relation
                    .evidence
                    .iter()
                    .map(|evidence| ProvenanceRef {
                        document_id: evidence.document_id.clone(),
                        note_id: evidence.note_id.clone(),
                        label: CompactString::from(evidence.label.as_str()),
                        kind: evidence.kind.as_ref().map(|kind| CompactString::from(kind.as_str())),
                        range: SourceRange::from(evidence.range),
                    })
                    .collect(),
            })
            .collect()
    }
}
