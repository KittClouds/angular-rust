use compact_str::CompactString;
use phoenix_automata::{CausalCueDirection, CausalCueFrame, CausalCueMatcher};
use phoenix_machine::SurfaceCompileArtifacts;
use phoenix_semantics::SemanticBundle;
use phoenix_time::TemporalBinding;
use phoenix_types::{
    CausalBundle, CausalCandidate, CausalDiagnostic, CausalEvidenceKind, CausalKind, CausalLink,
    Polarity, Proposition, ProvenanceRef, SemanticNodeRef, SemanticOrder, SourceRange,
    TruthStatus,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

pub struct CausalityRequest<'a> {
    pub text: &'a str,
    pub artifacts: &'a SurfaceCompileArtifacts,
    pub propositions: &'a [Proposition],
    pub semantics: &'a SemanticBundle,
    pub temporal_bindings: &'a [TemporalBinding],
}

pub struct CausalityLowerer;

#[derive(Clone, Debug)]
struct SemanticNodeRecord {
    node: SemanticNodeRef,
    proposition_id: CompactString,
    sentence_index: usize,
    trigger_range: SourceRange,
    order: SemanticOrder,
    attribution_entity_id: Option<phoenix_types::EntityId>,
    quoted: bool,
    negative: bool,
    evidence: SmallVec<[ProvenanceRef; 2]>,
}

impl CausalityLowerer {
    pub fn lower(request: CausalityRequest<'_>) -> CausalBundle {
        let matcher = CausalCueMatcher::default();
        let proposition_nodes = proposition_nodes(request.propositions, request.semantics);
        let mut sentence_nodes = FxHashMap::<usize, Vec<SemanticNodeRecord>>::default();
        for node in proposition_nodes.values() {
            sentence_nodes
                .entry(node.sentence_index)
                .or_default()
                .push(node.clone());
        }
        for nodes in sentence_nodes.values_mut() {
            nodes.sort_by_key(|node| (node.trigger_range.start, node.order.doc_ord));
        }

        let mut bundle = CausalBundle::default();
        for sentence in &request.artifacts.scan.sentences {
            let Some(nodes) = sentence_nodes.get(&sentence.index) else {
                continue;
            };
            let sentence_text = safe_slice(request.text, sentence.range);
            for cue_hit in matcher.find_iter(sentence_text) {
                let global_cue = SourceRange::new(
                    sentence.range.start + cue_hit.range.start,
                    sentence.range.start + cue_hit.range.end,
                );
                let Some((left, right)) = nearest_nodes(nodes, global_cue) else {
                    bundle.diagnostics.push(CausalDiagnostic {
                        code: CompactString::from("PX_CAUSALITY_ARGUMENT_GAP"),
                        message: CompactString::from(format!(
                            "Skipped causal cue '{}' without compatible semantic nodes.",
                            cue_hit.frame.cue
                        )),
                        proposition_id: None,
                        cue_span: Some(global_cue),
                    });
                    continue;
                };
                let (source, target) = orient_nodes(&cue_hit.frame, left, right);
                if source.node == target.node {
                    continue;
                }
                let (status, confidence_millis, polarity, attributed_to) = validate_candidate(
                    &cue_hit.frame,
                    source,
                    target,
                    request.propositions,
                    request.temporal_bindings,
                );
                let candidate = CausalCandidate {
                    source: source.node.clone(),
                    target: target.node.clone(),
                    kind: cue_hit.frame.kind,
                    confidence_millis,
                    status,
                    cue: Some(cue_hit.frame.cue.clone()),
                    cue_span: Some(global_cue),
                    evidence_kind: CausalEvidenceKind::ExplicitCue,
                    attributed_to,
                    polarity,
                    provenance: source
                        .evidence
                        .iter()
                        .chain(target.evidence.iter())
                        .take(2)
                        .cloned()
                        .collect(),
                };
                if candidate.status == TruthStatus::Asserted {
                    bundle.links.push(CausalLink {
                        edge_id: None,
                        source: candidate.source.clone(),
                        target: candidate.target.clone(),
                        kind: candidate.kind,
                        confidence_millis: candidate.confidence_millis,
                        status: candidate.status,
                        cue: candidate.cue.clone(),
                        cue_span: candidate.cue_span,
                        attributed_to: candidate.attributed_to.clone(),
                        polarity: candidate.polarity,
                        provenance: candidate.provenance.clone(),
                    });
                }
                bundle.candidates.push(candidate);
            }
        }
        bundle
    }
}

fn proposition_nodes(
    propositions: &[Proposition],
    semantics: &SemanticBundle,
) -> FxHashMap<CompactString, SemanticNodeRecord> {
    let proposition_by_id = propositions
        .iter()
        .map(|proposition| (proposition.proposition_id.clone(), proposition))
        .collect::<FxHashMap<_, _>>();
    let mut nodes = FxHashMap::default();

    for event in &semantics.events {
        if let (Some(event_id), Some(proposition)) = (
            event.event_id.as_ref(),
            proposition_by_id.get(&event.proposition_id),
        ) {
            nodes.insert(
                event.proposition_id.clone(),
                semantic_node_record(
                    proposition,
                    SemanticNodeRef::Event(event_id.clone()),
                    event.order,
                ),
            );
        }
    }
    for state in &semantics.states {
        if let (Some(state_id), Some(proposition)) = (
            state.state_id.as_ref(),
            proposition_by_id.get(&state.proposition_id),
        ) {
            nodes.insert(
                state.proposition_id.clone(),
                semantic_node_record(
                    proposition,
                    SemanticNodeRef::State(state_id.clone()),
                    state.order,
                ),
            );
        }
    }
    for claim in &semantics.claims {
        if let (Some(claim_id), Some(proposition)) = (
            claim.claim_id.as_ref(),
            proposition_by_id.get(&claim.proposition_id),
        ) {
            nodes.entry(claim.proposition_id.clone()).or_insert_with(|| {
                semantic_node_record(proposition, SemanticNodeRef::Claim(claim_id.clone()), claim.order)
            });
        }
    }
    nodes
}

fn semantic_node_record(
    proposition: &Proposition,
    node: SemanticNodeRef,
    order: SemanticOrder,
) -> SemanticNodeRecord {
    SemanticNodeRecord {
        node,
        proposition_id: proposition.proposition_id.clone(),
        sentence_index: proposition.sentence_index,
        trigger_range: proposition.predicate.trigger_range,
        order,
        attribution_entity_id: proposition
            .attribution
            .as_ref()
            .and_then(|frame| frame.source_entity_id.clone()),
        quoted: proposition.quote.is_some() || proposition.attribution.is_some(),
        negative: proposition
            .scope_ops
            .iter()
            .any(|scope| scope.polarity.as_deref() == Some("negative")),
        evidence: proposition.evidence.iter().take(2).cloned().collect(),
    }
}

fn nearest_nodes<'a>(
    nodes: &'a [SemanticNodeRecord],
    cue_range: SourceRange,
) -> Option<(&'a SemanticNodeRecord, &'a SemanticNodeRecord)> {
    let left = nodes
        .iter()
        .filter(|node| node.trigger_range.start < cue_range.start)
        .max_by_key(|node| node.trigger_range.start);
    let right = nodes
        .iter()
        .filter(|node| node.trigger_range.start >= cue_range.end)
        .min_by_key(|node| node.trigger_range.start);
    left.zip(right)
}

fn orient_nodes<'a>(
    frame: &CausalCueFrame,
    left: &'a SemanticNodeRecord,
    right: &'a SemanticNodeRecord,
) -> (&'a SemanticNodeRecord, &'a SemanticNodeRecord) {
    match frame.direction {
        CausalCueDirection::LeftToRight => (left, right),
        CausalCueDirection::RightToLeft => (right, left),
    }
}

fn validate_candidate(
    frame: &CausalCueFrame,
    source: &SemanticNodeRecord,
    target: &SemanticNodeRecord,
    propositions: &[Proposition],
    temporal_bindings: &[TemporalBinding],
) -> (TruthStatus, u16, Polarity, Option<phoenix_types::EntityId>) {
    let mut confidence = frame.priority;
    let mut status = TruthStatus::Asserted;
    let polarity = if source.negative || target.negative {
        Polarity::Negative
    } else {
        Polarity::Positive
    };
    let attributed_to = source
        .attribution_entity_id
        .clone()
        .or_else(|| target.attribution_entity_id.clone());

    if source.quoted || target.quoted || attributed_to.is_some() {
        confidence = confidence.saturating_sub(180);
        status = TruthStatus::Candidate;
    }
    if temporal_contradiction(source, target, propositions, temporal_bindings) {
        confidence = confidence.saturating_sub(240);
        status = TruthStatus::Candidate;
    }
    if matches!(frame.kind, CausalKind::Explains | CausalKind::Motivates | CausalKind::PurposeOf) {
        confidence = confidence.saturating_sub(60);
    }
    (status, confidence.max(100), polarity, attributed_to)
}

fn temporal_contradiction(
    source: &SemanticNodeRecord,
    target: &SemanticNodeRecord,
    propositions: &[Proposition],
    temporal_bindings: &[TemporalBinding],
) -> bool {
    if temporal_bindings.len() != propositions.len() {
        return false;
    }
    let source_ix = propositions
        .iter()
        .position(|proposition| proposition.proposition_id == source.proposition_id);
    let target_ix = propositions
        .iter()
        .position(|proposition| proposition.proposition_id == target.proposition_id);
    let (Some(source_ix), Some(target_ix)) = (source_ix, target_ix) else {
        return false;
    };
    let source_time = binding_time(&temporal_bindings[source_ix]);
    let target_time = binding_time(&temporal_bindings[target_ix]);
    matches!((source_time, target_time), (Some(source_time), Some(target_time)) if source_time > target_time)
}

fn binding_time(binding: &TemporalBinding) -> Option<i64> {
    binding
        .anchor
        .as_ref()
        .and_then(|anchor| anchor.interval.valid_from.or(anchor.interval.recorded_from))
        .or(binding.recorded_window.valid_from)
        .or(binding.recorded_window.recorded_from)
}

fn safe_slice<'a>(text: &'a str, range: phoenix_types::TextRange) -> &'a str {
    text.get(range.start as usize..range.end as usize)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use compact_str::CompactString;
    use phoenix_machine::SurfaceCompileArtifacts;
    use phoenix_semantics::SemanticBundle;
    use phoenix_types::{
        ClaimId, ClaimRecord, PredicateFrame, Proposition, ScanArtifact, SentenceSpan, SourceRange,
        StructureArtifact, SurfaceDocument,
    };

    fn sentence_artifacts(text: &str) -> SurfaceCompileArtifacts {
        SurfaceCompileArtifacts {
            scan: ScanArtifact {
                sentences: vec![SentenceSpan {
                    index: 0,
                    range: phoenix_types::TextRange {
                        start: 0,
                        end: text.len() as u32,
                    },
                }],
                ..ScanArtifact::default()
            },
            structure: StructureArtifact::default(),
            surface: SurfaceDocument::default(),
        }
    }

    #[test]
    fn because_maps_right_side_to_cause() {
        let text = "Alarm sounded because explosion happened.";
        let artifacts = sentence_artifacts(text);
        let propositions = vec![
            Proposition {
                proposition_id: CompactString::from("prop:0"),
                sentence_index: 0,
                predicate: PredicateFrame {
                    predicate: CompactString::from("sound"),
                    trigger_range: SourceRange::new(6, 13),
                    relation_type: CompactString::from("action"),
                },
                clause_range: None,
                ..Proposition::default()
            },
            Proposition {
                proposition_id: CompactString::from("prop:1"),
                sentence_index: 0,
                predicate: PredicateFrame {
                    predicate: CompactString::from("happen"),
                    trigger_range: SourceRange::new(32, 40),
                    relation_type: CompactString::from("action"),
                },
                clause_range: None,
                ..Proposition::default()
            },
        ];
        let semantics = SemanticBundle {
            claims: vec![
                ClaimRecord {
                    claim_id: Some(ClaimId("claim:prop:0".to_owned())),
                    label: CompactString::from("sound"),
                    proposition_id: CompactString::from("prop:0"),
                    order: SemanticOrder {
                        doc_ord: 0,
                        section_ord: 0,
                        sentence_ord: 0,
                        clause_ord: 0,
                        local_ord: 0,
                    },
                },
                ClaimRecord {
                    claim_id: Some(ClaimId("claim:prop:1".to_owned())),
                    label: CompactString::from("happen"),
                    proposition_id: CompactString::from("prop:1"),
                    order: SemanticOrder {
                        doc_ord: 1,
                        section_ord: 0,
                        sentence_ord: 0,
                        clause_ord: 1,
                        local_ord: 1,
                    },
                },
            ],
            ..SemanticBundle::default()
        };
        let bundle = CausalityLowerer::lower(CausalityRequest {
            text,
            artifacts: &artifacts,
            propositions: &propositions,
            semantics: &semantics,
            temporal_bindings: &[],
        });
        assert_eq!(bundle.candidates.len(), 1);
        assert_eq!(bundle.candidates[0].kind, CausalKind::Causes);
        assert_eq!(
            bundle.candidates[0].source,
            SemanticNodeRef::Claim(ClaimId("claim:prop:1".to_owned()))
        );
        assert_eq!(
            bundle.candidates[0].target,
            SemanticNodeRef::Claim(ClaimId("claim:prop:0".to_owned()))
        );
    }

    #[test]
    fn attributed_causality_stays_candidate() {
        let text = "Alarm sounded because explosion happened.";
        let artifacts = sentence_artifacts(text);
        let mut propositions = vec![
            Proposition {
                proposition_id: CompactString::from("prop:0"),
                sentence_index: 0,
                predicate: PredicateFrame {
                    predicate: CompactString::from("sound"),
                    trigger_range: SourceRange::new(6, 13),
                    relation_type: CompactString::from("action"),
                },
                clause_range: None,
                ..Proposition::default()
            },
            Proposition {
                proposition_id: CompactString::from("prop:1"),
                sentence_index: 0,
                predicate: PredicateFrame {
                    predicate: CompactString::from("happen"),
                    trigger_range: SourceRange::new(32, 40),
                    relation_type: CompactString::from("action"),
                },
                clause_range: None,
                attribution: Some(phoenix_types::AttributionFrame {
                    source_entity_id: Some("entity:narrator".into()),
                    quote_range: None,
                }),
                ..Proposition::default()
            },
        ];
        let semantics = SemanticBundle {
            claims: vec![
                ClaimRecord {
                    claim_id: Some(ClaimId("claim:prop:0".to_owned())),
                    label: CompactString::from("sound"),
                    proposition_id: CompactString::from("prop:0"),
                    order: SemanticOrder::default(),
                },
                ClaimRecord {
                    claim_id: Some(ClaimId("claim:prop:1".to_owned())),
                    label: CompactString::from("happen"),
                    proposition_id: CompactString::from("prop:1"),
                    order: SemanticOrder {
                        doc_ord: 1,
                        ..SemanticOrder::default()
                    },
                },
            ],
            ..SemanticBundle::default()
        };
        let bundle = CausalityLowerer::lower(CausalityRequest {
            text,
            artifacts: &artifacts,
            propositions: &propositions,
            semantics: &semantics,
            temporal_bindings: &[],
        });
        assert_eq!(bundle.candidates.len(), 1);
        assert!(bundle.links.is_empty());
        assert_eq!(bundle.candidates[0].status, TruthStatus::Candidate);
        propositions.clear();
    }
}
