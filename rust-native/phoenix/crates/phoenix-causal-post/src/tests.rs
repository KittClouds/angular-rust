use phoenix_semantic_v2::{DocumentArchive, DocumentManifest, ScopeOrd};
use phoenix_types::{
    BiTemporalWindow, CausalCandidate, CausalKind, CausalLink, ClaimId, ClaimRecord, EdgeId,
    EventId, EventRecord, Proposition, ScopeKey, SemanticNodeRef, SemanticOrder, TruthStatus,
};

use crate::{apply_causal_patch_sidecar, derive_scope_review_batch, run_causal_scope};

fn sample_archive() -> DocumentArchive {
    let proposition_one = Proposition {
        proposition_id: "prop:1".into(),
        sentence_index: 0,
        predicate: phoenix_types::PredicateFrame {
            predicate: "explode".into(),
            trigger_range: phoenix_types::SourceRange::new(0, 8),
            relation_type: "action".into(),
        },
        arguments: smallvec::smallvec![
            phoenix_types::Argument {
                role: "actor".into(),
                mention_index: None,
                entity_id: Some("bomb".into()),
                range: None,
            },
        ],
        ..Proposition::default()
    };
    let proposition_two = Proposition {
        proposition_id: "prop:2".into(),
        sentence_index: 0,
        predicate: phoenix_types::PredicateFrame {
            predicate: "collapse".into(),
            trigger_range: phoenix_types::SourceRange::new(12, 20),
            relation_type: "action".into(),
        },
        arguments: smallvec::smallvec![
            phoenix_types::Argument {
                role: "patient".into(),
                mention_index: None,
                entity_id: Some("tower".into()),
                range: None,
            },
        ],
        ..Proposition::default()
    };

    DocumentArchive {
        manifest: DocumentManifest {
            document_id: "doc-1".to_owned(),
            scope: ScopeKey::default(),
            scope_key: String::new(),
            scope_ord: ScopeOrd(1),
            created_at: 100,
            ..DocumentManifest::default()
        },
        causal_substrate: Some(phoenix_semantic_v2::DocumentCausalSubstrate {
            propositions: vec![proposition_one.clone(), proposition_two.clone()],
            semantic_events: vec![
                EventRecord {
                    event_id: Some(EventId("event:1".to_owned())),
                    label: "explode".into(),
                    proposition_id: proposition_one.proposition_id.clone(),
                    order: SemanticOrder::default(),
                },
                EventRecord {
                    event_id: Some(EventId("event:2".to_owned())),
                    label: "collapse".into(),
                    proposition_id: proposition_two.proposition_id.clone(),
                    order: SemanticOrder::default(),
                },
            ],
            semantic_claims: vec![ClaimRecord {
                claim_id: Some(ClaimId("claim:1".to_owned())),
                label: "witnessed".into(),
                proposition_id: proposition_one.proposition_id.clone(),
                order: SemanticOrder::default(),
            }],
            temporal_bindings: vec![
                phoenix_semantic_v2::RecordedTemporalBinding {
                    anchor: Some(phoenix_types::TimeAnchorRecord {
                        time_id: None,
                        label: "t1".into(),
                        interval: BiTemporalWindow {
                            valid_from: Some(100),
                            valid_to: None,
                            recorded_from: Some(100),
                            recorded_to: None,
                        },
                    }),
                    recorded_window: BiTemporalWindow {
                        valid_from: Some(100),
                        valid_to: None,
                        recorded_from: Some(100),
                        recorded_to: None,
                    },
                },
                phoenix_semantic_v2::RecordedTemporalBinding {
                    anchor: Some(phoenix_types::TimeAnchorRecord {
                        time_id: None,
                        label: "t2".into(),
                        interval: BiTemporalWindow {
                            valid_from: Some(110),
                            valid_to: None,
                            recorded_from: Some(110),
                            recorded_to: None,
                        },
                    }),
                    recorded_window: BiTemporalWindow {
                        valid_from: Some(110),
                        valid_to: None,
                        recorded_from: Some(110),
                        recorded_to: None,
                    },
                },
            ],
            causal_candidates: vec![CausalCandidate {
                source: SemanticNodeRef::Event(EventId("event:1".to_owned())),
                target: SemanticNodeRef::Event(EventId("event:2".to_owned())),
                kind: CausalKind::Causes,
                confidence_millis: 760,
                status: TruthStatus::Asserted,
                cue: Some("because".into()),
                cue_span: None,
                evidence_kind: phoenix_types::CausalEvidenceKind::ExplicitCue,
                attributed_to: None,
                polarity: phoenix_types::Polarity::Positive,
                provenance: Default::default(),
            }],
            causal_links: vec![CausalLink {
                edge_id: Some(EdgeId("edge:1".to_owned())),
                source: SemanticNodeRef::Event(EventId("event:1".to_owned())),
                target: SemanticNodeRef::Event(EventId("event:2".to_owned())),
                kind: CausalKind::Causes,
                confidence_millis: 810,
                status: TruthStatus::Asserted,
                cue: Some("because".into()),
                cue_span: None,
                attributed_to: None,
                polarity: phoenix_types::Polarity::Positive,
                provenance: Default::default(),
            }],
            ..Default::default()
        }),
        ..DocumentArchive::default()
    }
}

#[test]
fn compiles_causal_scope_from_substrate() {
    let archive = sample_archive();
    let mut batch = derive_scope_review_batch(&[archive], None, None, None);
    run_causal_scope(&mut batch, 200);

    assert!(!batch.event_profiles.is_empty());
    assert!(!batch.review_cases.is_empty());
    assert!(!batch.edge_records.is_empty());
    assert!(!batch.edge_additions.is_empty());
    assert!(!batch.memory_cards.is_empty());
    assert_eq!(batch.summary.edge_record_count, batch.edge_records.len());
    assert_eq!(batch.summary.committed_edge_count, batch.edge_additions.len());
    assert_eq!(batch.summary.accepted_edge_count, batch.edge_additions.len());
}

#[test]
fn replay_replaces_causal_outputs_idempotently() {
    let archive = sample_archive();
    let mut batch = derive_scope_review_batch(&[archive], None, None, None);
    run_causal_scope(&mut batch, 200);
    let sidecar = crate::build_causal_patch_sidecar(&batch, 200);

    let mut replayed = derive_scope_review_batch(&[], None, None, None);
    apply_causal_patch_sidecar(&mut replayed, &sidecar);
    apply_causal_patch_sidecar(&mut replayed, &sidecar);

    assert_eq!(replayed.edge_records, sidecar.edge_records);
    assert_eq!(replayed.edge_additions, sidecar.edge_additions);
    assert_eq!(replayed.summary, sidecar.summary);
}

#[test]
fn deferred_reviews_stay_out_of_committed_edges() {
    let mut batch = crate::CausalScopeReviewBatch {
        scope: ScopeKey::default(),
        scope_key: String::new(),
        scope_ord: phoenix_semantic_v2::ScopeOrd::default(),
        session_id: None,
        dirty: None,
        document_refs: Vec::new(),
        event_profiles: Vec::new(),
        review_cases: vec![crate::normalize::CausalReviewCase {
            case_id: "case:1".to_owned(),
            document_id: "doc-1".to_owned(),
            revision: 1,
            source: SemanticNodeRef::Event(EventId("event:1".to_owned())),
            target: SemanticNodeRef::Event(EventId("event:2".to_owned())),
            kind: CausalKind::ResultsIn,
            relation_kind: phoenix_semantic_v2::CausalRelationKind::DirectCause,
            base_confidence_millis: 560,
            base_status: TruthStatus::Asserted,
            cue: Some("because".to_owned()),
            polarity: phoenix_types::Polarity::Positive,
            attributed_to: None,
            temporal: BiTemporalWindow {
                valid_from: Some(100),
                valid_to: None,
                recorded_from: Some(100),
                recorded_to: None,
            },
            source_sentence_index: 0,
            target_sentence_index: 1,
            sentence_distance: 1,
            temporal_legal: true,
            quoted_or_attributed: false,
            shared_participant_count: 0,
            source_degree: 0,
            target_degree: 0,
            graph_support_count: 0,
            centrality_millis: 0,
            evidence_refs: vec!["sentence:0".to_owned()],
            seed_source: "candidate".to_owned(),
        }],
        claim_atoms: Vec::new(),
        decisions: Vec::new(),
        edge_records: Vec::new(),
        edge_additions: Vec::new(),
        decision_records: Vec::new(),
        decision_history: Vec::new(),
        invalidations: Vec::new(),
        edge_aliases: Vec::new(),
        review_queue: Vec::new(),
        chains: Vec::new(),
        counterfactual_reviews: Vec::new(),
        memory_cards: Vec::new(),
        metrics_snapshot: phoenix_semantic_v2::CausalMetricsSnapshot::default(),
        er_generation: None,
        causal_generation: None,
        summary: phoenix_semantic_v2::CausalCompilerSummary::default(),
        diagnostics: std::collections::BTreeMap::new(),
    };

    run_causal_scope(&mut batch, 200);

    assert_eq!(batch.edge_records.len(), 1);
    assert!(batch.edge_additions.is_empty());
    assert_eq!(batch.summary.edge_record_count, 1);
    assert_eq!(batch.summary.committed_edge_count, 0);
    assert_eq!(batch.summary.deferred_edge_count, 1);
}
