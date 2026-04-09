use phoenix_semantic_v2::{
    DocumentArchive, DocumentManifest, ErEntityLinkOverride, ErScopePatchSidecar, MentionId,
    ResolutionDecision, ResolvedMention, ScopeOrd,
};
use phoenix_types::{
    BiTemporalWindow, CausalCandidate, CausalKind, CausalLink, ClaimId, ClaimRecord, EdgeId,
    EventId, EventRecord, Proposition, ScopeKey, SemanticNodeRef, SemanticOrder, TextRange,
    TruthStatus,
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
        arguments: smallvec::smallvec![phoenix_types::Argument {
            role: "actor".into(),
            mention_index: None,
            entity_id: Some("bomb".into()),
            range: None,
        },],
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
        arguments: smallvec::smallvec![phoenix_types::Argument {
            role: "patient".into(),
            mention_index: None,
            entity_id: Some("tower".into()),
            range: None,
        },],
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
    assert_eq!(
        batch.summary.committed_edge_count,
        batch.edge_additions.len()
    );
    assert_eq!(
        batch.summary.accepted_edge_count + batch.summary.supported_edge_count,
        batch.edge_additions.len()
    );
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
            canonical_cause_event_id: None,
            target: SemanticNodeRef::Event(EventId("event:2".to_owned())),
            canonical_effect_event_id: None,
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
            quoted_evidence: true,
            attributed_evidence: false,
            quoted_or_attributed: true,
            source_semantics: crate::normalize::CausalSourceSemantics::ReportedSpeech,
            modality_semantics: crate::normalize::CausalModalitySemantics::Asserted,
            shared_participant_count: 0,
            source_degree: 0,
            target_degree: 0,
            graph_support_count: 0,
            centrality_millis: 0,
            evidence_refs: vec!["sentence:0".to_owned()],
            seed_source: "candidate".to_owned(),
        }],
        claim_atoms: Vec::new(),
        shadow_local_pair_cases: Vec::new(),
        shadow_local_pair_claim_atoms: Vec::new(),
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

#[test]
fn quoted_support_without_world_corroboration_stays_deferred() {
    let mut batch = crate::CausalScopeReviewBatch {
        scope: ScopeKey::default(),
        scope_key: String::new(),
        scope_ord: phoenix_semantic_v2::ScopeOrd::default(),
        session_id: None,
        dirty: None,
        document_refs: Vec::new(),
        event_profiles: Vec::new(),
        review_cases: vec![crate::normalize::CausalReviewCase {
            case_id: "case:reported".to_owned(),
            document_id: "doc-1".to_owned(),
            revision: 1,
            source: SemanticNodeRef::Event(EventId("event:1".to_owned())),
            canonical_cause_event_id: None,
            target: SemanticNodeRef::Event(EventId("event:2".to_owned())),
            canonical_effect_event_id: None,
            kind: CausalKind::Causes,
            relation_kind: phoenix_semantic_v2::CausalRelationKind::DirectCause,
            base_confidence_millis: 760,
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
            target_sentence_index: 0,
            sentence_distance: 0,
            temporal_legal: true,
            quoted_evidence: true,
            attributed_evidence: false,
            quoted_or_attributed: true,
            source_semantics: crate::normalize::CausalSourceSemantics::ReportedSpeech,
            modality_semantics: crate::normalize::CausalModalitySemantics::Asserted,
            shared_participant_count: 1,
            source_degree: 1,
            target_degree: 1,
            graph_support_count: 1,
            centrality_millis: 220,
            evidence_refs: vec!["sentence:0".to_owned()],
            seed_source: "candidate".to_owned(),
        }],
        claim_atoms: vec![phoenix_semantic_v2::CausalClaimAtom {
            claim_id: phoenix_semantic_v2::CausalClaimId("claim:test".to_owned()),
            edge_id: phoenix_semantic_v2::CausalEdgeId(
                "edge:doc-1:event:event:1:event:event:2:DirectCause".to_owned(),
            ),
            document_id: "doc-1".to_owned(),
            cause_event: SemanticNodeRef::Event(EventId("event:1".to_owned())),
            canonical_cause_event_id: None,
            effect_event: SemanticNodeRef::Event(EventId("event:2".to_owned())),
            canonical_effect_event_id: None,
            kind: CausalKind::Causes,
            relation_kind: phoenix_semantic_v2::CausalRelationKind::DirectCause,
            source_kind: phoenix_semantic_v2::CausalClaimSourceKind::CandidateCue,
            polarity: phoenix_semantic_v2::CausalClaimPolarity::Support,
            evidence_class: phoenix_semantic_v2::CausalEvidenceClass::ReportedSupport,
            strength_millis: 760,
            temporal: BiTemporalWindow {
                valid_from: Some(100),
                valid_to: None,
                recorded_from: Some(100),
                recorded_to: None,
            },
            evidence_refs: vec!["sentence:0".to_owned()],
            created_at: 100,
        }],
        shadow_local_pair_cases: Vec::new(),
        shadow_local_pair_claim_atoms: Vec::new(),
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
    assert_eq!(batch.summary.accepted_edge_count, 0);
    assert_eq!(batch.summary.supported_edge_count, 0);
    assert_eq!(batch.summary.deferred_edge_count, 1);
}

#[test]
fn quoted_candidate_with_world_corroboration_can_commit() {
    let edge_id = phoenix_semantic_v2::CausalEdgeId(
        "edge:doc-1:event:event:1:event:event:2:DirectCause".to_owned(),
    );
    let mut batch = crate::CausalScopeReviewBatch {
        scope: ScopeKey::default(),
        scope_key: String::new(),
        scope_ord: phoenix_semantic_v2::ScopeOrd::default(),
        session_id: None,
        dirty: None,
        document_refs: Vec::new(),
        event_profiles: Vec::new(),
        review_cases: vec![
            crate::normalize::CausalReviewCase {
                case_id: "case:world".to_owned(),
                document_id: "doc-1".to_owned(),
                revision: 1,
                source: SemanticNodeRef::Event(EventId("event:1".to_owned())),
                canonical_cause_event_id: None,
                target: SemanticNodeRef::Event(EventId("event:2".to_owned())),
                canonical_effect_event_id: None,
                kind: CausalKind::Causes,
                relation_kind: phoenix_semantic_v2::CausalRelationKind::DirectCause,
                base_confidence_millis: 780,
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
                target_sentence_index: 0,
                sentence_distance: 0,
                temporal_legal: true,
                quoted_evidence: false,
                attributed_evidence: false,
                quoted_or_attributed: false,
                source_semantics: crate::normalize::CausalSourceSemantics::WorldAssertion,
                modality_semantics: crate::normalize::CausalModalitySemantics::Asserted,
                shared_participant_count: 1,
                source_degree: 1,
                target_degree: 1,
                graph_support_count: 1,
                centrality_millis: 220,
                evidence_refs: vec!["sentence:0".to_owned()],
                seed_source: "link".to_owned(),
            },
            crate::normalize::CausalReviewCase {
                case_id: "case:reported".to_owned(),
                document_id: "doc-1".to_owned(),
                revision: 1,
                source: SemanticNodeRef::Event(EventId("event:1".to_owned())),
                canonical_cause_event_id: None,
                target: SemanticNodeRef::Event(EventId("event:2".to_owned())),
                canonical_effect_event_id: None,
                kind: CausalKind::Causes,
                relation_kind: phoenix_semantic_v2::CausalRelationKind::DirectCause,
                base_confidence_millis: 760,
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
                target_sentence_index: 0,
                sentence_distance: 0,
                temporal_legal: true,
                quoted_evidence: true,
                attributed_evidence: false,
                quoted_or_attributed: true,
                source_semantics: crate::normalize::CausalSourceSemantics::ReportedSpeech,
                modality_semantics: crate::normalize::CausalModalitySemantics::Asserted,
                shared_participant_count: 1,
                source_degree: 1,
                target_degree: 1,
                graph_support_count: 1,
                centrality_millis: 220,
                evidence_refs: vec!["sentence:0".to_owned()],
                seed_source: "candidate".to_owned(),
            },
        ],
        claim_atoms: vec![
            phoenix_semantic_v2::CausalClaimAtom {
                claim_id: phoenix_semantic_v2::CausalClaimId("claim:world".to_owned()),
                edge_id: edge_id.clone(),
                document_id: "doc-1".to_owned(),
                cause_event: SemanticNodeRef::Event(EventId("event:1".to_owned())),
                canonical_cause_event_id: None,
                effect_event: SemanticNodeRef::Event(EventId("event:2".to_owned())),
                canonical_effect_event_id: None,
                kind: CausalKind::Causes,
                relation_kind: phoenix_semantic_v2::CausalRelationKind::DirectCause,
                source_kind: phoenix_semantic_v2::CausalClaimSourceKind::ExplicitLink,
                polarity: phoenix_semantic_v2::CausalClaimPolarity::Support,
                evidence_class: phoenix_semantic_v2::CausalEvidenceClass::WorldSupport,
                strength_millis: 780,
                temporal: BiTemporalWindow {
                    valid_from: Some(100),
                    valid_to: None,
                    recorded_from: Some(100),
                    recorded_to: None,
                },
                evidence_refs: vec!["sentence:0".to_owned()],
                created_at: 100,
            },
            phoenix_semantic_v2::CausalClaimAtom {
                claim_id: phoenix_semantic_v2::CausalClaimId("claim:reported".to_owned()),
                edge_id,
                document_id: "doc-1".to_owned(),
                cause_event: SemanticNodeRef::Event(EventId("event:1".to_owned())),
                canonical_cause_event_id: None,
                effect_event: SemanticNodeRef::Event(EventId("event:2".to_owned())),
                canonical_effect_event_id: None,
                kind: CausalKind::Causes,
                relation_kind: phoenix_semantic_v2::CausalRelationKind::DirectCause,
                source_kind: phoenix_semantic_v2::CausalClaimSourceKind::CandidateCue,
                polarity: phoenix_semantic_v2::CausalClaimPolarity::Support,
                evidence_class: phoenix_semantic_v2::CausalEvidenceClass::ReportedSupport,
                strength_millis: 760,
                temporal: BiTemporalWindow {
                    valid_from: Some(100),
                    valid_to: None,
                    recorded_from: Some(100),
                    recorded_to: None,
                },
                evidence_refs: vec!["sentence:0".to_owned()],
                created_at: 100,
            },
        ],
        shadow_local_pair_cases: Vec::new(),
        shadow_local_pair_claim_atoms: Vec::new(),
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
    assert_eq!(batch.edge_additions.len(), 1);
    assert_eq!(batch.summary.committed_edge_count, 1);
    assert_eq!(batch.summary.deferred_edge_count, 0);
}

#[test]
fn temporal_illegality_cannot_be_recovered_by_score() {
    let mut batch = crate::CausalScopeReviewBatch {
        scope: ScopeKey::default(),
        scope_key: String::new(),
        scope_ord: phoenix_semantic_v2::ScopeOrd::default(),
        session_id: None,
        dirty: None,
        document_refs: Vec::new(),
        event_profiles: Vec::new(),
        review_cases: vec![crate::normalize::CausalReviewCase {
            case_id: "case:illegal".to_owned(),
            document_id: "doc-1".to_owned(),
            revision: 1,
            source: SemanticNodeRef::Event(EventId("event:2".to_owned())),
            canonical_cause_event_id: None,
            target: SemanticNodeRef::Event(EventId("event:1".to_owned())),
            canonical_effect_event_id: None,
            kind: CausalKind::Causes,
            relation_kind: phoenix_semantic_v2::CausalRelationKind::DirectCause,
            base_confidence_millis: 900,
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
            target_sentence_index: 0,
            sentence_distance: 0,
            temporal_legal: false,
            quoted_evidence: false,
            attributed_evidence: false,
            quoted_or_attributed: false,
            source_semantics: crate::normalize::CausalSourceSemantics::WorldAssertion,
            modality_semantics: crate::normalize::CausalModalitySemantics::Asserted,
            shared_participant_count: 2,
            source_degree: 2,
            target_degree: 2,
            graph_support_count: 2,
            centrality_millis: 300,
            evidence_refs: vec!["sentence:0".to_owned()],
            seed_source: "candidate".to_owned(),
        }],
        claim_atoms: Vec::new(),
        shadow_local_pair_cases: Vec::new(),
        shadow_local_pair_claim_atoms: Vec::new(),
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

    assert!(batch.edge_additions.is_empty());
    assert_eq!(batch.summary.rejected_edge_count, 1);
}

#[test]
fn repeated_runs_are_stable_for_same_input() {
    let archive = sample_archive();
    let mut left = derive_scope_review_batch(std::slice::from_ref(&archive), None, None, None);
    let mut right = derive_scope_review_batch(&[archive], None, None, None);
    run_causal_scope(&mut left, 200);
    run_causal_scope(&mut right, 200);

    assert_eq!(left.edge_records, right.edge_records);
    assert_eq!(left.edge_additions, right.edge_additions);
    assert_eq!(left.summary, right.summary);
    assert_eq!(left.diagnostics, right.diagnostics);
}

#[test]
fn participant_grounding_rejects_document_spillover_links() {
    let proposition = Proposition {
        proposition_id: "prop:1".into(),
        sentence_index: 0,
        predicate: phoenix_types::PredicateFrame {
            predicate: "explode".into(),
            trigger_range: phoenix_types::SourceRange::new(0, 8),
            relation_type: "action".into(),
        },
        clause_range: Some(phoenix_types::SourceRange::new(0, 20)),
        arguments: smallvec::smallvec![phoenix_types::Argument {
            role: "actor".into(),
            mention_index: Some(0),
            entity_id: Some("bomb".into()),
            range: Some(phoenix_types::SourceRange::new(0, 4)),
        }],
        ..Proposition::default()
    };
    let archive = DocumentArchive {
        manifest: DocumentManifest {
            document_id: "doc-1".to_owned(),
            scope: ScopeKey::default(),
            scope_key: String::new(),
            scope_ord: ScopeOrd(1),
            created_at: 100,
            ..DocumentManifest::default()
        },
        mentions: vec![
            phoenix_types::MentionSpan {
                range: TextRange { start: 0, end: 4 },
                surface: "bomb".to_owned(),
                sentence_index: 0,
                ..phoenix_types::MentionSpan::default()
            },
            phoenix_types::MentionSpan {
                range: TextRange { start: 60, end: 66 },
                surface: "ghost".to_owned(),
                sentence_index: 3,
                ..phoenix_types::MentionSpan::default()
            },
        ],
        resolved_mentions: vec![
            ResolvedMention {
                mention_id: MentionId("mention:local".to_owned()),
                mention_index: 0,
                range: TextRange { start: 0, end: 4 },
                surface: "bomb".to_owned(),
                normalized: "bomb".to_owned(),
                entity_id: Some("bomb".into()),
                decision: ResolutionDecision::default(),
                ..ResolvedMention::default()
            },
            ResolvedMention {
                mention_id: MentionId("mention:far".to_owned()),
                mention_index: 1,
                range: TextRange { start: 60, end: 66 },
                surface: "ghost".to_owned(),
                normalized: "ghost".to_owned(),
                entity_id: Some("ghost".into()),
                decision: ResolutionDecision::default(),
                ..ResolvedMention::default()
            },
        ],
        causal_substrate: Some(phoenix_semantic_v2::DocumentCausalSubstrate {
            propositions: vec![proposition.clone()],
            semantic_events: vec![EventRecord {
                event_id: Some(EventId("event:1".to_owned())),
                label: "explode".into(),
                proposition_id: proposition.proposition_id.clone(),
                order: SemanticOrder::default(),
            }],
            ..Default::default()
        }),
        ..DocumentArchive::default()
    };
    let er_sidecar = ErScopePatchSidecar {
        entity_links: vec![
            ErEntityLinkOverride {
                case_id: "case:local".to_owned(),
                document_id: "doc-1".to_owned(),
                mention_id: Some(MentionId("mention:local".to_owned())),
                entity_id: "bomb-link".into(),
                confidence_millis: 900,
                created_at: 100,
            },
            ErEntityLinkOverride {
                case_id: "case:far".to_owned(),
                document_id: "doc-1".to_owned(),
                mention_id: Some(MentionId("mention:far".to_owned())),
                entity_id: "ghost".into(),
                confidence_millis: 900,
                created_at: 100,
            },
        ],
        ..ErScopePatchSidecar::default()
    };

    let batch = derive_scope_review_batch(&[archive], None, None, Some(&er_sidecar));
    let profile = batch
        .event_profiles
        .iter()
        .find(|profile| matches!(profile.node, SemanticNodeRef::Event(_)))
        .expect("event profile");

    assert!(profile
        .participant_entity_ids
        .iter()
        .any(|value| value.0 == "bomb"));
    assert!(profile
        .participant_entity_ids
        .iter()
        .any(|value| value.0 == "bomb-link"));
    assert!(!profile
        .participant_entity_ids
        .iter()
        .any(|value| value.0 == "ghost"));
    assert_eq!(
        batch.diagnostics.get("participant_source:er_local_overlap"),
        Some(&1usize)
    );
    assert_eq!(
        batch
            .diagnostics
            .get("participant_source:er_document_spill_rejected"),
        Some(&1usize)
    );
}
