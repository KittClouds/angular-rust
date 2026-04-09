use phoenix_semantic_v2::{
    CanonicalEventId, DocumentArchive, DocumentManifest, EventIdentityMembershipId,
    EventIdentityMembershipRecord, EventIdentityScopeSidecar, EventIdentityState, EventMentionId,
    EventMentionPacket, EventModalitySemantics, EventSourceSemantics, ScopeOrd, TemporalAnchorId,
    TemporalAnchorRecord, TemporalAxisId, TemporalAxisKind, TemporalAxisRecord, TemporalClaimAtom,
    TemporalConstraintId, TemporalConstraintKind, TemporalConstraintRecord, TemporalReferenceEdge,
    TemporalTimexId, TemporalTimexRecord,
};
use phoenix_types::{
    BiTemporalWindow, EventId, EventRecord, PredicateFrame, Proposition, ScopeKey, SemanticOrder,
    SourceRange,
};

use crate::{
    apply_temporal_patch_sidecar, derive_scope_review_batch, run_temporal_scope,
    worker::annotate_temporal_batch_with_event_identity,
};

fn sample_archive() -> DocumentArchive {
    let proposition_one = Proposition {
        proposition_id: "prop:1".into(),
        sentence_index: 0,
        predicate: PredicateFrame {
            predicate: "arrive today".into(),
            trigger_range: SourceRange::new(0, 12),
            relation_type: "action".into(),
        },
        ..Proposition::default()
    };
    let proposition_two = Proposition {
        proposition_id: "prop:2".into(),
        sentence_index: 1,
        predicate: PredicateFrame {
            predicate: "report".into(),
            trigger_range: SourceRange::new(13, 19),
            relation_type: "action".into(),
        },
        quote: Some(phoenix_types::QuoteFrame {
            quote_range: SourceRange::new(13, 24),
            speaker_entity_id: None,
        }),
        ..Proposition::default()
    };
    let world_axis = TemporalAxisRecord {
        axis_id: TemporalAxisId("axis:world".to_owned()),
        document_id: "doc-1".to_owned(),
        kind: TemporalAxisKind::World,
        label: "world".to_owned(),
        evidence_refs: vec!["document_created_at".to_owned()],
    };
    let reported_axis = TemporalAxisRecord {
        axis_id: TemporalAxisId("axis:reported".to_owned()),
        document_id: "doc-1".to_owned(),
        kind: TemporalAxisKind::Reported,
        label: "reported".to_owned(),
        evidence_refs: vec!["quoted_context".to_owned()],
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
        temporal_substrate: Some(phoenix_semantic_v2::DocumentTemporalSubstrate {
            propositions: vec![proposition_one.clone(), proposition_two.clone()],
            semantic_events: vec![
                EventRecord {
                    event_id: Some(EventId("event:1".to_owned())),
                    label: "arrive".into(),
                    proposition_id: proposition_one.proposition_id.clone(),
                    order: SemanticOrder::default(),
                },
                EventRecord {
                    event_id: Some(EventId("event:2".to_owned())),
                    label: "report".into(),
                    proposition_id: proposition_two.proposition_id.clone(),
                    order: SemanticOrder::default(),
                },
            ],
            axis_records: vec![world_axis, reported_axis],
            timex_records: vec![
                TemporalTimexRecord {
                    timex_id: TemporalTimexId("timex:doc-1:dct".to_owned()),
                    document_id: "doc-1".to_owned(),
                    proposition_id: None,
                    sentence_index: 0,
                    label: "document_created_at".to_owned(),
                    normalized_value: Some("100".to_owned()),
                    range: None,
                    axis_id: TemporalAxisId("axis:world".to_owned()),
                    temporal: temporal_window(Some(100)),
                    confidence_millis: 1000,
                    source_class: "document_created_at".to_owned(),
                    evidence_refs: vec!["manifest.created_at".to_owned()],
                },
                TemporalTimexRecord {
                    timex_id: TemporalTimexId("timex:doc-1:today".to_owned()),
                    document_id: "doc-1".to_owned(),
                    proposition_id: Some("prop:1".to_owned()),
                    sentence_index: 0,
                    label: "today".to_owned(),
                    normalized_value: Some("today".to_owned()),
                    range: None,
                    axis_id: TemporalAxisId("axis:world".to_owned()),
                    temporal: temporal_window(Some(100)),
                    confidence_millis: 900,
                    source_class: "deictic_today".to_owned(),
                    evidence_refs: vec!["prop:1".to_owned()],
                },
            ],
            anchor_candidates: vec![
                TemporalAnchorRecord {
                    anchor_id: TemporalAnchorId("anchor:event:1".to_owned()),
                    document_id: "doc-1".to_owned(),
                    proposition_id: Some("prop:1".to_owned()),
                    event_id: Some("event:1".to_owned()),
                    canonical_event_id: None,
                    timex_id: Some(TemporalTimexId("timex:doc-1:today".to_owned())),
                    reference_event_id: None,
                    canonical_reference_event_id: None,
                    axis_id: TemporalAxisId("axis:world".to_owned()),
                    label: "explicit_timex".to_owned(),
                    anchor_kind: "explicit_timex".to_owned(),
                    temporal: temporal_window(Some(100)),
                    confidence_millis: 900,
                    source_class: "deictic_today".to_owned(),
                    evidence_refs: vec!["today".to_owned()],
                },
                TemporalAnchorRecord {
                    anchor_id: TemporalAnchorId("anchor:event:2".to_owned()),
                    document_id: "doc-1".to_owned(),
                    proposition_id: Some("prop:2".to_owned()),
                    event_id: Some("event:2".to_owned()),
                    canonical_event_id: None,
                    timex_id: Some(TemporalTimexId("timex:doc-1:dct".to_owned())),
                    reference_event_id: None,
                    canonical_reference_event_id: None,
                    axis_id: TemporalAxisId("axis:reported".to_owned()),
                    label: "document_created_at".to_owned(),
                    anchor_kind: "document_created_at".to_owned(),
                    temporal: temporal_window(Some(100)),
                    confidence_millis: 520,
                    source_class: "document_created_at".to_owned(),
                    evidence_refs: vec!["manifest.created_at".to_owned()],
                },
            ],
            reference_event_edges: vec![TemporalReferenceEdge {
                edge_id: "ref-event:event:1:event:2".to_owned(),
                document_id: "doc-1".to_owned(),
                axis_id: TemporalAxisId("axis:world".to_owned()),
                source_event_id: "event:1".to_owned(),
                canonical_source_event_id: None,
                target_event_id: Some("event:2".to_owned()),
                canonical_target_event_id: None,
                target_timex_id: None,
                relation: "narrative_sequence".to_owned(),
                confidence_millis: 640,
                evidence_refs: vec!["narrative_sequence".to_owned()],
            }],
            temporal_claims: vec![TemporalClaimAtom {
                claim_id: "tclaim:1".to_owned(),
                document_id: "doc-1".to_owned(),
                proposition_id: Some("prop:1".to_owned()),
                event_id: Some("event:1".to_owned()),
                canonical_event_id: None,
                axis_id: TemporalAxisId("axis:world".to_owned()),
                source_kind: "deictic_today".to_owned(),
                label: "event:1 anchored today".to_owned(),
                confidence_millis: 900,
                temporal: temporal_window(Some(100)),
                evidence_refs: vec!["today".to_owned()],
            }],
            temporal_constraints: vec![TemporalConstraintRecord {
                constraint_id: TemporalConstraintId("tconstraint:1".to_owned()),
                document_id: "doc-1".to_owned(),
                axis_id: TemporalAxisId("axis:world".to_owned()),
                source_event_id: Some("event:1".to_owned()),
                canonical_source_event_id: None,
                target_event_id: None,
                canonical_target_event_id: None,
                target_timex_id: Some(TemporalTimexId("timex:doc-1:today".to_owned())),
                kind: TemporalConstraintKind::AnchoredAt,
                confidence_millis: 900,
                hard: true,
                temporal: temporal_window(Some(100)),
                evidence_refs: vec!["today".to_owned()],
            }],
            ..Default::default()
        }),
        ..DocumentArchive::default()
    }
}

#[test]
fn compiles_temporal_scope_from_substrate() {
    let archive = sample_archive();
    let mut batch = derive_scope_review_batch(&[archive], None, None);
    run_temporal_scope(&mut batch, 200);

    assert!(!batch.event_profiles.is_empty());
    assert!(!batch.review_cases.is_empty());
    assert!(!batch.intervals.is_empty());
    assert!(!batch.timeline_segments.is_empty());
    assert!(!batch.memory_cards.is_empty());
    assert!(batch.summary.timex_count >= 2);
}

#[test]
fn replay_replaces_temporal_outputs_idempotently() {
    let archive = sample_archive();
    let mut batch = derive_scope_review_batch(&[archive], None, None);
    run_temporal_scope(&mut batch, 200);
    let sidecar = crate::build_temporal_patch_sidecar(&batch, 200);

    let mut replayed = derive_scope_review_batch(&[], None, None);
    apply_temporal_patch_sidecar(&mut replayed, &sidecar);
    apply_temporal_patch_sidecar(&mut replayed, &sidecar);

    assert_eq!(replayed.intervals, sidecar.intervals);
    assert_eq!(replayed.timeline_segments, sidecar.timeline_segments);
    assert_eq!(replayed.summary, sidecar.summary);
}

#[test]
fn reported_axis_without_explicit_timex_stays_as_gap() {
    let archive = sample_archive();
    let mut batch = derive_scope_review_batch(&[archive], None, None);
    run_temporal_scope(&mut batch, 200);

    assert!(batch
        .gaps
        .iter()
        .any(|gap| gap.event_id.as_deref() == Some("event:2")));
    assert!(!batch
        .intervals
        .iter()
        .any(|interval| interval.event_id == "event:2"));
}

#[test]
fn event_identity_sidecar_adds_canonical_ids() {
    let archive = sample_archive();
    let mut batch = derive_scope_review_batch(&[archive], None, None);
    let sidecar = EventIdentityScopeSidecar {
        mention_packets: vec![EventMentionPacket {
            mention_id: EventMentionId("mention:event:1".to_owned()),
            event_id: "event:1".to_owned(),
            document_id: "doc-1".to_owned(),
            proposition_id: "prop:1".to_owned(),
            revision: 1,
            label: "arrive".to_owned(),
            normalized_predicate: "arrive today".to_owned(),
            event_type: "action".to_owned(),
            participant_slots: Vec::new(),
            place_labels: Vec::new(),
            explicit_timex_ids: Vec::new(),
            time_anchor_ids: Vec::new(),
            causal_neighbor_event_ids: Vec::new(),
            temporal_neighbor_event_ids: Vec::new(),
            sentence_index: 0,
            clause_range: None,
            polarity_negative: false,
            source_semantics: EventSourceSemantics::WorldAssertion,
            modality_semantics: EventModalitySemantics::Asserted,
            realis: "actual".to_owned(),
            event_fingerprint: "fingerprint:event:1".to_owned(),
            evidence_refs: vec!["prop:1".to_owned()],
        }],
        memberships: vec![EventIdentityMembershipRecord {
            membership_id: EventIdentityMembershipId("membership:event:1".to_owned()),
            canonical_event_id: CanonicalEventId("canonical:event:1".to_owned()),
            mention_id: EventMentionId("mention:event:1".to_owned()),
            relation: EventIdentityState::FullIdentity,
            confidence_millis: 1000,
            created_at: 100,
        }],
        ..EventIdentityScopeSidecar::default()
    };

    annotate_temporal_batch_with_event_identity(&mut batch, &sidecar);
    run_temporal_scope(&mut batch, 200);

    assert_eq!(
        batch
            .event_profiles
            .iter()
            .find(|profile| profile.event_id == "event:1")
            .and_then(|profile| profile.canonical_event_id.clone()),
        Some(CanonicalEventId("canonical:event:1".to_owned()))
    );
    assert!(batch
        .intervals
        .iter()
        .any(|interval| interval.event_id == "event:1"
            && interval.canonical_event_id
                == Some(CanonicalEventId("canonical:event:1".to_owned()))));
    assert!(batch
        .memory_cards
        .iter()
        .any(|card| card.event_id == "event:1"
            && card.canonical_event_id == Some(CanonicalEventId("canonical:event:1".to_owned()))));
}

fn temporal_window(valid_from: Option<i64>) -> BiTemporalWindow {
    BiTemporalWindow {
        valid_from,
        valid_to: valid_from,
        recorded_from: Some(100),
        recorded_to: None,
    }
}
