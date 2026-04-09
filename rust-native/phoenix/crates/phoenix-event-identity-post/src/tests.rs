use phoenix_semantic_v2::{
    DocumentArchive, DocumentEventIdentitySubstrate, DocumentManifest, EventIdentityState,
    EventMentionId, EventMentionPacketSeed, EventModalitySemantics, EventParticipantSlot,
    EventSourceSemantics, ScopeOrd, TemporalAnchorId, TemporalTimexId,
};
use phoenix_types::{EntityId, ScopeKey, TextRange};

use crate::{
    apply_event_identity_patch_sidecar, build_identity_hypotheses, derive_scope_review_batch,
    run_event_identity_scope,
};

fn sample_archive(
    document_id: &str,
    revision: u64,
    seeds: Vec<EventMentionPacketSeed>,
) -> DocumentArchive {
    DocumentArchive {
        manifest: DocumentManifest {
            document_id: document_id.to_owned(),
            scope: ScopeKey {
                world_id: Some("world".to_owned()),
                narrative_id: Some("story".to_owned()),
                folder_id: None,
                folder_path: None,
            },
            scope_key: "world/story".to_owned(),
            scope_ord: ScopeOrd(1),
            revision,
            created_at: 100 + revision as i64,
            ..DocumentManifest::default()
        },
        event_identity_substrate: Some(DocumentEventIdentitySubstrate {
            mention_seeds: seeds,
            diagnostics: Vec::new(),
        }),
        ..DocumentArchive::default()
    }
}

fn seed(
    mention_id: &str,
    event_id: &str,
    document_id: &str,
    revision: u64,
    predicate: &str,
    participant_entity: &str,
    place: &str,
    timex: &str,
) -> EventMentionPacketSeed {
    EventMentionPacketSeed {
        mention_id: EventMentionId(mention_id.to_owned()),
        event_id: event_id.to_owned(),
        document_id: document_id.to_owned(),
        proposition_id: format!("prop:{mention_id}"),
        revision,
        label: predicate.to_owned(),
        normalized_predicate: predicate.to_ascii_lowercase(),
        event_type: "event".to_owned(),
        participant_slots: vec![EventParticipantSlot {
            role: "agent".to_owned(),
            entity_id: Some(EntityId(participant_entity.to_owned())),
            mention_index: None,
            label: Some(participant_entity.to_owned()),
            range: Some(TextRange { start: 0, end: 4 }),
        }],
        place_labels: vec![place.to_owned()],
        explicit_timex_ids: vec![TemporalTimexId(timex.to_owned())],
        time_anchor_ids: vec![TemporalAnchorId(format!("anchor:{timex}"))],
        causal_neighbor_event_ids: Vec::new(),
        temporal_neighbor_event_ids: Vec::new(),
        sentence_index: 0,
        clause_range: Some(TextRange { start: 0, end: 12 }),
        polarity_negative: false,
        source_semantics: EventSourceSemantics::WorldAssertion,
        modality_semantics: EventModalitySemantics::Asserted,
        realis: "asserted".to_owned(),
        event_fingerprint: format!("{predicate}:{participant_entity}:{place}:{timex}"),
        evidence_refs: vec![mention_id.to_owned()],
    }
}

#[test]
fn paraphrase_with_shared_frame_reaches_non_binary_identity() {
    let archives = vec![
        sample_archive(
            "doc-a",
            1,
            vec![seed(
                "m1", "event:1", "doc-a", 1, "attack", "alice", "rome", "timex:1",
            )],
        ),
        sample_archive(
            "doc-b",
            2,
            vec![seed(
                "m2", "event:2", "doc-b", 2, "assault", "alice", "rome", "timex:1",
            )],
        ),
    ];
    let batch = derive_scope_review_batch(&archives, None, None, None);
    let (hypotheses, _, _) = build_identity_hypotheses(&batch.scope_key, &batch.mention_packets);

    assert!(hypotheses.iter().any(|hypothesis| {
        matches!(
            hypothesis.relation,
            EventIdentityState::FullIdentity | EventIdentityState::QuasiIdentity
        )
    }));
}

#[test]
fn incompatible_participants_block_same_trigger_merge() {
    let archives = vec![sample_archive(
        "doc-a",
        1,
        vec![
            seed(
                "m1", "event:1", "doc-a", 1, "attack", "alice", "rome", "timex:1",
            ),
            seed(
                "m2", "event:2", "doc-a", 1, "attack", "bob", "rome", "timex:1",
            ),
        ],
    )];
    let batch = derive_scope_review_batch(&archives, None, None, None);
    let (hypotheses, _, _) = build_identity_hypotheses(&batch.scope_key, &batch.mention_packets);

    assert!(hypotheses
        .iter()
        .any(|hypothesis| hypothesis.relation == EventIdentityState::Incompatible));
}

#[test]
fn subevent_links_are_preserved_without_force_merge() {
    let mut parent = seed(
        "m1", "event:1", "doc-a", 1, "attack", "alice", "rome", "timex:1",
    );
    parent
        .temporal_neighbor_event_ids
        .push("event:2".to_owned());
    let child = seed(
        "m2", "event:2", "doc-a", 1, "strike", "alice", "rome", "timex:1",
    );
    let archives = vec![sample_archive("doc-a", 1, vec![parent, child])];
    let mut batch = derive_scope_review_batch(&archives, None, None, None);
    run_event_identity_scope(&mut batch, 200);

    assert!(batch
        .identity_hypotheses
        .iter()
        .any(|hypothesis| hypothesis.relation == EventIdentityState::SubeventOf));
    assert_eq!(batch.canonical_events.len(), 2);
}

#[test]
fn replay_keeps_canonical_ids_stable() {
    let archives = vec![sample_archive(
        "doc-a",
        1,
        vec![
            seed(
                "m1", "event:1", "doc-a", 1, "attack", "alice", "rome", "timex:1",
            ),
            seed(
                "m2", "event:2", "doc-a", 2, "attack", "alice", "rome", "timex:1",
            ),
        ],
    )];
    let mut batch = derive_scope_review_batch(&archives, None, None, None);
    run_event_identity_scope(&mut batch, 200);
    let sidecar = crate::build_event_identity_patch_sidecar(&batch, 200);

    let mut replayed = derive_scope_review_batch(&[], None, None, None);
    apply_event_identity_patch_sidecar(&mut replayed, &sidecar);
    apply_event_identity_patch_sidecar(&mut replayed, &sidecar);

    assert_eq!(replayed.canonical_events, sidecar.canonical_events);
    assert_eq!(replayed.summary, sidecar.summary);
}

#[test]
fn canonical_event_ids_do_not_collide_for_long_similar_prefixes() {
    let long_prefix = "sameprefixsameprefixsameprefixsameprefixsameprefixsameprefixsameprefix";
    let mut left = seed(
        &format!("{long_prefix}-mention-left"),
        "event:left",
        "doc-a",
        1,
        "signal",
        "alice",
        "rome",
        "timex:1",
    );
    left.event_fingerprint = format!("{long_prefix}-fingerprint-left");
    let mut right = seed(
        &format!("{long_prefix}-mention-right"),
        "event:right",
        "doc-a",
        1,
        "signal",
        "bob",
        "rome",
        "timex:1",
    );
    right.event_fingerprint = format!("{long_prefix}-fingerprint-right");

    let archives = vec![sample_archive("doc-a", 1, vec![left, right])];
    let mut batch = derive_scope_review_batch(&archives, None, None, None);
    run_event_identity_scope(&mut batch, 200);

    let unique_ids = batch
        .canonical_events
        .iter()
        .map(|row| row.canonical_event_id.0.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(unique_ids.len(), batch.canonical_events.len());
}
