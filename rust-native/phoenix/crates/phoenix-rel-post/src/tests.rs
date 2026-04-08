use crate::{
    apply_relation_patch_sidecar, build_relation_patch_sidecar, derive_scope_review_batch,
    draft_relation_decisions, persist_relation_patch_sidecar, GlirelRelationPrediction,
    GlirelRelationTypeSpec, RelationDecision, RelationDecisionKind,
};
use phoenix_semantic_v2::{
    scope_storage_key, AliasEntry, AliasPosting, CandidateEntity, DocumentArchive, DocumentManifest,
    ErAliasAddition, ErScopePatchSidecar, NativeCorefSummary, NativeErSummary, ResolutionDecision,
    RelationMentionSeedRecord, RelationMentionSeedScopeSidecar, ResolvedMention,
    ScopeLexSidecar, SemanticRelationRecord,
};
use phoenix_store_native_core::{PhoenixRelationMentionSeedStore, PhoenixRelationPatchStore};
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::{
    DocumentId, EntityId, EntityKind, FrameSlot, IngestDocumentSummary, MentionEntityRef,
    RelationCandidate, ScopeKey, SessionDocumentState, TextRange,
};

fn sample_manifest() -> DocumentManifest {
    let scope = ScopeKey {
        world_id: Some("world".to_owned()),
        narrative_id: Some("narr".to_owned()),
        folder_id: None,
        folder_path: None,
    };
    DocumentManifest {
        document_id: "doc-1".to_owned(),
        scope_key: phoenix_semantic_v2::scope_storage_key(&scope),
        scope,
        scope_ord: phoenix_semantic_v2::ScopeOrd(7),
        revision: 1,
        title: "Test".to_owned(),
        session_document: SessionDocumentState {
            document_id: DocumentId("doc-1".to_owned()),
            ..Default::default()
        },
        document_summary: IngestDocumentSummary::default(),
        ..Default::default()
    }
}

fn sample_archive() -> DocumentArchive {
    use phoenix_semantic_v2::{ChunkId, ChunkRecord, SemanticEntityRecord};
    use phoenix_types::SentenceSpan;

    DocumentArchive {
        manifest: sample_manifest(),
        sentences: vec![
            SentenceSpan {
                index: 0,
                range: TextRange { start: 0, end: 24 },
                ..Default::default()
            },
            SentenceSpan {
                index: 1,
                range: TextRange { start: 25, end: 49 },
                ..Default::default()
            },
        ],
        resolved_mentions: vec![
            ResolvedMention {
                mention_id: phoenix_semantic_v2::MentionId("m1".to_owned()),
                mention_index: 0,
                range: TextRange { start: 0, end: 5 },
                surface: "Alice".to_owned(),
                normalized: "alice".to_owned(),
                kind: Some(EntityKind::Character),
                entity_id: Some(EntityId("e1".to_owned())),
                decision: ResolutionDecision {
                    status: "resolved".to_owned(),
                    confidence_millis: 910,
                    margin_millis: 300,
                },
                candidates: vec![CandidateEntity {
                    entity_id: "e1".to_owned(),
                    source: "native".to_owned(),
                    score_millis: 910,
                    evidence: Vec::new(),
                }],
            },
            ResolvedMention {
                mention_id: phoenix_semantic_v2::MentionId("m2".to_owned()),
                mention_index: 1,
                range: TextRange { start: 16, end: 23 },
                surface: "Dynamis".to_owned(),
                normalized: "dynamis".to_owned(),
                kind: Some(EntityKind::Organization),
                entity_id: Some(EntityId("e2".to_owned())),
                decision: ResolutionDecision {
                    status: "resolved".to_owned(),
                    confidence_millis: 900,
                    margin_millis: 250,
                },
                candidates: Vec::new(),
            },
            ResolvedMention {
                mention_id: phoenix_semantic_v2::MentionId("m3".to_owned()),
                mention_index: 2,
                range: TextRange { start: 25, end: 32 },
                surface: "Dynamis".to_owned(),
                normalized: "dynamis".to_owned(),
                kind: Some(EntityKind::Organization),
                entity_id: Some(EntityId("e2".to_owned())),
                decision: ResolutionDecision {
                    status: "resolved".to_owned(),
                    confidence_millis: 905,
                    margin_millis: 250,
                },
                candidates: Vec::new(),
            },
            ResolvedMention {
                mention_id: phoenix_semantic_v2::MentionId("m4".to_owned()),
                mention_index: 3,
                range: TextRange { start: 39, end: 47 },
                surface: "New Rome".to_owned(),
                normalized: "new rome".to_owned(),
                kind: Some(EntityKind::Location),
                entity_id: Some(EntityId("e3".to_owned())),
                decision: ResolutionDecision {
                    status: "resolved".to_owned(),
                    confidence_millis: 880,
                    margin_millis: 200,
                },
                candidates: Vec::new(),
            },
        ],
        entities: vec![
            SemanticEntityRecord {
                entity_id: EntityId("e1".to_owned()),
                canonical_name: "Alice".to_owned(),
                aliases: vec!["Al".to_owned()],
                kind: Some(EntityKind::Character),
                mention_count: 1,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e2".to_owned()),
                canonical_name: "Dynamis".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Organization),
                mention_count: 2,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e3".to_owned()),
                canonical_name: "New Rome".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Location),
                mention_count: 1,
                chunk_ids: vec!["chunk-2".to_owned()],
            },
        ],
        chunks: vec![
            ChunkRecord {
                chunk_id: ChunkId("chunk-1".to_owned()),
                range: TextRange { start: 0, end: 24 },
                chapter_id: 0,
                boundary_label: None,
                text: "Alice works for Dynamis.".to_owned(),
            },
            ChunkRecord {
                chunk_id: ChunkId("chunk-2".to_owned()),
                range: TextRange { start: 25, end: 49 },
                chapter_id: 0,
                boundary_label: None,
                text: "Dynamis is in New Rome.".to_owned(),
            },
        ],
        er_summary: NativeErSummary::default(),
        coref_summary: NativeCorefSummary::default(),
        ..Default::default()
    }
}

fn relation_candidate_archive() -> DocumentArchive {
    let mut archive = sample_archive();
    archive.resolved_mentions.clear();
    archive.sentences.clear();
    archive.chunks.clear();
    for entity in &mut archive.entities {
        entity.chunk_ids.clear();
    }
    archive.relation_candidates = vec![RelationCandidate {
        sentence_index: 0,
        verb_range: TextRange { start: 6, end: 15 },
        lemma: "joined".to_owned(),
        event_class: "social".to_owned(),
        relation_type: "member_of".to_owned(),
        subject: Some(FrameSlot {
            range: TextRange { start: 0, end: 5 },
            entity_ref: Some(MentionEntityRef::Known(EntityId("e1".to_owned()))),
            confidence: 0.95,
        }),
        object: Some(FrameSlot {
            range: TextRange { start: 16, end: 23 },
            entity_ref: Some(MentionEntityRef::Known(EntityId("e2".to_owned()))),
            confidence: 0.95,
        }),
        recipient: None,
        attachments: Vec::new(),
        evidence: vec![phoenix_types::EvidenceSpan {
            document_id: Some(DocumentId("doc-1".to_owned())),
            note_id: None,
            label: "Alice joined Dynamis.".to_owned(),
            kind: Some("relation".to_owned()),
            range: TextRange { start: 0, end: 23 },
        }],
    }];
    archive
}

fn persisted_relation_archive() -> DocumentArchive {
    let mut archive = chunk_only_archive("Alice works for Dynamis in New Rome.");
    archive.relations = vec![SemanticRelationRecord {
        source_entity_id: EntityId("e1".to_owned()),
        target_entity_id: EntityId("e2".to_owned()),
        edge_type: "works_for".to_owned(),
        sentence_index: 0,
        chunk_id: Some("chunk-1".to_owned()),
    }];
    archive
}

fn chunk_only_archive(text: &str) -> DocumentArchive {
    use phoenix_semantic_v2::{ChunkId, ChunkRecord, SemanticEntityRecord};

    DocumentArchive {
        manifest: sample_manifest(),
        chunks: vec![ChunkRecord {
            chunk_id: ChunkId("chunk-1".to_owned()),
            range: TextRange {
                start: 0,
                end: text.len() as u32,
            },
            chapter_id: 0,
            boundary_label: None,
            text: text.to_owned(),
        }],
        entities: vec![
            SemanticEntityRecord {
                entity_id: EntityId("e1".to_owned()),
                canonical_name: "Alice".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                mention_count: 1,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e2".to_owned()),
                canonical_name: "Dynamis".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Organization),
                mention_count: 1,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e3".to_owned()),
                canonical_name: "New Rome".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Location),
                mention_count: 1,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
        ],
        er_summary: NativeErSummary::default(),
        coref_summary: NativeCorefSummary::default(),
        ..Default::default()
    }
}

#[test]
fn derive_scope_review_batch_builds_windowed_pair_cases() {
    let archive = sample_archive();
    let batch = derive_scope_review_batch(&[archive], None, None, None, None, None);
    assert!(!batch.windows.is_empty());
    assert!(batch
        .review_cases
        .iter()
        .any(|case| case.source_entity_id.0 == "e1" && case.target_entity_id.0 == "e2"));
}

#[test]
fn derive_scope_review_batch_falls_back_to_relation_candidate_windows() {
    let archive = relation_candidate_archive();
    let batch = derive_scope_review_batch(&[archive], None, None, None, None, None);
    assert_eq!(batch.windows.len(), 1);
    assert_eq!(
        batch.windows[0].candidate_relation_types,
        vec!["member_of".to_owned()]
    );
    assert!(batch.windows[0].text.contains("Alice joined Dynamis."));
    assert!(batch.review_cases.iter().any(|case| case
        .seed_evidence
        .iter()
        .any(|value| value == "candidate_relation_type:member_of")));
}

#[test]
fn derive_scope_review_batch_falls_back_to_archive_relation_windows() {
    let archive = persisted_relation_archive();
    let batch = derive_scope_review_batch(&[archive], None, None, None, None, None);
    assert_eq!(batch.windows.len(), 1);
    assert!(batch.windows[0]
        .candidate_relation_types
        .iter()
        .any(|value| value == "works_for"));
    assert!(batch.windows[0].text.contains("Alice works for Dynamis"));
    assert!(batch.review_cases.iter().any(|case| {
        case.source_entity_id.0 == "e1"
            && case.target_entity_id.0 == "e2"
            && case
                .seed_evidence
                .iter()
                .any(|value| value == "candidate_relation_type:works_for")
    }));
}

#[test]
fn synthetic_sentence_split_handles_crlf_and_dialogue() {
    use phoenix_semantic_v2::{ChunkId, ChunkRecord};

    let chunk = ChunkRecord {
        chunk_id: ChunkId("chunk-1".to_owned()),
        range: TextRange { start: 0, end: 92 },
        chapter_id: 0,
        boundary_label: None,
        text: "## Chapter 1\r\n\r\nAlice works for Dynamis.\r\n\"Dynamis is in New Rome!\"\r\n\r\nTable of Contents"
            .to_owned(),
    };
    let sentences = crate::worker::split_chunk_into_synthetic_sentences(&chunk, 0);
    assert!(sentences.iter().any(|row| row.text.contains("Alice works for Dynamis.")));
    assert!(sentences
        .iter()
        .any(|row| row.text.contains("Dynamis is in New Rome!")));
}

#[test]
fn derive_scope_review_batch_rebuilds_windows_from_chunk_only_archive() {
    let archive = chunk_only_archive(
        "## Chapter 1\r\n\r\nAlice works for Dynamis. Dynamis is in New Rome.",
    );
    let batch = derive_scope_review_batch(&[archive], None, None, None, None, None);
    assert!(!batch.windows.is_empty());
    assert!(!batch.review_cases.is_empty());
    assert!(batch
        .windows
        .iter()
        .all(|window| !window.text.contains("Chapter 1")));
    assert!(batch
        .window_build_stats
        .window_source_counts
        .contains_key("synthetic_sentence"));
}

#[test]
fn derive_scope_review_batch_uses_er_alias_additions_for_anchor_rebuild() {
    let archive = chunk_only_archive("Al works for Dynamis.");
    let sidecar = ErScopePatchSidecar {
        scope: archive.manifest.scope.clone(),
        scope_key: archive.manifest.scope_key.clone(),
        scope_ord: Some(archive.manifest.scope_ord),
        alias_additions: vec![ErAliasAddition {
            case_id: "case-1".to_owned(),
            document_id: archive.manifest.document_id.clone(),
            mention_id: None,
            entity_id: EntityId("e1".to_owned()),
            alias_surface: "Al".to_owned(),
            normalized: "al".to_owned(),
            confidence_millis: 900,
            created_at: 1,
        }],
        ..Default::default()
    };
    let batch = derive_scope_review_batch(&[archive], None, None, None, Some(&sidecar), None);
    assert!(batch.review_cases.iter().any(|case| {
        case.source_entity_id.0 == "e1"
            && case.target_entity_id.0 == "e2"
            && case
                .seed_evidence
                .iter()
                .any(|value| value == "candidate_relation_type:works_for")
    }));
    assert!(batch
        .windows
        .iter()
        .any(|window| window
            .evidence_labels
            .iter()
            .any(|value| value == "anchor_evidence:alex_exact_alias")));
}

#[test]
fn derive_scope_review_batch_uses_lexical_alias_postings_for_anchor_rebuild() {
    let archive = chunk_only_archive("corp is in New Rome.");
    let scope = archive.manifest.scope.clone();
    let scope_key = scope_storage_key(&scope);
    let sidecar = ScopeLexSidecar {
        scope,
        scope_key,
        alias_entries: vec![AliasEntry {
            normalized: "corp".to_owned(),
            postings: vec![AliasPosting {
                entity_id: "e2".to_owned(),
                document_id: archive.manifest.document_id.clone(),
                mention_count: 2,
            }],
        }],
        ..Default::default()
    };
    let batch = derive_scope_review_batch(&[archive], None, None, Some(&sidecar), None, None);
    assert!(batch.review_cases.iter().any(|case| {
        case.source_entity_id.0 == "e2" && case.target_entity_id.0 == "e3"
    }));
    assert!(batch
        .windows
        .iter()
        .any(|window| window
            .evidence_labels
            .iter()
            .any(|value| value == "anchor_evidence:alex_exact_alias")));
}

#[test]
fn derive_scope_review_batch_rejects_generic_chunk_only_windows() {
    use phoenix_semantic_v2::{ChunkId, ChunkRecord, SemanticEntityRecord};

    let archive = DocumentArchive {
        manifest: sample_manifest(),
        chunks: vec![ChunkRecord {
            chunk_id: ChunkId("chunk-1".to_owned()),
            range: TextRange { start: 0, end: 25 },
            chapter_id: 0,
            boundary_label: None,
            text: "Security guards Chapter.".to_owned(),
        }],
        entities: vec![
            SemanticEntityRecord {
                entity_id: EntityId("e1".to_owned()),
                canonical_name: "Security".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Organization),
                mention_count: 1,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e2".to_owned()),
                canonical_name: "Chapter".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Location),
                mention_count: 1,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
        ],
        ..Default::default()
    };
    let batch = derive_scope_review_batch(&[archive], None, None, None, None, None);
    assert!(batch.windows.is_empty());
    assert!(batch.review_cases.is_empty());
}

#[test]
fn derive_scope_review_batch_clips_chunk_windows_around_anchors() {
    let text = "Long lead text before the cue. Alice works for Dynamis while everyone else keeps talking for a while after the cue.";
    let archive = chunk_only_archive(text);
    let batch = derive_scope_review_batch(&[archive], None, None, None, None, None);
    assert!(!batch.windows.is_empty());
    assert!(batch.windows[0].text.len() < text.len());
}

#[test]
fn draft_relation_decisions_uses_family_thresholds() {
    let archive = sample_archive();
    let mut batch = derive_scope_review_batch(&[archive], None, None, None, None, None);
    let source_name = batch.review_cases[0].source_name.clone();
    let target_name = batch.review_cases[0].target_name.clone();
    batch.review_cases[0]
        .glirel_predictions
        .push(GlirelRelationPrediction {
            head_index: 0,
            tail_index: 1,
            head: source_name,
            tail: target_name,
            relation: "works_for".to_owned(),
            confidence: 0.82,
            evidence: vec!["glirel_score:1.2".to_owned()],
        });
    let decisions = draft_relation_decisions(
        &batch,
        &[GlirelRelationTypeSpec {
            label: "works_for".to_owned(),
            head_types: vec!["Character".to_owned()],
            tail_types: vec!["Organization".to_owned()],
            cue_phrases: vec!["works for".to_owned()],
            conflicts_with: vec!["member_of".to_owned()],
            priority_millis: 150,
            accept_threshold_millis: 500,
            review_threshold_millis: 450,
            max_predictions_per_window: 1,
            directed: true,
        }],
    );
    assert!(decisions
        .iter()
        .any(|decision| decision.kind == RelationDecisionKind::Accept));
}

#[test]
fn build_and_apply_relation_patch_sidecar_replays_edges() {
    let archive = sample_archive();
    let batch = derive_scope_review_batch(&[archive], None, None, None, None, None);
    let decision = RelationDecision {
        case_id: batch.review_cases[0].case_id.clone(),
        kind: RelationDecisionKind::Accept,
        edge_type: Some("works_for".to_owned()),
        score_millis: 811,
        rationale: "accepted".to_owned(),
        evidence: vec!["glirel_score:1.4".to_owned()],
        source_entity_id: Some(batch.review_cases[0].source_entity_id.clone()),
        target_entity_id: Some(batch.review_cases[0].target_entity_id.clone()),
        support_confidence_millis: Some(844),
        contradiction_confidence_millis: None,
    };
    let sidecar = build_relation_patch_sidecar(&batch, &[decision], 2222);
    let mut replayed = batch.clone();
    apply_relation_patch_sidecar(&mut replayed, &sidecar);
    assert!(replayed
        .review_cases
        .iter()
        .any(|case| case.decision_status.starts_with("relation_accept")));
    assert_eq!(sidecar.support_judgments.len(), 1);
    assert!(replayed
        .persisted_relations
        .iter()
        .any(|relation| relation.edge_type == "works_for"));
}

#[test]
fn persists_relation_patch_sidecar_in_overgraph_store() {
    let store_path =
        std::env::temp_dir().join(format!("phoenix-rel-post-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&store_path);
    let store = PhoenixOvergraphStore::open(&store_path).expect("open store");
    let batch = derive_scope_review_batch(&[sample_archive()], None, None, None, None, None);
    let decision = RelationDecision {
        case_id: batch.review_cases[0].case_id.clone(),
        kind: RelationDecisionKind::Accept,
        edge_type: Some("works_for".to_owned()),
        score_millis: 801,
        rationale: "accepted".to_owned(),
        evidence: vec!["glirel_score:1.3".to_owned()],
        source_entity_id: Some(batch.review_cases[0].source_entity_id.clone()),
        target_entity_id: Some(batch.review_cases[0].target_entity_id.clone()),
        support_confidence_millis: Some(820),
        contradiction_confidence_millis: None,
    };
    let persisted =
        persist_relation_patch_sidecar(&store, &batch, &[decision], 3333).expect("persist");
    let loaded = store
        .load_relation_patch_sidecar(&batch.scope)
        .expect("load")
        .expect("exists");
    assert_eq!(loaded.scope_key, persisted.scope_key);
    assert_eq!(loaded.edge_additions, persisted.edge_additions);
    let _ = std::fs::remove_dir_all(store_path);
}

#[test]
fn persists_relation_mention_seed_sidecar_in_overgraph_store() {
    let store_path =
        std::env::temp_dir().join(format!("phoenix-rel-seed-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&store_path);
    let store = PhoenixOvergraphStore::open(&store_path).expect("open store");
    let archive = sample_archive();
    let sidecar = RelationMentionSeedScopeSidecar {
        scope: archive.manifest.scope.clone(),
        scope_key: archive.manifest.scope_key.clone(),
        scope_ord: Some(archive.manifest.scope_ord),
        session_id: None,
        updated_at: 42,
        generation: 1,
        seeds: vec![RelationMentionSeedRecord {
            document_id: archive.manifest.document_id.clone(),
            revision: archive.manifest.revision,
            chunk_id: "chunk-1".to_owned(),
            entity_id: EntityId("e1".to_owned()),
            surface: "Alice".to_owned(),
            normalized: "alice".to_owned(),
            kind: Some(EntityKind::Character),
            range: TextRange { start: 0, end: 5 },
            sentence_index: Some(0),
            confidence_millis: 920,
            seed_label: "person".to_owned(),
            evidence: vec!["seed_source:chunker_microchunk".to_owned()],
            created_at: 42,
        }],
    };

    store
        .persist_relation_mention_seed_sidecar(&sidecar)
        .expect("persist seed sidecar");
    let loaded = store
        .load_relation_mention_seed_sidecar(&archive.manifest.scope)
        .expect("load seed sidecar")
        .expect("seed sidecar exists");
    assert_eq!(loaded.seeds.len(), 1);
    assert_eq!(loaded.seeds[0].surface, "Alice");
    assert_eq!(loaded.seeds[0].seed_label, "person");
    let _ = std::fs::remove_dir_all(store_path);
}
