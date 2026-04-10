use crate::{
    apply_memory_patch_sidecar, build_memory_patch_sidecar, derive_scope_review_batch,
    normalize_memory_inputs, persist_memory_patch_sidecar,
};
use phoenix_semantic_v2::{
    default_state_slot_definitions, scope_storage_key, DocumentArchive, DocumentManifest,
    ErAliasAddition, ErEntityLinkOverride, ErScopePatchSidecar, ErTypeOverride, MemoryGapKind,
    NativeCorefSummary, NativeErSummary, RelationDecisionOutcome, RelationDecisionRecord,
    RelationEdgeAddition, RelationJudgmentKind, RelationJudgmentRecord, RelationScopePatchSidecar,
    ScopeLexSidecar, SemanticEntityRecord, SemanticRelationRecord, SessionArchive,
    StateSchemaScopeSidecar, StateSlotLifecycle,
};
use phoenix_store_native_core::PhoenixMemoryPatchStore;
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::{
    DocumentId, EntityId, EntityKind, IngestDocumentSummary, ScopeKey, SessionDocumentState,
};

fn sample_scope() -> ScopeKey {
    ScopeKey {
        world_id: Some("world".to_owned()),
        narrative_id: Some("narr".to_owned()),
        folder_id: None,
        folder_path: None,
    }
}

fn sample_manifest(document_id: &str, created_at: i64) -> DocumentManifest {
    let scope = sample_scope();
    DocumentManifest {
        document_id: document_id.to_owned(),
        scope_key: scope_storage_key(&scope),
        scope,
        scope_ord: phoenix_semantic_v2::ScopeOrd(7),
        revision: 1,
        title: "Test".to_owned(),
        created_at,
        session_id: Some(phoenix_types::SessionId("session-1".to_owned())),
        session_document: SessionDocumentState {
            document_id: DocumentId(document_id.to_owned()),
            ..Default::default()
        },
        document_summary: IngestDocumentSummary::default(),
        ..Default::default()
    }
}

fn sample_archive() -> DocumentArchive {
    DocumentArchive {
        manifest: sample_manifest("doc-1", 100),
        entities: vec![
            SemanticEntityRecord {
                entity_id: EntityId("e1".to_owned()),
                canonical_name: "Alice".to_owned(),
                aliases: vec!["Al".to_owned()],
                kind: Some(EntityKind::Character),
                mention_count: 2,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e2".to_owned()),
                canonical_name: "Dynamis".to_owned(),
                aliases: vec!["Corp".to_owned()],
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
                chunk_ids: vec!["chunk-1".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e4".to_owned()),
                canonical_name: "Old Rome".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Location),
                mention_count: 1,
                chunk_ids: vec!["chunk-2".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e5".to_owned()),
                canonical_name: "Bob".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                mention_count: 1,
                chunk_ids: vec!["chunk-3".to_owned()],
            },
        ],
        relations: vec![SemanticRelationRecord {
            source_entity_id: EntityId("e5".to_owned()),
            target_entity_id: EntityId("e2".to_owned()),
            edge_type: "member_of".to_owned(),
            sentence_index: 0,
            chunk_id: Some("chunk-3".to_owned()),
        }],
        er_summary: NativeErSummary::default(),
        coref_summary: NativeCorefSummary::default(),
        ..Default::default()
    }
}

fn sample_session() -> SessionArchive {
    SessionArchive {
        session_id: phoenix_types::SessionId("session-1".to_owned()),
        updated_at: 150,
        ..Default::default()
    }
}

fn sample_er_sidecar() -> ErScopePatchSidecar {
    let scope = sample_scope();
    ErScopePatchSidecar {
        scope: scope.clone(),
        scope_key: scope_storage_key(&scope),
        scope_ord: Some(phoenix_semantic_v2::ScopeOrd(7)),
        session_id: Some(phoenix_types::SessionId("session-1".to_owned())),
        updated_at: 200,
        generation: 200,
        alias_additions: vec![ErAliasAddition {
            case_id: "case-alias".to_owned(),
            document_id: "doc-1".to_owned(),
            mention_id: None,
            entity_id: EntityId("e1".to_owned()),
            alias_surface: "Ace".to_owned(),
            normalized: "ace".to_owned(),
            confidence_millis: 910,
            created_at: 200,
        }],
        type_overrides: vec![ErTypeOverride {
            case_id: "case-type".to_owned(),
            document_id: "doc-1".to_owned(),
            mention_id: None,
            entity_id: EntityId("e1".to_owned()),
            kind: EntityKind::Npc,
            confidence_millis: 905,
            created_at: 205,
        }],
        entity_links: vec![ErEntityLinkOverride {
            case_id: "case-link".to_owned(),
            document_id: "doc-1".to_owned(),
            mention_id: None,
            entity_id: EntityId("e1".to_owned()),
            confidence_millis: 930,
            created_at: 210,
        }],
        ..Default::default()
    }
}

fn sample_relation_sidecar() -> RelationScopePatchSidecar {
    let scope = sample_scope();
    RelationScopePatchSidecar {
        scope: scope.clone(),
        scope_key: scope_storage_key(&scope),
        scope_ord: Some(phoenix_semantic_v2::ScopeOrd(7)),
        session_id: Some(phoenix_types::SessionId("session-1".to_owned())),
        updated_at: 300,
        generation: 300,
        edge_additions: vec![
            RelationEdgeAddition {
                case_id: "case-loc-1".to_owned(),
                document_id: "doc-1".to_owned(),
                window_id: "window-1".to_owned(),
                source_entity_id: EntityId("e1".to_owned()),
                target_entity_id: EntityId("e3".to_owned()),
                edge_type: "located_in".to_owned(),
                confidence_millis: 810,
                evidence_refs: vec!["ev:loc-1".to_owned()],
                created_at: 300,
            },
            RelationEdgeAddition {
                case_id: "case-loc-2".to_owned(),
                document_id: "doc-2".to_owned(),
                window_id: "window-2".to_owned(),
                source_entity_id: EntityId("e1".to_owned()),
                target_entity_id: EntityId("e4".to_owned()),
                edge_type: "located_in".to_owned(),
                confidence_millis: 870,
                evidence_refs: vec!["ev:loc-2".to_owned()],
                created_at: 400,
            },
            RelationEdgeAddition {
                case_id: "case-work".to_owned(),
                document_id: "doc-1".to_owned(),
                window_id: "window-3".to_owned(),
                source_entity_id: EntityId("e1".to_owned()),
                target_entity_id: EntityId("e2".to_owned()),
                edge_type: "works_for".to_owned(),
                confidence_millis: 850,
                evidence_refs: vec!["ev:work".to_owned()],
                created_at: 320,
            },
            RelationEdgeAddition {
                case_id: "case-command".to_owned(),
                document_id: "doc-1".to_owned(),
                window_id: "window-4".to_owned(),
                source_entity_id: EntityId("e1".to_owned()),
                target_entity_id: EntityId("e5".to_owned()),
                edge_type: "commands".to_owned(),
                confidence_millis: 700,
                evidence_refs: vec!["ev:command".to_owned()],
                created_at: 330,
            },
        ],
        support_judgments: vec![RelationJudgmentRecord {
            case_id: "case-work".to_owned(),
            document_id: "doc-1".to_owned(),
            window_id: "window-3".to_owned(),
            source_entity_id: EntityId("e1".to_owned()),
            target_entity_id: EntityId("e2".to_owned()),
            edge_type: "works_for".to_owned(),
            kind: RelationJudgmentKind::Support,
            confidence_millis: 900,
            evidence_refs: vec!["ev:support".to_owned()],
            created_at: 330,
        }],
        contradiction_judgments: vec![RelationJudgmentRecord {
            case_id: "case-loc-1".to_owned(),
            document_id: "doc-1".to_owned(),
            window_id: "window-1".to_owned(),
            source_entity_id: EntityId("e1".to_owned()),
            target_entity_id: EntityId("e3".to_owned()),
            edge_type: "located_in".to_owned(),
            kind: RelationJudgmentKind::Contradict,
            confidence_millis: 780,
            evidence_refs: vec!["ev:contradict".to_owned()],
            created_at: 410,
        }],
        decisions: vec![RelationDecisionRecord {
            case_id: "case-protect".to_owned(),
            document_id: "doc-1".to_owned(),
            window_id: "window-5".to_owned(),
            outcome: RelationDecisionOutcome::Defer,
            source_entity_id: Some(EntityId("e1".to_owned())),
            target_entity_id: Some(EntityId("e5".to_owned())),
            edge_type: Some("protects".to_owned()),
            score_millis: 430,
            rationale: "not enough evidence".to_owned(),
            evidence: vec!["review".to_owned()],
            reviewed_at: 420,
        }],
        ..Default::default()
    }
}

fn sample_prefixed_relation_sidecar() -> RelationScopePatchSidecar {
    let mut sidecar = sample_relation_sidecar();
    sidecar.edge_additions = vec![RelationEdgeAddition {
        case_id: "case-prefixed-loc".to_owned(),
        document_id: "doc-1".to_owned(),
        window_id: "window-prefixed".to_owned(),
        source_entity_id: EntityId("e1".to_owned()),
        target_entity_id: EntityId("e3".to_owned()),
        edge_type: "window::located_in".to_owned(),
        confidence_millis: 805,
        evidence_refs: vec!["ev:prefixed".to_owned()],
        created_at: 305,
    }];
    sidecar.support_judgments.clear();
    sidecar.contradiction_judgments.clear();
    sidecar.decisions.clear();
    sidecar
}

#[test]
fn normalizes_claims_from_relation_and_er_inputs() {
    let archive = sample_archive();
    let normalized = normalize_memory_inputs(
        &[archive],
        Some(&sample_session()),
        Some(&ScopeLexSidecar::default()),
        Some(&sample_er_sidecar()),
        Some(&sample_relation_sidecar()),
        None,
    );
    assert!(normalized
        .claims
        .iter()
        .any(|claim| claim.source_class == "relation_edge_addition"
            && claim.slot_key == "entity.location"));
    assert!(normalized
        .claims
        .iter()
        .any(|claim| claim.source_class == "relation_support_judgment"));
    assert!(normalized
        .claims
        .iter()
        .any(|claim| claim.source_class == "relation_contradiction_judgment"));
    assert!(normalized
        .claims
        .iter()
        .any(|claim| claim.source_class == "er_alias_addition"));
    assert!(!normalized.pending_reviews.is_empty());
}

#[test]
fn normalizes_prefixed_relation_families_into_state_slots() {
    let normalized = normalize_memory_inputs(
        &[sample_archive()],
        Some(&sample_session()),
        None,
        None,
        Some(&sample_prefixed_relation_sidecar()),
        None,
    );
    assert!(normalized.claims.iter().any(|claim| {
        claim.source_class == "relation_edge_addition"
            && claim.relation_family.as_deref() == Some("located_in")
            && claim.slot_key == "entity.location"
    }));

    let batch = derive_scope_review_batch(
        &[sample_archive()],
        Some(&sample_session()),
        None,
        None,
        None,
        Some(&sample_prefixed_relation_sidecar()),
        None,
    );
    assert!(batch
        .states
        .iter()
        .any(|state| state.slot_key == "entity.location" && state.value == "New Rome"));
}

#[test]
fn derives_states_deltas_conflicts_and_gaps() {
    let batch = derive_scope_review_batch(
        &[sample_archive()],
        Some(&sample_session()),
        None,
        None,
        Some(&sample_er_sidecar()),
        Some(&sample_relation_sidecar()),
        None,
    );
    assert!(batch
        .states
        .iter()
        .any(|state| state.slot_key == "entity.location" && state.value == "Old Rome"));
    assert!(batch
        .states
        .iter()
        .any(|state| state.slot_key == "entity.employer" && state.value == "Dynamis"));
    assert!(batch
        .deltas
        .iter()
        .any(|delta| delta.slot_key == "entity.location"
            && delta.old_value.as_deref() == Some("New Rome")
            && delta.new_value.as_deref() == Some("Old Rome")));
    assert!(batch
        .conflicts
        .iter()
        .any(|conflict| conflict.slot_key == "entity.location"));
    assert!(batch
        .gaps
        .iter()
        .any(|gap| gap.kind == MemoryGapKind::BrokenContinuity
            || gap.kind == MemoryGapKind::UnresolvedConflict));
    assert!(batch
        .relationship_ledgers
        .iter()
        .any(|ledger| ledger.relation_family == "commands"));
}

#[test]
fn cards_include_identity_state_and_relationship_views() {
    let batch = derive_scope_review_batch(
        &[sample_archive()],
        Some(&sample_session()),
        None,
        None,
        Some(&sample_er_sidecar()),
        Some(&sample_relation_sidecar()),
        None,
    );
    let card = batch
        .entity_cards
        .iter()
        .find(|card| card.entity_id.0 == "e1")
        .expect("alice card");
    assert_eq!(card.identity.canonical_name, "Alice");
    assert!(card.identity.aliases.iter().any(|value| value == "Ace"));
    assert!(card
        .current_state
        .iter()
        .any(|state| state.slot_key == "entity.location"));
    assert!(card
        .active_relationships
        .iter()
        .any(|row| row.relation_family == "commands"));
}

#[test]
fn build_and_apply_sidecar_is_idempotent() {
    let batch = derive_scope_review_batch(
        &[sample_archive()],
        Some(&sample_session()),
        None,
        None,
        Some(&sample_er_sidecar()),
        Some(&sample_relation_sidecar()),
        None,
    );
    let sidecar = build_memory_patch_sidecar(&batch, 999);
    let mut replayed = batch.clone();
    apply_memory_patch_sidecar(&mut replayed, &sidecar);
    apply_memory_patch_sidecar(&mut replayed, &sidecar);
    assert_eq!(replayed.summary, batch.summary);
    assert_eq!(replayed.entity_cards, batch.entity_cards);
    assert_eq!(replayed.memory_generation, Some(999));
}

#[test]
fn persists_memory_sidecar_in_overgraph_store() {
    let store_path =
        std::env::temp_dir().join(format!("phoenix-memory-post-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&store_path);
    let store = PhoenixOvergraphStore::open(&store_path).expect("open store");
    let batch = derive_scope_review_batch(
        &[sample_archive()],
        Some(&sample_session()),
        None,
        None,
        Some(&sample_er_sidecar()),
        Some(&sample_relation_sidecar()),
        None,
    );
    let persisted = persist_memory_patch_sidecar(&store, &batch, 1234).expect("persist");
    let loaded = store
        .load_memory_patch_sidecar(&batch.scope)
        .expect("load")
        .expect("present");
    assert_eq!(persisted.summary, loaded.summary);
    assert_eq!(persisted.entity_cards, loaded.entity_cards);
    let _ = std::fs::remove_dir_all(&store_path);
}

#[test]
fn dynamic_state_schema_slots_compile_into_memory_states() {
    let scope = sample_scope();
    let archive = DocumentArchive {
        manifest: sample_manifest("doc-task", 500),
        entities: vec![
            SemanticEntityRecord {
                entity_id: EntityId("task-1".to_owned()),
                canonical_name: "Find the vault".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Other),
                mention_count: 1,
                chunk_ids: Vec::new(),
            },
            SemanticEntityRecord {
                entity_id: EntityId("e1".to_owned()),
                canonical_name: "Alice".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                mention_count: 1,
                chunk_ids: Vec::new(),
            },
        ],
        ..Default::default()
    };
    let relation_sidecar = RelationScopePatchSidecar {
        scope: scope.clone(),
        scope_key: scope_storage_key(&scope),
        scope_ord: Some(phoenix_semantic_v2::ScopeOrd(7)),
        updated_at: 510,
        generation: 510,
        edge_additions: vec![RelationEdgeAddition {
            case_id: "case-assigned".to_owned(),
            document_id: "doc-task".to_owned(),
            window_id: "window-task".to_owned(),
            source_entity_id: EntityId("task-1".to_owned()),
            target_entity_id: EntityId("e1".to_owned()),
            edge_type: "assigned_to".to_owned(),
            confidence_millis: 910,
            evidence_refs: vec!["ev:assigned".to_owned()],
            created_at: 510,
        }],
        ..Default::default()
    };
    let mut slot_definitions = default_state_slot_definitions();
    let task_owner = slot_definitions
        .iter_mut()
        .find(|definition| definition.slot_key == "task.owner")
        .expect("task.owner definition");
    task_owner.lifecycle = StateSlotLifecycle::Active;
    if !task_owner
        .relation_families
        .iter()
        .any(|family| family == "assigned_to")
    {
        task_owner.relation_families.push("assigned_to".to_owned());
    }
    let state_schema_sidecar = StateSchemaScopeSidecar {
        scope: scope.clone(),
        scope_key: scope_storage_key(&scope),
        scope_ord: Some(phoenix_semantic_v2::ScopeOrd(7)),
        updated_at: 520,
        generation: 1,
        slot_definitions,
        ..Default::default()
    };

    let batch = derive_scope_review_batch(
        &[archive],
        None,
        None,
        None,
        None,
        Some(&relation_sidecar),
        Some(&state_schema_sidecar),
    );
    assert!(batch
        .states
        .iter()
        .any(|state| state.slot_key == "task.owner" && state.value == "Alice"));
}
