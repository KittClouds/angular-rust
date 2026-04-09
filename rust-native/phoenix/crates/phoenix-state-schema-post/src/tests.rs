use phoenix_semantic_v2::{
    scope_storage_key, DocumentArchive, DocumentManifest, RelationEdgeAddition,
    RelationScopePatchSidecar, SemanticEntityRecord, SemanticRelationRecord,
};
use phoenix_store_native_core::PhoenixStateSchemaPatchStore;
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::{EntityId, EntityKind, ScopeKey};

use crate::{
    derive_scope_review_batch, persist_state_schema_patch_sidecar, run_state_schema_scope,
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
        ..Default::default()
    }
}

fn sample_archives() -> Vec<DocumentArchive> {
    vec![
        DocumentArchive {
            manifest: sample_manifest("doc-1", 100),
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
                    entity_id: EntityId("person-1".to_owned()),
                    canonical_name: "Ava".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Character),
                    mention_count: 1,
                    chunk_ids: Vec::new(),
                },
            ],
            relations: vec![SemanticRelationRecord {
                source_entity_id: EntityId("task-1".to_owned()),
                target_entity_id: EntityId("person-1".to_owned()),
                edge_type: "assigned_to".to_owned(),
                sentence_index: 0,
                chunk_id: None,
            }],
            ..Default::default()
        },
        DocumentArchive {
            manifest: sample_manifest("doc-2", 120),
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
                    entity_id: EntityId("person-1".to_owned()),
                    canonical_name: "Ava".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Character),
                    mention_count: 1,
                    chunk_ids: Vec::new(),
                },
            ],
            ..Default::default()
        },
    ]
}

fn sample_relation_sidecar() -> RelationScopePatchSidecar {
    let scope = sample_scope();
    RelationScopePatchSidecar {
        scope: scope.clone(),
        scope_key: scope_storage_key(&scope),
        scope_ord: Some(phoenix_semantic_v2::ScopeOrd(7)),
        updated_at: 200,
        generation: 200,
        edge_additions: vec![RelationEdgeAddition {
            case_id: "case-owner".to_owned(),
            document_id: "doc-2".to_owned(),
            window_id: "window-owner".to_owned(),
            source_entity_id: EntityId("task-1".to_owned()),
            target_entity_id: EntityId("person-1".to_owned()),
            edge_type: "assigned_to".to_owned(),
            confidence_millis: 920,
            evidence_refs: vec!["ev:owner".to_owned()],
            created_at: 200,
        }],
        ..Default::default()
    }
}

#[test]
fn promotes_assignment_slot_from_relation_evidence() {
    let mut batch = derive_scope_review_batch(
        &sample_archives(),
        None,
        None,
        Some(&sample_relation_sidecar()),
    );
    run_state_schema_scope(&mut batch, 999);
    let task_owner = batch
        .slot_definitions
        .iter()
        .find(|definition| definition.slot_key == "task.owner")
        .expect("task.owner definition");
    assert!(matches!(
        task_owner.lifecycle,
        phoenix_semantic_v2::StateSlotLifecycle::Active
            | phoenix_semantic_v2::StateSlotLifecycle::Stable
    ));
    assert!(batch.write_proposals.iter().any(
        |proposal| proposal.slot_key == "task.owner" && proposal.owner_entity_id.0 == "task-1"
    ));
}

#[test]
fn discovered_slots_stay_candidate() {
    let scope = sample_scope();
    let sidecar = RelationScopePatchSidecar {
        scope: scope.clone(),
        scope_key: scope_storage_key(&scope),
        scope_ord: Some(phoenix_semantic_v2::ScopeOrd(7)),
        updated_at: 220,
        generation: 220,
        edge_additions: vec![RelationEdgeAddition {
            case_id: "case-discovered".to_owned(),
            document_id: "doc-2".to_owned(),
            window_id: "window-discovered".to_owned(),
            source_entity_id: EntityId("task-1".to_owned()),
            target_entity_id: EntityId("person-1".to_owned()),
            edge_type: "tracks_morale_for".to_owned(),
            confidence_millis: 700,
            evidence_refs: Vec::new(),
            created_at: 220,
        }],
        ..Default::default()
    };
    let mut batch = derive_scope_review_batch(&sample_archives(), None, None, Some(&sidecar));
    run_state_schema_scope(&mut batch, 999);
    let discovered = batch
        .slot_definitions
        .iter()
        .find(|definition| definition.slot_key == "state.tracks_morale_for")
        .expect("discovered definition");
    assert_eq!(
        discovered.lifecycle,
        phoenix_semantic_v2::StateSlotLifecycle::Candidate
    );
}

#[test]
fn persists_state_schema_sidecar_in_overgraph_store() {
    let store_path =
        std::env::temp_dir().join(format!("phoenix-state-schema-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&store_path);
    let store = PhoenixOvergraphStore::open(&store_path).expect("open store");
    let mut batch = derive_scope_review_batch(
        &sample_archives(),
        None,
        None,
        Some(&sample_relation_sidecar()),
    );
    run_state_schema_scope(&mut batch, 999);
    let sidecar = persist_state_schema_patch_sidecar(&store, &batch, 999).expect("persist sidecar");
    let loaded = store
        .load_state_schema_patch_sidecar(&sample_scope())
        .expect("load sidecar")
        .expect("stored sidecar");
    assert_eq!(loaded.summary, sidecar.summary);
    assert_eq!(loaded.slot_definitions, sidecar.slot_definitions);
}
