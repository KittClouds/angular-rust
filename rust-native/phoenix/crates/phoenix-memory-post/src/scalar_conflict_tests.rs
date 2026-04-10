use crate::derive_scope_review_batch;
use phoenix_semantic_v2::{
    scope_storage_key, DocumentArchive, DocumentManifest, MemoryConflictKind, RelationEdgeAddition,
    RelationScopePatchSidecar, ScopeOrd, SemanticEntityRecord, SessionArchive,
};
use phoenix_types::{EntityId, EntityKind, IngestDocumentSummary, ScopeKey, SessionDocumentState};

fn scope() -> ScopeKey {
    ScopeKey {
        world_id: Some("world".to_owned()),
        narrative_id: Some("narr".to_owned()),
        folder_id: None,
        folder_path: None,
    }
}

fn manifest(document_id: &str, created_at: i64) -> DocumentManifest {
    let scope = scope();
    DocumentManifest {
        document_id: document_id.to_owned(),
        scope_key: scope_storage_key(&scope),
        scope,
        scope_ord: ScopeOrd(11),
        revision: 1,
        title: "Scalar Conflict".to_owned(),
        created_at,
        session_id: Some(phoenix_types::SessionId("session-1".to_owned())),
        session_document: SessionDocumentState {
            document_id: phoenix_types::DocumentId(document_id.to_owned()),
            ..Default::default()
        },
        document_summary: IngestDocumentSummary::default(),
        ..Default::default()
    }
}

fn archive() -> DocumentArchive {
    DocumentArchive {
        manifest: manifest("doc-1", 100),
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
                canonical_name: "Acme".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Organization),
                mention_count: 1,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e3".to_owned()),
                canonical_name: "Globex".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Organization),
                mention_count: 1,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
        ],
        ..Default::default()
    }
}

fn session() -> SessionArchive {
    SessionArchive {
        session_id: phoenix_types::SessionId("session-1".to_owned()),
        updated_at: 150,
        ..Default::default()
    }
}

fn competing_relation_sidecar() -> RelationScopePatchSidecar {
    let scope = scope();
    RelationScopePatchSidecar {
        scope: scope.clone(),
        scope_key: scope_storage_key(&scope),
        scope_ord: Some(ScopeOrd(11)),
        session_id: Some(phoenix_types::SessionId("session-1".to_owned())),
        updated_at: 300,
        generation: 300,
        edge_additions: vec![
            RelationEdgeAddition {
                case_id: "case-acme".to_owned(),
                document_id: "doc-1".to_owned(),
                window_id: "window-1".to_owned(),
                source_entity_id: EntityId("e1".to_owned()),
                target_entity_id: EntityId("e2".to_owned()),
                edge_type: "works_for".to_owned(),
                confidence_millis: 840,
                evidence_refs: vec!["ev:acme".to_owned()],
                created_at: 300,
            },
            RelationEdgeAddition {
                case_id: "case-globex".to_owned(),
                document_id: "doc-1".to_owned(),
                window_id: "window-2".to_owned(),
                source_entity_id: EntityId("e1".to_owned()),
                target_entity_id: EntityId("e3".to_owned()),
                edge_type: "works_for".to_owned(),
                confidence_millis: 820,
                evidence_refs: vec!["ev:globex".to_owned()],
                created_at: 305,
            },
        ],
        ..Default::default()
    }
}

#[test]
fn overlapping_current_scalar_values_emit_conflict_pressure() {
    let batch = derive_scope_review_batch(
        &[archive()],
        Some(&session()),
        None,
        None,
        None,
        Some(&competing_relation_sidecar()),
        None,
    );

    assert!(batch
        .states
        .iter()
        .any(|state| state.slot_key == "entity.employer"));
    assert!(batch.conflicts.iter().any(|conflict| {
        conflict.slot_key == "entity.employer"
            && conflict.kind == MemoryConflictKind::TemporalOverlap
            && conflict.claim_ids.len() == 2
    }));
    assert!(batch.gaps.iter().any(|gap| {
        gap.gap_id.contains("gap:competitive:e1:entity.employer") && gap.claim_ids.len() == 2
    }));
}
