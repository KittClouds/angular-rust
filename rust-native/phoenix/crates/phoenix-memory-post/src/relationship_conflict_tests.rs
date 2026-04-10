use crate::derive_scope_review_batch;
use phoenix_semantic_v2::{
    scope_storage_key, DocumentArchive, DocumentManifest, RelationJudgmentKind,
    RelationJudgmentRecord, RelationScopePatchSidecar, ScopeOrd, SemanticEntityRecord,
    SessionArchive,
};
use phoenix_types::{EntityId, EntityKind, IngestDocumentSummary, ScopeKey, SessionDocumentState};

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
        scope_ord: ScopeOrd(9),
        revision: 1,
        title: "Relationship Conflict".to_owned(),
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

fn sample_archive() -> DocumentArchive {
    DocumentArchive {
        manifest: sample_manifest("doc-1", 100),
        entities: vec![
            SemanticEntityRecord {
                entity_id: EntityId("e1".to_owned()),
                canonical_name: "Alice".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                mention_count: 2,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
            SemanticEntityRecord {
                entity_id: EntityId("e2".to_owned()),
                canonical_name: "Bob".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                mention_count: 1,
                chunk_ids: vec!["chunk-1".to_owned()],
            },
        ],
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

fn relationship_conflict_sidecar() -> RelationScopePatchSidecar {
    let scope = sample_scope();
    RelationScopePatchSidecar {
        scope: scope.clone(),
        scope_key: scope_storage_key(&scope),
        scope_ord: Some(ScopeOrd(9)),
        session_id: Some(phoenix_types::SessionId("session-1".to_owned())),
        updated_at: 300,
        generation: 300,
        support_judgments: vec![RelationJudgmentRecord {
            case_id: "case-support".to_owned(),
            document_id: "doc-1".to_owned(),
            window_id: "window-1".to_owned(),
            source_entity_id: EntityId("e1".to_owned()),
            target_entity_id: EntityId("e2".to_owned()),
            edge_type: "commands".to_owned(),
            kind: RelationJudgmentKind::Support,
            confidence_millis: 910,
            evidence_refs: vec!["ev:support".to_owned()],
            created_at: 310,
        }],
        contradiction_judgments: vec![RelationJudgmentRecord {
            case_id: "case-contradict".to_owned(),
            document_id: "doc-1".to_owned(),
            window_id: "window-2".to_owned(),
            source_entity_id: EntityId("e1".to_owned()),
            target_entity_id: EntityId("e2".to_owned()),
            edge_type: "commands".to_owned(),
            kind: RelationJudgmentKind::Contradict,
            confidence_millis: 780,
            evidence_refs: vec!["ev:contradict".to_owned()],
            created_at: 320,
        }],
        ..Default::default()
    }
}

#[test]
fn relationship_ledgers_emit_conflicts_and_gaps() {
    let batch = derive_scope_review_batch(
        &[sample_archive()],
        Some(&sample_session()),
        None,
        None,
        None,
        Some(&relationship_conflict_sidecar()),
        None,
    );

    let conflict = batch
        .conflicts
        .iter()
        .find(|row| {
            row.conflict_id
                .contains("relationship:relationship:commands:e1:e2")
        })
        .expect("relationship conflict");
    assert_eq!(conflict.slot_key, "relation.commands");
    assert_eq!(conflict.claim_ids.len(), 2);

    assert!(batch.gaps.iter().any(|gap| {
        gap.gap_id
            .contains("relationship:relationship:commands:e1:e2")
            && gap.claim_ids.len() == 2
    }));
    assert!(batch.entity_cards.iter().any(|card| {
        card.entity_id.0 == "e1"
            && card
                .active_conflicts
                .iter()
                .any(|row| row.conflict_id == conflict.conflict_id)
    }));
}
