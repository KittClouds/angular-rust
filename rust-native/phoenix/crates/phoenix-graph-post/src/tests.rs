use phoenix_semantic_v2::{
    MemoryClaimAtom, MemoryClaimStatus, MemoryCompilerSummary, MemoryConflictKind,
    MemoryConflictRecord, MemoryContinuityGapRecord, MemoryGapKind, MemoryModality,
    MemoryScopeSidecar, MemoryStateRecord,
};
use phoenix_types::{BiTemporalWindow, EntityId, ScopeKey};

use crate::compile_graph_projection;

#[test]
fn compile_graph_projection_builds_claim_projection() {
    let sidecar = MemoryScopeSidecar {
        scope: ScopeKey::default(),
        scope_key: "__global__::__global__::__global__::__global__".to_owned(),
        claims: vec![MemoryClaimAtom {
            claim_id: "claim-1".to_owned(),
            document_id: "doc-1".to_owned(),
            slot_key: "entity.employer".to_owned(),
            subject_label: "Alice".to_owned(),
            object_label: "Acme".to_owned(),
            object_value: "Acme".to_owned(),
            status: MemoryClaimStatus::Active,
            modality: MemoryModality::Asserted,
            confidence_millis: 910,
            source_class: "world".to_owned(),
            provenance_label: "direct".to_owned(),
            temporal: BiTemporalWindow::default(),
            ..Default::default()
        }],
        summary: MemoryCompilerSummary::default(),
        ..Default::default()
    };

    let compiled = compile_graph_projection(
        &sidecar.scope_key,
        None,
        None,
        None,
        Some(&sidecar),
        Some(1),
    );

    assert_eq!(
        compiled.graph_batch.scope.scope_key(),
        "projection:__global__::__global__::__global__::__global__"
    );
    assert!(compiled
        .graph_batch
        .vertices
        .iter()
        .any(|vertex| vertex.kind == "claim"));
    assert!(compiled
        .graph_batch
        .edges
        .iter()
        .any(|edge| edge.edge_type.0 == "object"));
}

#[test]
fn compile_graph_projection_enriches_world_state_vertices() {
    let sidecar = MemoryScopeSidecar {
        scope: ScopeKey::default(),
        scope_key: "__global__::__global__::__global__::__global__".to_owned(),
        states: vec![MemoryStateRecord {
            state_id: "state-1".to_owned(),
            entity_id: EntityId("entity-alice".to_owned()),
            slot_key: "entity.employer".to_owned(),
            value: "Acme".to_owned(),
            status: MemoryClaimStatus::Active,
            source_class: "world".to_owned(),
            confidence_millis: 880,
            claim_ids: vec!["claim-1".to_owned()],
            temporal: BiTemporalWindow {
                valid_from: Some(10),
                valid_to: None,
                recorded_from: Some(20),
                recorded_to: None,
            },
            ..Default::default()
        }],
        conflicts: vec![MemoryConflictRecord {
            conflict_id: "conflict-1".to_owned(),
            entity_id: EntityId("entity-alice".to_owned()),
            slot_key: "entity.employer".to_owned(),
            kind: MemoryConflictKind::MutuallyExclusive,
            preferred_claim_id: Some("claim-1".to_owned()),
            status: MemoryClaimStatus::Supported,
            claim_ids: vec!["claim-1".to_owned()],
            temporal: BiTemporalWindow {
                valid_from: Some(10),
                valid_to: None,
                recorded_from: Some(20),
                recorded_to: None,
            },
        }],
        gaps: vec![MemoryContinuityGapRecord {
            gap_id: "gap-1".to_owned(),
            entity_id: EntityId("entity-alice".to_owned()),
            slot_key: "entity.employer".to_owned(),
            kind: MemoryGapKind::UnresolvedConflict,
            status: MemoryClaimStatus::Active,
            detail: "Need review".to_owned(),
            claim_ids: vec!["claim-1".to_owned()],
            temporal: BiTemporalWindow {
                valid_from: Some(10),
                valid_to: None,
                recorded_from: Some(20),
                recorded_to: None,
            },
        }],
        summary: MemoryCompilerSummary::default(),
        ..Default::default()
    };

    let compiled = compile_graph_projection(
        &sidecar.scope_key,
        None,
        None,
        None,
        Some(&sidecar),
        Some(20),
    );

    let state = compiled
        .graph_batch
        .vertices
        .iter()
        .find(|vertex| vertex.kind == "state")
        .expect("state vertex");
    assert_eq!(state.entity_id.as_deref(), Some("entity-alice"));
    assert_eq!(
        state
            .value
            .get("status")
            .and_then(serde_json::Value::as_str),
        Some("active")
    );
    assert_eq!(
        state
            .attributes
            .get("confidenceMillis")
            .and_then(serde_json::Value::as_u64),
        Some(880)
    );

    let conflict = compiled
        .graph_batch
        .vertices
        .iter()
        .find(|vertex| vertex.kind == "conflict")
        .expect("conflict vertex");
    assert_eq!(conflict.entity_id.as_deref(), Some("entity-alice"));
    assert_eq!(
        conflict
            .attributes
            .get("preferredClaimId")
            .and_then(serde_json::Value::as_str),
        Some("claim-1")
    );

    let gap = compiled
        .graph_batch
        .vertices
        .iter()
        .find(|vertex| vertex.kind == "gap")
        .expect("gap vertex");
    assert_eq!(gap.entity_id.as_deref(), Some("entity-alice"));
    assert_eq!(
        gap.value.get("detail").and_then(serde_json::Value::as_str),
        Some("Need review")
    );
}
