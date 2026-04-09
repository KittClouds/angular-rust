use phoenix_graph_kernel::{
    KernelBiTemporal, KernelEdge, KernelEdgeType, KernelGraphLayer, KernelGraphSnapshot,
    KernelProvenance, KernelRelationClass, KernelVertex, KernelVertexClass, KernelVertexId,
};
use serde_json::json;

use crate::retrieval::{
    GraphRetrievedCausalExplanationQueryRequest, GraphRetrievedHistoryQueryRequest,
    GraphRetrievedSeed, GraphRetrievedWorldStateQueryRequest,
};
use crate::retrieval_causal::build_causal_region;
use crate::retrieval_common::{kernel_from_snapshot, score_from_distance};
use crate::retrieval_history::build_history_region;
use crate::retrieval_world::build_world_state_region;
use phoenix_types::ScopeKey;

fn vertex(id: &str, kind: &str, entity_id: Option<&str>, slot_key: Option<&str>) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(id.to_owned()),
        kind: kind.to_owned(),
        class: match kind {
            "state" => KernelVertexClass::State,
            "entity" => KernelVertexClass::Entity,
            _ => KernelVertexClass::Generic,
        },
        labels: Vec::new(),
        weight: 1,
        value: json!({"slotKey": slot_key}),
        attributes: json!({}),
        temporal: KernelBiTemporal::default(),
        provenance: KernelProvenance::default(),
        entity_id: entity_id.map(str::to_owned),
        search_chunk_id: None,
        document_id: None,
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: None,
    }
}

fn edge(source: &str, target: &str, edge_type: &str) -> KernelEdge {
    KernelEdge {
        source_id: KernelVertexId(source.to_owned()),
        target_id: KernelVertexId(target.to_owned()),
        edge_type: KernelEdgeType(edge_type.to_owned()),
        relation_class: KernelRelationClass::Memory,
        weight: 1,
        attributes: json!({}),
        data: None,
        document_id: None,
        narrative_id: None,
        layer: KernelGraphLayer::Asserted,
        temporal: KernelBiTemporal::default(),
        provenance: KernelProvenance::default(),
        resolution_facet: None,
    }
}

#[test]
fn world_state_region_keeps_entity_slot_anchor_and_support_chain() {
    let snapshot = KernelGraphSnapshot {
        vertices: vec![
            vertex("graph::entity::alice", "entity", Some("alice"), None),
            vertex(
                "graph::state::1",
                "state",
                Some("alice"),
                Some("entity.employer"),
            ),
            vertex(
                "graph::claim::1",
                "claim",
                Some("alice"),
                Some("entity.employer"),
            ),
            vertex(
                "graph::claim::2",
                "claim",
                Some("alice"),
                Some("entity.location"),
            ),
            vertex("graph::chunk::1", "chunk", None, None),
        ],
        asserted_edges: vec![
            edge("graph::state::1", "graph::entity::alice", "state_of"),
            edge("graph::state::1", "graph::claim::1", "supported_by"),
            edge(
                "graph::claim::1",
                "graph::chunk::1",
                "semantic::chunk_neighbor",
            ),
            edge("graph::claim::2", "graph::entity::alice", "subject"),
        ],
        candidate_edges: Vec::new(),
    };
    let request = GraphRetrievedWorldStateQueryRequest {
        query_text: "where does alice work".to_owned(),
        entity_id: "alice".to_owned(),
        slot_key: "entity.employer".to_owned(),
        valid_at: None,
        recorded_at: None,
        include_candidate_graph: true,
        seed_limit: 6,
        oversample: 12,
        expansion_hops: 2,
        region_node_limit: 12,
    };
    let seeds = vec![GraphRetrievedSeed {
        node_id: "graph::claim::1".to_owned(),
        node_kind: "claim".to_owned(),
        score_millis: 930,
        distance_millis: 70,
        document_id: None,
        narrative_id: None,
        evidence_refs: Vec::new(),
    }];

    let (_, region) = build_world_state_region(&snapshot, &request, &seeds);

    assert!(region
        .included_vertex_ids
        .contains(&"graph::state::1".to_owned()));
    assert!(region
        .included_vertex_ids
        .contains(&"graph::claim::1".to_owned()));
    assert!(region
        .included_vertex_ids
        .contains(&"graph::entity::alice".to_owned()));
    assert!(!region
        .included_vertex_ids
        .contains(&"graph::claim::2".to_owned()));
    assert!(!region.truncated);
}

#[test]
fn semantic_score_from_distance_is_monotonic() {
    assert!(score_from_distance(0.0) > score_from_distance(0.5));
    assert!(score_from_distance(0.5) > score_from_distance(1.5));
}

#[test]
fn history_region_keeps_state_changes_and_issues_for_same_entity_slot() {
    let mut state_old = vertex(
        "graph::state::old",
        "state",
        Some("alice"),
        Some("entity.employer"),
    );
    state_old.temporal.valid_from = Some(10);
    let mut state_new = vertex(
        "graph::state::new",
        "state",
        Some("alice"),
        Some("entity.employer"),
    );
    state_new.temporal.valid_from = Some(20);
    let conflict = KernelVertex {
        value: json!({"kind": "contradiction", "status": "open", "slotKey": "entity.employer"}),
        attributes: json!({"claimIds": ["claim-1"], "reason": "conflict", "preferredClaimId": "claim-1"}),
        ..vertex(
            "graph::conflict::1",
            "conflict",
            Some("alice"),
            Some("entity.employer"),
        )
    };
    let snapshot = KernelGraphSnapshot {
        vertices: vec![
            vertex("graph::entity::alice", "entity", Some("alice"), None),
            state_old,
            state_new,
            conflict,
            vertex(
                "graph::state::other",
                "state",
                Some("alice"),
                Some("entity.location"),
            ),
        ],
        asserted_edges: vec![
            edge("graph::state::old", "graph::entity::alice", "state_of"),
            edge("graph::state::new", "graph::entity::alice", "state_of"),
            edge("graph::conflict::1", "graph::state::new", "about"),
        ],
        candidate_edges: Vec::new(),
    };
    let request = GraphRetrievedHistoryQueryRequest {
        query_text: "employment history for alice".to_owned(),
        entity_id: "alice".to_owned(),
        slot_key: Some("entity.employer".to_owned()),
        since_valid_at: 0,
        ..GraphRetrievedHistoryQueryRequest::default()
    };
    let seeds = vec![GraphRetrievedSeed {
        node_id: "graph::state::new".to_owned(),
        node_kind: "state".to_owned(),
        score_millis: 900,
        distance_millis: 100,
        document_id: None,
        narrative_id: None,
        evidence_refs: Vec::new(),
    }];

    let (_, region) = build_history_region(&snapshot, &request, &seeds);

    assert!(region
        .included_vertex_ids
        .contains(&"graph::state::old".to_owned()));
    assert!(region
        .included_vertex_ids
        .contains(&"graph::state::new".to_owned()));
    assert!(region
        .included_vertex_ids
        .contains(&"graph::conflict::1".to_owned()));
    assert!(!region
        .included_vertex_ids
        .contains(&"graph::state::other".to_owned()));
}

#[test]
fn causal_region_keeps_target_support_and_semantic_neighbors() {
    let snapshot = KernelGraphSnapshot {
        vertices: vec![
            vertex(
                "graph::event::1",
                "event",
                Some("alice"),
                Some("entity.employer"),
            ),
            vertex(
                "graph::claim::1",
                "claim",
                Some("alice"),
                Some("entity.employer"),
            ),
            vertex(
                "graph::event::2",
                "event",
                Some("alice"),
                Some("entity.employer"),
            ),
            vertex("graph::entity::alice", "entity", Some("alice"), None),
            vertex("graph::chunk::1", "chunk", None, None),
        ],
        asserted_edges: vec![
            edge("graph::event::2", "graph::event::1", "causal_link"),
            edge("graph::claim::1", "graph::event::1", "supported_by"),
            edge("graph::event::1", "graph::entity::alice", "subject"),
            edge(
                "graph::claim::1",
                "graph::chunk::1",
                "semantic::claim_support",
            ),
        ],
        candidate_edges: Vec::new(),
    };
    let request = GraphRetrievedCausalExplanationQueryRequest {
        query_text: "what led to alice employer change".to_owned(),
        target_vertex_id: "graph::event::1".to_owned(),
        ..GraphRetrievedCausalExplanationQueryRequest::default()
    };
    let seeds = vec![GraphRetrievedSeed {
        node_id: "graph::claim::1".to_owned(),
        node_kind: "claim".to_owned(),
        score_millis: 880,
        distance_millis: 120,
        document_id: None,
        narrative_id: None,
        evidence_refs: Vec::new(),
    }];

    let (_, region) = build_causal_region(&snapshot, &request, &seeds);

    assert!(region
        .included_vertex_ids
        .contains(&"graph::event::1".to_owned()));
    assert!(region
        .included_vertex_ids
        .contains(&"graph::event::2".to_owned()));
    assert!(region
        .included_vertex_ids
        .contains(&"graph::claim::1".to_owned()));
    assert!(region
        .included_vertex_ids
        .contains(&"graph::entity::alice".to_owned()));
}

#[test]
fn kernel_from_snapshot_accepts_candidate_edges_for_region_rebuilds() {
    let snapshot = KernelGraphSnapshot {
        vertices: vec![
            vertex(
                "graph::claim::1",
                "claim",
                Some("alice"),
                Some("entity.employer"),
            ),
            vertex(
                "graph::claim::2",
                "claim",
                Some("alice"),
                Some("entity.employer"),
            ),
        ],
        asserted_edges: Vec::new(),
        candidate_edges: vec![KernelEdge {
            layer: KernelGraphLayer::Candidate,
            ..edge(
                "graph::claim::1",
                "graph::claim::2",
                "semantic::claim_support",
            )
        }],
    };

    let kernel = kernel_from_snapshot(&ScopeKey::default(), &snapshot).expect("region kernel");
    let view = kernel.view_as_of(phoenix_graph_kernel::KernelViewRequest {
        valid_at: None,
        recorded_at: None,
        include_candidate_graph: true,
    });

    assert_eq!(view.candidate_edges.len(), 1);
}
