use phoenix_graph_kernel::{
    KernelBiTemporal, KernelEdge, KernelEdgeType, KernelGraphLayer, KernelGraphSnapshot,
    KernelProvenance, KernelRelationClass, KernelVertex, KernelVertexClass, KernelVertexId,
};
use serde_json::json;

use crate::retrieval::{
    GraphRetrievedCausalExplanationQueryRequest, GraphRetrievedHistoryQueryRequest,
    GraphRetrievedWorldStateQueryRequest,
};
use crate::retrieval_causal::build_causal_region;
use crate::retrieval_history::build_history_region;
use crate::retrieval_world::build_world_state_region;

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
fn world_state_region_expands_through_contradictory_support_region_edges() {
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
        ],
        asserted_edges: vec![edge("graph::state::1", "graph::entity::alice", "state_of")],
        candidate_edges: vec![KernelEdge {
            layer: KernelGraphLayer::Candidate,
            relation_class: KernelRelationClass::Candidate,
            ..edge(
                "graph::state::1",
                "graph::claim::1",
                "semantic::contradictory_support_region",
            )
        }],
    };
    let request = GraphRetrievedWorldStateQueryRequest {
        query_text: "who employs alice".to_owned(),
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

    let (_, region) = build_world_state_region(&snapshot, &request, &[]);

    assert!(region
        .included_vertex_ids
        .contains(&"graph::claim::1".to_owned()));
    assert_eq!(region.candidate_edge_count, 1);
}

#[test]
fn history_region_expands_through_contradictory_support_region_edges() {
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
        ],
        asserted_edges: vec![edge("graph::state::1", "graph::entity::alice", "state_of")],
        candidate_edges: vec![KernelEdge {
            layer: KernelGraphLayer::Candidate,
            relation_class: KernelRelationClass::Candidate,
            ..edge(
                "graph::state::1",
                "graph::claim::1",
                "semantic::contradictory_support_region",
            )
        }],
    };
    let request = GraphRetrievedHistoryQueryRequest {
        query_text: "employment history for alice".to_owned(),
        entity_id: "alice".to_owned(),
        slot_key: Some("entity.employer".to_owned()),
        since_valid_at: 0,
        ..GraphRetrievedHistoryQueryRequest::default()
    };

    let (_, region) = build_history_region(&snapshot, &request, &[]);

    assert!(region
        .included_vertex_ids
        .contains(&"graph::claim::1".to_owned()));
    assert_eq!(region.candidate_edge_count, 1);
}

#[test]
fn causal_region_does_not_expand_through_contradictory_support_region_edges() {
    let snapshot = KernelGraphSnapshot {
        vertices: vec![
            vertex(
                "graph::event::1",
                "event",
                Some("alice"),
                Some("entity.employer"),
            ),
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
        ],
        asserted_edges: Vec::new(),
        candidate_edges: vec![KernelEdge {
            layer: KernelGraphLayer::Candidate,
            relation_class: KernelRelationClass::Candidate,
            ..edge(
                "graph::state::1",
                "graph::claim::1",
                "semantic::contradictory_support_region",
            )
        }],
    };
    let request = GraphRetrievedCausalExplanationQueryRequest {
        query_text: "what led to alice employer change".to_owned(),
        target_vertex_id: "graph::event::1".to_owned(),
        ..GraphRetrievedCausalExplanationQueryRequest::default()
    };

    let (_, region) = build_causal_region(&snapshot, &request, &[]);

    assert!(!region
        .included_vertex_ids
        .contains(&"graph::claim::1".to_owned()));
    assert_eq!(region.candidate_edge_count, 0);
}

#[test]
fn causal_region_expands_through_missing_intermediate_cause_edges() {
    let snapshot = KernelGraphSnapshot {
        vertices: vec![
            vertex(
                "graph::event::1",
                "event",
                Some("alice"),
                Some("entity.employer"),
            ),
            vertex(
                "graph::event::2",
                "event",
                Some("alice"),
                Some("entity.location"),
            ),
        ],
        asserted_edges: Vec::new(),
        candidate_edges: vec![KernelEdge {
            layer: KernelGraphLayer::Candidate,
            relation_class: KernelRelationClass::Candidate,
            ..edge(
                "graph::event::2",
                "graph::event::1",
                "semantic::missing_intermediate_cause",
            )
        }],
    };
    let request = GraphRetrievedCausalExplanationQueryRequest {
        query_text: "what led to alice employer change".to_owned(),
        target_vertex_id: "graph::event::1".to_owned(),
        ..GraphRetrievedCausalExplanationQueryRequest::default()
    };

    let (_, region) = build_causal_region(&snapshot, &request, &[]);

    assert!(region
        .included_vertex_ids
        .contains(&"graph::event::2".to_owned()));
    assert_eq!(region.candidate_edge_count, 1);
}

#[test]
fn history_region_does_not_expand_through_missing_intermediate_cause_edges() {
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
                "graph::event::2",
                "event",
                Some("alice"),
                Some("entity.location"),
            ),
        ],
        asserted_edges: vec![edge("graph::state::1", "graph::entity::alice", "state_of")],
        candidate_edges: vec![KernelEdge {
            layer: KernelGraphLayer::Candidate,
            relation_class: KernelRelationClass::Candidate,
            ..edge(
                "graph::event::2",
                "graph::state::1",
                "semantic::missing_intermediate_cause",
            )
        }],
    };
    let request = GraphRetrievedHistoryQueryRequest {
        query_text: "employment history for alice".to_owned(),
        entity_id: "alice".to_owned(),
        slot_key: Some("entity.employer".to_owned()),
        since_valid_at: 0,
        ..GraphRetrievedHistoryQueryRequest::default()
    };

    let (_, region) = build_history_region(&snapshot, &request, &[]);

    assert!(!region
        .included_vertex_ids
        .contains(&"graph::event::2".to_owned()));
    assert_eq!(region.candidate_edge_count, 0);
}
