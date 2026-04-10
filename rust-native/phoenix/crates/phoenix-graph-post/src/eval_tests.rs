use phoenix_graph_kernel::{
    KernelBiTemporal, KernelEdge, KernelEdgeType, KernelGraphLayer, KernelGraphSnapshot,
    KernelProvenance, KernelRelationClass, KernelVertex, KernelVertexClass, KernelVertexId,
};
use serde_json::json;

use crate::eval::{filter_snapshot_for_families, GraphSoftFamily};
use crate::retrieval::GraphRetrievedWorldStateQueryRequest;
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

fn candidate_edge(source: &str, target: &str, edge_type: &str) -> KernelEdge {
    KernelEdge {
        source_id: KernelVertexId(source.to_owned()),
        target_id: KernelVertexId(target.to_owned()),
        edge_type: KernelEdgeType(edge_type.to_owned()),
        relation_class: KernelRelationClass::Candidate,
        weight: 1,
        attributes: json!({}),
        data: None,
        document_id: None,
        narrative_id: None,
        layer: KernelGraphLayer::Candidate,
        temporal: KernelBiTemporal::default(),
        provenance: KernelProvenance::default(),
        resolution_facet: None,
    }
}

#[test]
fn filter_snapshot_only_removes_targeted_soft_families() {
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
        asserted_edges: Vec::new(),
        candidate_edges: vec![
            candidate_edge(
                "graph::state::1",
                "graph::claim::1",
                "semantic::same_slot_family",
            ),
            candidate_edge(
                "graph::state::1",
                "graph::claim::1",
                "semantic::chunk_neighbor",
            ),
            candidate_edge("graph::state::1", "graph::claim::1", "claim_support"),
        ],
    };

    let filtered =
        filter_snapshot_for_families(&snapshot, &[GraphSoftFamily::ContradictorySupportRegion]);

    assert_eq!(filtered.candidate_edges.len(), 2);
    assert!(filtered
        .candidate_edges
        .iter()
        .all(|edge| edge.edge_type.0 != "semantic::same_slot_family"));
}

#[test]
fn no_soft_ablation_removes_same_slot_family_region_widening() {
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
        asserted_edges: Vec::new(),
        candidate_edges: vec![candidate_edge(
            "graph::state::1",
            "graph::claim::1",
            "semantic::same_slot_family",
        )],
    };
    let request = GraphRetrievedWorldStateQueryRequest {
        query_text: "current employer for alice".to_owned(),
        entity_id: "alice".to_owned(),
        slot_key: "entity.employer".to_owned(),
        valid_at: None,
        recorded_at: None,
        include_candidate_graph: true,
        seed_limit: 4,
        oversample: 8,
        expansion_hops: 2,
        region_node_limit: 12,
    };

    let hard_only = filter_snapshot_for_families(&snapshot, &[]);
    let (_, hard_region) = build_world_state_region(&hard_only, &request, &[]);
    let full_soft = filter_snapshot_for_families(&snapshot, &[GraphSoftFamily::SameSlotFamily]);
    let (_, full_region) = build_world_state_region(&full_soft, &request, &[]);

    assert_eq!(hard_region.candidate_edge_count, 0);
    assert_eq!(full_region.candidate_edge_count, 1);
    assert!(full_region
        .included_vertex_ids
        .contains(&"graph::claim::1".to_owned()));
}
