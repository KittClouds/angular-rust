use phoenix_graph_kernel::{
    KernelBiTemporal, KernelEdge, KernelEdgeType, KernelGraphLayer, KernelGraphSnapshot,
    KernelProvenance, KernelSlotState, KernelStateChange, KernelStateChangeKind, KernelVertex,
    KernelVertexClass, KernelVertexId,
};
use serde_json::json;

use crate::api::{
    GraphRankedCausalExplanationAnswer, GraphRankedCausalPath, GraphRankedHistoryAnswer,
    GraphRankedHistoryCandidate, GraphRankedSlotAnswer, GraphRankedStateCandidate, GraphTruthPlane,
};
use crate::phase4_graph_scoring::{
    apply_graph_structural_causal, apply_graph_structural_history,
    apply_graph_structural_world_state,
};

#[test]
fn structural_world_state_promotes_anchor_connected_state() {
    let snapshot = connected_snapshot();
    let mut answer = GraphRankedSlotAnswer {
        entity_id: "entity:ryan".to_owned(),
        slot_key: "entity.location".to_owned(),
        selected: None,
        candidates: vec![
            state_candidate("graph::state::connected", 2.0),
            state_candidate("graph::state::disconnected", 2.1),
        ],
        conflicts: Vec::new(),
        gaps: Vec::new(),
        abstain: false,
        abstain_reason: None,
    };

    apply_graph_structural_world_state(&["graph::entity::ryan".to_owned()], &snapshot, &mut answer);

    assert_eq!(
        answer
            .selected
            .as_ref()
            .map(|value| value.state.state_vertex_id.as_str()),
        Some("graph::state::connected")
    );
    assert!(answer.candidates[0].graph_structural_rerank.is_some());
}

#[test]
fn structural_history_promotes_anchor_connected_change() {
    let snapshot = connected_snapshot();
    let mut answer = GraphRankedHistoryAnswer {
        entity_id: "entity:ryan".to_owned(),
        slot_key: Some("entity.location".to_owned()),
        window_start_ms: 0,
        window_end_ms: 10,
        selected: None,
        candidates: vec![
            history_candidate("graph::state::connected", 2.0),
            history_candidate("graph::state::disconnected", 2.1),
        ],
        conflicts: Vec::new(),
        gaps: Vec::new(),
        abstain: false,
        abstain_reason: None,
    };

    apply_graph_structural_history(&["graph::entity::ryan".to_owned()], &snapshot, &mut answer);

    assert_eq!(
        answer
            .selected
            .as_ref()
            .map(|value| value.change.state.state_vertex_id.as_str()),
        Some("graph::state::connected")
    );
    assert!(answer.candidates[0].graph_structural_rerank.is_some());
}

#[test]
fn structural_causal_promotes_path_in_target_component() {
    let snapshot = connected_snapshot();
    let mut answer = GraphRankedCausalExplanationAnswer {
        target_vertex_id: "graph::event::target".to_owned(),
        target_kind: Some("event".to_owned()),
        selected: None,
        candidates: vec![
            causal_candidate(
                "graph::event::cause",
                vec!["graph::event::cause", "graph::event::target"],
                2.0,
            ),
            causal_candidate(
                "graph::event::far",
                vec!["graph::event::far", "graph::event::target"],
                2.1,
            ),
        ],
        abstain: false,
        abstain_reason: None,
    };

    apply_graph_structural_causal(&["graph::event::target".to_owned()], &snapshot, &mut answer);

    assert_eq!(
        answer
            .selected
            .as_ref()
            .map(|value| value.source_vertex_id.as_str()),
        Some("graph::event::cause")
    );
    assert!(answer.candidates[0].graph_structural_rerank.is_some());
}

fn connected_snapshot() -> KernelGraphSnapshot {
    KernelGraphSnapshot {
        vertices: vec![
            vertex("graph::entity::ryan", "entity", Some("entity:ryan")),
            vertex("graph::state::connected", "state", Some("entity:ryan")),
            vertex("graph::state::disconnected", "state", Some("entity:ryan")),
            vertex("graph::event::target", "event", None),
            vertex("graph::event::cause", "event", None),
            vertex("graph::event::far", "event", None),
        ],
        asserted_edges: vec![
            edge("graph::entity::ryan", "graph::state::connected", "state_of"),
            edge("graph::event::cause", "graph::event::target", "causal_link"),
        ],
        candidate_edges: Vec::new(),
    }
}

fn vertex(id: &str, kind: &str, entity_id: Option<&str>) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(id.to_owned()),
        kind: kind.to_owned(),
        class: KernelVertexClass::default(),
        labels: Vec::new(),
        weight: 1,
        value: json!({}),
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
        relation_class: Default::default(),
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

fn state_candidate(state_vertex_id: &str, answer_score: f64) -> GraphRankedStateCandidate {
    GraphRankedStateCandidate {
        state: KernelSlotState {
            state_vertex_id: state_vertex_id.to_owned(),
            entity_id: "entity:ryan".to_owned(),
            slot_key: "entity.location".to_owned(),
            value: state_vertex_id.to_owned(),
            value_entity_id: None,
            status: Some("active".to_owned()),
            source_class: Some("memory".to_owned()),
            confidence: Some(0.9),
            temporal: KernelBiTemporal::default(),
            supporting_claim_ids: Vec::new(),
            evidence_refs: Vec::new(),
        },
        truth_plane: GraphTruthPlane::WorldState,
        plane_allowed: true,
        answer_score,
        plane_gate: 1.0,
        status_prior: 1.0,
        support_strength: 0.9,
        temporal_fitness: 1.0,
        conflict_penalty: 0.0,
        gap_penalty: 0.0,
        contradiction_region_penalty: 0.0,
        speculative_penalty: 0.0,
        relevant_conflict_count: 0,
        relevant_gap_count: 0,
        contradiction_region_count: 0,
        supporting_modalities: Vec::new(),
        supporting_source_classes: Vec::new(),
        query_rerank: None,
        graph_structural_rerank: None,
    }
}

fn history_candidate(state_vertex_id: &str, answer_score: f64) -> GraphRankedHistoryCandidate {
    GraphRankedHistoryCandidate {
        change: KernelStateChange {
            change_kind: KernelStateChangeKind::Activated,
            state: state_candidate(state_vertex_id, answer_score).state,
        },
        truth_plane: GraphTruthPlane::WorldState,
        plane_allowed: true,
        answer_score,
        plane_gate: 1.0,
        status_prior: 1.0,
        support_strength: 0.9,
        temporal_fitness: 1.0,
        recency_score: 0.9,
        conflict_penalty: 0.0,
        gap_penalty: 0.0,
        contradiction_region_penalty: 0.0,
        speculative_penalty: 0.0,
        relevant_conflict_count: 0,
        relevant_gap_count: 0,
        contradiction_region_count: 0,
        supporting_modalities: Vec::new(),
        supporting_source_classes: Vec::new(),
        query_rerank: None,
        graph_structural_rerank: None,
    }
}

fn causal_candidate(
    source_vertex_id: &str,
    path_vertex_ids: Vec<&str>,
    answer_score: f64,
) -> GraphRankedCausalPath {
    GraphRankedCausalPath {
        target_vertex_id: "graph::event::target".to_owned(),
        source_vertex_id: source_vertex_id.to_owned(),
        path_vertex_ids: path_vertex_ids.into_iter().map(str::to_owned).collect(),
        hops: Vec::new(),
        truth_plane: GraphTruthPlane::WorldState,
        plane_allowed: true,
        answer_score,
        plane_gate: 1.0,
        path_stability: 0.9,
        support_strength: 0.9,
        temporal_fitness: 0.9,
        depth_penalty: 0.0,
        speculative_penalty: 0.0,
        supporting_modalities: Vec::new(),
        evidence_refs: Vec::new(),
        path_rerank: None,
        query_rerank: None,
        event_rerank: None,
        graph_structural_rerank: None,
    }
}
