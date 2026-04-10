use phoenix_graph_kernel::{
    KernelBiTemporal, KernelProvenance, KernelSlotState, KernelStateChange, KernelStateChangeKind,
    KernelVertex, KernelVertexClass, KernelVertexId,
};
use serde_json::json;

use crate::api::{
    GraphRankedCausalExplanationAnswer, GraphRankedCausalPath, GraphRankedHistoryAnswer,
    GraphRankedHistoryCandidate, GraphRankedSlotAnswer, GraphRankedStateCandidate, GraphTruthPlane,
};
use crate::phase4_contract::GraphPhase4RerankScore;
use crate::phase4_scoring::{
    apply_phase4_causal_with_scorer, apply_phase4_history_with_scorer,
    apply_phase4_world_state_with_scorer, Phase4Scorer,
};

struct StubScorer;

impl Phase4Scorer for StubScorer {
    fn score_state_candidate(
        &self,
        _query_text: &str,
        candidate: &GraphRankedStateCandidate,
    ) -> Option<GraphPhase4RerankScore> {
        Some(score_for_value(candidate.state.value.as_str()))
    }

    fn score_history_candidate(
        &self,
        _query_text: &str,
        candidate: &GraphRankedHistoryCandidate,
    ) -> Option<GraphPhase4RerankScore> {
        Some(score_for_value(candidate.change.state.value.as_str()))
    }

    fn score_causal_path(
        &self,
        _query_text: &str,
        candidate: &GraphRankedCausalPath,
    ) -> Option<GraphPhase4RerankScore> {
        Some(score_for_value(candidate.source_vertex_id.as_str()))
    }

    fn score_event_vertex(
        &self,
        _query_text: &str,
        vertex: &KernelVertex,
    ) -> Option<GraphPhase4RerankScore> {
        Some(score_for_value(vertex.id.0.as_str()))
    }
}

#[test]
fn phase4_world_state_promotes_more_answer_bearing_candidate() {
    let scorer = StubScorer;
    let mut answer = GraphRankedSlotAnswer {
        entity_id: "ryan".to_owned(),
        slot_key: "entity.location".to_owned(),
        selected: None,
        candidates: vec![
            state_candidate("graph::state::1", "rome", 2.4, true),
            state_candidate("graph::state::2", "new_rome", 2.1, true),
        ],
        conflicts: Vec::new(),
        gaps: Vec::new(),
        abstain: false,
        abstain_reason: None,
    };

    apply_phase4_world_state_with_scorer("current location for ryan", &mut answer, &scorer);

    assert_eq!(
        answer
            .selected
            .as_ref()
            .map(|candidate| candidate.state.state_vertex_id.as_str()),
        Some("graph::state::2")
    );
    assert!(answer.candidates[0].query_rerank.is_some());
}

#[test]
fn phase4_world_state_respects_plane_gate_even_with_high_model_score() {
    let scorer = StubScorer;
    let mut answer = GraphRankedSlotAnswer {
        entity_id: "ryan".to_owned(),
        slot_key: "entity.location".to_owned(),
        selected: None,
        candidates: vec![
            state_candidate("graph::state::1", "rome", 2.2, true),
            state_candidate("graph::state::2", "new_rome", 2.7, false),
        ],
        conflicts: Vec::new(),
        gaps: Vec::new(),
        abstain: false,
        abstain_reason: None,
    };

    apply_phase4_world_state_with_scorer("current location for ryan", &mut answer, &scorer);

    assert_eq!(
        answer
            .selected
            .as_ref()
            .map(|candidate| candidate.state.state_vertex_id.as_str()),
        Some("graph::state::1")
    );
}

#[test]
fn phase4_history_and_causal_apply_rerank_diagnostics() {
    let scorer = StubScorer;
    let mut history = GraphRankedHistoryAnswer {
        entity_id: "ryan".to_owned(),
        slot_key: Some("entity.location".to_owned()),
        window_start_ms: 0,
        window_end_ms: 10,
        selected: None,
        candidates: vec![
            history_candidate("graph::state::1", "rome", 2.3),
            history_candidate("graph::state::2", "new_rome", 2.0),
        ],
        conflicts: Vec::new(),
        gaps: Vec::new(),
        abstain: false,
        abstain_reason: None,
    };
    let mut causal = GraphRankedCausalExplanationAnswer {
        target_vertex_id: "graph::event::target".to_owned(),
        target_kind: Some("event".to_owned()),
        selected: None,
        candidates: vec![
            causal_candidate("source:weak", 2.2),
            causal_candidate("source:strong", 2.0),
        ],
        abstain: false,
        abstain_reason: None,
    };

    apply_phase4_history_with_scorer("history of location for ryan", &mut history, &scorer);
    apply_phase4_causal_with_scorer("what led to the battle", &[], &mut causal, &scorer);

    assert_eq!(
        history
            .selected
            .as_ref()
            .map(|candidate| candidate.change.state.state_vertex_id.as_str()),
        Some("graph::state::2")
    );
    assert_eq!(
        causal
            .selected
            .as_ref()
            .map(|candidate| candidate.source_vertex_id.as_str()),
        Some("source:strong")
    );
    assert!(history.candidates[0].query_rerank.is_some());
    assert!(causal.candidates[0].query_rerank.is_some());
}

#[test]
fn phase4_causal_event_scoring_boosts_main_event_path() {
    let scorer = StubScorer;
    let vertices = vec![
        event_vertex("graph::event::weak"),
        event_vertex("graph::event::strong"),
        event_vertex("graph::event::target"),
    ];
    let mut causal = GraphRankedCausalExplanationAnswer {
        target_vertex_id: "graph::event::target".to_owned(),
        target_kind: Some("event".to_owned()),
        selected: None,
        candidates: vec![
            causal_candidate_with_path(
                "source:path-weak",
                vec!["graph::event::weak", "graph::event::target"],
                2.2,
            ),
            causal_candidate_with_path(
                "source:path-strong",
                vec!["graph::event::strong", "graph::event::target"],
                2.0,
            ),
        ],
        abstain: false,
        abstain_reason: None,
    };

    apply_phase4_causal_with_scorer("what led to the battle", &vertices, &mut causal, &scorer);

    assert_eq!(
        causal
            .selected
            .as_ref()
            .map(|candidate| candidate.source_vertex_id.as_str()),
        Some("source:path-strong")
    );
    assert!(causal.candidates[0].event_rerank.is_some());
}

fn state_candidate(
    state_vertex_id: &str,
    value: &str,
    answer_score: f64,
    plane_allowed: bool,
) -> GraphRankedStateCandidate {
    GraphRankedStateCandidate {
        state: KernelSlotState {
            state_vertex_id: state_vertex_id.to_owned(),
            entity_id: "ryan".to_owned(),
            slot_key: "entity.location".to_owned(),
            value: value.to_owned(),
            value_entity_id: None,
            status: Some("active".to_owned()),
            source_class: Some("memory".to_owned()),
            confidence: Some(0.9),
            temporal: KernelBiTemporal::default(),
            supporting_claim_ids: Vec::new(),
            evidence_refs: Vec::new(),
        },
        truth_plane: GraphTruthPlane::WorldState,
        plane_allowed,
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

fn history_candidate(
    state_vertex_id: &str,
    value: &str,
    answer_score: f64,
) -> GraphRankedHistoryCandidate {
    GraphRankedHistoryCandidate {
        change: KernelStateChange {
            change_kind: KernelStateChangeKind::Activated,
            state: KernelSlotState {
                state_vertex_id: state_vertex_id.to_owned(),
                entity_id: "ryan".to_owned(),
                slot_key: "entity.location".to_owned(),
                value: value.to_owned(),
                value_entity_id: None,
                status: Some("active".to_owned()),
                source_class: Some("memory".to_owned()),
                confidence: Some(0.9),
                temporal: KernelBiTemporal::default(),
                supporting_claim_ids: Vec::new(),
                evidence_refs: Vec::new(),
            },
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

fn causal_candidate(source_vertex_id: &str, answer_score: f64) -> GraphRankedCausalPath {
    GraphRankedCausalPath {
        target_vertex_id: "graph::event::target".to_owned(),
        source_vertex_id: source_vertex_id.to_owned(),
        path_vertex_ids: vec![
            source_vertex_id.to_owned(),
            "graph::event::target".to_owned(),
        ],
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

fn causal_candidate_with_path(
    source_vertex_id: &str,
    path_vertex_ids: Vec<&str>,
    answer_score: f64,
) -> GraphRankedCausalPath {
    GraphRankedCausalPath {
        path_vertex_ids: path_vertex_ids.into_iter().map(str::to_owned).collect(),
        ..causal_candidate(source_vertex_id, answer_score)
    }
}

fn event_vertex(id: &str) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(id.to_owned()),
        kind: "event".to_owned(),
        class: KernelVertexClass::Event,
        labels: vec!["event".to_owned()],
        weight: 1,
        value: json!({"kind": "event"}),
        attributes: json!({}),
        temporal: KernelBiTemporal::default(),
        provenance: KernelProvenance::default(),
        entity_id: None,
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

fn score_for_value(key: &str) -> GraphPhase4RerankScore {
    let positive_score = if key.contains("new_rome") || key.contains("strong") {
        0.95
    } else {
        0.15
    };
    let negative_score = if positive_score > 0.8 { 0.05 } else { 0.65 };
    GraphPhase4RerankScore {
        model: "stub".to_owned(),
        positive_label: "direct_answer".to_owned(),
        positive_score,
        context_label: "supporting_context".to_owned(),
        context_score: 0.2,
        negative_label: "unsafe_answer".to_owned(),
        negative_score,
        applied_delta: (positive_score * 0.58) + 0.02 - (negative_score * 0.52),
    }
}
