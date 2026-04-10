use phoenix_graph_kernel::KernelVertex;

use crate::api::{
    GraphRankedCausalExplanationAnswer, GraphRankedCausalPath, GraphRankedHistoryCandidate,
    GraphRankedStateCandidate, GraphTruthPlane,
};
use crate::phase4_contract::GraphPhase4RerankScore;
use crate::phase4_scoring::Phase4Scorer;
use crate::phase5_path_rerank::apply_phase5_path_rerank_with_scorer;

struct PathScorer;

impl Phase4Scorer for PathScorer {
    fn score_state_candidate(
        &self,
        _query_text: &str,
        _candidate: &GraphRankedStateCandidate,
    ) -> Option<GraphPhase4RerankScore> {
        None
    }

    fn score_history_candidate(
        &self,
        _query_text: &str,
        _candidate: &GraphRankedHistoryCandidate,
    ) -> Option<GraphPhase4RerankScore> {
        None
    }

    fn score_causal_path(
        &self,
        _query_text: &str,
        candidate: &GraphRankedCausalPath,
    ) -> Option<GraphPhase4RerankScore> {
        let strong = candidate.source_vertex_id.contains("strong");
        Some(GraphPhase4RerankScore {
            model: "stub-path-reranker".to_owned(),
            positive_label: "explains_target".to_owned(),
            positive_score: if strong { 0.95 } else { 0.2 },
            context_label: "background_path".to_owned(),
            context_score: if strong { 0.05 } else { 0.8 },
            negative_label: "speculative_path".to_owned(),
            negative_score: if strong { 0.01 } else { 0.1 },
            applied_delta: if strong { 0.55 } else { -0.1 },
        })
    }

    fn score_event_vertex(
        &self,
        _query_text: &str,
        _vertex: &KernelVertex,
    ) -> Option<GraphPhase4RerankScore> {
        None
    }
}

#[test]
fn path_rerank_runs_after_deterministic_ranking_and_records_baseline() {
    let scorer = PathScorer;
    let mut answer = answer(vec![
        candidate("source:weak", 2.2, true),
        candidate("source:strong", 2.0, true),
    ]);

    apply_phase5_path_rerank_with_scorer("what explains the target", &mut answer, &scorer);

    let selected = answer.selected.as_ref().expect("expected selected path");
    assert_eq!(selected.source_vertex_id, "source:strong");
    let path_rerank = selected.path_rerank.as_ref().expect("expected path rerank");
    assert_eq!(path_rerank.deterministic_rank, 2);
    assert_eq!(path_rerank.deterministic_score, 2.0);
    assert_eq!(path_rerank.applied_delta, 0.55);
    assert!(selected.query_rerank.is_some());
}

#[test]
fn path_rerank_keeps_plane_gate_as_hard_boundary() {
    let scorer = PathScorer;
    let mut answer = answer(vec![
        candidate("source:weak", 2.2, true),
        candidate("source:strong", 2.0, false),
    ]);

    apply_phase5_path_rerank_with_scorer("what explains the target", &mut answer, &scorer);

    assert_eq!(
        answer
            .selected
            .as_ref()
            .map(|candidate| candidate.source_vertex_id.as_str()),
        Some("source:weak")
    );
    assert!(answer.candidates[0].path_rerank.is_some());
}

fn answer(candidates: Vec<GraphRankedCausalPath>) -> GraphRankedCausalExplanationAnswer {
    GraphRankedCausalExplanationAnswer {
        target_vertex_id: "target".to_owned(),
        target_kind: Some("event".to_owned()),
        selected: None,
        candidates,
        abstain: false,
        abstain_reason: None,
    }
}

fn candidate(
    source_vertex_id: &str,
    answer_score: f64,
    plane_allowed: bool,
) -> GraphRankedCausalPath {
    GraphRankedCausalPath {
        target_vertex_id: "target".to_owned(),
        source_vertex_id: source_vertex_id.to_owned(),
        path_vertex_ids: vec![source_vertex_id.to_owned(), "target".to_owned()],
        hops: Vec::new(),
        truth_plane: GraphTruthPlane::WorldState,
        plane_allowed,
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
