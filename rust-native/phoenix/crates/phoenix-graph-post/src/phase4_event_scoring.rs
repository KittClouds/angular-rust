use phoenix_graph_kernel::KernelVertex;
use rustc_hash::FxHashMap;

use crate::api::{GraphRankedCausalExplanationAnswer, GraphRankedCausalPath};
use crate::phase4_contract::GraphPhase4RerankScore;
use crate::phase4_scoring::Phase4Scorer;
use crate::phase4_scoring_support::{causal_abstain, phase4_event_disabled};

const MAX_EVENTS_PER_PATH: usize = 4;

pub(crate) fn apply_phase4_event_scores_to_causal(
    query_text: &str,
    vertices: &[KernelVertex],
    answer: &mut GraphRankedCausalExplanationAnswer,
    scorer: &impl Phase4Scorer,
) {
    if phase4_event_disabled() || query_text.trim().is_empty() || answer.candidates.is_empty() {
        return;
    }
    let vertex_by_id = vertices
        .iter()
        .map(|vertex| (vertex.id.0.as_str(), vertex))
        .collect::<FxHashMap<_, _>>();
    for candidate in answer.candidates.iter_mut() {
        if let Some(score) = best_event_score_for_path(query_text, candidate, &vertex_by_id, scorer)
        {
            candidate.answer_score += score.applied_delta;
            candidate.event_rerank = Some(score);
        }
    }
    answer.candidates.sort_by(|left, right| {
        right
            .answer_score
            .partial_cmp(&left.answer_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.source_vertex_id.cmp(&right.source_vertex_id))
    });
    answer.selected = answer
        .candidates
        .iter()
        .find(|candidate| candidate.plane_allowed)
        .cloned();
    let (abstain, abstain_reason) =
        causal_abstain(answer.selected.as_ref(), answer.candidates.is_empty());
    answer.abstain = abstain;
    answer.abstain_reason = abstain_reason;
}

fn best_event_score_for_path(
    query_text: &str,
    candidate: &GraphRankedCausalPath,
    vertex_by_id: &FxHashMap<&str, &KernelVertex>,
    scorer: &impl Phase4Scorer,
) -> Option<GraphPhase4RerankScore> {
    let mut best: Option<GraphPhase4RerankScore> = None;
    for vertex_id in candidate.path_vertex_ids.iter().take(MAX_EVENTS_PER_PATH) {
        let Some(vertex) = vertex_by_id.get(vertex_id.as_str()) else {
            continue;
        };
        if vertex.kind != "event" {
            continue;
        }
        let Some(score) = scorer.score_event_vertex(query_text, vertex) else {
            continue;
        };
        match best.as_ref() {
            Some(current)
                if current.applied_delta > score.applied_delta
                    || (current.applied_delta == score.applied_delta
                        && current.positive_score >= score.positive_score) => {}
            _ => best = Some(score),
        }
    }
    best
}
