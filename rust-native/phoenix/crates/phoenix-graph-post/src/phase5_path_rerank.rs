use crate::api::{GraphRankedCausalExplanationAnswer, GraphRankedCausalPath};
use crate::phase4_contract::GraphPathRerankScore;
use crate::phase4_scoring::Phase4Scorer;
use crate::phase4_scoring_support::causal_abstain;

const MAX_PATH_RERANK_CANDIDATES: usize = 8;

pub(crate) fn apply_phase5_path_rerank_with_scorer(
    query_text: &str,
    answer: &mut GraphRankedCausalExplanationAnswer,
    scorer: &impl Phase4Scorer,
) {
    if path_rerank_disabled() || query_text.trim().is_empty() || answer.candidates.is_empty() {
        return;
    }
    for (index, candidate) in answer
        .candidates
        .iter_mut()
        .take(MAX_PATH_RERANK_CANDIDATES)
        .enumerate()
    {
        let deterministic_score = candidate.answer_score;
        let Some(learned) = scorer.score_causal_path(query_text, candidate) else {
            continue;
        };
        candidate.answer_score += learned.applied_delta;
        candidate.query_rerank = Some(learned.clone());
        candidate.path_rerank = Some(GraphPathRerankScore {
            deterministic_rank: index + 1,
            deterministic_score,
            applied_delta: learned.applied_delta,
            learned,
        });
    }
    sort_and_select(answer);
}

fn sort_and_select(answer: &mut GraphRankedCausalExplanationAnswer) {
    answer.candidates.sort_by(path_order);
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

fn path_order(left: &GraphRankedCausalPath, right: &GraphRankedCausalPath) -> std::cmp::Ordering {
    right
        .answer_score
        .partial_cmp(&left.answer_score)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.source_vertex_id.cmp(&right.source_vertex_id))
}

fn path_rerank_disabled() -> bool {
    matches!(
        std::env::var("PHOENIX_GRAPH_PHASE5_PATH_DISABLED")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}
