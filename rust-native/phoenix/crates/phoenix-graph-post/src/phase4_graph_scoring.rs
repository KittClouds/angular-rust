use phoenix_graph_kernel::{KernelGraphSnapshot, KernelStructuralAnalytics, KernelStructuralScore};

use crate::api::{
    GraphRankedCausalExplanationAnswer, GraphRankedHistoryAnswer, GraphRankedHistoryCandidate,
    GraphRankedSlotAnswer, GraphRankedStateCandidate,
};
use crate::phase4_contract::GraphStructuralRerankScore;
use crate::phase4_scoring_support::{
    causal_abstain, history_abstain, phase4_structural_disabled, world_abstain,
};

const MAX_STRUCTURAL_CANDIDATES: usize = 12;

pub(crate) fn apply_graph_structural_world_state(
    anchor_vertex_ids: &[String],
    snapshot: &KernelGraphSnapshot,
    answer: &mut GraphRankedSlotAnswer,
) {
    if phase4_structural_disabled() || answer.candidates.is_empty() {
        return;
    }
    let analytics = KernelStructuralAnalytics::from_snapshot(snapshot, anchor_vertex_ids);
    if !analytics.is_active() {
        return;
    }
    for candidate in answer.candidates.iter_mut().take(MAX_STRUCTURAL_CANDIDATES) {
        let ids = structural_ids_for_state(candidate);
        if let Some(score) = analytics.score(ids.as_slice()) {
            candidate.answer_score += structural_delta(score.applied_delta_millis);
            candidate.graph_structural_rerank = Some(contract_score(score));
        }
    }
    answer.candidates.sort_by(|left, right| {
        right
            .answer_score
            .partial_cmp(&left.answer_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.state.state_vertex_id.cmp(&right.state.state_vertex_id))
    });
    answer.selected = answer
        .candidates
        .iter()
        .find(|candidate| candidate.plane_allowed)
        .cloned();
    let (abstain, abstain_reason) = world_abstain(answer.selected.as_ref());
    answer.abstain = abstain;
    answer.abstain_reason = abstain_reason;
}

pub(crate) fn apply_graph_structural_history(
    anchor_vertex_ids: &[String],
    snapshot: &KernelGraphSnapshot,
    answer: &mut GraphRankedHistoryAnswer,
) {
    if phase4_structural_disabled() || answer.candidates.is_empty() {
        return;
    }
    let analytics = KernelStructuralAnalytics::from_snapshot(snapshot, anchor_vertex_ids);
    if !analytics.is_active() {
        return;
    }
    for candidate in answer.candidates.iter_mut().take(MAX_STRUCTURAL_CANDIDATES) {
        let ids = structural_ids_for_history(candidate);
        if let Some(score) = analytics.score(ids.as_slice()) {
            candidate.answer_score += structural_delta(score.applied_delta_millis);
            candidate.graph_structural_rerank = Some(contract_score(score));
        }
    }
    answer.candidates.sort_by(|left, right| {
        right
            .answer_score
            .partial_cmp(&left.answer_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.change
                    .state
                    .state_vertex_id
                    .cmp(&right.change.state.state_vertex_id)
            })
    });
    answer.selected = answer
        .candidates
        .iter()
        .find(|candidate| candidate.plane_allowed)
        .cloned();
    let (abstain, abstain_reason) = history_abstain(answer.selected.as_ref());
    answer.abstain = abstain;
    answer.abstain_reason = abstain_reason;
}

pub(crate) fn apply_graph_structural_causal(
    anchor_vertex_ids: &[String],
    snapshot: &KernelGraphSnapshot,
    answer: &mut GraphRankedCausalExplanationAnswer,
) {
    if phase4_structural_disabled() || answer.candidates.is_empty() {
        return;
    }
    let analytics = KernelStructuralAnalytics::from_snapshot(snapshot, anchor_vertex_ids);
    if !analytics.is_active() {
        return;
    }
    for candidate in answer.candidates.iter_mut().take(MAX_STRUCTURAL_CANDIDATES) {
        if let Some(score) = analytics.score_non_anchor(candidate.path_vertex_ids.as_slice()) {
            candidate.answer_score += structural_delta(score.applied_delta_millis);
            candidate.graph_structural_rerank = Some(contract_score(score));
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

fn structural_ids_for_state(candidate: &GraphRankedStateCandidate) -> Vec<String> {
    let mut ids = Vec::with_capacity(candidate.state.supporting_claim_ids.len() + 1);
    ids.push(candidate.state.state_vertex_id.clone());
    ids.extend(candidate.state.supporting_claim_ids.iter().cloned());
    ids
}

fn structural_ids_for_history(candidate: &GraphRankedHistoryCandidate) -> Vec<String> {
    let mut ids = Vec::with_capacity(candidate.change.state.supporting_claim_ids.len() + 1);
    ids.push(candidate.change.state.state_vertex_id.clone());
    ids.extend(candidate.change.state.supporting_claim_ids.iter().cloned());
    ids
}

fn structural_delta(delta_millis: i32) -> f64 {
    (delta_millis as f64) / 1000.0
}

fn contract_score(score: KernelStructuralScore) -> GraphStructuralRerankScore {
    GraphStructuralRerankScore {
        model: "scirs2_retrieved_graph_structure".to_owned(),
        anchor_component: score.anchor_component,
        proximity_score_millis: score.proximity_score_millis,
        component_size: score.component_size,
        applied_delta_millis: score.applied_delta_millis,
    }
}
