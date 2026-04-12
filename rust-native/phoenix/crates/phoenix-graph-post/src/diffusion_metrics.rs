use phoenix_graph_kernel::KernelEdge;
use rustc_hash::FxHashMap;

use crate::api::{
    GraphRankedCausalExplanationAnswer, GraphRankedHistoryAnswer, GraphRankedSlotAnswer,
};
use crate::eval::{GraphEvalMetrics, GraphSoftEdgeCount, GraphSoftFamily};
use crate::retrieval::GraphRetrievedRegion;

pub(crate) fn metrics_from_world_state(
    pre_answer: &GraphRankedSlotAnswer,
    answer: &GraphRankedSlotAnswer,
    seed_count: usize,
    region: GraphRetrievedRegion,
    candidate_edges: &[KernelEdge],
) -> GraphEvalMetrics {
    let structural = answer
        .selected
        .as_ref()
        .and_then(|candidate| candidate.graph_structural_rerank.as_ref());
    let best_pre = pre_answer.candidates.first();
    let best_post = answer.candidates.first();
    GraphEvalMetrics {
        abstain: answer.abstain,
        abstain_reason: answer.abstain_reason.clone(),
        candidate_count: answer.candidates.len(),
        best_candidate_id: best_post.map(|candidate| candidate.state.state_vertex_id.clone()),
        best_pre_structural_score_millis: best_pre
            .map(|candidate| (candidate.answer_score * 1000.0).round() as i64),
        best_post_structural_score_millis: best_post
            .map(|candidate| (candidate.answer_score * 1000.0).round() as i64),
        best_candidate_hops: None,
        selected_id: answer
            .selected
            .as_ref()
            .map(|candidate| candidate.state.state_vertex_id.clone()),
        selected_label: answer
            .selected
            .as_ref()
            .map(|candidate| candidate.state.value.clone()),
        selected_score_millis: answer
            .selected
            .as_ref()
            .map(|candidate| (candidate.answer_score * 1000.0).round() as i64),
        selected_structural_model: structural.map(|score| score.model.clone()),
        selected_structural_delta_millis: structural.map(|score| score.applied_delta_millis),
        selected_structural_proximity_millis: structural.map(|score| score.proximity_score_millis),
        seed_count,
        region,
        soft_edge_counts: collect_soft_edge_counts(candidate_edges),
    }
}

pub(crate) fn metrics_from_history(
    pre_answer: &GraphRankedHistoryAnswer,
    answer: &GraphRankedHistoryAnswer,
    seed_count: usize,
    region: GraphRetrievedRegion,
    candidate_edges: &[KernelEdge],
) -> GraphEvalMetrics {
    let structural = answer
        .selected
        .as_ref()
        .and_then(|candidate| candidate.graph_structural_rerank.as_ref());
    let best_pre = pre_answer.candidates.first();
    let best_post = answer.candidates.first();
    GraphEvalMetrics {
        abstain: answer.abstain,
        abstain_reason: answer.abstain_reason.clone(),
        candidate_count: answer.candidates.len(),
        best_candidate_id: best_post
            .map(|candidate| candidate.change.state.state_vertex_id.clone()),
        best_pre_structural_score_millis: best_pre
            .map(|candidate| (candidate.answer_score * 1000.0).round() as i64),
        best_post_structural_score_millis: best_post
            .map(|candidate| (candidate.answer_score * 1000.0).round() as i64),
        best_candidate_hops: None,
        selected_id: answer
            .selected
            .as_ref()
            .map(|candidate| candidate.change.state.state_vertex_id.clone()),
        selected_label: answer.selected.as_ref().map(|candidate| {
            format!(
                "{:?} {}",
                candidate.change.change_kind, candidate.change.state.value
            )
        }),
        selected_score_millis: answer
            .selected
            .as_ref()
            .map(|candidate| (candidate.answer_score * 1000.0).round() as i64),
        selected_structural_model: structural.map(|score| score.model.clone()),
        selected_structural_delta_millis: structural.map(|score| score.applied_delta_millis),
        selected_structural_proximity_millis: structural.map(|score| score.proximity_score_millis),
        seed_count,
        region,
        soft_edge_counts: collect_soft_edge_counts(candidate_edges),
    }
}

pub(crate) fn metrics_from_causal(
    pre_answer: &GraphRankedCausalExplanationAnswer,
    answer: &GraphRankedCausalExplanationAnswer,
    seed_count: usize,
    region: GraphRetrievedRegion,
    candidate_edges: &[KernelEdge],
) -> GraphEvalMetrics {
    let structural = answer
        .selected
        .as_ref()
        .and_then(|candidate| candidate.graph_structural_rerank.as_ref());
    let best_pre = pre_answer.candidates.first();
    let best_post = answer.candidates.first();
    GraphEvalMetrics {
        abstain: answer.abstain,
        abstain_reason: answer.abstain_reason.clone(),
        candidate_count: answer.candidates.len(),
        best_candidate_id: best_post.map(|path| path.source_vertex_id.clone()),
        best_pre_structural_score_millis: best_pre
            .map(|path| (path.answer_score * 1000.0).round() as i64),
        best_post_structural_score_millis: best_post
            .map(|path| (path.answer_score * 1000.0).round() as i64),
        best_candidate_hops: best_post.map(|path| path.hops.len()),
        selected_id: answer
            .selected
            .as_ref()
            .map(|path| path.source_vertex_id.clone()),
        selected_label: answer
            .selected
            .as_ref()
            .map(|path| format!("{} -> {}", path.source_vertex_id, path.target_vertex_id)),
        selected_score_millis: answer
            .selected
            .as_ref()
            .map(|path| (path.answer_score * 1000.0).round() as i64),
        selected_structural_model: structural.map(|score| score.model.clone()),
        selected_structural_delta_millis: structural.map(|score| score.applied_delta_millis),
        selected_structural_proximity_millis: structural.map(|score| score.proximity_score_millis),
        seed_count,
        region,
        soft_edge_counts: collect_soft_edge_counts(candidate_edges),
    }
}

fn collect_soft_edge_counts(edges: &[KernelEdge]) -> Vec<GraphSoftEdgeCount> {
    let mut counts = FxHashMap::<GraphSoftFamily, usize>::default();
    for edge in edges {
        if let Some(family) = soft_family_for_edge(edge) {
            *counts.entry(family).or_default() += 1;
        }
    }
    let mut rows = counts
        .into_iter()
        .map(|(family, count)| GraphSoftEdgeCount { family, count })
        .collect::<Vec<_>>();
    rows.sort_by_key(|row| row.family.edge_type());
    rows
}

fn soft_family_for_edge(edge: &KernelEdge) -> Option<GraphSoftFamily> {
    match edge.edge_type.0.as_str() {
        "semantic::same_process" => Some(GraphSoftFamily::SameProcess),
        "semantic::same_slot_family" => Some(GraphSoftFamily::SameSlotFamily),
        "semantic::related_event" => Some(GraphSoftFamily::RelatedEvent),
        "semantic::contradictory_support_region" => {
            Some(GraphSoftFamily::ContradictorySupportRegion)
        }
        "semantic::missing_intermediate_cause" => Some(GraphSoftFamily::MissingIntermediateCause),
        _ => None,
    }
}
