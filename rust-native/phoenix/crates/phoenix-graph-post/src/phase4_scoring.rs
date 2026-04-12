use phoenix_graph_kernel::KernelVertex;

use crate::api::{
    GraphRankedCausalExplanationAnswer, GraphRankedCausalPath, GraphRankedHistoryAnswer,
    GraphRankedHistoryCandidate, GraphRankedSlotAnswer, GraphRankedStateCandidate,
};
use crate::phase4_contract::GraphPhase4RerankScore;
use crate::phase4_event_scoring::apply_phase4_event_scores_to_causal;
use crate::phase4_scoring_support::{
    build_rerank_score, history_abstain, label, phase4_disabled, with_default_scorer,
    world_abstain, GliclassPhase4Scorer,
};
use crate::phase4_scoring_text::{
    build_causal_candidate_text, build_event_candidate_text, build_history_candidate_text,
    build_state_candidate_text,
};
use crate::phase5_path_rerank::apply_phase5_path_rerank_with_scorer;

const STATE_PROMPT: &str = "Judge how well the candidate answers the user query. Prefer direct, concrete, well-supported answers over weak or merely adjacent context.";
const CAUSAL_PROMPT: &str = "Judge how well the candidate path explains the target for the user query. Prefer direct, coherent, well-supported explanations over indirect or speculative ones.";
const EVENT_PROMPT: &str = "Judge how important this event is for answering or centering the user query. Prefer the main event over background or unsafe anchors.";

const DIRECT_ANSWER: &str = "direct_answer";
const SUPPORTING_CONTEXT: &str = "supporting_context";
const UNSAFE_ANSWER: &str = "unsafe_answer";

const EXPLAINS_TARGET: &str = "explains_target";
const BACKGROUND_PATH: &str = "background_path";
const SPECULATIVE_PATH: &str = "speculative_path";

const DIRECT_EVENT_ANSWER: &str = "direct_event_answer";
const SUPPORTING_EVENT_CONTEXT: &str = "supporting_event_context";
const UNSAFE_EVENT_ANCHOR: &str = "unsafe_event_anchor";

const MAX_RERANK_CANDIDATES: usize = 8;

pub(crate) trait Phase4Scorer {
    fn score_state_candidate(
        &self,
        query_text: &str,
        candidate: &GraphRankedStateCandidate,
    ) -> Option<GraphPhase4RerankScore>;

    fn score_history_candidate(
        &self,
        query_text: &str,
        candidate: &GraphRankedHistoryCandidate,
    ) -> Option<GraphPhase4RerankScore>;

    fn score_causal_path(
        &self,
        query_text: &str,
        candidate: &GraphRankedCausalPath,
    ) -> Option<GraphPhase4RerankScore>;

    fn score_event_vertex(
        &self,
        query_text: &str,
        vertex: &KernelVertex,
    ) -> Option<GraphPhase4RerankScore>;
}

impl Phase4Scorer for GliclassPhase4Scorer {
    fn score_state_candidate(
        &self,
        query_text: &str,
        candidate: &GraphRankedStateCandidate,
    ) -> Option<GraphPhase4RerankScore> {
        let prediction = self.score_labels(
            build_state_candidate_text(query_text, candidate).as_str(),
            STATE_PROMPT,
            state_labels().as_slice(),
        )?;
        Some(build_rerank_score(
            &prediction,
            DIRECT_ANSWER,
            SUPPORTING_CONTEXT,
            UNSAFE_ANSWER,
            0.58,
            0.12,
            0.52,
            -0.55,
            0.65,
        ))
    }

    fn score_history_candidate(
        &self,
        query_text: &str,
        candidate: &GraphRankedHistoryCandidate,
    ) -> Option<GraphPhase4RerankScore> {
        let prediction = self.score_labels(
            build_history_candidate_text(query_text, candidate).as_str(),
            STATE_PROMPT,
            state_labels().as_slice(),
        )?;
        Some(build_rerank_score(
            &prediction,
            DIRECT_ANSWER,
            SUPPORTING_CONTEXT,
            UNSAFE_ANSWER,
            0.55,
            0.10,
            0.48,
            -0.55,
            0.65,
        ))
    }

    fn score_causal_path(
        &self,
        query_text: &str,
        candidate: &GraphRankedCausalPath,
    ) -> Option<GraphPhase4RerankScore> {
        let prediction = self.score_labels(
            build_causal_candidate_text(query_text, candidate).as_str(),
            CAUSAL_PROMPT,
            causal_labels().as_slice(),
        )?;
        Some(build_rerank_score(
            &prediction,
            EXPLAINS_TARGET,
            BACKGROUND_PATH,
            SPECULATIVE_PATH,
            0.62,
            0.08,
            0.55,
            -0.55,
            0.65,
        ))
    }

    fn score_event_vertex(
        &self,
        query_text: &str,
        vertex: &KernelVertex,
    ) -> Option<GraphPhase4RerankScore> {
        let prediction = self.score_labels(
            build_event_candidate_text(query_text, vertex).as_str(),
            EVENT_PROMPT,
            event_labels().as_slice(),
        )?;
        Some(build_rerank_score(
            &prediction,
            DIRECT_EVENT_ANSWER,
            SUPPORTING_EVENT_CONTEXT,
            UNSAFE_EVENT_ANCHOR,
            0.52,
            0.10,
            0.42,
            -0.45,
            0.55,
        ))
    }
}

pub(crate) fn apply_phase4_world_state(query_text: &str, answer: &mut GraphRankedSlotAnswer) {
    if phase4_disabled()
        || query_text.trim().is_empty()
        || answer.candidates.is_empty()
        || !should_phase4_world_state(answer)
    {
        return;
    }
    with_default_scorer(|scorer| {
        if let Some(scorer) = scorer {
            apply_phase4_world_state_with_scorer(query_text, answer, scorer);
        }
    });
}

pub(crate) fn apply_phase4_history(query_text: &str, answer: &mut GraphRankedHistoryAnswer) {
    if phase4_disabled()
        || query_text.trim().is_empty()
        || answer.candidates.is_empty()
        || !should_phase4_history(answer)
    {
        return;
    }
    with_default_scorer(|scorer| {
        if let Some(scorer) = scorer {
            apply_phase4_history_with_scorer(query_text, answer, scorer);
        }
    });
}

pub(crate) fn apply_phase4_causal(
    query_text: &str,
    vertices: &[KernelVertex],
    answer: &mut GraphRankedCausalExplanationAnswer,
) {
    if phase4_disabled()
        || query_text.trim().is_empty()
        || answer.candidates.is_empty()
        || !should_phase4_causal(answer)
    {
        return;
    }
    with_default_scorer(|scorer| {
        if let Some(scorer) = scorer {
            apply_phase4_causal_with_scorer(query_text, vertices, answer, scorer);
        }
    });
}

pub(crate) fn apply_phase4_world_state_with_scorer(
    query_text: &str,
    answer: &mut GraphRankedSlotAnswer,
    scorer: &impl Phase4Scorer,
) {
    if query_text.trim().is_empty()
        || answer.candidates.is_empty()
        || !should_phase4_world_state(answer)
    {
        return;
    }
    for candidate in answer.candidates.iter_mut().take(MAX_RERANK_CANDIDATES) {
        if let Some(score) = scorer.score_state_candidate(query_text, candidate) {
            candidate.answer_score += score.applied_delta;
            candidate.query_rerank = Some(score);
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

pub(crate) fn apply_phase4_history_with_scorer(
    query_text: &str,
    answer: &mut GraphRankedHistoryAnswer,
    scorer: &impl Phase4Scorer,
) {
    if query_text.trim().is_empty()
        || answer.candidates.is_empty()
        || !should_phase4_history(answer)
    {
        return;
    }
    for candidate in answer.candidates.iter_mut().take(MAX_RERANK_CANDIDATES) {
        if let Some(score) = scorer.score_history_candidate(query_text, candidate) {
            candidate.answer_score += score.applied_delta;
            candidate.query_rerank = Some(score);
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

pub(crate) fn apply_phase4_causal_with_scorer(
    query_text: &str,
    vertices: &[KernelVertex],
    answer: &mut GraphRankedCausalExplanationAnswer,
    scorer: &impl Phase4Scorer,
) {
    if query_text.trim().is_empty() || answer.candidates.is_empty() || !should_phase4_causal(answer)
    {
        return;
    }
    apply_phase5_path_rerank_with_scorer(query_text, answer, scorer);
    if vertices.is_empty() {
        return;
    }
    apply_phase4_event_scores_to_causal(query_text, vertices, answer, scorer);
}

fn state_labels() -> Vec<phoenix_rel_post::GliclassInstructLabel> {
    vec![
        label(
            DIRECT_ANSWER,
            "Directly answers the user query with the best state or state change.",
        ),
        label(
            SUPPORTING_CONTEXT,
            "Relevant support or context, but not the best direct answer.",
        ),
        label(
            UNSAFE_ANSWER,
            "Too speculative, contradicted, or weak to answer safely.",
        ),
    ]
}

fn causal_labels() -> Vec<phoenix_rel_post::GliclassInstructLabel> {
    vec![
        label(
            EXPLAINS_TARGET,
            "Directly explains why the target event or state happened.",
        ),
        label(
            BACKGROUND_PATH,
            "Related background context, but not the main explanatory path.",
        ),
        label(
            SPECULATIVE_PATH,
            "Too indirect, weak, or speculative to trust as the explanation.",
        ),
    ]
}

fn event_labels() -> Vec<phoenix_rel_post::GliclassInstructLabel> {
    vec![
        label(
            DIRECT_EVENT_ANSWER,
            "This event is the main event to center for the user query.",
        ),
        label(
            SUPPORTING_EVENT_CONTEXT,
            "This event is useful supporting context but not the main event.",
        ),
        label(
            UNSAFE_EVENT_ANCHOR,
            "This event is weak, incidental, or unsafe to center.",
        ),
    ]
}

fn should_phase4_world_state(answer: &GraphRankedSlotAnswer) -> bool {
    if answer.candidates.len() > 1 {
        return true;
    }
    if answer.selected.is_none() {
        return false;
    }
    answer.abstain
}

fn should_phase4_history(answer: &GraphRankedHistoryAnswer) -> bool {
    if answer.candidates.len() > 1 {
        return true;
    }
    if answer.selected.is_none() {
        return false;
    }
    answer.abstain
}

fn should_phase4_causal(answer: &GraphRankedCausalExplanationAnswer) -> bool {
    if answer.candidates.len() > 1 {
        return true;
    }
    if answer.selected.is_none() {
        return false;
    }
    answer.abstain
}
