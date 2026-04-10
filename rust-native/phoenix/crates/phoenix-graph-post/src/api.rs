//! Stable public entrypoints for the graph projection compiler.
//!
//! This stage compiles a shadow claim/event/state/view projection from
//! post-ingest sidecars and persists it as a `GraphScopeSidecar` without
//! disturbing the ingest hot path.

use phoenix_graph::GraphBackendError;
use phoenix_graph_kernel::{
    causal_path_candidate_views_from_snapshot, KernelBiTemporal, KernelCausalPathCandidateView,
    KernelEdge, KernelGraphLayer, KernelSlotAnswer, KernelSlotQueryRequest, KernelStateChange,
    KernelStateIssue, KernelUnresolvedQueryRequest, KernelVertex, KernelViewRequest,
    KernelWhatChangedRequest, PhoenixGraphKernel,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixCausalPatchStore, PhoenixEventIdentityPatchStore,
    PhoenixGraphPatchStore, PhoenixMemoryPatchStore, PhoenixSemanticGraphPatchStore,
    PhoenixTemporalPatchStore, StoreError,
};
use phoenix_types::{ScopeKey, SessionId};
use serde::{Deserialize, Serialize};

use crate::phase4_contract::{
    GraphPathRerankScore, GraphPhase4RerankScore, GraphStructuralRerankScore,
};
pub use crate::retrieval::{
    retrieved_causal_explanation, retrieved_history, retrieved_query, retrieved_world_state,
    GraphRetrievedCausalExplanationAnswer, GraphRetrievedCausalExplanationQueryRequest,
    GraphRetrievedHistoryAnswer, GraphRetrievedHistoryQueryRequest, GraphRetrievedQueryAnswer,
    GraphRetrievedQueryRequest, GraphRetrievedRegion, GraphRetrievedSeed,
    GraphRetrievedWorldStateAnswer, GraphRetrievedWorldStateQueryRequest,
};
use crate::{
    build_graph_patch_sidecar, compile_graph_projection, derive_dirty_scope_review_batches,
    derive_scope_review_batch, persist_graph_patch_sidecar, CompiledGraphProjection,
    GraphScopeReviewBatch,
};

#[derive(Debug, thiserror::Error)]
pub enum GraphQueryError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Kernel(#[from] GraphBackendError),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GraphTruthPlane {
    WorldState,
    Reported,
    Conditional,
    Hypothetical,
    Planned,
    Mixed,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphWorldStateQueryRequest {
    pub entity_id: String,
    pub slot_key: String,
    pub valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRankedStateCandidate {
    pub state: phoenix_graph_kernel::KernelSlotState,
    pub truth_plane: GraphTruthPlane,
    pub plane_allowed: bool,
    pub answer_score: f64,
    pub plane_gate: f64,
    pub status_prior: f64,
    pub support_strength: f64,
    pub temporal_fitness: f64,
    pub conflict_penalty: f64,
    pub gap_penalty: f64,
    pub contradiction_region_penalty: f64,
    pub speculative_penalty: f64,
    pub relevant_conflict_count: usize,
    pub relevant_gap_count: usize,
    pub contradiction_region_count: usize,
    #[serde(default)]
    pub supporting_modalities: Vec<String>,
    #[serde(default)]
    pub supporting_source_classes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_rerank: Option<GraphPhase4RerankScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_structural_rerank: Option<GraphStructuralRerankScore>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRankedSlotAnswer {
    pub entity_id: String,
    pub slot_key: String,
    pub selected: Option<GraphRankedStateCandidate>,
    #[serde(default)]
    pub candidates: Vec<GraphRankedStateCandidate>,
    #[serde(default)]
    pub conflicts: Vec<KernelStateIssue>,
    #[serde(default)]
    pub gaps: Vec<KernelStateIssue>,
    pub abstain: bool,
    pub abstain_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphHistoryQueryRequest {
    pub entity_id: String,
    pub slot_key: Option<String>,
    pub since_valid_at: i64,
    pub until_valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
    #[serde(default)]
    pub truth_plane: GraphTruthPlane,
    pub limit: Option<usize>,
}

impl Default for GraphHistoryQueryRequest {
    fn default() -> Self {
        Self {
            entity_id: String::new(),
            slot_key: None,
            since_valid_at: 0,
            until_valid_at: None,
            recorded_at: None,
            include_candidate_graph: false,
            truth_plane: GraphTruthPlane::WorldState,
            limit: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRankedHistoryCandidate {
    pub change: KernelStateChange,
    pub truth_plane: GraphTruthPlane,
    pub plane_allowed: bool,
    pub answer_score: f64,
    pub plane_gate: f64,
    pub status_prior: f64,
    pub support_strength: f64,
    pub temporal_fitness: f64,
    pub recency_score: f64,
    pub conflict_penalty: f64,
    pub gap_penalty: f64,
    pub contradiction_region_penalty: f64,
    pub speculative_penalty: f64,
    pub relevant_conflict_count: usize,
    pub relevant_gap_count: usize,
    pub contradiction_region_count: usize,
    #[serde(default)]
    pub supporting_modalities: Vec<String>,
    #[serde(default)]
    pub supporting_source_classes: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_rerank: Option<GraphPhase4RerankScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_structural_rerank: Option<GraphStructuralRerankScore>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRankedHistoryAnswer {
    pub entity_id: String,
    pub slot_key: Option<String>,
    pub window_start_ms: i64,
    pub window_end_ms: i64,
    pub selected: Option<GraphRankedHistoryCandidate>,
    #[serde(default)]
    pub candidates: Vec<GraphRankedHistoryCandidate>,
    #[serde(default)]
    pub conflicts: Vec<KernelStateIssue>,
    #[serde(default)]
    pub gaps: Vec<KernelStateIssue>,
    pub abstain: bool,
    pub abstain_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphCausalExplanationQueryRequest {
    pub target_vertex_id: String,
    pub valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
    pub max_depth: usize,
    pub limit: Option<usize>,
    #[serde(default)]
    pub truth_plane: GraphTruthPlane,
}

impl Default for GraphCausalExplanationQueryRequest {
    fn default() -> Self {
        Self {
            target_vertex_id: String::new(),
            valid_at: None,
            recorded_at: None,
            include_candidate_graph: false,
            max_depth: 3,
            limit: None,
            truth_plane: GraphTruthPlane::WorldState,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphCausalHop {
    pub source_id: String,
    pub target_id: String,
    pub source_kind: Option<String>,
    pub target_kind: Option<String>,
    pub relation_kind: Option<String>,
    pub status: Option<String>,
    pub polarity: Option<String>,
    pub confidence: Option<f64>,
    pub evidence_ref_count: usize,
    pub layer: KernelGraphLayer,
    #[serde(default)]
    pub temporal: KernelBiTemporal,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRankedCausalPath {
    pub target_vertex_id: String,
    pub source_vertex_id: String,
    #[serde(default)]
    pub path_vertex_ids: Vec<String>,
    #[serde(default)]
    pub hops: Vec<GraphCausalHop>,
    pub truth_plane: GraphTruthPlane,
    pub plane_allowed: bool,
    pub answer_score: f64,
    pub plane_gate: f64,
    pub path_stability: f64,
    pub support_strength: f64,
    pub temporal_fitness: f64,
    pub depth_penalty: f64,
    pub speculative_penalty: f64,
    #[serde(default)]
    pub supporting_modalities: Vec<String>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path_rerank: Option<GraphPathRerankScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_rerank: Option<GraphPhase4RerankScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_rerank: Option<GraphPhase4RerankScore>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub graph_structural_rerank: Option<GraphStructuralRerankScore>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRankedCausalExplanationAnswer {
    pub target_vertex_id: String,
    pub target_kind: Option<String>,
    pub selected: Option<GraphRankedCausalPath>,
    #[serde(default)]
    pub candidates: Vec<GraphRankedCausalPath>,
    pub abstain: bool,
    pub abstain_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum GraphRankedQueryRequest {
    WorldState {
        request: GraphWorldStateQueryRequest,
    },
    History {
        request: GraphHistoryQueryRequest,
    },
    CausalExplanation {
        request: GraphCausalExplanationQueryRequest,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum GraphRankedQueryAnswer {
    WorldState {
        answer: GraphRankedSlotAnswer,
    },
    History {
        answer: GraphRankedHistoryAnswer,
    },
    CausalExplanation {
        answer: GraphRankedCausalExplanationAnswer,
    },
}

pub fn derive_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<GraphScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2
        + PhoenixGraphPatchStore
        + PhoenixSemanticGraphPatchStore
        + PhoenixEventIdentityPatchStore
        + PhoenixTemporalPatchStore
        + PhoenixCausalPatchStore
        + PhoenixMemoryPatchStore,
{
    derive_dirty_scope_review_batches(store, session_id)
}

pub fn derive_batch(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    session: Option<&phoenix_semantic_v2::SessionArchive>,
    dirty: Option<&phoenix_semantic_v2::DirtyScopeRecord>,
    event_identity_sidecar: Option<&phoenix_semantic_v2::EventIdentityScopeSidecar>,
    temporal_sidecar: Option<&phoenix_semantic_v2::TemporalScopeSidecar>,
    causal_sidecar: Option<&phoenix_semantic_v2::CausalScopeSidecar>,
    memory_sidecar: Option<&phoenix_semantic_v2::MemoryScopeSidecar>,
) -> GraphScopeReviewBatch {
    derive_scope_review_batch(
        archives,
        session,
        dirty,
        event_identity_sidecar,
        temporal_sidecar,
        causal_sidecar,
        memory_sidecar,
    )
}

pub fn compile_from_inputs(
    scope_key: &str,
    event_identity_sidecar: Option<&phoenix_semantic_v2::EventIdentityScopeSidecar>,
    temporal_sidecar: Option<&phoenix_semantic_v2::TemporalScopeSidecar>,
    causal_sidecar: Option<&phoenix_semantic_v2::CausalScopeSidecar>,
    memory_sidecar: Option<&phoenix_semantic_v2::MemoryScopeSidecar>,
    recorded_at: Option<i64>,
) -> CompiledGraphProjection {
    compile_graph_projection(
        scope_key,
        event_identity_sidecar,
        temporal_sidecar,
        causal_sidecar,
        memory_sidecar,
        recorded_at,
    )
}

pub fn build_patch_sidecar(
    batch: &GraphScopeReviewBatch,
    created_at: i64,
) -> phoenix_semantic_v2::GraphScopeSidecar {
    build_graph_patch_sidecar(batch, created_at)
}

pub fn persist_patch_sidecar<S>(
    store: &S,
    batch: &GraphScopeReviewBatch,
    created_at: i64,
) -> Result<phoenix_semantic_v2::GraphScopeSidecar, StoreError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    persist_graph_patch_sidecar(store, batch, created_at)
}

pub fn current_slot<S>(
    store: &S,
    scope: &ScopeKey,
    entity_id: &str,
    slot_key: &str,
    recorded_at: Option<i64>,
) -> Result<Option<GraphRankedSlotAnswer>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    slot_at(
        store,
        scope,
        &GraphWorldStateQueryRequest {
            entity_id: entity_id.to_owned(),
            slot_key: slot_key.to_owned(),
            valid_at: Some(now_ms()),
            recorded_at,
            include_candidate_graph: false,
        },
    )
}

pub fn slot_at<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphWorldStateQueryRequest,
) -> Result<Option<GraphRankedSlotAnswer>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    let Some(kernel) = load_projection_kernel(store, scope)? else {
        return Ok(None);
    };
    let answer = kernel.slot_at(KernelSlotQueryRequest {
        entity_id: request.entity_id.clone(),
        slot_key: request.slot_key.clone(),
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    let snapshot = kernel.view_as_of(phoenix_graph_kernel::KernelViewRequest {
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    Ok(Some(rank_world_state_answer(
        &snapshot.vertices,
        &snapshot.candidate_edges,
        &answer,
    )))
}

pub fn what_is_unresolved<S>(
    store: &S,
    scope: &ScopeKey,
    request: &KernelUnresolvedQueryRequest,
) -> Result<Option<Vec<KernelStateIssue>>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    let Some(kernel) = load_projection_kernel(store, scope)? else {
        return Ok(None);
    };
    Ok(Some(kernel.what_is_unresolved(request.clone())))
}

pub fn what_changed<S>(
    store: &S,
    scope: &ScopeKey,
    request: &KernelWhatChangedRequest,
) -> Result<Option<Vec<KernelStateChange>>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    let Some(kernel) = load_projection_kernel(store, scope)? else {
        return Ok(None);
    };
    Ok(Some(kernel.what_changed(request.clone())))
}

pub fn history<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphHistoryQueryRequest,
) -> Result<Option<GraphRankedHistoryAnswer>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    let Some(kernel) = load_projection_kernel(store, scope)? else {
        return Ok(None);
    };
    let until_valid_at = request.until_valid_at.unwrap_or_else(now_ms);
    let snapshot = kernel.view_as_of(KernelViewRequest {
        valid_at: Some(until_valid_at),
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    let timeline = kernel.entity_timeline(
        &request.entity_id,
        Some((request.since_valid_at, until_valid_at)),
        request.recorded_at.or(Some(until_valid_at)),
    );
    let changes = kernel.what_changed(KernelWhatChangedRequest {
        entity_id: request.entity_id.clone(),
        slot_key: request.slot_key.clone(),
        since_valid_at: request.since_valid_at,
        until_valid_at: Some(until_valid_at),
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    let conflicts = collect_timeline_issues(
        &timeline.vertices,
        "conflict",
        &request.entity_id,
        request.slot_key.as_deref(),
    );
    let gaps = collect_timeline_issues(
        &timeline.vertices,
        "gap",
        &request.entity_id,
        request.slot_key.as_deref(),
    );
    Ok(Some(rank_history_answer(
        request,
        until_valid_at,
        &timeline.vertices,
        &snapshot.candidate_edges,
        &changes,
        &conflicts,
        &gaps,
    )))
}

pub fn causal_explanation<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphCausalExplanationQueryRequest,
) -> Result<Option<GraphRankedCausalExplanationAnswer>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    let Some(kernel) = load_projection_kernel(store, scope)? else {
        return Ok(None);
    };
    let snapshot = kernel.view_as_of(KernelViewRequest {
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    let answer = rank_causal_explanation_answer(request, &snapshot);
    if answer.selected.is_some() || !answer.candidates.is_empty() {
        return Ok(Some(answer));
    }

    let target_entity_id = snapshot
        .vertices
        .iter()
        .find(|vertex| vertex.id.0 == request.target_vertex_id)
        .and_then(|vertex| vertex.entity_id.clone());
    let Some(entity_id) = target_entity_id else {
        return Ok(Some(answer));
    };
    let timeline =
        kernel.entity_timeline(&entity_id, None, request.recorded_at.or(request.valid_at));
    let timeline_answer = rank_causal_explanation_answer(request, &timeline);
    if timeline_answer.selected.is_some() || !timeline_answer.candidates.is_empty() {
        Ok(Some(timeline_answer))
    } else {
        Ok(Some(answer))
    }
}

pub fn ranked_query<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRankedQueryRequest,
) -> Result<Option<GraphRankedQueryAnswer>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    match request {
        GraphRankedQueryRequest::WorldState { request } => slot_at(store, scope, request)
            .map(|answer| answer.map(|answer| GraphRankedQueryAnswer::WorldState { answer })),
        GraphRankedQueryRequest::History { request } => history(store, scope, request)
            .map(|answer| answer.map(|answer| GraphRankedQueryAnswer::History { answer })),
        GraphRankedQueryRequest::CausalExplanation { request } => {
            causal_explanation(store, scope, request).map(|answer| {
                answer.map(|answer| GraphRankedQueryAnswer::CausalExplanation { answer })
            })
        }
    }
}

pub(crate) fn rank_world_state_answer(
    vertices: &[KernelVertex],
    candidate_edges: &[KernelEdge],
    answer: &KernelSlotAnswer,
) -> GraphRankedSlotAnswer {
    let claim_by_id = claim_vertex_index(vertices);

    let mut candidates = Vec::new();
    if let Some(active) = answer.active_state.as_ref() {
        candidates.push(rank_candidate(
            active,
            answer,
            candidate_edges,
            &claim_by_id,
        ));
    }
    for state in &answer.competing_states {
        candidates.push(rank_candidate(state, answer, candidate_edges, &claim_by_id));
    }
    candidates.sort_by(|left, right| {
        right
            .answer_score
            .partial_cmp(&left.answer_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.state.state_vertex_id.cmp(&right.state.state_vertex_id))
    });

    let selected = candidates
        .iter()
        .find(|candidate| candidate.plane_allowed)
        .cloned();
    let (abstain, abstain_reason) = match selected.as_ref() {
        None => (
            true,
            Some("no world-state candidate passed the plane gate".to_owned()),
        ),
        Some(candidate) if candidate.answer_score < 1.75 => (
            true,
            Some("top candidate was too weak to answer safely".to_owned()),
        ),
        Some(candidate)
            if (candidate.relevant_conflict_count + candidate.relevant_gap_count) > 0
                && candidate.answer_score < 2.1 =>
        {
            (
                true,
                Some(
                    "top candidate remains unresolved under current conflict/gap pressure"
                        .to_owned(),
                ),
            )
        }
        Some(_) => (false, None),
    };

    GraphRankedSlotAnswer {
        entity_id: answer.entity_id.clone(),
        slot_key: answer.slot_key.clone(),
        selected,
        candidates,
        conflicts: answer.conflicts.clone(),
        gaps: answer.gaps.clone(),
        abstain,
        abstain_reason,
    }
}

pub fn rank_history_answer(
    request: &GraphHistoryQueryRequest,
    until_valid_at: i64,
    vertices: &[KernelVertex],
    candidate_edges: &[KernelEdge],
    changes: &[KernelStateChange],
    conflicts: &[KernelStateIssue],
    gaps: &[KernelStateIssue],
) -> GraphRankedHistoryAnswer {
    let claim_by_id = claim_vertex_index(vertices);
    let mut candidates = changes
        .iter()
        .map(|change| {
            rank_history_candidate(
                change,
                conflicts,
                gaps,
                candidate_edges,
                &claim_by_id,
                request.truth_plane,
                request.since_valid_at,
                until_valid_at,
            )
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
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
    candidates.truncate(request.limit.unwrap_or(12));

    let selected = candidates
        .iter()
        .find(|candidate| candidate.plane_allowed)
        .cloned();
    let (abstain, abstain_reason) = match selected.as_ref() {
        None => (
            true,
            Some("no history candidate passed the requested truth plane".to_owned()),
        ),
        Some(candidate) if candidate.answer_score < 1.8 => (
            true,
            Some("top history candidate was too weak to answer safely".to_owned()),
        ),
        Some(candidate)
            if (candidate.relevant_conflict_count + candidate.relevant_gap_count) > 0
                && candidate.answer_score < 2.15 =>
        {
            (
                true,
                Some("history window remains unresolved under conflict or gap pressure".to_owned()),
            )
        }
        Some(_) => (false, None),
    };

    GraphRankedHistoryAnswer {
        entity_id: request.entity_id.clone(),
        slot_key: request.slot_key.clone(),
        window_start_ms: request.since_valid_at,
        window_end_ms: until_valid_at,
        selected,
        candidates,
        conflicts: conflicts.to_vec(),
        gaps: gaps.to_vec(),
        abstain,
        abstain_reason,
    }
}

pub fn rank_causal_explanation_answer(
    request: &GraphCausalExplanationQueryRequest,
    snapshot: &phoenix_graph_kernel::KernelGraphSnapshot,
) -> GraphRankedCausalExplanationAnswer {
    let vertex_by_id = vertex_index(&snapshot.vertices);
    let target_kind = vertex_by_id
        .get(request.target_vertex_id.as_str())
        .map(|vertex| vertex.kind.clone());
    if target_kind.is_none() {
        return GraphRankedCausalExplanationAnswer {
            target_vertex_id: request.target_vertex_id.clone(),
            target_kind: None,
            selected: None,
            candidates: Vec::new(),
            abstain: true,
            abstain_reason: Some(
                "target vertex is not present in the current projection".to_owned(),
            ),
        };
    }

    let candidate_limit = request.limit.unwrap_or(8).saturating_mul(4).clamp(12, 64);
    let mut causal_candidates = causal_path_candidate_views_from_snapshot(
        snapshot,
        request.target_vertex_id.as_str(),
        request.max_depth,
        candidate_limit,
    )
    .into_iter()
    .map(|candidate| rank_causal_path(request, &vertex_by_id, &candidate))
    .collect::<Vec<_>>();
    causal_candidates.sort_by(|left, right| {
        right
            .answer_score
            .partial_cmp(&left.answer_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.source_vertex_id.cmp(&right.source_vertex_id))
    });
    causal_candidates.truncate(request.limit.unwrap_or(8));
    let selected = causal_candidates
        .iter()
        .find(|candidate| candidate.plane_allowed)
        .cloned();
    let (abstain, abstain_reason) = match selected.as_ref() {
        None if causal_candidates.is_empty() => (
            true,
            Some("no causal path was available for the target vertex".to_owned()),
        ),
        None => (
            true,
            Some("no causal path passed the requested truth plane".to_owned()),
        ),
        Some(candidate) if candidate.answer_score < 1.9 => (
            true,
            Some("top causal path was too weak to answer safely".to_owned()),
        ),
        Some(candidate) if candidate.temporal_fitness < 0.65 && candidate.path_stability < 0.7 => (
            true,
            Some("causal support is too brittle to answer confidently".to_owned()),
        ),
        Some(_) => (false, None),
    };

    GraphRankedCausalExplanationAnswer {
        target_vertex_id: request.target_vertex_id.clone(),
        target_kind,
        selected,
        candidates: causal_candidates,
        abstain,
        abstain_reason,
    }
}

fn claim_vertex_index(
    vertices: &[KernelVertex],
) -> std::collections::BTreeMap<String, &KernelVertex> {
    vertices
        .iter()
        .filter(|vertex| vertex.kind == "claim")
        .filter_map(|vertex| {
            vertex
                .id
                .0
                .strip_prefix("graph::claim::")
                .map(|claim_id| (claim_id.to_owned(), vertex))
        })
        .collect::<std::collections::BTreeMap<_, _>>()
}

fn vertex_index(vertices: &[KernelVertex]) -> std::collections::BTreeMap<&str, &KernelVertex> {
    vertices
        .iter()
        .map(|vertex| (vertex.id.0.as_str(), vertex))
        .collect::<std::collections::BTreeMap<_, _>>()
}

pub(crate) fn load_projection_kernel<S>(
    store: &S,
    scope: &ScopeKey,
) -> Result<Option<PhoenixGraphKernel>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
{
    let Some(sidecar) = store.load_graph_patch_sidecar(scope)? else {
        return Ok(None);
    };
    let mut kernel = PhoenixGraphKernel::new();
    kernel.apply_kernel_batch(sidecar.graph_batch)?;
    if let Some(semantic_sidecar) = store.load_semantic_graph_patch_sidecar(scope)? {
        kernel.apply_kernel_batch(semantic_sidecar.candidate_graph_batch)?;
    }
    Ok(Some(kernel))
}

fn rank_candidate(
    state: &phoenix_graph_kernel::KernelSlotState,
    answer: &KernelSlotAnswer,
    candidate_edges: &[KernelEdge],
    claim_by_id: &std::collections::BTreeMap<String, &KernelVertex>,
) -> GraphRankedStateCandidate {
    let (claim_vertices, supporting_modalities, supporting_source_classes) =
        claim_context_for_supporting_claims(state.supporting_claim_ids.as_slice(), claim_by_id);
    let truth_plane = derive_truth_plane(&supporting_modalities);
    let plane_allowed = plane_allowed_for(GraphTruthPlane::WorldState, truth_plane);
    let plane_gate = plane_gate(GraphTruthPlane::WorldState, truth_plane);
    let status_prior = status_prior(state.status.as_deref());
    let support_strength = support_strength(state, claim_vertices.as_slice());
    let temporal_fitness = temporal_fitness(&state.temporal);
    let relevant_conflict_count = relevant_issue_count(&answer.conflicts, state);
    let relevant_gap_count = relevant_issue_count(&answer.gaps, state);
    let contradiction_region_count = contradiction_region_count(
        candidate_edges,
        state.state_vertex_id.as_str(),
        state.supporting_claim_ids.as_slice(),
    );
    let conflict_penalty = (relevant_conflict_count as f64 * 0.18).min(0.45);
    let gap_penalty = (relevant_gap_count as f64 * 0.14).min(0.45);
    let contradiction_region_penalty = (contradiction_region_count as f64 * 0.12).min(0.36);
    let speculative_penalty = speculative_penalty(truth_plane);
    let answer_score = plane_gate + status_prior + support_strength + temporal_fitness
        - conflict_penalty
        - gap_penalty
        - contradiction_region_penalty
        - speculative_penalty;

    GraphRankedStateCandidate {
        state: state.clone(),
        truth_plane,
        plane_allowed,
        answer_score,
        plane_gate,
        status_prior,
        support_strength,
        temporal_fitness,
        conflict_penalty,
        gap_penalty,
        contradiction_region_penalty,
        speculative_penalty,
        relevant_conflict_count,
        relevant_gap_count,
        contradiction_region_count,
        supporting_modalities,
        supporting_source_classes,
        query_rerank: None,
        graph_structural_rerank: None,
    }
}

fn derive_truth_plane(modalities: &[String]) -> GraphTruthPlane {
    let mut saw_world = false;
    let mut saw_reported = false;
    let mut saw_conditional = false;
    let mut saw_hypothetical = false;
    let mut saw_planned = false;

    for modality in modalities {
        match modality.as_str() {
            "asserted" | "observed" | "inferred" => saw_world = true,
            "reported" => saw_reported = true,
            "conditional" => saw_conditional = true,
            "hypothetical" => saw_hypothetical = true,
            "planned" => saw_planned = true,
            _ => {}
        }
    }

    let count = [
        saw_world,
        saw_reported,
        saw_conditional,
        saw_hypothetical,
        saw_planned,
    ]
    .into_iter()
    .filter(|value| *value)
    .count();
    if count > 1 {
        return GraphTruthPlane::Mixed;
    }
    if saw_world {
        GraphTruthPlane::WorldState
    } else if saw_reported {
        GraphTruthPlane::Reported
    } else if saw_conditional {
        GraphTruthPlane::Conditional
    } else if saw_hypothetical {
        GraphTruthPlane::Hypothetical
    } else if saw_planned {
        GraphTruthPlane::Planned
    } else {
        GraphTruthPlane::Unknown
    }
}

fn status_prior(status: Option<&str>) -> f64 {
    match status.unwrap_or_default() {
        "supported" => 1.0,
        "active" => 0.95,
        "candidate" => 0.65,
        "deferred" => 0.4,
        "contradicted" => 0.15,
        "superseded" => 0.05,
        "rejected" => 0.0,
        _ => 0.2,
    }
}

fn support_strength(
    state: &phoenix_graph_kernel::KernelSlotState,
    claim_vertices: &[&KernelVertex],
) -> f64 {
    let confidence = state.confidence.unwrap_or(0.5).clamp(0.0, 1.0);
    let support_count = (state.supporting_claim_ids.len().min(4) as f64) / 4.0;
    let evidence_count = claim_vertices
        .iter()
        .map(|vertex| vertex.provenance.evidence_refs.len())
        .sum::<usize>()
        .min(6) as f64
        / 6.0;
    (confidence * 0.7) + (support_count * 0.2) + (evidence_count * 0.1)
}

fn temporal_fitness(temporal: &phoenix_graph_kernel::KernelBiTemporal) -> f64 {
    let mut score: f64 = if temporal.valid_to.is_none() {
        1.0
    } else {
        0.85
    };
    if temporal.valid_from.is_some() {
        score += 0.05;
    }
    score.min(1.0)
}

fn claim_context_for_supporting_claims<'a>(
    claim_ids: &[String],
    claim_by_id: &std::collections::BTreeMap<String, &'a KernelVertex>,
) -> (Vec<&'a KernelVertex>, Vec<String>, Vec<String>) {
    let claim_vertices = claim_ids
        .iter()
        .filter_map(|claim_id| claim_by_id.get(claim_id))
        .copied()
        .collect::<Vec<_>>();
    let mut supporting_modalities = claim_vertices
        .iter()
        .filter_map(|vertex| {
            vertex
                .value
                .get("modality")
                .and_then(serde_json::Value::as_str)
                .and_then(normalize_modality_label)
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let mut supporting_source_classes = claim_vertices
        .iter()
        .filter_map(|vertex| {
            vertex
                .attributes
                .get("sourceClass")
                .and_then(serde_json::Value::as_str)
        })
        .map(str::to_owned)
        .collect::<Vec<_>>();
    sort_and_dedup_strings(&mut supporting_modalities);
    sort_and_dedup_strings(&mut supporting_source_classes);
    (
        claim_vertices,
        supporting_modalities,
        supporting_source_classes,
    )
}

fn rank_history_candidate(
    change: &KernelStateChange,
    conflicts: &[KernelStateIssue],
    gaps: &[KernelStateIssue],
    candidate_edges: &[KernelEdge],
    claim_by_id: &std::collections::BTreeMap<String, &KernelVertex>,
    requested_plane: GraphTruthPlane,
    since_valid_at: i64,
    until_valid_at: i64,
) -> GraphRankedHistoryCandidate {
    let (claim_vertices, supporting_modalities, supporting_source_classes) =
        claim_context_for_supporting_claims(
            change.state.supporting_claim_ids.as_slice(),
            claim_by_id,
        );
    let truth_plane = derive_truth_plane(&supporting_modalities);
    let plane_allowed = plane_allowed_for(requested_plane, truth_plane);
    let plane_gate = plane_gate(requested_plane, truth_plane);
    let status_prior = status_prior(change.state.status.as_deref());
    let support_strength = support_strength(&change.state, claim_vertices.as_slice());
    let temporal_fitness = temporal_fitness(&change.state.temporal);
    let recency_score =
        history_recency_score(&change.state.temporal, since_valid_at, until_valid_at);
    let relevant_conflict_count = relevant_timeline_issue_count(conflicts, &change.state);
    let relevant_gap_count = relevant_timeline_issue_count(gaps, &change.state);
    let contradiction_region_count = contradiction_region_count(
        candidate_edges,
        change.state.state_vertex_id.as_str(),
        change.state.supporting_claim_ids.as_slice(),
    );
    let conflict_penalty = (relevant_conflict_count as f64 * 0.16).min(0.5);
    let gap_penalty = (relevant_gap_count as f64 * 0.12).min(0.4);
    let contradiction_region_penalty = (contradiction_region_count as f64 * 0.11).min(0.33);
    let speculative_penalty = speculative_penalty(truth_plane);
    let answer_score =
        plane_gate + status_prior + support_strength + temporal_fitness + recency_score
            - conflict_penalty
            - gap_penalty
            - contradiction_region_penalty
            - speculative_penalty;

    GraphRankedHistoryCandidate {
        change: change.clone(),
        truth_plane,
        plane_allowed,
        answer_score,
        plane_gate,
        status_prior,
        support_strength,
        temporal_fitness,
        recency_score,
        conflict_penalty,
        gap_penalty,
        contradiction_region_penalty,
        speculative_penalty,
        relevant_conflict_count,
        relevant_gap_count,
        contradiction_region_count,
        supporting_modalities,
        supporting_source_classes,
        query_rerank: None,
        graph_structural_rerank: None,
    }
}

fn relevant_issue_count(
    issues: &[KernelStateIssue],
    state: &phoenix_graph_kernel::KernelSlotState,
) -> usize {
    issues
        .iter()
        .filter(|issue| {
            issue.slot_key == state.slot_key
                && issue.entity_id == state.entity_id
                && (issue.supporting_claim_ids.is_empty()
                    || issue.supporting_claim_ids.iter().any(|claim_id| {
                        state
                            .supporting_claim_ids
                            .iter()
                            .any(|state_id| state_id == claim_id)
                    }))
        })
        .count()
}

fn contradiction_region_count(
    candidate_edges: &[KernelEdge],
    state_vertex_id: &str,
    supporting_claim_ids: &[String],
) -> usize {
    candidate_edges
        .iter()
        .filter(|edge| edge.edge_type.0 == "semantic::contradictory_support_region")
        .filter(|edge| {
            edge_touches_state_or_support_claims(edge, state_vertex_id, supporting_claim_ids)
        })
        .count()
}

fn edge_touches_state_or_support_claims(
    edge: &KernelEdge,
    state_vertex_id: &str,
    supporting_claim_ids: &[String],
) -> bool {
    edge.source_id.0 == state_vertex_id
        || edge.target_id.0 == state_vertex_id
        || edge_touches_supporting_claim(edge.source_id.0.as_str(), supporting_claim_ids)
        || edge_touches_supporting_claim(edge.target_id.0.as_str(), supporting_claim_ids)
}

fn edge_touches_supporting_claim(endpoint_id: &str, supporting_claim_ids: &[String]) -> bool {
    let Some(claim_id) = endpoint_id.strip_prefix("graph::claim::") else {
        return false;
    };
    supporting_claim_ids
        .iter()
        .any(|candidate| candidate == claim_id)
}

fn speculative_penalty(plane: GraphTruthPlane) -> f64 {
    match plane {
        GraphTruthPlane::WorldState | GraphTruthPlane::Unknown => 0.0,
        GraphTruthPlane::Reported => 0.35,
        GraphTruthPlane::Conditional => 0.55,
        GraphTruthPlane::Planned => 0.65,
        GraphTruthPlane::Hypothetical => 0.75,
        GraphTruthPlane::Mixed => 0.45,
    }
}

fn rank_causal_path(
    request: &GraphCausalExplanationQueryRequest,
    vertex_by_id: &std::collections::BTreeMap<&str, &KernelVertex>,
    candidate: &KernelCausalPathCandidateView<'_>,
) -> GraphRankedCausalPath {
    let supporting_modalities = candidate.supporting_modalities.clone();
    let truth_plane = derive_truth_plane(&supporting_modalities);
    let plane_allowed = plane_allowed_for(request.truth_plane, truth_plane);
    let plane_gate = plane_gate(request.truth_plane, truth_plane);
    let path_stability = candidate.features.path_stability;
    let support_strength = candidate.features.support_strength;
    let temporal_fitness = candidate.features.temporal_consistency_ratio;
    let depth_penalty = (candidate.features.depth.saturating_sub(1) as f64 * 0.12).min(0.48);
    let speculative_penalty = speculative_penalty(truth_plane);
    let answer_score = plane_gate + path_stability + support_strength + temporal_fitness
        - depth_penalty
        - speculative_penalty;
    let mut evidence_refs = candidate
        .path_edges
        .iter()
        .flat_map(|edge| edge.provenance.evidence_refs.iter().cloned())
        .collect::<Vec<_>>();
    sort_and_dedup_strings(&mut evidence_refs);
    let hops = candidate
        .path_edges
        .iter()
        .map(|edge| GraphCausalHop {
            source_id: edge.source_id.0.clone(),
            target_id: edge.target_id.0.clone(),
            source_kind: vertex_by_id
                .get(edge.source_id.0.as_str())
                .map(|vertex| vertex.kind.clone()),
            target_kind: vertex_by_id
                .get(edge.target_id.0.as_str())
                .map(|vertex| vertex.kind.clone()),
            relation_kind: edge
                .attributes
                .get("relationKind")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            status: edge
                .attributes
                .get("status")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            polarity: edge
                .attributes
                .get("polarity")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            confidence: edge.provenance.confidence,
            evidence_ref_count: edge.provenance.evidence_refs.len(),
            layer: edge.layer.clone(),
            temporal: edge.temporal.clone(),
        })
        .collect::<Vec<_>>();

    GraphRankedCausalPath {
        target_vertex_id: request.target_vertex_id.clone(),
        source_vertex_id: candidate.source_vertex_id.to_owned(),
        path_vertex_ids: candidate
            .path_vertex_ids
            .iter()
            .map(|vertex_id| (*vertex_id).to_owned())
            .collect(),
        hops,
        truth_plane,
        plane_allowed,
        answer_score,
        plane_gate,
        path_stability,
        support_strength,
        temporal_fitness,
        depth_penalty,
        speculative_penalty,
        supporting_modalities,
        evidence_refs,
        path_rerank: None,
        query_rerank: None,
        event_rerank: None,
        graph_structural_rerank: None,
    }
}

fn normalize_modality_label(label: &str) -> Option<&'static str> {
    match label {
        "asserted" | "observed" | "inferred" | "negated" => Some("asserted"),
        "reported" | "reportedSpeech" | "attributedClaim" => Some("reported"),
        "conditional" => Some("conditional"),
        "hypothetical" => Some("hypothetical"),
        "planned" => Some("planned"),
        _ => None,
    }
}

fn collect_timeline_issues(
    vertices: &[KernelVertex],
    issue_kind: &str,
    entity_id: &str,
    slot_key: Option<&str>,
) -> Vec<KernelStateIssue> {
    let mut issues = vertices
        .iter()
        .filter(|vertex| vertex.kind == issue_kind)
        .filter(|vertex| vertex.entity_id.as_deref() == Some(entity_id))
        .filter(|vertex| {
            slot_key
                .map(|slot_key| vertex_slot_key(vertex) == Some(slot_key))
                .unwrap_or(true)
        })
        .map(issue_from_vertex)
        .collect::<Vec<_>>();
    issues.sort_by(|left, right| {
        left.temporal
            .valid_from
            .cmp(&right.temporal.valid_from)
            .then_with(|| left.issue_vertex_id.cmp(&right.issue_vertex_id))
    });
    issues
}

fn issue_from_vertex(vertex: &KernelVertex) -> KernelStateIssue {
    KernelStateIssue {
        issue_vertex_id: vertex.id.0.clone(),
        issue_kind: vertex.kind.clone(),
        entity_id: vertex.entity_id.clone().unwrap_or_default(),
        slot_key: vertex_slot_key(vertex).unwrap_or_default().to_owned(),
        status: string_attr(&vertex.value, "status").map(str::to_owned),
        reason: string_attr(&vertex.value, "kind").map(str::to_owned),
        detail: string_attr(&vertex.attributes, "detail").map(str::to_owned),
        preferred_claim_id: string_attr(&vertex.attributes, "preferredClaimId").map(str::to_owned),
        temporal: vertex.temporal.clone(),
        supporting_claim_ids: string_list_attr(&vertex.attributes, "claimIds"),
    }
}

fn relevant_timeline_issue_count(
    issues: &[KernelStateIssue],
    state: &phoenix_graph_kernel::KernelSlotState,
) -> usize {
    issues
        .iter()
        .filter(|issue| {
            issue.entity_id == state.entity_id
                && issue.slot_key == state.slot_key
                && temporal_windows_overlap(&issue.temporal, &state.temporal)
                && (issue.supporting_claim_ids.is_empty()
                    || issue.supporting_claim_ids.iter().any(|claim_id| {
                        state
                            .supporting_claim_ids
                            .iter()
                            .any(|state_claim_id| state_claim_id == claim_id)
                    }))
        })
        .count()
}

fn temporal_windows_overlap(left: &KernelBiTemporal, right: &KernelBiTemporal) -> bool {
    let left_start = left.valid_from.unwrap_or(i64::MIN);
    let left_end = left.valid_to.unwrap_or(i64::MAX);
    let right_start = right.valid_from.unwrap_or(i64::MIN);
    let right_end = right.valid_to.unwrap_or(i64::MAX);
    left_start < right_end && left_end > right_start
}

fn history_recency_score(
    temporal: &KernelBiTemporal,
    since_valid_at: i64,
    until_valid_at: i64,
) -> f64 {
    if until_valid_at <= since_valid_at {
        return 0.35;
    }
    let anchor = temporal
        .valid_from
        .or(temporal.valid_to)
        .unwrap_or(until_valid_at)
        .clamp(since_valid_at, until_valid_at);
    let normalized = (anchor - since_valid_at) as f64 / (until_valid_at - since_valid_at) as f64;
    0.15 + (normalized * 0.35)
}

fn plane_allowed_for(requested_plane: GraphTruthPlane, candidate_plane: GraphTruthPlane) -> bool {
    match requested_plane {
        GraphTruthPlane::Unknown | GraphTruthPlane::Mixed => true,
        GraphTruthPlane::WorldState => {
            matches!(
                candidate_plane,
                GraphTruthPlane::WorldState | GraphTruthPlane::Unknown
            )
        }
        GraphTruthPlane::Reported
        | GraphTruthPlane::Conditional
        | GraphTruthPlane::Hypothetical
        | GraphTruthPlane::Planned => {
            candidate_plane == requested_plane || candidate_plane == GraphTruthPlane::Unknown
        }
    }
}

fn plane_gate(requested_plane: GraphTruthPlane, candidate_plane: GraphTruthPlane) -> f64 {
    if plane_allowed_for(requested_plane, candidate_plane) {
        1.0
    } else {
        0.0
    }
}

fn string_attr<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(serde_json::Value::as_str)
}

fn vertex_slot_key(vertex: &KernelVertex) -> Option<&str> {
    string_attr(&vertex.value, "slotKey").or_else(|| string_attr(&vertex.attributes, "slotKey"))
}

fn string_list_attr(value: &serde_json::Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(serde_json::Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(serde_json::Value::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>()
}

fn sort_and_dedup_strings(values: &mut Vec<String>) {
    values.sort();
    values.dedup();
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_graph_kernel::{
        KernelBiTemporal, KernelEdge, KernelEdgeType, KernelGraphLayer, KernelMutationBatch,
        KernelMutationScope, KernelProvenance, KernelRelationClass, KernelVertex,
        KernelVertexClass, KernelVertexId,
    };
    use phoenix_semantic_v2::{GraphScopeSidecar, SemanticGraphScopeSidecar};
    use phoenix_store_native_core::{PhoenixGraphPatchStore, PhoenixSemanticGraphPatchStore};

    #[derive(Clone, Default)]
    struct TestGraphStore {
        sidecar: Option<GraphScopeSidecar>,
    }

    impl PhoenixGraphPatchStore for TestGraphStore {
        fn init_graph_patch_schema(&self) -> Result<(), StoreError> {
            Ok(())
        }

        fn persist_graph_patch_sidecar(
            &self,
            _sidecar: &GraphScopeSidecar,
        ) -> Result<(), StoreError> {
            Ok(())
        }

        fn load_graph_patch_sidecar(
            &self,
            _scope: &ScopeKey,
        ) -> Result<Option<GraphScopeSidecar>, StoreError> {
            Ok(self.sidecar.clone())
        }
    }

    impl PhoenixSemanticGraphPatchStore for TestGraphStore {
        fn init_semantic_graph_patch_schema(&self) -> Result<(), StoreError> {
            Ok(())
        }

        fn persist_semantic_graph_patch_sidecar(
            &self,
            _sidecar: &SemanticGraphScopeSidecar,
        ) -> Result<(), StoreError> {
            Ok(())
        }

        fn load_semantic_graph_patch_sidecar(
            &self,
            _scope: &ScopeKey,
        ) -> Result<Option<SemanticGraphScopeSidecar>, StoreError> {
            Ok(None)
        }
    }

    #[test]
    fn current_slot_ranks_world_state_above_reported() {
        let scope = ScopeKey::default();
        let store = TestGraphStore {
            sidecar: Some(sidecar_for_states(vec![
                test_claim("claim-reported", "reported"),
                test_claim("claim-asserted", "asserted"),
                test_state(
                    "state-reported",
                    "alice",
                    "entity.employer",
                    "Press rumor",
                    0.82,
                    "claim-reported",
                ),
                test_state(
                    "state-asserted",
                    "alice",
                    "entity.employer",
                    "Acme",
                    0.81,
                    "claim-asserted",
                ),
            ])),
        };

        let answer = current_slot(&store, &scope, "alice", "entity.employer", Some(100))
            .expect("query")
            .expect("ranked answer");
        assert!(!answer.abstain);
        assert_eq!(
            answer
                .selected
                .as_ref()
                .map(|candidate| candidate.state.value.as_str()),
            Some("Acme")
        );
        assert_eq!(
            answer
                .selected
                .as_ref()
                .map(|candidate| candidate.truth_plane),
            Some(GraphTruthPlane::WorldState)
        );
    }

    #[test]
    fn current_slot_abstains_when_only_reported_candidates_exist() {
        let scope = ScopeKey::default();
        let store = TestGraphStore {
            sidecar: Some(sidecar_for_states(vec![
                test_claim("claim-reported", "reported"),
                test_state(
                    "state-reported",
                    "alice",
                    "entity.employer",
                    "RumoredCo",
                    0.93,
                    "claim-reported",
                ),
            ])),
        };

        let answer = current_slot(&store, &scope, "alice", "entity.employer", Some(100))
            .expect("query")
            .expect("ranked answer");
        assert!(answer.abstain);
        assert!(answer.selected.is_none());
        assert_eq!(answer.candidates.len(), 1);
        assert_eq!(answer.candidates[0].truth_plane, GraphTruthPlane::Reported);
    }

    #[test]
    fn history_ranks_recent_world_state_change_above_older_state() {
        let scope = ScopeKey::default();
        let store = TestGraphStore {
            sidecar: Some(sidecar_for_states(vec![
                test_claim("claim-old", "asserted"),
                test_claim("claim-new", "asserted"),
                test_state_with_temporal(
                    "state-old",
                    "alice",
                    "entity.employer",
                    "OldCo",
                    0.74,
                    "claim-old",
                    Some(10),
                    Some(40),
                ),
                test_state_with_temporal(
                    "state-new",
                    "alice",
                    "entity.employer",
                    "Acme",
                    0.79,
                    "claim-new",
                    Some(40),
                    None,
                ),
            ])),
        };

        let answer = history(
            &store,
            &scope,
            &GraphHistoryQueryRequest {
                entity_id: "alice".to_owned(),
                slot_key: Some("entity.employer".to_owned()),
                since_valid_at: 0,
                until_valid_at: Some(80),
                recorded_at: Some(100),
                include_candidate_graph: false,
                truth_plane: GraphTruthPlane::WorldState,
                limit: Some(4),
            },
        )
        .expect("query")
        .expect("ranked history");

        assert!(!answer.abstain);
        assert_eq!(
            answer
                .selected
                .as_ref()
                .map(|candidate| candidate.change.state.value.as_str()),
            Some("Acme")
        );
        assert!(answer.candidates.len() >= 2);
    }

    #[test]
    fn causal_explanation_prefers_world_state_path_over_reported_path() {
        let scope = ScopeKey::default();
        let world_cause = "graph::event::memory::cause-world";
        let reported_cause = "graph::event::memory::cause-reported";
        let effect = "graph::event::memory::effect";
        let store = TestGraphStore {
            sidecar: Some(sidecar_for_projection(
                vec![
                    test_claim("claim-world", "asserted"),
                    test_claim("claim-reported", "reported"),
                    test_claim("claim-effect", "asserted"),
                    test_event(world_cause, "alice", Some(0), Some(20)),
                    test_event(reported_cause, "alice", Some(0), Some(20)),
                    test_event(effect, "alice", Some(20), None),
                ],
                vec![
                    support_edge(world_cause, "claim-world"),
                    support_edge(reported_cause, "claim-reported"),
                    support_edge(effect, "claim-effect"),
                    causal_edge(world_cause, effect, "supported", 0.82),
                    causal_edge(reported_cause, effect, "supported", 0.96),
                ],
            )),
        };

        let answer = causal_explanation(
            &store,
            &scope,
            &GraphCausalExplanationQueryRequest {
                target_vertex_id: effect.to_owned(),
                valid_at: Some(30),
                recorded_at: Some(100),
                include_candidate_graph: false,
                max_depth: 3,
                limit: Some(4),
                truth_plane: GraphTruthPlane::WorldState,
            },
        )
        .expect("query")
        .expect("ranked explanation");

        assert!(!answer.abstain);
        assert_eq!(
            answer
                .selected
                .as_ref()
                .map(|candidate| candidate.source_vertex_id.as_str()),
            Some(world_cause)
        );
        assert_eq!(
            answer
                .selected
                .as_ref()
                .map(|candidate| candidate.truth_plane),
            Some(GraphTruthPlane::WorldState)
        );
    }

    fn sidecar_for_states(vertices: Vec<KernelVertex>) -> GraphScopeSidecar {
        let mut graph_vertices = Vec::new();
        let mut graph_edges = Vec::new();
        for vertex in vertices {
            if vertex.kind == "state" {
                let claim_id = vertex
                    .attributes
                    .get("claimIds")
                    .and_then(serde_json::Value::as_array)
                    .and_then(|value| value.first())
                    .and_then(serde_json::Value::as_str)
                    .expect("state claim id");
                graph_edges.push(KernelEdge {
                    source_id: vertex.id.clone(),
                    target_id: KernelVertexId(format!("graph::claim::{claim_id}")),
                    edge_type: KernelEdgeType("supported_by".to_owned()),
                    relation_class: KernelRelationClass::Resolution,
                    weight: 1,
                    attributes: serde_json::json!({}),
                    data: None,
                    document_id: None,
                    narrative_id: None,
                    layer: KernelGraphLayer::Asserted,
                    temporal: vertex.temporal.clone(),
                    provenance: KernelProvenance::default(),
                    resolution_facet: None,
                });
            }
            graph_vertices.push(vertex);
        }

        sidecar_for_projection(graph_vertices, graph_edges)
    }

    fn sidecar_for_projection(
        graph_vertices: Vec<KernelVertex>,
        graph_edges: Vec<KernelEdge>,
    ) -> GraphScopeSidecar {
        GraphScopeSidecar {
            graph_batch: KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Projection {
                    scope_key: "__graph__".to_owned(),
                },
                recorded_at: Some(100),
                vertices: graph_vertices,
                edges: graph_edges,
            },
            ..GraphScopeSidecar::default()
        }
    }

    fn test_claim(claim_id: &str, modality: &str) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(format!("graph::claim::{claim_id}")),
            kind: "claim".to_owned(),
            class: KernelVertexClass::Generic,
            labels: vec![],
            weight: 1,
            value: serde_json::json!({
                "slotKey": "entity.employer",
                "objectValue": "value",
                "status": "active",
                "modality": modality,
            }),
            attributes: serde_json::json!({
                "sourceClass": "archive_relation"
            }),
            temporal: KernelBiTemporal {
                valid_from: Some(0),
                valid_to: None,
                recorded_at: Some(100),
                expired_at: None,
            },
            provenance: KernelProvenance::default(),
            entity_id: Some("alice".to_owned()),
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

    fn test_state(
        state_id: &str,
        entity_id: &str,
        slot_key: &str,
        value: &str,
        confidence: f64,
        claim_id: &str,
    ) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(format!("graph::state::{state_id}")),
            kind: "state".to_owned(),
            class: KernelVertexClass::State,
            labels: vec![slot_key.to_owned(), value.to_owned()],
            weight: (confidence * 1000.0) as i64,
            value: serde_json::json!({
                "slotKey": slot_key,
                "value": value,
                "status": "active",
                "sourceClass": "world",
            }),
            attributes: serde_json::json!({
                "confidenceMillis": (confidence * 1000.0).round() as u64,
                "claimIds": [claim_id],
            }),
            temporal: KernelBiTemporal {
                valid_from: Some(0),
                valid_to: None,
                recorded_at: Some(100),
                expired_at: None,
            },
            provenance: KernelProvenance {
                confidence: Some(confidence),
                ..KernelProvenance::default()
            },
            entity_id: Some(entity_id.to_owned()),
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

    fn test_state_with_temporal(
        state_id: &str,
        entity_id: &str,
        slot_key: &str,
        value: &str,
        confidence: f64,
        claim_id: &str,
        valid_from: Option<i64>,
        valid_to: Option<i64>,
    ) -> KernelVertex {
        let mut vertex = test_state(state_id, entity_id, slot_key, value, confidence, claim_id);
        vertex.temporal.valid_from = valid_from;
        vertex.temporal.valid_to = valid_to;
        vertex
    }

    fn test_event(
        event_id: &str,
        entity_id: &str,
        valid_from: Option<i64>,
        valid_to: Option<i64>,
    ) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(event_id.to_owned()),
            kind: "event".to_owned(),
            class: KernelVertexClass::Event,
            labels: vec!["event".to_owned()],
            weight: 1,
            value: serde_json::json!({
                "kind": "stateChange",
                "slotKey": "entity.employer",
            }),
            attributes: serde_json::json!({}),
            temporal: KernelBiTemporal {
                valid_from,
                valid_to,
                recorded_at: Some(100),
                expired_at: None,
            },
            provenance: KernelProvenance::default(),
            entity_id: Some(entity_id.to_owned()),
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

    fn support_edge(source_id: &str, claim_id: &str) -> KernelEdge {
        KernelEdge {
            source_id: KernelVertexId(source_id.to_owned()),
            target_id: KernelVertexId(format!("graph::claim::{claim_id}")),
            edge_type: KernelEdgeType("supported_by".to_owned()),
            relation_class: KernelRelationClass::Resolution,
            weight: 1,
            attributes: serde_json::json!({}),
            data: None,
            document_id: None,
            narrative_id: None,
            layer: KernelGraphLayer::Asserted,
            temporal: KernelBiTemporal {
                valid_from: Some(0),
                valid_to: None,
                recorded_at: Some(100),
                expired_at: None,
            },
            provenance: KernelProvenance::default(),
            resolution_facet: None,
        }
    }

    fn causal_edge(source_id: &str, target_id: &str, status: &str, confidence: f64) -> KernelEdge {
        KernelEdge {
            source_id: KernelVertexId(source_id.to_owned()),
            target_id: KernelVertexId(target_id.to_owned()),
            edge_type: KernelEdgeType("causal_link".to_owned()),
            relation_class: KernelRelationClass::Semantic,
            weight: (confidence * 1000.0) as i64,
            attributes: serde_json::json!({
                "status": status,
                "relationKind": "direct",
                "polarity": "positive",
            }),
            data: None,
            document_id: None,
            narrative_id: None,
            layer: KernelGraphLayer::Asserted,
            temporal: KernelBiTemporal {
                valid_from: Some(0),
                valid_to: None,
                recorded_at: Some(100),
                expired_at: None,
            },
            provenance: KernelProvenance {
                confidence: Some(confidence),
                evidence_refs: vec!["evidence://1".to_owned()],
                ..KernelProvenance::default()
            },
            resolution_facet: None,
        }
    }
}
