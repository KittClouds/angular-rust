use phoenix_graph_kernel::{
    entity_timeline_from_snapshot, slot_at_snapshot, what_changed_from_snapshot,
    KernelLocalDiffusionKind, KernelSlotQueryRequest, KernelStateIssue, KernelVertex,
    KernelViewRequest, KernelWhatChangedRequest,
};
use phoenix_store_native_core::{
    PhoenixGraphPatchStore, PhoenixSemanticGraphPatchStore, PhoenixSemanticIndexStore,
};
use phoenix_types::ScopeKey;
use serde::{Deserialize, Serialize};

use crate::api::{
    load_projection_kernel, rank_causal_explanation_answer, rank_history_answer,
    rank_world_state_answer, GraphCausalExplanationQueryRequest, GraphHistoryQueryRequest,
    GraphQueryError,
};
use crate::diffusion_metrics::{
    metrics_from_causal, metrics_from_history, metrics_from_world_state,
};
use crate::eval::GraphEvalMetrics;
use crate::phase4_graph_scoring::{
    apply_graph_structural_causal_with_diffusion, apply_graph_structural_history_with_diffusion,
    apply_graph_structural_world_state_with_diffusion,
};
use crate::phase4_scoring::{apply_phase4_causal, apply_phase4_history, apply_phase4_world_state};
use crate::retrieval::{
    GraphRetrievedCausalExplanationQueryRequest, GraphRetrievedHistoryQueryRequest,
    GraphRetrievedWorldStateQueryRequest,
};
use crate::retrieval_causal::build_causal_region_from_view;
use crate::retrieval_common::{now_ms, retrieve_query_seeds};
use crate::retrieval_history::build_history_region_from_view;
use crate::retrieval_world::build_world_state_region_from_view;

const WORLD_RETRIEVAL_KINDS: [&str; 5] = ["state", "claim", "event", "chunk", "entity"];
const HISTORY_RETRIEVAL_KINDS: [&str; 5] = ["state", "claim", "event", "chunk", "entity"];
const CAUSAL_RETRIEVAL_KINDS: [&str; 5] = ["event", "claim", "entity", "chunk", "state"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphDiffusionCase {
    PersonalizedPagerank,
    HeatKernel,
}

impl GraphDiffusionCase {
    fn kernel_kind(self) -> KernelLocalDiffusionKind {
        match self {
            Self::PersonalizedPagerank => KernelLocalDiffusionKind::PersonalizedPagerank,
            Self::HeatKernel => KernelLocalDiffusionKind::HeatKernel,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::PersonalizedPagerank => "personalized_pagerank",
            Self::HeatKernel => "heat_kernel",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDiffusionCaseResult {
    pub case_name: String,
    pub diffusion: GraphDiffusionCase,
    pub metrics: GraphEvalMetrics,
}

pub fn default_diffusion_cases() -> Vec<GraphDiffusionCase> {
    vec![
        GraphDiffusionCase::PersonalizedPagerank,
        GraphDiffusionCase::HeatKernel,
    ]
}

pub fn evaluate_world_state_diffusion_cases<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedWorldStateQueryRequest,
    cases: &[GraphDiffusionCase],
) -> Result<Option<Vec<GraphDiffusionCaseResult>>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore + PhoenixSemanticIndexStore,
{
    let Some(kernel) = load_projection_kernel(store, scope)? else {
        return Ok(None);
    };
    let seeds = retrieve_query_seeds(
        store,
        scope,
        request.query_text.as_str(),
        &WORLD_RETRIEVAL_KINDS,
        request.seed_limit,
        request.oversample,
    )?;
    let view = kernel.query_view(KernelViewRequest {
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    let (region_snapshot, region) = build_world_state_region_from_view(&view, request, &seeds);
    let answer = slot_at_snapshot(
        &region_snapshot,
        &KernelSlotQueryRequest {
            entity_id: request.entity_id.clone(),
            slot_key: request.slot_key.clone(),
            valid_at: request.valid_at,
            recorded_at: request.recorded_at,
            include_candidate_graph: request.include_candidate_graph,
        },
    );
    let mut ranked = rank_world_state_answer(
        &region_snapshot.vertices,
        &region_snapshot.candidate_edges,
        &answer,
    );
    apply_phase4_world_state(request.query_text.as_str(), &mut ranked);
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let mut candidate = ranked.clone();
        apply_graph_structural_world_state_with_diffusion(
            region.anchor_vertex_ids.as_slice(),
            &region_snapshot,
            &mut candidate,
            case.kernel_kind(),
        );
        results.push(GraphDiffusionCaseResult {
            case_name: case.label().to_owned(),
            diffusion: *case,
            metrics: metrics_from_world_state(
                &ranked,
                &candidate,
                seeds.len(),
                region.clone(),
                &region_snapshot.candidate_edges,
            ),
        });
    }
    Ok(Some(results))
}

pub fn evaluate_history_diffusion_cases<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedHistoryQueryRequest,
    cases: &[GraphDiffusionCase],
) -> Result<Option<Vec<GraphDiffusionCaseResult>>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore + PhoenixSemanticIndexStore,
{
    let Some(kernel) = load_projection_kernel(store, scope)? else {
        return Ok(None);
    };
    let seeds = retrieve_query_seeds(
        store,
        scope,
        request.query_text.as_str(),
        &HISTORY_RETRIEVAL_KINDS,
        request.seed_limit,
        request.oversample,
    )?;
    let until_valid_at = request.until_valid_at.unwrap_or_else(now_ms);
    let view = kernel.query_view(KernelViewRequest {
        valid_at: Some(until_valid_at),
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    let (region_snapshot, region) = build_history_region_from_view(&view, request, &seeds);
    let timeline = entity_timeline_from_snapshot(
        &region_snapshot,
        &request.entity_id,
        Some((request.since_valid_at, until_valid_at)),
        request.recorded_at.or(Some(until_valid_at)),
    );
    let query = GraphHistoryQueryRequest {
        entity_id: request.entity_id.clone(),
        slot_key: request.slot_key.clone(),
        since_valid_at: request.since_valid_at,
        until_valid_at: Some(until_valid_at),
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
        truth_plane: request.truth_plane,
        limit: request.limit,
    };
    let changes = what_changed_from_snapshot(
        &timeline,
        &KernelWhatChangedRequest {
            entity_id: request.entity_id.clone(),
            slot_key: request.slot_key.clone(),
            since_valid_at: request.since_valid_at,
            until_valid_at: Some(until_valid_at),
            recorded_at: request.recorded_at,
            include_candidate_graph: request.include_candidate_graph,
        },
    );
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
    let mut ranked = rank_history_answer(
        &query,
        until_valid_at,
        &timeline.vertices,
        &timeline.vertices,
        &timeline.asserted_edges,
        &timeline.candidate_edges,
        &changes,
        &conflicts,
        &gaps,
    );
    apply_phase4_history(request.query_text.as_str(), &mut ranked);
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let mut candidate = ranked.clone();
        apply_graph_structural_history_with_diffusion(
            region.anchor_vertex_ids.as_slice(),
            &region_snapshot,
            &mut candidate,
            case.kernel_kind(),
        );
        results.push(GraphDiffusionCaseResult {
            case_name: case.label().to_owned(),
            diffusion: *case,
            metrics: metrics_from_history(
                &ranked,
                &candidate,
                seeds.len(),
                region.clone(),
                &region_snapshot.candidate_edges,
            ),
        });
    }
    Ok(Some(results))
}

pub fn evaluate_causal_diffusion_cases<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedCausalExplanationQueryRequest,
    cases: &[GraphDiffusionCase],
) -> Result<Option<Vec<GraphDiffusionCaseResult>>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore + PhoenixSemanticIndexStore,
{
    let Some(kernel) = load_projection_kernel(store, scope)? else {
        return Ok(None);
    };
    let seeds = retrieve_query_seeds(
        store,
        scope,
        request.query_text.as_str(),
        &CAUSAL_RETRIEVAL_KINDS,
        request.seed_limit,
        request.oversample,
    )?;
    let view = kernel.query_view(KernelViewRequest {
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    let (region_snapshot, region) = build_causal_region_from_view(&view, request, &seeds);
    let query = GraphCausalExplanationQueryRequest {
        target_vertex_id: request.target_vertex_id.clone(),
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
        max_depth: request.max_depth,
        limit: request.limit,
        truth_plane: request.truth_plane,
    };
    let mut ranked = rank_causal_explanation_answer(&query, &region_snapshot);
    apply_phase4_causal(
        request.query_text.as_str(),
        &region_snapshot.vertices,
        &mut ranked,
    );
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let mut candidate = ranked.clone();
        apply_graph_structural_causal_with_diffusion(
            region.anchor_vertex_ids.as_slice(),
            &region_snapshot,
            &mut candidate,
            case.kernel_kind(),
        );
        results.push(GraphDiffusionCaseResult {
            case_name: case.label().to_owned(),
            diffusion: *case,
            metrics: metrics_from_causal(
                &ranked,
                &candidate,
                seeds.len(),
                region.clone(),
                &region_snapshot.candidate_edges,
            ),
        });
    }
    Ok(Some(results))
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
                .map(|key| slot_key_of(vertex) == Some(key))
                .unwrap_or(true)
        })
        .map(|vertex| KernelStateIssue {
            issue_vertex_id: vertex.id.0.clone(),
            entity_id: vertex.entity_id.clone().unwrap_or_default(),
            slot_key: slot_key_of(vertex).unwrap_or_default().to_owned(),
            issue_kind: string_attr(&vertex.value, "kind")
                .unwrap_or(issue_kind)
                .to_owned(),
            reason: string_attr(&vertex.attributes, "reason").map(str::to_owned),
            detail: string_attr(&vertex.value, "detail").map(str::to_owned),
            status: string_attr(&vertex.value, "status").map(str::to_owned),
            preferred_claim_id: string_attr(&vertex.attributes, "preferredClaimId")
                .map(str::to_owned),
            temporal: vertex.temporal.clone(),
            supporting_claim_ids: string_list_attr(&vertex.attributes, "claimIds"),
        })
        .collect::<Vec<_>>();
    issues.sort_by(|left, right| {
        left.temporal
            .valid_from
            .cmp(&right.temporal.valid_from)
            .then_with(|| left.issue_vertex_id.cmp(&right.issue_vertex_id))
    });
    issues
}

fn slot_key_of(vertex: &KernelVertex) -> Option<&str> {
    string_attr(&vertex.value, "slotKey").or_else(|| string_attr(&vertex.attributes, "slotKey"))
}

fn string_attr<'a>(value: &'a serde_json::Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(serde_json::Value::as_str)
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
