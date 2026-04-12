use phoenix_graph_kernel::{
    entity_timeline_from_snapshot, slot_at_snapshot, what_changed_from_snapshot, KernelEdge,
    KernelGraphSnapshot, KernelSlotQueryRequest, KernelViewRequest, KernelWhatChangedRequest,
};
use phoenix_store_native_core::{
    PhoenixGraphPatchStore, PhoenixSemanticGraphPatchStore, PhoenixSemanticIndexStore,
};
use phoenix_types::ScopeKey;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::api::{
    load_projection_kernel, rank_causal_explanation_answer, rank_history_answer,
    rank_world_state_answer, GraphCausalExplanationQueryRequest, GraphHistoryQueryRequest,
    GraphQueryError, GraphRankedCausalExplanationAnswer, GraphRankedHistoryAnswer,
    GraphRankedSlotAnswer,
};
use crate::retrieval::{
    GraphRetrievedCausalExplanationQueryRequest, GraphRetrievedHistoryQueryRequest,
    GraphRetrievedRegion, GraphRetrievedWorldStateQueryRequest,
};
use crate::retrieval_causal::build_causal_region;
use crate::retrieval_common::retrieve_query_seeds;
use crate::retrieval_history::build_history_region;
use crate::retrieval_world::build_world_state_region;

const WORLD_RETRIEVAL_KINDS: [&str; 5] = ["state", "claim", "event", "chunk", "entity"];
const HISTORY_RETRIEVAL_KINDS: [&str; 5] = ["state", "claim", "event", "chunk", "entity"];
const CAUSAL_RETRIEVAL_KINDS: [&str; 5] = ["event", "claim", "entity", "chunk", "state"];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphSoftFamily {
    SameProcess,
    SameSlotFamily,
    RelatedEvent,
    ContradictorySupportRegion,
    MissingIntermediateCause,
}

impl GraphSoftFamily {
    pub fn edge_type(self) -> &'static str {
        match self {
            Self::SameProcess => "semantic::same_process",
            Self::SameSlotFamily => "semantic::same_slot_family",
            Self::RelatedEvent => "semantic::related_event",
            Self::ContradictorySupportRegion => "semantic::contradictory_support_region",
            Self::MissingIntermediateCause => "semantic::missing_intermediate_cause",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphAblationCase {
    pub name: String,
    #[serde(default)]
    pub families: Vec<GraphSoftFamily>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphSoftEdgeCount {
    pub family: GraphSoftFamily,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphEvalMetrics {
    pub abstain: bool,
    pub abstain_reason: Option<String>,
    pub candidate_count: usize,
    pub best_candidate_id: Option<String>,
    pub best_pre_structural_score_millis: Option<i64>,
    pub best_post_structural_score_millis: Option<i64>,
    pub best_candidate_hops: Option<usize>,
    pub selected_id: Option<String>,
    pub selected_label: Option<String>,
    pub selected_score_millis: Option<i64>,
    pub selected_structural_model: Option<String>,
    pub selected_structural_delta_millis: Option<i32>,
    pub selected_structural_proximity_millis: Option<u32>,
    pub seed_count: usize,
    pub region: GraphRetrievedRegion,
    #[serde(default)]
    pub soft_edge_counts: Vec<GraphSoftEdgeCount>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphAblationCaseResult {
    pub case_name: String,
    #[serde(default)]
    pub families: Vec<GraphSoftFamily>,
    pub metrics: GraphEvalMetrics,
}

pub fn default_ablation_cases() -> Vec<GraphAblationCase> {
    let full = all_soft_families();
    let mut cases = vec![
        GraphAblationCase {
            name: "hard_only".to_owned(),
            families: Vec::new(),
        },
        GraphAblationCase {
            name: "full_soft".to_owned(),
            families: full.clone(),
        },
    ];
    for family in full {
        cases.push(GraphAblationCase {
            name: format!(
                "minus_{}",
                family.edge_type().trim_start_matches("semantic::")
            ),
            families: all_soft_families()
                .into_iter()
                .filter(|candidate| *candidate != family)
                .collect(),
        });
    }
    cases
}

pub fn evaluate_world_state_cases<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedWorldStateQueryRequest,
    cases: &[GraphAblationCase],
) -> Result<Option<Vec<GraphAblationCaseResult>>, GraphQueryError>
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
    let snapshot = kernel.view_as_of(KernelViewRequest {
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let filtered = filter_snapshot_for_families(&snapshot, case.families.as_slice());
        let (region_snapshot, region) = build_world_state_region(&filtered, request, &seeds);
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
        let ranked = rank_world_state_answer(
            &region_snapshot.vertices,
            &region_snapshot.candidate_edges,
            &answer,
        );
        results.push(GraphAblationCaseResult {
            case_name: case.name.clone(),
            families: case.families.clone(),
            metrics: metrics_from_world_state(&ranked, seeds.len(), region, &region_snapshot),
        });
    }
    Ok(Some(results))
}

pub fn evaluate_history_cases<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedHistoryQueryRequest,
    cases: &[GraphAblationCase],
) -> Result<Option<Vec<GraphAblationCaseResult>>, GraphQueryError>
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
    let until_valid_at = request
        .until_valid_at
        .unwrap_or_else(crate::retrieval_common::now_ms);
    let snapshot = kernel.view_as_of(KernelViewRequest {
        valid_at: Some(until_valid_at),
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
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
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let filtered = filter_snapshot_for_families(&snapshot, case.families.as_slice());
        let (region_snapshot, region) = build_history_region(&filtered, request, &seeds);
        let timeline = entity_timeline_from_snapshot(
            &region_snapshot,
            &request.entity_id,
            Some((request.since_valid_at, until_valid_at)),
            request.recorded_at.or(Some(until_valid_at)),
        );
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
        let ranked = rank_history_answer(
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
        results.push(GraphAblationCaseResult {
            case_name: case.name.clone(),
            families: case.families.clone(),
            metrics: metrics_from_history(&ranked, seeds.len(), region, &region_snapshot),
        });
    }
    Ok(Some(results))
}

pub fn evaluate_causal_cases<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedCausalExplanationQueryRequest,
    cases: &[GraphAblationCase],
) -> Result<Option<Vec<GraphAblationCaseResult>>, GraphQueryError>
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
    let snapshot = kernel.view_as_of(KernelViewRequest {
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
    });
    let query = GraphCausalExplanationQueryRequest {
        target_vertex_id: request.target_vertex_id.clone(),
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
        max_depth: request.max_depth,
        limit: request.limit,
        truth_plane: request.truth_plane,
    };
    let mut results = Vec::with_capacity(cases.len());
    for case in cases {
        let filtered = filter_snapshot_for_families(&snapshot, case.families.as_slice());
        let (region_snapshot, region) = build_causal_region(&filtered, request, &seeds);
        let ranked = rank_causal_explanation_answer(&query, &region_snapshot);
        results.push(GraphAblationCaseResult {
            case_name: case.name.clone(),
            families: case.families.clone(),
            metrics: metrics_from_causal(&ranked, seeds.len(), region, &region_snapshot),
        });
    }
    Ok(Some(results))
}

pub(crate) fn filter_snapshot_for_families(
    snapshot: &KernelGraphSnapshot,
    families: &[GraphSoftFamily],
) -> KernelGraphSnapshot {
    let allowed = families
        .iter()
        .map(|family| family.edge_type())
        .collect::<std::collections::BTreeSet<_>>();
    let candidate_edges = snapshot
        .candidate_edges
        .iter()
        .filter(|edge| {
            if let Some(family) = soft_family_for_edge(edge) {
                allowed.contains(family.edge_type())
            } else {
                true
            }
        })
        .cloned()
        .collect::<Vec<_>>();
    KernelGraphSnapshot {
        vertices: snapshot.vertices.clone(),
        asserted_edges: snapshot.asserted_edges.clone(),
        candidate_edges,
    }
}

fn metrics_from_world_state(
    answer: &GraphRankedSlotAnswer,
    seed_count: usize,
    region: GraphRetrievedRegion,
    snapshot: &KernelGraphSnapshot,
) -> GraphEvalMetrics {
    let structural = answer
        .selected
        .as_ref()
        .and_then(|candidate| candidate.graph_structural_rerank.as_ref());
    let best = answer.candidates.first();
    GraphEvalMetrics {
        abstain: answer.abstain,
        abstain_reason: answer.abstain_reason.clone(),
        candidate_count: answer.candidates.len(),
        best_candidate_id: best.map(|candidate| candidate.state.state_vertex_id.clone()),
        best_pre_structural_score_millis: best
            .map(|candidate| (candidate.answer_score * 1000.0).round() as i64),
        best_post_structural_score_millis: best
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
        soft_edge_counts: collect_soft_edge_counts(&snapshot.candidate_edges),
    }
}

fn metrics_from_history(
    answer: &GraphRankedHistoryAnswer,
    seed_count: usize,
    region: GraphRetrievedRegion,
    snapshot: &KernelGraphSnapshot,
) -> GraphEvalMetrics {
    let structural = answer
        .selected
        .as_ref()
        .and_then(|candidate| candidate.graph_structural_rerank.as_ref());
    let best = answer.candidates.first();
    GraphEvalMetrics {
        abstain: answer.abstain,
        abstain_reason: answer.abstain_reason.clone(),
        candidate_count: answer.candidates.len(),
        best_candidate_id: best.map(|candidate| candidate.change.state.state_vertex_id.clone()),
        best_pre_structural_score_millis: best
            .map(|candidate| (candidate.answer_score * 1000.0).round() as i64),
        best_post_structural_score_millis: best
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
        soft_edge_counts: collect_soft_edge_counts(&snapshot.candidate_edges),
    }
}

fn metrics_from_causal(
    answer: &GraphRankedCausalExplanationAnswer,
    seed_count: usize,
    region: GraphRetrievedRegion,
    snapshot: &KernelGraphSnapshot,
) -> GraphEvalMetrics {
    let structural = answer
        .selected
        .as_ref()
        .and_then(|path| path.graph_structural_rerank.as_ref());
    let best = answer.candidates.first();
    GraphEvalMetrics {
        abstain: answer.abstain,
        abstain_reason: answer.abstain_reason.clone(),
        candidate_count: answer.candidates.len(),
        best_candidate_id: best.map(|path| path.source_vertex_id.clone()),
        best_pre_structural_score_millis: best
            .map(|path| (path.answer_score * 1000.0).round() as i64),
        best_post_structural_score_millis: best
            .map(|path| (path.answer_score * 1000.0).round() as i64),
        best_candidate_hops: best.map(|path| path.hops.len()),
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
        soft_edge_counts: collect_soft_edge_counts(&snapshot.candidate_edges),
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

fn all_soft_families() -> Vec<GraphSoftFamily> {
    vec![
        GraphSoftFamily::SameProcess,
        GraphSoftFamily::SameSlotFamily,
        GraphSoftFamily::RelatedEvent,
        GraphSoftFamily::ContradictorySupportRegion,
        GraphSoftFamily::MissingIntermediateCause,
    ]
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

fn collect_timeline_issues(
    vertices: &[phoenix_graph_kernel::KernelVertex],
    issue_kind: &str,
    entity_id: &str,
    slot_key: Option<&str>,
) -> Vec<phoenix_graph_kernel::KernelStateIssue> {
    let mut issues = vertices
        .iter()
        .filter(|vertex| vertex.kind == issue_kind)
        .filter(|vertex| vertex.entity_id.as_deref() == Some(entity_id))
        .filter(|vertex| {
            slot_key
                .map(|key| slot_key_of(vertex) == Some(key))
                .unwrap_or(true)
        })
        .map(|vertex| phoenix_graph_kernel::KernelStateIssue {
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

fn slot_key_of(vertex: &phoenix_graph_kernel::KernelVertex) -> Option<&str> {
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
