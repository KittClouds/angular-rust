use phoenix_store_native_core::{
    PhoenixGraphPatchStore, PhoenixSemanticGraphPatchStore, PhoenixSemanticIndexStore,
};
use phoenix_types::ScopeKey;
use serde::{Deserialize, Serialize};

use crate::api::{
    GraphCausalExplanationQueryRequest, GraphHistoryQueryRequest, GraphQueryError,
    GraphRankedCausalExplanationAnswer, GraphRankedHistoryAnswer, GraphRankedSlotAnswer,
    GraphTruthPlane, GraphWorldStateQueryRequest,
};
use crate::retrieval_causal::retrieved_causal_explanation_impl;
use crate::retrieval_history::retrieved_history_impl;
use crate::retrieval_world::retrieved_world_state_impl;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRetrievedSeed {
    pub node_id: String,
    pub node_kind: String,
    pub score_millis: u32,
    pub distance_millis: u32,
    pub document_id: Option<String>,
    pub narrative_id: Option<String>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRetrievedRegion {
    pub vertex_count: usize,
    pub asserted_edge_count: usize,
    pub candidate_edge_count: usize,
    pub truncated: bool,
    #[serde(default)]
    pub anchor_vertex_ids: Vec<String>,
    #[serde(default)]
    pub seed_vertex_ids: Vec<String>,
    #[serde(default)]
    pub included_vertex_ids: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRetrievedWorldStateQueryRequest {
    pub query_text: String,
    pub entity_id: String,
    pub slot_key: String,
    pub valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
    pub seed_limit: usize,
    pub oversample: usize,
    pub expansion_hops: usize,
    pub region_node_limit: usize,
}

impl Default for GraphRetrievedWorldStateQueryRequest {
    fn default() -> Self {
        Self {
            query_text: String::new(),
            entity_id: String::new(),
            slot_key: String::new(),
            valid_at: Some(now_ms()),
            recorded_at: None,
            include_candidate_graph: true,
            seed_limit: 8,
            oversample: 20,
            expansion_hops: 2,
            region_node_limit: 96,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRetrievedHistoryQueryRequest {
    pub query_text: String,
    pub entity_id: String,
    pub slot_key: Option<String>,
    pub since_valid_at: i64,
    pub until_valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
    pub truth_plane: GraphTruthPlane,
    pub limit: Option<usize>,
    pub seed_limit: usize,
    pub oversample: usize,
    pub expansion_hops: usize,
    pub region_node_limit: usize,
}

impl Default for GraphRetrievedHistoryQueryRequest {
    fn default() -> Self {
        Self {
            query_text: String::new(),
            entity_id: String::new(),
            slot_key: None,
            since_valid_at: 0,
            until_valid_at: None,
            recorded_at: None,
            include_candidate_graph: true,
            truth_plane: GraphTruthPlane::WorldState,
            limit: Some(12),
            seed_limit: 8,
            oversample: 20,
            expansion_hops: 2,
            region_node_limit: 128,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRetrievedCausalExplanationQueryRequest {
    pub query_text: String,
    pub target_vertex_id: String,
    pub valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
    pub max_depth: usize,
    pub limit: Option<usize>,
    pub truth_plane: GraphTruthPlane,
    pub seed_limit: usize,
    pub oversample: usize,
    pub expansion_hops: usize,
    pub region_node_limit: usize,
}

impl Default for GraphRetrievedCausalExplanationQueryRequest {
    fn default() -> Self {
        Self {
            query_text: String::new(),
            target_vertex_id: String::new(),
            valid_at: None,
            recorded_at: None,
            include_candidate_graph: true,
            max_depth: 3,
            limit: Some(8),
            truth_plane: GraphTruthPlane::WorldState,
            seed_limit: 8,
            oversample: 20,
            expansion_hops: 3,
            region_node_limit: 144,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRetrievedWorldStateAnswer {
    pub query_text: String,
    pub query: GraphWorldStateQueryRequest,
    pub answer: GraphRankedSlotAnswer,
    #[serde(default)]
    pub seeds: Vec<GraphRetrievedSeed>,
    pub region: GraphRetrievedRegion,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRetrievedHistoryAnswer {
    pub query_text: String,
    pub query: GraphHistoryQueryRequest,
    pub answer: GraphRankedHistoryAnswer,
    #[serde(default)]
    pub seeds: Vec<GraphRetrievedSeed>,
    pub region: GraphRetrievedRegion,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRetrievedCausalExplanationAnswer {
    pub query_text: String,
    pub query: GraphCausalExplanationQueryRequest,
    pub answer: GraphRankedCausalExplanationAnswer,
    #[serde(default)]
    pub seeds: Vec<GraphRetrievedSeed>,
    pub region: GraphRetrievedRegion,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum GraphRetrievedQueryRequest {
    WorldState {
        request: GraphRetrievedWorldStateQueryRequest,
    },
    History {
        request: GraphRetrievedHistoryQueryRequest,
    },
    CausalExplanation {
        request: GraphRetrievedCausalExplanationQueryRequest,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum GraphRetrievedQueryAnswer {
    WorldState {
        answer: GraphRetrievedWorldStateAnswer,
    },
    History {
        answer: GraphRetrievedHistoryAnswer,
    },
    CausalExplanation {
        answer: GraphRetrievedCausalExplanationAnswer,
    },
}

pub fn retrieved_world_state<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedWorldStateQueryRequest,
) -> Result<Option<GraphRetrievedWorldStateAnswer>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore + PhoenixSemanticIndexStore,
{
    retrieved_world_state_impl(store, scope, request)
}

pub fn retrieved_history<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedHistoryQueryRequest,
) -> Result<Option<GraphRetrievedHistoryAnswer>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore + PhoenixSemanticIndexStore,
{
    retrieved_history_impl(store, scope, request)
}

pub fn retrieved_causal_explanation<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedCausalExplanationQueryRequest,
) -> Result<Option<GraphRetrievedCausalExplanationAnswer>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore + PhoenixSemanticIndexStore,
{
    retrieved_causal_explanation_impl(store, scope, request)
}

pub fn retrieved_query<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedQueryRequest,
) -> Result<Option<GraphRetrievedQueryAnswer>, GraphQueryError>
where
    S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore + PhoenixSemanticIndexStore,
{
    match request {
        GraphRetrievedQueryRequest::WorldState { request } => {
            retrieved_world_state(store, scope, request)
                .map(|answer| answer.map(|answer| GraphRetrievedQueryAnswer::WorldState { answer }))
        }
        GraphRetrievedQueryRequest::History { request } => retrieved_history(store, scope, request)
            .map(|answer| answer.map(|answer| GraphRetrievedQueryAnswer::History { answer })),
        GraphRetrievedQueryRequest::CausalExplanation { request } => {
            retrieved_causal_explanation(store, scope, request).map(|answer| {
                answer.map(|answer| GraphRetrievedQueryAnswer::CausalExplanation { answer })
            })
        }
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
