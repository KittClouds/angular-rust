use phoenix_graph_kernel::{KernelEdge, KernelViewRequest};
use phoenix_store_native_core::{
    PhoenixGraphPatchStore, PhoenixSemanticGraphPatchStore, PhoenixSemanticIndexStore,
};
use phoenix_types::ScopeKey;

use crate::api::{
    load_projection_kernel, rank_causal_explanation_answer, GraphCausalExplanationQueryRequest,
    GraphQueryError,
};
use crate::retrieval::{
    GraphRetrievedCausalExplanationAnswer, GraphRetrievedCausalExplanationQueryRequest,
    GraphRetrievedSeed,
};
use crate::retrieval_common::{build_region_from_snapshot, retrieve_query_seeds};

const CAUSAL_RETRIEVAL_KINDS: [&str; 5] = ["event", "claim", "entity", "chunk", "state"];

pub(crate) fn retrieved_causal_explanation_impl<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedCausalExplanationQueryRequest,
) -> Result<Option<GraphRetrievedCausalExplanationAnswer>, GraphQueryError>
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
    let (region_snapshot, region) = build_causal_region(&snapshot, request, &seeds);
    let query = GraphCausalExplanationQueryRequest {
        target_vertex_id: request.target_vertex_id.clone(),
        valid_at: request.valid_at,
        recorded_at: request.recorded_at,
        include_candidate_graph: request.include_candidate_graph,
        max_depth: request.max_depth,
        limit: request.limit,
        truth_plane: request.truth_plane,
    };
    Ok(Some(GraphRetrievedCausalExplanationAnswer {
        query_text: request.query_text.clone(),
        answer: rank_causal_explanation_answer(&query, &region_snapshot),
        query,
        seeds,
        region,
    }))
}

pub(crate) fn build_causal_region(
    snapshot: &phoenix_graph_kernel::KernelGraphSnapshot,
    request: &GraphRetrievedCausalExplanationQueryRequest,
    seeds: &[GraphRetrievedSeed],
) -> (
    phoenix_graph_kernel::KernelGraphSnapshot,
    crate::retrieval::GraphRetrievedRegion,
) {
    let mut anchors = vec![request.target_vertex_id.clone()];
    if let Some(target) = snapshot
        .vertices
        .iter()
        .find(|vertex| vertex.id.0 == request.target_vertex_id)
    {
        if let Some(entity_id) = target.entity_id.as_deref() {
            anchors.extend(
                snapshot
                    .vertices
                    .iter()
                    .filter(|vertex| {
                        vertex.kind == "entity" && vertex.entity_id.as_deref() == Some(entity_id)
                    })
                    .map(|vertex| vertex.id.0.clone()),
            );
        }
    }
    anchors.sort();
    anchors.dedup();
    build_region_from_snapshot(
        snapshot,
        anchors,
        seeds,
        request.region_node_limit,
        request.expansion_hops,
        causal_edge_allowed,
    )
}

fn causal_edge_allowed(edge: &KernelEdge) -> bool {
    matches!(
        edge.edge_type.0.as_str(),
        "causal_link" | "supported_by" | "canonicalized_as" | "subject" | "object" | "under_view"
    ) || edge.edge_type.0.starts_with("semantic::")
}
