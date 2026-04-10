use phoenix_graph_kernel::{slot_at_snapshot, KernelEdge, KernelQueryView, KernelSlotQueryRequest, KernelViewRequest};
use phoenix_store_native_core::{
    PhoenixGraphPatchStore, PhoenixSemanticGraphPatchStore, PhoenixSemanticIndexStore,
};
use phoenix_types::ScopeKey;

use crate::api::{
    load_projection_kernel, rank_world_state_answer, GraphQueryError, GraphWorldStateQueryRequest,
};
use crate::phase4_graph_scoring::apply_graph_structural_world_state;
use crate::phase4_scoring::apply_phase4_world_state;
use crate::retrieval::{GraphRetrievedWorldStateAnswer, GraphRetrievedWorldStateQueryRequest};
use crate::retrieval_common::{build_region_from_snapshot, build_region_from_view, retrieve_query_seeds};

const WORLD_RETRIEVAL_KINDS: [&str; 5] = ["state", "claim", "event", "chunk", "entity"];

pub(crate) fn retrieved_world_state_impl<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedWorldStateQueryRequest,
) -> Result<Option<GraphRetrievedWorldStateAnswer>, GraphQueryError>
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
    apply_graph_structural_world_state(
        region.anchor_vertex_ids.as_slice(),
        &region_snapshot,
        &mut ranked,
    );
    Ok(Some(GraphRetrievedWorldStateAnswer {
        query_text: request.query_text.clone(),
        query: GraphWorldStateQueryRequest {
            entity_id: request.entity_id.clone(),
            slot_key: request.slot_key.clone(),
            valid_at: request.valid_at,
            recorded_at: request.recorded_at,
            include_candidate_graph: request.include_candidate_graph,
        },
        answer: ranked,
        seeds,
        region,
    }))
}

pub(crate) fn build_world_state_region(
    snapshot: &phoenix_graph_kernel::KernelGraphSnapshot,
    request: &GraphRetrievedWorldStateQueryRequest,
    seeds: &[crate::retrieval::GraphRetrievedSeed],
) -> (
    phoenix_graph_kernel::KernelGraphSnapshot,
    crate::retrieval::GraphRetrievedRegion,
) {
    let anchors = snapshot
        .vertices
        .iter()
        .filter(|vertex| vertex.entity_id.as_deref() == Some(request.entity_id.as_str()))
        .filter(|vertex| {
            vertex.kind == "entity" || slot_key_of(vertex) == Some(request.slot_key.as_str())
        })
        .map(|vertex| vertex.id.0.clone())
        .collect::<Vec<_>>();
    build_region_from_snapshot(
        snapshot,
        anchors,
        seeds,
        request.region_node_limit,
        request.expansion_hops,
        world_state_edge_allowed,
    )
}

pub(crate) fn build_world_state_region_from_view(
    view: &KernelQueryView<'_>,
    request: &GraphRetrievedWorldStateQueryRequest,
    seeds: &[crate::retrieval::GraphRetrievedSeed],
) -> (
    phoenix_graph_kernel::KernelGraphSnapshot,
    crate::retrieval::GraphRetrievedRegion,
) {
    let anchors = view
        .vertices()
        .iter()
        .filter(|vertex| vertex.entity_id.as_deref() == Some(request.entity_id.as_str()))
        .filter(|vertex| {
            vertex.kind == "entity" || slot_key_of(vertex) == Some(request.slot_key.as_str())
        })
        .map(|vertex| vertex.id.0.clone())
        .collect::<Vec<_>>();
    build_region_from_view(
        view,
        anchors,
        seeds,
        request.region_node_limit,
        request.expansion_hops,
        world_state_edge_allowed,
    )
}

fn slot_key_of(vertex: &phoenix_graph_kernel::KernelVertex) -> Option<&str> {
    vertex
        .value
        .get("slotKey")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            vertex
                .attributes
                .get("slotKey")
                .and_then(serde_json::Value::as_str)
        })
}

fn world_state_edge_allowed(edge: &KernelEdge) -> bool {
    matches!(
        edge.edge_type.0.as_str(),
        "state_of" | "state_value" | "supported_by" | "about" | "under_view"
    ) || (edge.edge_type.0.starts_with("semantic::")
        && edge.edge_type.0 != "semantic::related_event"
        && edge.edge_type.0 != "semantic::missing_intermediate_cause")
}
