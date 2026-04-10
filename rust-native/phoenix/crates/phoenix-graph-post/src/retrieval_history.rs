use phoenix_graph_kernel::{
    entity_timeline_from_snapshot, what_changed_from_snapshot, KernelEdge, KernelQueryView,
    KernelStateIssue, KernelVertex, KernelViewRequest, KernelWhatChangedRequest,
};
use phoenix_store_native_core::{
    PhoenixGraphPatchStore, PhoenixSemanticGraphPatchStore, PhoenixSemanticIndexStore,
};
use phoenix_types::ScopeKey;

use crate::api::{
    load_projection_kernel, rank_history_answer, GraphHistoryQueryRequest, GraphQueryError,
};
use crate::phase4_graph_scoring::apply_graph_structural_history;
use crate::phase4_scoring::apply_phase4_history;
use crate::retrieval::{
    GraphRetrievedHistoryAnswer, GraphRetrievedHistoryQueryRequest, GraphRetrievedSeed,
};
use crate::retrieval_common::{
    build_region_from_snapshot, build_region_from_view, now_ms, retrieve_query_seeds,
};

const HISTORY_RETRIEVAL_KINDS: [&str; 5] = ["state", "claim", "event", "chunk", "entity"];

pub(crate) fn retrieved_history_impl<S>(
    store: &S,
    scope: &ScopeKey,
    request: &GraphRetrievedHistoryQueryRequest,
) -> Result<Option<GraphRetrievedHistoryAnswer>, GraphQueryError>
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
    let mut ranked = rank_history_answer(
        &query,
        until_valid_at,
        &timeline.vertices,
        &region_snapshot.candidate_edges,
        &changes,
        &timeline_issues(
            &timeline.vertices,
            "conflict",
            &request.entity_id,
            request.slot_key.as_deref(),
        ),
        &timeline_issues(
            &timeline.vertices,
            "gap",
            &request.entity_id,
            request.slot_key.as_deref(),
        ),
    );
    apply_phase4_history(request.query_text.as_str(), &mut ranked);
    apply_graph_structural_history(
        region.anchor_vertex_ids.as_slice(),
        &region_snapshot,
        &mut ranked,
    );
    Ok(Some(GraphRetrievedHistoryAnswer {
        query_text: request.query_text.clone(),
        answer: ranked,
        query,
        seeds,
        region,
    }))
}

pub(crate) fn build_history_region(
    snapshot: &phoenix_graph_kernel::KernelGraphSnapshot,
    request: &GraphRetrievedHistoryQueryRequest,
    seeds: &[GraphRetrievedSeed],
) -> (
    phoenix_graph_kernel::KernelGraphSnapshot,
    crate::retrieval::GraphRetrievedRegion,
) {
    let anchors = snapshot
        .vertices
        .iter()
        .filter(|vertex| vertex.entity_id.as_deref() == Some(request.entity_id.as_str()))
        .filter(|vertex| {
            vertex.kind == "entity"
                || request
                    .slot_key
                    .as_deref()
                    .map(|slot_key| slot_key_of(vertex) == Some(slot_key))
                    .unwrap_or(true)
        })
        .map(|vertex| vertex.id.0.clone())
        .collect::<Vec<_>>();
    build_region_from_snapshot(
        snapshot,
        anchors,
        seeds,
        request.region_node_limit,
        request.expansion_hops,
        history_edge_allowed,
    )
}

pub(crate) fn build_history_region_from_view(
    view: &KernelQueryView<'_>,
    request: &GraphRetrievedHistoryQueryRequest,
    seeds: &[GraphRetrievedSeed],
) -> (
    phoenix_graph_kernel::KernelGraphSnapshot,
    crate::retrieval::GraphRetrievedRegion,
) {
    let anchors = view
        .vertices()
        .iter()
        .filter(|vertex| vertex.entity_id.as_deref() == Some(request.entity_id.as_str()))
        .filter(|vertex| {
            vertex.kind == "entity"
                || request
                    .slot_key
                    .as_deref()
                    .map(|slot_key| slot_key_of(vertex) == Some(slot_key))
                    .unwrap_or(true)
        })
        .map(|vertex| vertex.id.0.clone())
        .collect::<Vec<_>>();
    build_region_from_view(
        view,
        anchors,
        seeds,
        request.region_node_limit,
        request.expansion_hops,
        history_edge_allowed,
    )
}

fn timeline_issues(
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

fn history_edge_allowed(edge: &KernelEdge) -> bool {
    matches!(
        edge.edge_type.0.as_str(),
        "state_of" | "state_value" | "supported_by" | "about" | "under_view"
    ) || (edge.edge_type.0.starts_with("semantic::")
        && edge.edge_type.0 != "semantic::missing_intermediate_cause")
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
