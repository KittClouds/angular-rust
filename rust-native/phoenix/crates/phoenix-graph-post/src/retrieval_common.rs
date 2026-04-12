use phoenix_embed::{
    default_embedding_model_root, OrtTextEmbedConfig, OrtTextEmbedder, TextEmbeddingProfile,
};
use phoenix_graph::GraphBackendError;
use phoenix_graph_kernel::{
    expand_snapshot_region, KernelEdge, KernelGraphSnapshot, KernelQueryView, KernelRegionProfile,
};
#[cfg(test)]
use phoenix_graph_kernel::{
    KernelGraphLayer, KernelMutationBatch, KernelMutationScope, PhoenixGraphKernel,
};
#[cfg(test)]
use phoenix_semantic_v2::scope_storage_key;
use phoenix_store_native_core::{PhoenixSemanticIndexStore, SemanticNodeNeighbor};
use phoenix_types::ScopeKey;
use rustc_hash::FxHashMap;
use std::cell::RefCell;

use crate::api::GraphQueryError;
use crate::retrieval::{GraphRetrievedRegion, GraphRetrievedSeed};
use crate::runtime_telemetry::{measure_graph_runtime, record_region_build, GraphRuntimeMetric};
use crate::semantic::ensure_ort_dylib_path;

thread_local! {
    static QUERY_EMBEDDER_CACHE: RefCell<QueryEmbedderCache> =
        RefCell::new(QueryEmbedderCache::default());
}

#[derive(Default)]
struct QueryEmbedderCache {
    attempted: bool,
    embedder: Option<OrtTextEmbedder>,
}

pub(crate) fn clear_query_embedder_cache() {
    QUERY_EMBEDDER_CACHE.with(|cell| {
        *cell.borrow_mut() = QueryEmbedderCache::default();
    });
}

pub(crate) fn retrieve_query_seeds<S>(
    store: &S,
    scope: &ScopeKey,
    query_text: &str,
    kinds: &[&str],
    seed_limit: usize,
    oversample: usize,
) -> Result<Vec<GraphRetrievedSeed>, GraphQueryError>
where
    S: PhoenixSemanticIndexStore,
{
    let _timer = measure_graph_runtime(GraphRuntimeMetric::RetrieveQuerySeeds);
    if query_text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let embedding = embed_query(query_text)?;
    let seed_limit = seed_limit.clamp(1, 24);
    let oversample = oversample.max(seed_limit).clamp(seed_limit, 96);
    let mut best = FxHashMap::<String, GraphRetrievedSeed>::default();
    for kind in kinds {
        for hit in store
            .query_semantic_node_neighbors(&embedding, scope, kind, None, seed_limit, oversample)?
        {
            let candidate = seed_from_neighbor(hit);
            match best.get(candidate.node_id.as_str()) {
                Some(existing) if existing.score_millis >= candidate.score_millis => {}
                _ => {
                    best.insert(candidate.node_id.clone(), candidate);
                }
            }
        }
    }
    let mut seeds = best.into_values().collect::<Vec<_>>();
    seeds.sort_by(|left, right| {
        right
            .score_millis
            .cmp(&left.score_millis)
            .then_with(|| left.node_id.cmp(&right.node_id))
    });
    seeds.truncate(seed_limit);
    Ok(seeds)
}

pub(crate) fn build_region_from_snapshot(
    snapshot: &KernelGraphSnapshot,
    anchor_vertex_ids: Vec<String>,
    seeds: &[GraphRetrievedSeed],
    region_node_limit: usize,
    expansion_hops: usize,
    edge_allowed: fn(&KernelEdge) -> bool,
) -> (KernelGraphSnapshot, GraphRetrievedRegion) {
    let _timer = measure_graph_runtime(GraphRuntimeMetric::BuildRegionFromSnapshot);
    let seed_vertex_ids = seeds
        .iter()
        .filter(|seed| {
            snapshot
                .vertices
                .iter()
                .any(|vertex| vertex.id.0 == seed.node_id)
        })
        .map(|seed| seed.node_id.clone())
        .collect::<Vec<_>>();
    let expanded = expand_snapshot_region(
        snapshot,
        anchor_vertex_ids.as_slice(),
        seed_vertex_ids.as_slice(),
        region_node_limit,
        expansion_hops,
        edge_allowed,
    );
    let region = GraphRetrievedRegion {
        vertex_count: expanded.snapshot.vertices.len(),
        asserted_edge_count: expanded.snapshot.asserted_edges.len(),
        candidate_edge_count: expanded.snapshot.candidate_edges.len(),
        truncated: expanded.truncated,
        anchor_vertex_ids,
        seed_vertex_ids: expanded.seed_vertex_ids,
        included_vertex_ids: expanded.included_vertex_ids,
    };
    record_region_build(
        region.vertex_count,
        region.asserted_edge_count,
        region.candidate_edge_count,
    );
    (expanded.snapshot, region)
}

#[allow(dead_code)]
pub(crate) fn build_region_from_view(
    view: &KernelQueryView<'_>,
    anchor_vertex_ids: Vec<String>,
    seeds: &[GraphRetrievedSeed],
    region_node_limit: usize,
    expansion_hops: usize,
    edge_allowed: fn(&KernelEdge) -> bool,
) -> (KernelGraphSnapshot, GraphRetrievedRegion) {
    build_region_from_view_profile(
        view,
        anchor_vertex_ids,
        seeds,
        region_node_limit,
        expansion_hops,
        edge_allowed,
        KernelRegionProfile::Generic,
    )
}

pub(crate) fn build_region_from_view_profile(
    view: &KernelQueryView<'_>,
    anchor_vertex_ids: Vec<String>,
    seeds: &[GraphRetrievedSeed],
    region_node_limit: usize,
    expansion_hops: usize,
    edge_allowed: fn(&KernelEdge) -> bool,
    profile: KernelRegionProfile,
) -> (KernelGraphSnapshot, GraphRetrievedRegion) {
    let _timer = measure_graph_runtime(GraphRuntimeMetric::BuildRegionFromView);
    let seed_vertex_ids = seeds
        .iter()
        .filter(|seed| view.find_vertex(seed.node_id.as_str()).is_some())
        .map(|seed| seed.node_id.clone())
        .collect::<Vec<_>>();
    let expanded = view.expand_region_with_profile(
        anchor_vertex_ids.as_slice(),
        seed_vertex_ids.as_slice(),
        region_node_limit,
        expansion_hops,
        edge_allowed,
        profile,
    );
    let region = GraphRetrievedRegion {
        vertex_count: expanded.snapshot.vertices.len(),
        asserted_edge_count: expanded.snapshot.asserted_edges.len(),
        candidate_edge_count: expanded.snapshot.candidate_edges.len(),
        truncated: expanded.truncated,
        anchor_vertex_ids,
        seed_vertex_ids: expanded.seed_vertex_ids,
        included_vertex_ids: expanded.included_vertex_ids,
    };
    record_region_build(
        region.vertex_count,
        region.asserted_edge_count,
        region.candidate_edge_count,
    );
    (expanded.snapshot, region)
}

#[cfg(test)]
pub(crate) fn kernel_from_snapshot(
    scope: &ScopeKey,
    snapshot: &KernelGraphSnapshot,
) -> Result<PhoenixGraphKernel, GraphBackendError> {
    let mut kernel = PhoenixGraphKernel::new();
    if !snapshot.vertices.is_empty() || !snapshot.asserted_edges.is_empty() {
        kernel.apply_kernel_batch(KernelMutationBatch {
            layer: KernelGraphLayer::Asserted,
            scope: KernelMutationScope::Projection {
                scope_key: format!("region:{}", scope_storage_key(scope)),
            },
            recorded_at: None,
            vertices: snapshot.vertices.clone(),
            edges: snapshot.asserted_edges.clone(),
        })?;
    }
    if !snapshot.candidate_edges.is_empty() {
        kernel.apply_kernel_batch(KernelMutationBatch {
            layer: KernelGraphLayer::Candidate,
            scope: KernelMutationScope::Candidate {
                scope_key: format!("region:{}", scope_storage_key(scope)),
            },
            recorded_at: None,
            vertices: Vec::new(),
            edges: snapshot.candidate_edges.clone(),
        })?;
    }
    Ok(kernel)
}

pub(crate) fn score_from_distance(distance: f64) -> u32 {
    ((1.0 / (1.0 + distance.max(0.0))) * 1000.0)
        .round()
        .clamp(0.0, 1000.0) as u32
}

pub(crate) fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn seed_from_neighbor(hit: SemanticNodeNeighbor) -> GraphRetrievedSeed {
    GraphRetrievedSeed {
        score_millis: score_from_distance(hit.distance),
        distance_millis: (hit.distance.max(0.0) * 1000.0).round() as u32,
        node_id: hit.node_id,
        node_kind: hit.node_kind,
        document_id: hit.document_id,
        narrative_id: hit.narrative_id,
        evidence_refs: hit.evidence_refs,
    }
}

fn embed_query(query_text: &str) -> Result<Vec<f32>, GraphQueryError> {
    let _timer = measure_graph_runtime(GraphRuntimeMetric::EmbedQuery);
    with_query_embedder(|embedder| {
        let rows = embedder.embed_texts(&[query_text]).map_err(|error| {
            GraphBackendError::Operation(format!("query embed inference failed: {error}"))
        })?;
        rows.into_iter().next().ok_or_else(|| {
            GraphQueryError::Kernel(GraphBackendError::Operation(
                "query embedder returned no vector".to_owned(),
            ))
        })
    })
}

fn with_query_embedder<R>(
    f: impl FnOnce(&OrtTextEmbedder) -> Result<R, GraphQueryError>,
) -> Result<R, GraphQueryError> {
    QUERY_EMBEDDER_CACHE.with(|cell| {
        let mut cache = cell.borrow_mut();
        if !cache.attempted {
            let _timer = measure_graph_runtime(GraphRuntimeMetric::QueryEmbedderLoad);
            cache.attempted = true;
            let _ = ensure_ort_dylib_path();
            cache.embedder = Some(
                OrtTextEmbedder::load(&OrtTextEmbedConfig {
                    model_root: default_embedding_model_root(),
                    batch_size: 1,
                    max_length: 512,
                    profile: TextEmbeddingProfile::Native384,
                    prefix_passage: false,
                })
                .map_err(|error| {
                    GraphBackendError::Operation(format!("query embed load failed: {error}"))
                })?,
            );
        }
        let embedder = cache.embedder.as_ref().ok_or_else(|| {
            GraphQueryError::Kernel(GraphBackendError::Operation(
                "query embedder was unavailable".to_owned(),
            ))
        })?;
        f(embedder)
    })
}
