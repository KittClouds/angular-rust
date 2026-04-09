use phoenix_embed::{
    default_embedding_model_root, OrtTextEmbedConfig, OrtTextEmbedder, TextEmbeddingProfile,
};
use phoenix_graph::GraphBackendError;
use phoenix_graph_kernel::{
    KernelEdge, KernelGraphLayer, KernelGraphSnapshot, KernelMutationBatch, KernelMutationScope,
    PhoenixGraphKernel,
};
use phoenix_semantic_v2::scope_storage_key;
use phoenix_store_native_core::{PhoenixSemanticIndexStore, SemanticNodeNeighbor};
use phoenix_types::ScopeKey;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::api::GraphQueryError;
use crate::retrieval::{GraphRetrievedRegion, GraphRetrievedSeed};
use crate::semantic::ensure_ort_dylib_path;

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
    let mut included = FxHashSet::<String>::default();
    let mut frontier = Vec::<String>::new();
    for vertex_id in anchor_vertex_ids.iter().chain(seed_vertex_ids.iter()) {
        if included.insert(vertex_id.clone()) {
            frontier.push(vertex_id.clone());
        }
    }
    let all_edges = snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
        .collect::<Vec<_>>();
    let adjacency = region_adjacency(all_edges.as_slice());
    let node_limit = region_node_limit.clamp(8, 256);
    let mut truncated = false;
    for _ in 0..expansion_hops.clamp(1, 4) {
        if frontier.is_empty() || included.len() >= node_limit {
            break;
        }
        let mut next_frontier = Vec::new();
        for vertex_id in frontier {
            let Some(edges) = adjacency.get(vertex_id.as_str()) else {
                continue;
            };
            for edge in edges {
                if !edge_allowed(edge) {
                    continue;
                }
                for neighbor in [edge.source_id.0.as_str(), edge.target_id.0.as_str()] {
                    if included.len() >= node_limit && !included.contains(neighbor) {
                        truncated = true;
                        continue;
                    }
                    if included.insert(neighbor.to_owned()) {
                        next_frontier.push(neighbor.to_owned());
                    }
                }
            }
        }
        frontier = next_frontier;
    }
    let mut vertices = snapshot
        .vertices
        .iter()
        .filter(|vertex| included.contains(vertex.id.0.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let mut asserted_edges = snapshot
        .asserted_edges
        .iter()
        .filter(|edge| {
            included.contains(edge.source_id.0.as_str())
                && included.contains(edge.target_id.0.as_str())
                && edge_allowed(edge)
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut candidate_edges = snapshot
        .candidate_edges
        .iter()
        .filter(|edge| {
            included.contains(edge.source_id.0.as_str())
                && included.contains(edge.target_id.0.as_str())
                && edge_allowed(edge)
        })
        .cloned()
        .collect::<Vec<_>>();
    vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
    asserted_edges.sort_by(|left, right| left.source_id.0.cmp(&right.source_id.0));
    candidate_edges.sort_by(|left, right| left.source_id.0.cmp(&right.source_id.0));
    let mut included_vertex_ids = vertices
        .iter()
        .map(|vertex| vertex.id.0.clone())
        .collect::<Vec<_>>();
    included_vertex_ids.sort();
    let region = GraphRetrievedRegion {
        vertex_count: vertices.len(),
        asserted_edge_count: asserted_edges.len(),
        candidate_edge_count: candidate_edges.len(),
        truncated,
        anchor_vertex_ids,
        seed_vertex_ids,
        included_vertex_ids,
    };
    (
        KernelGraphSnapshot {
            vertices,
            asserted_edges,
            candidate_edges,
        },
        region,
    )
}

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

fn region_adjacency<'a>(edges: &[&'a KernelEdge]) -> FxHashMap<&'a str, Vec<&'a KernelEdge>> {
    let mut adjacency = FxHashMap::<&str, Vec<&KernelEdge>>::default();
    for edge in edges {
        adjacency
            .entry(edge.source_id.0.as_str())
            .or_default()
            .push(*edge);
        adjacency
            .entry(edge.target_id.0.as_str())
            .or_default()
            .push(*edge);
    }
    adjacency
}

fn embed_query(query_text: &str) -> Result<Vec<f32>, GraphQueryError> {
    let _ = ensure_ort_dylib_path();
    let embedder = OrtTextEmbedder::load(&OrtTextEmbedConfig {
        model_root: default_embedding_model_root(),
        batch_size: 1,
        max_length: 512,
        profile: TextEmbeddingProfile::Native384,
        prefix_passage: false,
    })
    .map_err(|error| GraphBackendError::Operation(format!("query embed load failed: {error}")))?;
    let rows = embedder.embed_texts(&[query_text]).map_err(|error| {
        GraphBackendError::Operation(format!("query embed inference failed: {error}"))
    })?;
    rows.into_iter().next().ok_or_else(|| {
        GraphQueryError::Kernel(GraphBackendError::Operation(
            "query embedder returned no vector".to_owned(),
        ))
    })
}
