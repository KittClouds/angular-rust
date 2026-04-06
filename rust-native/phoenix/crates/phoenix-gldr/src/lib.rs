use std::collections::{HashSet, VecDeque};

use phoenix_graptor::{
    load_graph_snapshot_with_candidate_graph, GraptorEdge, GraptorGraph, GraptorVertex,
};
use phoenix_lex::LexIndex;
use phoenix_store_cozo::{PhoenixCozoStore, SemanticNeighbor, StoreError};
use phoenix_types::{
    BoundaryKind, ChunkHit, Diagnostic, EntityId, NodeHit, QueryRequest, QueryResult,
    TemporalMarker,
};
use rustc_hash::FxHashMap;
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct GldrConfig {
    pub seed_limit: usize,
    pub subgraph_hops: usize,
    pub ppr_alpha: f64,
    pub ppr_alpha_leaf: f64,
    pub ppr_alpha_entity: f64,
    pub ppr_alpha_chapter: f64,
    pub ppr_iterations: usize,
    pub ppr_tolerance: f64,
}

impl Default for GldrConfig {
    fn default() -> Self {
        Self {
            seed_limit: 12,
            subgraph_hops: 3,
            ppr_alpha: 0.15,
            ppr_alpha_leaf: 0.05,
            ppr_alpha_entity: 0.25,
            ppr_alpha_chapter: 0.15,
            ppr_iterations: 24,
            ppr_tolerance: 1e-6,
        }
    }
}

pub struct PhoenixGldr {
    config: GldrConfig,
}

#[derive(Clone, Debug, Default)]
struct SemanticResolution {
    hits: Vec<SemanticNeighbor>,
    shortlisted_documents: usize,
    filtered_leaf_hits: usize,
    used_global_fallback: bool,
}

impl PhoenixGldr {
    pub fn new(config: GldrConfig) -> Self {
        Self { config }
    }

    pub fn query(
        &self,
        store: &PhoenixCozoStore,
        lex: &LexIndex,
        request: &QueryRequest,
    ) -> Result<QueryResult, StoreError> {
        self.query_with_snapshot(store, lex, request, None)
    }

    pub fn query_with_graph(
        &self,
        store: &PhoenixCozoStore,
        lex: &LexIndex,
        request: &QueryRequest,
        graph: &GraptorGraph,
    ) -> Result<QueryResult, StoreError> {
        self.query_with_snapshot(store, lex, request, Some(graph))
    }

    fn query_with_snapshot(
        &self,
        store: &PhoenixCozoStore,
        lex: &LexIndex,
        request: &QueryRequest,
        snapshot: Option<&GraptorGraph>,
    ) -> Result<QueryResult, StoreError> {
        let limit = request.limit.unwrap_or(5).max(1);
        let semantic_requested = request
            .targets
            .iter()
            .any(|target| matches!(target, phoenix_types::QueryTarget::Semantic));
        let lexical = lex.search(
            &request.query,
            &request.scope,
            limit.max(self.config.seed_limit) * 2,
        );
        let semantic_resolution = if semantic_requested {
            self.resolve_semantic_hits(store, request, limit)?
        } else {
            SemanticResolution::default()
        };
        let semantic_hits = &semantic_resolution.hits;
        let seed_vertex_ids: Vec<String> = lexical
            .span_hits
            .iter()
            .take(self.config.seed_limit.max(limit))
            .map(|hit| leaf_vertex_id(&hit.span_id))
            .chain(
                semantic_hits
                    .iter()
                    .take(self.config.seed_limit.max(limit))
                    .map(|hit| leaf_vertex_id(&hit.span_id)),
            )
            .collect();
        let graph = if let Some(snapshot) = snapshot {
            load_subgraph_from_graph(snapshot, &seed_vertex_ids, self.config.subgraph_hops)
        } else {
            load_subgraph(
                store,
                &seed_vertex_ids,
                self.config.subgraph_hops,
                request.include_candidate_graph,
            )?
        };
        let temporal = TemporalFilter::from_marker(request.temporal.as_ref());
        let wants_chunks = request.targets.is_empty()
            || request.targets.iter().any(|target| {
                matches!(
                    target,
                    phoenix_types::QueryTarget::Chunks
                        | phoenix_types::QueryTarget::Graph
                        | phoenix_types::QueryTarget::Semantic
                )
            });
        let wants_nodes = request.targets.iter().any(|target| {
            matches!(
                target,
                phoenix_types::QueryTarget::Nodes | phoenix_types::QueryTarget::Graph
            )
        });
        let seed_limit = self.config.seed_limit.max(limit);

        let mut seed_scores = FxHashMap::<String, f64>::default();

        for hit in lexical.span_hits.iter().take(seed_limit) {
            let leaf_id = leaf_vertex_id(&hit.span_id);
            let Some(leaf_vertex) = graph.vertices.get(&leaf_id) else {
                continue;
            };
            if !temporal.matches_vertex(leaf_vertex) {
                continue;
            }
            accumulate_score(&mut seed_scores, &leaf_id, hit.score.max(0.0));
        }

        for (rank, hit) in semantic_hits.iter().take(seed_limit).enumerate() {
            let leaf_id = leaf_vertex_id(&hit.span_id);
            let Some(leaf_vertex) = graph.vertices.get(&leaf_id) else {
                continue;
            };
            if !temporal.matches_vertex(leaf_vertex) {
                continue;
            }
            accumulate_score(
                &mut seed_scores,
                &leaf_id,
                semantic_seed_score(rank, hit.distance),
            );
        }

        let stationary_scores = self.personalized_pagerank(&graph, &temporal, &seed_scores);

        let mut diagnostics = lexical.diagnostics;
        diagnostics.push(Diagnostic {
            code: "PX_GLDR_OK".to_owned(),
            message: "GLDR diffused lexical and semantic seed mass across the local graph with type-annealed personalized PageRank.".to_owned(),
        });
        if semantic_requested {
            diagnostics.push(Diagnostic {
                code: if request.semantic_query_vector.is_some() {
                    "PX_GLDR_SEMANTIC".to_owned()
                } else {
                    "PX_GLDR_SEMANTIC_MISSING_VECTOR".to_owned()
                },
                message: if request.semantic_query_vector.is_some() {
                    format!(
                        "Semantic retrieval shortlisted {} documents and fused {} filtered leaf neighbors with lexical and graph expansion (fallback_to_global_leaf_ann={}).",
                        semantic_resolution.shortlisted_documents,
                        semantic_resolution.filtered_leaf_hits,
                        semantic_resolution.used_global_fallback
                    )
                } else {
                    "Semantic target requested without a query vector; GLDR used lexical and graph retrieval only."
                        .to_owned()
                },
            });
        }
        if request.include_candidate_graph {
            diagnostics.push(Diagnostic {
                code: "PX_GLDR_CANDIDATE_GRAPH".to_owned(),
                message: "GLDR included candidate embedding edges during local graph expansion."
                    .to_owned(),
            });
        }
        if request.temporal.is_some() {
            diagnostics.push(Diagnostic {
                code: "PX_GLDR_TEMPORAL".to_owned(),
                message: temporal.diagnostic_message(),
            });
        }

        let chunk_hits = if wants_chunks {
            ranked_chunk_hits(project_chunk_scores(&graph, &stationary_scores), limit)
        } else {
            Vec::new()
        };
        let node_hits = if wants_nodes {
            ranked_node_hits(project_node_scores(&graph, &stationary_scores), limit)
        } else {
            Vec::new()
        };

        Ok(QueryResult {
            session_id: request.session_id.clone(),
            chunk_hits,
            node_hits,
            diagnostics,
        })
    }

    fn resolve_semantic_hits(
        &self,
        store: &PhoenixCozoStore,
        request: &QueryRequest,
        limit: usize,
    ) -> Result<SemanticResolution, StoreError> {
        let Some(vector) = request.semantic_query_vector.as_ref() else {
            return Ok(SemanticResolution::default());
        };

        let doc_limit = (limit.saturating_mul(4)).max(24);
        let doc_oversample = doc_limit.saturating_mul(4);
        let leaf_limit = (limit.saturating_mul(8)).max(32);
        let leaf_oversample = leaf_limit.saturating_mul(4);

        let documents = store.query_semantic_documents(
            &vector.values,
            &request.scope,
            doc_limit,
            doc_oversample,
        )?;
        if documents.is_empty() {
            let hits = store.query_semantic_neighbors(
                &vector.values,
                &request.scope,
                leaf_limit,
                leaf_oversample,
            )?;
            return Ok(SemanticResolution {
                filtered_leaf_hits: hits.len(),
                hits,
                used_global_fallback: true,
                ..SemanticResolution::default()
            });
        }

        let document_ids = documents
            .iter()
            .map(|document| document.document_id.clone())
            .collect::<Vec<_>>();
        let hits = store.query_semantic_neighbors_in_documents(
            &vector.values,
            &request.scope,
            &document_ids,
            leaf_limit,
            leaf_oversample,
        )?;

        Ok(SemanticResolution {
            shortlisted_documents: document_ids.len(),
            filtered_leaf_hits: hits.len(),
            hits,
            used_global_fallback: false,
        })
    }

    fn personalized_pagerank(
        &self,
        graph: &GraptorGraph,
        temporal: &TemporalFilter,
        seed_scores: &FxHashMap<String, f64>,
    ) -> FxHashMap<String, f64> {
        if graph.vertices.is_empty() || seed_scores.is_empty() {
            return FxHashMap::default();
        }

        let mut vertex_ids = graph
            .vertices
            .iter()
            .filter_map(|(vertex_id, vertex)| {
                temporal.matches_vertex(vertex).then_some(vertex_id.clone())
            })
            .collect::<Vec<_>>();
        vertex_ids.sort();
        if vertex_ids.is_empty() {
            return FxHashMap::default();
        }

        let index_by_id = vertex_ids
            .iter()
            .enumerate()
            .map(|(index, vertex_id)| (vertex_id.clone(), index))
            .collect::<FxHashMap<_, _>>();

        let mut personalization = vec![0.0; vertex_ids.len()];
        let alpha_by_index = vertex_ids
            .iter()
            .map(|vertex_id| {
                graph
                    .vertices
                    .get(vertex_id)
                    .map(|vertex| self.alpha_for_vertex(vertex))
                    .unwrap_or_else(|| self.config.ppr_alpha.clamp(0.0, 1.0))
            })
            .collect::<Vec<_>>();
        let total_seed = seed_scores
            .iter()
            .filter_map(|(vertex_id, score)| {
                (index_by_id.contains_key(vertex_id) && *score > 0.0).then_some(*score)
            })
            .sum::<f64>();
        if total_seed <= f64::EPSILON {
            return FxHashMap::default();
        }
        for (vertex_id, score) in seed_scores {
            let Some(&index) = index_by_id.get(vertex_id) else {
                continue;
            };
            if *score > 0.0 {
                personalization[index] += *score / total_seed;
            }
        }

        let mut neighbor_weights = vec![FxHashMap::<usize, f64>::default(); vertex_ids.len()];
        for edge in graph.outgoing.values().flat_map(|edges| edges.iter()) {
            let (Some(&source_index), Some(&target_index)) = (
                index_by_id.get(&edge.source_id),
                index_by_id.get(&edge.target_id),
            ) else {
                continue;
            };
            let weight = transition_weight(edge.weight);
            add_transition(&mut neighbor_weights[source_index], target_index, weight);
            add_transition(&mut neighbor_weights[target_index], source_index, weight);
        }

        let outgoing = neighbor_weights
            .into_iter()
            .map(|neighbors| neighbors.into_iter().collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let outgoing_weight = outgoing
            .iter()
            .map(|neighbors| neighbors.iter().map(|(_, weight)| *weight).sum::<f64>())
            .collect::<Vec<_>>();

        let tolerance = if self.config.ppr_tolerance > 0.0 {
            self.config.ppr_tolerance
        } else {
            1e-6
        };
        let max_iterations = self.config.ppr_iterations.max(1);

        let mut scores = personalization.clone();
        let mut next_scores = vec![0.0; scores.len()];

        for _ in 0..max_iterations {
            next_scores.fill(0.0);

            let mut dangling_mass = 0.0;
            for (source_index, score) in scores.iter().copied().enumerate() {
                if score <= f64::EPSILON {
                    continue;
                }

                let degree = outgoing_weight[source_index];
                if degree <= f64::EPSILON {
                    dangling_mass += score;
                    continue;
                }

                let walk_mass = score / degree;
                for &(target_index, edge_weight) in &outgoing[source_index] {
                    next_scores[target_index] += walk_mass * edge_weight;
                }
            }

            let mut delta = 0.0;
            let mut total_mass = 0.0;
            for index in 0..next_scores.len() {
                let alpha = alpha_by_index[index];
                let base_mass = next_scores[index] + dangling_mass * personalization[index];
                next_scores[index] = alpha * personalization[index] + (1.0 - alpha) * base_mass;
                total_mass += next_scores[index];
                delta += (next_scores[index] - scores[index]).abs();
            }

            if total_mass > f64::EPSILON {
                for score in &mut next_scores {
                    *score /= total_mass;
                }
            }

            std::mem::swap(&mut scores, &mut next_scores);
            if delta < tolerance {
                break;
            }
        }

        vertex_ids
            .into_iter()
            .zip(scores)
            .filter(|(_, score)| *score > f64::EPSILON)
            .collect()
    }

    fn alpha_for_vertex(&self, vertex: &GraptorVertex) -> f64 {
        match vertex.kind.as_str() {
            "leaf" => self.config.ppr_alpha_leaf,
            "entity" => self.config.ppr_alpha_entity,
            "chapter" => self.config.ppr_alpha_chapter,
            _ => self.config.ppr_alpha,
        }
        .clamp(0.0, 1.0)
    }
}

impl Default for PhoenixGldr {
    fn default() -> Self {
        Self::new(GldrConfig::default())
    }
}

#[derive(Clone, Debug, Default)]
struct TemporalFilter {
    chapter: Option<u32>,
    boundary_doc_id: Option<String>,
    boundary_ordinal: Option<u32>,
    boundary_end_ordinal: Option<u32>,
}

impl TemporalFilter {
    fn from_marker(marker: Option<&TemporalMarker>) -> Self {
        let chapter = marker.and_then(|marker| marker.chapter);
        let boundary_doc_id = marker.and_then(|marker| {
            marker
                .boundary_doc_id
                .as_ref()
                .map(|document_id| document_id.0.clone())
        });
        let boundary_ordinal = marker.and_then(|marker| {
            marker
                .boundary_ordinal
                .or(marker.ordinal)
                .map(|value| value as u32)
        });
        let boundary_end_ordinal =
            marker.and_then(|marker| marker.boundary_end_ordinal.map(|value| value as u32));
        Self {
            chapter,
            boundary_doc_id,
            boundary_ordinal,
            boundary_end_ordinal,
        }
    }

    fn matches_vertex(&self, vertex: &GraptorVertex) -> bool {
        if let Some(boundary_ordinal) = self.boundary_ordinal {
            if let Some(document_id) = self.boundary_doc_id.as_ref() {
                if vertex.document_id.as_deref() != Some(document_id.as_str()) {
                    return false;
                }
            }
            let boundary_end = self.boundary_end_ordinal.unwrap_or(boundary_ordinal);
            let matches_boundary = vertex
                .boundary_ordinal
                .map(|value| value >= boundary_ordinal && value <= boundary_end)
                .unwrap_or(false)
                || vertex
                    .boundary_ordinals
                    .iter()
                    .any(|value| *value >= boundary_ordinal && *value <= boundary_end);
            if !matches_boundary {
                return false;
            }
        }
        match self.chapter {
            None => true,
            Some(chapter) => {
                vertex.chapter_id == Some(chapter)
                    || vertex.chapters.iter().any(|value| *value == chapter)
            }
        }
    }

    fn diagnostic_message(&self) -> String {
        if let Some(boundary_ordinal) = self.boundary_ordinal {
            let boundary_end = self.boundary_end_ordinal.unwrap_or(boundary_ordinal);
            let doc = self.boundary_doc_id.as_deref().unwrap_or("<any-document>");
            if boundary_end == boundary_ordinal {
                return format!(
                    "Applied boundary-local temporal filtering at {doc} ordinal {boundary_ordinal}."
                );
            }
            return format!(
                "Applied boundary-range temporal filtering at {doc} ordinals {boundary_ordinal}..={boundary_end}."
            );
        }
        match self.chapter {
            Some(chapter) => {
                format!("Applied chapter-local temporal filtering at chapter {chapter}.")
            }
            None => "No chapter-local temporal filter was applied.".to_owned(),
        }
    }
}

fn accumulate_score(scores: &mut FxHashMap<String, f64>, key: &str, score: f64) {
    scores
        .entry(key.to_owned())
        .and_modify(|value| *value += score)
        .or_insert(score);
}

fn add_transition(neighbors: &mut FxHashMap<usize, f64>, target_index: usize, weight: f64) {
    neighbors
        .entry(target_index)
        .and_modify(|value| *value += weight)
        .or_insert(weight);
}

fn transition_weight(weight: i64) -> f64 {
    1.0 + (weight.max(1) as f64).ln() * 0.2
}

fn reciprocal_rank_score(rank: usize) -> f64 {
    1.0 / (60.0 + rank as f64 + 1.0)
}

fn semantic_seed_score(rank: usize, distance: f64) -> f64 {
    reciprocal_rank_score(rank) * (1.0 / (1.0 + distance.max(0.0)))
}

fn project_chunk_scores(
    graph: &GraptorGraph,
    vertex_scores: &FxHashMap<String, f64>,
) -> FxHashMap<String, f64> {
    let mut chunk_scores = FxHashMap::default();
    for (vertex_id, score) in vertex_scores {
        let Some(vertex) = graph.vertices.get(vertex_id) else {
            continue;
        };
        if let Some(search_chunk_id) = vertex.search_chunk_id.as_ref() {
            accumulate_score(&mut chunk_scores, search_chunk_id, *score);
        }
    }
    chunk_scores
}

fn project_node_scores(
    graph: &GraptorGraph,
    vertex_scores: &FxHashMap<String, f64>,
) -> FxHashMap<String, f64> {
    let mut node_scores = FxHashMap::default();
    for (vertex_id, score) in vertex_scores {
        let Some(vertex) = graph.vertices.get(vertex_id) else {
            continue;
        };
        if let Some(entity_id) = vertex.entity_id.as_ref() {
            accumulate_score(&mut node_scores, entity_id, *score);
        }
    }
    node_scores
}

fn ranked_chunk_hits(scores: FxHashMap<String, f64>, limit: usize) -> Vec<ChunkHit> {
    let mut hits = scores
        .into_iter()
        .map(|(chunk_id, score)| ChunkHit { chunk_id, score })
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.chunk_id.cmp(&right.chunk_id))
    });
    if hits.len() > limit {
        hits.truncate(limit);
    }
    hits
}

fn ranked_node_hits(scores: FxHashMap<String, f64>, limit: usize) -> Vec<NodeHit> {
    let mut hits = scores
        .into_iter()
        .map(|(entity_id, score)| NodeHit {
            entity_id: Some(EntityId(entity_id)),
            score,
        })
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.entity_id.cmp(&right.entity_id))
    });
    if hits.len() > limit {
        hits.truncate(limit);
    }
    hits
}

fn leaf_vertex_id(search_chunk_id: &str) -> String {
    format!("leaf::{search_chunk_id}")
}

fn boundary_kind_from_str(kind: &str) -> BoundaryKind {
    match kind {
        "chapter" => BoundaryKind::Chapter,
        "heading" => BoundaryKind::Heading,
        "section" => BoundaryKind::Section,
        "act" => BoundaryKind::Act,
        _ => BoundaryKind::Other,
    }
}

/// Load only the local sub-graph around the given seeds.
/// The traversal is undirected so PPR can diffuse back from entity/chapter hubs
/// into supporting leaves while staying on a bounded query-time snapshot.
fn load_subgraph(
    store: &PhoenixCozoStore,
    seed_vertex_ids: &[String],
    max_hops: usize,
    include_candidate_graph: bool,
) -> Result<GraptorGraph, StoreError> {
    if seed_vertex_ids.is_empty() {
        return Ok(GraptorGraph::default());
    }

    if include_candidate_graph {
        let snapshot = load_graph_snapshot_with_candidate_graph(store, true)?;
        return Ok(build_bounded_subgraph(&snapshot, seed_vertex_ids, max_hops));
    }

    let max_hops = max_hops.max(1);
    let mut script = String::from(
        r#"
        seeds[id] <- $seeds

        neighbor[src, dst] := *graph_edges{ source_id: src, target_id: dst }
        neighbor[src, dst] := *graph_edges{ source_id: dst, target_id: src }

        touched[id] := seeds[id]
"#,
    );
    if include_candidate_graph {
        script.push_str(
            r#"
        neighbor[src, dst] := *graph_candidate_edges{ source_id: src, target_id: dst, edge_type, document_id, narrative_id, valid_from_doc, valid_from_boundary, valid_to_doc, valid_to_boundary, assertion_kind, weight, attributes, data }
        neighbor[src, dst] := *graph_candidate_edges{ source_id: dst, target_id: src, edge_type, document_id, narrative_id, valid_from_doc, valid_from_boundary, valid_to_doc, valid_to_boundary, assertion_kind, weight, attributes, data }
"#,
        );
    }
    for hop in 1..=max_hops {
        let frontier = if hop == 1 {
            "seeds".to_owned()
        } else {
            format!("hop{}", hop - 1)
        };
        script.push_str(&format!(
            r#"
        hop{hop}[id] := {frontier}[src], neighbor[src, id]
        touched[id] := hop{hop}[id]
"#
        ));
    }
    script.push_str(
        r#"

        ?[kind, c0, c1, c2, c3, c4, c5] := kind = "v", touched[c0],
            *graph_vertices{ id: c0, weight: w, value: c2, attributes: c3 },
            c1 = to_string(w), c4 = null, c5 = null

        ?[kind, c0, c1, c2, c3, c4, c5] := kind = "e",
            touched[c0], touched[c1],
            *graph_edges{ source_id: c0, target_id: c1, edge_type: c2, weight: w, attributes: c4, data: c5 },
            c3 = to_string(w)
    "#,
    );
    if include_candidate_graph {
        script.push_str(
            r#"

        ?[kind, c0, c1, c2, c3, c4, c5] := kind = "e",
            touched[c0], touched[c1],
            *graph_candidate_edges{ source_id: c0, target_id: c1, edge_type: c2, weight: w, attributes: c4, data: c5 },
            c3 = to_string(w)
    "#,
        );
    }

    let mut graph = GraptorGraph::default();

    // Single query returns both vertices and edges
    let rows = store.run_datalog_json(&script, seed_vertex_ids)?;
    for row in &rows {
        let Some(kind) = row.first().and_then(Value::as_str) else {
            continue;
        };
        match kind {
            "v" => {
                // Columns: ['v', id, weight_str, value, attributes, null, null]
                let Some(id) = row.get(1).and_then(Value::as_str) else {
                    continue;
                };
                let weight = row
                    .get(2)
                    .and_then(Value::as_str)
                    .and_then(|s| s.parse::<i64>().ok())
                    .or_else(|| row.get(2).and_then(Value::as_i64))
                    .unwrap_or(1);
                let value = row.get(3).cloned().unwrap_or(Value::Null);
                let attributes = row.get(4).cloned().unwrap_or(Value::Null);

                let vertex_kind = value
                    .get("kind")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_owned();
                let entity_id = value
                    .get("entityId")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let search_chunk_id = value
                    .get("searchChunkId")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        attributes
                            .get("searchChunkId")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    });
                let document_id = attributes
                    .get("documentId")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let chapter_id = attributes
                    .get("chapterId")
                    .and_then(Value::as_u64)
                    .map(|v| v as u32);
                let chapters = attributes
                    .get("chapters")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_u64)
                            .map(|v| v as u32)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let boundary_id = attributes
                    .get("boundaryId")
                    .and_then(Value::as_u64)
                    .map(|v| v as u32);
                let boundary_ordinal = attributes
                    .get("boundaryOrdinal")
                    .and_then(Value::as_u64)
                    .map(|v| v as u32);
                let boundary_kind = attributes
                    .get("boundaryKind")
                    .and_then(Value::as_str)
                    .map(boundary_kind_from_str);
                let boundary_ordinals = attributes
                    .get("boundaryOrdinals")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(Value::as_u64)
                            .map(|v| v as u32)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();

                let vertex = GraptorVertex {
                    id: id.to_owned(),
                    kind: vertex_kind,
                    weight,
                    value,
                    attributes: attributes.clone(),
                    entity_id,
                    search_chunk_id: search_chunk_id.clone(),
                    document_id: document_id.clone(),
                    chapter_id,
                    chapters,
                    boundary_id,
                    boundary_ordinal,
                    boundary_kind,
                    boundary_ordinals,
                };
                graph.vertices.insert(id.to_owned(), vertex);
                if let (Some(doc_id), Some(ch_id), Some(_)) =
                    (document_id, chapter_id, search_chunk_id)
                {
                    graph
                        .chapter_leaves
                        .entry((doc_id, ch_id))
                        .or_default()
                        .push(id.to_owned());
                }
            }
            "e" => {
                // Columns: ['e', source_id, target_id, edge_type, weight_str, attributes, data]
                let Some(source_id) = row.get(1).and_then(Value::as_str) else {
                    continue;
                };
                let Some(target_id) = row.get(2).and_then(Value::as_str) else {
                    continue;
                };
                let edge_weight = row
                    .get(4)
                    .and_then(Value::as_str)
                    .and_then(|s| s.parse::<i64>().ok())
                    .or_else(|| row.get(4).and_then(Value::as_i64))
                    .unwrap_or(1);
                let edge = GraptorEdge {
                    source_id: source_id.to_owned(),
                    target_id: target_id.to_owned(),
                    edge_type: row
                        .get(3)
                        .and_then(Value::as_str)
                        .unwrap_or("edge")
                        .to_owned(),
                    weight: edge_weight,
                    attributes: row.get(5).cloned().unwrap_or(Value::Null),
                    data: row.get(6).cloned().filter(|v| !v.is_null()),
                    layer: phoenix_graph::GraphLayer::Asserted,
                };
                graph
                    .outgoing
                    .entry(source_id.to_owned())
                    .or_default()
                    .push(edge.clone());
                graph
                    .incoming
                    .entry(target_id.to_owned())
                    .or_default()
                    .push(edge);
            }
            _ => continue,
        }
    }

    Ok(graph)
}

fn load_subgraph_from_graph(
    graph: &GraptorGraph,
    seed_vertex_ids: &[String],
    max_hops: usize,
) -> GraptorGraph {
    if seed_vertex_ids.is_empty() {
        return GraptorGraph::default();
    }
    build_bounded_subgraph(graph, seed_vertex_ids, max_hops)
}

fn build_bounded_subgraph(
    full_graph: &GraptorGraph,
    seed_vertex_ids: &[String],
    max_hops: usize,
) -> GraptorGraph {
    let max_hops = max_hops.max(1);
    let mut touched = HashSet::<String>::new();
    let mut queue = VecDeque::<(String, usize)>::new();

    for seed in seed_vertex_ids {
        if touched.insert(seed.clone()) {
            queue.push_back((seed.clone(), 0));
        }
    }

    while let Some((vertex_id, depth)) = queue.pop_front() {
        if depth >= max_hops {
            continue;
        }
        for edge in full_graph
            .outgoing_any(&vertex_id)
            .chain(full_graph.incoming_any(&vertex_id))
        {
            let neighbor_id = if edge.source_id == vertex_id {
                edge.target_id.as_str()
            } else {
                edge.source_id.as_str()
            };
            if touched.insert(neighbor_id.to_owned()) {
                queue.push_back((neighbor_id.to_owned(), depth + 1));
            }
        }
    }

    let mut graph = GraptorGraph::default();
    for vertex_id in &touched {
        if let Some(vertex) = full_graph.vertices.get(vertex_id) {
            graph.vertices.insert(vertex_id.clone(), vertex.clone());
            if let (Some(document_id), Some(chapter_id), Some(_)) = (
                vertex.document_id.clone(),
                vertex.chapter_id,
                vertex.search_chunk_id.clone(),
            ) {
                graph
                    .chapter_leaves
                    .entry((document_id, chapter_id))
                    .or_default()
                    .push(vertex_id.clone());
            }
        }
    }

    let mut seen_edges = HashSet::<(String, String, String)>::new();
    for source_id in &touched {
        for edge in full_graph.outgoing_any(source_id) {
            if !touched.contains(&edge.target_id) {
                continue;
            }
            let key = (
                edge.source_id.clone(),
                edge.target_id.clone(),
                edge.edge_type.clone(),
            );
            if !seen_edges.insert(key) {
                continue;
            }
            graph
                .outgoing
                .entry(edge.source_id.clone())
                .or_default()
                .push(edge.clone());
            graph
                .incoming
                .entry(edge.target_id.clone())
                .or_default()
                .push(edge.clone());
        }
    }

    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_lex::LexConfig;
    use phoenix_store_cozo::PhoenixCozoStore;
    use phoenix_types::{ScopeKey, SemanticQueryVector, TemporalSource};
    use serde_json::json;

    fn test_scope() -> ScopeKey {
        ScopeKey {
            world_id: Some("world-1".to_owned()),
            narrative_id: None,
            folder_id: None,
            folder_path: None,
        }
    }

    fn seed_note(store: &PhoenixCozoStore, id: &str, title: &str, content: &str) {
        store
            .put_row(
                "notes",
                json!({
                    "id": id,
                    "version": 1,
                    "world_id": "world-1",
                    "title": title,
                    "content": content,
                    "markdown_content": content,
                    "folder_id": null,
                    "entity_kind": null,
                    "entity_subtype": null,
                    "is_entity": false,
                    "is_pinned": false,
                    "favorite": false,
                    "owner_id": id,
                    "narrative_id": null,
                    "order": 0.0,
                    "created_at": 1,
                    "updated_at": 1,
                    "valid_from": 1,
                    "valid_to": null,
                    "is_current": true,
                    "change_reason": "seed"
                }),
            )
            .expect("seed note");
    }

    fn semantic_vector(primary_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; phoenix_store_cozo::SEMANTIC_VECTOR_DIM];
        if primary_index < values.len() {
            values[primary_index] = 1.0;
        }
        values
    }

    #[test]
    fn gldr_expands_from_seed_chunk_to_related_entity_and_chunk() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(&store, "doc-1", "Chapters", "Ryan woke up. He ran.");

        store
            .put_row(
                "chunks",
                json!({
                    "chunk_id": 1001,
                    "doc_id": "doc-1",
                    "level": 0,
                    "start": 0,
                    "end": 13,
                    "text": "Ryan woke up.",
                    "parent_id": null,
                    "scope_narrative": null,
                    "scope_folder": null,
                    "created_at": 1
                }),
            )
            .expect("chunk 1");
        store
            .put_row(
                "chunks",
                json!({
                    "chunk_id": 1002,
                    "doc_id": "doc-1",
                    "level": 0,
                    "start": 14,
                    "end": 26,
                    "text": "He ran hard.",
                    "parent_id": null,
                    "scope_narrative": null,
                    "scope_folder": null,
                    "created_at": 1
                }),
            )
            .expect("chunk 2");
        store
            .put_row(
                "chunkid_map",
                json!({
                    "id": 1001,
                    "chunk_key": "doc-1:1:0:0-13",
                    "doc_id": "doc-1",
                    "created_at": 1
                }),
            )
            .expect("chunkid 1");
        store
            .put_row(
                "chunkid_map",
                json!({
                    "id": 1002,
                    "chunk_key": "doc-1:2:0:14-26",
                    "doc_id": "doc-1",
                    "created_at": 1
                }),
            )
            .expect("chunkid 2");
        store
            .put_row(
                "graph_vertices",
                json!({
                    "id": "leaf::doc-1:1:0:0-13",
                    "value": { "kind": "leaf", "searchChunkId": "doc-1:1:0:0-13" },
                    "weight": 1,
                    "attributes": { "documentId": "doc-1", "chapterId": 1 }
                }),
            )
            .expect("leaf 1 vertex");
        store
            .put_row(
                "graph_vertices",
                json!({
                    "id": "leaf::doc-1:2:0:14-26",
                    "value": { "kind": "leaf", "searchChunkId": "doc-1:2:0:14-26" },
                    "weight": 1,
                    "attributes": { "documentId": "doc-1", "chapterId": 2 }
                }),
            )
            .expect("leaf 2 vertex");
        store.put_row(
            "graph_vertices",
            json!({
                "id": "entity::ryan",
                "value": { "kind": "entity", "entityId": "ryan", "label": "Ryan", "entityKind": "Character" },
                "weight": 2,
                "attributes": { "chapters": [1, 2] }
            }),
        )
        .expect("entity vertex");
        store
            .put_row(
                "graph_edges",
                json!({
                    "source_id": "leaf::doc-1:1:0:0-13",
                    "target_id": "entity::ryan",
                    "weight": 100,
                    "attributes": { "confidence": 1.0 },
                    "data": null,
                    "edge_type": "mentions"
                }),
            )
            .expect("mentions 1");
        store
            .put_row(
                "graph_edges",
                json!({
                    "source_id": "leaf::doc-1:2:0:14-26",
                    "target_id": "entity::ryan",
                    "weight": 85,
                    "attributes": { "confidence": 0.85 },
                    "data": null,
                    "edge_type": "mentions"
                }),
            )
            .expect("mentions 2");

        let lex = LexIndex::from_store(&store, LexConfig::default()).expect("lex");
        let gldr = PhoenixGldr::default();
        let result = gldr
            .query(
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "Ryan".to_owned(),
                    scope: test_scope(),
                    targets: vec![phoenix_types::QueryTarget::Graph],
                    limit: Some(5),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                },
            )
            .expect("gldr query");

        assert_eq!(result.node_hits.len(), 1);
        assert_eq!(
            result.node_hits[0].entity_id,
            Some(EntityId("ryan".to_owned()))
        );
        assert!(
            result
                .chunk_hits
                .iter()
                .any(|hit| hit.chunk_id == "doc-1:2:0:14-26"),
            "graph expansion should recover the second chunk through the shared entity"
        );
    }

    #[test]
    fn gldr_recovers_four_hop_cooccurrence_chunks_via_ppr() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(
            &store,
            "doc-1",
            "Cooccurrence",
            "Ryan checked the harbor. Len carried the map.",
        );

        for (chunk_id, chapter_id, key, text) in [
            (
                3001_i64,
                1_u32,
                "doc-1:1:0:0-24",
                "Ryan checked the harbor.",
            ),
            (3002_i64, 2_u32, "doc-1:2:0:25-45", "Len carried the map."),
        ] {
            store
                .put_row(
                    "chunks",
                    json!({
                        "chunk_id": chunk_id,
                        "doc_id": "doc-1",
                        "level": 0,
                        "start": if chapter_id == 1 { 0 } else { 25 },
                        "end": if chapter_id == 1 { 24 } else { 45 },
                        "text": text,
                        "parent_id": null,
                        "scope_narrative": null,
                        "scope_folder": null,
                        "created_at": 1
                    }),
                )
                .expect("chunk");
            store
                .put_row(
                    "chunkid_map",
                    json!({
                        "id": chunk_id,
                        "chunk_key": key,
                        "doc_id": "doc-1",
                        "created_at": 1
                    }),
                )
                .expect("chunkid");
            store
                .put_row(
                    "graph_vertices",
                    json!({
                        "id": format!("leaf::{key}"),
                        "value": { "kind": "leaf", "searchChunkId": key },
                        "weight": 1,
                        "attributes": { "documentId": "doc-1", "chapterId": chapter_id }
                    }),
                )
                .expect("leaf");
        }
        store
            .put_row(
                "graph_vertices",
                json!({
                    "id": "entity::ryan",
                    "value": { "kind": "entity", "entityId": "ryan", "label": "Ryan", "entityKind": "Character" },
                    "weight": 1,
                    "attributes": { "chapters": [1] }
                }),
            )
            .expect("ryan");
        store
            .put_row(
                "graph_vertices",
                json!({
                    "id": "entity::len",
                    "value": { "kind": "entity", "entityId": "len", "label": "Len", "entityKind": "Character" },
                    "weight": 1,
                    "attributes": { "chapters": [2] }
                }),
            )
            .expect("len");
        for (source_id, target_id, weight, edge_type) in [
            ("leaf::doc-1:1:0:0-24", "entity::ryan", 100_i64, "mentions"),
            ("leaf::doc-1:2:0:25-45", "entity::len", 100_i64, "mentions"),
            ("entity::ryan", "entity::len", 3_i64, "cooccurs"),
            ("entity::len", "entity::ryan", 3_i64, "cooccurs"),
        ] {
            store
                .put_row(
                    "graph_edges",
                    json!({
                        "source_id": source_id,
                        "target_id": target_id,
                        "weight": weight,
                        "attributes": { "confidence": 1.0 },
                        "data": null,
                        "edge_type": edge_type
                    }),
                )
                .expect("graph edge");
        }

        let lex = LexIndex::from_store(&store, LexConfig::default()).expect("lex");
        let gldr = PhoenixGldr::default();
        let result = gldr
            .query(
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "Ryan".to_owned(),
                    scope: test_scope(),
                    targets: vec![phoenix_types::QueryTarget::Graph],
                    limit: Some(5),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                },
            )
            .expect("gldr query");

        assert!(
            result
                .chunk_hits
                .iter()
                .any(|hit| hit.chunk_id == "doc-1:2:0:25-45"),
            "PPR over the loaded subgraph should surface the cooccurring entity's supporting chunk",
        );
        assert!(
            result
                .node_hits
                .iter()
                .any(|hit| hit.entity_id == Some(EntityId("len".to_owned()))),
            "PPR should surface the cooccurring entity node as well",
        );
    }

    #[test]
    fn gldr_candidate_graph_overlay_is_opt_in() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(&store, "doc-1", "Candidate One", "Ryan watched the harbor.");
        seed_note(
            &store,
            "doc-2",
            "Candidate Two",
            "Rian catalogued the harbor ledgers.",
        );

        for (chunk_id, key, document_id, text, entity_id, label) in [
            (
                3501_i64,
                "doc-1:1:0:0-24",
                "doc-1",
                "Ryan watched the harbor.",
                "entity::ryan",
                "Ryan",
            ),
            (
                3502_i64,
                "doc-2:1:0:0-36",
                "doc-2",
                "Rian catalogued the harbor ledgers.",
                "entity::rian",
                "Rian",
            ),
        ] {
            store
                .put_row(
                    "chunks",
                    json!({
                        "chunk_id": chunk_id,
                        "doc_id": document_id,
                        "level": 0,
                        "start": 0,
                        "end": text.len() as i64,
                        "text": text,
                        "parent_id": null,
                        "scope_narrative": null,
                        "scope_folder": null,
                        "created_at": 1
                    }),
                )
                .expect("chunk");
            store
                .put_row(
                    "chunkid_map",
                    json!({
                        "id": chunk_id,
                        "chunk_key": key,
                        "doc_id": document_id,
                        "created_at": 1
                    }),
                )
                .expect("chunk id");
            store
                .put_row(
                    "graph_vertices",
                    json!({
                        "id": format!("leaf::{key}"),
                        "value": { "kind": "leaf", "searchChunkId": key },
                        "weight": 1,
                        "attributes": { "documentId": document_id, "chapterId": 1 }
                    }),
                )
                .expect("leaf vertex");
            store
                .put_row(
                    "graph_vertices",
                    json!({
                        "id": entity_id,
                        "value": { "kind": "entity", "entityId": entity_id.trim_start_matches("entity::"), "label": label, "entityKind": "Character" },
                        "weight": 1,
                        "attributes": { "documentId": document_id, "chapters": [1] }
                    }),
                )
                .expect("entity vertex");
            store
                .put_row(
                    "graph_edges",
                    json!({
                        "source_id": format!("leaf::{key}"),
                        "target_id": entity_id,
                        "weight": 100,
                        "attributes": { "confidence": 1.0 },
                        "data": null,
                        "edge_type": "mentions"
                    }),
                )
                .expect("mentions");
        }

        store
            .put_row(
                "graph_candidate_edges",
                json!({
                    "source_id": "entity::ryan",
                    "target_id": "entity::rian",
                    "edge_type": "candidate_corefers_with",
                    "document_id": "doc-1",
                    "narrative_id": null,
                    "valid_from_doc": "doc-1",
                    "valid_from_boundary": null,
                    "valid_to_doc": null,
                    "valid_to_boundary": null,
                    "assertion_kind": "candidate",
                    "weight": 900,
                    "attributes": {
                        "graph": {
                            "layer": "candidate",
                            "status": "candidate",
                            "resolver": "test",
                            "confidence": 0.9,
                            "evidence_refs": []
                        }
                    },
                    "data": { "score": 0.9, "threshold": 0.78 }
                }),
            )
            .expect("candidate edge");

        let lex = LexIndex::from_store(&store, LexConfig::default()).expect("lex");
        let without_candidate = PhoenixGldr::default()
            .query(
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "Ryan".to_owned(),
                    scope: test_scope(),
                    targets: vec![phoenix_types::QueryTarget::Graph],
                    limit: Some(5),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                },
            )
            .expect("asserted-only query");
        let with_candidate = PhoenixGldr::default()
            .query(
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "Ryan".to_owned(),
                    scope: test_scope(),
                    targets: vec![phoenix_types::QueryTarget::Graph],
                    limit: Some(5),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: true,
                },
            )
            .expect("candidate query");

        assert!(
            !without_candidate
                .chunk_hits
                .iter()
                .any(|hit| hit.chunk_id == "doc-2:1:0:0-36"),
            "asserted-only traversal should not cross the candidate overlay",
        );
        assert!(
            with_candidate
                .chunk_hits
                .iter()
                .any(|hit| hit.chunk_id == "doc-2:1:0:0-36"),
            "opting into candidate graph traversal should surface the second chunk",
        );
        assert!(with_candidate
            .diagnostics
            .iter()
            .any(|diag| diag.code == "PX_GLDR_CANDIDATE_GRAPH"));

        store
            .put_row(
                "graph_candidate_edges",
                json!({
                    "source_id": "entity::ryan",
                    "target_id": "entity::rian",
                    "edge_type": "candidate_corefers_with",
                    "document_id": "doc-1",
                    "narrative_id": null,
                    "valid_from_doc": "doc-1",
                    "valid_from_boundary": null,
                    "valid_to_doc": null,
                    "valid_to_boundary": null,
                    "assertion_kind": "candidate",
                    "weight": 0,
                    "attributes": {
                        "graph": {
                            "layer": "candidate",
                            "status": "candidate_rejected",
                            "resolver": "test",
                            "confidence": 0.2,
                            "evidence_refs": []
                        }
                    },
                    "data": {
                        "base": { "score": 0.9, "threshold": 0.78 },
                        "nli": { "accepted": false }
                    }
                }),
            )
            .expect("rejected candidate edge");
        let with_rejected_candidate = PhoenixGldr::default()
            .query(
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "Ryan".to_owned(),
                    scope: test_scope(),
                    targets: vec![phoenix_types::QueryTarget::Graph],
                    limit: Some(5),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: true,
                },
            )
            .expect("rejected candidate query");
        assert!(
            !with_rejected_candidate
                .chunk_hits
                .iter()
                .any(|hit| hit.chunk_id == "doc-2:1:0:0-36"),
            "candidate_rejected rows must stay stored for audit but never participate in traversal",
        );
    }

    #[test]
    fn gldr_type_annealed_alpha_changes_stationary_distribution() {
        let mut graph = GraptorGraph::default();
        graph.vertices.insert(
            "leaf::seed".to_owned(),
            GraptorVertex {
                id: "leaf::seed".to_owned(),
                kind: "leaf".to_owned(),
                search_chunk_id: Some("seed".to_owned()),
                ..Default::default()
            },
        );
        graph.vertices.insert(
            "entity::hub".to_owned(),
            GraptorVertex {
                id: "entity::hub".to_owned(),
                kind: "entity".to_owned(),
                entity_id: Some("hub".to_owned()),
                ..Default::default()
            },
        );
        graph.vertices.insert(
            "chapter::doc-1::1".to_owned(),
            GraptorVertex {
                id: "chapter::doc-1::1".to_owned(),
                kind: "chapter".to_owned(),
                document_id: Some("doc-1".to_owned()),
                chapter_id: Some(1),
                ..Default::default()
            },
        );
        for edge in [
            GraptorEdge {
                source_id: "leaf::seed".to_owned(),
                target_id: "entity::hub".to_owned(),
                edge_type: "mentions".to_owned(),
                weight: 10,
                ..Default::default()
            },
            GraptorEdge {
                source_id: "entity::hub".to_owned(),
                target_id: "chapter::doc-1::1".to_owned(),
                edge_type: "in_chapter".to_owned(),
                weight: 3,
                ..Default::default()
            },
        ] {
            graph
                .outgoing
                .entry(edge.source_id.clone())
                .or_default()
                .push(edge.clone());
            graph
                .incoming
                .entry(edge.target_id.clone())
                .or_default()
                .push(edge);
        }

        let mut seed_scores = FxHashMap::default();
        seed_scores.insert("leaf::seed".to_owned(), 1.0);

        let uniform = PhoenixGldr::new(GldrConfig {
            ppr_alpha: 0.15,
            ppr_alpha_leaf: 0.15,
            ppr_alpha_entity: 0.15,
            ppr_alpha_chapter: 0.15,
            ..Default::default()
        })
        .personalized_pagerank(&graph, &TemporalFilter::default(), &seed_scores);

        let annealed = PhoenixGldr::new(GldrConfig {
            ppr_alpha: 0.15,
            ppr_alpha_leaf: 0.05,
            ppr_alpha_entity: 0.60,
            ppr_alpha_chapter: 0.60,
            ..Default::default()
        })
        .personalized_pagerank(&graph, &TemporalFilter::default(), &seed_scores);

        assert!(
            annealed["entity::hub"] < uniform["entity::hub"],
            "higher entity alpha should damp stationary mass landing on entity hubs",
        );
        assert!(
            annealed["chapter::doc-1::1"] < uniform["chapter::doc-1::1"],
            "higher chapter alpha should damp stationary mass landing on structural connectors",
        );
    }

    #[test]
    fn gldr_semantic_query_uses_document_shortlist_before_leaf_ann() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(
            &store,
            "doc-1",
            "Semantic One",
            "Crimson harbor bells at dawn.",
        );
        seed_note(
            &store,
            "doc-2",
            "Semantic Two",
            "Moonlit observatory above the desert.",
        );

        for (chunk_id, key, document_id, text) in [
            (
                4001_i64,
                "doc-1:1:0:0-29",
                "doc-1",
                "Crimson harbor bells at dawn.",
            ),
            (
                4002_i64,
                "doc-2:1:0:0-36",
                "doc-2",
                "Moonlit observatory above the desert.",
            ),
        ] {
            store
                .put_row(
                    "chunks",
                    json!({
                        "chunk_id": chunk_id,
                        "doc_id": document_id,
                        "level": 0,
                        "start": 0,
                        "end": text.len() as i64,
                        "text": text,
                        "parent_id": null,
                        "scope_narrative": "nar-1",
                        "scope_folder": "folder-1",
                        "created_at": 1
                    }),
                )
                .expect("chunk");
            store
                .put_row(
                    "chunkid_map",
                    json!({
                        "id": chunk_id,
                        "chunk_key": key,
                        "doc_id": document_id,
                        "created_at": 1
                    }),
                )
                .expect("chunkid");
            store
                .put_row(
                    "graph_vertices",
                    json!({
                        "id": format!("leaf::{key}"),
                        "value": { "kind": "leaf", "searchChunkId": key },
                        "weight": 1,
                        "attributes": { "documentId": document_id, "chapterId": 1 }
                    }),
                )
                .expect("leaf vertex");
        }

        let doc1_vector = semantic_vector(0);
        let doc2_vector = semantic_vector(1);
        store
            .upsert_semantic_vectors(&[
                phoenix_store_cozo::SemanticVectorRow {
                    span_id: "doc-1:1:0:0-29",
                    values: &doc1_vector,
                    model_id: phoenix_store_cozo::SEMANTIC_MODEL_ID,
                    updated_at: 10,
                },
                phoenix_store_cozo::SemanticVectorRow {
                    span_id: "doc-2:1:0:0-36",
                    values: &doc2_vector,
                    model_id: phoenix_store_cozo::SEMANTIC_MODEL_ID,
                    updated_at: 10,
                },
            ])
            .expect("leaf vectors");
        store
            .upsert_semantic_document_vectors(&[
                phoenix_store_cozo::SemanticDocumentVectorRow {
                    document_id: "doc-1",
                    values: &doc1_vector,
                    model_id: phoenix_store_cozo::SEMANTIC_MODEL_ID,
                    leaf_count: 1,
                    evidence_refs: &[],
                    updated_at: 10,
                },
                phoenix_store_cozo::SemanticDocumentVectorRow {
                    document_id: "doc-2",
                    values: &doc2_vector,
                    model_id: phoenix_store_cozo::SEMANTIC_MODEL_ID,
                    leaf_count: 1,
                    evidence_refs: &[],
                    updated_at: 10,
                },
            ])
            .expect("document vectors");

        let lex = LexIndex::from_store(&store, LexConfig::default()).expect("lex");
        let result = PhoenixGldr::default()
            .query(
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "astral signal".to_owned(),
                    scope: ScopeKey {
                        world_id: Some("world-1".to_owned()),
                        narrative_id: Some("nar-1".to_owned()),
                        folder_id: Some("folder-1".to_owned()),
                        folder_path: None,
                    },
                    targets: vec![phoenix_types::QueryTarget::Semantic],
                    limit: Some(3),
                    temporal: None,
                    semantic_query_vector: Some(SemanticQueryVector {
                        values: doc2_vector.clone(),
                    }),
                    include_candidate_graph: false,
                },
            )
            .expect("semantic query");

        assert!(
            !result.chunk_hits.is_empty(),
            "semantic query should return at least one chunk hit",
        );
        assert_eq!(result.chunk_hits[0].chunk_id, "doc-2:1:0:0-36");
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.code == "PX_GLDR_SEMANTIC"
                    && diag.message.contains("shortlisted 2 documents")
                    && diag.message.contains("fused 2 filtered leaf neighbors")
                    && diag.message.contains("fallback_to_global_leaf_ann=false")),
            "semantic diagnostic should report document shortlist usage",
        );
    }

    #[test]
    fn gldr_semantic_query_falls_back_to_global_leaf_ann_when_document_vectors_are_missing() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(
            &store,
            "doc-3",
            "Fallback",
            "The hidden archive shimmered quietly.",
        );
        store
            .put_row(
                "chunks",
                json!({
                    "chunk_id": 5001_i64,
                    "doc_id": "doc-3",
                    "level": 0,
                    "start": 0,
                    "end": 36,
                    "text": "The hidden archive shimmered quietly.",
                    "parent_id": null,
                    "scope_narrative": "nar-1",
                    "scope_folder": "folder-1",
                    "created_at": 1
                }),
            )
            .expect("chunk");
        store
            .put_row(
                "chunkid_map",
                json!({
                    "id": 5001_i64,
                    "chunk_key": "doc-3:1:0:0-36",
                    "doc_id": "doc-3",
                    "created_at": 1
                }),
            )
            .expect("chunkid");
        store
            .put_row(
                "graph_vertices",
                json!({
                    "id": "leaf::doc-3:1:0:0-36",
                    "value": { "kind": "leaf", "searchChunkId": "doc-3:1:0:0-36" },
                    "weight": 1,
                    "attributes": { "documentId": "doc-3", "chapterId": 1 }
                }),
            )
            .expect("leaf vertex");

        let vector = semantic_vector(2);
        store
            .upsert_semantic_vectors(&[phoenix_store_cozo::SemanticVectorRow {
                span_id: "doc-3:1:0:0-36",
                values: &vector,
                model_id: phoenix_store_cozo::SEMANTIC_MODEL_ID,
                updated_at: 10,
            }])
            .expect("leaf vectors");

        let lex = LexIndex::from_store(&store, LexConfig::default()).expect("lex");
        let result = PhoenixGldr::default()
            .query(
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "mysterious archive".to_owned(),
                    scope: ScopeKey {
                        world_id: Some("world-1".to_owned()),
                        narrative_id: Some("nar-1".to_owned()),
                        folder_id: Some("folder-1".to_owned()),
                        folder_path: None,
                    },
                    targets: vec![phoenix_types::QueryTarget::Semantic],
                    limit: Some(3),
                    temporal: None,
                    semantic_query_vector: Some(SemanticQueryVector {
                        values: vector.clone(),
                    }),
                    include_candidate_graph: false,
                },
            )
            .expect("fallback semantic query");

        assert_eq!(result.chunk_hits.len(), 1);
        assert_eq!(result.chunk_hits[0].chunk_id, "doc-3:1:0:0-36");
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.code == "PX_GLDR_SEMANTIC"
                    && diag.message.contains("shortlisted 0 documents")
                    && diag.message.contains("fallback_to_global_leaf_ann=true")),
            "semantic diagnostic should report fallback when document vectors are unavailable",
        );
    }

    #[test]
    fn gldr_honors_chapter_temporal_filter() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(&store, "doc-1", "Temporal", "Ryan stood. Ryan ran.");

        for (chunk_id, chapter_id, key, text) in [
            (2001_i64, 1_u32, "doc-1:1:0:0-11", "Ryan stood."),
            (2002_i64, 2_u32, "doc-1:2:0:12-21", "Ryan ran."),
        ] {
            store
                .put_row(
                    "chunks",
                    json!({
                        "chunk_id": chunk_id,
                        "doc_id": "doc-1",
                        "level": 0,
                        "start": if chapter_id == 1 { 0 } else { 12 },
                        "end": if chapter_id == 1 { 11 } else { 21 },
                        "text": text,
                        "parent_id": null,
                        "scope_narrative": null,
                        "scope_folder": null,
                        "created_at": 1
                    }),
                )
                .expect("chunk");
            store
                .put_row(
                    "chunkid_map",
                    json!({
                        "id": chunk_id,
                        "chunk_key": key,
                        "doc_id": "doc-1",
                        "created_at": 1
                    }),
                )
                .expect("chunkid");
            store
                .put_row(
                    "graph_vertices",
                    json!({
                        "id": format!("leaf::{key}"),
                        "value": { "kind": "leaf", "searchChunkId": key },
                        "weight": 1,
                        "attributes": { "documentId": "doc-1", "chapterId": chapter_id }
                    }),
                )
                .expect("leaf");
        }
        store.put_row(
            "graph_vertices",
            json!({
                "id": "entity::ryan",
                "value": { "kind": "entity", "entityId": "ryan", "label": "Ryan", "entityKind": "Character" },
                "weight": 2,
                "attributes": { "chapters": [1, 2] }
            }),
        )
        .expect("entity");
        store
            .put_row(
                "graph_edges",
                json!({
                    "source_id": "leaf::doc-1:1:0:0-11",
                    "target_id": "entity::ryan",
                    "weight": 100,
                    "attributes": { "confidence": 1.0 },
                    "data": null,
                    "edge_type": "mentions"
                }),
            )
            .expect("mentions 1");
        store
            .put_row(
                "graph_edges",
                json!({
                    "source_id": "leaf::doc-1:2:0:12-21",
                    "target_id": "entity::ryan",
                    "weight": 100,
                    "attributes": { "confidence": 1.0 },
                    "data": null,
                    "edge_type": "mentions"
                }),
            )
            .expect("mentions 2");

        let lex = LexIndex::from_store(&store, LexConfig::default()).expect("lex");
        let gldr = PhoenixGldr::default();
        let result = gldr
            .query(
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "Ryan".to_owned(),
                    scope: test_scope(),
                    targets: vec![phoenix_types::QueryTarget::Graph],
                    limit: Some(5),
                    temporal: Some(TemporalMarker {
                        source: Some(TemporalSource::Chapter),
                        chapter: Some(2),
                        calendar: None,
                        story_time: None,
                        ordinal: None,
                        ..Default::default()
                    }),
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                },
            )
            .expect("temporal query");

        assert_eq!(result.chunk_hits.len(), 1);
        assert_eq!(result.chunk_hits[0].chunk_id, "doc-1:2:0:12-21");
        assert_eq!(result.node_hits.len(), 1);
    }
}
