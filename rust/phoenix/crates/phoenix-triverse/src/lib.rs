use std::collections::{hash_map::Entry, VecDeque};

use phoenix_graptor::{GraptorEdge, GraptorGraph, GraptorVertex};
use phoenix_invarant::{
    CanonicalEntity, Claim, CoreferenceChainArtifact, Event, EvidenceAnchor,
    INVARANT_SEMANTIC_CLAIM_NAMESPACE, INVARANT_SEMANTIC_COREFERENCE_NAMESPACE,
    INVARANT_SEMANTIC_DOCUMENT_NAMESPACE, INVARANT_SEMANTIC_ENTITY_NAMESPACE,
    INVARANT_SEMANTIC_EVENT_NAMESPACE, INVARANT_SEMANTIC_EVIDENCE_NAMESPACE,
};
use phoenix_lex::LexIndex;
use phoenix_store_cozo::{PhoenixCozoStore, SemanticNeighbor, StoreError};
use phoenix_store_native::{PhoenixNativeRowStore, ScopedDefinitionFilter, ScopedDocumentFilter};
use phoenix_types::{
    ChunkHit, Diagnostic, EntityId, LexicalSearchResult, NodeHit, QueryRequest, QueryResult,
    SessionId, TemporalMarker,
};
use rustc_hash::{FxHashMap, FxHashSet};
use scirs2_graph::algorithms::shortest_path::dijkstra_path_digraph;
use scirs2_graph::link_prediction::{
    adamic_adar_index, common_neighbors_score, jaccard_coefficient, preferential_attachment,
    resource_allocation_index,
};
use scirs2_graph::temporal::{
    TemporalEdge as StreamTemporalEdge, TemporalGraph as StreamTemporalGraph,
};
use scirs2_graph::{
    csr_pagerank, louvain_communities_result,
    personalized_pagerank as scirs2_personalized_pagerank, CsrGraph, DiGraph, Graph as SciRsGraph,
    PageRankConfig,
};
use serde::de::DeserializeOwned;
use serde_json::Value;

#[derive(Clone, Debug)]
pub struct TriverseConfig {
    pub seed_limit: usize,
    pub walk_hops: usize,
    pub context_seed_limit: usize,
    pub context_seed_score: f64,
    pub ppr_alpha: f64,
    pub ppr_iterations: usize,
    pub ppr_tolerance: f64,
    pub teleport_leaf_bias: f64,
    pub teleport_entity_bias: f64,
    pub teleport_chapter_bias: f64,
    pub path_explanation_limit: usize,
    pub community_compression: f64,
    pub candidate_score_strength: f64,
    pub temporal_projection_decay: f64,
}

impl Default for TriverseConfig {
    fn default() -> Self {
        Self {
            seed_limit: 12,
            walk_hops: 3,
            context_seed_limit: 8,
            context_seed_score: 0.05,
            ppr_alpha: 0.85,
            ppr_iterations: 48,
            ppr_tolerance: 1e-6,
            teleport_leaf_bias: 1.0,
            teleport_entity_bias: 1.15,
            teleport_chapter_bias: 1.05,
            path_explanation_limit: 3,
            community_compression: 0.2,
            candidate_score_strength: 1.0,
            temporal_projection_decay: 0.35,
        }
    }
}

pub struct PhoenixTriverse {
    config: TriverseConfig,
}

#[derive(Clone, Debug, Default)]
struct SemanticResolution {
    hits: Vec<SemanticNeighbor>,
    shortlisted_documents: usize,
    filtered_leaf_hits: usize,
    used_global_fallback: bool,
}

#[derive(Clone, Debug, Default)]
struct TriverseSeedSet {
    seed_scores: FxHashMap<String, f64>,
    seed_vertex_ids: Vec<String>,
    semantic_resolution: SemanticResolution,
    invarant_resolution: InvarantSemanticResolution,
    context_seed_count: usize,
}

#[derive(Clone, Debug, Default)]
struct InvarantSemanticResolution {
    shortlisted_documents: usize,
    entity_matches: usize,
    claim_matches: usize,
    event_matches: usize,
    coreference_matches: usize,
    evidence_chunk_seeds: usize,
}

#[derive(Clone, Debug, Default)]
struct TriverseNeighborhood {
    graph: GraptorGraph,
    temporal: TemporalFilter,
}

#[derive(Clone, Debug, Default)]
struct TriverseRanking {
    stationary_scores: FxHashMap<String, f64>,
    chunk_scores: FxHashMap<String, f64>,
    node_scores: FxHashMap<String, f64>,
    pagerank_iterations: usize,
    pagerank_residual: f64,
    pagerank_converged: bool,
    explanation_paths: Vec<String>,
    community_count: usize,
    community_modularity: f64,
    candidate_edge_scores: usize,
    temporal_projection_vertices: usize,
    temporal_projection_paths: usize,
}

#[derive(Clone, Debug, Default)]
struct PageRankRun {
    scores: FxHashMap<String, f64>,
    iterations: usize,
    residual: f64,
    converged: bool,
}

#[derive(Clone, Debug, Default)]
struct CommunityProjection {
    assignments: FxHashMap<String, usize>,
    sizes: FxHashMap<usize, usize>,
    seed_mass: FxHashMap<usize, f64>,
    modularity: f64,
    community_count: usize,
}

#[derive(Clone, Debug, Default)]
struct CandidateEdgeModel {
    scores: FxHashMap<(String, String, String), f64>,
}

#[derive(Clone, Debug, Default)]
struct TemporalProjection {
    vertex_weights: FxHashMap<String, f64>,
    path_count: usize,
}

#[derive(Clone, Debug)]
struct TemporalProjectionGraph {
    graph: StreamTemporalGraph,
    index_by_id: FxHashMap<String, usize>,
}

struct TriverseAnalysisViews {
    walk_graph: SciRsGraph<String, f64>,
    temporal_projection_graph: Option<TemporalProjectionGraph>,
}

impl PhoenixTriverse {
    pub fn new(config: TriverseConfig) -> Self {
        Self { config }
    }

    pub fn query(
        &self,
        native_store: Option<&dyn PhoenixNativeRowStore>,
        store: &PhoenixCozoStore,
        lex: &LexIndex,
        request: &QueryRequest,
        full_graph: &GraptorGraph,
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
        let seed_set = self.collect_seeds(
            native_store,
            store,
            request,
            full_graph,
            semantic_requested,
            limit,
            &lexical,
        )?;
        let neighborhood = self.build_neighborhood(request, full_graph, &seed_set);
        let mut analysis_views = build_analysis_views(&neighborhood.graph, &neighborhood.temporal);
        let ranking = self.rank(
            &neighborhood.graph,
            &neighborhood.temporal,
            &seed_set,
            &mut analysis_views,
        );

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

        let mut diagnostics = lexical.diagnostics;
        diagnostics.push(Diagnostic {
            code: "PX_TRIVERSE_OK".to_owned(),
            message: "Phoenix Triverse fused lexical, semantic, and graph-context seeds across the native graph using SciRS2 CSR PageRank.".to_owned(),
        });
        diagnostics.push(Diagnostic {
            code: "PX_TRIVERSE_GRAPH".to_owned(),
            message: format!(
                "Triverse ranked {} native graph vertices across {} projected chunk hits and {} projected node hits (iterations={}, converged={}, residual={:.3e}).",
                ranking.stationary_scores.len(),
                ranking.chunk_scores.len(),
                ranking.node_scores.len(),
                ranking.pagerank_iterations,
                ranking.pagerank_converged,
                ranking.pagerank_residual,
            ),
        });
        if semantic_requested {
            diagnostics.push(Diagnostic {
                code: if request.semantic_query_vector.is_some() {
                    "PX_TRIVERSE_SEMANTIC".to_owned()
                } else {
                    "PX_TRIVERSE_SEMANTIC_MISSING_VECTOR".to_owned()
                },
                message: if request.semantic_query_vector.is_some() {
                    format!(
                        "Triverse shortlisted {} documents and fused {} filtered leaf neighbors from store-backed semantic ANN (fallback_to_global_leaf_ann={}).",
                        seed_set.semantic_resolution.shortlisted_documents,
                        seed_set.semantic_resolution.filtered_leaf_hits,
                        seed_set.semantic_resolution.used_global_fallback
                    )
                } else {
                    "Semantic target requested without a query vector; Triverse used lexical and graph-context retrieval only."
                        .to_owned()
                },
            });
        }
        if seed_set.context_seed_count > 0 {
            diagnostics.push(Diagnostic {
                code: "PX_TRIVERSE_CONTEXT".to_owned(),
                message: format!(
                    "Triverse incorporated {} native session-context seeds from the active runtime graph.",
                    seed_set.context_seed_count
                ),
            });
        }
        if seed_set.invarant_resolution.entity_matches > 0
            || seed_set.invarant_resolution.claim_matches > 0
            || seed_set.invarant_resolution.event_matches > 0
            || seed_set.invarant_resolution.coreference_matches > 0
        {
            diagnostics.push(Diagnostic {
                code: "PX_TRIVERSE_INVARANT".to_owned(),
                message: format!(
                    "Triverse fused {} scoped canonical entities, {} claims, {} events, and {} coreference chains from Invarant semantic planes into {} evidence-backed chunk seeds across {} semantic documents.",
                    seed_set.invarant_resolution.entity_matches,
                    seed_set.invarant_resolution.claim_matches,
                    seed_set.invarant_resolution.event_matches,
                    seed_set.invarant_resolution.coreference_matches,
                    seed_set.invarant_resolution.evidence_chunk_seeds,
                    seed_set.invarant_resolution.shortlisted_documents,
                ),
            });
        }
        if request.include_candidate_graph {
            diagnostics.push(Diagnostic {
                code: "PX_TRIVERSE_CANDIDATE_GRAPH".to_owned(),
                message: "Triverse included candidate graph edges during native graph walking and ranking."
                    .to_owned(),
            });
        }
        if request.temporal.is_some() {
            diagnostics.push(Diagnostic {
                code: "PX_TRIVERSE_TEMPORAL".to_owned(),
                message: neighborhood.temporal.diagnostic_message(),
            });
        }
        if ranking.community_count > 0 {
            diagnostics.push(Diagnostic {
                code: "PX_TRIVERSE_COMMUNITIES".to_owned(),
                message: format!(
                    "Triverse compressed the walked neighborhood into {} Louvain communities (modularity={:.3}).",
                    ranking.community_count,
                    ranking.community_modularity
                ),
            });
        }
        if ranking.candidate_edge_scores > 0 {
            diagnostics.push(Diagnostic {
                code: "PX_TRIVERSE_LINK_PREDICTION".to_owned(),
                message: format!(
                    "Triverse rescored {} candidate edges with native link-prediction heuristics before ranking.",
                    ranking.candidate_edge_scores
                ),
            });
        }
        if ranking.temporal_projection_vertices > 0 {
            diagnostics.push(Diagnostic {
                code: "PX_TRIVERSE_TEMPORAL_GRAPH".to_owned(),
                message: format!(
                    "Triverse projected {} temporally-positioned vertices through {} foremost temporal paths.",
                    ranking.temporal_projection_vertices,
                    ranking.temporal_projection_paths
                ),
            });
        }
        if !ranking.explanation_paths.is_empty() {
            diagnostics.push(Diagnostic {
                code: "PX_TRIVERSE_PATHS".to_owned(),
                message: format!(
                    "Representative native graph paths: {}",
                    ranking.explanation_paths.join(" | ")
                ),
            });
        }

        Ok(QueryResult {
            session_id: request.session_id.clone(),
            chunk_hits: if wants_chunks {
                ranked_chunk_hits(ranking.chunk_scores, limit)
            } else {
                Vec::new()
            },
            node_hits: if wants_nodes {
                ranked_node_hits(ranking.node_scores, limit)
            } else {
                Vec::new()
            },
            diagnostics,
        })
    }

    fn collect_seeds(
        &self,
        native_store: Option<&dyn PhoenixNativeRowStore>,
        store: &PhoenixCozoStore,
        request: &QueryRequest,
        full_graph: &GraptorGraph,
        semantic_requested: bool,
        limit: usize,
        lexical: &LexicalSearchResult,
    ) -> Result<TriverseSeedSet, StoreError> {
        let semantic_resolution = if semantic_requested {
            self.resolve_semantic_hits(store, request, limit)?
        } else {
            SemanticResolution::default()
        };
        let seed_limit = self.config.seed_limit.max(limit);
        let mut seed_scores = FxHashMap::<String, f64>::default();
        let mut seed_vertex_ids = Vec::<String>::new();

        for hit in lexical.span_hits.iter().take(seed_limit) {
            let leaf_id = leaf_vertex_id(&hit.span_id);
            if full_graph.vertices.contains_key(&leaf_id) {
                push_seed(
                    &mut seed_scores,
                    &mut seed_vertex_ids,
                    leaf_id,
                    hit.score.max(0.0),
                );
            }
        }

        for (rank, hit) in semantic_resolution.hits.iter().take(seed_limit).enumerate() {
            let leaf_id = leaf_vertex_id(&hit.span_id);
            if full_graph.vertices.contains_key(&leaf_id) {
                push_seed(
                    &mut seed_scores,
                    &mut seed_vertex_ids,
                    leaf_id,
                    semantic_seed_score(rank, hit.distance),
                );
            }
        }

        let invarant_resolution = self.collect_invarant_seeds(
            native_store,
            store,
            request,
            full_graph,
            &mut seed_scores,
            &mut seed_vertex_ids,
        )?;

        let context_vertex_ids = session_context_vertex_ids(
            full_graph,
            request.session_id.as_ref(),
            self.config.context_seed_limit,
        );
        for vertex_id in &context_vertex_ids {
            push_seed(
                &mut seed_scores,
                &mut seed_vertex_ids,
                vertex_id.clone(),
                self.config.context_seed_score,
            );
        }

        Ok(TriverseSeedSet {
            seed_scores,
            seed_vertex_ids,
            semantic_resolution,
            invarant_resolution,
            context_seed_count: context_vertex_ids.len(),
        })
    }

    fn build_neighborhood(
        &self,
        request: &QueryRequest,
        full_graph: &GraptorGraph,
        seed_set: &TriverseSeedSet,
    ) -> TriverseNeighborhood {
        TriverseNeighborhood {
            graph: load_subgraph_from_graph(
                full_graph,
                &seed_set.seed_vertex_ids,
                self.config.walk_hops,
            ),
            temporal: TemporalFilter::from_marker(request.temporal.as_ref()),
        }
    }

    fn rank(
        &self,
        graph: &GraptorGraph,
        temporal: &TemporalFilter,
        seed_set: &TriverseSeedSet,
        analysis_views: &mut TriverseAnalysisViews,
    ) -> TriverseRanking {
        let community_projection =
            self.detect_communities(&analysis_views.walk_graph, &seed_set.seed_scores);
        let candidate_edge_model =
            self.score_candidate_edges(graph, temporal, &analysis_views.walk_graph);
        let temporal_projection = self.build_temporal_projection(
            analysis_views.temporal_projection_graph.as_mut(),
            &seed_set.seed_scores,
        );
        let pagerank = self.personalized_pagerank(
            graph,
            temporal,
            &seed_set.seed_scores,
            &candidate_edge_model,
            &temporal_projection,
        );
        let stationary_scores = self.apply_post_rank_adjustments(
            graph,
            pagerank.scores,
            &community_projection,
            &temporal_projection,
        );
        let explanation_paths =
            self.explain_ranked_paths(graph, temporal, &seed_set.seed_scores, &stationary_scores);
        TriverseRanking {
            stationary_scores: stationary_scores.clone(),
            chunk_scores: project_chunk_scores(graph, &stationary_scores),
            node_scores: project_node_scores(graph, &stationary_scores),
            pagerank_iterations: pagerank.iterations,
            pagerank_residual: pagerank.residual,
            pagerank_converged: pagerank.converged,
            explanation_paths,
            community_count: community_projection.community_count,
            community_modularity: community_projection.modularity,
            candidate_edge_scores: candidate_edge_model.scores.len(),
            temporal_projection_vertices: temporal_projection.vertex_weights.len(),
            temporal_projection_paths: temporal_projection.path_count,
        }
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

    fn collect_invarant_seeds(
        &self,
        native_store: Option<&dyn PhoenixNativeRowStore>,
        store: &PhoenixCozoStore,
        request: &QueryRequest,
        full_graph: &GraptorGraph,
        seed_scores: &mut FxHashMap<String, f64>,
        seed_vertex_ids: &mut Vec<String>,
    ) -> Result<InvarantSemanticResolution, StoreError> {
        let query_terms = normalized_terms(&request.query);
        if query_terms.is_empty() {
            return Ok(InvarantSemanticResolution::default());
        }

        let scoped_documents = load_invarant_document_keys(native_store, store, request)?;
        let restrict_documents = !scoped_documents.is_empty();
        let evidence_rows = fetch_invarant_definition_rows(
            native_store,
            store,
            INVARANT_SEMANTIC_EVIDENCE_NAMESPACE,
        )?;

        let mut evidence_by_id = FxHashMap::<String, EvidenceAnchor>::default();
        for row in &evidence_rows {
            for anchor in deserialize_scoped_payloads::<EvidenceAnchor>(
                row,
                INVARANT_SEMANTIC_EVIDENCE_NAMESPACE,
                "evidenceAnchors",
            ) {
                if !restrict_documents || scoped_documents.contains(&anchor.document_id) {
                    evidence_by_id.insert(anchor.evidence_id.0.clone(), anchor);
                }
            }
        }

        let mut resolution = InvarantSemanticResolution {
            shortlisted_documents: scoped_documents.len(),
            ..InvarantSemanticResolution::default()
        };

        for row in fetch_invarant_definition_rows(
            native_store,
            store,
            INVARANT_SEMANTIC_ENTITY_NAMESPACE,
        )? {
            for entity in deserialize_scoped_payloads::<CanonicalEntity>(
                &row,
                INVARANT_SEMANTIC_ENTITY_NAMESPACE,
                "entities",
            ) {
                if !scope_matches(&request.scope, &entity.scope) {
                    continue;
                }
                let entity_match = semantic_text_match(
                    &query_terms,
                    std::iter::once(entity.label.as_str())
                        .chain(entity.aliases.iter().map(String::as_str)),
                );
                if entity_match <= 0.0 {
                    continue;
                }
                let vertex_id = format!("entity::{}", entity.entity_id.0);
                if full_graph.vertices.contains_key(&vertex_id) {
                    push_seed(
                        seed_scores,
                        seed_vertex_ids,
                        vertex_id,
                        0.2 + entity_match * 0.5,
                    );
                    resolution.entity_matches += 1;
                }
            }
        }

        for row in fetch_invarant_definition_rows(
            native_store,
            store,
            INVARANT_SEMANTIC_CLAIM_NAMESPACE,
        )? {
            for claim in deserialize_scoped_payloads::<Claim>(
                &row,
                INVARANT_SEMANTIC_CLAIM_NAMESPACE,
                "claims",
            ) {
                let claim_match = semantic_text_match(
                    &query_terms,
                    [
                        Some(claim.relation_type.as_str()),
                        Some(claim.event_class.as_str()),
                        claim.subject_text.as_deref(),
                        claim.object_text.as_deref(),
                        claim.recipient_text.as_deref(),
                    ]
                    .into_iter()
                    .flatten(),
                );
                if claim_match <= 0.0 {
                    continue;
                }
                resolution.claim_matches += 1;
                resolution.evidence_chunk_seeds += seed_claim_artifacts(
                    full_graph,
                    &claim.evidence_chunk_ids,
                    &claim.evidence_ids,
                    &evidence_by_id,
                    seed_scores,
                    seed_vertex_ids,
                    0.15 + claim_match * 0.35,
                );
                for entity_id in [
                    claim.subject_entity_id.as_ref(),
                    claim.object_entity_id.as_ref(),
                    claim.recipient_entity_id.as_ref(),
                ]
                .into_iter()
                .flatten()
                {
                    let vertex_id = format!("entity::{}", entity_id.0);
                    if full_graph.vertices.contains_key(&vertex_id) {
                        push_seed(
                            seed_scores,
                            seed_vertex_ids,
                            vertex_id,
                            0.1 + claim_match * 0.2,
                        );
                    }
                }
            }
        }

        for row in fetch_invarant_definition_rows(
            native_store,
            store,
            INVARANT_SEMANTIC_EVENT_NAMESPACE,
        )? {
            for event in deserialize_scoped_payloads::<Event>(
                &row,
                INVARANT_SEMANTIC_EVENT_NAMESPACE,
                "events",
            ) {
                let event_match = semantic_text_match(
                    &query_terms,
                    [event.label.as_str(), event.event_class.as_str()].into_iter(),
                );
                if event_match <= 0.0 {
                    continue;
                }
                resolution.event_matches += 1;
                resolution.evidence_chunk_seeds += seed_evidence_chunks(
                    full_graph,
                    &event.evidence_ids,
                    &evidence_by_id,
                    seed_scores,
                    seed_vertex_ids,
                    0.12 + event_match * 0.25,
                );
                for entity_id in &event.participant_entity_ids {
                    let vertex_id = format!("entity::{}", entity_id.0);
                    if full_graph.vertices.contains_key(&vertex_id) {
                        push_seed(
                            seed_scores,
                            seed_vertex_ids,
                            vertex_id,
                            0.08 + event_match * 0.18,
                        );
                    }
                }
            }
        }

        for row in fetch_invarant_definition_rows(
            native_store,
            store,
            INVARANT_SEMANTIC_COREFERENCE_NAMESPACE,
        )? {
            for chain in deserialize_scoped_payloads::<CoreferenceChainArtifact>(
                &row,
                INVARANT_SEMANTIC_COREFERENCE_NAMESPACE,
                "chains",
            ) {
                let coref_match = semantic_text_match(
                    &query_terms,
                    std::iter::once(chain.canonical.as_str()).chain(
                        chain.mentions.iter().map(|mention| mention.surface.as_str()),
                    ),
                );
                if coref_match <= 0.0 {
                    continue;
                }
                resolution.coreference_matches += 1;
                for chunk_id in &chain.chunk_ids {
                    let vertex_id = leaf_vertex_id(&chunk_id.0);
                    if full_graph.vertices.contains_key(&vertex_id) {
                        push_seed(
                            seed_scores,
                            seed_vertex_ids,
                            vertex_id,
                            0.1 + coref_match * 0.2,
                        );
                        resolution.evidence_chunk_seeds += 1;
                    }
                }
                resolution.evidence_chunk_seeds += seed_evidence_chunks(
                    full_graph,
                    &chain.evidence_ids,
                    &evidence_by_id,
                    seed_scores,
                    seed_vertex_ids,
                    0.1 + coref_match * 0.2,
                );
            }
        }

        Ok(resolution)
    }

    fn personalized_pagerank(
        &self,
        graph: &GraptorGraph,
        temporal: &TemporalFilter,
        seed_scores: &FxHashMap<String, f64>,
        candidate_edge_model: &CandidateEdgeModel,
        temporal_projection: &TemporalProjection,
    ) -> PageRankRun {
        if graph.vertices.is_empty() || seed_scores.is_empty() {
            return PageRankRun::default();
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
            return PageRankRun::default();
        }

        let index_by_id = vertex_ids
            .iter()
            .enumerate()
            .map(|(index, vertex_id)| (vertex_id.clone(), index))
            .collect::<FxHashMap<_, _>>();
        let mut personalization = vec![0.0; vertex_ids.len()];
        let total_seed = seed_scores
            .iter()
            .filter_map(|(vertex_id, score)| {
                (index_by_id.contains_key(vertex_id) && *score > 0.0).then_some(*score)
            })
            .sum::<f64>();
        if total_seed <= f64::EPSILON {
            return PageRankRun::default();
        }
        for (vertex_id, score) in seed_scores {
            let Some(&index) = index_by_id.get(vertex_id) else {
                continue;
            };
            let teleport_bias = graph
                .vertices
                .get(vertex_id)
                .map(|vertex| self.teleport_bias_for_vertex(vertex))
                .unwrap_or(1.0);
            let temporal_weight = temporal_projection
                .vertex_weights
                .get(vertex_id)
                .copied()
                .unwrap_or(1.0);
            personalization[index] += (*score / total_seed) * teleport_bias * temporal_weight;
        }
        normalize_distribution(&mut personalization);

        let mut adjacency = Vec::<(usize, usize, f64)>::new();
        let mut walk_graph = SciRsGraph::new();
        for vertex_id in &vertex_ids {
            walk_graph.add_node(vertex_id.clone());
        }
        for edge in graph.outgoing.values().flat_map(|edges| edges.iter()) {
            let (Some(&source_index), Some(&target_index)) = (
                index_by_id.get(&edge.source_id),
                index_by_id.get(&edge.target_id),
            ) else {
                continue;
            };
            let weight = transition_weight(
                edge,
                candidate_edge_model,
                self.config.candidate_score_strength,
                temporal_projection,
            );
            adjacency.push((source_index, target_index, weight));
            adjacency.push((target_index, source_index, weight));
            let _ = walk_graph.add_edge(edge.source_id.clone(), edge.target_id.clone(), weight);
        }

        let csr_graph = CsrGraph::from_edges(vertex_ids.len(), adjacency, true);
        let global_result = csr_graph.and_then(|csr_graph| {
            csr_pagerank(
                &csr_graph,
                &PageRankConfig {
                    damping: self.config.ppr_alpha.clamp(0.0, 1.0),
                    max_iterations: self.config.ppr_iterations.max(1),
                    tolerance: self.config.ppr_tolerance.max(1e-9),
                },
            )
        });

        let personalized_scores = self.multi_seed_personalized_scores(
            &walk_graph,
            &vertex_ids,
            &index_by_id,
            &personalization,
        );
        let mut combined = vec![0.0; vertex_ids.len()];
        let mut iterations = self.config.ppr_iterations.max(1);
        let mut residual = f64::NAN;
        let mut converged = false;

        if let Ok(result) = global_result {
            iterations = result.iterations;
            residual = result.residual;
            converged = result.converged;
            for (index, score) in result.scores.into_iter().enumerate() {
                combined[index] += score * 0.25;
            }
        }

        if !personalized_scores.is_empty() {
            for (index, score) in personalized_scores.into_iter().enumerate() {
                combined[index] += score * 0.75;
            }
        }

        if combined.iter().all(|score| *score <= f64::EPSILON) {
            let scores = self.personalized_pagerank_fallback(graph, &index_by_id, &personalization);
            return PageRankRun {
                scores,
                iterations,
                residual,
                converged,
            };
        }

        normalize_distribution(&mut combined);
        let scores = vertex_ids
            .into_iter()
            .zip(combined)
            .filter(|(_, score)| *score > f64::EPSILON)
            .collect::<FxHashMap<_, _>>();
        PageRankRun {
            scores,
            iterations,
            residual,
            converged,
        }
    }

    fn personalized_pagerank_fallback(
        &self,
        graph: &GraptorGraph,
        index_by_id: &FxHashMap<String, usize>,
        personalization: &[f64],
    ) -> FxHashMap<String, f64> {
        let mut neighbor_weights = vec![FxHashMap::<usize, f64>::default(); index_by_id.len()];
        for edge in graph.outgoing.values().flat_map(|edges| edges.iter()) {
            let (Some(&source_index), Some(&target_index)) = (
                index_by_id.get(&edge.source_id),
                index_by_id.get(&edge.target_id),
            ) else {
                continue;
            };
            let weight = edge_transition_weight_without_models(edge);
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

        let mut scores = personalization.to_vec();
        let mut next_scores = vec![0.0; scores.len()];
        for _ in 0..self.config.ppr_iterations.max(1) {
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

            for index in 0..next_scores.len() {
                let teleport = personalization[index] * (1.0 - self.config.ppr_alpha);
                let base_mass = next_scores[index] + dangling_mass * personalization[index];
                next_scores[index] = teleport + self.config.ppr_alpha * base_mass;
            }
            normalize_distribution(&mut next_scores);
            std::mem::swap(&mut scores, &mut next_scores);
        }

        index_by_id
            .iter()
            .filter_map(|(vertex_id, index)| {
                let score = scores[*index];
                (score > f64::EPSILON).then_some((vertex_id.clone(), score))
            })
            .collect()
    }

    fn detect_communities(
        &self,
        walk_graph: &SciRsGraph<String, f64>,
        seed_scores: &FxHashMap<String, f64>,
    ) -> CommunityProjection {
        if walk_graph.node_count() < 2 {
            return CommunityProjection::default();
        }
        let result = louvain_communities_result(walk_graph);
        self.community_projection_from_result(seed_scores, result)
    }

    fn community_projection_from_result(
        &self,
        seed_scores: &FxHashMap<String, f64>,
        result: scirs2_graph::algorithms::community::types::CommunityResult<String>,
    ) -> CommunityProjection {
        let mut assignments = FxHashMap::default();
        let mut sizes = FxHashMap::<usize, usize>::default();
        let mut seed_mass = FxHashMap::<usize, f64>::default();
        for (vertex_id, community) in result.node_communities {
            assignments.insert(vertex_id.clone(), community);
            *sizes.entry(community).or_insert(0) += 1;
            if let Some(score) = seed_scores.get(&vertex_id) {
                *seed_mass.entry(community).or_insert(0.0) += *score;
            }
        }
        CommunityProjection {
            assignments,
            sizes,
            seed_mass,
            modularity: result.quality_score.unwrap_or(0.0),
            community_count: result.num_communities,
        }
    }

    fn score_candidate_edges(
        &self,
        graph: &GraptorGraph,
        temporal: &TemporalFilter,
        walk_graph: &SciRsGraph<String, f64>,
    ) -> CandidateEdgeModel {
        let mut scores = FxHashMap::default();
        for edge in graph.outgoing.values().flat_map(|edges| edges.iter()) {
            if !edge.edge_type.starts_with("candidate_") {
                continue;
            }
            let (Some(source), Some(target)) = (
                graph.vertices.get(&edge.source_id),
                graph.vertices.get(&edge.target_id),
            ) else {
                continue;
            };
            if !temporal.matches_vertex(source) || !temporal.matches_vertex(target) {
                continue;
            }
            let score = candidate_link_score(walk_graph, &edge.source_id, &edge.target_id);
            scores.insert(
                (
                    edge.source_id.clone(),
                    edge.target_id.clone(),
                    edge.edge_type.clone(),
                ),
                score,
            );
        }
        CandidateEdgeModel { scores }
    }

    fn build_temporal_projection(
        &self,
        temporal_graph: Option<&mut TemporalProjectionGraph>,
        seed_scores: &FxHashMap<String, f64>,
    ) -> TemporalProjection {
        let Some(temporal_graph) = temporal_graph else {
            return TemporalProjection::default();
        };
        let Some(primary_seed_id) = strongest_seed_id(seed_scores) else {
            return TemporalProjection::default();
        };
        let Some(seed_index) = temporal_graph.index_by_id.get(&primary_seed_id).copied() else {
            return TemporalProjection::default();
        };
        let mut vertex_weights = FxHashMap::default();
        let mut path_count = 0usize;
        for (vertex_id, &target_index) in &temporal_graph.index_by_id {
            if vertex_id == &primary_seed_id {
                vertex_weights.insert(vertex_id.clone(), 1.0);
                continue;
            }
            if let Some(path) =
                temporal_graph
                    .graph
                    .foremost_path(seed_index, target_index, 0.0, f64::INFINITY)
            {
                path_count += 1;
                let distance = path.len().saturating_sub(1) as f64;
                let weight = (1.0 / (1.0 + distance * self.config.temporal_projection_decay))
                    .clamp(0.25, 1.0);
                vertex_weights.insert(vertex_id.clone(), weight);
            }
        }
        TemporalProjection {
            vertex_weights,
            path_count,
        }
    }

    fn apply_post_rank_adjustments(
        &self,
        graph: &GraptorGraph,
        mut stationary_scores: FxHashMap<String, f64>,
        community_projection: &CommunityProjection,
        temporal_projection: &TemporalProjection,
    ) -> FxHashMap<String, f64> {
        for (vertex_id, score) in stationary_scores.iter_mut() {
            if let Some(community) = community_projection.assignments.get(vertex_id) {
                let size = community_projection
                    .sizes
                    .get(community)
                    .copied()
                    .unwrap_or(1)
                    .max(1) as f64;
                let seed_mass = community_projection
                    .seed_mass
                    .get(community)
                    .copied()
                    .unwrap_or(0.0);
                let compression =
                    1.0 / size.powf(self.config.community_compression.clamp(0.0, 1.0));
                let seed_boost = 1.0 + seed_mass.min(1.0) * 0.35;
                *score *= compression * seed_boost;
            }
            if let Some(weight) = temporal_projection.vertex_weights.get(vertex_id) {
                *score *= *weight;
            } else if graph.vertices.contains_key(vertex_id)
                && !temporal_projection.vertex_weights.is_empty()
            {
                *score *= 0.7;
            }
        }
        renormalize_scores(&mut stationary_scores);
        stationary_scores
    }

    fn explain_ranked_paths(
        &self,
        graph: &GraptorGraph,
        temporal: &TemporalFilter,
        seed_scores: &FxHashMap<String, f64>,
        stationary_scores: &FxHashMap<String, f64>,
    ) -> Vec<String> {
        let Some(primary_seed_id) = strongest_seed_id(seed_scores) else {
            return Vec::new();
        };
        let Ok(explanation_graph) = build_explanation_graph(graph, temporal) else {
            return Vec::new();
        };
        top_explanation_targets(graph, stationary_scores, self.config.path_explanation_limit)
            .into_iter()
            .filter(|target_id| target_id != &primary_seed_id)
            .filter_map(|target_id| {
                dijkstra_path_digraph(&explanation_graph, &primary_seed_id, &target_id)
                    .ok()
                    .and_then(|path| path)
                    .map(|path| format_explanation_path(graph, &path.nodes))
            })
            .collect()
    }

    fn multi_seed_personalized_scores(
        &self,
        walk_graph: &SciRsGraph<String, f64>,
        vertex_ids: &[String],
        index_by_id: &FxHashMap<String, usize>,
        personalization: &[f64],
    ) -> Vec<f64> {
        let top_seeds = vertex_ids
            .iter()
            .enumerate()
            .filter_map(|(index, vertex_id)| {
                (personalization.get(index).copied().unwrap_or_default() > f64::EPSILON)
                    .then_some((vertex_id.clone(), personalization[index]))
            })
            .collect::<Vec<_>>();
        if top_seeds.is_empty() {
            return Vec::new();
        }

        let mut aggregated = vec![0.0; vertex_ids.len()];
        let seed_total = top_seeds.iter().map(|(_, weight)| *weight).sum::<f64>();
        if seed_total <= f64::EPSILON {
            return aggregated;
        }

        for (vertex_id, seed_weight) in top_seeds
            .into_iter()
            .take(self.config.seed_limit.min(4).max(1))
        {
            let Ok(result) = scirs2_personalized_pagerank(
                walk_graph,
                &vertex_id,
                self.config.ppr_alpha.clamp(0.0, 1.0),
                self.config.ppr_tolerance.max(1e-9),
                self.config.ppr_iterations.max(1),
            ) else {
                continue;
            };
            let normalized_weight = seed_weight / seed_total;
            for (ranked_vertex_id, score) in result {
                if let Some(&index) = index_by_id.get(&ranked_vertex_id) {
                    aggregated[index] += normalized_weight * score;
                }
            }
        }

        normalize_distribution(&mut aggregated);
        aggregated
    }

    fn teleport_bias_for_vertex(&self, vertex: &GraptorVertex) -> f64 {
        match vertex.kind.as_str() {
            "leaf" => self.config.teleport_leaf_bias,
            "entity" => self.config.teleport_entity_bias,
            "chapter" => self.config.teleport_chapter_bias,
            _ => 1.0,
        }
        .max(0.0)
    }
}

impl Default for PhoenixTriverse {
    fn default() -> Self {
        Self::new(TriverseConfig::default())
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
        Self {
            chapter: marker.and_then(|marker| marker.chapter),
            boundary_doc_id: marker.and_then(|marker| {
                marker
                    .boundary_doc_id
                    .as_ref()
                    .map(|document_id| document_id.0.clone())
            }),
            boundary_ordinal: marker.and_then(|marker| {
                marker
                    .boundary_ordinal
                    .or(marker.ordinal)
                    .map(|value| value as u32)
            }),
            boundary_end_ordinal: marker
                .and_then(|marker| marker.boundary_end_ordinal.map(|value| value as u32)),
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
                return format!("Applied boundary-local temporal filtering at {doc} ordinal {boundary_ordinal}.");
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

fn push_seed(
    seed_scores: &mut FxHashMap<String, f64>,
    seed_vertex_ids: &mut Vec<String>,
    vertex_id: String,
    score: f64,
) {
    if score <= 0.0 {
        return;
    }
    match seed_scores.entry(vertex_id) {
        Entry::Occupied(mut occupied) => {
            *occupied.get_mut() += score;
        }
        Entry::Vacant(vacant) => {
            seed_vertex_ids.push(vacant.key().clone());
            vacant.insert(score);
        }
    }
}

fn session_context_vertex_ids(
    graph: &GraptorGraph,
    session_id: Option<&SessionId>,
    limit: usize,
) -> Vec<String> {
    let Some(session_id) = session_id else {
        return Vec::new();
    };
    let mut vertex_ids = graph
        .vertices
        .iter()
        .filter(|(_, vertex)| {
            vertex
                .attributes
                .get("sessionId")
                .and_then(|value| value.as_str())
                == Some(session_id.0.as_str())
        })
        .map(|(vertex_id, _)| vertex_id.clone())
        .collect::<Vec<_>>();
    vertex_ids.sort();
    if vertex_ids.len() > limit {
        vertex_ids.truncate(limit);
    }
    vertex_ids
}

fn accumulate_score(scores: &mut FxHashMap<String, f64>, key: &str, score: f64) {
    scores
        .entry(key.to_owned())
        .and_modify(|value| *value += score)
        .or_insert(score);
}

fn load_invarant_document_keys(
    native_store: Option<&dyn PhoenixNativeRowStore>,
    store: &PhoenixCozoStore,
    request: &QueryRequest,
) -> Result<FxHashSet<String>, StoreError> {
    Ok(fetch_invarant_document_rows(native_store, store)?
        .into_iter()
        .filter(|row| scoped_document_matches_request(row, request))
        .filter_map(|row| {
            row.get("payload")
                .and_then(|payload| payload.get("documentId"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect())
}

fn fetch_invarant_document_rows(
    native_store: Option<&dyn PhoenixNativeRowStore>,
    store: &PhoenixCozoStore,
) -> Result<Vec<Value>, StoreError> {
    if let Some(native_store) = native_store {
        native_store.fetch_scoped_documents(ScopedDocumentFilter {
            namespace: Some(INVARANT_SEMANTIC_DOCUMENT_NAMESPACE),
            ..ScopedDocumentFilter::default()
        })
    } else {
        store.fetch_rows("scoped_documents").map(|rows| {
            rows.into_iter()
                .filter(|row| {
                    row.get("namespace").and_then(Value::as_str)
                        == Some(INVARANT_SEMANTIC_DOCUMENT_NAMESPACE)
                })
                .collect()
        })
    }
}

fn fetch_invarant_definition_rows(
    native_store: Option<&dyn PhoenixNativeRowStore>,
    store: &PhoenixCozoStore,
    namespace: &str,
) -> Result<Vec<Value>, StoreError> {
    if let Some(native_store) = native_store {
        native_store.fetch_scoped_definitions(ScopedDefinitionFilter {
            namespace: Some(namespace),
            ..ScopedDefinitionFilter::default()
        })
    } else {
        store.fetch_rows("scoped_definitions").map(|rows| {
            rows.into_iter()
                .filter(|row| row.get("namespace").and_then(Value::as_str) == Some(namespace))
                .collect()
        })
    }
}

fn scoped_document_matches_request(row: &Value, request: &QueryRequest) -> bool {
    let Some(payload) = row.get("payload") else {
        return false;
    };
    if let Some(session_id) = request.session_id.as_ref() {
        if payload.get("sessionId").and_then(Value::as_str) != Some(session_id.0.as_str()) {
            return false;
        }
    }
    let Some(scope_value) = payload.get("scope").cloned() else {
        return false;
    };
    let Ok(scope) = serde_json::from_value::<phoenix_types::ScopeKey>(scope_value) else {
        return false;
    };
    scope_matches(&request.scope, &scope)
}

fn scope_matches(
    query_scope: &phoenix_types::ScopeKey,
    candidate_scope: &phoenix_types::ScopeKey,
) -> bool {
    query_scope
        .world_id
        .as_ref()
        .map(|value| candidate_scope.world_id.as_ref() == Some(value))
        .unwrap_or(true)
        && query_scope
            .narrative_id
            .as_ref()
            .map(|value| candidate_scope.narrative_id.as_ref() == Some(value))
            .unwrap_or(true)
        && query_scope
            .folder_id
            .as_ref()
            .map(|value| candidate_scope.folder_id.as_ref() == Some(value))
            .unwrap_or(true)
        && query_scope
            .folder_path
            .as_ref()
            .map(|value| candidate_scope.folder_path.as_ref() == Some(value))
            .unwrap_or(true)
}

fn deserialize_scoped_payload<T>(row: &Value, namespace: &str) -> Option<T>
where
    T: DeserializeOwned,
{
    (row.get("namespace").and_then(Value::as_str) == Some(namespace))
        .then(|| row.get("payload").cloned())
        .flatten()
        .and_then(|payload| serde_json::from_value(payload).ok())
}

fn deserialize_scoped_payloads<T>(row: &Value, namespace: &str, array_field: &str) -> Vec<T>
where
    T: DeserializeOwned,
{
    if row.get("namespace").and_then(Value::as_str) != Some(namespace) {
        return Vec::new();
    }
    let Some(payload) = row.get("payload").cloned() else {
        return Vec::new();
    };
    if let Ok(item) = serde_json::from_value::<T>(payload.clone()) {
        return vec![item];
    }
    payload
        .get(array_field)
        .cloned()
        .and_then(|value| serde_json::from_value::<Vec<T>>(value).ok())
        .unwrap_or_default()
}

fn normalized_terms(text: &str) -> Vec<String> {
    let normalized = normalize_text(text);
    normalized
        .split_whitespace()
        .filter(|part| part.len() >= 2)
        .map(ToOwned::to_owned)
        .collect()
}

fn normalize_text(text: &str) -> String {
    text.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect()
}

fn semantic_text_match<'a>(
    query_terms: &[String],
    candidates: impl IntoIterator<Item = &'a str>,
) -> f64 {
    if query_terms.is_empty() {
        return 0.0;
    }
    let surfaces = candidates
        .into_iter()
        .map(normalize_text)
        .filter(|surface| !surface.trim().is_empty())
        .collect::<Vec<_>>();
    if surfaces.is_empty() {
        return 0.0;
    }

    let mut best = 0.0_f64;
    for surface in &surfaces {
        let matched = query_terms
            .iter()
            .filter(|term| surface.contains(term.as_str()))
            .count();
        if matched == 0 {
            continue;
        }
        let ratio = matched as f64 / query_terms.len() as f64;
        let exact = surface.contains(&query_terms.join(" ")) as u8 as f64 * 0.25;
        best = best.max((ratio + exact).min(1.0));
    }
    best
}

fn seed_claim_artifacts(
    full_graph: &GraptorGraph,
    evidence_chunk_ids: &[phoenix_invarant::ChunkId],
    evidence_ids: &[phoenix_invarant::EvidenceId],
    evidence_by_id: &FxHashMap<String, EvidenceAnchor>,
    seed_scores: &mut FxHashMap<String, f64>,
    seed_vertex_ids: &mut Vec<String>,
    score: f64,
) -> usize {
    let mut chunk_ids = evidence_chunk_ids
        .iter()
        .map(|chunk_id| chunk_id.0.clone())
        .collect::<FxHashSet<_>>();
    for evidence_id in evidence_ids {
        if let Some(anchor) = evidence_by_id.get(&evidence_id.0) {
            if let Some(chunk_id) = anchor.chunk_id.as_ref() {
                chunk_ids.insert(chunk_id.0.clone());
            }
        }
    }
    seed_chunk_ids(full_graph, &chunk_ids, seed_scores, seed_vertex_ids, score)
}

fn seed_evidence_chunks(
    full_graph: &GraptorGraph,
    evidence_ids: &[phoenix_invarant::EvidenceId],
    evidence_by_id: &FxHashMap<String, EvidenceAnchor>,
    seed_scores: &mut FxHashMap<String, f64>,
    seed_vertex_ids: &mut Vec<String>,
    score: f64,
) -> usize {
    let chunk_ids = evidence_ids
        .iter()
        .filter_map(|evidence_id| evidence_by_id.get(&evidence_id.0))
        .filter_map(|anchor| anchor.chunk_id.as_ref().map(|chunk_id| chunk_id.0.clone()))
        .collect::<FxHashSet<_>>();
    seed_chunk_ids(full_graph, &chunk_ids, seed_scores, seed_vertex_ids, score)
}

fn seed_chunk_ids(
    full_graph: &GraptorGraph,
    chunk_ids: &FxHashSet<String>,
    seed_scores: &mut FxHashMap<String, f64>,
    seed_vertex_ids: &mut Vec<String>,
    score: f64,
) -> usize {
    let mut seeded = 0usize;
    for chunk_id in chunk_ids {
        let vertex_id = leaf_vertex_id(chunk_id);
        if full_graph.vertices.contains_key(&vertex_id) {
            push_seed(seed_scores, seed_vertex_ids, vertex_id, score);
            seeded += 1;
        }
    }
    seeded
}

fn add_transition(neighbors: &mut FxHashMap<usize, f64>, target_index: usize, weight: f64) {
    neighbors
        .entry(target_index)
        .and_modify(|value| *value += weight)
        .or_insert(weight);
}

fn normalize_distribution(values: &mut [f64]) {
    let total = values.iter().sum::<f64>();
    if total > f64::EPSILON {
        for value in values {
            *value /= total;
        }
    }
}

fn renormalize_scores(scores: &mut FxHashMap<String, f64>) {
    let total = scores.values().sum::<f64>();
    if total > f64::EPSILON {
        for score in scores.values_mut() {
            *score /= total;
        }
    }
}

fn edge_transition_weight_without_models(edge: &GraptorEdge) -> f64 {
    let base = 1.0 + (edge.weight.max(1) as f64).ln() * 0.2;
    let multiplier = match edge.edge_type.as_str() {
        "mentions" | "observed_in" | "depends_on" => 1.15,
        "supported_by" | "contradicted_by" | "valid_under" => 1.10,
        "about" | "relevant_to" => 0.95,
        edge_type if edge_type.starts_with("candidate_") => 0.80,
        _ => 1.0,
    };
    (base * multiplier).max(0.05)
}

fn transition_weight(
    edge: &GraptorEdge,
    candidate_edge_model: &CandidateEdgeModel,
    candidate_score_strength: f64,
    temporal_projection: &TemporalProjection,
) -> f64 {
    let mut weight = edge_transition_weight_without_models(edge);
    if edge.edge_type.starts_with("candidate_") {
        let key = (
            edge.source_id.clone(),
            edge.target_id.clone(),
            edge.edge_type.clone(),
        );
        if let Some(score) = candidate_edge_model.scores.get(&key) {
            let strength = candidate_score_strength.clamp(0.0, 1.0);
            let scored_multiplier = 0.65 + *score;
            weight *= 1.0 + (scored_multiplier - 1.0) * strength;
        }
    }
    let source_temporal = temporal_projection
        .vertex_weights
        .get(&edge.source_id)
        .copied()
        .unwrap_or(1.0);
    let target_temporal = temporal_projection
        .vertex_weights
        .get(&edge.target_id)
        .copied()
        .unwrap_or(1.0);
    weight * ((source_temporal + target_temporal) / 2.0).clamp(0.25, 1.0)
}

fn reciprocal_rank_score(rank: usize) -> f64 {
    1.0 / (60.0 + rank as f64 + 1.0)
}

fn semantic_seed_score(rank: usize, distance: f64) -> f64 {
    reciprocal_rank_score(rank) * (1.0 / (1.0 + distance.max(0.0)))
}

fn strongest_seed_id(seed_scores: &FxHashMap<String, f64>) -> Option<String> {
    seed_scores
        .iter()
        .max_by(|left, right| left.1.total_cmp(right.1).then_with(|| left.0.cmp(right.0)))
        .map(|(vertex_id, _)| vertex_id.clone())
}

fn build_analysis_views(graph: &GraptorGraph, temporal: &TemporalFilter) -> TriverseAnalysisViews {
    let mut vertex_ids = graph
        .vertices
        .iter()
        .filter_map(|(vertex_id, vertex)| {
            temporal.matches_vertex(vertex).then_some(vertex_id.clone())
        })
        .collect::<Vec<_>>();
    vertex_ids.sort();

    let mut walk_graph = SciRsGraph::new();
    for vertex_id in &vertex_ids {
        walk_graph.add_node(vertex_id.clone());
    }

    let temporal_index_by_id = (!vertex_ids.is_empty()
        && (temporal.chapter.is_some() || temporal.boundary_ordinal.is_some()))
    .then(|| {
        vertex_ids
            .iter()
            .enumerate()
            .map(|(index, vertex_id)| (vertex_id.clone(), index))
            .collect::<FxHashMap<_, _>>()
    });
    let mut temporal_graph = temporal_index_by_id
        .as_ref()
        .map(|_| StreamTemporalGraph::new(vertex_ids.len()));

    let mut walk_seen = FxHashSet::<(String, String)>::default();
    let mut temporal_seen = FxHashSet::<(usize, usize, u64)>::default();
    for edge in graph.outgoing.values().flat_map(|edges| edges.iter()) {
        let (Some(source), Some(target)) = (
            graph.vertices.get(&edge.source_id),
            graph.vertices.get(&edge.target_id),
        ) else {
            continue;
        };
        if !temporal.matches_vertex(source) || !temporal.matches_vertex(target) {
            continue;
        }

        if !edge.edge_type.starts_with("candidate_") {
            let key = if edge.source_id <= edge.target_id {
                (edge.source_id.clone(), edge.target_id.clone())
            } else {
                (edge.target_id.clone(), edge.source_id.clone())
            };
            if walk_seen.insert(key) {
                let _ = walk_graph.add_edge(
                    edge.source_id.clone(),
                    edge.target_id.clone(),
                    edge_transition_weight_without_models(edge),
                );
            }
        }

        let (Some(index_by_id), Some(temporal_graph)) =
            (temporal_index_by_id.as_ref(), temporal_graph.as_mut())
        else {
            continue;
        };
        let (Some(&source_index), Some(&target_index)) = (
            index_by_id.get(&edge.source_id),
            index_by_id.get(&edge.target_id),
        ) else {
            continue;
        };
        let Some(timestamp) = temporal_edge_timestamp(source, target) else {
            continue;
        };
        let stamp_key = timestamp.to_bits();
        if temporal_seen.insert((source_index, target_index, stamp_key)) {
            temporal_graph.add_edge(StreamTemporalEdge::with_weight(
                source_index,
                target_index,
                timestamp,
                edge_transition_weight_without_models(edge),
            ));
        }
        if temporal_seen.insert((target_index, source_index, stamp_key)) {
            temporal_graph.add_edge(StreamTemporalEdge::with_weight(
                target_index,
                source_index,
                timestamp,
                edge_transition_weight_without_models(edge),
            ));
        }
    }

    TriverseAnalysisViews {
        walk_graph,
        temporal_projection_graph: temporal_graph
            .zip(temporal_index_by_id)
            .map(|(graph, index_by_id)| TemporalProjectionGraph { graph, index_by_id }),
    }
}

fn candidate_link_score(
    walk_graph: &SciRsGraph<String, f64>,
    source_id: &str,
    target_id: &str,
) -> f64 {
    let source = source_id.to_owned();
    let target = target_id.to_owned();
    let cn = common_neighbors_score(walk_graph, &source, &target).unwrap_or(0.0);
    let jc = jaccard_coefficient(walk_graph, &source, &target).unwrap_or(0.0);
    let aa = adamic_adar_index(walk_graph, &source, &target).unwrap_or(0.0);
    let ra = resource_allocation_index(walk_graph, &source, &target).unwrap_or(0.0);
    let pa = preferential_attachment(walk_graph, &source, &target).unwrap_or(0.0);

    let cn_norm = cn / (cn + 2.0);
    let aa_norm = aa / (aa + 2.0);
    let ra_norm = ra / (ra + 1.0);
    let pa_norm = pa.ln_1p() / 10.0_f64.ln_1p();

    ((cn_norm + jc + aa_norm + ra_norm + pa_norm.clamp(0.0, 1.0)) / 5.0).clamp(0.0, 1.0)
}

fn temporal_edge_timestamp(source: &GraptorVertex, target: &GraptorVertex) -> Option<f64> {
    match (source.boundary_ordinal, target.boundary_ordinal) {
        (Some(left), Some(right)) => Some(left.max(right) as f64),
        (Some(left), None) => Some(left as f64),
        (None, Some(right)) => Some(right as f64),
        (None, None) => match (source.chapter_id, target.chapter_id) {
            (Some(left), Some(right)) => Some(left.max(right) as f64),
            (Some(left), None) => Some(left as f64),
            (None, Some(right)) => Some(right as f64),
            (None, None) => None,
        },
    }
}

fn top_explanation_targets(
    graph: &GraptorGraph,
    stationary_scores: &FxHashMap<String, f64>,
    limit: usize,
) -> Vec<String> {
    let mut ranked = stationary_scores
        .iter()
        .filter(|(vertex_id, _)| {
            graph.vertices.get(*vertex_id).is_some_and(|vertex| {
                vertex.search_chunk_id.is_some() || vertex.entity_id.is_some()
            })
        })
        .map(|(vertex_id, score)| (vertex_id.clone(), *score))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked
        .into_iter()
        .take(limit.max(1))
        .map(|(vertex_id, _)| vertex_id)
        .collect()
}

fn build_explanation_graph(
    graph: &GraptorGraph,
    temporal: &TemporalFilter,
) -> Result<DiGraph<String, f64>, String> {
    let mut explanation = DiGraph::new();
    let mut seen = FxHashSet::<(String, String)>::default();
    for (vertex_id, vertex) in &graph.vertices {
        if temporal.matches_vertex(vertex) {
            let _ = explanation.add_node(vertex_id.clone());
        }
    }
    for edge in graph.outgoing.values().flat_map(|edges| edges.iter()) {
        let (Some(source), Some(target)) = (
            graph.vertices.get(&edge.source_id),
            graph.vertices.get(&edge.target_id),
        ) else {
            continue;
        };
        if !temporal.matches_vertex(source) || !temporal.matches_vertex(target) {
            continue;
        }
        let traversal_cost = 1.0 / edge_transition_weight_without_models(edge).max(0.05);
        let forward = (edge.source_id.clone(), edge.target_id.clone());
        if seen.insert(forward) {
            explanation
                .add_edge(
                    edge.source_id.clone(),
                    edge.target_id.clone(),
                    traversal_cost,
                )
                .map_err(|error| error.to_string())?;
        }
        let reverse = (edge.target_id.clone(), edge.source_id.clone());
        if seen.insert(reverse) {
            explanation
                .add_edge(
                    edge.target_id.clone(),
                    edge.source_id.clone(),
                    traversal_cost,
                )
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(explanation)
}

fn format_explanation_path(graph: &GraptorGraph, path: &[String]) -> String {
    path.iter()
        .map(|vertex_id| vertex_display_label(graph, vertex_id))
        .collect::<Vec<_>>()
        .join(" -> ")
}

fn vertex_display_label(graph: &GraptorGraph, vertex_id: &str) -> String {
    let Some(vertex) = graph.vertices.get(vertex_id) else {
        return vertex_id.to_owned();
    };
    let kind = vertex.kind.as_str();
    let label = vertex
        .value
        .get("label")
        .and_then(|value| value.as_str())
        .or_else(|| vertex.value.get("lemma").and_then(|value| value.as_str()))
        .or_else(|| vertex.search_chunk_id.as_deref())
        .or_else(|| vertex.entity_id.as_deref())
        .unwrap_or(vertex_id);
    format!("{kind}:{label}")
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
    let mut touched = FxHashSet::<String>::default();
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
            let neighbor_id = neighbor_id.to_owned();
            if touched.insert(neighbor_id.clone()) {
                queue.push_back((neighbor_id, depth + 1));
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

    let mut seen_edges = FxHashSet::<(String, String, String)>::default();
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
    use phoenix_store_cozo::{
        PhoenixCozoStore, SemanticDocumentVectorRow, SemanticVectorRow, SEMANTIC_MODEL_ID,
        SEMANTIC_VECTOR_DIM,
    };
    use phoenix_types::{
        BoundaryKind, DocumentId, IndexedSpan, IndexedTextField, LexicalField, QueryTarget,
        ScopeKey, SemanticQueryVector, TemporalMarker, TemporalSource,
    };
    use serde_json::json;

    fn scope() -> ScopeKey {
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

    fn seed_chunk(
        store: &PhoenixCozoStore,
        chunk_id: i64,
        chunk_key: &str,
        document_id: &str,
        text: &str,
        chapter_id: u32,
    ) {
        store
            .put_row(
                "chunks",
                json!({
                    "chunk_id": chunk_id,
                    "doc_id": document_id,
                    "level": 0,
                    "start": 0,
                    "end": text.len(),
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
                    "chunk_key": chunk_key,
                    "doc_id": document_id,
                    "created_at": 1
                }),
            )
            .expect("chunkid map");
        store
            .put_row(
                "document_boundaries",
                json!({
                    "doc_id": document_id,
                    "boundary_id": chapter_id,
                    "kind": "chapter",
                    "depth": 0,
                    "ordinal": chapter_id,
                    "label": format!("Chapter {chapter_id}"),
                    "parent_boundary_id": null,
                    "note_id": document_id,
                    "start_char": 0,
                    "end_char": text.len(),
                    "created_at": 1
                }),
            )
            .expect("boundary");
    }

    fn seed_invarant_semantic_rows(store: &PhoenixCozoStore, document_id: &str, chunk_key: &str) {
        store
            .put_row(
                "scoped_documents",
                json!({
                    "id": format!("semantic-doc::{document_id}"),
                    "scope_folder_id": "__root__",
                    "narrative_id": "__global__",
                    "namespace": INVARANT_SEMANTIC_DOCUMENT_NAMESPACE,
                    "document_key": document_id,
                    "payload": {
                        "documentId": document_id,
                        "documentVersionId": format!("version::{document_id}"),
                        "sessionId": null,
                        "scope": scope(),
                        "summary": {
                            "documentId": document_id,
                            "noteId": null,
                            "title": "Harbor ledger",
                            "chunkCount": 1,
                            "chapterCount": 1,
                            "boundaryCount": 1,
                            "parentCount": 0,
                            "leafCount": 1,
                            "entityCount": 1,
                            "mentionCount": 1,
                            "edgeCount": 1,
                            "crossChapterLinks": 0,
                            "discoveryCount": 0
                        }
                    },
                    "seeded_from_scope_folder_id": null,
                    "created_at": 1,
                    "updated_at": 1
                }),
            )
            .expect("semantic doc row");
        store
            .put_row(
                "scoped_definitions",
                json!({
                    "id": "semantic-entity-row",
                    "narrative_id": "__global__",
                    "namespace": INVARANT_SEMANTIC_ENTITY_NAMESPACE,
                    "definition_key": "entity:harbor_authority",
                    "payload": {
                        "entityId": "harbor_authority",
                        "label": "Harbor Authority",
                        "aliases": ["Port Authority"],
                        "kind": "Organization",
                        "scope": scope(),
                        "status": "resolved",
                        "mentionIds": ["mention::1"],
                        "evidenceIds": ["evidence::harbor"],
                        "confidence": 0.92
                    },
                    "created_at": 1,
                    "updated_at": 1
                }),
            )
            .expect("semantic entity row");
        store
            .put_row(
                "scoped_definitions",
                json!({
                    "id": "semantic-claim-row",
                    "narrative_id": "__global__",
                    "namespace": INVARANT_SEMANTIC_CLAIM_NAMESPACE,
                    "definition_key": "claim:harbor-control",
                    "payload": {
                        "claimId": "claim::harbor-control",
                        "relationType": "controls",
                        "eventClass": "governance",
                        "subjectEntityId": "harbor_authority",
                        "objectEntityId": null,
                        "recipientEntityId": null,
                        "subjectText": "Harbor Authority",
                        "objectText": "harbor traffic",
                        "recipientText": null,
                        "evidenceIds": ["evidence::harbor"],
                        "evidenceChunkIds": [chunk_key],
                        "confidence": 0.88
                    },
                    "created_at": 1,
                    "updated_at": 1
                }),
            )
            .expect("semantic claim row");
        store
            .put_row(
                "scoped_definitions",
                json!({
                    "id": "semantic-evidence-row",
                    "narrative_id": "__global__",
                    "namespace": INVARANT_SEMANTIC_EVIDENCE_NAMESPACE,
                    "definition_key": "evidence:evidence::harbor",
                    "payload": {
                        "evidenceId": "evidence::harbor",
                        "documentId": document_id,
                        "chunkId": chunk_key,
                        "spanPath": { "value": "root/0/sentence:0" },
                        "range": { "start": 0, "end": 34 },
                        "sentenceIndex": 0,
                        "label": "Harbor Authority controls harbor traffic.",
                        "kind": "claim"
                    },
                    "created_at": 1,
                    "updated_at": 1
                }),
            )
            .expect("semantic evidence row");
        store
            .put_row(
                "scoped_definitions",
                json!({
                    "id": "semantic-coref-row",
                    "narrative_id": "__global__",
                    "namespace": INVARANT_SEMANTIC_COREFERENCE_NAMESPACE,
                    "definition_key": "coref:harbor-authority",
                    "payload": {
                        "chainId": "coref::harbor-authority",
                        "canonical": "Harbor Authority",
                        "mentions": [
                            {
                                "mentionId": "mention::1",
                                "surface": "Harbor Authority",
                                "canonicalSurface": "Harbor Authority",
                                "range": { "start": 0, "end": 16 },
                                "sentenceIndex": 0
                            },
                            {
                                "mentionId": "mention::2",
                                "surface": "they",
                                "canonicalSurface": "Harbor Authority",
                                "range": { "start": 24, "end": 28 },
                                "sentenceIndex": 0
                            }
                        ],
                        "evidenceIds": ["evidence::harbor"],
                        "chunkIds": [chunk_key],
                        "confidence": 0.74,
                        "provider": "scirs2-text-coref",
                        "providerVersion": "0.4.1",
                        "configHash": "cfg"
                    },
                    "created_at": 1,
                    "updated_at": 1
                }),
            )
            .expect("semantic coreference row");
    }

    fn semantic_vector(primary_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; SEMANTIC_VECTOR_DIM];
        if primary_index < values.len() {
            values[primary_index] = 1.0;
        }
        values
    }

    fn leaf_vertex(chunk_key: &str, document_id: &str, chapter_id: u32) -> GraptorVertex {
        GraptorVertex {
            id: format!("leaf::{chunk_key}"),
            kind: "leaf".to_owned(),
            weight: 1,
            value: json!({ "kind": "leaf", "searchChunkId": chunk_key }),
            attributes: json!({
                "documentId": document_id,
                "chapterId": chapter_id,
            }),
            entity_id: None,
            search_chunk_id: Some(chunk_key.to_owned()),
            document_id: Some(document_id.to_owned()),
            chapter_id: Some(chapter_id),
            chapters: vec![chapter_id],
            boundary_id: Some(chapter_id),
            boundary_ordinal: Some(chapter_id),
            boundary_kind: Some(BoundaryKind::Chapter),
            boundary_ordinals: vec![chapter_id],
        }
    }

    fn entity_vertex(entity_id: &str, label: &str, chapters: Vec<u32>) -> GraptorVertex {
        GraptorVertex {
            id: format!("entity::{entity_id}"),
            kind: "entity".to_owned(),
            weight: 1,
            value: json!({
                "kind": "entity",
                "entityId": entity_id,
                "label": label,
            }),
            attributes: json!({
                "chapters": chapters,
            }),
            entity_id: Some(entity_id.to_owned()),
            search_chunk_id: None,
            document_id: None,
            chapter_id: None,
            chapters,
            boundary_id: None,
            boundary_ordinal: None,
            boundary_kind: None,
            boundary_ordinals: Vec::new(),
        }
    }

    fn add_vertex(graph: &mut GraptorGraph, vertex: GraptorVertex) {
        if let (Some(document_id), Some(chapter_id), Some(search_chunk_id)) = (
            vertex.document_id.clone(),
            vertex.chapter_id,
            vertex.search_chunk_id.clone(),
        ) {
            graph
                .chapter_leaves
                .entry((document_id, chapter_id))
                .or_default()
                .push(vertex.id.clone());
            graph.vertices.insert(vertex.id.clone(), vertex);
            let _ = search_chunk_id;
            return;
        }
        graph.vertices.insert(vertex.id.clone(), vertex);
    }

    fn add_edge(
        graph: &mut GraptorGraph,
        source_id: &str,
        target_id: &str,
        edge_type: &str,
        weight: i64,
    ) {
        let edge = GraptorEdge {
            source_id: source_id.to_owned(),
            target_id: target_id.to_owned(),
            edge_type: edge_type.to_owned(),
            weight,
            attributes: json!({}),
            data: None,
            layer: Default::default(),
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

    fn build_lex(store: &PhoenixCozoStore) -> LexIndex {
        LexIndex::from_store(store, phoenix_lex::LexConfig::default()).expect("lex")
    }

    #[test]
    fn triverse_walks_from_lexical_leaf_to_related_entity_and_chunk() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(
            &store,
            "doc-1",
            "Harbor",
            "Ryan woke early. Ryan crossed the harbor.",
        );
        seed_chunk(
            &store,
            1001,
            "doc-1:1:0:0-17",
            "doc-1",
            "Ryan woke early.",
            1,
        );
        seed_chunk(
            &store,
            1002,
            "doc-1:2:0:18-42",
            "doc-1",
            "Ryan crossed the harbor.",
            2,
        );

        let lex = build_lex(&store);
        let mut graph = GraptorGraph::default();
        add_vertex(&mut graph, leaf_vertex("doc-1:1:0:0-17", "doc-1", 1));
        add_vertex(&mut graph, leaf_vertex("doc-1:2:0:18-42", "doc-1", 2));
        add_vertex(&mut graph, entity_vertex("ryan", "Ryan", vec![1, 2]));
        add_edge(
            &mut graph,
            "leaf::doc-1:1:0:0-17",
            "entity::ryan",
            "mentions",
            4,
        );
        add_edge(
            &mut graph,
            "leaf::doc-1:2:0:18-42",
            "entity::ryan",
            "mentions",
            4,
        );

        let result = PhoenixTriverse::default()
            .query(
                None,
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "Ryan".to_owned(),
                    scope: scope(),
                    targets: vec![QueryTarget::Graph],
                    limit: Some(4),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                },
                &graph,
            )
            .expect("query");

        assert!(
            result
                .chunk_hits
                .iter()
                .any(|hit| hit.chunk_id == "doc-1:1:0:0-17"),
            "seed chunk should remain visible"
        );
        assert!(
            result
                .chunk_hits
                .iter()
                .any(|hit| hit.chunk_id == "doc-1:2:0:18-42"),
            "graph walking should surface the related chunk"
        );
        assert!(
            result
                .node_hits
                .iter()
                .any(|hit| hit.entity_id.as_ref().map(|id| id.0.as_str()) == Some("ryan")),
            "entity ranking should survive projection"
        );
        assert!(result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "PX_TRIVERSE_OK"));
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.code == "PX_TRIVERSE_GRAPH"),
            "native SciRS2 graph ranking should emit graph telemetry"
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.code == "PX_TRIVERSE_PATHS"),
            "native graph ranking should emit representative explanation paths"
        );
    }

    #[test]
    fn triverse_preserves_explanation_paths_when_parallel_edges_exist() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(&store, "doc-dup", "Harbor", "Ryan crossed the harbor.");
        seed_chunk(
            &store,
            1101,
            "doc-dup:1:0:0-25",
            "doc-dup",
            "Ryan crossed the harbor.",
            1,
        );

        let lex = build_lex(&store);
        let mut graph = GraptorGraph::default();
        add_vertex(&mut graph, leaf_vertex("doc-dup:1:0:0-25", "doc-dup", 1));
        add_vertex(&mut graph, entity_vertex("ryan", "Ryan", vec![1]));
        add_edge(
            &mut graph,
            "leaf::doc-dup:1:0:0-25",
            "entity::ryan",
            "mentions",
            4,
        );
        add_edge(
            &mut graph,
            "leaf::doc-dup:1:0:0-25",
            "entity::ryan",
            "observed_in",
            2,
        );

        let result = PhoenixTriverse::default()
            .query(
                None,
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "Ryan".to_owned(),
                    scope: scope(),
                    targets: vec![QueryTarget::Graph],
                    limit: Some(4),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                },
                &graph,
            )
            .expect("query");

        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.code == "PX_TRIVERSE_PATHS"),
            "explanation telemetry should survive duplicate source/target edges"
        );
    }

    #[test]
    fn triverse_uses_store_backed_semantic_shortlists() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(&store, "doc-a", "Camp", "Cold campfire ash.");
        seed_note(
            &store,
            "doc-b",
            "Harbor",
            "Lanterns glow on the harbor wall.",
        );
        seed_chunk(
            &store,
            2001,
            "doc-a:1:0:0-18",
            "doc-a",
            "Cold campfire ash.",
            1,
        );
        seed_chunk(
            &store,
            2002,
            "doc-b:1:0:0-33",
            "doc-b",
            "Lanterns glow on the harbor wall.",
            1,
        );

        let vector_a = semantic_vector(0);
        let vector_b = semantic_vector(1);
        store
            .upsert_semantic_document_vectors(&[
                SemanticDocumentVectorRow {
                    document_id: "doc-a",
                    values: &vector_a,
                    model_id: SEMANTIC_MODEL_ID,
                    leaf_count: 1,
                    evidence_refs: &[],
                    updated_at: 1,
                },
                SemanticDocumentVectorRow {
                    document_id: "doc-b",
                    values: &vector_b,
                    model_id: SEMANTIC_MODEL_ID,
                    leaf_count: 1,
                    evidence_refs: &[],
                    updated_at: 1,
                },
            ])
            .expect("semantic docs");
        store
            .upsert_semantic_vectors(&[
                SemanticVectorRow {
                    span_id: "doc-a:1:0:0-18",
                    values: &vector_a,
                    model_id: SEMANTIC_MODEL_ID,
                    updated_at: 1,
                },
                SemanticVectorRow {
                    span_id: "doc-b:1:0:0-33",
                    values: &vector_b,
                    model_id: SEMANTIC_MODEL_ID,
                    updated_at: 1,
                },
            ])
            .expect("semantic leaves");

        let lex = LexIndex::build(
            &[IndexedSpan {
                span_id: "doc-a:1:0:0-18".to_owned(),
                note_id: None,
                document_id: Some(DocumentId("doc-a".to_owned())),
                scope: scope(),
                fields: vec![IndexedTextField {
                    field: LexicalField::Body,
                    text: "campfire".to_owned(),
                }],
            }],
            phoenix_lex::LexConfig::default(),
        );
        let mut graph = GraptorGraph::default();
        add_vertex(&mut graph, leaf_vertex("doc-a:1:0:0-18", "doc-a", 1));
        add_vertex(&mut graph, leaf_vertex("doc-b:1:0:0-33", "doc-b", 1));

        let result = PhoenixTriverse::default()
            .query(
                None,
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "harbor".to_owned(),
                    scope: scope(),
                    targets: vec![QueryTarget::Semantic],
                    limit: Some(3),
                    temporal: None,
                    semantic_query_vector: Some(SemanticQueryVector {
                        values: vector_b.clone(),
                    }),
                    include_candidate_graph: false,
                },
                &graph,
            )
            .expect("semantic query");

        assert_eq!(
            result.chunk_hits.first().map(|hit| hit.chunk_id.as_str()),
            Some("doc-b:1:0:0-33")
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.code == "PX_TRIVERSE_SEMANTIC"),
            "semantic path should report Triverse semantic fusion"
        );
    }

    #[test]
    fn triverse_uses_invarant_entities_and_claims_as_native_seeds() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(
            &store,
            "doc-semantic",
            "Harbor ledger",
            "Harbor Authority controls harbor traffic.",
        );
        seed_chunk(
            &store,
            4001,
            "doc-semantic:1:0:0-41",
            "doc-semantic",
            "Harbor Authority controls harbor traffic.",
            1,
        );
        seed_invarant_semantic_rows(&store, "doc-semantic", "doc-semantic:1:0:0-41");

        let lex = LexIndex::build(&[], phoenix_lex::LexConfig::default());
        let mut graph = GraptorGraph::default();
        add_vertex(
            &mut graph,
            leaf_vertex("doc-semantic:1:0:0-41", "doc-semantic", 1),
        );
        add_vertex(
            &mut graph,
            entity_vertex("harbor_authority", "Harbor Authority", vec![1]),
        );
        add_edge(
            &mut graph,
            "leaf::doc-semantic:1:0:0-41",
            "entity::harbor_authority",
            "mentions",
            4,
        );

        let result = PhoenixTriverse::default()
            .query(
                None,
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "port authority controls harbor".to_owned(),
                    scope: scope(),
                    targets: vec![QueryTarget::Graph],
                    limit: Some(4),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                },
                &graph,
            )
            .expect("invarant semantic query");

        assert_eq!(
            result.chunk_hits.first().map(|hit| hit.chunk_id.as_str()),
            Some("doc-semantic:1:0:0-41")
        );
        assert!(
            result
                .node_hits
                .iter()
                .any(|hit| hit.entity_id.as_ref().map(|id| id.0.as_str())
                    == Some("harbor_authority")),
            "entity-backed seeds should project canonical entity hits"
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.code == "PX_TRIVERSE_INVARANT"),
            "native semantic rows should surface an Invarant diagnostic"
        );
    }

    #[test]
    fn triverse_applies_temporal_filtering_to_native_rank_projection() {
        let store = PhoenixCozoStore::new().expect("store");
        seed_note(
            &store,
            "doc-temporal",
            "Voyage",
            "Ryan woke early. Ryan sailed at dusk.",
        );
        seed_chunk(
            &store,
            3001,
            "doc-temporal:1:0:0-17",
            "doc-temporal",
            "Ryan woke early.",
            1,
        );
        seed_chunk(
            &store,
            3002,
            "doc-temporal:2:0:18-38",
            "doc-temporal",
            "Ryan sailed at dusk.",
            2,
        );

        let lex = build_lex(&store);
        let mut graph = GraptorGraph::default();
        add_vertex(
            &mut graph,
            leaf_vertex("doc-temporal:1:0:0-17", "doc-temporal", 1),
        );
        add_vertex(
            &mut graph,
            leaf_vertex("doc-temporal:2:0:18-38", "doc-temporal", 2),
        );
        add_vertex(&mut graph, entity_vertex("ryan", "Ryan", vec![1, 2]));
        add_edge(
            &mut graph,
            "leaf::doc-temporal:1:0:0-17",
            "entity::ryan",
            "mentions",
            4,
        );
        add_edge(
            &mut graph,
            "leaf::doc-temporal:2:0:18-38",
            "entity::ryan",
            "mentions",
            4,
        );

        let result = PhoenixTriverse::default()
            .query(
                None,
                &store,
                &lex,
                &QueryRequest {
                    session_id: None,
                    query: "Ryan".to_owned(),
                    scope: scope(),
                    targets: vec![QueryTarget::Graph],
                    limit: Some(4),
                    temporal: Some(TemporalMarker {
                        source: Some(TemporalSource::Chapter),
                        chapter: Some(1),
                        boundary_doc_id: None,
                        boundary_id: None,
                        boundary_ordinal: None,
                        boundary_end_ordinal: None,
                        boundary_kind: None,
                        calendar: None,
                        story_time: None,
                        ordinal: None,
                    }),
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                },
                &graph,
            )
            .expect("temporal query");

        assert!(
            result
                .chunk_hits
                .iter()
                .all(|hit| hit.chunk_id != "doc-temporal:2:0:18-38"),
            "chapter-local filtering should exclude chapter 2 chunks"
        );
        assert!(
            result
                .diagnostics
                .iter()
                .any(|diag| diag.code == "PX_TRIVERSE_TEMPORAL"),
            "temporal filtering should surface a Triverse diagnostic"
        );
    }
}
