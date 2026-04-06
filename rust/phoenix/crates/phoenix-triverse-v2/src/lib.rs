use phoenix_graph::{GraptorGraph, PhoenixGraphBackend};
use phoenix_graph_kernel::PhoenixGraphKernel;
use phoenix_lex::LexIndex;
use phoenix_store_cozo::StoreError;
use phoenix_store_native::PhoenixArchiveStoreV2;
use phoenix_types::{Diagnostic, EntityId, LexicalSearchResult, NodeHit, QueryRequest, QueryResult};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TriverseV2Config {
    pub max_graph_hops: usize,
}

impl Default for TriverseV2Config {
    fn default() -> Self {
        Self { max_graph_hops: 2 }
    }
}

#[derive(Default)]
pub struct PhoenixTriverseV2 {
    config: TriverseV2Config,
}

impl PhoenixTriverseV2 {
    pub fn new(config: TriverseV2Config) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &TriverseV2Config {
        &self.config
    }

    pub fn query(
        &self,
        _store: &dyn PhoenixArchiveStoreV2,
        backend: &dyn PhoenixGraphBackend,
    ) -> Result<(GraptorGraph, QueryResult), StoreError> {
        let graph = backend
            .snapshot(true)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok((graph, QueryResult::default()))
    }

    pub fn query_kernel_request(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        kernel: &PhoenixGraphKernel,
        lex: &LexIndex,
        request: &QueryRequest,
    ) -> Result<QueryResult, StoreError> {
        let lexical = lex.search(&request.query, &request.scope, request.limit.unwrap_or(5));
        self.query_kernel_request_with_lexical(store, kernel, lexical, request)
    }

    pub fn query_kernel_request_with_lexical(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        kernel: &PhoenixGraphKernel,
        lexical: LexicalSearchResult,
        request: &QueryRequest,
    ) -> Result<QueryResult, StoreError> {
        let chunk_hits = lexical
            .span_hits
            .iter()
            .map(|hit| phoenix_types::ChunkHit {
                chunk_id: hit.span_id.clone(),
                score: hit.score,
            })
            .collect::<Vec<_>>();

        let query = normalize_surface(&request.query);
        let mut entity_scores = self.semantic_entity_scores(store, &request.scope, &query)?;
        self.apply_graph_boosts(kernel, &chunk_hits, &mut entity_scores);

        let mut node_hits = entity_scores
            .into_iter()
            .map(|(entity_id, score)| NodeHit {
                entity_id: Some(EntityId(entity_id)),
                score,
            })
            .collect::<Vec<_>>();
        node_hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.entity_id.cmp(&right.entity_id))
        });
        node_hits.truncate(request.limit.unwrap_or(5));

        let graph_view = kernel.graph_view();
        let mut diagnostics = lexical.diagnostics;
        diagnostics.push(Diagnostic {
            code: "PX_TRIVERSE_V2_KERNEL".to_owned(),
            message: format!(
                "Triverse V2 queried the kernel graph with {} vertices, {} asserted edges, and {} CSR rows.",
                graph_view.vertices.len(),
                graph_view.asserted_edges.len(),
                kernel.csr_sidecar().offsets.len().saturating_sub(1),
            ),
        });
        diagnostics.push(Diagnostic {
            code: "PX_TRIVERSE_V2_SIDECAR".to_owned(),
            message: "Entity seed lookup used scope lex sidecars instead of semantic bundle scans.".to_owned(),
        });
        if request.temporal.is_some() {
            diagnostics.push(Diagnostic {
                code: "PX_TRIVERSE_V2_TEMPORAL".to_owned(),
                message: "Temporal filtering is applied from V2 archive metadata on the native path.".to_owned(),
            });
        }
        Ok(QueryResult {
            session_id: request.session_id.clone(),
            chunk_hits,
            node_hits,
            diagnostics,
        })
    }

    fn semantic_entity_scores(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        scope: &phoenix_types::ScopeKey,
        query: &str,
    ) -> Result<FxHashMap<String, f64>, StoreError> {
        let mut scores = FxHashMap::default();
        let postings = store.lookup_alias_postings(scope, query)?;
        if postings.is_empty() {
            return Ok(scores);
        }
        let score = term_score(query, query);
        for posting in postings {
            scores
                .entry(posting.entity_id.clone())
                .and_modify(|existing| *existing += score + (posting.mention_count as f64).ln_1p() * 0.05)
                .or_insert(score + (posting.mention_count as f64).ln_1p() * 0.05);
        }
        Ok(scores)
    }

    fn apply_graph_boosts(
        &self,
        kernel: &PhoenixGraphKernel,
        chunk_hits: &[phoenix_types::ChunkHit],
        scores: &mut FxHashMap<String, f64>,
    ) {
        let csr = kernel.csr_sidecar();
        for hit in chunk_hits {
            let Some(source_index) = kernel.vertex_ordinal(&hit.chunk_id) else {
                continue;
            };
            let start = csr.offsets.get(source_index).copied().unwrap_or_default();
            let end = csr.offsets.get(source_index + 1).copied().unwrap_or(start);
            for edge_ix in start..end {
                let Some(target_ix) = csr.targets.get(edge_ix).copied() else {
                    continue;
                };
                let Some(vertex_id) = csr.vertex_ids.get(target_ix) else {
                    continue;
                };
                let Some(vertex) = kernel.vertex(vertex_id) else {
                    continue;
                };
                let Some(entity_id) = vertex.entity_id.clone() else {
                    continue;
                };
                let boost = hit.score * 0.2;
                scores
                    .entry(entity_id)
                    .and_modify(|existing| *existing += boost)
                    .or_insert(boost);
            }
        }
    }
}

fn normalize_surface(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ch.is_whitespace())
        .flat_map(|ch| ch.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn term_score(query: &str, candidate: &str) -> f64 {
    if query.is_empty() || candidate.is_empty() {
        return 0.0;
    }
    if query == candidate {
        return 1.0;
    }
    if candidate.contains(query) || query.contains(candidate) {
        return 0.8;
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_invarant_v2::PhoenixInvarantV2;
    use phoenix_graph_kernel::{
        KernelEdgeType, KernelGraphLayer, KernelMutationBatch, KernelMutationScope, KernelVertex,
        KernelVertexId,
    };
    use phoenix_store_lmdb::PhoenixLmdbStore;
    use phoenix_store_native::PhoenixArchiveStoreV2;
    use phoenix_types::{ChunkHit, DocumentId, IngestDocument, ScopeKey};
    use serde_json::json;
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!(
            "phoenix-triverse-v2-test-{name}-{}",
            SystemTime::now().duration_since(UNIX_EPOCH).expect("time").as_nanos()
        ))
    }

    #[test]
    fn query_uses_sidecar_and_kernel_csr() {
        let store = PhoenixLmdbStore::open(temp_path("sidecar")).expect("store");
        store.init_archive_schema().expect("schema");
        let scope = ScopeKey::default();
        let invarant = PhoenixInvarantV2::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-1".to_owned()),
            note_id: None,
            title: "Ryan".to_owned(),
            text: "Ryan found the map. Ryan shared it.".to_owned(),
            scope: scope.clone(),
        };
        invarant
            .ingest_documents_native(&store, None, &[document], 0, 1)
            .expect("ingest");

        let mut kernel = PhoenixGraphKernel::default();
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Document {
                    document_id: "doc-1".to_owned(),
                },
                recorded_at: None,
                vertices: vec![
                    KernelVertex {
                        id: KernelVertexId("doc-1:0".to_owned()),
                        kind: "chunk".to_owned(),
                        value: json!({}),
                        attributes: json!({}),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("entity::ryan".to_owned()),
                        kind: "entity".to_owned(),
                        value: json!({}),
                        attributes: json!({}),
                        entity_id: Some("ryan".to_owned()),
                        ..KernelVertex::default()
                    },
                ],
                edges: vec![phoenix_graph_kernel::KernelEdge {
                    source_id: KernelVertexId("doc-1:0".to_owned()),
                    target_id: KernelVertexId("entity::ryan".to_owned()),
                    edge_type: KernelEdgeType("mentions".to_owned()),
                    weight: 1,
                    attributes: json!({}),
                    layer: KernelGraphLayer::Asserted,
                    ..phoenix_graph_kernel::KernelEdge::default()
                }],
            })
            .expect("batch");
        let lex = LexIndex::default();
        let mut boosted_scores = FxHashMap::default();
        PhoenixTriverseV2::default().apply_graph_boosts(
            &kernel,
            &[ChunkHit {
                chunk_id: "doc-1:0".to_owned(),
                score: 1.0,
            }],
            &mut boosted_scores,
        );
        let query = QueryRequest {
            query: "Ryan".to_owned(),
            scope,
            limit: Some(5),
            ..QueryRequest::default()
        };
        let result = PhoenixTriverseV2::default()
            .query_kernel_request(&store, &kernel, &lex, &query)
            .expect("query");
        assert!(boosted_scores.contains_key("ryan"));
        assert!(result
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "PX_TRIVERSE_V2_SIDECAR"));
    }
}
