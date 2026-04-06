use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

mod binary;
mod planner;
mod view;

use phoenix_analytics::TextAnalytics;
use phoenix_chat::PhoenixChat;
use phoenix_gldr::PhoenixGldr;
use phoenix_graph::{
    GraphBackendError, GraphCounts, GraphEdgeRecord, GraphLayer, GraphMutationBatch,
    GraphMutationScope, GraphVertexRecord,
};
use phoenix_kernel::{
    KernelGraphLayer, KernelGraphSnapshot, KernelMutationBatch as KernelGraphMutationBatch,
    KernelMutationScope,
};
#[cfg(test)]
use phoenix_kernel::{
    KernelBiTemporal, KernelEntityFacet, KernelVertex, KernelVertexClass, KernelVertexId,
    KernelViewRequest,
};
use phoenix_graptor::{
    build_graph_delta_from_snapshot, load_graph_snapshot_with_candidate_graph, load_session_state,
    BorrowedIngestDocument, BorrowedIngestRequest, GraptorGraph, PhoenixGraptor,
};
use phoenix_invarant_v2::PhoenixInvarantV2;
use phoenix_lex::{indexed_spans_from_store, LexConfig, LexIndex};
use phoenix_om::{OmEngine, OmNativeBridge};
use phoenix_runtime_native::PhoenixRuntimeNative;
use phoenix_semantic_v2::{scope_storage_key, DocumentManifest, DocumentRevisionRef, SessionArchive};
use phoenix_scanner::PhoenixScanner;
pub use phoenix_store_cozo::SnapshotPartition;
use phoenix_store_cozo::{
    schema::{CONTENT_SNAPSHOT_RELATIONS, DERIVED_SNAPSHOT_RELATIONS},
    CompactRow, CompactRowView, PhoenixCozoStore, SnapshotEnvelope, StoreConfig, StoreError,
};
use phoenix_store_native::{
    GraphCheckpointData, PhoenixArchiveStoreV2, PhoenixBundleStoreV2,
    NativeSemanticDocumentVectorRecord, NativeSemanticNodeVectorRecord,
    PhoenixGraphKernelStoreV2, PhoenixNativeRowStore, PhoenixSemanticIndexStore,
};
use phoenix_store_lmdb::PhoenixLmdbStore;
use phoenix_structure::PhoenixStructure;
use phoenix_triverse_v2::PhoenixTriverseV2;
use phoenix_types::{
    ChatPlannerModelResponse, ChatRunEvent, ChatRuntimeConfig, CommitId, CommitRequest,
    CommitResult, CreateSessionRequest, Diagnostic, DocumentId, EntityCard, FolderSchema,
    GraphDeltaRequest, GraphDeltaResult, IndexedSpan, IngestRequest, IngestResult,
    LexicalField, LexicalSearchResult, NetworkInstance,
    OmPendingAction, OmRecord, OmReflectorModelResponse, OmReflectorToolResult, QueryRequest,
    QueryResult, RebuildRequest, RebuildResult, RelationCount, RunOptions, RuntimeConfig,
    RuntimeInitResult, RuntimeTarget, SavedNetworkView, ScanArtifact, ScanRequest, ScopeKey,
    SessionId, SessionRecord, SessionState, SessionStats, SnapshotDto, SpanHit,
    StoreCommandRequest, StoreCommandResult,
    StructureArtifact, StructureRequest, Thread, ThreadMessage, ToolResultSubmission,
};
use planner::{list_run_artifacts, set_artifact_pinned, ChatPlannerRunner};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
pub use view::{
    AnalyzeTextRequestView, IngestDocumentView, IngestRequestView, QueryRequestView,
    ScanRequestView, ScopeKeyView, StructureRequestView,
};

pub struct PhoenixRuntime {
    pub config: RuntimeConfig,
    pub store: MaybeCozoStore,
    v2_store: Option<PhoenixLmdbStore>,
    pub scanner: PhoenixScanner,
    pub structure: PhoenixStructure,
    pub graptor: PhoenixGraptor,
    pub invarant_v2: PhoenixInvarantV2,
    pub native_runtime: PhoenixRuntimeNative,
    pub om_engine: OmEngine,
    pub om_bridge: OmNativeBridge,
    pub chat: PhoenixChat,
    pub planner: ChatPlannerRunner,
    pub lex: RefCell<Option<LexIndex>>,
    native_scope_lex: RefCell<HashMap<String, NativeScopeLexCacheEntry>>,
    pub gldr: PhoenixGldr,
    pub triverse_v2: PhoenixTriverseV2,
}

static NATIVE_STORE_INSTANCE_COUNTER: AtomicU64 = AtomicU64::new(0);
const NATIVE_RUNTIME_SCHEMA_VERSION: &str = "phoenix.native.v2";
const NATIVE_SCOPE_LEX_CACHE_LIMIT: usize = 16;
const NATIVE_SCOPE_QGRAM_MIN_SPANS: usize = 32;

const STORE_API_VERSION: u32 = 1;
const RUNTIME_CAPABILITIES: &[&str] = &[
    "note:list",
    "note:get",
    "note:listByIds",
    "persistence:applyWalBatch",
    "persistence:clearDerived",
    "session:close",
];
const DERIVED_EPHEMERA_RELATIONS: &[&str] = &[
    "phoenix_sessions",
    "phoenix_commits",
    "phoenix_ingest_log",
    "phoenix_query_log",
];
const NATIVE_CANDIDATE_GRAPH_NAMESPACE: &str = "phoenix.graph.candidate";
const NATIVE_GRAPH_JOURNAL_COMPACTION_THRESHOLD: usize = 32;
#[derive(Debug, Deserialize)]
struct PersistenceWalBatchRequest {
    records: Vec<PersistenceWalRecord>,
}

#[derive(Debug, Deserialize)]
struct PersistenceWalRecord {
    seq: u64,
    command: String,
    payload: Value,
    partition: String,
    #[serde(rename = "writtenAt")]
    written_at: Option<u64>,
}

#[derive(Clone, Debug)]
struct NativeScopeLexCacheEntry {
    generation: u64,
    lex: LexIndex,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticDocumentVectorUpsertRow {
    document_id: String,
    values: Vec<f32>,
    leaf_count: usize,
    #[serde(default)]
    evidence_refs: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticNodeVectorUpsertRow {
    node_id: String,
    node_kind: String,
    document_id: Option<String>,
    narrative_id: Option<String>,
    folder_id: Option<String>,
    values: Vec<f32>,
    #[serde(default)]
    evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticCandidatePrototypeInput {
    node_id: String,
    node_kind: String,
    document_id: Option<String>,
    narrative_id: Option<String>,
    folder_id: Option<String>,
    text: String,
    evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticNliJudgmentInput {
    judgment_id: String,
    group_id: String,
    source_id: String,
    target_id: String,
    edge_type: String,
    direction: String,
    premise: String,
    hypothesis: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SemanticNliJudgmentResultRow {
    judgment_id: String,
    group_id: String,
    source_id: String,
    target_id: String,
    edge_type: String,
    direction: String,
    premise: String,
    hypothesis: String,
    entailment: f64,
    neutral: f64,
    contradiction: f64,
    predicted_label: String,
    confidence: f64,
}

#[derive(Clone, Debug, Default)]
struct Phase2NliJudgmentAggregate {
    group_id: String,
    judgments: Vec<SemanticNliJudgmentResultRow>,
}

#[derive(Clone, Debug, Default)]
struct Phase2CandidateEdgeRecord {
    source_id: String,
    target_id: String,
    edge_type: String,
    document_id: Option<String>,
    base_score: f64,
}

#[derive(Clone, Debug, Default)]
struct Phase2NliDecision {
    accepted: bool,
    threshold: f64,
    entailment: f64,
    neutral: f64,
    contradiction: f64,
    nli_score: f64,
    final_score: f64,
}

#[derive(Clone, Debug, Default)]
struct StoredSemanticNodeVector {
    node_id: String,
    node_kind: String,
    document_id: Option<String>,
    narrative_id: Option<String>,
    folder_id: Option<String>,
    values: Vec<f32>,
    evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct StoredSemanticDocumentVector {
    values: Vec<f32>,
    evidence_refs: Vec<String>,
}

#[derive(Default)]
#[allow(dead_code)]
struct NativeGraphEnrichmentStats {
    thread_count: usize,
    vertex_count: usize,
    edge_count: usize,
    removed_vertex_count: usize,
    removed_edge_count: usize,
}

#[derive(Clone, Debug, Default)]
struct Phase2LeafContext {
    document_id: String,
    narrative_id: Option<String>,
    folder_id: Option<String>,
    text: String,
}

pub struct MaybeCozoStore {
    inner: Option<PhoenixCozoStore>,
}

impl MaybeCozoStore {
    fn open(
        enabled: bool,
        config: &RuntimeConfig,
        storage_path: Option<PathBuf>,
    ) -> Result<Self, StoreError> {
        let inner = if enabled {
            Some(PhoenixCozoStore::open(StoreConfig {
                mode: config.storage.clone(),
                path: storage_path,
            })?)
        } else {
            None
        };
        Ok(Self { inner })
    }

}

impl Deref for MaybeCozoStore {
    type Target = PhoenixCozoStore;

    fn deref(&self) -> &Self::Target {
        self.inner
            .as_ref()
            .expect("PhoenixCozoStore is unavailable on the native runtime path")
    }
}

impl PhoenixRuntime {
    pub fn new(config: RuntimeConfig) -> Result<Self, StoreError> {
        Self::open(config, None)
    }

    pub fn open(config: RuntimeConfig, storage_path: Option<PathBuf>) -> Result<Self, StoreError> {
        let use_native_graph = config.target == RuntimeTarget::Native;
        let v2_store = if use_native_graph {
            let store = PhoenixLmdbStore::open(native_lmdb_store_path(storage_path.as_deref()))?;
            store.init_schema()?;
            store.init_archive_schema()?;
            store.init_bundle_schema()?;
            store.init_graph_kernel_schema()?;
            Some(store)
        } else {
            None
        };
        let store = MaybeCozoStore::open(!use_native_graph, &config, storage_path)?;
        Ok(Self {
            config,
            store,
            v2_store,
            scanner: PhoenixScanner::default(),
            structure: PhoenixStructure::default(),
            graptor: PhoenixGraptor::default(),
            invarant_v2: PhoenixInvarantV2::default(),
            native_runtime: PhoenixRuntimeNative::default(),
            om_engine: OmEngine::default(),
            om_bridge: OmNativeBridge::default(),
            chat: PhoenixChat::default(),
            planner: ChatPlannerRunner::default(),
            lex: RefCell::new(None),
            native_scope_lex: RefCell::new(HashMap::new()),
            gldr: PhoenixGldr::default(),
            triverse_v2: PhoenixTriverseV2::default(),
        })
    }

    pub fn init(&self) -> Result<RuntimeInitResult, StoreError> {
        if self.native_graph_enabled() {
            self.v2_store()
                .ok_or_else(|| StoreError::Init("missing native LMDB store".to_owned()))?
                .init_schema()?;
        } else {
            self.store.init_schema()?;
        }
        if let Some(store) = self.v2_store() {
            store.init_archive_schema()?;
            store.init_bundle_schema()?;
            store.init_graph_kernel_schema()?;
        }
        self.rebuild_lex_index()?;
        self.ensure_native_graph_ready()?;
        let relation_counts = self.relation_counts()?;
        Ok(RuntimeInitResult {
            ready: true,
            schema_version: if self.native_graph_enabled() {
                self.v2_store()
                    .ok_or_else(|| StoreError::Init("missing native LMDB store".to_owned()))?
                    .schema_version()
                    .to_owned()
            } else {
                self.store.schema_version().to_owned()
            },
            relation_count: relation_counts.len(),
            relation_counts,
            diagnostics: Vec::new(),
        })
    }

    fn native_graph_enabled(&self) -> bool {
        self.config.target == RuntimeTarget::Native
    }

    fn v2_store(&self) -> Option<&PhoenixLmdbStore> {
        self.v2_store.as_ref()
    }

    fn native_store(&self) -> Result<&PhoenixLmdbStore, StoreError> {
        self.v2_store()
            .ok_or_else(|| StoreError::Query("missing native LMDB store".to_owned()))
    }

    fn native_unsupported(&self, feature: &str) -> StoreError {
        StoreError::Query(format!(
            "{feature} is unavailable on the native runtime path"
        ))
    }

    fn legacy_store(&self, feature: &str) -> Result<&PhoenixCozoStore, StoreError> {
        if self.native_graph_enabled() {
            Err(self.native_unsupported(feature))
        } else {
            self.store
                .inner
                .as_ref()
                .ok_or_else(|| self.native_unsupported(feature))
        }
    }

    fn replace_native_relation_rows_with_keys(
        &self,
        relation: &str,
        rows: &[Value],
        key_fields: &[&str],
    ) -> Result<(), StoreError> {
        let store = self.native_store()?;
        let mut existing = store.fetch_rows(relation)?;
        existing.retain(|existing_row| {
            !rows
                .iter()
                .any(|candidate| relation_rows_match_keys(existing_row, candidate, key_fields))
        });
        existing.extend(rows.iter().cloned());
        store.replace_relation_rows(relation, &existing)
    }

    fn put_relation_row(&self, relation: &str, row: Value) -> Result<(), StoreError> {
        if self.native_graph_enabled() {
            self.native_store()?.put_row(relation, row)
        } else {
            self.store.put_row(relation, row)
        }
    }

    fn fetch_relation_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        if self.native_graph_enabled() {
            self.native_store()?.fetch_rows(relation)
        } else {
            self.store.fetch_rows(relation)
        }
    }

    fn delete_relation_rows(&self, relation: &str, rows: &[Value]) -> Result<usize, StoreError> {
        if self.native_graph_enabled() {
            self.v2_store()
                .ok_or_else(|| StoreError::Query("missing native LMDB store".to_owned()))?
                .delete_rows(relation, rows)
        } else {
            let existing_rows = self.store.fetch_rows(relation)?;
            let existing_compact = self.store.fetch_compact_rows(relation)?;
            let matched = existing_rows
                .iter()
                .zip(existing_compact.into_iter())
                .filter_map(|(existing, compact)| {
                    rows.iter().any(|row| row == existing).then_some(compact)
                })
                .collect::<Vec<_>>();
            let deleted = matched.len();
            self.store.delete_key_rows(relation, &matched)?;
            Ok(deleted)
        }
    }

    fn sync_relation_from_cozo_to_native(&self, _relation: &str) -> Result<(), StoreError> {
        Ok(())
    }

    #[allow(dead_code)]
    fn bootstrap_or_hydrate_native_store(&self) -> Result<(), StoreError> {
        Ok(())
    }

    pub fn relation_counts(&self) -> Result<Vec<RelationCount>, StoreError> {
        if self.native_graph_enabled() {
            Ok(self
                .native_store()?
                .relation_counts()?
                .into_iter()
                .map(|(relation, rows)| RelationCount { relation, rows })
                .collect())
        } else {
            self.store.relation_counts()
        }
    }

    pub fn relation_name_count(&self) -> Result<usize, StoreError> {
        if self.native_graph_enabled() {
            Ok(self.native_store()?.relation_names().len())
        } else {
            Ok(self.store.relation_names().len())
        }
    }

    fn fetch_store_command_relation_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        if self.native_graph_enabled() && phoenix_store_native::is_graph_compat_relation(relation) {
            return self.project_native_graph_relation_rows(relation);
        }
        self.fetch_relation_rows(relation)
    }

    fn project_native_graph_relation_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        let graph = self.graph_snapshot(true)?;
        match relation {
            "graph_vertices" => Ok(project_graph_vertices(&graph)),
            "graph_vertex_labels" => Ok(project_graph_vertex_labels(&graph)),
            "graph_edges" => Ok(project_graph_edges(&graph, GraphLayer::Asserted)),
            "graph_candidate_edges" => Ok(project_graph_edges(&graph, GraphLayer::Candidate)),
            "graph_node_index" => Ok(project_graph_node_index(&graph)),
            "graph_properties" => Ok(project_graph_properties(&graph)),
            other => Err(StoreError::UnknownRelation(other.to_owned())),
        }
    }

    fn persist_native_graph_batch(
        &self,
        batch: &KernelGraphMutationBatch,
        source_revision: Option<String>,
    ) -> Result<(), StoreError> {
        let Some(store) = self.v2_store() else {
            return Ok(());
        };
        let source_revision = match source_revision {
            Some(source_revision) => source_revision,
            None => self.native_graph_rebuild_token()?,
        };
        let generation = store.kernel_current_generation()? + 1;
        let created_at = now_ms();
        store.append_kernel_batch(generation, &source_revision, batch, created_at)?;
        if store.kernel_journal_len()? >= NATIVE_GRAPH_JOURNAL_COMPACTION_THRESHOLD {
            let _ = self.write_native_graph_checkpoint(Some(generation), Some(source_revision))?;
        }
        Ok(())
    }

    fn write_native_graph_checkpoint(
        &self,
        generation: Option<u64>,
        source_revision: Option<String>,
    ) -> Result<Option<GraphCheckpointData>, StoreError> {
        let Some(store) = self.v2_store() else {
            return Ok(None);
        };
        let graph = self.graph_snapshot(true)?;
        let generation = generation.unwrap_or(store.kernel_current_generation()?);
        let source_revision = match source_revision {
            Some(source_revision) => source_revision,
            None => self.native_graph_rebuild_token()?,
        };
        let checkpoint = store.write_kernel_checkpoint(
            generation,
            &source_revision,
            &KernelGraphSnapshot::from(&graph),
        )?;
        Ok(Some(GraphCheckpointData {
            meta: phoenix_store_native::GraphCheckpointMeta {
                checkpoint_id: checkpoint.meta.checkpoint_id,
                generation: checkpoint.meta.generation,
                source_revision: checkpoint.meta.source_revision,
                created_at: checkpoint.meta.created_at,
            },
            asserted_batch: GraphMutationBatch {
                layer: GraphLayer::Asserted,
                scope: GraphMutationScope::Full,
                vertices: checkpoint
                    .snapshot
                    .vertices
                    .iter()
                    .cloned()
                    .map(GraphVertexRecord::from)
                    .collect(),
                edges: checkpoint
                    .snapshot
                    .asserted_edges
                    .iter()
                    .cloned()
                    .map(GraphEdgeRecord::from)
                    .collect(),
            },
            candidate_batch: GraphMutationBatch {
                layer: GraphLayer::Candidate,
                scope: GraphMutationScope::Full,
                vertices: checkpoint
                    .snapshot
                    .vertices
                    .into_iter()
                    .map(GraphVertexRecord::from)
                    .collect(),
                edges: checkpoint
                    .snapshot
                    .candidate_edges
                    .into_iter()
                    .map(GraphEdgeRecord::from)
                    .collect(),
            },
        }))
    }

    fn augment_graph_delta_request_from_journal(
        &self,
        request: &mut GraphDeltaRequest,
    ) -> Result<(), StoreError> {
        let Some(store) = self.v2_store() else {
            return Ok(());
        };
        let Some(since_commit) = request.since_commit.as_ref() else {
            return Ok(());
        };
        let Some(generation) = store.kernel_generation_for_commit(&since_commit.0)? else {
            return Ok(());
        };
        let mut changed = request
            .changed_documents
            .iter()
            .map(|document_id| document_id.0.clone())
            .collect::<BTreeSet<_>>();
        for entry in store.load_kernel_journal_after(generation)? {
            let Some(batch) = entry.batch else {
                continue;
            };
            for vertex in batch.vertices {
                if let Some(document_id) = vertex.document_id {
                    changed.insert(document_id);
                }
            }
            for edge in batch.edges {
                if let Some(document_id) = edge.document_id {
                    changed.insert(document_id);
                }
            }
        }
        request.changed_documents = changed.into_iter().map(DocumentId).collect();
        Ok(())
    }

    fn graph_backend_error(error: GraphBackendError) -> StoreError {
        StoreError::Query(format!("native graph backend: {error}"))
    }

    fn graph_snapshot(&self, include_candidate_graph: bool) -> Result<GraptorGraph, StoreError> {
        if self.native_graph_enabled() {
            self.ensure_native_graph_ready()?;
            self.native_runtime
                .deterministic_kernel
                .snapshot_legacy(include_candidate_graph)
                .map_err(Self::graph_backend_error)
        } else {
            load_graph_snapshot_with_candidate_graph(&self.store, include_candidate_graph)
        }
    }

    fn estimate_session_graph_counts(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        existing_summary: Option<&SessionArchive>,
        new_manifests: &[DocumentManifest],
    ) -> Result<GraphCounts, StoreError> {
        let mut vertex_count = existing_summary
            .map(|summary| summary.graph_vertex_count)
            .unwrap_or_default();
        let mut asserted_edge_count = existing_summary
            .map(|summary| summary.graph_edge_count)
            .unwrap_or_default();
        let existing_refs = existing_summary
            .map(|summary| {
                summary
                    .document_refs
                    .iter()
                    .cloned()
                    .map(|document_ref| (document_ref.document_id.clone(), document_ref))
                    .collect::<HashMap<String, DocumentRevisionRef>>()
            })
            .unwrap_or_default();
        let replacement_manifests = new_manifests
            .iter()
            .map(|manifest| (manifest.document_id.clone(), manifest))
            .collect::<HashMap<String, &DocumentManifest>>();

        for (document_id, manifest) in replacement_manifests {
            if let Some(existing_ref) = existing_refs.get(&document_id) {
                if let Some(existing_manifest) = store.load_document_manifest(existing_ref)? {
                    vertex_count = vertex_count.saturating_sub(existing_manifest.graph_vertex_count);
                    asserted_edge_count =
                        asserted_edge_count.saturating_sub(existing_manifest.graph_edge_count);
                }
            }
            vertex_count = vertex_count.saturating_add(manifest.graph_vertex_count);
            asserted_edge_count = asserted_edge_count.saturating_add(manifest.graph_edge_count);
        }

        Ok(GraphCounts {
            vertex_count,
            asserted_edge_count,
            candidate_edge_count: 0,
        })
    }

    fn native_lexical_search(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        query: &str,
        scope: &ScopeKey,
        limit: usize,
    ) -> Result<LexicalSearchResult, StoreError> {
        let scope_key = scope_storage_key(scope);
        if let Some(cached) = self.native_scope_lex.borrow().get(&scope_key) {
            let mut result = cached.lex.search(query, scope, limit);
            result.diagnostics.push(Diagnostic {
                code: "PX_QUERY_NATIVE_SCOPE_QGRAM_CACHE".to_owned(),
                message: format!(
                    "Native query used cached scope-local qgram search at lexical generation {}.",
                    cached.generation
                ),
            });
            return Ok(result);
        }

        let sidecar = store.load_materialized_scope_lexical(scope)?;
        if sidecar.spans.len() < NATIVE_SCOPE_QGRAM_MIN_SPANS {
            return Ok(native_linear_lexical_search_result(
                &sidecar.spans,
                query,
                scope,
                limit,
            ));
        }

        let lex = LexIndex::build(&sidecar.spans, LexConfig::default());
        let generation = sidecar.generation;
        let mut result = lex.search(query, scope, limit);
        result.diagnostics.push(Diagnostic {
            code: "PX_QUERY_NATIVE_SCOPE_QGRAM".to_owned(),
            message: format!(
                "Native query built a scope-local qgram index at lexical generation {} without rebuilding the global in-memory lex index.",
                generation
            ),
        });

        self.insert_native_scope_lex_cache(scope_key, generation, lex);
        Ok(result)
    }

    fn candidate_edge_rows(&self) -> Result<Vec<Value>, StoreError> {
        if self.native_graph_enabled() {
            self.ensure_native_graph_ready()?;
            let edges = self
                .native_runtime
                .deterministic_kernel
                .candidate_edge_records()
                .map_err(Self::graph_backend_error)?;
            Ok(edges
                .into_iter()
                .map(graph_edge_record_to_row_value)
                .collect::<Vec<_>>())
        } else {
            self.store.fetch_rows("graph_candidate_edges")
        }
    }

    fn native_graph_rebuild_token(&self) -> Result<String, StoreError> {
        let graph_generation = self
            .v2_store()
            .map(|store| store.kernel_current_generation())
            .transpose()?
            .unwrap_or_default();
        Ok(format!("graph:{graph_generation}"))
    }

    #[allow(dead_code)]
    fn list_session_ids(&self) -> Result<Vec<SessionId>, StoreError> {
        Ok(self
            .fetch_relation_rows("phoenix_sessions")?
            .into_iter()
            .filter_map(|row| {
                row.get("session_id")
                    .and_then(Value::as_str)
                    .map(|session_id| SessionId(session_id.to_owned()))
            })
            .collect())
    }

    fn rebuild_native_graph(&self, rebuild_token: String) -> Result<(), StoreError> {
        if let Some(store) = self.v2_store() {
            if let Some(checkpoint) = store.load_kernel_checkpoint()? {
                let journal = store.load_kernel_journal_after(checkpoint.meta.generation)?;
                let mut batches = vec![
                    KernelGraphMutationBatch {
                        layer: KernelGraphLayer::Asserted,
                        scope: KernelMutationScope::Full,
                        recorded_at: None,
                        vertices: checkpoint.snapshot.vertices.clone(),
                        edges: checkpoint.snapshot.asserted_edges.clone(),
                    },
                    KernelGraphMutationBatch {
                        layer: KernelGraphLayer::Candidate,
                        scope: KernelMutationScope::Full,
                        recorded_at: None,
                        vertices: Vec::new(),
                        edges: checkpoint.snapshot.candidate_edges,
                    },
                ];
                batches.extend(journal.into_iter().filter_map(|entry| entry.batch));
                self.native_runtime
                    .deterministic_kernel
                    .rebuild_from_kernel_batches(batches, Some(rebuild_token))
                    .map_err(Self::graph_backend_error)?;
                return Ok(());
            }
            let batches = store
                .load_kernel_journal_after(0)?
                .into_iter()
                .filter_map(|entry| entry.batch)
                .collect::<Vec<_>>();
            self.native_runtime
                .deterministic_kernel
                .rebuild_from_kernel_batches(batches, Some(rebuild_token))
                .map_err(Self::graph_backend_error)?;
            return Ok(());
        }
        self.native_runtime
            .deterministic_kernel
            .rebuild_from_kernel_batches(Vec::new(), Some(rebuild_token))
            .map_err(Self::graph_backend_error)
    }

    fn ensure_native_graph_ready(&self) -> Result<(), StoreError> {
        if !self.native_graph_enabled() {
            return Ok(());
        }
        let rebuild_token = self.native_graph_rebuild_token()?;
        let current_token = self.native_runtime.deterministic_kernel.rebuild_token();
        if runtime_ingest_progress_enabled() {
            eprintln!(
                "[runtime-graph] ensure_ready current_token={:?} rebuild_token={}",
                current_token,
                rebuild_token
            );
        }
        if current_token.as_deref() == Some(rebuild_token.as_str()) {
            if runtime_ingest_progress_enabled() {
                eprintln!("[runtime-graph] ensure_ready status=up_to_date");
            }
            return Ok(());
        }
        if runtime_ingest_progress_enabled() {
            eprintln!("[runtime-graph] ensure_ready status=rebuild");
        }
        self.rebuild_native_graph(rebuild_token)
    }

    fn refresh_native_graph_rebuild_token(&self) -> Result<(), StoreError> {
        if !self.native_graph_enabled() {
            return Ok(());
        }
        let rebuild_token = self.native_graph_rebuild_token()?;
        self.native_runtime
            .deterministic_kernel
            .set_rebuild_token(Some(rebuild_token));
        Ok(())
    }

    fn persist_native_candidate_scope_batches(
        &self,
        batches: Vec<GraphMutationBatch>,
    ) -> Result<(), StoreError> {
        if !self.native_graph_enabled() || batches.is_empty() {
            return Ok(());
        }

        let touched_scope_keys = batches
            .iter()
            .filter_map(|batch| match &batch.scope {
                GraphMutationScope::Candidate { scope_key } => Some(scope_key.clone()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        if touched_scope_keys.is_empty() {
            return Ok(());
        }

        let mut existing_rows = HashMap::<String, Value>::new();
        for row in self.fetch_relation_rows("scoped_definitions")? {
            if row.get("namespace").and_then(Value::as_str)
                != Some(NATIVE_CANDIDATE_GRAPH_NAMESPACE)
            {
                continue;
            }
            let Some(definition_key) = row.get("definition_key").and_then(Value::as_str) else {
                continue;
            };
            if touched_scope_keys.contains(definition_key) {
                existing_rows.insert(definition_key.to_owned(), row);
            }
        }

        let updated_at = now_ms();
        let mut delete_rows = Vec::new();
        let mut stored_batches = Vec::new();

        for batch in batches {
            let GraphMutationScope::Candidate { scope_key } = &batch.scope else {
                return Err(StoreError::Query(
                    "native candidate persistence received a non-candidate scope".to_owned(),
                ));
            };
            let scope_key = scope_key.clone();
            if batch.edges.is_empty() {
                if let Some(row) = existing_rows.remove(&scope_key) {
                    delete_rows.push(row);
                }
                stored_batches.push(batch);
                continue;
            }

            let narrative_id = batch
                .edges
                .iter()
                .find_map(|edge| edge.narrative_id.clone());
            let mut payload = serde_json::to_value(&batch).map_err(|error| {
                StoreError::Query(format!(
                    "failed to serialize native candidate graph batch for {scope_key}: {error}"
                ))
            })?;
            if let Some(object) = payload.as_object_mut() {
                object.insert("updatedAt".to_owned(), json!(updated_at));
            }
            self.put_relation_row(
                "scoped_definitions",
                json!({
                    "id": native_candidate_definition_row_id(&scope_key),
                    "narrative_id": narrative_id,
                    "namespace": NATIVE_CANDIDATE_GRAPH_NAMESPACE,
                    "definition_key": scope_key,
                    "payload": payload,
                    "created_at": updated_at,
                    "updated_at": updated_at,
                }),
            )?;
            stored_batches.push(batch);
        }

        if !delete_rows.is_empty() {
            self.delete_relation_rows("scoped_definitions", &delete_rows)?;
        }

        let apply_result = (|| {
            for batch in &stored_batches {
                self.native_runtime
                    .deterministic_kernel
                    .apply_compat_batch(batch.clone())
                    .map_err(Self::graph_backend_error)?;
            }
            Ok(())
        })();
        if let Err(error) = apply_result {
            self.native_runtime.deterministic_kernel.invalidate();
            return Err(error);
        }

        let source_revision = self.native_graph_rebuild_token()?;
        for batch in &stored_batches {
            let kernel_batch = KernelGraphMutationBatch::from(batch.clone());
            self.persist_native_graph_batch(&kernel_batch, Some(source_revision.clone()))?;
        }

        self.refresh_native_graph_rebuild_token()
    }

    pub fn create_session(
        &self,
        request: CreateSessionRequest,
    ) -> Result<SessionRecord, StoreError> {
        let now = now_ms();
        let session_id = request
            .session_id
            .unwrap_or_else(|| SessionId(format!("session-{}", now)));
        let record = SessionRecord {
            session_id,
            label: request.label,
            scope: request.scope,
            status: "active".to_owned(),
            revision: 0,
            created_at: now,
            updated_at: now,
        };

        self.put_relation_row(
            "phoenix_sessions",
            serde_json::json!({
                "session_id": record.session_id.0,
                "label": record.label,
                "world_id": record.scope.world_id,
                "narrative_id": record.scope.narrative_id,
                "folder_id": record.scope.folder_id,
                "folder_path": record.scope.folder_path,
                "status": record.status,
                "revision": record.revision,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }),
        )?;

        Ok(record)
    }

    pub fn commit(&self, request: CommitRequest) -> Result<CommitResult, StoreError> {
        let mut session = self.load_session(&request.session_id)?;
        let committed_at = now_ms();
        session.revision += 1;
        session.updated_at = committed_at;
        let session_row = serde_json::json!({
            "session_id": session.session_id.0,
            "label": session.label,
            "world_id": session.scope.world_id,
            "narrative_id": session.scope.narrative_id,
            "folder_id": session.scope.folder_id,
            "folder_path": session.scope.folder_path,
            "status": session.status,
            "revision": session.revision,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        });

        let commit_id = CommitId(format!(
            "commit-{}-{}",
            session.session_id.0, session.revision
        ));
        let commit_row = serde_json::json!({
            "commit_id": commit_id.0,
            "session_id": session.session_id.0,
            "reason": request.reason,
            "revision": session.revision,
            "committed_at": committed_at,
        });

        if let Some(store) = self.v2_store() {
            let generation = store.kernel_current_generation()?;
            let source_revision = format!("graph:{generation}");
            store.commit_native_session(
                &session_row,
                &commit_row,
                generation,
                &source_revision,
                &commit_id.0,
                committed_at,
            )?;
        } else {
            self.put_relation_row("phoenix_sessions", session_row)?;
            self.put_relation_row("phoenix_commits", commit_row)?;
        }

        Ok(CommitResult {
            session_id: session.session_id,
            commit_id,
            revision: session.revision,
            committed_at,
            relation_counts: if self.native_graph_enabled() {
                Vec::new()
            } else {
                self.relation_counts()?
            },
            diagnostics: Vec::new(),
        })
    }

    pub fn rebuild(&self, _request: RebuildRequest) -> Result<RebuildResult, StoreError> {
        let rebuilt_at = now_ms();
        let dirty_scope_count = if let Some(store) = self.v2_store() {
            if runtime_ingest_progress_enabled() {
                eprintln!("[runtime-ingest] subphase=rebuild_dirty_scope_sidecars start");
            }
            let started = std::time::Instant::now();
            let rebuilt = store.rebuild_dirty_scope_sidecars(rebuilt_at)?;
            if runtime_ingest_progress_enabled() {
                eprintln!(
                    "[runtime-ingest] subphase=rebuild_dirty_scope_sidecars finish dirty_scopes={} wall_ms={}",
                    rebuilt,
                    started.elapsed().as_millis()
                );
            }
            rebuilt
        } else {
            0
        };
        let span_count = if self.native_graph_enabled() {
            self.invalidate_lex_caches();
            self.v2_store()
                .ok_or_else(|| StoreError::Query("missing native LMDB store".to_owned()))?
                .load_lex_spans(None)?
                .len()
        } else {
            self.rebuild_lex_index()?
        };
        Ok(RebuildResult {
            rebuilt_at,
            relation_counts: if self.native_graph_enabled() {
                Vec::new()
            } else {
                self.relation_counts()?
            },
            diagnostics: vec![Diagnostic {
                code: "PX_REBUILD_LEX".to_owned(),
                message: format!(
                    "Rebuilt lexical sidecars from {span_count} canonical spans across {dirty_scope_count} dirty scopes."
                ),
            }],
        })
    }

    pub fn ingest(&self, request: IngestRequest) -> Result<IngestResult, StoreError> {
        self.ingest_view(IngestRequestView::from(&request))
    }

    pub fn ingest_view(&self, request: IngestRequestView<'_>) -> Result<IngestResult, StoreError> {
        let created_at = now_ms();
        let request_summary = serde_json::json!({
            "sessionId": request.session_id.as_ref().map(|value| value.0.clone()),
            "documentCount": request.documents.len(),
            "documentIds": request.documents.iter().map(|document| document.document_id.0.clone()).collect::<Vec<_>>(),
            "titles": request.documents.iter().map(|document| document.title.to_owned()).collect::<Vec<_>>(),
            "commit": request.commit,
        });
        self.put_relation_row(
            "phoenix_ingest_log",
            serde_json::json!({
                "id": format!("ingest-{}", created_at),
                "session_id": request.session_id.as_ref().map(|value| value.0.clone()),
                "document_count": request.documents.len(),
                "commit_requested": request.commit,
                "request_json": request_summary,
                "created_at": created_at,
            }),
        )?;
        let documents = request
            .documents
            .iter()
            .map(|document| BorrowedIngestDocument {
                document_id: document.document_id.clone(),
                note_id: document.note_id.clone(),
                title: document.title,
                text: document.text,
                scope: document.scope.to_owned(),
            })
            .collect::<Vec<_>>();
        let borrowed_request = BorrowedIngestRequest {
            session_id: request.session_id.clone(),
            documents: &documents,
        };
        let mut ingest = if self.native_graph_enabled() {
            let store = self
                .v2_store()
                .ok_or_else(|| StoreError::Query("missing native LMDB store".to_owned()))?;
            let base_revision = request
                .session_id
                .as_ref()
                .map(|session_id| self.load_session(session_id).map(|session| session.revision))
                .transpose()?
                .unwrap_or_default();
            let owned_documents = request
                .documents
                .iter()
                .map(|document| phoenix_types::IngestDocument {
                    document_id: document.document_id.clone(),
                    note_id: document.note_id.clone(),
                    title: document.title.to_owned(),
                    text: document.text.to_owned(),
                    scope: document.scope.to_owned(),
                })
                .collect::<Vec<_>>();
            let (mut ingest, artifacts) = self
                .native_runtime
                .ingest_documents_native(
                    store,
                    request.session_id.as_ref(),
                    &owned_documents,
                    base_revision,
                    created_at,
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
            if !artifacts.kernel_batches.is_empty() {
                let source_revision = self.native_graph_rebuild_token()?;
                let apply_result = (|| {
                    for batch in &artifacts.kernel_batches {
                        self.native_runtime
                            .deterministic_kernel
                            .apply_batch(batch.clone())
                            .map_err(Self::graph_backend_error)?;
                    }
                    Ok(())
                })();
                if let Err(error) = apply_result {
                    self.native_runtime.deterministic_kernel.invalidate();
                    return Err(error);
                }
                for batch in &artifacts.kernel_batches {
                    self.persist_native_graph_batch(batch, Some(source_revision.clone()))?;
                }
            }
            if request.commit {
                if let Some(session_id) = request.session_id.clone() {
                    let commit = self.commit(CommitRequest {
                        session_id,
                        reason: Some("invarant-v3-ingest".to_owned()),
                    })?;
                    ingest.diagnostics.extend(commit.diagnostics);
                }
            }
            if let Some(session_id) = request.session_id.as_ref() {
                let persist_started = std::time::Instant::now();
                if runtime_ingest_progress_enabled() {
                    eprintln!(
                        "[runtime-ingest] subphase=persist_session_summary start session_id={}",
                        session_id.0
                    );
                }
                let step_started = std::time::Instant::now();
                let existing_summary = self
                    .native_runtime
                    .load_latest_session_summary(store, session_id)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                let graph_counts = self.estimate_session_graph_counts(
                    store,
                    existing_summary.as_ref(),
                    &artifacts.document_manifests,
                )?;
                if runtime_ingest_progress_enabled() {
                    eprintln!(
                        "[runtime-ingest] persist_session_summary_step=estimate_session_graph_counts wall_ms={} vertices={} asserted_edges={} candidate_edges={}",
                        step_started.elapsed().as_millis(),
                        graph_counts.vertex_count,
                        graph_counts.asserted_edge_count,
                        graph_counts.candidate_edge_count,
                    );
                }
                let step_started = std::time::Instant::now();
                let session = self.load_session(session_id)?;
                if runtime_ingest_progress_enabled() {
                    eprintln!(
                        "[runtime-ingest] persist_session_summary_step=load_session wall_ms={} revision={} updated_at={}",
                        step_started.elapsed().as_millis(),
                        session.revision,
                        session.updated_at,
                    );
                }
                let step_started = std::time::Instant::now();
                let summary = self.native_runtime.merge_session_summary(
                    existing_summary,
                    session_id.clone(),
                    artifacts.session_documents,
                    artifacts.document_refs,
                    artifacts.discovery_candidate_count,
                    artifacts.span_count,
                    graph_counts.vertex_count,
                    graph_counts.asserted_edge_count + graph_counts.candidate_edge_count,
                    session.updated_at.max(created_at),
                );
                if runtime_ingest_progress_enabled() {
                    eprintln!(
                        "[runtime-ingest] persist_session_summary_step=merge_session_summary wall_ms={} documents={} document_refs={}",
                        step_started.elapsed().as_millis(),
                        summary.documents.len(),
                        summary.document_refs.len(),
                    );
                }
                let step_started = std::time::Instant::now();
                self.native_runtime
                    .persist_session_summary(
                        store,
                        &summary,
                        session.revision,
                        session.updated_at.max(created_at),
                    )
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                if runtime_ingest_progress_enabled() {
                    eprintln!(
                        "[runtime-ingest] persist_session_summary_step=persist_session_summary_native wall_ms={}",
                        step_started.elapsed().as_millis(),
                    );
                }
                if runtime_ingest_progress_enabled() {
                    eprintln!(
                        "[runtime-ingest] subphase=persist_session_summary finish documents={} wall_ms={}",
                        summary.documents.len(),
                        persist_started.elapsed().as_millis()
                    );
                }
            }
            ingest
        } else {
            self.graptor.ingest_view(
                &self.store,
                &self.scanner,
                &self.structure,
                &borrowed_request,
            )?
        };
        let mut diagnostics = ingest.diagnostics.clone();
        diagnostics.push(Diagnostic {
            code: if self.native_graph_enabled() {
                "PX_INGEST_INVARANT_V2".to_owned()
            } else {
                "PX_INGEST_GRAPTOR".to_owned()
            },
            message: if self.native_graph_enabled() {
                "Phoenix Invarant V2 ingested native bundles, semantic state, and kernel graph mutations.".to_owned()
            } else {
                "Phoenix Graptor ingested canonical chunk and graph facts.".to_owned()
            },
        });

        self.invalidate_lex_caches();
        if runtime_ingest_progress_enabled() {
            eprintln!("[runtime-ingest] phase=final_relation_counts");
        }
        ingest.diagnostics = diagnostics;
        ingest.warning_count = ingest.diagnostics.len();
        ingest.relation_counts = self.relation_counts()?;
        Ok(ingest)
    }

    #[allow(dead_code)]
    fn refresh_asserted_native_graph(
        &self,
        session_id: &SessionId,
    ) -> Result<NativeGraphEnrichmentStats, StoreError> {
        let session = self.load_session(session_id)?;
        let session_state = if self.native_graph_enabled() {
            self.session_state(session_id)?
        } else {
            load_session_state(&self.store, session_id)?
        };
        let graph = self.graph_snapshot(false)?;
        let message_document_links = phase1_message_document_links(&session_state, &graph);
        let singleton_document_id = (session_state.documents.len() == 1)
            .then(|| session_state.documents[0].document_id.0.clone());
        let mut threads = self
            .chat
            .list_threads(
                &self.store,
                session
                    .scope
                    .world_id
                    .as_deref()
                    .filter(|value| !value.is_empty()),
            )?
            .into_iter()
            .filter(|thread| phase1_thread_matches_session(&session, thread))
            .collect::<Vec<_>>();
        threads.sort_by(|left, right| left.id.cmp(&right.id));

        let mut stats = NativeGraphEnrichmentStats::default();
        let mut vertex_ids = BTreeSet::new();
        let mut edge_pairs = BTreeSet::new();
        let mut vertex_rows = Vec::new();
        let mut label_rows = Vec::new();
        let mut edge_rows = Vec::new();

        for thread in threads {
            let mut messages = self.chat.list_messages(&self.store, &thread.id.0)?;
            if messages.is_empty() {
                continue;
            }
            messages.sort_by(|left, right| {
                left.created_at
                    .cmp(&right.created_at)
                    .then_with(|| left.id.cmp(&right.id))
            });
            stats.thread_count += 1;

            let mut events =
                self.chat
                    .list_run_events_for_thread(&self.store, &thread.id.0, 256)?;
            events.sort_by(|left, right| {
                left.created_at
                    .cmp(&right.created_at)
                    .then_with(|| left.id.cmp(&right.id))
            });
            let om_record = self.load_latest_om_record_for_thread(&thread.id.0)?;
            let mut message_links = HashMap::<String, Vec<String>>::new();

            for message in &messages {
                let document_links = phase1_document_links_for_message(
                    &message.id,
                    &message_document_links,
                    singleton_document_id.as_deref(),
                );
                message_links.insert(message.id.clone(), document_links.clone());

                let turn_id = phase1_turn_vertex_id(&message.id);
                let turn_document_id =
                    (document_links.len() == 1).then(|| document_links[0].clone());
                let turn_label = phase1_turn_label(message);
                let mut turn_attributes = Map::new();
                turn_attributes.insert("sessionId".to_owned(), json!(session_id.0));
                turn_attributes.insert("threadId".to_owned(), json!(thread.id.0));
                turn_attributes.insert("messageId".to_owned(), json!(message.id));
                turn_attributes.insert("role".to_owned(), json!(message.role));
                turn_attributes.insert("createdAt".to_owned(), json!(message.created_at));
                turn_attributes.insert("documentIds".to_owned(), json!(document_links.clone()));
                phase1_push_vertex(
                    &mut vertex_ids,
                    &mut vertex_rows,
                    &mut label_rows,
                    phase1_vertex_row(
                        &turn_id,
                        "turn",
                        &turn_label,
                        turn_document_id.as_deref(),
                        phase1_narrative_id(&thread, message, &session).as_deref(),
                        turn_attributes,
                        vec![format!("thread_message:{}", message.id)],
                    ),
                );

                let agent_id = phase1_agent_vertex_id(session_id, &message.role);
                let mut agent_attributes = Map::new();
                agent_attributes.insert("sessionId".to_owned(), json!(session_id.0));
                agent_attributes.insert("role".to_owned(), json!(message.role));
                phase1_push_vertex(
                    &mut vertex_ids,
                    &mut vertex_rows,
                    &mut label_rows,
                    phase1_vertex_row(
                        &agent_id,
                        "agent",
                        &message.role,
                        None,
                        phase1_narrative_id(&thread, message, &session).as_deref(),
                        agent_attributes,
                        vec![format!("thread_role:{}", message.role)],
                    ),
                );
                phase1_push_edge(
                    &mut edge_pairs,
                    &mut edge_rows,
                    phase1_edge_row(
                        &turn_id,
                        &agent_id,
                        "seen_by",
                        None,
                        phase1_narrative_id(&thread, message, &session).as_deref(),
                        {
                            let mut attributes = Map::new();
                            attributes.insert("sessionId".to_owned(), json!(session_id.0));
                            attributes.insert("threadId".to_owned(), json!(thread.id.0));
                            attributes.insert("messageId".to_owned(), json!(message.id));
                            attributes.insert("role".to_owned(), json!(message.role));
                            attributes
                        },
                        vec![
                            format!("thread_message:{}", message.id),
                            format!("thread_role:{}", message.role),
                        ],
                    ),
                );

                let time_id = phase1_time_vertex_id("message", &message.id);
                phase1_push_vertex(
                    &mut vertex_ids,
                    &mut vertex_rows,
                    &mut label_rows,
                    phase1_time_vertex_row(
                        &time_id,
                        session_id,
                        "message",
                        &message.id,
                        message.created_at,
                        phase1_narrative_id(&thread, message, &session).as_deref(),
                        vec![format!("thread_message:{}", message.id)],
                    ),
                );
                phase1_push_edge(
                    &mut edge_pairs,
                    &mut edge_rows,
                    phase1_edge_row(
                        &turn_id,
                        &time_id,
                        "active_during",
                        turn_document_id.as_deref(),
                        phase1_narrative_id(&thread, message, &session).as_deref(),
                        {
                            let mut attributes = Map::new();
                            attributes.insert("sessionId".to_owned(), json!(session_id.0));
                            attributes.insert("threadId".to_owned(), json!(thread.id.0));
                            attributes.insert("messageId".to_owned(), json!(message.id));
                            attributes.insert("timestampMs".to_owned(), json!(message.created_at));
                            attributes
                        },
                        vec![format!("thread_message:{}", message.id)],
                    ),
                );
                phase1_push_edge(
                    &mut edge_pairs,
                    &mut edge_rows,
                    phase1_edge_row(
                        &time_id,
                        &turn_id,
                        "derived_from",
                        turn_document_id.as_deref(),
                        phase1_narrative_id(&thread, message, &session).as_deref(),
                        {
                            let mut attributes = Map::new();
                            attributes.insert("sessionId".to_owned(), json!(session_id.0));
                            attributes.insert("threadId".to_owned(), json!(thread.id.0));
                            attributes.insert("messageId".to_owned(), json!(message.id));
                            attributes.insert("timestampMs".to_owned(), json!(message.created_at));
                            attributes
                        },
                        vec![format!("thread_message:{}", message.id)],
                    ),
                );

                for document_id in &document_links {
                    phase1_push_edge(
                        &mut edge_pairs,
                        &mut edge_rows,
                        phase1_edge_row(
                            &turn_id,
                            &phase1_document_vertex_id(document_id),
                            "about",
                            Some(document_id.as_str()),
                            phase1_narrative_id(&thread, message, &session).as_deref(),
                            {
                                let mut attributes = Map::new();
                                attributes.insert("sessionId".to_owned(), json!(session_id.0));
                                attributes.insert("threadId".to_owned(), json!(thread.id.0));
                                attributes.insert("messageId".to_owned(), json!(message.id));
                                attributes.insert("documentId".to_owned(), json!(document_id));
                                attributes
                            },
                            vec![
                                format!("thread_message:{}", message.id),
                                format!("document:{document_id}"),
                            ],
                        ),
                    );
                }
            }

            for event in &events {
                let state_id = phase1_state_vertex_id(&event.id);
                let anchor_message =
                    phase1_latest_message_at_or_before(&messages, event.created_at);
                let event_document_id = anchor_message
                    .and_then(|message| message_links.get(&message.id))
                    .filter(|document_ids| document_ids.len() == 1)
                    .map(|document_ids| document_ids[0].clone());
                let mut state_attributes = Map::new();
                state_attributes.insert("sessionId".to_owned(), json!(session_id.0));
                state_attributes.insert("threadId".to_owned(), json!(thread.id.0));
                state_attributes.insert("runEventId".to_owned(), json!(event.id));
                state_attributes.insert("phase".to_owned(), json!(event.phase));
                state_attributes.insert("kind".to_owned(), json!(event.kind));
                state_attributes.insert("status".to_owned(), json!(event.status.clone()));
                state_attributes.insert("createdAt".to_owned(), json!(event.created_at));
                phase1_push_vertex(
                    &mut vertex_ids,
                    &mut vertex_rows,
                    &mut label_rows,
                    phase1_vertex_row(
                        &state_id,
                        "state",
                        &phase1_state_label(event),
                        event_document_id.as_deref(),
                        phase1_thread_narrative_id(&thread, &session).as_deref(),
                        state_attributes,
                        vec![format!("chat_run_event:{}", event.id)],
                    ),
                );

                let state_time_id = phase1_time_vertex_id("state", &event.id);
                phase1_push_vertex(
                    &mut vertex_ids,
                    &mut vertex_rows,
                    &mut label_rows,
                    phase1_time_vertex_row(
                        &state_time_id,
                        session_id,
                        "state",
                        &event.id,
                        event.created_at,
                        phase1_thread_narrative_id(&thread, &session).as_deref(),
                        vec![format!("chat_run_event:{}", event.id)],
                    ),
                );
                phase1_push_edge(
                    &mut edge_pairs,
                    &mut edge_rows,
                    phase1_edge_row(
                        &state_id,
                        &state_time_id,
                        "active_during",
                        event_document_id.as_deref(),
                        phase1_thread_narrative_id(&thread, &session).as_deref(),
                        {
                            let mut attributes = Map::new();
                            attributes.insert("sessionId".to_owned(), json!(session_id.0));
                            attributes.insert("threadId".to_owned(), json!(thread.id.0));
                            attributes.insert("runEventId".to_owned(), json!(event.id));
                            attributes.insert("timestampMs".to_owned(), json!(event.created_at));
                            attributes
                        },
                        vec![format!("chat_run_event:{}", event.id)],
                    ),
                );
                phase1_push_edge(
                    &mut edge_pairs,
                    &mut edge_rows,
                    phase1_edge_row(
                        &state_time_id,
                        &state_id,
                        "derived_from",
                        event_document_id.as_deref(),
                        phase1_thread_narrative_id(&thread, &session).as_deref(),
                        {
                            let mut attributes = Map::new();
                            attributes.insert("sessionId".to_owned(), json!(session_id.0));
                            attributes.insert("threadId".to_owned(), json!(thread.id.0));
                            attributes.insert("runEventId".to_owned(), json!(event.id));
                            attributes.insert("timestampMs".to_owned(), json!(event.created_at));
                            attributes
                        },
                        vec![format!("chat_run_event:{}", event.id)],
                    ),
                );

                if let Some(message) = anchor_message {
                    phase1_push_edge(
                        &mut edge_pairs,
                        &mut edge_rows,
                        phase1_edge_row(
                            &state_id,
                            &phase1_turn_vertex_id(&message.id),
                            "depends_on",
                            event_document_id.as_deref(),
                            phase1_thread_narrative_id(&thread, &session).as_deref(),
                            {
                                let mut attributes = Map::new();
                                attributes.insert("sessionId".to_owned(), json!(session_id.0));
                                attributes.insert("threadId".to_owned(), json!(thread.id.0));
                                attributes.insert("runEventId".to_owned(), json!(event.id));
                                attributes.insert("messageId".to_owned(), json!(message.id));
                                attributes
                            },
                            vec![
                                format!("chat_run_event:{}", event.id),
                                format!("thread_message:{}", message.id),
                            ],
                        ),
                    );
                }
            }

            if let Some(record) = om_record
                .as_ref()
                .filter(|record| !record.current_task.trim().is_empty())
            {
                let task_id = phase1_task_vertex_id(&thread.id.0);
                let observed_message = phase1_latest_message_at_or_before(
                    &messages,
                    record.updated_at.max(record.last_observed_at),
                );
                let task_document_links = observed_message
                    .and_then(|message| message_links.get(&message.id).cloned())
                    .unwrap_or_else(|| {
                        singleton_document_id
                            .as_ref()
                            .map(|document_id| vec![document_id.clone()])
                            .unwrap_or_default()
                    });
                let task_document_id =
                    (task_document_links.len() == 1).then(|| task_document_links[0].clone());
                let mut task_attributes = Map::new();
                task_attributes.insert("sessionId".to_owned(), json!(session_id.0));
                task_attributes.insert("threadId".to_owned(), json!(thread.id.0));
                task_attributes.insert("currentTask".to_owned(), json!(record.current_task));
                task_attributes.insert("updatedAt".to_owned(), json!(record.updated_at));
                task_attributes
                    .insert("documentIds".to_owned(), json!(task_document_links.clone()));
                phase1_push_vertex(
                    &mut vertex_ids,
                    &mut vertex_rows,
                    &mut label_rows,
                    phase1_vertex_row(
                        &task_id,
                        "task",
                        record.current_task.trim(),
                        task_document_id.as_deref(),
                        phase1_thread_narrative_id(&thread, &session).as_deref(),
                        task_attributes,
                        vec![format!("om_record:{}", record.thread_id)],
                    ),
                );

                let task_time_id = phase1_time_vertex_id("task", &thread.id.0);
                phase1_push_vertex(
                    &mut vertex_ids,
                    &mut vertex_rows,
                    &mut label_rows,
                    phase1_time_vertex_row(
                        &task_time_id,
                        session_id,
                        "task",
                        &thread.id.0,
                        record.updated_at.max(record.last_observed_at),
                        phase1_thread_narrative_id(&thread, &session).as_deref(),
                        vec![format!("om_record:{}", record.thread_id)],
                    ),
                );
                phase1_push_edge(
                    &mut edge_pairs,
                    &mut edge_rows,
                    phase1_edge_row(
                        &task_id,
                        &task_time_id,
                        "active_during",
                        task_document_id.as_deref(),
                        phase1_thread_narrative_id(&thread, &session).as_deref(),
                        {
                            let mut attributes = Map::new();
                            attributes.insert("sessionId".to_owned(), json!(session_id.0));
                            attributes.insert("threadId".to_owned(), json!(thread.id.0));
                            attributes.insert(
                                "timestampMs".to_owned(),
                                json!(record.updated_at.max(record.last_observed_at)),
                            );
                            attributes
                        },
                        vec![format!("om_record:{}", record.thread_id)],
                    ),
                );
                phase1_push_edge(
                    &mut edge_pairs,
                    &mut edge_rows,
                    phase1_edge_row(
                        &task_time_id,
                        &task_id,
                        "derived_from",
                        task_document_id.as_deref(),
                        phase1_thread_narrative_id(&thread, &session).as_deref(),
                        {
                            let mut attributes = Map::new();
                            attributes.insert("sessionId".to_owned(), json!(session_id.0));
                            attributes.insert("threadId".to_owned(), json!(thread.id.0));
                            attributes.insert(
                                "timestampMs".to_owned(),
                                json!(record.updated_at.max(record.last_observed_at)),
                            );
                            attributes
                        },
                        vec![format!("om_record:{}", record.thread_id)],
                    ),
                );

                if let Some(message) = observed_message {
                    phase1_push_edge(
                        &mut edge_pairs,
                        &mut edge_rows,
                        phase1_edge_row(
                            &task_id,
                            &phase1_turn_vertex_id(&message.id),
                            "observed_in",
                            task_document_id.as_deref(),
                            phase1_thread_narrative_id(&thread, &session).as_deref(),
                            {
                                let mut attributes = Map::new();
                                attributes.insert("sessionId".to_owned(), json!(session_id.0));
                                attributes.insert("threadId".to_owned(), json!(thread.id.0));
                                attributes.insert("messageId".to_owned(), json!(message.id));
                                attributes
                            },
                            vec![
                                format!("om_record:{}", record.thread_id),
                                format!("thread_message:{}", message.id),
                            ],
                        ),
                    );
                }

                if let Some(state) = phase1_event_for_task(&events, record.updated_at) {
                    phase1_push_edge(
                        &mut edge_pairs,
                        &mut edge_rows,
                        phase1_edge_row(
                            &task_id,
                            &phase1_state_vertex_id(&state.id),
                            "valid_under",
                            task_document_id.as_deref(),
                            phase1_thread_narrative_id(&thread, &session).as_deref(),
                            {
                                let mut attributes = Map::new();
                                attributes.insert("sessionId".to_owned(), json!(session_id.0));
                                attributes.insert("threadId".to_owned(), json!(thread.id.0));
                                attributes.insert("runEventId".to_owned(), json!(state.id));
                                attributes
                            },
                            vec![
                                format!("om_record:{}", record.thread_id),
                                format!("chat_run_event:{}", state.id),
                            ],
                        ),
                    );
                }

                for document_id in &task_document_links {
                    phase1_push_edge(
                        &mut edge_pairs,
                        &mut edge_rows,
                        phase1_edge_row(
                            &task_id,
                            &phase1_document_vertex_id(document_id),
                            "about",
                            Some(document_id.as_str()),
                            phase1_thread_narrative_id(&thread, &session).as_deref(),
                            {
                                let mut attributes = Map::new();
                                attributes.insert("sessionId".to_owned(), json!(session_id.0));
                                attributes.insert("threadId".to_owned(), json!(thread.id.0));
                                attributes.insert("documentId".to_owned(), json!(document_id));
                                attributes
                            },
                            vec![
                                format!("om_record:{}", record.thread_id),
                                format!("document:{document_id}"),
                            ],
                        ),
                    );
                }
            }
        }

        stats.vertex_count = vertex_rows.len();
        stats.edge_count = edge_rows.len();
        if self.native_graph_enabled() {
            let batch = GraphMutationBatch {
                layer: GraphLayer::Asserted,
                scope: GraphMutationScope::Session {
                    session_id: session_id.0.clone(),
                },
                vertices: vertex_rows
                    .iter()
                    .filter_map(graph_vertex_record_from_row_value)
                    .collect(),
                edges: edge_rows
                    .iter()
                    .filter_map(|row| graph_edge_record_from_row_value(row, GraphLayer::Asserted))
                    .collect(),
            };
            self.native_runtime
                .deterministic_kernel
                .apply_compat_batch(batch.clone())
                .map_err(Self::graph_backend_error)?;
            let kernel_batch = KernelGraphMutationBatch::from(batch);
            self.persist_native_graph_batch(&kernel_batch, None)?;
            return Ok(stats);
        }

        let (removed_vertex_count, removed_edge_count) =
            self.prune_stale_native_graph_rows(session_id, &vertex_ids, &edge_pairs)?;
        stats.removed_vertex_count = removed_vertex_count;
        stats.removed_edge_count = removed_edge_count;
        for row in vertex_rows {
            self.store.put_row("graph_vertices", row)?;
        }
        for row in label_rows {
            self.store.put_row("graph_vertex_labels", row)?;
        }
        for row in edge_rows {
            self.store.put_row("graph_edges", row)?;
        }
        Ok(stats)
    }

    #[allow(dead_code)]
    fn prune_stale_native_graph_rows(
        &self,
        session_id: &SessionId,
        keep_vertex_ids: &BTreeSet<String>,
        keep_edge_pairs: &BTreeSet<(String, String)>,
    ) -> Result<(usize, usize), StoreError> {
        let existing_vertex_rows = self.store.fetch_rows("graph_vertices")?;
        let existing_vertex_compact = self.store.fetch_compact_rows("graph_vertices")?;
        let mut removed_vertex_ids = BTreeSet::new();
        let stale_vertices = existing_vertex_rows
            .iter()
            .zip(existing_vertex_compact)
            .filter_map(|(row, compact)| {
                phase1_native_graph_row_matches_session(row, session_id)
                    .then(|| row.get("id").and_then(Value::as_str).map(str::to_owned))
                    .flatten()
                    .filter(|vertex_id| !keep_vertex_ids.contains(vertex_id))
                    .map(|vertex_id| {
                        removed_vertex_ids.insert(vertex_id);
                        compact
                    })
            })
            .collect::<Vec<_>>();
        let removed_vertex_count = stale_vertices.len();
        if !stale_vertices.is_empty() {
            self.store
                .delete_key_rows("graph_vertices", &stale_vertices)?;
        }

        if !removed_vertex_ids.is_empty() {
            let label_rows = self.store.fetch_rows("graph_vertex_labels")?;
            let label_compact = self.store.fetch_compact_rows("graph_vertex_labels")?;
            let stale_labels = label_rows
                .iter()
                .zip(label_compact)
                .filter_map(|(row, compact)| {
                    row.get("vertex_id")
                        .and_then(Value::as_str)
                        .filter(|vertex_id| removed_vertex_ids.contains(*vertex_id))
                        .map(|_| compact)
                })
                .collect::<Vec<_>>();
            if !stale_labels.is_empty() {
                self.store
                    .delete_key_rows("graph_vertex_labels", &stale_labels)?;
            }
        }

        let edge_rows = self.store.fetch_rows("graph_edges")?;
        let edge_compact = self.store.fetch_compact_rows("graph_edges")?;
        let stale_edges = edge_rows
            .iter()
            .zip(edge_compact)
            .filter_map(|(row, compact)| {
                let source_id = row.get("source_id").and_then(Value::as_str)?;
                let target_id = row.get("target_id").and_then(Value::as_str)?;
                phase1_native_graph_row_matches_session(row, session_id).then_some((
                    source_id.to_owned(),
                    target_id.to_owned(),
                    compact,
                ))
            })
            .filter_map(|(source_id, target_id, compact)| {
                (!keep_edge_pairs.contains(&(source_id, target_id))).then_some(compact)
            })
            .collect::<Vec<_>>();
        let removed_edge_count = stale_edges.len();
        if !stale_edges.is_empty() {
            self.store.delete_key_rows("graph_edges", &stale_edges)?;
        }

        Ok((removed_vertex_count, removed_edge_count))
    }

    #[allow(dead_code)]
    fn load_latest_om_record_for_thread(
        &self,
        thread_id: &str,
    ) -> Result<Option<OmRecord>, StoreError> {
        let mut records = self
            .store
            .fetch_rows("om_records")?
            .into_iter()
            .filter(|row| row.get("thread_id").and_then(Value::as_str) == Some(thread_id))
            .map(om_record_from_value)
            .collect::<Result<Vec<_>, _>>()?;
        records.sort_by(|left, right| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| right.created_at.cmp(&left.created_at))
        });
        Ok(records.into_iter().next())
    }

    fn list_candidate_prototype_inputs(
        &self,
        document_ids: &[String],
    ) -> Result<Vec<SemanticCandidatePrototypeInput>, StoreError> {
        if document_ids.is_empty() {
            return Ok(Vec::new());
        }

        let graph = self.graph_snapshot(false)?;
        let leaf_chunks = self.semantic_leaf_chunks_for_documents(document_ids)?;
        let leaf_context = phase2_leaf_context_map(&leaf_chunks);
        let document_scopes = phase2_document_scope_from_leaf_chunks(&leaf_chunks);
        let messages = self.semantic_thread_message_map()?;
        let allowed_documents = document_ids.iter().cloned().collect::<HashSet<_>>();

        let mut entity_ids = BTreeSet::new();
        let mut event_ids = BTreeSet::new();
        for leaf in &leaf_chunks {
            let leaf_id = phase2_leaf_vertex_id(&leaf.span_id);
            for edge in graph
                .outgoing_any(&leaf_id)
                .chain(graph.incoming_any(&leaf_id))
            {
                let neighbor_id = if edge.source_id == leaf_id {
                    edge.target_id.as_str()
                } else {
                    edge.source_id.as_str()
                };
                let Some(vertex) = graph.vertices.get(neighbor_id) else {
                    continue;
                };
                match vertex.kind.as_str() {
                    "entity" => {
                        entity_ids.insert(vertex.id.clone());
                    }
                    "event" => {
                        event_ids.insert(vertex.id.clone());
                    }
                    _ => {}
                }
            }
        }

        let mut inputs = Vec::new();
        let mut seen = BTreeSet::new();
        for entity_id in entity_ids {
            if let Some(input) =
                phase2_entity_prototype_input(&graph, &leaf_context, &document_scopes, &entity_id)
            {
                if seen.insert(input.node_id.clone()) {
                    inputs.push(input);
                }
            }
        }
        for event_id in event_ids {
            if let Some(input) =
                phase2_event_prototype_input(&graph, &leaf_context, &document_scopes, &event_id)
            {
                if seen.insert(input.node_id.clone()) {
                    inputs.push(input);
                }
            }
        }
        for vertex in graph.vertices.values() {
            if !matches!(vertex.kind.as_str(), "turn" | "task" | "state") {
                continue;
            }
            if !vertex
                .document_id
                .as_ref()
                .map(|document_id| allowed_documents.contains(document_id))
                .unwrap_or(false)
            {
                continue;
            }
            let input = match vertex.kind.as_str() {
                "turn" => phase2_turn_prototype_input(vertex, &messages, &document_scopes),
                "task" => phase2_task_prototype_input(vertex, &document_scopes),
                "state" => phase2_state_prototype_input(vertex, &document_scopes),
                _ => None,
            };
            if let Some(input) = input {
                if seen.insert(input.node_id.clone()) {
                    inputs.push(input);
                }
            }
        }
        inputs.sort_by(|left, right| {
            (&left.node_kind, &left.node_id).cmp(&(&right.node_kind, &right.node_id))
        });
        Ok(inputs)
    }

    fn list_nli_judgment_inputs(
        &self,
        document_ids: &[String],
        node_ids: &[String],
    ) -> Result<Vec<SemanticNliJudgmentInput>, StoreError> {
        if document_ids.is_empty() && node_ids.is_empty() {
            return Ok(Vec::new());
        }

        let keep_docs = document_ids.iter().cloned().collect::<HashSet<_>>();
        let keep_nodes = node_ids.iter().cloned().collect::<HashSet<_>>();
        let graph = self.graph_snapshot(true)?;
        let rows = self.candidate_edge_rows()?;

        let mut relevant = rows
            .into_iter()
            .filter_map(|row| {
                let source_id = row.get("source_id").and_then(Value::as_str)?.to_owned();
                let target_id = row.get("target_id").and_then(Value::as_str)?.to_owned();
                let edge_type = row.get("edge_type").and_then(Value::as_str)?.to_owned();
                if !matches!(
                    edge_type.as_str(),
                    "candidate_corefers_with" | "candidate_same_event_as" | "about" | "relevant_to"
                ) {
                    return None;
                }

                let attributes = row.get("attributes").cloned().unwrap_or(Value::Null);
                if !phase2_candidate_row_is_active(&attributes) {
                    return None;
                }

                let document_id = row
                    .get("document_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        attributes
                            .get("documentId")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    });
                let touched = document_id
                    .as_ref()
                    .map(|value| keep_docs.contains(value))
                    .unwrap_or(false)
                    || keep_nodes.contains(&source_id)
                    || keep_nodes.contains(&target_id);
                if !touched {
                    return None;
                }

                Some(Phase2CandidateEdgeRecord {
                    source_id,
                    target_id,
                    edge_type,
                    document_id,
                    base_score: phase2_candidate_row_base_score(
                        row.get("data"),
                        row.get("attributes"),
                    ),
                })
            })
            .collect::<Vec<_>>();

        relevant.sort_by(|left, right| {
            right
                .base_score
                .partial_cmp(&left.base_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    (
                        left.edge_type.as_str(),
                        left.source_id.as_str(),
                        left.target_id.as_str(),
                    )
                        .cmp(&(
                            right.edge_type.as_str(),
                            right.source_id.as_str(),
                            right.target_id.as_str(),
                        ))
                })
        });
        let mut seen_groups = HashSet::<String>::new();
        relevant.retain(|record| {
            if matches!(
                record.edge_type.as_str(),
                "candidate_corefers_with" | "candidate_same_event_as"
            ) {
                seen_groups.insert(phase2_nli_group_id(record))
            } else {
                true
            }
        });

        let mut relevant_documents = keep_docs.clone();
        for record in &relevant {
            if let Some(document_id) = record.document_id.as_ref() {
                relevant_documents.insert(document_id.clone());
            }
            if let Some(document_id) = graph
                .vertices
                .get(&record.source_id)
                .and_then(|vertex| vertex.document_id.clone())
            {
                relevant_documents.insert(document_id);
            }
            if let Some(document_id) = graph
                .vertices
                .get(&record.target_id)
                .and_then(|vertex| vertex.document_id.clone())
            {
                relevant_documents.insert(document_id);
            }
        }

        let relevant_document_ids = relevant_documents.into_iter().collect::<Vec<_>>();
        let leaf_chunks = self.semantic_leaf_chunks_for_documents(&relevant_document_ids)?;
        let leaf_context = phase2_leaf_context_map(&leaf_chunks);
        let document_scopes = phase2_document_scope_from_leaf_chunks(&leaf_chunks);
        let document_support = phase2_document_support_from_leaf_chunks(&leaf_chunks);
        let messages = self.semantic_thread_message_map()?;
        let mut profile_cache = HashMap::<String, SemanticCandidatePrototypeInput>::new();
        let mut inputs = Vec::new();
        let mut seen_judgment_ids = HashSet::<String>::new();

        for record in relevant {
            let Some(source_profile) = phase2_nli_profile_for_vertex(
                &graph,
                &leaf_context,
                &document_scopes,
                &document_support,
                &messages,
                &record.source_id,
                &mut profile_cache,
            ) else {
                continue;
            };
            let Some(target_profile) = phase2_nli_profile_for_vertex(
                &graph,
                &leaf_context,
                &document_scopes,
                &document_support,
                &messages,
                &record.target_id,
                &mut profile_cache,
            ) else {
                continue;
            };
            for input in
                phase2_nli_inputs_for_edge(&record, &source_profile.text, &target_profile.text)
            {
                if !seen_judgment_ids.insert(input.judgment_id.clone()) {
                    continue;
                }
                inputs.push(input);
                if inputs.len() >= PHASE2_NLI_MAX_INPUTS {
                    return Ok(inputs);
                }
            }
        }

        let mut evidence_source_ids = graph
            .vertices
            .values()
            .filter(|vertex| matches!(vertex.kind.as_str(), "task" | "state"))
            .filter(|vertex| {
                keep_nodes.contains(&vertex.id)
                    || vertex
                        .document_id
                        .as_ref()
                        .map(|document_id| keep_docs.contains(document_id))
                        .unwrap_or(false)
            })
            .map(|vertex| vertex.id.clone())
            .collect::<Vec<_>>();
        evidence_source_ids.sort();

        for source_id in evidence_source_ids {
            let Some(source_vertex) = graph.vertices.get(&source_id) else {
                continue;
            };
            let Some(source_profile) = phase2_nli_profile_for_vertex(
                &graph,
                &leaf_context,
                &document_scopes,
                &document_support,
                &messages,
                &source_id,
                &mut profile_cache,
            ) else {
                continue;
            };

            let mut evidence_target_ids = Vec::new();
            let mut seen_targets = BTreeSet::<String>::new();

            for edge in graph.outgoing_any(&source_id) {
                if !matches!(edge.edge_type.as_str(), "observed_in" | "depends_on") {
                    continue;
                }
                let Some(target_vertex) = graph.vertices.get(&edge.target_id) else {
                    continue;
                };
                if target_vertex.kind != "turn" {
                    continue;
                }
                if seen_targets.insert(edge.target_id.clone()) {
                    evidence_target_ids.push(edge.target_id.clone());
                }
            }

            if let Some(document_id) = source_vertex.document_id.as_deref() {
                let mut leaf_candidates = leaf_context
                    .iter()
                    .filter(|(_, context)| context.document_id == document_id)
                    .map(|(leaf_id, context)| {
                        (
                            phase2_text_overlap_score(&source_profile.text, &context.text),
                            leaf_id.clone(),
                        )
                    })
                    .collect::<Vec<_>>();
                leaf_candidates
                    .sort_by(|left, right| right.0.cmp(&left.0).then_with(|| left.1.cmp(&right.1)));

                let mut added_leafs = 0usize;
                for (score, leaf_id) in leaf_candidates.iter().filter(|(score, _)| *score > 0) {
                    if added_leafs >= PHASE2_NLI_MAX_LEAF_EVIDENCE_PER_SOURCE {
                        break;
                    }
                    let _ = score;
                    if seen_targets.insert(leaf_id.clone()) {
                        evidence_target_ids.push(leaf_id.clone());
                        added_leafs += 1;
                    }
                }
                if added_leafs == 0 {
                    if let Some((_, leaf_id)) = leaf_candidates.first() {
                        if seen_targets.insert(leaf_id.clone()) {
                            evidence_target_ids.push(leaf_id.clone());
                        }
                    }
                }
            }

            evidence_target_ids.truncate(PHASE2_NLI_MAX_EVIDENCE_TARGETS);
            for target_id in evidence_target_ids {
                let Some(target_profile) = phase2_nli_profile_for_vertex(
                    &graph,
                    &leaf_context,
                    &document_scopes,
                    &document_support,
                    &messages,
                    &target_id,
                    &mut profile_cache,
                ) else {
                    continue;
                };
                for input in phase2_nli_inputs_for_evidence_edge(
                    &source_id,
                    &source_profile.text,
                    &target_id,
                    &target_profile.text,
                ) {
                    if !seen_judgment_ids.insert(input.judgment_id.clone()) {
                        continue;
                    }
                    inputs.push(input);
                    if inputs.len() >= PHASE2_NLI_MAX_INPUTS {
                        return Ok(inputs);
                    }
                }
            }
        }

        Ok(inputs)
    }

    fn semantic_leaf_chunks_for_documents(
        &self,
        document_ids: &[String],
    ) -> Result<Vec<phoenix_store_cozo::SemanticLeafChunk>, StoreError> {
        if self.native_graph_enabled() {
            self.native_semantic_leaf_chunks_for_documents(document_ids)
        } else {
            self.store.list_leaf_chunks_for_documents(document_ids)
        }
    }

    fn native_semantic_leaf_chunks_for_documents(
        &self,
        document_ids: &[String],
    ) -> Result<Vec<phoenix_store_cozo::SemanticLeafChunk>, StoreError> {
        if document_ids.is_empty() {
            return Ok(Vec::new());
        }
        let allowed = document_ids.iter().cloned().collect::<HashSet<_>>();
        let mut chunks = self
            .native_store()?
            .load_lex_spans(None)?
            .into_iter()
            .filter_map(|span| {
                let document_id = span.document_id.as_ref()?.0.clone();
                if !allowed.contains(&document_id) {
                    return None;
                }
                let text = span
                    .fields
                    .iter()
                    .map(|field| field.text.trim())
                    .filter(|text| !text.is_empty())
                    .collect::<Vec<_>>()
                    .join(" ");
                if text.is_empty() {
                    return None;
                }
                Some(phoenix_store_cozo::SemanticLeafChunk {
                    span_id: span.span_id,
                    document_id,
                    text,
                    narrative_id: span.scope.narrative_id,
                    folder_id: span.scope.folder_id,
                })
            })
            .collect::<Vec<_>>();
        chunks.sort_by(|left, right| {
            left.document_id
                .cmp(&right.document_id)
                .then_with(|| left.span_id.cmp(&right.span_id))
        });
        Ok(chunks)
    }

    fn semantic_thread_message_map(&self) -> Result<HashMap<String, ThreadMessage>, StoreError> {
        if self.native_graph_enabled() {
            return Ok(HashMap::new());
        }
        phase2_thread_message_map(&self.store)
    }

    fn semantic_document_scope_map(
        &self,
        document_ids: &[String],
    ) -> Result<HashMap<String, ScopeKey>, StoreError> {
        let leaf_chunks = self.semantic_leaf_chunks_for_documents(document_ids)?;
        Ok(phase2_document_scope_from_leaf_chunks(&leaf_chunks))
    }

    fn native_semantic_document_vector_map(
        &self,
        document_ids: &[String],
    ) -> Result<HashMap<String, StoredSemanticDocumentVector>, StoreError> {
        Ok(self
            .native_store()?
            .load_semantic_document_vector_records(document_ids)?
            .into_iter()
            .map(|record| {
                (
                    record.document_id,
                    StoredSemanticDocumentVector {
                        values: record.values,
                        evidence_refs: record.evidence_refs,
                    },
                )
            })
            .collect())
    }

    fn native_semantic_node_vector_map(
        &self,
        node_ids: &[String],
    ) -> Result<HashMap<String, StoredSemanticNodeVector>, StoreError> {
        Ok(self
            .native_store()?
            .load_semantic_node_vector_records(node_ids)?
            .into_iter()
            .map(|record| {
                let node_id = record.node_id.clone();
                (
                    node_id.clone(),
                    StoredSemanticNodeVector {
                        node_id,
                        node_kind: record.node_kind,
                        document_id: record.document_id,
                        narrative_id: record.scope.narrative_id,
                        folder_id: record.scope.folder_id,
                        values: record.values,
                        evidence_refs: record.evidence_refs,
                    },
                )
            })
            .collect())
    }

    fn native_semantic_document_upsert_rows(
        &self,
        rows: &[SemanticDocumentVectorUpsertRow],
        updated_at: i64,
    ) -> Result<Vec<NativeSemanticDocumentVectorRecord>, StoreError> {
        let document_ids = rows
            .iter()
            .map(|row| row.document_id.clone())
            .collect::<Vec<_>>();
        let scope_by_document = self.semantic_document_scope_map(&document_ids)?;
        let mut native_rows = Vec::with_capacity(rows.len());
        for row in rows {
            let Some(scope) = scope_by_document.get(&row.document_id).cloned() else {
                return Err(StoreError::Query(format!(
                    "missing native scope for semantic document vector {}",
                    row.document_id
                )));
            };
            native_rows.push(NativeSemanticDocumentVectorRecord {
                scope,
                document_id: row.document_id.clone(),
                values: row.values.clone(),
                leaf_count: row.leaf_count,
                evidence_refs: row.evidence_refs.clone(),
                updated_at,
            });
        }
        Ok(native_rows)
    }

    fn native_semantic_node_upsert_rows(
        &self,
        rows: &[SemanticNodeVectorUpsertRow],
        updated_at: i64,
    ) -> Result<Vec<NativeSemanticNodeVectorRecord>, StoreError> {
        let document_ids = rows
            .iter()
            .filter_map(|row| row.document_id.clone())
            .collect::<Vec<_>>();
        let scope_by_document = self.semantic_document_scope_map(&document_ids)?;
        let mut native_rows = Vec::with_capacity(rows.len());
        for row in rows {
            let scope = match (
                row.narrative_id.clone(),
                row.folder_id.clone(),
                row.document_id.as_ref(),
            ) {
                (narrative_id, folder_id, _) if narrative_id.is_some() || folder_id.is_some() => {
                    ScopeKey {
                        world_id: None,
                        narrative_id,
                        folder_id,
                        folder_path: None,
                    }
                }
                (_, _, Some(document_id)) => scope_by_document.get(document_id).cloned().ok_or_else(
                    || {
                        StoreError::Query(format!(
                            "missing native scope for semantic node vector {}",
                            row.node_id
                        ))
                    },
                )?,
                _ => {
                    return Err(StoreError::Query(format!(
                        "semantic node vector {} requires scope or document_id",
                        row.node_id
                    )))
                }
            };
            native_rows.push(NativeSemanticNodeVectorRecord {
                scope,
                node_id: row.node_id.clone(),
                node_kind: row.node_kind.clone(),
                document_id: row.document_id.clone(),
                narrative_id: row.narrative_id.clone(),
                folder_id: row.folder_id.clone(),
                values: row.values.clone(),
                evidence_refs: row.evidence_refs.clone(),
                updated_at,
            });
        }
        Ok(native_rows)
    }

    fn refresh_candidate_graph_edges(
        &self,
        document_ids: &[String],
        node_ids: &[String],
    ) -> Result<usize, StoreError> {
        let graph = self.graph_snapshot(false)?;
        let active_vertices = graph.vertices.keys().cloned().collect::<HashSet<_>>();
        let doc_scope = self.semantic_document_scope_map(document_ids)?;
        let document_vectors = if self.native_graph_enabled() {
            self.native_semantic_document_vector_map(document_ids)?
        } else {
            phase2_document_vector_map(&self.store, document_ids)?
        };
        let node_vectors = if self.native_graph_enabled() {
            self.native_semantic_node_vector_map(node_ids)?
        } else {
            phase2_node_vector_map(&self.store, node_ids)?
        };

        if !self.native_graph_enabled() {
            self.prune_candidate_graph_edges(document_ids, node_ids)?;
        }

        let mut candidate_rows = Vec::new();
        let mut edge_keys = BTreeSet::new();

        for document_id in document_ids {
            let Some(source_vector) = document_vectors.get(document_id) else {
                continue;
            };
            let Some(scope) = doc_scope.get(document_id).cloned() else {
                continue;
            };
            let neighbors = if self.native_graph_enabled() {
                self.native_store()?
                    .query_semantic_documents(&source_vector.values, &scope, 4, 16)?
            } else {
                self.store
                    .query_semantic_documents(&source_vector.values, &scope, 4, 16)?
            };
            for neighbor in neighbors {
                if neighbor.document_id == *document_id {
                    continue;
                }
                if !active_vertices.contains(&phase1_document_vertex_id(&neighbor.document_id)) {
                    continue;
                }
                let score = phase2_similarity_score(neighbor.distance);
                if score < PHASE2_DOC_SIMILAR_TO_THRESHOLD {
                    continue;
                }
                if phase2_graph_has_edge(
                    &graph,
                    &phase1_document_vertex_id(document_id),
                    &phase1_document_vertex_id(&neighbor.document_id),
                    "similar_to",
                ) {
                    continue;
                }
                phase2_push_candidate_edge(
                    &mut edge_keys,
                    &mut candidate_rows,
                    phase2_candidate_edge_row(
                        &phase1_document_vertex_id(document_id),
                        &phase1_document_vertex_id(&neighbor.document_id),
                        "similar_to",
                        Some(document_id.as_str()),
                        scope.narrative_id.as_deref(),
                        score,
                        PHASE2_DOC_SIMILAR_TO_THRESHOLD,
                        {
                            let mut attributes = Map::new();
                            attributes.insert("sourceKind".to_owned(), json!("document"));
                            attributes.insert("targetKind".to_owned(), json!("document"));
                            attributes.insert("layer".to_owned(), json!("candidate"));
                            attributes
                        },
                        phase2_merge_evidence_refs(
                            &source_vector.evidence_refs,
                            &neighbor.evidence_refs,
                        ),
                    ),
                );
            }
        }

        for node_id in node_ids {
            let Some(source_vector) = node_vectors.get(node_id) else {
                continue;
            };
            if !active_vertices.contains(&source_vector.node_id) {
                continue;
            }
            let scope = ScopeKey {
                world_id: None,
                narrative_id: source_vector.narrative_id.clone(),
                folder_id: source_vector.folder_id.clone(),
                folder_path: None,
            };
            match source_vector.node_kind.as_str() {
                "entity" => {
                    let neighbors = if self.native_graph_enabled() {
                        self.native_store()?.query_semantic_node_neighbors(
                            &source_vector.values,
                            &scope,
                            "entity",
                            Some(&source_vector.node_id),
                            4,
                            16,
                        )?
                    } else {
                        self.store.query_semantic_node_neighbors(
                            &source_vector.values,
                            &scope,
                            "entity",
                            Some(&source_vector.node_id),
                            4,
                            16,
                        )?
                    };
                    for neighbor in neighbors {
                        if !active_vertices.contains(&neighbor.node_id) {
                            continue;
                        }
                        if !phase2_entity_candidate_is_coherent(
                            &graph,
                            &source_vector.node_id,
                            &neighbor.node_id,
                        ) {
                            continue;
                        }
                        let score = phase2_similarity_score(neighbor.distance);
                        if score < PHASE2_ENTITY_COREF_THRESHOLD {
                            continue;
                        }
                        if phase2_graph_has_symmetric_edge(
                            &graph,
                            &source_vector.node_id,
                            &neighbor.node_id,
                            "candidate_corefers_with",
                        ) {
                            continue;
                        }
                        phase2_push_candidate_edge(
                            &mut edge_keys,
                            &mut candidate_rows,
                            phase2_candidate_edge_row(
                                &source_vector.node_id,
                                &neighbor.node_id,
                                "candidate_corefers_with",
                                source_vector.document_id.as_deref(),
                                source_vector.narrative_id.as_deref(),
                                score,
                                PHASE2_ENTITY_COREF_THRESHOLD,
                                {
                                    let mut attributes = Map::new();
                                    attributes.insert("sourceKind".to_owned(), json!("entity"));
                                    attributes.insert("targetKind".to_owned(), json!("entity"));
                                    attributes
                                },
                                phase2_merge_evidence_refs(
                                    &source_vector.evidence_refs,
                                    &neighbor.evidence_refs,
                                ),
                            ),
                        );
                    }
                }
                "event" => {
                    let neighbors = if self.native_graph_enabled() {
                        self.native_store()?.query_semantic_node_neighbors(
                            &source_vector.values,
                            &scope,
                            "event",
                            Some(&source_vector.node_id),
                            4,
                            16,
                        )?
                    } else {
                        self.store.query_semantic_node_neighbors(
                            &source_vector.values,
                            &scope,
                            "event",
                            Some(&source_vector.node_id),
                            4,
                            16,
                        )?
                    };
                    for neighbor in neighbors {
                        if !active_vertices.contains(&neighbor.node_id) {
                            continue;
                        }
                        let score = phase2_similarity_score(neighbor.distance);
                        if score < PHASE2_EVENT_MATCH_THRESHOLD {
                            continue;
                        }
                        if phase2_graph_has_edge(
                            &graph,
                            &source_vector.node_id,
                            &neighbor.node_id,
                            "candidate_same_event_as",
                        ) {
                            continue;
                        }
                        phase2_push_candidate_edge(
                            &mut edge_keys,
                            &mut candidate_rows,
                            phase2_candidate_edge_row(
                                &source_vector.node_id,
                                &neighbor.node_id,
                                "candidate_same_event_as",
                                source_vector.document_id.as_deref(),
                                source_vector.narrative_id.as_deref(),
                                score,
                                PHASE2_EVENT_MATCH_THRESHOLD,
                                {
                                    let mut attributes = Map::new();
                                    attributes.insert("sourceKind".to_owned(), json!("event"));
                                    attributes.insert("targetKind".to_owned(), json!("event"));
                                    attributes
                                },
                                phase2_merge_evidence_refs(
                                    &source_vector.evidence_refs,
                                    &neighbor.evidence_refs,
                                ),
                            ),
                        );
                    }
                }
                "turn" | "task" | "state" => {
                    let document_neighbors = if self.native_graph_enabled() {
                        self.native_store()?
                            .query_semantic_documents(&source_vector.values, &scope, 3, 12)?
                    } else {
                        self.store
                            .query_semantic_documents(&source_vector.values, &scope, 3, 12)?
                    };
                    for neighbor in document_neighbors {
                        if !active_vertices
                            .contains(&phase1_document_vertex_id(&neighbor.document_id))
                        {
                            continue;
                        }
                        let score = phase2_similarity_score(neighbor.distance);
                        if score < PHASE2_ABOUT_THRESHOLD {
                            continue;
                        }
                        if phase2_graph_has_edge(
                            &graph,
                            &source_vector.node_id,
                            &phase1_document_vertex_id(&neighbor.document_id),
                            "about",
                        ) {
                            continue;
                        }
                        phase2_push_candidate_edge(
                            &mut edge_keys,
                            &mut candidate_rows,
                            phase2_candidate_edge_row(
                                &source_vector.node_id,
                                &phase1_document_vertex_id(&neighbor.document_id),
                                "about",
                                source_vector.document_id.as_deref(),
                                source_vector.narrative_id.as_deref(),
                                score,
                                PHASE2_ABOUT_THRESHOLD,
                                {
                                    let mut attributes = Map::new();
                                    attributes.insert(
                                        "sourceKind".to_owned(),
                                        json!(source_vector.node_kind),
                                    );
                                    attributes.insert("targetKind".to_owned(), json!("document"));
                                    attributes
                                },
                                phase2_merge_evidence_refs(
                                    &source_vector.evidence_refs,
                                    &neighbor.evidence_refs,
                                ),
                            ),
                        );
                    }
                    for target_kind in ["entity", "event"] {
                        let neighbors = if self.native_graph_enabled() {
                            self.native_store()?.query_semantic_node_neighbors(
                                &source_vector.values,
                                &scope,
                                target_kind,
                                None,
                                3,
                                12,
                            )?
                        } else {
                            self.store.query_semantic_node_neighbors(
                                &source_vector.values,
                                &scope,
                                target_kind,
                                None,
                                3,
                                12,
                            )?
                        };
                        for neighbor in neighbors {
                            if !active_vertices.contains(&neighbor.node_id) {
                                continue;
                            }
                            let score = phase2_similarity_score(neighbor.distance);
                            if score < PHASE2_RELEVANT_TO_THRESHOLD {
                                continue;
                            }
                            if phase2_graph_has_edge(
                                &graph,
                                &source_vector.node_id,
                                &neighbor.node_id,
                                "relevant_to",
                            ) {
                                continue;
                            }
                            phase2_push_candidate_edge(
                                &mut edge_keys,
                                &mut candidate_rows,
                                phase2_candidate_edge_row(
                                    &source_vector.node_id,
                                    &neighbor.node_id,
                                    "relevant_to",
                                    source_vector.document_id.as_deref(),
                                    source_vector.narrative_id.as_deref(),
                                    score,
                                    PHASE2_RELEVANT_TO_THRESHOLD,
                                    {
                                        let mut attributes = Map::new();
                                        attributes.insert(
                                            "sourceKind".to_owned(),
                                            json!(source_vector.node_kind),
                                        );
                                        attributes.insert(
                                            "targetKind".to_owned(),
                                            json!(neighbor.node_kind),
                                        );
                                        attributes
                                    },
                                    phase2_merge_evidence_refs(
                                        &source_vector.evidence_refs,
                                        &neighbor.evidence_refs,
                                    ),
                                ),
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        if self.native_graph_enabled() {
            let touched_scope_keys = document_ids
                .iter()
                .map(|document_id| native_candidate_scope_key_for_document(document_id))
                .chain(
                    node_ids
                        .iter()
                        .map(|node_id| native_candidate_scope_key_for_node(node_id)),
                )
                .collect::<BTreeSet<_>>();
            let batches = native_candidate_batches_from_rows(candidate_rows, &touched_scope_keys);
            let inserted = batches.iter().map(|batch| batch.edges.len()).sum();
            self.persist_native_candidate_scope_batches(batches)?;
            Ok(inserted)
        } else {
            let inserted = candidate_rows.len();
            for row in candidate_rows {
                self.store.put_row("graph_candidate_edges", row)?;
            }
            Ok(inserted)
        }
    }

    fn prune_candidate_graph_edges(
        &self,
        document_ids: &[String],
        node_ids: &[String],
    ) -> Result<(), StoreError> {
        if document_ids.is_empty() && node_ids.is_empty() {
            return Ok(());
        }
        let keep_docs = document_ids.iter().cloned().collect::<HashSet<_>>();
        let keep_nodes = node_ids.iter().cloned().collect::<HashSet<_>>();
        let rows = self.store.fetch_rows("graph_candidate_edges")?;
        let compact = self.store.fetch_compact_rows("graph_candidate_edges")?;
        let stale = rows
            .iter()
            .zip(compact)
            .filter_map(|(row, compact)| {
                let source_id = row
                    .get("source_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let target_id = row
                    .get("target_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let document_id = row
                    .get("document_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let target_document_id = target_id.strip_prefix("doc::").unwrap_or_default();
                (keep_nodes.contains(source_id)
                    || keep_nodes.contains(target_id)
                    || keep_docs.contains(document_id)
                    || keep_docs.contains(target_document_id))
                .then_some(compact)
            })
            .collect::<Vec<_>>();
        if !stale.is_empty() {
            self.store
                .delete_key_rows("graph_candidate_edges", &stale)?;
        }
        Ok(())
    }

    fn apply_nli_judgments(
        &self,
        model_id: &str,
        device: Option<&str>,
        rows: Vec<SemanticNliJudgmentResultRow>,
    ) -> Result<Value, StoreError> {
        let graph = self.graph_snapshot(true)?;
        let mut aggregates =
            BTreeMap::<(String, String, String), Phase2NliJudgmentAggregate>::new();
        for row in rows {
            let key = (
                row.source_id.clone(),
                row.target_id.clone(),
                row.edge_type.clone(),
            );
            let aggregate = aggregates
                .entry(key)
                .or_insert_with(|| Phase2NliJudgmentAggregate {
                    group_id: row.group_id.clone(),
                    judgments: Vec::new(),
                });
            aggregate.judgments.push(row);
        }

        if aggregates.is_empty() {
            return Ok(json!({
                "judged": 0,
                "kept": 0,
                "rejected": 0,
                "byEdgeType": {},
                "modelId": model_id,
                "device": device,
                "diagnostics": [
                    {
                        "code": "PX_NLI_SKIP",
                        "message": "No NLI judgments were provided; candidate graph edges were left unchanged."
                    }
                ],
            }));
        }

        let existing_rows = self.candidate_edge_rows()?;
        let mut row_map = HashMap::<(String, String, String), Value>::new();
        for row in existing_rows {
            let Some(source_id) = row.get("source_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(target_id) = row.get("target_id").and_then(Value::as_str) else {
                continue;
            };
            let Some(edge_type) = row.get("edge_type").and_then(Value::as_str) else {
                continue;
            };
            row_map.insert(
                (
                    source_id.to_owned(),
                    target_id.to_owned(),
                    edge_type.to_owned(),
                ),
                row,
            );
        }

        let mut kept = 0usize;
        let mut rejected = 0usize;
        let mut by_edge_type = BTreeMap::<String, Value>::new();
        let judged_at = now_ms();
        let mut touched_scope_keys = BTreeSet::<String>::new();

        for ((source_id, target_id, edge_type), aggregate) in aggregates {
            let Some(mut row) = row_map
                .remove(&(source_id.clone(), target_id.clone(), edge_type.clone()))
                .or_else(|| {
                    phase2_nli_candidate_edge_seed_row(&graph, &source_id, &target_id, &edge_type)
                })
            else {
                continue;
            };
            let decision = phase2_apply_nli_decision(
                &edge_type,
                phase2_candidate_row_base_score(row.get("data"), row.get("attributes")),
                &aggregate.judgments,
            );
            let attributes_snapshot = row
                .get("attributes")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new()));
            let existing_data = row
                .get("data")
                .cloned()
                .filter(|value| value.is_object())
                .unwrap_or_else(|| Value::Object(Map::new()));
            let base_data = existing_data
                .get("base")
                .cloned()
                .or_else(|| (!existing_data.is_null()).then_some(existing_data.clone()))
                .unwrap_or_else(|| {
                    json!({
                        "score": phase2_candidate_row_base_score(None, Some(&attributes_snapshot)),
                        "resolver": attributes_snapshot
                            .get("graph")
                            .and_then(Value::as_object)
                            .and_then(|graph| graph.get("resolver"))
                            .and_then(Value::as_str)
                            .unwrap_or(PHASE2_EMBEDDING_RESOLVER),
                    })
                });

            let evidence_refs = phase2_merge_evidence_refs(
                &phase2_graph_evidence_refs(&attributes_snapshot),
                &phase2_nli_evidence_refs(&aggregate.judgments),
            );
            let mut attributes_object =
                attributes_snapshot.as_object().cloned().unwrap_or_default();

            attributes_object.insert("score".to_owned(), json!(decision.final_score));
            attributes_object.insert("nliScore".to_owned(), json!(decision.nli_score));
            attributes_object.insert(
                "graph".to_owned(),
                json!({
                    "layer": "candidate",
                    "status": if decision.accepted { "candidate" } else { "candidate_rejected" },
                    "resolver": PHASE2_NLI_RESOLVER,
                    "confidence": decision.final_score,
                    "evidence_refs": evidence_refs,
                }),
            );
            row["attributes"] = Value::Object(attributes_object);
            row["data"] = json!({
                "base": base_data,
                "nli": {
                    "modelId": model_id,
                    "device": device,
                    "judgedAt": judged_at,
                    "groupId": aggregate.group_id,
                    "accepted": decision.accepted,
                    "threshold": decision.threshold,
                    "aggregated": {
                        "entailment": decision.entailment,
                        "neutral": decision.neutral,
                        "contradiction": decision.contradiction,
                        "nliScore": decision.nli_score,
                        "finalScore": decision.final_score,
                    },
                    "judgments": aggregate.judgments.iter().map(|judgment| {
                        json!({
                            "judgmentId": judgment.judgment_id,
                            "direction": judgment.direction,
                            "premise": judgment.premise,
                            "hypothesis": judgment.hypothesis,
                            "entailment": judgment.entailment,
                            "neutral": judgment.neutral,
                            "contradiction": judgment.contradiction,
                            "predictedLabel": judgment.predicted_label,
                            "confidence": judgment.confidence,
                        })
                    }).collect::<Vec<_>>(),
                }
            });
            row["weight"] = json!(if decision.accepted {
                ((decision.final_score * 1000.0).round() as i64).max(1)
            } else {
                0
            });

            if let Some(scope_key) = native_candidate_scope_key_for_row(&row) {
                touched_scope_keys.insert(scope_key);
            }
            row_map.insert(
                (source_id.clone(), target_id.clone(), edge_type.clone()),
                row.clone(),
            );
            if !self.native_graph_enabled() {
                self.store.put_row("graph_candidate_edges", row)?;
            }

            let counter = by_edge_type
                .entry(edge_type.clone())
                .or_insert_with(|| json!({ "kept": 0, "rejected": 0 }));
            if let Some(object) = counter.as_object_mut() {
                let key = if decision.accepted {
                    "kept"
                } else {
                    "rejected"
                };
                let current = object.get(key).and_then(Value::as_u64).unwrap_or(0);
                object.insert(key.to_owned(), json!(current + 1));
            }

            if decision.accepted {
                kept += 1;
            } else {
                rejected += 1;
            }
        }

        if self.native_graph_enabled() && !touched_scope_keys.is_empty() {
            let batches = native_candidate_batches_from_rows(
                row_map.into_values().collect::<Vec<_>>(),
                &touched_scope_keys,
            );
            self.persist_native_candidate_scope_batches(batches)?;
        }

        Ok(json!({
            "judged": kept + rejected,
            "kept": kept,
            "rejected": rejected,
            "byEdgeType": by_edge_type,
            "modelId": model_id,
            "device": device,
            "diagnostics": [
                {
                    "code": "PX_NLI_OK",
                    "message": format!(
                        "Applied local NLI judgments to {} candidate edges (kept {}, rejected {}).",
                        kept + rejected,
                        kept,
                        rejected
                    ),
                }
            ],
        }))
    }

    pub fn query(&self, request: QueryRequest) -> Result<QueryResult, StoreError> {
        self.query_view(QueryRequestView::from(&request))
    }

    pub fn query_view(&self, request: QueryRequestView<'_>) -> Result<QueryResult, StoreError> {
        let created_at = now_ms();
        self.put_relation_row(
            "phoenix_query_log",
            serde_json::json!({
                "id": format!("query-{}", created_at),
                "session_id": request.session_id.as_ref().map(|value| value.0.clone()),
                "query": request.query,
                "limit": request.limit,
                "request_json": serde_json::json!({
                    "sessionId": request.session_id.as_ref().map(|value| value.0.clone()),
                    "query": request.query,
                    "scope": request.scope.to_owned(),
                    "targets": request.targets,
                    "limit": request.limit,
                    "temporal": request.temporal,
                    "hasSemanticVector": request.semantic_query_vector.is_some(),
                    "includeCandidateGraph": request.include_candidate_graph,
                }),
                "created_at": created_at,
            }),
        )?;

        let graph_requested = request.targets.iter().any(|target| {
            matches!(
                target,
                phoenix_types::QueryTarget::Nodes
                    | phoenix_types::QueryTarget::Graph
                    | phoenix_types::QueryTarget::Semantic
            )
        });
        let native_store = if self.native_graph_enabled() {
            Some(
                self.v2_store()
                    .ok_or_else(|| StoreError::Query("missing native LMDB store".to_owned()))?,
            )
        } else {
            None
        };
        if graph_requested {
            let semantic_requested = request
                .targets
                .iter()
                .any(|target| matches!(target, phoenix_types::QueryTarget::Semantic));
            let requested_candidate_graph = request.include_candidate_graph;
            let mut owned_request = request.to_owned();
            if semantic_requested && !self.config.feature_flags.semantic {
                owned_request.semantic_query_vector = None;
                owned_request
                    .targets
                    .retain(|target| !matches!(target, phoenix_types::QueryTarget::Semantic));
            }
            if owned_request.include_candidate_graph && !self.config.feature_flags.candidate_graph {
                owned_request.include_candidate_graph = false;
            }
            let mut result = if let Some(store) = native_store {
                let native_query_plan = self.native_runtime.plan_query(&owned_request);
                let lexical_limit = native_query_plan.lexical_limit;
                let lexical_query = owned_request.query.as_str();
                let lexical = if let Some(lex) = self.lex.borrow().as_ref() {
                    lex.search(
                        lexical_query,
                        &owned_request.scope,
                        lexical_limit,
                    )
                } else {
                    self.native_lexical_search(
                        store,
                        lexical_query,
                        &owned_request.scope,
                        lexical_limit,
                    )?
                };
                self.native_runtime
                    .query_with_lexical(store, lexical, &owned_request)
                    .map_err(|error| StoreError::Query(error.to_string()))?
            } else {
                self.ensure_lex_index()?;
                let lex_borrow = self.lex.borrow();
                let lex = lex_borrow.as_ref().expect("lex should exist after ensure");
                self.gldr.query(&self.store, lex, &owned_request)?
            };
            if semantic_requested && !self.config.feature_flags.semantic {
                let engine_name = if self.native_graph_enabled() {
                    "Triverse"
                } else {
                    "GLDR"
                };
                result.diagnostics.push(Diagnostic {
                    code: "PX_QUERY_SEMANTIC_DISABLED".to_owned(),
                    message: format!(
                        "Semantic retrieval is disabled in runtime feature flags; {engine_name} used lexical and graph retrieval only."
                    ),
                });
            }
            if requested_candidate_graph && !self.config.feature_flags.candidate_graph {
                result.diagnostics.push(Diagnostic {
                    code: "PX_QUERY_CANDIDATE_GRAPH_DISABLED".to_owned(),
                    message:
                        "Candidate graph overlay was requested but runtime feature flags left it disabled."
                            .to_owned(),
                });
            }
            return Ok(result);
        }

        let lexical = if let Some(store) = native_store {
            if let Some(lex) = self.lex.borrow().as_ref() {
                lex.search(request.query, &request.scope.to_owned(), request.limit.unwrap_or(5))
            } else {
                self.native_lexical_search(
                    store,
                    request.query,
                    &request.scope.to_owned(),
                    request.limit.unwrap_or(5),
                )?
            }
        } else {
            self.ensure_lex_index()?;
            self.lex
                .borrow()
                .as_ref()
                .expect("lex should exist after ensure")
                .search(
                    request.query,
                    &request.scope.to_owned(),
                    request.limit.unwrap_or(5),
                )
        };

        Ok(QueryResult {
            session_id: request.session_id.clone(),
            chunk_hits: lexical
                .span_hits
                .into_iter()
                .map(|hit| phoenix_types::ChunkHit {
                    chunk_id: hit.span_id,
                    score: hit.score,
                })
                .collect(),
            node_hits: Vec::new(),
            diagnostics: {
                let mut diagnostics = lexical.diagnostics;
                diagnostics.push(Diagnostic {
                    code: "PX_QUERY_LEX".to_owned(),
                    message: "Query executed via the Phoenix lexical facade.".to_owned(),
                });
                diagnostics
            },
        })
    }

    pub fn query_binary(&self, request: QueryRequest) -> Result<Vec<u8>, StoreError> {
        let result = self.query_view(QueryRequestView::from(&request))?;
        binary::encode_query_result(&result)
    }

    pub fn query_binary_into(
        &self,
        request: QueryRequestView<'_>,
        buffer: &mut [u8],
    ) -> Result<usize, StoreError> {
        let result = self.query_view(request)?;
        binary::encode_query_result_into(buffer, &result)
    }

    pub fn encode_query_result_into(
        &self,
        result: &QueryResult,
        buffer: &mut [u8],
    ) -> Result<usize, StoreError> {
        binary::encode_query_result_into(buffer, result)
    }

    pub fn graph_delta(&self, request: GraphDeltaRequest) -> Result<GraphDeltaResult, StoreError> {
        let mut request = request;
        if request.include_candidate_graph && !self.config.feature_flags.candidate_graph {
            request.include_candidate_graph = false;
        }
        if self.native_graph_enabled() {
            self.augment_graph_delta_request_from_journal(&mut request)?;
            let graph = self.graph_snapshot(request.include_candidate_graph)?;
            let state = self.session_state(&request.session_id)?;
            Ok(build_graph_delta_from_snapshot(&graph, &state, &request))
        } else {
            self.graptor.graph_delta(&self.store, &request)
        }
    }

    pub fn graph_delta_binary(&self, request: GraphDeltaRequest) -> Result<Vec<u8>, StoreError> {
        let result = self.graph_delta(request)?;
        binary::encode_graph_delta(&result)
    }

    pub fn graph_delta_binary_into(
        &self,
        request: GraphDeltaRequest,
        buffer: &mut [u8],
    ) -> Result<usize, StoreError> {
        let result = self.graph_delta(request)?;
        binary::encode_graph_delta_into(buffer, &result)
    }

    pub fn ingest_stub(&self, request: IngestRequest) -> Result<IngestResult, StoreError> {
        self.ingest(request)
    }

    pub fn query_stub(&self, request: QueryRequest) -> Result<QueryResult, StoreError> {
        self.query(request)
    }

    pub fn scan_text(&self, request: ScanRequest) -> ScanArtifact {
        self.scan_text_view(ScanRequestView::from(&request))
    }

    pub fn scan_text_view(&self, request: ScanRequestView<'_>) -> ScanArtifact {
        if self.native_graph_enabled() {
            self.native_runtime.scan_text(request.to_owned())
        } else {
            self.scanner.scan_parts(
                request.text,
                &request.scope.to_owned(),
                request.session_id.as_ref(),
                request.resolver_seed,
            )
        }
    }

    pub fn build_structure(&self, request: StructureRequest) -> StructureArtifact {
        self.build_structure_view(StructureRequestView::from(&request))
    }

    pub fn build_structure_view(&self, request: StructureRequestView<'_>) -> StructureArtifact {
        if self.native_graph_enabled() {
            self.native_runtime.build_structure(request.to_owned())
        } else {
            self.structure.build_parts(request.text, request.scan)
        }
    }

    pub fn analyze_text(&self, text: &str) -> TextAnalytics {
        self.analyze_text_view(AnalyzeTextRequestView { text })
    }

    pub fn analyze_text_view(&self, request: AnalyzeTextRequestView<'_>) -> TextAnalytics {
        if self.native_graph_enabled() {
            self.native_runtime.analyze_text(request.text)
        } else {
            phoenix_analytics::analyze_text(request.text)
        }
    }

    pub fn export_snapshot(&self) -> Result<Vec<u8>, StoreError> {
        self.export_snapshot_partition(SnapshotPartition::All)
    }

    pub fn export_snapshot_partition(
        &self,
        partition: SnapshotPartition,
    ) -> Result<Vec<u8>, StoreError> {
        if self.native_graph_enabled() {
            self.ensure_native_graph_ready()?;
            let store = self.native_store()?;
            let graph_generation = store.kernel_current_generation()?;
            let rebuild_token = self.native_graph_rebuild_token()?;
            let kernel_snapshot = self.native_runtime.deterministic_kernel.snapshot();
            store.export_native_snapshot_with_kernel_snapshot(
                store.schema_version(),
                partition,
                Some(kernel_snapshot.as_ref()),
                Some(&rebuild_token),
                Some(graph_generation),
            )
        } else {
            self.store.export_snapshot_partition(partition)
        }
    }

    pub fn import_snapshot(&self, bytes: &[u8]) -> Result<SnapshotEnvelope, StoreError> {
        let envelope = if self.native_graph_enabled() {
            if !PhoenixLmdbStore::is_native_snapshot(bytes) {
                return Err(self.native_unsupported("legacy Cozo snapshots"));
            }
            let imported = self.native_store()?.import_native_snapshot(bytes)?;
            let phoenix_store_lmdb::ImportedNativeSnapshot {
                schema_version,
                created_at,
                relation_count,
                kernel_generation,
                kernel_snapshot,
            } = imported;
            if let Some(snapshot) = kernel_snapshot {
                self.native_runtime.deterministic_kernel.install_snapshot(
                    snapshot,
                    Some(format!("graph:{kernel_generation}")),
                );
            } else {
                self.native_runtime.deterministic_kernel.invalidate();
                self.native_runtime.deterministic_kernel.set_rebuild_token(None);
            }
            SnapshotEnvelope {
                schema_version,
                relation_count,
                created_at,
                relations: BTreeMap::new(),
                checksum: None,
            }
        } else {
            self.store.import_snapshot(bytes)?
        };
        self.invalidate_lex_caches();
        if self.native_graph_enabled() {
            if let Some(store) = self.v2_store() {
                self.warm_native_scope_lex_caches(store.load_lex_spans(None)?);
            }
            self.ensure_native_graph_ready()?;
        } else {
            self.rebuild_lex_index()?;
        }
        Ok(envelope)
    }

    pub fn snapshot_descriptor(&self, created_at: i64, payload_bytes: usize) -> SnapshotDto {
        if self.native_graph_enabled() {
            SnapshotDto {
                schema_version: NATIVE_RUNTIME_SCHEMA_VERSION.to_owned(),
                created_at,
                payload_bytes,
                relation_counts: self.relation_counts().unwrap_or_default(),
            }
        } else {
            self.store.snapshot_descriptor(created_at, payload_bytes)
        }
    }

    pub fn session_state(&self, session_id: &SessionId) -> Result<SessionState, StoreError> {
        if self.native_graph_enabled() {
            let store = self
                .v2_store()
                .ok_or_else(|| StoreError::Query("missing native LMDB store".to_owned()))?;
            if let Some(summary) = self
                .native_runtime
                .load_latest_session_summary(store, session_id)
                .map_err(|error| StoreError::Query(error.to_string()))?
            {
                let session = self.load_session(session_id)?;
                return Ok(SessionState {
                    session_id: session_id.clone(),
                    documents: summary.documents,
                    manifest_namespaces: vec!["invarant-v3.session".to_owned()],
                    updated_at: session.updated_at.max(summary.updated_at),
                });
            }
            let session = self.load_session(session_id)?;
            return Ok(SessionState {
                session_id: session_id.clone(),
                documents: Vec::new(),
                manifest_namespaces: vec!["invarant-v3.session".to_owned()],
                updated_at: session.updated_at,
            });
        }
        load_session_state(&self.store, session_id)
    }

    pub fn session_stats(&self, session_id: &SessionId) -> Result<SessionStats, StoreError> {
        if self.native_graph_enabled() {
            let store = self
                .v2_store()
                .ok_or_else(|| StoreError::Query("missing native LMDB store".to_owned()))?;
            if let Some(summary) = self
                .native_runtime
                .load_latest_session_summary(store, session_id)
                .map_err(|error| StoreError::Query(error.to_string()))?
            {
                let session = self.load_session(session_id)?;
                let document_count = summary.documents.len();
                let chapter_count = summary
                    .documents
                    .iter()
                    .map(|document| document.chapter_count)
                    .sum();
                let boundary_count = summary
                    .documents
                    .iter()
                    .map(|document| document.boundary_count)
                    .sum();
                let parent_count = summary
                    .documents
                    .iter()
                    .map(|document| document.parent_count)
                    .sum();
                let leaf_count = summary
                    .documents
                    .iter()
                    .map(|document| document.leaf_count)
                    .sum();
                let entity_count = summary
                    .documents
                    .iter()
                    .map(|document| document.entity_count)
                    .sum();
                return Ok(SessionStats {
                    session_id: session_id.clone(),
                    document_count,
                    chapter_count,
                    boundary_count,
                    parent_count,
                    leaf_count,
                    entity_count,
                    discovery_candidate_count: summary.discovery_candidate_count,
                    graph_vertex_count: summary.graph_vertex_count,
                    graph_edge_count: summary.graph_edge_count,
                    span_count: summary.span_count,
                    updated_at: session.updated_at.max(summary.updated_at),
                });
            }
            let session = self.load_session(session_id)?;
            return Ok(SessionStats {
                session_id: session_id.clone(),
                document_count: 0,
                chapter_count: 0,
                boundary_count: 0,
                parent_count: 0,
                leaf_count: 0,
                entity_count: 0,
                discovery_candidate_count: 0,
                graph_vertex_count: 0,
                graph_edge_count: 0,
                span_count: 0,
                updated_at: session.updated_at,
            });
        }
        self.graptor.session_stats(&self.store, session_id)
    }

    pub fn upsert_entity_card(&self, card: &EntityCard) -> Result<(), StoreError> {
        if self.native_graph_enabled() {
            self.upsert_entity_cards_batch(std::slice::from_ref(card))?;
        } else {
            self.legacy_store("entity cards")?.upsert_entity_card(card)?;
        }
        self.sync_relation_from_cozo_to_native("entity_cards")
    }

    pub fn upsert_entity_cards_batch(&self, cards: &[EntityCard]) -> Result<(), StoreError> {
        if self.native_graph_enabled() {
            let rows = cards
                .iter()
                .map(entity_card_row)
                .collect::<Vec<_>>();
            self.replace_native_relation_rows_with_keys(
                "entity_cards",
                &rows,
                &["entity_id", "card_id"],
            )?;
        } else {
            self.legacy_store("entity cards")?
                .upsert_entity_cards_batch(cards)?;
        }
        self.sync_relation_from_cozo_to_native("entity_cards")
    }

    pub fn get_entity_cards(
        &self,
        entity_id: &phoenix_types::EntityId,
    ) -> Result<Vec<EntityCard>, StoreError> {
        if self.native_graph_enabled() {
            let mut cards = self
                .fetch_relation_rows("entity_cards")?
                .into_iter()
                .filter(|row| row.get("entity_id").and_then(Value::as_str) == Some(entity_id.0.as_str()))
                .map(entity_card_from_row)
                .collect::<Result<Vec<_>, _>>()?;
            cards.sort_by(|left: &EntityCard, right: &EntityCard| {
                left.display_order
                    .cmp(&right.display_order)
                    .then_with(|| left.card_id.cmp(&right.card_id))
            });
            Ok(cards)
        } else {
            self.legacy_store("entity cards")?.get_entity_cards(entity_id)
        }
    }

    pub fn upsert_folder_schema(&self, schema: &FolderSchema) -> Result<(), StoreError> {
        if self.native_graph_enabled() {
            let row = folder_schema_row(schema);
            self.replace_native_relation_rows_with_keys("folder_schemas", &[row], &["id"])?;
        } else {
            self.legacy_store("folder schemas")?
                .upsert_folder_schema(schema)?;
        }
        self.sync_relation_from_cozo_to_native("folder_schemas")
    }

    pub fn get_folder_schema(&self, id: &str) -> Result<Option<FolderSchema>, StoreError> {
        if self.native_graph_enabled() {
            self.fetch_relation_rows("folder_schemas")?
                .into_iter()
                .find(|row| row.get("id").and_then(Value::as_str) == Some(id))
                .map(folder_schema_from_row)
                .transpose()
        } else {
            self.legacy_store("folder schemas")?.get_folder_schema(id)
        }
    }

    pub fn save_network_view(&self, view: &SavedNetworkView) -> Result<(), StoreError> {
        if self.native_graph_enabled() {
            let store = self.native_store()?;
            let instance_row = network_instance_row(&view.instance);
            let member_rows = view.members.iter().map(network_membership_row).collect::<Vec<_>>();
            let relationship_rows = view
                .relationships
                .iter()
                .map(network_relationship_row)
                .collect::<Vec<_>>();

            let mut instances = store.fetch_rows("network_instance")?;
            instances.retain(|row| row.get("id").and_then(Value::as_str) != Some(view.instance.id.as_str()));
            instances.push(instance_row);
            store.replace_relation_rows("network_instance", &instances)?;

            let mut members = store.fetch_rows("network_membership")?;
            members.retain(|row| {
                row.get("network_id").and_then(Value::as_str) != Some(view.instance.id.as_str())
            });
            members.extend(member_rows);
            store.replace_relation_rows("network_membership", &members)?;

            let mut relationships = store.fetch_rows("network_relationship")?;
            relationships.retain(|row| {
                row.get("network_id").and_then(Value::as_str) != Some(view.instance.id.as_str())
            });
            relationships.extend(relationship_rows);
            store.replace_relation_rows("network_relationship", &relationships)?;

            return Ok(());
        }

        let existing = self.get_network_view(&view.instance.id)?;

        self.legacy_store("network views")?
            .upsert_network_instance(&view.instance)?;
        self.legacy_store("network views")?
            .upsert_network_memberships(&view.members)?;
        self.legacy_store("network views")?
            .upsert_network_relationships(&view.relationships)?;

        if let Some(existing) = existing {
            let new_member_keys = view
                .members
                .iter()
                .map(|member| (member.network_id.clone(), member.entity_id.0.clone()))
                .collect::<BTreeSet<_>>();
            let stale_members = existing
                .members
                .into_iter()
                .filter(|member| {
                    !new_member_keys
                        .contains(&(member.network_id.clone(), member.entity_id.0.clone()))
                })
                .collect::<Vec<_>>();
            self.legacy_store("network views")?
                .delete_network_memberships(&stale_members)?;

            let new_relationship_keys = view
                .relationships
                .iter()
                .map(|relationship| {
                    (
                        relationship.network_id.clone(),
                        relationship.relationship_id.clone(),
                    )
                })
                .collect::<BTreeSet<_>>();
            let stale_relationships = existing
                .relationships
                .into_iter()
                .filter(|relationship| {
                    !new_relationship_keys.contains(&(
                        relationship.network_id.clone(),
                        relationship.relationship_id.clone(),
                    ))
                })
                .collect::<Vec<_>>();
            self.legacy_store("network views")?
                .delete_network_relationships(&stale_relationships)?;
        }

        self.sync_relation_from_cozo_to_native("network_instance")?;
        self.sync_relation_from_cozo_to_native("network_membership")?;
        self.sync_relation_from_cozo_to_native("network_relationship")?;
        Ok(())
    }

    pub fn get_network_view(&self, id: &str) -> Result<Option<SavedNetworkView>, StoreError> {
        if self.native_graph_enabled() {
            let Some(instance) = self
                .fetch_relation_rows("network_instance")?
                .into_iter()
                .find(|row| row.get("id").and_then(Value::as_str) == Some(id))
            else {
                return Ok(None);
            };
            let instance = network_instance_from_row(instance)?;
            let members = self
                .fetch_relation_rows("network_membership")?
                .into_iter()
                .filter(|row| row.get("network_id").and_then(Value::as_str) == Some(id))
                .map(network_membership_from_row)
                .collect::<Result<Vec<_>, _>>()?;
            let relationships = self
                .fetch_relation_rows("network_relationship")?
                .into_iter()
                .filter(|row| row.get("network_id").and_then(Value::as_str) == Some(id))
                .map(network_relationship_from_row)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Some(SavedNetworkView {
                instance,
                members,
                relationships,
            }))
        } else {
            let Some(instance) = self.legacy_store("network views")?.get_network_instance(id)? else {
                return Ok(None);
            };
            let members = self.legacy_store("network views")?.get_network_members(id)?;
            let relationships = self
                .legacy_store("network views")?
                .get_network_relationships(id)?;
            Ok(Some(SavedNetworkView {
                instance,
                members,
                relationships,
            }))
        }
    }

    pub fn list_network_views(&self) -> Result<Vec<NetworkInstance>, StoreError> {
        if self.native_graph_enabled() {
            let mut views = self
                .fetch_relation_rows("network_instance")?
                .into_iter()
                .map(network_instance_from_row)
                .collect::<Result<Vec<_>, _>>()?;
            views.sort_by(|left: &NetworkInstance, right: &NetworkInstance| {
                right
                    .updated_at
                    .cmp(&left.updated_at)
                    .then_with(|| left.id.cmp(&right.id))
            });
            Ok(views)
        } else {
            self.legacy_store("network views")?.list_network_instances()
        }
    }

    pub fn delete_network_view(&self, id: &str) -> Result<(), StoreError> {
        if self.native_graph_enabled() {
            let store = self.native_store()?;
            let mut instances = store.fetch_rows("network_instance")?;
            instances.retain(|row| row.get("id").and_then(Value::as_str) != Some(id));
            store.replace_relation_rows("network_instance", &instances)?;

            let mut members = store.fetch_rows("network_membership")?;
            members.retain(|row| row.get("network_id").and_then(Value::as_str) != Some(id));
            store.replace_relation_rows("network_membership", &members)?;

            let mut relationships = store.fetch_rows("network_relationship")?;
            relationships.retain(|row| row.get("network_id").and_then(Value::as_str) != Some(id));
            store.replace_relation_rows("network_relationship", &relationships)?;
            return Ok(());
        }

        let members = self.legacy_store("network views")?.get_network_members(id)?;
        let relationships = self
            .legacy_store("network views")?
            .get_network_relationships(id)?;
        self.legacy_store("network views")?
            .delete_network_relationships(&relationships)?;
        self.legacy_store("network views")?
            .delete_network_memberships(&members)?;
        self.legacy_store("network views")?
            .delete_network_instance(id)?;
        self.sync_relation_from_cozo_to_native("network_instance")?;
        self.sync_relation_from_cozo_to_native("network_membership")?;
        self.sync_relation_from_cozo_to_native("network_relationship")
    }

    fn clear_derived_partition(&self) -> Result<(), StoreError> {
        if self.native_graph_enabled() {
            self.native_store()?
                .clear_relations(DERIVED_SNAPSHOT_RELATIONS)?;
            self.native_store()?
                .clear_relations(DERIVED_EPHEMERA_RELATIONS)?;
        } else {
            self.store.clear_relations(DERIVED_SNAPSHOT_RELATIONS)?;
            self.store.clear_relations(DERIVED_EPHEMERA_RELATIONS)?;
            for relation in DERIVED_SNAPSHOT_RELATIONS {
                self.sync_relation_from_cozo_to_native(relation)?;
            }
            for relation in DERIVED_EPHEMERA_RELATIONS {
                self.sync_relation_from_cozo_to_native(relation)?;
            }
        }
        self.rebuild_lex_index()?;
        Ok(())
    }

    fn clear_derived_ephemera(&self) -> Result<(), StoreError> {
        if self.native_graph_enabled() {
            self.native_store()?
                .clear_relations(DERIVED_EPHEMERA_RELATIONS)?;
        } else {
            self.store.clear_relations(DERIVED_EPHEMERA_RELATIONS)?;
            for relation in DERIVED_EPHEMERA_RELATIONS {
                self.sync_relation_from_cozo_to_native(relation)?;
            }
        }
        Ok(())
    }

    fn delete_rows_by_session(
        &self,
        relation: &str,
        key_columns: &[&str],
        session_id: &str,
    ) -> Result<usize, StoreError> {
        if self.native_graph_enabled() {
            let rows = self
                .fetch_relation_rows(relation)?
                .into_iter()
                .filter(|row| row.get("session_id").and_then(Value::as_str) == Some(session_id))
                .collect::<Vec<_>>();
            self.delete_relation_rows(relation, &rows)
        } else {
            let rows = self.legacy_store("session cleanup")?.fetch_compact_rows_where_str(
                relation,
                key_columns,
                "session_id",
                session_id,
            )?;
            let count = rows.len();
            self.legacy_store("session cleanup")?
                .delete_key_rows(relation, &rows)?;
            Ok(count)
        }
    }

    fn close_session(&self, session_id: &str) -> Result<usize, StoreError> {
        let mut deleted = 0usize;
        deleted += self.delete_rows_by_session("phoenix_commits", &["commit_id"], session_id)?;
        deleted += self.delete_rows_by_session("phoenix_ingest_log", &["id"], session_id)?;
        deleted += self.delete_rows_by_session("phoenix_query_log", &["id"], session_id)?;
        deleted += self.delete_rows_by_session("phoenix_sessions", &["session_id"], session_id)?;
        self.sync_relation_from_cozo_to_native("phoenix_commits")?;
        self.sync_relation_from_cozo_to_native("phoenix_ingest_log")?;
        self.sync_relation_from_cozo_to_native("phoenix_query_log")?;
        self.sync_relation_from_cozo_to_native("phoenix_sessions")?;
        Ok(deleted)
    }

    fn apply_persistence_wal_batch(
        &self,
        records: &[PersistenceWalRecord],
    ) -> Result<(), StoreError> {
        for record in records {
            if record.seq == 0 {
                return Err(StoreError::Query("invalid WAL seq: 0".to_owned()));
            }
            if record.partition != "content" {
                return Err(StoreError::Query(format!(
                    "unsupported WAL partition: {}",
                    record.partition
                )));
            }
            let _written_at = record.written_at;

            match record.command.as_str() {
                "note:upsert" => {
                    let row = require_payload_value(&record.payload, "row")?;
                    self.upsert_note_row(row)?;
                }
                "note:delete" => {
                    let id = require_payload_str(&record.payload, "id")?;
                    self.delete_note_rows(id)?;
                }
                "relation:upsert" => {
                    let relation = require_payload_str(&record.payload, "relation")?;
                    ensure_allowed_content_relation(relation)?;
                    let row = require_payload_value(&record.payload, "row")?;
                    self.put_relation_row(relation, row.clone())?;
                }
                "relation:delete" => {
                    let relation = require_payload_str(&record.payload, "relation")?;
                    ensure_allowed_content_relation(relation)?;
                    let filter = payload_object(record.payload.get("filter"));
                    let rows = self.fetch_relation_rows(relation)?;
                    let matched = rows
                        .into_iter()
                        .filter(|row| row_matches_filter(row, filter))
                        .collect::<Vec<_>>();
                    let _ = self.delete_relation_rows(relation, &matched)?;
                }
                "entityCards:upsertBatch" => {
                    let cards: Vec<EntityCard> = serde_json::from_value(
                        record
                            .payload
                            .get("cards")
                            .cloned()
                            .unwrap_or_else(|| Value::Array(Vec::new())),
                    )
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                    self.upsert_entity_cards_batch(&cards)?;
                }
                "folderSchema:upsert" => {
                    let schema: FolderSchema = serde_json::from_value(
                        record.payload.get("schema").cloned().unwrap_or(Value::Null),
                    )
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                    self.upsert_folder_schema(&schema)?;
                }
                "networkView:save" => {
                    let view: SavedNetworkView = serde_json::from_value(
                        record.payload.get("view").cloned().unwrap_or(Value::Null),
                    )
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                    self.save_network_view(&view)?;
                }
                "networkView:delete" => {
                    let id = require_payload_str(&record.payload, "id")?;
                    self.delete_network_view(id)?;
                }
                other => {
                    return Err(StoreError::Query(format!(
                        "unsupported WAL command: {other}"
                    )));
                }
            }
        }

        self.rebuild_lex_index()?;
        Ok(())
    }

    pub fn init_chat_config(&self, config: ChatRuntimeConfig) -> ChatRuntimeConfig {
        self.chat.init_config(config)
    }

    pub fn store_command(
        &self,
        request: StoreCommandRequest,
    ) -> Result<StoreCommandResult, StoreError> {
        if self.native_graph_enabled()
            && request.command != "runtime:capabilities"
            && ["chat:", "om:"]
                .iter()
                .any(|prefix| request.command.starts_with(prefix))
        {
            return Ok(StoreCommandResult {
                success: false,
                payload: None,
                error: Some(format!(
                    "{} is unavailable on the native runtime path",
                    request.command
                )),
            });
        }
        match request.command.as_str() {
            "relation:upsert" => {
                let relation = require_payload_str(&request.payload, "relation")?;
                let row = require_payload_value(&request.payload, "row")?;
                self.put_relation_row(relation, row.clone())?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            "relation:getFirst" => {
                let relation = require_payload_str(&request.payload, "relation")?;
                let filter = payload_object(request.payload.get("filter"));
                let row = self
                    .fetch_store_command_relation_rows(relation)?
                    .into_iter()
                    .find(|row| row_matches_filter(row, filter));
                Ok(StoreCommandResult {
                    success: true,
                    payload: row,
                    error: None,
                })
            }
            "relation:list" => {
                let relation = require_payload_str(&request.payload, "relation")?;
                let filter = payload_object(request.payload.get("filter"));
                let rows = self
                    .fetch_store_command_relation_rows(relation)?
                    .into_iter()
                    .filter(|row| row_matches_filter(row, filter))
                    .collect::<Vec<_>>();
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(Value::Array(rows)),
                    error: None,
                })
            }
            "relation:delete" => {
                let relation = require_payload_str(&request.payload, "relation")?;
                let filter = payload_object(request.payload.get("filter"));
                let rows = self.fetch_relation_rows(relation)?;
                let matched = rows
                    .into_iter()
                    .filter(|row| row_matches_filter(row, filter))
                    .collect::<Vec<_>>();
                let deleted = self.delete_relation_rows(relation, &matched)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(serde_json::json!({ "deleted": deleted })),
                    error: None,
                })
            }
            "note:upsert" => {
                let row = require_payload_value(&request.payload, "row")?;
                self.upsert_note_row(row)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            "note:get" => {
                let id = require_payload_str(&request.payload, "id")?;
                let include_body = payload_bool(request.payload.get("includeBody"), true);
                let note = self.get_note_value(id, include_body)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: note,
                    error: None,
                })
            }
            "note:list" => {
                let folder_id = request.payload.get("folderId").and_then(Value::as_str);
                let include_body = payload_bool(request.payload.get("includeBody"), true);
                let notes = self.list_note_values(folder_id, include_body)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(Value::Array(notes)),
                    error: None,
                })
            }
            "note:listByIds" => {
                let ids = payload_string_array(request.payload.get("ids"));
                let include_body = payload_bool(request.payload.get("includeBody"), true);
                let notes = self.list_note_values_by_ids(&ids, include_body)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(Value::Array(notes)),
                    error: None,
                })
            }
            "note:delete" => {
                let id = require_payload_str(&request.payload, "id")?;
                let deleted = self.delete_note_rows(id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(serde_json::json!({ "deleted": deleted })),
                    error: None,
                })
            }
            "runtime:capabilities" => Ok(StoreCommandResult {
                success: true,
                payload: Some(serde_json::json!({
                    "storeApiVersion": STORE_API_VERSION,
                    "capabilities": RUNTIME_CAPABILITIES,
                })),
                error: None,
            }),
            "session:close" => {
                let session_id = require_payload_str(&request.payload, "sessionId")?;
                let deleted = self.close_session(session_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(serde_json::json!({ "deleted": deleted })),
                    error: None,
                })
            }
            "persistence:applyWalBatch" => {
                let batch: PersistenceWalBatchRequest = serde_json::from_value(request.payload)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                self.apply_persistence_wal_batch(&batch.records)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(serde_json::json!({ "replayed": batch.records.len() })),
                    error: None,
                })
            }
            "persistence:clearDerived" => {
                self.clear_derived_partition()?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            "persistence:clearDerivedEphemera" => {
                self.clear_derived_ephemera()?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            "chat:init" => {
                let config: ChatRuntimeConfig = serde_json::from_value(
                    request
                        .payload
                        .get("config")
                        .cloned()
                        .unwrap_or(Value::Null),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let config = self.init_chat_config(config);
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(config)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:createThread" => {
                let world_id = request.payload.get("worldId").and_then(Value::as_str);
                let narrative_id = request.payload.get("narrativeId").and_then(Value::as_str);
                let title = request.payload.get("title").and_then(Value::as_str);
                let thread = self
                    .chat
                    .create_thread(&self.store, world_id, narrative_id, title)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(thread)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:getThread" => {
                let id = require_payload_str(&request.payload, "id")?;
                let thread = self.chat.get_thread(&self.store, id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: thread
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:listThreads" => {
                let world_id = request.payload.get("worldId").and_then(Value::as_str);
                let threads = self.chat.list_threads(&self.store, world_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(threads)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:deleteThread" => {
                let id = require_payload_str(&request.payload, "id")?;
                self.chat.delete_thread(&self.store, id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            "chat:addMessage" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                let role = require_payload_str(&request.payload, "role")?;
                let content = request
                    .payload
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let narrative_id = request.payload.get("narrativeId").and_then(Value::as_str);
                let message =
                    self.chat
                        .add_message(&self.store, thread_id, role, content, narrative_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(message)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:listMessages" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                let messages = self.chat.list_messages(&self.store, thread_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(messages)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:updateMessage" => {
                let message_id = require_payload_str(&request.payload, "messageId")?;
                let content = request
                    .payload
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let message = self.chat.update_message(&self.store, message_id, content)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: message
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:appendMessage" => {
                let message_id = require_payload_str(&request.payload, "messageId")?;
                let chunk = request
                    .payload
                    .get("chunk")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let message = self.chat.append_message(&self.store, message_id, chunk)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: message
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:startStreamingMessage" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                let narrative_id = request.payload.get("narrativeId").and_then(Value::as_str);
                let message =
                    self.chat
                        .start_streaming_message(&self.store, thread_id, narrative_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(message)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:clearThread" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                self.chat.clear_thread(&self.store, thread_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            "chat:exportThread" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                let exported = self.chat.export_thread(&self.store, thread_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(Value::String(exported)),
                    error: None,
                })
            }
            "chat:startRun" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                let prompt = request
                    .payload
                    .get("prompt")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let options: RunOptions = serde_json::from_value(
                    request
                        .payload
                        .get("options")
                        .cloned()
                        .unwrap_or(Value::Null),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let run = self
                    .chat
                    .start_run(&self.store, thread_id, prompt, options)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(run)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:pollRun" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let snapshot = self
                    .chat
                    .poll_run(&self.store, run_id)?
                    .map(|mut snapshot| {
                        snapshot.planner_step = self.planner.peek_step(run_id);
                        if let Ok(artifacts) = list_run_artifacts(self, &snapshot.run) {
                            snapshot.artifacts = artifacts;
                        }
                        snapshot
                    });
                Ok(StoreCommandResult {
                    success: true,
                    payload: snapshot
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:resumeRun" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let run = self.chat.resume_run(&self.store, run_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: run
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:cancelRun" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                self.planner.drop_session(run_id);
                let run = self.chat.cancel_run(&self.store, run_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: run
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:listRunEvents" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                let limit = request
                    .payload
                    .get("limit")
                    .and_then(Value::as_u64)
                    .unwrap_or(100) as usize;
                let events = self
                    .chat
                    .list_run_events_for_thread(&self.store, thread_id, limit)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(events)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:markRunStreaming" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let assistant_message_id =
                    require_payload_str(&request.payload, "assistantMessageId")?;
                let snapshot =
                    self.chat
                        .mark_run_streaming(&self.store, run_id, assistant_message_id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: snapshot
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:completeRun" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                self.planner.drop_session(run_id);
                let assistant_message_id =
                    require_payload_str(&request.payload, "assistantMessageId")?;
                let final_response = request
                    .payload
                    .get("finalResponse")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let final_error = request.payload.get("finalError").and_then(Value::as_str);
                let snapshot = self.chat.complete_run(
                    &self.store,
                    run_id,
                    assistant_message_id,
                    final_response,
                    final_error,
                )?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: snapshot
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:getPlannerStep" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let run = self
                    .chat
                    .get_run(&self.store, run_id)?
                    .ok_or_else(|| StoreError::Query(format!("run not found: {run_id}")))?;
                let step = self.planner.get_step(self, &run)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: step
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:submitPlannerModelResponse" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let response: ChatPlannerModelResponse = serde_json::from_value(
                    request
                        .payload
                        .get("response")
                        .cloned()
                        .unwrap_or(Value::Null),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let run = self
                    .chat
                    .get_run(&self.store, run_id)?
                    .ok_or_else(|| StoreError::Query(format!("run not found: {run_id}")))?;
                let step = self.planner.submit_model_response(self, &run, response)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: step
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:advancePlannerRun" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let run = self
                    .chat
                    .get_run(&self.store, run_id)?
                    .ok_or_else(|| StoreError::Query(format!("run not found: {run_id}")))?;
                let step = self.planner.advance(self, &run)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: step
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:degradePlannerRun" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let reason = request
                    .payload
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("Planner failed.");
                let run = self
                    .chat
                    .get_run(&self.store, run_id)?
                    .ok_or_else(|| StoreError::Query(format!("run not found: {run_id}")))?;
                self.planner.degrade_run(self, &run, reason, None)?;
                self.planner.drop_session(run_id);
                let snapshot = self
                    .chat
                    .poll_run(&self.store, run_id)?
                    .map(|mut snapshot| {
                        snapshot.planner_step = self.planner.peek_step(run_id);
                        if let Ok(artifacts) = list_run_artifacts(self, &snapshot.run) {
                            snapshot.artifacts = artifacts;
                        }
                        snapshot
                    });
                Ok(StoreCommandResult {
                    success: true,
                    payload: snapshot
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:listPlannerArtifacts" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let run = self
                    .chat
                    .get_run(&self.store, run_id)?
                    .ok_or_else(|| StoreError::Query(format!("run not found: {run_id}")))?;
                let artifacts = list_run_artifacts(self, &run)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(artifacts)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:pinPlannerArtifact" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let key = require_payload_str(&request.payload, "key")?;
                let pinned = request
                    .payload
                    .get("pinned")
                    .and_then(Value::as_bool)
                    .unwrap_or(true);
                let run = self
                    .chat
                    .get_run(&self.store, run_id)?
                    .ok_or_else(|| StoreError::Query(format!("run not found: {run_id}")))?;
                let Some(artifact) = set_artifact_pinned(self, &run, key, pinned)? else {
                    return Err(StoreError::Query(format!("artifact not found: {key}")));
                };
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(artifact)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:prepareOm" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                let config = OmEngine::config_from_runtime(&self.chat.current_config());
                let action = self
                    .om_engine
                    .prepare_pending_action_with_graph(
                        &self.store,
                        &self.scanner,
                        &self.structure,
                        &self.om_bridge,
                        thread_id,
                        &config,
                    )
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: action
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "chat:applyOmAction" => {
                let action: OmPendingAction = serde_json::from_value(
                    request
                        .payload
                        .get("action")
                        .cloned()
                        .unwrap_or(Value::Null),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let response = request
                    .payload
                    .get("response")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let config = OmEngine::config_from_runtime(&self.chat.current_config());
                let applied = self
                    .om_engine
                    .apply_pending_action_with_graph(
                        &self.store,
                        &self.scanner,
                        &self.structure,
                        &self.om_bridge,
                        &config,
                        &action,
                        response,
                    )
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(Value::Bool(applied)),
                    error: None,
                })
            }
            "om:startReflector" => {
                let action: OmPendingAction = serde_json::from_value(
                    request
                        .payload
                        .get("action")
                        .cloned()
                        .unwrap_or(Value::Null),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let step = self
                    .om_engine
                    .start_reflector(&action)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(step)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "om:submitReflectorModelResponse" => {
                let session_id = require_payload_str(&request.payload, "sessionId")?;
                let response: OmReflectorModelResponse = serde_json::from_value(
                    request
                        .payload
                        .get("response")
                        .cloned()
                        .unwrap_or(Value::Null),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let step = self
                    .om_engine
                    .submit_reflector_model_response(session_id, response)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(step)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "om:submitReflectorToolResults" => {
                let session_id = require_payload_str(&request.payload, "sessionId")?;
                let results: Vec<OmReflectorToolResult> = serde_json::from_value(
                    request
                        .payload
                        .get("results")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new())),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let step = self
                    .om_engine
                    .submit_reflector_tool_results(session_id, &results)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(step)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "om:dropReflectorSession" => {
                let session_id = require_payload_str(&request.payload, "sessionId")?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(Value::Bool(
                        self.om_engine.drop_reflector_session(session_id),
                    )),
                    error: None,
                })
            }
            "om:recoverLostMemory" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                let limit = request
                    .payload
                    .get("limit")
                    .and_then(Value::as_u64)
                    .unwrap_or(10) as usize;
                let focus = request.payload.get("focus").and_then(Value::as_str);
                let hits = self
                    .om_bridge
                    .recover_lost_memory(&self.store, thread_id, limit, focus)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(hits)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "om:memoryGraphSearch" => {
                let thread_id = require_payload_str(&request.payload, "threadId")?;
                let query = request
                    .payload
                    .get("query")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let limit = request
                    .payload
                    .get("limit")
                    .and_then(Value::as_u64)
                    .unwrap_or(10) as usize;
                let hits = self
                    .om_bridge
                    .memory_graph_search(&self.store, thread_id, query, limit)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(hits)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "semantic:listLeafChunks" => {
                let document_ids = request
                    .payload
                    .get("documentIds")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let chunks = self.semantic_leaf_chunks_for_documents(&document_ids)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(chunks)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "semantic:listCandidatePrototypeInputs" => {
                let document_ids = request
                    .payload
                    .get("documentIds")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let rows = self.list_candidate_prototype_inputs(&document_ids)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(rows)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "semantic:listNliJudgmentInputs" => {
                let document_ids = request
                    .payload
                    .get("documentIds")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let node_ids = request
                    .payload
                    .get("nodeIds")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let rows = self.list_nli_judgment_inputs(&document_ids, &node_ids)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(rows)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "semantic:upsertDocumentVectors" => {
                let rows: Vec<SemanticDocumentVectorUpsertRow> = serde_json::from_value(
                    request
                        .payload
                        .get("rows")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new())),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let updated_at = now_ms();
                let inserted = rows.len();
                if self.native_graph_enabled() {
                    let vectors = self.native_semantic_document_upsert_rows(&rows, updated_at)?;
                    self.native_store()?
                        .upsert_semantic_document_vectors_native(&vectors)?;
                } else {
                    let vectors = rows
                        .iter()
                        .map(|row| phoenix_store_cozo::SemanticDocumentVectorRow {
                            document_id: row.document_id.as_str(),
                            values: row.values.as_slice(),
                            model_id: phoenix_store_cozo::SEMANTIC_MODEL_ID,
                            leaf_count: row.leaf_count,
                            evidence_refs: row.evidence_refs.as_slice(),
                            updated_at,
                        })
                        .collect::<Vec<_>>();
                    self.store.upsert_semantic_document_vectors(&vectors)?;
                }
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(serde_json::json!({
                        "inserted": inserted,
                        "modelId": phoenix_store_cozo::SEMANTIC_MODEL_ID,
                        "dimension": phoenix_store_cozo::SEMANTIC_VECTOR_DIM,
                    })),
                    error: None,
                })
            }
            "semantic:upsertPrototypeVectors" => {
                let rows: Vec<SemanticNodeVectorUpsertRow> = serde_json::from_value(
                    request
                        .payload
                        .get("rows")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new())),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let updated_at = now_ms();
                let inserted = rows.len();
                if self.native_graph_enabled() {
                    let vectors = self.native_semantic_node_upsert_rows(&rows, updated_at)?;
                    self.native_store()?.upsert_semantic_node_vectors_native(&vectors)?;
                } else {
                    let vectors = rows
                        .iter()
                        .map(|row| phoenix_store_cozo::SemanticNodeVectorRow {
                            node_id: row.node_id.as_str(),
                            node_kind: row.node_kind.as_str(),
                            document_id: row.document_id.as_deref(),
                            narrative_id: row.narrative_id.as_deref(),
                            folder_id: row.folder_id.as_deref(),
                            values: row.values.as_slice(),
                            model_id: phoenix_store_cozo::SEMANTIC_MODEL_ID,
                            evidence_refs: row.evidence_refs.as_slice(),
                            updated_at,
                        })
                        .collect::<Vec<_>>();
                    self.store.upsert_semantic_node_vectors(&vectors)?;
                }
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(serde_json::json!({
                        "inserted": inserted,
                        "modelId": phoenix_store_cozo::SEMANTIC_MODEL_ID,
                        "dimension": phoenix_store_cozo::SEMANTIC_VECTOR_DIM,
                    })),
                    error: None,
                })
            }
            "semantic:refreshCandidateGraphEdges" => {
                let document_ids = request
                    .payload
                    .get("documentIds")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let node_ids = request
                    .payload
                    .get("nodeIds")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let refreshed = self.refresh_candidate_graph_edges(&document_ids, &node_ids)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(serde_json::json!({
                        "inserted": refreshed,
                    })),
                    error: None,
                })
            }
            "semantic:applyNliJudgments" => {
                let model_id = request
                    .payload
                    .get("modelId")
                    .and_then(Value::as_str)
                    .unwrap_or(SEMANTIC_NLI_MODEL_ID);
                let device = request.payload.get("device").and_then(Value::as_str);
                let rows: Vec<SemanticNliJudgmentResultRow> = serde_json::from_value(
                    request
                        .payload
                        .get("results")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new())),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                let payload = self.apply_nli_judgments(model_id, device, rows)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(payload),
                    error: None,
                })
            }
            "chat:submitToolResults" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let results: Vec<ToolResultSubmission> = serde_json::from_value(
                    request
                        .payload
                        .get("results")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new())),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                self.planner.drop_session(run_id);
                let snapshot = self
                    .chat
                    .submit_tool_results(&self.store, run_id, &results)
                    .map(|mut snapshot| {
                        snapshot.planner_step = self.planner.peek_step(run_id);
                        if let Ok(artifacts) = list_run_artifacts(self, &snapshot.run) {
                            snapshot.artifacts = artifacts;
                        }
                        snapshot
                    })?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(snapshot)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "chat:submitApproval" => {
                let run_id = require_payload_str(&request.payload, "runId")?;
                let approval_id = require_payload_str(&request.payload, "approvalId")?;
                let approved = request
                    .payload
                    .get("approved")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let decision_json = request.payload.get("decisionJson").and_then(Value::as_str);
                self.planner.drop_session(run_id);
                let snapshot = self
                    .chat
                    .submit_approval(&self.store, run_id, approval_id, approved, decision_json)
                    .map(|mut snapshot| {
                        snapshot.planner_step = self.planner.peek_step(run_id);
                        if let Ok(artifacts) = list_run_artifacts(self, &snapshot.run) {
                            snapshot.artifacts = artifacts;
                        }
                        snapshot
                    })?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(snapshot)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "entityCards:upsertBatch" => {
                let cards: Vec<EntityCard> = serde_json::from_value(
                    request
                        .payload
                        .get("cards")
                        .cloned()
                        .unwrap_or_else(|| Value::Array(Vec::new())),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                self.upsert_entity_cards_batch(&cards)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            "entityCards:get" => {
                let entity_id = require_payload_str(&request.payload, "entityId")?;
                let cards =
                    self.get_entity_cards(&phoenix_types::EntityId(entity_id.to_owned()))?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(cards)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "folderSchema:upsert" => {
                let schema: FolderSchema = serde_json::from_value(
                    request
                        .payload
                        .get("schema")
                        .cloned()
                        .unwrap_or(Value::Null),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                self.upsert_folder_schema(&schema)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            "folderSchema:get" => {
                let id = require_payload_str(&request.payload, "id")?;
                let schema = self.get_folder_schema(id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: schema
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "networkView:save" => {
                let view: SavedNetworkView = serde_json::from_value(
                    request.payload.get("view").cloned().unwrap_or(Value::Null),
                )
                .map_err(|error| StoreError::Query(error.to_string()))?;
                self.save_network_view(&view)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            "networkView:get" => {
                let id = require_payload_str(&request.payload, "id")?;
                let view = self.get_network_view(id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: view
                        .map(|value| {
                            serde_json::to_value(value)
                                .map_err(|error| StoreError::Query(error.to_string()))
                        })
                        .transpose()?,
                    error: None,
                })
            }
            "networkView:list" => {
                let views = self.list_network_views()?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: Some(
                        serde_json::to_value(views)
                            .map_err(|error| StoreError::Query(error.to_string()))?,
                    ),
                    error: None,
                })
            }
            "networkView:delete" => {
                let id = require_payload_str(&request.payload, "id")?;
                self.delete_network_view(id)?;
                Ok(StoreCommandResult {
                    success: true,
                    payload: None,
                    error: None,
                })
            }
            other => Ok(StoreCommandResult {
                success: false,
                payload: None,
                error: Some(format!("unsupported store command: {other}")),
            }),
        }
    }

    fn upsert_note_row(&self, row: &Value) -> Result<(), StoreError> {
        let id = row
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| StoreError::Query("missing store command field: row.id".to_owned()))?;
        if self.native_graph_enabled() {
            self.replace_native_relation_rows_with_keys("notes", &[row.clone()], NOTE_KEY_COLUMNS)?;
        } else {
            let existing = self
                .legacy_store("notes")?
                .fetch_compact_rows_where_str("notes", NOTE_KEY_COLUMNS, "id", id)?;
            if !existing.is_empty() {
                self.legacy_store("notes")?
                    .delete_key_rows("notes", &existing)?;
            }
            self.put_relation_row("notes", row.clone())?;
        }
        Ok(())
    }

    pub(crate) fn get_note_value(
        &self,
        id: &str,
        include_body: bool,
    ) -> Result<Option<Value>, StoreError> {
        if self.native_graph_enabled() {
            let rows = self
                .fetch_relation_rows("notes")?
                .into_iter()
                .filter(|row| row.get("id").and_then(Value::as_str) == Some(id))
                .collect::<Vec<_>>();
            Ok(select_latest_note_value(rows, include_body))
        } else {
            let columns = note_columns(include_body);
            let rows = self
                .legacy_store("notes")?
                .fetch_compact_rows_where_str("notes", columns, "id", id)?;
            Ok(select_latest_note(&rows, columns).map(|view| note_value_from_row(view, include_body)))
        }
    }

    pub(crate) fn list_note_values(
        &self,
        folder_id: Option<&str>,
        include_body: bool,
    ) -> Result<Vec<Value>, StoreError> {
        if self.native_graph_enabled() {
            Ok(note_values_from_value_rows(
                self.fetch_relation_rows("notes")?,
                folder_id,
                include_body,
            ))
        } else {
            let columns = note_columns(include_body);
            let rows = match folder_id {
                Some(value) if !value.is_empty() => self
                    .legacy_store("notes")?
                    .fetch_compact_rows_where_str("notes", columns, "folder_id", value)?,
                _ => self
                    .legacy_store("notes")?
                    .fetch_compact_rows_with_columns("notes", columns)?,
            };
            Ok(note_values_from_rows(
                rows,
                columns,
                folder_id,
                include_body,
            ))
        }
    }

    pub(crate) fn list_note_values_by_ids(
        &self,
        ids: &[String],
        include_body: bool,
    ) -> Result<Vec<Value>, StoreError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        if self.native_graph_enabled() {
            let ids = ids.iter().map(String::as_str).collect::<HashSet<_>>();
            let rows = self
                .fetch_relation_rows("notes")?
                .into_iter()
                .filter(|row| {
                    row.get("id")
                        .and_then(Value::as_str)
                        .map(|id| ids.contains(id))
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>();
            Ok(note_values_from_value_rows(rows, None, include_body))
        } else {
            let columns = note_columns(include_body);
            let rows = self
                .legacy_store("notes")?
                .fetch_compact_rows_where_in_strings("notes", columns, "id", ids)?;
            Ok(note_values_from_rows(rows, columns, None, include_body))
        }
    }

    fn delete_note_rows(&self, id: &str) -> Result<usize, StoreError> {
        if self.native_graph_enabled() {
            let rows = self
                .fetch_relation_rows("notes")?
                .into_iter()
                .filter(|row| row.get("id").and_then(Value::as_str) == Some(id))
                .collect::<Vec<_>>();
            self.delete_relation_rows("notes", &rows)
        } else {
            let rows = self
                .legacy_store("notes")?
                .fetch_compact_rows_where_str("notes", NOTE_KEY_COLUMNS, "id", id)?;
            let deleted = rows.len();
            if deleted > 0 {
                self.legacy_store("notes")?
                    .delete_key_rows("notes", &rows)?;
            }
            Ok(deleted)
        }
    }

    pub fn session_state_binary(&self, session_id: &SessionId) -> Result<Vec<u8>, StoreError> {
        let state = self.session_state(session_id)?;
        binary::encode_session_state(&state)
    }

    pub fn session_state_binary_into(
        &self,
        session_id: &SessionId,
        buffer: &mut [u8],
    ) -> Result<usize, StoreError> {
        let state = self.session_state(session_id)?;
        binary::encode_session_state_into(buffer, &state)
    }

    pub fn session_stats_binary(&self, session_id: &SessionId) -> Result<Vec<u8>, StoreError> {
        let stats = self.session_stats(session_id)?;
        binary::encode_session_stats(&stats)
    }

    pub fn session_stats_binary_into(
        &self,
        session_id: &SessionId,
        buffer: &mut [u8],
    ) -> Result<usize, StoreError> {
        let stats = self.session_stats(session_id)?;
        binary::encode_session_stats_into(buffer, &stats)
    }

    fn ensure_lex_index(&self) -> Result<(), StoreError> {
        if self.lex.borrow().is_none() {
            self.rebuild_lex_index()?;
        }
        Ok(())
    }

    fn rebuild_lex_index(&self) -> Result<usize, StoreError> {
        let spans = if let Some(store) = self.v2_store() {
            self.native_runtime
                .load_latest_lex_spans(store, None)
                .map_err(|error| StoreError::Query(error.to_string()))?
        } else {
            indexed_spans_from_store(&self.store)?
        };
        self.native_scope_lex.borrow_mut().clear();
        let mut borrow = self.lex.borrow_mut();
        let span_count = spans.len();
        let span_count = if let Some(index) = borrow.as_mut() {
            index.rebuild_from_spans(&spans);
            span_count
        } else {
            let mut index = LexIndex::default();
            index.rebuild_from_spans(&spans);
            *borrow = Some(index);
            span_count
        };
        Ok(span_count)
    }

    fn invalidate_lex_caches(&self) {
        *self.lex.borrow_mut() = None;
        self.native_scope_lex.borrow_mut().clear();
    }

    fn insert_native_scope_lex_cache(&self, scope_key: String, generation: u64, lex: LexIndex) {
        let mut cache = self.native_scope_lex.borrow_mut();
        if !cache.contains_key(&scope_key) && cache.len() >= NATIVE_SCOPE_LEX_CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(
            scope_key,
            NativeScopeLexCacheEntry {
                generation,
                lex,
            },
        );
    }

    fn warm_native_scope_lex_caches(&self, spans: Vec<IndexedSpan>) {
        let mut scoped = HashMap::<String, Vec<IndexedSpan>>::new();
        for span in spans {
            let scope_key = scope_storage_key(&span.scope);
            scoped.entry(scope_key).or_default().push(span);
        }

        for (scope_key, scoped_spans) in scoped.into_iter().take(NATIVE_SCOPE_LEX_CACHE_LIMIT) {
            if scoped_spans.len() < NATIVE_SCOPE_QGRAM_MIN_SPANS {
                continue;
            }
            let lex = LexIndex::build(&scoped_spans, LexConfig::default());
            self.insert_native_scope_lex_cache(scope_key, 0, lex);
        }
    }

    fn load_session(&self, session_id: &SessionId) -> Result<SessionRecord, StoreError> {
        let rows = self.fetch_relation_rows("phoenix_sessions")?;
        let row = rows
            .into_iter()
            .find(|row| {
                row.get("session_id").and_then(Value::as_str) == Some(session_id.0.as_str())
            })
            .ok_or_else(|| StoreError::Query(format!("session not found: {}", session_id.0)))?;
        session_record_from_row(&row)
    }
}

#[allow(dead_code)]
fn boundary_kind_from_str(value: &str) -> phoenix_types::BoundaryKind {
    match value.to_ascii_lowercase().as_str() {
        "chapter" => phoenix_types::BoundaryKind::Chapter,
        "heading" => phoenix_types::BoundaryKind::Heading,
        "section" => phoenix_types::BoundaryKind::Section,
        "act" => phoenix_types::BoundaryKind::Act,
        _ => phoenix_types::BoundaryKind::Other,
    }
}

fn runtime_ingest_progress_enabled() -> bool {
    matches!(
        std::env::var("PHOENIX_INGEST_PROGRESS").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    ) || matches!(
        std::env::var("PHOENIX_PERF_PROGRESS").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn native_linear_lexical_search_result(
    spans: &[IndexedSpan],
    query: &str,
    scope: &ScopeKey,
    limit: usize,
) -> LexicalSearchResult {
    let normalized_query = normalize_lexical_query(query);
    let terms = normalized_query
        .split_whitespace()
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>();
    if terms.is_empty() {
        return LexicalSearchResult::default();
    }

    let mut hits = spans
        .iter()
        .filter(|span| span.scope == *scope)
        .filter_map(|span| native_span_hit(span, &normalized_query, &terms))
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.span_id.cmp(&right.span_id))
    });
    hits.truncate(limit);

    LexicalSearchResult {
        span_hits: hits,
        diagnostics: vec![Diagnostic {
            code: "PX_QUERY_NATIVE_SCOPE_LINEAR".to_owned(),
            message:
                "Native query used linear scope-local lexical spans without rebuilding the global in-memory lex index."
                    .to_owned(),
        }],
    }
}

fn native_span_hit(
    span: &IndexedSpan,
    normalized_query: &str,
    terms: &[&str],
) -> Option<SpanHit> {
    let mut matched_terms = 0usize;
    let mut score = 0.0f64;
    for field in &span.fields {
        let normalized_text = normalize_lexical_query(&field.text);
        if normalized_text.is_empty() {
            continue;
        }
        let field_weight = match field.field {
            LexicalField::Title => 3.0,
            LexicalField::Summary => 2.0,
            LexicalField::Body => 1.0,
            LexicalField::Tags => 1.5,
            LexicalField::Other => 1.0,
        };
        if normalized_text.contains(normalized_query) {
            score += field_weight * 4.0;
        }
        for term in terms {
            if normalized_text.contains(term) {
                matched_terms += 1;
                score += field_weight;
            }
        }
    }
    if matched_terms == 0 {
        return None;
    }
    Some(SpanHit {
        span_id: span.span_id.clone(),
        note_id: span.note_id.clone(),
        document_id: span.document_id.clone(),
        score,
        coverage: matched_terms as f32 / terms.len() as f32,
    })
}

fn normalize_lexical_query(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ch.is_whitespace())
        .flat_map(|ch| ch.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn runtime_row_document_id(row: &Map<String, Value>) -> Option<String> {
    row.get("document_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            row.get("attributes")
                .and_then(Value::as_object)
                .and_then(|attributes| attributes.get("documentId"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            row.get("value")
                .and_then(Value::as_object)
                .and_then(|value| value.get("documentId"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

#[allow(dead_code)]
fn graph_vertex_record_from_row_value(row: &Value) -> Option<GraphVertexRecord> {
    let object = row.as_object()?;
    let id = object.get("id")?.as_str()?.to_owned();
    let value = object.get("value").cloned().unwrap_or(Value::Null);
    let attributes = object.get("attributes").cloned().unwrap_or(Value::Null);
    let chapters = attributes
        .get("chapters")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_u64)
                .map(|value| value as u32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let boundary_ordinals = attributes
        .get("boundaryOrdinals")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_u64)
                .map(|value| value as u32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(GraphVertexRecord {
        id,
        kind: value
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned(),
        weight: object.get("weight").and_then(Value::as_i64).unwrap_or(1),
        value: value.clone(),
        attributes: attributes.clone(),
        entity_id: value
            .get("entityId")
            .and_then(Value::as_str)
            .map(str::to_owned),
        search_chunk_id: value
            .get("searchChunkId")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                attributes
                    .get("searchChunkId")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }),
        document_id: runtime_row_document_id(object),
        chapter_id: attributes
            .get("chapterId")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        chapters,
        boundary_id: attributes
            .get("boundaryId")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        boundary_ordinal: attributes
            .get("boundaryOrdinal")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        boundary_kind: attributes
            .get("boundaryKind")
            .and_then(Value::as_str)
            .map(boundary_kind_from_str),
        boundary_ordinals,
    })
}

fn graph_edge_record_from_row_value(row: &Value, layer: GraphLayer) -> Option<GraphEdgeRecord> {
    let object = row.as_object()?;
    Some(GraphEdgeRecord {
        source_id: object.get("source_id")?.as_str()?.to_owned(),
        target_id: object.get("target_id")?.as_str()?.to_owned(),
        edge_type: object
            .get("edge_type")
            .and_then(Value::as_str)
            .unwrap_or("edge")
            .to_owned(),
        weight: object.get("weight").and_then(Value::as_i64).unwrap_or(1),
        attributes: object.get("attributes").cloned().unwrap_or(Value::Null),
        data: object.get("data").cloned().filter(|value| !value.is_null()),
        document_id: object
            .get("document_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                object
                    .get("attributes")
                    .and_then(Value::as_object)
                    .and_then(|attributes| attributes.get("documentId"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }),
        narrative_id: object
            .get("narrative_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                object
                    .get("attributes")
                    .and_then(Value::as_object)
                    .and_then(|attributes| attributes.get("narrativeId"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }),
        layer,
    })
}

fn graph_edge_record_to_row_value(record: GraphEdgeRecord) -> Value {
    let assertion_kind = match record.layer {
        GraphLayer::Asserted => "asserted",
        GraphLayer::Candidate => "candidate",
    };
    json!({
        "source_id": record.source_id,
        "target_id": record.target_id,
        "edge_type": record.edge_type,
        "document_id": record.document_id,
        "narrative_id": record.narrative_id,
        "valid_from_doc": record.document_id,
        "valid_from_boundary": null,
        "valid_to_doc": null,
        "valid_to_boundary": null,
        "assertion_kind": assertion_kind,
        "weight": record.weight,
        "attributes": record.attributes,
        "data": record.data,
    })
}

fn graph_vertex_to_row_value(vertex: &phoenix_graph::GraptorVertex) -> Value {
    json!({
        "id": vertex.id,
        "document_id": vertex.document_id,
        "narrative_id": vertex
            .value
            .get("narrativeId")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                vertex
                    .attributes
                    .get("narrativeId")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }),
        "value": vertex.value,
        "weight": vertex.weight,
        "attributes": vertex.attributes,
    })
}

fn project_graph_vertices(graph: &GraptorGraph) -> Vec<Value> {
    let mut vertex_ids = graph.vertices.keys().cloned().collect::<Vec<_>>();
    vertex_ids.sort();
    vertex_ids
        .into_iter()
        .filter_map(|vertex_id| graph.vertices.get(&vertex_id))
        .map(graph_vertex_to_row_value)
        .collect()
}

fn project_graph_vertex_labels(graph: &GraptorGraph) -> Vec<Value> {
    let mut rows = Vec::with_capacity(graph.vertices.len());
    let mut vertex_ids = graph.vertices.keys().cloned().collect::<Vec<_>>();
    vertex_ids.sort();
    for vertex_id in vertex_ids {
        let Some(vertex) = graph.vertices.get(&vertex_id) else {
            continue;
        };
        rows.push(json!({
            "vertex_id": vertex.id,
            "label": vertex
                .value
                .get("label")
                .and_then(Value::as_str)
                .unwrap_or(vertex.kind.as_str()),
        }));
    }
    rows
}

fn project_graph_edges(graph: &GraptorGraph, layer: GraphLayer) -> Vec<Value> {
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for edges in graph.outgoing.values() {
        for edge in edges {
            if edge.layer != layer {
                continue;
            }
            let key = (
                edge.source_id.clone(),
                edge.target_id.clone(),
                edge.edge_type.clone(),
            );
            if !seen.insert(key) {
                continue;
            }
            rows.push(graph_edge_record_to_row_value(GraphEdgeRecord::from(edge)));
        }
    }
    rows
}

fn project_graph_node_index(graph: &GraptorGraph) -> Vec<Value> {
    let mut vertex_ids = graph.vertices.keys().cloned().collect::<Vec<_>>();
    vertex_ids.sort();
    vertex_ids
        .into_iter()
        .enumerate()
        .map(|(index, vertex_id)| {
            json!({
                "id": vertex_id,
                "idx": index as i64,
            })
        })
        .collect()
}

fn project_graph_properties(graph: &GraptorGraph) -> Vec<Value> {
    let mut rows = Vec::new();
    for vertex in graph.vertices.values() {
        if let Some(attributes) = vertex.attributes.as_object() {
            for (key, value) in attributes {
                rows.push(json!({
                    "owner_id": vertex.id,
                    "owner_type": "vertex",
                    "key": key,
                    "valid_from": 0,
                    "value_type": graph_property_value_type(value),
                    "value_blob": value,
                    "valid_until": null,
                    "txn_id": 0,
                }));
            }
        }
    }
    for edges in graph.outgoing.values() {
        for edge in edges {
            if let Some(attributes) = edge.attributes.as_object() {
                let owner_id =
                    format!("{}::{}::{}", edge.source_id, edge.edge_type, edge.target_id);
                for (key, value) in attributes {
                    rows.push(json!({
                        "owner_id": owner_id,
                        "owner_type": "edge",
                        "key": key,
                        "valid_from": 0,
                        "value_type": graph_property_value_type(value),
                        "value_blob": value,
                        "valid_until": null,
                        "txn_id": 0,
                    }));
                }
            }
        }
    }
    rows
}

fn graph_property_value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_i64() || number.is_u64() => "int",
        Value::Number(_) => "float",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn native_candidate_definition_row_id(scope_key: &str) -> String {
    format!("phoenix.graph.candidate::{scope_key}")
}

fn native_candidate_scope_key_for_document(document_id: &str) -> String {
    format!("document:{document_id}")
}

fn native_candidate_scope_key_for_node(node_id: &str) -> String {
    format!("node:{node_id}")
}

fn native_candidate_scope_key_for_row(row: &Value) -> Option<String> {
    row.as_object().and_then(|object| {
        runtime_row_document_id(object)
            .map(|document_id| native_candidate_scope_key_for_document(&document_id))
            .or_else(|| {
                object
                    .get("source_id")
                    .and_then(Value::as_str)
                    .map(native_candidate_scope_key_for_node)
            })
    })
}

fn native_candidate_batches_from_rows(
    rows: Vec<Value>,
    touched_scope_keys: &BTreeSet<String>,
) -> Vec<GraphMutationBatch> {
    let mut rows_by_scope = BTreeMap::<String, Vec<Value>>::new();
    for scope_key in touched_scope_keys {
        rows_by_scope.entry(scope_key.clone()).or_default();
    }
    for row in rows {
        let Some(scope_key) = native_candidate_scope_key_for_row(&row) else {
            continue;
        };
        if touched_scope_keys.contains(&scope_key) {
            rows_by_scope.entry(scope_key).or_default().push(row);
        }
    }
    touched_scope_keys
        .iter()
        .map(|scope_key| GraphMutationBatch {
            layer: GraphLayer::Candidate,
            scope: GraphMutationScope::Candidate {
                scope_key: scope_key.clone(),
            },
            vertices: Vec::new(),
            edges: rows_by_scope
                .remove(scope_key)
                .unwrap_or_default()
                .into_iter()
                .filter_map(|row| graph_edge_record_from_row_value(&row, GraphLayer::Candidate))
                .collect(),
        })
        .collect()
}

#[allow(dead_code)]
fn phase1_thread_matches_session(session: &SessionRecord, thread: &Thread) -> bool {
    phase1_scope_field_matches(session.scope.world_id.as_deref(), &thread.world_id)
        && phase1_scope_field_matches(session.scope.narrative_id.as_deref(), &thread.narrative_id)
}

#[allow(dead_code)]
fn phase1_scope_field_matches(expected: Option<&str>, actual: &str) -> bool {
    match expected.filter(|value| !value.is_empty()) {
        Some(expected) => actual == expected,
        None => actual.is_empty(),
    }
}

#[allow(dead_code)]
fn phase1_message_document_links(
    session_state: &SessionState,
    graph: &GraptorGraph,
) -> HashMap<String, BTreeSet<String>> {
    let allowed_documents = session_state
        .documents
        .iter()
        .map(|document| document.document_id.0.clone())
        .collect::<BTreeSet<_>>();
    let mut links = HashMap::<String, BTreeSet<String>>::new();
    for vertex in graph.vertices.values() {
        if vertex.kind != "thread_document" {
            continue;
        }
        let Some(document_id) = vertex.document_id.as_ref().cloned().or_else(|| {
            vertex
                .value
                .get("documentId")
                .and_then(Value::as_str)
                .map(str::to_owned)
        }) else {
            continue;
        };
        if !allowed_documents.contains(&document_id) {
            continue;
        }
        let Some(message_ids) = vertex
            .attributes
            .get("messageIds")
            .and_then(Value::as_array)
        else {
            continue;
        };
        for message_id in message_ids.iter().filter_map(Value::as_str) {
            links
                .entry(message_id.to_owned())
                .or_default()
                .insert(document_id.clone());
        }
    }
    links
}

#[allow(dead_code)]
fn phase1_document_links_for_message(
    message_id: &str,
    message_document_links: &HashMap<String, BTreeSet<String>>,
    singleton_document_id: Option<&str>,
) -> Vec<String> {
    if let Some(document_ids) = message_document_links.get(message_id) {
        return document_ids.iter().cloned().collect();
    }
    singleton_document_id
        .map(|document_id| vec![document_id.to_owned()])
        .unwrap_or_default()
}

#[allow(dead_code)]
fn phase1_graph_metadata(evidence_refs: Vec<String>) -> Value {
    json!({
        "layer": "asserted",
        "status": "asserted",
        "resolver": "phoenix-runtime/native",
        "confidence": 1.0,
        "evidence_refs": evidence_refs,
    })
}

#[allow(dead_code)]
fn phase1_vertex_row(
    id: &str,
    kind: &str,
    label: &str,
    document_id: Option<&str>,
    narrative_id: Option<&str>,
    mut attributes: Map<String, Value>,
    evidence_refs: Vec<String>,
) -> Value {
    if let Some(document_id) = document_id {
        attributes.insert("documentId".to_owned(), json!(document_id));
    }
    attributes.insert("graph".to_owned(), phase1_graph_metadata(evidence_refs));
    json!({
        "id": id,
        "document_id": document_id,
        "narrative_id": narrative_id,
        "value": {
            "kind": kind,
            "label": label,
        },
        "weight": 1,
        "attributes": Value::Object(attributes),
    })
}

#[allow(dead_code)]
fn phase1_time_vertex_row(
    id: &str,
    session_id: &SessionId,
    source_kind: &str,
    source_id: &str,
    timestamp_ms: i64,
    narrative_id: Option<&str>,
    evidence_refs: Vec<String>,
) -> Value {
    let mut attributes = Map::new();
    attributes.insert("sessionId".to_owned(), json!(session_id.0));
    attributes.insert("sourceKind".to_owned(), json!(source_kind));
    attributes.insert("sourceId".to_owned(), json!(source_id));
    attributes.insert("timestampMs".to_owned(), json!(timestamp_ms));
    phase1_vertex_row(
        id,
        "time_window",
        &format!("{source_kind}@{timestamp_ms}"),
        None,
        narrative_id,
        attributes,
        evidence_refs,
    )
}

#[allow(dead_code)]
fn phase1_edge_row(
    source_id: &str,
    target_id: &str,
    edge_type: &str,
    document_id: Option<&str>,
    narrative_id: Option<&str>,
    mut attributes: Map<String, Value>,
    evidence_refs: Vec<String>,
) -> Value {
    if let Some(document_id) = document_id {
        attributes
            .entry("documentId".to_owned())
            .or_insert_with(|| json!(document_id));
    }
    attributes.insert("graph".to_owned(), phase1_graph_metadata(evidence_refs));
    json!({
        "source_id": source_id,
        "target_id": target_id,
        "document_id": document_id,
        "narrative_id": narrative_id,
        "valid_from_doc": document_id,
        "valid_from_boundary": null,
        "valid_to_doc": null,
        "valid_to_boundary": null,
        "assertion_kind": "asserted",
        "weight": 1,
        "attributes": Value::Object(attributes),
        "data": null,
        "edge_type": edge_type,
    })
}

#[allow(dead_code)]
fn phase1_push_vertex(
    vertex_ids: &mut BTreeSet<String>,
    vertex_rows: &mut Vec<Value>,
    label_rows: &mut Vec<Value>,
    row: Value,
) {
    let Some(id) = row.get("id").and_then(Value::as_str).map(str::to_owned) else {
        return;
    };
    if !vertex_ids.insert(id.clone()) {
        return;
    }
    let label = row
        .get("value")
        .and_then(Value::as_object)
        .and_then(|value| value.get("label"))
        .and_then(Value::as_str)
        .unwrap_or(id.as_str())
        .to_owned();
    label_rows.push(json!({
        "vertex_id": id,
        "label": label,
    }));
    vertex_rows.push(row);
}

#[allow(dead_code)]
fn phase1_push_edge(
    edge_pairs: &mut BTreeSet<(String, String)>,
    edge_rows: &mut Vec<Value>,
    row: Value,
) {
    let Some(source_id) = row
        .get("source_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return;
    };
    let Some(target_id) = row
        .get("target_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return;
    };
    if edge_pairs.insert((source_id, target_id)) {
        edge_rows.push(row);
    }
}

const PHASE2_DOC_SIMILAR_TO_THRESHOLD: f64 = 0.72;
const PHASE2_ENTITY_COREF_THRESHOLD: f64 = 0.78;
const PHASE2_EVENT_MATCH_THRESHOLD: f64 = 0.74;
const PHASE2_ABOUT_THRESHOLD: f64 = 0.66;
const PHASE2_RELEVANT_TO_THRESHOLD: f64 = 0.62;
const PHASE2_EMBEDDING_RESOLVER: &str = "phoenix-runtime/embedding-candidate";
const SEMANTIC_NLI_MODEL_ID: &str = "onnx-community/ModernBERT-base-nli-ONNX";
const PHASE2_NLI_RESOLVER: &str = "phoenix-runtime/nli-edge-judge";
const PHASE2_NLI_MAX_INPUTS: usize = 256;
const PHASE2_NLI_MAX_EVIDENCE_TARGETS: usize = 4;
const PHASE2_NLI_MAX_LEAF_EVIDENCE_PER_SOURCE: usize = 3;
const PHASE2_NLI_COREF_THRESHOLD: f64 = 0.68;
const PHASE2_NLI_EVENT_THRESHOLD: f64 = 0.66;
const PHASE2_NLI_ABOUT_THRESHOLD: f64 = 0.58;
const PHASE2_NLI_RELEVANT_TO_THRESHOLD: f64 = 0.55;
const PHASE2_NLI_SUPPORTED_BY_THRESHOLD: f64 = 0.62;
const PHASE2_NLI_CONTRADICTED_BY_THRESHOLD: f64 = 0.66;

fn phase2_leaf_vertex_id(span_id: &str) -> String {
    format!("leaf::{span_id}")
}

fn phase2_leaf_context_map(
    leaf_chunks: &[phoenix_store_cozo::SemanticLeafChunk],
) -> HashMap<String, Phase2LeafContext> {
    leaf_chunks
        .iter()
        .map(|leaf| {
            (
                phase2_leaf_vertex_id(&leaf.span_id),
                Phase2LeafContext {
                    document_id: leaf.document_id.clone(),
                    narrative_id: leaf.narrative_id.clone(),
                    folder_id: leaf.folder_id.clone(),
                    text: leaf.text.clone(),
                },
            )
        })
        .collect()
}

fn phase2_document_scope_from_leaf_chunks(
    leaf_chunks: &[phoenix_store_cozo::SemanticLeafChunk],
) -> HashMap<String, ScopeKey> {
    let mut scopes = HashMap::new();
    for leaf in leaf_chunks {
        scopes
            .entry(leaf.document_id.clone())
            .or_insert_with(|| ScopeKey {
                world_id: None,
                narrative_id: leaf.narrative_id.clone(),
                folder_id: leaf.folder_id.clone(),
                folder_path: None,
            });
    }
    scopes
}

fn phase2_document_support_from_leaf_chunks(
    leaf_chunks: &[phoenix_store_cozo::SemanticLeafChunk],
) -> HashMap<String, Vec<String>> {
    let mut support = HashMap::<String, Vec<String>>::new();
    for leaf in leaf_chunks {
        let entry = support.entry(leaf.document_id.clone()).or_default();
        if entry.len() < 3 {
            let text = leaf.text.trim();
            if !text.is_empty() {
                entry.push(text.to_owned());
            }
        }
    }
    support
}

fn phase2_thread_message_map(
    store: &PhoenixCozoStore,
) -> Result<HashMap<String, ThreadMessage>, StoreError> {
    let mut messages = HashMap::new();
    for row in store.fetch_rows("thread_messages")? {
        let Ok(message) = serde_json::from_value::<ThreadMessage>(row) else {
            continue;
        };
        messages.insert(message.id.clone(), message);
    }
    Ok(messages)
}

fn phase2_document_vector_map(
    store: &PhoenixCozoStore,
    document_ids: &[String],
) -> Result<HashMap<String, StoredSemanticDocumentVector>, StoreError> {
    let allowed = document_ids.iter().cloned().collect::<HashSet<_>>();
    let mut vectors = HashMap::new();
    for row in store.fetch_rows("semantic_documents")? {
        let Some(document_id) = row.get("document_id").and_then(Value::as_str) else {
            continue;
        };
        if !allowed.is_empty() && !allowed.contains(document_id) {
            continue;
        }
        let Some(values) = phase2_json_vector(row.get("vec")) else {
            continue;
        };
        vectors.insert(
            document_id.to_owned(),
            StoredSemanticDocumentVector {
                values,
                evidence_refs: phase2_json_string_array(row.get("evidence_refs")),
            },
        );
    }
    Ok(vectors)
}

fn phase2_node_vector_map(
    store: &PhoenixCozoStore,
    node_ids: &[String],
) -> Result<HashMap<String, StoredSemanticNodeVector>, StoreError> {
    let allowed = node_ids.iter().cloned().collect::<HashSet<_>>();
    let mut vectors = HashMap::new();
    for row in store.fetch_rows("semantic_node_prototypes")? {
        let Some(node_id) = row.get("node_id").and_then(Value::as_str) else {
            continue;
        };
        if !allowed.is_empty() && !allowed.contains(node_id) {
            continue;
        }
        let Some(values) = phase2_json_vector(row.get("vec")) else {
            continue;
        };
        vectors.insert(
            node_id.to_owned(),
            StoredSemanticNodeVector {
                node_id: node_id.to_owned(),
                node_kind: row
                    .get("node_kind")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                document_id: row
                    .get("document_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                narrative_id: row
                    .get("narrative_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                folder_id: row
                    .get("folder_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                values,
                evidence_refs: phase2_json_string_array(row.get("evidence_refs")),
            },
        );
    }
    Ok(vectors)
}

fn phase2_entity_prototype_input(
    graph: &GraptorGraph,
    leaf_context: &HashMap<String, Phase2LeafContext>,
    document_scopes: &HashMap<String, ScopeKey>,
    entity_id: &str,
) -> Option<SemanticCandidatePrototypeInput> {
    let vertex = graph.vertices.get(entity_id)?;
    let label = vertex
        .value
        .get("label")
        .and_then(Value::as_str)
        .unwrap_or(entity_id)
        .trim()
        .to_owned();
    if label.is_empty() {
        return None;
    }

    let aliases = phase2_json_string_array(vertex.attributes.get("aliases"));
    let mut support = BTreeSet::new();
    let mut evidence_refs = phase2_graph_evidence_refs(&vertex.attributes);
    let mut document_id = vertex.document_id.clone();
    let mut narrative_id = None;
    let mut folder_id = None;

    for edge in graph
        .outgoing_any(entity_id)
        .chain(graph.incoming_any(entity_id))
    {
        if edge.edge_type != "mentions" {
            continue;
        }
        let neighbor_id = if edge.source_id == entity_id {
            edge.target_id.as_str()
        } else {
            edge.source_id.as_str()
        };
        if let Some(context) = leaf_context.get(neighbor_id) {
            support.insert(context.text.clone());
            evidence_refs.push(format!("graph_vertex:{neighbor_id}"));
            if document_id.is_none() {
                document_id = Some(context.document_id.clone());
            }
            if narrative_id.is_none() {
                narrative_id = context.narrative_id.clone();
            }
            if folder_id.is_none() {
                folder_id = context.folder_id.clone();
            }
        }
    }

    if let Some(scope) = document_id
        .as_ref()
        .and_then(|document_id| document_scopes.get(document_id))
    {
        if narrative_id.is_none() {
            narrative_id = scope.narrative_id.clone();
        }
        if folder_id.is_none() {
            folder_id = scope.folder_id.clone();
        }
    }

    let mut sections = vec![format!("entity: {label}")];
    if !aliases.is_empty() {
        sections.push(format!("aliases: {}", aliases.join(", ")));
    }
    if !support.is_empty() {
        sections.push(format!(
            "support: {}",
            support.into_iter().take(3).collect::<Vec<_>>().join(" | ")
        ));
    }

    Some(SemanticCandidatePrototypeInput {
        node_id: entity_id.to_owned(),
        node_kind: "entity".to_owned(),
        document_id,
        narrative_id,
        folder_id,
        text: sections.join("\n"),
        evidence_refs: phase2_unique_strings(evidence_refs),
    })
}

fn phase2_normalize_entity_surface(value: &str) -> Option<String> {
    let normalized = value
        .trim()
        .to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_alphanumeric() { ch } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    (!normalized.is_empty()).then_some(normalized)
}

fn phase2_entity_surface_set(graph: &GraptorGraph, entity_id: &str) -> BTreeSet<String> {
    let mut surfaces = BTreeSet::new();
    let Some(vertex) = graph.vertices.get(entity_id) else {
        return surfaces;
    };
    if let Some(label) = vertex.value.get("label").and_then(Value::as_str) {
        if let Some(normalized) = phase2_normalize_entity_surface(label) {
            surfaces.insert(normalized);
        }
    }
    for alias in phase2_json_string_array(vertex.attributes.get("aliases")) {
        if let Some(normalized) = phase2_normalize_entity_surface(&alias) {
            surfaces.insert(normalized);
        }
    }
    surfaces
}

fn phase2_entity_candidate_is_coherent(
    graph: &GraptorGraph,
    source_id: &str,
    target_id: &str,
) -> bool {
    let source_surfaces = phase2_entity_surface_set(graph, source_id);
    let target_surfaces = phase2_entity_surface_set(graph, target_id);
    !source_surfaces.is_empty() && !target_surfaces.is_empty() && !source_surfaces.is_disjoint(&target_surfaces)
}

fn phase2_event_prototype_input(
    graph: &GraptorGraph,
    leaf_context: &HashMap<String, Phase2LeafContext>,
    document_scopes: &HashMap<String, ScopeKey>,
    event_id: &str,
) -> Option<SemanticCandidatePrototypeInput> {
    let vertex = graph.vertices.get(event_id)?;
    let lemma = vertex
        .value
        .get("lemma")
        .and_then(Value::as_str)
        .unwrap_or(event_id)
        .trim()
        .to_owned();
    if lemma.is_empty() {
        return None;
    }

    let relation_type = vertex
        .value
        .get("relationType")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();
    let event_class = vertex
        .value
        .get("eventClass")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_owned();

    let mut subjects = BTreeSet::new();
    let mut objects = BTreeSet::new();
    let mut recipients = BTreeSet::new();
    let mut support = BTreeSet::new();
    let mut evidence_refs = phase2_graph_evidence_refs(&vertex.attributes);
    let mut document_id = vertex.document_id.clone();
    let mut narrative_id = None;
    let mut folder_id = None;

    for edge in graph.incoming_any(event_id) {
        match edge.edge_type.as_str() {
            "has_event" => {
                if let Some(context) = leaf_context.get(edge.source_id.as_str()) {
                    support.insert(context.text.clone());
                    evidence_refs.push(format!("graph_vertex:{}", edge.source_id));
                    if document_id.is_none() {
                        document_id = Some(context.document_id.clone());
                    }
                    if narrative_id.is_none() {
                        narrative_id = context.narrative_id.clone();
                    }
                    if folder_id.is_none() {
                        folder_id = context.folder_id.clone();
                    }
                }
            }
            "event_subject" => {
                if let Some(label) = phase2_vertex_label(graph, &edge.source_id) {
                    subjects.insert(label);
                }
            }
            _ => {}
        }
    }
    for edge in graph.outgoing_any(event_id) {
        match edge.edge_type.as_str() {
            "event_object" => {
                if let Some(label) = phase2_vertex_label(graph, &edge.target_id) {
                    objects.insert(label);
                }
            }
            "event_recipient" => {
                if let Some(label) = phase2_vertex_label(graph, &edge.target_id) {
                    recipients.insert(label);
                }
            }
            _ => {}
        }
    }

    if let Some(scope) = document_id
        .as_ref()
        .and_then(|document_id| document_scopes.get(document_id))
    {
        if narrative_id.is_none() {
            narrative_id = scope.narrative_id.clone();
        }
        if folder_id.is_none() {
            folder_id = scope.folder_id.clone();
        }
    }

    let mut sections = vec![format!("event: {lemma}")];
    if !relation_type.is_empty() {
        sections.push(format!("relation: {relation_type}"));
    }
    if !event_class.is_empty() {
        sections.push(format!("class: {event_class}"));
    }
    if !subjects.is_empty() {
        sections.push(format!(
            "subjects: {}",
            subjects.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    if !objects.is_empty() {
        sections.push(format!(
            "objects: {}",
            objects.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    if !recipients.is_empty() {
        sections.push(format!(
            "recipients: {}",
            recipients.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }
    if !support.is_empty() {
        sections.push(format!(
            "support: {}",
            support.into_iter().take(3).collect::<Vec<_>>().join(" | ")
        ));
    }

    Some(SemanticCandidatePrototypeInput {
        node_id: event_id.to_owned(),
        node_kind: "event".to_owned(),
        document_id,
        narrative_id,
        folder_id,
        text: sections.join("\n"),
        evidence_refs: phase2_unique_strings(evidence_refs),
    })
}

fn phase2_turn_prototype_input(
    vertex: &phoenix_graptor::GraptorVertex,
    messages: &HashMap<String, ThreadMessage>,
    document_scopes: &HashMap<String, ScopeKey>,
) -> Option<SemanticCandidatePrototypeInput> {
    let message_id = vertex
        .attributes
        .get("messageId")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| vertex.id.strip_prefix("turn::").map(str::to_owned))?;
    let message = messages.get(&message_id)?;
    let document_id = vertex.document_id.clone();
    let (narrative_id, folder_id) =
        phase2_scope_for_document(document_scopes, document_id.as_deref());
    Some(SemanticCandidatePrototypeInput {
        node_id: vertex.id.clone(),
        node_kind: "turn".to_owned(),
        document_id,
        narrative_id: if !message.narrative_id.trim().is_empty() {
            Some(message.narrative_id.clone())
        } else {
            narrative_id
        },
        folder_id,
        text: format!("turn [{}]: {}", message.role, message.content.trim()),
        evidence_refs: phase2_unique_strings(vec![
            format!("thread_message:{}", message.id),
            format!("graph_vertex:{}", vertex.id),
        ]),
    })
}

fn phase2_task_prototype_input(
    vertex: &phoenix_graptor::GraptorVertex,
    document_scopes: &HashMap<String, ScopeKey>,
) -> Option<SemanticCandidatePrototypeInput> {
    let task = vertex
        .attributes
        .get("currentTask")
        .and_then(Value::as_str)
        .or_else(|| vertex.value.get("label").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())?
        .to_owned();
    let document_id = vertex.document_id.clone();
    let (narrative_id, folder_id) =
        phase2_scope_for_document(document_scopes, document_id.as_deref());
    Some(SemanticCandidatePrototypeInput {
        node_id: vertex.id.clone(),
        node_kind: "task".to_owned(),
        document_id,
        narrative_id,
        folder_id,
        text: format!("task: {task}"),
        evidence_refs: phase2_unique_strings(phase2_graph_evidence_refs(&vertex.attributes)),
    })
}

fn phase2_state_prototype_input(
    vertex: &phoenix_graptor::GraptorVertex,
    document_scopes: &HashMap<String, ScopeKey>,
) -> Option<SemanticCandidatePrototypeInput> {
    let label = vertex
        .value
        .get("label")
        .and_then(Value::as_str)
        .or_else(|| vertex.attributes.get("kind").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())?
        .to_owned();
    let document_id = vertex.document_id.clone();
    let (narrative_id, folder_id) =
        phase2_scope_for_document(document_scopes, document_id.as_deref());
    let phase = vertex
        .attributes
        .get("phase")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let kind = vertex
        .attributes
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let status = vertex
        .attributes
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut parts = vec![format!("state: {label}")];
    if !phase.is_empty() || !kind.is_empty() || !status.is_empty() {
        parts.push(
            [phase, kind, status]
                .into_iter()
                .filter(|value| !value.is_empty())
                .collect::<Vec<_>>()
                .join(" / "),
        );
    }
    Some(SemanticCandidatePrototypeInput {
        node_id: vertex.id.clone(),
        node_kind: "state".to_owned(),
        document_id,
        narrative_id,
        folder_id,
        text: parts.join("\n"),
        evidence_refs: phase2_unique_strings(phase2_graph_evidence_refs(&vertex.attributes)),
    })
}

fn phase2_document_prototype_input(
    graph: &GraptorGraph,
    document_support: &HashMap<String, Vec<String>>,
    document_id: &str,
) -> Option<SemanticCandidatePrototypeInput> {
    let vertex_id = phase1_document_vertex_id(document_id);
    let vertex = graph.vertices.get(&vertex_id)?;
    let label = phase2_vertex_label(graph, &vertex_id).unwrap_or_else(|| document_id.to_owned());
    let mut parts = vec![format!("document: {label}")];
    if let Some(support) = document_support.get(document_id) {
        if !support.is_empty() {
            parts.push(format!("support: {}", support.join(" | ")));
        }
    }
    Some(SemanticCandidatePrototypeInput {
        node_id: vertex.id.clone(),
        node_kind: "document".to_owned(),
        document_id: Some(document_id.to_owned()),
        narrative_id: vertex
            .attributes
            .get("narrativeId")
            .and_then(Value::as_str)
            .map(str::to_owned),
        folder_id: vertex
            .attributes
            .get("folderId")
            .and_then(Value::as_str)
            .map(str::to_owned),
        text: parts.join("\n"),
        evidence_refs: phase2_unique_strings(vec![format!("graph_vertex:{}", vertex.id)]),
    })
}

fn phase2_leaf_prototype_input(
    leaf_context: &HashMap<String, Phase2LeafContext>,
    vertex_id: &str,
) -> Option<SemanticCandidatePrototypeInput> {
    let context = leaf_context.get(vertex_id)?;
    Some(SemanticCandidatePrototypeInput {
        node_id: vertex_id.to_owned(),
        node_kind: "leaf".to_owned(),
        document_id: Some(context.document_id.clone()),
        narrative_id: context.narrative_id.clone(),
        folder_id: context.folder_id.clone(),
        text: format!("leaf: {}", context.text.trim()),
        evidence_refs: vec![format!("graph_vertex:{vertex_id}")],
    })
}

fn phase2_nli_profile_for_vertex(
    graph: &GraptorGraph,
    leaf_context: &HashMap<String, Phase2LeafContext>,
    document_scopes: &HashMap<String, ScopeKey>,
    document_support: &HashMap<String, Vec<String>>,
    messages: &HashMap<String, ThreadMessage>,
    vertex_id: &str,
    cache: &mut HashMap<String, SemanticCandidatePrototypeInput>,
) -> Option<SemanticCandidatePrototypeInput> {
    if let Some(existing) = cache.get(vertex_id) {
        return Some(existing.clone());
    }
    let profile = if let Some(document_id) = vertex_id.strip_prefix("doc::") {
        phase2_document_prototype_input(graph, document_support, document_id)
    } else if vertex_id.starts_with("leaf::") {
        phase2_leaf_prototype_input(leaf_context, vertex_id)
    } else {
        let vertex = graph.vertices.get(vertex_id)?;
        match vertex.kind.as_str() {
            "entity" => {
                phase2_entity_prototype_input(graph, leaf_context, document_scopes, vertex_id)
            }
            "event" => {
                phase2_event_prototype_input(graph, leaf_context, document_scopes, vertex_id)
            }
            "turn" => phase2_turn_prototype_input(vertex, messages, document_scopes),
            "task" => phase2_task_prototype_input(vertex, document_scopes),
            "state" => phase2_state_prototype_input(vertex, document_scopes),
            "document" => vertex.document_id.as_deref().and_then(|document_id| {
                phase2_document_prototype_input(graph, document_support, document_id)
            }),
            "leaf" => phase2_leaf_prototype_input(leaf_context, vertex_id),
            _ => None,
        }
    }?;
    cache.insert(vertex_id.to_owned(), profile.clone());
    Some(profile)
}

fn phase2_scope_for_document(
    document_scopes: &HashMap<String, ScopeKey>,
    document_id: Option<&str>,
) -> (Option<String>, Option<String>) {
    let Some(document_id) = document_id else {
        return (None, None);
    };
    let Some(scope) = document_scopes.get(document_id) else {
        return (None, None);
    };
    (scope.narrative_id.clone(), scope.folder_id.clone())
}

fn phase2_vertex_label(graph: &GraptorGraph, vertex_id: &str) -> Option<String> {
    let vertex = graph.vertices.get(vertex_id)?;
    vertex
        .value
        .get("label")
        .and_then(Value::as_str)
        .or_else(|| vertex.value.get("lemma").and_then(Value::as_str))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn phase2_json_string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn phase2_json_vector(value: Option<&Value>) -> Option<Vec<f32>> {
    let values = value?.as_array()?;
    let vector = values
        .iter()
        .map(|value| value.as_f64().map(|value| value as f32))
        .collect::<Option<Vec<_>>>()?;
    (!vector.is_empty()).then_some(vector)
}

fn phase2_graph_evidence_refs(attributes: &Value) -> Vec<String> {
    attributes
        .get("graph")
        .and_then(Value::as_object)
        .and_then(|graph| graph.get("evidence_refs"))
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn phase2_candidate_graph_status(attributes: &Value) -> Option<&str> {
    attributes
        .get("graph")
        .and_then(Value::as_object)
        .and_then(|graph| graph.get("status"))
        .and_then(Value::as_str)
}

fn phase2_candidate_row_is_active(attributes: &Value) -> bool {
    !matches!(
        phase2_candidate_graph_status(attributes),
        Some("candidate_rejected")
    )
}

fn phase2_candidate_row_base_score(data: Option<&Value>, attributes: Option<&Value>) -> f64 {
    data.and_then(|value| value.get("base"))
        .and_then(|value| value.get("score"))
        .and_then(Value::as_f64)
        .or_else(|| {
            data.and_then(|value| value.get("score"))
                .and_then(Value::as_f64)
        })
        .or_else(|| {
            attributes
                .and_then(|value| value.get("score"))
                .and_then(Value::as_f64)
        })
        .unwrap_or(0.0)
}

fn phase2_similarity_score(distance: f64) -> f64 {
    1.0 / (1.0 + distance.max(0.0))
}

fn phase2_merge_evidence_refs(left: &[String], right: &[String]) -> Vec<String> {
    phase2_unique_strings(left.iter().chain(right.iter()).cloned().collect::<Vec<_>>())
}

fn phase2_unique_strings(values: Vec<String>) -> Vec<String> {
    let mut unique = BTreeSet::new();
    for value in values {
        if !value.trim().is_empty() {
            unique.insert(value);
        }
    }
    unique.into_iter().collect()
}

fn phase2_tokenize_terms(text: &str) -> BTreeSet<String> {
    text.split(|ch: char| !ch.is_alphanumeric())
        .map(|part| part.trim().to_ascii_lowercase())
        .filter(|part| part.len() >= 3)
        .collect()
}

fn phase2_text_overlap_score(left: &str, right: &str) -> usize {
    let left_terms = phase2_tokenize_terms(left);
    if left_terms.is_empty() {
        return 0;
    }
    let right_terms = phase2_tokenize_terms(right);
    left_terms.intersection(&right_terms).count()
}

fn phase2_nli_hypothesis(edge_type: &str, target_text: &str) -> String {
    match edge_type {
        "candidate_corefers_with" => {
            format!("This refers to the same entity as:\n{target_text}")
        }
        "candidate_same_event_as" => {
            format!("This describes the same event as:\n{target_text}")
        }
        "about" => format!("This text is about:\n{target_text}"),
        "relevant_to" => format!("This text is relevant to:\n{target_text}"),
        _ => target_text.to_owned(),
    }
}

fn phase2_nli_statement_hypothesis(statement_text: &str) -> String {
    format!("The following statement is true:\n{statement_text}")
}

fn phase2_nli_group_id(record: &Phase2CandidateEdgeRecord) -> String {
    match record.edge_type.as_str() {
        "candidate_corefers_with" | "candidate_same_event_as" => {
            let (left, right) = if record.source_id <= record.target_id {
                (record.source_id.as_str(), record.target_id.as_str())
            } else {
                (record.target_id.as_str(), record.source_id.as_str())
            };
            format!("{}::{left}::{right}", record.edge_type)
        }
        _ => format!(
            "{}::{}::{}",
            record.edge_type, record.source_id, record.target_id
        ),
    }
}

fn phase2_nli_inputs_for_edge(
    record: &Phase2CandidateEdgeRecord,
    source_text: &str,
    target_text: &str,
) -> Vec<SemanticNliJudgmentInput> {
    let group_id = phase2_nli_group_id(record);
    let mut inputs = vec![SemanticNliJudgmentInput {
        judgment_id: format!(
            "{}::{}::{}::forward",
            record.edge_type, record.source_id, record.target_id
        ),
        group_id: group_id.clone(),
        source_id: record.source_id.clone(),
        target_id: record.target_id.clone(),
        edge_type: record.edge_type.clone(),
        direction: "forward".to_owned(),
        premise: source_text.to_owned(),
        hypothesis: phase2_nli_hypothesis(&record.edge_type, target_text),
    }];

    if matches!(
        record.edge_type.as_str(),
        "candidate_corefers_with" | "candidate_same_event_as"
    ) {
        inputs.push(SemanticNliJudgmentInput {
            judgment_id: format!(
                "{}::{}::{}::reverse",
                record.edge_type, record.source_id, record.target_id
            ),
            group_id,
            source_id: record.source_id.clone(),
            target_id: record.target_id.clone(),
            edge_type: record.edge_type.clone(),
            direction: "reverse".to_owned(),
            premise: target_text.to_owned(),
            hypothesis: phase2_nli_hypothesis(&record.edge_type, source_text),
        });
    }

    inputs
}

fn phase2_nli_inputs_for_evidence_edge(
    source_id: &str,
    source_text: &str,
    target_id: &str,
    target_text: &str,
) -> Vec<SemanticNliJudgmentInput> {
    let hypothesis = phase2_nli_statement_hypothesis(source_text);
    ["supported_by", "contradicted_by"]
        .into_iter()
        .map(|edge_type| SemanticNliJudgmentInput {
            judgment_id: format!("{edge_type}::{source_id}::{target_id}::forward"),
            group_id: format!("{edge_type}::{source_id}::{target_id}"),
            source_id: source_id.to_owned(),
            target_id: target_id.to_owned(),
            edge_type: edge_type.to_owned(),
            direction: "forward".to_owned(),
            premise: target_text.to_owned(),
            hypothesis: hypothesis.clone(),
        })
        .collect()
}

fn phase2_nli_threshold(edge_type: &str) -> f64 {
    match edge_type {
        "candidate_corefers_with" => PHASE2_NLI_COREF_THRESHOLD,
        "candidate_same_event_as" => PHASE2_NLI_EVENT_THRESHOLD,
        "about" => PHASE2_NLI_ABOUT_THRESHOLD,
        "relevant_to" => PHASE2_NLI_RELEVANT_TO_THRESHOLD,
        "supported_by" => PHASE2_NLI_SUPPORTED_BY_THRESHOLD,
        "contradicted_by" => PHASE2_NLI_CONTRADICTED_BY_THRESHOLD,
        _ => 0.6,
    }
}

fn phase2_apply_nli_decision(
    edge_type: &str,
    base_score: f64,
    judgments: &[SemanticNliJudgmentResultRow],
) -> Phase2NliDecision {
    if judgments.is_empty() {
        return Phase2NliDecision::default();
    }

    let symmetric = matches!(
        edge_type,
        "candidate_corefers_with" | "candidate_same_event_as"
    );
    let entailment = if symmetric {
        judgments
            .iter()
            .map(|judgment| judgment.entailment)
            .fold(1.0_f64, |acc, value| acc.min(value))
    } else {
        judgments[0].entailment
    };
    let contradiction = judgments
        .iter()
        .map(|judgment| judgment.contradiction)
        .fold(0.0_f64, |acc, value| acc.max(value));
    let neutral = if symmetric {
        judgments
            .iter()
            .map(|judgment| judgment.neutral)
            .sum::<f64>()
            / judgments.len() as f64
    } else {
        judgments[0].neutral
    };
    let threshold = phase2_nli_threshold(edge_type);
    let (accepted, nli_score, final_score) = if edge_type == "contradicted_by" {
        let nli_score = (contradiction - entailment).max(0.0);
        let accepted = contradiction >= threshold
            && contradiction >= neutral
            && entailment <= (1.0 - threshold);
        let final_score = if accepted {
            ((base_score * 0.2) + (nli_score * 0.8)).clamp(0.0, 1.0)
        } else {
            (base_score * 0.2).clamp(0.0, 1.0)
        };
        (accepted, nli_score, final_score)
    } else {
        let nli_score = (entailment - contradiction).max(0.0);
        let accepted =
            entailment >= threshold && entailment >= neutral && contradiction <= (1.0 - threshold);
        let final_score = if accepted {
            ((base_score * 0.4) + (nli_score * 0.6)).clamp(0.0, 1.0)
        } else {
            (base_score * 0.35).clamp(0.0, 1.0)
        };
        (accepted, nli_score, final_score)
    };

    Phase2NliDecision {
        accepted,
        threshold,
        entailment,
        neutral,
        contradiction,
        nli_score,
        final_score,
    }
}

fn phase2_nli_evidence_refs(judgments: &[SemanticNliJudgmentResultRow]) -> Vec<String> {
    phase2_unique_strings(
        judgments
            .iter()
            .map(|judgment| format!("nli_judgment:{}", judgment.judgment_id))
            .collect::<Vec<_>>(),
    )
}

fn phase2_graph_has_edge(
    graph: &GraptorGraph,
    source_id: &str,
    target_id: &str,
    edge_type: &str,
) -> bool {
    graph
        .outgoing_matching(source_id, edge_type)
        .any(|edge| edge.target_id == target_id)
}

fn phase2_graph_has_symmetric_edge(
    graph: &GraptorGraph,
    left_id: &str,
    right_id: &str,
    edge_type: &str,
) -> bool {
    phase2_graph_has_edge(graph, left_id, right_id, edge_type)
        || phase2_graph_has_edge(graph, right_id, left_id, edge_type)
}

fn phase2_candidate_edge_row(
    source_id: &str,
    target_id: &str,
    edge_type: &str,
    document_id: Option<&str>,
    narrative_id: Option<&str>,
    score: f64,
    threshold: f64,
    mut attributes: Map<String, Value>,
    evidence_refs: Vec<String>,
) -> Value {
    if let Some(document_id) = document_id {
        attributes
            .entry("documentId".to_owned())
            .or_insert_with(|| json!(document_id));
    }
    attributes.insert("score".to_owned(), json!(score));
    attributes.insert("threshold".to_owned(), json!(threshold));
    attributes.insert(
        "graph".to_owned(),
        json!({
            "layer": "candidate",
            "status": "candidate",
            "resolver": PHASE2_EMBEDDING_RESOLVER,
            "confidence": score,
            "evidence_refs": evidence_refs,
        }),
    );
    json!({
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge_type,
        "document_id": document_id,
        "narrative_id": narrative_id,
        "valid_from_doc": document_id,
        "valid_from_boundary": null,
        "valid_to_doc": null,
        "valid_to_boundary": null,
        "assertion_kind": "candidate",
        "weight": ((score * 1000.0).round() as i64).max(1),
        "attributes": Value::Object(attributes),
        "data": {
            "base": {
                "score": score,
                "threshold": threshold,
                "resolver": PHASE2_EMBEDDING_RESOLVER,
            },
        },
    })
}

fn phase2_nli_candidate_edge_seed_row(
    graph: &GraptorGraph,
    source_id: &str,
    target_id: &str,
    edge_type: &str,
) -> Option<Value> {
    let source_vertex = graph.vertices.get(source_id)?;
    let target_vertex = graph.vertices.get(target_id)?;
    let document_id = source_vertex
        .document_id
        .clone()
        .or_else(|| target_vertex.document_id.clone());
    let narrative_id = source_vertex
        .attributes
        .get("narrativeId")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            target_vertex
                .attributes
                .get("narrativeId")
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
    let evidence_refs = phase2_unique_strings(vec![
        format!("graph_vertex:{source_id}"),
        format!("graph_vertex:{target_id}"),
    ]);
    Some(json!({
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": edge_type,
        "document_id": document_id,
        "narrative_id": narrative_id,
        "valid_from_doc": document_id,
        "valid_from_boundary": null,
        "valid_to_doc": null,
        "valid_to_boundary": null,
        "assertion_kind": "candidate",
        "weight": 0,
        "attributes": {
            "documentId": document_id,
            "sourceKind": source_vertex.kind,
            "targetKind": target_vertex.kind,
            "score": 0.0,
            "threshold": phase2_nli_threshold(edge_type),
            "graph": {
                "layer": "candidate",
                "status": "candidate",
                "resolver": PHASE2_NLI_RESOLVER,
                "confidence": 0.0,
                "evidence_refs": evidence_refs,
            }
        },
        "data": {
            "base": {
                "score": 0.0,
                "threshold": phase2_nli_threshold(edge_type),
                "resolver": PHASE2_NLI_RESOLVER,
            }
        }
    }))
}

fn phase2_push_candidate_edge(
    edge_keys: &mut BTreeSet<(String, String, String)>,
    edge_rows: &mut Vec<Value>,
    row: Value,
) {
    let Some(source_id) = row
        .get("source_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return;
    };
    let Some(target_id) = row
        .get("target_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return;
    };
    let Some(edge_type) = row
        .get("edge_type")
        .and_then(Value::as_str)
        .map(str::to_owned)
    else {
        return;
    };
    if edge_keys.insert((source_id, target_id, edge_type)) {
        edge_rows.push(row);
    }
}

#[allow(dead_code)]
fn phase1_turn_vertex_id(message_id: &str) -> String {
    format!("turn::{message_id}")
}

#[allow(dead_code)]
fn phase1_agent_vertex_id(session_id: &SessionId, role: &str) -> String {
    format!("agent::{}::{role}", session_id.0)
}

#[allow(dead_code)]
fn phase1_task_vertex_id(thread_id: &str) -> String {
    format!("task::{thread_id}::current")
}

#[allow(dead_code)]
fn phase1_state_vertex_id(event_id: &str) -> String {
    format!("state::{event_id}")
}

#[allow(dead_code)]
fn phase1_time_vertex_id(kind: &str, source_id: &str) -> String {
    format!("time::{kind}::{source_id}")
}

fn phase1_document_vertex_id(document_id: &str) -> String {
    format!("doc::{document_id}")
}

#[allow(dead_code)]
fn phase1_turn_label(message: &ThreadMessage) -> String {
    let snippet = phase1_snippet(&message.content, 72);
    if snippet.is_empty() {
        message.role.clone()
    } else {
        format!("{}: {snippet}", message.role)
    }
}

#[allow(dead_code)]
fn phase1_state_label(event: &ChatRunEvent) -> String {
    let mut label = event.label.trim().to_owned();
    if label.is_empty() {
        label = format!("{} {}", event.phase, event.kind).trim().to_owned();
    }
    if label.is_empty() {
        label = event.id.clone();
    }
    label
}

#[allow(dead_code)]
fn phase1_snippet(text: &str, limit: usize) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.len() <= limit {
        collapsed
    } else {
        let mut snippet = collapsed
            .chars()
            .take(limit.saturating_sub(1))
            .collect::<String>();
        snippet.push('…');
        snippet
    }
}

#[allow(dead_code)]
fn phase1_thread_narrative_id(thread: &Thread, session: &SessionRecord) -> Option<String> {
    if !thread.narrative_id.trim().is_empty() {
        return Some(thread.narrative_id.clone());
    }
    session
        .scope
        .narrative_id
        .as_ref()
        .filter(|value| !value.is_empty())
        .cloned()
}

#[allow(dead_code)]
fn phase1_narrative_id(
    thread: &Thread,
    message: &ThreadMessage,
    session: &SessionRecord,
) -> Option<String> {
    if !message.narrative_id.trim().is_empty() {
        return Some(message.narrative_id.clone());
    }
    phase1_thread_narrative_id(thread, session)
}

#[allow(dead_code)]
fn phase1_latest_message_at_or_before<'a>(
    messages: &'a [ThreadMessage],
    timestamp_ms: i64,
) -> Option<&'a ThreadMessage> {
    messages
        .iter()
        .rev()
        .find(|message| message.created_at <= timestamp_ms)
        .or_else(|| messages.last())
}

#[allow(dead_code)]
fn phase1_event_for_task<'a>(
    events: &'a [ChatRunEvent],
    timestamp_ms: i64,
) -> Option<&'a ChatRunEvent> {
    events
        .iter()
        .rev()
        .find(|event| event.created_at <= timestamp_ms)
        .or_else(|| events.last())
}

#[allow(dead_code)]
fn phase1_native_graph_row_matches_session(row: &Value, session_id: &SessionId) -> bool {
    row.get("attributes")
        .and_then(Value::as_object)
        .filter(|attributes| {
            attributes.get("sessionId").and_then(Value::as_str) == Some(session_id.0.as_str())
        })
        .and_then(|attributes| attributes.get("graph"))
        .and_then(Value::as_object)
        .and_then(|graph| graph.get("resolver"))
        .and_then(Value::as_str)
        == Some("phoenix-runtime/native")
}

#[allow(dead_code)]
fn om_record_from_value(row: Value) -> Result<OmRecord, StoreError> {
    let Some(object) = row.as_object() else {
        return Err(StoreError::Query(
            "OM record row should be an object.".to_owned(),
        ));
    };
    Ok(OmRecord {
        thread_id: object
            .get("thread_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        observations: object
            .get("observations")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        current_task: object
            .get("current_task")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        suggested_continuation: object
            .get("suggested_continuation")
            .and_then(Value::as_str)
            .map(str::to_owned),
        last_observed_at: object
            .get("last_observed_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        obs_token_count: object
            .get("obs_token_count")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        generation_num: object
            .get("generation_num")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        created_at: object
            .get("created_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        updated_at: object
            .get("updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
    })
}

const NOTE_KEY_COLUMNS: &[&str] = &["id", "version"];
const ALLOWED_WAL_RELATIONS: &[&str] = &[
    "entities",
    "edges",
    "folders",
    "scoped_documents",
    "scoped_entity_fields",
    "scoped_definitions",
    "discovery_candidates",
];
const NOTE_HEADER_COLUMNS: &[&str] = &[
    "id",
    "version",
    "world_id",
    "title",
    "folder_id",
    "entity_kind",
    "entity_subtype",
    "is_entity",
    "is_pinned",
    "favorite",
    "owner_id",
    "narrative_id",
    "order",
    "created_at",
    "updated_at",
    "is_current",
];
const NOTE_BODY_COLUMNS: &[&str] = &[
    "id",
    "version",
    "world_id",
    "title",
    "content",
    "markdown_content",
    "folder_id",
    "entity_kind",
    "entity_subtype",
    "is_entity",
    "is_pinned",
    "favorite",
    "owner_id",
    "narrative_id",
    "order",
    "created_at",
    "updated_at",
    "is_current",
];

fn note_columns(include_body: bool) -> &'static [&'static str] {
    if include_body {
        NOTE_BODY_COLUMNS
    } else {
        NOTE_HEADER_COLUMNS
    }
}

fn payload_bool(value: Option<&Value>, default: bool) -> bool {
    value.and_then(Value::as_bool).unwrap_or(default)
}

fn payload_string_array(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn note_priority(row: &CompactRowView<'_>) -> (u8, i64, i64) {
    (
        u8::from(row.get_bool("is_current").unwrap_or(false)),
        row.get_i64("version").unwrap_or_default(),
        row.get_i64("updated_at").unwrap_or_default(),
    )
}

fn select_latest_note<'a>(
    rows: &'a [CompactRow],
    columns: &'a [&'a str],
) -> Option<CompactRowView<'a>> {
    let mut best: Option<CompactRowView<'a>> = None;
    for row in rows {
        let candidate = CompactRowView::new(columns, row);
        let is_better = best
            .as_ref()
            .map(|current| note_priority(&candidate) > note_priority(current))
            .unwrap_or(true);
        if is_better {
            best = Some(candidate);
        }
    }
    best
}

fn note_value_from_row(row: CompactRowView<'_>, include_body: bool) -> Value {
    let mut object = serde_json::Map::new();
    object.insert(
        "id".to_owned(),
        Value::String(row.get_str("id").unwrap_or_default().to_owned()),
    );
    object.insert(
        "version".to_owned(),
        Value::from(row.get_i64("version").unwrap_or_default()),
    );
    object.insert(
        "world_id".to_owned(),
        Value::String(row.get_str("world_id").unwrap_or_default().to_owned()),
    );
    object.insert(
        "title".to_owned(),
        Value::String(row.get_str("title").unwrap_or_default().to_owned()),
    );
    if include_body {
        object.insert(
            "content".to_owned(),
            Value::String(row.get_str("content").unwrap_or_default().to_owned()),
        );
        object.insert(
            "markdown_content".to_owned(),
            Value::String(
                row.get_str("markdown_content")
                    .unwrap_or_default()
                    .to_owned(),
            ),
        );
    }
    object.insert(
        "folder_id".to_owned(),
        row.get_str("folder_id")
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "entity_kind".to_owned(),
        row.get_str("entity_kind")
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "entity_subtype".to_owned(),
        row.get_str("entity_subtype")
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "is_entity".to_owned(),
        Value::Bool(row.get_bool("is_entity").unwrap_or(false)),
    );
    object.insert(
        "is_pinned".to_owned(),
        Value::Bool(row.get_bool("is_pinned").unwrap_or(false)),
    );
    object.insert(
        "favorite".to_owned(),
        Value::Bool(row.get_bool("favorite").unwrap_or(false)),
    );
    object.insert(
        "owner_id".to_owned(),
        row.get_str("owner_id")
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "narrative_id".to_owned(),
        row.get_str("narrative_id")
            .map(|value| Value::String(value.to_owned()))
            .unwrap_or(Value::Null),
    );
    object.insert(
        "order".to_owned(),
        row.get_json("order").unwrap_or(Value::from(0)),
    );
    object.insert(
        "created_at".to_owned(),
        Value::from(row.get_i64("created_at").unwrap_or_default()),
    );
    object.insert(
        "updated_at".to_owned(),
        Value::from(row.get_i64("updated_at").unwrap_or_default()),
    );
    Value::Object(object)
}

fn note_values_from_rows(
    rows: Vec<CompactRow>,
    columns: &[&str],
    folder_id: Option<&str>,
    include_body: bool,
) -> Vec<Value> {
    let mut best_by_id = HashMap::<String, CompactRow>::new();

    for row in rows {
        let candidate = CompactRowView::new(columns, &row);
        if let Some(expected_folder_id) = folder_id {
            let actual_folder_id = candidate.get_str("folder_id").unwrap_or_default();
            let folder_matches = if expected_folder_id.is_empty() {
                actual_folder_id.is_empty()
            } else {
                actual_folder_id == expected_folder_id
            };
            if !folder_matches {
                continue;
            }
        }

        let Some(id) = candidate.get_str("id").map(str::to_owned) else {
            continue;
        };
        match best_by_id.get(&id) {
            Some(existing) => {
                let existing_view = CompactRowView::new(columns, existing);
                if note_priority(&candidate) > note_priority(&existing_view) {
                    best_by_id.insert(id, row);
                }
            }
            None => {
                best_by_id.insert(id, row);
            }
        }
    }

    let mut values = best_by_id
        .values()
        .map(|row| note_value_from_row(CompactRowView::new(columns, row), include_body))
        .collect::<Vec<_>>();
    values.sort_by(|left, right| {
        let left_updated = left
            .get("updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        let right_updated = right
            .get("updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        right_updated.cmp(&left_updated).then_with(|| {
            left.get("title")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .cmp(
                    right
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )
        })
    });
    values
}

fn relation_rows_match_keys(left: &Value, right: &Value, key_fields: &[&str]) -> bool {
    key_fields
        .iter()
        .all(|field| left.get(*field) == right.get(*field))
}

fn entity_card_row(card: &EntityCard) -> Value {
    json!({
        "entity_id": card.entity_id.0,
        "card_id": card.card_id,
        "name": card.name,
        "color": card.color,
        "icon": card.icon,
        "display_order": card.display_order,
        "is_collapsed": card.is_collapsed,
        "created_at": card.created_at,
        "updated_at": card.updated_at,
    })
}

fn entity_card_from_row(row: Value) -> Result<EntityCard, StoreError> {
    Ok(EntityCard {
        entity_id: phoenix_types::EntityId(
            required_string_field(&row, "entity_cards", "entity_id")?.to_owned(),
        ),
        card_id: required_string_field(&row, "entity_cards", "card_id")?.to_owned(),
        name: row
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        color: row
            .get("color")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        icon: row
            .get("icon")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        display_order: row
            .get("display_order")
            .and_then(Value::as_i64)
            .unwrap_or_default() as i32,
        is_collapsed: row
            .get("is_collapsed")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        created_at: row
            .get("created_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        updated_at: row
            .get("updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
    })
}

fn folder_schema_row(schema: &FolderSchema) -> Value {
    json!({
        "id": schema.id,
        "entity_kind": schema.entity_kind,
        "subtype": null_if_empty_value(&schema.subtype),
        "name": schema.name,
        "description": null_if_empty_value(&schema.description),
        "allowed_subfolders": parse_json_string_array_value(&schema.allowed_subfolders),
        "allowed_note_types": parse_json_string_array_value(&schema.allowed_note_types),
        "is_vault_root": schema.is_vault_root,
        "container_only": schema.container_only,
        "propagate_kind_to_children": schema.propagate_kind_to_children,
        "icon": null_if_empty_value(&schema.icon),
        "is_system": schema.is_system,
        "created_at": schema.created_at,
        "updated_at": schema.updated_at,
    })
}

fn folder_schema_from_row(row: Value) -> Result<FolderSchema, StoreError> {
    Ok(FolderSchema {
        id: required_string_field(&row, "folder_schemas", "id")?.to_owned(),
        entity_kind: row
            .get("entity_kind")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        subtype: row
            .get("subtype")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        name: row
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        description: row
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        allowed_subfolders: json_value_to_string_array(
            row.get("allowed_subfolders").cloned().unwrap_or(Value::Array(Vec::new())),
        ),
        allowed_note_types: json_value_to_string_array(
            row.get("allowed_note_types").cloned().unwrap_or(Value::Array(Vec::new())),
        ),
        is_vault_root: row
            .get("is_vault_root")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        container_only: row
            .get("container_only")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        propagate_kind_to_children: row
            .get("propagate_kind_to_children")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        icon: row
            .get("icon")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        is_system: row
            .get("is_system")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        created_at: row
            .get("created_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        updated_at: row
            .get("updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
    })
}

fn network_instance_row(network: &NetworkInstance) -> Value {
    json!({
        "id": network.id,
        "name": network.name,
        "schema_id": null_if_empty_value(&network.schema_id),
        "network_kind": network.network_kind,
        "network_subtype": null_if_empty_value(&network.network_subtype),
        "root_folder_id": null_if_empty_value(&network.root_folder_id),
        "root_entity_id": null_if_empty_value(&network.root_entity_id),
        "namespace": network.namespace,
        "description": null_if_empty_value(&network.description),
        "tags": network.tags,
        "member_count": network.member_count as i64,
        "relationship_count": network.relationship_count as i64,
        "max_depth": network.max_depth as i64,
        "created_at": network.created_at,
        "updated_at": network.updated_at,
        "group_id": null_if_empty_value(&network.group_id),
        "scope_type": network.scope_type,
        "narrative_id": null_if_empty_value(&network.narrative_id),
    })
}

fn network_instance_from_row(row: Value) -> Result<NetworkInstance, StoreError> {
    Ok(NetworkInstance {
        id: required_string_field(&row, "network_instance", "id")?.to_owned(),
        name: row
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        schema_id: row
            .get("schema_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        network_kind: row
            .get("network_kind")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        network_subtype: row
            .get("network_subtype")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        root_folder_id: row
            .get("root_folder_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        root_entity_id: row
            .get("root_entity_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        namespace: row
            .get("namespace")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        description: row
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        tags: row
            .get("tags")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default(),
        member_count: row
            .get("member_count")
            .and_then(Value::as_i64)
            .unwrap_or_default() as usize,
        relationship_count: row
            .get("relationship_count")
            .and_then(Value::as_i64)
            .unwrap_or_default() as usize,
        max_depth: row
            .get("max_depth")
            .and_then(Value::as_i64)
            .unwrap_or_default() as usize,
        created_at: row
            .get("created_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        updated_at: row
            .get("updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        group_id: row
            .get("group_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        scope_type: row
            .get("scope_type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        narrative_id: row
            .get("narrative_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
    })
}

fn network_membership_row(member: &phoenix_types::NetworkMembership) -> Value {
    json!({
        "network_id": member.network_id,
        "entity_id": member.entity_id.0,
        "x": member.x,
        "y": member.y,
        "fixed": member.fixed,
    })
}

fn network_membership_from_row(row: Value) -> Result<phoenix_types::NetworkMembership, StoreError> {
    Ok(phoenix_types::NetworkMembership {
        network_id: required_string_field(&row, "network_membership", "network_id")?.to_owned(),
        entity_id: phoenix_types::EntityId(
            required_string_field(&row, "network_membership", "entity_id")?.to_owned(),
        ),
        x: row.get("x").and_then(Value::as_f64).unwrap_or_default(),
        y: row.get("y").and_then(Value::as_f64).unwrap_or_default(),
        fixed: row.get("fixed").and_then(Value::as_bool).unwrap_or(false),
    })
}

fn network_relationship_row(relationship: &phoenix_types::NetworkRelationship) -> Value {
    json!({
        "network_id": relationship.network_id,
        "source_entity_id": relationship.source_entity_id.0,
        "target_entity_id": relationship.target_entity_id.0,
        "relationship_id": relationship.relationship_id,
    })
}

fn network_relationship_from_row(
    row: Value,
) -> Result<phoenix_types::NetworkRelationship, StoreError> {
    Ok(phoenix_types::NetworkRelationship {
        network_id: required_string_field(&row, "network_relationship", "network_id")?.to_owned(),
        source_entity_id: phoenix_types::EntityId(
            required_string_field(&row, "network_relationship", "source_entity_id")?.to_owned(),
        ),
        target_entity_id: phoenix_types::EntityId(
            required_string_field(&row, "network_relationship", "target_entity_id")?.to_owned(),
        ),
        relationship_id: required_string_field(&row, "network_relationship", "relationship_id")?
            .to_owned(),
    })
}

fn required_string_field<'a>(
    row: &'a Value,
    relation: &str,
    field: &str,
) -> Result<&'a str, StoreError> {
    row.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| StoreError::MissingColumn {
            relation: relation.to_owned(),
            column: field.to_owned(),
        })
}

fn null_if_empty_value(value: &str) -> Value {
    if value.is_empty() {
        Value::Null
    } else {
        Value::String(value.to_owned())
    }
}

fn parse_json_string_array_value(value: &str) -> Value {
    serde_json::from_str(value).unwrap_or_else(|_| Value::Array(Vec::new()))
}

fn json_value_to_string_array(value: Value) -> String {
    match value {
        Value::Array(_) => value.to_string(),
        Value::Null => "[]".to_owned(),
        other => serde_json::to_string(&other).unwrap_or_else(|_| "[]".to_owned()),
    }
}

fn note_value_priority(row: &Value) -> (u8, i64, i64) {
    (
        u8::from(row.get("is_current").and_then(Value::as_bool).unwrap_or(false)),
        row.get("version").and_then(Value::as_i64).unwrap_or_default(),
        row.get("updated_at").and_then(Value::as_i64).unwrap_or_default(),
    )
}

fn normalize_note_value(mut row: Value, include_body: bool) -> Value {
    if let Some(object) = row.as_object_mut() {
        object.entry("id").or_insert_with(|| Value::String(String::new()));
        object.entry("version").or_insert_with(|| Value::from(0));
        object
            .entry("world_id")
            .or_insert_with(|| Value::String(String::new()));
        object
            .entry("title")
            .or_insert_with(|| Value::String(String::new()));
        if include_body {
            object
                .entry("content")
                .or_insert_with(|| Value::String(String::new()));
            object
                .entry("markdown_content")
                .or_insert_with(|| Value::String(String::new()));
        } else {
            object.remove("content");
            object.remove("markdown_content");
        }
        for key in [
            "folder_id",
            "entity_kind",
            "entity_subtype",
            "owner_id",
            "narrative_id",
        ] {
            object.entry(key.to_owned()).or_insert(Value::Null);
        }
        object
            .entry("order")
            .or_insert_with(|| Value::from(0));
        object
            .entry("created_at")
            .or_insert_with(|| Value::from(0));
        object
            .entry("updated_at")
            .or_insert_with(|| Value::from(0));
        object
            .entry("is_entity")
            .or_insert_with(|| Value::Bool(false));
        object
            .entry("is_pinned")
            .or_insert_with(|| Value::Bool(false));
        object
            .entry("favorite")
            .or_insert_with(|| Value::Bool(false));
    }
    row
}

fn select_latest_note_value(rows: Vec<Value>, include_body: bool) -> Option<Value> {
    rows.into_iter()
        .max_by(|left, right| note_value_priority(left).cmp(&note_value_priority(right)))
        .map(|row| normalize_note_value(row, include_body))
}

fn note_values_from_value_rows(
    rows: Vec<Value>,
    folder_id: Option<&str>,
    include_body: bool,
) -> Vec<Value> {
    let mut best_by_id = HashMap::<String, Value>::new();
    for row in rows {
        if let Some(expected_folder_id) = folder_id {
            let actual_folder_id = row.get("folder_id").and_then(Value::as_str).unwrap_or_default();
            let folder_matches = if expected_folder_id.is_empty() {
                actual_folder_id.is_empty()
            } else {
                actual_folder_id == expected_folder_id
            };
            if !folder_matches {
                continue;
            }
        }
        let Some(id) = row.get("id").and_then(Value::as_str).map(str::to_owned) else {
            continue;
        };
        let should_replace = best_by_id
            .get(&id)
            .map(|existing| note_value_priority(&row) > note_value_priority(existing))
            .unwrap_or(true);
        if should_replace {
            best_by_id.insert(id, row);
        }
    }

    let mut values = best_by_id
        .into_values()
        .map(|row| normalize_note_value(row, include_body))
        .collect::<Vec<_>>();
    values.sort_by(|left, right| {
        let left_updated = left
            .get("updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        let right_updated = right
            .get("updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default();
        right_updated.cmp(&left_updated).then_with(|| {
            left.get("title")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .cmp(
                    right
                        .get("title")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                )
        })
    });
    values
}

fn session_record_from_row(row: &Value) -> Result<SessionRecord, StoreError> {
    let object = row.as_object().ok_or(StoreError::InvalidRow)?;
    Ok(SessionRecord {
        session_id: SessionId(
            object
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
        ),
        label: object
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned(),
        scope: phoenix_types::ScopeKey {
            world_id: object
                .get("world_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            narrative_id: object
                .get("narrative_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            folder_id: object
                .get("folder_id")
                .and_then(Value::as_str)
                .map(str::to_owned),
            folder_path: object
                .get("folder_path")
                .and_then(Value::as_str)
                .map(str::to_owned),
        },
        status: object
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("active")
            .to_owned(),
        revision: object.get("revision").and_then(Value::as_u64).unwrap_or(0),
        created_at: object
            .get("created_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
        updated_at: object
            .get("updated_at")
            .and_then(Value::as_i64)
            .unwrap_or_default(),
    })
}

fn require_payload_str<'a>(payload: &'a Value, key: &str) -> Result<&'a str, StoreError> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| StoreError::Query(format!("missing store command field: {key}")))
}

fn require_payload_value<'a>(payload: &'a Value, key: &str) -> Result<&'a Value, StoreError> {
    payload
        .get(key)
        .ok_or_else(|| StoreError::Query(format!("missing store command field: {key}")))
}

fn payload_object(value: Option<&Value>) -> Option<&serde_json::Map<String, Value>> {
    value.and_then(Value::as_object)
}

fn ensure_allowed_content_relation(relation: &str) -> Result<(), StoreError> {
    if ALLOWED_WAL_RELATIONS.contains(&relation) && CONTENT_SNAPSHOT_RELATIONS.contains(&relation) {
        return Ok(());
    }
    Err(StoreError::Query(format!(
        "unsupported WAL relation: {relation}"
    )))
}

fn row_matches_filter(row: &Value, filter: Option<&serde_json::Map<String, Value>>) -> bool {
    let Some(filter) = filter else {
        return true;
    };
    let Some(object) = row.as_object() else {
        return false;
    };
    filter
        .iter()
        .all(|(key, expected)| object.get(key) == Some(expected))
}

pub(crate) fn now_ms() -> i64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now() as i64
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_millis() as i64
    }
}

fn native_lmdb_store_path(storage_path: Option<&Path>) -> PathBuf {
    match storage_path {
        Some(path) => path.join("phoenix-lmdb-v2"),
        None => std::env::temp_dir().join(format!(
            "phoenix-native-lmdb-v2-{}-{}-{}",
            std::process::id(),
            now_ms().max(0),
            NATIVE_STORE_INSTANCE_COUNTER.fetch_add(1, AtomicOrdering::Relaxed),
        )),
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FixtureManifest {
    pub fixtures: Vec<GoldenFixture>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoldenFixture {
    pub id: String,
    pub title: String,
    pub source_path: String,
    pub document_id: String,
    pub file_path: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpectedFixtureBaseline {
    pub fixture_id: String,
    pub source_path: String,
    pub expected_scanner: Option<serde_json::Value>,
    pub expected_structure: Option<serde_json::Value>,
    pub expected_ingest: Option<serde_json::Value>,
    pub expected_query: Option<serde_json::Value>,
}

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("workspace root")
}

pub fn fixtures_root() -> PathBuf {
    workspace_root().join("fixtures")
}

pub fn load_fixture_manifest() -> FixtureManifest {
    let path = fixtures_root().join("manifest.json");
    let content = fs::read_to_string(path).expect("fixture manifest");
    serde_json::from_str(&content).expect("fixture manifest json")
}

pub fn load_expected_baseline(fixture_id: &str) -> ExpectedFixtureBaseline {
    let path = fixtures_root()
        .join("expected")
        .join(format!("{fixture_id}.json"));
    let content = fs::read_to_string(path).expect("expected baseline");
    serde_json::from_str(&content).expect("expected baseline json")
}

pub fn fixture_body(fixture: &GoldenFixture) -> String {
    let path = fixtures_root().join(&fixture.file_path);
    fs::read_to_string(path).expect("fixture body")
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_store_native::{
        BundleKind, PhoenixArchiveStoreV2, PhoenixBundleStoreV2, PhoenixGraphKernelStoreV2,
    };
    use phoenix_types::{
        ChatPlannerModelResponse, ChatPlannerStep, ChatRunSnapshot, ChatRunStatus,
        ChatWorkspaceArtifact, CreateSessionRequest, DocumentId, EntityId, EntityKind, GenderHint,
        GraphDeltaRequest, MentionEntityRef, QueryResultHeader, QueryTarget, RunOptions, ScopeKey,
        SessionStateResultHeader, SessionStatsResultHeader,
    };
    use serde_json::{json, Value};

    #[test]
    fn fixture_manifest_loads() {
        let manifest = load_fixture_manifest();
        assert!(
            !manifest.fixtures.is_empty(),
            "fixtures should not be empty"
        );
    }

    #[test]
    fn fixture_bodies_exist_and_are_non_empty() {
        let manifest = load_fixture_manifest();

        for fixture in &manifest.fixtures {
            let body = fixture_body(fixture);
            assert!(
                !body.trim().is_empty(),
                "fixture {} should have non-empty body",
                fixture.id
            );
        }
    }

    #[test]
    fn expected_baselines_match_manifest() {
        let manifest = load_fixture_manifest();

        for fixture in &manifest.fixtures {
            let baseline = load_expected_baseline(&fixture.id);
            assert_eq!(baseline.fixture_id, fixture.id);
            assert_eq!(baseline.source_path, fixture.source_path);
        }
    }

    #[test]
    fn runtime_init_reports_schema() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let init = runtime.init().expect("init");
        assert!(init.ready);
        assert_eq!(init.schema_version, NATIVE_RUNTIME_SCHEMA_VERSION);
    }

    #[test]
    fn runtime_capabilities_report_required_store_contract() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");

        let result = runtime
            .store_command(StoreCommandRequest {
                command: "runtime:capabilities".to_owned(),
                payload: json!({}),
            })
            .expect("runtime capabilities");

        let payload = result.payload.expect("payload");
        let object = payload.as_object().expect("object");
        let store_api_version = object
            .get("storeApiVersion")
            .and_then(Value::as_u64)
            .expect("storeApiVersion");
        let capabilities = object
            .get("capabilities")
            .and_then(Value::as_array)
            .expect("capabilities");

        assert_eq!(store_api_version, STORE_API_VERSION as u64);
        for capability in RUNTIME_CAPABILITIES {
            assert!(
                capabilities
                    .iter()
                    .any(|candidate| candidate.as_str() == Some(*capability)),
                "missing capability {}",
                capability
            );
        }
    }

    #[test]
    fn native_store_command_rejects_legacy_cozo_namespaces() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");

        let result = runtime
            .store_command(StoreCommandRequest {
                command: "chat:listThreads".to_owned(),
                payload: json!({}),
            })
            .expect("store command");

        assert!(!result.success);
        assert_eq!(
            result.error.as_deref(),
            Some("chat:listThreads is unavailable on the native runtime path")
        );
    }

    #[test]
    fn native_runtime_rejects_legacy_cozo_snapshot_bytes() {
        let legacy = wasm_runtime();
        let snapshot = legacy.export_snapshot().expect("legacy snapshot");

        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");
        let error = runtime
            .import_snapshot(&snapshot)
            .expect_err("native should reject legacy snapshot bytes");

        assert!(matches!(error, StoreError::Query(_)));
        assert!(error
            .to_string()
            .contains("legacy Cozo snapshots is unavailable on the native runtime path"));
    }

    #[test]
    fn session_commit_cycle_updates_revision() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Foundation".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        let commit = runtime
            .commit(CommitRequest {
                session_id: session.session_id,
                reason: Some("phase-2".to_owned()),
            })
            .expect("commit");

        assert_eq!(commit.revision, 1);
    }

    #[test]
    fn native_ingest_leaves_lex_dirty_until_explicit_rebuild() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");
        runtime.rebuild_lex_index().expect("seed lex");
        assert!(runtime.lex.borrow().is_some());

        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Dirty Lex".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");
        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-dirty-lex".to_owned()),
                    note_id: None,
                    title: "Dirty".to_owned(),
                    text: "Ryan crossed the harbor.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        let store = runtime.v2_store().expect("v2 store");
        assert!(runtime.lex.borrow().is_none());
        assert_eq!(store.list_dirty_scopes().expect("dirty scopes").len(), 1);
        assert!(store
            .load_scope_sidecar(&ScopeKey::default())
            .expect("sidecar")
            .is_none());

        runtime
            .rebuild(RebuildRequest::default())
            .expect("rebuild");

        assert!(runtime.lex.borrow().is_none());
        assert!(store.list_dirty_scopes().expect("dirty scopes").is_empty());
        assert!(store
            .load_scope_sidecar(&ScopeKey::default())
            .expect("sidecar")
            .is_some());
        let query = runtime
            .query(QueryRequest {
                session_id: Some(session.session_id.clone()),
                query: "Ryan".to_owned(),
                scope: ScopeKey::default(),
                targets: vec![phoenix_types::QueryTarget::Chunks],
                limit: Some(5),
                temporal: None,
                semantic_query_vector: None,
                include_candidate_graph: false,
            })
            .expect("query");
        assert!(!query.chunk_hits.is_empty());
    }

    #[test]
    fn native_scope_qgram_cache_reuses_and_invalidates_across_ingest() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Scope Cache".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");
        let large_text = (0..256)
            .map(|index| {
                format!(
                    "## Sector {index}\nRyan mapped harbor sector {index}. The harbor route {index} stayed open for dawn traffic.\n\n"
                )
            })
            .collect::<String>();
        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-cache-one".to_owned()),
                    note_id: None,
                    title: "Harbor Ledger".to_owned(),
                    text: large_text,
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest one");

        let store = runtime.v2_store().expect("v2 store");
        let sidecar = store
            .load_materialized_scope_lexical(&ScopeKey::default())
            .expect("materialized sidecar");
        assert!(sidecar.spans.len() >= NATIVE_SCOPE_QGRAM_MIN_SPANS);
        assert!(runtime.native_scope_lex.borrow().is_empty());

        let first = runtime
            .native_lexical_search(store, "harbor", &ScopeKey::default(), 5)
            .expect("first search");
        assert!(!first.span_hits.is_empty());
        assert!(first
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "PX_QUERY_NATIVE_SCOPE_QGRAM"));
        assert_eq!(runtime.native_scope_lex.borrow().len(), 1);

        let second = runtime
            .native_lexical_search(store, "route", &ScopeKey::default(), 5)
            .expect("second search");
        assert!(!second.span_hits.is_empty());
        assert!(second
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "PX_QUERY_NATIVE_SCOPE_QGRAM_CACHE"));
        assert_eq!(runtime.native_scope_lex.borrow().len(), 1);

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-cache-two".to_owned()),
                    note_id: None,
                    title: "Citadel Notes".to_owned(),
                    text: "The citadel lantern stayed lit above the harbor gate.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest two");
        assert!(runtime.native_scope_lex.borrow().is_empty());

        let refreshed = runtime
            .native_lexical_search(store, "citadel", &ScopeKey::default(), 5)
            .expect("refreshed search");
        assert!(!refreshed.span_hits.is_empty());
        assert_eq!(runtime.native_scope_lex.borrow().len(), 1);
    }

    #[test]
    fn native_commit_keeps_session_archive_content_while_session_rows_advance() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Commit".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");
        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-commit".to_owned()),
                    note_id: None,
                    title: "Commit".to_owned(),
                    text: "Ryan catalogued the harbor ledger.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        let store = runtime.v2_store().expect("v2 store");
        let archive_before = store
            .load_latest_session_archive(&session.session_id)
            .expect("session archive before")
            .expect("session archive");

        let commit = runtime
            .commit(CommitRequest {
                session_id: session.session_id.clone(),
                reason: Some("verify commit".to_owned()),
            })
            .expect("commit");

        let archive_after = store
            .load_latest_session_archive(&session.session_id)
            .expect("session archive after")
            .expect("session archive");
        let state = runtime
            .session_state(&session.session_id)
            .expect("session state");
        let stats = runtime
            .session_stats(&session.session_id)
            .expect("session stats");

        assert_eq!(archive_after, archive_before);
        assert_eq!(state.updated_at, commit.committed_at);
        assert_eq!(stats.updated_at, commit.committed_at);
    }

    #[test]
    fn native_snapshot_import_skips_sidecars_but_preserves_lazy_query_and_rebuild() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Snapshot Dirty".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-snapshot-dirty".to_owned()),
                    note_id: None,
                    title: "Snapshot".to_owned(),
                    text: "Ryan mapped the harbor before dawn.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");
        runtime
            .rebuild(RebuildRequest::default())
            .expect("rebuild sidecars");
        assert!(runtime
            .v2_store()
            .expect("v2 store")
            .load_scope_sidecar(&ScopeKey::default())
            .expect("sidecar before export")
            .is_some());

        let snapshot = runtime.export_snapshot().expect("export snapshot");
        let restored = PhoenixRuntime::new(RuntimeConfig::default()).expect("restored runtime");
        restored.init().expect("restored init");
        restored.import_snapshot(&snapshot).expect("import snapshot");

        let restored_store = restored.v2_store().expect("restored store");
        assert!(restored_store
            .load_scope_sidecar(&ScopeKey::default())
            .expect("sidecar after import")
            .is_none());
        assert_eq!(restored_store.list_dirty_scopes().expect("dirty scopes").len(), 1);

        let state = restored
            .session_state(&session.session_id)
            .expect("session state");
        let stats = restored
            .session_stats(&session.session_id)
            .expect("session stats");
        let query = restored
            .query(QueryRequest {
                session_id: Some(session.session_id.clone()),
                query: "Ryan".to_owned(),
                scope: ScopeKey::default(),
                targets: vec![QueryTarget::Chunks, QueryTarget::Nodes],
                limit: Some(5),
                temporal: None,
                semantic_query_vector: None,
                include_candidate_graph: false,
            })
            .expect("query");
        let delta = restored
            .graph_delta(GraphDeltaRequest {
                session_id: session.session_id.clone(),
                scope: ScopeKey::default(),
                changed_documents: Vec::new(),
                limit: None,
                since_commit: None,
                include_candidate_graph: false,
            })
            .expect("graph delta");

        assert_eq!(state.documents.len(), 1);
        assert_eq!(stats.document_count, 1);
        assert!(!query.chunk_hits.is_empty());
        assert!(!query.node_hits.is_empty());
        assert!(!delta.nodes.is_empty());

        restored
            .rebuild(RebuildRequest::default())
            .expect("rebuild imported");
        assert!(restored_store.list_dirty_scopes().expect("dirty scopes").is_empty());
        assert!(restored_store
            .load_scope_sidecar(&ScopeKey::default())
            .expect("sidecar after rebuild")
            .is_some());
    }

    #[test]
    fn ingest_and_query_roundtrip() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Stub".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        let ingest = runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-1".to_owned()),
                    note_id: None,
                    title: "Ash Song".to_owned(),
                    text: "The phoenix rose from ash.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: true,
            })
            .expect("ingest");
        assert_eq!(ingest.document_count, 1);

        let query = runtime
            .query(QueryRequest {
                session_id: Some(session.session_id),
                query: "phoenix".to_owned(),
                scope: ScopeKey::default(),
                targets: vec![QueryTarget::Chunks],
                limit: Some(3),
                temporal: None,
                semantic_query_vector: None,
                include_candidate_graph: false,
            })
            .expect("query");

        assert_eq!(query.chunk_hits.len(), 1);
        assert!(query.chunk_hits[0].chunk_id.starts_with("doc-1:"));
    }

    #[test]
    fn native_v2_ingest_persists_source_and_session_summary_bundles() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "V2 Bundles".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-v2-bundle".to_owned()),
                    note_id: None,
                    title: "V2 Bundle".to_owned(),
                    text: "The phoenix wrote its own bundle.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: true,
            })
            .expect("ingest");

        let store = runtime.v2_store().expect("v2 store");
        let document_headers = store
            .list_bundle_headers(BundleKind::DocumentArchive, Some("__global__::__global__::__global__::__global__"))
            .expect("document headers");
        assert!(document_headers
            .iter()
            .any(|header| header.key.entity_key == "doc-v2-bundle"));

        let session_headers = store
            .list_bundle_headers(BundleKind::SessionArchive, Some(&session.session_id.0))
            .expect("session headers");
        assert!(session_headers
            .iter()
            .any(|header| header.key.entity_key == session.session_id.0));
        assert!(store.kernel_current_generation().expect("kernel generation") >= 1);
        assert!(!store
            .load_kernel_journal_after(0)
            .expect("kernel journal")
            .is_empty());

        let state = runtime
            .session_state(&session.session_id)
            .expect("session state");
        let stats = runtime
            .session_stats(&session.session_id)
            .expect("session stats");
        assert_eq!(state.documents.len(), 1);
        assert_eq!(stats.document_count, 1);
        assert!(stats.graph_vertex_count >= 1);
    }

    #[test]
    fn native_v2_scan_query_and_graph_delta_use_v2_paths() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "V2 Query".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        let scan = runtime.scan_text(ScanRequest {
            session_id: Some(session.session_id.clone()),
            text: "Ryan crossed the harbor before dawn.".to_owned(),
            scope: ScopeKey::default(),
            resolver_seed: Vec::new(),
        });
        assert!(scan
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "PX_INVARANT_V2_SCAN"));

        let structure = runtime.build_structure(StructureRequest {
            text: "Ryan crossed the harbor before dawn.".to_owned(),
            scan: scan.clone(),
        });
        assert!(structure
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "PX_INVARANT_V2_STRUCTURE"));

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-v2-query".to_owned()),
                    note_id: None,
                    title: "V2 Query".to_owned(),
                    text: "Ryan crossed the harbor before dawn.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: true,
            })
            .expect("ingest");

        let query = runtime
            .query(QueryRequest {
                session_id: Some(session.session_id.clone()),
                query: "Ryan".to_owned(),
                scope: ScopeKey::default(),
                targets: vec![QueryTarget::Chunks, QueryTarget::Nodes],
                limit: Some(5),
                temporal: None,
                semantic_query_vector: None,
                include_candidate_graph: false,
            })
            .expect("query");
        assert!(!query.chunk_hits.is_empty());
        assert!(!query.node_hits.is_empty());
        assert!(query
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "PX_TRIVERSE_V2_KERNEL"));

        let delta = runtime
            .graph_delta(GraphDeltaRequest {
                session_id: session.session_id.clone(),
                scope: ScopeKey::default(),
                changed_documents: Vec::new(),
                limit: None,
                since_commit: None,
                include_candidate_graph: false,
            })
            .expect("graph delta");
        assert!(!delta.nodes.is_empty());
    }

    #[test]
    fn native_v2_snapshot_roundtrip_preserves_bundles_and_query_paths() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "V2 Snapshot".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-v2-snapshot".to_owned()),
                    note_id: None,
                    title: "Snapshot".to_owned(),
                    text: "Ryan mapped the harbor before dawn.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: true,
            })
            .expect("ingest");

        let snapshot = runtime.export_snapshot().expect("export snapshot");

        let restored = PhoenixRuntime::new(RuntimeConfig::default()).expect("restored runtime");
        restored.init().expect("restored init");
        let envelope = restored.import_snapshot(&snapshot).expect("import snapshot");
        assert_eq!(envelope.schema_version, NATIVE_RUNTIME_SCHEMA_VERSION);

        let restored_session = restored
            .session_state(&session.session_id)
            .expect("session state");
        assert_eq!(restored_session.documents.len(), 1);

        let restored_stats = restored
            .session_stats(&session.session_id)
            .expect("session stats");
        assert_eq!(restored_stats.document_count, 1);
        assert!(restored_stats.graph_vertex_count >= 1);

        let query = restored
            .query(QueryRequest {
                session_id: Some(session.session_id.clone()),
                query: "Ryan".to_owned(),
                scope: ScopeKey::default(),
                targets: vec![QueryTarget::Chunks, QueryTarget::Nodes],
                limit: Some(5),
                temporal: None,
                semantic_query_vector: None,
                include_candidate_graph: false,
            })
            .expect("query");
        assert!(!query.chunk_hits.is_empty());
        assert!(!query.node_hits.is_empty());
        assert!(query
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "PX_TRIVERSE_V2_KERNEL"));

        let store = restored.v2_store().expect("v2 store");
        assert!(store
            .list_bundle_headers(
                BundleKind::DocumentArchive,
                Some("__global__::__global__::__global__::__global__")
            )
            .expect("source headers")
            .is_empty());
        assert_eq!(
            store
                .load_latest_document_archives(Some(&ScopeKey::default()))
                .expect("native archives")
                .len(),
            1
        );
        assert!(store.kernel_current_generation().expect("generation") >= 1);
    }

    #[test]
    fn native_semantic_store_commands_refresh_document_candidate_edges_from_lmdb_ann() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Native semantic".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![
                    phoenix_types::IngestDocument {
                        document_id: DocumentId("doc-sem-a".to_owned()),
                        note_id: None,
                        title: "Dock A".to_owned(),
                        text: "Ryan mapped dock alpha before dawn.".to_owned(),
                        scope: ScopeKey::default(),
                    },
                    phoenix_types::IngestDocument {
                        document_id: DocumentId("doc-sem-b".to_owned()),
                        note_id: None,
                        title: "Dock B".to_owned(),
                        text: "Rian mapped dock beta before dawn.".to_owned(),
                        scope: ScopeKey::default(),
                    },
                ],
                commit: false,
            })
            .expect("ingest");

        let leaf_payload = runtime
            .store_command(StoreCommandRequest {
                command: "semantic:listLeafChunks".to_owned(),
                payload: json!({ "documentIds": ["doc-sem-a", "doc-sem-b"] }),
            })
            .expect("list leaf chunks")
            .payload
            .expect("leaf payload");
        let leaf_chunks: Vec<phoenix_store_cozo::SemanticLeafChunk> =
            serde_json::from_value(leaf_payload).expect("leaf chunks");
        assert!(!leaf_chunks.is_empty());

        runtime
            .store_command(StoreCommandRequest {
                command: "semantic:upsertDocumentVectors".to_owned(),
                payload: json!({
                    "rows": [
                        {
                            "documentId": "doc-sem-a",
                            "values": semantic_test_vector(0),
                            "leafCount": 1,
                            "evidenceRefs": ["span:doc-sem-a"]
                        },
                        {
                            "documentId": "doc-sem-b",
                            "values": semantic_test_vector(0),
                            "leafCount": 1,
                            "evidenceRefs": ["span:doc-sem-b"]
                        }
                    ]
                }),
            })
            .expect("upsert document vectors");

        runtime
            .store_command(StoreCommandRequest {
                command: "semantic:refreshCandidateGraphEdges".to_owned(),
                payload: json!({
                    "documentIds": ["doc-sem-a", "doc-sem-b"],
                    "nodeIds": [],
                }),
            })
            .expect("refresh candidate graph");

        let candidate_edges = runtime.candidate_edge_rows().expect("candidate edges");
        assert!(candidate_edges.iter().any(|row| {
            row.get("source_id").and_then(Value::as_str) == Some("doc::doc-sem-a")
                && row.get("target_id").and_then(Value::as_str) == Some("doc::doc-sem-b")
                && row.get("edge_type").and_then(Value::as_str) == Some("similar_to")
        }));
    }

    #[test]
    fn session_state_and_stats_persist_after_ingest() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "State".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-state".to_owned()),
                    note_id: None,
                    title: "Stateful".to_owned(),
                    text: "# Prologue\nRyan woke up.\n\nChapter 1\nRyan met Len.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        let state = runtime
            .session_state(&session.session_id)
            .expect("session state");
        let stats = runtime
            .session_stats(&session.session_id)
            .expect("session stats");

        assert_eq!(state.documents.len(), 1);
        assert!(stats.chapter_count >= 1);
        assert!(stats.graph_vertex_count >= 1);
    }

    #[test]
    fn phase2_candidate_prototypes_and_edges_refresh_from_store_commands() {
        let runtime = wasm_runtime();
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Phase2".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-phase2".to_owned()),
                    note_id: None,
                    title: "Harbor".to_owned(),
                    text: "Ryan met Rian at the harbor.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        let leaf_id = runtime
            .store
            .fetch_rows("graph_vertices")
            .expect("graph vertices")
            .into_iter()
            .find(|row| {
                row.get("document_id").and_then(Value::as_str) == Some("doc-phase2")
                    && row
                        .get("value")
                        .and_then(Value::as_object)
                        .and_then(|value| value.get("kind"))
                        .and_then(Value::as_str)
                        == Some("leaf")
            })
            .and_then(|row| row.get("id").and_then(Value::as_str).map(str::to_owned))
            .expect("leaf vertex");
        for (entity_id, label, aliases) in [
            ("entity::ryan", "Ryan", json!(["Ryan Hale"])),
            ("entity::ryan-alt", "Ryan Hale", json!(["Ryan"])),
        ] {
            runtime
                .store
                .put_row(
                    "graph_vertices",
                    json!({
                        "id": entity_id,
                        "document_id": "doc-phase2",
                        "narrative_id": null,
                        "value": { "kind": "entity", "entityId": entity_id.trim_start_matches("entity::"), "label": label, "entityKind": "Character" },
                        "weight": 1,
                        "attributes": {
                            "aliases": aliases,
                            "documentId": "doc-phase2",
                            "graph": {
                                "layer": "asserted",
                                "status": "asserted",
                                "resolver": "test",
                                "confidence": 1.0,
                                "evidence_refs": [format!("graph_vertex:{}", leaf_id)]
                            }
                        }
                    }),
                )
                .expect("entity vertex");
            runtime
                .store
                .put_row(
                    "graph_edges",
                    json!({
                        "source_id": leaf_id.as_str(),
                        "target_id": entity_id,
                        "document_id": "doc-phase2",
                        "narrative_id": null,
                        "valid_from_doc": "doc-phase2",
                        "valid_from_boundary": null,
                        "valid_to_doc": null,
                        "valid_to_boundary": null,
                        "assertion_kind": "asserted",
                        "weight": 100,
                        "attributes": {
                            "documentId": "doc-phase2",
                            "graph": {
                                "layer": "asserted",
                                "status": "asserted",
                                "resolver": "test",
                                "confidence": 1.0,
                                "evidence_refs": [format!("graph_vertex:{}", leaf_id)]
                            }
                        },
                        "data": null,
                        "edge_type": "mentions"
                    }),
                )
                .expect("mentions edge");
        }

        let payload = runtime
            .store_command(StoreCommandRequest {
                command: "semantic:listCandidatePrototypeInputs".to_owned(),
                payload: json!({ "documentIds": ["doc-phase2"] }),
            })
            .expect("prototype inputs")
            .payload
            .expect("prototype payload");
        let inputs: Vec<SemanticCandidatePrototypeInput> =
            serde_json::from_value(payload).expect("prototype rows");
        let entities = inputs
            .iter()
            .filter(|row| row.node_kind == "entity")
            .take(2)
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(entities.len(), 2, "expected two entity prototypes");

        runtime
            .store_command(StoreCommandRequest {
                command: "semantic:upsertPrototypeVectors".to_owned(),
                payload: json!({
                    "rows": entities.iter().map(|entity| {
                        json!({
                            "nodeId": entity.node_id,
                            "nodeKind": entity.node_kind,
                            "documentId": entity.document_id,
                            "narrativeId": entity.narrative_id,
                            "folderId": entity.folder_id,
                            "values": semantic_test_vector(0),
                            "evidenceRefs": entity.evidence_refs,
                        })
                    }).collect::<Vec<_>>()
                }),
            })
            .expect("upsert prototype vectors");

        runtime
            .store_command(StoreCommandRequest {
                command: "semantic:refreshCandidateGraphEdges".to_owned(),
                payload: json!({
                    "documentIds": ["doc-phase2"],
                    "nodeIds": entities.iter().map(|entity| entity.node_id.clone()).collect::<Vec<_>>(),
                }),
            })
            .expect("refresh candidate graph");

        let candidate_edges = runtime
            .store
            .fetch_rows("graph_candidate_edges")
            .expect("candidate edges");
        let relevant_edge = candidate_edges
            .iter()
            .find(|row| {
                row.get("source_id").and_then(Value::as_str) == Some(entities[0].node_id.as_str())
                    && row.get("target_id").and_then(Value::as_str)
                        == Some(entities[1].node_id.as_str())
                    && row.get("edge_type").and_then(Value::as_str)
                        == Some("candidate_corefers_with")
            })
            .expect("candidate coref edge");

        let graph_meta = relevant_edge
            .get("attributes")
            .and_then(Value::as_object)
            .and_then(|attributes| attributes.get("graph"))
            .and_then(Value::as_object)
            .expect("candidate graph metadata");
        assert_eq!(
            graph_meta.get("layer").and_then(Value::as_str),
            Some("candidate")
        );
        assert_eq!(
            graph_meta.get("resolver").and_then(Value::as_str),
            Some(PHASE2_EMBEDDING_RESOLVER)
        );
        assert_eq!(
            relevant_edge
                .get("data")
                .and_then(Value::as_object)
                .and_then(|data| data.get("base"))
                .and_then(Value::as_object)
                .and_then(|base| base.get("resolver"))
                .and_then(Value::as_str),
            Some(PHASE2_EMBEDDING_RESOLVER)
        );
    }

    #[test]
    fn phase2_entity_candidate_edges_require_surface_coherence() {
        let runtime = wasm_runtime();
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Phase2 coherence".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-phase2-coherence".to_owned()),
                    note_id: None,
                    title: "Harbor".to_owned(),
                    text: "Ryan met Rian at the harbor.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        let leaf_id = runtime
            .store
            .fetch_rows("graph_vertices")
            .expect("graph vertices")
            .into_iter()
            .find(|row| {
                row.get("document_id").and_then(Value::as_str) == Some("doc-phase2-coherence")
                    && row
                        .get("value")
                        .and_then(Value::as_object)
                        .and_then(|value| value.get("kind"))
                        .and_then(Value::as_str)
                        == Some("leaf")
            })
            .and_then(|row| row.get("id").and_then(Value::as_str).map(str::to_owned))
            .expect("leaf vertex");
        for (entity_id, label) in [("entity::ryan-coherence", "Ryan"), ("entity::rian-coherence", "Rian")] {
            runtime
                .store
                .put_row(
                    "graph_vertices",
                    json!({
                        "id": entity_id,
                        "document_id": "doc-phase2-coherence",
                        "narrative_id": null,
                        "value": { "kind": "entity", "entityId": entity_id.trim_start_matches("entity::"), "label": label, "entityKind": "Character" },
                        "weight": 1,
                        "attributes": {
                            "documentId": "doc-phase2-coherence",
                            "graph": {
                                "layer": "asserted",
                                "status": "asserted",
                                "resolver": "test",
                                "confidence": 1.0,
                                "evidence_refs": [format!("graph_vertex:{}", leaf_id)]
                            }
                        }
                    }),
                )
                .expect("entity vertex");
            runtime
                .store
                .put_row(
                    "graph_edges",
                    json!({
                        "source_id": leaf_id.as_str(),
                        "target_id": entity_id,
                        "document_id": "doc-phase2-coherence",
                        "narrative_id": null,
                        "valid_from_doc": "doc-phase2-coherence",
                        "valid_from_boundary": null,
                        "valid_to_doc": null,
                        "valid_to_boundary": null,
                        "assertion_kind": "asserted",
                        "weight": 100,
                        "attributes": {
                            "documentId": "doc-phase2-coherence",
                            "graph": {
                                "layer": "asserted",
                                "status": "asserted",
                                "resolver": "test",
                                "confidence": 1.0,
                                "evidence_refs": [format!("graph_vertex:{}", leaf_id)]
                            }
                        },
                        "data": null,
                        "edge_type": "mentions"
                    }),
                )
                .expect("mentions edge");
        }

        let payload = runtime
            .store_command(StoreCommandRequest {
                command: "semantic:listCandidatePrototypeInputs".to_owned(),
                payload: json!({ "documentIds": ["doc-phase2-coherence"] }),
            })
            .expect("prototype inputs")
            .payload
            .expect("prototype payload");
        let inputs: Vec<SemanticCandidatePrototypeInput> =
            serde_json::from_value(payload).expect("prototype rows");
        let entities = inputs
            .iter()
            .filter(|row| row.node_kind == "entity")
            .take(2)
            .cloned()
            .collect::<Vec<_>>();

        runtime
            .store_command(StoreCommandRequest {
                command: "semantic:upsertPrototypeVectors".to_owned(),
                payload: json!({
                    "rows": entities.iter().map(|entity| {
                        json!({
                            "nodeId": entity.node_id,
                            "nodeKind": entity.node_kind,
                            "documentId": entity.document_id,
                            "narrativeId": entity.narrative_id,
                            "folderId": entity.folder_id,
                            "values": semantic_test_vector(0),
                            "evidenceRefs": entity.evidence_refs,
                        })
                    }).collect::<Vec<_>>()
                }),
            })
            .expect("upsert prototype vectors");

        runtime
            .store_command(StoreCommandRequest {
                command: "semantic:refreshCandidateGraphEdges".to_owned(),
                payload: json!({
                    "documentIds": ["doc-phase2-coherence"],
                    "nodeIds": entities.iter().map(|entity| entity.node_id.clone()).collect::<Vec<_>>(),
                }),
            })
            .expect("refresh candidate graph");

        let candidate_edges = runtime
            .store
            .fetch_rows("graph_candidate_edges")
            .expect("candidate edges");
        assert!(candidate_edges.iter().all(|row| {
            row.get("edge_type").and_then(Value::as_str) != Some("candidate_corefers_with")
        }));
    }

    #[test]
    fn phase2_nli_inputs_are_bounded_and_apply_promotes_or_rejects_candidate_edges() {
        let runtime = wasm_runtime();
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Phase2 NLI".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-phase2-nli".to_owned()),
                    note_id: None,
                    title: "Harbor".to_owned(),
                    text: "Ryan met Rian at the harbor.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        let leaf_id = runtime
            .store
            .fetch_rows("graph_vertices")
            .expect("graph vertices")
            .into_iter()
            .find(|row| {
                row.get("document_id").and_then(Value::as_str) == Some("doc-phase2-nli")
                    && row
                        .get("value")
                        .and_then(Value::as_object)
                        .and_then(|value| value.get("kind"))
                        .and_then(Value::as_str)
                        == Some("leaf")
            })
            .and_then(|row| row.get("id").and_then(Value::as_str).map(str::to_owned))
            .expect("leaf vertex");
        for (entity_id, label, aliases) in [
            ("entity::ryan-nli", "Ryan", json!(["Ryan Hale"])),
            ("entity::ryan-hale-nli", "Ryan Hale", json!(["Ryan"])),
        ] {
            runtime
                .store
                .put_row(
                    "graph_vertices",
                    json!({
                        "id": entity_id,
                        "document_id": "doc-phase2-nli",
                        "narrative_id": null,
                        "value": { "kind": "entity", "entityId": entity_id.trim_start_matches("entity::"), "label": label, "entityKind": "Character" },
                        "weight": 1,
                        "attributes": {
                            "aliases": aliases,
                            "documentId": "doc-phase2-nli",
                            "graph": {
                                "layer": "asserted",
                                "status": "asserted",
                                "resolver": "test",
                                "confidence": 1.0,
                                "evidence_refs": [format!("graph_vertex:{}", leaf_id)]
                            }
                        }
                    }),
                )
                .expect("entity vertex");
            runtime
                .store
                .put_row(
                    "graph_edges",
                    json!({
                        "source_id": leaf_id.as_str(),
                        "target_id": entity_id,
                        "document_id": "doc-phase2-nli",
                        "narrative_id": null,
                        "valid_from_doc": "doc-phase2-nli",
                        "valid_from_boundary": null,
                        "valid_to_doc": null,
                        "valid_to_boundary": null,
                        "assertion_kind": "asserted",
                        "weight": 100,
                        "attributes": {
                            "documentId": "doc-phase2-nli",
                            "graph": {
                                "layer": "asserted",
                                "status": "asserted",
                                "resolver": "test",
                                "confidence": 1.0,
                                "evidence_refs": [format!("graph_vertex:{}", leaf_id)]
                            }
                        },
                        "data": null,
                        "edge_type": "mentions"
                    }),
                )
                .expect("mentions edge");
        }

        let payload = runtime
            .store_command(StoreCommandRequest {
                command: "semantic:listCandidatePrototypeInputs".to_owned(),
                payload: json!({ "documentIds": ["doc-phase2-nli"] }),
            })
            .expect("prototype inputs")
            .payload
            .expect("prototype payload");
        let inputs: Vec<SemanticCandidatePrototypeInput> =
            serde_json::from_value(payload).expect("prototype rows");
        let entities = inputs
            .iter()
            .filter(|row| row.node_kind == "entity")
            .take(2)
            .cloned()
            .collect::<Vec<_>>();

        runtime
            .store_command(StoreCommandRequest {
                command: "semantic:upsertPrototypeVectors".to_owned(),
                payload: json!({
                    "rows": entities.iter().map(|entity| {
                        json!({
                            "nodeId": entity.node_id,
                            "nodeKind": entity.node_kind,
                            "documentId": entity.document_id,
                            "narrativeId": entity.narrative_id,
                            "folderId": entity.folder_id,
                            "values": semantic_test_vector(0),
                            "evidenceRefs": entity.evidence_refs,
                        })
                    }).collect::<Vec<_>>()
                }),
            })
            .expect("upsert prototype vectors");

        runtime
            .store_command(StoreCommandRequest {
                command: "semantic:refreshCandidateGraphEdges".to_owned(),
                payload: json!({
                    "documentIds": ["doc-phase2-nli"],
                    "nodeIds": entities.iter().map(|entity| entity.node_id.clone()).collect::<Vec<_>>(),
                }),
            })
            .expect("refresh candidate graph");

        let nli_payload = runtime
            .store_command(StoreCommandRequest {
                command: "semantic:listNliJudgmentInputs".to_owned(),
                payload: json!({
                    "documentIds": ["doc-phase2-nli"],
                    "nodeIds": entities.iter().map(|entity| entity.node_id.clone()).collect::<Vec<_>>(),
                }),
            })
            .expect("nli inputs")
            .payload
            .expect("nli input payload");
        let nli_inputs: Vec<SemanticNliJudgmentInput> =
            serde_json::from_value(nli_payload).expect("nli judgment inputs");
        assert_eq!(nli_inputs.len(), 2);
        assert!(nli_inputs[0].hypothesis.contains("same entity"));
        assert!(nli_inputs.len() <= PHASE2_NLI_MAX_INPUTS);

        let accepted = runtime
            .store_command(StoreCommandRequest {
                command: "semantic:applyNliJudgments".to_owned(),
                payload: json!({
                    "modelId": SEMANTIC_NLI_MODEL_ID,
                    "device": "wasm",
                    "results": nli_inputs.iter().map(|input| {
                        json!({
                            "judgmentId": input.judgment_id,
                            "groupId": input.group_id,
                            "sourceId": input.source_id,
                            "targetId": input.target_id,
                            "edgeType": input.edge_type,
                            "direction": input.direction,
                            "premise": input.premise,
                            "hypothesis": input.hypothesis,
                            "entailment": 0.84,
                            "neutral": 0.09,
                            "contradiction": 0.07,
                            "predictedLabel": "entailment",
                            "confidence": 0.84,
                        })
                    }).collect::<Vec<_>>()
                }),
            })
            .expect("apply accepted nli")
            .payload
            .expect("accepted payload");
        assert_eq!(accepted.get("kept").and_then(Value::as_u64), Some(1));

        let rejected = runtime
            .store_command(StoreCommandRequest {
                command: "semantic:applyNliJudgments".to_owned(),
                payload: json!({
                    "modelId": SEMANTIC_NLI_MODEL_ID,
                    "device": "wasm",
                    "results": nli_inputs.iter().map(|input| {
                        json!({
                            "judgmentId": input.judgment_id,
                            "groupId": input.group_id,
                            "sourceId": input.source_id,
                            "targetId": input.target_id,
                            "edgeType": input.edge_type,
                            "direction": input.direction,
                            "premise": input.premise,
                            "hypothesis": input.hypothesis,
                            "entailment": 0.11,
                            "neutral": 0.18,
                            "contradiction": 0.71,
                            "predictedLabel": "contradiction",
                            "confidence": 0.71,
                        })
                    }).collect::<Vec<_>>()
                }),
            })
            .expect("apply rejected nli")
            .payload
            .expect("rejected payload");
        assert_eq!(rejected.get("rejected").and_then(Value::as_u64), Some(1));

        let row = runtime
            .store
            .fetch_rows("graph_candidate_edges")
            .expect("candidate rows")
            .into_iter()
            .find(|row| {
                row.get("source_id").and_then(Value::as_str) == Some(entities[0].node_id.as_str())
                    && row.get("target_id").and_then(Value::as_str)
                        == Some(entities[1].node_id.as_str())
                    && row.get("edge_type").and_then(Value::as_str)
                        == Some("candidate_corefers_with")
            })
            .expect("candidate edge row");
        let graph_meta = row
            .get("attributes")
            .and_then(Value::as_object)
            .and_then(|attributes| attributes.get("graph"))
            .and_then(Value::as_object)
            .expect("graph metadata");
        assert_eq!(
            graph_meta.get("status").and_then(Value::as_str),
            Some("candidate_rejected")
        );
        assert_eq!(
            graph_meta.get("resolver").and_then(Value::as_str),
            Some(PHASE2_NLI_RESOLVER)
        );
        let data = row
            .get("data")
            .and_then(Value::as_object)
            .expect("candidate edge data");
        assert!(data.get("base").is_some());
        assert!(data.get("nli").is_some());
    }

    #[test]
    fn phase2_nli_creates_task_evidence_candidate_rows_for_support_and_rejection() {
        let runtime = wasm_runtime();
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Phase2 NLI Evidence".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-phase2-nli-evidence".to_owned()),
                    note_id: None,
                    title: "Harbor task".to_owned(),
                    text: "Map the harbor before dawn.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        let leaf_id = runtime
            .store
            .fetch_rows("graph_vertices")
            .expect("graph vertices")
            .into_iter()
            .find(|row| {
                row.get("document_id").and_then(Value::as_str) == Some("doc-phase2-nli-evidence")
                    && row
                        .get("value")
                        .and_then(Value::as_object)
                        .and_then(|value| value.get("kind"))
                        .and_then(Value::as_str)
                        == Some("leaf")
            })
            .and_then(|row| row.get("id").and_then(Value::as_str).map(str::to_owned))
            .expect("leaf vertex");

        runtime
            .store
            .put_row(
                "graph_vertices",
                json!({
                    "id": "task::phase2-evidence",
                    "document_id": "doc-phase2-nli-evidence",
                    "narrative_id": null,
                    "value": { "kind": "task", "label": "Map the harbor" },
                    "weight": 1,
                    "attributes": {
                        "documentId": "doc-phase2-nli-evidence",
                        "currentTask": "Map the harbor",
                        "graph": {
                            "layer": "asserted",
                            "status": "asserted",
                            "resolver": "test",
                            "confidence": 1.0,
                            "evidence_refs": [format!("graph_vertex:{leaf_id}")]
                        }
                    }
                }),
            )
            .expect("task vertex");

        let nli_payload = runtime
            .store_command(StoreCommandRequest {
                command: "semantic:listNliJudgmentInputs".to_owned(),
                payload: json!({
                    "documentIds": ["doc-phase2-nli-evidence"],
                    "nodeIds": ["task::phase2-evidence"],
                }),
            })
            .expect("nli inputs")
            .payload
            .expect("nli input payload");
        let nli_inputs: Vec<SemanticNliJudgmentInput> =
            serde_json::from_value(nli_payload).expect("nli judgment inputs");
        assert!(nli_inputs
            .iter()
            .any(|input| input.edge_type == "supported_by" && input.target_id == leaf_id));
        assert!(nli_inputs
            .iter()
            .any(|input| input.edge_type == "contradicted_by" && input.target_id == leaf_id));

        let apply_payload = runtime
            .store_command(StoreCommandRequest {
                command: "semantic:applyNliJudgments".to_owned(),
                payload: json!({
                    "modelId": SEMANTIC_NLI_MODEL_ID,
                    "device": "wasm",
                    "results": nli_inputs.iter().map(|input| {
                        if input.edge_type == "supported_by" {
                            json!({
                                "judgmentId": input.judgment_id,
                                "groupId": input.group_id,
                                "sourceId": input.source_id,
                                "targetId": input.target_id,
                                "edgeType": input.edge_type,
                                "direction": input.direction,
                                "premise": input.premise,
                                "hypothesis": input.hypothesis,
                                "entailment": 0.83,
                                "neutral": 0.11,
                                "contradiction": 0.06,
                                "predictedLabel": "entailment",
                                "confidence": 0.83,
                            })
                        } else {
                            json!({
                                "judgmentId": input.judgment_id,
                                "groupId": input.group_id,
                                "sourceId": input.source_id,
                                "targetId": input.target_id,
                                "edgeType": input.edge_type,
                                "direction": input.direction,
                                "premise": input.premise,
                                "hypothesis": input.hypothesis,
                                "entailment": 0.54,
                                "neutral": 0.30,
                                "contradiction": 0.16,
                                "predictedLabel": "entailment",
                                "confidence": 0.54,
                            })
                        }
                    }).collect::<Vec<_>>()
                }),
            })
            .expect("apply nli judgments")
            .payload
            .expect("nli apply payload");
        assert_eq!(apply_payload.get("kept").and_then(Value::as_u64), Some(1));
        assert_eq!(
            apply_payload.get("rejected").and_then(Value::as_u64),
            Some(1)
        );

        let candidate_rows = runtime
            .store
            .fetch_rows("graph_candidate_edges")
            .expect("candidate rows");
        let supported_row = candidate_rows
            .iter()
            .find(|row| {
                row.get("source_id").and_then(Value::as_str) == Some("task::phase2-evidence")
                    && row.get("target_id").and_then(Value::as_str) == Some(leaf_id.as_str())
                    && row.get("edge_type").and_then(Value::as_str) == Some("supported_by")
            })
            .expect("supported row");
        let supported_graph = supported_row
            .get("attributes")
            .and_then(Value::as_object)
            .and_then(|attributes| attributes.get("graph"))
            .and_then(Value::as_object)
            .expect("supported graph metadata");
        assert_eq!(
            supported_graph.get("status").and_then(Value::as_str),
            Some("candidate")
        );
        assert_eq!(
            supported_row
                .get("data")
                .and_then(Value::as_object)
                .and_then(|data| data.get("base"))
                .and_then(Value::as_object)
                .and_then(|base| base.get("resolver"))
                .and_then(Value::as_str),
            Some(PHASE2_NLI_RESOLVER)
        );

        let contradicted_row = candidate_rows
            .iter()
            .find(|row| {
                row.get("source_id").and_then(Value::as_str) == Some("task::phase2-evidence")
                    && row.get("target_id").and_then(Value::as_str) == Some(leaf_id.as_str())
                    && row.get("edge_type").and_then(Value::as_str) == Some("contradicted_by")
            })
            .expect("contradicted row");
        let contradicted_graph = contradicted_row
            .get("attributes")
            .and_then(Value::as_object)
            .and_then(|attributes| attributes.get("graph"))
            .and_then(Value::as_object)
            .expect("contradicted graph metadata");
        assert_eq!(
            contradicted_graph.get("status").and_then(Value::as_str),
            Some("candidate_rejected")
        );
        assert!(contradicted_row
            .get("data")
            .and_then(Value::as_object)
            .and_then(|data| data.get("nli"))
            .is_some());
    }

    #[test]
    fn graph_delta_and_binary_payloads_are_rebuildable_from_canonical_state() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Binary".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-binary".to_owned()),
                    note_id: None,
                    title: "Binary".to_owned(),
                    text: "Ryan attacked Len. Ryan gave Len a blade.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        let graph_delta = runtime
            .graph_delta(GraphDeltaRequest {
                session_id: session.session_id.clone(),
                scope: ScopeKey::default(),
                changed_documents: vec![DocumentId("doc-binary".to_owned())],
                limit: Some(8),
                since_commit: None,
                include_candidate_graph: false,
            })
            .expect("graph delta");
        assert!(!graph_delta.nodes.is_empty() || !graph_delta.chunks.is_empty());
        assert!(!graph_delta.edges.is_empty());

        let query_bytes = runtime
            .query_binary(QueryRequest {
                session_id: Some(session.session_id.clone()),
                query: "Ryan".to_owned(),
                scope: ScopeKey::default(),
                targets: vec![QueryTarget::Chunks],
                limit: Some(5),
                temporal: None,
                semantic_query_vector: None,
                include_candidate_graph: false,
            })
            .expect("query bytes");
        let graph_bytes = runtime
            .graph_delta_binary(GraphDeltaRequest {
                session_id: session.session_id.clone(),
                scope: ScopeKey::default(),
                changed_documents: vec![DocumentId("doc-binary".to_owned())],
                limit: Some(8),
                since_commit: None,
                include_candidate_graph: false,
            })
            .expect("graph bytes");
        let state_bytes = runtime
            .session_state_binary(&session.session_id)
            .expect("state bytes");
        let stats_bytes = runtime
            .session_stats_binary(&session.session_id)
            .expect("stats bytes");

        assert!(query_bytes.len() >= QueryResultHeader::BYTE_LEN);
        assert!(graph_bytes.len() >= phoenix_types::GraphDeltaResultHeader::BYTE_LEN);
        assert!(state_bytes.len() >= SessionStateResultHeader::BYTE_LEN);
        assert!(stats_bytes.len() >= SessionStatsResultHeader::BYTE_LEN);
    }

    #[test]
    fn runtime_scan_and_structure_roundtrip() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let scan = runtime.scan_text(ScanRequest {
            text: "Luffy attacked Zoro.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(SessionId("scan-1".to_owned())),
            resolver_seed: vec![
                phoenix_types::ResolverEntitySeed {
                    entity_id: EntityId("luffy".to_owned()),
                    canonical_name: "Luffy".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Character),
                    gender: Some(GenderHint::Male),
                    number: None,
                    scope: ScopeKey::default(),
                },
                phoenix_types::ResolverEntitySeed {
                    entity_id: EntityId("zoro".to_owned()),
                    canonical_name: "Zoro".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Character),
                    gender: Some(GenderHint::Male),
                    number: None,
                    scope: ScopeKey::default(),
                },
            ],
        });
        assert_eq!(scan.mentions.len(), 2);
        assert!(scan.mentions.iter().any(|mention| mention.entity_ref
            == Some(MentionEntityRef::Known(EntityId("luffy".to_owned())))));

        let structure = runtime.build_structure(StructureRequest {
            text: "Luffy attacked Zoro.".to_owned(),
            scan,
        });
        assert_eq!(structure.sentence_frames.len(), 1);
        assert_eq!(structure.sentence_frames[0].verb_frames.len(), 1);
    }

    #[test]
    fn entity_cards_and_folder_schema_persist() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");

        runtime
            .upsert_entity_cards_batch(&[
                phoenix_types::EntityCard {
                    entity_id: EntityId("CHARACTER".to_owned()),
                    card_id: "summary".to_owned(),
                    name: "Summary".to_owned(),
                    color: "#ff8800".to_owned(),
                    icon: "spark".to_owned(),
                    display_order: 2,
                    is_collapsed: false,
                    created_at: 10,
                    updated_at: 10,
                },
                phoenix_types::EntityCard {
                    entity_id: EntityId("CHARACTER".to_owned()),
                    card_id: "traits".to_owned(),
                    name: "Traits".to_owned(),
                    color: "#00aaff".to_owned(),
                    icon: "bolt".to_owned(),
                    display_order: 1,
                    is_collapsed: true,
                    created_at: 11,
                    updated_at: 11,
                },
            ])
            .expect("cards");

        runtime
            .upsert_folder_schema(&phoenix_types::FolderSchema {
                id: "schema-character".to_owned(),
                entity_kind: "character".to_owned(),
                subtype: "crew".to_owned(),
                name: "Character Vault".to_owned(),
                description: "Stores character notes.".to_owned(),
                allowed_subfolders: "[\"profiles\",\"chapters\"]".to_owned(),
                allowed_note_types: "[\"bio\",\"scene\"]".to_owned(),
                is_vault_root: true,
                container_only: false,
                propagate_kind_to_children: true,
                icon: "user".to_owned(),
                is_system: false,
                created_at: 20,
                updated_at: 21,
            })
            .expect("schema");

        let cards = runtime
            .get_entity_cards(&EntityId("CHARACTER".to_owned()))
            .expect("entity cards");
        let schema = runtime
            .get_folder_schema("schema-character")
            .expect("folder schema")
            .expect("folder schema present");

        assert_eq!(cards.len(), 2);
        assert_eq!(cards[0].card_id, "traits");
        assert_eq!(schema.allowed_subfolders, "[\"profiles\",\"chapters\"]");
        assert_eq!(schema.allowed_note_types, "[\"bio\",\"scene\"]");
    }

    #[test]
    fn saved_network_view_replaces_members_and_relationships() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");

        runtime
            .save_network_view(&phoenix_types::SavedNetworkView {
                instance: phoenix_types::NetworkInstance {
                    id: "net-1".to_owned(),
                    name: "Crew".to_owned(),
                    schema_id: "schema-character".to_owned(),
                    network_kind: "mindmap".to_owned(),
                    network_subtype: "entity".to_owned(),
                    root_folder_id: "folder-1".to_owned(),
                    root_entity_id: String::new(),
                    namespace: "world".to_owned(),
                    description: "Crew graph".to_owned(),
                    tags: vec!["crew".to_owned(), "battle".to_owned()],
                    member_count: 2,
                    relationship_count: 1,
                    max_depth: 2,
                    created_at: 100,
                    updated_at: 100,
                    group_id: String::new(),
                    scope_type: "folder".to_owned(),
                    narrative_id: "nar-1".to_owned(),
                },
                members: vec![
                    phoenix_types::NetworkMembership {
                        network_id: "net-1".to_owned(),
                        entity_id: EntityId("luffy".to_owned()),
                        x: 10.0,
                        y: 20.0,
                        fixed: true,
                    },
                    phoenix_types::NetworkMembership {
                        network_id: "net-1".to_owned(),
                        entity_id: EntityId("zoro".to_owned()),
                        x: 30.0,
                        y: 40.0,
                        fixed: false,
                    },
                ],
                relationships: vec![phoenix_types::NetworkRelationship {
                    network_id: "net-1".to_owned(),
                    source_entity_id: EntityId("luffy".to_owned()),
                    target_entity_id: EntityId("zoro".to_owned()),
                    relationship_id: "edge-1".to_owned(),
                }],
            })
            .expect("initial save");

        runtime
            .save_network_view(&phoenix_types::SavedNetworkView {
                instance: phoenix_types::NetworkInstance {
                    id: "net-1".to_owned(),
                    name: "Crew Revised".to_owned(),
                    schema_id: "schema-character".to_owned(),
                    network_kind: "mindmap".to_owned(),
                    network_subtype: "entity".to_owned(),
                    root_folder_id: "folder-1".to_owned(),
                    root_entity_id: String::new(),
                    namespace: "world".to_owned(),
                    description: "Crew graph revised".to_owned(),
                    tags: vec!["crew".to_owned()],
                    member_count: 1,
                    relationship_count: 0,
                    max_depth: 1,
                    created_at: 100,
                    updated_at: 200,
                    group_id: String::new(),
                    scope_type: "folder".to_owned(),
                    narrative_id: "nar-1".to_owned(),
                },
                members: vec![phoenix_types::NetworkMembership {
                    network_id: "net-1".to_owned(),
                    entity_id: EntityId("luffy".to_owned()),
                    x: 12.0,
                    y: 24.0,
                    fixed: false,
                }],
                relationships: Vec::new(),
            })
            .expect("replacement save");

        let view = runtime
            .get_network_view("net-1")
            .expect("network fetch")
            .expect("network exists");
        let listed = runtime.list_network_views().expect("network list");

        assert_eq!(view.instance.name, "Crew Revised");
        assert_eq!(view.members.len(), 1);
        assert_eq!(view.members[0].entity_id.0, "luffy");
        assert!(view.relationships.is_empty());
        assert_eq!(listed.len(), 1);

        runtime
            .delete_network_view("net-1")
            .expect("delete network");
        assert!(runtime
            .get_network_view("net-1")
            .expect("network fetch after delete")
            .is_none());
    }

    #[test]
    fn persistence_batch_replays_content_mutations_in_order() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");

        let result = runtime
            .store_command(StoreCommandRequest {
                command: "persistence:applyWalBatch".to_owned(),
                payload: json!({
                    "records": [
                        {
                            "seq": 1,
                            "command": "note:upsert",
                            "partition": "content",
                            "writtenAt": 100,
                            "payload": {
                                "row": {
                                    "id": "note-1",
                                    "version": 1,
                                    "world_id": "world-1",
                                    "title": "Alpha",
                                    "content": "Hello",
                                    "markdown_content": "Hello",
                                    "folder_id": null,
                                    "entity_kind": null,
                                    "entity_subtype": null,
                                    "is_entity": false,
                                    "is_pinned": false,
                                    "favorite": false,
                                    "owner_id": null,
                                    "narrative_id": null,
                                    "order": 0,
                                    "created_at": 10,
                                    "updated_at": 10,
                                    "valid_from": 10,
                                    "valid_to": null,
                                    "is_current": true,
                                    "change_reason": null
                                }
                            }
                        },
                        {
                            "seq": 2,
                            "command": "relation:upsert",
                            "partition": "content",
                            "writtenAt": 101,
                            "payload": {
                                "relation": "entities",
                                "row": {
                                    "id": "entity-1",
                                    "label": "Hero",
                                    "kind": "character",
                                    "subtype": null,
                                    "aliases": [],
                                    "first_note": "note-1",
                                    "total_mentions": 1,
                                    "narrative_id": null,
                                    "created_by": "user",
                                    "created_at": 10,
                                    "updated_at": 10
                                }
                            }
                        }
                    ]
                }),
            })
            .expect("wal batch");

        assert!(result.success);
        let note = runtime.get_note_value("note-1", true).expect("note");
        let entity = runtime
            .fetch_relation_rows("entities")
            .expect("entities")
            .into_iter()
            .find(|row| row.get("id").and_then(Value::as_str) == Some("entity-1"));

        assert_eq!(
            note.and_then(|value| value
                .get("title")
                .and_then(Value::as_str)
                .map(str::to_owned)),
            Some("Alpha".to_owned())
        );
        assert!(entity.is_some());
    }

    #[test]
    fn persistence_clear_derived_keeps_content_rows() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");

        runtime
            .put_relation_row(
                "notes",
                json!({
                    "id": "note-keep",
                    "version": 1,
                    "world_id": "world-1",
                    "title": "Keep",
                    "content": "persist",
                    "markdown_content": "persist",
                    "folder_id": null,
                    "entity_kind": null,
                    "entity_subtype": null,
                    "is_entity": false,
                    "is_pinned": false,
                    "favorite": false,
                    "owner_id": null,
                    "narrative_id": null,
                    "order": 0,
                    "created_at": 10,
                    "updated_at": 10,
                    "valid_from": 10,
                    "valid_to": null,
                    "is_current": true,
                    "change_reason": null
                }),
            )
            .expect("note row");
        runtime
            .put_relation_row(
                "phoenix_sessions",
                json!({
                    "session_id": "session-1",
                    "label": "Derived",
                    "world_id": null,
                    "narrative_id": null,
                    "folder_id": null,
                    "folder_path": null,
                    "status": "active",
                    "revision": 0,
                    "created_at": 1,
                    "updated_at": 1
                }),
            )
            .expect("session row");

        runtime
            .store_command(StoreCommandRequest {
                command: "persistence:clearDerived".to_owned(),
                payload: json!({}),
            })
            .expect("clear derived");

        assert!(runtime
            .fetch_relation_rows("phoenix_sessions")
            .expect("derived rows")
            .is_empty());
        assert_eq!(
            runtime
                .get_note_value("note-keep", true)
                .expect("note")
                .is_some(),
            true
        );
    }

    #[test]
    fn session_close_drops_session_and_session_logs() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: Some(SessionId("session-close".to_owned())),
                label: "Closable".to_owned(),
                scope: Default::default(),
            })
            .expect("session");

        runtime
            .store
            .put_row(
                "phoenix_ingest_log",
                json!({
                    "id": "ingest-1",
                    "session_id": session.session_id.0,
                    "document_count": 1,
                    "commit_requested": false,
                    "request_json": {"sessionId": session.session_id.0},
                    "created_at": 1
                }),
            )
            .expect("ingest log");
        runtime
            .store
            .put_row(
                "phoenix_query_log",
                json!({
                    "id": "query-1",
                    "session_id": session.session_id.0,
                    "query": "kai",
                    "limit": 10,
                    "request_json": {"sessionId": session.session_id.0},
                    "created_at": 1
                }),
            )
            .expect("query log");

        let result = runtime
            .store_command(StoreCommandRequest {
                command: "session:close".to_owned(),
                payload: json!({ "sessionId": session.session_id.0 }),
            })
            .expect("close session");

        assert_eq!(result.success, true);
        assert!(runtime
            .store
            .fetch_rows("phoenix_sessions")
            .expect("sessions")
            .is_empty());
        assert!(runtime
            .store
            .fetch_rows("phoenix_ingest_log")
            .expect("ingest logs")
            .is_empty());
        assert!(runtime
            .store
            .fetch_rows("phoenix_query_log")
            .expect("query logs")
            .is_empty());
    }

    #[test]
    fn planner_disabled_runs_stay_ready_to_answer() {
        let runtime = chat_runtime();
        let thread = runtime
            .chat
            .create_thread(
                &runtime.store,
                Some("world-1"),
                Some("nar-1"),
                Some("Thread"),
            )
            .expect("thread");
        runtime
            .chat
            .add_message(
                &runtime.store,
                &thread.id.0,
                "user",
                "Summarize the scoped notes.",
                Some("nar-1"),
            )
            .expect("message");

        let run = runtime
            .chat
            .start_run(
                &runtime.store,
                &thread.id.0,
                "Summarize the scoped notes.",
                run_options("nar-1", false, false),
            )
            .expect("run");

        assert_eq!(run.status, ChatRunStatus::ReadyToAnswer);

        let snapshot = runtime
            .chat
            .poll_run(&runtime.store, &run.id)
            .expect("snapshot")
            .expect("run snapshot");
        assert_eq!(snapshot.run.status, ChatRunStatus::ReadyToAnswer);
        assert!(snapshot.planner_step.is_none());
        assert!(snapshot.artifacts.is_empty());
    }

    #[test]
    fn planner_run_executes_tools_and_promotes_scoped_artifacts() {
        let runtime = chat_runtime();
        let thread = runtime
            .chat
            .create_thread(
                &runtime.store,
                Some("world-1"),
                Some("nar-1"),
                Some("Thread"),
            )
            .expect("thread");
        runtime
            .chat
            .add_message(
                &runtime.store,
                &thread.id.0,
                "user",
                "Find the in-scope note and prepare a grounded answer.",
                Some("nar-1"),
            )
            .expect("message");

        insert_note(
            &runtime,
            "note-in",
            "nar-1",
            "Inside scope",
            "Ryan waited at the harbor.",
        );
        insert_note(
            &runtime,
            "note-out",
            "nar-2",
            "Outside scope",
            "This note should never be surfaced.",
        );

        let run = runtime
            .chat
            .start_run(
                &runtime.store,
                &thread.id.0,
                "Find the in-scope note and prepare a grounded answer.",
                run_options("nar-1", true, false),
            )
            .expect("run");
        assert_eq!(run.status, ChatRunStatus::Planning);
        let run_id = run.id.clone();

        let step =
            planner_step_command(&runtime, "chat:getPlannerStep", json!({ "runId": run_id }));
        let request = match step {
            ChatPlannerStep::ModelRequest { request } => request,
            other => panic!("expected initial model request, got {other:?}"),
        };
        assert!(request.allow_tools);

        let tool_step = planner_step_from_payload(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:submitPlannerModelResponse".to_owned(),
                    payload: json!({
                        "runId": run_id,
                        "response": ChatPlannerModelResponse {
                            content: String::new(),
                            tool_calls: vec![
                                phoenix_types::ChatPlannerToolCall {
                                    id: "tool-note-list".to_owned(),
                                    name: "note_list".to_owned(),
                                    arguments_json: r#"{"limit":10}"#.to_owned(),
                                },
                                phoenix_types::ChatPlannerToolCall {
                                    id: "tool-artifact-put".to_owned(),
                                    name: "artifact_put".to_owned(),
                                    arguments_json: r#"{"kind":"claim_list","payload":{"claim":"Ryan waited at the harbor."},"pinned":false}"#.to_owned(),
                                },
                            ],
                        },
                    }),
                })
                .expect("submit planner response")
                .payload
                .expect("tool step payload"),
        );

        match tool_step {
            ChatPlannerStep::ToolCalls { tool_calls, .. } => assert_eq!(tool_calls.len(), 2),
            other => panic!("expected tool calls step, got {other:?}"),
        }

        let step_after_tools = planner_step_command(
            &runtime,
            "chat:advancePlannerRun",
            json!({ "runId": run_id }),
        );
        let request_after_tools = match step_after_tools {
            ChatPlannerStep::ModelRequest { request } => request,
            other => panic!("expected follow-up model request, got {other:?}"),
        };
        let note_list_message = request_after_tools
            .messages
            .iter()
            .find(|message| message.tool_call_id.as_deref() == Some("tool-note-list"))
            .expect("note_list tool message");
        let note_list_payload: Value =
            serde_json::from_str(&note_list_message.content).expect("note list payload");
        let note_ids = note_list_payload
            .get("notes")
            .and_then(Value::as_array)
            .expect("notes array")
            .iter()
            .filter_map(|note| note.get("id").and_then(Value::as_str))
            .collect::<Vec<_>>();
        assert_eq!(note_ids, vec!["note-in"]);

        let complete_step = planner_step_from_payload(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:submitPlannerModelResponse".to_owned(),
                    payload: json!({
                        "runId": run_id,
                        "response": ChatPlannerModelResponse {
                            content: "Ground the answer in note-in and the saved claims.".to_owned(),
                            tool_calls: Vec::new(),
                        },
                    }),
                })
                .expect("submit planner summary")
                .payload
                .expect("complete payload"),
        );
        match complete_step {
            ChatPlannerStep::Complete { response, .. } => {
                assert_eq!(
                    response,
                    "Ground the answer in note-in and the saved claims."
                )
            }
            other => panic!("expected planner completion, got {other:?}"),
        }

        let snapshot: ChatRunSnapshot = serde_json::from_value(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:pollRun".to_owned(),
                    payload: json!({ "runId": run_id }),
                })
                .expect("poll run")
                .payload
                .expect("snapshot payload"),
        )
        .expect("snapshot");
        assert_eq!(snapshot.run.status, ChatRunStatus::ReadyToAnswer);
        assert!(snapshot
            .run
            .prepared_system_prompt
            .contains("Ground the answer in note-in and the saved claims."));
        assert!(snapshot.planner_step.is_none());
        assert!(snapshot
            .artifacts
            .iter()
            .any(|artifact| artifact.kind == "draft_answer" && artifact.pinned));
        let claim_artifact = snapshot
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == "claim_list")
            .cloned()
            .expect("claim_list artifact");
        assert!(!claim_artifact.pinned);

        let artifacts: Vec<ChatWorkspaceArtifact> = serde_json::from_value(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:listPlannerArtifacts".to_owned(),
                    payload: json!({ "runId": run_id }),
                })
                .expect("list artifacts")
                .payload
                .expect("artifacts payload"),
        )
        .expect("artifacts");
        assert!(artifacts
            .iter()
            .any(|artifact| artifact.key == claim_artifact.key));

        let pinned: ChatWorkspaceArtifact = serde_json::from_value(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:pinPlannerArtifact".to_owned(),
                    payload: json!({
                        "runId": run_id,
                        "key": claim_artifact.key,
                        "pinned": true,
                    }),
                })
                .expect("pin artifact")
                .payload
                .expect("pin payload"),
        )
        .expect("pinned artifact");
        assert!(pinned.pinned);
    }

    #[test]
    fn planner_canvas_tools_pause_for_ts_host_and_resume_after_approval() {
        let runtime = chat_runtime();
        let thread = runtime
            .chat
            .create_thread(
                &runtime.store,
                Some("world-1"),
                Some("nar-1"),
                Some("Thread"),
            )
            .expect("thread");
        runtime
            .chat
            .add_message(
                &runtime.store,
                &thread.id.0,
                "user",
                "Rewrite the highlighted paragraph in the open note.",
                Some("nar-1"),
            )
            .expect("message");

        let run = runtime
            .chat
            .start_run(
                &runtime.store,
                &thread.id.0,
                "Rewrite the highlighted paragraph in the open note.",
                run_options("nar-1", true, true),
            )
            .expect("run");
        let run_id = run.id.clone();

        let initial_step =
            planner_step_command(&runtime, "chat:getPlannerStep", json!({ "runId": run_id }));
        let initial_request = match initial_step {
            ChatPlannerStep::ModelRequest { request } => request,
            other => panic!("expected initial model request, got {other:?}"),
        };
        assert!(initial_request
            .tools
            .iter()
            .any(|tool| tool.name == "get_active_note_snapshot"));
        assert!(initial_request
            .tools
            .iter()
            .any(|tool| tool.name == "replace_text_proposal"));

        let tool_step = planner_step_from_payload(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:submitPlannerModelResponse".to_owned(),
                    payload: json!({
                        "runId": run_id,
                        "response": ChatPlannerModelResponse {
                            content: String::new(),
                            tool_calls: vec![
                                phoenix_types::ChatPlannerToolCall {
                                    id: "tool-note-snapshot".to_owned(),
                                    name: "get_active_note_snapshot".to_owned(),
                                    arguments_json: "{}".to_owned(),
                                },
                                phoenix_types::ChatPlannerToolCall {
                                    id: "tool-highlight".to_owned(),
                                    name: "highlight_range".to_owned(),
                                    arguments_json: r#"{"from":12,"to":48}"#.to_owned(),
                                },
                                phoenix_types::ChatPlannerToolCall {
                                    id: "tool-replace".to_owned(),
                                    name: "replace_text_proposal".to_owned(),
                                    arguments_json: r#"{"from":12,"to":48,"replacement":"Rewritten paragraph.","expectedRevision":42}"#.to_owned(),
                                },
                            ],
                        },
                    }),
                })
                .expect("submit planner response")
                .payload
                .expect("tool step payload"),
        );
        match tool_step {
            ChatPlannerStep::ToolCalls { tool_calls, .. } => assert_eq!(tool_calls.len(), 3),
            other => panic!("expected tool calls step, got {other:?}"),
        }

        let after_advance = runtime
            .store_command(StoreCommandRequest {
                command: "chat:advancePlannerRun".to_owned(),
                payload: json!({ "runId": run_id }),
            })
            .expect("advance planner run");
        assert!(after_advance.payload.is_none());

        let awaiting_host: ChatRunSnapshot = serde_json::from_value(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:pollRun".to_owned(),
                    payload: json!({ "runId": run_id }),
                })
                .expect("poll awaiting host")
                .payload
                .expect("awaiting host payload"),
        )
        .expect("awaiting host snapshot");
        assert_eq!(awaiting_host.run.status, ChatRunStatus::AwaitingToolHost);
        assert_eq!(awaiting_host.tool_calls.len(), 3);
        assert!(awaiting_host
            .tool_calls
            .iter()
            .all(|call| call.host == "typescript" && call.status == "pending_host"));

        let after_tools: ChatRunSnapshot = serde_json::from_value(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:submitToolResults".to_owned(),
                    payload: json!({
                        "runId": run_id,
                        "results": [
                            {
                                "toolCallId": "tool-note-snapshot",
                                "resultJson": "{\"noteId\":\"note-1\",\"revision\":42,\"text\":\"Original paragraph.\"}"
                            },
                            {
                                "toolCallId": "tool-highlight",
                                "resultJson": "{\"ok\":true}"
                            },
                            {
                                "toolCallId": "tool-replace",
                                "proposal": {
                                    "proposalId": "approval-1",
                                    "toolName": "replace_text_proposal",
                                    "affectedNoteId": "note-1",
                                    "summary": "Replace the highlighted paragraph",
                                    "diffPreview": "Replace the highlighted paragraph with a tighter rewrite.",
                                    "expectedRevision": 42,
                                    "rollbackToken": "note-1:42",
                                    "payloadJson": "{\"kind\":\"replace_text\",\"from\":12,\"to\":48,\"replacement\":\"Rewritten paragraph.\",\"expectedRevision\":42}"
                                }
                            }
                        ]
                    }),
                })
                .expect("submit tool results")
                .payload
                .expect("tool results payload"),
        )
        .expect("after tools snapshot");
        assert_eq!(after_tools.run.status, ChatRunStatus::AwaitingApproval);
        assert_eq!(after_tools.approvals.len(), 1);
        assert!(after_tools
            .tool_calls
            .iter()
            .any(|call| call.tool_call_id == "tool-replace" && call.status == "awaiting_approval"));

        let approval_id = after_tools.approvals[0].id.clone();
        let after_approval: ChatRunSnapshot = serde_json::from_value(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:submitApproval".to_owned(),
                    payload: json!({
                        "runId": run_id,
                        "approvalId": approval_id,
                        "approved": true,
                        "decisionJson": "{\"approved\":true,\"applied\":true,\"autoApplied\":false}"
                    }),
                })
                .expect("submit approval")
                .payload
                .expect("approval payload"),
        )
        .expect("after approval snapshot");
        assert_eq!(after_approval.run.status, ChatRunStatus::Planning);

        let resumed_step =
            planner_step_command(&runtime, "chat:getPlannerStep", json!({ "runId": run_id }));
        let resumed_request = match resumed_step {
            ChatPlannerStep::ModelRequest { request } => request,
            other => panic!("expected resumed model request, got {other:?}"),
        };
        assert!(resumed_request
            .messages
            .iter()
            .any(|message| message.tool_call_id.as_deref() == Some("tool-note-snapshot")));
        assert!(resumed_request
            .messages
            .iter()
            .any(|message| message.tool_call_id.as_deref() == Some("tool-replace")));

        let complete_step = planner_step_from_payload(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:submitPlannerModelResponse".to_owned(),
                    payload: json!({
                        "runId": run_id,
                        "response": ChatPlannerModelResponse {
                            content: "The active note edit has been planned and approved.".to_owned(),
                            tool_calls: Vec::new(),
                        },
                    }),
                })
                .expect("submit final planner response")
                .payload
                .expect("final planner payload"),
        );
        match complete_step {
            ChatPlannerStep::Complete { response, .. } => {
                assert_eq!(
                    response,
                    "The active note edit has been planned and approved."
                )
            }
            other => panic!("expected completion after approval, got {other:?}"),
        }

        let final_snapshot: ChatRunSnapshot = serde_json::from_value(
            runtime
                .store_command(StoreCommandRequest {
                    command: "chat:pollRun".to_owned(),
                    payload: json!({ "runId": run_id }),
                })
                .expect("poll final snapshot")
                .payload
                .expect("final snapshot payload"),
        )
        .expect("final snapshot");
        assert_eq!(final_snapshot.run.status, ChatRunStatus::ReadyToAnswer);
    }

    #[test]
    fn runtime_text_analytics_matches_gokitt_shape() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        let analytics = runtime.analyze_text(
            "The iron gate slammed shut. The iron gate rattled again. The iron gate shook against the wall. \
Bright embers glowed beside the ember-lit grate. Bright embers hissed in the ash.",
        );

        assert!(analytics.word_count > 0);
        assert!(!analytics.repetition.items.is_empty());
        assert!(!analytics.proximity.items.is_empty());
        assert_eq!(analytics.cadence.sentences.len(), 5);
    }

    #[test]
    fn fixture_structure_relations_match_expected_baselines() {
        let runtime = PhoenixRuntime::new(RuntimeConfig {
            target: RuntimeTarget::Wasm,
            ..RuntimeConfig::default()
        })
        .expect("runtime");
        let manifest = load_fixture_manifest();

        for fixture in &manifest.fixtures {
            let text = fixture_body(fixture);
            let scan = runtime.scan_text(ScanRequest {
                text: text.clone(),
                scope: ScopeKey::default(),
                session_id: Some(SessionId(format!("fixture-{}", fixture.id))),
                resolver_seed: fixture_resolver_seed(&fixture.id),
            });
            let structure = runtime.build_structure(StructureRequest { text, scan });
            let normalized = normalize_structure_relations(&structure);
            let baseline = load_expected_baseline(&fixture.id);
            assert_eq!(
                baseline.expected_structure,
                Some(normalized),
                "fixture {} structure relations drifted",
                fixture.id
            );
        }
    }

    fn fixture_resolver_seed(fixture_id: &str) -> Vec<phoenix_types::ResolverEntitySeed> {
        match fixture_id {
            "shortrun-opening" => vec![
                fixture_seed("ryan", "Ryan", EntityKind::Character),
                fixture_seed("bakuto", "Bakuto", EntityKind::Faction),
            ],
            "perfect-run-dialogue" => vec![
                fixture_seed("ryan", "Ryan", EntityKind::Character),
                fixture_seed("zanbato", "Zanbato", EntityKind::Character),
                fixture_seed("ghoul", "Ghoul", EntityKind::Character),
                fixture_seed("augusti", "Augusti", EntityKind::Organization),
                fixture_seed("meta_gang", "Meta-Gang", EntityKind::Faction),
                fixture_seed("len", "Len", EntityKind::Character),
            ],
            _ => Vec::new(),
        }
    }

    fn fixture_seed(
        id: &str,
        canonical_name: &str,
        kind: EntityKind,
    ) -> phoenix_types::ResolverEntitySeed {
        phoenix_types::ResolverEntitySeed {
            entity_id: EntityId(id.to_owned()),
            canonical_name: canonical_name.to_owned(),
            aliases: Vec::new(),
            kind: Some(kind),
            gender: Some(GenderHint::Unknown),
            number: None,
            scope: ScopeKey::default(),
        }
    }

    fn semantic_test_vector(primary_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; phoenix_store_cozo::SEMANTIC_VECTOR_DIM];
        if primary_index < values.len() {
            values[primary_index] = 1.0;
        }
        values
    }

    fn chat_runtime() -> PhoenixRuntime {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");
        runtime
    }

    fn wasm_runtime() -> PhoenixRuntime {
        let runtime = PhoenixRuntime::new(RuntimeConfig {
            target: RuntimeTarget::Wasm,
            ..RuntimeConfig::default()
        })
        .expect("runtime");
        runtime.init().expect("init");
        runtime
    }

    #[test]
    fn wasm_query_stays_on_gldr_diagnostics() {
        let runtime = wasm_runtime();
        let session = runtime
            .create_session(CreateSessionRequest {
                session_id: None,
                label: "Wasm query".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("session");

        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![phoenix_types::IngestDocument {
                    document_id: DocumentId("doc-wasm-query".to_owned()),
                    note_id: None,
                    title: "Harbor".to_owned(),
                    text: "Ryan mapped the harbor before dawn.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        let result = runtime
            .query(QueryRequest {
                session_id: Some(session.session_id.clone()),
                query: "Ryan".to_owned(),
                scope: ScopeKey::default(),
                targets: vec![QueryTarget::Graph],
                limit: Some(5),
                temporal: None,
                semantic_query_vector: None,
                include_candidate_graph: true,
            })
            .expect("query");

        assert!(result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "PX_GLDR_OK"));
        assert!(!result
            .diagnostics
            .iter()
            .any(|diag| diag.code == "PX_TRIVERSE_OK"));
    }

    fn run_options(
        narrative_id: &str,
        planner_enabled: bool,
        mutations_enabled: bool,
    ) -> RunOptions {
        RunOptions {
            final_provider: "openrouter".to_owned(),
            final_model: "meta-llama/llama-3.3-70b-instruct:free".to_owned(),
            planner_model: Some("meta-llama/llama-3.3-70b-instruct:free".to_owned()),
            om_model: None,
            planner_enabled,
            om_enabled: false,
            workspace_enabled: planner_enabled,
            mutations_enabled,
            deadline_ms: 8_000,
            mutation_policy: "confirm".to_owned(),
            narrative_id: Some(narrative_id.to_owned()),
            folder_id: None,
            scope_id: Some(narrative_id.to_owned()),
            base_system_prompt: Some("You are Kammi.".to_owned()),
            initial_external_context: None,
        }
    }

    fn insert_note(
        runtime: &PhoenixRuntime,
        note_id: &str,
        narrative_id: &str,
        title: &str,
        content: &str,
    ) {
        runtime
            .put_relation_row(
                "notes",
                json!({
                    "id": note_id,
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
                    "owner_id": null,
                    "narrative_id": narrative_id,
                    "order": 0,
                    "created_at": 10,
                    "updated_at": 10,
                    "valid_from": 10,
                    "valid_to": null,
                    "is_current": true,
                    "change_reason": null
                }),
            )
            .expect("note row");
    }

    fn planner_step_command(
        runtime: &PhoenixRuntime,
        command: &str,
        payload: Value,
    ) -> ChatPlannerStep {
        planner_step_from_payload(
            runtime
                .store_command(StoreCommandRequest {
                    command: command.to_owned(),
                    payload,
                })
                .expect("planner command")
                .payload
                .expect("planner payload"),
        )
    }

    #[test]
    fn native_rebuild_preserves_kernel_checkpoint_temporal_payloads() {
        let runtime = PhoenixRuntime::new(RuntimeConfig::default()).expect("runtime");
        runtime.init().expect("init");
        let store = runtime.v2_store().expect("native store");
        let snapshot = KernelGraphSnapshot {
            vertices: vec![KernelVertex {
                id: KernelVertexId("entity::dock".to_owned()),
                kind: "entity".to_owned(),
                class: KernelVertexClass::Entity,
                value: json!({"name":"Dock Authority"}),
                attributes: json!({}),
                temporal: KernelBiTemporal {
                    valid_from: Some(10),
                    valid_to: Some(30),
                    recorded_at: Some(20),
                    expired_at: None,
                },
                entity_id: Some("dock-authority".to_owned()),
                entity_facet: Some(KernelEntityFacet {
                    canonical_entity_id: Some("dock-authority".to_owned()),
                    surface: Some("Dock Authority".to_owned()),
                    entity_kind: Some("organization".to_owned()),
                }),
                ..KernelVertex::default()
            }],
            asserted_edges: Vec::new(),
            candidate_edges: Vec::new(),
        };
        store
            .write_kernel_checkpoint(7, "rev-7", &snapshot)
            .expect("write checkpoint");
        runtime.native_runtime.deterministic_kernel.invalidate();
        runtime
            .native_runtime
            .deterministic_kernel
            .set_rebuild_token(None);
        runtime
            .rebuild_native_graph("graph:test:checkpoint".to_owned())
            .expect("rebuild native graph");
        let view = runtime
            .native_runtime
            .deterministic_kernel
            .view_as_of(KernelViewRequest {
                valid_at: Some(15),
                recorded_at: Some(25),
                include_candidate_graph: false,
            });
        assert_eq!(view.vertices.len(), 1);
        assert_eq!(
            view.vertices[0].entity_facet.as_ref().and_then(|facet| facet.surface.as_deref()),
            Some("Dock Authority")
        );
        assert_eq!(view.vertices[0].temporal.recorded_at, Some(20));
    }

    fn planner_step_from_payload(payload: Value) -> ChatPlannerStep {
        serde_json::from_value(payload).expect("planner step")
    }

    fn normalize_structure_relations(structure: &StructureArtifact) -> Value {
        json!({
            "relations": structure.relations.iter().map(|relation| {
                json!({
                    "sentenceIndex": relation.sentence_index,
                    "lemma": relation.lemma,
                    "eventClass": relation.event_class,
                    "relationType": relation.relation_type,
                    "subjectEntity": frame_slot_entity_id(relation.subject.as_ref()),
                    "objectEntity": frame_slot_entity_id(relation.object.as_ref()),
                    "recipientEntity": frame_slot_entity_id(relation.recipient.as_ref()),
                    "subjectRange": frame_slot_range(relation.subject.as_ref()),
                    "objectRange": frame_slot_range(relation.object.as_ref()),
                    "recipientRange": frame_slot_range(relation.recipient.as_ref()),
                })
            }).collect::<Vec<_>>()
        })
    }

    fn frame_slot_entity_id(slot: Option<&phoenix_types::FrameSlot>) -> Option<String> {
        slot.and_then(|slot| match &slot.entity_ref {
            Some(MentionEntityRef::Known(entity_id)) => Some(entity_id.0.clone()),
            Some(MentionEntityRef::Speculative(key)) => Some(format!("spec:{key}")),
            None => None,
        })
    }

    fn frame_slot_range(slot: Option<&phoenix_types::FrameSlot>) -> Option<Value> {
        slot.map(|slot| {
            json!({
                "start": slot.range.start,
                "end": slot.range.end,
            })
        })
    }
}
