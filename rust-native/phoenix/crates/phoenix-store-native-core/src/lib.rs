use phoenix_graph::{GraphMutationBatch, GraptorGraph};
use phoenix_graph_kernel::{
    KernelCheckpointData, KernelGraphSnapshot, KernelJournalEntry, KernelMutationBatch,
};
use phoenix_semantic_v2::{
    AliasPosting, CausalScopeSidecar, DirtyScopeRecord, DocumentArchive, DocumentManifest,
    DocumentOrdinalAssignment, DocumentRevisionRef, ErScopePatchSidecar, MemoryScopeSidecar,
    PreparedDocument, RelationMentionSeedScopeSidecar, RelationScopePatchSidecar,
    ScopeLexSidecar, ScopeOrd, SessionArchive, SessionOrd,
};
use phoenix_types::{IndexedSpan, IngestDocument, ScopeKey, SessionId};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;

pub mod schema;
pub use schema::{
    PhoenixColumnSpec, PhoenixColumnType, PhoenixRelationSpec, ALL_RELATIONS,
    CONTENT_SNAPSHOT_RELATIONS, DERIVED_SNAPSHOT_RELATIONS,
};

pub const SEMANTIC_VECTOR_DIM: usize = 384;
pub const SEMANTIC_MODEL_ID: &str = "MongoDB/mdbr-leaf-ir";

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotEnvelope {
    pub schema_version: String,
    pub relation_count: usize,
    pub created_at: i64,
    pub relations: std::collections::BTreeMap<String, Vec<Value>>,
    pub checksum: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SnapshotPartition {
    All,
    Content,
    Derived,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticNeighbor {
    pub span_id: String,
    pub distance: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticDocumentNeighbor {
    pub document_id: String,
    pub distance: f64,
    pub leaf_count: usize,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticNodeNeighbor {
    pub node_id: String,
    pub node_kind: String,
    pub distance: f64,
    pub document_id: Option<String>,
    pub narrative_id: Option<String>,
    pub folder_id: Option<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("unknown relation: {0}")]
    UnknownRelation(String),
    #[error("row must be a JSON object")]
    InvalidRow,
    #[error("missing required column '{column}' in relation '{relation}'")]
    MissingColumn { relation: String, column: String },
    #[error("native store init failed: {0}")]
    Init(String),
    #[error("native store schema failed: {0}")]
    Schema(String),
    #[error("native store query failed: {0}")]
    Query(String),
    #[error("snapshot decode failed: {0}")]
    Snapshot(String),
}

pub const NATIVE_COVERED_RELATIONS: &[&str] = &[
    "phoenix_schema_state",
    "phoenix_sessions",
    "phoenix_commits",
    "phoenix_ingest_log",
    "phoenix_query_log",
    "notes",
    "entities",
    "edges",
    "folders",
    "docid_map",
    "chunkid_map",
    "chunks",
    "raptor_nodes",
    "raptor_edges",
    "episodes",
    "spans",
    "wormholes",
    "span_mentions",
    "discovery_candidates",
    "blocks",
    "document_boundaries",
    "entity_cards",
    "folder_schemas",
    "scoped_documents",
    "scoped_entity_fields",
    "scoped_definitions",
    "network_instance",
    "network_membership",
    "network_relationship",
];

pub const NATIVE_GRAPH_COMPAT_RELATIONS: &[&str] = &[
    "graph_vertices",
    "graph_edges",
    "graph_candidate_edges",
    "graph_node_index",
    "graph_properties",
    "graph_vertex_labels",
];

pub const PHOENIX_GRAPH_CHECKPOINT_META: &[PhoenixColumnSpec] = &[
    PhoenixColumnSpec::new("checkpoint_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("generation", PhoenixColumnType::Int, false, false),
    PhoenixColumnSpec::new("source_revision", PhoenixColumnType::String, false, false),
    PhoenixColumnSpec::new("created_at", PhoenixColumnType::Int, false, false),
];

pub const PHOENIX_GRAPH_VERTEX_SNAPSHOT: &[PhoenixColumnSpec] = &[
    PhoenixColumnSpec::new("checkpoint_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("vertex_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("record_json", PhoenixColumnType::Json, false, false),
    PhoenixColumnSpec::new("created_at", PhoenixColumnType::Int, false, false),
];

pub const PHOENIX_GRAPH_EDGE_SNAPSHOT: &[PhoenixColumnSpec] = &[
    PhoenixColumnSpec::new("checkpoint_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("layer", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("source_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("target_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("edge_type", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("record_json", PhoenixColumnType::Json, false, false),
    PhoenixColumnSpec::new("created_at", PhoenixColumnType::Int, false, false),
];

pub const PHOENIX_GRAPH_DELTA_JOURNAL: &[PhoenixColumnSpec] = &[
    PhoenixColumnSpec::new("entry_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("generation", PhoenixColumnType::Int, false, false),
    PhoenixColumnSpec::new("source_revision", PhoenixColumnType::String, false, false),
    PhoenixColumnSpec::new("batch_json", PhoenixColumnType::Json, true, false),
    PhoenixColumnSpec::new("commit_id", PhoenixColumnType::String, true, false),
    PhoenixColumnSpec::new("created_at", PhoenixColumnType::Int, false, false),
];

pub const INTERNAL_RELATIONS: &[PhoenixRelationSpec] = &[
    PhoenixRelationSpec::new(
        "phoenix_graph_checkpoint_meta",
        PHOENIX_GRAPH_CHECKPOINT_META,
    ),
    PhoenixRelationSpec::new(
        "phoenix_graph_vertex_snapshot",
        PHOENIX_GRAPH_VERTEX_SNAPSHOT,
    ),
    PhoenixRelationSpec::new("phoenix_graph_edge_snapshot", PHOENIX_GRAPH_EDGE_SNAPSHOT),
    PhoenixRelationSpec::new("phoenix_graph_delta_journal", PHOENIX_GRAPH_DELTA_JOURNAL),
];

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScopedDocumentFilter<'a> {
    pub namespace: Option<&'a str>,
    pub narrative_id: Option<&'a str>,
    pub document_key: Option<&'a str>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ScopedDefinitionFilter<'a> {
    pub namespace: Option<&'a str>,
    pub narrative_id: Option<&'a str>,
    pub definition_key: Option<&'a str>,
    pub definition_key_prefix: Option<&'a str>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphCheckpointMeta {
    pub checkpoint_id: String,
    pub generation: u64,
    pub source_revision: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphCheckpointData {
    pub meta: GraphCheckpointMeta,
    pub asserted_batch: GraphMutationBatch,
    pub candidate_batch: GraphMutationBatch,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphJournalEntry {
    pub generation: u64,
    pub source_revision: String,
    pub batch: Option<GraphMutationBatch>,
    pub commit_id: Option<String>,
    pub created_at: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BundleKind {
    #[default]
    DocumentArchive,
    SessionArchive,
    ScopeLexSidecar,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BundleKey {
    pub kind: BundleKind,
    pub scope: String,
    pub entity_key: String,
    pub revision: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BundleHeader {
    pub key: BundleKey,
    pub byte_len: usize,
    pub created_at: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum IngestMode {
    #[default]
    Safe,
    BulkBuild,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedIngestContext {
    pub session_id: Option<SessionId>,
    pub session_ord: Option<SessionOrd>,
    #[serde(default)]
    pub assignments: Vec<DocumentOrdinalAssignment>,
    pub kernel_snapshot: Option<KernelGraphSnapshot>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AnnIndexFamily {
    Document,
    Leaf,
    NodePrototype,
}

#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct AnnGenerationId(pub u64);

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnnIndexKey {
    pub scope_ord: ScopeOrd,
    pub family: AnnIndexFamily,
    pub kind: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnnManifest {
    pub index: AnnIndexKey,
    pub generation_id: AnnGenerationId,
    pub built_at: i64,
    pub dimension: usize,
    pub model_id: String,
    pub count: usize,
    pub entry_point: u32,
    pub max_level: u32,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub level_mult: f32,
    pub metric: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnnPackedSegments {
    pub vectors: Vec<u8>,
    pub levels: Vec<u8>,
    pub offsets: Vec<u8>,
    pub adjacency: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeSemanticLeafVectorRecord {
    pub scope: ScopeKey,
    pub span_id: String,
    pub document_id: String,
    pub values: Vec<f32>,
    pub updated_at: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeSemanticDocumentVectorRecord {
    pub scope: ScopeKey,
    pub document_id: String,
    pub values: Vec<f32>,
    pub leaf_count: usize,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub updated_at: i64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeSemanticNodeVectorRecord {
    pub scope: ScopeKey,
    pub node_id: String,
    pub node_kind: String,
    pub document_id: Option<String>,
    pub narrative_id: Option<String>,
    pub folder_id: Option<String>,
    pub values: Vec<f32>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub updated_at: i64,
}

pub trait PhoenixNativeRowStore {
    fn init_schema(&self) -> Result<(), StoreError>;
    fn relation_names(&self) -> Vec<&'static str>;
    fn relation_counts(&self) -> Result<Vec<(String, usize)>, StoreError>;
    fn fetch_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError>;
    fn put_row(&self, relation: &str, row: Value) -> Result<(), StoreError>;
    fn put_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError>;
    fn replace_relation_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError>;
    fn delete_rows(&self, relation: &str, rows: &[Value]) -> Result<usize, StoreError>;
    fn clear_relations(&self, relations: &[&str]) -> Result<(), StoreError>;
    fn export_snapshot_partition(
        &self,
        partition: SnapshotPartition,
    ) -> Result<Vec<u8>, StoreError>;
    fn import_snapshot(&self, bytes: &[u8]) -> Result<SnapshotEnvelope, StoreError>;

    fn fetch_scoped_documents(
        &self,
        filter: ScopedDocumentFilter<'_>,
    ) -> Result<Vec<Value>, StoreError> {
        let mut rows = self.fetch_rows("scoped_documents")?;
        rows.retain(|row| matches_scoped_document_filter(row, &filter));
        Ok(rows)
    }

    fn fetch_scoped_definitions(
        &self,
        filter: ScopedDefinitionFilter<'_>,
    ) -> Result<Vec<Value>, StoreError> {
        let mut rows = self.fetch_rows("scoped_definitions")?;
        rows.retain(|row| matches_scoped_definition_filter(row, &filter));
        Ok(rows)
    }
}

pub trait PhoenixGraphDurabilityStore {
    fn load_graph_checkpoint(&self) -> Result<Option<GraphCheckpointData>, StoreError>;
    fn write_graph_checkpoint(
        &self,
        generation: u64,
        source_revision: &str,
        graph: &GraptorGraph,
    ) -> Result<GraphCheckpointData, StoreError>;
    fn load_graph_journal_after(
        &self,
        generation: u64,
    ) -> Result<Vec<GraphJournalEntry>, StoreError>;
    fn append_graph_batch(
        &self,
        generation: u64,
        source_revision: &str,
        batch: &GraphMutationBatch,
        created_at: i64,
    ) -> Result<(), StoreError>;
    fn append_commit_marker(
        &self,
        generation: u64,
        source_revision: &str,
        commit_id: &str,
        created_at: i64,
    ) -> Result<(), StoreError>;
    fn generation_for_commit(&self, commit_id: &str) -> Result<Option<u64>, StoreError>;
    fn current_generation(&self) -> Result<u64, StoreError>;
    fn journal_len(&self) -> Result<usize, StoreError>;
}

pub trait PhoenixBundleStoreV2 {
    fn init_bundle_schema(&self) -> Result<(), StoreError>;
    fn put_bundle(&self, header: &BundleHeader, payload: &[u8]) -> Result<(), StoreError>;
    fn get_bundle(&self, key: &BundleKey) -> Result<Option<Vec<u8>>, StoreError>;
    fn get_bundle_header(&self, key: &BundleKey) -> Result<Option<BundleHeader>, StoreError>;
    fn list_bundle_headers(
        &self,
        kind: BundleKind,
        scope: Option<&str>,
    ) -> Result<Vec<BundleHeader>, StoreError>;
    fn delete_bundle(&self, key: &BundleKey) -> Result<bool, StoreError>;
}

pub trait PhoenixArchiveStoreV2 {
    fn init_archive_schema(&self) -> Result<(), StoreError>;
    fn ingest_mode(&self) -> IngestMode;
    fn prepare_ingest_context(
        &self,
        session_id: Option<&SessionId>,
        documents: &[IngestDocument],
        revision: u64,
    ) -> Result<PreparedIngestContext, StoreError>;
    fn persist_prepared_documents(
        &self,
        prepared: &[PreparedDocument],
        session_archive: Option<&SessionArchive>,
        touched_scopes: &[DirtyScopeRecord],
        created_at: i64,
    ) -> Result<(), StoreError>;
    fn persist_session_archive(
        &self,
        archive: &SessionArchive,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError>;
    fn load_latest_session_archive(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<SessionArchive>, StoreError>;
    fn load_latest_document_archives(
        &self,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<DocumentArchive>, StoreError>;
    fn load_document_manifest(
        &self,
        document_ref: &DocumentRevisionRef,
    ) -> Result<Option<DocumentManifest>, StoreError>;
    fn load_scope_sidecar(&self, scope: &ScopeKey) -> Result<Option<ScopeLexSidecar>, StoreError>;
    fn load_materialized_scope_lexical(
        &self,
        scope: &ScopeKey,
    ) -> Result<ScopeLexSidecar, StoreError>;
    fn load_lex_spans(&self, scope: Option<&ScopeKey>) -> Result<Vec<IndexedSpan>, StoreError>;
    fn lookup_alias_postings(
        &self,
        scope: &ScopeKey,
        normalized: &str,
    ) -> Result<Vec<AliasPosting>, StoreError>;
    fn rebuild_dirty_scope_sidecars(&self, created_at: i64) -> Result<usize, StoreError>;
    fn list_dirty_scopes(&self) -> Result<Vec<DirtyScopeRecord>, StoreError>;
}

pub trait PhoenixErPatchStore {
    fn init_er_patch_schema(&self) -> Result<(), StoreError>;
    fn persist_er_patch_sidecar(&self, sidecar: &ErScopePatchSidecar) -> Result<(), StoreError>;
    fn load_er_patch_sidecar(
        &self,
        scope: &ScopeKey,
    ) -> Result<Option<ErScopePatchSidecar>, StoreError>;
}

pub trait PhoenixRelationPatchStore {
    fn init_relation_patch_schema(&self) -> Result<(), StoreError>;
    fn persist_relation_patch_sidecar(
        &self,
        sidecar: &RelationScopePatchSidecar,
    ) -> Result<(), StoreError>;
    fn load_relation_patch_sidecar(
        &self,
        scope: &ScopeKey,
    ) -> Result<Option<RelationScopePatchSidecar>, StoreError>;
}

pub trait PhoenixMemoryPatchStore {
    fn init_memory_patch_schema(&self) -> Result<(), StoreError>;
    fn persist_memory_patch_sidecar(&self, sidecar: &MemoryScopeSidecar) -> Result<(), StoreError>;
    fn load_memory_patch_sidecar(
        &self,
        scope: &ScopeKey,
    ) -> Result<Option<MemoryScopeSidecar>, StoreError>;
}

pub trait PhoenixCausalPatchStore {
    fn init_causal_patch_schema(&self) -> Result<(), StoreError>;
    fn persist_causal_patch_sidecar(&self, sidecar: &CausalScopeSidecar) -> Result<(), StoreError>;
    fn load_causal_patch_sidecar(
        &self,
        scope: &ScopeKey,
    ) -> Result<Option<CausalScopeSidecar>, StoreError>;
}

pub trait PhoenixRelationMentionSeedStore {
    fn init_relation_mention_seed_schema(&self) -> Result<(), StoreError>;
    fn persist_relation_mention_seed_sidecar(
        &self,
        sidecar: &RelationMentionSeedScopeSidecar,
    ) -> Result<(), StoreError>;
    fn load_relation_mention_seed_sidecar(
        &self,
        scope: &ScopeKey,
    ) -> Result<Option<RelationMentionSeedScopeSidecar>, StoreError>;
}

pub trait PhoenixSemanticIndexStore {
    fn semantic_model_id(&self) -> &'static str {
        SEMANTIC_MODEL_ID
    }

    fn semantic_vector_dim(&self) -> usize {
        SEMANTIC_VECTOR_DIM
    }

    fn upsert_semantic_leaf_vectors(
        &self,
        rows: &[NativeSemanticLeafVectorRecord],
    ) -> Result<(), StoreError>;
    fn upsert_semantic_document_vectors_native(
        &self,
        rows: &[NativeSemanticDocumentVectorRecord],
    ) -> Result<(), StoreError>;
    fn upsert_semantic_node_vectors_native(
        &self,
        rows: &[NativeSemanticNodeVectorRecord],
    ) -> Result<(), StoreError>;
    fn query_semantic_neighbors(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticNeighbor>, StoreError>;
    fn query_semantic_documents(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticDocumentNeighbor>, StoreError>;
    fn query_semantic_neighbors_in_documents(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        document_ids: &[String],
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticNeighbor>, StoreError>;
    fn query_semantic_node_neighbors(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        kind: &str,
        exclude_node_id: Option<&str>,
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticNodeNeighbor>, StoreError>;
    fn load_semantic_document_vector_records(
        &self,
        document_ids: &[String],
    ) -> Result<Vec<NativeSemanticDocumentVectorRecord>, StoreError>;
    fn load_semantic_node_vector_records(
        &self,
        node_ids: &[String],
    ) -> Result<Vec<NativeSemanticNodeVectorRecord>, StoreError>;
}

pub trait PhoenixDirectGraphStoreV2 {
    fn init_direct_graph_schema(&self) -> Result<(), StoreError>;
    fn load_direct_graph_checkpoint(&self) -> Result<Option<GraphCheckpointData>, StoreError>;
    fn write_direct_graph_checkpoint(
        &self,
        generation: u64,
        source_revision: &str,
        graph: &GraptorGraph,
    ) -> Result<GraphCheckpointData, StoreError>;
    fn load_direct_graph_journal_after(
        &self,
        generation: u64,
    ) -> Result<Vec<GraphJournalEntry>, StoreError>;
    fn append_direct_graph_batch(
        &self,
        generation: u64,
        source_revision: &str,
        batch: &GraphMutationBatch,
        created_at: i64,
    ) -> Result<(), StoreError>;
    fn append_direct_graph_commit_marker(
        &self,
        generation: u64,
        source_revision: &str,
        commit_id: &str,
        created_at: i64,
    ) -> Result<(), StoreError>;
    fn direct_graph_generation_for_commit(
        &self,
        commit_id: &str,
    ) -> Result<Option<u64>, StoreError>;
    fn direct_graph_current_generation(&self) -> Result<u64, StoreError>;
    fn direct_graph_journal_len(&self) -> Result<usize, StoreError>;
}

pub trait PhoenixGraphKernelStoreV2 {
    fn init_graph_kernel_schema(&self) -> Result<(), StoreError>;
    fn load_kernel_checkpoint(&self) -> Result<Option<KernelCheckpointData>, StoreError>;
    fn write_kernel_checkpoint(
        &self,
        generation: u64,
        source_revision: &str,
        snapshot: &KernelGraphSnapshot,
    ) -> Result<KernelCheckpointData, StoreError>;
    fn load_kernel_journal_after(
        &self,
        generation: u64,
    ) -> Result<Vec<KernelJournalEntry>, StoreError>;
    fn append_kernel_batch(
        &self,
        generation: u64,
        source_revision: &str,
        batch: &KernelMutationBatch,
        created_at: i64,
    ) -> Result<(), StoreError>;
    fn append_kernel_commit_marker(
        &self,
        generation: u64,
        source_revision: &str,
        commit_id: &str,
        created_at: i64,
    ) -> Result<(), StoreError>;
    fn kernel_generation_for_commit(&self, commit_id: &str) -> Result<Option<u64>, StoreError>;
    fn kernel_current_generation(&self) -> Result<u64, StoreError>;
    fn kernel_journal_len(&self) -> Result<usize, StoreError>;
}

pub trait PhoenixNativeStore: PhoenixNativeRowStore + PhoenixGraphDurabilityStore {}

impl<T> PhoenixNativeStore for T where T: PhoenixNativeRowStore + PhoenixGraphDurabilityStore {}

pub fn relation_spec(relation: &str) -> Result<&'static PhoenixRelationSpec, StoreError> {
    ALL_RELATIONS
        .iter()
        .chain(INTERNAL_RELATIONS.iter())
        .find(|spec| spec.name == relation)
        .ok_or_else(|| StoreError::UnknownRelation(relation.to_owned()))
}

pub fn internal_relation_spec(relation: &str) -> Option<&'static PhoenixRelationSpec> {
    INTERNAL_RELATIONS.iter().find(|spec| spec.name == relation)
}

pub fn is_covered_relation(relation: &str) -> bool {
    NATIVE_COVERED_RELATIONS.contains(&relation)
}

pub fn is_graph_compat_relation(relation: &str) -> bool {
    NATIVE_GRAPH_COMPAT_RELATIONS.contains(&relation)
}

pub fn is_supported_relation(relation: &str) -> bool {
    is_covered_relation(relation) || internal_relation_spec(relation).is_some()
}

pub fn snapshot_relations_for_partition(partition: SnapshotPartition) -> &'static [&'static str] {
    match partition {
        SnapshotPartition::All => CONTENT_SNAPSHOT_RELATIONS,
        SnapshotPartition::Content => CONTENT_SNAPSHOT_RELATIONS,
        SnapshotPartition::Derived => DERIVED_SNAPSHOT_RELATIONS,
    }
}

pub fn matches_scoped_document_filter(row: &Value, filter: &ScopedDocumentFilter<'_>) -> bool {
    string_field_matches(row, "namespace", filter.namespace)
        && string_field_matches(row, "narrative_id", filter.narrative_id)
        && string_field_matches(row, "document_key", filter.document_key)
}

pub fn matches_scoped_definition_filter(row: &Value, filter: &ScopedDefinitionFilter<'_>) -> bool {
    string_field_matches(row, "namespace", filter.namespace)
        && string_field_matches(row, "narrative_id", filter.narrative_id)
        && string_field_matches(row, "definition_key", filter.definition_key)
        && string_prefix_field_matches(row, "definition_key", filter.definition_key_prefix)
}

fn string_field_matches(row: &Value, field: &str, expected: Option<&str>) -> bool {
    match expected {
        Some(expected) => row.get(field).and_then(Value::as_str) == Some(expected),
        None => true,
    }
}

fn string_prefix_field_matches(row: &Value, field: &str, expected_prefix: Option<&str>) -> bool {
    match expected_prefix {
        Some(expected_prefix) => row
            .get(field)
            .and_then(Value::as_str)
            .map(|value| value.starts_with(expected_prefix))
            .unwrap_or(false),
        None => true,
    }
}
