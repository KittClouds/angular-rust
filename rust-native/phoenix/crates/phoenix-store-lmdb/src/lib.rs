use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

use heed3::types::{Bytes, Str};
use heed3::{Database, DatabaseFlags, DatabaseOpenOptions, Env, EnvFlags, EnvOpenOptions, PutFlags, RwTxn};
use phoenix_kernel::{
    DeterministicKernel, KernelCheckpointData, KernelCheckpointMeta, KernelGraphLayer,
    KernelGraphSnapshot, KernelJournalEntry, KernelMutationBatch, KernelMutationScope,
};
use phoenix_hyperbolic::{
    Candidate as HnswCandidate, HnswBuildParams, HyperbolicDiskHnsw, HyperbolicHnswBuilder,
    PackedHnswGraph, PoincareMetric,
};
use phoenix_semantic_v2::{
    scope_storage_key, AliasEntry, AliasPosting, DirtyScopeRecord, DocumentArchive,
    DocumentManifest, DocumentOrd, DocumentOrdinalAssignment, DocumentRevisionRef,
    DocumentSegmentKind, LexicalPostingsSegment, PreparedDocument, PreparedDocumentSegment,
    ScopeLexSidecar, ScopeOrd, SessionArchive, SessionOrd,
};
use phoenix_store_cozo::{
    SemanticDocumentNeighbor, SemanticNeighbor, SemanticNodeNeighbor, SnapshotPartition,
    StoreError, SEMANTIC_MODEL_ID, SEMANTIC_VECTOR_DIM,
};
use phoenix_store_native_core::{
    BundleHeader as CoreBundleHeader, BundleKey as CoreBundleKey, BundleKind as CoreBundleKind,
    IngestMode as CoreIngestMode, PhoenixArchiveStoreV2 as CorePhoenixArchiveStoreV2,
    PhoenixBundleStoreV2 as CorePhoenixBundleStoreV2, StoreError as CoreStoreError,
};
use phoenix_store_native::{
    relation_spec, AnnGenerationId, AnnIndexFamily, AnnIndexKey, AnnManifest,
    BundleHeader, BundleKey, BundleKind, IngestMode, NativeSemanticDocumentVectorRecord,
    NativeSemanticLeafVectorRecord, NativeSemanticNodeVectorRecord, PhoenixArchiveStoreV2,
    PhoenixBundleStoreV2, PhoenixGraphKernelStoreV2, PhoenixNativeRowStore,
    PhoenixSemanticIndexStore, PreparedIngestContext, NATIVE_COVERED_RELATIONS,
};
use phoenix_types::{IndexedSpan, IngestDocument, ScopeKey, SessionId};
use serde::{de::DeserializeOwned, Serialize};
use serde_json::Value;

const LMDB_MAP_SIZE: usize = 1 << 30;
const LMDB_MAX_DBS: u32 = 64;
const LMDB_PERIODIC_SYNC_EVERY_COMMITS: u64 = 32;
const CHECKPOINT_KEY: &str = "current";
const PHOENIX_NATIVE_SCHEMA_VERSION: &str = "phoenix.native.v2";
const COMPAT_RELATION_ROWS_DB: &str = "compat.relation_rows";
const COMPAT_BUNDLE_HEADERS_DB: &str = "compat.bundle_header";
const COMPAT_BUNDLE_PAYLOADS_DB: &str = "compat.bundle_payload";
const SCOPE_BY_VALUE_DB: &str = "id.scope_by_value";
const SCOPE_BY_ORD_DB: &str = "id.scope_by_ord";
const DOC_BY_VALUE_DB: &str = "id.doc_by_value";
const DOC_BY_ORD_DB: &str = "id.doc_by_ord";
const SESSION_BY_VALUE_DB: &str = "id.session_by_value";
const SESSION_BY_ORD_DB: &str = "id.session_by_ord";
const ORD_COUNTER_DB: &str = "id.ord_counter";
const DOC_MANIFEST_DB: &str = "doc.manifest";
const DOC_SEGMENT_DB: &str = "doc.segment";
const SESSION_ARCHIVE_DB: &str = "session.archive";
const SCOPE_SIDECAR_META_DB: &str = "scope.sidecar.meta";
const SCOPE_SIDECAR_FST_DB: &str = "scope.sidecar.fst";
const SCOPE_SIDECAR_POSTINGS_DB: &str = "scope.sidecar.postings";
const IDX_DOC_LATEST_DB: &str = "idx.doc_latest";
const IDX_SESSION_LATEST_DB: &str = "idx.session_latest";
const IDX_SESSION_DOCS_DB: &str = "idx.session_docs";
const IDX_SCOPE_DOCS_DB: &str = "idx.scope_docs";
const IDX_SCOPE_DIRTY_DB: &str = "idx.scope_dirty";
const GRAPH_KERNEL_CHECKPOINT_DB: &str = "graph.kernel.checkpoint";
const GRAPH_KERNEL_JOURNAL_DB: &str = "graph.kernel.journal";
const GRAPH_KERNEL_COMMIT_INDEX_DB: &str = "graph.kernel.commit_index";
const ANN_SOURCE_LEAF_DB: &str = "ann.source.leaf";
const ANN_SOURCE_DOCUMENT_DB: &str = "ann.source.document";
const ANN_SOURCE_NODE_DB: &str = "ann.source.node";
const ANN_HEAD_DB: &str = "ann.head";
const ANN_MANIFEST_DB: &str = "ann.manifest";
const ANN_VECTORS_DB: &str = "ann.vectors";
const ANN_LEVELS_DB: &str = "ann.levels";
const ANN_OFFSETS_DB: &str = "ann.offsets";
const ANN_ADJACENCY_DB: &str = "ann.adjacency";
const ANN_ID_BY_ORD_DB: &str = "ann.id_by_ord";
const ANN_ORD_BY_ID_DB: &str = "ann.ord_by_id";
const ANN_PAYLOAD_DB: &str = "ann.payload";
const ANN_DIRTY_DB: &str = "ann.dirty";
const ORD_COUNTER_SCOPE_KEY: &[u8] = b"scope";
const ORD_COUNTER_SESSION_KEY: &[u8] = b"session";
const ORD_COUNTER_DOCUMENT_KEY: &[u8] = b"document";
const NATIVE_SNAPSHOT_MAGIC: &[u8; 8] = b"PXNATV01";
const SEP: char = '\u{1f}';
const NATIVE_AUTHORITATIVE_RELATIONS: &[&str] = &[
    "phoenix_schema_state",
    "phoenix_sessions",
    "phoenix_commits",
    "phoenix_ingest_log",
    "notes",
    "entities",
    "edges",
    "folders",
    "entity_cards",
    "folder_schemas",
    "network_instance",
    "network_membership",
    "network_relationship",
];

fn core_store_error(error: StoreError) -> CoreStoreError {
    CoreStoreError::Query(error.to_string())
}

fn core_bundle_kind(kind: BundleKind) -> CoreBundleKind {
    match kind {
        BundleKind::DocumentArchive => CoreBundleKind::DocumentArchive,
        BundleKind::SessionArchive => CoreBundleKind::SessionArchive,
        BundleKind::ScopeLexSidecar => CoreBundleKind::ScopeLexSidecar,
    }
}

fn legacy_bundle_kind(kind: CoreBundleKind) -> BundleKind {
    match kind {
        CoreBundleKind::DocumentArchive => BundleKind::DocumentArchive,
        CoreBundleKind::SessionArchive => BundleKind::SessionArchive,
        CoreBundleKind::ScopeLexSidecar => BundleKind::ScopeLexSidecar,
    }
}

fn core_bundle_key(key: BundleKey) -> CoreBundleKey {
    CoreBundleKey {
        kind: core_bundle_kind(key.kind),
        scope: key.scope,
        entity_key: key.entity_key,
        revision: key.revision,
    }
}

fn legacy_bundle_key(key: &CoreBundleKey) -> BundleKey {
    BundleKey {
        kind: legacy_bundle_kind(key.kind),
        scope: key.scope.clone(),
        entity_key: key.entity_key.clone(),
        revision: key.revision,
    }
}

fn core_bundle_header(header: BundleHeader) -> CoreBundleHeader {
    CoreBundleHeader {
        key: core_bundle_key(header.key),
        byte_len: header.byte_len,
        created_at: header.created_at,
    }
}

fn legacy_bundle_header(header: &CoreBundleHeader) -> BundleHeader {
    BundleHeader {
        key: legacy_bundle_key(&header.key),
        byte_len: header.byte_len,
        created_at: header.created_at,
    }
}

fn core_ingest_mode(mode: IngestMode) -> CoreIngestMode {
    match mode {
        IngestMode::Safe => CoreIngestMode::Safe,
        IngestMode::BulkBuild => CoreIngestMode::BulkBuild,
    }
}

fn core_prepared_ingest_context(
    context: phoenix_store_native::PreparedIngestContext,
) -> phoenix_store_native_core::PreparedIngestContext {
    phoenix_store_native_core::PreparedIngestContext {
        session_id: context.session_id,
        session_ord: context.session_ord,
        assignments: context.assignments,
        kernel_snapshot: context.kernel_snapshot,
    }
}

static JOURNAL_ENTRY_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IngestDurabilityMode {
    Safe,
    NoMetaSync,
    NoSync,
    NoSyncPeriodic { sync_every_commits: u64 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LmdbTuning {
    pub map_size_bytes: usize,
    pub max_dbs: u32,
    pub max_readers: Option<u32>,
    pub durability: IngestDurabilityMode,
    pub no_read_ahead: bool,
    pub mem_init: bool,
    pub write_map: bool,
    pub map_async: bool,
}

impl Default for LmdbTuning {
    fn default() -> Self {
        Self {
            map_size_bytes: LMDB_MAP_SIZE,
            max_dbs: LMDB_MAX_DBS,
            max_readers: None,
            durability: IngestDurabilityMode::Safe,
            no_read_ahead: false,
            mem_init: true,
            write_map: false,
            map_async: false,
        }
    }
}

impl LmdbTuning {
    fn from_env() -> Self {
        let mut tuning = Self::default();
        if let Some(value) = read_env_usize("PHOENIX_LMDB_MAP_SIZE") {
            tuning.map_size_bytes = value;
        }
        if let Some(value) = read_env_u32("PHOENIX_LMDB_MAX_DBS") {
            tuning.max_dbs = value;
        }
        tuning.max_readers = read_env_u32("PHOENIX_LMDB_MAX_READERS");
        tuning.no_read_ahead = read_env_bool("PHOENIX_LMDB_NO_READ_AHEAD").unwrap_or(false);
        tuning.mem_init = !read_env_bool("PHOENIX_LMDB_NO_MEM_INIT").unwrap_or(false);
        tuning.write_map = read_env_bool("PHOENIX_LMDB_WRITE_MAP").unwrap_or(false);
        tuning.map_async = read_env_bool("PHOENIX_LMDB_MAP_ASYNC").unwrap_or(false);
        if let Some(mode) = std::env::var("PHOENIX_LMDB_DURABILITY").ok() {
            tuning.durability = match mode.to_ascii_lowercase().as_str() {
                "nometasync" | "no_meta_sync" => IngestDurabilityMode::NoMetaSync,
                "nosync" | "no_sync" => IngestDurabilityMode::NoSync,
                "nosync_periodic" | "no_sync_periodic" => IngestDurabilityMode::NoSyncPeriodic {
                    sync_every_commits: read_env_u64("PHOENIX_LMDB_SYNC_EVERY_COMMITS")
                        .unwrap_or(LMDB_PERIODIC_SYNC_EVERY_COMMITS),
                },
                _ => IngestDurabilityMode::Safe,
            };
        }
        tuning
    }
}

#[derive(Clone, Debug)]
pub struct LmdbWriteBatch {
    pub puts: Vec<LmdbPut>,
}

#[derive(Clone, Debug)]
pub struct LmdbPut {
    pub db: LmdbDbId,
    pub key: LmdbKey,
    pub value: Vec<u8>,
    pub flags: PutFlags,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LmdbKey {
    Str(String),
    Bytes(Vec<u8>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LmdbDbId {
    CompatBundleHeader,
    DocManifest,
    DocSegment,
    IdxDocLatest,
    IdxScopeDocs,
    IdxScopeDirty,
    SessionArchive,
    IdxSessionLatest,
    IdxSessionDocs,
}

pub struct PhoenixLmdbStore {
    env: Env,
    path: PathBuf,
    tuning: LmdbTuning,
    compat_relation_rows: Database<Str, Bytes>,
    compat_bundle_headers: Database<Str, Bytes>,
    compat_bundle_payloads: Database<Str, Bytes>,
    scope_by_value: Database<Str, Bytes>,
    scope_by_ord: Database<Bytes, Bytes>,
    doc_by_value: Database<Str, Bytes>,
    doc_by_ord: Database<Bytes, Bytes>,
    session_by_value: Database<Str, Bytes>,
    session_by_ord: Database<Bytes, Bytes>,
    ord_counter: Database<Bytes, Bytes>,
    doc_manifest: Database<Bytes, Bytes>,
    doc_segment: Database<Bytes, Bytes>,
    session_archive: Database<Bytes, Bytes>,
    scope_sidecar_meta: Database<Bytes, Bytes>,
    scope_sidecar_fst: Database<Bytes, Bytes>,
    scope_sidecar_postings: Database<Bytes, Bytes>,
    idx_doc_latest: Database<Bytes, Bytes>,
    idx_session_latest: Database<Bytes, Bytes>,
    idx_session_docs: Database<Bytes, Bytes>,
    idx_scope_docs: Database<Bytes, Bytes>,
    idx_scope_dirty: Database<Bytes, Bytes>,
    graph_kernel_checkpoint: Database<Str, Bytes>,
    graph_kernel_journal: Database<Str, Bytes>,
    graph_kernel_commit_index: Database<Str, Bytes>,
    ann_source_leaf: Database<Bytes, Bytes>,
    ann_source_document: Database<Bytes, Bytes>,
    ann_source_node: Database<Bytes, Bytes>,
    ann_head: Database<Bytes, Bytes>,
    ann_manifest: Database<Bytes, Bytes>,
    ann_vectors: Database<Bytes, Bytes>,
    ann_levels: Database<Bytes, Bytes>,
    ann_offsets: Database<Bytes, Bytes>,
    ann_adjacency: Database<Bytes, Bytes>,
    ann_id_by_ord: Database<Bytes, Bytes>,
    ann_ord_by_id: Database<Bytes, Bytes>,
    ann_payload: Database<Bytes, Bytes>,
    ann_dirty: Database<Bytes, Bytes>,
    live_kernel_snapshot: RwLock<Option<(u64, KernelGraphSnapshot)>>,
    live_kernel_generation: AtomicU64,
    commit_count: AtomicU64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct NativeSnapshotEnvelope {
    schema_version: String,
    partition: String,
    created_at: i64,
    relation_rows: Vec<(String, Vec<u8>)>,
    compat_bundle_headers: Vec<(String, Vec<u8>)>,
    compat_bundle_payloads: Vec<(String, Vec<u8>)>,
    scope_by_value: Vec<(String, Vec<u8>)>,
    scope_by_ord: Vec<(Vec<u8>, Vec<u8>)>,
    doc_by_value: Vec<(String, Vec<u8>)>,
    doc_by_ord: Vec<(Vec<u8>, Vec<u8>)>,
    session_by_value: Vec<(String, Vec<u8>)>,
    session_by_ord: Vec<(Vec<u8>, Vec<u8>)>,
    ord_counter: Vec<(Vec<u8>, Vec<u8>)>,
    doc_manifest: Vec<(Vec<u8>, Vec<u8>)>,
    doc_segment: Vec<(Vec<u8>, Vec<u8>)>,
    session_archive: Vec<(Vec<u8>, Vec<u8>)>,
    scope_sidecar_meta: Vec<(Vec<u8>, Vec<u8>)>,
    scope_sidecar_fst: Vec<(Vec<u8>, Vec<u8>)>,
    scope_sidecar_postings: Vec<(Vec<u8>, Vec<u8>)>,
    idx_doc_latest: Vec<(Vec<u8>, Vec<u8>)>,
    idx_session_latest: Vec<(Vec<u8>, Vec<u8>)>,
    idx_session_docs: Vec<(Vec<u8>, Vec<u8>)>,
    idx_scope_docs: Vec<(Vec<u8>, Vec<u8>)>,
    idx_scope_dirty: Vec<(Vec<u8>, Vec<u8>)>,
    graph_kernel_checkpoint: Vec<(String, Vec<u8>)>,
    graph_kernel_journal: Vec<(String, Vec<u8>)>,
    graph_kernel_commit_index: Vec<(String, Vec<u8>)>,
    ann_source_leaf: Vec<(Vec<u8>, Vec<u8>)>,
    ann_source_document: Vec<(Vec<u8>, Vec<u8>)>,
    ann_source_node: Vec<(Vec<u8>, Vec<u8>)>,
    ann_head: Vec<(Vec<u8>, Vec<u8>)>,
    ann_manifest: Vec<(Vec<u8>, Vec<u8>)>,
    ann_vectors: Vec<(Vec<u8>, Vec<u8>)>,
    ann_levels: Vec<(Vec<u8>, Vec<u8>)>,
    ann_offsets: Vec<(Vec<u8>, Vec<u8>)>,
    ann_adjacency: Vec<(Vec<u8>, Vec<u8>)>,
    ann_id_by_ord: Vec<(Vec<u8>, Vec<u8>)>,
    ann_ord_by_id: Vec<(Vec<u8>, Vec<u8>)>,
    ann_payload: Vec<(Vec<u8>, Vec<u8>)>,
    ann_dirty: Vec<(Vec<u8>, Vec<u8>)>,
}

#[derive(Clone, Debug)]
pub struct ImportedNativeSnapshot {
    pub schema_version: String,
    pub created_at: i64,
    pub relation_count: usize,
    pub kernel_generation: u64,
    pub kernel_snapshot: Option<KernelGraphSnapshot>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
enum AnnPayload {
    Document {
        document_id: String,
        leaf_count: usize,
        evidence_refs: Vec<String>,
    },
    Leaf {
        span_id: String,
        document_id: String,
    },
    Node {
        node_id: String,
        node_kind: String,
        document_id: Option<String>,
        narrative_id: Option<String>,
        folder_id: Option<String>,
        evidence_refs: Vec<String>,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnnSourceLeafRecord {
    scope: ScopeKey,
    scope_key: String,
    span_id: String,
    document_id: String,
    values: Vec<f32>,
    updated_at: i64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnnSourceDocumentRecord {
    scope: ScopeKey,
    scope_key: String,
    document_id: String,
    values: Vec<f32>,
    leaf_count: usize,
    evidence_refs: Vec<String>,
    updated_at: i64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct AnnSourceNodeRecord {
    scope: ScopeKey,
    scope_key: String,
    node_id: String,
    node_kind: String,
    document_id: Option<String>,
    narrative_id: Option<String>,
    folder_id: Option<String>,
    values: Vec<f32>,
    evidence_refs: Vec<String>,
    updated_at: i64,
}

impl PhoenixLmdbStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        Self::open_with_tuning(path, LmdbTuning::from_env())
    }

    pub fn open_with_tuning(
        path: impl AsRef<Path>,
        tuning: LmdbTuning,
    ) -> Result<Self, StoreError> {
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path).map_err(|error| StoreError::Init(error.to_string()))?;

        let env = open_lmdb_env(&path, &tuning)?;

        let mut wtxn = env
            .write_txn()
            .map_err(|error| StoreError::Init(error.to_string()))?;

        let compat_relation_rows =
            open_str_bytes_db(&env, &mut wtxn, COMPAT_RELATION_ROWS_DB, false)?;
        let compat_bundle_headers =
            open_str_bytes_db(&env, &mut wtxn, COMPAT_BUNDLE_HEADERS_DB, false)?;
        let compat_bundle_payloads =
            open_str_bytes_db(&env, &mut wtxn, COMPAT_BUNDLE_PAYLOADS_DB, false)?;
        let scope_by_value = open_str_bytes_db(&env, &mut wtxn, SCOPE_BY_VALUE_DB, false)?;
        let scope_by_ord = open_raw_bytes_db(&env, &mut wtxn, SCOPE_BY_ORD_DB, false)?;
        let doc_by_value = open_str_bytes_db(&env, &mut wtxn, DOC_BY_VALUE_DB, false)?;
        let doc_by_ord = open_raw_bytes_db(&env, &mut wtxn, DOC_BY_ORD_DB, false)?;
        let session_by_value = open_str_bytes_db(&env, &mut wtxn, SESSION_BY_VALUE_DB, false)?;
        let session_by_ord = open_raw_bytes_db(&env, &mut wtxn, SESSION_BY_ORD_DB, false)?;
        let ord_counter = open_raw_bytes_db(&env, &mut wtxn, ORD_COUNTER_DB, false)?;
        let doc_manifest = open_raw_bytes_db(&env, &mut wtxn, DOC_MANIFEST_DB, false)?;
        let doc_segment = open_raw_bytes_db(&env, &mut wtxn, DOC_SEGMENT_DB, false)?;
        let session_archive = open_raw_bytes_db(&env, &mut wtxn, SESSION_ARCHIVE_DB, false)?;
        let scope_sidecar_meta =
            open_raw_bytes_db(&env, &mut wtxn, SCOPE_SIDECAR_META_DB, false)?;
        let scope_sidecar_fst = open_raw_bytes_db(&env, &mut wtxn, SCOPE_SIDECAR_FST_DB, false)?;
        let scope_sidecar_postings =
            open_raw_bytes_db(&env, &mut wtxn, SCOPE_SIDECAR_POSTINGS_DB, false)?;
        let idx_doc_latest = open_raw_bytes_db(&env, &mut wtxn, IDX_DOC_LATEST_DB, false)?;
        let idx_session_latest =
            open_raw_bytes_db(&env, &mut wtxn, IDX_SESSION_LATEST_DB, false)?;
        let idx_session_docs = open_raw_bytes_db(&env, &mut wtxn, IDX_SESSION_DOCS_DB, false)?;
        let idx_scope_docs = open_raw_bytes_db(&env, &mut wtxn, IDX_SCOPE_DOCS_DB, false)?;
        let idx_scope_dirty = open_raw_bytes_db(&env, &mut wtxn, IDX_SCOPE_DIRTY_DB, false)?;
        let graph_kernel_checkpoint =
            open_str_bytes_db(&env, &mut wtxn, GRAPH_KERNEL_CHECKPOINT_DB, false)?;
        let graph_kernel_journal =
            open_str_bytes_db(&env, &mut wtxn, GRAPH_KERNEL_JOURNAL_DB, false)?;
        let graph_kernel_commit_index =
            open_str_bytes_db(&env, &mut wtxn, GRAPH_KERNEL_COMMIT_INDEX_DB, false)?;
        let ann_source_leaf = open_raw_bytes_db(&env, &mut wtxn, ANN_SOURCE_LEAF_DB, false)?;
        let ann_source_document =
            open_raw_bytes_db(&env, &mut wtxn, ANN_SOURCE_DOCUMENT_DB, false)?;
        let ann_source_node = open_raw_bytes_db(&env, &mut wtxn, ANN_SOURCE_NODE_DB, false)?;
        let ann_head = open_raw_bytes_db(&env, &mut wtxn, ANN_HEAD_DB, false)?;
        let ann_manifest = open_raw_bytes_db(&env, &mut wtxn, ANN_MANIFEST_DB, false)?;
        let ann_vectors = open_raw_bytes_db(&env, &mut wtxn, ANN_VECTORS_DB, false)?;
        let ann_levels = open_raw_bytes_db(&env, &mut wtxn, ANN_LEVELS_DB, false)?;
        let ann_offsets = open_raw_bytes_db(&env, &mut wtxn, ANN_OFFSETS_DB, false)?;
        let ann_adjacency = open_raw_bytes_db(&env, &mut wtxn, ANN_ADJACENCY_DB, false)?;
        let ann_id_by_ord = open_raw_bytes_db(&env, &mut wtxn, ANN_ID_BY_ORD_DB, false)?;
        let ann_ord_by_id = open_raw_bytes_db(&env, &mut wtxn, ANN_ORD_BY_ID_DB, false)?;
        let ann_payload = open_raw_bytes_db(&env, &mut wtxn, ANN_PAYLOAD_DB, false)?;
        let ann_dirty = open_raw_bytes_db(&env, &mut wtxn, ANN_DIRTY_DB, false)?;

        wtxn.commit()
            .map_err(|error| StoreError::Init(error.to_string()))?;
        let initial_generation = {
            let checkpoint_generation = {
                let rtxn = env
                    .read_txn()
                    .map_err(|error| StoreError::Init(error.to_string()))?;
                let checkpoint = graph_kernel_checkpoint
                    .get(&rtxn, CHECKPOINT_KEY)
                    .map_err(|error| StoreError::Init(error.to_string()))?
                    .map(|bytes| decode_value::<KernelCheckpointData>(bytes))
                    .transpose()
                    .map_err(|error| StoreError::Init(error.to_string()))?;
                checkpoint
                    .map(|checkpoint| checkpoint.meta.generation)
                    .unwrap_or_default()
            };
            let journal_generation = {
                let rtxn = env
                    .read_txn()
                    .map_err(|error| StoreError::Init(error.to_string()))?;
                let iter = graph_kernel_journal
                    .rev_iter(&rtxn)
                    .map_err(|error| StoreError::Init(error.to_string()))?;
                let mut generation = 0;
                for item in iter {
                    let (_, bytes) =
                        item.map_err(|error| StoreError::Init(error.to_string()))?;
                    let entry: KernelJournalEntry =
                        decode_value(bytes).map_err(|error| StoreError::Init(error.to_string()))?;
                    generation = entry.generation;
                    break;
                }
                generation
            };
            checkpoint_generation.max(journal_generation)
        };

        Ok(Self {
            env,
            path,
            tuning,
            compat_relation_rows,
            compat_bundle_headers,
            compat_bundle_payloads,
            scope_by_value,
            scope_by_ord,
            doc_by_value,
            doc_by_ord,
            session_by_value,
            session_by_ord,
            ord_counter,
            doc_manifest,
            doc_segment,
            session_archive,
            scope_sidecar_meta,
            scope_sidecar_fst,
            scope_sidecar_postings,
            idx_doc_latest,
            idx_session_latest,
            idx_session_docs,
            idx_scope_docs,
            idx_scope_dirty,
            graph_kernel_checkpoint,
            graph_kernel_journal,
            graph_kernel_commit_index,
            ann_source_leaf,
            ann_source_document,
            ann_source_node,
            ann_head,
            ann_manifest,
            ann_vectors,
            ann_levels,
            ann_offsets,
            ann_adjacency,
            ann_id_by_ord,
            ann_ord_by_id,
            ann_payload,
            ann_dirty,
            live_kernel_snapshot: RwLock::new(None),
            live_kernel_generation: AtomicU64::new(initial_generation),
            commit_count: AtomicU64::new(0),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn schema_version(&self) -> &'static str {
        PHOENIX_NATIVE_SCHEMA_VERSION
    }

    fn commit_write_txn(&self, wtxn: RwTxn) -> Result<(), StoreError> {
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let commits = self.commit_count.fetch_add(1, Ordering::Relaxed) + 1;
        if let IngestDurabilityMode::NoSyncPeriodic { sync_every_commits } = self.tuning.durability {
            if sync_every_commits > 0 && commits % sync_every_commits == 0 {
                self.env
                    .force_sync()
                    .map_err(|error| StoreError::Query(error.to_string()))?;
            }
        }
        Ok(())
    }

    fn invalidate_live_kernel_snapshot(&self) {
        if let Ok(mut cached) = self.live_kernel_snapshot.write() {
            *cached = None;
        }
    }

    fn cache_live_kernel_snapshot(&self, generation: u64, snapshot: KernelGraphSnapshot) {
        self.live_kernel_generation.store(generation, Ordering::Release);
        if let Ok(mut cached) = self.live_kernel_snapshot.write() {
            *cached = Some((generation, snapshot));
        }
    }

    fn live_kernel_snapshot_cached(
        &self,
        generation: u64,
    ) -> Option<KernelGraphSnapshot> {
        self.live_kernel_snapshot
            .read()
            .ok()
            .and_then(|cached| cached.as_ref().cloned())
            .and_then(|(cached_generation, snapshot)| {
                (cached_generation == generation).then_some(snapshot)
            })
    }

    pub fn commit_native_session(
        &self,
        session_row: &Value,
        commit_row: &Value,
        generation: u64,
        source_revision: &str,
        commit_id: &str,
        committed_at: i64,
    ) -> Result<(), StoreError> {
        let session_key = relation_row_storage_key("phoenix_sessions", session_row)?;
        let session_bytes = encode_value(session_row)?;
        let commit_key = relation_row_storage_key("phoenix_commits", commit_row)?;
        let commit_bytes = encode_value(commit_row)?;
        let journal_entry = KernelJournalEntry {
            generation,
            source_revision: source_revision.to_owned(),
            batch: None,
            commit_id: Some(commit_id.to_owned()),
            created_at: committed_at,
        };
        let journal_key = journal_entry_key(journal_entry.generation, journal_entry.created_at);
        let journal_bytes = encode_value(&journal_entry)?;

        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        self.write_bytes(&mut wtxn, self.compat_relation_rows, &session_key, &session_bytes)?;
        self.write_bytes(&mut wtxn, self.compat_relation_rows, &commit_key, &commit_bytes)?;
        self.write_bytes(
            &mut wtxn,
            self.graph_kernel_journal,
            &journal_key,
            &journal_bytes,
        )?;
        self.write_bytes(
            &mut wtxn,
            self.graph_kernel_commit_index,
            commit_id,
            &generation.to_be_bytes(),
        )?;
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn latest_kernel_journal_generation(&self) -> Result<Option<u64>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let iter = self
            .graph_kernel_journal
            .rev_iter(&rtxn)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        for item in iter {
            let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            let entry: KernelJournalEntry = decode_value(bytes)?;
            return Ok(Some(entry.generation));
        }
        Ok(None)
    }

    pub fn export_native_snapshot(
        &self,
        schema_version: &str,
        partition: SnapshotPartition,
    ) -> Result<Vec<u8>, StoreError> {
        self.export_native_snapshot_with_kernel_snapshot(schema_version, partition, None, None, None)
    }

    pub fn export_native_snapshot_with_kernel_snapshot(
        &self,
        schema_version: &str,
        partition: SnapshotPartition,
        kernel_snapshot: Option<&KernelGraphSnapshot>,
        source_revision: Option<&str>,
        generation: Option<u64>,
    ) -> Result<Vec<u8>, StoreError> {
        let include_authoritative = matches!(
            partition,
            SnapshotPartition::All | SnapshotPartition::Content
        );
        let include_derived = matches!(partition, SnapshotPartition::Derived);
        let injected_checkpoint = if include_authoritative {
            kernel_snapshot
                .map(|snapshot| {
                    let generation = generation.unwrap_or(self.kernel_current_generation()?);
                    Ok::<_, StoreError>(KernelCheckpointData {
                        meta: KernelCheckpointMeta {
                            checkpoint_id: format!("kernel-checkpoint-{generation}"),
                            generation,
                            source_revision: source_revision
                                .map(str::to_owned)
                                .unwrap_or_else(|| format!("graph:{generation}")),
                            created_at: now_ms(),
                        },
                        snapshot: snapshot.clone(),
                    })
                })
                .transpose()?
        } else {
            None
        };
        let envelope = NativeSnapshotEnvelope {
            schema_version: schema_version.to_owned(),
            partition: match partition {
                SnapshotPartition::All => "all",
                SnapshotPartition::Content => "content",
                SnapshotPartition::Derived => "derived",
            }
            .to_owned(),
            created_at: now_ms(),
            relation_rows: if include_authoritative {
                dump_relation_db(
                    &self.env,
                    self.compat_relation_rows,
                    &native_relation_snapshot_names(partition),
                )?
            } else {
                Vec::new()
            },
            compat_bundle_headers: Vec::new(),
            compat_bundle_payloads: Vec::new(),
            scope_by_value: dump_snapshot_section(
                "scope_by_value",
                include_authoritative,
                || dump_db(&self.env, self.scope_by_value),
            )?,
            scope_by_ord: dump_snapshot_section("scope_by_ord", include_authoritative, || {
                dump_raw_db(&self.env, self.scope_by_ord)
            })?,
            doc_by_value: dump_snapshot_section("doc_by_value", include_authoritative, || {
                dump_db(&self.env, self.doc_by_value)
            })?,
            doc_by_ord: dump_snapshot_section("doc_by_ord", include_authoritative, || {
                dump_raw_db(&self.env, self.doc_by_ord)
            })?,
            session_by_value: dump_snapshot_section(
                "session_by_value",
                include_authoritative,
                || dump_db(&self.env, self.session_by_value),
            )?,
            session_by_ord: dump_snapshot_section("session_by_ord", include_authoritative, || {
                dump_raw_db(&self.env, self.session_by_ord)
            })?,
            ord_counter: dump_snapshot_section("ord_counter", include_authoritative, || {
                dump_raw_db(&self.env, self.ord_counter)
            })?,
            doc_manifest: dump_snapshot_section("doc_manifest", include_authoritative, || {
                dump_raw_db(&self.env, self.doc_manifest)
            })?,
            doc_segment: dump_snapshot_section("doc_segment", include_authoritative, || {
                dump_raw_db(&self.env, self.doc_segment)
            })?,
            session_archive: dump_snapshot_section(
                "session_archive",
                include_authoritative,
                || dump_raw_db(&self.env, self.session_archive),
            )?,
            scope_sidecar_meta: dump_snapshot_section(
                "scope_sidecar_meta",
                include_derived,
                || dump_raw_db(&self.env, self.scope_sidecar_meta),
            )?,
            scope_sidecar_fst: dump_snapshot_section("scope_sidecar_fst", include_derived, || {
                dump_raw_db(&self.env, self.scope_sidecar_fst)
            })?,
            scope_sidecar_postings: dump_snapshot_section(
                "scope_sidecar_postings",
                include_derived,
                || dump_raw_db(&self.env, self.scope_sidecar_postings),
            )?,
            idx_doc_latest: dump_snapshot_section("idx_doc_latest", include_authoritative, || {
                dump_raw_db(&self.env, self.idx_doc_latest)
            })?,
            idx_session_latest: dump_snapshot_section(
                "idx_session_latest",
                include_authoritative,
                || dump_raw_db(&self.env, self.idx_session_latest),
            )?,
            idx_session_docs: dump_snapshot_section("idx_session_docs", include_authoritative, || {
                dump_raw_db(&self.env, self.idx_session_docs)
            })?,
            idx_scope_docs: dump_snapshot_section("idx_scope_docs", include_authoritative, || {
                dump_raw_db(&self.env, self.idx_scope_docs)
            })?,
            idx_scope_dirty: dump_snapshot_section("idx_scope_dirty", include_authoritative, || {
                dump_raw_db(&self.env, self.idx_scope_dirty)
            })?,
            graph_kernel_checkpoint: dump_snapshot_section(
                "graph_kernel_checkpoint",
                include_authoritative,
                || {
                    if let Some(checkpoint) = injected_checkpoint.as_ref() {
                        return Ok(vec![(CHECKPOINT_KEY.to_owned(), encode_value(checkpoint)?)]);
                    }
                    dump_db(&self.env, self.graph_kernel_checkpoint)
                },
            )?,
            graph_kernel_journal: dump_snapshot_section(
                "graph_kernel_journal",
                include_authoritative,
                || {
                    if injected_checkpoint.is_some() {
                        return Ok(Vec::new());
                    }
                    dump_db(&self.env, self.graph_kernel_journal)
                },
            )?,
            graph_kernel_commit_index: dump_snapshot_section(
                "graph_kernel_commit_index",
                include_authoritative,
                || {
                    if injected_checkpoint.is_some() {
                        return Ok(Vec::new());
                    }
                    dump_db(&self.env, self.graph_kernel_commit_index)
                },
            )?,
            ann_source_leaf: dump_snapshot_section("ann_source_leaf", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_source_leaf)
            })?,
            ann_source_document: dump_snapshot_section(
                "ann_source_document",
                include_authoritative,
                || dump_raw_db(&self.env, self.ann_source_document),
            )?,
            ann_source_node: dump_snapshot_section("ann_source_node", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_source_node)
            })?,
            ann_head: dump_snapshot_section("ann_head", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_head)
            })?,
            ann_manifest: dump_snapshot_section("ann_manifest", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_manifest)
            })?,
            ann_vectors: dump_snapshot_section("ann_vectors", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_vectors)
            })?,
            ann_levels: dump_snapshot_section("ann_levels", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_levels)
            })?,
            ann_offsets: dump_snapshot_section("ann_offsets", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_offsets)
            })?,
            ann_adjacency: dump_snapshot_section("ann_adjacency", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_adjacency)
            })?,
            ann_id_by_ord: dump_snapshot_section("ann_id_by_ord", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_id_by_ord)
            })?,
            ann_ord_by_id: dump_snapshot_section("ann_ord_by_id", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_ord_by_id)
            })?,
            ann_payload: dump_snapshot_section("ann_payload", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_payload)
            })?,
            ann_dirty: dump_snapshot_section("ann_dirty", include_authoritative, || {
                dump_raw_db(&self.env, self.ann_dirty)
            })?,
        };
        let mut bytes = Vec::from(&NATIVE_SNAPSHOT_MAGIC[..]);
        bytes.extend_from_slice(&encode_value(&envelope)?);
        Ok(bytes)
    }

    pub fn is_native_snapshot(bytes: &[u8]) -> bool {
        bytes.starts_with(NATIVE_SNAPSHOT_MAGIC)
    }

    pub fn import_native_snapshot(&self, bytes: &[u8]) -> Result<ImportedNativeSnapshot, StoreError> {
        if !Self::is_native_snapshot(bytes) {
            return Err(StoreError::Snapshot("not a native LMDB snapshot".to_owned()));
        }
        let envelope: NativeSnapshotEnvelope = decode_value(&bytes[NATIVE_SNAPSHOT_MAGIC.len()..])?;
        let authoritative_partition = envelope.partition == "all" || envelope.partition == "content";
        let derived_partition = envelope.partition == "derived";
        let imported_checkpoint = envelope
            .graph_kernel_checkpoint
            .iter()
            .find_map(|(key, value)| {
                (key == CHECKPOINT_KEY)
                    .then(|| decode_value::<KernelCheckpointData>(value))
            })
            .transpose()?;
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Snapshot(error.to_string()))?;
        if authoritative_partition {
            restore_snapshot_section("compat", || {
                restore_db(&mut wtxn, self.compat_bundle_headers, &[])?;
                restore_db(&mut wtxn, self.compat_bundle_payloads, &[])?;
                restore_db(&mut wtxn, self.compat_relation_rows, &envelope.relation_rows)
            })?;
            restore_snapshot_section("ids", || {
                restore_db(&mut wtxn, self.scope_by_value, &envelope.scope_by_value)?;
                restore_raw_db(&mut wtxn, self.scope_by_ord, &envelope.scope_by_ord)?;
                restore_db(&mut wtxn, self.doc_by_value, &envelope.doc_by_value)?;
                restore_raw_db(&mut wtxn, self.doc_by_ord, &envelope.doc_by_ord)?;
                restore_db(&mut wtxn, self.session_by_value, &envelope.session_by_value)?;
                restore_raw_db(&mut wtxn, self.session_by_ord, &envelope.session_by_ord)?;
                restore_raw_db(&mut wtxn, self.ord_counter, &envelope.ord_counter)
            })?;
            restore_snapshot_section("documents", || {
                restore_raw_db(&mut wtxn, self.doc_manifest, &envelope.doc_manifest)?;
                restore_raw_db(&mut wtxn, self.doc_segment, &envelope.doc_segment)
            })?;
            restore_snapshot_section("session_archive", || {
                restore_raw_db(&mut wtxn, self.session_archive, &envelope.session_archive)
            })?;
            restore_snapshot_section("indexes", || {
                restore_raw_db(&mut wtxn, self.idx_doc_latest, &envelope.idx_doc_latest)?;
                restore_raw_db(&mut wtxn, self.idx_session_latest, &envelope.idx_session_latest)?;
                restore_raw_db(&mut wtxn, self.idx_session_docs, &envelope.idx_session_docs)?;
                restore_raw_db(&mut wtxn, self.idx_scope_docs, &envelope.idx_scope_docs)?;
                restore_raw_db(&mut wtxn, self.idx_scope_dirty, &envelope.idx_scope_dirty)
            })?;
            restore_snapshot_section("ann", || {
                restore_raw_db(&mut wtxn, self.ann_source_leaf, &envelope.ann_source_leaf)?;
                restore_raw_db(
                    &mut wtxn,
                    self.ann_source_document,
                    &envelope.ann_source_document,
                )?;
                restore_raw_db(&mut wtxn, self.ann_source_node, &envelope.ann_source_node)?;
                restore_raw_db(&mut wtxn, self.ann_head, &envelope.ann_head)?;
                restore_raw_db(&mut wtxn, self.ann_manifest, &envelope.ann_manifest)?;
                restore_raw_db(&mut wtxn, self.ann_vectors, &envelope.ann_vectors)?;
                restore_raw_db(&mut wtxn, self.ann_levels, &envelope.ann_levels)?;
                restore_raw_db(&mut wtxn, self.ann_offsets, &envelope.ann_offsets)?;
                restore_raw_db(&mut wtxn, self.ann_adjacency, &envelope.ann_adjacency)?;
                restore_raw_db(&mut wtxn, self.ann_id_by_ord, &envelope.ann_id_by_ord)?;
                restore_raw_db(&mut wtxn, self.ann_ord_by_id, &envelope.ann_ord_by_id)?;
                restore_raw_db(&mut wtxn, self.ann_payload, &envelope.ann_payload)?;
                restore_raw_db(&mut wtxn, self.ann_dirty, &envelope.ann_dirty)
            })?;
            restore_snapshot_section("graph_kernel", || {
                restore_db(
                    &mut wtxn,
                    self.graph_kernel_checkpoint,
                    &envelope.graph_kernel_checkpoint,
                )?;
                restore_db(&mut wtxn, self.graph_kernel_journal, &envelope.graph_kernel_journal)?;
                restore_db(
                    &mut wtxn,
                    self.graph_kernel_commit_index,
                    &envelope.graph_kernel_commit_index,
                )
            })?;
            restore_snapshot_section("sidecar_cache", || {
                restore_raw_db(&mut wtxn, self.scope_sidecar_meta, &[])?;
                restore_raw_db(&mut wtxn, self.scope_sidecar_fst, &[])?;
                restore_raw_db(&mut wtxn, self.scope_sidecar_postings, &[])
            })?;
            restore_snapshot_section("dirty_scope_schedule", || {
                refresh_dirty_scope_records_in_txn(self, &mut wtxn, now_ms())
            })?;
        } else if derived_partition {
            restore_snapshot_section("sidecar_cache", || {
                restore_raw_db(&mut wtxn, self.scope_sidecar_meta, &envelope.scope_sidecar_meta)?;
                restore_raw_db(&mut wtxn, self.scope_sidecar_fst, &envelope.scope_sidecar_fst)?;
                restore_raw_db(
                    &mut wtxn,
                    self.scope_sidecar_postings,
                    &envelope.scope_sidecar_postings,
                )
            })?;
        } else {
            return Err(StoreError::Snapshot(format!(
                "unsupported native snapshot partition {}",
                envelope.partition
            )));
        }
        wtxn.commit()
            .map_err(|error| StoreError::Snapshot(error.to_string()))?;
        let kernel_generation = self.kernel_current_generation().unwrap_or_default();
        let mut kernel_snapshot = None;
        if let Some(checkpoint) = imported_checkpoint {
            if checkpoint.meta.generation == kernel_generation {
                self.cache_live_kernel_snapshot(kernel_generation, checkpoint.snapshot.clone());
                kernel_snapshot = Some(checkpoint.snapshot);
            } else {
                self.invalidate_live_kernel_snapshot();
            }
        } else {
            self.invalidate_live_kernel_snapshot();
        }
        self.live_kernel_generation
            .store(kernel_generation, Ordering::Release);
        Ok(ImportedNativeSnapshot {
            schema_version: envelope.schema_version,
            created_at: envelope.created_at,
            relation_count: envelope.relation_rows.len(),
            kernel_generation,
            kernel_snapshot,
        })
    }

    fn write_bytes(
        &self,
        txn: &mut RwTxn,
        db: Database<Str, Bytes>,
        key: &str,
        bytes: &[u8],
    ) -> Result<(), StoreError> {
        db.put_reserved(txn, key, bytes.len(), |reserved| {
            reserved.write_all(bytes)
        })
        .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn write_raw_bytes(
        &self,
        txn: &mut RwTxn,
        db: Database<Bytes, Bytes>,
        key: &[u8],
        bytes: &[u8],
    ) -> Result<(), StoreError> {
        db.put_reserved(txn, key, bytes.len(), |reserved| {
            reserved.write_all(bytes)
        })
        .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn apply_write_batch(
        &self,
        txn: &mut RwTxn,
        mut batch: LmdbWriteBatch,
    ) -> Result<(), StoreError> {
        batch.puts.sort_by(|left, right| {
            left.db.cmp(&right.db).then_with(|| left.key.cmp(&right.key))
        });
        for put in batch.puts {
            self.apply_put(txn, put)?;
        }
        Ok(())
    }

    fn apply_put(&self, txn: &mut RwTxn, put: LmdbPut) -> Result<(), StoreError> {
        match (put.db, put.key) {
            (LmdbDbId::CompatBundleHeader, LmdbKey::Str(key)) => self.write_str_bytes_with_flags(
                txn,
                self.compat_bundle_headers,
                &key,
                &put.value,
                put.flags,
            ),
            (LmdbDbId::DocManifest, LmdbKey::Bytes(key)) => self.write_raw_bytes_with_flags(
                txn,
                self.doc_manifest,
                &key,
                &put.value,
                put.flags,
            ),
            (LmdbDbId::DocSegment, LmdbKey::Bytes(key)) => self.write_raw_bytes_with_flags(
                txn,
                self.doc_segment,
                &key,
                &put.value,
                put.flags,
            ),
            (LmdbDbId::IdxDocLatest, LmdbKey::Bytes(key)) => self.write_raw_bytes_with_flags(
                txn,
                self.idx_doc_latest,
                &key,
                &put.value,
                put.flags,
            ),
            (LmdbDbId::IdxScopeDocs, LmdbKey::Bytes(key)) => self.write_raw_bytes_with_flags(
                txn,
                self.idx_scope_docs,
                &key,
                &put.value,
                put.flags,
            ),
            (LmdbDbId::IdxScopeDirty, LmdbKey::Bytes(key)) => self.write_raw_bytes_with_flags(
                txn,
                self.idx_scope_dirty,
                &key,
                &put.value,
                put.flags,
            ),
            (LmdbDbId::SessionArchive, LmdbKey::Bytes(key)) => self.write_raw_bytes_with_flags(
                txn,
                self.session_archive,
                &key,
                &put.value,
                put.flags,
            ),
            (LmdbDbId::IdxSessionLatest, LmdbKey::Bytes(key)) => self.write_raw_bytes_with_flags(
                txn,
                self.idx_session_latest,
                &key,
                &put.value,
                put.flags,
            ),
            (LmdbDbId::IdxSessionDocs, LmdbKey::Bytes(key)) => self.write_raw_bytes_with_flags(
                txn,
                self.idx_session_docs,
                &key,
                &put.value,
                put.flags,
            ),
            (db, key) => Err(StoreError::Query(format!(
                "invalid LMDB batch key for {:?}: {:?}",
                db, key
            ))),
        }
    }

    fn write_str_bytes_with_flags(
        &self,
        txn: &mut RwTxn,
        db: Database<Str, Bytes>,
        key: &str,
        bytes: &[u8],
        flags: PutFlags,
    ) -> Result<(), StoreError> {
        if flags.is_empty() {
            return self.write_bytes(txn, db, key, bytes);
        }
        db.put_with_flags(txn, flags, key, bytes)
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn write_raw_bytes_with_flags(
        &self,
        txn: &mut RwTxn,
        db: Database<Bytes, Bytes>,
        key: &[u8],
        bytes: &[u8],
        flags: PutFlags,
    ) -> Result<(), StoreError> {
        if flags.is_empty() {
            return self.write_raw_bytes(txn, db, key, bytes);
        }
        db.put_with_flags(txn, flags, key, bytes)
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn read_struct<T: DeserializeOwned>(
        &self,
        db: Database<Str, Bytes>,
        key: &str,
    ) -> Result<Option<T>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let value = db
            .get(&rtxn, key)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        value.map(decode_value).transpose()
    }

    fn iter_structs_with_prefix<T: DeserializeOwned>(
        &self,
        db: Database<Str, Bytes>,
        prefix: Option<&str>,
    ) -> Result<Vec<T>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut values = Vec::new();
        match prefix {
            Some(prefix) => {
                let iter = db
                    .prefix_iter(&rtxn, prefix)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                for item in iter {
                    let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
                    values.push(decode_value(bytes)?);
                }
            }
            None => {
                let iter = db
                    .iter(&rtxn)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                for item in iter {
                    let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
                    values.push(decode_value(bytes)?);
                }
            }
        }
        Ok(values)
    }

    fn read_raw_struct<T: DeserializeOwned>(
        &self,
        db: Database<Bytes, Bytes>,
        key: &[u8],
    ) -> Result<Option<T>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        self.read_raw_struct_in_txn(&rtxn, db, key)
    }

    fn read_raw_struct_in_txn<T: DeserializeOwned>(
        &self,
        rtxn: &heed3::RoTxn,
        db: Database<Bytes, Bytes>,
        key: &[u8],
    ) -> Result<Option<T>, StoreError> {
        let value = db
            .get(rtxn, key)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        value.map(decode_value).transpose()
    }

    fn iter_raw_structs_with_prefix<T: DeserializeOwned>(
        &self,
        db: Database<Bytes, Bytes>,
        prefix: Option<&[u8]>,
    ) -> Result<Vec<T>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut values = Vec::new();
        match prefix {
            Some(prefix) => {
                let iter = db
                    .prefix_iter(&rtxn, prefix)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                for item in iter {
                    let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
                    values.push(decode_value(bytes)?);
                }
            }
            None => {
                let iter = db
                    .iter(&rtxn)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                for item in iter {
                    let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
                    values.push(decode_value(bytes)?);
                }
            }
        }
        Ok(values)
    }
}

impl PhoenixLmdbStore {
    fn lookup_scope_ord(&self, scope_key: &str) -> Result<Option<ScopeOrd>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        self.lookup_scope_ord_in_txn(&rtxn, scope_key)
    }

    fn lookup_scope_ord_in_txn(
        &self,
        rtxn: &heed3::RoTxn,
        scope_key: &str,
    ) -> Result<Option<ScopeOrd>, StoreError> {
        let value = self
            .scope_by_value
            .get(rtxn, scope_key)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(value.and_then(decode_ord).map(ScopeOrd))
    }

    fn lookup_session_ord(&self, session_id: &SessionId) -> Result<Option<SessionOrd>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let value = self
            .session_by_value
            .get(&rtxn, &session_id.0)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(value.and_then(decode_ord).map(SessionOrd))
    }

    fn lookup_latest_session_revision(&self, session_ord: SessionOrd) -> Result<Option<u64>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let value = self
            .idx_session_latest
            .get(&rtxn, &session_latest_key(session_ord))
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(value.and_then(decode_u64))
    }

    fn list_all_scopes(&self) -> Result<Vec<ScopeKey>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let iter = self
            .scope_by_value
            .iter(&rtxn)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut scopes = Vec::new();
        for item in iter {
            let (scope_key, _) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            scopes.push(parse_scope_key(scope_key));
        }
        scopes.sort_by_key(scope_storage_key);
        scopes.dedup_by(|left, right| scope_storage_key(left) == scope_storage_key(right));
        Ok(scopes)
    }

    fn load_latest_document_manifests(
        &self,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<DocumentManifest>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        self.load_latest_document_manifests_in_txn(&rtxn, scope)
    }

    fn load_latest_document_manifests_in_txn(
        &self,
        rtxn: &heed3::RoTxn,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<DocumentManifest>, StoreError> {
        let mut revisions = Vec::<(ScopeOrd, DocumentOrd, u64)>::new();
        match scope {
            Some(scope) => {
                let Some(scope_ord) =
                    self.lookup_scope_ord_in_txn(rtxn, &scope_storage_key(scope))?
                else {
                    return Ok(Vec::new());
                };
                let iter = self
                    .idx_scope_docs
                    .prefix_iter(rtxn, &scope_membership_prefix(scope_ord))
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                for item in iter {
                    let (key, _) = item.map_err(|error| StoreError::Query(error.to_string()))?;
                    let (_, document_ord) = decode_scope_doc_membership_key(key)?;
                    if let Some(revision) =
                        self.latest_document_revision_with_txn(rtxn, scope_ord, document_ord)?
                    {
                        revisions.push((scope_ord, document_ord, revision));
                    }
                }
            }
            None => {
                let iter = self
                    .idx_doc_latest
                    .iter(rtxn)
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                for item in iter {
                    let (key, value) = item.map_err(|error| StoreError::Query(error.to_string()))?;
                    let (scope_ord, document_ord) = decode_document_latest_key(key)?;
                    let revision = decode_u64(value).unwrap_or_default();
                    revisions.push((scope_ord, document_ord, revision));
                }
            }
        }
        revisions.sort_by(|left, right| {
            left.0
                .0
                .cmp(&right.0.0)
                .then_with(|| left.1.0.cmp(&right.1.0))
                .then_with(|| left.2.cmp(&right.2))
        });
        let mut manifests = Vec::with_capacity(revisions.len());
        for (scope_ord, document_ord, revision) in revisions {
            if let Some(manifest) = self.read_raw_struct_in_txn(
                rtxn,
                self.doc_manifest,
                &document_revision_key(scope_ord, document_ord, revision),
            )? {
                manifests.push(manifest);
            }
        }
        manifests.sort_by(|left: &DocumentManifest, right: &DocumentManifest| {
            left.document_id.cmp(&right.document_id)
        });
        Ok(manifests)
    }

    fn load_latest_document_manifests_for_ords_in_txn(
        &self,
        rtxn: &heed3::RoTxn,
        scope_ord: ScopeOrd,
        document_ords: &[DocumentOrd],
    ) -> Result<Vec<DocumentManifest>, StoreError> {
        let mut manifests = Vec::new();
        for document_ord in document_ords {
            if let Some(revision) =
                self.latest_document_revision_with_txn(rtxn, scope_ord, *document_ord)?
            {
                if let Some(manifest) = self.read_raw_struct_in_txn(
                    rtxn,
                    self.doc_manifest,
                    &document_revision_key(scope_ord, *document_ord, revision),
                )? {
                    manifests.push(manifest);
                }
            }
        }
        manifests.sort_by(|left: &DocumentManifest, right: &DocumentManifest| {
            left.document_id.cmp(&right.document_id)
        });
        Ok(manifests)
    }

    fn latest_document_revision_with_txn(
        &self,
        rtxn: &heed3::RoTxn,
        scope_ord: ScopeOrd,
        document_ord: DocumentOrd,
    ) -> Result<Option<u64>, StoreError> {
        let value = self
            .idx_doc_latest
            .get(rtxn, &document_latest_key(scope_ord, document_ord))
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(value.and_then(decode_u64))
    }

    fn count_relation_rows(&self, relation: &str) -> Result<usize, StoreError> {
        ensure_native_relation_supported(relation)?;
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let iter = self
            .compat_relation_rows
            .prefix_iter(&rtxn, &relation_prefix(relation))
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut count = 0usize;
        for item in iter {
            let _ = item.map_err(|error| StoreError::Query(error.to_string()))?;
            count += 1;
        }
        Ok(count)
    }

    fn load_document_archive_from_manifest(
        &self,
        manifest: &DocumentManifest,
    ) -> Result<DocumentArchive, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut archive = DocumentArchive {
            manifest: manifest.clone(),
            ..Default::default()
        };
        let prefix = document_segment_prefix(
            manifest.scope_ord,
            manifest.document_ord,
            manifest.revision,
        );
        let iter = self
            .doc_segment
            .prefix_iter(&rtxn, &prefix)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut lexical = None::<LexicalPostingsSegment>;
        for item in iter {
            let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            let segment: PreparedDocumentSegment = decode_value(bytes)?;
            match segment.header.kind() {
                DocumentSegmentKind::StringArena => {
                    archive.tokens = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::SentenceTable => {
                    archive.sentences = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::MentionTable => {
                    archive.mentions = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::ResolverLinkTable => {
                    archive.resolver_links = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::ResolvedMentionTable => {
                    archive.resolved_mentions = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::AliasConfirmationTable => {
                    archive.alias_confirmations = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::CorefClusterTable => {
                    archive.coref_clusters = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::CausalSubstrateTable => {
                    archive.causal_substrate = Some(decode_segment_payload(&segment.payload)?);
                }
                DocumentSegmentKind::ChunkTable => {
                    archive.chunks = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::EntityTable => {
                    archive.entities = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::RelationTable => {
                    archive.relations = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::EvidenceTable => {
                    archive.evidence_spans = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::LexicalPostings => {
                    lexical = Some(decode_segment_payload(&segment.payload)?);
                }
                DocumentSegmentKind::NarrativeHitTable => {
                    archive.relation_candidates = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::GraphMutation => {
                    archive.graph_batch = decode_segment_payload(&segment.payload)?;
                }
                DocumentSegmentKind::StructureRelations => {
                    archive.structure = Some(decode_segment_payload(&segment.payload)?);
                }
                _ => {}
            }
        }
        if let Some(lexical) = lexical {
            archive.indexed_spans = lexical.spans;
        }
        Ok(archive)
    }

    fn load_live_kernel_snapshot(&self) -> Result<KernelGraphSnapshot, StoreError> {
        let generation = self.kernel_current_generation()?;
        self.live_kernel_generation.store(generation, Ordering::Release);
        if let Some(snapshot) = self.live_kernel_snapshot_cached(generation) {
            return Ok(snapshot);
        }
        let kernel = DeterministicKernel::default();
        if let Some(checkpoint) = self.load_kernel_checkpoint()? {
            kernel
                .rebuild_from_kernel_batches(vec![
                    KernelMutationBatch {
                        layer: KernelGraphLayer::Asserted,
                        scope: KernelMutationScope::Full,
                        recorded_at: None,
                        vertices: checkpoint.snapshot.vertices.clone(),
                        edges: checkpoint.snapshot.asserted_edges.clone(),
                    },
                    KernelMutationBatch {
                        layer: KernelGraphLayer::Candidate,
                        scope: KernelMutationScope::Full,
                        recorded_at: None,
                        vertices: Vec::new(),
                        edges: checkpoint.snapshot.candidate_edges,
                    },
                ], None)
                .map_err(|error| StoreError::Query(error.to_string()))?;
            for entry in self.load_kernel_journal_after(checkpoint.meta.generation)? {
                if let Some(batch) = entry.batch {
                    kernel
                        .apply_batch(batch)
                        .map_err(|error| StoreError::Query(error.to_string()))?;
                }
            }
            let snapshot = kernel.snapshot().as_ref().clone();
            self.cache_live_kernel_snapshot(generation, snapshot.clone());
            return Ok(snapshot);
        }

        let batches = self
            .load_kernel_journal_after(0)?
            .into_iter()
            .filter_map(|entry| entry.batch)
            .collect::<Vec<_>>();
        kernel
            .rebuild_from_kernel_batches(batches, None)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let snapshot = kernel.snapshot().as_ref().clone();
        self.cache_live_kernel_snapshot(generation, snapshot.clone());
        Ok(snapshot)
    }

    fn load_lexical_postings_from_manifest_in_txn(
        &self,
        rtxn: &heed3::RoTxn,
        manifest: &DocumentManifest,
    ) -> Result<LexicalPostingsSegment, StoreError> {
        let mut lexical = LexicalPostingsSegment::default();
        let refs = manifest
            .segment_refs
            .iter()
            .filter(|segment_ref| segment_ref.kind == DocumentSegmentKind::LexicalPostings);
        for segment_ref in refs {
            let key = document_segment_key(
                manifest.scope_ord,
                manifest.document_ord,
                manifest.revision,
                DocumentSegmentKind::LexicalPostings,
                segment_ref.ordinal,
            );
            let bytes = self
                .doc_segment
                .get(rtxn, &key)
                .map_err(|error| StoreError::Query(error.to_string()))?
                .ok_or_else(|| {
                    StoreError::Query(format!(
                        "missing lexical postings segment for {}@{}",
                        manifest.document_id, manifest.revision
                    ))
                })?;
            let segment: PreparedDocumentSegment = decode_value(bytes)?;
            let mut decoded: LexicalPostingsSegment = decode_segment_payload(&segment.payload)?;
            lexical.spans.append(&mut decoded.spans);
            lexical.alias_entries.append(&mut decoded.alias_entries);
        }
        Ok(lexical)
    }

    fn materialize_scope_lexical(&self, scope: &ScopeKey) -> Result<ScopeLexSidecar, StoreError> {
        let Some(scope_ord) = self.lookup_scope_ord(&scope_storage_key(scope))? else {
            return Ok(ScopeLexSidecar {
                scope: scope.clone(),
                scope_key: scope_storage_key(scope),
                ..Default::default()
            });
        };
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let dirty = self.read_raw_struct_in_txn::<DirtyScopeRecord>(
            &rtxn,
            self.idx_scope_dirty,
            &scope_dirty_key(scope_ord),
        )?;
        let persisted = self.read_raw_struct_in_txn::<ScopeLexSidecar>(
            &rtxn,
            self.scope_sidecar_meta,
            &scope_sidecar_key(scope_ord),
        )?;
        if dirty.is_none() {
            if let Some(sidecar) = persisted {
                return Ok(sidecar);
            }
        }
        self.aggregate_scope_sidecar_with_txn(&rtxn, scope, scope_ord, persisted, dirty.as_ref())
    }

    fn aggregate_scope_sidecar_with_txn(
        &self,
        rtxn: &heed3::RoTxn,
        scope: &ScopeKey,
        scope_ord: ScopeOrd,
        persisted: Option<ScopeLexSidecar>,
        dirty: Option<&DirtyScopeRecord>,
    ) -> Result<ScopeLexSidecar, StoreError> {
        let had_persisted = persisted.is_some();
        let scope_key = scope_storage_key(scope);
        let mut sidecar = persisted.unwrap_or_else(|| ScopeLexSidecar {
            scope: scope.clone(),
            scope_key: scope_key.clone(),
            scope_ord: Some(scope_ord),
            ..Default::default()
        });
        sidecar.scope = scope.clone();
        sidecar.scope_key = scope_key;
        sidecar.scope_ord = Some(scope_ord);

        let mut generation = sidecar.generation;
        let dirty_manifests = if let Some(record) = dirty {
            if !had_persisted {
                let manifests = self.load_latest_document_manifests_in_txn(rtxn, Some(scope))?;
                generation = generation.max(record.updated_at as u64);
                manifests
            } else {
                let manifests = self.load_latest_document_manifests_for_ords_in_txn(
                    rtxn,
                    record.scope_ord,
                    &record.document_ords,
                )?;
                let dirty_document_ids = manifests
                    .iter()
                    .map(|manifest| manifest.document_id.clone())
                    .collect::<BTreeSet<_>>();
                if !dirty_document_ids.is_empty() {
                    sidecar.spans.retain(|span| {
                        span.document_id
                            .as_ref()
                            .map(|document_id| !dirty_document_ids.contains(&document_id.0))
                            .unwrap_or(true)
                    });
                    sidecar
                        .document_ids
                        .retain(|document_id| !dirty_document_ids.contains(document_id));
                    sidecar.alias_entries =
                        filter_alias_entries(&sidecar.alias_entries, &dirty_document_ids);
                }
                generation = generation.max(record.updated_at as u64);
                manifests
            }
        } else {
            self.load_latest_document_manifests_in_txn(rtxn, Some(scope))?
        };

        let mut alias_entries = alias_entries_to_map(&sidecar.alias_entries);
        for manifest in &dirty_manifests {
            let lexical = self.load_lexical_postings_from_manifest_in_txn(rtxn, manifest)?;
            generation = generation.max(manifest.revision);
            sidecar.document_ids.push(manifest.document_id.clone());
            sidecar.spans.extend(lexical.spans);
            merge_alias_entries(&mut alias_entries, &lexical.alias_entries);
        }

        sidecar.document_ids.sort();
        sidecar.document_ids.dedup();
        sidecar.spans.sort_by(|left, right| left.span_id.cmp(&right.span_id));
        sidecar.spans.dedup_by(|left, right| left.span_id == right.span_id);
        sidecar.alias_entries = alias_entries_from_map(alias_entries);
        sidecar.entity_count = entity_count_from_alias_entries(&sidecar.alias_entries);
        sidecar.generated_at = now_ms();
        sidecar.generation = generation;
        Ok(sidecar)
    }

    fn validate_semantic_vector(values: &[f32], label: &str) -> Result<(), StoreError> {
        if values.len() != SEMANTIC_VECTOR_DIM {
            return Err(StoreError::Query(format!(
                "semantic vector dimension mismatch for {label}: expected {}, got {}",
                SEMANTIC_VECTOR_DIM,
                values.len()
            )));
        }
        Ok(())
    }

    fn persist_ann_generation_in_txn(
        &self,
        wtxn: &mut RwTxn,
        index: &AnnIndexKey,
        vectors: &[Vec<f32>],
        stable_ids: &[String],
        payloads: &[AnnPayload],
        built_at: i64,
    ) -> Result<(), StoreError> {
        if vectors.is_empty() {
            return Ok(());
        }
        let mut builder = HyperbolicHnswBuilder::new(
            SEMANTIC_VECTOR_DIM,
            PoincareMetric { curvature: 1.0 },
            HnswBuildParams::default(),
        );
        for vector in vectors {
            builder.insert(vector.clone());
        }
        let packed = builder.into_packed();
        let prefix = ann_index_prefix(index);
        let generation = AnnGenerationId(
            self.ann_head
                .get(&*wtxn, &prefix)
                .map_err(|error| StoreError::Query(error.to_string()))?
                .and_then(decode_u64)
                .unwrap_or_default()
                + 1,
        );
        let generation_key = ann_generation_key(index, generation);
        let manifest = AnnManifest {
            index: index.clone(),
            generation_id: generation,
            built_at,
            dimension: packed.metadata.dim(),
            model_id: SEMANTIC_MODEL_ID.to_owned(),
            count: packed.metadata.num_vectors(),
            entry_point: packed.metadata.entry_point(),
            max_level: packed.metadata.max_level(),
            m: HnswBuildParams::default().m,
            m0: HnswBuildParams::default().m0,
            ef_construction: HnswBuildParams::default().ef_construction,
            level_mult: HnswBuildParams::default().level_mult,
            metric: "hyperbolic:poincare".to_owned(),
        };
        self.write_raw_bytes(
            wtxn,
            self.ann_manifest,
            &generation_key,
            &encode_value(&manifest)?,
        )?;
        self.write_raw_bytes(wtxn, self.ann_vectors, &generation_key, &packed.vectors)?;
        self.write_raw_bytes(wtxn, self.ann_levels, &generation_key, &packed.levels)?;
        self.write_raw_bytes(wtxn, self.ann_offsets, &generation_key, &packed.offsets)?;
        self.write_raw_bytes(wtxn, self.ann_adjacency, &generation_key, &packed.adjacency)?;
        for (ordinal, stable_id) in stable_ids.iter().enumerate() {
            let ord = ordinal as u32;
            self.write_raw_bytes(
                wtxn,
                self.ann_id_by_ord,
                &ann_generation_ord_key(index, generation, ord),
                stable_id.as_bytes(),
            )?;
            self.write_raw_bytes(
                wtxn,
                self.ann_ord_by_id,
                &ann_generation_id_key(index, generation, stable_id),
                &ord.to_be_bytes(),
            )?;
            self.write_raw_bytes(
                wtxn,
                self.ann_payload,
                &ann_generation_ord_key(index, generation, ord),
                &encode_value(&payloads[ordinal])?,
            )?;
        }
        self.write_raw_bytes(wtxn, self.ann_head, &prefix, &generation.0.to_be_bytes())?;
        self.ann_dirty
            .delete(wtxn, &ann_dirty_key(index))
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(())
    }

    fn rebuild_document_ann_index_in_txn(
        &self,
        wtxn: &mut RwTxn,
        index: &AnnIndexKey,
        built_at: i64,
    ) -> Result<(), StoreError> {
        let prefix = ann_index_prefix(index);
        let iter = self
            .ann_source_document
            .prefix_iter(&*wtxn, &prefix)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut records = Vec::new();
        for item in iter {
            let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            records.push(decode_value::<AnnSourceDocumentRecord>(bytes)?);
        }
        records.sort_by(|left, right| left.document_id.cmp(&right.document_id));
        let stable_ids = records.iter().map(|record| record.document_id.clone()).collect::<Vec<_>>();
        let vectors = records.iter().map(|record| record.values.clone()).collect::<Vec<_>>();
        let payloads = records
            .iter()
            .map(|record| AnnPayload::Document {
                document_id: record.document_id.clone(),
                leaf_count: record.leaf_count,
                evidence_refs: record.evidence_refs.clone(),
            })
            .collect::<Vec<_>>();
        self.persist_ann_generation_in_txn(wtxn, index, &vectors, &stable_ids, &payloads, built_at)
    }

    fn rebuild_leaf_ann_index_in_txn(
        &self,
        wtxn: &mut RwTxn,
        index: &AnnIndexKey,
        built_at: i64,
    ) -> Result<(), StoreError> {
        let prefix = ann_index_prefix(index);
        let iter = self
            .ann_source_leaf
            .prefix_iter(&*wtxn, &prefix)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut records = Vec::new();
        for item in iter {
            let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            records.push(decode_value::<AnnSourceLeafRecord>(bytes)?);
        }
        records.sort_by(|left, right| left.span_id.cmp(&right.span_id));
        let stable_ids = records.iter().map(|record| record.span_id.clone()).collect::<Vec<_>>();
        let vectors = records.iter().map(|record| record.values.clone()).collect::<Vec<_>>();
        let payloads = records
            .iter()
            .map(|record| AnnPayload::Leaf {
                span_id: record.span_id.clone(),
                document_id: record.document_id.clone(),
            })
            .collect::<Vec<_>>();
        self.persist_ann_generation_in_txn(wtxn, index, &vectors, &stable_ids, &payloads, built_at)
    }

    fn rebuild_node_ann_index_in_txn(
        &self,
        wtxn: &mut RwTxn,
        index: &AnnIndexKey,
        built_at: i64,
    ) -> Result<(), StoreError> {
        let prefix = ann_index_prefix(index);
        let iter = self
            .ann_source_node
            .prefix_iter(&*wtxn, &prefix)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut records = Vec::new();
        for item in iter {
            let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            records.push(decode_value::<AnnSourceNodeRecord>(bytes)?);
        }
        records.sort_by(|left, right| left.node_id.cmp(&right.node_id));
        let stable_ids = records.iter().map(|record| record.node_id.clone()).collect::<Vec<_>>();
        let vectors = records.iter().map(|record| record.values.clone()).collect::<Vec<_>>();
        let payloads = records
            .iter()
            .map(|record| AnnPayload::Node {
                node_id: record.node_id.clone(),
                node_kind: record.node_kind.clone(),
                document_id: record.document_id.clone(),
                narrative_id: record.narrative_id.clone(),
                folder_id: record.folder_id.clone(),
                evidence_refs: record.evidence_refs.clone(),
            })
            .collect::<Vec<_>>();
        self.persist_ann_generation_in_txn(wtxn, index, &vectors, &stable_ids, &payloads, built_at)
    }

    fn load_ann_query_state(
        &self,
        scope: &ScopeKey,
        family: AnnIndexFamily,
        kind: Option<&str>,
    ) -> Result<Option<(AnnManifest, HyperbolicDiskHnsw<PoincareMetric>, Vec<AnnPayload>)>, StoreError> {
        self.ensure_ann_index_ready(scope, family, kind)?;
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let Some(scope_ord) = self.lookup_scope_ord_in_txn(&rtxn, &scope_storage_key(scope))? else {
            return Ok(None);
        };
        let index = AnnIndexKey {
            scope_ord,
            family,
            kind: kind.map(str::to_owned),
        };
        let prefix = ann_index_prefix(&index);
        let Some(generation) = self
            .ann_head
            .get(&rtxn, &prefix)
            .map_err(|error| StoreError::Query(error.to_string()))?
            .and_then(decode_u64)
        else {
            return Ok(None);
        };
        let generation = AnnGenerationId(generation);
        let generation_key = ann_generation_key(&index, generation);
        let Some(manifest): Option<AnnManifest> =
            self.read_raw_struct_in_txn(&rtxn, self.ann_manifest, &generation_key)?
        else {
            return Ok(None);
        };
        let vectors = self
            .ann_vectors
            .get(&rtxn, &generation_key)
            .map_err(|error| StoreError::Query(error.to_string()))?
            .map(|value| value.to_vec())
            .unwrap_or_default();
        let levels = self
            .ann_levels
            .get(&rtxn, &generation_key)
            .map_err(|error| StoreError::Query(error.to_string()))?
            .map(|value| value.to_vec())
            .unwrap_or_default();
        let offsets = self
            .ann_offsets
            .get(&rtxn, &generation_key)
            .map_err(|error| StoreError::Query(error.to_string()))?
            .map(|value| value.to_vec())
            .unwrap_or_default();
        let adjacency = self
            .ann_adjacency
            .get(&rtxn, &generation_key)
            .map_err(|error| StoreError::Query(error.to_string()))?
            .map(|value| value.to_vec())
            .unwrap_or_default();
        let payload_iter = self
            .ann_payload
            .prefix_iter(&rtxn, &generation_key)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut payloads = vec![None; manifest.count];
        for item in payload_iter {
            let (key, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            let ordinal = decode_ann_generation_ord_key(key, &generation_key)?;
            if let Some(slot) = payloads.get_mut(ordinal as usize) {
                *slot = Some(decode_value(bytes)?);
            }
        }
        let payloads = payloads.into_iter().flatten().collect::<Vec<_>>();
        if payloads.len() != manifest.count {
            return Ok(None);
        }
        let index = HyperbolicDiskHnsw::from_packed(
            PackedHnswGraph {
                metadata: phoenix_hyperbolic::PackedHnswMetadata::new(
                    manifest.dimension,
                    manifest.count,
                    manifest.max_level,
                    manifest.entry_point,
                    4,
                ),
                vectors,
                levels,
                offsets,
                adjacency,
            },
            PoincareMetric { curvature: 1.0 },
        )
        .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(Some((manifest, index, payloads)))
    }

    fn ensure_ann_index_ready(
        &self,
        scope: &ScopeKey,
        family: AnnIndexFamily,
        kind: Option<&str>,
    ) -> Result<(), StoreError> {
        let Some(scope_ord) = self.lookup_scope_ord(&scope_storage_key(scope))? else {
            return Ok(());
        };
        let index = AnnIndexKey {
            scope_ord,
            family,
            kind: kind.map(str::to_owned),
        };
        let dirty_key = ann_dirty_key(&index);
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let dirty = self
            .ann_dirty
            .get(&rtxn, &dirty_key)
            .map_err(|error| StoreError::Query(error.to_string()))?
            .is_some();
        drop(rtxn);
        if !dirty {
            return Ok(());
        }
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let built_at = now_ms();
        match family {
            AnnIndexFamily::Document => {
                self.rebuild_document_ann_index_in_txn(&mut wtxn, &index, built_at)?
            }
            AnnIndexFamily::Leaf => self.rebuild_leaf_ann_index_in_txn(&mut wtxn, &index, built_at)?,
            AnnIndexFamily::NodePrototype => {
                self.rebuild_node_ann_index_in_txn(&mut wtxn, &index, built_at)?
            }
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn search_ann_payloads(
        &self,
        scope: &ScopeKey,
        family: AnnIndexFamily,
        kind: Option<&str>,
        query_vector: &[f32],
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<(HnswCandidate, AnnPayload)>, StoreError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        Self::validate_semantic_vector(query_vector, "query")?;
        let Some((_, index, payloads)) = self.load_ann_query_state(scope, family, kind)? else {
            return Ok(Vec::new());
        };
        let search_limit = oversample.max(limit);
        let results = index.search(query_vector, search_limit, search_limit.max(16));
        Ok(results
            .into_iter()
            .filter_map(|candidate| {
                payloads
                    .get(candidate.id as usize)
                    .cloned()
                    .map(|payload| (candidate, payload))
            })
            .collect())
    }
}

impl CorePhoenixBundleStoreV2 for PhoenixLmdbStore {
    fn init_bundle_schema(&self) -> Result<(), CoreStoreError> {
        <Self as PhoenixBundleStoreV2>::init_bundle_schema(self).map_err(core_store_error)
    }

    fn put_bundle(&self, header: &CoreBundleHeader, payload: &[u8]) -> Result<(), CoreStoreError> {
        <Self as PhoenixBundleStoreV2>::put_bundle(self, &legacy_bundle_header(header), payload)
            .map_err(core_store_error)
    }

    fn get_bundle(&self, key: &CoreBundleKey) -> Result<Option<Vec<u8>>, CoreStoreError> {
        <Self as PhoenixBundleStoreV2>::get_bundle(self, &legacy_bundle_key(key))
            .map_err(core_store_error)
    }

    fn get_bundle_header(
        &self,
        key: &CoreBundleKey,
    ) -> Result<Option<CoreBundleHeader>, CoreStoreError> {
        <Self as PhoenixBundleStoreV2>::get_bundle_header(self, &legacy_bundle_key(key))
            .map(|header| header.map(core_bundle_header))
            .map_err(core_store_error)
    }

    fn list_bundle_headers(
        &self,
        kind: CoreBundleKind,
        scope: Option<&str>,
    ) -> Result<Vec<CoreBundleHeader>, CoreStoreError> {
        <Self as PhoenixBundleStoreV2>::list_bundle_headers(self, legacy_bundle_kind(kind), scope)
            .map(|headers| headers.into_iter().map(core_bundle_header).collect())
            .map_err(core_store_error)
    }

    fn delete_bundle(&self, key: &CoreBundleKey) -> Result<bool, CoreStoreError> {
        <Self as PhoenixBundleStoreV2>::delete_bundle(self, &legacy_bundle_key(key))
            .map_err(core_store_error)
    }
}

impl CorePhoenixArchiveStoreV2 for PhoenixLmdbStore {
    fn init_archive_schema(&self) -> Result<(), CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::init_archive_schema(self).map_err(core_store_error)
    }

    fn ingest_mode(&self) -> CoreIngestMode {
        core_ingest_mode(<Self as PhoenixArchiveStoreV2>::ingest_mode(self))
    }

    fn prepare_ingest_context(
        &self,
        session_id: Option<&SessionId>,
        documents: &[IngestDocument],
        revision: u64,
    ) -> Result<phoenix_store_native_core::PreparedIngestContext, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::prepare_ingest_context(self, session_id, documents, revision)
            .map(core_prepared_ingest_context)
            .map_err(core_store_error)
    }

    fn persist_prepared_documents(
        &self,
        prepared: &[PreparedDocument],
        session_archive: Option<&SessionArchive>,
        touched_scopes: &[DirtyScopeRecord],
        created_at: i64,
    ) -> Result<(), CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::persist_prepared_documents(
            self,
            prepared,
            session_archive,
            touched_scopes,
            created_at,
        )
        .map_err(core_store_error)
    }

    fn persist_session_archive(
        &self,
        archive: &SessionArchive,
        revision: u64,
        created_at: i64,
    ) -> Result<(), CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::persist_session_archive(self, archive, revision, created_at)
            .map_err(core_store_error)
    }

    fn load_latest_session_archive(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<SessionArchive>, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::load_latest_session_archive(self, session_id)
            .map_err(core_store_error)
    }

    fn load_latest_document_archives(
        &self,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<DocumentArchive>, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::load_latest_document_archives(self, scope)
            .map_err(core_store_error)
    }

    fn load_document_manifest(
        &self,
        document_ref: &DocumentRevisionRef,
    ) -> Result<Option<DocumentManifest>, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::load_document_manifest(self, document_ref)
            .map_err(core_store_error)
    }

    fn load_scope_sidecar(&self, scope: &ScopeKey) -> Result<Option<ScopeLexSidecar>, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::load_scope_sidecar(self, scope).map_err(core_store_error)
    }

    fn load_materialized_scope_lexical(
        &self,
        scope: &ScopeKey,
    ) -> Result<ScopeLexSidecar, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::load_materialized_scope_lexical(self, scope)
            .map_err(core_store_error)
    }

    fn load_lex_spans(&self, scope: Option<&ScopeKey>) -> Result<Vec<IndexedSpan>, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::load_lex_spans(self, scope).map_err(core_store_error)
    }

    fn lookup_alias_postings(
        &self,
        scope: &ScopeKey,
        normalized: &str,
    ) -> Result<Vec<AliasPosting>, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::lookup_alias_postings(self, scope, normalized)
            .map_err(core_store_error)
    }

    fn rebuild_dirty_scope_sidecars(&self, created_at: i64) -> Result<usize, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::rebuild_dirty_scope_sidecars(self, created_at)
            .map_err(core_store_error)
    }

    fn list_dirty_scopes(&self) -> Result<Vec<DirtyScopeRecord>, CoreStoreError> {
        <Self as PhoenixArchiveStoreV2>::list_dirty_scopes(self).map_err(core_store_error)
    }
}

impl PhoenixBundleStoreV2 for PhoenixLmdbStore {
    fn init_bundle_schema(&self) -> Result<(), StoreError> {
        Ok(())
    }

    fn put_bundle(&self, header: &BundleHeader, payload: &[u8]) -> Result<(), StoreError> {
        let key = bundle_storage_key(&header.key);
        let encoded_header = encode_value(header)?;
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        self.write_bytes(&mut wtxn, self.compat_bundle_headers, &key, &encoded_header)?;
        self.write_bytes(&mut wtxn, self.compat_bundle_payloads, &key, payload)?;
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn get_bundle(&self, key: &BundleKey) -> Result<Option<Vec<u8>>, StoreError> {
        let storage_key = bundle_storage_key(key);
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(self
            .compat_bundle_payloads
            .get(&rtxn, &storage_key)
            .map_err(|error| StoreError::Query(error.to_string()))?
            .map(|bytes| bytes.to_vec()))
    }

    fn get_bundle_header(&self, key: &BundleKey) -> Result<Option<BundleHeader>, StoreError> {
        self.read_struct(self.compat_bundle_headers, &bundle_storage_key(key))
    }

    fn list_bundle_headers(
        &self,
        kind: BundleKind,
        scope: Option<&str>,
    ) -> Result<Vec<BundleHeader>, StoreError> {
        self.iter_structs_with_prefix(self.compat_bundle_headers, Some(&bundle_prefix(kind, scope)))
    }

    fn delete_bundle(&self, key: &BundleKey) -> Result<bool, StoreError> {
        let storage_key = bundle_storage_key(key);
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let deleted_header = self
            .compat_bundle_headers
            .delete(&mut wtxn, &storage_key)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let deleted_payload = self
            .compat_bundle_payloads
            .delete(&mut wtxn, &storage_key)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(deleted_header || deleted_payload)
    }
}

impl PhoenixSemanticIndexStore for PhoenixLmdbStore {
    fn upsert_semantic_leaf_vectors(
        &self,
        rows: &[NativeSemanticLeafVectorRecord],
    ) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut affected = HashSet::new();
        for row in rows {
            Self::validate_semantic_vector(&row.values, &row.span_id)?;
            let scope_key = scope_storage_key(&row.scope);
            let scope_ord = ensure_scope_ord_in_txn(
                &mut wtxn,
                self.scope_by_value,
                self.scope_by_ord,
                self.ord_counter,
                &scope_key,
            )?;
            let index = AnnIndexKey {
                scope_ord,
                family: AnnIndexFamily::Leaf,
                kind: None,
            };
            affected.insert(index.clone());
            let record = AnnSourceLeafRecord {
                scope: row.scope.clone(),
                scope_key,
                span_id: row.span_id.clone(),
                document_id: row.document_id.clone(),
                values: row.values.clone(),
                updated_at: row.updated_at,
            };
            self.write_raw_bytes(
                &mut wtxn,
                self.ann_source_leaf,
                &ann_source_entry_key(scope_ord, AnnIndexFamily::Leaf, None, &row.span_id),
                &encode_value(&record)?,
            )?;
        }
        let dirty_at = now_ms() as u64;
        for index in affected {
            self.write_raw_bytes(
                &mut wtxn,
                self.ann_dirty,
                &ann_dirty_key(&index),
                &dirty_at.to_be_bytes(),
            )?;
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn upsert_semantic_document_vectors_native(
        &self,
        rows: &[NativeSemanticDocumentVectorRecord],
    ) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut affected = HashSet::new();
        for row in rows {
            Self::validate_semantic_vector(&row.values, &row.document_id)?;
            let scope_key = scope_storage_key(&row.scope);
            let scope_ord = ensure_scope_ord_in_txn(
                &mut wtxn,
                self.scope_by_value,
                self.scope_by_ord,
                self.ord_counter,
                &scope_key,
            )?;
            let index = AnnIndexKey {
                scope_ord,
                family: AnnIndexFamily::Document,
                kind: None,
            };
            affected.insert(index.clone());
            let record = AnnSourceDocumentRecord {
                scope: row.scope.clone(),
                scope_key,
                document_id: row.document_id.clone(),
                values: row.values.clone(),
                leaf_count: row.leaf_count,
                evidence_refs: row.evidence_refs.clone(),
                updated_at: row.updated_at,
            };
            self.write_raw_bytes(
                &mut wtxn,
                self.ann_source_document,
                &ann_source_entry_key(scope_ord, AnnIndexFamily::Document, None, &row.document_id),
                &encode_value(&record)?,
            )?;
        }
        let dirty_at = now_ms() as u64;
        for index in affected {
            self.write_raw_bytes(
                &mut wtxn,
                self.ann_dirty,
                &ann_dirty_key(&index),
                &dirty_at.to_be_bytes(),
            )?;
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn upsert_semantic_node_vectors_native(
        &self,
        rows: &[NativeSemanticNodeVectorRecord],
    ) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut affected = HashSet::new();
        for row in rows {
            Self::validate_semantic_vector(&row.values, &row.node_id)?;
            let scope_key = scope_storage_key(&row.scope);
            let scope_ord = ensure_scope_ord_in_txn(
                &mut wtxn,
                self.scope_by_value,
                self.scope_by_ord,
                self.ord_counter,
                &scope_key,
            )?;
            let index = AnnIndexKey {
                scope_ord,
                family: AnnIndexFamily::NodePrototype,
                kind: Some(row.node_kind.clone()),
            };
            affected.insert(index.clone());
            let record = AnnSourceNodeRecord {
                scope: row.scope.clone(),
                scope_key,
                node_id: row.node_id.clone(),
                node_kind: row.node_kind.clone(),
                document_id: row.document_id.clone(),
                narrative_id: row.narrative_id.clone(),
                folder_id: row.folder_id.clone(),
                values: row.values.clone(),
                evidence_refs: row.evidence_refs.clone(),
                updated_at: row.updated_at,
            };
            self.write_raw_bytes(
                &mut wtxn,
                self.ann_source_node,
                &ann_source_entry_key(
                    scope_ord,
                    AnnIndexFamily::NodePrototype,
                    Some(&row.node_kind),
                    &row.node_id,
                ),
                &encode_value(&record)?,
            )?;
        }
        let dirty_at = now_ms() as u64;
        for index in affected {
            self.write_raw_bytes(
                &mut wtxn,
                self.ann_dirty,
                &ann_dirty_key(&index),
                &dirty_at.to_be_bytes(),
            )?;
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn query_semantic_neighbors(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticNeighbor>, StoreError> {
        Ok(self
            .search_ann_payloads(
                scope,
                AnnIndexFamily::Leaf,
                None,
                query_vector,
                limit,
                oversample,
            )?
            .into_iter()
            .filter_map(|(candidate, payload)| match payload {
                AnnPayload::Leaf { span_id, .. } => Some(SemanticNeighbor {
                    span_id,
                    distance: candidate.dist as f64,
                }),
                _ => None,
            })
            .take(limit)
            .collect())
    }

    fn query_semantic_documents(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticDocumentNeighbor>, StoreError> {
        Ok(self
            .search_ann_payloads(
                scope,
                AnnIndexFamily::Document,
                None,
                query_vector,
                limit,
                oversample,
            )?
            .into_iter()
            .filter_map(|(candidate, payload)| match payload {
                AnnPayload::Document {
                    document_id,
                    leaf_count,
                    evidence_refs,
                } => Some(SemanticDocumentNeighbor {
                    document_id,
                    distance: candidate.dist as f64,
                    leaf_count,
                    evidence_refs,
                }),
                _ => None,
            })
            .take(limit)
            .collect())
    }

    fn query_semantic_neighbors_in_documents(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        document_ids: &[String],
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticNeighbor>, StoreError> {
        if limit == 0 || document_ids.is_empty() {
            return Ok(Vec::new());
        }
        let allowed = document_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let Some((manifest, index, payloads)) =
            self.load_ann_query_state(scope, AnnIndexFamily::Leaf, None)?
        else {
            return Ok(Vec::new());
        };
        Self::validate_semantic_vector(query_vector, "query")?;
        let mut search_k = oversample
            .max(limit)
            .max(8)
            .min(manifest.count.max(limit).max(1));
        let mut hits = Vec::new();
        while search_k <= manifest.count.max(limit) {
            hits.clear();
            for candidate in index.search(query_vector, search_k, search_k.max(16)) {
                let Some(payload) = payloads.get(candidate.id as usize) else {
                    continue;
                };
                if let AnnPayload::Leaf { span_id, document_id } = payload {
                    if allowed.contains(document_id.as_str()) {
                        hits.push(SemanticNeighbor {
                            span_id: span_id.clone(),
                            distance: candidate.dist as f64,
                        });
                        if hits.len() >= limit {
                            return Ok(hits);
                        }
                    }
                }
            }
            if search_k >= manifest.count {
                break;
            }
            search_k = (search_k * 2).min(manifest.count);
        }
        Ok(hits)
    }

    fn query_semantic_node_neighbors(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        kind: &str,
        exclude_node_id: Option<&str>,
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticNodeNeighbor>, StoreError> {
        Ok(self
            .search_ann_payloads(
                scope,
                AnnIndexFamily::NodePrototype,
                Some(kind),
                query_vector,
                limit.saturating_add(exclude_node_id.is_some() as usize),
                oversample.max(limit),
            )?
            .into_iter()
            .filter_map(|(candidate, payload)| match payload {
                AnnPayload::Node {
                    node_id,
                    node_kind,
                    document_id,
                    narrative_id,
                    folder_id,
                    evidence_refs,
                } if exclude_node_id != Some(node_id.as_str()) => Some(SemanticNodeNeighbor {
                    node_id,
                    node_kind,
                    distance: candidate.dist as f64,
                    document_id,
                    narrative_id,
                    folder_id,
                    evidence_refs,
                }),
                _ => None,
            })
            .take(limit)
            .collect())
    }

    fn load_semantic_document_vector_records(
        &self,
        document_ids: &[String],
    ) -> Result<Vec<NativeSemanticDocumentVectorRecord>, StoreError> {
        let allowed = document_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let iter = self
            .ann_source_document
            .iter(&rtxn)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut records = Vec::new();
        for item in iter {
            let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            let record: AnnSourceDocumentRecord = decode_value(bytes)?;
            if allowed.is_empty() || allowed.contains(record.document_id.as_str()) {
                records.push(NativeSemanticDocumentVectorRecord {
                    scope: record.scope,
                    document_id: record.document_id,
                    values: record.values,
                    leaf_count: record.leaf_count,
                    evidence_refs: record.evidence_refs,
                    updated_at: record.updated_at,
                });
            }
        }
        Ok(records)
    }

    fn load_semantic_node_vector_records(
        &self,
        node_ids: &[String],
    ) -> Result<Vec<NativeSemanticNodeVectorRecord>, StoreError> {
        let allowed = node_ids.iter().map(String::as_str).collect::<BTreeSet<_>>();
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let iter = self
            .ann_source_node
            .iter(&rtxn)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut records = Vec::new();
        for item in iter {
            let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            let record: AnnSourceNodeRecord = decode_value(bytes)?;
            if allowed.is_empty() || allowed.contains(record.node_id.as_str()) {
                records.push(NativeSemanticNodeVectorRecord {
                    scope: record.scope,
                    node_id: record.node_id,
                    node_kind: record.node_kind,
                    document_id: record.document_id,
                    narrative_id: record.narrative_id,
                    folder_id: record.folder_id,
                    values: record.values,
                    evidence_refs: record.evidence_refs,
                    updated_at: record.updated_at,
                });
            }
        }
        Ok(records)
    }
}

impl PhoenixArchiveStoreV2 for PhoenixLmdbStore {
    fn init_archive_schema(&self) -> Result<(), StoreError> {
        Ok(())
    }

    fn ingest_mode(&self) -> IngestMode {
        IngestMode::Safe
    }

    fn prepare_ingest_context(
        &self,
        session_id: Option<&SessionId>,
        documents: &[IngestDocument],
        revision: u64,
    ) -> Result<PreparedIngestContext, StoreError> {
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let session_ord = match session_id {
            Some(session_id) => Some(ensure_session_ord_in_txn(
                &mut wtxn,
                self.session_by_value,
                self.session_by_ord,
                self.ord_counter,
                session_id,
            )?),
            None => None,
        };
        let mut assignments = Vec::with_capacity(documents.len());
        for document in documents {
            let scope_key = scope_storage_key(&document.scope);
            let scope_ord = ensure_scope_ord_in_txn(
                &mut wtxn,
                self.scope_by_value,
                self.scope_by_ord,
                self.ord_counter,
                &scope_key,
            )?;
            let document_ord = ensure_document_ord_in_txn(
                &mut wtxn,
                self.doc_by_value,
                self.doc_by_ord,
                self.ord_counter,
                &scope_key,
                &document.document_id.0,
            )?;
            assignments.push(DocumentOrdinalAssignment {
                document_id: document.document_id.0.clone(),
                scope: document.scope.clone(),
                scope_key,
                scope_ord,
                document_ord,
                revision: revision + 1,
            });
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(PreparedIngestContext {
            session_id: session_id.cloned(),
            session_ord,
            assignments,
            kernel_snapshot: Some(self.load_live_kernel_snapshot()?),
        })
    }

    fn persist_prepared_documents(
        &self,
        prepared: &[PreparedDocument],
        session_archive: Option<&SessionArchive>,
        touched_scopes: &[DirtyScopeRecord],
        _created_at: i64,
    ) -> Result<(), StoreError> {
        let document_batch = build_prepared_documents_write_batch(prepared, touched_scopes)?;
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        self.apply_write_batch(&mut wtxn, document_batch)?;
        if let Some(session_archive) = session_archive {
            persist_session_archive_in_txn(
                self,
                &mut wtxn,
                session_archive,
                session_archive
                    .document_refs
                    .iter()
                    .map(|document| document.revision)
                    .max()
                    .unwrap_or(0),
                )?;
        }
        self.commit_write_txn(wtxn)
    }

    fn persist_session_archive(
        &self,
        archive: &SessionArchive,
        revision: u64,
        _created_at: i64,
    ) -> Result<(), StoreError> {
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        persist_session_archive_in_txn(self, &mut wtxn, archive, revision)?;
        self.commit_write_txn(wtxn)
    }

    fn load_latest_session_archive(
        &self,
        session_id: &SessionId,
    ) -> Result<Option<SessionArchive>, StoreError> {
        let Some(session_ord) = self.lookup_session_ord(session_id)? else {
            return Ok(None);
        };
        let Some(revision) = self.lookup_latest_session_revision(session_ord)? else {
            return Ok(None);
        };
        self.read_raw_struct(self.session_archive, &session_revision_key(session_ord, revision))
    }

    fn load_latest_document_archives(
        &self,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<DocumentArchive>, StoreError> {
        let manifests = self.load_latest_document_manifests(scope)?;
        let mut archives = Vec::with_capacity(manifests.len());
        for manifest in manifests {
            archives.push(self.load_document_archive_from_manifest(&manifest)?);
        }
        Ok(archives)
    }

    fn load_document_manifest(
        &self,
        document_ref: &DocumentRevisionRef,
    ) -> Result<Option<DocumentManifest>, StoreError> {
        self.read_raw_struct(
            self.doc_manifest,
            &document_revision_key(
                document_ref.scope_ord,
                document_ref.document_ord,
                document_ref.revision,
            ),
        )
    }

    fn load_scope_sidecar(&self, scope: &ScopeKey) -> Result<Option<ScopeLexSidecar>, StoreError> {
        let Some(scope_ord) = self.lookup_scope_ord(&scope_storage_key(scope))? else {
            return Ok(None);
        };
        self.read_raw_struct(self.scope_sidecar_meta, &scope_sidecar_key(scope_ord))
    }

    fn load_materialized_scope_lexical(
        &self,
        scope: &ScopeKey,
    ) -> Result<ScopeLexSidecar, StoreError> {
        self.materialize_scope_lexical(scope)
    }

    fn load_lex_spans(&self, scope: Option<&ScopeKey>) -> Result<Vec<IndexedSpan>, StoreError> {
        if let Some(scope) = scope {
            return Ok(self.materialize_scope_lexical(scope)?.spans);
        }

        let mut spans = Vec::new();
        for scope in self.list_all_scopes()? {
            spans.extend(self.materialize_scope_lexical(&scope)?.spans);
        }
        spans.sort_by(|left, right| left.span_id.cmp(&right.span_id));
        spans.dedup_by(|left, right| left.span_id == right.span_id);
        Ok(spans)
    }

    fn lookup_alias_postings(
        &self,
        scope: &ScopeKey,
        normalized: &str,
    ) -> Result<Vec<AliasPosting>, StoreError> {
        let sidecar = self.materialize_scope_lexical(scope)?;
        Ok(sidecar
            .alias_entries
            .into_iter()
            .find(|entry| entry.normalized == normalized)
            .map(|entry| entry.postings)
            .unwrap_or_default())
    }

    fn rebuild_dirty_scope_sidecars(&self, created_at: i64) -> Result<usize, StoreError> {
        let dirty = <Self as PhoenixArchiveStoreV2>::list_dirty_scopes(self)?;
        if dirty.is_empty() {
            return Ok(0);
        }
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut rebuilt = Vec::with_capacity(dirty.len());
        for record in &dirty {
            let persisted = self.read_raw_struct_in_txn::<ScopeLexSidecar>(
                &rtxn,
                self.scope_sidecar_meta,
                &scope_sidecar_key(record.scope_ord),
            )?;
            rebuilt.push(self.aggregate_scope_sidecar_with_txn(
                &rtxn,
                &record.scope,
                record.scope_ord,
                persisted,
                Some(record),
            )?);
        }
        drop(rtxn);
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        for (record, mut sidecar) in dirty.iter().zip(rebuilt.into_iter()) {
            sidecar.scope_ord = Some(record.scope_ord);
            sidecar.generated_at = created_at;
            sidecar.generation = sidecar.generation.max(record.updated_at as u64);
            self.write_raw_bytes(
                &mut wtxn,
                self.scope_sidecar_meta,
                &scope_sidecar_key(record.scope_ord),
                &encode_value(&sidecar)?,
            )?;
            self.write_raw_bytes(
                &mut wtxn,
                self.scope_sidecar_postings,
                &scope_sidecar_key(record.scope_ord),
                &encode_value(&sidecar.alias_entries)?,
            )?;
            self.write_raw_bytes(
                &mut wtxn,
                self.scope_sidecar_fst,
                &scope_sidecar_key(record.scope_ord),
                &[],
            )?;
            self.idx_scope_dirty
                .delete(&mut wtxn, &scope_dirty_key(record.scope_ord))
                .map_err(|error| StoreError::Query(error.to_string()))?;
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(dirty.len())
    }

    fn list_dirty_scopes(&self) -> Result<Vec<DirtyScopeRecord>, StoreError> {
        let mut values = self.iter_raw_structs_with_prefix(self.idx_scope_dirty, None)?;
        values.sort_by(|left: &DirtyScopeRecord, right: &DirtyScopeRecord| {
            left.scope_key.cmp(&right.scope_key)
        });
        Ok(values)
    }
}

impl PhoenixNativeRowStore for PhoenixLmdbStore {
    fn init_schema(&self) -> Result<(), StoreError> {
        self.put_row(
            "phoenix_schema_state",
            serde_json::json!({
                "version": self.schema_version(),
                "updated_at": now_ms(),
            }),
        )
    }

    fn relation_names(&self) -> Vec<&'static str> {
        NATIVE_COVERED_RELATIONS.to_vec()
    }

    fn relation_counts(&self) -> Result<Vec<(String, usize)>, StoreError> {
        let mut counts = Vec::with_capacity(NATIVE_COVERED_RELATIONS.len());
        for relation in NATIVE_COVERED_RELATIONS {
            counts.push(((*relation).to_owned(), self.count_relation_rows(relation)?));
        }
        Ok(counts)
    }

    fn fetch_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        ensure_native_relation_supported(relation)?;
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let iter = self
            .compat_relation_rows
            .prefix_iter(&rtxn, &relation_prefix(relation))
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut rows = Vec::new();
        for item in iter {
            let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            rows.push(decode_value(bytes)?);
        }
        Ok(rows)
    }

    fn put_row(&self, relation: &str, row: Value) -> Result<(), StoreError> {
        let key = relation_row_storage_key(relation, &row)?;
        let bytes = encode_value(&row)?;
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        self.write_bytes(&mut wtxn, self.compat_relation_rows, &key, &bytes)?;
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn put_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError> {
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        for row in rows {
            let key = relation_row_storage_key(relation, row)?;
            let bytes = encode_value(row)?;
            self.write_bytes(&mut wtxn, self.compat_relation_rows, &key, &bytes)?;
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn replace_relation_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError> {
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        clear_relation_in_txn(&mut wtxn, self.compat_relation_rows, relation)?;
        for row in rows {
            let key = relation_row_storage_key(relation, row)?;
            let bytes = encode_value(row)?;
            self.write_bytes(&mut wtxn, self.compat_relation_rows, &key, &bytes)?;
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn delete_rows(&self, relation: &str, rows: &[Value]) -> Result<usize, StoreError> {
        ensure_native_relation_supported(relation)?;
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut deleted = 0usize;
        for row in rows {
            let key = relation_row_storage_key(relation, row)?;
            if self
                .compat_relation_rows
                .delete(&mut wtxn, &key)
                .map_err(|error| StoreError::Query(error.to_string()))?
            {
                deleted += 1;
            }
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(deleted)
    }

    fn clear_relations(&self, relations: &[&str]) -> Result<(), StoreError> {
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        for relation in relations {
            clear_relation_in_txn(&mut wtxn, self.compat_relation_rows, relation)?;
        }
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn export_snapshot_partition(
        &self,
        partition: SnapshotPartition,
    ) -> Result<Vec<u8>, StoreError> {
        let relation_names = native_relation_snapshot_names(partition);
        let mut relations = BTreeMap::new();
        for relation in relation_names {
            relations.insert(relation.to_owned(), self.fetch_rows(relation)?);
        }
        let envelope = phoenix_store_cozo::SnapshotEnvelope {
            schema_version: self.schema_version().to_owned(),
            relation_count: relations.len(),
            created_at: now_ms(),
            relations,
            checksum: None,
        };
        serde_json::to_vec(&envelope).map_err(|error| StoreError::Snapshot(error.to_string()))
    }

    fn import_snapshot(&self, bytes: &[u8]) -> Result<phoenix_store_cozo::SnapshotEnvelope, StoreError> {
        let mut envelope: phoenix_store_cozo::SnapshotEnvelope =
            serde_json::from_slice(bytes).map_err(|error| StoreError::Snapshot(error.to_string()))?;
        let relations = envelope.relations.clone();
        let relation_names = relations.keys().map(String::as_str).collect::<Vec<_>>();
        self.clear_relations(&relation_names)?;
        for (relation, rows) in relations {
            self.put_rows(&relation, &rows)?;
        }
        envelope.relations.clear();
        Ok(envelope)
    }
}

impl PhoenixGraphKernelStoreV2 for PhoenixLmdbStore {
    fn init_graph_kernel_schema(&self) -> Result<(), StoreError> {
        Ok(())
    }

    fn load_kernel_checkpoint(&self) -> Result<Option<KernelCheckpointData>, StoreError> {
        self.read_struct(self.graph_kernel_checkpoint, CHECKPOINT_KEY)
    }

    fn write_kernel_checkpoint(
        &self,
        generation: u64,
        source_revision: &str,
        snapshot: &KernelGraphSnapshot,
    ) -> Result<KernelCheckpointData, StoreError> {
        let checkpoint = KernelCheckpointData {
            meta: KernelCheckpointMeta {
                checkpoint_id: format!("kernel-checkpoint-{generation}"),
                generation,
                source_revision: source_revision.to_owned(),
                created_at: now_ms(),
            },
            snapshot: snapshot.clone(),
        };
        let encoded = encode_value(&checkpoint)?;
        let mut wtxn = self
            .env
            .write_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        self.write_bytes(&mut wtxn, self.graph_kernel_checkpoint, CHECKPOINT_KEY, &encoded)?;
        compact_kernel_journal(&mut wtxn, self.graph_kernel_journal, generation)?;
        wtxn.commit()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        self.cache_live_kernel_snapshot(generation, snapshot.clone());
        Ok(checkpoint)
    }

    fn load_kernel_journal_after(
        &self,
        generation: u64,
    ) -> Result<Vec<KernelJournalEntry>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let iter = self
            .graph_kernel_journal
            .iter(&rtxn)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut entries = Vec::new();
        for item in iter {
            let (_, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            let entry: KernelJournalEntry = decode_value(bytes)?;
            if entry.generation > generation {
                entries.push(entry);
            }
        }
        entries.sort_by(|left, right| {
            left.generation.cmp(&right.generation).then_with(|| left.created_at.cmp(&right.created_at))
        });
        Ok(entries)
    }

    fn append_kernel_batch(
        &self,
        generation: u64,
        source_revision: &str,
        batch: &KernelMutationBatch,
        created_at: i64,
    ) -> Result<(), StoreError> {
        let entry = KernelJournalEntry {
            generation,
            source_revision: source_revision.to_owned(),
            batch: Some(batch.clone()),
            commit_id: None,
            created_at,
        };
        append_kernel_entry(self, entry, None)?;
        self.invalidate_live_kernel_snapshot();
        self.live_kernel_generation
            .store(generation, Ordering::Release);
        Ok(())
    }

    fn append_kernel_commit_marker(
        &self,
        generation: u64,
        source_revision: &str,
        commit_id: &str,
        created_at: i64,
    ) -> Result<(), StoreError> {
        let entry = KernelJournalEntry {
            generation,
            source_revision: source_revision.to_owned(),
            batch: None,
            commit_id: Some(commit_id.to_owned()),
            created_at,
        };
        append_kernel_entry(self, entry, Some((commit_id, generation)))
    }

    fn kernel_generation_for_commit(&self, commit_id: &str) -> Result<Option<u64>, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let value = self
            .graph_kernel_commit_index
            .get(&rtxn, commit_id)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(value.and_then(decode_u64))
    }

    fn kernel_current_generation(&self) -> Result<u64, StoreError> {
        let checkpoint_generation = self
            .load_kernel_checkpoint()?
            .map(|checkpoint| checkpoint.meta.generation)
            .unwrap_or_default();
        let journal_generation = self
            .latest_kernel_journal_generation()?
            .unwrap_or_default();
        Ok(checkpoint_generation.max(journal_generation))
    }

    fn kernel_journal_len(&self) -> Result<usize, StoreError> {
        let rtxn = self
            .env
            .read_txn()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let iter = self
            .graph_kernel_journal
            .iter(&rtxn)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut len = 0usize;
        for item in iter {
            item.map_err(|error| StoreError::Query(error.to_string()))?;
            len += 1;
        }
        Ok(len)
    }
}

fn open_str_bytes_db(
    env: &Env,
    wtxn: &mut RwTxn,
    name: &str,
    dup_sort: bool,
) -> Result<Database<Str, Bytes>, StoreError> {
    let mut options = DatabaseOpenOptions::new(env).types::<Str, Bytes>();
    options.name(name);
    if dup_sort {
        options.flags(DatabaseFlags::DUP_SORT);
    }
    options
        .create(wtxn)
        .map_err(|error| StoreError::Init(error.to_string()))
}

fn open_raw_bytes_db(
    env: &Env,
    wtxn: &mut RwTxn,
    name: &str,
    dup_sort: bool,
) -> Result<Database<Bytes, Bytes>, StoreError> {
    let mut options = DatabaseOpenOptions::new(env).types::<Bytes, Bytes>();
    options.name(name);
    if dup_sort {
        options.flags(DatabaseFlags::DUP_SORT);
    }
    options
        .create(wtxn)
        .map_err(|error| StoreError::Init(error.to_string()))
}

fn append_kernel_entry(
    store: &PhoenixLmdbStore,
    entry: KernelJournalEntry,
    commit_index: Option<(&str, u64)>,
) -> Result<(), StoreError> {
    let storage_key = journal_entry_key(entry.generation, entry.created_at);
    let encoded = encode_value(&entry)?;
    let mut wtxn = store
        .env
        .write_txn()
        .map_err(|error| StoreError::Query(error.to_string()))?;
    store.write_bytes(&mut wtxn, store.graph_kernel_journal, &storage_key, &encoded)?;
    if let Some((commit_id, generation)) = commit_index {
        store.write_bytes(
            &mut wtxn,
            store.graph_kernel_commit_index,
            commit_id,
            &generation.to_be_bytes(),
        )?;
    }
    wtxn.commit()
        .map_err(|error| StoreError::Query(error.to_string()))
}

fn compact_kernel_journal(
    wtxn: &mut RwTxn,
    db: Database<Str, Bytes>,
    up_to_generation: u64,
) -> Result<(), StoreError> {
    let mut stale_keys = Vec::new();
    {
        let iter = db
            .iter(wtxn)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        for item in iter {
            let (key, bytes) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            let entry: KernelJournalEntry = decode_value(bytes)?;
            if entry.generation <= up_to_generation {
                stale_keys.push(key.to_owned());
            }
        }
    }
    for key in stale_keys {
        db.delete(wtxn, &key)
            .map_err(|error| StoreError::Query(error.to_string()))?;
    }
    Ok(())
}

fn open_lmdb_env(path: &Path, tuning: &LmdbTuning) -> Result<Env, StoreError> {
    let mut options = EnvOpenOptions::new();
    options
        .map_size(tuning.map_size_bytes)
        .max_dbs(tuning.max_dbs);
    if let Some(readers) = tuning.max_readers {
        options.max_readers(readers);
    }
    let mut flags = EnvFlags::empty();
    match tuning.durability {
        IngestDurabilityMode::Safe => {}
        IngestDurabilityMode::NoMetaSync => flags |= EnvFlags::NO_META_SYNC,
        IngestDurabilityMode::NoSync | IngestDurabilityMode::NoSyncPeriodic { .. } => {
            flags |= EnvFlags::NO_SYNC
        }
    }
    if tuning.no_read_ahead {
        flags |= EnvFlags::NO_READ_AHEAD;
    }
    if !tuning.mem_init {
        flags |= EnvFlags::NO_MEM_INIT;
    }
    if tuning.write_map {
        flags |= EnvFlags::WRITE_MAP;
    }
    if tuning.map_async {
        flags |= EnvFlags::MAP_ASYNC;
    }
    unsafe {
        options.flags(flags);
        options
            .open(path)
            .map_err(|error| StoreError::Init(error.to_string()))
    }
}

fn read_env_bool(name: &str) -> Option<bool> {
    std::env::var(name).ok().and_then(|value| match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" => Some(false),
        _ => None,
    })
}

fn read_env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok()?.parse().ok()
}

fn read_env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.parse().ok()
}

fn read_env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.parse().ok()
}

fn encode_value<T: Serialize>(value: &T) -> Result<Vec<u8>, StoreError> {
    rmp_serde::to_vec_named(value).map_err(|error| StoreError::Query(error.to_string()))
}

fn decode_value<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, StoreError> {
    rmp_serde::from_slice(bytes).map_err(|error| StoreError::Query(error.to_string()))
}

fn decode_u64(bytes: &[u8]) -> Option<u64> {
    (bytes.len() == 8).then(|| {
        let mut buffer = [0u8; 8];
        buffer.copy_from_slice(bytes);
        u64::from_be_bytes(buffer)
    })
}

fn decode_ord(bytes: &[u8]) -> Option<u64> {
    decode_u64(bytes)
}

fn dump_db(env: &Env, db: Database<Str, Bytes>) -> Result<Vec<(String, Vec<u8>)>, StoreError> {
    let rtxn = env
        .read_txn()
        .map_err(|error| StoreError::Snapshot(error.to_string()))?;
    let iter = db
        .iter(&rtxn)
        .map_err(|error| StoreError::Snapshot(error.to_string()))?;
    let mut rows = Vec::new();
    for item in iter {
        let (key, bytes) = item.map_err(|error| StoreError::Snapshot(error.to_string()))?;
        rows.push((key.to_owned(), bytes.to_vec()));
    }
    Ok(rows)
}

fn dump_raw_db(env: &Env, db: Database<Bytes, Bytes>) -> Result<Vec<(Vec<u8>, Vec<u8>)>, StoreError> {
    let rtxn = env
        .read_txn()
        .map_err(|error| StoreError::Snapshot(error.to_string()))?;
    let iter = db
        .iter(&rtxn)
        .map_err(|error| StoreError::Snapshot(error.to_string()))?;
    let mut rows = Vec::new();
    for item in iter {
        let (key, bytes) = item.map_err(|error| StoreError::Snapshot(error.to_string()))?;
        rows.push((key.to_vec(), bytes.to_vec()));
    }
    Ok(rows)
}

fn dump_relation_db(
    env: &Env,
    db: Database<Str, Bytes>,
    relations: &[&str],
) -> Result<Vec<(String, Vec<u8>)>, StoreError> {
    let rtxn = env
        .read_txn()
        .map_err(|error| StoreError::Snapshot(error.to_string()))?;
    let mut rows = Vec::new();
    for relation in relations {
        if !NATIVE_COVERED_RELATIONS.contains(relation) {
            continue;
        }
        let iter = db
            .prefix_iter(&rtxn, &relation_prefix(relation))
            .map_err(|error| StoreError::Snapshot(error.to_string()))?;
        for item in iter {
            let (key, bytes) = item.map_err(|error| StoreError::Snapshot(error.to_string()))?;
            rows.push((key.to_owned(), bytes.to_vec()));
        }
    }
    Ok(rows)
}

fn dump_snapshot_section<T, F>(
    section: &str,
    enabled: bool,
    dump: F,
) -> Result<Vec<(T, Vec<u8>)>, StoreError>
where
    F: FnOnce() -> Result<Vec<(T, Vec<u8>)>, StoreError>,
{
    if !enabled {
        return Ok(Vec::new());
    }
    let started = std::time::Instant::now();
    let rows = dump()?;
    if native_progress_enabled() {
        eprintln!(
            "[runtime-snapshot] section={} rows={} wall_ms={}",
            section,
            rows.len(),
            started.elapsed().as_millis()
        );
    }
    Ok(rows)
}

fn restore_snapshot_section<T, F>(section: &str, restore: F) -> Result<T, StoreError>
where
    F: FnOnce() -> Result<T, StoreError>,
{
    let started = std::time::Instant::now();
    let value = restore()?;
    if native_progress_enabled() {
        eprintln!(
            "[runtime-snapshot] import_section={} wall_ms={}",
            section,
            started.elapsed().as_millis()
        );
    }
    Ok(value)
}

fn restore_db(
    wtxn: &mut RwTxn,
    db: Database<Str, Bytes>,
    rows: &[(String, Vec<u8>)],
) -> Result<(), StoreError> {
    db.clear(wtxn)
        .map_err(|error| StoreError::Snapshot(error.to_string()))?;
    for (key, bytes) in rows {
        db.put_reserved(wtxn, key, bytes.len(), |reserved| reserved.write_all(bytes))
            .map_err(|error| StoreError::Snapshot(error.to_string()))?;
    }
    Ok(())
}

fn restore_raw_db(
    wtxn: &mut RwTxn,
    db: Database<Bytes, Bytes>,
    rows: &[(Vec<u8>, Vec<u8>)],
) -> Result<(), StoreError> {
    db.clear(wtxn)
        .map_err(|error| StoreError::Snapshot(error.to_string()))?;
    for (key, bytes) in rows {
        db.put_reserved(wtxn, key, bytes.len(), |reserved| reserved.write_all(bytes))
            .map_err(|error| StoreError::Snapshot(error.to_string()))?;
    }
    Ok(())
}

fn ensure_scope_ord_in_txn(
    wtxn: &mut RwTxn,
    value_db: Database<Str, Bytes>,
    ord_db: Database<Bytes, Bytes>,
    counter_db: Database<Bytes, Bytes>,
    scope_key: &str,
) -> Result<ScopeOrd, StoreError> {
    ensure_named_ord_in_txn(wtxn, value_db, ord_db, counter_db, ORD_COUNTER_SCOPE_KEY, scope_key)
        .map(ScopeOrd)
}

fn ensure_session_ord_in_txn(
    wtxn: &mut RwTxn,
    value_db: Database<Str, Bytes>,
    ord_db: Database<Bytes, Bytes>,
    counter_db: Database<Bytes, Bytes>,
    session_id: &SessionId,
) -> Result<SessionOrd, StoreError> {
    ensure_named_ord_in_txn(
        wtxn,
        value_db,
        ord_db,
        counter_db,
        ORD_COUNTER_SESSION_KEY,
        &session_id.0,
    )
    .map(SessionOrd)
}

fn ensure_document_ord_in_txn(
    wtxn: &mut RwTxn,
    value_db: Database<Str, Bytes>,
    ord_db: Database<Bytes, Bytes>,
    counter_db: Database<Bytes, Bytes>,
    scope_key: &str,
    document_id: &str,
) -> Result<DocumentOrd, StoreError> {
    let value_key = document_value_key(scope_key, document_id);
    ensure_named_ord_in_txn(
        wtxn,
        value_db,
        ord_db,
        counter_db,
        ORD_COUNTER_DOCUMENT_KEY,
        &value_key,
    )
    .map(DocumentOrd)
}

fn ensure_named_ord_in_txn(
    wtxn: &mut RwTxn,
    value_db: Database<Str, Bytes>,
    ord_db: Database<Bytes, Bytes>,
    counter_db: Database<Bytes, Bytes>,
    counter_key: &[u8],
    value_key: &str,
) -> Result<u64, StoreError> {
    if let Some(existing) = value_db
        .get(&*wtxn, value_key)
        .map_err(|error| StoreError::Query(error.to_string()))?
        .and_then(decode_u64)
    {
        return Ok(existing);
    }
    let next_ord = next_ord_from_counter_in_txn(wtxn, ord_db, counter_db, counter_key)? + 1;
    value_db
        .put_reserved(wtxn, value_key, 8, |reserved| reserved.write_all(&next_ord.to_be_bytes()))
        .map_err(|error| StoreError::Query(error.to_string()))?;
    ord_db
        .put_reserved(wtxn, &next_ord.to_be_bytes(), value_key.len(), |reserved| {
            reserved.write_all(value_key.as_bytes())
        })
        .map_err(|error| StoreError::Query(error.to_string()))?;
    counter_db
        .put_reserved(wtxn, counter_key, 8, |reserved| {
            reserved.write_all(&next_ord.to_be_bytes())
        })
        .map_err(|error| StoreError::Query(error.to_string()))?;
    Ok(next_ord)
}

fn next_ord_from_counter_in_txn(
    wtxn: &mut RwTxn,
    ord_db: Database<Bytes, Bytes>,
    counter_db: Database<Bytes, Bytes>,
    counter_key: &[u8],
) -> Result<u64, StoreError> {
    if let Some(existing) = counter_db
        .get(&*wtxn, counter_key)
        .map_err(|error| StoreError::Query(error.to_string()))?
        .and_then(decode_u64)
    {
        return Ok(existing);
    }
    let next = next_ord_in_txn(wtxn, ord_db)?;
    counter_db
        .put_reserved(wtxn, counter_key, 8, |reserved| {
            reserved.write_all(&next.to_be_bytes())
        })
        .map_err(|error| StoreError::Query(error.to_string()))?;
    Ok(next)
}

fn next_ord_in_txn(wtxn: &mut RwTxn, ord_db: Database<Bytes, Bytes>) -> Result<u64, StoreError> {
    let iter = ord_db
        .rev_iter(&*wtxn)
        .map_err(|error| StoreError::Query(error.to_string()))?;
    for item in iter {
        let (key, _) = item.map_err(|error| StoreError::Query(error.to_string()))?;
        if let Some(value) = decode_u64(key) {
            return Ok(value);
        }
    }
    Ok(0)
}

fn persist_session_archive_in_txn(
    store: &PhoenixLmdbStore,
    wtxn: &mut RwTxn,
    archive: &SessionArchive,
    revision: u64,
) -> Result<(), StoreError> {
    let session_ord = match archive.session_ord {
        Some(session_ord) => session_ord,
        None => ensure_session_ord_in_txn(
            wtxn,
            store.session_by_value,
            store.session_by_ord,
            store.ord_counter,
            &archive.session_id,
        )?,
    };
    let mut archived = archive.clone();
    archived.session_ord = Some(session_ord);
    let key = session_revision_key(session_ord, revision);
    store.write_raw_bytes(
        wtxn,
        store.session_archive,
        &key,
        &encode_value(&archived)?,
    )?;
    store.write_raw_bytes(
        wtxn,
        store.idx_session_latest,
        &session_latest_key(session_ord),
        &revision.to_be_bytes(),
    )?;
    let bundle_header = BundleHeader {
        key: BundleKey {
            kind: BundleKind::SessionArchive,
            scope: archived.session_id.0.clone(),
            entity_key: archived.session_id.0.clone(),
            revision,
        },
        byte_len: 0,
        created_at: archived.updated_at,
    };
    store.write_bytes(
        wtxn,
        store.compat_bundle_headers,
        &bundle_storage_key(&bundle_header.key),
        &encode_value(&bundle_header)?,
    )?;
    for document in &archived.document_refs {
        store.write_raw_bytes(
            wtxn,
            store.idx_session_docs,
            &session_doc_membership_key(session_ord, document.document_ord),
            &document.revision.to_be_bytes(),
        )?;
    }
    Ok(())
}

fn build_prepared_documents_write_batch(
    prepared: &[PreparedDocument],
    touched_scopes: &[DirtyScopeRecord],
) -> Result<LmdbWriteBatch, StoreError> {
    let mut puts = Vec::new();
    for document in prepared {
        let manifest_key = document_revision_key(
            document.manifest.scope_ord,
            document.manifest.document_ord,
            document.manifest.revision,
        );
        puts.push(LmdbPut {
            db: LmdbDbId::DocManifest,
            key: LmdbKey::Bytes(manifest_key),
            value: encode_value(&document.manifest)?,
            flags: PutFlags::empty(),
        });
        puts.push(LmdbPut {
            db: LmdbDbId::IdxDocLatest,
            key: LmdbKey::Bytes(document_latest_key(
                document.manifest.scope_ord,
                document.manifest.document_ord,
            )),
            value: document.manifest.revision.to_be_bytes().to_vec(),
            flags: PutFlags::empty(),
        });
        puts.push(LmdbPut {
            db: LmdbDbId::IdxScopeDocs,
            key: LmdbKey::Bytes(scope_doc_membership_key(
                document.manifest.scope_ord,
                document.manifest.document_ord,
            )),
            value: document.manifest.revision.to_be_bytes().to_vec(),
            flags: PutFlags::empty(),
        });
        let bundle_header = BundleHeader {
            key: BundleKey {
                kind: BundleKind::DocumentArchive,
                scope: document.manifest.scope_key.clone(),
                entity_key: document.manifest.document_id.clone(),
                revision: document.manifest.revision,
            },
            byte_len: document
                .segments
                .iter()
                .map(|segment| segment.payload.len())
                .sum::<usize>(),
            created_at: document.manifest.created_at,
        };
        puts.push(LmdbPut {
            db: LmdbDbId::CompatBundleHeader,
            key: LmdbKey::Str(bundle_storage_key(&bundle_header.key)),
            value: encode_value(&bundle_header)?,
            flags: PutFlags::empty(),
        });
        for segment in &document.segments {
            puts.push(LmdbPut {
                db: LmdbDbId::DocSegment,
                key: LmdbKey::Bytes(document_segment_key(
                    document.manifest.scope_ord,
                    document.manifest.document_ord,
                    document.manifest.revision,
                    segment.header.kind(),
                    segment.header.ordinal,
                )),
                value: encode_value(segment)?,
                flags: PutFlags::empty(),
            });
        }
    }
    for dirty_scope in touched_scopes {
        puts.push(LmdbPut {
            db: LmdbDbId::IdxScopeDirty,
            key: LmdbKey::Bytes(scope_dirty_key(dirty_scope.scope_ord)),
            value: encode_value(dirty_scope)?,
            flags: PutFlags::empty(),
        });
    }
    Ok(LmdbWriteBatch { puts })
}

fn refresh_dirty_scope_records_in_txn(
    store: &PhoenixLmdbStore,
    wtxn: &mut RwTxn,
    updated_at: i64,
) -> Result<(), StoreError> {
    store
        .idx_scope_dirty
        .clear(wtxn)
        .map_err(|error| StoreError::Query(error.to_string()))?;
    let iter = store
        .idx_scope_docs
        .iter(&*wtxn)
        .map_err(|error| StoreError::Query(error.to_string()))?;
    let mut grouped = BTreeMap::<u64, BTreeSet<u64>>::new();
    for item in iter {
        let (key, _) = item.map_err(|error| StoreError::Query(error.to_string()))?;
        let (scope_ord, document_ord) = decode_scope_doc_membership_key(key)?;
        grouped
            .entry(scope_ord.0)
            .or_default()
            .insert(document_ord.0);
    }
    for (scope_ord_value, document_ords) in grouped {
        let scope_ord = ScopeOrd(scope_ord_value);
        let scope_key_bytes = store
            .scope_by_ord
            .get(&*wtxn, &scope_ord.0.to_be_bytes())
            .map_err(|error| StoreError::Query(error.to_string()))?
            .ok_or_else(|| {
                StoreError::Query(format!("missing scope key for ordinal {}", scope_ord.0))
            })?;
        let scope_key = String::from_utf8(scope_key_bytes.to_vec())
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let record = DirtyScopeRecord {
            scope: parse_scope_key(&scope_key),
            scope_key,
            scope_ord,
            document_ords: document_ords.into_iter().map(DocumentOrd).collect(),
            updated_at,
        };
        store.write_raw_bytes(
            wtxn,
            store.idx_scope_dirty,
            &scope_dirty_key(scope_ord),
            &encode_value(&record)?,
        )?;
    }
    Ok(())
}

fn parse_scope_key(scope_key: &str) -> ScopeKey {
    let mut parts = scope_key.split("::");
    ScopeKey {
        world_id: parse_scope_component(parts.next()),
        narrative_id: parse_scope_component(parts.next()),
        folder_id: parse_scope_component(parts.next()),
        folder_path: parse_scope_component(parts.next()),
    }
}

fn parse_scope_component(value: Option<&str>) -> Option<String> {
    value
        .filter(|value| *value != "__global__")
        .map(str::to_owned)
}

fn document_value_key(scope_key: &str, document_id: &str) -> String {
    format!("{scope_key}{SEP}{document_id}")
}

fn document_revision_key(scope_ord: ScopeOrd, document_ord: DocumentOrd, revision: u64) -> Vec<u8> {
    [
        scope_ord.0.to_be_bytes().as_slice(),
        document_ord.0.to_be_bytes().as_slice(),
        revision.to_be_bytes().as_slice(),
    ]
    .concat()
}

fn document_segment_prefix(scope_ord: ScopeOrd, document_ord: DocumentOrd, revision: u64) -> Vec<u8> {
    document_revision_key(scope_ord, document_ord, revision)
}

fn document_segment_key(
    scope_ord: ScopeOrd,
    document_ord: DocumentOrd,
    revision: u64,
    kind: DocumentSegmentKind,
    ordinal: u32,
) -> Vec<u8> {
    [
        document_revision_key(scope_ord, document_ord, revision).as_slice(),
        &[kind.as_u8()],
        &ordinal.to_be_bytes(),
    ]
    .concat()
}

fn document_latest_key(scope_ord: ScopeOrd, document_ord: DocumentOrd) -> Vec<u8> {
    [scope_ord.0.to_be_bytes().as_slice(), document_ord.0.to_be_bytes().as_slice()].concat()
}

fn decode_document_latest_key(bytes: &[u8]) -> Result<(ScopeOrd, DocumentOrd), StoreError> {
    if bytes.len() != 16 {
        return Err(StoreError::Query("invalid latest document key".to_owned()));
    }
    Ok((
        ScopeOrd(decode_u64(&bytes[0..8]).unwrap_or_default()),
        DocumentOrd(decode_u64(&bytes[8..16]).unwrap_or_default()),
    ))
}

fn session_revision_key(session_ord: SessionOrd, revision: u64) -> Vec<u8> {
    [session_ord.0.to_be_bytes().as_slice(), revision.to_be_bytes().as_slice()].concat()
}

fn session_latest_key(session_ord: SessionOrd) -> Vec<u8> {
    session_ord.0.to_be_bytes().to_vec()
}

fn session_doc_membership_key(session_ord: SessionOrd, document_ord: DocumentOrd) -> Vec<u8> {
    [session_ord.0.to_be_bytes().as_slice(), document_ord.0.to_be_bytes().as_slice()].concat()
}

fn scope_membership_prefix(scope_ord: ScopeOrd) -> Vec<u8> {
    scope_ord.0.to_be_bytes().to_vec()
}

fn scope_doc_membership_key(scope_ord: ScopeOrd, document_ord: DocumentOrd) -> Vec<u8> {
    [scope_ord.0.to_be_bytes().as_slice(), document_ord.0.to_be_bytes().as_slice()].concat()
}

fn decode_scope_doc_membership_key(bytes: &[u8]) -> Result<(ScopeOrd, DocumentOrd), StoreError> {
    decode_document_latest_key(bytes)
}

fn scope_dirty_key(scope_ord: ScopeOrd) -> Vec<u8> {
    scope_ord.0.to_be_bytes().to_vec()
}

fn scope_sidecar_key(scope_ord: ScopeOrd) -> Vec<u8> {
    scope_ord.0.to_be_bytes().to_vec()
}

fn ann_index_prefix(index: &AnnIndexKey) -> Vec<u8> {
    let kind = index.kind.as_deref().unwrap_or_default().as_bytes().to_vec();
    [
        index.scope_ord.0.to_be_bytes().as_slice(),
        &[ann_family_tag(index.family)],
        &(kind.len() as u16).to_be_bytes(),
        kind.as_slice(),
    ]
    .concat()
}

fn ann_generation_key(index: &AnnIndexKey, generation: AnnGenerationId) -> Vec<u8> {
    [ann_index_prefix(index).as_slice(), generation.0.to_be_bytes().as_slice()].concat()
}

fn ann_generation_ord_key(index: &AnnIndexKey, generation: AnnGenerationId, ordinal: u32) -> Vec<u8> {
    [ann_generation_key(index, generation).as_slice(), ordinal.to_be_bytes().as_slice()].concat()
}

fn ann_generation_id_key(index: &AnnIndexKey, generation: AnnGenerationId, stable_id: &str) -> Vec<u8> {
    [ann_generation_key(index, generation).as_slice(), stable_id.as_bytes()].concat()
}

fn ann_source_entry_key(
    scope_ord: ScopeOrd,
    family: AnnIndexFamily,
    kind: Option<&str>,
    stable_id: &str,
) -> Vec<u8> {
    let index = AnnIndexKey {
        scope_ord,
        family,
        kind: kind.map(str::to_owned),
    };
    [ann_index_prefix(&index).as_slice(), stable_id.as_bytes()].concat()
}

fn ann_dirty_key(index: &AnnIndexKey) -> Vec<u8> {
    ann_index_prefix(index)
}

fn decode_ann_generation_ord_key(key: &[u8], generation_key: &[u8]) -> Result<u32, StoreError> {
    if key.len() != generation_key.len() + 4 || !key.starts_with(generation_key) {
        return Err(StoreError::Query("invalid ANN ordinal key".to_owned()));
    }
    let mut buffer = [0u8; 4];
    buffer.copy_from_slice(&key[generation_key.len()..]);
    Ok(u32::from_be_bytes(buffer))
}

fn ann_family_tag(family: AnnIndexFamily) -> u8 {
    match family {
        AnnIndexFamily::Document => 1,
        AnnIndexFamily::Leaf => 2,
        AnnIndexFamily::NodePrototype => 3,
    }
}

fn decode_segment_payload<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, StoreError> {
    let payload = lz4_flex::decompress_size_prepended(bytes)
        .map_err(|error| StoreError::Query(error.to_string()))?;
    decode_value(&payload)
}

fn alias_entries_to_map(
    entries: &[AliasEntry],
) -> BTreeMap<String, BTreeMap<(String, String), usize>> {
    let mut merged = BTreeMap::<String, BTreeMap<(String, String), usize>>::new();
    merge_alias_entries(&mut merged, entries);
    merged
}

fn merge_alias_entries(
    merged: &mut BTreeMap<String, BTreeMap<(String, String), usize>>,
    entries: &[AliasEntry],
) {
    for entry in entries {
        let postings = merged.entry(entry.normalized.clone()).or_default();
        for posting in &entry.postings {
            postings
                .entry((posting.entity_id.clone(), posting.document_id.clone()))
                .and_modify(|count| *count += posting.mention_count)
                .or_insert(posting.mention_count);
        }
    }
}

fn alias_entries_from_map(
    alias_entries: BTreeMap<String, BTreeMap<(String, String), usize>>,
) -> Vec<AliasEntry> {
    alias_entries
        .into_iter()
        .map(|(normalized, postings)| AliasEntry {
            normalized,
            postings: postings
                .into_iter()
                .map(|((entity_id, document_id), mention_count)| AliasPosting {
                    entity_id,
                    document_id,
                    mention_count,
                })
                .collect(),
        })
        .collect()
}

fn filter_alias_entries(entries: &[AliasEntry], excluded_document_ids: &BTreeSet<String>) -> Vec<AliasEntry> {
    entries
        .iter()
        .filter_map(|entry| {
            let postings = entry
                .postings
                .iter()
                .filter(|posting| !excluded_document_ids.contains(&posting.document_id))
                .cloned()
                .collect::<Vec<_>>();
            (!postings.is_empty()).then(|| AliasEntry {
                normalized: entry.normalized.clone(),
                postings,
            })
        })
        .collect()
}

fn entity_count_from_alias_entries(entries: &[AliasEntry]) -> usize {
    entries
        .iter()
        .flat_map(|entry| entry.postings.iter().map(|posting| posting.entity_id.as_str()))
        .collect::<BTreeSet<_>>()
        .len()
}

fn clear_relation_in_txn(
    wtxn: &mut RwTxn,
    db: Database<Str, Bytes>,
    relation: &str,
) -> Result<(), StoreError> {
    ensure_native_relation_supported(relation)?;
    let mut keys = Vec::new();
    {
        let iter = db
            .prefix_iter(&*wtxn, &relation_prefix(relation))
            .map_err(|error| StoreError::Query(error.to_string()))?;
        for item in iter {
            let (key, _) = item.map_err(|error| StoreError::Query(error.to_string()))?;
            keys.push(key.to_owned());
        }
    }
    for key in keys {
        db.delete(wtxn, &key)
            .map_err(|error| StoreError::Query(error.to_string()))?;
    }
    Ok(())
}

fn relation_row_storage_key(relation: &str, row: &Value) -> Result<String, StoreError> {
    ensure_native_relation_supported(relation)?;
    let object = row.as_object().ok_or(StoreError::InvalidRow)?;
    let key_values = relation_spec(relation)?
        .key_columns()
        .map(|column| {
            object
                .get(column.name)
                .cloned()
                .ok_or_else(|| StoreError::MissingColumn {
                    relation: relation.to_owned(),
                    column: column.name.to_owned(),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let encoded = serde_json::to_string(&key_values)
        .map_err(|error| StoreError::Query(error.to_string()))?;
    Ok(format!("{}{SEP}{encoded}", relation))
}

fn relation_prefix(relation: &str) -> String {
    format!("{relation}{SEP}")
}

fn ensure_native_relation_supported(relation: &str) -> Result<(), StoreError> {
    if NATIVE_COVERED_RELATIONS.contains(&relation) {
        Ok(())
    } else {
        Err(StoreError::UnknownRelation(relation.to_owned()))
    }
}

fn bundle_storage_key(key: &BundleKey) -> String {
    format!(
        "{}{SEP}{}{SEP}{}{SEP}{:020}",
        bundle_kind_tag(key.kind),
        key.scope,
        key.entity_key,
        key.revision
    )
}

fn bundle_prefix(kind: BundleKind, scope: Option<&str>) -> String {
    match scope {
        Some(scope) => format!("{}{SEP}{}{SEP}", bundle_kind_tag(kind), scope),
        None => format!("{}{SEP}", bundle_kind_tag(kind)),
    }
}

fn bundle_kind_tag(kind: BundleKind) -> &'static str {
    match kind {
        BundleKind::DocumentArchive => "document",
        BundleKind::SessionArchive => "session",
        BundleKind::ScopeLexSidecar => "scope_lex",
    }
}

fn journal_entry_key(generation: u64, created_at: i64) -> String {
    let ordinal = JOURNAL_ENTRY_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{generation:020}{SEP}{created_at:020}{SEP}{ordinal:020}")
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

fn native_progress_enabled() -> bool {
    std::env::var_os("PHOENIX_PERF_PROGRESS").is_some()
        || std::env::var_os("PHOENIX_INGEST_PROGRESS").is_some()
}

fn native_relation_snapshot_names(partition: SnapshotPartition) -> Vec<&'static str> {
    match partition {
        SnapshotPartition::All | SnapshotPartition::Content => {
            NATIVE_AUTHORITATIVE_RELATIONS.to_vec()
        }
        SnapshotPartition::Derived => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_kernel::{
        KernelBiTemporal, KernelEdge, KernelEdgeType, KernelEntityFacet, KernelGraphLayer,
        KernelMutationScope, KernelProvenance, KernelRelationClass, KernelVertex,
        KernelVertexClass, KernelVertexId,
    };
    use phoenix_store_native::{
        PhoenixArchiveStoreV2 as LegacyArchiveStore, PhoenixBundleStoreV2 as LegacyBundleStore,
    };
    use phoenix_semantic_v2::DocumentSegmentHeader;
    use phoenix_types::{DocumentId, IngestDocumentSummary, SessionDocumentState};
    use serde_json::json;

    fn temp_store(name: &str) -> PhoenixLmdbStore {
        let path = std::env::temp_dir().join(format!(
            "phoenix-lmdb-test-{name}-{}-{}",
            std::process::id(),
            now_ms()
        ));
        PhoenixLmdbStore::open(path).expect("open store")
    }

    fn test_document(document_id: &str, scope: &ScopeKey) -> IngestDocument {
        IngestDocument {
            document_id: DocumentId(document_id.to_owned()),
            note_id: None,
            title: document_id.to_owned(),
            text: format!("{document_id} text"),
            scope: scope.clone(),
        }
    }

    fn lexical_prepared_document(
        assignment: &DocumentOrdinalAssignment,
        alias: &str,
        entity_id: &str,
        created_at: i64,
    ) -> PreparedDocument {
        let span = IndexedSpan {
            span_id: format!("span::{}::{}", assignment.document_id, assignment.revision),
            note_id: None,
            document_id: Some(DocumentId(assignment.document_id.clone())),
            scope: assignment.scope.clone(),
            fields: Vec::new(),
        };
        let lexical = LexicalPostingsSegment {
            spans: vec![span],
            alias_entries: vec![AliasEntry {
                normalized: alias.to_owned(),
                postings: vec![AliasPosting {
                    entity_id: entity_id.to_owned(),
                    document_id: assignment.document_id.clone(),
                    mention_count: 1,
                }],
            }],
        };
        let payload = rmp_serde::to_vec_named(&lexical).expect("encode lexical");
        let compressed = lz4_flex::compress_prepend_size(&payload);
        let header = DocumentSegmentHeader::new(
            DocumentSegmentKind::LexicalPostings,
            0,
            (lexical.spans.len() + lexical.alias_entries.len()) as u32,
            payload.len(),
            compressed.len(),
        );
        let manifest = DocumentManifest {
            document_id: assignment.document_id.clone(),
            document_version_id: phoenix_semantic_v2::DocumentVersionId(format!(
                "{}::{}",
                assignment.document_id, assignment.revision
            )),
            note_id: None,
            scope: assignment.scope.clone(),
            scope_key: assignment.scope_key.clone(),
            scope_ord: assignment.scope_ord,
            document_ord: assignment.document_ord,
            revision: assignment.revision,
            title: assignment.document_id.clone(),
            text_len: assignment.document_id.len(),
            fingerprint: format!("{}::{}", assignment.document_id, assignment.revision),
            config_hash: "test-config".to_owned(),
            session_id: None,
            document_summary: IngestDocumentSummary {
                document_id: DocumentId(assignment.document_id.clone()),
                note_id: None,
                chapter_count: 0,
                boundary_count: 0,
                parent_count: 0,
                leaf_count: 1,
                entity_count: 1,
                edge_count: 0,
                has_front_matter_chapter: false,
                has_front_matter_boundary: false,
            },
            session_document: SessionDocumentState {
                document_id: DocumentId(assignment.document_id.clone()),
                note_id: None,
                chapter_count: 0,
                boundary_count: 0,
                chapter_titles: Vec::new(),
                boundary_labels: Vec::new(),
                parent_count: 0,
                leaf_count: 1,
                entity_count: 1,
                discovery_count: 0,
                has_front_matter_chapter: false,
                has_front_matter_boundary: false,
                updated_at: created_at,
            },
            discovery_count: 0,
            mention_count: 1,
            span_count: 1,
            entity_count: 1,
            alias_count: 1,
            graph_edge_count: 0,
            graph_vertex_count: 0,
            segment_refs: vec![phoenix_semantic_v2::DocumentSegmentRef {
                kind: DocumentSegmentKind::LexicalPostings,
                ordinal: 0,
                row_count: (lexical.spans.len() + lexical.alias_entries.len()) as u32,
                byte_len: compressed.len() as u32,
                uncompressed_len: payload.len() as u32,
            }],
            created_at,
            archive_version: 1,
        };
        PreparedDocument {
            assignment: assignment.clone(),
            manifest,
            segments: vec![PreparedDocumentSegment {
                header,
                payload: compressed,
            }],
            kernel_batch: KernelMutationBatch::default(),
        }
    }

    fn semantic_test_vector(primary_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; SEMANTIC_VECTOR_DIM];
        if primary_index < values.len() {
            values[primary_index] = 1.0;
        }
        values
    }

    #[test]
    fn bundle_round_trip() {
        let store = temp_store("bundle");
        let header = BundleHeader {
            key: BundleKey {
                kind: BundleKind::DocumentArchive,
                scope: "__global__::__global__::__global__::__global__".to_owned(),
                entity_key: "doc-1".to_owned(),
                revision: 1,
            },
            byte_len: 3,
            created_at: 10,
        };
        LegacyBundleStore::put_bundle(&store, &header, b"hey").expect("put bundle");
        let loaded = LegacyBundleStore::get_bundle_header(&store, &header.key)
            .expect("header")
            .expect("present");
        assert_eq!(loaded.key.entity_key, "doc-1");
        assert_eq!(
            LegacyBundleStore::get_bundle(&store, &header.key).expect("payload"),
            Some(b"hey".to_vec())
        );
    }

    #[test]
    fn kernel_journal_round_trip() {
        let store = temp_store("kernel");
        let batch = KernelMutationBatch {
            layer: KernelGraphLayer::Asserted,
            scope: KernelMutationScope::Full,
            recorded_at: None,
            vertices: vec![KernelVertex {
                id: KernelVertexId("vertex-1".to_owned()),
                kind: "document".to_owned(),
                ..Default::default()
            }],
            edges: Vec::new(),
        };
        store
            .append_kernel_batch(2, "rev-2", &batch, 22)
            .expect("append batch");
        store
            .append_kernel_commit_marker(2, "rev-2", "commit-2", 23)
            .expect("append commit");
        let entries = store.load_kernel_journal_after(0).expect("entries");
        assert_eq!(entries.len(), 2);
        assert_eq!(
            store.kernel_generation_for_commit("commit-2").expect("commit generation"),
            Some(2)
        );
    }

    #[test]
    fn kernel_checkpoint_round_trip_preserves_temporal_and_identity_fields() {
        let store = temp_store("kernel-checkpoint");
        let snapshot = KernelGraphSnapshot {
            vertices: vec![KernelVertex {
                id: KernelVertexId("entity::port".to_owned()),
                kind: "entity".to_owned(),
                class: KernelVertexClass::Entity,
                value: json!({"name":"Port Authority"}),
                attributes: json!({"source":"checkpoint"}),
                temporal: KernelBiTemporal {
                    valid_from: Some(10),
                    valid_to: Some(20),
                    recorded_at: Some(30),
                    expired_at: Some(40),
                },
                provenance: KernelProvenance {
                    resolver: Some("resolver-x".to_owned()),
                    source: Some("source-a".to_owned()),
                    confidence: Some(0.82),
                    evidence_refs: vec!["ev-1".to_owned()],
                },
                entity_id: Some("port-authority".to_owned()),
                entity_facet: Some(KernelEntityFacet {
                    canonical_entity_id: Some("port-authority".to_owned()),
                    surface: Some("Port Authority".to_owned()),
                    entity_kind: Some("organization".to_owned()),
                }),
                ..KernelVertex::default()
            }],
            asserted_edges: vec![KernelEdge {
                source_id: KernelVertexId("mention::1".to_owned()),
                target_id: KernelVertexId("entity::port".to_owned()),
                edge_type: KernelEdgeType("resolved_to".to_owned()),
                relation_class: KernelRelationClass::Resolution,
                attributes: json!({}),
                temporal: KernelBiTemporal {
                    valid_from: Some(12),
                    valid_to: None,
                    recorded_at: Some(30),
                    expired_at: None,
                },
                provenance: KernelProvenance {
                    resolver: Some("resolver-x".to_owned()),
                    source: Some("source-a".to_owned()),
                    confidence: Some(0.91),
                    evidence_refs: vec!["ev-2".to_owned()],
                },
                ..KernelEdge::default()
            }],
            candidate_edges: Vec::new(),
        };
        let written = store
            .write_kernel_checkpoint(3, "rev-3", &snapshot)
            .expect("write checkpoint");
        let loaded = store
            .load_kernel_checkpoint()
            .expect("checkpoint load")
            .expect("checkpoint present");
        assert_eq!(loaded, written);
        assert_eq!(loaded.snapshot.vertices[0].temporal.valid_from, Some(10));
        assert_eq!(
            loaded.snapshot.vertices[0]
                .entity_facet
                .as_ref()
                .and_then(|facet| facet.surface.as_deref()),
            Some("Port Authority")
        );
        assert_eq!(
            loaded.snapshot.asserted_edges[0]
                .provenance
                .confidence,
            Some(0.91)
        );
    }

    #[test]
    fn sidecar_rebuild_uses_lexical_postings_segments_only() {
        let store = temp_store("lexical-sidecar");
        let scope = ScopeKey::default();
        let created_at = now_ms();
        let documents = vec![test_document("doc-a", &scope), test_document("doc-b", &scope)];
        let context = LegacyArchiveStore::prepare_ingest_context(&store, None, &documents, 0)
            .expect("prepare ingest context");
        let prepared = vec![
            lexical_prepared_document(&context.assignments[0], "ryan", "entity-ryan", created_at),
            lexical_prepared_document(&context.assignments[1], "len", "entity-len", created_at),
        ];
        let dirty = DirtyScopeRecord {
            scope: scope.clone(),
            scope_key: scope_storage_key(&scope),
            scope_ord: context.assignments[0].scope_ord,
            document_ords: context.assignments.iter().map(|assignment| assignment.document_ord).collect(),
            updated_at: created_at,
        };
        LegacyArchiveStore::persist_prepared_documents(&store, &prepared, None, &[dirty], created_at)
            .expect("persist prepared documents");

        assert_eq!(
            LegacyArchiveStore::rebuild_dirty_scope_sidecars(&store, created_at + 1)
                .expect("rebuild dirty sidecars"),
            1
        );

        let sidecar = LegacyArchiveStore::load_scope_sidecar(&store, &scope)
            .expect("load sidecar")
            .expect("sidecar");
        assert_eq!(sidecar.spans.len(), 2);
        assert_eq!(sidecar.document_ids.len(), 2);
        assert_eq!(
            LegacyArchiveStore::lookup_alias_postings(&store, &scope, "ryan")
                .expect("ryan")
                .len(),
            1
        );
        assert_eq!(
            LegacyArchiveStore::lookup_alias_postings(&store, &scope, "len")
                .expect("len")
                .len(),
            1
        );
    }

    #[test]
    fn dirty_scope_fallback_merges_persisted_sidecar_with_dirty_docs() {
        let store = temp_store("dirty-fallback");
        let scope = ScopeKey::default();
        let created_at = now_ms();
        let documents = vec![test_document("doc-a", &scope), test_document("doc-b", &scope)];
        let context = LegacyArchiveStore::prepare_ingest_context(&store, None, &documents, 0)
            .expect("prepare ingest context");
        let prepared = vec![
            lexical_prepared_document(&context.assignments[0], "alpha", "entity-alpha", created_at),
            lexical_prepared_document(&context.assignments[1], "beta", "entity-beta", created_at),
        ];
        let dirty = DirtyScopeRecord {
            scope: scope.clone(),
            scope_key: scope_storage_key(&scope),
            scope_ord: context.assignments[0].scope_ord,
            document_ords: context.assignments.iter().map(|assignment| assignment.document_ord).collect(),
            updated_at: created_at,
        };
        LegacyArchiveStore::persist_prepared_documents(&store, &prepared, None, &[dirty], created_at)
            .expect("persist prepared documents");
        LegacyArchiveStore::rebuild_dirty_scope_sidecars(&store, created_at + 1)
            .expect("rebuild sidecars");

        let updated_context = LegacyArchiveStore::prepare_ingest_context(
            &store,
            None,
            &[test_document("doc-b", &scope)],
            1,
        )
        .expect("prepare second revision");
        let updated_prepared = vec![lexical_prepared_document(
            &updated_context.assignments[0],
            "gamma",
            "entity-gamma",
            created_at + 2,
        )];
        let dirty_update = DirtyScopeRecord {
            scope: scope.clone(),
            scope_key: scope_storage_key(&scope),
            scope_ord: updated_context.assignments[0].scope_ord,
            document_ords: vec![updated_context.assignments[0].document_ord],
            updated_at: created_at + 2,
        };
        LegacyArchiveStore::persist_prepared_documents(
            &store,
            &updated_prepared,
            None,
            &[dirty_update],
            created_at + 2,
        )
            .expect("persist second revision");

        assert_eq!(
            LegacyArchiveStore::list_dirty_scopes(&store)
                .expect("dirty scopes")
                .len(),
            1
        );
        let spans = LegacyArchiveStore::load_lex_spans(&store, Some(&scope)).expect("lex spans");
        assert_eq!(spans.len(), 2);
        assert_eq!(
            LegacyArchiveStore::lookup_alias_postings(&store, &scope, "alpha")
                .expect("alpha")
                .len(),
            1
        );
        assert!(
            LegacyArchiveStore::lookup_alias_postings(&store, &scope, "beta")
                .expect("beta")
                .is_empty()
        );
        assert_eq!(
            LegacyArchiveStore::lookup_alias_postings(&store, &scope, "gamma")
                .expect("gamma")
                .len(),
            1
        );

        LegacyArchiveStore::rebuild_dirty_scope_sidecars(&store, created_at + 3)
            .expect("rebuild updated sidecar");
        assert!(
            LegacyArchiveStore::list_dirty_scopes(&store)
                .expect("dirty scopes")
                .is_empty()
        );
        let sidecar = LegacyArchiveStore::load_scope_sidecar(&store, &scope)
            .expect("load sidecar")
            .expect("sidecar");
        assert_eq!(sidecar.spans.len(), 2);
        assert!(
            LegacyArchiveStore::lookup_alias_postings(&store, &scope, "beta")
                .expect("beta")
                .is_empty()
        );
        assert_eq!(
            LegacyArchiveStore::lookup_alias_postings(&store, &scope, "gamma")
                .expect("gamma")
                .len(),
            1
        );
    }

    #[test]
    fn snapshot_all_excludes_sidecar_caches() {
        let store = temp_store("snapshot");
        let scope = ScopeKey::default();
        let created_at = now_ms();
        let documents = vec![test_document("doc-a", &scope)];
        let context = LegacyArchiveStore::prepare_ingest_context(&store, None, &documents, 0)
            .expect("prepare ingest context");
        let prepared = vec![lexical_prepared_document(
            &context.assignments[0],
            "alpha",
            "entity-alpha",
            created_at,
        )];
        let dirty = DirtyScopeRecord {
            scope: scope.clone(),
            scope_key: scope_storage_key(&scope),
            scope_ord: context.assignments[0].scope_ord,
            document_ords: vec![context.assignments[0].document_ord],
            updated_at: created_at,
        };
        LegacyArchiveStore::persist_prepared_documents(&store, &prepared, None, &[dirty], created_at)
            .expect("persist prepared documents");
        LegacyArchiveStore::rebuild_dirty_scope_sidecars(&store, created_at + 1)
            .expect("rebuild sidecars");

        let all_bytes = store
            .export_native_snapshot(store.schema_version(), SnapshotPartition::All)
            .expect("export all");
        let derived_bytes = store
            .export_native_snapshot(store.schema_version(), SnapshotPartition::Derived)
            .expect("export derived");
        let all_envelope: NativeSnapshotEnvelope =
            decode_value(&all_bytes[NATIVE_SNAPSHOT_MAGIC.len()..]).expect("decode all");
        let derived_envelope: NativeSnapshotEnvelope =
            decode_value(&derived_bytes[NATIVE_SNAPSHOT_MAGIC.len()..]).expect("decode derived");

        assert!(all_envelope.scope_sidecar_meta.is_empty());
        assert!(all_envelope.scope_sidecar_fst.is_empty());
        assert!(all_envelope.scope_sidecar_postings.is_empty());
        assert!(all_envelope.doc_manifest.len() >= 1);
        assert!(!derived_envelope.scope_sidecar_meta.is_empty());
        assert!(!derived_envelope.scope_sidecar_postings.is_empty());
        assert!(derived_envelope.doc_manifest.is_empty());

        let mut legacy_like = all_envelope.clone();
        legacy_like.scope_sidecar_meta =
            dump_raw_db(&store.env, store.scope_sidecar_meta).expect("dump meta");
        legacy_like.scope_sidecar_fst =
            dump_raw_db(&store.env, store.scope_sidecar_fst).expect("dump fst");
        legacy_like.scope_sidecar_postings =
            dump_raw_db(&store.env, store.scope_sidecar_postings).expect("dump postings");
        let mut legacy_like_bytes = Vec::from(&NATIVE_SNAPSHOT_MAGIC[..]);
        legacy_like_bytes.extend_from_slice(&encode_value(&legacy_like).expect("encode legacy-like"));

        assert!(all_bytes.len() < legacy_like_bytes.len());
    }

    #[test]
    fn ann_generation_swap_and_query_roundtrip() {
        let store = temp_store("ann-generation");
        let scope = ScopeKey::default();

        store
            .upsert_semantic_leaf_vectors(&[
                NativeSemanticLeafVectorRecord {
                    scope: scope.clone(),
                    span_id: "span-a".to_owned(),
                    document_id: "doc-a".to_owned(),
                    values: semantic_test_vector(0),
                    updated_at: 1,
                },
                NativeSemanticLeafVectorRecord {
                    scope: scope.clone(),
                    span_id: "span-b".to_owned(),
                    document_id: "doc-b".to_owned(),
                    values: semantic_test_vector(1),
                    updated_at: 1,
                },
            ])
            .expect("upsert leaf vectors");
        store
            .upsert_semantic_document_vectors_native(&[
                NativeSemanticDocumentVectorRecord {
                    scope: scope.clone(),
                    document_id: "doc-a".to_owned(),
                    values: semantic_test_vector(0),
                    leaf_count: 1,
                    evidence_refs: vec!["span:span-a".to_owned()],
                    updated_at: 1,
                },
                NativeSemanticDocumentVectorRecord {
                    scope: scope.clone(),
                    document_id: "doc-b".to_owned(),
                    values: semantic_test_vector(1),
                    leaf_count: 1,
                    evidence_refs: vec!["span:span-b".to_owned()],
                    updated_at: 1,
                },
            ])
            .expect("upsert document vectors");
        store
            .upsert_semantic_node_vectors_native(&[
                NativeSemanticNodeVectorRecord {
                    scope: scope.clone(),
                    node_id: "entity::a".to_owned(),
                    node_kind: "entity".to_owned(),
                    document_id: Some("doc-a".to_owned()),
                    narrative_id: None,
                    folder_id: None,
                    values: semantic_test_vector(0),
                    evidence_refs: vec!["graph_vertex:entity::a".to_owned()],
                    updated_at: 1,
                },
                NativeSemanticNodeVectorRecord {
                    scope: scope.clone(),
                    node_id: "entity::b".to_owned(),
                    node_kind: "entity".to_owned(),
                    document_id: Some("doc-b".to_owned()),
                    narrative_id: None,
                    folder_id: None,
                    values: semantic_test_vector(1),
                    evidence_refs: vec!["graph_vertex:entity::b".to_owned()],
                    updated_at: 1,
                },
            ])
            .expect("upsert node vectors");

        let scope_ord = store
            .lookup_scope_ord(&scope_storage_key(&scope))
            .expect("scope ord")
            .expect("present");
        let document_index = AnnIndexKey {
            scope_ord,
            family: AnnIndexFamily::Document,
            kind: None,
        };
        let head_key = ann_index_prefix(&document_index);
        let dirty_key = ann_dirty_key(&document_index);
        let before_query = {
            let rtxn = store.env.read_txn().expect("read txn");
            (
                store.ann_head.get(&rtxn, &head_key).expect("head row").is_some(),
                store.ann_dirty.get(&rtxn, &dirty_key).expect("dirty row").is_some(),
            )
        };
        assert!(!before_query.0);
        assert!(before_query.1);

        let doc_hits = store
            .query_semantic_documents(&semantic_test_vector(0), &scope, 2, 8)
            .expect("document query");
        assert_eq!(doc_hits.first().map(|hit| hit.document_id.as_str()), Some("doc-a"));

        let generation_one = {
            let rtxn = store.env.read_txn().expect("read txn");
            store
                .ann_head
                .get(&rtxn, &head_key)
                .expect("head row")
                .and_then(decode_u64)
                .expect("generation one")
        };
        assert_eq!(generation_one, 1);

        let leaf_hits = store
            .query_semantic_neighbors_in_documents(
                &semantic_test_vector(0),
                &scope,
                &["doc-a".to_owned()],
                2,
                8,
            )
            .expect("leaf query");
        assert_eq!(leaf_hits.first().map(|hit| hit.span_id.as_str()), Some("span-a"));

        let node_hits = store
            .query_semantic_node_neighbors(&semantic_test_vector(0), &scope, "entity", None, 2, 8)
            .expect("node query");
        assert_eq!(node_hits.first().map(|hit| hit.node_id.as_str()), Some("entity::a"));

        store
            .upsert_semantic_document_vectors_native(&[
                NativeSemanticDocumentVectorRecord {
                    scope: scope.clone(),
                    document_id: "doc-a".to_owned(),
                    values: semantic_test_vector(2),
                    leaf_count: 1,
                    evidence_refs: vec!["span:span-a".to_owned()],
                    updated_at: 2,
                },
                NativeSemanticDocumentVectorRecord {
                    scope: scope.clone(),
                    document_id: "doc-b".to_owned(),
                    values: semantic_test_vector(0),
                    leaf_count: 1,
                    evidence_refs: vec!["span:span-b".to_owned()],
                    updated_at: 2,
                },
            ])
            .expect("update document vectors");

        let before_second_query = {
            let rtxn = store.env.read_txn().expect("read txn");
            (
                store
                    .ann_head
                    .get(&rtxn, &head_key)
                    .expect("head row")
                    .and_then(decode_u64)
                    .expect("stale generation"),
                store.ann_dirty.get(&rtxn, &dirty_key).expect("dirty row").is_some(),
            )
        };
        assert_eq!(before_second_query.0, 1);
        assert!(before_second_query.1);

        let updated_hits = store
            .query_semantic_documents(&semantic_test_vector(0), &scope, 2, 8)
            .expect("updated document query");
        assert_eq!(updated_hits.first().map(|hit| hit.document_id.as_str()), Some("doc-b"));

        let generation_two = {
            let rtxn = store.env.read_txn().expect("read txn");
            store
                .ann_head
                .get(&rtxn, &head_key)
                .expect("head row")
                .and_then(decode_u64)
                .expect("generation two")
        };
        assert_eq!(generation_two, 2);
    }

    #[test]
    fn ann_snapshot_roundtrip_preserves_active_generation_queries() {
        let store = temp_store("ann-snapshot-source");
        let scope = ScopeKey::default();
        store
            .upsert_semantic_document_vectors_native(&[
                NativeSemanticDocumentVectorRecord {
                    scope: scope.clone(),
                    document_id: "doc-a".to_owned(),
                    values: semantic_test_vector(0),
                    leaf_count: 1,
                    evidence_refs: vec!["span:span-a".to_owned()],
                    updated_at: 1,
                },
                NativeSemanticDocumentVectorRecord {
                    scope: scope.clone(),
                    document_id: "doc-b".to_owned(),
                    values: semantic_test_vector(1),
                    leaf_count: 1,
                    evidence_refs: vec!["span:span-b".to_owned()],
                    updated_at: 1,
                },
            ])
            .expect("upsert document vectors");

        let snapshot = store
            .export_native_snapshot(store.schema_version(), SnapshotPartition::All)
            .expect("export snapshot");

        let restored = temp_store("ann-snapshot-restored");
        restored
            .import_native_snapshot(&snapshot)
            .expect("import snapshot");

        let hits = restored
            .query_semantic_documents(&semantic_test_vector(0), &scope, 2, 8)
            .expect("restored query");
        assert_eq!(hits.first().map(|hit| hit.document_id.as_str()), Some("doc-a"));
    }
}
