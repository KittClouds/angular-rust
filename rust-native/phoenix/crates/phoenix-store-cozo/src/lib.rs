use std::collections::BTreeMap;
use std::path::PathBuf;

use cozo;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use phoenix_types::{
    EntityCard, EntityId, FolderSchema, NetworkInstance, NetworkMembership, NetworkRelationship,
    RelationCount, ScopeKey, SnapshotDto, StorageMode,
};
use rustc_hash::FxHashMap;
use schema::{
    PhoenixColumnSpec, PhoenixColumnType, PhoenixRelationSpec, ALL_RELATIONS,
    CONTENT_SNAPSHOT_RELATIONS, DERIVED_SNAPSHOT_RELATIONS,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use smallvec::SmallVec;

pub mod schema;

pub const SCHEMA_VERSION: &str = "phoenix.cozo.v3";
pub const SEMANTIC_VECTOR_DIM: usize = 384;
pub const SEMANTIC_MODEL_ID: &str = "MongoDB/mdbr-leaf-ir";
const PUT_ROWS_BATCH_LIMIT: usize = if cfg!(target_arch = "wasm32") {
    256
} else {
    512
};
const SNAPSHOT_MAGIC: [u8; 8] = *b"PXSNAP02";
const SNAPSHOT_VERSION: u16 = 2;
const SNAPSHOT_COMPRESS_THRESHOLD: usize = 4 * 1024;

pub type CompactRow = SmallVec<[cozo::DataValue; 16]>;
type CompactRowKey = SmallVec<[cozo::DataValue; 4]>;

#[derive(Clone, Copy, Debug)]
pub struct CompactRowView<'a> {
    columns: &'a [&'a str],
    row: &'a CompactRow,
}

impl<'a> CompactRowView<'a> {
    pub fn new(columns: &'a [&'a str], row: &'a CompactRow) -> Self {
        Self { columns, row }
    }

    pub fn get_value(&self, column: &str) -> Option<&'a cozo::DataValue> {
        self.columns
            .iter()
            .position(|candidate| *candidate == column)
            .and_then(|index| self.row.get(index))
    }

    pub fn get_str(&self, column: &str) -> Option<&'a str> {
        match self.get_value(column)? {
            cozo::DataValue::Str(value) => Some(value.as_str()),
            _ => None,
        }
    }

    pub fn get_i64(&self, column: &str) -> Option<i64> {
        match self.get_value(column)? {
            cozo::DataValue::Num(cozo::Num::Int(value)) => Some(*value),
            cozo::DataValue::Num(cozo::Num::Float(value)) => Some(*value as i64),
            _ => None,
        }
    }

    pub fn get_u64(&self, column: &str) -> Option<u64> {
        self.get_i64(column)
            .and_then(|value| u64::try_from(value).ok())
    }

    pub fn get_bool(&self, column: &str) -> Option<bool> {
        match self.get_value(column)? {
            cozo::DataValue::Bool(value) => Some(*value),
            _ => None,
        }
    }

    pub fn get_json(&self, column: &str) -> Option<Value> {
        Some(datavalue_to_json_ref(self.get_value(column)?))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SnapshotCodec {
    Raw = 0,
    Lz4 = 1,
}

impl SnapshotCodec {
    fn from_u8(value: u8) -> Result<Self, StoreError> {
        match value {
            0 => Ok(Self::Raw),
            1 => Ok(Self::Lz4),
            _ => Err(StoreError::Snapshot(format!(
                "unsupported snapshot codec: {value}"
            ))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotWireHeader {
    magic: [u8; 8],
    version: u16,
    relation_count: u16,
    created_at: i64,
}

impl SnapshotWireHeader {
    fn encode(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.magic);
        bytes.extend_from_slice(&self.version.to_le_bytes());
        bytes.extend_from_slice(&self.relation_count.to_le_bytes());
        bytes.extend_from_slice(&self.created_at.to_le_bytes());
    }

    fn decode(bytes: &[u8], offset: &mut usize) -> Result<Self, StoreError> {
        let magic = read_array::<8>(bytes, offset)?;
        let version = read_u16(bytes, offset)?;
        let relation_count = read_u16(bytes, offset)?;
        let created_at = read_i64(bytes, offset)?;
        Ok(Self {
            magic,
            version,
            relation_count,
            created_at,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SnapshotRelationBlockHeader {
    relation_id: u16,
    codec: SnapshotCodec,
    row_count: u32,
    encoded_len: u32,
}

impl SnapshotRelationBlockHeader {
    const BYTE_LEN: usize = 2 + 1 + 1 + 4 + 4;

    fn encode(self, bytes: &mut Vec<u8>) {
        bytes.extend_from_slice(&self.relation_id.to_le_bytes());
        bytes.push(self.codec as u8);
        bytes.push(0);
        bytes.extend_from_slice(&self.row_count.to_le_bytes());
        bytes.extend_from_slice(&self.encoded_len.to_le_bytes());
    }

    fn decode(bytes: &[u8], offset: &mut usize) -> Result<Self, StoreError> {
        let relation_id = read_u16(bytes, offset)?;
        let codec = SnapshotCodec::from_u8(read_u8(bytes, offset)?)?;
        let _reserved = read_u8(bytes, offset)?;
        let row_count = read_u32(bytes, offset)?;
        let encoded_len = read_u32(bytes, offset)?;
        Ok(Self {
            relation_id,
            codec,
            row_count,
            encoded_len,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct CompactRelationBuffer {
    relation: &'static str,
    rows: FxHashMap<CompactRowKey, CompactRow>,
}

impl CompactRelationBuffer {
    pub fn new(relation: &'static str) -> Result<Self, StoreError> {
        relation_spec(relation)?;
        Ok(Self {
            relation,
            rows: FxHashMap::default(),
        })
    }

    pub fn insert_value(&mut self, row: Value) -> Result<(), StoreError> {
        let compact = compact_row_from_value(self.relation, &row)?;
        self.insert_row(compact)
    }

    pub fn insert_row(&mut self, row: CompactRow) -> Result<(), StoreError> {
        let key = compact_row_key(self.relation, &row)?;
        self.rows.insert(key, row);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn relation(&self) -> &'static str {
        self.relation
    }

    pub fn rows(&self) -> Vec<CompactRow> {
        self.rows.values().cloned().collect()
    }

    pub fn into_rows(self) -> Vec<CompactRow> {
        self.rows.into_values().collect()
    }

    pub fn drain_all(&mut self) -> Vec<CompactRow> {
        self.rows.drain().map(|(_, row)| row).collect()
    }

    pub fn drain_if_len_ge(&mut self, limit: usize) -> Option<Vec<CompactRow>> {
        (self.rows.len() >= limit).then(|| self.drain_all())
    }

    pub fn json_rows(&self) -> Result<Vec<Value>, StoreError> {
        let spec = relation_spec(self.relation)?;
        Ok(self
            .rows
            .values()
            .map(|row| compact_row_to_json_object(spec, row))
            .collect())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoreConfig {
    pub mode: StorageMode,
    pub path: Option<PathBuf>,
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            mode: StorageMode::CozoMem,
            path: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotEnvelope {
    pub schema_version: String,
    pub relation_count: usize,
    pub created_at: i64,
    pub relations: BTreeMap<String, Vec<Value>>,
    pub checksum: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("unsupported storage mode on this target: {0:?}")]
    UnsupportedMode(StorageMode),
    #[error("sqlite path is required for persistent Cozo storage")]
    MissingPath,
    #[error("unknown relation: {0}")]
    UnknownRelation(String),
    #[error("row must be a JSON object")]
    InvalidRow,
    #[error("missing required column '{column}' in relation '{relation}'")]
    MissingColumn { relation: String, column: String },
    #[error("cozo init failed: {0}")]
    Init(String),
    #[error("cozo schema failed: {0}")]
    Schema(String),
    #[error("cozo query failed: {0}")]
    Query(String),
    #[error("snapshot decode failed: {0}")]
    Snapshot(String),
}

pub struct PhoenixCozoStore {
    db: cozo::DbInstance,
    config: StoreConfig,
    schema_version: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotPartition {
    All,
    Content,
    Derived,
}

impl SnapshotPartition {
    pub fn from_str(value: &str) -> Option<Self> {
        match value {
            "all" => Some(Self::All),
            "content" => Some(Self::Content),
            "derived" => Some(Self::Derived),
            _ => None,
        }
    }

    fn relation_names(self) -> &'static [&'static str] {
        match self {
            Self::All => &[],
            Self::Content => CONTENT_SNAPSHOT_RELATIONS,
            Self::Derived => DERIVED_SNAPSHOT_RELATIONS,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticVectorRow<'a> {
    pub span_id: &'a str,
    pub values: &'a [f32],
    pub model_id: &'a str,
    pub updated_at: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticDocumentVectorRow<'a> {
    pub document_id: &'a str,
    pub values: &'a [f32],
    pub model_id: &'a str,
    pub leaf_count: usize,
    pub evidence_refs: &'a [String],
    pub updated_at: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SemanticNodeVectorRow<'a> {
    pub node_id: &'a str,
    pub node_kind: &'a str,
    pub document_id: Option<&'a str>,
    pub narrative_id: Option<&'a str>,
    pub folder_id: Option<&'a str>,
    pub values: &'a [f32],
    pub model_id: &'a str,
    pub evidence_refs: &'a [String],
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticLeafChunk {
    pub span_id: String,
    pub document_id: String,
    pub text: String,
    pub narrative_id: Option<String>,
    pub folder_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticNeighbor {
    pub span_id: String,
    pub distance: f64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticDocumentNeighbor {
    pub document_id: String,
    pub distance: f64,
    pub leaf_count: usize,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SemanticNodeNeighbor {
    pub node_id: String,
    pub node_kind: String,
    pub distance: f64,
    pub document_id: Option<String>,
    pub narrative_id: Option<String>,
    pub folder_id: Option<String>,
    pub evidence_refs: Vec<String>,
}

impl PhoenixCozoStore {
    pub fn new() -> Result<Self, StoreError> {
        Self::open(StoreConfig::default())
    }

    pub fn open(config: StoreConfig) -> Result<Self, StoreError> {
        let db = open_db(&config)?;
        let store = Self {
            db,
            config,
            schema_version: SCHEMA_VERSION,
        };
        store.init_schema()?;
        Ok(store)
    }

    pub fn config(&self) -> &StoreConfig {
        &self.config
    }

    pub fn schema_version(&self) -> &'static str {
        self.schema_version
    }

    pub fn relation_names(&self) -> Vec<&'static str> {
        ALL_RELATIONS.iter().map(|relation| relation.name).collect()
    }

    pub fn init_schema(&self) -> Result<(), StoreError> {
        for relation in ALL_RELATIONS {
            let script = build_create_relation_script(relation);
            match self
                .db
                .run_script(&script, Default::default(), cozo::ScriptMutability::Mutable)
            {
                Ok(_) => {}
                Err(error) if relation_already_exists(&error.to_string()) => {}
                Err(error) => return Err(StoreError::Schema(error.to_string())),
            }
        }

        self.ensure_semantic_relations()?;

        self.put_row(
            "phoenix_schema_state",
            serde_json::json!({
                "version": self.schema_version,
                "updated_at": now_ms(),
            }),
        )?;

        Ok(())
    }

    pub fn ensure_semantic_relations(&self) -> Result<(), StoreError> {
        self.ensure_semantic_index("semantic_vectors", "vec_idx")?;
        self.ensure_semantic_index("semantic_documents", "doc_idx")?;
        self.ensure_semantic_index("semantic_node_prototypes", "node_idx")
    }

    fn ensure_semantic_index(&self, relation: &str, index_name: &str) -> Result<(), StoreError> {
        let script = format!(
            r#"
::hnsw create {relation}:{index_name} {{
    dim: {dim},
    dtype: F32,
    fields: [vec],
    distance: Cosine,
    m: 16,
    ef_construction: 200,
    filter: model_id == "{model_id}",
}}
"#,
            dim = SEMANTIC_VECTOR_DIM,
            model_id = SEMANTIC_MODEL_ID,
        );
        match self
            .db
            .run_script(&script, Default::default(), cozo::ScriptMutability::Mutable)
        {
            Ok(_) => Ok(()),
            Err(error) if relation_already_exists(&error.to_string()) => Ok(()),
            Err(error) => Err(StoreError::Schema(error.to_string())),
        }
    }

    pub fn upsert_semantic_vectors(
        &self,
        rows: &[SemanticVectorRow<'_>],
    ) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }
        let data = rows
            .iter()
            .map(|row| {
                if row.values.len() != SEMANTIC_VECTOR_DIM {
                    return Err(StoreError::Query(format!(
                        "semantic vector dimension mismatch for {}: expected {}, got {}",
                        row.span_id,
                        SEMANTIC_VECTOR_DIM,
                        row.values.len()
                    )));
                }
                if row.model_id != SEMANTIC_MODEL_ID {
                    return Err(StoreError::Query(format!(
                        "semantic model mismatch for {}: expected {}, got {}",
                        row.span_id, SEMANTIC_MODEL_ID, row.model_id
                    )));
                }
                Ok(cozo::DataValue::List(vec![
                    cozo::DataValue::Str(row.span_id.to_owned().into()),
                    cozo::DataValue::List(
                        row.values
                            .iter()
                            .map(|value| cozo::DataValue::from(*value as f64))
                            .collect(),
                    ),
                    cozo::DataValue::Str(row.model_id.to_owned().into()),
                    cozo::DataValue::from(row.updated_at),
                ]))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let params = [("rows".to_owned(), cozo::DataValue::List(data))]
            .into_iter()
            .collect();
        self.db
            .run_script(
                r#"
rows[span_id, raw_vec, model_id, updated_at] <- $rows
?[span_id, vec, model_id, updated_at] := rows[span_id, raw_vec, model_id, updated_at],
    vec = vec(raw_vec, "F32")
:put semantic_vectors { span_id => vec, model_id, updated_at }
"#,
                params,
                cozo::ScriptMutability::Mutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(())
    }

    pub fn upsert_semantic_document_vectors(
        &self,
        rows: &[SemanticDocumentVectorRow<'_>],
    ) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }
        let data = rows
            .iter()
            .map(|row| {
                if row.values.len() != SEMANTIC_VECTOR_DIM {
                    return Err(StoreError::Query(format!(
                        "semantic document vector dimension mismatch for {}: expected {}, got {}",
                        row.document_id,
                        SEMANTIC_VECTOR_DIM,
                        row.values.len()
                    )));
                }
                if row.model_id != SEMANTIC_MODEL_ID {
                    return Err(StoreError::Query(format!(
                        "semantic document model mismatch for {}: expected {}, got {}",
                        row.document_id, SEMANTIC_MODEL_ID, row.model_id
                    )));
                }
                Ok(cozo::DataValue::List(vec![
                    cozo::DataValue::Str(row.document_id.to_owned().into()),
                    cozo::DataValue::List(
                        row.values
                            .iter()
                            .map(|value| cozo::DataValue::from(*value as f64))
                            .collect(),
                    ),
                    cozo::DataValue::Str(row.model_id.to_owned().into()),
                    cozo::DataValue::from(row.leaf_count as i64),
                    json_to_datavalue(&json!(row.evidence_refs)),
                    cozo::DataValue::from(row.updated_at),
                ]))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let params = [("rows".to_owned(), cozo::DataValue::List(data))]
            .into_iter()
            .collect();
        self.db
            .run_script(
                r#"
rows[document_id, raw_vec, model_id, leaf_count, evidence_refs, updated_at] <- $rows
?[document_id, vec, model_id, leaf_count, evidence_refs, updated_at] := rows[document_id, raw_vec, model_id, leaf_count, evidence_refs, updated_at],
    vec = vec(raw_vec, "F32")
:put semantic_documents { document_id => vec, model_id, leaf_count, evidence_refs, updated_at }
"#,
                params,
                cozo::ScriptMutability::Mutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(())
    }

    pub fn upsert_semantic_node_vectors(
        &self,
        rows: &[SemanticNodeVectorRow<'_>],
    ) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }
        let data = rows
            .iter()
            .map(|row| {
                if row.values.len() != SEMANTIC_VECTOR_DIM {
                    return Err(StoreError::Query(format!(
                        "semantic node vector dimension mismatch for {}: expected {}, got {}",
                        row.node_id,
                        SEMANTIC_VECTOR_DIM,
                        row.values.len()
                    )));
                }
                if row.model_id != SEMANTIC_MODEL_ID {
                    return Err(StoreError::Query(format!(
                        "semantic node model mismatch for {}: expected {}, got {}",
                        row.node_id, SEMANTIC_MODEL_ID, row.model_id
                    )));
                }
                Ok(cozo::DataValue::List(vec![
                    cozo::DataValue::Str(row.node_id.to_owned().into()),
                    cozo::DataValue::Str(row.node_kind.to_owned().into()),
                    nullable_str_to_datavalue(row.document_id),
                    nullable_str_to_datavalue(row.narrative_id),
                    nullable_str_to_datavalue(row.folder_id),
                    cozo::DataValue::List(
                        row.values
                            .iter()
                            .map(|value| cozo::DataValue::from(*value as f64))
                            .collect(),
                    ),
                    cozo::DataValue::Str(row.model_id.to_owned().into()),
                    json_to_datavalue(&json!(row.evidence_refs)),
                    cozo::DataValue::from(row.updated_at),
                ]))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let params = [("rows".to_owned(), cozo::DataValue::List(data))]
            .into_iter()
            .collect();
        self.db
            .run_script(
                r#"
rows[node_id, node_kind, document_id, narrative_id, folder_id, raw_vec, model_id, evidence_refs, updated_at] <- $rows
?[node_id, node_kind, document_id, narrative_id, folder_id, vec, model_id, evidence_refs, updated_at] :=
    rows[node_id, node_kind, document_id, narrative_id, folder_id, raw_vec, model_id, evidence_refs, updated_at],
    vec = vec(raw_vec, "F32")
:put semantic_node_prototypes {
    node_id => node_kind,
    document_id,
    narrative_id,
    folder_id,
    vec,
    model_id,
    evidence_refs,
    updated_at
}
"#,
                params,
                cozo::ScriptMutability::Mutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(())
    }

    pub fn list_leaf_chunks_for_documents(
        &self,
        document_ids: &[String],
    ) -> Result<Vec<SemanticLeafChunk>, StoreError> {
        if document_ids.is_empty() {
            return Ok(Vec::new());
        }
        let params = [(
            "docs".to_owned(),
            cozo::DataValue::List(
                document_ids
                    .iter()
                    .map(|value| {
                        cozo::DataValue::List(vec![cozo::DataValue::Str(value.clone().into())])
                    })
                    .collect(),
            ),
        )]
        .into_iter()
        .collect();
        let rows = self
            .db
            .run_script(
                r#"
selected[doc_id] <- $docs
?[span_id, doc_id, text, narrative_id, folder_id] := selected[doc_id],
    *chunks{chunk_id, doc_id, level, text, scope_narrative: narrative_id, scope_folder: folder_id},
    level = 0,
    *chunkid_map{id: chunk_id, chunk_key: span_id}
:order doc_id, span_id
"#,
                params,
                cozo::ScriptMutability::Immutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(rows
            .rows
            .into_iter()
            .map(|row| SemanticLeafChunk {
                span_id: row
                    .first()
                    .and_then(datavalue_as_str)
                    .unwrap_or_default()
                    .to_owned(),
                document_id: row
                    .get(1)
                    .and_then(datavalue_as_str)
                    .unwrap_or_default()
                    .to_owned(),
                text: row
                    .get(2)
                    .and_then(datavalue_as_str)
                    .unwrap_or_default()
                    .to_owned(),
                narrative_id: row.get(3).and_then(datavalue_as_str).map(str::to_owned),
                folder_id: row.get(4).and_then(datavalue_as_str).map(str::to_owned),
            })
            .filter(|row| !row.span_id.is_empty() && !row.text.is_empty())
            .collect())
    }

    pub fn query_semantic_neighbors(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticNeighbor>, StoreError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        if query_vector.len() != SEMANTIC_VECTOR_DIM {
            return Err(StoreError::Query(format!(
                "semantic query vector dimension mismatch: expected {}, got {}",
                SEMANTIC_VECTOR_DIM,
                query_vector.len()
            )));
        }
        let params = build_semantic_query_params(query_vector, oversample.max(limit));
        let rows = self
            .db
            .run_script(
                r#"
q_vec[v] := v = vec($query, "F32")
?[span_id, dist, narrative_id, folder_id] := q_vec[q],
    ~semantic_vectors:vec_idx{
        span_id,
        vec,
        model_id,
        updated_at |
        query: q,
        k: $k,
        ef: $ef,
        bind_distance: dist,
        filter: model_id == $model_id
    },
    *chunkid_map{id: chunk_id, chunk_key: span_id},
    *chunks{chunk_id, level, scope_narrative: narrative_id, scope_folder: folder_id},
    level = 0
:order dist
"#,
                params,
                cozo::ScriptMutability::Immutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut filtered = Vec::with_capacity(limit);
        for row in rows.rows {
            let Some(span_id) = row.first().and_then(datavalue_as_str) else {
                continue;
            };
            if !matches_scope(
                scope,
                row.get(2).and_then(datavalue_as_str),
                row.get(3).and_then(datavalue_as_str),
            ) {
                continue;
            }
            let Some(distance) = row.get(1).and_then(datavalue_to_f64) else {
                continue;
            };
            filtered.push(SemanticNeighbor {
                span_id: span_id.to_owned(),
                distance,
            });
            if filtered.len() >= limit {
                break;
            }
        }
        Ok(filtered)
    }

    pub fn query_semantic_documents(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticDocumentNeighbor>, StoreError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        if query_vector.len() != SEMANTIC_VECTOR_DIM {
            return Err(StoreError::Query(format!(
                "semantic document query vector dimension mismatch: expected {}, got {}",
                SEMANTIC_VECTOR_DIM,
                query_vector.len()
            )));
        }
        let params = build_semantic_query_params(query_vector, oversample.max(limit));
        let rows = self
            .db
            .run_script(
                r#"
q_vec[v] := v = vec($query, "F32")
doc_scope[document_id, narrative_id, folder_id] := *chunks{
    doc_id: document_id,
    level,
    scope_narrative: narrative_id,
    scope_folder: folder_id
},
    level = 0
?[document_id, dist, leaf_count, evidence_refs, narrative_id, folder_id] := q_vec[q],
    ~semantic_documents:doc_idx{
        document_id,
        vec,
        model_id,
        leaf_count,
        evidence_refs,
        updated_at |
        query: q,
        k: $k,
        ef: $ef,
        bind_distance: dist,
        filter: model_id == $model_id
    },
    doc_scope[document_id, narrative_id, folder_id]
:order dist
"#,
                params,
                cozo::ScriptMutability::Immutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut filtered = Vec::with_capacity(limit);
        for row in rows.rows {
            let Some(document_id) = row.first().and_then(datavalue_as_str) else {
                continue;
            };
            if !matches_scope(
                scope,
                row.get(4).and_then(datavalue_as_str),
                row.get(5).and_then(datavalue_as_str),
            ) {
                continue;
            }
            let Some(distance) = row.get(1).and_then(datavalue_to_f64) else {
                continue;
            };
            let leaf_count = row
                .get(2)
                .and_then(datavalue_to_f64)
                .and_then(|value| usize::try_from(value as i64).ok())
                .unwrap_or_default();
            filtered.push(SemanticDocumentNeighbor {
                document_id: document_id.to_owned(),
                distance,
                leaf_count,
                evidence_refs: row
                    .get(3)
                    .map(datavalue_to_json_ref)
                    .and_then(|value| serde_json::from_value::<Vec<String>>(value).ok())
                    .unwrap_or_default(),
            });
            if filtered.len() >= limit {
                break;
            }
        }
        Ok(filtered)
    }

    pub fn query_semantic_node_neighbors(
        &self,
        query_vector: &[f32],
        scope: &ScopeKey,
        kind: &str,
        exclude_node_id: Option<&str>,
        limit: usize,
        oversample: usize,
    ) -> Result<Vec<SemanticNodeNeighbor>, StoreError> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        if query_vector.len() != SEMANTIC_VECTOR_DIM {
            return Err(StoreError::Query(format!(
                "semantic node query vector dimension mismatch: expected {}, got {}",
                SEMANTIC_VECTOR_DIM,
                query_vector.len()
            )));
        }
        let mut params = build_semantic_query_params(query_vector, oversample.max(limit));
        params.insert(
            "kind".to_owned(),
            cozo::DataValue::Str(kind.to_owned().into()),
        );
        params.insert(
            "exclude_node_id".to_owned(),
            nullable_str_to_datavalue(exclude_node_id),
        );
        let rows = self
            .db
            .run_script(
                r#"
q_vec[v] := v = vec($query, "F32")
?[node_id, dist, node_kind, document_id, narrative_id, folder_id, evidence_refs] := q_vec[q],
    ~semantic_node_prototypes:node_idx{
        node_id,
        node_kind,
        document_id,
        narrative_id,
        folder_id,
        vec,
        model_id,
        evidence_refs,
        updated_at |
        query: q,
        k: $k,
        ef: $ef,
        bind_distance: dist,
        filter: model_id == $model_id
    },
    node_kind == $kind,
    (is_null($exclude_node_id) || node_id != $exclude_node_id)
:order dist
"#,
                params,
                cozo::ScriptMutability::Immutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut filtered = Vec::with_capacity(limit);
        for row in rows.rows {
            let Some(node_id) = row.first().and_then(datavalue_as_str) else {
                continue;
            };
            if !matches_scope(
                scope,
                row.get(4).and_then(datavalue_as_str),
                row.get(5).and_then(datavalue_as_str),
            ) {
                continue;
            }
            let Some(distance) = row.get(1).and_then(datavalue_to_f64) else {
                continue;
            };
            filtered.push(SemanticNodeNeighbor {
                node_id: node_id.to_owned(),
                node_kind: row
                    .get(2)
                    .and_then(datavalue_as_str)
                    .unwrap_or_default()
                    .to_owned(),
                distance,
                document_id: row.get(3).and_then(datavalue_as_str).map(str::to_owned),
                narrative_id: row.get(4).and_then(datavalue_as_str).map(str::to_owned),
                folder_id: row.get(5).and_then(datavalue_as_str).map(str::to_owned),
                evidence_refs: row
                    .get(6)
                    .map(datavalue_to_json_ref)
                    .and_then(|value| serde_json::from_value::<Vec<String>>(value).ok())
                    .unwrap_or_default(),
            });
            if filtered.len() >= limit {
                break;
            }
        }
        Ok(filtered)
    }

    pub fn query_semantic_neighbors_in_documents(
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
        if query_vector.len() != SEMANTIC_VECTOR_DIM {
            return Err(StoreError::Query(format!(
                "semantic query vector dimension mismatch: expected {}, got {}",
                SEMANTIC_VECTOR_DIM,
                query_vector.len()
            )));
        }
        let mut params = build_semantic_query_params(query_vector, oversample.max(limit));
        params.insert(
            "docs".to_owned(),
            cozo::DataValue::List(
                document_ids
                    .iter()
                    .map(|value| {
                        cozo::DataValue::List(vec![cozo::DataValue::Str(value.clone().into())])
                    })
                    .collect(),
            ),
        );
        let rows = self
            .db
            .run_script(
                r#"
selected_docs[doc_id] <- $docs
q_vec[v] := v = vec($query, "F32")
?[span_id, dist, narrative_id, folder_id] := q_vec[q],
    ~semantic_vectors:vec_idx{
        span_id,
        vec,
        model_id,
        updated_at |
        query: q,
        k: $k,
        ef: $ef,
        bind_distance: dist,
        filter: model_id == $model_id
    },
    *chunkid_map{id: chunk_id, chunk_key: span_id},
    *chunks{
        chunk_id,
        doc_id,
        level,
        scope_narrative: narrative_id,
        scope_folder: folder_id
    },
    level = 0,
    selected_docs[doc_id]
:order dist
"#,
                params,
                cozo::ScriptMutability::Immutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut filtered = Vec::with_capacity(limit);
        for row in rows.rows {
            let Some(span_id) = row.first().and_then(datavalue_as_str) else {
                continue;
            };
            if !matches_scope(
                scope,
                row.get(2).and_then(datavalue_as_str),
                row.get(3).and_then(datavalue_as_str),
            ) {
                continue;
            }
            let Some(distance) = row.get(1).and_then(datavalue_to_f64) else {
                continue;
            };
            filtered.push(SemanticNeighbor {
                span_id: span_id.to_owned(),
                distance,
            });
            if filtered.len() >= limit {
                break;
            }
        }
        Ok(filtered)
    }

    pub fn put_row(&self, relation: &str, row: Value) -> Result<(), StoreError> {
        let compact = compact_row_from_value(relation, &row)?;
        self.put_compact_rows(relation, &[compact])
    }

    pub fn put_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }

        let compact_rows = rows
            .iter()
            .map(|row| compact_row_from_value(relation, row))
            .collect::<Result<Vec<_>, _>>()?;
        self.put_compact_rows(relation, &compact_rows)
    }

    pub fn put_compact_rows(&self, relation: &str, rows: &[CompactRow]) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }

        let spec = relation_spec(relation)?;
        for chunk in rows.chunks(PUT_ROWS_BATCH_LIMIT) {
            let (script, params) = build_put_compact_payload(spec, chunk)?;
            self.db
                .run_script(&script, params, cozo::ScriptMutability::Mutable)
                .map_err(|error| StoreError::Query(error.to_string()))?;
        }
        Ok(())
    }

    pub fn put_compact_rows_owned(
        &self,
        relation: &str,
        rows: Vec<CompactRow>,
    ) -> Result<(), StoreError> {
        self.put_compact_rows(relation, &rows)
    }

    pub fn delete_key_rows(&self, relation: &str, rows: &[CompactRow]) -> Result<(), StoreError> {
        if rows.is_empty() {
            return Ok(());
        }

        let spec = relation_spec(relation)?;
        for chunk in rows.chunks(PUT_ROWS_BATCH_LIMIT) {
            let (script, params) = build_delete_key_payload(spec, chunk)?;
            self.db
                .run_script(&script, params, cozo::ScriptMutability::Mutable)
                .map_err(|error| StoreError::Query(error.to_string()))?;
        }
        Ok(())
    }

    pub fn fetch_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        let spec = relation_spec(relation)?;
        let script = build_fetch_script(spec, None)?;
        let rows = self
            .db
            .run_script(
                &script,
                Default::default(),
                cozo::ScriptMutability::Immutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(named_rows_to_json_objects(rows))
    }

    pub fn fetch_compact_rows(&self, relation: &str) -> Result<Vec<CompactRow>, StoreError> {
        let spec = relation_spec(relation)?;
        let script = build_fetch_script(spec, None)?;
        let rows = self
            .db
            .run_script(
                &script,
                Default::default(),
                cozo::ScriptMutability::Immutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(rows
            .rows
            .into_iter()
            .map(|row| row.into_iter().collect::<CompactRow>())
            .collect())
    }

    pub fn fetch_rows_with_columns(
        &self,
        relation: &str,
        columns: &[&str],
    ) -> Result<Vec<Value>, StoreError> {
        let spec = relation_spec(relation)?;
        let script = build_fetch_script(spec, Some(columns))?;
        let rows = self
            .db
            .run_script(
                &script,
                Default::default(),
                cozo::ScriptMutability::Immutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(named_rows_to_json_objects(rows))
    }

    pub fn fetch_compact_rows_with_columns(
        &self,
        relation: &str,
        columns: &[&str],
    ) -> Result<Vec<CompactRow>, StoreError> {
        let spec = relation_spec(relation)?;
        let script = build_fetch_script(spec, Some(columns))?;
        let rows = self
            .db
            .run_script(
                &script,
                Default::default(),
                cozo::ScriptMutability::Immutable,
            )
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(rows
            .rows
            .into_iter()
            .map(|row| row.into_iter().collect::<CompactRow>())
            .collect())
    }

    pub fn fetch_compact_rows_where_str(
        &self,
        relation: &str,
        columns: &[&str],
        filter_column: &str,
        filter_value: &str,
    ) -> Result<Vec<CompactRow>, StoreError> {
        let spec = relation_spec(relation)?;
        let script = build_fetch_where_eq_script(spec, columns, filter_column)?;
        let params = [(
            "filter".to_owned(),
            cozo::DataValue::Str(filter_value.to_owned().into()),
        )]
        .into_iter()
        .collect();
        let rows = self
            .db
            .run_script(&script, params, cozo::ScriptMutability::Immutable)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(rows
            .rows
            .into_iter()
            .map(|row| row.into_iter().collect::<CompactRow>())
            .collect())
    }

    pub fn fetch_compact_rows_where_in_strings(
        &self,
        relation: &str,
        columns: &[&str],
        filter_column: &str,
        filter_values: &[String],
    ) -> Result<Vec<CompactRow>, StoreError> {
        if filter_values.is_empty() {
            return Ok(Vec::new());
        }
        let spec = relation_spec(relation)?;
        let script = build_fetch_where_in_script(spec, columns, filter_column)?;
        let params = [(
            "data".to_owned(),
            cozo::DataValue::List(
                filter_values
                    .iter()
                    .map(|value| {
                        cozo::DataValue::List(vec![cozo::DataValue::Str(value.clone().into())])
                    })
                    .collect(),
            ),
        )]
        .into_iter()
        .collect();
        let rows = self
            .db
            .run_script(&script, params, cozo::ScriptMutability::Immutable)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(rows
            .rows
            .into_iter()
            .map(|row| row.into_iter().collect::<CompactRow>())
            .collect())
    }

    pub fn run_datalog_json(
        &self,
        script: &str,
        seed_strings: &[String],
    ) -> Result<Vec<Vec<Value>>, StoreError> {
        let seed_values: Vec<cozo::DataValue> = seed_strings
            .iter()
            .map(|s| cozo::DataValue::List(vec![cozo::DataValue::Str(s.clone().into())]))
            .collect();
        let params: BTreeMap<String, cozo::DataValue> =
            [("seeds".to_owned(), cozo::DataValue::List(seed_values))]
                .into_iter()
                .collect();
        let rows = self
            .db
            .run_script(script, params, cozo::ScriptMutability::Immutable)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(rows
            .rows
            .into_iter()
            .map(|row| row.into_iter().map(datavalue_to_json).collect())
            .collect())
    }

    pub fn clear_relation(&self, relation: &str) -> Result<(), StoreError> {
        let spec = relation_spec(relation)?;
        let script = build_clear_script(spec);
        self.db
            .run_script(&script, Default::default(), cozo::ScriptMutability::Mutable)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        Ok(())
    }

    pub fn clear_relations(&self, relations: &[&str]) -> Result<(), StoreError> {
        for relation in ALL_RELATIONS.iter().rev() {
            if relations
                .iter()
                .any(|candidate| *candidate == relation.name)
            {
                self.clear_relation(relation.name)?;
            }
        }
        self.put_row(
            "phoenix_schema_state",
            serde_json::json!({
                "version": self.schema_version,
                "updated_at": now_ms(),
            }),
        )?;
        Ok(())
    }

    pub fn clear_relations_by_ids(&self, relation_ids: &[u16]) -> Result<(), StoreError> {
        let relation_names = relation_ids
            .iter()
            .filter_map(|relation_id| {
                ALL_RELATIONS
                    .get(*relation_id as usize)
                    .map(|relation| relation.name)
            })
            .collect::<Vec<_>>();
        self.clear_relations(&relation_names)
    }

    pub fn clear_all_relations(&self) -> Result<(), StoreError> {
        let relation_names = ALL_RELATIONS
            .iter()
            .map(|relation| relation.name)
            .collect::<Vec<_>>();
        self.clear_relations(&relation_names)
    }

    pub fn relation_counts(&self) -> Result<Vec<RelationCount>, StoreError> {
        let mut counts = Vec::with_capacity(ALL_RELATIONS.len());
        for relation in ALL_RELATIONS {
            let key_columns = relation
                .key_columns()
                .map(|column| column.name)
                .collect::<Vec<_>>();
            let rows = self.fetch_compact_rows_with_columns(relation.name, &key_columns)?;
            counts.push(RelationCount {
                relation: relation.name.to_owned(),
                rows: rows.len(),
            });
        }
        Ok(counts)
    }

    pub fn upsert_entity_card(&self, card: &EntityCard) -> Result<(), StoreError> {
        self.put_row(
            "entity_cards",
            serde_json::json!({
                "entity_id": card.entity_id.0.as_str(),
                "card_id": &card.card_id,
                "name": &card.name,
                "color": &card.color,
                "icon": &card.icon,
                "display_order": card.display_order,
                "is_collapsed": card.is_collapsed,
                "created_at": card.created_at,
                "updated_at": card.updated_at,
            }),
        )
    }

    pub fn upsert_entity_cards_batch(&self, cards: &[EntityCard]) -> Result<(), StoreError> {
        if cards.is_empty() {
            return Ok(());
        }
        let rows = cards
            .iter()
            .map(|card| {
                serde_json::json!({
                    "entity_id": card.entity_id.0.as_str(),
                    "card_id": &card.card_id,
                    "name": &card.name,
                    "color": &card.color,
                    "icon": &card.icon,
                    "display_order": card.display_order,
                    "is_collapsed": card.is_collapsed,
                    "created_at": card.created_at,
                    "updated_at": card.updated_at,
                })
            })
            .collect::<Vec<_>>();
        self.put_rows("entity_cards", &rows)
    }

    pub fn get_entity_cards(&self, entity_id: &EntityId) -> Result<Vec<EntityCard>, StoreError> {
        let rows = self.fetch_compact_rows_with_columns(
            "entity_cards",
            &[
                "entity_id",
                "card_id",
                "name",
                "color",
                "icon",
                "display_order",
                "is_collapsed",
                "created_at",
                "updated_at",
            ],
        )?;
        let mut cards = rows
            .iter()
            .filter_map(|row| {
                let row = CompactRowView::new(
                    &[
                        "entity_id",
                        "card_id",
                        "name",
                        "color",
                        "icon",
                        "display_order",
                        "is_collapsed",
                        "created_at",
                        "updated_at",
                    ],
                    row,
                );
                (row.get_str("entity_id") == Some(entity_id.0.as_str())).then(|| EntityCard {
                    entity_id: EntityId(row.get_str("entity_id").unwrap_or_default().to_owned()),
                    card_id: row.get_str("card_id").unwrap_or_default().to_owned(),
                    name: row.get_str("name").unwrap_or_default().to_owned(),
                    color: row.get_str("color").unwrap_or_default().to_owned(),
                    icon: row.get_str("icon").unwrap_or_default().to_owned(),
                    display_order: row.get_i64("display_order").unwrap_or_default() as i32,
                    is_collapsed: row.get_bool("is_collapsed").unwrap_or(false),
                    created_at: row.get_i64("created_at").unwrap_or_default(),
                    updated_at: row.get_i64("updated_at").unwrap_or_default(),
                })
            })
            .collect::<Vec<_>>();
        cards.sort_by_key(|card| (card.display_order, card.card_id.clone()));
        Ok(cards)
    }

    pub fn upsert_folder_schema(&self, schema: &FolderSchema) -> Result<(), StoreError> {
        self.put_row(
            "folder_schemas",
            serde_json::json!({
                "id": &schema.id,
                "entity_kind": &schema.entity_kind,
                "subtype": null_if_empty(&schema.subtype),
                "name": &schema.name,
                "description": null_if_empty(&schema.description),
                "allowed_subfolders": parse_json_string_array(&schema.allowed_subfolders),
                "allowed_note_types": parse_json_string_array(&schema.allowed_note_types),
                "is_vault_root": schema.is_vault_root,
                "container_only": schema.container_only,
                "propagate_kind_to_children": schema.propagate_kind_to_children,
                "icon": null_if_empty(&schema.icon),
                "is_system": schema.is_system,
                "created_at": schema.created_at,
                "updated_at": schema.updated_at,
            }),
        )
    }

    pub fn get_folder_schema(&self, id: &str) -> Result<Option<FolderSchema>, StoreError> {
        let rows = self.fetch_compact_rows_with_columns(
            "folder_schemas",
            &[
                "id",
                "entity_kind",
                "subtype",
                "name",
                "description",
                "allowed_subfolders",
                "allowed_note_types",
                "is_vault_root",
                "container_only",
                "propagate_kind_to_children",
                "icon",
                "is_system",
                "created_at",
                "updated_at",
            ],
        )?;
        Ok(rows.into_iter().find_map(|row| {
            let row = CompactRowView::new(
                &[
                    "id",
                    "entity_kind",
                    "subtype",
                    "name",
                    "description",
                    "allowed_subfolders",
                    "allowed_note_types",
                    "is_vault_root",
                    "container_only",
                    "propagate_kind_to_children",
                    "icon",
                    "is_system",
                    "created_at",
                    "updated_at",
                ],
                &row,
            );
            (row.get_str("id") == Some(id)).then(|| FolderSchema {
                id: row.get_str("id").unwrap_or_default().to_owned(),
                entity_kind: row.get_str("entity_kind").unwrap_or_default().to_owned(),
                subtype: row.get_str("subtype").unwrap_or_default().to_owned(),
                name: row.get_str("name").unwrap_or_default().to_owned(),
                description: row.get_str("description").unwrap_or_default().to_owned(),
                allowed_subfolders: json_to_string_array(row.get_json("allowed_subfolders")),
                allowed_note_types: json_to_string_array(row.get_json("allowed_note_types")),
                is_vault_root: row.get_bool("is_vault_root").unwrap_or(false),
                container_only: row.get_bool("container_only").unwrap_or(false),
                propagate_kind_to_children: row
                    .get_bool("propagate_kind_to_children")
                    .unwrap_or(false),
                icon: row.get_str("icon").unwrap_or_default().to_owned(),
                is_system: row.get_bool("is_system").unwrap_or(false),
                created_at: row.get_i64("created_at").unwrap_or_default(),
                updated_at: row.get_i64("updated_at").unwrap_or_default(),
            })
        }))
    }

    pub fn upsert_network_instance(&self, network: &NetworkInstance) -> Result<(), StoreError> {
        self.put_row(
            "network_instance",
            serde_json::json!({
                "id": &network.id,
                "name": &network.name,
                "schema_id": null_if_empty(&network.schema_id),
                "network_kind": &network.network_kind,
                "network_subtype": null_if_empty(&network.network_subtype),
                "root_folder_id": null_if_empty(&network.root_folder_id),
                "root_entity_id": null_if_empty(&network.root_entity_id),
                "namespace": &network.namespace,
                "description": null_if_empty(&network.description),
                "tags": &network.tags,
                "member_count": network.member_count as i64,
                "relationship_count": network.relationship_count as i64,
                "max_depth": network.max_depth as i64,
                "created_at": network.created_at,
                "updated_at": network.updated_at,
                "group_id": null_if_empty(&network.group_id),
                "scope_type": &network.scope_type,
                "narrative_id": null_if_empty(&network.narrative_id),
            }),
        )
    }

    pub fn get_network_instance(&self, id: &str) -> Result<Option<NetworkInstance>, StoreError> {
        let rows = self.fetch_compact_rows_with_columns(
            "network_instance",
            &[
                "id",
                "name",
                "schema_id",
                "network_kind",
                "network_subtype",
                "root_folder_id",
                "root_entity_id",
                "namespace",
                "description",
                "tags",
                "member_count",
                "relationship_count",
                "max_depth",
                "created_at",
                "updated_at",
                "group_id",
                "scope_type",
                "narrative_id",
            ],
        )?;
        Ok(rows.into_iter().find_map(|row| {
            let row = CompactRowView::new(
                &[
                    "id",
                    "name",
                    "schema_id",
                    "network_kind",
                    "network_subtype",
                    "root_folder_id",
                    "root_entity_id",
                    "namespace",
                    "description",
                    "tags",
                    "member_count",
                    "relationship_count",
                    "max_depth",
                    "created_at",
                    "updated_at",
                    "group_id",
                    "scope_type",
                    "narrative_id",
                ],
                &row,
            );
            (row.get_str("id") == Some(id)).then(|| network_instance_from_row(&row))
        }))
    }

    pub fn list_network_instances(&self) -> Result<Vec<NetworkInstance>, StoreError> {
        let columns = &[
            "id",
            "name",
            "schema_id",
            "network_kind",
            "network_subtype",
            "root_folder_id",
            "root_entity_id",
            "namespace",
            "description",
            "tags",
            "member_count",
            "relationship_count",
            "max_depth",
            "created_at",
            "updated_at",
            "group_id",
            "scope_type",
            "narrative_id",
        ];
        let rows = self.fetch_compact_rows_with_columns("network_instance", columns)?;
        let mut networks = rows
            .iter()
            .map(|row| network_instance_from_row(&CompactRowView::new(columns, row)))
            .collect::<Vec<_>>();
        networks.sort_by(|left, right| {
            right
                .updated_at
                .cmp(&left.updated_at)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(networks)
    }

    pub fn delete_network_instance(&self, id: &str) -> Result<(), StoreError> {
        let mut row = CompactRow::new();
        row.push(cozo::DataValue::Str(id.to_owned().into()));
        self.delete_key_rows("network_instance", &[row])
    }

    pub fn upsert_network_memberships(
        &self,
        members: &[NetworkMembership],
    ) -> Result<(), StoreError> {
        if members.is_empty() {
            return Ok(());
        }
        let rows = members
            .iter()
            .map(|member| {
                serde_json::json!({
                    "network_id": &member.network_id,
                    "entity_id": member.entity_id.0.as_str(),
                    "x": member.x,
                    "y": member.y,
                    "fixed": member.fixed,
                })
            })
            .collect::<Vec<_>>();
        self.put_rows("network_membership", &rows)
    }

    pub fn get_network_members(
        &self,
        network_id: &str,
    ) -> Result<Vec<NetworkMembership>, StoreError> {
        let columns = &["network_id", "entity_id", "x", "y", "fixed"];
        let rows = self.fetch_compact_rows_with_columns("network_membership", columns)?;
        let mut members = rows
            .iter()
            .filter_map(|row| {
                let row = CompactRowView::new(columns, row);
                (row.get_str("network_id") == Some(network_id)).then(|| NetworkMembership {
                    network_id: row.get_str("network_id").unwrap_or_default().to_owned(),
                    entity_id: EntityId(row.get_str("entity_id").unwrap_or_default().to_owned()),
                    x: row
                        .get_value("x")
                        .and_then(datavalue_to_f64)
                        .unwrap_or_default(),
                    y: row
                        .get_value("y")
                        .and_then(datavalue_to_f64)
                        .unwrap_or_default(),
                    fixed: row.get_bool("fixed").unwrap_or(false),
                })
            })
            .collect::<Vec<_>>();
        members.sort_by(|left, right| left.entity_id.0.cmp(&right.entity_id.0));
        Ok(members)
    }

    pub fn delete_network_memberships(
        &self,
        members: &[NetworkMembership],
    ) -> Result<(), StoreError> {
        if members.is_empty() {
            return Ok(());
        }
        let rows = members
            .iter()
            .map(|member| {
                let mut row = CompactRow::new();
                row.push(cozo::DataValue::Str(member.network_id.clone().into()));
                row.push(cozo::DataValue::Str(member.entity_id.0.clone().into()));
                row
            })
            .collect::<Vec<_>>();
        self.delete_key_rows("network_membership", &rows)
    }

    pub fn upsert_network_relationships(
        &self,
        relationships: &[NetworkRelationship],
    ) -> Result<(), StoreError> {
        if relationships.is_empty() {
            return Ok(());
        }
        let rows = relationships
            .iter()
            .map(|relationship| {
                serde_json::json!({
                    "network_id": &relationship.network_id,
                    "source_entity_id": relationship.source_entity_id.0.as_str(),
                    "target_entity_id": relationship.target_entity_id.0.as_str(),
                    "relationship_id": &relationship.relationship_id,
                })
            })
            .collect::<Vec<_>>();
        self.put_rows("network_relationship", &rows)
    }

    pub fn get_network_relationships(
        &self,
        network_id: &str,
    ) -> Result<Vec<NetworkRelationship>, StoreError> {
        let columns = &[
            "network_id",
            "source_entity_id",
            "target_entity_id",
            "relationship_id",
        ];
        let rows = self.fetch_compact_rows_with_columns("network_relationship", columns)?;
        let mut relationships = rows
            .iter()
            .filter_map(|row| {
                let row = CompactRowView::new(columns, row);
                (row.get_str("network_id") == Some(network_id)).then(|| NetworkRelationship {
                    network_id: row.get_str("network_id").unwrap_or_default().to_owned(),
                    source_entity_id: EntityId(
                        row.get_str("source_entity_id")
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    target_entity_id: EntityId(
                        row.get_str("target_entity_id")
                            .unwrap_or_default()
                            .to_owned(),
                    ),
                    relationship_id: row
                        .get_str("relationship_id")
                        .unwrap_or_default()
                        .to_owned(),
                })
            })
            .collect::<Vec<_>>();
        relationships.sort_by(|left, right| left.relationship_id.cmp(&right.relationship_id));
        Ok(relationships)
    }

    pub fn delete_network_relationships(
        &self,
        relationships: &[NetworkRelationship],
    ) -> Result<(), StoreError> {
        if relationships.is_empty() {
            return Ok(());
        }
        let rows = relationships
            .iter()
            .map(|relationship| {
                let mut row = CompactRow::new();
                row.push(cozo::DataValue::Str(relationship.network_id.clone().into()));
                row.push(cozo::DataValue::Str(
                    relationship.relationship_id.clone().into(),
                ));
                row
            })
            .collect::<Vec<_>>();
        self.delete_key_rows("network_relationship", &rows)
    }

    pub fn snapshot_descriptor(&self, created_at: i64, payload_bytes: usize) -> SnapshotDto {
        let relation_counts = self.relation_counts().unwrap_or_default();
        SnapshotDto {
            schema_version: self.schema_version.to_owned(),
            created_at,
            payload_bytes,
            relation_counts,
        }
    }

    pub fn export_snapshot(&self) -> Result<Vec<u8>, StoreError> {
        self.export_snapshot_partition(SnapshotPartition::All)
    }

    pub fn export_snapshot_partition(
        &self,
        partition: SnapshotPartition,
    ) -> Result<Vec<u8>, StoreError> {
        let created_at = now_ms();
        let relations = relations_for_partition(partition);
        let mut bytes = Vec::with_capacity(relations.len() * SnapshotRelationBlockHeader::BYTE_LEN);
        SnapshotWireHeader {
            magic: SNAPSHOT_MAGIC,
            version: SNAPSHOT_VERSION,
            relation_count: relations.len() as u16,
            created_at,
        }
        .encode(&mut bytes);

        for relation in relations {
            let relation_id = relation_id_for_name(relation.name)? as u16;
            let rows = self.fetch_compact_rows(relation.name)?;
            let payload = encode_snapshot_block(&rows)?;
            let (codec, block_bytes) = maybe_compress_snapshot_block(&payload);
            SnapshotRelationBlockHeader {
                relation_id,
                codec,
                row_count: rows.len() as u32,
                encoded_len: block_bytes.len() as u32,
            }
            .encode(&mut bytes);
            bytes.extend_from_slice(&block_bytes);
        }

        Ok(bytes)
    }

    pub fn import_snapshot(&self, bytes: &[u8]) -> Result<SnapshotEnvelope, StoreError> {
        if bytes.starts_with(&SNAPSHOT_MAGIC) {
            return self.import_binary_snapshot(bytes);
        }
        self.import_legacy_json_snapshot(bytes)
    }
}

impl PhoenixCozoStore {
    fn import_binary_snapshot(&self, bytes: &[u8]) -> Result<SnapshotEnvelope, StoreError> {
        let mut offset = 0usize;
        let header = SnapshotWireHeader::decode(bytes, &mut offset)?;
        if header.magic != SNAPSHOT_MAGIC {
            return Err(StoreError::Snapshot("snapshot magic mismatch".to_owned()));
        }
        if header.version != SNAPSHOT_VERSION {
            return Err(StoreError::Snapshot(format!(
                "unsupported snapshot version: {}",
                header.version
            )));
        }

        let mut relation_ids = Vec::with_capacity(header.relation_count as usize);
        let mut relation_blocks = Vec::with_capacity(header.relation_count as usize);
        for _ in 0..header.relation_count {
            let block_header = SnapshotRelationBlockHeader::decode(bytes, &mut offset)?;
            let payload = read_bytes(bytes, &mut offset, block_header.encoded_len as usize)?;
            relation_ids.push(block_header.relation_id);
            relation_blocks.push((block_header, payload.to_vec()));
        }

        self.clear_relations_by_ids(&relation_ids)?;

        for (block_header, payload) in relation_blocks {
            let decoded = decode_snapshot_block(block_header.codec, &payload)?;
            let rows = decode_snapshot_rows(&decoded)?;
            let relation = ALL_RELATIONS
                .get(block_header.relation_id as usize)
                .ok_or_else(|| {
                    StoreError::Snapshot(format!(
                        "unknown relation id in snapshot: {}",
                        block_header.relation_id
                    ))
                })?
                .name;
            self.put_compact_rows(relation, &rows)?;
        }

        self.put_row(
            "phoenix_schema_state",
            serde_json::json!({
                "version": self.schema_version,
                "updated_at": now_ms(),
            }),
        )?;

        Ok(SnapshotEnvelope {
            schema_version: self.schema_version.to_owned(),
            relation_count: header.relation_count as usize,
            created_at: header.created_at,
            relations: BTreeMap::new(),
            checksum: None,
        })
    }

    fn import_legacy_json_snapshot(&self, bytes: &[u8]) -> Result<SnapshotEnvelope, StoreError> {
        let mut envelope: SnapshotEnvelope = serde_json::from_slice(bytes)
            .map_err(|error| StoreError::Snapshot(error.to_string()))?;
        let relation_names = envelope
            .relations
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        self.clear_relations(&relation_names)?;

        for (relation, rows) in &envelope.relations {
            relation_spec(relation)?;
            self.put_rows(relation, rows)?;
        }

        self.put_row(
            "phoenix_schema_state",
            serde_json::json!({
                "version": envelope.schema_version,
                "updated_at": now_ms(),
            }),
        )?;
        envelope.relations.clear();
        Ok(envelope)
    }
}

fn open_db(config: &StoreConfig) -> Result<cozo::DbInstance, StoreError> {
    match config.mode {
        StorageMode::CozoMem => cozo::DbInstance::new("mem", "", "")
            .map_err(|error| StoreError::Init(error.to_string())),
        StorageMode::CozoSqlite => Err(StoreError::UnsupportedMode(StorageMode::CozoSqlite)),
    }
}

fn relations_for_partition(partition: SnapshotPartition) -> Vec<&'static PhoenixRelationSpec> {
    match partition {
        SnapshotPartition::All => ALL_RELATIONS.iter().collect(),
        SnapshotPartition::Content | SnapshotPartition::Derived => partition
            .relation_names()
            .iter()
            .filter_map(|name| ALL_RELATIONS.iter().find(|relation| relation.name == *name))
            .collect(),
    }
}

fn relation_spec(name: &str) -> Result<&'static PhoenixRelationSpec, StoreError> {
    ALL_RELATIONS
        .iter()
        .find(|relation| relation.name == name)
        .ok_or_else(|| StoreError::UnknownRelation(name.to_owned()))
}

fn relation_id_for_name(name: &str) -> Result<usize, StoreError> {
    ALL_RELATIONS
        .iter()
        .position(|relation| relation.name == name)
        .ok_or_else(|| StoreError::UnknownRelation(name.to_owned()))
}

fn build_create_relation_script(spec: &PhoenixRelationSpec) -> String {
    let keys = spec
        .key_columns()
        .map(column_declaration)
        .collect::<Vec<_>>()
        .join(",\n    ");
    let values = spec
        .value_columns()
        .map(column_declaration)
        .collect::<Vec<_>>()
        .join(",\n    ");

    if values.is_empty() {
        format!(":create {} {{\n    {}\n}}", spec.name, keys)
    } else {
        format!(
            ":create {} {{\n    {}\n    =>\n    {}\n}}",
            spec.name, keys, values
        )
    }
}

fn build_put_compact_payload(
    spec: &PhoenixRelationSpec,
    rows: &[CompactRow],
) -> Result<(String, BTreeMap<String, cozo::DataValue>), StoreError> {
    let column_names = spec
        .columns
        .iter()
        .map(|column| column.name)
        .collect::<Vec<_>>();
    let row_values = rows
        .iter()
        .map(|row| cozo::DataValue::List(row.iter().cloned().collect()))
        .collect::<Vec<_>>();
    let columns = column_names.join(", ");
    let query = format!("?[{columns}] <- $data :put {} {{ {columns} }}", spec.name);
    let params = [("data".to_owned(), cozo::DataValue::List(row_values))]
        .into_iter()
        .collect();
    Ok((query, params))
}

fn build_delete_key_payload(
    spec: &PhoenixRelationSpec,
    rows: &[CompactRow],
) -> Result<(String, BTreeMap<String, cozo::DataValue>), StoreError> {
    let key_columns = spec.key_columns().collect::<Vec<_>>();
    let key_count = key_columns.len();
    let column_names = key_columns
        .iter()
        .map(|column| column.name)
        .collect::<Vec<_>>();
    let row_values = rows
        .iter()
        .map(|row| {
            if row.len() < key_count {
                return Err(StoreError::InvalidRow);
            }
            Ok(cozo::DataValue::List(
                row.iter().take(key_count).cloned().collect(),
            ))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let columns = column_names.join(", ");
    let query = format!("?[{columns}] <- $data :rm {} {{ {columns} }}", spec.name);
    let params = [("data".to_owned(), cozo::DataValue::List(row_values))]
        .into_iter()
        .collect();
    Ok((query, params))
}

fn build_fetch_script(
    spec: &PhoenixRelationSpec,
    columns: Option<&[&str]>,
) -> Result<String, StoreError> {
    let columns = resolve_column_names(spec, columns.unwrap_or(&[]), &[])?;
    let column_list = columns.join(", ");
    Ok(format!(
        "?[{column_list}] := *{}{{{column_list}}}",
        spec.name
    ))
}

fn build_fetch_where_eq_script(
    spec: &PhoenixRelationSpec,
    columns: &[&str],
    filter_column: &str,
) -> Result<String, StoreError> {
    let head_columns = resolve_column_names(spec, columns, &[])?;
    let body_columns = resolve_column_names(spec, columns, &[filter_column])?;
    let head_list = head_columns.join(", ");
    let body_list = body_columns.join(", ");
    Ok(format!(
        "?[{head_list}] := *{}{{{body_list}}}, {filter_column} = $filter",
        spec.name
    ))
}

fn build_fetch_where_in_script(
    spec: &PhoenixRelationSpec,
    columns: &[&str],
    filter_column: &str,
) -> Result<String, StoreError> {
    let head_columns = resolve_column_names(spec, columns, &[])?;
    let body_columns = resolve_column_names(spec, columns, &[filter_column])?;
    let head_list = head_columns.join(", ");
    let body_list = body_columns.join(", ");
    Ok(format!(
        "selected[{filter_column}] <- $data\n?[{head_list}] := selected[{filter_column}], *{}{{{body_list}}}",
        spec.name
    ))
}

fn resolve_column_names<'a>(
    spec: &'a PhoenixRelationSpec,
    requested: &[&str],
    required: &[&str],
) -> Result<Vec<&'a str>, StoreError> {
    let mut names = if requested.is_empty() {
        spec.columns
            .iter()
            .map(|column| column.name)
            .collect::<Vec<_>>()
    } else {
        requested
            .iter()
            .map(|name| resolve_column_name(spec, name))
            .collect::<Result<Vec<_>, _>>()?
    };
    for required_name in required {
        let column = resolve_column_name(spec, required_name)?;
        if !names.contains(&column) {
            names.push(column);
        }
    }
    Ok(names)
}

fn resolve_column_name<'a>(
    spec: &'a PhoenixRelationSpec,
    name: &str,
) -> Result<&'a str, StoreError> {
    spec.columns
        .iter()
        .find(|column| column.name == name)
        .map(|column| column.name)
        .ok_or_else(|| StoreError::MissingColumn {
            relation: spec.name.to_owned(),
            column: name.to_owned(),
        })
}

fn build_clear_script(spec: &PhoenixRelationSpec) -> String {
    let keys = spec
        .key_columns()
        .map(|column| column.name)
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "?[{keys}] := *{}{{{keys}}} :rm {}{{{keys}}}",
        spec.name, spec.name
    )
}

fn null_if_empty(value: &str) -> Value {
    if value.is_empty() {
        Value::Null
    } else {
        Value::String(value.to_owned())
    }
}

fn parse_json_string_array(value: &str) -> Value {
    if value.trim().is_empty() {
        Value::Array(Vec::new())
    } else {
        serde_json::from_str(value).unwrap_or_else(|_| Value::Array(Vec::new()))
    }
}

fn json_to_string_array(value: Option<Value>) -> String {
    let value = value.unwrap_or_else(|| Value::Array(Vec::new()));
    serde_json::to_string(&value).unwrap_or_else(|_| "[]".to_owned())
}

fn datavalue_to_f64(value: &cozo::DataValue) -> Option<f64> {
    match value {
        cozo::DataValue::Num(cozo::Num::Float(float)) => Some(*float),
        cozo::DataValue::Num(cozo::Num::Int(int)) => Some(*int as f64),
        _ => None,
    }
}

fn json_to_string_vec(value: Option<Value>) -> Vec<String> {
    match value {
        Some(Value::Array(items)) => items
            .into_iter()
            .filter_map(|item| item.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

fn network_instance_from_row(row: &CompactRowView<'_>) -> NetworkInstance {
    NetworkInstance {
        id: row.get_str("id").unwrap_or_default().to_owned(),
        name: row.get_str("name").unwrap_or_default().to_owned(),
        schema_id: row.get_str("schema_id").unwrap_or_default().to_owned(),
        network_kind: row.get_str("network_kind").unwrap_or_default().to_owned(),
        network_subtype: row
            .get_str("network_subtype")
            .unwrap_or_default()
            .to_owned(),
        root_folder_id: row.get_str("root_folder_id").unwrap_or_default().to_owned(),
        root_entity_id: row.get_str("root_entity_id").unwrap_or_default().to_owned(),
        namespace: row.get_str("namespace").unwrap_or_default().to_owned(),
        description: row.get_str("description").unwrap_or_default().to_owned(),
        tags: json_to_string_vec(row.get_json("tags")),
        member_count: row.get_u64("member_count").unwrap_or_default() as usize,
        relationship_count: row.get_u64("relationship_count").unwrap_or_default() as usize,
        max_depth: row.get_u64("max_depth").unwrap_or_default() as usize,
        created_at: row.get_i64("created_at").unwrap_or_default(),
        updated_at: row.get_i64("updated_at").unwrap_or_default(),
        group_id: row.get_str("group_id").unwrap_or_default().to_owned(),
        scope_type: row.get_str("scope_type").unwrap_or_default().to_owned(),
        narrative_id: row.get_str("narrative_id").unwrap_or_default().to_owned(),
    }
}

fn column_declaration(column: &PhoenixColumnSpec) -> String {
    if column.optional {
        format!("{}: {}?", column.name, column.ty.as_cozo())
    } else {
        format!("{}: {}", column.name, column.ty.as_cozo())
    }
}

fn row_datavalue(
    spec: &PhoenixRelationSpec,
    column: &PhoenixColumnSpec,
    row: &Map<String, Value>,
) -> Result<cozo::DataValue, StoreError> {
    let value = row.get(column.name).unwrap_or(&Value::Null);
    if value.is_null() && !column.optional {
        return Err(StoreError::MissingColumn {
            relation: spec.name.to_owned(),
            column: column.name.to_owned(),
        });
    }
    let datavalue = match column.ty {
        PhoenixColumnType::VectorF32(expected_dim) => {
            let values = value
                .as_array()
                .ok_or_else(|| {
                    StoreError::Query(format!(
                        "vector column '{}' in relation '{}' must be an array",
                        column.name, spec.name
                    ))
                })?
                .iter()
                .map(|item| {
                    item.as_f64().map(|value| value as f32).ok_or_else(|| {
                        StoreError::Query(format!(
                            "vector column '{}' in relation '{}' must contain only numbers",
                            column.name, spec.name
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if values.len() != expected_dim {
                return Err(StoreError::Query(format!(
                    "vector column '{}' in relation '{}' expected {} values, got {}",
                    column.name,
                    spec.name,
                    expected_dim,
                    values.len()
                )));
            }
            cozo::DataValue::Vec(cozo::Vector::F32(values.into()))
        }
        _ => cozo::DataValue::from(value.clone()),
    };
    Ok(datavalue)
}

fn compact_row_from_value(relation: &str, row: &Value) -> Result<CompactRow, StoreError> {
    let spec = relation_spec(relation)?;
    let object = row.as_object().ok_or(StoreError::InvalidRow)?;
    compact_row_from_object(spec, object)
}

fn compact_row_from_object(
    spec: &PhoenixRelationSpec,
    row: &Map<String, Value>,
) -> Result<CompactRow, StoreError> {
    let mut compact = CompactRow::with_capacity(spec.columns.len());
    for column in spec.columns {
        compact.push(row_datavalue(spec, column, row)?);
    }
    Ok(compact)
}

fn compact_row_key(relation: &str, row: &CompactRow) -> Result<CompactRowKey, StoreError> {
    let spec = relation_spec(relation)?;
    let mut key = CompactRowKey::new();
    for (index, column) in spec.columns.iter().enumerate() {
        if column.key {
            key.push(row[index].clone());
        }
    }
    Ok(key)
}

fn named_rows_to_json_objects(rows: cozo::NamedRows) -> Vec<Value> {
    let headers = rows.headers.clone();
    rows.rows
        .into_iter()
        .map(|row| {
            let object = headers
                .iter()
                .cloned()
                .zip(row.into_iter().map(datavalue_to_json))
                .collect::<Map<String, Value>>();
            Value::Object(object)
        })
        .collect()
}

fn compact_row_to_json_object(spec: &PhoenixRelationSpec, row: &CompactRow) -> Value {
    let object = spec
        .columns
        .iter()
        .zip(row.iter().map(datavalue_to_json_ref))
        .map(|(column, value)| (column.name.to_owned(), value))
        .collect::<Map<String, Value>>();
    Value::Object(object)
}

fn datavalue_to_json(value: cozo::DataValue) -> Value {
    datavalue_to_json_ref(&value)
}

fn datavalue_to_json_ref(value: &cozo::DataValue) -> Value {
    use cozo::DataValue;

    match value {
        DataValue::Null | DataValue::Bot => Value::Null,
        DataValue::Json(json) => serde_json::to_value(json).unwrap_or(Value::Null),
        DataValue::Vec(cozo::Vector::F32(values)) => Value::Array(
            values
                .iter()
                .map(|value| Value::from(*value as f64))
                .collect(),
        ),
        DataValue::Vec(cozo::Vector::F64(values)) => {
            Value::Array(values.iter().map(|value| Value::from(*value)).collect())
        }
        other => match serde_json::to_value(&other) {
            Ok(Value::Object(map)) if map.len() == 1 => {
                let inner = map.into_iter().next().expect("one value").1;
                match inner {
                    Value::Object(map) if map.len() == 1 => {
                        map.into_iter().next().expect("inner value").1
                    }
                    value => value,
                }
            }
            Ok(value) => value,
            Err(_) => Value::Null,
        },
    }
}

fn encode_snapshot_block(rows: &[CompactRow]) -> Result<Vec<u8>, StoreError> {
    let payload = rows
        .iter()
        .map(|row| row.iter().cloned().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    rmp_serde::to_vec(&payload).map_err(|error| StoreError::Snapshot(error.to_string()))
}

fn maybe_compress_snapshot_block(payload: &[u8]) -> (SnapshotCodec, Vec<u8>) {
    if payload.len() < SNAPSHOT_COMPRESS_THRESHOLD {
        return (SnapshotCodec::Raw, payload.to_vec());
    }
    (SnapshotCodec::Lz4, compress_prepend_size(payload))
}

fn decode_snapshot_block(codec: SnapshotCodec, bytes: &[u8]) -> Result<Vec<u8>, StoreError> {
    match codec {
        SnapshotCodec::Raw => Ok(bytes.to_vec()),
        SnapshotCodec::Lz4 => decompress_size_prepended(bytes)
            .map_err(|error| StoreError::Snapshot(error.to_string())),
    }
}

fn decode_snapshot_rows(bytes: &[u8]) -> Result<Vec<CompactRow>, StoreError> {
    let rows: Vec<Vec<cozo::DataValue>> =
        rmp_serde::from_slice(bytes).map_err(|error| StoreError::Snapshot(error.to_string()))?;
    Ok(rows
        .into_iter()
        .map(|row| row.into_iter().collect::<CompactRow>())
        .collect())
}

fn read_array<const N: usize>(bytes: &[u8], offset: &mut usize) -> Result<[u8; N], StoreError> {
    let slice = read_bytes(bytes, offset, N)?;
    let mut array = [0u8; N];
    array.copy_from_slice(slice);
    Ok(array)
}

fn read_u8(bytes: &[u8], offset: &mut usize) -> Result<u8, StoreError> {
    Ok(read_bytes(bytes, offset, 1)?[0])
}

fn read_u16(bytes: &[u8], offset: &mut usize) -> Result<u16, StoreError> {
    Ok(u16::from_le_bytes(read_array::<2>(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, StoreError> {
    Ok(u32::from_le_bytes(read_array::<4>(bytes, offset)?))
}

fn read_i64(bytes: &[u8], offset: &mut usize) -> Result<i64, StoreError> {
    Ok(i64::from_le_bytes(read_array::<8>(bytes, offset)?))
}

fn read_bytes<'a>(bytes: &'a [u8], offset: &mut usize, len: usize) -> Result<&'a [u8], StoreError> {
    let end = offset.saturating_add(len);
    let slice = bytes
        .get(*offset..end)
        .ok_or_else(|| StoreError::Snapshot("unexpected end of snapshot".to_owned()))?;
    *offset = end;
    Ok(slice)
}

fn relation_already_exists(message: &str) -> bool {
    message.contains("exists")
        || message.contains("already")
        || message.contains("conflicts with an existing one")
}

fn nullable_str_to_datavalue(value: Option<&str>) -> cozo::DataValue {
    value
        .map(|value| cozo::DataValue::Str(value.to_owned().into()))
        .unwrap_or(cozo::DataValue::Null)
}

fn json_to_datavalue(value: &Value) -> cozo::DataValue {
    serde_json::from_value(value.clone())
        .map(cozo::DataValue::Json)
        .unwrap_or(cozo::DataValue::Null)
}

fn build_semantic_query_params(
    query_vector: &[f32],
    candidate_count: usize,
) -> BTreeMap<String, cozo::DataValue> {
    [
        (
            "query".to_owned(),
            cozo::DataValue::List(
                query_vector
                    .iter()
                    .map(|value| cozo::DataValue::from(*value as f64))
                    .collect(),
            ),
        ),
        (
            "k".to_owned(),
            cozo::DataValue::from(candidate_count as i64),
        ),
        ("ef".to_owned(), cozo::DataValue::from(50_i64)),
        (
            "model_id".to_owned(),
            cozo::DataValue::Str(SEMANTIC_MODEL_ID.to_owned().into()),
        ),
    ]
    .into_iter()
    .collect()
}

fn datavalue_as_str(value: &cozo::DataValue) -> Option<&str> {
    match value {
        cozo::DataValue::Str(text) => Some(text.as_str()),
        _ => None,
    }
}

fn matches_scope(scope: &ScopeKey, narrative_id: Option<&str>, folder_id: Option<&str>) -> bool {
    if let Some(expected) = scope.narrative_id.as_deref() {
        if narrative_id != Some(expected) {
            return false;
        }
    }
    if let Some(expected) = scope.folder_id.as_deref() {
        if folder_id != Some(expected) {
            return false;
        }
    }
    true
}

fn now_ms() -> i64 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::CORE_RELATIONS;
    use phoenix_types::ScopeKey;

    fn sample_note() -> Value {
        serde_json::json!({
            "id": "note-1",
            "version": 1,
            "world_id": "world-1",
            "title": "Phoenix",
            "content": "Ash and rebirth",
            "markdown_content": "Ash and rebirth",
            "folder_id": "folder-1",
            "entity_kind": null,
            "entity_subtype": null,
            "is_entity": false,
            "is_pinned": false,
            "favorite": false,
            "owner_id": null,
            "narrative_id": "nar-1",
            "order": 1000.0,
            "created_at": 10,
            "updated_at": 10,
            "valid_from": 10,
            "valid_to": null,
            "is_current": true,
            "change_reason": "seed"
        })
    }

    fn semantic_test_vector(primary_index: usize) -> Vec<f32> {
        let mut values = vec![0.0; SEMANTIC_VECTOR_DIM];
        if primary_index < values.len() {
            values[primary_index] = 1.0;
        }
        values
    }

    fn seed_semantic_leaf(
        store: &PhoenixCozoStore,
        chunk_id: i64,
        span_id: &str,
        document_id: &str,
        narrative_id: &str,
        folder_id: &str,
        text: &str,
    ) {
        store
            .put_row(
                "chunks",
                serde_json::json!({
                    "chunk_id": chunk_id,
                    "doc_id": document_id,
                    "level": 0,
                    "start": 0,
                    "end": text.len() as i64,
                    "text": text,
                    "parent_id": null,
                    "scope_narrative": narrative_id,
                    "scope_folder": folder_id,
                    "created_at": 1,
                }),
            )
            .expect("seed chunk");
        store
            .put_row(
                "chunkid_map",
                serde_json::json!({
                    "id": chunk_id,
                    "chunk_key": span_id,
                    "doc_id": document_id,
                    "created_at": 1,
                }),
            )
            .expect("seed chunkid");
    }

    #[test]
    fn core_relations_include_ui_backbone_tables() {
        assert!(CORE_RELATIONS.contains(&"notes"));
        assert!(CORE_RELATIONS.contains(&"entities"));
        assert!(CORE_RELATIONS.contains(&"edges"));
        assert!(CORE_RELATIONS.contains(&"folders"));
        assert!(CORE_RELATIONS.contains(&"graph_vertices"));
    }

    #[test]
    fn schema_init_is_idempotent() {
        let store = PhoenixCozoStore::new().expect("mem store");
        store.init_schema().expect("second init");
        let rows = store
            .fetch_rows("phoenix_schema_state")
            .expect("schema rows");
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn snapshot_export_import_restores_rows() {
        let store = PhoenixCozoStore::new().expect("mem store");
        store.put_row("notes", sample_note()).expect("seed note");

        let snapshot = store.export_snapshot().expect("snapshot bytes");

        let restored = PhoenixCozoStore::new().expect("restored store");
        restored
            .import_snapshot(&snapshot)
            .expect("import snapshot");

        let rows = restored.fetch_rows("notes").expect("restored notes");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"], "note-1");
    }

    #[test]
    fn wasm_in_memory_restore_uses_same_snapshot_format() {
        let store = PhoenixCozoStore::open(StoreConfig {
            mode: StorageMode::CozoMem,
            path: None,
        })
        .expect("mem store");
        store.put_row("notes", sample_note()).expect("seed note");
        let snapshot = store.export_snapshot().expect("snapshot");

        let restored = PhoenixCozoStore::open(StoreConfig {
            mode: StorageMode::CozoMem,
            path: None,
        })
        .expect("restored store");
        let envelope = restored.import_snapshot(&snapshot).expect("import");
        assert_eq!(envelope.schema_version, SCHEMA_VERSION);
        assert_eq!(restored.fetch_rows("notes").expect("rows").len(), 1);
    }

    #[test]
    fn metadata_relations_support_cards_schemas_and_network_views() {
        let store = PhoenixCozoStore::new().expect("mem store");

        store
            .upsert_entity_cards_batch(&[
                EntityCard {
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
                EntityCard {
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
        store
            .upsert_folder_schema(&FolderSchema {
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
        store
            .upsert_network_instance(&NetworkInstance {
                id: "net-1".to_owned(),
                name: "Crew".to_owned(),
                schema_id: "schema-character".to_owned(),
                network_kind: "mindmap".to_owned(),
                network_subtype: "entity".to_owned(),
                root_folder_id: "folder-1".to_owned(),
                root_entity_id: String::new(),
                namespace: "world".to_owned(),
                description: "Crew graph".to_owned(),
                tags: vec!["crew".to_owned()],
                member_count: 2,
                relationship_count: 1,
                max_depth: 2,
                created_at: 100,
                updated_at: 100,
                group_id: String::new(),
                scope_type: "folder".to_owned(),
                narrative_id: "nar-1".to_owned(),
            })
            .expect("network");
        let luffy = NetworkMembership {
            network_id: "net-1".to_owned(),
            entity_id: EntityId("luffy".to_owned()),
            x: 10.0,
            y: 20.0,
            fixed: true,
        };
        let zoro = NetworkMembership {
            network_id: "net-1".to_owned(),
            entity_id: EntityId("zoro".to_owned()),
            x: 30.0,
            y: 40.0,
            fixed: false,
        };
        store
            .upsert_network_memberships(&[luffy.clone(), zoro.clone()])
            .expect("members");
        let relationship = NetworkRelationship {
            network_id: "net-1".to_owned(),
            source_entity_id: EntityId("luffy".to_owned()),
            target_entity_id: EntityId("zoro".to_owned()),
            relationship_id: "edge-1".to_owned(),
        };
        store
            .upsert_network_relationships(std::slice::from_ref(&relationship))
            .expect("relationships");

        let cards = store
            .get_entity_cards(&EntityId("CHARACTER".to_owned()))
            .expect("cards");
        let schema = store
            .get_folder_schema("schema-character")
            .expect("schema fetch")
            .expect("schema exists");
        let network = store
            .get_network_instance("net-1")
            .expect("network fetch")
            .expect("network exists");
        assert_eq!(cards[0].card_id, "traits");
        assert_eq!(schema.allowed_note_types, "[\"bio\",\"scene\"]");
        assert_eq!(network.name, "Crew");

        store
            .delete_network_memberships(std::slice::from_ref(&zoro))
            .expect("delete stale member");
        store
            .delete_network_relationships(std::slice::from_ref(&relationship))
            .expect("delete stale relationship");

        let members = store.get_network_members("net-1").expect("members");
        let relationships = store
            .get_network_relationships("net-1")
            .expect("relationships");
        assert_eq!(members.len(), 1);
        assert_eq!(members[0].entity_id.0, "luffy");
        assert!(relationships.is_empty());
    }

    #[test]
    fn semantic_document_vectors_upsert_and_query_are_scoped_and_replaceable() {
        let store = PhoenixCozoStore::new().expect("mem store");
        seed_semantic_leaf(
            &store,
            1,
            "doc-a:leaf-1",
            "doc-a",
            "nar-a",
            "folder-a",
            "alpha harbor",
        );
        seed_semantic_leaf(
            &store,
            2,
            "doc-b:leaf-1",
            "doc-b",
            "nar-b",
            "folder-b",
            "beta forest",
        );

        let vector_a = semantic_test_vector(0);
        let vector_b = semantic_test_vector(1);
        store
            .upsert_semantic_document_vectors(&[
                SemanticDocumentVectorRow {
                    document_id: "doc-a",
                    values: &vector_a,
                    model_id: SEMANTIC_MODEL_ID,
                    leaf_count: 1,
                    evidence_refs: &[],
                    updated_at: 10,
                },
                SemanticDocumentVectorRow {
                    document_id: "doc-b",
                    values: &vector_b,
                    model_id: SEMANTIC_MODEL_ID,
                    leaf_count: 2,
                    evidence_refs: &[],
                    updated_at: 10,
                },
            ])
            .expect("upsert docs");

        let scoped = store
            .query_semantic_documents(
                &vector_a,
                &ScopeKey {
                    narrative_id: Some("nar-a".to_owned()),
                    ..ScopeKey::default()
                },
                4,
                8,
            )
            .expect("query docs");
        assert_eq!(scoped.len(), 1);
        assert_eq!(scoped[0].document_id, "doc-a");
        assert_eq!(scoped[0].leaf_count, 1);

        store
            .upsert_semantic_document_vectors(&[SemanticDocumentVectorRow {
                document_id: "doc-a",
                values: &vector_b,
                model_id: SEMANTIC_MODEL_ID,
                leaf_count: 3,
                evidence_refs: &[],
                updated_at: 11,
            }])
            .expect("replace doc");
        let replaced = store
            .query_semantic_documents(
                &vector_b,
                &ScopeKey {
                    narrative_id: Some("nar-a".to_owned()),
                    ..ScopeKey::default()
                },
                4,
                8,
            )
            .expect("query replaced doc");
        assert_eq!(replaced.len(), 1);
        assert_eq!(replaced[0].document_id, "doc-a");
        assert_eq!(replaced[0].leaf_count, 3);
    }

    #[test]
    fn semantic_node_prototypes_upsert_and_query_round_trip_evidence() {
        let store = PhoenixCozoStore::new().expect("mem store");
        let vector_a = semantic_test_vector(0);
        let vector_b = semantic_test_vector(1);
        let refs_a = vec!["graph_vertex:entity::ryan".to_owned()];
        let refs_b = vec!["graph_vertex:entity::len".to_owned()];

        store
            .upsert_semantic_node_vectors(&[
                SemanticNodeVectorRow {
                    node_id: "entity::ryan",
                    node_kind: "entity",
                    document_id: Some("doc-a"),
                    narrative_id: Some("nar-a"),
                    folder_id: Some("folder-a"),
                    values: &vector_a,
                    model_id: SEMANTIC_MODEL_ID,
                    evidence_refs: &refs_a,
                    updated_at: 10,
                },
                SemanticNodeVectorRow {
                    node_id: "entity::len",
                    node_kind: "entity",
                    document_id: Some("doc-b"),
                    narrative_id: Some("nar-a"),
                    folder_id: Some("folder-a"),
                    values: &vector_b,
                    model_id: SEMANTIC_MODEL_ID,
                    evidence_refs: &refs_b,
                    updated_at: 10,
                },
            ])
            .expect("upsert node prototypes");

        let neighbors = store
            .query_semantic_node_neighbors(
                &vector_a,
                &ScopeKey {
                    narrative_id: Some("nar-a".to_owned()),
                    folder_id: Some("folder-a".to_owned()),
                    ..ScopeKey::default()
                },
                "entity",
                Some("entity::ryan"),
                4,
                8,
            )
            .expect("query node neighbors");

        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].node_id, "entity::len");
        assert_eq!(neighbors[0].node_kind, "entity");
        assert_eq!(neighbors[0].evidence_refs, refs_b);
    }

    #[test]
    fn semantic_neighbors_can_be_filtered_to_document_shortlist() {
        let store = PhoenixCozoStore::new().expect("mem store");
        seed_semantic_leaf(
            &store,
            1,
            "doc-a:leaf-1",
            "doc-a",
            "nar-a",
            "folder-a",
            "alpha harbor",
        );
        seed_semantic_leaf(
            &store,
            2,
            "doc-b:leaf-1",
            "doc-b",
            "nar-a",
            "folder-a",
            "alpha harbor annex",
        );

        let vector = semantic_test_vector(0);
        store
            .upsert_semantic_vectors(&[
                SemanticVectorRow {
                    span_id: "doc-a:leaf-1",
                    values: &vector,
                    model_id: SEMANTIC_MODEL_ID,
                    updated_at: 10,
                },
                SemanticVectorRow {
                    span_id: "doc-b:leaf-1",
                    values: &vector,
                    model_id: SEMANTIC_MODEL_ID,
                    updated_at: 10,
                },
            ])
            .expect("upsert leaf vectors");

        let filtered = store
            .query_semantic_neighbors_in_documents(
                &vector,
                &ScopeKey {
                    narrative_id: Some("nar-a".to_owned()),
                    ..ScopeKey::default()
                },
                &["doc-b".to_owned()],
                5,
                10,
            )
            .expect("filtered query");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].span_id, "doc-b:leaf-1");
    }
}
