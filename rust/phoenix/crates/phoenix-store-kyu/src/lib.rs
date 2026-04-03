use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use kyu_graph::{Database, DeltaBatchBuilder, DeltaValue, TypedValue};
use phoenix_graph::{
    GraphEdgeRecord, GraphLayer, GraphMutationBatch, GraphVertexRecord, GraptorGraph,
};
use phoenix_store_cozo::{
    schema::{
        PhoenixColumnSpec, PhoenixColumnType, PhoenixRelationSpec, ALL_RELATIONS,
        CONTENT_SNAPSHOT_RELATIONS, DERIVED_SNAPSHOT_RELATIONS,
    },
    SnapshotEnvelope, SnapshotPartition, StoreError,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const KYU_SCHEMA_VERSION: &str = "phoenix.kyu.v1";

pub const KYU_COVERED_RELATIONS: &[&str] = &[
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

pub const KYU_GRAPH_COMPAT_RELATIONS: &[&str] = &[
    "graph_vertices",
    "graph_edges",
    "graph_candidate_edges",
    "graph_node_index",
    "graph_properties",
    "graph_vertex_labels",
];

const PHOENIX_GRAPH_CHECKPOINT_META: &[PhoenixColumnSpec] = &[
    PhoenixColumnSpec::new("checkpoint_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("generation", PhoenixColumnType::Int, false, false),
    PhoenixColumnSpec::new("source_revision", PhoenixColumnType::String, false, false),
    PhoenixColumnSpec::new("created_at", PhoenixColumnType::Int, false, false),
];

const PHOENIX_GRAPH_VERTEX_SNAPSHOT: &[PhoenixColumnSpec] = &[
    PhoenixColumnSpec::new("checkpoint_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("vertex_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("record_json", PhoenixColumnType::Json, false, false),
    PhoenixColumnSpec::new("created_at", PhoenixColumnType::Int, false, false),
];

const PHOENIX_GRAPH_EDGE_SNAPSHOT: &[PhoenixColumnSpec] = &[
    PhoenixColumnSpec::new("checkpoint_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("layer", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("source_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("target_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("edge_type", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("record_json", PhoenixColumnType::Json, false, false),
    PhoenixColumnSpec::new("created_at", PhoenixColumnType::Int, false, false),
];

const PHOENIX_GRAPH_DELTA_JOURNAL: &[PhoenixColumnSpec] = &[
    PhoenixColumnSpec::new("entry_id", PhoenixColumnType::String, false, true),
    PhoenixColumnSpec::new("generation", PhoenixColumnType::Int, false, false),
    PhoenixColumnSpec::new("source_revision", PhoenixColumnType::String, false, false),
    PhoenixColumnSpec::new("batch_json", PhoenixColumnType::Json, true, false),
    PhoenixColumnSpec::new("commit_id", PhoenixColumnType::String, true, false),
    PhoenixColumnSpec::new("created_at", PhoenixColumnType::Int, false, false),
];

const INTERNAL_TABLES: &[PhoenixRelationSpec] = &[
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

pub struct PhoenixKyuStore {
    db: Database,
    path: PathBuf,
}

impl PhoenixKyuStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StoreError> {
        let path = path.as_ref().to_path_buf();
        let db = Database::open(&path).map_err(|error| StoreError::Init(error.to_string()))?;
        Ok(Self { db, path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn is_covered_relation(relation: &str) -> bool {
        KYU_COVERED_RELATIONS.contains(&relation)
    }

    pub fn is_graph_compat_relation(relation: &str) -> bool {
        KYU_GRAPH_COMPAT_RELATIONS.contains(&relation)
    }

    pub fn is_supported_relation(relation: &str) -> bool {
        Self::is_covered_relation(relation) || internal_relation_spec(relation).is_some()
    }

    fn create_table(&self, spec: &PhoenixRelationSpec) -> Result<(), StoreError> {
        let mut declarations = vec!["__pk STRING".to_owned()];
        declarations.extend(spec.columns.iter().map(kyu_column_declaration));
        declarations.push("PRIMARY KEY (__pk)".to_owned());
        let query = format!(
            "CREATE NODE TABLE {} ({})",
            spec.name,
            declarations.join(", ")
        );
        match self.db.connect().query(&query) {
            Ok(_) => Ok(()),
            Err(error) if relation_already_exists(&error.to_string()) => Ok(()),
            Err(error) => Err(StoreError::Schema(error.to_string())),
        }
    }

    fn query_rows(
        &self,
        cypher: &str,
        columns: &[&PhoenixColumnSpec],
        _spec: &PhoenixRelationSpec,
    ) -> Result<Vec<Value>, StoreError> {
        let result = self
            .db
            .connect()
            .query(cypher)
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut rows = Vec::with_capacity(result.num_rows());
        for index in 0..result.num_rows() {
            let row = result.row(index);
            let mut object = serde_json::Map::with_capacity(columns.len());
            for (column_index, column_spec) in columns.iter().enumerate() {
                let value = row
                    .get(column_index)
                    .map(|value| typed_value_to_json(value, column_spec.ty))
                    .unwrap_or(Value::Null);
                object.insert(column_spec.name.to_owned(), value);
            }
            rows.push(Value::Object(object));
        }
        Ok(rows)
    }

    fn row_primary_key(
        &self,
        spec: &PhoenixRelationSpec,
        row: &Value,
    ) -> Result<String, StoreError> {
        let object = row.as_object().ok_or(StoreError::InvalidRow)?;
        let key_values = spec
            .key_columns()
            .map(|column| {
                object
                    .get(column.name)
                    .cloned()
                    .ok_or_else(|| StoreError::MissingColumn {
                        relation: spec.name.to_owned(),
                        column: column.name.to_owned(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        serde_json::to_string(&key_values).map_err(|error| StoreError::Query(error.to_string()))
    }

    fn delta_builder_for_row(
        &self,
        spec: &PhoenixRelationSpec,
        row: &Value,
    ) -> Result<DeltaBatchBuilder, StoreError> {
        let object = row.as_object().ok_or(StoreError::InvalidRow)?;
        let pk = self.row_primary_key(spec, row)?;
        let props = spec
            .columns
            .iter()
            .map(|column| {
                let value = object.get(column.name).cloned().unwrap_or(Value::Null);
                let typed = json_to_typed_value(spec.name, column, value)?;
                Ok((kyu_column_name(column.name), typed))
            })
            .collect::<Result<Vec<_>, StoreError>>()?;
        Ok(DeltaBatchBuilder::new(spec.name, now_ms_u64()).upsert_node(
            spec.name,
            pk,
            Vec::new(),
            props,
        ))
    }

    fn fetch_internal_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        let spec = internal_relation_spec(relation)
            .ok_or_else(|| StoreError::UnknownRelation(relation.to_owned()))?;
        let columns = spec.columns.iter().collect::<Vec<_>>();
        let select = columns
            .iter()
            .map(|column| format!("n.{}", kyu_column_name(column.name)))
            .collect::<Vec<_>>()
            .join(", ");
        let query = format!("MATCH (n:{relation}) RETURN {select} ORDER BY n.__pk");
        self.query_rows(&query, &columns, spec)
    }

    pub fn compact_graph_journal(&self, up_to_generation: u64) -> Result<(), StoreError> {
        let rows = self
            .fetch_rows("phoenix_graph_delta_journal")?
            .into_iter()
            .filter(|row| {
                row.get("generation")
                    .and_then(Value::as_u64)
                    .unwrap_or_default()
                    <= up_to_generation
            })
            .collect::<Vec<_>>();
        let _ = self.delete_rows("phoenix_graph_delta_journal", &rows)?;
        Ok(())
    }
}

impl PhoenixNativeRowStore for PhoenixKyuStore {
    fn init_schema(&self) -> Result<(), StoreError> {
        for relation in KYU_COVERED_RELATIONS {
            let spec = covered_relation_spec(relation)?;
            self.create_table(spec)?;
        }
        for relation in INTERNAL_TABLES {
            self.create_table(relation)?;
        }
        self.put_row(
            "phoenix_schema_state",
            json!({
                "version": KYU_SCHEMA_VERSION,
                "updated_at": now_ms(),
            }),
        )?;
        Ok(())
    }

    fn relation_names(&self) -> Vec<&'static str> {
        KYU_COVERED_RELATIONS.to_vec()
    }

    fn relation_counts(&self) -> Result<Vec<(String, usize)>, StoreError> {
        let mut counts = Vec::with_capacity(KYU_COVERED_RELATIONS.len());
        for relation in KYU_COVERED_RELATIONS {
            counts.push(((*relation).to_owned(), self.fetch_rows(relation)?.len()));
        }
        Ok(counts)
    }

    fn fetch_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        let spec = if let Ok(spec) = covered_relation_spec(relation) {
            spec
        } else {
            return self.fetch_internal_rows(relation);
        };
        let columns = spec.columns.iter().collect::<Vec<_>>();
        let select = columns
            .iter()
            .map(|column| format!("n.{}", kyu_column_name(column.name)))
            .collect::<Vec<_>>()
            .join(", ");
        let query = format!("MATCH (n:{relation}) RETURN {select} ORDER BY n.__pk");
        self.query_rows(&query, &columns, spec)
    }

    fn put_row(&self, relation: &str, row: Value) -> Result<(), StoreError> {
        let spec = supported_relation_spec(relation)?;
        let batch = self.delta_builder_for_row(spec, &row)?.build();
        self.db
            .connect()
            .apply_delta(batch)
            .map(|_| ())
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    fn put_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError> {
        for row in rows {
            self.put_row(relation, row.clone())?;
        }
        Ok(())
    }

    fn replace_relation_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError> {
        self.clear_relations(&[relation])?;
        self.put_rows(relation, rows)
    }

    fn delete_rows(&self, relation: &str, rows: &[Value]) -> Result<usize, StoreError> {
        let spec = supported_relation_spec(relation)?;
        let mut deleted = 0usize;
        for row in rows {
            let pk = self.row_primary_key(spec, row)?;
            let batch = DeltaBatchBuilder::new(relation, now_ms_u64())
                .delete_node(relation, pk)
                .build();
            self.db
                .connect()
                .apply_delta(batch)
                .map_err(|error| StoreError::Query(error.to_string()))?;
            deleted += 1;
        }
        Ok(deleted)
    }

    fn clear_relations(&self, relations: &[&str]) -> Result<(), StoreError> {
        for relation in relations {
            let rows = self.fetch_rows(relation)?;
            let _ = self.delete_rows(relation, &rows)?;
        }
        Ok(())
    }

    fn export_snapshot_partition(
        &self,
        partition: SnapshotPartition,
    ) -> Result<Vec<u8>, StoreError> {
        let relation_names = match partition {
            SnapshotPartition::All => {
                let mut relations = KYU_COVERED_RELATIONS
                    .iter()
                    .map(|relation| (*relation).to_owned())
                    .collect::<Vec<_>>();
                relations.extend(
                    INTERNAL_TABLES
                        .iter()
                        .map(|relation| relation.name.to_owned()),
                );
                relations
            }
            SnapshotPartition::Content => CONTENT_SNAPSHOT_RELATIONS
                .iter()
                .filter(|relation| Self::is_covered_relation(relation))
                .map(|relation| (*relation).to_owned())
                .collect(),
            SnapshotPartition::Derived => {
                let mut relations = DERIVED_SNAPSHOT_RELATIONS
                    .iter()
                    .filter(|relation| Self::is_covered_relation(relation))
                    .map(|relation| (*relation).to_owned())
                    .collect::<Vec<_>>();
                relations.extend(
                    INTERNAL_TABLES
                        .iter()
                        .map(|relation| relation.name.to_owned()),
                );
                relations
            }
        };
        let mut relations = BTreeMap::new();
        for relation in relation_names {
            relations.insert(relation.clone(), self.fetch_rows(&relation)?);
        }
        let envelope = SnapshotEnvelope {
            schema_version: KYU_SCHEMA_VERSION.to_owned(),
            relation_count: relations.len(),
            created_at: now_ms(),
            relations,
            checksum: None,
        };
        serde_json::to_vec(&envelope).map_err(|error| StoreError::Snapshot(error.to_string()))
    }

    fn import_snapshot(&self, bytes: &[u8]) -> Result<SnapshotEnvelope, StoreError> {
        let mut envelope: SnapshotEnvelope = serde_json::from_slice(bytes)
            .map_err(|error| StoreError::Snapshot(error.to_string()))?;
        let relation_names = envelope
            .relations
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>();
        self.clear_relations(&relation_names)?;
        for (relation, rows) in &envelope.relations {
            self.put_rows(relation, rows)?;
        }
        self.put_row(
            "phoenix_schema_state",
            json!({
                "version": envelope.schema_version.clone(),
                "updated_at": now_ms(),
            }),
        )?;
        envelope.relations.clear();
        Ok(envelope)
    }
}

impl PhoenixGraphDurabilityStore for PhoenixKyuStore {
    fn load_graph_checkpoint(&self) -> Result<Option<GraphCheckpointData>, StoreError> {
        let meta_rows = self.fetch_rows("phoenix_graph_checkpoint_meta")?;
        let Some(meta_row) = meta_rows.into_iter().max_by_key(|row| {
            row.get("generation")
                .and_then(Value::as_u64)
                .unwrap_or_default()
        }) else {
            return Ok(None);
        };
        let meta = GraphCheckpointMeta {
            checkpoint_id: meta_row
                .get("checkpoint_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            generation: meta_row
                .get("generation")
                .and_then(Value::as_u64)
                .unwrap_or_default(),
            source_revision: meta_row
                .get("source_revision")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned(),
            created_at: meta_row
                .get("created_at")
                .and_then(Value::as_i64)
                .unwrap_or_default(),
        };
        let vertex_rows = self.fetch_rows("phoenix_graph_vertex_snapshot")?;
        let edge_rows = self.fetch_rows("phoenix_graph_edge_snapshot")?;
        let vertices = vertex_rows
            .into_iter()
            .filter(|row| {
                row.get("checkpoint_id").and_then(Value::as_str)
                    == Some(meta.checkpoint_id.as_str())
            })
            .filter_map(|row| row.get("record_json").cloned())
            .map(serde_json::from_value::<GraphVertexRecord>)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| StoreError::Query(error.to_string()))?;
        let mut asserted_edges = Vec::new();
        let mut candidate_edges = Vec::new();
        for row in edge_rows.into_iter().filter(|row| {
            row.get("checkpoint_id").and_then(Value::as_str) == Some(meta.checkpoint_id.as_str())
        }) {
            let Some(record_value) = row.get("record_json").cloned() else {
                continue;
            };
            let record: GraphEdgeRecord = serde_json::from_value(record_value)
                .map_err(|error| StoreError::Query(error.to_string()))?;
            match row.get("layer").and_then(Value::as_str) {
                Some("candidate") => candidate_edges.push(record),
                _ => asserted_edges.push(record),
            }
        }
        Ok(Some(GraphCheckpointData {
            meta,
            asserted_batch: GraphMutationBatch {
                layer: GraphLayer::Asserted,
                scope: phoenix_graph::GraphMutationScope::Full,
                vertices,
                edges: asserted_edges,
            },
            candidate_batch: GraphMutationBatch {
                layer: GraphLayer::Candidate,
                scope: phoenix_graph::GraphMutationScope::Full,
                vertices: Vec::new(),
                edges: candidate_edges,
            },
        }))
    }

    fn write_graph_checkpoint(
        &self,
        generation: u64,
        source_revision: &str,
        graph: &GraptorGraph,
    ) -> Result<GraphCheckpointData, StoreError> {
        let checkpoint_id = format!("checkpoint-{generation}");
        let created_at = now_ms();
        let meta = GraphCheckpointMeta {
            checkpoint_id: checkpoint_id.clone(),
            generation,
            source_revision: source_revision.to_owned(),
            created_at,
        };
        let vertices = graph
            .vertices
            .values()
            .map(GraphVertexRecord::from)
            .collect::<Vec<_>>();
        let mut seen_edges = BTreeSet::new();
        let mut asserted_edges = Vec::new();
        let mut candidate_edges = Vec::new();
        for edges in graph.outgoing.values() {
            for edge in edges {
                let key = (
                    edge.source_id.clone(),
                    edge.target_id.clone(),
                    edge.edge_type.clone(),
                    matches!(edge.layer, GraphLayer::Candidate),
                );
                if !seen_edges.insert(key) {
                    continue;
                }
                let record = GraphEdgeRecord::from(edge);
                match edge.layer {
                    GraphLayer::Asserted => asserted_edges.push(record),
                    GraphLayer::Candidate => candidate_edges.push(record),
                }
            }
        }
        self.clear_relations(&[
            "phoenix_graph_checkpoint_meta",
            "phoenix_graph_vertex_snapshot",
            "phoenix_graph_edge_snapshot",
        ])?;
        self.put_row(
            "phoenix_graph_checkpoint_meta",
            json!({
                "checkpoint_id": checkpoint_id,
                "generation": generation as i64,
                "source_revision": source_revision,
                "created_at": created_at,
            }),
        )?;
        for vertex in &vertices {
            self.put_row(
                "phoenix_graph_vertex_snapshot",
                json!({
                    "checkpoint_id": meta.checkpoint_id,
                    "vertex_id": vertex.id,
                    "record_json": vertex,
                    "created_at": created_at,
                }),
            )?;
        }
        for edge in &asserted_edges {
            self.put_row(
                "phoenix_graph_edge_snapshot",
                json!({
                    "checkpoint_id": meta.checkpoint_id,
                    "layer": "asserted",
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type,
                    "record_json": edge,
                    "created_at": created_at,
                }),
            )?;
        }
        for edge in &candidate_edges {
            self.put_row(
                "phoenix_graph_edge_snapshot",
                json!({
                    "checkpoint_id": meta.checkpoint_id,
                    "layer": "candidate",
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "edge_type": edge.edge_type,
                    "record_json": edge,
                    "created_at": created_at,
                }),
            )?;
        }
        let _ = self.compact_graph_journal(generation);
        Ok(GraphCheckpointData {
            meta,
            asserted_batch: GraphMutationBatch {
                layer: GraphLayer::Asserted,
                scope: phoenix_graph::GraphMutationScope::Full,
                vertices,
                edges: asserted_edges,
            },
            candidate_batch: GraphMutationBatch {
                layer: GraphLayer::Candidate,
                scope: phoenix_graph::GraphMutationScope::Full,
                vertices: Vec::new(),
                edges: candidate_edges,
            },
        })
    }

    fn load_graph_journal_after(
        &self,
        generation: u64,
    ) -> Result<Vec<GraphJournalEntry>, StoreError> {
        let mut entries = self
            .fetch_rows("phoenix_graph_delta_journal")?
            .into_iter()
            .filter_map(|row| {
                let row_generation = row.get("generation").and_then(Value::as_u64)?;
                (row_generation > generation).then_some((row_generation, row))
            })
            .collect::<Vec<_>>();
        entries.sort_by(|left, right| {
            left.0.cmp(&right.0).then_with(|| {
                left.1
                    .get("created_at")
                    .and_then(Value::as_i64)
                    .unwrap_or_default()
                    .cmp(
                        &right
                            .1
                            .get("created_at")
                            .and_then(Value::as_i64)
                            .unwrap_or_default(),
                    )
            })
        });
        entries
            .into_iter()
            .map(|(_, row)| {
                let batch = row
                    .get("batch_json")
                    .cloned()
                    .filter(|value| !value.is_null())
                    .map(serde_json::from_value::<GraphMutationBatch>)
                    .transpose()
                    .map_err(|error| StoreError::Query(error.to_string()))?;
                Ok(GraphJournalEntry {
                    generation: row
                        .get("generation")
                        .and_then(Value::as_u64)
                        .unwrap_or_default(),
                    source_revision: row
                        .get("source_revision")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                    batch,
                    commit_id: row
                        .get("commit_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    created_at: row
                        .get("created_at")
                        .and_then(Value::as_i64)
                        .unwrap_or_default(),
                })
            })
            .collect()
    }

    fn append_graph_batch(
        &self,
        generation: u64,
        source_revision: &str,
        batch: &GraphMutationBatch,
        created_at: i64,
    ) -> Result<(), StoreError> {
        self.put_row(
            "phoenix_graph_delta_journal",
            json!({
                "entry_id": format!("batch-{generation}-{created_at}"),
                "generation": generation as i64,
                "source_revision": source_revision,
                "batch_json": batch,
                "commit_id": null,
                "created_at": created_at,
            }),
        )
    }

    fn append_commit_marker(
        &self,
        generation: u64,
        source_revision: &str,
        commit_id: &str,
        created_at: i64,
    ) -> Result<(), StoreError> {
        self.put_row(
            "phoenix_graph_delta_journal",
            json!({
                "entry_id": format!("commit-{generation}-{commit_id}"),
                "generation": generation as i64,
                "source_revision": source_revision,
                "batch_json": null,
                "commit_id": commit_id,
                "created_at": created_at,
            }),
        )
    }

    fn generation_for_commit(&self, commit_id: &str) -> Result<Option<u64>, StoreError> {
        Ok(self
            .fetch_rows("phoenix_graph_delta_journal")?
            .into_iter()
            .filter(|row| row.get("commit_id").and_then(Value::as_str) == Some(commit_id))
            .filter_map(|row| row.get("generation").and_then(Value::as_u64))
            .max())
    }

    fn current_generation(&self) -> Result<u64, StoreError> {
        let checkpoint_generation = self
            .load_graph_checkpoint()?
            .map(|checkpoint| checkpoint.meta.generation)
            .unwrap_or_default();
        let journal_generation = self
            .fetch_rows("phoenix_graph_delta_journal")?
            .into_iter()
            .filter_map(|row| row.get("generation").and_then(Value::as_u64))
            .max()
            .unwrap_or_default();
        Ok(checkpoint_generation.max(journal_generation))
    }

    fn journal_len(&self) -> Result<usize, StoreError> {
        Ok(self.fetch_rows("phoenix_graph_delta_journal")?.len())
    }
}

fn relation_already_exists(message: &str) -> bool {
    let lower = message.to_ascii_lowercase();
    lower.contains("already exists") || lower.contains("exists")
}

fn covered_relation_spec(name: &str) -> Result<&'static PhoenixRelationSpec, StoreError> {
    ALL_RELATIONS
        .iter()
        .find(|relation| KYU_COVERED_RELATIONS.contains(&relation.name) && relation.name == name)
        .ok_or_else(|| StoreError::UnknownRelation(name.to_owned()))
}

fn internal_relation_spec(name: &str) -> Option<&'static PhoenixRelationSpec> {
    INTERNAL_TABLES
        .iter()
        .find(|relation| relation.name == name)
}

fn supported_relation_spec(name: &str) -> Result<&'static PhoenixRelationSpec, StoreError> {
    covered_relation_spec(name).or_else(|_| {
        internal_relation_spec(name).ok_or_else(|| StoreError::UnknownRelation(name.to_owned()))
    })
}

fn kyu_column_declaration(column: &PhoenixColumnSpec) -> String {
    format!(
        "{} {}",
        kyu_column_name(column.name),
        kyu_type_name(column.ty)
    )
}

fn kyu_column_name(column: &str) -> String {
    format!("c_{column}")
}

fn kyu_type_name(ty: PhoenixColumnType) -> &'static str {
    match ty {
        PhoenixColumnType::String => "STRING",
        PhoenixColumnType::Int => "INT64",
        PhoenixColumnType::Float => "DOUBLE",
        PhoenixColumnType::Bool => "BOOL",
        PhoenixColumnType::Json => "STRING",
        PhoenixColumnType::VectorF32(_) => "STRING",
    }
}

fn json_to_typed_value(
    relation: &str,
    column: &PhoenixColumnSpec,
    value: Value,
) -> Result<DeltaValue, StoreError> {
    match (column.ty, value) {
        (_, Value::Null) if column.optional || column.ty == PhoenixColumnType::Json => {
            Ok(TypedValue::Null)
        }
        (PhoenixColumnType::String, Value::String(value)) => {
            Ok(TypedValue::from(Value::String(value)))
        }
        (PhoenixColumnType::String, value) => Err(StoreError::Query(format!(
            "relation '{relation}' column '{}' expected string, got {value}",
            column.name
        ))),
        (PhoenixColumnType::Int, Value::Number(number)) => {
            number.as_i64().map(TypedValue::Int64).ok_or_else(|| {
                StoreError::Query(format!(
                    "relation '{relation}' column '{}' expected int64-compatible number",
                    column.name
                ))
            })
        }
        (PhoenixColumnType::Int, value) => Err(StoreError::Query(format!(
            "relation '{relation}' column '{}' expected number, got {value}",
            column.name
        ))),
        (PhoenixColumnType::Float, Value::Number(number)) => {
            number.as_f64().map(TypedValue::Double).ok_or_else(|| {
                StoreError::Query(format!(
                    "relation '{relation}' column '{}' expected float-compatible number",
                    column.name
                ))
            })
        }
        (PhoenixColumnType::Float, value) => Err(StoreError::Query(format!(
            "relation '{relation}' column '{}' expected float, got {value}",
            column.name
        ))),
        (PhoenixColumnType::Bool, Value::Bool(value)) => Ok(TypedValue::Bool(value)),
        (PhoenixColumnType::Bool, value) => Err(StoreError::Query(format!(
            "relation '{relation}' column '{}' expected bool, got {value}",
            column.name
        ))),
        (PhoenixColumnType::Json, value) => serde_json::to_string(&value)
            .map(|value| TypedValue::from(Value::String(value)))
            .map_err(|error| StoreError::Query(error.to_string())),
        (PhoenixColumnType::VectorF32(_), value) => serde_json::to_string(&value)
            .map(|value| TypedValue::from(Value::String(value)))
            .map_err(|error| StoreError::Query(error.to_string())),
    }
}

fn typed_value_to_json(value: &TypedValue, ty: PhoenixColumnType) -> Value {
    match ty {
        PhoenixColumnType::Json | PhoenixColumnType::VectorF32(_) => match value {
            TypedValue::Null => Value::Null,
            TypedValue::String(encoded) => {
                serde_json::from_str(encoded.as_str()).unwrap_or(Value::Null)
            }
            other => Value::from(other.clone()),
        },
        _ => Value::from(value.clone()),
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

fn now_ms_u64() -> u64 {
    now_ms().max(0) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_store() -> PhoenixKyuStore {
        let path = std::env::temp_dir().join(format!(
            "phoenix-kyu-test-{}-{}",
            std::process::id(),
            now_ms()
        ));
        let store = PhoenixKyuStore::open(path).expect("open store");
        store.init_schema().expect("init schema");
        store
    }

    #[test]
    fn note_round_trip_rows() {
        let store = temp_store();
        store
            .put_row(
                "notes",
                json!({
                    "id": "note-1",
                    "version": 1,
                    "world_id": "world-1",
                    "title": "Title",
                    "content": "Body",
                    "markdown_content": null,
                    "folder_id": null,
                    "entity_kind": null,
                    "entity_subtype": null,
                    "is_entity": false,
                    "is_pinned": false,
                    "favorite": false,
                    "owner_id": null,
                    "narrative_id": null,
                    "order": null,
                    "created_at": 1,
                    "updated_at": 1,
                    "valid_from": 1,
                    "valid_to": null,
                    "is_current": true,
                    "change_reason": null
                }),
            )
            .expect("put note");
        let rows = store.fetch_rows("notes").expect("fetch notes");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].get("id").and_then(Value::as_str), Some("note-1"));
    }

    #[test]
    fn graph_checkpoint_round_trip() {
        let store = temp_store();
        let mut graph = GraptorGraph::default();
        graph.vertices.insert(
            "doc::1".to_owned(),
            phoenix_graph::GraptorVertex {
                id: "doc::1".to_owned(),
                kind: "document".to_owned(),
                weight: 1,
                value: json!({"kind":"document","documentId":"doc-1"}),
                attributes: json!({"documentId":"doc-1"}),
                document_id: Some("doc-1".to_owned()),
                ..Default::default()
            },
        );
        graph.outgoing.insert(
            "doc::1".to_owned(),
            vec![phoenix_graph::GraptorEdge {
                source_id: "doc::1".to_owned(),
                target_id: "entity::1".to_owned(),
                edge_type: "mentions".to_owned(),
                weight: 1,
                attributes: json!({"documentId":"doc-1"}),
                layer: GraphLayer::Asserted,
                ..Default::default()
            }],
        );
        store
            .write_graph_checkpoint(3, "rev-3", &graph)
            .expect("write checkpoint");
        let checkpoint = store
            .load_graph_checkpoint()
            .expect("load checkpoint")
            .expect("checkpoint exists");
        assert_eq!(checkpoint.meta.generation, 3);
        assert_eq!(checkpoint.asserted_batch.vertices.len(), 1);
        assert_eq!(checkpoint.asserted_batch.edges.len(), 1);
    }
}
