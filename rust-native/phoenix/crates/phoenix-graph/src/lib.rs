use std::any::Any;

use phoenix_types::BoundaryKind;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GraphLayer {
    #[default]
    Asserted,
    Candidate,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum GraphMutationScope {
    Document { document_id: String },
    Session { session_id: String },
    Candidate { scope_key: String },
    Full,
}

impl Default for GraphMutationScope {
    fn default() -> Self {
        Self::Full
    }
}

impl GraphMutationScope {
    pub fn scope_key(&self) -> String {
        match self {
            Self::Document { document_id } => format!("document:{document_id}"),
            Self::Session { session_id } => format!("session:{session_id}"),
            Self::Candidate { scope_key } => format!("candidate:{scope_key}"),
            Self::Full => "__full__".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphVertexRecord {
    pub id: String,
    pub kind: String,
    pub weight: i64,
    pub value: Value,
    pub attributes: Value,
    pub entity_id: Option<String>,
    pub search_chunk_id: Option<String>,
    pub document_id: Option<String>,
    pub chapter_id: Option<u32>,
    pub chapters: Vec<u32>,
    pub boundary_id: Option<u32>,
    pub boundary_ordinal: Option<u32>,
    pub boundary_kind: Option<BoundaryKind>,
    pub boundary_ordinals: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphEdgeRecord {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: String,
    pub weight: i64,
    pub attributes: Value,
    pub data: Option<Value>,
    pub document_id: Option<String>,
    pub narrative_id: Option<String>,
    #[serde(default)]
    pub layer: GraphLayer,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphMutationBatch {
    #[serde(default)]
    pub layer: GraphLayer,
    #[serde(default)]
    pub scope: GraphMutationScope,
    #[serde(default)]
    pub vertices: Vec<GraphVertexRecord>,
    #[serde(default)]
    pub edges: Vec<GraphEdgeRecord>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraptorVertex {
    pub id: String,
    pub kind: String,
    pub weight: i64,
    pub value: Value,
    pub attributes: Value,
    pub entity_id: Option<String>,
    pub search_chunk_id: Option<String>,
    pub document_id: Option<String>,
    pub chapter_id: Option<u32>,
    pub chapters: Vec<u32>,
    pub boundary_id: Option<u32>,
    pub boundary_ordinal: Option<u32>,
    pub boundary_kind: Option<BoundaryKind>,
    pub boundary_ordinals: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraptorEdge {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: String,
    pub weight: i64,
    pub attributes: Value,
    pub data: Option<Value>,
    #[serde(default)]
    pub layer: GraphLayer,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphCounts {
    pub vertex_count: usize,
    pub asserted_edge_count: usize,
    pub candidate_edge_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraptorGraph {
    pub vertices: FxHashMap<String, GraptorVertex>,
    pub outgoing: FxHashMap<String, Vec<GraptorEdge>>,
    pub incoming: FxHashMap<String, Vec<GraptorEdge>>,
    pub chapter_leaves: FxHashMap<(String, u32), Vec<String>>,
}

impl GraptorGraph {
    pub fn outgoing_matching<'a>(
        &'a self,
        vertex_id: &str,
        edge_type: &'a str,
    ) -> impl Iterator<Item = &'a GraptorEdge> {
        self.outgoing_any(vertex_id)
            .filter(move |edge| edge.edge_type == edge_type)
    }

    pub fn incoming_matching<'a>(
        &'a self,
        vertex_id: &str,
        edge_type: &'a str,
    ) -> impl Iterator<Item = &'a GraptorEdge> {
        self.incoming_any(vertex_id)
            .filter(move |edge| edge.edge_type == edge_type)
    }

    pub fn outgoing_any<'a>(&'a self, vertex_id: &str) -> impl Iterator<Item = &'a GraptorEdge> {
        self.outgoing
            .get(vertex_id)
            .into_iter()
            .flat_map(|edges| edges.iter())
    }

    pub fn incoming_any<'a>(&'a self, vertex_id: &str) -> impl Iterator<Item = &'a GraptorEdge> {
        self.incoming
            .get(vertex_id)
            .into_iter()
            .flat_map(|edges| edges.iter())
    }

    pub fn chapter_leaves(
        &self,
        document_id: &str,
        chapter_id: u32,
    ) -> impl Iterator<Item = &String> {
        self.chapter_leaves
            .get(&(document_id.to_owned(), chapter_id))
            .into_iter()
            .flat_map(|leaves| leaves.iter())
    }
}

impl From<GraphVertexRecord> for GraptorVertex {
    fn from(value: GraphVertexRecord) -> Self {
        Self {
            id: value.id,
            kind: value.kind,
            weight: value.weight,
            value: value.value,
            attributes: value.attributes,
            entity_id: value.entity_id,
            search_chunk_id: value.search_chunk_id,
            document_id: value.document_id,
            chapter_id: value.chapter_id,
            chapters: value.chapters,
            boundary_id: value.boundary_id,
            boundary_ordinal: value.boundary_ordinal,
            boundary_kind: value.boundary_kind,
            boundary_ordinals: value.boundary_ordinals,
        }
    }
}

impl From<&GraptorVertex> for GraphVertexRecord {
    fn from(value: &GraptorVertex) -> Self {
        Self {
            id: value.id.clone(),
            kind: value.kind.clone(),
            weight: value.weight,
            value: value.value.clone(),
            attributes: value.attributes.clone(),
            entity_id: value.entity_id.clone(),
            search_chunk_id: value.search_chunk_id.clone(),
            document_id: value.document_id.clone(),
            chapter_id: value.chapter_id,
            chapters: value.chapters.clone(),
            boundary_id: value.boundary_id,
            boundary_ordinal: value.boundary_ordinal,
            boundary_kind: value.boundary_kind.clone(),
            boundary_ordinals: value.boundary_ordinals.clone(),
        }
    }
}

impl From<GraphEdgeRecord> for GraptorEdge {
    fn from(value: GraphEdgeRecord) -> Self {
        Self {
            source_id: value.source_id,
            target_id: value.target_id,
            edge_type: value.edge_type,
            weight: value.weight,
            attributes: value.attributes,
            data: value.data,
            layer: value.layer,
        }
    }
}

impl From<&GraptorEdge> for GraphEdgeRecord {
    fn from(value: &GraptorEdge) -> Self {
        Self {
            source_id: value.source_id.clone(),
            target_id: value.target_id.clone(),
            edge_type: value.edge_type.clone(),
            weight: value.weight,
            attributes: value.attributes.clone(),
            data: value.data.clone(),
            document_id: value
                .attributes
                .get("documentId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            narrative_id: value
                .attributes
                .get("narrativeId")
                .and_then(Value::as_str)
                .map(str::to_owned),
            layer: value.layer.clone(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GraphBackendError {
    #[error("graph backend is invalidated and must rebuild before use")]
    Invalidated,
    #[error("graph backend operation failed: {0}")]
    Operation(String),
}

pub trait PhoenixGraphBackend {
    fn apply_batch(&mut self, batch: GraphMutationBatch) -> Result<(), GraphBackendError>;
    fn rebuild_from_batches(
        &mut self,
        batches: Vec<GraphMutationBatch>,
    ) -> Result<(), GraphBackendError>;
    fn snapshot(&self, include_candidate_graph: bool) -> Result<GraptorGraph, GraphBackendError>;
    fn counts(&self) -> Result<GraphCounts, GraphBackendError>;
    fn candidate_edges(&self) -> Result<Vec<GraphEdgeRecord>, GraphBackendError>;
    fn invalidate(&mut self);
    fn rebuild_token(&self) -> Option<&str>;
    fn set_rebuild_token(&mut self, token: Option<String>);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
