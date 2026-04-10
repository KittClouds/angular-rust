use phoenix_graph::{
    GraphBackendError, GraphCounts, GraphEdgeRecord, GraphLayer, GraphMutationBatch,
    GraphMutationScope, GraphVertexRecord, GraptorEdge, GraptorGraph, GraptorVertex,
    PhoenixGraphBackend,
};
use phoenix_types::BoundaryKind;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::any::Any;
use std::sync::RwLock;
use std::time::{SystemTime, UNIX_EPOCH};

mod borrowed;
mod causal;
mod causal_view;
mod query_view;
mod region;
mod snapshot_query;
mod structural;

pub use borrowed::{KernelCsrRef, KernelGraphRef};
pub use causal::{causal_path_candidates_from_snapshot, KernelCausalPathCandidate};
pub use causal_view::{
    causal_path_candidate_views_from_snapshot, KernelCausalPathCandidateView,
    KernelCausalPathFeatures,
};
pub use query_view::KernelQueryView;
pub use region::{expand_snapshot_region, KernelExpandedRegion};
pub use snapshot_query::{
    entity_timeline_from_snapshot, slot_at_snapshot, unresolved_from_snapshot,
    what_changed_from_snapshot,
};
pub use structural::{KernelStructuralAnalytics, KernelStructuralScore};

#[cfg(test)]
mod causal_view_tests;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KernelVertexId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct KernelEdgeType(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum KernelGraphLayer {
    #[default]
    Asserted,
    Candidate,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum KernelMutationScope {
    Document { document_id: String },
    Session { session_id: String },
    Candidate { scope_key: String },
    Projection { scope_key: String },
    Full,
}

impl Default for KernelMutationScope {
    fn default() -> Self {
        Self::Full
    }
}

impl KernelMutationScope {
    pub fn scope_key(&self) -> String {
        match self {
            Self::Document { document_id } => format!("document:{document_id}"),
            Self::Session { session_id } => format!("session:{session_id}"),
            Self::Candidate { scope_key } => format!("candidate:{scope_key}"),
            Self::Projection { scope_key } => format!("projection:{scope_key}"),
            Self::Full => "__full__".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum KernelVertexClass {
    Document,
    Chunk,
    Entity,
    Alias,
    Mention,
    TimeAnchor,
    CalendarAnchor,
    Narrative,
    Episode,
    Memory,
    Task,
    State,
    Event,
    #[default]
    Generic,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum KernelRelationClass {
    Structural,
    Semantic,
    Identity,
    Resolution,
    Temporal,
    Calendar,
    Memory,
    Narrative,
    Candidate,
    #[default]
    Custom,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum KernelCalendarGranularity {
    Year,
    Month,
    Week,
    Day,
    Hour,
    #[default]
    Instant,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelBiTemporal {
    pub valid_from: Option<i64>,
    pub valid_to: Option<i64>,
    pub recorded_at: Option<i64>,
    pub expired_at: Option<i64>,
}

impl KernelBiTemporal {
    fn with_default_recorded_at(mut self, recorded_at: i64) -> Self {
        if self.recorded_at.is_none() {
            self.recorded_at = Some(recorded_at);
        }
        self
    }

    fn is_visible_at(&self, valid_at: i64, tx_at: i64) -> bool {
        if let Some(valid_from) = self.valid_from {
            if valid_at < valid_from {
                return false;
            }
        }
        if let Some(valid_to) = self.valid_to {
            if valid_at >= valid_to {
                return false;
            }
        }
        if let Some(recorded_at) = self.recorded_at {
            if tx_at < recorded_at {
                return false;
            }
        }
        if let Some(expired_at) = self.expired_at {
            if tx_at >= expired_at {
                return false;
            }
        }
        true
    }

    fn overlaps_valid_window(&self, start: Option<i64>, end: Option<i64>) -> bool {
        let self_start = self.valid_from.unwrap_or(i64::MIN);
        let self_end = self.valid_to.unwrap_or(i64::MAX);
        let window_start = start.unwrap_or(i64::MIN);
        let window_end = end.unwrap_or(i64::MAX);
        self_start < window_end && self_end > window_start
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelProvenance {
    pub resolver: Option<String>,
    pub source: Option<String>,
    pub confidence: Option<f64>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelEntityFacet {
    pub canonical_entity_id: Option<String>,
    pub surface: Option<String>,
    pub entity_kind: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelCalendarFacet {
    #[serde(default)]
    pub granularity: KernelCalendarGranularity,
    pub anchor_key: Option<String>,
    pub year: Option<i32>,
    pub month: Option<u32>,
    pub week_start_day: Option<String>,
    pub day: Option<u32>,
    pub hour: Option<u32>,
    pub timestamp_ms: Option<i64>,
    pub interval_start_ms: Option<i64>,
    pub interval_end_ms: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelResolutionFacet {
    pub strategy: Option<String>,
    pub candidate_rank: Option<u32>,
    pub confidence: Option<f64>,
    pub replaced_edge_key: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelVertex {
    pub id: KernelVertexId,
    pub kind: String,
    #[serde(default)]
    pub class: KernelVertexClass,
    #[serde(default)]
    pub labels: Vec<String>,
    pub weight: i64,
    pub value: Value,
    pub attributes: Value,
    #[serde(default)]
    pub temporal: KernelBiTemporal,
    #[serde(default)]
    pub provenance: KernelProvenance,
    pub entity_id: Option<String>,
    pub search_chunk_id: Option<String>,
    pub document_id: Option<String>,
    pub chapter_id: Option<u32>,
    #[serde(default)]
    pub chapters: Vec<u32>,
    pub boundary_id: Option<u32>,
    pub boundary_ordinal: Option<u32>,
    pub boundary_kind: Option<BoundaryKind>,
    #[serde(default)]
    pub boundary_ordinals: Vec<u32>,
    pub entity_facet: Option<KernelEntityFacet>,
    pub calendar_facet: Option<KernelCalendarFacet>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelEdge {
    pub source_id: KernelVertexId,
    pub target_id: KernelVertexId,
    pub edge_type: KernelEdgeType,
    #[serde(default)]
    pub relation_class: KernelRelationClass,
    pub weight: i64,
    pub attributes: Value,
    pub data: Option<Value>,
    pub document_id: Option<String>,
    pub narrative_id: Option<String>,
    #[serde(default)]
    pub layer: KernelGraphLayer,
    #[serde(default)]
    pub temporal: KernelBiTemporal,
    #[serde(default)]
    pub provenance: KernelProvenance,
    pub resolution_facet: Option<KernelResolutionFacet>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelMutationBatch {
    #[serde(default)]
    pub layer: KernelGraphLayer,
    #[serde(default)]
    pub scope: KernelMutationScope,
    pub recorded_at: Option<i64>,
    #[serde(default)]
    pub vertices: Vec<KernelVertex>,
    #[serde(default)]
    pub edges: Vec<KernelEdge>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelGraphSnapshot {
    #[serde(default)]
    pub vertices: Vec<KernelVertex>,
    #[serde(default)]
    pub asserted_edges: Vec<KernelEdge>,
    #[serde(default)]
    pub candidate_edges: Vec<KernelEdge>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelCheckpointMeta {
    pub checkpoint_id: String,
    pub generation: u64,
    pub source_revision: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelCheckpointData {
    pub meta: KernelCheckpointMeta,
    pub snapshot: KernelGraphSnapshot,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelJournalEntry {
    pub generation: u64,
    pub source_revision: String,
    pub batch: Option<KernelMutationBatch>,
    pub commit_id: Option<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelCsrSidecar {
    pub vertex_ids: Vec<String>,
    pub offsets: Vec<usize>,
    pub targets: Vec<usize>,
    pub weights: Vec<f64>,
    pub dense: FxHashMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelTemporalIndexEntry {
    pub record_id: String,
    pub valid_from: Option<i64>,
    pub valid_to: Option<i64>,
    pub recorded_at: Option<i64>,
    pub expired_at: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelTemporalIndexSidecar {
    pub vertices: Vec<KernelTemporalIndexEntry>,
    pub asserted_edges: Vec<KernelTemporalIndexEntry>,
    pub candidate_edges: Vec<KernelTemporalIndexEntry>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelEntityCandidate {
    pub entity_id: String,
    pub score: f64,
    pub source_vertex_id: Option<String>,
    pub relation_type: Option<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelEntitySupport {
    pub alias_vertex_ids: Vec<String>,
    pub mention_vertex_ids: Vec<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelEntitySidecar {
    pub alias_candidates: FxHashMap<String, Vec<KernelEntityCandidate>>,
    pub mention_entities: FxHashMap<String, String>,
    pub canonical_support: FxHashMap<String, KernelEntitySupport>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelCalendarSidecar {
    pub anchors: FxHashMap<String, KernelVertex>,
    pub adjacency: FxHashMap<String, Vec<KernelEdge>>,
    pub anchor_members: FxHashMap<String, Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelGraphView {
    pub vertices: FxHashMap<String, KernelVertex>,
    pub asserted_edges: FxHashMap<KernelEdgeKey, KernelEdge>,
    pub candidate_edges: FxHashMap<KernelEdgeKey, KernelEdge>,
    pub csr: KernelCsrSidecar,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelViewRequest {
    pub valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelEntityResolveRequest {
    pub surface: Option<String>,
    pub mention_vertex_id: Option<String>,
    pub canonical_entity_id: Option<String>,
    pub valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
    pub limit: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelCalendarWindowRequest {
    pub start_ms: Option<i64>,
    pub end_ms: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelSlotQueryRequest {
    pub entity_id: String,
    pub slot_key: String,
    pub valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelUnresolvedQueryRequest {
    pub entity_id: String,
    pub slot_key: Option<String>,
    pub valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelWhatChangedRequest {
    pub entity_id: String,
    pub slot_key: Option<String>,
    pub since_valid_at: i64,
    pub until_valid_at: Option<i64>,
    pub recorded_at: Option<i64>,
    pub include_candidate_graph: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelSlotState {
    pub state_vertex_id: String,
    pub entity_id: String,
    pub slot_key: String,
    pub value: String,
    pub value_entity_id: Option<String>,
    pub status: Option<String>,
    pub source_class: Option<String>,
    pub confidence: Option<f64>,
    pub temporal: KernelBiTemporal,
    #[serde(default)]
    pub supporting_claim_ids: Vec<String>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelStateIssue {
    pub issue_vertex_id: String,
    pub issue_kind: String,
    pub entity_id: String,
    pub slot_key: String,
    pub status: Option<String>,
    pub reason: Option<String>,
    pub detail: Option<String>,
    pub preferred_claim_id: Option<String>,
    pub temporal: KernelBiTemporal,
    #[serde(default)]
    pub supporting_claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelSlotAnswer {
    pub entity_id: String,
    pub slot_key: String,
    pub active_state: Option<KernelSlotState>,
    #[serde(default)]
    pub competing_states: Vec<KernelSlotState>,
    #[serde(default)]
    pub conflicts: Vec<KernelStateIssue>,
    #[serde(default)]
    pub gaps: Vec<KernelStateIssue>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum KernelStateChangeKind {
    #[default]
    Activated,
    Expired,
    ActivatedAndExpired,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelStateChange {
    pub change_kind: KernelStateChangeKind,
    pub state: KernelSlotState,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelCalendarAnchorSet {
    pub year_id: String,
    pub month_id: String,
    pub week_id: String,
    pub day_id: String,
    pub hour_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct KernelEdgeKey {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: String,
}

impl KernelEdgeKey {
    fn new(source_id: &str, target_id: &str, edge_type: &str) -> Self {
        Self {
            source_id: source_id.to_owned(),
            target_id: target_id.to_owned(),
            edge_type: edge_type.to_owned(),
        }
    }

    fn from_edge(edge: &KernelEdge) -> Self {
        Self::new(&edge.source_id.0, &edge.target_id.0, &edge.edge_type.0)
    }

    fn storage_key(&self) -> String {
        format!("{}|{}|{}", self.source_id, self.target_id, self.edge_type)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct KernelDirtyFlags {
    csr: bool,
    valid_time_index: bool,
    transaction_time_index: bool,
    entity_index: bool,
    calendar_index: bool,
}

#[derive(Default)]
pub struct PhoenixGraphKernel {
    rebuild_token: Option<String>,
    invalidated: bool,
    vertex_history: FxHashMap<String, Vec<KernelVertex>>,
    asserted_edge_history: FxHashMap<KernelEdgeKey, Vec<KernelEdge>>,
    candidate_edge_history: FxHashMap<KernelEdgeKey, Vec<KernelEdge>>,
    vertices: FxHashMap<String, KernelVertex>,
    asserted_edges: FxHashMap<KernelEdgeKey, KernelEdge>,
    candidate_edges: FxHashMap<KernelEdgeKey, KernelEdge>,
    document_scope_vertices: FxHashMap<String, FxHashSet<String>>,
    document_scope_edges: FxHashMap<String, FxHashSet<KernelEdgeKey>>,
    session_scope_vertices: FxHashMap<String, FxHashSet<String>>,
    session_scope_edges: FxHashMap<String, FxHashSet<KernelEdgeKey>>,
    projection_scope_vertices: FxHashMap<String, FxHashSet<String>>,
    projection_scope_edges: FxHashMap<String, FxHashSet<KernelEdgeKey>>,
    candidate_scope_edges: FxHashMap<String, FxHashSet<KernelEdgeKey>>,
    dirty_flags: RwLock<KernelDirtyFlags>,
    csr: RwLock<KernelCsrSidecar>,
    valid_time_index: RwLock<KernelTemporalIndexSidecar>,
    transaction_time_index: RwLock<KernelTemporalIndexSidecar>,
    entity_index: RwLock<KernelEntitySidecar>,
    calendar_index: RwLock<KernelCalendarSidecar>,
}

impl PhoenixGraphKernel {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_snapshot(snapshot: KernelGraphSnapshot, rebuild_token: Option<String>) -> Self {
        let mut kernel = Self {
            rebuild_token,
            invalidated: false,
            ..Self::default()
        };
        for vertex in snapshot.vertices {
            kernel
                .vertex_history
                .entry(vertex.id.0.clone())
                .or_default()
                .push(vertex);
        }
        for edge in snapshot.asserted_edges {
            kernel
                .asserted_edge_history
                .entry(KernelEdgeKey::from_edge(&edge))
                .or_default()
                .push(edge);
        }
        for edge in snapshot.candidate_edges {
            kernel
                .candidate_edge_history
                .entry(KernelEdgeKey::from_edge(&edge))
                .or_default()
                .push(edge);
        }
        kernel.rebuild_active_views_from_history();
        kernel.mark_all_sidecars_dirty();
        kernel
    }

    pub fn entity_sidecar(&self) -> KernelEntitySidecar {
        self.ensure_entity_index();
        self.entity_index
            .read()
            .expect("kernel entity sidecar poisoned")
            .clone()
    }

    pub fn graph_view(&self) -> KernelGraphView {
        self.ensure_csr_sidecar();
        KernelGraphView {
            vertices: self.vertices.clone(),
            asserted_edges: self.asserted_edges.clone(),
            candidate_edges: self.candidate_edges.clone(),
            csr: self
                .csr
                .read()
                .expect("kernel csr sidecar poisoned")
                .clone(),
        }
    }

    pub fn snapshot_kernel(&self) -> KernelGraphSnapshot {
        let mut vertices = self
            .vertex_history
            .values()
            .flat_map(|records| records.iter().cloned())
            .collect::<Vec<_>>();
        vertices.sort_by(|left, right| {
            left.id
                .0
                .cmp(&right.id.0)
                .then_with(|| left.temporal.recorded_at.cmp(&right.temporal.recorded_at))
                .then_with(|| left.temporal.valid_from.cmp(&right.temporal.valid_from))
                .then_with(|| left.temporal.expired_at.cmp(&right.temporal.expired_at))
        });

        let mut asserted_edges = self
            .asserted_edge_history
            .values()
            .flat_map(|records| records.iter().cloned())
            .collect::<Vec<_>>();
        asserted_edges.sort_by(|left, right| {
            KernelEdgeKey::from_edge(left)
                .storage_key()
                .cmp(&KernelEdgeKey::from_edge(right).storage_key())
                .then_with(|| left.temporal.recorded_at.cmp(&right.temporal.recorded_at))
                .then_with(|| left.temporal.valid_from.cmp(&right.temporal.valid_from))
        });

        let mut candidate_edges = self
            .candidate_edge_history
            .values()
            .flat_map(|records| records.iter().cloned())
            .collect::<Vec<_>>();
        candidate_edges.sort_by(|left, right| {
            KernelEdgeKey::from_edge(left)
                .storage_key()
                .cmp(&KernelEdgeKey::from_edge(right).storage_key())
                .then_with(|| left.temporal.recorded_at.cmp(&right.temporal.recorded_at))
                .then_with(|| left.temporal.valid_from.cmp(&right.temporal.valid_from))
        });

        KernelGraphSnapshot {
            vertices,
            asserted_edges,
            candidate_edges,
        }
    }

    pub fn csr_sidecar(&self) -> KernelCsrSidecar {
        self.ensure_csr_sidecar();
        self.csr
            .read()
            .expect("kernel csr sidecar poisoned")
            .clone()
    }

    pub fn valid_time_index(&self) -> KernelTemporalIndexSidecar {
        self.ensure_valid_time_index();
        self.valid_time_index
            .read()
            .expect("kernel valid-time sidecar poisoned")
            .clone()
    }

    pub fn transaction_time_index(&self) -> KernelTemporalIndexSidecar {
        self.ensure_transaction_time_index();
        self.transaction_time_index
            .read()
            .expect("kernel transaction-time sidecar poisoned")
            .clone()
    }

    pub fn entity_index(&self) -> KernelEntitySidecar {
        self.ensure_entity_index();
        self.entity_index
            .read()
            .expect("kernel entity sidecar poisoned")
            .clone()
    }

    pub fn calendar_index(&self) -> KernelCalendarSidecar {
        self.ensure_calendar_index();
        self.calendar_index
            .read()
            .expect("kernel calendar sidecar poisoned")
            .clone()
    }

    pub fn vertex_ordinal(&self, vertex_id: &str) -> Option<usize> {
        self.ensure_csr_sidecar();
        self.csr
            .read()
            .expect("kernel csr sidecar poisoned")
            .dense
            .get(vertex_id)
            .copied()
    }

    pub fn vertex(&self, vertex_id: &str) -> Option<&KernelVertex> {
        self.vertices.get(vertex_id)
    }

    pub fn view_as_of(&self, request: KernelViewRequest) -> KernelGraphSnapshot {
        let valid_at = request.valid_at.unwrap_or_else(now_ms);
        let tx_at = request.recorded_at.unwrap_or(valid_at);

        let mut vertices = visible_vertices(&self.vertex_history, valid_at, tx_at);
        let vertex_ids = vertices
            .iter()
            .map(|vertex| vertex.id.0.clone())
            .collect::<FxHashSet<_>>();
        let mut asserted_edges =
            visible_edges(&self.asserted_edge_history, valid_at, tx_at, &vertex_ids);
        let mut candidate_edges = if request.include_candidate_graph {
            visible_edges(&self.candidate_edge_history, valid_at, tx_at, &vertex_ids)
        } else {
            Vec::new()
        };

        vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        asserted_edges.sort_by(|left, right| {
            KernelEdgeKey::from_edge(left)
                .storage_key()
                .cmp(&KernelEdgeKey::from_edge(right).storage_key())
        });
        candidate_edges.sort_by(|left, right| {
            KernelEdgeKey::from_edge(left)
                .storage_key()
                .cmp(&KernelEdgeKey::from_edge(right).storage_key())
        });

        KernelGraphSnapshot {
            vertices,
            asserted_edges,
            candidate_edges,
        }
    }

    pub fn entity_candidates(
        &self,
        request: KernelEntityResolveRequest,
    ) -> Vec<KernelEntityCandidate> {
        let sidecar = if request.valid_at.is_some() || request.recorded_at.is_some() {
            let snapshot = self.view_as_of(KernelViewRequest {
                valid_at: request.valid_at,
                recorded_at: request.recorded_at,
                include_candidate_graph: request.include_candidate_graph,
            });
            build_entity_sidecar(
                &snapshot.vertices,
                &snapshot.asserted_edges,
                &snapshot.candidate_edges,
                request.include_candidate_graph,
            )
        } else {
            self.entity_sidecar()
        };

        let mut candidates = Vec::new();
        if let Some(surface) = request.surface.as_deref() {
            candidates.extend(
                sidecar
                    .alias_candidates
                    .get(&normalize_surface(surface))
                    .cloned()
                    .unwrap_or_default(),
            );
        }
        if let Some(mention_vertex_id) = request.mention_vertex_id.as_deref() {
            if let Some(entity_id) = sidecar.mention_entities.get(mention_vertex_id) {
                candidates.push(KernelEntityCandidate {
                    entity_id: entity_id.clone(),
                    score: 1.0,
                    source_vertex_id: Some(mention_vertex_id.to_owned()),
                    relation_type: Some("resolved_to".to_owned()),
                    evidence_refs: sidecar
                        .canonical_support
                        .get(entity_id)
                        .map(|support| support.evidence_refs.clone())
                        .unwrap_or_default(),
                });
            }
        }
        if let Some(entity_id) = request.canonical_entity_id.as_deref() {
            candidates.push(KernelEntityCandidate {
                entity_id: entity_id.to_owned(),
                score: 1.0,
                source_vertex_id: None,
                relation_type: Some("canonical".to_owned()),
                evidence_refs: sidecar
                    .canonical_support
                    .get(entity_id)
                    .map(|support| support.evidence_refs.clone())
                    .unwrap_or_default(),
            });
        }
        candidates.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.entity_id.cmp(&right.entity_id))
        });
        candidates.dedup_by(|left, right| left.entity_id == right.entity_id);
        candidates.truncate(request.limit.unwrap_or(8));
        candidates
    }

    pub fn entity_timeline(
        &self,
        entity_id: &str,
        window: Option<(i64, i64)>,
        tx_point: Option<i64>,
    ) -> KernelGraphSnapshot {
        let entity_vertex_ids = self
            .snapshot_kernel()
            .vertices
            .into_iter()
            .filter(|vertex| {
                vertex.id.0 == entity_id
                    || vertex.entity_id.as_deref() == Some(entity_id)
                    || vertex
                        .entity_facet
                        .as_ref()
                        .and_then(|facet| facet.canonical_entity_id.as_deref())
                        == Some(entity_id)
            })
            .map(|vertex| vertex.id.0)
            .collect::<FxHashSet<_>>();
        let (start, end) = window
            .map(|(start, end)| (Some(start), Some(end)))
            .unwrap_or((None, None));
        let tx_point = tx_point.unwrap_or_else(now_ms);
        let mut vertices = self
            .vertex_history
            .values()
            .flat_map(|records| records.iter())
            .filter(|vertex| {
                (entity_vertex_ids.contains(&vertex.id.0)
                    || vertex.entity_id.as_deref() == Some(entity_id)
                    || vertex
                        .entity_facet
                        .as_ref()
                        .and_then(|facet| facet.canonical_entity_id.as_deref())
                        == Some(entity_id))
                    && vertex.temporal.overlaps_valid_window(start, end)
                    && vertex
                        .temporal
                        .is_visible_at(vertex.temporal.valid_from.unwrap_or(i64::MIN), tx_point)
            })
            .cloned()
            .collect::<Vec<_>>();
        let vertex_ids = vertices
            .iter()
            .map(|vertex| vertex.id.0.clone())
            .collect::<FxHashSet<_>>();
        let mut asserted_edges = self
            .asserted_edge_history
            .values()
            .flat_map(|records| records.iter())
            .filter(|edge| {
                (vertex_ids.contains(&edge.source_id.0) || vertex_ids.contains(&edge.target_id.0))
                    && edge.temporal.overlaps_valid_window(start, end)
                    && edge
                        .temporal
                        .is_visible_at(edge.temporal.valid_from.unwrap_or(i64::MIN), tx_point)
            })
            .cloned()
            .collect::<Vec<_>>();
        let mut candidate_edges = self
            .candidate_edge_history
            .values()
            .flat_map(|records| records.iter())
            .filter(|edge| {
                (vertex_ids.contains(&edge.source_id.0) || vertex_ids.contains(&edge.target_id.0))
                    && edge.temporal.overlaps_valid_window(start, end)
                    && edge
                        .temporal
                        .is_visible_at(edge.temporal.valid_from.unwrap_or(i64::MIN), tx_point)
            })
            .cloned()
            .collect::<Vec<_>>();
        vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        asserted_edges.sort_by(|left, right| {
            KernelEdgeKey::from_edge(left)
                .storage_key()
                .cmp(&KernelEdgeKey::from_edge(right).storage_key())
        });
        candidate_edges.sort_by(|left, right| {
            KernelEdgeKey::from_edge(left)
                .storage_key()
                .cmp(&KernelEdgeKey::from_edge(right).storage_key())
        });
        KernelGraphSnapshot {
            vertices,
            asserted_edges,
            candidate_edges,
        }
    }

    pub fn current_slot(
        &self,
        entity_id: &str,
        slot_key: &str,
        recorded_at: Option<i64>,
    ) -> KernelSlotAnswer {
        self.slot_at(KernelSlotQueryRequest {
            entity_id: entity_id.to_owned(),
            slot_key: slot_key.to_owned(),
            valid_at: Some(now_ms()),
            recorded_at,
            include_candidate_graph: false,
        })
    }

    pub fn slot_at(&self, request: KernelSlotQueryRequest) -> KernelSlotAnswer {
        let snapshot = self.snapshot_for_query(
            request.valid_at,
            request.recorded_at,
            request.include_candidate_graph,
        );
        slot_at_snapshot(&snapshot, &request)
    }

    pub fn what_is_unresolved(
        &self,
        request: KernelUnresolvedQueryRequest,
    ) -> Vec<KernelStateIssue> {
        let snapshot = self.snapshot_for_query(
            request.valid_at,
            request.recorded_at,
            request.include_candidate_graph,
        );
        unresolved_from_snapshot(&snapshot, &request)
    }

    pub fn what_changed(&self, request: KernelWhatChangedRequest) -> Vec<KernelStateChange> {
        let until = request.until_valid_at.unwrap_or_else(now_ms);
        let timeline = self.entity_timeline(
            &request.entity_id,
            Some((request.since_valid_at, until)),
            request.recorded_at.or(Some(until)),
        );
        what_changed_from_snapshot(&timeline, &request)
    }

    pub fn calendar_slice(&self, request: KernelCalendarWindowRequest) -> KernelGraphSnapshot {
        let snapshot = self.view_as_of(KernelViewRequest {
            valid_at: request.end_ms.or(request.start_ms),
            recorded_at: request.recorded_at,
            include_candidate_graph: request.include_candidate_graph,
        });
        let start = request.start_ms;
        let end = request.end_ms;
        let mut vertices = snapshot
            .vertices
            .into_iter()
            .filter(|vertex| {
                if matches!(
                    vertex.class,
                    KernelVertexClass::CalendarAnchor | KernelVertexClass::TimeAnchor
                ) {
                    return vertex.temporal.overlaps_valid_window(start, end)
                        || vertex
                            .calendar_facet
                            .as_ref()
                            .map(|facet| {
                                let anchor_ts = facet
                                    .timestamp_ms
                                    .or(facet.interval_start_ms)
                                    .unwrap_or(i64::MIN);
                                anchor_ts >= start.unwrap_or(i64::MIN)
                                    && anchor_ts < end.unwrap_or(i64::MAX)
                            })
                            .unwrap_or(false);
                }
                vertex.temporal.overlaps_valid_window(start, end)
            })
            .collect::<Vec<_>>();
        let vertex_ids = vertices
            .iter()
            .map(|vertex| vertex.id.0.clone())
            .collect::<FxHashSet<_>>();
        let asserted_edges = snapshot
            .asserted_edges
            .into_iter()
            .filter(|edge| {
                vertex_ids.contains(&edge.source_id.0) || vertex_ids.contains(&edge.target_id.0)
            })
            .collect::<Vec<_>>();
        let candidate_edges = snapshot
            .candidate_edges
            .into_iter()
            .filter(|edge| {
                vertex_ids.contains(&edge.source_id.0) || vertex_ids.contains(&edge.target_id.0)
            })
            .collect::<Vec<_>>();
        vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        KernelGraphSnapshot {
            vertices,
            asserted_edges,
            candidate_edges,
        }
    }

    fn snapshot_for_query(
        &self,
        valid_at: Option<i64>,
        recorded_at: Option<i64>,
        include_candidate_graph: bool,
    ) -> KernelGraphSnapshot {
        if valid_at.is_some() || recorded_at.is_some() || include_candidate_graph {
            return self.view_as_of(KernelViewRequest {
                valid_at,
                recorded_at,
                include_candidate_graph,
            });
        }

        let mut vertices = self.vertices.values().cloned().collect::<Vec<_>>();
        let mut asserted_edges = self.asserted_edges.values().cloned().collect::<Vec<_>>();
        let mut candidate_edges = if include_candidate_graph {
            self.candidate_edges.values().cloned().collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        asserted_edges.sort_by(|left, right| {
            KernelEdgeKey::from_edge(left)
                .storage_key()
                .cmp(&KernelEdgeKey::from_edge(right).storage_key())
        });
        candidate_edges.sort_by(|left, right| {
            KernelEdgeKey::from_edge(left)
                .storage_key()
                .cmp(&KernelEdgeKey::from_edge(right).storage_key())
        });
        KernelGraphSnapshot {
            vertices,
            asserted_edges,
            candidate_edges,
        }
    }

    pub fn build_calendar_anchor_artifacts(
        timestamp_ms: i64,
        recorded_at: Option<i64>,
    ) -> (Vec<KernelVertex>, Vec<KernelEdge>, KernelCalendarAnchorSet) {
        let recorded_at = recorded_at.unwrap_or(timestamp_ms);
        let (year, month, day, hour, week_start_key) = calendar_components(timestamp_ms);
        let year_id = format!("calendar::year::{year}");
        let month_id = format!("calendar::month::{year}-{month:02}");
        let week_id = format!("calendar::week::{week_start_key}");
        let day_id = format!("calendar::day::{year}-{month:02}-{day:02}");
        let hour_id = format!("calendar::hour::{year}-{month:02}-{day:02}T{hour:02}");
        let temporal = KernelBiTemporal {
            valid_from: Some(timestamp_ms),
            valid_to: None,
            recorded_at: Some(recorded_at),
            expired_at: None,
        };
        let mut vertices = vec![
            calendar_anchor_vertex(
                &year_id,
                "year",
                KernelCalendarGranularity::Year,
                timestamp_ms,
                temporal.clone(),
                year,
                Some(month),
                Some(day),
                Some(hour),
                Some(week_start_key.clone()),
            ),
            calendar_anchor_vertex(
                &month_id,
                "month",
                KernelCalendarGranularity::Month,
                timestamp_ms,
                temporal.clone(),
                year,
                Some(month),
                Some(day),
                Some(hour),
                Some(week_start_key.clone()),
            ),
            calendar_anchor_vertex(
                &week_id,
                "week",
                KernelCalendarGranularity::Week,
                timestamp_ms,
                temporal.clone(),
                year,
                Some(month),
                Some(day),
                Some(hour),
                Some(week_start_key.clone()),
            ),
            calendar_anchor_vertex(
                &day_id,
                "day",
                KernelCalendarGranularity::Day,
                timestamp_ms,
                temporal.clone(),
                year,
                Some(month),
                Some(day),
                Some(hour),
                Some(week_start_key.clone()),
            ),
            calendar_anchor_vertex(
                &hour_id,
                "hour",
                KernelCalendarGranularity::Hour,
                timestamp_ms,
                temporal,
                year,
                Some(month),
                Some(day),
                Some(hour),
                Some(week_start_key.clone()),
            ),
        ];
        vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        let edges = vec![
            calendar_edge(&year_id, &month_id, "contains", recorded_at),
            calendar_edge(&month_id, &day_id, "contains", recorded_at),
            calendar_edge(&week_id, &day_id, "contains", recorded_at),
            calendar_edge(&day_id, &hour_id, "contains", recorded_at),
        ];
        (
            vertices,
            edges,
            KernelCalendarAnchorSet {
                year_id,
                month_id,
                week_id,
                day_id,
                hour_id,
            },
        )
    }

    pub fn build_interval_anchor_links(
        owner_vertex_id: &KernelVertexId,
        temporal: &KernelBiTemporal,
        layer: KernelGraphLayer,
        document_id: Option<String>,
        recorded_at: Option<i64>,
    ) -> (Vec<KernelVertex>, Vec<KernelEdge>) {
        let mut vertices = Vec::new();
        let mut edges = Vec::new();
        if let Some(valid_from) = temporal.valid_from {
            let (anchor_vertices, mut anchor_edges, anchors) =
                Self::build_calendar_anchor_artifacts(
                    valid_from,
                    recorded_at.or(temporal.recorded_at),
                );
            vertices.extend(anchor_vertices);
            edges.append(&mut anchor_edges);
            edges.push(KernelEdge {
                source_id: owner_vertex_id.clone(),
                target_id: KernelVertexId(anchors.hour_id.clone()),
                edge_type: KernelEdgeType("starts_at".to_owned()),
                relation_class: KernelRelationClass::Temporal,
                weight: 1,
                attributes: json!({}),
                data: None,
                document_id: document_id.clone(),
                narrative_id: None,
                layer: layer.clone(),
                temporal: temporal.clone(),
                provenance: KernelProvenance::default(),
                resolution_facet: None,
            });
            if temporal.valid_to.is_none() {
                edges.push(KernelEdge {
                    source_id: owner_vertex_id.clone(),
                    target_id: KernelVertexId(anchors.hour_id),
                    edge_type: KernelEdgeType("occurs_at".to_owned()),
                    relation_class: KernelRelationClass::Temporal,
                    weight: 1,
                    attributes: json!({}),
                    data: None,
                    document_id: document_id.clone(),
                    narrative_id: None,
                    layer: layer.clone(),
                    temporal: temporal.clone(),
                    provenance: KernelProvenance::default(),
                    resolution_facet: None,
                });
            }
        }
        if let Some(valid_to) = temporal.valid_to {
            let (anchor_vertices, mut anchor_edges, anchors) =
                Self::build_calendar_anchor_artifacts(
                    valid_to,
                    recorded_at.or(temporal.recorded_at),
                );
            vertices.extend(anchor_vertices);
            edges.append(&mut anchor_edges);
            edges.push(KernelEdge {
                source_id: owner_vertex_id.clone(),
                target_id: KernelVertexId(anchors.hour_id),
                edge_type: KernelEdgeType("ends_at".to_owned()),
                relation_class: KernelRelationClass::Temporal,
                weight: 1,
                attributes: json!({}),
                data: None,
                document_id,
                narrative_id: None,
                layer,
                temporal: temporal.clone(),
                provenance: KernelProvenance::default(),
                resolution_facet: None,
            });
        }
        dedup_vertices(&mut vertices);
        dedup_edges(&mut edges);
        (vertices, edges)
    }

    pub fn snapshot_legacy(&self, include_candidate_graph: bool) -> GraptorGraph {
        let mut graph = GraptorGraph::default();
        for (vertex_id, vertex) in &self.vertices {
            let legacy = GraptorVertex::from(vertex.clone());
            graph.vertices.insert(vertex_id.clone(), legacy.clone());
            if let (Some(document_id), Some(chapter_id), Some(_)) = (
                legacy.document_id.clone(),
                legacy.chapter_id,
                legacy.search_chunk_id.clone(),
            ) {
                graph
                    .chapter_leaves
                    .entry((document_id, chapter_id))
                    .or_default()
                    .push(vertex_id.clone());
            }
        }
        for edge in self.asserted_edges.values() {
            let legacy = GraptorEdge::from(edge.clone());
            graph
                .outgoing
                .entry(legacy.source_id.clone())
                .or_default()
                .push(legacy.clone());
            graph
                .incoming
                .entry(legacy.target_id.clone())
                .or_default()
                .push(legacy);
        }
        if include_candidate_graph {
            for edge in self.candidate_edges.values() {
                let legacy = GraptorEdge::from(edge.clone());
                if !candidate_edge_is_active(&legacy) {
                    continue;
                }
                graph
                    .outgoing
                    .entry(legacy.source_id.clone())
                    .or_default()
                    .push(legacy.clone());
                graph
                    .incoming
                    .entry(legacy.target_id.clone())
                    .or_default()
                    .push(legacy);
            }
        }
        graph
    }

    pub fn from_legacy_graph(graph: &GraptorGraph) -> Self {
        let mut kernel = Self::default();
        for vertex in graph.vertices.values() {
            let vertex = KernelVertex::from(vertex.clone());
            kernel
                .vertex_history
                .entry(vertex.id.0.clone())
                .or_default()
                .push(vertex);
        }
        for edge in graph.outgoing.values().flat_map(|edges| edges.iter()) {
            let edge = KernelEdge::from(edge.clone());
            let key = KernelEdgeKey::from_edge(&edge);
            match edge.layer {
                KernelGraphLayer::Asserted => {
                    kernel
                        .asserted_edge_history
                        .entry(key)
                        .or_default()
                        .push(edge);
                }
                KernelGraphLayer::Candidate => {
                    kernel
                        .candidate_edge_history
                        .entry(key)
                        .or_default()
                        .push(edge);
                }
            }
        }
        kernel.rebuild_active_views_from_history();
        kernel.mark_all_sidecars_dirty();
        kernel
    }

    pub fn rebuild_from_kernel_batches(
        &mut self,
        batches: Vec<KernelMutationBatch>,
    ) -> Result<(), GraphBackendError> {
        self.invalidated = false;
        self.vertex_history.clear();
        self.asserted_edge_history.clear();
        self.candidate_edge_history.clear();
        self.vertices.clear();
        self.asserted_edges.clear();
        self.candidate_edges.clear();
        self.document_scope_vertices.clear();
        self.document_scope_edges.clear();
        self.session_scope_vertices.clear();
        self.session_scope_edges.clear();
        self.projection_scope_vertices.clear();
        self.projection_scope_edges.clear();
        self.candidate_scope_edges.clear();
        *self.csr.write().expect("kernel csr sidecar poisoned") = KernelCsrSidecar::default();
        *self
            .valid_time_index
            .write()
            .expect("kernel valid-time sidecar poisoned") = KernelTemporalIndexSidecar::default();
        *self
            .transaction_time_index
            .write()
            .expect("kernel transaction-time sidecar poisoned") =
            KernelTemporalIndexSidecar::default();
        *self
            .entity_index
            .write()
            .expect("kernel entity sidecar poisoned") = KernelEntitySidecar::default();
        *self
            .calendar_index
            .write()
            .expect("kernel calendar sidecar poisoned") = KernelCalendarSidecar::default();
        *self
            .dirty_flags
            .write()
            .expect("kernel dirty flags poisoned") = KernelDirtyFlags::default();
        for batch in batches {
            self.apply_kernel_batch_inner(batch, false)?;
        }
        Ok(())
    }

    pub fn apply_kernel_batch(
        &mut self,
        batch: KernelMutationBatch,
    ) -> Result<(), GraphBackendError> {
        self.apply_kernel_batch_inner(batch, true)
    }

    fn apply_kernel_batch_inner(
        &mut self,
        batch: KernelMutationBatch,
        _rebuild_sidecars: bool,
    ) -> Result<(), GraphBackendError> {
        self.ensure_ready()?;
        let scope_key = batch.scope.scope_key();
        let recorded_at = batch.recorded_at.unwrap_or_else(now_ms);
        let layer = batch.layer.clone();
        let vertices = batch
            .vertices
            .into_iter()
            .map(|vertex| normalize_vertex(vertex, recorded_at))
            .collect::<Vec<_>>();
        let edges = batch
            .edges
            .into_iter()
            .map(|edge| normalize_edge(edge, recorded_at, &layer))
            .collect::<Vec<_>>();

        match (&layer, &batch.scope) {
            (KernelGraphLayer::Asserted, KernelMutationScope::Document { .. }) => {
                Self::replace_vertex_scope(
                    &mut self.vertex_history,
                    &mut self.vertices,
                    &mut self.document_scope_vertices,
                    scope_key.clone(),
                    recorded_at,
                    vertices,
                );
                Self::replace_edge_scope(
                    &mut self.asserted_edge_history,
                    &mut self.asserted_edges,
                    &self.vertices,
                    &mut self.document_scope_edges,
                    scope_key,
                    recorded_at,
                    edges,
                );
                Self::prune_edge_map_with_vertices(&mut self.asserted_edges, &self.vertices);
                Self::prune_edge_map_with_vertices(&mut self.candidate_edges, &self.vertices);
            }
            (KernelGraphLayer::Asserted, KernelMutationScope::Session { .. }) => {
                Self::replace_vertex_scope(
                    &mut self.vertex_history,
                    &mut self.vertices,
                    &mut self.session_scope_vertices,
                    scope_key.clone(),
                    recorded_at,
                    vertices,
                );
                Self::replace_edge_scope(
                    &mut self.asserted_edge_history,
                    &mut self.asserted_edges,
                    &self.vertices,
                    &mut self.session_scope_edges,
                    scope_key,
                    recorded_at,
                    edges,
                );
                Self::prune_edge_map_with_vertices(&mut self.asserted_edges, &self.vertices);
                Self::prune_edge_map_with_vertices(&mut self.candidate_edges, &self.vertices);
            }
            (KernelGraphLayer::Asserted, KernelMutationScope::Projection { .. }) => {
                Self::replace_vertex_scope(
                    &mut self.vertex_history,
                    &mut self.vertices,
                    &mut self.projection_scope_vertices,
                    scope_key.clone(),
                    recorded_at,
                    vertices,
                );
                Self::replace_edge_scope(
                    &mut self.asserted_edge_history,
                    &mut self.asserted_edges,
                    &self.vertices,
                    &mut self.projection_scope_edges,
                    scope_key,
                    recorded_at,
                    edges,
                );
                Self::prune_edge_map_with_vertices(&mut self.asserted_edges, &self.vertices);
                Self::prune_edge_map_with_vertices(&mut self.candidate_edges, &self.vertices);
            }
            (KernelGraphLayer::Candidate, KernelMutationScope::Candidate { .. }) => {
                Self::replace_edge_scope(
                    &mut self.candidate_edge_history,
                    &mut self.candidate_edges,
                    &self.vertices,
                    &mut self.candidate_scope_edges,
                    scope_key,
                    recorded_at,
                    edges,
                );
            }
            (KernelGraphLayer::Asserted, KernelMutationScope::Full) => {
                Self::expire_all_vertices(&mut self.vertex_history, recorded_at);
                Self::expire_all_edges(&mut self.asserted_edge_history, recorded_at);
                self.document_scope_vertices.clear();
                self.document_scope_edges.clear();
                self.session_scope_vertices.clear();
                self.session_scope_edges.clear();
                self.projection_scope_vertices.clear();
                self.projection_scope_edges.clear();
                self.vertices.clear();
                self.asserted_edges.clear();
                for vertex in vertices {
                    self.vertices.insert(vertex.id.0.clone(), vertex.clone());
                    self.vertex_history
                        .entry(vertex.id.0.clone())
                        .or_default()
                        .push(vertex);
                }
                for edge in edges {
                    if self.vertices.contains_key(&edge.source_id.0)
                        && self.vertices.contains_key(&edge.target_id.0)
                    {
                        self.asserted_edges
                            .insert(KernelEdgeKey::from_edge(&edge), edge.clone());
                    }
                    self.asserted_edge_history
                        .entry(KernelEdgeKey::from_edge(&edge))
                        .or_default()
                        .push(edge);
                }
                Self::prune_edge_map_with_vertices(&mut self.candidate_edges, &self.vertices);
            }
            (KernelGraphLayer::Candidate, KernelMutationScope::Full) => {
                Self::expire_all_edges(&mut self.candidate_edge_history, recorded_at);
                self.candidate_scope_edges.clear();
                self.candidate_edges.clear();
                for edge in edges {
                    if self.vertices.contains_key(&edge.source_id.0)
                        && self.vertices.contains_key(&edge.target_id.0)
                    {
                        self.candidate_edges
                            .insert(KernelEdgeKey::from_edge(&edge), edge.clone());
                    }
                    self.candidate_edge_history
                        .entry(KernelEdgeKey::from_edge(&edge))
                        .or_default()
                        .push(edge);
                }
            }
            _ => {
                return Err(GraphBackendError::Operation(
                    "kernel graph mutation scope and layer were incompatible".to_owned(),
                ));
            }
        }
        self.mark_all_sidecars_dirty();
        Ok(())
    }

    fn ensure_ready(&self) -> Result<(), GraphBackendError> {
        if self.invalidated {
            return Err(GraphBackendError::Invalidated);
        }
        Ok(())
    }

    fn replace_vertex_scope(
        history: &mut FxHashMap<String, Vec<KernelVertex>>,
        active: &mut FxHashMap<String, KernelVertex>,
        scope_map: &mut FxHashMap<String, FxHashSet<String>>,
        scope_key: String,
        recorded_at: i64,
        new_vertices: Vec<KernelVertex>,
    ) {
        if let Some(existing) = scope_map.remove(&scope_key) {
            for vertex_id in existing {
                expire_current_vertex(history, &vertex_id, recorded_at);
                active.remove(&vertex_id);
            }
        }
        let scope_vertices = new_vertices
            .iter()
            .map(|vertex| vertex.id.0.clone())
            .collect::<FxHashSet<_>>();
        for vertex in new_vertices {
            active.insert(vertex.id.0.clone(), vertex.clone());
            history.entry(vertex.id.0.clone()).or_default().push(vertex);
        }
        scope_map.insert(scope_key, scope_vertices);
    }

    fn replace_edge_scope(
        history: &mut FxHashMap<KernelEdgeKey, Vec<KernelEdge>>,
        active: &mut FxHashMap<KernelEdgeKey, KernelEdge>,
        visible_vertices: &FxHashMap<String, KernelVertex>,
        scope_map: &mut FxHashMap<String, FxHashSet<KernelEdgeKey>>,
        scope_key: String,
        recorded_at: i64,
        new_edges: Vec<KernelEdge>,
    ) {
        if let Some(existing) = scope_map.remove(&scope_key) {
            for edge_key in existing {
                expire_current_edge(history, &edge_key, recorded_at);
                active.remove(&edge_key);
            }
        }
        let scope_edges = new_edges
            .iter()
            .map(KernelEdgeKey::from_edge)
            .collect::<FxHashSet<_>>();
        for edge in new_edges {
            let edge_key = KernelEdgeKey::from_edge(&edge);
            if visible_vertices.contains_key(&edge.source_id.0)
                && visible_vertices.contains_key(&edge.target_id.0)
            {
                active.insert(edge_key.clone(), edge.clone());
            } else {
                active.remove(&edge_key);
            }
            history.entry(edge_key).or_default().push(edge);
        }
        scope_map.insert(scope_key, scope_edges);
    }

    fn expire_all_vertices(history: &mut FxHashMap<String, Vec<KernelVertex>>, recorded_at: i64) {
        for vertex_id in history.keys().cloned().collect::<Vec<_>>() {
            expire_current_vertex(history, &vertex_id, recorded_at);
        }
    }

    fn expire_all_edges(history: &mut FxHashMap<KernelEdgeKey, Vec<KernelEdge>>, recorded_at: i64) {
        for edge_key in history.keys().cloned().collect::<Vec<_>>() {
            expire_current_edge(history, &edge_key, recorded_at);
        }
    }

    fn rebuild_active_views_from_history(&mut self) {
        let valid_at = now_ms();
        let tx_at = valid_at;
        self.vertices = visible_vertices(&self.vertex_history, valid_at, tx_at)
            .into_iter()
            .map(|vertex| (vertex.id.0.clone(), vertex))
            .collect();
        let vertex_ids = self.vertices.keys().cloned().collect::<FxHashSet<_>>();
        self.asserted_edges =
            visible_edges(&self.asserted_edge_history, valid_at, tx_at, &vertex_ids)
                .into_iter()
                .map(|edge| (KernelEdgeKey::from_edge(&edge), edge))
                .collect();
        self.candidate_edges =
            visible_edges(&self.candidate_edge_history, valid_at, tx_at, &vertex_ids)
                .into_iter()
                .map(|edge| (KernelEdgeKey::from_edge(&edge), edge))
                .collect();
    }

    fn prune_edge_map_with_vertices(
        edges: &mut FxHashMap<KernelEdgeKey, KernelEdge>,
        vertices: &FxHashMap<String, KernelVertex>,
    ) {
        edges.retain(|_, edge| {
            vertices.contains_key(&edge.source_id.0) && vertices.contains_key(&edge.target_id.0)
        });
    }

    fn mark_all_sidecars_dirty(&self) {
        *self
            .dirty_flags
            .write()
            .expect("kernel dirty flags poisoned") = KernelDirtyFlags {
            csr: true,
            valid_time_index: true,
            transaction_time_index: true,
            entity_index: true,
            calendar_index: true,
        };
    }

    fn ensure_csr_sidecar(&self) {
        let mut flags = self
            .dirty_flags
            .write()
            .expect("kernel dirty flags poisoned");
        if !flags.csr {
            return;
        }
        *self.csr.write().expect("kernel csr sidecar poisoned") =
            build_csr_sidecar(&self.vertices, &self.asserted_edges);
        flags.csr = false;
    }

    fn ensure_valid_time_index(&self) {
        let mut flags = self
            .dirty_flags
            .write()
            .expect("kernel dirty flags poisoned");
        if !flags.valid_time_index {
            return;
        }
        *self
            .valid_time_index
            .write()
            .expect("kernel valid-time sidecar poisoned") = build_temporal_sidecar(
            &self.vertex_history,
            &self.asserted_edge_history,
            &self.candidate_edge_history,
            TemporalAxis::Valid,
        );
        flags.valid_time_index = false;
    }

    fn ensure_transaction_time_index(&self) {
        let mut flags = self
            .dirty_flags
            .write()
            .expect("kernel dirty flags poisoned");
        if !flags.transaction_time_index {
            return;
        }
        *self
            .transaction_time_index
            .write()
            .expect("kernel transaction-time sidecar poisoned") = build_temporal_sidecar(
            &self.vertex_history,
            &self.asserted_edge_history,
            &self.candidate_edge_history,
            TemporalAxis::Transaction,
        );
        flags.transaction_time_index = false;
    }

    fn ensure_entity_index(&self) {
        let mut flags = self
            .dirty_flags
            .write()
            .expect("kernel dirty flags poisoned");
        if !flags.entity_index {
            return;
        }
        let active_vertices = self.vertices.values().cloned().collect::<Vec<_>>();
        let active_asserted = self.asserted_edges.values().cloned().collect::<Vec<_>>();
        let active_candidate = self.candidate_edges.values().cloned().collect::<Vec<_>>();
        *self
            .entity_index
            .write()
            .expect("kernel entity sidecar poisoned") =
            build_entity_sidecar(&active_vertices, &active_asserted, &active_candidate, true);
        flags.entity_index = false;
    }

    fn ensure_calendar_index(&self) {
        let mut flags = self
            .dirty_flags
            .write()
            .expect("kernel dirty flags poisoned");
        if !flags.calendar_index {
            return;
        }
        let active_vertices = self.vertices.values().cloned().collect::<Vec<_>>();
        let active_asserted = self.asserted_edges.values().cloned().collect::<Vec<_>>();
        *self
            .calendar_index
            .write()
            .expect("kernel calendar sidecar poisoned") =
            build_calendar_sidecar(&active_vertices, &active_asserted);
        flags.calendar_index = false;
    }
}

impl PhoenixGraphBackend for PhoenixGraphKernel {
    fn apply_batch(&mut self, batch: GraphMutationBatch) -> Result<(), GraphBackendError> {
        self.apply_kernel_batch(KernelMutationBatch::from(batch))
    }

    fn rebuild_from_batches(
        &mut self,
        batches: Vec<GraphMutationBatch>,
    ) -> Result<(), GraphBackendError> {
        self.rebuild_from_kernel_batches(
            batches.into_iter().map(KernelMutationBatch::from).collect(),
        )
    }

    fn snapshot(&self, include_candidate_graph: bool) -> Result<GraptorGraph, GraphBackendError> {
        self.ensure_ready()?;
        Ok(self.snapshot_legacy(include_candidate_graph))
    }

    fn counts(&self) -> Result<GraphCounts, GraphBackendError> {
        self.ensure_ready()?;
        Ok(GraphCounts {
            vertex_count: self.vertices.len(),
            asserted_edge_count: self.asserted_edges.len(),
            candidate_edge_count: self
                .candidate_edges
                .values()
                .cloned()
                .map(GraptorEdge::from)
                .filter(candidate_edge_is_active)
                .count(),
        })
    }

    fn candidate_edges(&self) -> Result<Vec<GraphEdgeRecord>, GraphBackendError> {
        self.ensure_ready()?;
        Ok(self
            .candidate_edges
            .values()
            .cloned()
            .map(GraphEdgeRecord::from)
            .collect())
    }

    fn invalidate(&mut self) {
        self.invalidated = true;
    }

    fn rebuild_token(&self) -> Option<&str> {
        self.rebuild_token.as_deref()
    }

    fn set_rebuild_token(&mut self, token: Option<String>) {
        self.rebuild_token = token;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl From<GraphLayer> for KernelGraphLayer {
    fn from(value: GraphLayer) -> Self {
        match value {
            GraphLayer::Asserted => Self::Asserted,
            GraphLayer::Candidate => Self::Candidate,
        }
    }
}

impl From<KernelGraphLayer> for GraphLayer {
    fn from(value: KernelGraphLayer) -> Self {
        match value {
            KernelGraphLayer::Asserted => Self::Asserted,
            KernelGraphLayer::Candidate => Self::Candidate,
        }
    }
}

impl From<GraphMutationScope> for KernelMutationScope {
    fn from(value: GraphMutationScope) -> Self {
        match value {
            GraphMutationScope::Document { document_id } => Self::Document { document_id },
            GraphMutationScope::Session { session_id } => Self::Session { session_id },
            GraphMutationScope::Candidate { scope_key } => Self::Candidate { scope_key },
            GraphMutationScope::Projection { scope_key } => Self::Projection { scope_key },
            GraphMutationScope::Full => Self::Full,
        }
    }
}

impl From<KernelMutationScope> for GraphMutationScope {
    fn from(value: KernelMutationScope) -> Self {
        match value {
            KernelMutationScope::Document { document_id } => Self::Document { document_id },
            KernelMutationScope::Session { session_id } => Self::Session { session_id },
            KernelMutationScope::Candidate { scope_key } => Self::Candidate { scope_key },
            KernelMutationScope::Projection { scope_key } => Self::Projection { scope_key },
            KernelMutationScope::Full => Self::Full,
        }
    }
}

impl From<GraphVertexRecord> for KernelVertex {
    fn from(value: GraphVertexRecord) -> Self {
        let labels = value
            .attributes
            .get("labels")
            .and_then(Value::as_array)
            .map(|labels| {
                labels
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let class = kernel_meta_field(&value.attributes, "class")
            .unwrap_or_else(|| infer_vertex_class(&value.kind));
        let temporal = kernel_meta_field(&value.attributes, "temporal").unwrap_or_default();
        let provenance = kernel_meta_field(&value.attributes, "provenance").unwrap_or_default();
        let entity_facet = kernel_meta_field(&value.attributes, "entityFacet");
        let calendar_facet = kernel_meta_field(&value.attributes, "calendarFacet");
        Self {
            id: KernelVertexId(value.id),
            kind: value.kind,
            class,
            labels,
            weight: value.weight,
            value: value.value,
            attributes: strip_kernel_meta(value.attributes),
            temporal,
            provenance,
            entity_id: value.entity_id,
            search_chunk_id: value.search_chunk_id,
            document_id: value.document_id,
            chapter_id: value.chapter_id,
            chapters: value.chapters,
            boundary_id: value.boundary_id,
            boundary_ordinal: value.boundary_ordinal,
            boundary_kind: value.boundary_kind,
            boundary_ordinals: value.boundary_ordinals,
            entity_facet,
            calendar_facet,
        }
    }
}

impl From<KernelVertex> for GraphVertexRecord {
    fn from(value: KernelVertex) -> Self {
        let mut attributes = value.attributes.clone();
        if !value.labels.is_empty() {
            if let Some(object) = attributes.as_object_mut() {
                object.insert(
                    "labels".to_owned(),
                    Value::Array(value.labels.iter().cloned().map(Value::String).collect()),
                );
            }
        }
        attach_kernel_meta(
            &mut attributes,
            &[
                ("class", serde_json::to_value(&value.class).ok()),
                ("temporal", serde_json::to_value(&value.temporal).ok()),
                ("provenance", serde_json::to_value(&value.provenance).ok()),
                (
                    "entityFacet",
                    serde_json::to_value(&value.entity_facet).ok(),
                ),
                (
                    "calendarFacet",
                    serde_json::to_value(&value.calendar_facet).ok(),
                ),
            ],
        );
        Self {
            id: value.id.0,
            kind: value.kind,
            weight: value.weight,
            value: value.value,
            attributes,
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

impl From<GraptorVertex> for KernelVertex {
    fn from(value: GraptorVertex) -> Self {
        Self::from(GraphVertexRecord::from(&value))
    }
}

impl From<KernelVertex> for GraptorVertex {
    fn from(value: KernelVertex) -> Self {
        GraptorVertex::from(GraphVertexRecord::from(value))
    }
}

impl From<GraphEdgeRecord> for KernelEdge {
    fn from(value: GraphEdgeRecord) -> Self {
        let relation_class = kernel_meta_field(&value.attributes, "relationClass")
            .unwrap_or_else(|| infer_relation_class(&value.edge_type));
        let temporal = kernel_meta_field(&value.attributes, "temporal").unwrap_or_default();
        let provenance = kernel_meta_field(&value.attributes, "provenance").unwrap_or_default();
        let resolution_facet = kernel_meta_field(&value.attributes, "resolutionFacet");
        Self {
            source_id: KernelVertexId(value.source_id),
            target_id: KernelVertexId(value.target_id),
            edge_type: KernelEdgeType(value.edge_type),
            relation_class,
            weight: value.weight,
            attributes: strip_kernel_meta(value.attributes),
            data: value.data,
            document_id: value.document_id,
            narrative_id: value.narrative_id,
            layer: value.layer.into(),
            temporal,
            provenance,
            resolution_facet,
        }
    }
}

impl From<KernelEdge> for GraphEdgeRecord {
    fn from(value: KernelEdge) -> Self {
        let mut attributes = value.attributes.clone();
        attach_kernel_meta(
            &mut attributes,
            &[
                (
                    "relationClass",
                    serde_json::to_value(&value.relation_class).ok(),
                ),
                ("temporal", serde_json::to_value(&value.temporal).ok()),
                ("provenance", serde_json::to_value(&value.provenance).ok()),
                (
                    "resolutionFacet",
                    serde_json::to_value(&value.resolution_facet).ok(),
                ),
            ],
        );
        Self {
            source_id: value.source_id.0,
            target_id: value.target_id.0,
            edge_type: value.edge_type.0,
            weight: value.weight,
            attributes,
            data: value.data,
            document_id: value.document_id,
            narrative_id: value.narrative_id,
            layer: value.layer.into(),
        }
    }
}

impl From<GraptorEdge> for KernelEdge {
    fn from(value: GraptorEdge) -> Self {
        Self::from(GraphEdgeRecord::from(&value))
    }
}

impl From<KernelEdge> for GraptorEdge {
    fn from(value: KernelEdge) -> Self {
        GraptorEdge::from(GraphEdgeRecord::from(value))
    }
}

impl From<GraphMutationBatch> for KernelMutationBatch {
    fn from(value: GraphMutationBatch) -> Self {
        Self {
            layer: value.layer.into(),
            scope: value.scope.into(),
            recorded_at: None,
            vertices: value.vertices.into_iter().map(KernelVertex::from).collect(),
            edges: value.edges.into_iter().map(KernelEdge::from).collect(),
        }
    }
}

impl From<KernelMutationBatch> for GraphMutationBatch {
    fn from(value: KernelMutationBatch) -> Self {
        Self {
            layer: value.layer.into(),
            scope: value.scope.into(),
            vertices: value
                .vertices
                .into_iter()
                .map(GraphVertexRecord::from)
                .collect(),
            edges: value.edges.into_iter().map(GraphEdgeRecord::from).collect(),
        }
    }
}

impl From<&GraptorGraph> for KernelGraphSnapshot {
    fn from(value: &GraptorGraph) -> Self {
        let kernel = PhoenixGraphKernel::from_legacy_graph(value);
        kernel.snapshot_kernel()
    }
}

fn candidate_edge_is_active(edge: &GraptorEdge) -> bool {
    !matches!(
        edge.attributes
            .get("graph")
            .and_then(|graph| graph.get("status"))
            .and_then(|status| status.as_str()),
        Some("candidate_rejected")
    )
}

#[derive(Clone, Copy)]
enum TemporalAxis {
    Valid,
    Transaction,
}

fn normalize_vertex(mut vertex: KernelVertex, recorded_at: i64) -> KernelVertex {
    vertex.class = if matches!(vertex.class, KernelVertexClass::Generic) {
        infer_vertex_class(&vertex.kind)
    } else {
        vertex.class
    };
    vertex.temporal = vertex.temporal.with_default_recorded_at(recorded_at);
    if vertex.entity_facet.is_none()
        && (matches!(
            vertex.class,
            KernelVertexClass::Entity | KernelVertexClass::Alias | KernelVertexClass::Mention
        ) || vertex.entity_id.is_some())
    {
        vertex.entity_facet = Some(KernelEntityFacet {
            canonical_entity_id: vertex.entity_id.clone(),
            surface: surface_for_vertex(&vertex),
            entity_kind: Some(vertex.kind.clone()),
        });
    }
    vertex
}

fn normalize_edge(mut edge: KernelEdge, recorded_at: i64, layer: &KernelGraphLayer) -> KernelEdge {
    edge.layer = layer.clone();
    edge.relation_class = if matches!(edge.relation_class, KernelRelationClass::Custom) {
        infer_relation_class(&edge.edge_type.0)
    } else {
        edge.relation_class
    };
    edge.temporal = edge.temporal.with_default_recorded_at(recorded_at);
    if edge.provenance.confidence.is_none() && edge.weight > 0 {
        edge.provenance.confidence = Some(edge.weight as f64);
    }
    edge
}

fn infer_vertex_class(kind: &str) -> KernelVertexClass {
    match kind {
        "document" | "doc" => KernelVertexClass::Document,
        "chunk" | "leaf" => KernelVertexClass::Chunk,
        "entity" => KernelVertexClass::Entity,
        "alias" => KernelVertexClass::Alias,
        "mention" => KernelVertexClass::Mention,
        "time_anchor" => KernelVertexClass::TimeAnchor,
        "calendar_anchor" | "year" | "month" | "week" | "day" | "hour" => {
            KernelVertexClass::CalendarAnchor
        }
        "narrative" => KernelVertexClass::Narrative,
        "episode" => KernelVertexClass::Episode,
        "memory" => KernelVertexClass::Memory,
        "task" => KernelVertexClass::Task,
        "state" => KernelVertexClass::State,
        "event" => KernelVertexClass::Event,
        _ => KernelVertexClass::Generic,
    }
}

fn infer_relation_class(edge_type: &str) -> KernelRelationClass {
    match edge_type {
        "contains" | "precedes" => KernelRelationClass::Calendar,
        "occurs_at" | "starts_at" | "ends_at" | "scheduled_for" => KernelRelationClass::Temporal,
        "alias_of" => KernelRelationClass::Identity,
        "mentions" | "candidate_same_as" | "resolved_to" | "evidence_for" => {
            KernelRelationClass::Resolution
        }
        "entity" | "related_to" | "supports" | "similar_to" => KernelRelationClass::Semantic,
        "candidate_corefers_with" => KernelRelationClass::Candidate,
        _ => KernelRelationClass::Custom,
    }
}

fn expire_current_vertex(
    history: &mut FxHashMap<String, Vec<KernelVertex>>,
    vertex_id: &str,
    expired_at: i64,
) {
    if let Some(records) = history.get_mut(vertex_id) {
        if let Some(record) = records
            .iter_mut()
            .rev()
            .find(|record| record.temporal.expired_at.is_none())
        {
            record.temporal.expired_at = Some(expired_at);
        }
    }
}

fn expire_current_edge(
    history: &mut FxHashMap<KernelEdgeKey, Vec<KernelEdge>>,
    edge_key: &KernelEdgeKey,
    expired_at: i64,
) {
    if let Some(records) = history.get_mut(edge_key) {
        if let Some(record) = records
            .iter_mut()
            .rev()
            .find(|record| record.temporal.expired_at.is_none())
        {
            record.temporal.expired_at = Some(expired_at);
        }
    }
}

fn visible_vertices(
    history: &FxHashMap<String, Vec<KernelVertex>>,
    valid_at: i64,
    tx_at: i64,
) -> Vec<KernelVertex> {
    let mut visible = Vec::new();
    for records in history.values() {
        if let Some(record) = records
            .iter()
            .rev()
            .find(|record| record.temporal.is_visible_at(valid_at, tx_at))
        {
            visible.push(record.clone());
        }
    }
    visible
}

fn visible_edges(
    history: &FxHashMap<KernelEdgeKey, Vec<KernelEdge>>,
    valid_at: i64,
    tx_at: i64,
    visible_vertices: &FxHashSet<String>,
) -> Vec<KernelEdge> {
    let mut visible = Vec::new();
    for records in history.values() {
        if let Some(record) = records
            .iter()
            .rev()
            .find(|record| record.temporal.is_visible_at(valid_at, tx_at))
        {
            if visible_vertices.contains(&record.source_id.0)
                && visible_vertices.contains(&record.target_id.0)
            {
                visible.push(record.clone());
            }
        }
    }
    visible
}

fn build_csr_sidecar(
    vertices: &FxHashMap<String, KernelVertex>,
    asserted_edges: &FxHashMap<KernelEdgeKey, KernelEdge>,
) -> KernelCsrSidecar {
    let mut vertex_ids = vertices.keys().cloned().collect::<Vec<_>>();
    vertex_ids.sort();
    let dense = vertex_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (id.clone(), index))
        .collect::<FxHashMap<_, _>>();
    let mut adjacency = vec![Vec::<(usize, f64)>::new(); vertex_ids.len()];
    for edge in asserted_edges.values() {
        let Some(source_ix) = dense.get(&edge.source_id.0).copied() else {
            continue;
        };
        let Some(target_ix) = dense.get(&edge.target_id.0).copied() else {
            continue;
        };
        adjacency[source_ix].push((target_ix, edge.weight.max(1) as f64));
    }
    for row in &mut adjacency {
        row.sort_by_key(|(target, _)| *target);
    }
    let mut offsets = Vec::with_capacity(vertex_ids.len() + 1);
    let mut targets = Vec::new();
    let mut weights = Vec::new();
    offsets.push(0);
    for row in adjacency {
        for (target, weight) in row {
            targets.push(target);
            weights.push(weight);
        }
        offsets.push(targets.len());
    }
    KernelCsrSidecar {
        vertex_ids,
        offsets,
        targets,
        weights,
        dense,
    }
}

fn build_temporal_sidecar(
    vertices: &FxHashMap<String, Vec<KernelVertex>>,
    asserted_edges: &FxHashMap<KernelEdgeKey, Vec<KernelEdge>>,
    candidate_edges: &FxHashMap<KernelEdgeKey, Vec<KernelEdge>>,
    axis: TemporalAxis,
) -> KernelTemporalIndexSidecar {
    let mut sidecar = KernelTemporalIndexSidecar::default();
    sidecar.vertices = vertices
        .iter()
        .flat_map(|(vertex_id, records)| {
            records.iter().map(|record| KernelTemporalIndexEntry {
                record_id: vertex_id.clone(),
                valid_from: record.temporal.valid_from,
                valid_to: record.temporal.valid_to,
                recorded_at: record.temporal.recorded_at,
                expired_at: record.temporal.expired_at,
            })
        })
        .collect();
    sidecar.asserted_edges = asserted_edges
        .iter()
        .flat_map(|(edge_key, records)| {
            records.iter().map(|record| KernelTemporalIndexEntry {
                record_id: edge_key.storage_key(),
                valid_from: record.temporal.valid_from,
                valid_to: record.temporal.valid_to,
                recorded_at: record.temporal.recorded_at,
                expired_at: record.temporal.expired_at,
            })
        })
        .collect();
    sidecar.candidate_edges = candidate_edges
        .iter()
        .flat_map(|(edge_key, records)| {
            records.iter().map(|record| KernelTemporalIndexEntry {
                record_id: edge_key.storage_key(),
                valid_from: record.temporal.valid_from,
                valid_to: record.temporal.valid_to,
                recorded_at: record.temporal.recorded_at,
                expired_at: record.temporal.expired_at,
            })
        })
        .collect();

    let key = move |entry: &KernelTemporalIndexEntry| match axis {
        TemporalAxis::Valid => (
            entry.valid_from.unwrap_or(i64::MIN),
            entry.valid_to.unwrap_or(i64::MAX),
        ),
        TemporalAxis::Transaction => (
            entry.recorded_at.unwrap_or(i64::MIN),
            entry.expired_at.unwrap_or(i64::MAX),
        ),
    };
    sidecar.vertices.sort_by_key(key);
    sidecar.asserted_edges.sort_by_key(key);
    sidecar.candidate_edges.sort_by_key(key);
    sidecar
}

fn build_entity_sidecar(
    vertices: &[KernelVertex],
    asserted_edges: &[KernelEdge],
    candidate_edges: &[KernelEdge],
    include_candidate_graph: bool,
) -> KernelEntitySidecar {
    let mut sidecar = KernelEntitySidecar::default();
    let vertex_by_id = vertices
        .iter()
        .map(|vertex| (vertex.id.0.clone(), vertex))
        .collect::<FxHashMap<_, _>>();

    for vertex in vertices {
        if matches!(vertex.class, KernelVertexClass::Entity) {
            let entity_id = vertex
                .entity_id
                .clone()
                .or_else(|| {
                    vertex
                        .entity_facet
                        .as_ref()
                        .and_then(|facet| facet.canonical_entity_id.clone())
                })
                .unwrap_or_else(|| vertex.id.0.clone());
            sidecar.canonical_support.entry(entity_id).or_default();
            if let Some(surface) = surface_for_vertex(vertex) {
                sidecar
                    .alias_candidates
                    .entry(normalize_surface(&surface))
                    .or_default()
                    .push(KernelEntityCandidate {
                        entity_id: vertex
                            .entity_id
                            .clone()
                            .or_else(|| {
                                vertex
                                    .entity_facet
                                    .as_ref()
                                    .and_then(|facet| facet.canonical_entity_id.clone())
                            })
                            .unwrap_or_else(|| vertex.id.0.clone()),
                        score: 1.0,
                        source_vertex_id: Some(vertex.id.0.clone()),
                        relation_type: Some("canonical".to_owned()),
                        evidence_refs: Vec::new(),
                    });
            }
        }
    }

    let mut register_edge = |edge: &KernelEdge| {
        let source = vertex_by_id.get(&edge.source_id.0).copied();
        let target = vertex_by_id.get(&edge.target_id.0).copied();
        let Some(target) = target else {
            return;
        };
        let target_entity_id = target
            .entity_id
            .clone()
            .or_else(|| {
                target
                    .entity_facet
                    .as_ref()
                    .and_then(|facet| facet.canonical_entity_id.clone())
            })
            .unwrap_or_else(|| target.id.0.clone());
        let support = sidecar
            .canonical_support
            .entry(target_entity_id.clone())
            .or_default();
        support
            .evidence_refs
            .extend(edge.provenance.evidence_refs.iter().cloned());

        match edge.edge_type.0.as_str() {
            "alias_of" => {
                if let Some(source) = source {
                    support.alias_vertex_ids.push(source.id.0.clone());
                    if let Some(surface) = surface_for_vertex(source) {
                        sidecar
                            .alias_candidates
                            .entry(normalize_surface(&surface))
                            .or_default()
                            .push(KernelEntityCandidate {
                                entity_id: target_entity_id,
                                score: edge
                                    .resolution_facet
                                    .as_ref()
                                    .and_then(|facet| facet.confidence)
                                    .or(edge.provenance.confidence)
                                    .unwrap_or(edge.weight as f64),
                                source_vertex_id: Some(source.id.0.clone()),
                                relation_type: Some("alias_of".to_owned()),
                                evidence_refs: edge.provenance.evidence_refs.clone(),
                            });
                    }
                }
            }
            "mentions" | "resolved_to" => {
                if let Some(source) = source {
                    if matches!(source.class, KernelVertexClass::Mention) {
                        support.mention_vertex_ids.push(source.id.0.clone());
                        if edge.edge_type.0 == "resolved_to" {
                            sidecar
                                .mention_entities
                                .insert(source.id.0.clone(), target_entity_id.clone());
                        }
                    }
                }
            }
            "candidate_same_as" => {
                if let Some(source) = source {
                    let surface = source
                        .entity_facet
                        .as_ref()
                        .and_then(|facet| facet.surface.clone())
                        .unwrap_or_else(|| source.id.0.clone());
                    sidecar
                        .alias_candidates
                        .entry(normalize_surface(&surface))
                        .or_default()
                        .push(KernelEntityCandidate {
                            entity_id: target_entity_id,
                            score: edge
                                .resolution_facet
                                .as_ref()
                                .and_then(|facet| facet.confidence)
                                .or(edge.provenance.confidence)
                                .unwrap_or(edge.weight as f64),
                            source_vertex_id: Some(source.id.0.clone()),
                            relation_type: Some("candidate_same_as".to_owned()),
                            evidence_refs: edge.provenance.evidence_refs.clone(),
                        });
                }
            }
            "evidence_for" => {
                if let Some(source) = source {
                    support.evidence_refs.push(source.id.0.clone());
                }
            }
            _ => {}
        }
    };

    for edge in asserted_edges {
        register_edge(edge);
    }
    if include_candidate_graph {
        for edge in candidate_edges {
            register_edge(edge);
        }
    }

    for support in sidecar.canonical_support.values_mut() {
        support.alias_vertex_ids.sort();
        support.alias_vertex_ids.dedup();
        support.mention_vertex_ids.sort();
        support.mention_vertex_ids.dedup();
        support.evidence_refs.sort();
        support.evidence_refs.dedup();
    }
    for candidates in sidecar.alias_candidates.values_mut() {
        candidates.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.entity_id.cmp(&right.entity_id))
        });
        candidates.dedup_by(|left, right| left.entity_id == right.entity_id);
    }
    sidecar
}

fn build_calendar_sidecar(
    vertices: &[KernelVertex],
    asserted_edges: &[KernelEdge],
) -> KernelCalendarSidecar {
    let mut sidecar = KernelCalendarSidecar::default();
    for vertex in vertices {
        if matches!(
            vertex.class,
            KernelVertexClass::CalendarAnchor | KernelVertexClass::TimeAnchor
        ) {
            sidecar.anchors.insert(vertex.id.0.clone(), vertex.clone());
        }
    }
    for edge in asserted_edges {
        if matches!(
            edge.relation_class,
            KernelRelationClass::Calendar | KernelRelationClass::Temporal
        ) {
            sidecar
                .adjacency
                .entry(edge.source_id.0.clone())
                .or_default()
                .push(edge.clone());
            if sidecar.anchors.contains_key(&edge.target_id.0) {
                sidecar
                    .anchor_members
                    .entry(edge.target_id.0.clone())
                    .or_default()
                    .push(edge.source_id.0.clone());
            }
        }
    }
    for members in sidecar.anchor_members.values_mut() {
        members.sort();
        members.dedup();
    }
    sidecar
}

fn collect_slot_states(
    vertices: &[KernelVertex],
    asserted_edges: &[KernelEdge],
    entity_id: &str,
    slot_key: &str,
) -> Vec<KernelSlotState> {
    vertices
        .iter()
        .filter(|vertex| {
            vertex.kind == "state"
                && vertex.entity_id.as_deref() == Some(entity_id)
                && vertex_slot_key(vertex) == Some(slot_key)
        })
        .map(|vertex| build_slot_state(vertex, asserted_edges, entity_id))
        .collect()
}

fn collect_state_issues(
    vertices: &[KernelVertex],
    asserted_edges: &[KernelEdge],
    issue_kind: &str,
    entity_id: &str,
    slot_key: Option<&str>,
    unresolved_only: bool,
) -> Vec<KernelStateIssue> {
    vertices
        .iter()
        .filter(|vertex| {
            vertex.kind == issue_kind
                && vertex.entity_id.as_deref() == Some(entity_id)
                && slot_key
                    .map(|slot_key| vertex_slot_key(vertex) == Some(slot_key))
                    .unwrap_or(true)
        })
        .filter_map(|vertex| {
            let issue = KernelStateIssue {
                issue_vertex_id: vertex.id.0.clone(),
                issue_kind: issue_kind.to_owned(),
                entity_id: entity_id.to_owned(),
                slot_key: vertex_slot_key(vertex).unwrap_or_default().to_owned(),
                status: vertex_string_field(vertex, "status").map(str::to_owned),
                reason: vertex_string_field(vertex, "kind").map(str::to_owned),
                detail: vertex_string_field(vertex, "detail").map(str::to_owned),
                preferred_claim_id: vertex_string_field(vertex, "preferredClaimId")
                    .map(str::to_owned),
                temporal: vertex.temporal.clone(),
                supporting_claim_ids: supporting_claim_ids(&vertex.id.0, asserted_edges),
            };
            if unresolved_only
                && !issue
                    .status
                    .as_deref()
                    .map(is_unresolved_status)
                    .unwrap_or(true)
            {
                return None;
            }
            Some(issue)
        })
        .collect()
}

fn build_slot_state(
    vertex: &KernelVertex,
    asserted_edges: &[KernelEdge],
    entity_id: &str,
) -> KernelSlotState {
    KernelSlotState {
        state_vertex_id: vertex.id.0.clone(),
        entity_id: entity_id.to_owned(),
        slot_key: vertex_slot_key(vertex).unwrap_or_default().to_owned(),
        value: vertex_string_field(vertex, "value")
            .unwrap_or_default()
            .to_owned(),
        value_entity_id: vertex_string_field(vertex, "valueEntityId").map(str::to_owned),
        status: vertex_string_field(vertex, "status").map(str::to_owned),
        source_class: vertex_string_field(vertex, "sourceClass").map(str::to_owned),
        confidence: vertex_confidence(vertex),
        temporal: vertex.temporal.clone(),
        supporting_claim_ids: supporting_claim_ids(&vertex.id.0, asserted_edges),
        evidence_refs: vertex.provenance.evidence_refs.clone(),
    }
}

fn vertex_slot_key(vertex: &KernelVertex) -> Option<&str> {
    vertex_string_field(vertex, "slotKey")
}

fn vertex_string_field<'a>(vertex: &'a KernelVertex, field: &str) -> Option<&'a str> {
    vertex
        .value
        .get(field)
        .and_then(Value::as_str)
        .or_else(|| vertex.attributes.get(field).and_then(Value::as_str))
}

fn vertex_confidence(vertex: &KernelVertex) -> Option<f64> {
    vertex.provenance.confidence.or_else(|| {
        vertex
            .attributes
            .get("confidenceMillis")
            .and_then(Value::as_u64)
            .map(|value| value as f64 / 1000.0)
    })
}

fn supporting_claim_ids(source_vertex_id: &str, asserted_edges: &[KernelEdge]) -> Vec<String> {
    let mut claim_ids = asserted_edges
        .iter()
        .filter(|edge| edge.source_id.0 == source_vertex_id && edge.edge_type.0 == "supported_by")
        .filter_map(|edge| claim_id_from_vertex_id(&edge.target_id.0))
        .collect::<Vec<_>>();
    claim_ids.sort();
    claim_ids.dedup();
    claim_ids
}

fn claim_id_from_vertex_id(vertex_id: &str) -> Option<String> {
    vertex_id
        .strip_prefix("graph::claim::")
        .map(str::to_owned)
        .or_else(|| vertex_id.strip_prefix("claim::").map(str::to_owned))
}

fn compare_slot_states(left: &KernelSlotState, right: &KernelSlotState) -> std::cmp::Ordering {
    state_status_rank(right.status.as_deref())
        .cmp(&state_status_rank(left.status.as_deref()))
        .then_with(|| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| {
            right
                .supporting_claim_ids
                .len()
                .cmp(&left.supporting_claim_ids.len())
        })
        .then_with(|| right.temporal.valid_from.cmp(&left.temporal.valid_from))
        .then_with(|| left.state_vertex_id.cmp(&right.state_vertex_id))
}

fn compare_state_issues(left: &KernelStateIssue, right: &KernelStateIssue) -> std::cmp::Ordering {
    left.slot_key
        .cmp(&right.slot_key)
        .then_with(|| left.issue_kind.cmp(&right.issue_kind))
        .then_with(|| right.temporal.valid_from.cmp(&left.temporal.valid_from))
        .then_with(|| left.issue_vertex_id.cmp(&right.issue_vertex_id))
}

fn state_status_rank(status: Option<&str>) -> u8 {
    match status.unwrap_or_default() {
        "supported" => 6,
        "active" => 5,
        "candidate" => 4,
        "deferred" => 3,
        "contradicted" => 2,
        "superseded" => 1,
        "rejected" => 0,
        _ => 0,
    }
}

fn is_unresolved_status(status: &str) -> bool {
    !matches!(status, "rejected" | "superseded")
}

fn classify_state_change(
    temporal: &KernelBiTemporal,
    since_valid_at: i64,
    until_valid_at: i64,
) -> Option<KernelStateChangeKind> {
    let activated = temporal
        .valid_from
        .is_some_and(|value| value >= since_valid_at && value < until_valid_at);
    let expired = temporal
        .valid_to
        .is_some_and(|value| value > since_valid_at && value <= until_valid_at);
    match (activated, expired) {
        (true, true) => Some(KernelStateChangeKind::ActivatedAndExpired),
        (true, false) => Some(KernelStateChangeKind::Activated),
        (false, true) => Some(KernelStateChangeKind::Expired),
        (false, false) => None,
    }
}

fn surface_for_vertex(vertex: &KernelVertex) -> Option<String> {
    vertex
        .entity_facet
        .as_ref()
        .and_then(|facet| facet.surface.clone())
        .or_else(|| {
            vertex
                .value
                .get("name")
                .or_else(|| vertex.value.get("text"))
                .or_else(|| vertex.value.get("label"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            vertex
                .attributes
                .get("surface")
                .or_else(|| vertex.attributes.get("label"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

fn normalize_surface(surface: &str) -> String {
    surface.trim().to_ascii_lowercase()
}

fn attach_kernel_meta(attributes: &mut Value, pairs: &[(&str, Option<Value>)]) {
    if !attributes.is_object() {
        *attributes = json!({});
    }
    let Some(root) = attributes.as_object_mut() else {
        return;
    };
    let kernel = root
        .entry("__kernel".to_owned())
        .or_insert_with(|| Value::Object(Default::default()));
    let Some(kernel_object) = kernel.as_object_mut() else {
        return;
    };
    for (field, value) in pairs {
        if let Some(value) = value {
            if !value.is_null() {
                kernel_object.insert((*field).to_owned(), value.clone());
            }
        }
    }
}

fn strip_kernel_meta(mut attributes: Value) -> Value {
    if let Some(object) = attributes.as_object_mut() {
        object.remove("__kernel");
    }
    attributes
}

fn kernel_meta_field<T>(attributes: &Value, field: &str) -> Option<T>
where
    T: DeserializeOwned,
{
    attributes
        .get("__kernel")
        .and_then(|kernel| kernel.get(field))
        .cloned()
        .and_then(|value| serde_json::from_value(value).ok())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

fn floor_div(value: i64, divisor: i64) -> i64 {
    let mut quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && ((remainder > 0) != (divisor > 0)) {
        quotient -= 1;
    }
    quotient
}

fn civil_from_days(days: i64) -> (i32, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if month <= 2 { 1 } else { 0 };
    (year as i32, month as u32, day as u32)
}

fn calendar_components(timestamp_ms: i64) -> (i32, u32, u32, u32, String) {
    let total_seconds = floor_div(timestamp_ms, 1_000);
    let days = floor_div(total_seconds, 86_400);
    let seconds_of_day = total_seconds - days * 86_400;
    let hour = floor_div(seconds_of_day, 3_600) as u32;
    let (year, month, day) = civil_from_days(days);
    let weekday_monday = (days + 3).rem_euclid(7);
    let week_start_days = days - weekday_monday;
    let (week_year, week_month, week_day) = civil_from_days(week_start_days);
    (
        year,
        month,
        day,
        hour,
        format!("{week_year}-{week_month:02}-{week_day:02}"),
    )
}

fn calendar_anchor_vertex(
    id: &str,
    kind: &str,
    granularity: KernelCalendarGranularity,
    timestamp_ms: i64,
    temporal: KernelBiTemporal,
    year: i32,
    month: Option<u32>,
    day: Option<u32>,
    hour: Option<u32>,
    week_start_day: Option<String>,
) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(id.to_owned()),
        kind: kind.to_owned(),
        class: KernelVertexClass::CalendarAnchor,
        labels: vec!["calendar".to_owned(), kind.to_owned()],
        weight: 1,
        value: json!({ "anchorKey": id }),
        attributes: json!({}),
        temporal,
        provenance: KernelProvenance::default(),
        entity_id: None,
        search_chunk_id: None,
        document_id: None,
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: Some(KernelCalendarFacet {
            granularity,
            anchor_key: Some(id.to_owned()),
            year: Some(year),
            month,
            week_start_day,
            day,
            hour,
            timestamp_ms: Some(timestamp_ms),
            interval_start_ms: Some(timestamp_ms),
            interval_end_ms: None,
        }),
    }
}

fn calendar_edge(
    source_id: &str,
    target_id: &str,
    edge_type: &str,
    recorded_at: i64,
) -> KernelEdge {
    KernelEdge {
        source_id: KernelVertexId(source_id.to_owned()),
        target_id: KernelVertexId(target_id.to_owned()),
        edge_type: KernelEdgeType(edge_type.to_owned()),
        relation_class: KernelRelationClass::Calendar,
        weight: 1,
        attributes: json!({}),
        data: None,
        document_id: None,
        narrative_id: None,
        layer: KernelGraphLayer::Asserted,
        temporal: KernelBiTemporal {
            valid_from: Some(recorded_at),
            valid_to: None,
            recorded_at: Some(recorded_at),
            expired_at: None,
        },
        provenance: KernelProvenance::default(),
        resolution_facet: None,
    }
}

fn dedup_vertices(vertices: &mut Vec<KernelVertex>) {
    let mut seen = FxHashSet::default();
    vertices.retain(|vertex| seen.insert(vertex.id.0.clone()));
}

fn dedup_edges(edges: &mut Vec<KernelEdge>) {
    let mut seen = FxHashSet::default();
    edges.retain(|edge| seen.insert(KernelEdgeKey::from_edge(edge)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn kernel_batches_build_deterministic_csr() {
        let mut kernel = PhoenixGraphKernel::new();
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Document {
                    document_id: "doc-1".to_owned(),
                },
                recorded_at: Some(10),
                vertices: vec![
                    KernelVertex {
                        id: KernelVertexId("a".to_owned()),
                        kind: "doc".to_owned(),
                        value: json!({}),
                        attributes: json!({}),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("b".to_owned()),
                        kind: "entity".to_owned(),
                        value: json!({}),
                        attributes: json!({}),
                        ..KernelVertex::default()
                    },
                ],
                edges: vec![KernelEdge {
                    source_id: KernelVertexId("a".to_owned()),
                    target_id: KernelVertexId("b".to_owned()),
                    edge_type: KernelEdgeType("mentions".to_owned()),
                    weight: 3,
                    attributes: json!({}),
                    layer: KernelGraphLayer::Asserted,
                    ..KernelEdge::default()
                }],
            })
            .expect("batch");
        let csr = kernel.csr_sidecar();
        assert_eq!(csr.vertex_ids, vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(csr.offsets, vec![0, 1, 1]);
        assert_eq!(csr.targets, vec![1]);
        assert_eq!(csr.weights, vec![3.0]);
    }

    #[test]
    fn legacy_roundtrip_adapters_preserve_shape() {
        let batch = GraphMutationBatch {
            layer: GraphLayer::Asserted,
            scope: GraphMutationScope::Document {
                document_id: "doc-1".to_owned(),
            },
            vertices: vec![GraphVertexRecord {
                id: "doc::1".to_owned(),
                kind: "document".to_owned(),
                weight: 1,
                value: json!({"kind":"document"}),
                attributes: json!({"documentId":"doc-1"}),
                document_id: Some("doc-1".to_owned()),
                ..GraphVertexRecord::default()
            }],
            edges: vec![GraphEdgeRecord {
                source_id: "doc::1".to_owned(),
                target_id: "entity::1".to_owned(),
                edge_type: "mentions".to_owned(),
                weight: 1,
                attributes: json!({"documentId":"doc-1"}),
                document_id: Some("doc-1".to_owned()),
                layer: GraphLayer::Asserted,
                ..GraphEdgeRecord::default()
            }],
        };
        let kernel_batch = KernelMutationBatch::from(batch.clone());
        let legacy = GraphMutationBatch::from(kernel_batch);
        assert_eq!(legacy.scope, batch.scope);
        assert_eq!(legacy.layer, batch.layer);
        assert_eq!(legacy.vertices.len(), 1);
        assert_eq!(legacy.edges.len(), 1);
    }

    #[test]
    fn view_as_of_preserves_prior_truth_and_prior_belief() {
        let mut kernel = PhoenixGraphKernel::new();
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Document {
                    document_id: "doc-1".to_owned(),
                },
                recorded_at: Some(10),
                vertices: vec![KernelVertex {
                    id: KernelVertexId("entity::river".to_owned()),
                    kind: "entity".to_owned(),
                    class: KernelVertexClass::Entity,
                    value: json!({"name":"River"}),
                    attributes: json!({}),
                    entity_id: Some("river".to_owned()),
                    temporal: KernelBiTemporal {
                        valid_from: Some(0),
                        valid_to: Some(50),
                        recorded_at: Some(10),
                        expired_at: None,
                    },
                    ..KernelVertex::default()
                }],
                edges: Vec::new(),
            })
            .expect("batch 1");
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Document {
                    document_id: "doc-1".to_owned(),
                },
                recorded_at: Some(20),
                vertices: vec![KernelVertex {
                    id: KernelVertexId("entity::river".to_owned()),
                    kind: "entity".to_owned(),
                    class: KernelVertexClass::Entity,
                    value: json!({"name":"River Logistics"}),
                    attributes: json!({}),
                    entity_id: Some("river".to_owned()),
                    temporal: KernelBiTemporal {
                        valid_from: Some(50),
                        valid_to: None,
                        recorded_at: Some(20),
                        expired_at: None,
                    },
                    ..KernelVertex::default()
                }],
                edges: Vec::new(),
            })
            .expect("batch 2");

        let old_view = kernel.view_as_of(KernelViewRequest {
            valid_at: Some(25),
            recorded_at: Some(15),
            include_candidate_graph: false,
        });
        assert_eq!(old_view.vertices.len(), 1);
        assert_eq!(
            old_view.vertices[0]
                .value
                .get("name")
                .and_then(Value::as_str),
            Some("River")
        );

        let corrected_view = kernel.view_as_of(KernelViewRequest {
            valid_at: Some(75),
            recorded_at: Some(25),
            include_candidate_graph: false,
        });
        assert_eq!(
            corrected_view.vertices[0]
                .value
                .get("name")
                .and_then(Value::as_str),
            Some("River Logistics")
        );
    }

    #[test]
    fn entity_resolution_replacement_expires_old_edge_and_updates_sidecar() {
        let mut kernel = PhoenixGraphKernel::new();
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Document {
                    document_id: "doc-1".to_owned(),
                },
                recorded_at: Some(10),
                vertices: vec![
                    KernelVertex {
                        id: KernelVertexId("alias::harbor".to_owned()),
                        kind: "alias".to_owned(),
                        class: KernelVertexClass::Alias,
                        value: json!({"name":"Harbor Authority"}),
                        attributes: json!({}),
                        entity_facet: Some(KernelEntityFacet {
                            canonical_entity_id: None,
                            surface: Some("Harbor Authority".to_owned()),
                            entity_kind: Some("org".to_owned()),
                        }),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("entity::a".to_owned()),
                        kind: "entity".to_owned(),
                        class: KernelVertexClass::Entity,
                        value: json!({"name":"Harbor Authority A"}),
                        attributes: json!({}),
                        entity_id: Some("entity-a".to_owned()),
                        ..KernelVertex::default()
                    },
                ],
                edges: vec![KernelEdge {
                    source_id: KernelVertexId("alias::harbor".to_owned()),
                    target_id: KernelVertexId("entity::a".to_owned()),
                    edge_type: KernelEdgeType("alias_of".to_owned()),
                    relation_class: KernelRelationClass::Identity,
                    weight: 1,
                    attributes: json!({}),
                    temporal: KernelBiTemporal {
                        valid_from: Some(0),
                        valid_to: None,
                        recorded_at: Some(10),
                        expired_at: None,
                    },
                    ..KernelEdge::default()
                }],
            })
            .expect("first resolution");
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Document {
                    document_id: "doc-1".to_owned(),
                },
                recorded_at: Some(20),
                vertices: vec![
                    KernelVertex {
                        id: KernelVertexId("alias::harbor".to_owned()),
                        kind: "alias".to_owned(),
                        class: KernelVertexClass::Alias,
                        value: json!({"name":"Harbor Authority"}),
                        attributes: json!({}),
                        entity_facet: Some(KernelEntityFacet {
                            canonical_entity_id: None,
                            surface: Some("Harbor Authority".to_owned()),
                            entity_kind: Some("org".to_owned()),
                        }),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("entity::b".to_owned()),
                        kind: "entity".to_owned(),
                        class: KernelVertexClass::Entity,
                        value: json!({"name":"Harbor Authority B"}),
                        attributes: json!({}),
                        entity_id: Some("entity-b".to_owned()),
                        ..KernelVertex::default()
                    },
                ],
                edges: vec![KernelEdge {
                    source_id: KernelVertexId("alias::harbor".to_owned()),
                    target_id: KernelVertexId("entity::b".to_owned()),
                    edge_type: KernelEdgeType("alias_of".to_owned()),
                    relation_class: KernelRelationClass::Identity,
                    weight: 1,
                    attributes: json!({}),
                    temporal: KernelBiTemporal {
                        valid_from: Some(0),
                        valid_to: None,
                        recorded_at: Some(20),
                        expired_at: None,
                    },
                    ..KernelEdge::default()
                }],
            })
            .expect("second resolution");

        let current = kernel.entity_candidates(KernelEntityResolveRequest {
            surface: Some("Harbor Authority".to_owned()),
            limit: Some(4),
            ..KernelEntityResolveRequest::default()
        });
        assert_eq!(
            current
                .first()
                .map(|candidate| candidate.entity_id.as_str()),
            Some("entity-b")
        );

        let old = kernel.view_as_of(KernelViewRequest {
            valid_at: Some(1),
            recorded_at: Some(15),
            include_candidate_graph: false,
        });
        assert!(old
            .asserted_edges
            .iter()
            .any(|edge| edge.target_id.0 == "entity::a"));
        assert!(!old
            .asserted_edges
            .iter()
            .any(|edge| edge.target_id.0 == "entity::b"));
    }

    #[test]
    fn calendar_anchor_helper_emits_stable_hierarchy() {
        let (vertices, edges, anchors) = PhoenixGraphKernel::build_calendar_anchor_artifacts(
            1_710_000_000_000,
            Some(1_710_000_000_000),
        );
        assert!(vertices.iter().any(|vertex| vertex.id.0 == anchors.year_id));
        assert!(vertices
            .iter()
            .any(|vertex| vertex.id.0 == anchors.month_id));
        assert!(vertices.iter().any(|vertex| vertex.id.0 == anchors.week_id));
        assert!(vertices.iter().any(|vertex| vertex.id.0 == anchors.day_id));
        assert!(vertices.iter().any(|vertex| vertex.id.0 == anchors.hour_id));
        assert!(edges.iter().any(|edge| {
            edge.source_id.0 == anchors.year_id
                && edge.target_id.0 == anchors.month_id
                && edge.edge_type.0 == "contains"
        }));
    }

    #[test]
    fn world_state_queries_return_active_slot_and_unresolved_issues() {
        let mut kernel = PhoenixGraphKernel::new();
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Projection {
                    scope_key: "scope-1".to_owned(),
                },
                recorded_at: Some(100),
                vertices: vec![
                    KernelVertex {
                        id: KernelVertexId("graph::claim::claim-1".to_owned()),
                        kind: "claim".to_owned(),
                        value: json!({"slotKey":"entity.employer","objectValue":"Acme"}),
                        attributes: json!({}),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("graph::claim::claim-2".to_owned()),
                        kind: "claim".to_owned(),
                        value: json!({"slotKey":"entity.employer","objectValue":"Zenith"}),
                        attributes: json!({}),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("graph::state::old".to_owned()),
                        kind: "state".to_owned(),
                        class: KernelVertexClass::State,
                        value: json!({
                            "slotKey":"entity.employer",
                            "value":"Acme",
                            "status":"superseded",
                            "sourceClass":"world",
                        }),
                        attributes: json!({"confidenceMillis": 400}),
                        temporal: KernelBiTemporal {
                            valid_from: Some(0),
                            valid_to: Some(50),
                            recorded_at: Some(100),
                            expired_at: None,
                        },
                        provenance: KernelProvenance {
                            confidence: Some(0.4),
                            ..KernelProvenance::default()
                        },
                        entity_id: Some("alice".to_owned()),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("graph::state::current".to_owned()),
                        kind: "state".to_owned(),
                        class: KernelVertexClass::State,
                        value: json!({
                            "slotKey":"entity.employer",
                            "value":"Zenith",
                            "status":"active",
                            "sourceClass":"world",
                        }),
                        attributes: json!({"confidenceMillis": 900}),
                        temporal: KernelBiTemporal {
                            valid_from: Some(50),
                            valid_to: None,
                            recorded_at: Some(100),
                            expired_at: None,
                        },
                        provenance: KernelProvenance {
                            confidence: Some(0.9),
                            ..KernelProvenance::default()
                        },
                        entity_id: Some("alice".to_owned()),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("graph::conflict::conflict-1".to_owned()),
                        kind: "conflict".to_owned(),
                        value: json!({
                            "slotKey":"entity.employer",
                            "kind":"mutuallyExclusive",
                            "status":"supported",
                        }),
                        attributes: json!({"preferredClaimId":"claim-2"}),
                        temporal: KernelBiTemporal {
                            valid_from: Some(60),
                            valid_to: None,
                            recorded_at: Some(100),
                            expired_at: None,
                        },
                        entity_id: Some("alice".to_owned()),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("graph::gap::gap-1".to_owned()),
                        kind: "gap".to_owned(),
                        value: json!({
                            "slotKey":"entity.employer",
                            "kind":"unresolvedConflict",
                            "detail":"Need review",
                            "status":"active",
                        }),
                        attributes: json!({}),
                        temporal: KernelBiTemporal {
                            valid_from: Some(70),
                            valid_to: None,
                            recorded_at: Some(100),
                            expired_at: None,
                        },
                        entity_id: Some("alice".to_owned()),
                        ..KernelVertex::default()
                    },
                ],
                edges: vec![
                    KernelEdge {
                        source_id: KernelVertexId("graph::state::old".to_owned()),
                        target_id: KernelVertexId("graph::claim::claim-1".to_owned()),
                        edge_type: KernelEdgeType("supported_by".to_owned()),
                        relation_class: KernelRelationClass::Resolution,
                        weight: 1,
                        attributes: json!({}),
                        ..KernelEdge::default()
                    },
                    KernelEdge {
                        source_id: KernelVertexId("graph::state::current".to_owned()),
                        target_id: KernelVertexId("graph::claim::claim-2".to_owned()),
                        edge_type: KernelEdgeType("supported_by".to_owned()),
                        relation_class: KernelRelationClass::Resolution,
                        weight: 1,
                        attributes: json!({}),
                        ..KernelEdge::default()
                    },
                    KernelEdge {
                        source_id: KernelVertexId("graph::conflict::conflict-1".to_owned()),
                        target_id: KernelVertexId("graph::claim::claim-2".to_owned()),
                        edge_type: KernelEdgeType("supported_by".to_owned()),
                        relation_class: KernelRelationClass::Resolution,
                        weight: 1,
                        attributes: json!({}),
                        ..KernelEdge::default()
                    },
                    KernelEdge {
                        source_id: KernelVertexId("graph::gap::gap-1".to_owned()),
                        target_id: KernelVertexId("graph::claim::claim-2".to_owned()),
                        edge_type: KernelEdgeType("supported_by".to_owned()),
                        relation_class: KernelRelationClass::Resolution,
                        weight: 1,
                        attributes: json!({}),
                        ..KernelEdge::default()
                    },
                ],
            })
            .expect("projection batch");

        let current = kernel.current_slot("alice", "entity.employer", Some(100));
        assert_eq!(
            current
                .active_state
                .as_ref()
                .map(|state| state.value.as_str()),
            Some("Zenith")
        );
        assert_eq!(
            current
                .active_state
                .as_ref()
                .map(|state| state.supporting_claim_ids.clone()),
            Some(vec!["claim-2".to_owned()])
        );
        assert_eq!(current.competing_states.len(), 0);
        assert_eq!(current.conflicts.len(), 1);
        assert_eq!(current.gaps.len(), 1);

        let past = kernel.slot_at(KernelSlotQueryRequest {
            entity_id: "alice".to_owned(),
            slot_key: "entity.employer".to_owned(),
            valid_at: Some(25),
            recorded_at: Some(100),
            include_candidate_graph: false,
        });
        assert_eq!(
            past.active_state.as_ref().map(|state| state.value.as_str()),
            Some("Acme")
        );

        let unresolved = kernel.what_is_unresolved(KernelUnresolvedQueryRequest {
            entity_id: "alice".to_owned(),
            slot_key: Some("entity.employer".to_owned()),
            valid_at: Some(100),
            recorded_at: Some(100),
            include_candidate_graph: false,
        });
        assert_eq!(unresolved.len(), 2);
    }

    #[test]
    fn what_changed_reports_state_activations_and_expirations() {
        let mut kernel = PhoenixGraphKernel::new();
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Projection {
                    scope_key: "scope-2".to_owned(),
                },
                recorded_at: Some(100),
                vertices: vec![
                    KernelVertex {
                        id: KernelVertexId("graph::state::old".to_owned()),
                        kind: "state".to_owned(),
                        class: KernelVertexClass::State,
                        value: json!({
                            "slotKey":"entity.location",
                            "value":"Austin",
                            "status":"superseded",
                            "sourceClass":"world",
                        }),
                        temporal: KernelBiTemporal {
                            valid_from: Some(0),
                            valid_to: Some(50),
                            recorded_at: Some(100),
                            expired_at: None,
                        },
                        entity_id: Some("alice".to_owned()),
                        ..KernelVertex::default()
                    },
                    KernelVertex {
                        id: KernelVertexId("graph::state::new".to_owned()),
                        kind: "state".to_owned(),
                        class: KernelVertexClass::State,
                        value: json!({
                            "slotKey":"entity.location",
                            "value":"Boston",
                            "status":"active",
                            "sourceClass":"world",
                        }),
                        temporal: KernelBiTemporal {
                            valid_from: Some(50),
                            valid_to: None,
                            recorded_at: Some(100),
                            expired_at: None,
                        },
                        entity_id: Some("alice".to_owned()),
                        ..KernelVertex::default()
                    },
                ],
                edges: Vec::new(),
            })
            .expect("projection batch");

        let changes = kernel.what_changed(KernelWhatChangedRequest {
            entity_id: "alice".to_owned(),
            slot_key: Some("entity.location".to_owned()),
            since_valid_at: 40,
            until_valid_at: Some(100),
            recorded_at: Some(100),
            include_candidate_graph: false,
        });

        assert_eq!(changes.len(), 2);
        assert!(changes.iter().any(|change| change.change_kind
            == KernelStateChangeKind::Activated
            && change.state.value == "Boston"));
        assert!(changes.iter().any(
            |change| change.change_kind == KernelStateChangeKind::Expired
                && change.state.value == "Austin"
        ));
    }
}
