use phoenix_graph_kernel::KernelMutationBatch;
use phoenix_types::{
    EntityId, EntityKind, EvidenceSpan, IndexedSpan, IngestDocumentSummary, MentionSpan, NoteId,
    RelationCandidate, ResolverLink, ScopeKey, SentenceSpan, SessionDocumentState, SessionId,
    StructureArtifact, TextRange, TokenSpan,
};
use serde::{Deserialize, Serialize};
use zerocopy::{AsBytes, FromBytes, FromZeroes};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentVersionId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpanId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MentionId(pub String);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScopeOrd(pub u64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocumentOrd(pub u64);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionOrd(pub u64);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkRecord {
    pub chunk_id: ChunkId,
    pub range: TextRange,
    pub chapter_id: u32,
    pub boundary_label: Option<String>,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateEvidence {
    pub kind: String,
    pub detail: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateEntity {
    pub entity_id: String,
    pub source: String,
    pub score_millis: i32,
    #[serde(default)]
    pub evidence: Vec<CandidateEvidence>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionDecision {
    pub status: String,
    pub confidence_millis: u32,
    pub margin_millis: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedMention {
    pub mention_id: MentionId,
    pub mention_index: usize,
    pub range: TextRange,
    pub surface: String,
    pub normalized: String,
    pub kind: Option<EntityKind>,
    pub entity_id: Option<EntityId>,
    pub decision: ResolutionDecision,
    #[serde(default)]
    pub candidates: Vec<CandidateEntity>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AliasConfirmation {
    pub alias_surface: String,
    pub normalized: String,
    pub entity_id: EntityId,
    pub confidence_millis: u32,
    pub mention_id: MentionId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorefClusterRecord {
    pub cluster_id: String,
    pub representative_surface: String,
    pub mention_count: usize,
    pub first_sentence_index: usize,
    pub last_sentence_index: usize,
    #[serde(default)]
    pub chunk_ids: Vec<String>,
    pub named_count: usize,
    pub nominal_count: usize,
    pub pronoun_count: usize,
    #[serde(default)]
    pub resolved_entity_ids: Vec<EntityId>,
    pub confidence_millis: u32,
    pub ambiguous: bool,
    pub route_mix_bits: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeCorefSummary {
    #[serde(default)]
    pub cluster_count: usize,
    #[serde(default)]
    pub attached_mention_count: usize,
    #[serde(default)]
    pub candidate_link_count: usize,
    #[serde(default)]
    pub conflict_cluster_count: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CompactResolutionKind {
    Resolved,
    Ambiguous,
    #[default]
    Unresolved,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompactResolutionRow {
    pub mention_index: usize,
    pub entity_id: Option<EntityId>,
    pub chunk_index: Option<u32>,
    pub kind: CompactResolutionKind,
    pub confidence_millis: u32,
    pub margin_millis: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeExtractionSummary {
    #[serde(default)]
    pub detected_mention_count: usize,
    #[serde(default)]
    pub detected_named_count: usize,
    #[serde(default)]
    pub detected_nominal_count: usize,
    #[serde(default)]
    pub detected_pronoun_count: usize,
    pub resolved_count: usize,
    pub ambiguous_count: usize,
    pub unresolved_count: usize,
    pub alias_confirmation_count: usize,
    #[serde(default)]
    pub verifier_task_count: usize,
    #[serde(default)]
    pub verifier_supported_alias_count: usize,
    #[serde(default)]
    pub verifier_supported_type_count: usize,
}

pub type NativeErSummary = NativeExtractionSummary;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticEntityRecord {
    pub entity_id: EntityId,
    pub canonical_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub kind: Option<EntityKind>,
    pub mention_count: usize,
    #[serde(default)]
    pub chunk_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticRelationRecord {
    pub source_entity_id: EntityId,
    pub target_entity_id: EntityId,
    pub edge_type: String,
    pub sentence_index: usize,
    pub chunk_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AliasPosting {
    pub entity_id: String,
    pub document_id: String,
    pub mention_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AliasEntry {
    pub normalized: String,
    #[serde(default)]
    pub postings: Vec<AliasPosting>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LexicalPostingsSegment {
    #[serde(default)]
    pub spans: Vec<IndexedSpan>,
    #[serde(default)]
    pub alias_entries: Vec<AliasEntry>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DocumentSegmentKind {
    #[default]
    StringArena = 1,
    SentenceTable = 2,
    BoundaryTable = 3,
    ChunkTable = 4,
    MentionTable = 5,
    ResolverLinkTable = 6,
    NarrativeHitTable = 7,
    EntityTable = 8,
    RelationTable = 9,
    EvidenceTable = 10,
    LexicalPostings = 11,
    GraphMutation = 12,
    StructureRelations = 13,
    ResolvedMentionTable = 14,
    AliasConfirmationTable = 15,
    CorefClusterTable = 16,
}

impl DocumentSegmentKind {
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::StringArena),
            2 => Some(Self::SentenceTable),
            3 => Some(Self::BoundaryTable),
            4 => Some(Self::ChunkTable),
            5 => Some(Self::MentionTable),
            6 => Some(Self::ResolverLinkTable),
            7 => Some(Self::NarrativeHitTable),
            8 => Some(Self::EntityTable),
            9 => Some(Self::RelationTable),
            10 => Some(Self::EvidenceTable),
            11 => Some(Self::LexicalPostings),
            12 => Some(Self::GraphMutation),
            13 => Some(Self::StructureRelations),
            14 => Some(Self::ResolvedMentionTable),
            15 => Some(Self::AliasConfirmationTable),
            16 => Some(Self::CorefClusterTable),
            _ => None,
        }
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    AsBytes,
    FromZeroes,
    FromBytes,
)]
#[repr(C)]
pub struct DocumentSegmentHeader {
    pub version: u16,
    pub kind: u8,
    pub flags: u8,
    pub ordinal: u32,
    pub row_count: u32,
    pub uncompressed_len: u32,
    pub payload_len: u32,
}

impl DocumentSegmentHeader {
    pub const VERSION: u16 = 1;

    pub fn new(
        kind: DocumentSegmentKind,
        ordinal: u32,
        row_count: u32,
        uncompressed_len: usize,
        payload_len: usize,
    ) -> Self {
        Self {
            version: Self::VERSION,
            kind: kind.as_u8(),
            flags: 0,
            ordinal,
            row_count,
            uncompressed_len: uncompressed_len as u32,
            payload_len: payload_len as u32,
        }
    }

    pub fn kind(self) -> DocumentSegmentKind {
        DocumentSegmentKind::from_u8(self.kind).unwrap_or_default()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSegmentRef {
    pub kind: DocumentSegmentKind,
    pub ordinal: u32,
    pub row_count: u32,
    pub byte_len: u32,
    pub uncompressed_len: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentManifest {
    pub document_id: String,
    pub document_version_id: DocumentVersionId,
    pub note_id: Option<NoteId>,
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub document_ord: DocumentOrd,
    pub revision: u64,
    pub title: String,
    pub text_len: usize,
    pub fingerprint: String,
    pub config_hash: String,
    pub session_id: Option<SessionId>,
    pub document_summary: IngestDocumentSummary,
    pub session_document: SessionDocumentState,
    pub discovery_count: usize,
    pub mention_count: usize,
    pub span_count: usize,
    pub entity_count: usize,
    pub alias_count: usize,
    pub graph_edge_count: usize,
    pub graph_vertex_count: usize,
    #[serde(default)]
    pub segment_refs: Vec<DocumentSegmentRef>,
    pub created_at: i64,
    pub archive_version: u16,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentArchive {
    pub manifest: DocumentManifest,
    #[serde(default)]
    pub tokens: Vec<TokenSpan>,
    #[serde(default)]
    pub sentences: Vec<SentenceSpan>,
    #[serde(default)]
    pub mentions: Vec<MentionSpan>,
    #[serde(default)]
    pub resolver_links: Vec<ResolverLink>,
    #[serde(default)]
    pub resolved_mentions: Vec<ResolvedMention>,
    #[serde(default)]
    pub alias_confirmations: Vec<AliasConfirmation>,
    #[serde(default)]
    pub coref_clusters: Vec<CorefClusterRecord>,
    #[serde(default)]
    pub er_summary: NativeErSummary,
    #[serde(default)]
    pub coref_summary: NativeCorefSummary,
    #[serde(default)]
    pub chunks: Vec<ChunkRecord>,
    #[serde(default)]
    pub indexed_spans: Vec<IndexedSpan>,
    #[serde(default)]
    pub entities: Vec<SemanticEntityRecord>,
    #[serde(default)]
    pub relations: Vec<SemanticRelationRecord>,
    #[serde(default)]
    pub evidence_spans: Vec<EvidenceSpan>,
    #[serde(default)]
    pub relation_candidates: Vec<RelationCandidate>,
    pub graph_batch: KernelMutationBatch,
    pub structure: Option<StructureArtifact>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentOrdinalAssignment {
    pub document_id: String,
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub document_ord: DocumentOrd,
    pub revision: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedDocumentSegment {
    pub header: DocumentSegmentHeader,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedDocument {
    pub assignment: DocumentOrdinalAssignment,
    pub manifest: DocumentManifest,
    #[serde(default)]
    pub segments: Vec<PreparedDocumentSegment>,
    pub kernel_batch: KernelMutationBatch,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentRevisionRef {
    pub document_id: String,
    pub scope: ScopeKey,
    pub scope_ord: ScopeOrd,
    pub document_ord: DocumentOrd,
    pub revision: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionArchive {
    pub session_id: SessionId,
    pub session_ord: Option<SessionOrd>,
    #[serde(default)]
    pub documents: Vec<SessionDocumentState>,
    #[serde(default)]
    pub document_refs: Vec<DocumentRevisionRef>,
    pub discovery_candidate_count: usize,
    pub span_count: usize,
    pub graph_vertex_count: usize,
    pub graph_edge_count: usize,
    pub graph_generation: u64,
    pub lex_generation: u64,
    pub updated_at: i64,
    pub archive_version: u16,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeLexSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    #[serde(default)]
    pub spans: Vec<IndexedSpan>,
    #[serde(default)]
    pub alias_entries: Vec<AliasEntry>,
    #[serde(default)]
    pub document_ids: Vec<String>,
    pub entity_count: usize,
    pub generated_at: i64,
    pub generation: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DirtyScopeRecord {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    #[serde(default)]
    pub document_ords: Vec<DocumentOrd>,
    pub updated_at: i64,
}

pub fn scope_storage_key(scope: &ScopeKey) -> String {
    let world_id = scope
        .world_id
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or("__global__");
    let narrative_id = scope
        .narrative_id
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or("__global__");
    let folder_id = scope
        .folder_id
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or("__global__");
    let folder_path = scope
        .folder_path
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or("__global__");
    format!("{world_id}::{narrative_id}::{folder_id}::{folder_path}")
}
