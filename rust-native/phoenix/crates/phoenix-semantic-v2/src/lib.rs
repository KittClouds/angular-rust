use phoenix_graph_kernel::KernelMutationBatch;
use phoenix_types::{
    BiTemporalWindow, CausalCandidate, CausalDiagnostic, CausalKind, CausalLink, ClaimRecord,
    EntityId, EntityKind, EventRecord, EvidenceSpan, IndexedSpan, IngestDocumentSummary,
    MentionSpan, NoteId, Polarity, Proposition, RelationCandidate, ResolverLink, ScopeKey,
    SemanticNodeRef, SemanticRelation, SentenceSpan, SessionDocumentState, SessionId,
    StateRecord, StructureArtifact, TextRange, TimeAnchorRecord, TokenSpan,
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
pub struct RecordedTemporalBinding {
    pub anchor: Option<TimeAnchorRecord>,
    pub recorded_window: BiTemporalWindow,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentCausalSubstrate {
    #[serde(default)]
    pub propositions: Vec<Proposition>,
    #[serde(default)]
    pub semantic_events: Vec<EventRecord>,
    #[serde(default)]
    pub semantic_states: Vec<StateRecord>,
    #[serde(default)]
    pub semantic_claims: Vec<ClaimRecord>,
    #[serde(default)]
    pub semantic_relations: Vec<SemanticRelation>,
    #[serde(default)]
    pub temporal_bindings: Vec<RecordedTemporalBinding>,
    #[serde(default)]
    pub causal_candidates: Vec<CausalCandidate>,
    #[serde(default)]
    pub causal_links: Vec<CausalLink>,
    #[serde(default)]
    pub causal_diagnostics: Vec<CausalDiagnostic>,
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
    CausalSubstrateTable = 17,
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
            17 => Some(Self::CausalSubstrateTable),
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
    #[serde(default)]
    pub causal_substrate: Option<DocumentCausalSubstrate>,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErDecisionOutcome {
    Link,
    ConfirmAlias,
    PatchType,
    Defer,
    #[default]
    Reject,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErAliasAddition {
    pub case_id: String,
    pub document_id: String,
    pub mention_id: Option<MentionId>,
    pub entity_id: EntityId,
    pub alias_surface: String,
    pub normalized: String,
    pub confidence_millis: u32,
    pub created_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErTypeOverride {
    pub case_id: String,
    pub document_id: String,
    pub mention_id: Option<MentionId>,
    pub entity_id: EntityId,
    pub kind: EntityKind,
    pub confidence_millis: u32,
    pub created_at: i64,
}

impl Default for ErTypeOverride {
    fn default() -> Self {
        Self {
            case_id: String::new(),
            document_id: String::new(),
            mention_id: None,
            entity_id: EntityId::default(),
            kind: EntityKind::Other,
            confidence_millis: 0,
            created_at: 0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErEntityLinkOverride {
    pub case_id: String,
    pub document_id: String,
    pub mention_id: Option<MentionId>,
    pub entity_id: EntityId,
    pub confidence_millis: u32,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErDecisionRecord {
    pub case_id: String,
    pub document_id: String,
    pub mention_id: Option<MentionId>,
    pub outcome: ErDecisionOutcome,
    pub entity_id: Option<EntityId>,
    pub patched_kind: Option<EntityKind>,
    pub score_millis: i32,
    pub rationale: String,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub surface: String,
    pub normalized_surface: String,
    pub reviewed_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErScopePatchSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    #[serde(default)]
    pub alias_additions: Vec<ErAliasAddition>,
    #[serde(default)]
    pub type_overrides: Vec<ErTypeOverride>,
    #[serde(default)]
    pub entity_links: Vec<ErEntityLinkOverride>,
    #[serde(default)]
    pub decisions: Vec<ErDecisionRecord>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RelationDecisionOutcome {
    Accept,
    Support,
    Contradict,
    Defer,
    #[default]
    Reject,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationEdgeAddition {
    pub case_id: String,
    pub document_id: String,
    pub window_id: String,
    pub source_entity_id: EntityId,
    pub target_entity_id: EntityId,
    pub edge_type: String,
    pub confidence_millis: u32,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RelationJudgmentKind {
    #[default]
    Support,
    Contradict,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationJudgmentRecord {
    pub case_id: String,
    pub document_id: String,
    pub window_id: String,
    pub source_entity_id: EntityId,
    pub target_entity_id: EntityId,
    pub edge_type: String,
    pub kind: RelationJudgmentKind,
    pub confidence_millis: u32,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationDecisionRecord {
    pub case_id: String,
    pub document_id: String,
    pub window_id: String,
    pub outcome: RelationDecisionOutcome,
    pub source_entity_id: Option<EntityId>,
    pub target_entity_id: Option<EntityId>,
    pub edge_type: Option<String>,
    pub score_millis: i32,
    pub rationale: String,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub reviewed_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationScopePatchSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    #[serde(default)]
    pub edge_additions: Vec<RelationEdgeAddition>,
    #[serde(default)]
    pub support_judgments: Vec<RelationJudgmentRecord>,
    #[serde(default)]
    pub contradiction_judgments: Vec<RelationJudgmentRecord>,
    #[serde(default)]
    pub decisions: Vec<RelationDecisionRecord>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationMentionSeedRecord {
    pub document_id: String,
    pub revision: u64,
    pub chunk_id: String,
    pub entity_id: EntityId,
    pub surface: String,
    pub normalized: String,
    pub kind: Option<EntityKind>,
    pub range: TextRange,
    pub sentence_index: Option<usize>,
    pub confidence_millis: u32,
    pub seed_label: String,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationMentionSeedScopeSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    #[serde(default)]
    pub seeds: Vec<RelationMentionSeedRecord>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalDecisionOutcome {
    Accept,
    Support,
    Invalidate,
    Defer,
    #[default]
    Reject,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalClaimStatus {
    #[default]
    Candidate,
    Active,
    Supported,
    Contradicted,
    Superseded,
    Invalidated,
    Deferred,
    Rejected,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CausalEdgeId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CausalDecisionId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CausalClaimId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CausalChainId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct CausalReviewId(pub String);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalRelationKind {
    DirectCause,
    ContributingCause,
    EnablingCondition,
    PreventingFactor,
    Trigger,
    MediatedCause,
    #[default]
    HypothesizedCause,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalClaimPolarity {
    #[default]
    Support,
    Contradict,
    Underspecify,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalClaimSourceKind {
    ExplicitLink,
    ExplicitCue,
    CandidateCue,
    LocalTemporalPair,
    GraphSupport,
    ReverseConflict,
    QuoteAttribution,
    CounterfactualCompetition,
    #[default]
    ChainBridge,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CounterfactualReason {
    #[default]
    CompetingCause,
    BlockedByEvent,
    MissingIntermediate,
    BrittleSupportPath,
    DirectionDispute,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalClaimAtom {
    pub claim_id: CausalClaimId,
    pub edge_id: CausalEdgeId,
    pub document_id: String,
    pub cause_event: SemanticNodeRef,
    pub effect_event: SemanticNodeRef,
    pub kind: CausalKind,
    pub relation_kind: CausalRelationKind,
    pub source_kind: CausalClaimSourceKind,
    pub polarity: CausalClaimPolarity,
    pub strength_millis: u32,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalEdgeAddition {
    pub edge_id: CausalEdgeId,
    pub case_id: String,
    pub document_id: String,
    pub source: SemanticNodeRef,
    pub target: SemanticNodeRef,
    pub kind: CausalKind,
    pub relation_kind: CausalRelationKind,
    pub status: CausalClaimStatus,
    pub first_seen_revision: u64,
    pub latest_decision_id: Option<CausalDecisionId>,
    pub confidence_millis: u32,
    pub cue: Option<String>,
    pub attributed_to: Option<EntityId>,
    pub polarity: Polarity,
    #[serde(default)]
    pub claim_atom_ids: Vec<CausalClaimId>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub effective_interval: BiTemporalWindow,
    pub observation_interval: BiTemporalWindow,
    pub temporal_certainty_millis: u32,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalChainRecord {
    pub chain_id: CausalChainId,
    pub document_id: String,
    pub kind: CausalKind,
    pub relation_kind: CausalRelationKind,
    #[serde(default)]
    pub nodes: Vec<SemanticNodeRef>,
    #[serde(default)]
    pub edge_ids: Vec<CausalEdgeId>,
    pub weakest_status: CausalClaimStatus,
    pub confidence_millis: u32,
    pub temporal: BiTemporalWindow,
    pub temporal_consistency_millis: u32,
    pub explanatory_strength_millis: u32,
    pub speculative: bool,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CounterfactualReviewRecord {
    pub review_id: CausalReviewId,
    pub case_id: String,
    pub focal_edge_id: CausalEdgeId,
    pub document_id: String,
    pub source: SemanticNodeRef,
    pub target: SemanticNodeRef,
    pub kind: CausalKind,
    pub relation_kind: CausalRelationKind,
    pub confidence_millis: u32,
    pub review_reason: CounterfactualReason,
    #[serde(default)]
    pub competing_cause_ids: Vec<CausalEdgeId>,
    #[serde(default)]
    pub blocker_events: Vec<SemanticNodeRef>,
    #[serde(default)]
    pub missing_intermediate_events: Vec<SemanticNodeRef>,
    pub only_support_path: Option<CausalChainId>,
    pub rationale: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalDecisionRecord {
    pub decision_id: CausalDecisionId,
    pub edge_id: CausalEdgeId,
    pub case_id: String,
    pub document_id: String,
    pub outcome: CausalDecisionOutcome,
    pub source: Option<SemanticNodeRef>,
    pub target: Option<SemanticNodeRef>,
    pub kind: Option<CausalKind>,
    pub relation_kind: Option<CausalRelationKind>,
    pub score_millis: i32,
    pub rationale: String,
    pub supersedes: Option<CausalDecisionId>,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub reviewed_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalInvalidationRecord {
    pub invalidation_id: String,
    pub edge_id: CausalEdgeId,
    pub decision_id: CausalDecisionId,
    pub document_id: String,
    pub rationale: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalEdgeAliasRecord {
    pub alias_key: String,
    pub edge_id: CausalEdgeId,
    pub document_id: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalReviewQueueItem {
    pub queue_id: String,
    pub edge_id: CausalEdgeId,
    pub latest_decision_id: Option<CausalDecisionId>,
    pub document_id: String,
    pub priority_millis: u32,
    pub rationale: String,
    pub unresolved: bool,
    pub created_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalMemoryCard {
    pub node: SemanticNodeRef,
    pub document_id: String,
    pub label: String,
    pub sentence_index: usize,
    #[serde(default)]
    pub incoming_edge_ids: Vec<CausalEdgeId>,
    #[serde(default)]
    pub outgoing_edge_ids: Vec<CausalEdgeId>,
    #[serde(default)]
    pub chain_ids: Vec<CausalChainId>,
    #[serde(default)]
    pub counterfactual_review_ids: Vec<CausalReviewId>,
    pub why_this_event_matters: Option<String>,
    pub strongest_upstream_cause: Option<CausalEdgeId>,
    pub most_fragile_downstream_effect: Option<CausalEdgeId>,
    #[serde(default)]
    pub open_disputes: Vec<CausalReviewId>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalMetricsSnapshot {
    #[serde(default)]
    pub edge_record_count: usize,
    #[serde(default)]
    pub accepted_count: usize,
    #[serde(default)]
    pub supported_count: usize,
    #[serde(default)]
    pub deferred_count: usize,
    #[serde(default)]
    pub rejected_count: usize,
    #[serde(default)]
    pub invalidated_count: usize,
    #[serde(default)]
    pub contradicted_count: usize,
    #[serde(default)]
    pub contradiction_rate_per_1k_events_millis: u32,
    #[serde(default)]
    pub edge_survival_rate_millis: u32,
    #[serde(default)]
    pub chain_collapse_rate_millis: u32,
    #[serde(default)]
    pub avg_claim_atoms_per_edge_millis: u32,
    #[serde(default)]
    pub cue_only_edge_rate_millis: u32,
    #[serde(default)]
    pub card_open_dispute_rate_millis: u32,
    #[serde(default)]
    pub temporal_illegality_rejection_rate_millis: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalCompilerSummary {
    #[serde(default)]
    pub claim_atom_count: usize,
    #[serde(default)]
    pub review_case_count: usize,
    #[serde(default)]
    pub edge_record_count: usize,
    #[serde(default)]
    pub committed_edge_count: usize,
    #[serde(default)]
    pub accepted_edge_count: usize,
    #[serde(default)]
    pub supported_edge_count: usize,
    #[serde(default)]
    pub deferred_edge_count: usize,
    #[serde(default)]
    pub rejected_edge_count: usize,
    #[serde(default)]
    pub contradicted_edge_count: usize,
    #[serde(default)]
    pub chain_count: usize,
    #[serde(default)]
    pub counterfactual_review_count: usize,
    #[serde(default)]
    pub memory_card_count: usize,
    #[serde(default)]
    pub invalidation_count: usize,
    #[serde(default)]
    pub review_queue_count: usize,
    #[serde(default)]
    pub kind_counts: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    pub outcome_counts: std::collections::BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalScopeSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    #[serde(default)]
    pub claim_atoms: Vec<CausalClaimAtom>,
    #[serde(default)]
    pub edge_records: Vec<CausalEdgeAddition>,
    #[serde(default)]
    pub edge_additions: Vec<CausalEdgeAddition>,
    #[serde(default)]
    pub chains: Vec<CausalChainRecord>,
    #[serde(default)]
    pub counterfactual_reviews: Vec<CounterfactualReviewRecord>,
    #[serde(default)]
    pub decisions: Vec<CausalDecisionRecord>,
    #[serde(default)]
    pub decision_history: Vec<CausalDecisionRecord>,
    #[serde(default)]
    pub invalidations: Vec<CausalInvalidationRecord>,
    #[serde(default)]
    pub edge_aliases: Vec<CausalEdgeAliasRecord>,
    #[serde(default)]
    pub review_queue: Vec<CausalReviewQueueItem>,
    #[serde(default)]
    pub memory_cards: Vec<CausalMemoryCard>,
    pub metrics_snapshot: CausalMetricsSnapshot,
    pub summary: CausalCompilerSummary,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryClaimStatus {
    #[default]
    Candidate,
    Active,
    Supported,
    Contradicted,
    Superseded,
    Deferred,
    Rejected,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryModality {
    #[default]
    Asserted,
    Reported,
    Observed,
    Inferred,
    Planned,
    Conditional,
    Hypothetical,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryConflictKind {
    #[default]
    MutuallyExclusive,
    TemporalOverlap,
    SupportVsContradiction,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MemoryGapKind {
    #[default]
    MissingCurrentValue,
    UnresolvedConflict,
    MissingSuccessor,
    BrokenContinuity,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryClaimAtom {
    pub claim_id: String,
    pub document_id: String,
    pub source_entity_id: Option<EntityId>,
    pub target_entity_id: Option<EntityId>,
    pub slot_key: String,
    pub relation_family: Option<String>,
    pub subject_label: String,
    pub object_label: String,
    pub object_entity_id: Option<EntityId>,
    pub object_value: String,
    pub status: MemoryClaimStatus,
    pub modality: MemoryModality,
    pub confidence_millis: u32,
    pub source_class: String,
    pub provenance_label: String,
    pub window_id: Option<String>,
    pub source_case_id: Option<String>,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryEventRecord {
    pub event_id: String,
    pub document_id: String,
    pub kind: String,
    pub slot_key: String,
    pub subject_entity_id: Option<EntityId>,
    pub object_entity_id: Option<EntityId>,
    pub old_value: Option<String>,
    pub new_value: Option<String>,
    pub conflict_id: Option<String>,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub claim_ids: Vec<String>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStateRecord {
    pub state_id: String,
    pub entity_id: EntityId,
    pub slot_key: String,
    pub value: String,
    pub value_entity_id: Option<EntityId>,
    pub status: MemoryClaimStatus,
    pub source_class: String,
    pub confidence_millis: u32,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryDeltaRecord {
    pub delta_id: String,
    pub entity_id: EntityId,
    pub slot_key: String,
    pub old_value: Option<String>,
    pub old_value_entity_id: Option<EntityId>,
    pub new_value: Option<String>,
    pub new_value_entity_id: Option<EntityId>,
    pub caused_by_event_id: Option<String>,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryConflictRecord {
    pub conflict_id: String,
    pub entity_id: EntityId,
    pub slot_key: String,
    pub kind: MemoryConflictKind,
    pub preferred_claim_id: Option<String>,
    pub status: MemoryClaimStatus,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryContinuityGapRecord {
    pub gap_id: String,
    pub entity_id: EntityId,
    pub slot_key: String,
    pub kind: MemoryGapKind,
    pub status: MemoryClaimStatus,
    pub detail: String,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityMemoryIdentityCard {
    pub entity_id: EntityId,
    pub canonical_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub effective_kind: Option<EntityKind>,
    pub linked_mention_count: usize,
    #[serde(default)]
    pub continuity_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityMemoryStateView {
    pub slot_key: String,
    pub value: String,
    pub value_entity_id: Option<EntityId>,
    pub confidence_millis: u32,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipMemoryRef {
    pub relation_family: String,
    pub target_entity_id: EntityId,
    pub status: MemoryClaimStatus,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub supporting_claim_ids: Vec<String>,
    #[serde(default)]
    pub contradicting_claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityMemoryCard {
    pub entity_id: EntityId,
    pub identity: EntityMemoryIdentityCard,
    #[serde(default)]
    pub current_state: Vec<EntityMemoryStateView>,
    #[serde(default)]
    pub recent_deltas: Vec<MemoryDeltaRecord>,
    #[serde(default)]
    pub active_relationships: Vec<RelationshipMemoryRef>,
    #[serde(default)]
    pub active_conflicts: Vec<MemoryConflictRecord>,
    #[serde(default)]
    pub open_gaps: Vec<MemoryContinuityGapRecord>,
    #[serde(default)]
    pub top_evidence_claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationshipMemoryLedger {
    pub ledger_id: String,
    pub relation_family: String,
    pub source_entity_id: EntityId,
    pub target_entity_id: EntityId,
    pub current_status: MemoryClaimStatus,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub supporting_claim_ids: Vec<String>,
    #[serde(default)]
    pub contradicting_claim_ids: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryCompilerSummary {
    pub claim_count: usize,
    pub event_count: usize,
    pub state_count: usize,
    pub delta_count: usize,
    pub conflict_count: usize,
    pub gap_count: usize,
    pub entity_card_count: usize,
    pub relationship_ledger_count: usize,
    #[serde(default)]
    pub active_slot_counts: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    pub unresolved_gap_counts: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    pub source_class_counts: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    pub status_counts: std::collections::BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryScopeSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    #[serde(default)]
    pub claims: Vec<MemoryClaimAtom>,
    #[serde(default)]
    pub events: Vec<MemoryEventRecord>,
    #[serde(default)]
    pub states: Vec<MemoryStateRecord>,
    #[serde(default)]
    pub deltas: Vec<MemoryDeltaRecord>,
    #[serde(default)]
    pub conflicts: Vec<MemoryConflictRecord>,
    #[serde(default)]
    pub gaps: Vec<MemoryContinuityGapRecord>,
    #[serde(default)]
    pub entity_cards: Vec<EntityMemoryCard>,
    #[serde(default)]
    pub relationship_ledgers: Vec<RelationshipMemoryLedger>,
    pub summary: MemoryCompilerSummary,
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
