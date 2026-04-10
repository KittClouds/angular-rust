use phoenix_graph_kernel::KernelMutationBatch;
use phoenix_types::{
    BiTemporalWindow, CausalCandidate, CausalDiagnostic, CausalKind, CausalLink, ClaimRecord,
    EntityId, EntityKind, EventRecord, EvidenceSpan, IndexedSpan, IngestDocumentSummary,
    MentionSpan, NoteId, Polarity, Proposition, RelationCandidate, ResolverLink, ScopeKey,
    SemanticNodeRef, SemanticRelation, SentenceSpan, SessionDocumentState, SessionId, StateRecord,
    StructureArtifact, TextRange, TimeAnchorRecord as NativeTimeAnchorRecord, TokenSpan,
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventMentionId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalEventId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityHypothesisId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityDecisionId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityMembershipId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentitySplitId(pub String);

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
    pub anchor: Option<NativeTimeAnchorRecord>,
    pub recorded_window: BiTemporalWindow,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TemporalTimexId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TemporalAnchorId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TemporalAxisId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TemporalConstraintId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TimelineSegmentId(pub String);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TemporalAxisKind {
    #[default]
    World,
    Reported,
    Conditional,
    Hypothetical,
    Planned,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TemporalConstraintKind {
    #[default]
    AnchoredAt,
    StartBeforeStart,
    EndBeforeStart,
    NotLaterThan,
    ReferenceEvent,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TimelineSegmentKind {
    #[default]
    Main,
    Branch,
    Subordinate,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TemporalGapKind {
    #[default]
    MissingAnchor,
    UnresolvedOrder,
    UnderspecifiedInterval,
    ConflictingAnchors,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TemporalConflictKind {
    #[default]
    IncompatibleConstraints,
    ImpossibleOrdering,
    IncompatibleAnchors,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceTemporalCueRecord {
    pub cue_id: String,
    pub proposition_id: Option<String>,
    pub sentence_index: usize,
    pub cue_kind: String,
    pub label: String,
    pub range: Option<TextRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalAxisRecord {
    pub axis_id: TemporalAxisId,
    pub document_id: String,
    pub kind: TemporalAxisKind,
    pub label: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalTimexRecord {
    pub timex_id: TemporalTimexId,
    pub document_id: String,
    pub proposition_id: Option<String>,
    pub sentence_index: usize,
    pub label: String,
    pub normalized_value: Option<String>,
    pub range: Option<TextRange>,
    pub axis_id: TemporalAxisId,
    pub temporal: BiTemporalWindow,
    pub confidence_millis: u32,
    pub source_class: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalAnchorRecord {
    pub anchor_id: TemporalAnchorId,
    pub document_id: String,
    pub proposition_id: Option<String>,
    pub event_id: Option<String>,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub timex_id: Option<TemporalTimexId>,
    pub reference_event_id: Option<String>,
    pub canonical_reference_event_id: Option<CanonicalEventId>,
    pub axis_id: TemporalAxisId,
    pub label: String,
    pub anchor_kind: String,
    pub temporal: BiTemporalWindow,
    pub confidence_millis: u32,
    pub source_class: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalReferenceEdge {
    pub edge_id: String,
    pub document_id: String,
    pub axis_id: TemporalAxisId,
    pub source_event_id: String,
    pub canonical_source_event_id: Option<CanonicalEventId>,
    pub target_event_id: Option<String>,
    pub canonical_target_event_id: Option<CanonicalEventId>,
    pub target_timex_id: Option<TemporalTimexId>,
    pub relation: String,
    pub confidence_millis: u32,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalClaimAtom {
    pub claim_id: String,
    pub document_id: String,
    pub proposition_id: Option<String>,
    pub event_id: Option<String>,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub axis_id: TemporalAxisId,
    pub source_kind: String,
    pub label: String,
    pub confidence_millis: u32,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalConstraintRecord {
    pub constraint_id: TemporalConstraintId,
    pub document_id: String,
    pub axis_id: TemporalAxisId,
    pub source_event_id: Option<String>,
    pub canonical_source_event_id: Option<CanonicalEventId>,
    pub target_event_id: Option<String>,
    pub canonical_target_event_id: Option<CanonicalEventId>,
    pub target_timex_id: Option<TemporalTimexId>,
    pub kind: TemporalConstraintKind,
    pub confidence_millis: u32,
    pub hard: bool,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalIntervalRecord {
    pub interval_id: String,
    pub document_id: String,
    pub event_id: String,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub axis_id: TemporalAxisId,
    pub anchor_id: Option<TemporalAnchorId>,
    pub temporal: BiTemporalWindow,
    pub confidence_millis: u32,
    pub source_class: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineSegmentRecord {
    pub segment_id: TimelineSegmentId,
    pub document_id: String,
    pub axis_id: TemporalAxisId,
    pub segment_kind: TimelineSegmentKind,
    #[serde(default)]
    pub event_ids: Vec<String>,
    #[serde(default)]
    pub canonical_event_ids: Vec<CanonicalEventId>,
    pub anchor_coverage_millis: u32,
    #[serde(default)]
    pub indeterminate_event_ids: Vec<String>,
    #[serde(default)]
    pub indeterminate_canonical_event_ids: Vec<CanonicalEventId>,
    pub temporal: BiTemporalWindow,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalConflictRecord {
    pub conflict_id: String,
    pub document_id: String,
    pub axis_id: TemporalAxisId,
    pub kind: TemporalConflictKind,
    pub event_id: Option<String>,
    pub canonical_event_id: Option<CanonicalEventId>,
    #[serde(default)]
    pub constraint_ids: Vec<TemporalConstraintId>,
    pub reason: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalGapRecord {
    pub gap_id: String,
    pub document_id: String,
    pub axis_id: TemporalAxisId,
    pub event_id: Option<String>,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub kind: TemporalGapKind,
    pub reason: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalMemoryCard {
    pub card_id: String,
    pub document_id: String,
    pub event_id: String,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub label: String,
    pub sentence_index: usize,
    pub axis_kind: TemporalAxisKind,
    pub strongest_interval: Option<BiTemporalWindow>,
    pub anchor_source: Option<String>,
    #[serde(default)]
    pub before_event_ids: Vec<String>,
    #[serde(default)]
    pub before_canonical_event_ids: Vec<CanonicalEventId>,
    #[serde(default)]
    pub after_event_ids: Vec<String>,
    #[serde(default)]
    pub after_canonical_event_ids: Vec<CanonicalEventId>,
    #[serde(default)]
    pub open_conflict_ids: Vec<String>,
    #[serde(default)]
    pub open_gap_ids: Vec<String>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalDiagnosticRecord {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentTemporalSubstrate {
    #[serde(default)]
    pub propositions: Vec<Proposition>,
    #[serde(default)]
    pub semantic_events: Vec<EventRecord>,
    #[serde(default)]
    pub semantic_states: Vec<StateRecord>,
    #[serde(default)]
    pub semantic_claims: Vec<ClaimRecord>,
    #[serde(default)]
    pub surface_temporal_cues: Vec<SurfaceTemporalCueRecord>,
    #[serde(default)]
    pub timex_records: Vec<TemporalTimexRecord>,
    #[serde(default)]
    pub anchor_candidates: Vec<TemporalAnchorRecord>,
    #[serde(default)]
    pub axis_records: Vec<TemporalAxisRecord>,
    #[serde(default)]
    pub reference_timex_edges: Vec<TemporalReferenceEdge>,
    #[serde(default)]
    pub reference_event_edges: Vec<TemporalReferenceEdge>,
    #[serde(default)]
    pub temporal_claims: Vec<TemporalClaimAtom>,
    #[serde(default)]
    pub temporal_constraints: Vec<TemporalConstraintRecord>,
    #[serde(default)]
    pub temporal_diagnostics: Vec<TemporalDiagnosticRecord>,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EventIdentityState {
    FullIdentity,
    #[default]
    QuasiIdentity,
    MemberOfCollection,
    SubeventOf,
    VersionOf,
    ReportsOn,
    Incompatible,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EventIdentityDecisionKind {
    #[default]
    Merge,
    Link,
    Split,
    Promote,
    Demote,
    Invalidate,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EventSourceSemantics {
    #[default]
    WorldAssertion,
    ReportedSpeech,
    AttributedClaim,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EventModalitySemantics {
    #[default]
    Asserted,
    Conditional,
    Planned,
    Hypothetical,
    Negated,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventParticipantSlot {
    pub role: String,
    pub entity_id: Option<EntityId>,
    pub mention_index: Option<usize>,
    pub label: Option<String>,
    pub range: Option<TextRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventMentionPacketSeed {
    pub mention_id: EventMentionId,
    pub event_id: String,
    pub document_id: String,
    pub proposition_id: String,
    pub revision: u64,
    pub label: String,
    pub normalized_predicate: String,
    pub event_type: String,
    #[serde(default)]
    pub participant_slots: Vec<EventParticipantSlot>,
    #[serde(default)]
    pub place_labels: Vec<String>,
    #[serde(default)]
    pub explicit_timex_ids: Vec<TemporalTimexId>,
    #[serde(default)]
    pub time_anchor_ids: Vec<TemporalAnchorId>,
    #[serde(default)]
    pub causal_neighbor_event_ids: Vec<String>,
    #[serde(default)]
    pub temporal_neighbor_event_ids: Vec<String>,
    pub sentence_index: usize,
    pub clause_range: Option<TextRange>,
    pub polarity_negative: bool,
    pub source_semantics: EventSourceSemantics,
    pub modality_semantics: EventModalitySemantics,
    pub realis: String,
    pub event_fingerprint: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityDiagnosticRecord {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentEventIdentitySubstrate {
    #[serde(default)]
    pub mention_seeds: Vec<EventMentionPacketSeed>,
    #[serde(default)]
    pub diagnostics: Vec<EventIdentityDiagnosticRecord>,
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
    TemporalSubstrateTable = 18,
    EventIdentitySubstrateTable = 19,
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
            18 => Some(Self::TemporalSubstrateTable),
            19 => Some(Self::EventIdentitySubstrateTable),
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
    #[serde(default)]
    pub temporal_substrate: Option<DocumentTemporalSubstrate>,
    #[serde(default)]
    pub event_identity_substrate: Option<DocumentEventIdentitySubstrate>,
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
pub enum CausalEvidenceClass {
    #[default]
    WorldSupport,
    ReportedSupport,
    AttributedSupport,
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
    pub canonical_cause_event_id: Option<CanonicalEventId>,
    pub effect_event: SemanticNodeRef,
    pub canonical_effect_event_id: Option<CanonicalEventId>,
    pub kind: CausalKind,
    pub relation_kind: CausalRelationKind,
    pub source_kind: CausalClaimSourceKind,
    pub polarity: CausalClaimPolarity,
    #[serde(default)]
    pub evidence_class: CausalEvidenceClass,
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
    pub canonical_cause_event_id: Option<CanonicalEventId>,
    pub target: SemanticNodeRef,
    pub canonical_effect_event_id: Option<CanonicalEventId>,
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
    pub canonical_event_ids: Vec<CanonicalEventId>,
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
    pub canonical_cause_event_id: Option<CanonicalEventId>,
    pub target: SemanticNodeRef,
    pub canonical_effect_event_id: Option<CanonicalEventId>,
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
    pub canonical_event_id: Option<CanonicalEventId>,
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
    #[serde(default)]
    pub pass_a_accept_count: usize,
    #[serde(default)]
    pub pass_a_defer_count: usize,
    #[serde(default)]
    pub pass_a_reject_count: usize,
    #[serde(default)]
    pub pass_b_demoted_count: usize,
    #[serde(default)]
    pub world_support_count: usize,
    #[serde(default)]
    pub reported_support_count: usize,
    #[serde(default)]
    pub attributed_support_count: usize,
    #[serde(default)]
    pub shadow_local_pair_candidate_count: usize,
    #[serde(default)]
    pub shadow_local_pair_committed_count: usize,
    #[serde(default)]
    pub shadow_local_pair_deferred_count: usize,
    #[serde(default)]
    pub shadow_local_pair_overlap_count: usize,
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalCompilerSummary {
    #[serde(default)]
    pub timex_count: usize,
    #[serde(default)]
    pub anchor_count: usize,
    #[serde(default)]
    pub claim_count: usize,
    #[serde(default)]
    pub constraint_count: usize,
    #[serde(default)]
    pub review_case_count: usize,
    #[serde(default)]
    pub interval_count: usize,
    #[serde(default)]
    pub segment_count: usize,
    #[serde(default)]
    pub conflict_count: usize,
    #[serde(default)]
    pub gap_count: usize,
    #[serde(default)]
    pub memory_card_count: usize,
    #[serde(default)]
    pub axis_counts: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    pub source_class_counts: std::collections::BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalScopeSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    #[serde(default)]
    pub timex_records: Vec<TemporalTimexRecord>,
    #[serde(default)]
    pub anchors: Vec<TemporalAnchorRecord>,
    #[serde(default)]
    pub axes: Vec<TemporalAxisRecord>,
    #[serde(default)]
    pub reference_edges: Vec<TemporalReferenceEdge>,
    #[serde(default)]
    pub claim_atoms: Vec<TemporalClaimAtom>,
    #[serde(default)]
    pub constraints: Vec<TemporalConstraintRecord>,
    #[serde(default)]
    pub intervals: Vec<TemporalIntervalRecord>,
    #[serde(default)]
    pub timeline_segments: Vec<TimelineSegmentRecord>,
    #[serde(default)]
    pub conflicts: Vec<TemporalConflictRecord>,
    #[serde(default)]
    pub gaps: Vec<TemporalGapRecord>,
    #[serde(default)]
    pub memory_cards: Vec<TemporalMemoryCard>,
    pub summary: TemporalCompilerSummary,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventMentionPacket {
    pub mention_id: EventMentionId,
    pub event_id: String,
    pub document_id: String,
    pub proposition_id: String,
    pub revision: u64,
    pub label: String,
    pub normalized_predicate: String,
    pub event_type: String,
    #[serde(default)]
    pub participant_slots: Vec<EventParticipantSlot>,
    #[serde(default)]
    pub place_labels: Vec<String>,
    #[serde(default)]
    pub explicit_timex_ids: Vec<TemporalTimexId>,
    #[serde(default)]
    pub time_anchor_ids: Vec<TemporalAnchorId>,
    #[serde(default)]
    pub causal_neighbor_event_ids: Vec<String>,
    #[serde(default)]
    pub temporal_neighbor_event_ids: Vec<String>,
    pub sentence_index: usize,
    pub clause_range: Option<TextRange>,
    pub polarity_negative: bool,
    pub source_semantics: EventSourceSemantics,
    pub modality_semantics: EventModalitySemantics,
    pub realis: String,
    pub event_fingerprint: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityHypothesis {
    pub hypothesis_id: EventIdentityHypothesisId,
    pub left_mention_id: EventMentionId,
    pub right_mention_id: EventMentionId,
    pub relation: EventIdentityState,
    pub score_millis: i32,
    pub argument_role_score_millis: u32,
    pub time_score_millis: u32,
    pub place_score_millis: u32,
    pub neighborhood_score_millis: u32,
    pub discourse_score_millis: u32,
    pub lexical_score_millis: u32,
    pub blocked: bool,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalEventRecord {
    pub canonical_event_id: CanonicalEventId,
    pub scope_key: String,
    pub canonical_label: String,
    pub normalized_predicate: String,
    pub event_type: String,
    pub source_semantics: EventSourceSemantics,
    pub modality_semantics: EventModalitySemantics,
    pub realis: String,
    #[serde(default)]
    pub mention_ids: Vec<EventMentionId>,
    #[serde(default)]
    pub document_ids: Vec<String>,
    #[serde(default)]
    pub participant_slots: Vec<EventParticipantSlot>,
    #[serde(default)]
    pub place_labels: Vec<String>,
    #[serde(default)]
    pub time_anchor_ids: Vec<TemporalAnchorId>,
    pub first_seen_revision: u64,
    pub latest_seen_revision: u64,
    pub confidence_millis: u32,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityMembershipRecord {
    pub membership_id: EventIdentityMembershipId,
    pub canonical_event_id: CanonicalEventId,
    pub mention_id: EventMentionId,
    pub relation: EventIdentityState,
    pub confidence_millis: u32,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityLedgerRecord {
    pub decision_id: EventIdentityDecisionId,
    pub hypothesis_id: Option<EventIdentityHypothesisId>,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub left_mention_id: Option<EventMentionId>,
    pub right_mention_id: Option<EventMentionId>,
    pub relation: EventIdentityState,
    pub decision_kind: EventIdentityDecisionKind,
    pub rationale: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityInvalidationRecord {
    pub invalidation_id: String,
    pub decision_id: EventIdentityDecisionId,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub rationale: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentitySplitRecord {
    pub split_id: EventIdentitySplitId,
    pub source_canonical_event_id: CanonicalEventId,
    #[serde(default)]
    pub target_canonical_event_ids: Vec<CanonicalEventId>,
    pub rationale: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanonicalEventCard {
    pub canonical_event_id: CanonicalEventId,
    pub canonical_label: String,
    pub normalized_predicate: String,
    pub event_type: String,
    #[serde(default)]
    pub mention_ids: Vec<EventMentionId>,
    #[serde(default)]
    pub document_ids: Vec<String>,
    #[serde(default)]
    pub strongest_time_anchor_ids: Vec<TemporalAnchorId>,
    #[serde(default)]
    pub strongest_participant_slots: Vec<EventParticipantSlot>,
    #[serde(default)]
    pub related_temporal_event_ids: Vec<String>,
    #[serde(default)]
    pub related_causal_event_ids: Vec<String>,
    #[serde(default)]
    pub open_dispute_ids: Vec<EventIdentityHypothesisId>,
    #[serde(default)]
    pub incompatible_hypothesis_ids: Vec<EventIdentityHypothesisId>,
    pub revision_start: u64,
    pub revision_end: u64,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityCompilerSummary {
    #[serde(default)]
    pub mention_packet_count: usize,
    #[serde(default)]
    pub hypothesis_count: usize,
    #[serde(default)]
    pub canonical_event_count: usize,
    #[serde(default)]
    pub membership_count: usize,
    #[serde(default)]
    pub decision_count: usize,
    #[serde(default)]
    pub invalidation_count: usize,
    #[serde(default)]
    pub split_count: usize,
    #[serde(default)]
    pub card_count: usize,
    #[serde(default)]
    pub relation_counts: std::collections::BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityScopeSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    #[serde(default)]
    pub mention_packets: Vec<EventMentionPacket>,
    #[serde(default)]
    pub identity_hypotheses: Vec<EventIdentityHypothesis>,
    #[serde(default)]
    pub canonical_events: Vec<CanonicalEventRecord>,
    #[serde(default)]
    pub memberships: Vec<EventIdentityMembershipRecord>,
    #[serde(default)]
    pub decisions: Vec<EventIdentityLedgerRecord>,
    #[serde(default)]
    pub decision_history: Vec<EventIdentityLedgerRecord>,
    #[serde(default)]
    pub invalidations: Vec<EventIdentityInvalidationRecord>,
    #[serde(default)]
    pub splits: Vec<EventIdentitySplitRecord>,
    #[serde(default)]
    pub canonical_event_cards: Vec<CanonicalEventCard>,
    pub summary: EventIdentityCompilerSummary,
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
    pub canonical_event_id: Option<CanonicalEventId>,
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
    pub canonical_caused_by_event_id: Option<CanonicalEventId>,
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

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphCompilerSummary {
    pub claim_node_count: usize,
    pub event_node_count: usize,
    pub state_node_count: usize,
    pub view_node_count: usize,
    pub value_node_count: usize,
    pub time_anchor_node_count: usize,
    pub conflict_node_count: usize,
    pub gap_node_count: usize,
    pub temporal_edge_count: usize,
    pub causal_edge_count: usize,
    pub support_edge_count: usize,
    pub projection_vertex_count: usize,
    pub projection_edge_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphScopeSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    pub graph_batch: KernelMutationBatch,
    pub event_identity_generation: Option<u64>,
    pub temporal_generation: Option<u64>,
    pub causal_generation: Option<u64>,
    pub memory_generation: Option<u64>,
    pub summary: GraphCompilerSummary,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SemanticGraphNodeKind {
    Chunk,
    Claim,
    State,
    Event,
    Entity,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SemanticEdgeFamily {
    ChunkNeighbor,
    ClaimSupport,
    ClaimContradiction,
    StateSupport,
    StateContradiction,
    ContradictorySupportRegion,
    SameSlotFamily,
    SameProcess,
    RelatedEvent,
    MissingIntermediateCause,
    EntityStateSupport,
    EntityEventSupport,
    EventNeighbor,
    EntityRoleNeighbor,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SemanticCandidateStatus {
    #[default]
    Generated,
    ReviewedSupport,
    ReviewedContradiction,
    Deferred,
    Rejected,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticGraphNodeRecord {
    pub node_id: String,
    #[serde(default)]
    pub node_kind: SemanticGraphNodeKind,
    pub document_id: Option<String>,
    pub narrative_id: Option<String>,
    pub text_key: String,
    pub text_hash: u64,
    pub truth_plane: Option<String>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticGraphEdgeCandidate {
    pub edge_id: String,
    #[serde(default)]
    pub family: SemanticEdgeFamily,
    pub source_node_id: String,
    pub source_kind: SemanticGraphNodeKind,
    pub target_node_id: String,
    pub target_kind: SemanticGraphNodeKind,
    pub score_millis: u32,
    pub distance_millis: u32,
    #[serde(default)]
    pub candidate_status: SemanticCandidateStatus,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    #[serde(default)]
    pub model_evidence: Vec<String>,
    pub nli_support_millis: Option<u32>,
    pub nli_contradiction_millis: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticGraphCompilerSummary {
    pub node_count: usize,
    pub edge_count: usize,
    pub reviewed_support_count: usize,
    pub reviewed_contradiction_count: usize,
    #[serde(default)]
    pub node_kind_counts: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    pub edge_family_counts: std::collections::BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticGraphScopeSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    pub model_id: String,
    pub embedding_profile: String,
    pub embedding_dim: usize,
    #[serde(default)]
    pub candidate_nodes: Vec<SemanticGraphNodeRecord>,
    #[serde(default)]
    pub candidate_edges: Vec<SemanticGraphEdgeCandidate>,
    pub candidate_graph_batch: KernelMutationBatch,
    pub graph_generation: Option<u64>,
    pub memory_generation: Option<u64>,
    pub event_identity_generation: Option<u64>,
    pub summary: SemanticGraphCompilerSummary,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StateSlotOwnerType {
    #[default]
    Entity,
    Project,
    Task,
    Relationship,
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StateSlotValueType {
    #[default]
    EntityRef,
    Enum,
    Date,
    Interval,
    String,
    RankedChoice,
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StateSlotCardinality {
    #[default]
    Single,
    Multi,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StateSlotTemporalMode {
    Point,
    Interval,
    #[default]
    DurableUntilChanged,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StateSlotUpdateOperator {
    Exists,
    Add,
    #[default]
    Replace,
    CloseInterval,
    Deprecate,
    Infer,
    Defer,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StateSlotLifecycle {
    #[default]
    Reserved,
    Candidate,
    Active,
    Stable,
    Deprecated,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StateSlotFamilyId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StateSlotDefinitionId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StateSlotCandidateId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StateSlotPromotionDecisionId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StateWriteProposalId(pub String);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSlotFamilyRecord {
    pub family_id: StateSlotFamilyId,
    pub family_key: String,
    pub label: String,
    pub description: String,
    pub owner_type: StateSlotOwnerType,
    pub lifecycle: StateSlotLifecycle,
    pub salience_millis: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSlotDefinitionRecord {
    pub slot_id: StateSlotDefinitionId,
    pub family_id: StateSlotFamilyId,
    pub slot_key: String,
    pub slot_name: String,
    pub owner_type: StateSlotOwnerType,
    pub value_type: StateSlotValueType,
    pub cardinality: StateSlotCardinality,
    pub temporal_mode: StateSlotTemporalMode,
    pub update_operator: StateSlotUpdateOperator,
    pub evidence_threshold_millis: u32,
    pub contradiction_policy: String,
    pub salience_millis: u32,
    pub lifecycle: StateSlotLifecycle,
    pub single_value: bool,
    pub relationship_only: bool,
    #[serde(default)]
    pub relation_families: Vec<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSlotCandidateRecord {
    pub candidate_id: StateSlotCandidateId,
    pub family_id: StateSlotFamilyId,
    pub slot_key: String,
    pub normalized_name: String,
    pub source_phrase: String,
    pub owner_type: StateSlotOwnerType,
    pub value_type: StateSlotValueType,
    pub support_count: usize,
    pub document_count: usize,
    pub canonicalization_score_millis: u32,
    pub utility_score_millis: u32,
    pub conflict_count: usize,
    #[serde(default)]
    pub relation_families: Vec<String>,
    #[serde(default)]
    pub value_samples: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSlotPromotionDecisionRecord {
    pub decision_id: StateSlotPromotionDecisionId,
    pub slot_id: StateSlotDefinitionId,
    #[serde(default)]
    pub candidate_ids: Vec<StateSlotCandidateId>,
    pub previous_lifecycle: StateSlotLifecycle,
    pub next_lifecycle: StateSlotLifecycle,
    pub rationale: String,
    pub support_count: usize,
    pub conflict_count: usize,
    pub utility_score_millis: u32,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateWriteProposal {
    pub proposal_id: StateWriteProposalId,
    pub owner_entity_id: EntityId,
    pub owner_type: StateSlotOwnerType,
    pub slot_key: String,
    pub before_value: Option<String>,
    pub after_value: Option<String>,
    pub after_value_entity_id: Option<EntityId>,
    pub operation: StateSlotUpdateOperator,
    pub effective_time: Option<i64>,
    pub source_document_id: String,
    pub source_event_id: Option<String>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSchemaCompilerSummary {
    pub family_count: usize,
    pub definition_count: usize,
    pub active_definition_count: usize,
    pub stable_definition_count: usize,
    pub candidate_definition_count: usize,
    pub candidate_count: usize,
    pub promotion_decision_count: usize,
    pub write_proposal_count: usize,
    #[serde(default)]
    pub family_counts: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    pub lifecycle_counts: std::collections::BTreeMap<String, usize>,
    #[serde(default)]
    pub owner_type_counts: std::collections::BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSchemaScopeSidecar {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: Option<ScopeOrd>,
    pub session_id: Option<SessionId>,
    pub updated_at: i64,
    pub generation: u64,
    #[serde(default)]
    pub slot_families: Vec<StateSlotFamilyRecord>,
    #[serde(default)]
    pub slot_definitions: Vec<StateSlotDefinitionRecord>,
    #[serde(default)]
    pub slot_candidates: Vec<StateSlotCandidateRecord>,
    #[serde(default)]
    pub promotion_decisions: Vec<StateSlotPromotionDecisionRecord>,
    #[serde(default)]
    pub write_proposals: Vec<StateWriteProposal>,
    pub summary: StateSchemaCompilerSummary,
    #[serde(default)]
    pub diagnostics: std::collections::BTreeMap<String, usize>,
}

pub fn default_state_slot_families() -> Vec<StateSlotFamilyRecord> {
    vec![
        state_slot_family(
            "location",
            "Location",
            "Entity location and physical placement.",
            StateSlotOwnerType::Entity,
            StateSlotLifecycle::Active,
            900,
        ),
        state_slot_family(
            "affiliation",
            "Affiliation",
            "Employment, membership, and durable affiliation state.",
            StateSlotOwnerType::Entity,
            StateSlotLifecycle::Active,
            900,
        ),
        state_slot_family(
            "relationship",
            "Relationship",
            "Relationship-ledger slots that stay pair-shaped instead of scalar state.",
            StateSlotOwnerType::Relationship,
            StateSlotLifecycle::Active,
            700,
        ),
        state_slot_family(
            "lifecycle",
            "Lifecycle",
            "Lifecycle and status state for projects and tasks.",
            StateSlotOwnerType::Project,
            StateSlotLifecycle::Reserved,
            860,
        ),
        state_slot_family(
            "assignment",
            "Assignment",
            "Ownership and assignment state for tasks and work items.",
            StateSlotOwnerType::Task,
            StateSlotLifecycle::Reserved,
            840,
        ),
        state_slot_family(
            "schedule",
            "Schedule",
            "Durable deadlines and temporal commitments.",
            StateSlotOwnerType::Task,
            StateSlotLifecycle::Reserved,
            840,
        ),
        state_slot_family(
            "role_preference",
            "RolePreference",
            "Role and preference state that usually needs stronger corroboration.",
            StateSlotOwnerType::Entity,
            StateSlotLifecycle::Reserved,
            680,
        ),
        state_slot_family(
            "discovered",
            "Discovered",
            "Bottom-up discovered slot candidates awaiting schema promotion.",
            StateSlotOwnerType::Unknown,
            StateSlotLifecycle::Candidate,
            500,
        ),
    ]
}

pub fn default_state_slot_definitions() -> Vec<StateSlotDefinitionRecord> {
    vec![
        state_slot_definition(
            "slot:entity.location",
            "location",
            "entity.location",
            StateSlotOwnerType::Entity,
            StateSlotValueType::EntityRef,
            StateSlotCardinality::Single,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Replace,
            650,
            "prefer newer location evidence and keep conflicts open",
            920,
            StateSlotLifecycle::Active,
            true,
            false,
            &["located_in"],
            &["location", "place"],
        ),
        state_slot_definition(
            "slot:entity.employer",
            "affiliation",
            "entity.employer",
            StateSlotOwnerType::Entity,
            StateSlotValueType::EntityRef,
            StateSlotCardinality::Single,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Replace,
            700,
            "prefer supported employment evidence and preserve contradictions",
            900,
            StateSlotLifecycle::Active,
            true,
            false,
            &["works_for"],
            &["employer", "employment"],
        ),
        state_slot_definition(
            "slot:entity.membership",
            "affiliation",
            "entity.membership",
            StateSlotOwnerType::Entity,
            StateSlotValueType::EntityRef,
            StateSlotCardinality::Single,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Replace,
            680,
            "prefer supported membership evidence and preserve contradictions",
            860,
            StateSlotLifecycle::Active,
            true,
            false,
            &["member_of"],
            &["membership"],
        ),
        state_slot_definition(
            "slot:relationship.commands",
            "relationship",
            "relationship.commands",
            StateSlotOwnerType::Relationship,
            StateSlotValueType::EntityRef,
            StateSlotCardinality::Multi,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Add,
            600,
            "relationship ledgers accept multiple supported links",
            700,
            StateSlotLifecycle::Active,
            false,
            true,
            &["commands"],
            &["command"],
        ),
        state_slot_definition(
            "slot:relationship.protects",
            "relationship",
            "relationship.protects",
            StateSlotOwnerType::Relationship,
            StateSlotValueType::EntityRef,
            StateSlotCardinality::Multi,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Add,
            600,
            "relationship ledgers accept multiple supported links",
            700,
            StateSlotLifecycle::Active,
            false,
            true,
            &["protects"],
            &["protect"],
        ),
        state_slot_definition(
            "slot:project.status",
            "lifecycle",
            "project.status",
            StateSlotOwnerType::Project,
            StateSlotValueType::Enum,
            StateSlotCardinality::Single,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Replace,
            780,
            "project lifecycle values supersede prior current state when explicit",
            860,
            StateSlotLifecycle::Reserved,
            true,
            false,
            &["project_status", "has_status", "status", "phase"],
            &["project status", "status"],
        ),
        state_slot_definition(
            "slot:task.owner",
            "assignment",
            "task.owner",
            StateSlotOwnerType::Task,
            StateSlotValueType::EntityRef,
            StateSlotCardinality::Single,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Replace,
            760,
            "task ownership closes the previous current owner when explicit",
            840,
            StateSlotLifecycle::Reserved,
            true,
            false,
            &["assigned_to", "owned_by", "task_owner", "assignee"],
            &["owner", "assignee"],
        ),
        state_slot_definition(
            "slot:task.due_date",
            "schedule",
            "task.due_date",
            StateSlotOwnerType::Task,
            StateSlotValueType::Date,
            StateSlotCardinality::Single,
            StateSlotTemporalMode::Point,
            StateSlotUpdateOperator::Replace,
            760,
            "task due dates replace the previous deadline when explicit",
            830,
            StateSlotLifecycle::Reserved,
            true,
            false,
            &["due_date", "due_on", "deadline", "scheduled_for"],
            &["due date", "deadline"],
        ),
        state_slot_definition(
            "slot:task.completion_state",
            "lifecycle",
            "task.completion_state",
            StateSlotOwnerType::Task,
            StateSlotValueType::Enum,
            StateSlotCardinality::Single,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Replace,
            760,
            "task completion values supersede earlier completion state when explicit",
            840,
            StateSlotLifecycle::Reserved,
            true,
            false,
            &["task_status", "completion_state", "completed", "task_state"],
            &["completion", "task status"],
        ),
        state_slot_definition(
            "slot:entity.preference",
            "role_preference",
            "entity.preference",
            StateSlotOwnerType::Entity,
            StateSlotValueType::RankedChoice,
            StateSlotCardinality::Multi,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Infer,
            860,
            "preferences stay candidate until repeatedly corroborated",
            620,
            StateSlotLifecycle::Reserved,
            false,
            false,
            &["preference", "prefers", "likes"],
            &["preference", "likes", "prefers"],
        ),
        state_slot_definition(
            "slot:entity.role",
            "role_preference",
            "entity.role",
            StateSlotOwnerType::Entity,
            StateSlotValueType::String,
            StateSlotCardinality::Multi,
            StateSlotTemporalMode::DurableUntilChanged,
            StateSlotUpdateOperator::Replace,
            820,
            "roles need repeated explicit support before becoming durable truth",
            640,
            StateSlotLifecycle::Reserved,
            false,
            false,
            &["role", "acts_as", "serves_as"],
            &["role", "title"],
        ),
    ]
}

fn state_slot_family(
    family_key: &str,
    label: &str,
    description: &str,
    owner_type: StateSlotOwnerType,
    lifecycle: StateSlotLifecycle,
    salience_millis: u32,
) -> StateSlotFamilyRecord {
    StateSlotFamilyRecord {
        family_id: StateSlotFamilyId(format!("family:{family_key}")),
        family_key: family_key.to_owned(),
        label: label.to_owned(),
        description: description.to_owned(),
        owner_type,
        lifecycle,
        salience_millis,
    }
}

fn state_slot_definition(
    slot_id: &str,
    family_key: &str,
    slot_key: &str,
    owner_type: StateSlotOwnerType,
    value_type: StateSlotValueType,
    cardinality: StateSlotCardinality,
    temporal_mode: StateSlotTemporalMode,
    update_operator: StateSlotUpdateOperator,
    evidence_threshold_millis: u32,
    contradiction_policy: &str,
    salience_millis: u32,
    lifecycle: StateSlotLifecycle,
    single_value: bool,
    relationship_only: bool,
    relation_families: &[&str],
    aliases: &[&str],
) -> StateSlotDefinitionRecord {
    let slot_name = slot_key.rsplit('.').next().unwrap_or(slot_key).to_owned();
    StateSlotDefinitionRecord {
        slot_id: StateSlotDefinitionId(slot_id.to_owned()),
        family_id: StateSlotFamilyId(format!("family:{family_key}")),
        slot_key: slot_key.to_owned(),
        slot_name,
        owner_type,
        value_type,
        cardinality,
        temporal_mode,
        update_operator,
        evidence_threshold_millis,
        contradiction_policy: contradiction_policy.to_owned(),
        salience_millis,
        lifecycle,
        single_value,
        relationship_only,
        relation_families: relation_families
            .iter()
            .map(|value| (*value).to_owned())
            .collect(),
        aliases: aliases.iter().map(|value| (*value).to_owned()).collect(),
    }
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
