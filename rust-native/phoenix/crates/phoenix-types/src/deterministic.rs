use compact_str::CompactString;
use rowan::{TextRange as RowanTextRange, TextSize};
use serde::{Deserialize, Serialize};
use slotmap::new_key_type;
use smallvec::SmallVec;

use crate::{
    ClaimId, ConceptId, DocumentId, EntityId, EntityKind, EventId, MentionEntityRef, NoteId,
    PosTag, StateId, TextRange, TokenClass, ValueId,
};

new_key_type! { pub struct DocumentKey; }
new_key_type! { pub struct ChunkKey; }
new_key_type! { pub struct UnitKey; }
new_key_type! { pub struct MentionKey; }
new_key_type! { pub struct EntityKey; }
new_key_type! { pub struct AliasKey; }
new_key_type! { pub struct ClaimKey; }
new_key_type! { pub struct EventKey; }
new_key_type! { pub struct StateKey; }
new_key_type! { pub struct ValueKey; }
new_key_type! { pub struct ConceptKey; }
new_key_type! { pub struct QuoteKey; }
new_key_type! { pub struct TimeKey; }
new_key_type! { pub struct EdgeKey; }

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceRange {
    pub start: u32,
    pub end: u32,
}

impl SourceRange {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    pub fn as_rowan(self) -> RowanTextRange {
        RowanTextRange::new(TextSize::from(self.start), TextSize::from(self.end))
    }
}

impl From<RowanTextRange> for SourceRange {
    fn from(value: RowanTextRange) -> Self {
        Self {
            start: u32::from(value.start()),
            end: u32::from(value.end()),
        }
    }
}

impl From<TextRange> for SourceRange {
    fn from(value: TextRange) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProvenanceRef {
    pub document_id: Option<DocumentId>,
    pub note_id: Option<NoteId>,
    pub label: CompactString,
    pub kind: Option<CompactString>,
    pub range: SourceRange,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TruthStatus {
    Candidate,
    Asserted,
    Rejected,
    Expired,
    #[default]
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Polarity {
    #[default]
    Positive,
    Negative,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalKind {
    #[default]
    Causes,
    Enables,
    Prevents,
    Hinders,
    Motivates,
    PurposeOf,
    ResultsIn,
    Explains,
    ConditionFor,
    TriggerFor,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalEvidenceKind {
    #[default]
    ExplicitCue,
    StructuralPattern,
    StateTransitionTrigger,
    SemanticSchema,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind", content = "id")]
pub enum SemanticNodeRef {
    Event(EventId),
    Claim(ClaimId),
    State(StateId),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticOrder {
    pub doc_ord: u32,
    pub section_ord: u32,
    pub sentence_ord: u32,
    pub clause_ord: u32,
    pub local_ord: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BiTemporalWindow {
    pub valid_from: Option<i64>,
    pub valid_to: Option<i64>,
    pub recorded_from: Option<i64>,
    pub recorded_to: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SurfaceUnitKind {
    #[default]
    Sentence,
    Paragraph,
    Clause,
    Quote,
    SpeakerCue,
    Phrase,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PhraseKind {
    #[default]
    Clause,
    Np,
    Vp,
    Pp,
    Ap,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Token {
    pub range: SourceRange,
    pub surface: CompactString,
    pub normalized: CompactString,
    pub class: Option<TokenClass>,
    pub pos: Option<PosTag>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sentence {
    pub index: usize,
    pub range: SourceRange,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Clause {
    pub sentence_index: usize,
    pub range: SourceRange,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuoteBlock {
    pub range: SourceRange,
    pub sentence_index: usize,
    pub cue_range: Option<SourceRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerCue {
    pub range: SourceRange,
    pub sentence_index: usize,
    pub text: CompactString,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhraseNode {
    pub kind: PhraseKind,
    pub range: SourceRange,
    pub head: Option<SourceRange>,
    #[serde(default)]
    pub modifiers: SmallVec<[SourceRange; 4]>,
    pub sentence_index: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attachment {
    pub source: SourceRange,
    pub target: SourceRange,
    pub sentence_index: usize,
    pub label: CompactString,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceUnit {
    pub kind: SurfaceUnitKind,
    pub key: Option<UnitKey>,
    pub range: SourceRange,
    pub sentence_index: usize,
    pub chunk_id_hint: Option<CompactString>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SurfaceDocument {
    #[serde(default)]
    pub tokens: Vec<Token>,
    #[serde(default)]
    pub sentences: Vec<Sentence>,
    #[serde(default)]
    pub clauses: Vec<Clause>,
    #[serde(default)]
    pub quote_blocks: Vec<QuoteBlock>,
    #[serde(default)]
    pub speaker_cues: Vec<SpeakerCue>,
    #[serde(default)]
    pub phrases: Vec<PhraseNode>,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
    #[serde(default)]
    pub units: Vec<SurfaceUnit>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MentionClass {
    #[default]
    Named,
    Nominal,
    Pronoun,
    Discovery,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MentionFeatures {
    pub normalized: CompactString,
    pub sentence_index: usize,
    pub chunk_index: Option<u32>,
    pub kind: Option<EntityKind>,
    pub entity_ref: Option<MentionEntityRef>,
    pub confidence_millis: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MentionContext {
    pub unit_key: Option<UnitKey>,
    pub sentence_range: Option<SourceRange>,
    pub clause_range: Option<SourceRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PreparedMentionRecord {
    pub mention_id: Option<MentionKey>,
    pub range: SourceRange,
    pub surface: CompactString,
    pub class: MentionClass,
    pub features: MentionFeatures,
    pub context: MentionContext,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorefCluster {
    pub cluster_id: CompactString,
    #[serde(default)]
    pub member_mentions: SmallVec<[usize; 8]>,
    pub representative_surface: CompactString,
    pub confidence_millis: u32,
    pub ambiguous: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateEntityRef {
    pub entity_id: EntityId,
    pub source: CompactString,
    pub score_millis: i32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionDecisionRecord {
    pub mention_index: usize,
    pub entity_id: Option<EntityId>,
    pub status: TruthStatus,
    pub confidence_millis: u32,
    pub margin_millis: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredicateFrame {
    pub predicate: CompactString,
    pub trigger_range: SourceRange,
    pub relation_type: CompactString,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Argument {
    pub role: CompactString,
    pub mention_index: Option<usize>,
    pub entity_id: Option<EntityId>,
    pub range: Option<SourceRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeOp {
    pub kind: CompactString,
    pub polarity: Option<CompactString>,
    pub modality: Option<CompactString>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AttributionFrame {
    pub source_entity_id: Option<EntityId>,
    pub quote_range: Option<SourceRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConditionalFrame {
    pub condition_range: Option<SourceRange>,
    pub consequent_range: Option<SourceRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuoteFrame {
    pub quote_range: SourceRange,
    pub speaker_entity_id: Option<EntityId>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Proposition {
    pub proposition_id: CompactString,
    pub sentence_index: usize,
    pub predicate: PredicateFrame,
    pub clause_range: Option<SourceRange>,
    #[serde(default)]
    pub arguments: SmallVec<[Argument; 4]>,
    #[serde(default)]
    pub scope_ops: SmallVec<[ScopeOp; 2]>,
    pub attribution: Option<AttributionFrame>,
    pub conditional: Option<ConditionalFrame>,
    pub quote: Option<QuoteFrame>,
    #[serde(default)]
    pub evidence: SmallVec<[ProvenanceRef; 2]>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventRecord {
    pub event_id: Option<EventId>,
    pub label: CompactString,
    pub proposition_id: CompactString,
    pub order: SemanticOrder,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ClaimRecord {
    pub claim_id: Option<ClaimId>,
    pub label: CompactString,
    pub proposition_id: CompactString,
    pub order: SemanticOrder,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateRecord {
    pub state_id: Option<StateId>,
    pub label: CompactString,
    pub proposition_id: CompactString,
    pub order: SemanticOrder,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ValueRecord {
    pub value_id: Option<ValueId>,
    pub label: CompactString,
    pub proposition_id: CompactString,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConceptRecord {
    pub concept_id: Option<ConceptId>,
    pub label: CompactString,
    pub proposition_id: CompactString,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalCandidate {
    pub source: SemanticNodeRef,
    pub target: SemanticNodeRef,
    pub kind: CausalKind,
    pub confidence_millis: u16,
    pub status: TruthStatus,
    pub cue: Option<CompactString>,
    pub cue_span: Option<SourceRange>,
    pub evidence_kind: CausalEvidenceKind,
    pub attributed_to: Option<EntityId>,
    pub polarity: Polarity,
    #[serde(default)]
    pub provenance: SmallVec<[ProvenanceRef; 2]>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalLink {
    pub edge_id: Option<crate::EdgeId>,
    pub source: SemanticNodeRef,
    pub target: SemanticNodeRef,
    pub kind: CausalKind,
    pub confidence_millis: u16,
    pub status: TruthStatus,
    pub cue: Option<CompactString>,
    pub cue_span: Option<SourceRange>,
    pub attributed_to: Option<EntityId>,
    pub polarity: Polarity,
    #[serde(default)]
    pub provenance: SmallVec<[ProvenanceRef; 2]>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalDiagnostic {
    pub code: CompactString,
    pub message: CompactString,
    pub proposition_id: Option<CompactString>,
    pub cue_span: Option<SourceRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalBundle {
    #[serde(default)]
    pub candidates: Vec<CausalCandidate>,
    #[serde(default)]
    pub links: Vec<CausalLink>,
    #[serde(default)]
    pub diagnostics: Vec<CausalDiagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticRelation {
    pub edge_type: CompactString,
    pub source_id: CompactString,
    pub target_id: CompactString,
    pub status: TruthStatus,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimeAnchorRecord {
    pub time_id: Option<TimeKey>,
    pub label: CompactString,
    pub interval: BiTemporalWindow,
}
