use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

mod binary;
mod deterministic;

pub use binary::*;
pub use deterministic::*;

macro_rules! string_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
    };
}

string_id!(DocumentId);
string_id!(NoteId);
string_id!(EntityId);
string_id!(AliasId);
string_id!(ChunkId);
string_id!(MentionId);
string_id!(ClaimId);
string_id!(EventId);
string_id!(StateId);
string_id!(ValueId);
string_id!(ConceptId);
string_id!(QuoteId);
string_id!(TimeId);
string_id!(EdgeId);
string_id!(SessionId);
string_id!(ThreadId);
string_id!(CommitId);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum EntityKind {
    Character,
    Location,
    Npc,
    Item,
    Faction,
    Organization,
    Event,
    Concept,
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum GenderHint {
    Unknown,
    Male,
    Female,
    Neutral,
    Plural,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NumberHint {
    Singular,
    Plural,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeKey {
    pub world_id: Option<String>,
    pub narrative_id: Option<String>,
    pub folder_id: Option<String>,
    pub folder_path: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TemporalSource {
    Chapter,
    Boundary,
    Calendar,
    Story,
    Ordinal,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum BoundaryKind {
    #[default]
    Chapter,
    Heading,
    Section,
    Act,
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BoundaryRef {
    pub document_id: DocumentId,
    pub boundary_id: u32,
    pub kind: BoundaryKind,
    pub ordinal: u32,
    pub depth: u8,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DetectedBoundary {
    pub start: u32,
    pub end: Option<u32>,
    pub boundary_id: u32,
    pub parent_boundary_id: Option<u32>,
    pub ordinal: u32,
    pub kind: BoundaryKind,
    pub depth: u8,
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "mode")]
pub enum BoundaryDetectionStrategy {
    Disabled,
    Keywords {
        keywords: Vec<String>,
    },
    MarkdownHeadings {
        max_depth: u8,
    },
    Both {
        keywords: Vec<String>,
        max_depth: u8,
    },
}

impl Default for BoundaryDetectionStrategy {
    fn default() -> Self {
        Self::Both {
            keywords: Vec::new(),
            max_depth: 6,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalMarker {
    pub source: Option<TemporalSource>,
    pub chapter: Option<u32>,
    pub boundary_doc_id: Option<DocumentId>,
    pub boundary_id: Option<u32>,
    pub boundary_ordinal: Option<i64>,
    pub boundary_end_ordinal: Option<i64>,
    pub boundary_kind: Option<BoundaryKind>,
    pub calendar: Option<i64>,
    pub story_time: Option<String>,
    pub ordinal: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextRange {
    pub start: u32,
    pub end: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceSpan {
    pub document_id: Option<DocumentId>,
    pub note_id: Option<NoteId>,
    pub label: String,
    pub kind: Option<String>,
    pub range: TextRange,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RuntimeTarget {
    Native,
    Wasm,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StorageMode {
    CozoMem,
    CozoSqlite,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SnapshotPolicy {
    Manual,
    OnCommit,
    Debounced,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FeatureFlags {
    pub scanner: bool,
    pub structure: bool,
    pub graptor: bool,
    pub gldr: bool,
    pub semantic: bool,
    pub candidate_graph: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeConfig {
    pub target: RuntimeTarget,
    pub storage: StorageMode,
    pub snapshot_policy: SnapshotPolicy,
    pub feature_flags: FeatureFlags,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            target: RuntimeTarget::Native,
            storage: StorageMode::CozoMem,
            snapshot_policy: SnapshotPolicy::Manual,
            feature_flags: FeatureFlags {
                scanner: true,
                structure: true,
                graptor: true,
                gldr: true,
                semantic: false,
                candidate_graph: true,
            },
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestDocument {
    pub document_id: DocumentId,
    pub note_id: Option<NoteId>,
    pub title: String,
    pub text: String,
    pub scope: ScopeKey,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestRequest {
    pub session_id: Option<SessionId>,
    pub documents: Vec<IngestDocument>,
    pub commit: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestResult {
    pub session_id: Option<SessionId>,
    pub document_count: usize,
    pub warning_count: usize,
    pub documents: Vec<IngestDocumentSummary>,
    pub chunk_stats: Option<ChunkStats>,
    pub graph_summary: Option<GraphSummary>,
    pub entity_summary: Option<EntitySummary>,
    pub discovery_summary: Option<DiscoverySummary>,
    pub retrieval_summary: Option<RetrievalSummary>,
    pub relation_counts: Vec<RelationCount>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestDocumentSummary {
    pub document_id: DocumentId,
    pub note_id: Option<NoteId>,
    pub chapter_count: usize,
    pub boundary_count: usize,
    pub parent_count: usize,
    pub leaf_count: usize,
    pub entity_count: usize,
    pub edge_count: usize,
    pub has_front_matter_chapter: bool,
    pub has_front_matter_boundary: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkStats {
    pub documents: usize,
    pub total_chapters: usize,
    pub total_boundaries: usize,
    pub total_parents: usize,
    pub total_leaves: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphSummary {
    pub documents: usize,
    pub total_chapters: usize,
    pub total_boundaries: usize,
    pub total_leaves: usize,
    pub total_entities: usize,
    pub total_mentions: usize,
    pub total_edges: usize,
    pub cross_chapter_links: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntitySummary {
    pub total_entities: usize,
    pub total_aliases: usize,
    pub total_mentions: usize,
    pub multi_chapter_entities: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoverySummary {
    pub candidate_count: usize,
    pub mention_count: usize,
    pub persisted_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RetrievalSummary {
    pub qgram_documents: usize,
    pub gldr_chunks: usize,
    pub gldr_entities: usize,
    pub gldr_edges: usize,
    pub raptor_documents: usize,
    pub raptor_leaves: usize,
    pub raptor_enabled: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum QueryTarget {
    Chunks,
    Nodes,
    Graph,
    Semantic,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SemanticQueryVector {
    pub values: Vec<f32>,
}

impl PartialEq for SemanticQueryVector {
    fn eq(&self, other: &Self) -> bool {
        self.values.len() == other.values.len()
            && self
                .values
                .iter()
                .zip(other.values.iter())
                .all(|(left, right)| left.to_bits() == right.to_bits())
    }
}

impl Eq for SemanticQueryVector {}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryRequest {
    pub session_id: Option<SessionId>,
    pub query: String,
    pub scope: ScopeKey,
    pub targets: Vec<QueryTarget>,
    pub limit: Option<usize>,
    pub temporal: Option<TemporalMarker>,
    pub semantic_query_vector: Option<SemanticQueryVector>,
    #[serde(default)]
    pub include_candidate_graph: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkHit {
    pub chunk_id: String,
    pub score: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NodeHit {
    pub entity_id: Option<EntityId>,
    pub score: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Diagnostic {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationCount {
    pub relation: String,
    pub rows: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryResult {
    pub session_id: Option<SessionId>,
    pub chunk_hits: Vec<ChunkHit>,
    pub node_hits: Vec<NodeHit>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyzeTextRequest {
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDeltaRequest {
    pub session_id: SessionId,
    pub scope: ScopeKey,
    pub changed_documents: Vec<DocumentId>,
    pub limit: Option<usize>,
    pub since_commit: Option<CommitId>,
    #[serde(default)]
    pub include_candidate_graph: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDeltaChunk {
    pub vertex_id: String,
    pub chunk_id: String,
    pub document_id: DocumentId,
    pub note_id: Option<NoteId>,
    pub chapter_id: u32,
    pub boundary_id: Option<u32>,
    pub boundary_ordinal: Option<u32>,
    pub range: TextRange,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDeltaNode {
    pub node_id: String,
    pub kind: String,
    pub label: String,
    pub entity_id: Option<EntityId>,
    pub document_id: Option<DocumentId>,
    pub chapter_id: Option<u32>,
    pub boundary_id: Option<u32>,
    pub boundary_ordinal: Option<u32>,
    pub weight: i32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDeltaEdge {
    pub source_id: String,
    pub target_id: String,
    pub edge_type: String,
    pub weight: i32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphDeltaResult {
    pub session_id: SessionId,
    pub chunks: Vec<GraphDeltaChunk>,
    pub nodes: Vec<GraphDeltaNode>,
    pub edges: Vec<GraphDeltaEdge>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LexicalField {
    Title,
    Body,
    Tags,
    Summary,
    Other,
}

impl Default for LexicalField {
    fn default() -> Self {
        Self::Body
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexedTextField {
    pub field: LexicalField,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexedSpan {
    pub span_id: String,
    pub note_id: Option<NoteId>,
    pub document_id: Option<DocumentId>,
    pub scope: ScopeKey,
    pub fields: Vec<IndexedTextField>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImplicitMatchHit {
    pub range: TextRange,
    pub surface: String,
    pub label: String,
    pub kind: Option<EntityKind>,
    pub resolved_entity_id: Option<EntityId>,
    pub candidate_entity_ids: Vec<EntityId>,
    pub candidate_labels: Vec<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpanHit {
    pub span_id: String,
    pub note_id: Option<NoteId>,
    pub document_id: Option<DocumentId>,
    pub score: f64,
    pub coverage: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LexicalSearchResult {
    pub span_hits: Vec<SpanHit>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SnapshotDto {
    pub schema_version: String,
    pub created_at: i64,
    pub payload_bytes: usize,
    pub relation_counts: Vec<RelationCount>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LexiconSurfaceSource {
    Canonical,
    Alias,
    AutoAlias,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LexiconEntry {
    pub entity_id: EntityId,
    pub label: String,
    pub aliases: Vec<String>,
    pub kind: Option<EntityKind>,
    pub gender: Option<GenderHint>,
    pub number: Option<NumberHint>,
    pub scope: ScopeKey,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum KnownMatchSource {
    ExactCanonical,
    ExactAlias,
    ExactAutoAlias,
    FuzzyAnchor,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KnownMatch {
    pub range: TextRange,
    pub surface: String,
    pub entries: Vec<LexiconEntry>,
    pub source: Option<KnownMatchSource>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LexiconStats {
    pub entity_count: usize,
    pub exact_surface_count: usize,
    pub anchor_count: usize,
    pub unique_token_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LexiconSnapshot {
    pub version: u64,
    pub compiled_at: i64,
    pub fst_bytes: Vec<u8>,
    pub entries: Vec<LexiconEntry>,
    pub buckets: Vec<Vec<usize>>,
    pub exact_surfaces: Vec<String>,
    pub exact_surface_bucket_indices: Vec<usize>,
    pub exact_surface_sources: Vec<LexiconSurfaceSource>,
    pub unique_token_to_entry: BTreeMap<String, usize>,
    pub anchor_to_entries: BTreeMap<String, Vec<usize>>,
    pub entry_tokens: BTreeMap<String, Vec<String>>,
    pub stats: LexiconStats,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FuzzyMode {
    Off,
    AnchorOnly,
}

impl Default for FuzzyMode {
    fn default() -> Self {
        Self::AnchorOnly
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum NarrativeTransitivity {
    Intransitive,
    Transitive,
    Ditransitive,
}

impl Default for NarrativeTransitivity {
    fn default() -> Self {
        Self::Transitive
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrativeRule {
    pub lemma: String,
    pub event_class: String,
    pub relation_type: String,
    pub transitivity: NarrativeTransitivity,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DiscoveryThresholds {
    pub min_occurrences: u32,
    pub min_score: f32,
    pub narrative_bonus: f32,
    pub np_head_bonus: f32,
    pub capitalized_bonus: f32,
    pub sentence_start_penalty: f32,
    pub lowercase_alias_penalty: f32,
    pub dialogue_lead_penalty: f32,
}

impl Default for DiscoveryThresholds {
    fn default() -> Self {
        Self {
            min_occurrences: 2,
            min_score: 2.0,
            narrative_bonus: 0.5,
            np_head_bonus: 0.75,
            capitalized_bonus: 0.5,
            sentence_start_penalty: 0.75,
            lowercase_alias_penalty: 1.5,
            dialogue_lead_penalty: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScannerConfig {
    pub discovery_thresholds: DiscoveryThresholds,
    pub fuzzy_mode: FuzzyMode,
    pub stopword_profile: String,
    pub alias_rules: Vec<String>,
    pub narrative_overlay: Vec<NarrativeRule>,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            discovery_thresholds: DiscoveryThresholds::default(),
            fuzzy_mode: FuzzyMode::AnchorOnly,
            stopword_profile: "default".to_owned(),
            alias_rules: vec![
                "parenthetical".to_owned(),
                "aka".to_owned(),
                "also_known_as".to_owned(),
                "called".to_owned(),
                "appositive".to_owned(),
            ],
            narrative_overlay: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolverEntitySeed {
    pub entity_id: EntityId,
    pub canonical_name: String,
    pub aliases: Vec<String>,
    pub kind: Option<EntityKind>,
    pub gender: Option<GenderHint>,
    pub number: Option<NumberHint>,
    pub scope: ScopeKey,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanRequest {
    pub text: String,
    pub scope: ScopeKey,
    pub session_id: Option<SessionId>,
    pub resolver_seed: Vec<ResolverEntitySeed>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TokenClass {
    Word,
    Number,
    Punctuation,
    Symbol,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PosTag {
    Noun,
    Pronoun,
    ProperNoun,
    Verb,
    Auxiliary,
    Modal,
    Adjective,
    Adverb,
    Determiner,
    Preposition,
    Conjunction,
    RelativePronoun,
    Punctuation,
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TokenSpan {
    pub range: TextRange,
    pub token_class: Option<TokenClass>,
    pub pos: Option<PosTag>,
    pub masked: bool,
    pub capitalized: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SentenceSpan {
    pub index: usize,
    pub range: TextRange,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MentionSource {
    Known,
    Alias,
    Fuzzy,
    Discovery,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MentionEntityRef {
    Known(EntityId),
    Speculative(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MentionSpan {
    pub range: TextRange,
    pub surface: String,
    pub kind: Option<EntityKind>,
    pub entity_ref: Option<MentionEntityRef>,
    pub source: Option<MentionSource>,
    pub confidence: f32,
    pub sentence_index: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ChunkKind {
    Np,
    Vp,
    Pp,
    Clause,
    AdjP,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChunkSpan {
    pub kind: Option<ChunkKind>,
    pub range: TextRange,
    pub head: TextRange,
    pub modifiers: Vec<TextRange>,
    pub sentence_index: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NarrativeVerbHit {
    pub range: TextRange,
    pub lemma: String,
    pub event_class: String,
    pub relation_type: String,
    pub transitivity: Option<NarrativeTransitivity>,
    pub sentence_index: usize,
    pub confidence: f32,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ResolverLinkKind {
    Pronoun,
    AliasCandidate,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolverLink {
    pub source_range: TextRange,
    pub target_range: Option<TextRange>,
    pub target_entity: Option<MentionEntityRef>,
    pub link_kind: Option<ResolverLinkKind>,
    pub confidence: f32,
    pub sentence_index: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanArtifact {
    pub sentences: Vec<SentenceSpan>,
    pub tokens: Vec<TokenSpan>,
    pub mentions: Vec<MentionSpan>,
    pub chunks: Vec<ChunkSpan>,
    pub resolver_links: Vec<ResolverLink>,
    pub narrative_hits: Vec<NarrativeVerbHit>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FrameSlot {
    pub range: TextRange,
    pub entity_ref: Option<MentionEntityRef>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VerbFrame {
    pub verb_range: TextRange,
    pub lemma: String,
    pub event_class: String,
    pub relation_type: String,
    pub transitivity: Option<NarrativeTransitivity>,
    pub subject_candidates: Vec<FrameSlot>,
    pub object_candidates: Vec<FrameSlot>,
    pub recipient_candidates: Vec<FrameSlot>,
    pub pp_attachments: Vec<TextRange>,
    pub clause_range: TextRange,
    pub evidence: Vec<EvidenceSpan>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationCandidate {
    pub sentence_index: usize,
    pub verb_range: TextRange,
    pub lemma: String,
    pub event_class: String,
    pub relation_type: String,
    pub subject: Option<FrameSlot>,
    pub object: Option<FrameSlot>,
    pub recipient: Option<FrameSlot>,
    pub attachments: Vec<TextRange>,
    pub evidence: Vec<EvidenceSpan>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SentenceFrame {
    pub sentence: SentenceSpan,
    pub mentions: Vec<MentionSpan>,
    pub chunks: Vec<ChunkSpan>,
    pub verb_frames: Vec<VerbFrame>,
    pub clause_ranges: Vec<TextRange>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StructureRequest {
    pub text: String,
    pub scan: ScanArtifact,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StructureArtifact {
    pub sentence_frames: Vec<SentenceFrame>,
    pub relations: Vec<RelationCandidate>,
    pub evidence_spans: Vec<EvidenceSpan>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u32)]
pub enum PacketKind {
    None = 0,
    Status = 1,
    InitRuntimeRequest = 2,
    InitRuntimeResult = 3,
    CreateSessionRequest = 4,
    CreateSessionResult = 5,
    CommitRequest = 6,
    CommitResult = 7,
    RebuildRequest = 8,
    RebuildResult = 9,
    IngestRequest = 10,
    IngestResult = 11,
    QueryRequest = 12,
    QueryResult = 13,
    SnapshotExportRequest = 14,
    SnapshotResult = 15,
    SnapshotImportRequest = 16,
    ScanRequest = 17,
    ScanResult = 18,
    StructureRequest = 19,
    StructureResult = 20,
    GraphDeltaRequest = 21,
    GraphDeltaResult = 22,
    SessionStateRequest = 23,
    SessionStateResult = 24,
    SessionStatsRequest = 25,
    SessionStatsResult = 26,
    AnalyzeTextRequest = 27,
    AnalyzeTextResult = 28,
    QueryBinaryRequest = 29,
    AnalyzeTextBinaryRequest = 30,
    IngestBinaryRequest = 31,
    ScanBinaryRequest = 32,
    StructureBinaryRequest = 33,
    StoreCommandRequest = 34,
    StoreCommandResult = 35,
    EmbedUpsertBinaryRequest = 36,
    EmbedUpsertResult = 37,
    Ack = 255,
}

impl Default for PacketKind {
    fn default() -> Self {
        Self::None
    }
}

impl From<u32> for PacketKind {
    fn from(value: u32) -> Self {
        match value {
            1 => Self::Status,
            2 => Self::InitRuntimeRequest,
            3 => Self::InitRuntimeResult,
            4 => Self::CreateSessionRequest,
            5 => Self::CreateSessionResult,
            6 => Self::CommitRequest,
            7 => Self::CommitResult,
            8 => Self::RebuildRequest,
            9 => Self::RebuildResult,
            10 => Self::IngestRequest,
            11 => Self::IngestResult,
            12 => Self::QueryRequest,
            13 => Self::QueryResult,
            14 => Self::SnapshotExportRequest,
            15 => Self::SnapshotResult,
            16 => Self::SnapshotImportRequest,
            17 => Self::ScanRequest,
            18 => Self::ScanResult,
            19 => Self::StructureRequest,
            20 => Self::StructureResult,
            21 => Self::GraphDeltaRequest,
            22 => Self::GraphDeltaResult,
            23 => Self::SessionStateRequest,
            24 => Self::SessionStateResult,
            25 => Self::SessionStatsRequest,
            26 => Self::SessionStatsResult,
            27 => Self::AnalyzeTextRequest,
            28 => Self::AnalyzeTextResult,
            29 => Self::QueryBinaryRequest,
            30 => Self::AnalyzeTextBinaryRequest,
            31 => Self::IngestBinaryRequest,
            32 => Self::ScanBinaryRequest,
            33 => Self::StructureBinaryRequest,
            34 => Self::StoreCommandRequest,
            35 => Self::StoreCommandResult,
            36 => Self::EmbedUpsertBinaryRequest,
            37 => Self::EmbedUpsertResult,
            255 => Self::Ack,
            _ => Self::None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeInitRequest {
    pub config: RuntimeConfig,
    pub storage_path: Option<String>,
    pub force_reset: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimeInitResult {
    pub ready: bool,
    pub schema_version: String,
    pub relation_count: usize,
    pub relation_counts: Vec<RelationCount>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionRequest {
    pub session_id: Option<SessionId>,
    pub label: String,
    pub scope: ScopeKey,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionRecord {
    pub session_id: SessionId,
    pub label: String,
    pub scope: ScopeKey,
    pub status: String,
    pub revision: u64,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionDocumentState {
    pub document_id: DocumentId,
    pub note_id: Option<NoteId>,
    pub chapter_count: usize,
    pub boundary_count: usize,
    pub chapter_titles: Vec<String>,
    pub boundary_labels: Vec<String>,
    pub parent_count: usize,
    pub leaf_count: usize,
    pub entity_count: usize,
    pub discovery_count: usize,
    pub has_front_matter_chapter: bool,
    pub has_front_matter_boundary: bool,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionState {
    pub session_id: SessionId,
    pub documents: Vec<SessionDocumentState>,
    pub manifest_namespaces: Vec<String>,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStats {
    pub session_id: SessionId,
    pub document_count: usize,
    pub chapter_count: usize,
    pub boundary_count: usize,
    pub parent_count: usize,
    pub leaf_count: usize,
    pub entity_count: usize,
    pub discovery_candidate_count: usize,
    pub graph_vertex_count: usize,
    pub graph_edge_count: usize,
    pub span_count: usize,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkInstance {
    pub id: String,
    pub name: String,
    pub schema_id: String,
    pub network_kind: String,
    pub network_subtype: String,
    pub root_folder_id: String,
    pub root_entity_id: String,
    pub namespace: String,
    pub description: String,
    pub tags: Vec<String>,
    pub member_count: usize,
    pub relationship_count: usize,
    pub max_depth: usize,
    pub created_at: i64,
    pub updated_at: i64,
    pub group_id: String,
    pub scope_type: String,
    pub narrative_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkMembership {
    pub network_id: String,
    pub entity_id: EntityId,
    pub x: f64,
    pub y: f64,
    pub fixed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NetworkRelationship {
    pub network_id: String,
    pub source_entity_id: EntityId,
    pub target_entity_id: EntityId,
    pub relationship_id: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SavedNetworkView {
    pub instance: NetworkInstance,
    pub members: Vec<NetworkMembership>,
    pub relationships: Vec<NetworkRelationship>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EntityCard {
    pub entity_id: EntityId,
    pub card_id: String,
    pub name: String,
    pub color: String,
    pub icon: String,
    pub display_order: i32,
    pub is_collapsed: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FolderSchema {
    pub id: String,
    pub entity_kind: String,
    pub subtype: String,
    pub name: String,
    pub description: String,
    pub allowed_subfolders: String,
    pub allowed_note_types: String,
    pub is_vault_root: bool,
    pub container_only: bool,
    pub propagate_kind_to_children: bool,
    pub icon: String,
    pub is_system: bool,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Thread {
    pub id: ThreadId,
    pub world_id: String,
    pub narrative_id: String,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThreadMessage {
    pub id: String,
    pub thread_id: String,
    pub role: String,
    pub content: String,
    pub narrative_id: String,
    pub created_at: i64,
    pub updated_at: i64,
    pub is_streaming: bool,
    pub token_count: Option<i64>,
    pub is_observed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ReasoningEffort {
    Low,
    #[default]
    Medium,
    High,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatRuntimeConfig {
    pub model: String,
    pub temperature_milli: Option<u16>,
    pub max_tokens: Option<u32>,
    pub reasoning_enabled: bool,
    pub reasoning_effort: ReasoningEffort,
    pub reasoning_max_tokens: Option<u32>,
    pub include_reasoning: bool,
    pub om_enabled: bool,
    pub om_model: Option<String>,
    pub observe_threshold: Option<u32>,
    pub reflect_threshold: Option<u32>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmRecord {
    pub thread_id: String,
    pub observations: String,
    pub current_task: String,
    pub suggested_continuation: Option<String>,
    pub last_observed_at: i64,
    pub obs_token_count: i64,
    pub generation_num: i64,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmGeneration {
    pub id: String,
    pub thread_id: String,
    pub generation: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub input_text: String,
    pub output_text: String,
    pub created_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmConfig {
    pub enabled: bool,
    pub model: String,
    pub observe_threshold: u32,
    pub reflect_threshold: u32,
    pub graph_index_enabled: bool,
    pub index_raw_messages: bool,
    pub index_observations: bool,
    pub index_reflections: bool,
    pub reflector_tooling_enabled: bool,
    pub reflector_max_tool_rounds: u8,
}

impl Default for OmConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            model: String::new(),
            observe_threshold: 2_000,
            reflect_threshold: 4_000,
            graph_index_enabled: true,
            index_raw_messages: true,
            index_observations: true,
            index_reflections: true,
            reflector_tooling_enabled: true,
            reflector_max_tool_rounds: 2,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmObservationResult {
    pub observations: String,
    pub current_task: Option<String>,
    pub suggested_continuation: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmPendingAction {
    pub kind: String,
    pub thread_id: String,
    pub model: String,
    pub system_prompt: String,
    pub user_prompt: String,
    pub message_ids: Vec<String>,
    pub reflector_tooling_enabled: bool,
    pub reflector_max_tool_rounds: u8,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmReflectorToolSpec {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub parameters_json: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmReflectorToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmReflectorToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub result_json: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmReflectorMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub tool_calls: Vec<OmReflectorToolCall>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmReflectorModelRequest {
    pub session_id: String,
    pub thread_id: String,
    pub model: String,
    pub allow_tools: bool,
    #[serde(default)]
    pub tools: Vec<OmReflectorToolSpec>,
    #[serde(default)]
    pub messages: Vec<OmReflectorMessage>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmReflectorModelResponse {
    pub content: String,
    #[serde(default)]
    pub tool_calls: Vec<OmReflectorToolCall>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmReflectorSession {
    pub session_id: String,
    pub thread_id: String,
    pub model: String,
    pub tool_rounds_used: u8,
    pub max_tool_rounds: u8,
    pub final_request_sent: bool,
    pub awaiting_tool_results: bool,
    #[serde(default)]
    pub messages: Vec<OmReflectorMessage>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum OmReflectorStep {
    ModelRequest {
        request: OmReflectorModelRequest,
    },
    ToolCalls {
        session_id: String,
        thread_id: String,
        #[serde(default)]
        tool_calls: Vec<OmReflectorToolCall>,
    },
    Complete {
        session_id: String,
        thread_id: String,
        response: String,
    },
}

impl Default for OmReflectorStep {
    fn default() -> Self {
        Self::Complete {
            session_id: String::new(),
            thread_id: String::new(),
            response: String::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmGraphIndexRecord {
    pub thread_id: String,
    pub kind: String,
    pub source_key: String,
    pub document_id: String,
    pub entity_count: i64,
    pub edge_count: i64,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmIndexResult {
    pub kind: String,
    pub source_key: String,
    pub document_id: String,
    pub entity_count: i64,
    pub edge_count: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmMemorySearchHit {
    pub label: String,
    pub kind: String,
    pub document_id: String,
    pub source_kind: String,
    pub source_key: String,
    pub snippet: String,
    pub relation_summaries: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OmLostMemoryHit {
    pub entity_id: String,
    pub label: String,
    pub total_mentions: i64,
    pub source_keys: Vec<String>,
    pub relation_summaries: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ChatRunStatus {
    #[default]
    Queued,
    Gathering,
    Planning,
    ExecutingTools,
    AwaitingToolHost,
    AwaitingApproval,
    ReadyToAnswer,
    Streaming,
    Completed,
    Degraded,
    Failed,
    Cancelled,
}

impl ChatRunStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Gathering => "gathering",
            Self::Planning => "planning",
            Self::ExecutingTools => "executing_tools",
            Self::AwaitingToolHost => "awaiting_tool_host",
            Self::AwaitingApproval => "awaiting_approval",
            Self::ReadyToAnswer => "ready_to_answer",
            Self::Streaming => "streaming",
            Self::Completed => "completed",
            Self::Degraded => "degraded",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn from_str(value: &str) -> Self {
        match value {
            "queued" => Self::Queued,
            "gathering" => Self::Gathering,
            "planning" => Self::Planning,
            "executing_tools" => Self::ExecutingTools,
            "awaiting_tool_host" => Self::AwaitingToolHost,
            "awaiting_approval" => Self::AwaitingApproval,
            "ready_to_answer" => Self::ReadyToAnswer,
            "streaming" => Self::Streaming,
            "completed" => Self::Completed,
            "degraded" => Self::Degraded,
            "failed" => Self::Failed,
            "cancelled" => Self::Cancelled,
            _ => Self::Queued,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RunOptions {
    pub final_provider: String,
    pub final_model: String,
    pub planner_model: Option<String>,
    pub om_model: Option<String>,
    pub planner_enabled: bool,
    pub om_enabled: bool,
    pub workspace_enabled: bool,
    pub mutations_enabled: bool,
    pub deadline_ms: i64,
    pub mutation_policy: String,
    pub narrative_id: Option<String>,
    pub folder_id: Option<String>,
    pub scope_id: Option<String>,
    pub base_system_prompt: Option<String>,
    pub initial_external_context: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CapabilityProfile {
    pub om_enabled: bool,
    pub workspace_enabled: bool,
    pub planner_enabled: bool,
    pub go_tool_host: bool,
    pub ts_tool_host: bool,
    pub block_search: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceItem {
    pub id: String,
    pub source: String,
    pub title: Option<String>,
    pub content: String,
    pub score: Option<f64>,
    pub metadata: Option<BTreeMap<String, Value>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatRun {
    pub id: String,
    pub thread_id: ThreadId,
    pub user_prompt: String,
    pub status: ChatRunStatus,
    pub options: RunOptions,
    pub capabilities: CapabilityProfile,
    pub prepared_context: String,
    pub prepared_system_prompt: String,
    pub planner_messages_json: String,
    pub evidence_json: String,
    pub missing_capabilities_json: String,
    pub error: Option<String>,
    pub final_response: Option<String>,
    pub assistant_message_id: Option<String>,
    pub deadline_at: i64,
    pub completed_at: Option<i64>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatRunEvent {
    pub id: String,
    pub run_id: String,
    pub phase: String,
    pub kind: String,
    pub label: String,
    pub detail: Option<String>,
    pub status: Option<String>,
    pub payload: Option<String>,
    pub latency_ms: Option<i64>,
    pub created_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatToolCall {
    pub id: String,
    pub run_id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub host: String,
    pub class: String,
    pub status: String,
    pub arguments_json: String,
    pub result_json: Option<String>,
    pub error: Option<String>,
    pub idempotency_key: Option<String>,
    pub approval_id: Option<String>,
    pub started_at: Option<i64>,
    pub completed_at: Option<i64>,
    pub latency_ms: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolProposal {
    pub proposal_id: String,
    pub tool_name: String,
    pub affected_note_id: Option<String>,
    pub summary: String,
    pub diff_preview: Option<String>,
    pub expected_revision: Option<i64>,
    pub rollback_token: Option<String>,
    pub payload_json: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatApprovalRequest {
    pub id: String,
    pub run_id: String,
    pub tool_call_id: String,
    pub tool_name: String,
    pub status: String,
    pub affected_note_id: Option<String>,
    pub summary: String,
    pub diff_preview: Option<String>,
    pub expected_revision: Option<i64>,
    pub rollback_token: Option<String>,
    pub proposal_json: Option<String>,
    pub decision_json: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatWorkspaceArtifact {
    pub key: String,
    pub run_id: String,
    pub narrative_id: String,
    pub folder_id: String,
    pub kind: String,
    pub payload: Value,
    pub pinned: bool,
    pub produced_by: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatPlannerToolSpec {
    pub name: String,
    pub description: String,
    pub parameters_json: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatPlannerToolCall {
    pub id: String,
    pub name: String,
    pub arguments_json: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatPlannerMessage {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
    pub tool_calls: Vec<ChatPlannerToolCall>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatPlannerModelRequest {
    pub run_id: String,
    pub thread_id: String,
    pub model: String,
    pub allow_tools: bool,
    pub tools: Vec<ChatPlannerToolSpec>,
    pub messages: Vec<ChatPlannerMessage>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatPlannerModelResponse {
    pub content: String,
    pub tool_calls: Vec<ChatPlannerToolCall>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum ChatPlannerStep {
    ModelRequest {
        request: ChatPlannerModelRequest,
    },
    ToolCalls {
        run_id: String,
        tool_calls: Vec<ChatPlannerToolCall>,
    },
    Complete {
        run_id: String,
        response: String,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatRunSnapshot {
    pub run: ChatRun,
    pub events: Vec<ChatRunEvent>,
    pub tool_calls: Vec<ChatToolCall>,
    pub approvals: Vec<ChatApprovalRequest>,
    pub evidence: Vec<EvidenceItem>,
    pub missing_capabilities: Vec<String>,
    pub planner_step: Option<ChatPlannerStep>,
    pub artifacts: Vec<ChatWorkspaceArtifact>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolResultSubmission {
    pub call_id: Option<String>,
    pub tool_call_id: Option<String>,
    pub result_json: Option<String>,
    pub error: Option<String>,
    pub proposal: Option<ToolProposal>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoreCommandRequest {
    pub command: String,
    #[serde(default)]
    pub payload: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoreCommandResult {
    pub success: bool,
    pub payload: Option<Value>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStateRequest {
    pub session_id: SessionId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionStatsRequest {
    pub session_id: SessionId,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitRequest {
    pub session_id: SessionId,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommitResult {
    pub session_id: SessionId,
    pub commit_id: CommitId,
    pub revision: u64,
    pub committed_at: i64,
    pub relation_counts: Vec<RelationCount>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RebuildRequest {
    pub session_id: Option<SessionId>,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RebuildResult {
    pub rebuilt_at: i64,
    pub relation_counts: Vec<RelationCount>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)]
pub struct PacketHeader {
    pub ready: u32,
    pub kind: u32,
    pub request_id: u32,
    pub payload_len: u32,
}

impl PacketHeader {
    pub const BYTE_LEN: usize = 16;

    pub fn new(ready: u32, kind: PacketKind, request_id: u32, payload_len: u32) -> Self {
        Self {
            ready,
            kind: kind as u32,
            request_id,
            payload_len,
        }
    }

    pub fn packet_kind(&self) -> PacketKind {
        self.kind.into()
    }

    pub fn to_le_bytes(self) -> [u8; Self::BYTE_LEN] {
        let mut bytes = [0_u8; Self::BYTE_LEN];
        bytes[0..4].copy_from_slice(&self.ready.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.kind.to_le_bytes());
        bytes[8..12].copy_from_slice(&self.request_id.to_le_bytes());
        bytes[12..16].copy_from_slice(&self.payload_len.to_le_bytes());
        bytes
    }

    pub fn from_le_bytes(bytes: [u8; Self::BYTE_LEN]) -> Self {
        Self {
            ready: u32::from_le_bytes(bytes[0..4].try_into().expect("ready bytes")),
            kind: u32::from_le_bytes(bytes[4..8].try_into().expect("kind bytes")),
            request_id: u32::from_le_bytes(bytes[8..12].try_into().expect("request bytes")),
            payload_len: u32::from_le_bytes(bytes[12..16].try_into().expect("payload bytes")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet_header_round_trip() {
        let header = PacketHeader::new(1, PacketKind::QueryResult, 42, 4096);
        let bytes = header.to_le_bytes();
        let decoded = PacketHeader::from_le_bytes(bytes);

        assert_eq!(decoded.ready, 1);
        assert_eq!(decoded.packet_kind(), PacketKind::QueryResult);
        assert_eq!(decoded.request_id, 42);
        assert_eq!(decoded.payload_len, 4096);
    }

    #[test]
    fn runtime_config_serializes_camel_case() {
        let config = RuntimeConfig::default();
        let json = serde_json::to_string(&config).expect("serialize runtime config");

        assert!(json.contains("snapshotPolicy"));
        assert!(json.contains("featureFlags"));
    }

    #[test]
    fn scan_artifact_serializes_camel_case() {
        let artifact = ScanArtifact::default();
        let json = serde_json::to_string(&artifact).expect("serialize scan artifact");

        assert!(json.contains("resolverLinks"));
        assert!(json.contains("narrativeHits"));
    }

    #[test]
    fn packet_kind_round_trip_for_new_binary_kinds() {
        assert_eq!(PacketKind::from(21), PacketKind::GraphDeltaRequest);
        assert_eq!(PacketKind::from(22), PacketKind::GraphDeltaResult);
        assert_eq!(PacketKind::from(23), PacketKind::SessionStateRequest);
        assert_eq!(PacketKind::from(24), PacketKind::SessionStateResult);
        assert_eq!(PacketKind::from(25), PacketKind::SessionStatsRequest);
        assert_eq!(PacketKind::from(26), PacketKind::SessionStatsResult);
        assert_eq!(PacketKind::from(27), PacketKind::AnalyzeTextRequest);
        assert_eq!(PacketKind::from(28), PacketKind::AnalyzeTextResult);
        assert_eq!(PacketKind::from(29), PacketKind::QueryBinaryRequest);
        assert_eq!(PacketKind::from(30), PacketKind::AnalyzeTextBinaryRequest);
        assert_eq!(PacketKind::from(31), PacketKind::IngestBinaryRequest);
        assert_eq!(PacketKind::from(32), PacketKind::ScanBinaryRequest);
        assert_eq!(PacketKind::from(33), PacketKind::StructureBinaryRequest);
        assert_eq!(PacketKind::from(34), PacketKind::StoreCommandRequest);
        assert_eq!(PacketKind::from(35), PacketKind::StoreCommandResult);
    }
}
