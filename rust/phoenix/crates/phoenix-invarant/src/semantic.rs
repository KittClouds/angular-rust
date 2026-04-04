use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use phoenix_chunker::{build_chunks, ChunkerConfig};
use phoenix_graptor::BorrowedIngestDocument;
use phoenix_store_cozo::StoreError;
use phoenix_types::{
    Diagnostic, EntityId, EntityKind, EvidenceSpan, IngestDocumentSummary, MentionEntityRef,
    MentionSource, RelationCandidate, ResolverEntitySeed, ResolverLink, ScopeKey,
    SessionDocumentState, SessionId, StructureArtifact, TextRange,
};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    nlp::{
        CoreferenceProvider, InvarantNlpPipeline, NerProvider, NormalizedTextRecord,
        ProviderCoreferenceChain, ProviderMention, TextNormalizationProvider,
    },
    completed_stage, emit_ingest_progress, maybe_timeout_ingest_stage, AnalysisContext,
    InvarantConfig,
};

macro_rules! string_id {
    ($name:ident) => {
        #[derive(
            Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

string_id!(DocumentVersionId);
string_id!(SpanId);
string_id!(ChunkId);
string_id!(MentionId);
string_id!(ClaimId);
string_id!(EventId);
string_id!(RelationId);
string_id!(EvidenceId);

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OffsetRange {
    pub start: u32,
    pub end: u32,
}

impl From<TextRange> for OffsetRange {
    fn from(value: TextRange) -> Self {
        Self {
            start: value.start,
            end: value.end,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpanPath {
    pub value: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum StructuralKind {
    #[default]
    Root,
    Heading,
    Paragraph,
    ListItem,
    Quote,
    CodeBlock,
    TableRow,
    SpeakerTurn,
    Sentence,
    Link,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StructuralSpan {
    pub span_id: SpanId,
    pub kind: StructuralKind,
    pub range: OffsetRange,
    pub path: SpanPath,
    pub parent_span_id: Option<SpanId>,
    pub depth: u8,
    pub label: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScannedDocument {
    pub document_id: String,
    pub document_version_id: DocumentVersionId,
    pub spans: Vec<StructuralSpan>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SemanticChunkKind {
    #[default]
    Leaf,
    Window,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticChunk {
    pub chunk_id: ChunkId,
    pub kind: SemanticChunkKind,
    pub range: OffsetRange,
    pub span_ids: Vec<SpanId>,
    pub source_chunk_ids: Vec<ChunkId>,
    pub text_preview: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct MentionCandidate {
    pub mention_id: MentionId,
    pub surface: String,
    pub normalized_surface: String,
    pub kind: Option<EntityKind>,
    pub entity_ref: Option<MentionEntityRef>,
    pub source: Option<MentionSource>,
    pub range: OffsetRange,
    pub sentence_index: usize,
    pub chunk_id: Option<ChunkId>,
    pub span_id: Option<SpanId>,
    pub evidence_id: EvidenceId,
    pub confidence: f32,
    pub provider: String,
    pub provider_version: String,
    pub config_hash: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AliasCandidate {
    pub alias: String,
    pub normalized_alias: String,
    pub canonical_hint: String,
    pub normalized_canonical_hint: String,
    pub entity_ref: Option<MentionEntityRef>,
    pub range: OffsetRange,
    pub sentence_index: usize,
    pub coreference_chain_id: Option<String>,
    pub confidence: f32,
    pub provider: String,
    pub provider_version: String,
    pub config_hash: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyTermCandidate {
    pub term: String,
    pub range: OffsetRange,
    pub score: f32,
    pub provider: String,
    pub provider_version: String,
    pub config_hash: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TimeCandidate {
    pub surface: String,
    pub normalized: Option<String>,
    pub range: OffsetRange,
    pub sentence_index: usize,
    pub confidence: f32,
    pub provider: String,
    pub provider_version: String,
    pub config_hash: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RelationCueCandidate {
    pub relation_type: String,
    pub event_class: String,
    pub lemma: String,
    pub sentence_index: usize,
    pub evidence_ranges: Vec<OffsetRange>,
    pub confidence: f32,
    pub provider: String,
    pub provider_version: String,
    pub config_hash: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct NormalizedTextArtifact {
    pub normalized_text: String,
    pub folded_text: String,
    pub provider: String,
    pub provider_version: String,
    pub config_hash: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CoreferenceMentionArtifact {
    pub mention_id: Option<MentionId>,
    pub surface: String,
    pub canonical_surface: String,
    pub range: OffsetRange,
    pub sentence_index: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CoreferenceChainArtifact {
    pub chain_id: String,
    pub canonical: String,
    pub mentions: Vec<CoreferenceMentionArtifact>,
    pub evidence_ids: Vec<EvidenceId>,
    pub chunk_ids: Vec<ChunkId>,
    pub confidence: f32,
    pub provider: String,
    pub provider_version: String,
    pub config_hash: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct AnnotationBundle {
    pub normalized_text: NormalizedTextArtifact,
    pub mention_candidates: Vec<MentionCandidate>,
    pub alias_candidates: Vec<AliasCandidate>,
    pub key_term_candidates: Vec<KeyTermCandidate>,
    pub time_candidates: Vec<TimeCandidate>,
    pub coreference_chains: Vec<CoreferenceChainArtifact>,
    pub relation_cues: Vec<RelationCueCandidate>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ResolutionStatus {
    #[default]
    Unresolved,
    Proposed,
    Resolved,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ResolutionProvenance {
    pub providers: Vec<String>,
    pub reasons: Vec<String>,
    pub alias_hits: Vec<String>,
    pub seed_entity_ids: Vec<EntityId>,
    pub coreference_chain_ids: Vec<String>,
    pub relation_cue_count: usize,
    pub score_breakdown: BTreeMap<String, f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct UnresolvedMention {
    pub mention_id: MentionId,
    pub surface: String,
    pub range: OffsetRange,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ProposedEntityLink {
    pub mention_id: MentionId,
    pub entity_id: EntityId,
    pub reason: String,
    pub confidence: f32,
    pub provenance: ResolutionProvenance,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct ResolvedMention {
    pub mention_id: MentionId,
    pub entity_id: Option<EntityId>,
    pub status: ResolutionStatus,
    pub confidence: f32,
    pub evidence_id: EvidenceId,
    pub provenance: ResolutionProvenance,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CanonicalEntity {
    pub entity_id: EntityId,
    pub label: String,
    pub aliases: Vec<String>,
    pub kind: Option<EntityKind>,
    pub scope: ScopeKey,
    pub status: ResolutionStatus,
    pub mention_ids: Vec<MentionId>,
    pub evidence_ids: Vec<EvidenceId>,
    pub confidence: f32,
    pub provenance: ResolutionProvenance,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceAnchor {
    pub evidence_id: EvidenceId,
    pub document_id: String,
    pub chunk_id: Option<ChunkId>,
    pub span_path: Option<SpanPath>,
    pub range: OffsetRange,
    pub sentence_index: Option<usize>,
    pub label: String,
    pub kind: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Relation {
    pub relation_id: RelationId,
    pub relation_type: String,
    pub source_entity_id: Option<EntityId>,
    pub target_entity_id: Option<EntityId>,
    pub evidence_ids: Vec<EvidenceId>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Event {
    pub event_id: EventId,
    pub label: String,
    pub event_class: String,
    pub trigger_range: OffsetRange,
    pub participant_entity_ids: Vec<EntityId>,
    pub evidence_ids: Vec<EvidenceId>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Claim {
    pub claim_id: ClaimId,
    pub relation_type: String,
    pub event_class: String,
    pub subject_entity_id: Option<EntityId>,
    pub object_entity_id: Option<EntityId>,
    pub recipient_entity_id: Option<EntityId>,
    pub subject_text: Option<String>,
    pub object_text: Option<String>,
    pub recipient_text: Option<String>,
    pub evidence_ids: Vec<EvidenceId>,
    pub evidence_chunk_ids: Vec<ChunkId>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolutionBundle {
    pub unresolved_mentions: Vec<UnresolvedMention>,
    pub proposed_links: Vec<ProposedEntityLink>,
    pub resolved_mentions: Vec<ResolvedMention>,
    pub canonical_entities: Vec<CanonicalEntity>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticProjectionManifest {
    pub document_version_id: DocumentVersionId,
    pub assertion_version: String,
    pub canonical_entity_count: usize,
    pub claim_count: usize,
    pub event_count: usize,
    pub relation_count: usize,
    pub evidence_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticBundle {
    pub evidence_anchors: Vec<EvidenceAnchor>,
    pub claims: Vec<Claim>,
    pub events: Vec<Event>,
    pub relations: Vec<Relation>,
    pub projection: SemanticProjectionManifest,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentAnalysisStage {
    pub name: String,
    pub wall_ms: u64,
    pub counters: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentAnalysisProfile {
    pub document_id: String,
    pub input_bytes: usize,
    pub total_wall_ms: u64,
    pub stages: Vec<DocumentAnalysisStage>,
    pub counters: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSemanticBundle {
    pub context: AnalysisContext,
    pub document_version_id: DocumentVersionId,
    pub source_fingerprint: String,
    pub config_hash: String,
    pub scanned_document: ScannedDocument,
    pub leaf_chunks: Vec<SemanticChunk>,
    pub window_chunks: Vec<SemanticChunk>,
    pub annotation: AnnotationBundle,
    pub resolution: ResolutionBundle,
    pub semantics: SemanticBundle,
    pub session_document: SessionDocumentState,
    pub created_at: i64,
    #[serde(default)]
    pub analysis_profile: DocumentAnalysisProfile,
}

trait EntityResolutionProvider {
    fn resolve(
        &self,
        scope: &ScopeKey,
        annotation: &AnnotationBundle,
        resolver_seed: &[ResolverEntitySeed],
    ) -> ResolutionBundle;
}

#[derive(Clone, Debug, Default)]
struct ScoredEntityResolver;

#[derive(Clone, Debug)]
struct ResolutionSeed {
    entity_id: EntityId,
    label: String,
    normalized_label: String,
    aliases: BTreeSet<String>,
    kind: Option<EntityKind>,
}

#[derive(Clone, Debug)]
struct ObservedMentionCluster<'a> {
    mention_id: MentionId,
    surface: String,
    normalized_surface: String,
    kind: Option<EntityKind>,
    entity_ref: Option<MentionEntityRef>,
    source: Option<MentionSource>,
    range: OffsetRange,
    sentence_index: usize,
    evidence_id: EvidenceId,
    confidence: f32,
    providers: BTreeSet<String>,
    members: Vec<&'a MentionCandidate>,
}

#[derive(Clone, Debug)]
struct ScoredCandidate {
    entity_id: EntityId,
    label: String,
    kind: Option<EntityKind>,
    status: ResolutionStatus,
    score: f32,
    provenance: ResolutionProvenance,
}

struct ResolutionLookup<'a> {
    annotation: &'a AnnotationBundle,
    seeds: Vec<ResolutionSeed>,
    seeds_by_entity: FxHashMap<String, usize>,
    seeds_by_surface: FxHashMap<String, SmallVec<[usize; 4]>>,
    alias_candidates_by_surface: FxHashMap<String, SmallVec<[usize; 4]>>,
    alias_candidates_by_range: FxHashMap<(u32, u32), SmallVec<[usize; 4]>>,
    coreference_chains_by_range: FxHashMap<(u32, u32), SmallVec<[usize; 2]>>,
    normalized_chain_canonicals: Vec<String>,
    relation_cues_by_sentence: FxHashMap<usize, usize>,
}

const MAX_SURFACE_ALIAS_MATCHES: usize = 12;

impl<'a> ResolutionLookup<'a> {
    fn new(
        scope: &ScopeKey,
        annotation: &'a AnnotationBundle,
        resolver_seed: &[ResolverEntitySeed],
    ) -> Self {
        let seeds = build_resolution_seeds(scope, annotation, resolver_seed);
        let mut seeds_by_entity = FxHashMap::default();
        let mut seeds_by_surface = FxHashMap::<String, SmallVec<[usize; 4]>>::default();
        for (index, seed) in seeds.iter().enumerate() {
            seeds_by_entity.insert(seed.entity_id.0.clone(), index);
            seeds_by_surface
                .entry(seed.normalized_label.clone())
                .or_default()
                .push(index);
            for alias in &seed.aliases {
                seeds_by_surface.entry(alias.clone()).or_default().push(index);
            }
        }

        let mut alias_candidates_by_surface =
            FxHashMap::<String, SmallVec<[usize; 4]>>::default();
        let mut alias_candidates_by_range =
            FxHashMap::<(u32, u32), SmallVec<[usize; 4]>>::default();
        for (index, alias) in annotation.alias_candidates.iter().enumerate() {
            let surface_bucket = alias_candidates_by_surface
                .entry(alias.normalized_alias.clone())
                .or_default();
            if surface_bucket.len() < MAX_SURFACE_ALIAS_MATCHES {
                surface_bucket.push(index);
            }
            alias_candidates_by_range
                .entry((alias.range.start, alias.range.end))
                .or_default()
                .push(index);
        }

        let mut coreference_chains_by_range =
            FxHashMap::<(u32, u32), SmallVec<[usize; 2]>>::default();
        let mut normalized_chain_canonicals = Vec::with_capacity(annotation.coreference_chains.len());
        for (chain_index, chain) in annotation.coreference_chains.iter().enumerate() {
            normalized_chain_canonicals.push(normalize_surface(&chain.canonical));
            for mention in &chain.mentions {
                coreference_chains_by_range
                    .entry((mention.range.start, mention.range.end))
                    .or_default()
                    .push(chain_index);
            }
        }

        let mut relation_cues_by_sentence = FxHashMap::<usize, usize>::default();
        for cue in &annotation.relation_cues {
            *relation_cues_by_sentence.entry(cue.sentence_index).or_default() += 1;
        }

        Self {
            annotation,
            seeds,
            seeds_by_entity,
            seeds_by_surface,
            alias_candidates_by_surface,
            alias_candidates_by_range,
            coreference_chains_by_range,
            normalized_chain_canonicals,
            relation_cues_by_sentence,
        }
    }

    fn seed_for_entity(&self, entity_id: &EntityId) -> Option<&ResolutionSeed> {
        self.seeds_by_entity
            .get(entity_id.0.as_str())
            .and_then(|index| self.seeds.get(*index))
    }

    fn relation_cue_count(&self, sentence_index: usize) -> usize {
        self.relation_cues_by_sentence
            .get(&sentence_index)
            .copied()
            .unwrap_or_default()
    }

    fn matching_alias_indices(
        &self,
        cluster: &ObservedMentionCluster<'_>,
    ) -> SmallVec<[usize; 8]> {
        let mut indices = SmallVec::<[usize; 8]>::new();
        if let Some(by_range) = self
            .alias_candidates_by_range
            .get(&(cluster.range.start, cluster.range.end))
        {
            for &index in by_range {
                indices.push(index);
            }
        }
        if indices.is_empty() {
            if let Some(by_surface) = self
                .alias_candidates_by_surface
                .get(cluster.normalized_surface.as_str())
            {
                indices.extend(by_surface.iter().copied());
            }
        }
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    fn matching_chain_indices(
        &self,
        cluster: &ObservedMentionCluster<'_>,
    ) -> SmallVec<[usize; 4]> {
        let mut indices = SmallVec::<[usize; 4]>::new();
        if let Some(matches) = self
            .coreference_chains_by_range
            .get(&(cluster.range.start, cluster.range.end))
        {
            indices.extend(matches.iter().copied());
        }
        indices.sort_unstable();
        indices
    }

    fn candidate_seed_indices(
        &self,
        scope: &ScopeKey,
        cluster: &ObservedMentionCluster<'_>,
        matching_aliases: &[usize],
        matching_chains: &[usize],
    ) -> SmallVec<[usize; 8]> {
        let mut seen = FxHashSet::default();
        let mut indices = SmallVec::<[usize; 8]>::new();
        let mut push_seed = |index: usize| {
            if seen.insert(index) {
                indices.push(index);
            }
        };

        if let Some(MentionEntityRef::Known(entity_id)) = cluster.entity_ref.as_ref() {
            if let Some(index) = self.seeds_by_entity.get(entity_id.0.as_str()) {
                push_seed(*index);
            }
        }

        if let Some(matches) = self
            .seeds_by_surface
            .get(cluster.normalized_surface.as_str())
        {
            for &index in matches {
                push_seed(index);
            }
        }

        for &alias_index in matching_aliases {
            let alias = &self.annotation.alias_candidates[alias_index];
            if let Some(entity_ref) = alias.entity_ref.as_ref() {
                let entity_id = match entity_ref {
                    MentionEntityRef::Known(entity_id) => entity_id.clone(),
                    MentionEntityRef::Speculative(key) => proposal_entity_id(
                        scope,
                        &alias.normalized_canonical_hint,
                        None,
                        Some(key.as_str()),
                    ),
                };
                if let Some(index) = self.seeds_by_entity.get(entity_id.0.as_str()) {
                    push_seed(*index);
                }
            }
            if let Some(matches) = self
                .seeds_by_surface
                .get(alias.normalized_canonical_hint.as_str())
            {
                for &index in matches {
                    push_seed(index);
                }
            }
        }

        for &chain_index in matching_chains {
            if let Some(matches) = self
                .seeds_by_surface
                .get(self.normalized_chain_canonicals[chain_index].as_str())
            {
                for &index in matches {
                    push_seed(index);
                }
            }
        }

        indices.sort_unstable();
        indices
    }
}

pub fn analyze_document(
    document: &BorrowedIngestDocument<'_>,
    session_id: Option<&SessionId>,
    context: AnalysisContext,
    config: &InvarantConfig,
    config_hash: &str,
    scan: &phoenix_types::ScanArtifact,
    structure: &StructureArtifact,
    summary: Option<&IngestDocumentSummary>,
    resolver_seed: &[ResolverEntitySeed],
) -> Result<DocumentSemanticBundle, StoreError> {
    let total_started = Instant::now();
    let mut analysis_stages = Vec::new();
    let document_version_id = DocumentVersionId(crate::stable_hex(
        "invarant_document_version",
        &[
            document.document_id.0.as_str(),
            document.title,
            document.text,
            config_hash,
        ],
    ));
    let structural_started = Instant::now();
    let scanned_document = build_structural_document(document, &document_version_id, scan);
    analysis_stages.push(completed_stage(
        "structural_document",
        structural_started,
        BTreeMap::from([("spans".to_owned(), scanned_document.spans.len())]),
    )?);
    let chunking_started = Instant::now();
    let (leaf_chunks, window_chunks) = build_semantic_chunks(
        document.text,
        &scanned_document,
        &document_version_id,
        config,
    );
    analysis_stages.push(completed_stage(
        "semantic_chunks",
        chunking_started,
        BTreeMap::from([
            ("leafChunks".to_owned(), leaf_chunks.len()),
            ("windowChunks".to_owned(), window_chunks.len()),
        ]),
    )?);
    let annotation_started = Instant::now();
    let annotation = build_annotation_bundle(
        document.text,
        config_hash,
        &scanned_document,
        &leaf_chunks,
        scan,
        structure,
        resolver_seed,
    );
    analysis_stages.push(completed_stage(
        "annotation_bundle",
        annotation_started,
        BTreeMap::from([
            (
                "mentionCandidates".to_owned(),
                annotation.mention_candidates.len(),
            ),
            ("aliasCandidates".to_owned(), annotation.alias_candidates.len()),
            ("timeCandidates".to_owned(), annotation.time_candidates.len()),
            ("relationCues".to_owned(), annotation.relation_cues.len()),
            (
                "coreferenceChains".to_owned(),
                annotation.coreference_chains.len(),
            ),
        ]),
    )?);
    if let Some(stage) = analysis_stages.last() {
        maybe_timeout_ingest_stage(document.document_id.0.as_str(), stage)?;
    }
    let resolution_started = Instant::now();
    let resolution = ScoredEntityResolver.resolve(&context.scope, &annotation, resolver_seed);
    analysis_stages.push(completed_stage(
        "entity_resolution",
        resolution_started,
        BTreeMap::from([
            (
                "unresolvedMentions".to_owned(),
                resolution.unresolved_mentions.len(),
            ),
            ("proposedLinks".to_owned(), resolution.proposed_links.len()),
            (
                "resolvedMentions".to_owned(),
                resolution.resolved_mentions.len(),
            ),
            (
                "canonicalEntities".to_owned(),
                resolution.canonical_entities.len(),
            ),
        ]),
    )?);
    if let Some(stage) = analysis_stages.last() {
        maybe_timeout_ingest_stage(document.document_id.0.as_str(), stage)?;
    }
    let semantic_started = Instant::now();
    let semantics = build_semantic_bundle(
        document,
        document.text,
        &document_version_id,
        &scanned_document,
        &leaf_chunks,
        &annotation,
        &resolution,
        structure,
    );
    analysis_stages.push(completed_stage(
        "semantic_projection",
        semantic_started,
        BTreeMap::from([
            (
                "evidenceAnchors".to_owned(),
                semantics.evidence_anchors.len(),
            ),
            ("claims".to_owned(), semantics.claims.len()),
            ("events".to_owned(), semantics.events.len()),
            ("relations".to_owned(), semantics.relations.len()),
        ]),
    )?);
    if let Some(stage) = analysis_stages.last() {
        maybe_timeout_ingest_stage(document.document_id.0.as_str(), stage)?;
    }
    let session_started = Instant::now();
    let session_document = build_session_document_state(
        document,
        session_id,
        summary,
        &scanned_document,
        &annotation,
        &resolution,
        &semantics,
    );
    analysis_stages.push(completed_stage(
        "session_document_state",
        session_started,
        BTreeMap::from([
            (
                "sessionEntityCount".to_owned(),
                session_document.entity_count,
            ),
            (
                "sessionDiscoveryCount".to_owned(),
                session_document.discovery_count,
            ),
            (
                "sessionLeafCount".to_owned(),
                session_document.leaf_count,
            ),
        ]),
    )?);
    if let Some(stage) = analysis_stages.last() {
        maybe_timeout_ingest_stage(document.document_id.0.as_str(), stage)?;
    }
    let analysis_profile = DocumentAnalysisProfile {
        document_id: document.document_id.0.clone(),
        input_bytes: document.text.len(),
        total_wall_ms: total_started.elapsed().as_millis() as u64,
        stages: analysis_stages,
        counters: BTreeMap::from([
            ("structuralSpans".to_owned(), scanned_document.spans.len()),
            ("leafChunks".to_owned(), leaf_chunks.len()),
            ("windowChunks".to_owned(), window_chunks.len()),
            (
                "mentionCandidates".to_owned(),
                annotation.mention_candidates.len(),
            ),
            (
                "canonicalEntities".to_owned(),
                resolution.canonical_entities.len(),
            ),
            ("claims".to_owned(), semantics.claims.len()),
            ("events".to_owned(), semantics.events.len()),
            ("relations".to_owned(), semantics.relations.len()),
        ]),
    };
    maybe_timeout_ingest_stage(
        document.document_id.0.as_str(),
        &DocumentAnalysisStage {
            name: "document_total".to_owned(),
            wall_ms: analysis_profile.total_wall_ms,
            counters: analysis_profile.counters.clone(),
        },
    )?;
    Ok(DocumentSemanticBundle {
        context,
        document_version_id,
        source_fingerprint: crate::stable_hex(
            "invarant_source",
            &[
                document.document_id.0.as_str(),
                document.title,
                document.text,
            ],
        ),
        config_hash: config_hash.to_owned(),
        scanned_document,
        leaf_chunks,
        window_chunks,
        annotation,
        resolution,
        semantics,
        session_document,
        created_at: crate::now_ms(),
        analysis_profile,
    })
}

fn build_structural_document(
    document: &BorrowedIngestDocument<'_>,
    document_version_id: &DocumentVersionId,
    scan: &phoenix_types::ScanArtifact,
) -> ScannedDocument {
    let blocks = detect_blocks(document.text);
    let mut spans = Vec::with_capacity(1 + blocks.len() + scan.sentences.len());
    let root_id = SpanId(format!(
        "span::{}",
        crate::stable_hex("invarant_root_span", &[document_version_id.0.as_str()])
    ));
    spans.push(StructuralSpan {
        span_id: root_id.clone(),
        kind: StructuralKind::Root,
        range: OffsetRange {
            start: 0,
            end: document.text.len().min(u32::MAX as usize) as u32,
        },
        path: SpanPath {
            value: "root".to_owned(),
        },
        parent_span_id: None,
        depth: 0,
        label: Some(document.title.to_owned()),
    });

    let mut block_spans = Vec::with_capacity(blocks.len());
    for (block_index, block) in blocks.into_iter().enumerate() {
        block_spans.push(StructuralSpan {
            span_id: SpanId(format!(
                "span::{}",
                crate::stable_hex(
                    "invarant_block_span",
                    &[
                        document_version_id.0.as_str(),
                        block.kind.as_str(),
                        &block.range.start.to_string(),
                        &block.range.end.to_string(),
                    ],
                )
            )),
            kind: match block.kind.as_str() {
                "heading" => StructuralKind::Heading,
                "listItem" => StructuralKind::ListItem,
                "quote" => StructuralKind::Quote,
                "codeBlock" => StructuralKind::CodeBlock,
                "tableRow" => StructuralKind::TableRow,
                "speakerTurn" => StructuralKind::SpeakerTurn,
                _ => StructuralKind::Paragraph,
            },
            range: block.range,
            path: SpanPath {
                value: format!("root/{block_index}"),
            },
            parent_span_id: Some(root_id.clone()),
            depth: 1,
            label: block.label,
        });
    }
    spans.extend(block_spans.iter().cloned());

    let mut block_cursor = 0usize;
    for sentence in &scan.sentences {
        while block_cursor < block_spans.len()
            && block_spans[block_cursor].range.end <= sentence.range.start
        {
            block_cursor += 1;
        }
        let parent = block_spans.get(block_cursor).filter(|span| {
            span.range.start <= sentence.range.start && span.range.end >= sentence.range.end
        });
        spans.push(StructuralSpan {
            span_id: SpanId(format!(
                "span::{}",
                crate::stable_hex(
                    "invarant_sentence_span",
                    &[
                        document_version_id.0.as_str(),
                        &sentence.index.to_string(),
                        &sentence.range.start.to_string(),
                        &sentence.range.end.to_string(),
                    ],
                )
            )),
            kind: StructuralKind::Sentence,
            range: sentence.range.into(),
            path: SpanPath {
                value: parent
                    .map(|span| format!("{}/sentence:{}", span.path.value, sentence.index))
                    .unwrap_or_else(|| format!("root/sentence:{}", sentence.index)),
            },
            parent_span_id: parent.map(|span| span.span_id.clone()),
            depth: 2,
            label: None,
        });
    }

    let mut diagnostics = Vec::new();
    diagnostics.push(Diagnostic {
        code: "PX_INVARANT_STRUCTURE_MAP".to_owned(),
        message: format!(
            "Invarant preserved {} structural spans before semantic chunking.",
            spans.len()
        ),
    });

    ScannedDocument {
        document_id: document.document_id.0.clone(),
        document_version_id: document_version_id.clone(),
        spans,
        diagnostics,
    }
}

fn build_semantic_chunks(
    text: &str,
    scanned_document: &ScannedDocument,
    document_version_id: &DocumentVersionId,
    config: &InvarantConfig,
) -> (Vec<SemanticChunk>, Vec<SemanticChunk>) {
    let chunker_config = ChunkerConfig {
        chunk_size: config.chunk_size,
        overlap: config.overlap,
    };
    let mut leaf_chunks = Vec::new();

    for span in scanned_document.spans.iter().filter(|span| {
        matches!(
            span.kind,
            StructuralKind::Heading
                | StructuralKind::Paragraph
                | StructuralKind::ListItem
                | StructuralKind::Quote
                | StructuralKind::CodeBlock
                | StructuralKind::TableRow
                | StructuralKind::SpeakerTurn
        )
    }) {
        let (start, end) = normalized_bounds(
            text.len(),
            span.range.start as usize,
            span.range.end as usize,
        );
        if start >= end {
            continue;
        }
        let mut local_chunks = build_chunks(&text[start..end], &chunker_config);
        if local_chunks.is_empty() {
            local_chunks.push(phoenix_chunker::Chunk {
                start: 0,
                end: end - start,
            });
        }
        for local_chunk in local_chunks {
            let global_start = start + local_chunk.start;
            let global_end = start + local_chunk.end;
            leaf_chunks.push(SemanticChunk {
                chunk_id: ChunkId(format!(
                    "leaf::{}",
                    crate::stable_hex(
                        "invarant_leaf_chunk",
                        &[
                            document_version_id.0.as_str(),
                            &global_start.to_string(),
                            &global_end.to_string(),
                        ],
                    )
                )),
                kind: SemanticChunkKind::Leaf,
                range: OffsetRange {
                    start: global_start.min(u32::MAX as usize) as u32,
                    end: global_end.min(u32::MAX as usize) as u32,
                },
                span_ids: vec![span.span_id.clone()],
                source_chunk_ids: Vec::new(),
                text_preview: slice_preview(text, global_start, global_end, 180),
            });
        }
    }

    if leaf_chunks.is_empty() && !text.trim().is_empty() {
        leaf_chunks.push(SemanticChunk {
            chunk_id: ChunkId(format!(
                "leaf::{}",
                crate::stable_hex(
                    "invarant_leaf_chunk",
                    &[document_version_id.0.as_str(), "0", &text.len().to_string()],
                )
            )),
            kind: SemanticChunkKind::Leaf,
            range: OffsetRange {
                start: 0,
                end: text.len().min(u32::MAX as usize) as u32,
            },
            span_ids: vec![scanned_document.spans[0].span_id.clone()],
            source_chunk_ids: Vec::new(),
            text_preview: slice_preview(text, 0, text.len(), 180),
        });
    }

    let window_chunks = leaf_chunks
        .windows(2)
        .map(|window| {
            let start = window[0].range.start.min(window[1].range.start) as usize;
            let end = window[0].range.end.max(window[1].range.end) as usize;
            SemanticChunk {
                chunk_id: ChunkId(format!(
                    "window::{}",
                    crate::stable_hex(
                        "invarant_window_chunk",
                        &[
                            document_version_id.0.as_str(),
                            window[0].chunk_id.0.as_str(),
                            window[1].chunk_id.0.as_str(),
                        ],
                    )
                )),
                kind: SemanticChunkKind::Window,
                range: OffsetRange {
                    start: start.min(u32::MAX as usize) as u32,
                    end: end.min(u32::MAX as usize) as u32,
                },
                span_ids: window
                    .iter()
                    .flat_map(|chunk| chunk.span_ids.iter().cloned())
                    .collect(),
                source_chunk_ids: window.iter().map(|chunk| chunk.chunk_id.clone()).collect(),
                text_preview: slice_preview(text, start, end, 220),
            }
        })
        .collect();

    (leaf_chunks, window_chunks)
}

fn build_annotation_bundle(
    text: &str,
    config_hash: &str,
    scanned_document: &ScannedDocument,
    leaf_chunks: &[SemanticChunk],
    scan: &phoenix_types::ScanArtifact,
    structure: &StructureArtifact,
    resolver_seed: &[ResolverEntitySeed],
) -> AnnotationBundle {
    let annotation_started = Instant::now();
    let nlp = InvarantNlpPipeline::default();
    let span_index = SpanIndex::new(scanned_document);
    let chunk_index = ChunkIndex::new(leaf_chunks);
    let normalized_text = normalized_artifact(
        config_hash,
        nlp.normalizer.normalize(text),
    );
    emit_ingest_progress(format!(
        "annotation_substage=normalize wall_ms={}",
        annotation_started.elapsed().as_millis()
    ));

    let mention_projection_started = Instant::now();
    let mut mention_candidates = scan
        .mentions
        .iter()
        .enumerate()
        .map(|(index, mention)| MentionCandidate {
            mention_id: MentionId(format!(
                "mention::{}",
                crate::stable_hex(
                    "invarant_mention",
                    &[
                        scanned_document.document_version_id.0.as_str(),
                        &index.to_string(),
                        &mention.range.start.to_string(),
                        &mention.range.end.to_string(),
                    ],
                )
            )),
            surface: slice_preview(
                text,
                mention.range.start as usize,
                mention.range.end as usize,
                160,
            ),
            normalized_surface: normalize_surface(&mention.surface),
            kind: mention.kind.clone(),
            entity_ref: mention.entity_ref.clone(),
            source: mention.source.clone(),
            range: mention.range.into(),
            sentence_index: mention.sentence_index,
            chunk_id: chunk_index.chunk_id_for(mention.range.start, mention.range.end),
            span_id: span_index.span_id_for(mention.range.start, mention.range.end),
            evidence_id: EvidenceId(format!(
                "evidence::{}",
                crate::stable_hex(
                    "invarant_mention_evidence",
                    &[
                        scanned_document.document_version_id.0.as_str(),
                        &index.to_string(),
                        &mention.range.start.to_string(),
                        &mention.range.end.to_string(),
                    ],
                )
            )),
            confidence: mention.confidence,
            provider: "phoenix-scanner".to_owned(),
            provider_version: "v1".to_owned(),
            config_hash: config_hash.to_owned(),
        })
        .collect::<Vec<_>>();
    emit_ingest_progress(format!(
        "annotation_substage=observed_mentions wall_ms={} counters={{\"mentions\":{}}}",
        mention_projection_started.elapsed().as_millis(),
        mention_candidates.len()
    ));

    let observed_mentions = scan
        .mentions
        .iter()
        .map(|mention| (mention.range, mention.kind.clone()))
        .collect::<Vec<_>>();
    let provider_ner_started = Instant::now();
    let provider_mentions = nlp.ner.extract_mentions(
        text,
        &scan.sentences,
        &scan.tokens,
        &observed_mentions,
        resolver_seed,
    );
    for (index, mention) in provider_mentions.into_iter().enumerate() {
        mention_candidates.push(provider_mention_candidate(
            scanned_document,
            &span_index,
            &chunk_index,
            config_hash,
            index,
            mention,
        ));
    }
    emit_ingest_progress(format!(
        "annotation_substage=provider_mentions wall_ms={} counters={{\"mentions\":{}}}",
        provider_ner_started.elapsed().as_millis(),
        mention_candidates.len()
    ));

    let alias_started = Instant::now();
    let mut alias_candidates = scan
        .resolver_links
        .iter()
        .filter_map(|link| alias_candidate_from_link(text, config_hash, link))
        .collect::<Vec<_>>();
    emit_ingest_progress(format!(
        "annotation_substage=alias_candidates wall_ms={} counters={{\"aliases\":{}}}",
        alias_started.elapsed().as_millis(),
        alias_candidates.len()
    ));

    let key_term_started = Instant::now();
    let mut seen_terms = BTreeSet::new();
    let mut key_term_candidates = Vec::new();
    for span in scanned_document.spans.iter().filter(|span| {
        matches!(
            span.kind,
            StructuralKind::Heading | StructuralKind::Paragraph
        )
    }) {
        let surface = slice_preview(text, span.range.start as usize, span.range.end as usize, 80);
        let normalized = normalize_surface(&surface);
        if normalized.len() >= 3 && seen_terms.insert(normalized) {
            key_term_candidates.push(KeyTermCandidate {
                term: surface,
                range: span.range.clone(),
                score: if matches!(span.kind, StructuralKind::Heading) {
                    1.0
                } else {
                    0.6
                },
                provider: "invarant-structure".to_owned(),
                provider_version: "v1".to_owned(),
                config_hash: config_hash.to_owned(),
            });
        }
    }
    emit_ingest_progress(format!(
        "annotation_substage=key_terms wall_ms={} counters={{\"keyTerms\":{}}}",
        key_term_started.elapsed().as_millis(),
        key_term_candidates.len()
    ));

    let time_started = Instant::now();
    let mut time_candidates = nlp
        .ner
        .extract_time_candidates(text, &scan.sentences)
        .into_iter()
        .map(|candidate| TimeCandidate {
            surface: candidate.surface,
            normalized: candidate.normalized,
            range: candidate.range.into(),
            sentence_index: candidate.sentence_index,
            confidence: candidate.confidence,
            provider: candidate.provider,
            provider_version: candidate.provider_version,
            config_hash: config_hash.to_owned(),
        })
        .collect::<Vec<_>>();
    if time_candidates.is_empty() {
        for sentence in &scan.sentences {
            let surface = slice_preview(
                text,
                sentence.range.start as usize,
                sentence.range.end as usize,
                240,
            );
            let normalized = normalize_surface(&surface);
            if normalized.contains("before ")
                || normalized.contains("after ")
                || normalized.contains("during ")
                || normalized.contains("today")
                || normalized.contains("tomorrow")
                || normalized.contains("yesterday")
                || surface
                    .split(|ch: char| !ch.is_ascii_alphanumeric())
                    .any(|part| part.len() == 4 && part.chars().all(|ch| ch.is_ascii_digit()))
            {
                time_candidates.push(TimeCandidate {
                    surface,
                    normalized: Some(normalized),
                    range: sentence.range.into(),
                    sentence_index: sentence.index,
                    confidence: 0.7,
                    provider: "invarant-time".to_owned(),
                    provider_version: "v1".to_owned(),
                    config_hash: config_hash.to_owned(),
                });
            }
        }
    }
    emit_ingest_progress(format!(
        "annotation_substage=time_candidates wall_ms={} counters={{\"times\":{}}}",
        time_started.elapsed().as_millis(),
        time_candidates.len()
    ));

    let coref_started = Instant::now();
    let provider_chains = nlp
        .coreference
        .resolve(text, &scan.sentences, resolver_seed);
    emit_ingest_progress(format!(
        "annotation_substage=provider_coref wall_ms={} counters={{\"chains\":{}}}",
        coref_started.elapsed().as_millis(),
        provider_chains.len()
    ));
    let coref_projection_started = Instant::now();
    add_coreference_mentions(
        &mut mention_candidates,
        scanned_document,
        &span_index,
        &chunk_index,
        resolver_seed,
        config_hash,
        &provider_chains,
    );
    let coreference_chains = coreference_artifacts(
        config_hash,
        &mention_candidates,
        &provider_chains,
    );
    alias_candidates.extend(coreference_alias_candidates(
        config_hash,
        &coreference_chains,
    ));
    emit_ingest_progress(format!(
        "annotation_substage=coref_projection wall_ms={} counters={{\"mentions\":{},\"chains\":{},\"aliases\":{}}}",
        coref_projection_started.elapsed().as_millis(),
        mention_candidates.len(),
        coreference_chains.len(),
        alias_candidates.len()
    ));

    let relation_cues_started = Instant::now();
    let relation_cues = structure
        .relations
        .iter()
        .map(|relation| RelationCueCandidate {
            relation_type: relation.relation_type.clone(),
            event_class: relation.event_class.clone(),
            lemma: relation.lemma.clone(),
            sentence_index: relation.sentence_index,
            evidence_ranges: relation
                .evidence
                .iter()
                .map(|evidence| evidence.range.into())
                .collect(),
            confidence: relation_confidence(relation),
            provider: "phoenix-structure".to_owned(),
            provider_version: "v1".to_owned(),
            config_hash: config_hash.to_owned(),
        })
        .collect::<Vec<_>>();
    emit_ingest_progress(format!(
        "annotation_substage=relation_cues wall_ms={} counters={{\"relations\":{}}}",
        relation_cues_started.elapsed().as_millis(),
        relation_cues.len()
    ));

    let mut diagnostics = scanned_document.diagnostics.clone();
    diagnostics.push(Diagnostic {
        code: "PX_INVARANT_ANNOTATION".to_owned(),
        message: format!(
            "Invarant normalized {} mention candidates, {} alias candidates, {} key terms, {} temporal cues, {} coreference chains, and {} relation cues.",
            mention_candidates.len(),
            alias_candidates.len(),
            key_term_candidates.len(),
            time_candidates.len(),
            coreference_chains.len(),
            relation_cues.len(),
        ),
    });

    AnnotationBundle {
        normalized_text,
        mention_candidates,
        alias_candidates,
        key_term_candidates,
        time_candidates,
        coreference_chains,
        relation_cues,
        diagnostics,
    }
}

impl EntityResolutionProvider for ScoredEntityResolver {
    fn resolve(
        &self,
        scope: &ScopeKey,
        annotation: &AnnotationBundle,
        resolver_seed: &[ResolverEntitySeed],
    ) -> ResolutionBundle {
        let lookup_started = Instant::now();
        let lookup = ResolutionLookup::new(scope, annotation, resolver_seed);
        emit_ingest_progress(format!(
            "resolution_substage=lookup wall_ms={} counters={{\"seeds\":{},\"seedSurfaces\":{},\"aliasSurfaceKeys\":{},\"corefRanges\":{}}}",
            lookup_started.elapsed().as_millis(),
            lookup.seeds.len(),
            lookup.seeds_by_surface.len(),
            lookup.alias_candidates_by_surface.len(),
            lookup.coreference_chains_by_range.len()
        ));

        let cluster_started = Instant::now();
        let mention_clusters = cluster_mentions(annotation);
        emit_ingest_progress(format!(
            "resolution_substage=cluster_mentions wall_ms={} counters={{\"clusters\":{}}}",
            cluster_started.elapsed().as_millis(),
            mention_clusters.len()
        ));

        let mut surface_counts = FxHashMap::<String, usize>::default();
        for cluster in &mention_clusters {
            *surface_counts
                .entry(cluster.normalized_surface.clone())
                .or_default() += 1;
        }

        let mut unresolved_mentions = Vec::new();
        let mut proposed_links = Vec::new();
        let mut resolved_mentions = Vec::new();
        let mut grouped_mentions =
            BTreeMap::<String, (ScoredCandidate, Vec<ObservedMentionCluster<'_>>)>::new();

        let scoring_started = Instant::now();
        let mut scored_cluster_count = 0usize;
        for cluster in mention_clusters {
            let best = choose_resolution_candidate(
                scope,
                &cluster,
                &lookup,
                surface_counts
                    .get(cluster.normalized_surface.as_str())
                    .copied()
                    .unwrap_or_default(),
            );

            match best {
                Some(best) => {
                    if matches!(best.status, ResolutionStatus::Proposed) {
                        proposed_links.push(ProposedEntityLink {
                            mention_id: cluster.mention_id.clone(),
                            entity_id: best.entity_id.clone(),
                            reason: best
                                .provenance
                                .reasons
                                .first()
                                .cloned()
                                .unwrap_or_else(|| "scored_resolution".to_owned()),
                            confidence: best.score,
                            provenance: best.provenance.clone(),
                        });
                    }
                    resolved_mentions.push(ResolvedMention {
                        mention_id: cluster.mention_id.clone(),
                        entity_id: Some(best.entity_id.clone()),
                        status: best.status.clone(),
                        confidence: best.score,
                        evidence_id: cluster.evidence_id.clone(),
                        provenance: best.provenance.clone(),
                    });
                    grouped_mentions
                        .entry(best.entity_id.0.clone())
                        .or_insert_with(|| (best.clone(), Vec::new()))
                        .1
                        .push(cluster);
                }
                None => {
                    unresolved_mentions.push(UnresolvedMention {
                        mention_id: cluster.mention_id.clone(),
                        surface: cluster.surface.clone(),
                        range: cluster.range.clone(),
                        confidence: cluster.confidence,
                    });
                    resolved_mentions.push(ResolvedMention {
                        mention_id: cluster.mention_id,
                        entity_id: None,
                        status: ResolutionStatus::Unresolved,
                        confidence: cluster.confidence,
                        evidence_id: cluster.evidence_id,
                        provenance: ResolutionProvenance {
                            providers: cluster.providers.into_iter().collect(),
                            reasons: vec!["insufficient_resolution_evidence".to_owned()],
                            relation_cue_count: lookup.relation_cue_count(cluster.sentence_index),
                            ..ResolutionProvenance::default()
                        },
                    });
                }
            }
            scored_cluster_count += 1;
            if scored_cluster_count % 5_000 == 0 {
                emit_ingest_progress(format!(
                    "resolution_substage=score_clusters_progress wall_ms={} counters={{\"processed\":{},\"resolved\":{},\"proposed\":{},\"unresolved\":{}}}",
                    scoring_started.elapsed().as_millis(),
                    scored_cluster_count,
                    resolved_mentions
                        .iter()
                        .filter(|mention| mention.entity_id.is_some())
                        .count(),
                    proposed_links.len(),
                    unresolved_mentions.len()
                ));
            }
        }
        emit_ingest_progress(format!(
            "resolution_substage=score_clusters wall_ms={} counters={{\"resolved\":{},\"proposed\":{},\"unresolved\":{}}}",
            scoring_started.elapsed().as_millis(),
            resolved_mentions
                .iter()
                .filter(|mention| mention.entity_id.is_some())
                .count(),
            proposed_links.len(),
            unresolved_mentions.len()
        ));

        let canonical_started = Instant::now();
        let canonical_entities = grouped_mentions
            .into_iter()
            .map(|(entity_key, (best, mentions))| {
                let aliases = mentions
                    .iter()
                    .flat_map(|cluster| cluster.members.iter().map(|mention| mention.surface.clone()))
                    .chain(mentions.iter().map(|cluster| cluster.surface.clone()))
                    .filter(|surface| !surface.is_empty())
                    .collect::<BTreeSet<_>>();
                let evidence_ids = mentions
                    .iter()
                    .map(|cluster| cluster.evidence_id.clone())
                    .collect::<Vec<_>>();
                let mention_ids = mentions
                    .iter()
                    .map(|cluster| cluster.mention_id.clone())
                    .collect::<Vec<_>>();
                let label = choose_entity_label(&best.label, &aliases);
                CanonicalEntity {
                    entity_id: EntityId(entity_key),
                    label,
                    aliases: aliases.into_iter().collect(),
                    kind: best
                        .kind
                        .clone()
                        .or_else(|| mentions.iter().find_map(|cluster| cluster.kind.clone())),
                    scope: scope.clone(),
                    status: best.status,
                    mention_ids,
                    evidence_ids,
                    confidence: (mentions
                        .iter()
                        .map(|cluster| cluster.confidence as f64)
                        .sum::<f64>()
                        / mentions.len().max(1) as f64)
                        .max(best.score as f64) as f32,
                    provenance: merge_provenance(
                        best.provenance,
                        mentions.iter().map(|cluster| {
                            ResolutionProvenance {
                                providers: cluster.providers.iter().cloned().collect(),
                                relation_cue_count: lookup
                                    .relation_cue_count(cluster.sentence_index),
                                ..ResolutionProvenance::default()
                            }
                        }),
                    ),
                }
            })
            .collect::<Vec<_>>();
        emit_ingest_progress(format!(
            "resolution_substage=canonicalize wall_ms={} counters={{\"entities\":{}}}",
            canonical_started.elapsed().as_millis(),
            canonical_entities.len()
        ));

        let mut diagnostics = annotation.diagnostics.clone();
        diagnostics.push(Diagnostic {
            code: "PX_INVARANT_RESOLUTION".to_owned(),
            message: format!(
                "Invarant preserved {} unresolved mentions, proposed {} links, and built {} canonical entities with scored seed/alias/coreference resolution.",
                unresolved_mentions.len(),
                proposed_links.len(),
                canonical_entities.len(),
            ),
        });

        ResolutionBundle {
            unresolved_mentions,
            proposed_links,
            resolved_mentions,
            canonical_entities,
            diagnostics,
        }
    }
}

fn build_semantic_bundle(
    document: &BorrowedIngestDocument<'_>,
    text: &str,
    document_version_id: &DocumentVersionId,
    scanned_document: &ScannedDocument,
    leaf_chunks: &[SemanticChunk],
    _annotation: &AnnotationBundle,
    resolution: &ResolutionBundle,
    structure: &StructureArtifact,
) -> SemanticBundle {
    let span_index = SpanIndex::new(scanned_document);
    let chunk_index = ChunkIndex::new(leaf_chunks);
    let mut evidence_anchors = BTreeMap::<String, EvidenceAnchor>::new();
    let mut claims = Vec::new();
    let mut events = Vec::new();
    let mut relations = Vec::new();

    for entity in &resolution.canonical_entities {
        for evidence_id in &entity.evidence_ids {
            evidence_anchors
                .entry(evidence_id.0.clone())
                .or_insert(EvidenceAnchor {
                    evidence_id: evidence_id.clone(),
                    document_id: document.document_id.0.clone(),
                    chunk_id: None,
                    span_path: None,
                    range: OffsetRange::default(),
                    sentence_index: None,
                    label: entity.label.clone(),
                    kind: "entity".to_owned(),
                });
        }
    }

    for (relation_index, relation) in structure.relations.iter().enumerate() {
        let evidence_ids = relation
            .evidence
            .iter()
            .enumerate()
            .map(|(evidence_index, evidence)| {
                let evidence_id = EvidenceId(format!(
                    "evidence::{}",
                    crate::stable_hex(
                        "invarant_relation_evidence",
                        &[
                            document_version_id.0.as_str(),
                            &relation_index.to_string(),
                            &evidence_index.to_string(),
                            &evidence.range.start.to_string(),
                            &evidence.range.end.to_string(),
                        ],
                    )
                ));
                evidence_anchors
                    .entry(evidence_id.0.clone())
                    .or_insert_with(|| {
                        evidence_anchor_from_span(
                            document.document_id.0.as_str(),
                            &span_index,
                            &chunk_index,
                            evidence_id.clone(),
                            evidence,
                        )
                    });
                evidence_id
            })
            .collect::<Vec<_>>();

        let subject_entity_id = slot_entity_id(&relation.subject);
        let object_entity_id = slot_entity_id(&relation.object);
        let recipient_entity_id = slot_entity_id(&relation.recipient);
        let confidence = relation_confidence(relation);

        claims.push(Claim {
            claim_id: ClaimId(format!(
                "claim::{}",
                crate::stable_hex(
                    "invarant_claim",
                    &[
                        document_version_id.0.as_str(),
                        &relation_index.to_string(),
                        relation.relation_type.as_str(),
                        relation.lemma.as_str(),
                    ],
                )
            )),
            relation_type: relation.relation_type.clone(),
            event_class: relation.event_class.clone(),
            subject_entity_id: subject_entity_id.clone(),
            object_entity_id: object_entity_id.clone(),
            recipient_entity_id: recipient_entity_id.clone(),
            subject_text: slot_text(text, &relation.subject),
            object_text: slot_text(text, &relation.object),
            recipient_text: slot_text(text, &relation.recipient),
            evidence_ids: evidence_ids.clone(),
            evidence_chunk_ids: evidence_ids
                .iter()
                .filter_map(|evidence_id| evidence_anchors.get(&evidence_id.0))
                .filter_map(|anchor| anchor.chunk_id.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            confidence,
        });

        events.push(Event {
            event_id: EventId(format!(
                "event::{}",
                crate::stable_hex(
                    "invarant_event",
                    &[
                        document_version_id.0.as_str(),
                        &relation_index.to_string(),
                        relation.event_class.as_str(),
                        relation.lemma.as_str(),
                    ],
                )
            )),
            label: relation.lemma.clone(),
            event_class: relation.event_class.clone(),
            trigger_range: relation.verb_range.into(),
            participant_entity_ids: [
                subject_entity_id.clone(),
                object_entity_id.clone(),
                recipient_entity_id.clone(),
            ]
            .into_iter()
            .flatten()
            .collect(),
            evidence_ids: evidence_ids.clone(),
            confidence,
        });

        if subject_entity_id.is_some() || object_entity_id.is_some() {
            relations.push(Relation {
                relation_id: RelationId(format!(
                    "relation::{}",
                    crate::stable_hex(
                        "invarant_relation",
                        &[
                            document_version_id.0.as_str(),
                            &relation_index.to_string(),
                            relation.relation_type.as_str(),
                        ],
                    )
                )),
                relation_type: relation.relation_type.clone(),
                source_entity_id: subject_entity_id,
                target_entity_id: object_entity_id.or(recipient_entity_id),
                evidence_ids,
                confidence,
            });
        }
    }

    let projection = SemanticProjectionManifest {
        document_version_id: document_version_id.clone(),
        assertion_version: crate::stable_hex(
            "invarant_assertion_version",
            &[
                document_version_id.0.as_str(),
                &resolution.canonical_entities.len().to_string(),
                &claims.len().to_string(),
                &events.len().to_string(),
                &relations.len().to_string(),
                &evidence_anchors.len().to_string(),
            ],
        ),
        canonical_entity_count: resolution.canonical_entities.len(),
        claim_count: claims.len(),
        event_count: events.len(),
        relation_count: relations.len(),
        evidence_count: evidence_anchors.len(),
    };

    let mut diagnostics = resolution.diagnostics.clone();
    diagnostics.push(Diagnostic {
        code: "PX_INVARANT_SEMANTICS".to_owned(),
        message: format!(
            "Invarant assembled {} claims, {} events, {} relations, and {} evidence anchors from staged native artifacts.",
            claims.len(),
            events.len(),
            relations.len(),
            evidence_anchors.len(),
        ),
    });

    SemanticBundle {
        evidence_anchors: evidence_anchors.into_values().collect(),
        claims,
        events,
        relations,
        projection,
        diagnostics,
    }
}

fn build_session_document_state(
    document: &BorrowedIngestDocument<'_>,
    _session_id: Option<&SessionId>,
    summary: Option<&IngestDocumentSummary>,
    scanned_document: &ScannedDocument,
    annotation: &AnnotationBundle,
    resolution: &ResolutionBundle,
    semantics: &SemanticBundle,
) -> SessionDocumentState {
    let chapter_titles = scanned_document
        .spans
        .iter()
        .filter(|span| matches!(span.kind, StructuralKind::Heading))
        .filter_map(|span| span.label.clone())
        .collect::<Vec<_>>();
    let boundary_labels = scanned_document
        .spans
        .iter()
        .filter(|span| {
            matches!(
                span.kind,
                StructuralKind::Heading | StructuralKind::Paragraph
            )
        })
        .filter_map(|span| span.label.clone())
        .take(32)
        .collect::<Vec<_>>();
    let discovery_count = annotation
        .mention_candidates
        .iter()
        .filter(|mention| matches!(mention.source, Some(MentionSource::Discovery)))
        .count();

    SessionDocumentState {
        document_id: document.document_id.clone(),
        note_id: document.note_id.clone(),
        chapter_count: summary
            .map(|summary| summary.chapter_count)
            .unwrap_or_else(|| chapter_titles.len()),
        boundary_count: summary
            .map(|summary| summary.boundary_count)
            .unwrap_or_else(|| boundary_labels.len()),
        chapter_titles,
        boundary_labels,
        parent_count: summary
            .map(|summary| summary.parent_count)
            .unwrap_or_default(),
        leaf_count: summary
            .map(|summary| summary.leaf_count)
            .unwrap_or_else(|| semantics.projection.evidence_count),
        entity_count: summary
            .map(|summary| {
                summary
                    .entity_count
                    .max(resolution.canonical_entities.len())
            })
            .unwrap_or_else(|| resolution.canonical_entities.len()),
        discovery_count,
        has_front_matter_chapter: summary
            .map(|summary| summary.has_front_matter_chapter)
            .unwrap_or(false),
        has_front_matter_boundary: summary
            .map(|summary| summary.has_front_matter_boundary)
            .unwrap_or(false),
        updated_at: crate::now_ms(),
    }
}

fn normalized_artifact(
    config_hash: &str,
    record: NormalizedTextRecord,
) -> NormalizedTextArtifact {
    NormalizedTextArtifact {
        normalized_text: record.normalized_text,
        folded_text: record.folded_text,
        provider: record.provider,
        provider_version: record.provider_version,
        config_hash: config_hash.to_owned(),
    }
}

fn provider_mention_candidate(
    scanned_document: &ScannedDocument,
    span_index: &SpanIndex,
    chunk_index: &ChunkIndex,
    config_hash: &str,
    index: usize,
    mention: ProviderMention,
) -> MentionCandidate {
    MentionCandidate {
        mention_id: MentionId(format!(
            "mention::{}",
            crate::stable_hex(
                "invarant_provider_mention",
                &[
                    scanned_document.document_version_id.0.as_str(),
                    mention.provider.as_str(),
                    &index.to_string(),
                    &mention.range.start.to_string(),
                    &mention.range.end.to_string(),
                ],
            )
        )),
        surface: mention.surface.clone(),
        normalized_surface: mention.normalized_surface,
        kind: mention.kind,
        entity_ref: None,
        source: None,
        range: mention.range.into(),
        sentence_index: mention.sentence_index,
        chunk_id: chunk_index.chunk_id_for(mention.range.start, mention.range.end),
        span_id: span_index.span_id_for(mention.range.start, mention.range.end),
        evidence_id: EvidenceId(format!(
            "evidence::{}",
            crate::stable_hex(
                "invarant_provider_mention_evidence",
                &[
                    scanned_document.document_version_id.0.as_str(),
                    mention.provider.as_str(),
                    &index.to_string(),
                    &mention.range.start.to_string(),
                    &mention.range.end.to_string(),
                ],
            )
        )),
        confidence: mention.confidence,
        provider: mention.provider,
        provider_version: mention.provider_version,
        config_hash: config_hash.to_owned(),
    }
}

fn add_coreference_mentions(
    mention_candidates: &mut Vec<MentionCandidate>,
    scanned_document: &ScannedDocument,
    span_index: &SpanIndex,
    chunk_index: &ChunkIndex,
    resolver_seed: &[ResolverEntitySeed],
    config_hash: &str,
    chains: &[ProviderCoreferenceChain],
) {
    let mut seen_ranges = mention_candidates
        .iter()
        .map(|mention| (mention.range.start, mention.range.end))
        .collect::<BTreeSet<_>>();
    let mut index = mention_candidates.len();
    for chain in chains {
        for mention in &chain.mentions {
            if !seen_ranges.insert((mention.range.start, mention.range.end)) {
                continue;
            }
            let inferred_kind = infer_kind_from_coreference(
                &chain.canonical,
                mention_candidates,
                resolver_seed,
            );
            mention_candidates.push(MentionCandidate {
                mention_id: MentionId(format!(
                    "mention::{}",
                    crate::stable_hex(
                        "invarant_coref_mention",
                        &[
                            scanned_document.document_version_id.0.as_str(),
                            chain.canonical.as_str(),
                            &mention.range.start.to_string(),
                            &mention.range.end.to_string(),
                        ],
                    )
                )),
                surface: mention.surface.clone(),
                normalized_surface: normalize_surface(&mention.surface),
                kind: inferred_kind,
                entity_ref: None,
                source: None,
                range: mention.range.into(),
                sentence_index: mention.sentence_index,
                chunk_id: chunk_index.chunk_id_for(mention.range.start, mention.range.end),
                span_id: span_index.span_id_for(mention.range.start, mention.range.end),
                evidence_id: EvidenceId(format!(
                    "evidence::{}",
                    crate::stable_hex(
                        "invarant_coref_evidence",
                        &[
                            scanned_document.document_version_id.0.as_str(),
                            &index.to_string(),
                            chain.canonical.as_str(),
                            &mention.range.start.to_string(),
                            &mention.range.end.to_string(),
                        ],
                    )
                )),
                confidence: (chain.confidence * 0.8).clamp(0.35, 0.88),
                provider: chain.provider.clone(),
                provider_version: chain.provider_version.clone(),
                config_hash: config_hash.to_owned(),
            });
            index += 1;
        }
    }
}

fn coreference_artifacts(
    config_hash: &str,
    mention_candidates: &[MentionCandidate],
    chains: &[ProviderCoreferenceChain],
) -> Vec<CoreferenceChainArtifact> {
    chains
        .iter()
        .enumerate()
        .map(|(index, chain)| {
            let chain_id = crate::stable_hex(
                "invarant_coref_chain",
                &[
                    chain.canonical.as_str(),
                    &index.to_string(),
                    &chain.confidence.to_string(),
                ],
            );
            let mentions = chain
                .mentions
                .iter()
                .map(|mention| {
                    let matched = mention_candidates.iter().find(|candidate| {
                        candidate.range.start == mention.range.start
                            && candidate.range.end == mention.range.end
                    });
                    CoreferenceMentionArtifact {
                        mention_id: matched.map(|candidate| candidate.mention_id.clone()),
                        surface: mention.surface.clone(),
                        canonical_surface: mention.canonical_surface.clone(),
                        range: mention.range.into(),
                        sentence_index: mention.sentence_index,
                    }
                })
                .collect::<Vec<_>>();
            let evidence_ids = chain
                .mentions
                .iter()
                .filter_map(|mention| {
                    mention_candidates
                        .iter()
                        .find(|candidate| {
                            candidate.range.start == mention.range.start
                                && candidate.range.end == mention.range.end
                        })
                        .map(|candidate| candidate.evidence_id.clone())
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let chunk_ids = chain
                .mentions
                .iter()
                .filter_map(|mention| {
                    mention_candidates
                        .iter()
                        .find(|candidate| {
                            candidate.range.start == mention.range.start
                                && candidate.range.end == mention.range.end
                        })
                        .and_then(|candidate| candidate.chunk_id.clone())
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            CoreferenceChainArtifact {
                chain_id,
                canonical: chain.canonical.clone(),
                mentions,
                evidence_ids,
                chunk_ids,
                confidence: chain.confidence,
                provider: chain.provider.clone(),
                provider_version: chain.provider_version.clone(),
                config_hash: config_hash.to_owned(),
            }
        })
        .collect()
}

fn coreference_alias_candidates(
    config_hash: &str,
    chains: &[CoreferenceChainArtifact],
) -> Vec<AliasCandidate> {
    let mut aliases = Vec::new();
    let mut seen = BTreeSet::new();
    for chain in chains {
        let normalized_canonical = normalize_surface(&chain.canonical);
        for mention in &chain.mentions {
            let normalized_alias = normalize_surface(&mention.surface);
            if normalized_alias.is_empty() || normalized_alias == normalized_canonical {
                continue;
            }
            let key = (
                normalized_alias.clone(),
                normalized_canonical.clone(),
                mention.range.start,
                mention.range.end,
            );
            if !seen.insert(key) {
                continue;
            }
            aliases.push(AliasCandidate {
                alias: mention.surface.clone(),
                normalized_alias,
                canonical_hint: chain.canonical.clone(),
                normalized_canonical_hint: normalized_canonical.clone(),
                entity_ref: None,
                range: mention.range.clone(),
                sentence_index: mention.sentence_index,
                coreference_chain_id: Some(chain.chain_id.clone()),
                confidence: (chain.confidence * 0.78).clamp(0.35, 0.9),
                provider: chain.provider.clone(),
                provider_version: chain.provider_version.clone(),
                config_hash: config_hash.to_owned(),
            });
        }
    }
    aliases
}

fn infer_kind_from_coreference(
    canonical: &str,
    mention_candidates: &[MentionCandidate],
    resolver_seed: &[ResolverEntitySeed],
) -> Option<EntityKind> {
    let normalized = normalize_surface(canonical);
    mention_candidates
        .iter()
        .find(|candidate| candidate.normalized_surface == normalized)
        .and_then(|candidate| candidate.kind.clone())
        .or_else(|| {
            resolver_seed.iter().find_map(|seed| {
                (normalize_surface(&seed.canonical_name) == normalized
                    || seed
                        .aliases
                        .iter()
                        .any(|alias| normalize_surface(alias) == normalized))
                .then(|| seed.kind.clone())
                .flatten()
            })
        })
}

#[derive(Clone)]
struct SpanLookupEntry {
    start: u32,
    end: u32,
    span_id: SpanId,
    path: SpanPath,
}

struct SpanIndex {
    root: Option<SpanLookupEntry>,
    blocks: Vec<SpanLookupEntry>,
    sentences: Vec<SpanLookupEntry>,
}

impl SpanIndex {
    fn new(scanned_document: &ScannedDocument) -> Self {
        let mut root = None;
        let mut blocks = Vec::new();
        let mut sentences = Vec::new();
        for span in &scanned_document.spans {
            let entry = SpanLookupEntry {
                start: span.range.start,
                end: span.range.end,
                span_id: span.span_id.clone(),
                path: span.path.clone(),
            };
            match span.kind {
                StructuralKind::Root => root = Some(entry),
                StructuralKind::Sentence => sentences.push(entry),
                _ => blocks.push(entry),
            }
        }
        blocks.sort_by_key(|entry| (entry.start, entry.end));
        sentences.sort_by_key(|entry| (entry.start, entry.end));
        Self {
            root,
            blocks,
            sentences,
        }
    }

    fn span_id_for(&self, start: u32, end: u32) -> Option<SpanId> {
        self.locate(start, end).map(|entry| entry.span_id.clone())
    }

    fn span_path_for(&self, start: u32, end: u32) -> Option<SpanPath> {
        self.locate(start, end).map(|entry| entry.path.clone())
    }

    fn locate(&self, start: u32, end: u32) -> Option<&SpanLookupEntry> {
        Self::find_containing(&self.sentences, start, end)
            .or_else(|| Self::find_containing(&self.blocks, start, end))
            .or(self.root.as_ref())
    }

    fn find_containing<'a>(
        entries: &'a [SpanLookupEntry],
        start: u32,
        end: u32,
    ) -> Option<&'a SpanLookupEntry> {
        let idx = entries.partition_point(|entry| entry.start <= start);
        entries
            .get(idx.saturating_sub(1))
            .filter(|entry| entry.start <= start && entry.end >= end)
    }
}

#[derive(Clone)]
struct ChunkLookupEntry {
    start: u32,
    end: u32,
    chunk_id: ChunkId,
}

struct ChunkIndex {
    entries: Vec<ChunkLookupEntry>,
}

impl ChunkIndex {
    fn new(chunks: &[SemanticChunk]) -> Self {
        let mut entries = chunks
            .iter()
            .map(|chunk| ChunkLookupEntry {
                start: chunk.range.start,
                end: chunk.range.end,
                chunk_id: chunk.chunk_id.clone(),
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| (entry.start, entry.end));
        Self { entries }
    }

    fn chunk_id_for(&self, start: u32, end: u32) -> Option<ChunkId> {
        if self.entries.is_empty() {
            return None;
        }
        let mut idx = self.entries.partition_point(|entry| entry.start <= start);
        let mut best: Option<&ChunkLookupEntry> = None;
        while idx > 0 {
            idx -= 1;
            let entry = &self.entries[idx];
            if entry.start > start {
                continue;
            }
            if entry.end < end {
                if best.is_some() || entry.end < start {
                    break;
                }
                continue;
            }
            best = match best {
                Some(current)
                    if current.end.saturating_sub(current.start)
                        <= entry.end.saturating_sub(entry.start) =>
                {
                    Some(current)
                }
                _ => Some(entry),
            };
            if entry.start == start {
                break;
            }
        }
        best.map(|entry| entry.chunk_id.clone())
    }
}

fn build_resolution_seeds(
    scope: &ScopeKey,
    annotation: &AnnotationBundle,
    resolver_seed: &[ResolverEntitySeed],
) -> Vec<ResolutionSeed> {
    let mut seeds_by_entity = FxHashMap::<String, ResolutionSeed>::default();

    for seed in resolver_seed
        .iter()
        .filter(|seed| scope_compatible(scope, &seed.scope))
    {
        let entry = seeds_by_entity
            .entry(seed.entity_id.0.clone())
            .or_insert_with(|| ResolutionSeed {
                entity_id: seed.entity_id.clone(),
                label: seed.canonical_name.clone(),
                normalized_label: normalize_surface(&seed.canonical_name),
                aliases: BTreeSet::new(),
                kind: seed.kind.clone(),
            });
        if entry.label.is_empty() && !seed.canonical_name.is_empty() {
            entry.label = seed.canonical_name.clone();
            entry.normalized_label = normalize_surface(&seed.canonical_name);
        }
        if entry.kind.is_none() {
            entry.kind = seed.kind.clone();
        }
        entry
            .aliases
            .extend(
                std::iter::once(seed.canonical_name.as_str())
                    .chain(seed.aliases.iter().map(String::as_str))
                    .map(normalize_surface)
                    .filter(|alias| !alias.is_empty()),
            );
    }

    for alias in &annotation.alias_candidates {
        let Some(entity_ref) = alias.entity_ref.as_ref() else {
            continue;
        };
        let entity_id = match entity_ref {
            MentionEntityRef::Known(entity_id) => entity_id.clone(),
            MentionEntityRef::Speculative(key) => proposal_entity_id(
                scope,
                &alias.normalized_canonical_hint,
                None,
                Some(key.as_str()),
            ),
        };
        let aliases = [
            alias.normalized_alias.as_str(),
            alias.normalized_canonical_hint.as_str(),
        ]
        .into_iter()
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        if aliases.is_empty() {
            continue;
        }
        let entry = seeds_by_entity
            .entry(entity_id.0.clone())
            .or_insert_with(|| ResolutionSeed {
                entity_id,
                label: alias.canonical_hint.clone(),
                normalized_label: normalize_surface(&alias.canonical_hint),
                aliases: BTreeSet::new(),
                kind: None,
            });
        if entry.label.is_empty() && !alias.canonical_hint.is_empty() {
            entry.label = alias.canonical_hint.clone();
            entry.normalized_label = normalize_surface(&alias.canonical_hint);
        }
        entry.aliases.extend(aliases);
    }

    let mut seeds = seeds_by_entity.into_values().collect::<Vec<_>>();
    seeds.sort_by(|left, right| left.entity_id.cmp(&right.entity_id));
    seeds
}

fn cluster_mentions<'a>(annotation: &'a AnnotationBundle) -> Vec<ObservedMentionCluster<'a>> {
    let mut grouped = FxHashMap::<(u32, u32, String), Vec<&MentionCandidate>>::default();
    for mention in &annotation.mention_candidates {
        grouped
            .entry((
                mention.range.start,
                mention.range.end,
                mention.normalized_surface.clone(),
            ))
            .or_default()
            .push(mention);
    }

    let mut clusters = grouped
        .into_values()
        .map(|mentions| {
            let best = mentions
                .iter()
                .max_by(|left, right| {
                    left.confidence
                        .partial_cmp(&right.confidence)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or_else(|| mentions[0]);
            let entity_ref = mentions
                .iter()
                .find_map(|mention| match mention.entity_ref.as_ref() {
                    Some(MentionEntityRef::Known(entity_id)) => {
                        Some(MentionEntityRef::Known(entity_id.clone()))
                    }
                    Some(MentionEntityRef::Speculative(key)) => {
                        Some(MentionEntityRef::Speculative(key.clone()))
                    }
                    None => None,
                });
            ObservedMentionCluster {
                mention_id: best.mention_id.clone(),
                surface: best.surface.clone(),
                normalized_surface: best.normalized_surface.clone(),
                kind: mentions.iter().find_map(|mention| mention.kind.clone()),
                entity_ref,
                source: best.source.clone(),
                range: best.range.clone(),
                sentence_index: best.sentence_index,
                evidence_id: best.evidence_id.clone(),
                confidence: mentions
                    .iter()
                    .map(|mention| mention.confidence)
                    .fold(best.confidence, f32::max),
                providers: mentions.iter().map(|mention| mention.provider.clone()).collect(),
                members: mentions,
            }
        })
        .collect::<Vec<_>>();
    clusters.sort_by(|left, right| {
        (left.range.start, left.range.end, left.normalized_surface.as_str()).cmp(&(
            right.range.start,
            right.range.end,
            right.normalized_surface.as_str(),
        ))
    });
    clusters
}

fn choose_resolution_candidate(
    scope: &ScopeKey,
    cluster: &ObservedMentionCluster<'_>,
    lookup: &ResolutionLookup<'_>,
    surface_count: usize,
) -> Option<ScoredCandidate> {
    let relation_hint_count = lookup.relation_cue_count(cluster.sentence_index);
    let matching_aliases = lookup.matching_alias_indices(cluster);
    let matching_chains = lookup.matching_chain_indices(cluster);
    let candidate_seed_indices =
        lookup.candidate_seed_indices(scope, cluster, &matching_aliases, &matching_chains);
    let providers = cluster.providers.iter().cloned().collect::<Vec<_>>();
    let coreference_chain_ids = matching_chains
        .iter()
        .map(|chain_index| {
            lookup.annotation.coreference_chains[*chain_index]
                .chain_id
                .clone()
        })
        .collect::<Vec<_>>();
    let mut best_candidate: Option<ScoredCandidate> = None;

    if let Some(MentionEntityRef::Known(entity_id)) = cluster.entity_ref.as_ref() {
        let seed = lookup.seed_for_entity(entity_id);
        let label = seed
            .map(|seed| seed.label.clone())
            .unwrap_or_else(|| cluster.surface.clone());
        upsert_best_candidate(
            &mut best_candidate,
            ScoredCandidate {
            entity_id: entity_id.clone(),
            label,
            kind: seed.and_then(|seed| seed.kind.clone()).or_else(|| cluster.kind.clone()),
            status: ResolutionStatus::Resolved,
            score: cluster.confidence.max(0.94),
            provenance: ResolutionProvenance {
                providers: providers.clone(),
                reasons: vec!["seeded_exact_entity_match".to_owned()],
                seed_entity_ids: vec![entity_id.clone()],
                relation_cue_count: relation_hint_count,
                score_breakdown: BTreeMap::from([
                    ("entity_ref".to_owned(), 0.94),
                    ("confidence".to_owned(), cluster.confidence),
                ]),
                ..ResolutionProvenance::default()
            },
        },
        );
    }

    if let Some(MentionEntityRef::Speculative(key)) = cluster.entity_ref.as_ref() {
        let proposal_id = proposal_entity_id(
            scope,
            &cluster.normalized_surface,
            cluster.kind.clone(),
            Some(key.as_str()),
        );
        upsert_best_candidate(
            &mut best_candidate,
            ScoredCandidate {
            entity_id: proposal_id,
            label: cluster.surface.clone(),
            kind: cluster.kind.clone(),
            status: ResolutionStatus::Proposed,
            score: cluster.confidence.max(0.72),
            provenance: ResolutionProvenance {
                providers: providers.clone(),
                reasons: vec!["scanner_speculative_resolution".to_owned()],
                relation_cue_count: relation_hint_count,
                score_breakdown: BTreeMap::from([
                    ("speculative_ref".to_owned(), 0.72),
                    ("confidence".to_owned(), cluster.confidence),
                ]),
                ..ResolutionProvenance::default()
            },
        },
        );
    }

    for seed_index in candidate_seed_indices {
        let seed = &lookup.seeds[seed_index];
        let mut score = 0.0_f32;
        let mut label_exact = false;
        let mut alias_exact = false;
        let mut surface_similarity_score = 0.0_f32;
        let mut alias_candidate_score = 0.0_f32;
        let mut sentence_locality_score = 0.0_f32;
        let mut coreference_score = 0.0_f32;
        let mut kind_score = 0.0_f32;
        let mut repeated_surface_score = 0.0_f32;
        let mut alias_hit_names = SmallVec::<[String; 4]>::new();

        if seed.normalized_label == cluster.normalized_surface {
            score += 0.7;
            label_exact = true;
        } else if seed.aliases.contains(&cluster.normalized_surface) {
            score += 0.72;
            alias_exact = true;
        } else {
            let similarity = surface_similarity(&cluster.normalized_surface, &seed.normalized_label)
                .max(
                    seed.aliases
                        .iter()
                        .map(|alias| surface_similarity(&cluster.normalized_surface, alias))
                        .fold(0.0_f32, f32::max),
                );
            if similarity >= 0.7 {
                surface_similarity_score = 0.12 + similarity * 0.18;
                score += surface_similarity_score;
            }
        }

        for &alias_index in &matching_aliases {
            let alias = &lookup.annotation.alias_candidates[alias_index];
            let canonical_matches = alias.normalized_canonical_hint == seed.normalized_label
                || seed.aliases.contains(&alias.normalized_canonical_hint);
            if canonical_matches {
                let value = if alias.entity_ref.is_some() { 0.24 } else { 0.16 };
                score += value;
                alias_candidate_score += value;
                alias_hit_names.push(alias.alias.clone());
                if alias.sentence_index == cluster.sentence_index {
                    score += 0.06;
                    sentence_locality_score += 0.06;
                }
            }
        }

        for &chain_index in &matching_chains {
            let canonical_normalized = &lookup.normalized_chain_canonicals[chain_index];
            if canonical_normalized == &seed.normalized_label
                || seed.aliases.contains(canonical_normalized)
            {
                score += 0.18;
                coreference_score += 0.18;
            }
        }

        if let (Some(cluster_kind), Some(seed_kind)) = (cluster.kind.as_ref(), seed.kind.as_ref()) {
            if cluster_kind == seed_kind {
                score += 0.12;
                kind_score = 0.12;
            } else {
                score -= 0.18;
                kind_score = -0.18;
            }
        }

        if relation_hint_count > 0 {
            score += (relation_hint_count.min(2) as f32) * 0.04;
        }

        if surface_count > 1
            && (seed.normalized_label == cluster.normalized_surface
                || seed.aliases.contains(&cluster.normalized_surface))
        {
            score += 0.07;
            repeated_surface_score = 0.07;
        }

        if score <= 0.34 {
            continue;
        }

        let final_score = score.max(cluster.confidence * 0.85);
        let replace = best_candidate
            .as_ref()
            .map(|current| final_score > current.score)
            .unwrap_or(true);
        if !replace {
            continue;
        }

        let status = if score >= 0.82 {
            ResolutionStatus::Resolved
        } else {
            ResolutionStatus::Proposed
        };
        let mut reasons = SmallVec::<[String; 6]>::new();
        let mut score_breakdown = BTreeMap::<String, f32>::new();
        let mut alias_hits = SmallVec::<[String; 6]>::new();
        if label_exact {
            reasons.push("seed_label_exact_match".to_owned());
            score_breakdown.insert("label_exact".to_owned(), 0.7);
        }
        if alias_exact {
            reasons.push("seed_alias_exact_match".to_owned());
            score_breakdown.insert("alias_exact".to_owned(), 0.72);
            alias_hits.push(cluster.surface.clone());
        }
        if surface_similarity_score > 0.0 {
            reasons.push("seed_surface_similarity".to_owned());
            score_breakdown.insert("surface_similarity".to_owned(), surface_similarity_score);
        }
        if alias_candidate_score > 0.0 {
            reasons.push("annotation_alias_match".to_owned());
            score_breakdown.insert("alias_candidate".to_owned(), alias_candidate_score);
            alias_hits.extend(alias_hit_names.into_iter());
        }
        if sentence_locality_score > 0.0 {
            score_breakdown.insert("sentence_locality".to_owned(), sentence_locality_score);
        }
        if coreference_score > 0.0 {
            reasons.push("coreference_canonical_match".to_owned());
            score_breakdown.insert("coreference".to_owned(), coreference_score);
        }
        if kind_score != 0.0 {
            score_breakdown.insert(
                if kind_score > 0.0 {
                    "kind_compatibility".to_owned()
                } else {
                    "kind_penalty".to_owned()
                },
                kind_score,
            );
        }
        if relation_hint_count > 0 {
            score_breakdown.insert(
                "relation_cues".to_owned(),
                (relation_hint_count.min(2) as f32) * 0.04,
            );
        }
        if repeated_surface_score > 0.0 {
            score_breakdown.insert("repeated_surface".to_owned(), repeated_surface_score);
        }

        upsert_best_candidate(
            &mut best_candidate,
            ScoredCandidate {
            entity_id: seed.entity_id.clone(),
            label: seed.label.clone(),
            kind: seed.kind.clone().or_else(|| cluster.kind.clone()),
            status,
            score: final_score,
            provenance: ResolutionProvenance {
                providers: providers.clone(),
                reasons: reasons.into_iter().collect(),
                alias_hits: alias_hits.into_iter().collect(),
                seed_entity_ids: vec![seed.entity_id.clone()],
                coreference_chain_ids: coreference_chain_ids.clone(),
                relation_cue_count: relation_hint_count,
                score_breakdown,
            },
        },
        );
    }

    if best_candidate.is_none() && likely_entity_surface(&cluster.surface, cluster.kind.as_ref()) {
        let chain_boost = if matching_chains.is_empty() { 0.0 } else { 0.12 };
        let repeated_boost = if surface_count > 1 { 0.1 } else { 0.0 };
        let relation_boost = if relation_hint_count > 0 { 0.05 } else { 0.0 };
        let score = (cluster.confidence * 0.55 + chain_boost + repeated_boost + relation_boost)
            .max(0.42);
        if score >= 0.44 {
            upsert_best_candidate(
                &mut best_candidate,
                ScoredCandidate {
                entity_id: proposal_entity_id(
                    scope,
                    &cluster.normalized_surface,
                    cluster.kind.clone(),
                    cluster.source.as_ref().map(|_| cluster.surface.as_str()),
                ),
                label: cluster.surface.clone(),
                kind: cluster.kind.clone(),
                status: ResolutionStatus::Proposed,
                score,
                provenance: ResolutionProvenance {
                    providers,
                    reasons: vec!["scoped_surface_resolution".to_owned()],
                    coreference_chain_ids,
                    relation_cue_count: relation_hint_count,
                    score_breakdown: BTreeMap::from([
                        ("base_confidence".to_owned(), cluster.confidence * 0.55),
                        ("coreference".to_owned(), chain_boost),
                        ("repeated_surface".to_owned(), repeated_boost),
                        ("relation_cues".to_owned(), relation_boost),
                    ]),
                    ..ResolutionProvenance::default()
                },
            },
            );
        }
    }

    best_candidate
}

fn upsert_best_candidate(
    best_candidate: &mut Option<ScoredCandidate>,
    candidate: ScoredCandidate,
) {
    let replace = best_candidate
        .as_ref()
        .map(|current| candidate.score > current.score)
        .unwrap_or(true);
    if replace {
        *best_candidate = Some(candidate);
    }
}

fn choose_entity_label(preferred: &str, aliases: &BTreeSet<String>) -> String {
    if !preferred.trim().is_empty() && !is_pronoun(preferred) {
        return preferred.to_owned();
    }
    aliases
        .iter()
        .filter(|alias| !is_pronoun(alias))
        .max_by_key(|alias| alias.len())
        .cloned()
        .or_else(|| aliases.iter().next().cloned())
        .unwrap_or_default()
}

fn merge_provenance(
    base: ResolutionProvenance,
    others: impl IntoIterator<Item = ResolutionProvenance>,
) -> ResolutionProvenance {
    let mut providers = base.providers.into_iter().collect::<BTreeSet<_>>();
    let mut reasons = base.reasons.into_iter().collect::<BTreeSet<_>>();
    let mut alias_hits = base.alias_hits.into_iter().collect::<BTreeSet<_>>();
    let mut seed_entity_ids = base.seed_entity_ids.into_iter().collect::<BTreeSet<_>>();
    let mut coreference_chain_ids = base.coreference_chain_ids.into_iter().collect::<BTreeSet<_>>();
    let mut score_breakdown = base.score_breakdown;
    let mut relation_cue_count = base.relation_cue_count;

    for other in others {
        providers.extend(other.providers);
        reasons.extend(other.reasons);
        alias_hits.extend(other.alias_hits);
        seed_entity_ids.extend(other.seed_entity_ids);
        coreference_chain_ids.extend(other.coreference_chain_ids);
        relation_cue_count = relation_cue_count.max(other.relation_cue_count);
        for (key, value) in other.score_breakdown {
            score_breakdown
                .entry(key)
                .and_modify(|entry| *entry = entry.max(value))
                .or_insert(value);
        }
    }

    ResolutionProvenance {
        providers: providers.into_iter().collect(),
        reasons: reasons.into_iter().collect(),
        alias_hits: alias_hits.into_iter().collect(),
        seed_entity_ids: seed_entity_ids.into_iter().collect(),
        coreference_chain_ids: coreference_chain_ids.into_iter().collect(),
        relation_cue_count,
        score_breakdown,
    }
}

fn relation_cue_count(annotation: &AnnotationBundle, sentence_index: usize) -> usize {
    annotation
        .relation_cues
        .iter()
        .filter(|cue| cue.sentence_index == sentence_index)
        .count()
}

fn proposal_entity_id(
    scope: &ScopeKey,
    normalized_surface: &str,
    kind: Option<EntityKind>,
    origin: Option<&str>,
) -> EntityId {
    EntityId(format!(
        "proposal::{}",
        crate::stable_hex(
            "invarant_scoped_proposal",
            &[
                scope_fingerprint(scope).as_str(),
                normalized_surface,
                kind.as_ref()
                    .map(entity_kind_key)
                    .unwrap_or("unknown"),
                origin.unwrap_or("__origin__"),
            ],
        )
    ))
}

fn scope_compatible(left: &ScopeKey, right: &ScopeKey) -> bool {
    fn field_matches(left: &Option<String>, right: &Option<String>) -> bool {
        match (left.as_deref(), right.as_deref()) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
    }
    field_matches(&left.world_id, &right.world_id)
        && field_matches(&left.narrative_id, &right.narrative_id)
        && field_matches(&left.folder_id, &right.folder_id)
        && field_matches(&left.folder_path, &right.folder_path)
}

fn likely_entity_surface(surface: &str, kind: Option<&EntityKind>) -> bool {
    kind.is_some()
        || surface
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
        || surface.split_whitespace().count() >= 2
        || is_pronoun(surface)
}

fn is_pronoun(surface: &str) -> bool {
    matches!(
        normalize_surface(surface).as_str(),
        "he"
            | "she"
            | "they"
            | "them"
            | "him"
            | "her"
            | "his"
            | "hers"
            | "their"
            | "theirs"
            | "it"
            | "its"
            | "we"
            | "us"
            | "i"
            | "me"
            | "you"
    )
}

fn surface_similarity(left: &str, right: &str) -> f32 {
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let left_tokens = left.split_whitespace().collect::<BTreeSet<_>>();
    let right_tokens = right.split_whitespace().collect::<BTreeSet<_>>();
    let overlap = left_tokens.intersection(&right_tokens).count() as f32;
    let denom = left_tokens.len().max(right_tokens.len()) as f32;
    if denom <= 0.0 {
        0.0
    } else {
        overlap / denom
    }
}

fn entity_kind_key(kind: &EntityKind) -> &'static str {
    match kind {
        EntityKind::Character => "character",
        EntityKind::Location => "location",
        EntityKind::Npc => "npc",
        EntityKind::Item => "item",
        EntityKind::Faction => "faction",
        EntityKind::Organization => "organization",
        EntityKind::Event => "event",
        EntityKind::Concept => "concept",
        EntityKind::Other => "other",
    }
}

#[derive(Clone)]
struct BlockSpan {
    kind: String,
    range: OffsetRange,
    label: Option<String>,
}

fn detect_blocks(text: &str) -> Vec<BlockSpan> {
    let mut blocks = Vec::new();
    let mut offset = 0usize;
    for raw in text.split_inclusive('\n') {
        let trimmed = raw.trim_end_matches(['\r', '\n']);
        let trimmed_text = trimmed.trim();
        if !trimmed_text.is_empty() {
            let kind = if trimmed_text.starts_with('#') {
                "heading"
            } else if trimmed_text.starts_with('>') {
                "quote"
            } else if is_list_item(trimmed_text) {
                "listItem"
            } else if trimmed_text.starts_with("```") {
                "codeBlock"
            } else if looks_like_table_row(trimmed_text) {
                "tableRow"
            } else if looks_like_speaker_turn(trimmed_text) {
                "speakerTurn"
            } else {
                "paragraph"
            };
            blocks.push(BlockSpan {
                kind: kind.to_owned(),
                range: OffsetRange {
                    start: offset.min(u32::MAX as usize) as u32,
                    end: (offset + trimmed.len()).min(u32::MAX as usize) as u32,
                },
                label: heading_label(trimmed_text)
                    .or_else(|| Some(trimmed_text.chars().take(120).collect())),
            });
        }
        offset += raw.len();
    }
    if blocks.is_empty() && !text.trim().is_empty() {
        blocks.push(BlockSpan {
            kind: "paragraph".to_owned(),
            range: OffsetRange {
                start: 0,
                end: text.len().min(u32::MAX as usize) as u32,
            },
            label: Some(text.trim().chars().take(120).collect()),
        });
    }
    blocks
}

fn alias_candidate_from_link(
    text: &str,
    config_hash: &str,
    link: &ResolverLink,
) -> Option<AliasCandidate> {
    let alias = slice_preview(
        text,
        link.source_range.start as usize,
        link.source_range.end as usize,
        120,
    );
    if alias.is_empty() {
        return None;
    }
    let canonical_hint = link
        .target_range
        .map(|range| slice_preview(text, range.start as usize, range.end as usize, 120))
        .filter(|value| !value.is_empty())
        .or_else(|| {
            link.target_entity.as_ref().map(|entity| match entity {
                MentionEntityRef::Known(entity_id) => entity_id.0.clone(),
                MentionEntityRef::Speculative(key) => key.clone(),
            })
        })
        .unwrap_or_else(|| alias.clone());
    Some(AliasCandidate {
        alias,
        normalized_alias: normalize_surface(&slice_preview(
            text,
            link.source_range.start as usize,
            link.source_range.end as usize,
            120,
        )),
        canonical_hint: canonical_hint.clone(),
        normalized_canonical_hint: normalize_surface(&canonical_hint),
        entity_ref: link.target_entity.clone(),
        range: link.source_range.into(),
        sentence_index: link.sentence_index,
        coreference_chain_id: None,
        confidence: link.confidence,
        provider: "phoenix-scanner".to_owned(),
        provider_version: "v1".to_owned(),
        config_hash: config_hash.to_owned(),
    })
}

fn deepest_containing_span<'a>(
    spans: &'a [StructuralSpan],
    start: u32,
    end: u32,
) -> Option<&'a StructuralSpan> {
    spans
        .iter()
        .filter(|span| span.range.start <= start && span.range.end >= end)
        .max_by_key(|span| (span.depth, span.range.end.saturating_sub(span.range.start)))
}

fn containing_chunk<'a>(
    chunks: &'a [SemanticChunk],
    start: u32,
    end: u32,
) -> Option<&'a SemanticChunk> {
    chunks
        .iter()
        .filter(|chunk| chunk.range.start <= start && chunk.range.end >= end)
        .min_by_key(|chunk| chunk.range.end.saturating_sub(chunk.range.start))
        .or_else(|| {
            chunks
                .iter()
                .find(|chunk| chunk.range.end > start && chunk.range.start < end)
        })
}

fn evidence_anchor_from_span(
    document_id: &str,
    span_index: &SpanIndex,
    chunk_index: &ChunkIndex,
    evidence_id: EvidenceId,
    evidence: &EvidenceSpan,
) -> EvidenceAnchor {
    EvidenceAnchor {
        evidence_id,
        document_id: document_id.to_owned(),
        chunk_id: chunk_index.chunk_id_for(evidence.range.start, evidence.range.end),
        span_path: span_index.span_path_for(evidence.range.start, evidence.range.end),
        range: evidence.range.into(),
        sentence_index: None,
        label: evidence.label.clone(),
        kind: evidence
            .kind
            .clone()
            .unwrap_or_else(|| "relationEvidence".to_owned()),
    }
}

fn slot_entity_id(slot: &Option<phoenix_types::FrameSlot>) -> Option<EntityId> {
    slot.as_ref().and_then(|slot| match &slot.entity_ref {
        Some(MentionEntityRef::Known(entity_id)) => Some(entity_id.clone()),
        Some(MentionEntityRef::Speculative(key)) => Some(EntityId(format!(
            "proposal::{}",
            crate::stable_hex("invarant_slot_proposal", &[key.as_str()])
        ))),
        None => None,
    })
}

fn slot_text(text: &str, slot: &Option<phoenix_types::FrameSlot>) -> Option<String> {
    slot.as_ref().map(|slot| {
        slice_preview(
            text,
            slot.range.start as usize,
            slot.range.end as usize,
            140,
        )
    })
}

fn relation_confidence(relation: &RelationCandidate) -> f32 {
    [
        relation.subject.as_ref().map(|slot| slot.confidence),
        relation.object.as_ref().map(|slot| slot.confidence),
        relation.recipient.as_ref().map(|slot| slot.confidence),
    ]
    .into_iter()
    .flatten()
    .fold(0.35_f32, f32::max)
}

fn heading_label(trimmed: &str) -> Option<String> {
    if trimmed.starts_with('#') {
        return Some(trimmed.trim_start_matches('#').trim().to_owned());
    }
    let lowered = trimmed.to_ascii_lowercase();
    if lowered.starts_with("chapter ")
        || lowered.starts_with("section ")
        || lowered.starts_with("act ")
    {
        return Some(trimmed.to_owned());
    }
    None
}

fn is_list_item(trimmed: &str) -> bool {
    trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("+ ")
        || (trimmed
            .chars()
            .next()
            .map(|ch| ch.is_ascii_digit())
            .unwrap_or(false)
            && trimmed.contains(". "))
}

fn looks_like_table_row(trimmed: &str) -> bool {
    trimmed.matches('|').count() >= 2
}

fn looks_like_speaker_turn(trimmed: &str) -> bool {
    let Some((speaker, _)) = trimmed.split_once(':') else {
        return false;
    };
    !speaker.is_empty()
        && speaker.len() <= 32
        && speaker
            .chars()
            .all(|ch| ch.is_ascii_alphabetic() || ch == ' ' || ch == '-')
        && speaker
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
}

fn slice_preview(text: &str, start: usize, end: usize, max_len: usize) -> String {
    let (start, end) = normalized_bounds(text.len(), start, end);
    if start >= end {
        return String::new();
    }
    let slice = text[start..end].trim();
    let mut preview = slice.chars().take(max_len).collect::<String>();
    if slice.chars().count() > max_len {
        preview.push_str("...");
    }
    preview
}

fn normalize_surface(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalized_bounds(len: usize, start: usize, end: usize) -> (usize, usize) {
    let start = start.min(len);
    let end = end.min(len);
    if end < start {
        (end, start)
    } else {
        (start, end)
    }
}

fn scope_fingerprint(scope: &ScopeKey) -> String {
    crate::stable_hex(
        "invarant_scope",
        &[
            scope.world_id.as_deref().unwrap_or("__world__"),
            scope.narrative_id.as_deref().unwrap_or("__narrative__"),
            scope.folder_id.as_deref().unwrap_or("__folder__"),
            scope.folder_path.as_deref().unwrap_or("__path__"),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_types::{DocumentId, SentenceSpan};

    fn empty_structure() -> StructureArtifact {
        StructureArtifact::default()
    }

    fn make_context(scope: ScopeKey) -> AnalysisContext {
        AnalysisContext {
            session_id: None,
            scope,
            document_key: Some("doc-1".to_owned()),
        }
    }

    fn make_scan(text: &str, mentions: Vec<phoenix_types::MentionSpan>) -> phoenix_types::ScanArtifact {
        let mut sentences = Vec::new();
        let mut start = 0usize;
        for (index, part) in text.split_terminator('.').enumerate() {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                start += part.len() + 1;
                continue;
            }
            let local = text[start..]
                .find(trimmed)
                .map(|offset| start + offset)
                .unwrap_or(start);
            let end = local + trimmed.len();
            sentences.push(SentenceSpan {
                index,
                range: TextRange {
                    start: local.min(u32::MAX as usize) as u32,
                    end: end.min(u32::MAX as usize) as u32,
                },
            });
            start = end + 1;
        }
        phoenix_types::ScanArtifact {
            sentences,
            mentions,
            ..Default::default()
        }
    }

    #[test]
    fn structural_scan_preserves_headings_and_sentences() {
        let title = "Harbor".to_owned();
        let document = BorrowedIngestDocument {
            document_id: phoenix_types::DocumentId("doc-1".to_owned()),
            note_id: None,
            title: title.as_str(),
            text: "# Chapter 1\nRyan crossed the harbor.\n\nLen answered.\n",
            scope: ScopeKey::default(),
        };
        let scan = phoenix_types::ScanArtifact {
            sentences: vec![
                phoenix_types::SentenceSpan {
                    index: 0,
                    range: TextRange { start: 12, end: 37 },
                },
                phoenix_types::SentenceSpan {
                    index: 1,
                    range: TextRange { start: 39, end: 51 },
                },
            ],
            ..Default::default()
        };
        let scanned =
            build_structural_document(&document, &DocumentVersionId("ver-1".to_owned()), &scan);
        assert!(scanned
            .spans
            .iter()
            .any(|span| matches!(span.kind, StructuralKind::Heading)));
        assert!(scanned
            .spans
            .iter()
            .any(|span| matches!(span.kind, StructuralKind::Sentence)));
    }

    #[test]
    fn resolution_uses_seed_alias_for_known_entity() {
        let document = BorrowedIngestDocument {
            document_id: DocumentId("doc-1".to_owned()),
            note_id: None,
            title: "Harbor",
            text: "Port Authority controls the docks.",
            scope: ScopeKey::default(),
        };
        let scan = make_scan(
            document.text,
            vec![phoenix_types::MentionSpan {
                range: TextRange { start: 0, end: 14 },
                surface: "Port Authority".to_owned(),
                kind: Some(EntityKind::Organization),
                entity_ref: None,
                source: Some(MentionSource::Discovery),
                confidence: 0.74,
                sentence_index: 0,
            }],
        );
        let seed = ResolverEntitySeed {
            entity_id: EntityId("harbor_authority".to_owned()),
            canonical_name: "Harbor Authority".to_owned(),
            aliases: vec!["Port Authority".to_owned()],
            kind: Some(EntityKind::Organization),
            gender: None,
            number: None,
            scope: ScopeKey::default(),
        };

        let bundle = analyze_document(
            &document,
            None,
            make_context(ScopeKey::default()),
            &InvarantConfig::default(),
            "cfg",
            &scan,
            &empty_structure(),
            None,
            &[seed],
        )
        .expect("bundle");

        assert!(bundle.resolution.resolved_mentions.iter().any(|mention| {
            mention.entity_id.as_ref().map(|entity| entity.0.as_str())
                == Some("harbor_authority")
                && matches!(mention.status, ResolutionStatus::Resolved)
        }));
        assert!(bundle.resolution.canonical_entities.iter().any(|entity| {
            entity.entity_id.0 == "harbor_authority"
                && entity.aliases.iter().any(|alias| alias == "Port Authority")
        }));
    }

    #[test]
    fn resolution_preserves_unresolved_when_evidence_is_weak() {
        let document = BorrowedIngestDocument {
            document_id: DocumentId("doc-weak".to_owned()),
            note_id: None,
            title: "Field notes",
            text: "shadow moved quickly.",
            scope: ScopeKey::default(),
        };
        let scan = make_scan(
            document.text,
            vec![phoenix_types::MentionSpan {
                range: TextRange { start: 0, end: 6 },
                surface: "shadow".to_owned(),
                kind: None,
                entity_ref: None,
                source: Some(MentionSource::Discovery),
                confidence: 0.28,
                sentence_index: 0,
            }],
        );

        let bundle = analyze_document(
            &document,
            None,
            make_context(ScopeKey::default()),
            &InvarantConfig::default(),
            "cfg",
            &scan,
            &empty_structure(),
            None,
            &[],
        )
        .expect("bundle");

        assert_eq!(bundle.resolution.canonical_entities.len(), 0);
        assert_eq!(bundle.resolution.unresolved_mentions.len(), 1);
        assert!(matches!(
            bundle.resolution.resolved_mentions[0].status,
            ResolutionStatus::Unresolved
        ));
    }

    #[test]
    fn resolution_uses_coreference_to_link_pronoun_mentions() {
        let scope = ScopeKey::default();
        let document = BorrowedIngestDocument {
            document_id: DocumentId("doc-coref".to_owned()),
            note_id: None,
            title: "Harbor",
            text: "Ryan crossed the harbor. He waved.",
            scope: scope.clone(),
        };
        let scan = make_scan(
            document.text,
            vec![phoenix_types::MentionSpan {
                range: TextRange { start: 0, end: 4 },
                surface: "Ryan".to_owned(),
                kind: Some(EntityKind::Character),
                entity_ref: None,
                source: Some(MentionSource::Discovery),
                confidence: 0.82,
                sentence_index: 0,
            }],
        );
        let seed = ResolverEntitySeed {
            entity_id: EntityId("ryan".to_owned()),
            canonical_name: "Ryan".to_owned(),
            aliases: vec!["Ryan".to_owned()],
            kind: Some(EntityKind::Character),
            gender: None,
            number: None,
            scope: scope.clone(),
        };

        let bundle = analyze_document(
            &document,
            None,
            make_context(scope),
            &InvarantConfig::default(),
            "cfg",
            &scan,
            &empty_structure(),
            None,
            &[seed],
        )
        .expect("bundle");

        assert!(!bundle.annotation.coreference_chains.is_empty());
        assert!(bundle.resolution.resolved_mentions.iter().any(|mention| {
            mention.entity_id.as_ref().map(|entity| entity.0.as_str()) == Some("ryan")
                && mention
                    .provenance
                    .coreference_chain_ids
                    .iter()
                    .any(|value| !value.is_empty())
        }));
    }
}
