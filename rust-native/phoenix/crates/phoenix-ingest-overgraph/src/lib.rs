use std::cmp::Ordering;
use std::collections::BTreeMap;
#[cfg(feature = "background-verifier")]
use std::path::Path;
use std::path::PathBuf;

use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use phoenix_alex::Lexicon;
use phoenix_causality::{CausalityLowerer, CausalityRequest, SemanticLowerer};
use phoenix_chunker::{build_chunks, ChunkerConfig};
use phoenix_kernel::{
    DeterministicKernel, KernelEdge, KernelEdgeType, KernelEntityFacet, KernelEntitySidecar,
    KernelGraphLayer, KernelGraphSnapshot, KernelMutationBatch, KernelMutationScope,
    KernelProvenance, KernelResolutionFacet, KernelVertex, KernelVertexId,
};
use phoenix_machine::{
    MachineConfig, MachineExtractionConfig, SurfaceCompileArtifacts, SurfaceCompiler,
};
use phoenix_proposition::PropositionLowerer;
use phoenix_semantic_v2::{
    scope_storage_key, AliasConfirmation, AliasEntry, AliasPosting, CandidateEntity,
    CandidateEvidence, ChunkId, ChunkRecord, CompactResolutionKind, CompactResolutionRow,
    CorefClusterRecord, DirtyScopeRecord, DocumentArchive, DocumentCausalSubstrate,
    DocumentEventIdentitySubstrate, DocumentManifest, DocumentOrd, DocumentOrdinalAssignment,
    DocumentRevisionRef, DocumentSegmentHeader, DocumentSegmentKind, DocumentSegmentRef,
    DocumentTemporalSubstrate, DocumentVersionId, EventIdentityDiagnosticRecord, EventMentionId,
    EventMentionPacketSeed, EventModalitySemantics, EventParticipantSlot, EventSourceSemantics,
    LexicalPostingsSegment, NativeCorefSummary, NativeErSummary, PreparedDocument,
    PreparedDocumentSegment, RecordedTemporalBinding, ResolutionDecision, ResolvedMention,
    ScopeLexSidecar, ScopeOrd, SemanticEntityRecord, SemanticRelationRecord, SessionArchive,
    SurfaceTemporalCueRecord, TemporalAnchorId, TemporalAnchorRecord, TemporalAxisId,
    TemporalAxisKind, TemporalAxisRecord, TemporalClaimAtom, TemporalConstraintId,
    TemporalConstraintKind, TemporalConstraintRecord, TemporalDiagnosticRecord,
    TemporalReferenceEdge, TemporalTimexId, TemporalTimexRecord,
};
use phoenix_store_native_core::{
    BundleHeader, BundleKey, BundleKind, PhoenixArchiveStoreV2, PhoenixBundleStoreV2, StoreError,
};
use phoenix_time::TimeKernel;
use phoenix_types::{
    BiTemporalWindow, BoundaryKind, ChunkSpan, Diagnostic, DocumentId, EntityId,
    EntityKind, EvidenceSpan, FrameSlot, IndexedSpan, IndexedTextField, IngestDocument,
    IngestDocumentSummary, IngestResult, KnownMatch, KnownMatchSource, LexicalField, LexiconEntry,
    MentionEntityRef, MentionSource, MentionSpan, NarrativeVerbHit, PosTag, RelationCandidate,
    ResolverEntitySeed, ResolverLink, ResolverLinkKind, ScanArtifact, ScopeKey,
    SemanticNodeRef, SentenceFrame, SentenceSpan, SessionDocumentState, SessionId,
    StructureArtifact, TextRange, TokenClass, TokenSpan, VerbFrame,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use scirs2_text::information_extraction::{
    Entity as IeEntity, EntityType as IeEntityType, RuleBasedNER,
};
use scirs2_text::named_entity_recognition::{
    extract_entities, NerEntity, NerEntityType as PatternEntityType, NerPatternConfig,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;
use smallvec::SmallVec;
use std::sync::OnceLock;
use std::time::Instant;
use stop_words::{get, LANGUAGE};
#[cfg(feature = "background-verifier")]
use thiserror::Error;

mod bench;

pub use bench::{IngestBenchmarkCounts, IngestBenchmarkReport};

#[cfg(feature = "background-verifier")]
use gliner::model::input::text::TextInput;
#[cfg(feature = "background-verifier")]
use gliner::model::params::Parameters;
#[cfg(feature = "background-verifier")]
use gliner::model::pipeline::span::SpanMode;
#[cfg(feature = "background-verifier")]
use gliner::model::GLiNER;
#[cfg(feature = "background-verifier")]
use orp::params::RuntimeParameters;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InvarantV3ExtractionConfig {
    pub enable_scirs2_rule_ner: bool,
    pub enable_scirs2_pattern_ner: bool,
    pub enable_native_refinement: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InvarantV3VerificationConfig {
    pub enable_background_ner_verifier: bool,
    pub gliner_model_path: Option<PathBuf>,
    pub gliner_tokenizer_path: Option<PathBuf>,
    pub max_windows_per_document: usize,
    pub window_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InvarantV3CorefConfig {
    pub enable: bool,
    pub max_named_sent_window: usize,
    pub max_nominal_sent_window: usize,
    pub max_pronoun_sent_window: usize,
    pub max_named_antecedents: usize,
    pub max_nominal_antecedents: usize,
    pub max_pronoun_antecedents: usize,
    pub emit_phase2_conflict_edges: bool,
    pub persist_chunk_cap: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InvarantV3Config {
    pub chunk_size: usize,
    pub overlap: usize,
    #[serde(default)]
    pub extraction: InvarantV3ExtractionConfig,
    #[serde(default)]
    pub coref: InvarantV3CorefConfig,
    #[serde(default)]
    pub verification: InvarantV3VerificationConfig,
}

impl Default for InvarantV3ExtractionConfig {
    fn default() -> Self {
        Self {
            enable_scirs2_rule_ner: true,
            enable_scirs2_pattern_ner: true,
            enable_native_refinement: true,
        }
    }
}

impl Default for InvarantV3VerificationConfig {
    fn default() -> Self {
        Self {
            enable_background_ner_verifier: false,
            gliner_model_path: None,
            gliner_tokenizer_path: None,
            max_windows_per_document: 128,
            window_bytes: 1200,
        }
    }
}

impl Default for InvarantV3CorefConfig {
    fn default() -> Self {
        Self {
            enable: true,
            max_named_sent_window: 64,
            max_nominal_sent_window: 24,
            max_pronoun_sent_window: 12,
            max_named_antecedents: 256,
            max_nominal_antecedents: 96,
            max_pronoun_antecedents: 48,
            emit_phase2_conflict_edges: true,
            persist_chunk_cap: 8,
        }
    }
}

fn machine_extraction_config(
    extraction: &InvarantV3ExtractionConfig,
) -> MachineExtractionConfig {
    MachineExtractionConfig {
        enable_scirs2_rule_ner: extraction.enable_scirs2_rule_ner,
        enable_scirs2_pattern_ner: extraction.enable_scirs2_pattern_ner,
        enable_native_refinement: extraction.enable_native_refinement,
    }
}

fn machine_compiler_for_extraction(extraction: &InvarantV3ExtractionConfig) -> SurfaceCompiler {
    SurfaceCompiler::new(MachineConfig {
        extraction: machine_extraction_config(extraction),
    })
}

fn encode_archive<T: Serialize>(value: &T) -> Result<Vec<u8>, StoreError> {
    let payload =
        rmp_serde::to_vec_named(value).map_err(|error| StoreError::Query(error.to_string()))?;
    Ok(compress_prepend_size(&payload))
}

fn decode_archive<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, StoreError> {
    let payload =
        decompress_size_prepended(bytes).map_err(|error| StoreError::Query(error.to_string()))?;
    rmp_serde::from_slice(&payload).map_err(|error| StoreError::Query(error.to_string()))
}

fn encode_segment_payload<T: Serialize>(value: &T) -> Result<(Vec<u8>, usize), StoreError> {
    let payload =
        rmp_serde::to_vec_named(value).map_err(|error| StoreError::Query(error.to_string()))?;
    let uncompressed_len = payload.len();
    Ok((compress_prepend_size(&payload), uncompressed_len))
}

fn document_archive_header(
    scope: &ScopeKey,
    document_id: &str,
    revision: u64,
    byte_len: usize,
    created_at: i64,
) -> BundleHeader {
    BundleHeader {
        key: BundleKey {
            kind: BundleKind::DocumentArchive,
            scope: scope_storage_key(scope),
            entity_key: document_id.to_owned(),
            revision,
        },
        byte_len,
        created_at,
    }
}

fn session_archive_header(
    session_id: &SessionId,
    revision: u64,
    byte_len: usize,
    created_at: i64,
) -> BundleHeader {
    BundleHeader {
        key: BundleKey {
            kind: BundleKind::SessionArchive,
            scope: session_id.0.clone(),
            entity_key: session_id.0.clone(),
            revision,
        },
        byte_len,
        created_at,
    }
}

fn scope_lex_sidecar_header(
    scope: &ScopeKey,
    revision: u64,
    byte_len: usize,
    created_at: i64,
) -> BundleHeader {
    let scope_key = scope_storage_key(scope);
    BundleHeader {
        key: BundleKey {
            kind: BundleKind::ScopeLexSidecar,
            scope: scope_key.clone(),
            entity_key: scope_key,
            revision,
        },
        byte_len,
        created_at,
    }
}

const RULE_NER_MAX_BYTES_WITHOUT_SEEDS: usize = 256 * 1024;
const HOT_PATH_PATTERN_NER_MAX_BYTES: usize = 512 * 1024;
const EXPENSIVE_NER_SENTENCE_PAD: usize = 1;
const EXPENSIVE_NER_MAX_SENTENCE_FRACTION_DIVISOR: usize = 8;
const EXPENSIVE_NER_MAX_SENTENCES_ABSOLUTE: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DetectedMentionSourceKind {
    SeedGazetteer,
    Scirs2Rule,
    Scirs2Pattern,
    NativeHeuristic,
    Pronoun,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DetectedMentionKind {
    Named,
    Nominal,
    Pronoun,
}

#[derive(Clone, Debug)]
struct DetectedMention {
    range: TextRange,
    surface: String,
    normalized: String,
    mention_kind: DetectedMentionKind,
    type_hint: Option<EntityKind>,
    entity_ref: Option<MentionEntityRef>,
    source: DetectedMentionSourceKind,
    confidence: f32,
    sentence_index: usize,
}

struct CompiledSeedGazetteer {
    scope: ScopeKey,
    lexicon: Lexicon,
}

#[derive(Clone, Copy, Debug, Default)]
struct SentenceNeed {
    has_seed: bool,
    has_cap_span: bool,
    has_nominal_role: bool,
    has_pronoun: bool,
    repeated_surface_pressure: bool,
    has_discourse_cue: bool,
    proposal_count: u16,
    named_like_count: u16,
}

fn mention_source_for_detected(
    source: DetectedMentionSourceKind,
    entity_ref: Option<&MentionEntityRef>,
) -> MentionSource {
    match source {
        DetectedMentionSourceKind::SeedGazetteer if entity_ref.is_some() => MentionSource::Known,
        DetectedMentionSourceKind::SeedGazetteer => MentionSource::Alias,
        DetectedMentionSourceKind::Scirs2Rule => MentionSource::Alias,
        DetectedMentionSourceKind::Scirs2Pattern
        | DetectedMentionSourceKind::NativeHeuristic
        | DetectedMentionSourceKind::Pronoun => MentionSource::Discovery,
    }
}

fn detected_priority(mention: &DetectedMention) -> (u8, bool, usize, i32) {
    let source_rank = match mention.source {
        DetectedMentionSourceKind::SeedGazetteer => 5,
        DetectedMentionSourceKind::Scirs2Rule => 4,
        DetectedMentionSourceKind::Scirs2Pattern => 3,
        DetectedMentionSourceKind::NativeHeuristic => 2,
        DetectedMentionSourceKind::Pronoun => 1,
    };
    (
        source_rank,
        mention.entity_ref.is_some(),
        (mention.range.end - mention.range.start) as usize,
        (mention.confidence * 1000.0).round() as i32,
    )
}

fn compare_detected_mentions(left: &DetectedMention, right: &DetectedMention) -> Ordering {
    detected_priority(left)
        .cmp(&detected_priority(right))
        .then_with(|| left.range.start.cmp(&right.range.start).reverse())
}

fn range_overlaps(left: TextRange, right: TextRange) -> bool {
    left.start < right.end && right.start < left.end
}

fn map_ie_entity_kind(kind: &IeEntityType) -> Option<EntityKind> {
    match kind {
        IeEntityType::Person => Some(EntityKind::Character),
        IeEntityType::Organization => Some(EntityKind::Organization),
        IeEntityType::Location => Some(EntityKind::Location),
        IeEntityType::Custom(label) => {
            let lowered = label.to_ascii_lowercase();
            if lowered.contains("event") {
                Some(EntityKind::Event)
            } else if lowered.contains("concept") {
                Some(EntityKind::Concept)
            } else if lowered.contains("item") {
                Some(EntityKind::Item)
            } else if lowered.starts_with("temporal_") {
                None
            } else {
                Some(EntityKind::Other)
            }
        }
        IeEntityType::Date
        | IeEntityType::Time
        | IeEntityType::Money
        | IeEntityType::Percentage
        | IeEntityType::Email
        | IeEntityType::Url
        | IeEntityType::Phone
        | IeEntityType::Other => None,
    }
}

fn map_pattern_entity_kind(kind: &PatternEntityType) -> Option<EntityKind> {
    match kind {
        PatternEntityType::Person => Some(EntityKind::Character),
        PatternEntityType::Organisation => Some(EntityKind::Organization),
        PatternEntityType::Location => Some(EntityKind::Location),
        PatternEntityType::Custom(label) => {
            let lowered = label.to_ascii_lowercase();
            if lowered.contains("event") {
                Some(EntityKind::Event)
            } else if lowered.contains("concept") {
                Some(EntityKind::Concept)
            } else if lowered.contains("item") {
                Some(EntityKind::Item)
            } else if lowered.contains("role") {
                Some(EntityKind::Other)
            } else {
                Some(EntityKind::Other)
            }
        }
        PatternEntityType::Date
        | PatternEntityType::Time
        | PatternEntityType::Email
        | PatternEntityType::Url
        | PatternEntityType::IpAddress
        | PatternEntityType::Hashtag
        | PatternEntityType::Mention
        | PatternEntityType::Money
        | PatternEntityType::Percentage
        | PatternEntityType::Phone
        | PatternEntityType::Number => None,
    }
}

fn infer_seed_entity_ref(surface: &str, seeds: &[ResolverEntitySeed]) -> Option<MentionEntityRef> {
    let normalized = normalize_surface(surface);
    seeds.iter().find_map(|seed| {
        (normalize_surface(&seed.canonical_name) == normalized
            || seed
                .aliases
                .iter()
                .any(|alias| normalize_surface(alias) == normalized))
        .then(|| MentionEntityRef::Known(seed.entity_id.clone()))
    })
}

fn infer_seed_kind(surface: &str, seeds: &[ResolverEntitySeed]) -> Option<EntityKind> {
    let normalized = normalize_surface(surface);
    seeds.iter().find_map(|seed| {
        (normalize_surface(&seed.canonical_name) == normalized
            || seed
                .aliases
                .iter()
                .any(|alias| normalize_surface(alias) == normalized))
        .then(|| seed.kind.clone())
        .flatten()
    })
}

fn lexicon_entry_from_seed(seed: &ResolverEntitySeed) -> LexiconEntry {
    LexiconEntry {
        entity_id: seed.entity_id.clone(),
        label: seed.canonical_name.clone(),
        aliases: seed.aliases.clone(),
        kind: seed.kind.clone(),
        gender: seed.gender.clone(),
        number: seed.number.clone(),
        scope: seed.scope.clone(),
    }
}

fn seed_match_binding(entries: &[LexiconEntry]) -> (Option<EntityKind>, Option<MentionEntityRef>) {
    let mut entity_id = None::<EntityId>;
    let mut ambiguous_entity = false;
    let mut type_hint = None::<EntityKind>;
    let mut ambiguous_kind = false;

    for entry in entries {
        match (&entity_id, &entry.entity_id) {
            (None, next) => entity_id = Some(next.clone()),
            (Some(existing), next) if existing == next => {}
            (Some(_), _) => ambiguous_entity = true,
        }
        match (&type_hint, &entry.kind) {
            (None, Some(next)) => type_hint = Some(next.clone()),
            (Some(existing), Some(next)) if existing == next => {}
            (Some(_), Some(_)) => ambiguous_kind = true,
            _ => {}
        }
    }

    let entity_ref = if ambiguous_entity {
        None
    } else {
        entity_id.map(MentionEntityRef::Known)
    };
    let type_hint = if ambiguous_kind { None } else { type_hint };
    (type_hint, entity_ref)
}

fn build_seed_gazetteer(
    scope: &ScopeKey,
    resolver_seed: &[ResolverEntitySeed],
) -> Option<CompiledSeedGazetteer> {
    let entries = resolver_seed
        .iter()
        .map(lexicon_entry_from_seed)
        .collect::<Vec<_>>();
    if entries.is_empty() {
        return None;
    }
    let lexicon = Lexicon::from_entries(&entries).ok()?;
    Some(CompiledSeedGazetteer {
        scope: scope.clone(),
        lexicon,
    })
}

fn seeded_gazetteer_mentions(
    text: &str,
    _tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
    gazetteer: Option<&CompiledSeedGazetteer>,
) -> Vec<DetectedMention> {
    let Some(gazetteer) = gazetteer else {
        return Vec::new();
    };
    gazetteer
        .lexicon
        .scan(text, &gazetteer.scope)
        .into_iter()
        .filter_map(|matched| detected_seed_mention_from_match(sentences, matched))
        .collect()
}

fn detected_seed_mention_from_match(
    sentences: &[SentenceSpan],
    matched: KnownMatch,
) -> Option<DetectedMention> {
    let surface = matched.surface.trim().to_owned();
    if surface.is_empty() {
        return None;
    }
    let normalized = normalize_surface(&surface);
    if normalized.is_empty() {
        return None;
    }
    let (type_hint, entity_ref) = seed_match_binding(&matched.entries);
    let sentence_index = locate_sentence(sentences, matched.range).unwrap_or_else(|| {
        sentences
            .last()
            .map(|sentence| sentence.index)
            .unwrap_or_default()
    });
    let confidence = match matched.source {
        Some(KnownMatchSource::ExactCanonical) => 1.0,
        Some(KnownMatchSource::ExactAlias) => 0.99,
        Some(KnownMatchSource::ExactAutoAlias) => 0.96,
        Some(KnownMatchSource::FuzzyAnchor) | None => matched.confidence,
    };
    Some(DetectedMention {
        range: matched.range,
        surface,
        normalized,
        mention_kind: DetectedMentionKind::Named,
        type_hint,
        entity_ref,
        source: DetectedMentionSourceKind::SeedGazetteer,
        confidence,
        sentence_index,
    })
}

fn build_rule_entities(text: &str, seeds: &[ResolverEntitySeed]) -> Vec<IeEntity> {
    if seeds.is_empty() && text.len() > RULE_NER_MAX_BYTES_WITHOUT_SEEDS {
        return Vec::new();
    }
    let mut ner = RuleBasedNER::with_basic_knowledge();
    for seed in seeds {
        let values = std::iter::once(seed.canonical_name.clone())
            .chain(seed.aliases.iter().cloned())
            .collect::<Vec<_>>();
        match seed.kind.as_ref() {
            Some(EntityKind::Character) | Some(EntityKind::Npc) => ner.add_person_names(values),
            Some(EntityKind::Organization) | Some(EntityKind::Faction) => {
                ner.add_organizations(values)
            }
            Some(EntityKind::Location) => ner.add_locations(values),
            _ => {}
        }
    }
    ner.extract_entities(text).unwrap_or_default()
}

fn scirs2_rule_mentions(
    text: &str,
    sentences: &[SentenceSpan],
    resolver_seed: &[ResolverEntitySeed],
) -> Vec<DetectedMention> {
    let mut sentence_cursor = 0usize;
    build_rule_entities(text, resolver_seed)
        .into_iter()
        .filter_map(|entity| {
            let type_hint = map_ie_entity_kind(&entity.entity_type)?;
            let range = TextRange {
                start: entity.start.min(u32::MAX as usize) as u32,
                end: entity.end.min(u32::MAX as usize) as u32,
            };
            let entity_ref = infer_seed_entity_ref(&entity.text, resolver_seed);
            let inferred_kind = infer_seed_kind(&entity.text, resolver_seed).or(Some(type_hint));
            Some(DetectedMention {
                range,
                surface: safe_text_slice(text, range).to_owned(),
                normalized: normalize_surface(&entity.text),
                mention_kind: DetectedMentionKind::Named,
                type_hint: inferred_kind,
                entity_ref,
                source: DetectedMentionSourceKind::Scirs2Rule,
                confidence: entity.confidence.clamp(0.0, 1.0) as f32,
                sentence_index: locate_sentence_cursor(sentences, &mut sentence_cursor, range),
            })
        })
        .collect()
}

fn scirs2_pattern_mentions(text: &str, sentences: &[SentenceSpan]) -> Vec<DetectedMention> {
    let mut config = NerPatternConfig::none();
    config.heuristic_entities = true;
    let mut sentence_cursor = 0usize;
    extract_entities(text, &config)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|entity: NerEntity| {
            let type_hint = map_pattern_entity_kind(&entity.entity_type)?;
            let range = TextRange {
                start: entity.start.min(u32::MAX as usize) as u32,
                end: entity.end.min(u32::MAX as usize) as u32,
            };
            Some(DetectedMention {
                range,
                surface: safe_text_slice(text, range).to_owned(),
                normalized: normalize_surface(&entity.text),
                mention_kind: DetectedMentionKind::Named,
                type_hint: Some(type_hint),
                entity_ref: None,
                source: DetectedMentionSourceKind::Scirs2Pattern,
                confidence: entity.confidence.clamp(0.0, 1.0) as f32,
                sentence_index: locate_sentence_cursor(sentences, &mut sentence_cursor, range),
            })
        })
        .collect()
}

fn connective_token(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "of" | "the" | "and" | "&"
    )
}

fn looks_like_entity_token(value: &str) -> bool {
    value
        .chars()
        .next()
        .map(|ch| ch.is_uppercase())
        .unwrap_or(false)
        && value.chars().any(|ch| ch.is_alphabetic())
}

fn title_token(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().trim_end_matches('.'),
        "mr" | "mrs"
            | "ms"
            | "dr"
            | "prof"
            | "captain"
            | "capt"
            | "sir"
            | "lord"
            | "lady"
            | "king"
            | "queen"
            | "prince"
            | "princess"
    )
}

fn nominal_role_token(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "manager"
            | "captain"
            | "doctor"
            | "professor"
            | "teacher"
            | "brother"
            | "sister"
            | "mother"
            | "father"
            | "leader"
            | "chief"
            | "assistant"
            | "guard"
            | "agent"
            | "priest"
            | "king"
            | "queen"
    )
}

fn native_refinement_mentions(
    text: &str,
    tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
    observed_ranges: &[TextRange],
    resolver_seed: &[ResolverEntitySeed],
) -> Vec<DetectedMention> {
    let mut mentions = Vec::new();
    let mut index = 0usize;
    let mut sentence_cursor = 0usize;

    while index < tokens.len() {
        let token = &tokens[index];
        let range = token.range;
        if observed_ranges
            .iter()
            .any(|existing| range_overlaps(*existing, range))
        {
            index += 1;
            continue;
        }
        let token_text = slice_or_empty(text, token.range);
        let sentence_index = locate_sentence_cursor(sentences, &mut sentence_cursor, token.range);

        if matches!(token.pos, Some(PosTag::Pronoun)) {
            mentions.push(DetectedMention {
                range,
                surface: token_text.to_owned(),
                normalized: normalize_token_surface(token_text),
                mention_kind: DetectedMentionKind::Pronoun,
                type_hint: None,
                entity_ref: None,
                source: DetectedMentionSourceKind::Pronoun,
                confidence: 0.65,
                sentence_index,
            });
            index += 1;
            continue;
        }

        if title_token(token_text) {
            let mut last = index;
            let mut end = token.range.end;
            while let Some(next) = tokens.get(last + 1) {
                let next_text = slice_or_empty(text, next.range);
                if looks_like_entity_token(next_text) || connective_token(next_text) {
                    end = next.range.end;
                    last += 1;
                } else {
                    break;
                }
            }
            if last > index {
                let range = TextRange {
                    start: token.range.start,
                    end,
                };
                let surface = safe_text_slice(text, range).to_owned();
                mentions.push(DetectedMention {
                    range,
                    surface: surface.clone(),
                    normalized: normalize_surface(&surface),
                    mention_kind: DetectedMentionKind::Named,
                    type_hint: infer_seed_kind(&surface, resolver_seed)
                        .or(Some(EntityKind::Character)),
                    entity_ref: infer_seed_entity_ref(&surface, resolver_seed).or_else(|| {
                        Some(MentionEntityRef::Speculative(normalize_surface(&surface)))
                    }),
                    source: DetectedMentionSourceKind::NativeHeuristic,
                    confidence: 0.8,
                    sentence_index,
                });
                index = last + 1;
                continue;
            }
        }

        if token.capitalized && matches!(token.token_class, Some(TokenClass::Word)) {
            let mut last = index;
            let mut end = token.range.end;
            while let Some(next) = tokens.get(last + 1) {
                let next_text = slice_or_empty(text, next.range);
                if (next.capitalized && matches!(next.token_class, Some(TokenClass::Word)))
                    || connective_token(next_text)
                {
                    end = next.range.end;
                    last += 1;
                } else {
                    break;
                }
            }
            let range = TextRange {
                start: token.range.start,
                end,
            };
            let surface = safe_text_slice(text, range).to_owned();
            mentions.push(DetectedMention {
                range,
                surface: surface.clone(),
                normalized: normalize_surface(&surface),
                mention_kind: DetectedMentionKind::Named,
                type_hint: infer_seed_kind(&surface, resolver_seed)
                    .or_else(|| infer_heuristic_kind(&surface)),
                entity_ref: infer_seed_entity_ref(&surface, resolver_seed)
                    .or_else(|| Some(MentionEntityRef::Speculative(normalize_surface(&surface)))),
                source: DetectedMentionSourceKind::NativeHeuristic,
                confidence: 0.78,
                sentence_index,
            });
            index = last + 1;
            continue;
        }

        if nominal_role_token(token_text) {
            mentions.push(DetectedMention {
                range,
                surface: token_text.to_owned(),
                normalized: normalize_surface(token_text),
                mention_kind: DetectedMentionKind::Nominal,
                type_hint: None,
                entity_ref: None,
                source: DetectedMentionSourceKind::NativeHeuristic,
                confidence: 0.52,
                sentence_index,
            });
            index += 1;
            continue;
        }

        let determiner = matches!(
            token_text.to_ascii_lowercase().as_str(),
            "the" | "a" | "an" | "my" | "our" | "his" | "her" | "their"
        );
        if determiner {
            if let Some(next) = tokens.get(index + 1) {
                let next_text = slice_or_empty(text, next.range);
                if nominal_role_token(next_text) {
                    let mut last = index + 1;
                    let mut end = next.range.end;
                    while let Some(trailing) = tokens.get(last + 1) {
                        let trailing_text = slice_or_empty(text, trailing.range);
                        let lower = trailing_text.to_ascii_lowercase();
                        if matches!(trailing.token_class, Some(TokenClass::Word))
                            && lower.chars().all(|ch| ch.is_ascii_lowercase() || ch == '-')
                            && lower.len() > 2
                        {
                            end = trailing.range.end;
                            last += 1;
                        } else {
                            break;
                        }
                    }
                    let range = TextRange {
                        start: next.range.start,
                        end,
                    };
                    let surface = safe_text_slice(text, range).to_owned();
                    mentions.push(DetectedMention {
                        range,
                        surface: surface.clone(),
                        normalized: normalize_surface(&surface),
                        mention_kind: DetectedMentionKind::Nominal,
                        type_hint: None,
                        entity_ref: None,
                        source: DetectedMentionSourceKind::NativeHeuristic,
                        confidence: 0.58,
                        sentence_index,
                    });
                    index = last + 1;
                    continue;
                }
            }
        }

        index += 1;
    }

    mentions
}

fn infer_heuristic_kind(surface: &str) -> Option<EntityKind> {
    let lowered = surface.to_ascii_lowercase();
    if lowered.contains("city")
        || lowered.contains("harbor")
        || lowered.contains("island")
        || lowered.contains("kingdom")
    {
        Some(EntityKind::Location)
    } else if lowered.ends_with(" corp")
        || lowered.ends_with(" corporation")
        || lowered.ends_with(" company")
        || lowered.ends_with(" guild")
        || lowered.ends_with(" crew")
        || lowered.ends_with(" pirates")
    {
        Some(EntityKind::Organization)
    } else {
        Some(EntityKind::Character)
    }
}

fn dedupe_detected_mentions(mut mentions: Vec<DetectedMention>) -> Vec<DetectedMention> {
    if mentions.is_empty() {
        return mentions;
    }
    mentions.sort_by(|left, right| {
        left.range
            .start
            .cmp(&right.range.start)
            .then_with(|| right.range.end.cmp(&left.range.end))
            .then_with(|| compare_detected_mentions(left, right).reverse())
    });
    let mut deduped = Vec::with_capacity(mentions.len());
    let mut index = 0usize;
    while index < mentions.len() {
        let mut best = mentions[index].clone();
        let mut cluster_end = mentions[index].range.end;
        let mut cursor = index + 1;
        while cursor < mentions.len() && mentions[cursor].range.start < cluster_end {
            cluster_end = cluster_end.max(mentions[cursor].range.end);
            if compare_detected_mentions(&mentions[cursor], &best).is_gt() {
                best = mentions[cursor].clone();
            }
            cursor += 1;
        }
        deduped.push(best);
        index = cursor;
    }
    deduped.sort_by_key(|mention| mention.range.start);
    deduped
}

fn sentence_has_discourse_cue(text: &str, sentence: SentenceSpan) -> bool {
    let lowered = safe_text_slice(text, sentence.range).to_ascii_lowercase();
    [
        " said ",
        " told ",
        " asked ",
        " called ",
        " known as ",
        " according to ",
        " wrote ",
        " says ",
        " says,",
        " replied ",
        " answered ",
    ]
    .iter()
    .any(|cue| lowered.contains(cue))
}

fn build_sentence_needs(
    text: &str,
    sentences: &[SentenceSpan],
    proposals: &[DetectedMention],
) -> Vec<SentenceNeed> {
    let mut needs = vec![SentenceNeed::default(); sentences.len()];
    let mut normalized_counts = FxHashMap::<&str, usize>::default();
    for proposal in proposals {
        if proposal.mention_kind != DetectedMentionKind::Pronoun && !proposal.normalized.is_empty()
        {
            *normalized_counts
                .entry(proposal.normalized.as_str())
                .or_insert(0) += 1;
        }
    }
    for proposal in proposals {
        let Some(need) = needs.get_mut(proposal.sentence_index) else {
            continue;
        };
        need.proposal_count = need.proposal_count.saturating_add(1);
        if proposal.entity_ref.is_some() {
            need.has_seed = true;
        }
        match proposal.mention_kind {
            DetectedMentionKind::Pronoun => need.has_pronoun = true,
            DetectedMentionKind::Nominal => need.has_nominal_role = true,
            DetectedMentionKind::Named => {
                need.has_cap_span = true;
                need.named_like_count = need.named_like_count.saturating_add(1);
            }
        }
        if normalized_counts
            .get(proposal.normalized.as_str())
            .copied()
            .unwrap_or_default()
            > 1
        {
            need.repeated_surface_pressure = true;
        }
    }
    for sentence in sentences {
        if let Some(need) = needs.get_mut(sentence.index) {
            need.has_discourse_cue = sentence_has_discourse_cue(text, sentence.clone());
        }
    }
    needs
}

fn sentence_needs_expensive_ner(need: SentenceNeed) -> bool {
    need.repeated_surface_pressure
        || need.named_like_count >= 3
        || (need.has_pronoun
            && need.named_like_count > 0
            && (need.has_seed || need.has_discourse_cue || need.repeated_surface_pressure))
        || (need.has_nominal_role && (need.has_cap_span || need.has_discourse_cue))
        || (need.has_seed && need.repeated_surface_pressure)
}

fn sentence_need_priority(need: SentenceNeed) -> u16 {
    let mut priority = 0u16;
    if need.repeated_surface_pressure {
        priority += 5;
    }
    priority += need.named_like_count.min(4);
    if need.has_discourse_cue {
        priority += 3;
    }
    if need.has_nominal_role {
        priority += 2;
    }
    if need.has_pronoun {
        priority += 1;
    }
    if need.has_seed {
        priority += 1;
    }
    priority
}

fn plan_expensive_sentence_windows(
    sentences: &[SentenceSpan],
    needs: &[SentenceNeed],
) -> Vec<(usize, usize)> {
    if sentences.is_empty() {
        return Vec::new();
    }
    let mut selected = needs
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(index, need)| sentence_needs_expensive_ner(need).then_some((index, need)))
        .collect::<Vec<_>>();
    let max_sentences = (sentences.len() / EXPENSIVE_NER_MAX_SENTENCE_FRACTION_DIVISOR)
        .max(1)
        .min(EXPENSIVE_NER_MAX_SENTENCES_ABSOLUTE);
    if selected.len() > max_sentences {
        selected.sort_by(|left, right| {
            sentence_need_priority(right.1)
                .cmp(&sentence_need_priority(left.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        selected.truncate(max_sentences);
        selected.sort_by_key(|(index, _)| *index);
    }
    let mut marked = vec![false; sentences.len()];
    for (index, _) in selected {
        let start = index.saturating_sub(EXPENSIVE_NER_SENTENCE_PAD);
        let end = (index + EXPENSIVE_NER_SENTENCE_PAD + 1).min(sentences.len());
        for slot in &mut marked[start..end] {
            *slot = true;
        }
    }
    let mut windows = Vec::new();
    let mut index = 0usize;
    while index < marked.len() {
        if !marked[index] {
            index += 1;
            continue;
        }
        let start = index;
        while index < marked.len() && marked[index] {
            index += 1;
        }
        windows.push((start, index));
    }
    windows
}

fn shifted_sentence_spans(sentences: &[SentenceSpan], base_start: u32) -> Vec<SentenceSpan> {
    sentences
        .iter()
        .enumerate()
        .map(|(index, sentence)| SentenceSpan {
            index,
            range: TextRange {
                start: sentence.range.start.saturating_sub(base_start),
                end: sentence.range.end.saturating_sub(base_start),
            },
        })
        .collect()
}

fn remap_detected_mentions(
    mentions: Vec<DetectedMention>,
    byte_offset: u32,
    sentence_offset: usize,
) -> Vec<DetectedMention> {
    mentions
        .into_iter()
        .map(|mut mention| {
            mention.range = TextRange {
                start: mention.range.start + byte_offset,
                end: mention.range.end + byte_offset,
            };
            mention.sentence_index += sentence_offset;
            mention
        })
        .collect()
}

fn selective_expensive_mentions(
    text: &str,
    sentences: &[SentenceSpan],
    resolver_seed: &[ResolverEntitySeed],
    config: &InvarantV3ExtractionConfig,
    sentence_needs: &[SentenceNeed],
) -> Vec<DetectedMention> {
    let windows = plan_expensive_sentence_windows(sentences, sentence_needs);
    if windows.is_empty() {
        return Vec::new();
    }
    let mut mentions = Vec::new();
    for (sentence_start, sentence_end) in windows {
        let Some(first_sentence) = sentences.get(sentence_start) else {
            continue;
        };
        let Some(last_sentence) = sentences.get(sentence_end.saturating_sub(1)) else {
            continue;
        };
        let byte_start = first_sentence.range.start;
        let byte_end = last_sentence.range.end;
        let window_range = TextRange {
            start: byte_start,
            end: byte_end,
        };
        let window_text = safe_text_slice(text, window_range);
        let local_sentences =
            shifted_sentence_spans(&sentences[sentence_start..sentence_end], byte_start);
        if config.enable_scirs2_rule_ner {
            mentions.extend(remap_detected_mentions(
                scirs2_rule_mentions(window_text, &local_sentences, resolver_seed),
                byte_start,
                sentence_start,
            ));
        }
        if config.enable_scirs2_pattern_ner {
            mentions.extend(remap_detected_mentions(
                scirs2_pattern_mentions(window_text, &local_sentences),
                byte_start,
                sentence_start,
            ));
        }
    }
    mentions
}

fn detect_mentions(
    text: &str,
    tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
    seed_gazetteer: Option<&CompiledSeedGazetteer>,
    resolver_seed: &[ResolverEntitySeed],
    config: &InvarantV3ExtractionConfig,
) -> Vec<MentionSpan> {
    let mut detected = seeded_gazetteer_mentions(text, tokens, sentences, seed_gazetteer);
    if config.enable_native_refinement {
        detected.extend(native_refinement_mentions(
            text,
            tokens,
            sentences,
            &detected
                .iter()
                .map(|mention| mention.range)
                .collect::<Vec<_>>(),
            resolver_seed,
        ));
    }
    let detected = dedupe_detected_mentions(detected);
    let sentence_needs = build_sentence_needs(text, sentences, &detected);
    let mut detected = detected;
    detected.extend(selective_expensive_mentions(
        text,
        sentences,
        resolver_seed,
        config,
        &sentence_needs,
    ));

    dedupe_detected_mentions(detected)
        .into_iter()
        .map(|mention| MentionSpan {
            range: mention.range,
            surface: mention.surface,
            kind: mention.type_hint,
            entity_ref: mention.entity_ref.clone().or_else(|| {
                (mention.mention_kind == DetectedMentionKind::Named
                    && !mention.normalized.is_empty()
                    && !is_pronoun(&mention.normalized))
                .then(|| MentionEntityRef::Speculative(mention.normalized))
            }),
            source: Some(mention_source_for_detected(
                mention.source,
                mention.entity_ref.as_ref(),
            )),
            confidence: mention.confidence,
            sentence_index: mention.sentence_index,
        })
        .collect()
}

#[allow(dead_code)]
fn discover_mentions(
    text: &str,
    tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
    scope: &ScopeKey,
    resolver_seed: &[ResolverEntitySeed],
) -> Vec<MentionSpan> {
    let seed_gazetteer = build_seed_gazetteer(scope, resolver_seed);
    detect_mentions(
        text,
        tokens,
        sentences,
        seed_gazetteer.as_ref(),
        resolver_seed,
        &InvarantV3ExtractionConfig::default(),
    )
}

fn hot_path_extraction_config(
    text_len: usize,
    resolver_seed: &[ResolverEntitySeed],
    config: &InvarantV3ExtractionConfig,
) -> InvarantV3ExtractionConfig {
    let mut hot_config = config.clone();
    if text_len > HOT_PATH_PATTERN_NER_MAX_BYTES {
        hot_config.enable_scirs2_pattern_ner = false;
    }
    if resolver_seed.is_empty() && text_len > RULE_NER_MAX_BYTES_WITHOUT_SEEDS {
        hot_config.enable_scirs2_rule_ner = false;
    }
    hot_config
}

fn should_register_surface_binding(
    mention: &MentionSpan,
    coref_kind: CorefMentionKind,
    surface_count: u32,
) -> bool {
    if coref_kind != CorefMentionKind::Named {
        return false;
    }
    match mention.entity_ref.as_ref() {
        Some(MentionEntityRef::Known(_)) => true,
        Some(MentionEntityRef::Speculative(_)) => {
            surface_count > 1
                && mention.confidence >= 0.78
                && !matches!(mention.source, Some(MentionSource::Fuzzy))
        }
        None => false,
    }
}

fn merge_surface_binding(binding: &mut SurfaceLibraryBinding, entity_ref: &MentionEntityRef) {
    if binding.ambiguous {
        return;
    }
    match (&binding.entity_ref, entity_ref) {
        (None, next) => binding.entity_ref = Some(next.clone()),
        (Some(MentionEntityRef::Known(existing)), MentionEntityRef::Known(next))
            if existing == next => {}
        (Some(MentionEntityRef::Known(_)), MentionEntityRef::Known(_)) => {
            binding.entity_ref = None;
            binding.ambiguous = true;
        }
        (Some(MentionEntityRef::Speculative(existing)), MentionEntityRef::Speculative(next))
            if existing == next => {}
        (Some(MentionEntityRef::Speculative(_)), MentionEntityRef::Speculative(_)) => {
            binding.entity_ref = None;
            binding.ambiguous = true;
        }
        (Some(MentionEntityRef::Speculative(_)), MentionEntityRef::Known(next)) => {
            binding.entity_ref = Some(MentionEntityRef::Known(next.clone()));
        }
        (Some(MentionEntityRef::Known(_)), MentionEntityRef::Speculative(_)) => {}
    }
}

fn build_surface_library_bindings(
    mentions: &[MentionSpan],
    mention_surface_ords: &[u32],
    mention_coref_kinds: &[CorefMentionKind],
    surface_counts: &[u32],
) -> Vec<SurfaceLibraryBinding> {
    let mut bindings = vec![SurfaceLibraryBinding::default(); surface_counts.len()];
    for preferred_known in [true, false] {
        for (mention_ix, mention) in mentions.iter().enumerate() {
            let Some(entity_ref) = mention.entity_ref.as_ref() else {
                continue;
            };
            if preferred_known != matches!(entity_ref, MentionEntityRef::Known(_)) {
                continue;
            }
            let surface_ord = mention_surface_ords[mention_ix] as usize;
            if !should_register_surface_binding(
                mention,
                mention_coref_kinds[mention_ix],
                surface_counts[surface_ord],
            ) {
                continue;
            }
            merge_surface_binding(&mut bindings[surface_ord], entity_ref);
        }
    }
    bindings
}

fn build_entity_library(
    mentions: &[MentionSpan],
    mention_surface_ords: &[u32],
    mention_coref_kinds: &[CorefMentionKind],
    surface_counts: &[u32],
) -> AlexEntityLibrary {
    AlexEntityLibrary {
        surface_bindings: build_surface_library_bindings(
            mentions,
            mention_surface_ords,
            mention_coref_kinds,
            surface_counts,
        ),
    }
}

fn apply_surface_library_bindings(
    mentions: &mut [MentionSpan],
    mention_surface_ords: &[u32],
    mention_coref_kinds: &[CorefMentionKind],
    surface_library_bindings: &[SurfaceLibraryBinding],
) {
    for (mention_ix, mention) in mentions.iter_mut().enumerate() {
        if mention_coref_kinds[mention_ix] != CorefMentionKind::Named {
            continue;
        }
        let binding = &surface_library_bindings[mention_surface_ords[mention_ix] as usize];
        if binding.ambiguous {
            continue;
        }
        let Some(entity_ref) = binding.entity_ref.as_ref() else {
            continue;
        };
        let should_bind = match (mention.entity_ref.as_ref(), entity_ref) {
            (None, _) => true,
            (Some(MentionEntityRef::Speculative(_)), MentionEntityRef::Known(_)) => true,
            (
                Some(MentionEntityRef::Speculative(existing)),
                MentionEntityRef::Speculative(next),
            ) => existing == next,
            _ => false,
        };
        if !should_bind {
            continue;
        }
        mention.entity_ref = Some(entity_ref.clone());
        if matches!(
            mention.source,
            None | Some(MentionSource::Discovery | MentionSource::Fuzzy)
        ) {
            mention.source = Some(match entity_ref {
                MentionEntityRef::Known(_) => MentionSource::Alias,
                MentionEntityRef::Speculative(_) => MentionSource::Discovery,
            });
        }
    }
}

fn build_occurrence_layers(
    mentions: &[MentionSpan],
    mention_surface_ords: &[u32],
    mention_coref_kinds: &[CorefMentionKind],
    entity_library: &AlexEntityLibrary,
) -> (Vec<MentionFamily>, Vec<Option<u32>>) {
    let mut mention_families = Vec::<MentionFamily>::new();
    let mut family_ord_by_surface = FxHashMap::<u32, u32>::default();
    let mut family_ord_by_mention = vec![None; mentions.len()];

    for (mention_ix, mention) in mentions.iter().enumerate() {
        let surface_ord = mention_surface_ords[mention_ix];
        let mention_kind = mention_coref_kinds[mention_ix];
        let family_ord = if mention_kind == CorefMentionKind::Pronoun {
            None
        } else if let Some(existing) = family_ord_by_surface.get(&surface_ord).copied() {
            Some(existing)
        } else {
            let family_ord = mention_families.len() as u32;
            let binding = &entity_library.surface_bindings[surface_ord as usize];
            mention_families.push(MentionFamily {
                mention_kind,
                representative_mention_ix: mention_ix,
                member_indexes: Vec::new(),
                resolved_entity_ref: binding.entity_ref.clone(),
                ambiguous: binding.ambiguous,
            });
            family_ord_by_surface.insert(surface_ord, family_ord);
            Some(family_ord)
        };
        if let Some(family_ord) = family_ord {
            family_ord_by_mention[mention_ix] = Some(family_ord);
            if let Some(family) = mention_families.get_mut(family_ord as usize) {
                family.member_indexes.push(mention_ix);
                let current = &mentions[family.representative_mention_ix];
                let preferred = match (&mention.entity_ref, &current.entity_ref) {
                    (Some(MentionEntityRef::Known(_)), Some(MentionEntityRef::Known(_))) => {
                        mention_ix < family.representative_mention_ix
                    }
                    (Some(MentionEntityRef::Known(_)), _) => true,
                    (Some(MentionEntityRef::Speculative(_)), None) => true,
                    (
                        Some(MentionEntityRef::Speculative(_)),
                        Some(MentionEntityRef::Speculative(_)),
                    ) => mention_ix < family.representative_mention_ix,
                    _ => false,
                };
                if preferred {
                    family.representative_mention_ix = mention_ix;
                }
            }
        }
    }

    (mention_families, family_ord_by_mention)
}

fn build_resolver_links_native(
    mentions: &[MentionSpan],
    mention_surface_ords: &[u32],
    mention_coref_kinds: &[CorefMentionKind],
    surface_library_bindings: &[SurfaceLibraryBinding],
) -> Vec<ResolverLink> {
    let mut links = Vec::new();
    let mut last_entity_by_surface = FxHashMap::<u32, usize>::default();
    let mut antecedent = None::<usize>;
    for (index, mention) in mentions.iter().enumerate() {
        let surface_ord = mention_surface_ords[index];
        if mention_coref_kinds[index] == CorefMentionKind::Pronoun {
            if let Some(target_ix) = antecedent {
                let target = &mentions[target_ix];
                links.push(ResolverLink {
                    source_range: mention.range,
                    target_range: Some(target.range),
                    target_entity: target.entity_ref.clone(),
                    link_kind: Some(ResolverLinkKind::Pronoun),
                    confidence: 0.72,
                    sentence_index: mention.sentence_index,
                });
            }
            continue;
        }
        let family_bound_exact = mention_coref_kinds[index] == CorefMentionKind::Named
            && !surface_library_bindings[surface_ord as usize].ambiguous
            && surface_library_bindings[surface_ord as usize]
                .entity_ref
                .is_some();
        if !family_bound_exact {
            if let Some(previous_ix) = last_entity_by_surface.get(&surface_ord).copied() {
                let previous = &mentions[previous_ix];
                let target_entity = surface_library_bindings[surface_ord as usize]
                    .entity_ref
                    .clone()
                    .or_else(|| previous.entity_ref.clone());
                links.push(ResolverLink {
                    source_range: mention.range,
                    target_range: Some(previous.range),
                    target_entity,
                    link_kind: Some(ResolverLinkKind::AliasCandidate),
                    confidence: 0.61,
                    sentence_index: mention.sentence_index,
                });
            }
        }
        if mention.entity_ref.is_some() {
            antecedent = Some(index);
        }
        last_entity_by_surface.insert(surface_ord, index);
    }
    links
}

fn build_chunk_records(
    document: &IngestDocument,
    boundaries: &[BoundaryRecord],
    chunk_ranges: &[phoenix_chunker::Chunk],
) -> (Vec<ChunkRecord>, Vec<IndexedSpan>) {
    let mut chunks = Vec::with_capacity(chunk_ranges.len());
    let mut indexed_spans = Vec::with_capacity(chunk_ranges.len());
    let mut boundary_ix = 0usize;

    for (index, chunk) in chunk_ranges.iter().enumerate() {
        while boundary_ix + 1 < boundaries.len()
            && boundaries[boundary_ix + 1].range.start <= chunk.start as u32
        {
            boundary_ix += 1;
        }
        let boundary = boundaries.get(boundary_ix);
        let chapter_id = boundary.map(|value| value.chapter_id).unwrap_or(0);
        let boundary_label = boundary.map(|value| value.label.clone());
        let chunk_id = format!("{}:{index}", document.document_id.0);
        let text = document.text[chunk.start..chunk.end].trim().to_owned();
        let record = ChunkRecord {
            chunk_id: ChunkId(chunk_id.clone()),
            range: to_range(chunk.start, chunk.end),
            chapter_id,
            boundary_label: boundary_label.clone(),
            text: text.clone(),
        };
        let mut fields = Vec::new();
        if !document.title.trim().is_empty() {
            fields.push(IndexedTextField {
                field: LexicalField::Title,
                text: document.title.clone(),
            });
        }
        if !text.is_empty() {
            fields.push(IndexedTextField {
                field: LexicalField::Body,
                text,
            });
        }
        if let Some(summary) = boundary_label
            .as_deref()
            .filter(|value| !value.trim().is_empty())
        {
            fields.push(IndexedTextField {
                field: LexicalField::Summary,
                text: summary.to_owned(),
            });
        }
        indexed_spans.push(IndexedSpan {
            span_id: chunk_id,
            note_id: document.note_id.clone(),
            document_id: Some(document.document_id.clone()),
            scope: document.scope.clone(),
            fields,
        });
        chunks.push(record);
    }

    (chunks, indexed_spans)
}

fn scan_native_compact(
    text: &str,
    scope: &ScopeKey,
    resolver_seed: &[ResolverEntitySeed],
    extraction: &InvarantV3ExtractionConfig,
) -> NativeScanRows {
    let hot_config = hot_path_extraction_config(text.len(), resolver_seed, extraction);
    let compiler = machine_compiler_for_extraction(&hot_config);
    let scan = compiler.compatibility_scan_parts(text, scope, resolver_seed);
    let mut mentions = scan.mentions;
    mentions.retain(hot_path_should_keep_mention);
    let sentences = scan.sentences;
    let narrative_hits = scan.narrative_hits;
    let mut surface_ord_by_normalized = FxHashMap::<String, u32>::default();
    let mut acronym_ord_by_value = FxHashMap::<String, u32>::default();
    let mut surface_atoms = Vec::<SurfaceAtom>::new();
    let mut surface_counts = Vec::<u32>::new();
    let mut acronym_values = Vec::<Box<str>>::new();
    let mut surface_acronym_ords = Vec::<Option<u32>>::new();
    let mut mention_surface_ords = Vec::<u32>::with_capacity(mentions.len());
    let mut mention_coref_kinds = Vec::<CorefMentionKind>::with_capacity(mentions.len());
    let mut detected_pronoun_count = 0usize;
    let mut detected_nominal_count = 0usize;

    for mention in &mentions {
        let normalized = normalize_surface(&mention.surface);
        let is_pronoun_surface = is_pronoun(&normalized);
        let coref_kind = classify_coref_mention_with_pronoun(mention, is_pronoun_surface);
        let surface_ord =
            if let Some(existing) = surface_ord_by_normalized.get(&normalized).copied() {
                existing
            } else {
                let ord = surface_atoms.len() as u32;
                let acronym_ord = acronym_of_normalized(&normalized).map(|acronym| {
                    if let Some(existing) = acronym_ord_by_value.get(&acronym).copied() {
                        existing
                    } else {
                        let ord = acronym_values.len() as u32;
                        acronym_ord_by_value.insert(acronym.clone(), ord);
                        acronym_values.push(acronym.into_boxed_str());
                        ord
                    }
                });
                surface_ord_by_normalized.insert(normalized.clone(), ord);
                surface_atoms.push(SurfaceAtom {
                    normalized: normalized.into_boxed_str(),
                    is_pronoun: is_pronoun_surface,
                });
                surface_counts.push(0);
                surface_acronym_ords.push(acronym_ord);
                ord
            };
        surface_counts[surface_ord as usize] += 1;
        mention_surface_ords.push(surface_ord);
        mention_coref_kinds.push(coref_kind);
        match coref_kind {
            CorefMentionKind::Pronoun => detected_pronoun_count += 1,
            CorefMentionKind::Nominal if mention.entity_ref.is_none() => {
                detected_nominal_count += 1;
            }
            _ => {}
        }
    }
    let entity_library = build_entity_library(
        &mentions,
        &mention_surface_ords,
        &mention_coref_kinds,
        &surface_counts,
    );
    apply_surface_library_bindings(
        &mut mentions,
        &mention_surface_ords,
        &mention_coref_kinds,
        &entity_library.surface_bindings,
    );
    let (mention_families, family_ord_by_mention) =
        build_occurrence_layers(
            &mentions,
            &mention_surface_ords,
            &mention_coref_kinds,
            &entity_library,
        );
    let resolver_links = build_resolver_links_native(
        &mentions,
        &mention_surface_ords,
        &mention_coref_kinds,
        &entity_library.surface_bindings,
    );
    let detected_named_count = mentions
        .len()
        .saturating_sub(detected_nominal_count + detected_pronoun_count);
    let discovery_count = mentions
        .iter()
        .filter(|mention| {
            matches!(
                mention.source,
                Some(MentionSource::Discovery | MentionSource::Fuzzy)
            )
        })
        .count();
    NativeScanRows {
        sentences,
        mentions,
        resolver_links,
        narrative_hits,
        mention_families,
        family_ord_by_mention,
        entity_library,
        surface_atoms,
        surface_counts,
        acronym_values,
        surface_acronym_ords,
        mention_surface_ords,
        mention_coref_kinds,
        discovery_count,
        detected_named_count,
        detected_nominal_count,
        detected_pronoun_count,
    }
}

fn sentence_chunk_indexes(sentences: &[SentenceSpan], chunks: &[ChunkRecord]) -> Vec<Option<u32>> {
    let mut indexes = vec![None; sentences.len()];
    let mut chunk_cursor = 0usize;
    for sentence in sentences {
        while chunk_cursor + 1 < chunks.len() && chunks[chunk_cursor].range.end < sentence.range.end
        {
            chunk_cursor += 1;
        }
        let chunk_ix = chunks
            .get(chunk_cursor)
            .filter(|chunk| range_contains(chunk.range, sentence.range))
            .map(|_| chunk_cursor as u32)
            .or_else(|| {
                chunks
                    .iter()
                    .position(|chunk| range_contains(chunk.range, sentence.range))
                    .map(|index| index as u32)
            });
        if let Some(slot) = indexes.get_mut(sentence.index) {
            *slot = chunk_ix;
        }
    }
    indexes
}

fn build_native_structure_rows(
    _text: &str,
    scan: &NativeScanRows,
    chunks: &[ChunkRecord],
) -> NativeStructureRows {
    let relation_canonical_mentions = build_relation_canonical_mentions(scan);
    let mut sentence_mentions = vec![SmallVec::<[usize; 8]>::new(); scan.sentences.len()];
    for (mention_ix, mention) in scan.mentions.iter().enumerate() {
        if let Some(bucket) = sentence_mentions.get_mut(mention.sentence_index) {
            bucket.push(mention_ix);
        }
    }
    let mut sentence_hits = vec![SmallVec::<[usize; 4]>::new(); scan.sentences.len()];
    for (hit_ix, hit) in scan.narrative_hits.iter().enumerate() {
        if let Some(bucket) = sentence_hits.get_mut(hit.sentence_index) {
            bucket.push(hit_ix);
        }
    }

    let mut relation_seeds = Vec::new();
    for sentence in &scan.sentences {
        let mention_indexes = sentence_mentions
            .get(sentence.index)
            .cloned()
            .unwrap_or_default();
        let hit_indexes = sentence_hits
            .get(sentence.index)
            .cloned()
            .unwrap_or_default();
        for hit_ix in hit_indexes {
            let hit = &scan.narrative_hits[hit_ix];
            let (subject_mention_ix, object_mention_ix) = select_relation_seed_pair(
                scan,
                &mention_indexes,
                hit,
                &relation_canonical_mentions,
            );
            relation_seeds.push(NativeRelationSeed {
                sentence_index: sentence.index,
                relation_type: hit.relation_type.clone(),
                subject_mention_ix,
                object_mention_ix,
            });
        }
    }

    NativeStructureRows {
        relation_seeds,
        sentence_chunk_indexes: sentence_chunk_indexes(&scan.sentences, chunks),
    }
}

fn build_relation_canonical_mentions(scan: &NativeScanRows) -> Vec<usize> {
    let mut mention_ix_by_range = FxHashMap::<(u32, u32), usize>::default();
    for (mention_ix, mention) in scan.mentions.iter().enumerate() {
        mention_ix_by_range.insert((mention.range.start, mention.range.end), mention_ix);
    }

    let mut pronoun_target_by_mention_ix = FxHashMap::<usize, usize>::default();
    for link in &scan.resolver_links {
        if link.link_kind != Some(ResolverLinkKind::Pronoun) {
            continue;
        }
        let Some(target_range) = link.target_range else {
            continue;
        };
        let Some(&source_ix) =
            mention_ix_by_range.get(&(link.source_range.start, link.source_range.end))
        else {
            continue;
        };
        let Some(&target_ix) = mention_ix_by_range.get(&(target_range.start, target_range.end))
        else {
            continue;
        };
        pronoun_target_by_mention_ix.insert(source_ix, target_ix);
    }

    let mut canonical_mentions = Vec::with_capacity(scan.mentions.len());
    for mention_ix in 0..scan.mentions.len() {
        let mut canonical_ix = mention_ix;
        if scan.mention_coref_kinds[mention_ix] == CorefMentionKind::Pronoun {
            if let Some(target_ix) = pronoun_target_by_mention_ix.get(&mention_ix).copied() {
                canonical_ix = target_ix;
            }
        }
        if let Some(family_ord) = scan.family_ord_by_mention[canonical_ix] {
            if let Some(family) = scan.mention_families.get(family_ord as usize) {
                canonical_ix = family.representative_mention_ix;
            }
        }
        canonical_mentions.push(canonical_ix);
    }
    canonical_mentions
}

fn select_relation_seed_pair(
    scan: &NativeScanRows,
    mention_indexes: &[usize],
    hit: &NarrativeVerbHit,
    relation_canonical_mentions: &[usize],
) -> (Option<usize>, Option<usize>) {
    let subject_candidates = mention_indexes
        .iter()
        .copied()
        .filter(|mention_ix| scan.mentions[*mention_ix].range.end <= hit.range.start)
        .collect::<SmallVec<[usize; 8]>>();
    let object_candidates = mention_indexes
        .iter()
        .copied()
        .filter(|mention_ix| scan.mentions[*mention_ix].range.start >= hit.range.end)
        .collect::<SmallVec<[usize; 8]>>();

    if let Some((subject_mention_ix, object_mention_ix)) = select_best_relation_pair(
        scan,
        &subject_candidates,
        &object_candidates,
        hit,
        relation_canonical_mentions,
        false,
    ) {
        return (Some(subject_mention_ix), Some(object_mention_ix));
    }

    if let Some((subject_mention_ix, object_mention_ix)) = select_best_relation_pair(
        scan,
        mention_indexes,
        mention_indexes,
        hit,
        relation_canonical_mentions,
        true,
    ) {
        return (Some(subject_mention_ix), Some(object_mention_ix));
    }

    let subject_mention_ix = subject_candidates.iter().copied().max_by_key(|mention_ix| {
        relation_argument_candidate_score(
            scan,
            *mention_ix,
            hit,
            relation_canonical_mentions,
            true,
            false,
        )
    });
    let object_mention_ix = select_best_distinct_relation_partner(
        scan,
        &object_candidates,
        subject_mention_ix,
        hit,
        relation_canonical_mentions,
        false,
        false,
    )
    .or_else(|| {
        select_best_distinct_relation_partner(
            scan,
            mention_indexes,
            subject_mention_ix,
            hit,
            relation_canonical_mentions,
            false,
            true,
        )
    });
    (subject_mention_ix, object_mention_ix)
}

fn select_best_relation_pair(
    scan: &NativeScanRows,
    subject_candidates: &[usize],
    object_candidates: &[usize],
    hit: &NarrativeVerbHit,
    relation_canonical_mentions: &[usize],
    relaxed: bool,
) -> Option<(usize, usize)> {
    let mut best_pair = None::<(i32, usize, usize)>;
    for &subject_mention_ix in subject_candidates {
        let subject_score = relation_argument_candidate_score(
            scan,
            subject_mention_ix,
            hit,
            relation_canonical_mentions,
            true,
            relaxed,
        );
        for &object_mention_ix in object_candidates {
            if subject_mention_ix == object_mention_ix
                || relation_mentions_collapse_to_same_entity(
                    scan,
                    subject_mention_ix,
                    object_mention_ix,
                    relation_canonical_mentions,
                )
            {
                continue;
            }
            let object_score = relation_argument_candidate_score(
                scan,
                object_mention_ix,
                hit,
                relation_canonical_mentions,
                false,
                relaxed,
            );
            let pair_score = subject_score + object_score;
            match best_pair {
                Some((best_score, _, _)) if best_score >= pair_score => {}
                _ => best_pair = Some((pair_score, subject_mention_ix, object_mention_ix)),
            }
        }
    }
    best_pair
        .map(|(_, subject_mention_ix, object_mention_ix)| (subject_mention_ix, object_mention_ix))
}

fn select_best_distinct_relation_partner(
    scan: &NativeScanRows,
    candidates: &[usize],
    anchor_mention_ix: Option<usize>,
    hit: &NarrativeVerbHit,
    relation_canonical_mentions: &[usize],
    subject_role: bool,
    relaxed: bool,
) -> Option<usize> {
    let anchor_mention_ix = anchor_mention_ix?;
    candidates
        .iter()
        .copied()
        .filter(|candidate_ix| {
            *candidate_ix != anchor_mention_ix
                && !relation_mentions_collapse_to_same_entity(
                    scan,
                    anchor_mention_ix,
                    *candidate_ix,
                    relation_canonical_mentions,
                )
        })
        .max_by_key(|mention_ix| {
            relation_argument_candidate_score(
                scan,
                *mention_ix,
                hit,
                relation_canonical_mentions,
                subject_role,
                relaxed,
            )
        })
}

fn relation_argument_candidate_score(
    scan: &NativeScanRows,
    mention_ix: usize,
    hit: &NarrativeVerbHit,
    relation_canonical_mentions: &[usize],
    subject_role: bool,
    relaxed: bool,
) -> i32 {
    let mention = &scan.mentions[mention_ix];
    let canonical_mention_ix = relation_canonical_mentions[mention_ix];
    let canonical_mention = &scan.mentions[canonical_mention_ix];
    let distance = if subject_role {
        hit.range.start.saturating_sub(mention.range.end)
    } else {
        mention.range.start.saturating_sub(hit.range.end)
    } as i32;
    let mention_kind_score = match scan.mention_coref_kinds[mention_ix] {
        CorefMentionKind::Named => 220,
        CorefMentionKind::Nominal => 120,
        CorefMentionKind::Pronoun => -40,
    };
    let entity_ref_score = match canonical_mention.entity_ref.as_ref() {
        Some(MentionEntityRef::Known(_)) => 220,
        Some(MentionEntityRef::Speculative(_)) => 120,
        None => 0,
    };
    let kind_score = canonical_mention
        .kind
        .as_ref()
        .map(|_| 90)
        .unwrap_or_default();
    let family_score = scan.family_ord_by_mention[canonical_mention_ix]
        .and_then(|family_ord| scan.mention_families.get(family_ord as usize))
        .map(|family| if family.ambiguous { -90 } else { 80 })
        .unwrap_or_default();
    let canonicalization_bonus = if canonical_mention_ix != mention_ix {
        140
    } else {
        0
    };
    let surface_quality_score = relation_surface_quality_score(canonical_mention);
    let side_penalty = if relaxed {
        relation_argument_wrong_side_penalty(mention, hit, subject_role)
    } else {
        0
    };

    1400 - distance.min(900)
        + mention_kind_score
        + entity_ref_score
        + kind_score
        + family_score
        + canonicalization_bonus
        + surface_quality_score
        + side_penalty
}

fn relation_argument_wrong_side_penalty(
    mention: &MentionSpan,
    hit: &NarrativeVerbHit,
    subject_role: bool,
) -> i32 {
    if subject_role {
        if mention.range.end > hit.range.start {
            return -260;
        }
    } else if mention.range.start < hit.range.end {
        return -260;
    }
    0
}

fn relation_mentions_collapse_to_same_entity(
    scan: &NativeScanRows,
    left_mention_ix: usize,
    right_mention_ix: usize,
    relation_canonical_mentions: &[usize],
) -> bool {
    let left_canonical_ix = relation_canonical_mentions[left_mention_ix];
    let right_canonical_ix = relation_canonical_mentions[right_mention_ix];
    if left_canonical_ix == right_canonical_ix {
        return true;
    }
    if scan.family_ord_by_mention[left_canonical_ix].is_some()
        && scan.family_ord_by_mention[left_canonical_ix]
            == scan.family_ord_by_mention[right_canonical_ix]
    {
        return true;
    }
    match (
        scan.mentions[left_canonical_ix].entity_ref.as_ref(),
        scan.mentions[right_canonical_ix].entity_ref.as_ref(),
    ) {
        (Some(MentionEntityRef::Known(left)), Some(MentionEntityRef::Known(right))) => {
            left == right
        }
        (Some(MentionEntityRef::Speculative(left)), Some(MentionEntityRef::Speculative(right))) => {
            left == right
        }
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RelationEndpointRoute {
    Resolved,
    CanonicalKnownRef,
    CanonicalSpeculativeRef,
    MentionKnownRef,
    MentionSpeculativeRef,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RelationEndpointResolution {
    entity_ord: u32,
    mention_ix: usize,
    route: RelationEndpointRoute,
}

fn relation_surface_quality_score(mention: &MentionSpan) -> i32 {
    relation_surface_quality_score_name(
        mention.surface.trim(),
        mention.kind.is_some(),
        matches!(mention.entity_ref, Some(MentionEntityRef::Speculative(_))),
    )
}

fn relation_surface_quality_score_name(
    surface: &str,
    has_kind: bool,
    speculative_ref: bool,
) -> i32 {
    if surface.is_empty() {
        return -360;
    }
    if is_ascii_pronoun_surface(surface) {
        return -900;
    }

    let mut score = 0;
    if !surface.contains(char::is_whitespace) {
        if relation_noise_token(surface) {
            score -= 300;
        }
        if surface.len() > 4 && ascii_word_ends_with(surface, "ing") {
            score -= 240;
        } else if surface.len() > 4 && ascii_word_ends_with(surface, "ly") {
            score -= 220;
        }
    } else if relation_noise_phrase(surface) {
        score -= 260;
    }

    if !has_kind {
        score -= 120;
    }
    if speculative_ref {
        score -= 80;
        if !has_kind {
            score -= 180;
        }
    }
    score
}

fn relation_noise_token(surface: &str) -> bool {
    matches_ci(
        surface,
        &[
            "finally",
            "thankfully",
            "meanwhile",
            "however",
            "therefore",
            "instead",
            "maybe",
            "meh",
            "hey",
            "poor",
            "chapter",
            "driving",
            "arriving",
            "resting",
            "moving",
            "italian",
            "latin",
            "french",
            "english",
            "spanish",
            "miami-like",
        ],
    )
}

fn relation_noise_phrase(surface: &str) -> bool {
    matches_ci(
        surface,
        &[
            "table of contents",
            "french and english",
            "italy and",
            "to quicksave",
            "so the augusti",
        ],
    )
}

fn matches_ci(surface: &str, values: &[&str]) -> bool {
    values
        .iter()
        .any(|value| surface.eq_ignore_ascii_case(value))
}

fn ascii_word_ends_with(surface: &str, suffix: &str) -> bool {
    surface.len() >= suffix.len()
        && surface
            .chars()
            .rev()
            .zip(suffix.chars().rev())
            .all(|(left, right)| left.eq_ignore_ascii_case(&right))
}

fn is_ascii_pronoun_surface(surface: &str) -> bool {
    matches_ci(
        surface.trim_matches(|ch: char| !ch.is_alphanumeric()),
        &[
            "he", "she", "they", "them", "him", "her", "we", "us", "i", "you", "it",
        ],
    )
}

fn classify_coref_mention_with_pronoun(
    mention: &MentionSpan,
    is_pronoun_surface: bool,
) -> CorefMentionKind {
    if is_pronoun_surface {
        CorefMentionKind::Pronoun
    } else if mention.kind.is_none()
        && mention
            .surface
            .chars()
            .next()
            .map(|ch| ch.is_lowercase())
            .unwrap_or(false)
    {
        CorefMentionKind::Nominal
    } else {
        CorefMentionKind::Named
    }
}

fn acronym_of_normalized(normalized: &str) -> Option<String> {
    let parts = normalized
        .split_whitespace()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    if parts.len() < 2 {
        return None;
    }
    let mut acronym = String::new();
    for part in parts {
        let Some(ch) = part.chars().next() else {
            continue;
        };
        acronym.push(ch.to_ascii_uppercase());
    }
    (!acronym.is_empty()).then_some(acronym)
}

fn coref_pair_route(
    current: &MentionSpan,
    antecedent: &MentionSpan,
    current_norm: &str,
    antecedent_norm: &str,
    current_kind: CorefMentionKind,
    antecedent_kind: CorefMentionKind,
    current_acronym: Option<&str>,
    antecedent_acronym: Option<&str>,
) -> Option<CorefPairRoute> {
    if let (Some(MentionEntityRef::Known(left)), Some(MentionEntityRef::Known(right))) =
        (current.entity_ref.as_ref(), antecedent.entity_ref.as_ref())
    {
        if left != right {
            return None;
        }
    }
    if matches!(current_kind, CorefMentionKind::Pronoun)
        && matches!(antecedent_kind, CorefMentionKind::Pronoun)
    {
        return None;
    }
    if let (Some(left), Some(right)) = (current.kind.as_ref(), antecedent.kind.as_ref()) {
        if left != right {
            return None;
        }
    }

    if !current_norm.is_empty() && current_norm == antecedent_norm {
        return Some(CorefPairRoute::ExactSurface);
    }

    if current.surface.eq_ignore_ascii_case(&antecedent.surface)
        || current_acronym == Some(antecedent.surface.as_str())
        || antecedent_acronym == Some(current.surface.as_str())
    {
        return Some(CorefPairRoute::AliasOrAcronym);
    }

    let current_words = current_norm.split_whitespace().collect::<Vec<_>>();
    let antecedent_words = antecedent_norm.split_whitespace().collect::<Vec<_>>();
    if current_words.len() >= 2
        && antecedent_words.len() >= 1
        && (current_words.ends_with(&antecedent_words)
            || antecedent_words.ends_with(&current_words)
            || current_words.contains(&antecedent_words[0])
            || antecedent_words.contains(&current_words[0]))
        && (title_token(current_words[0]) || title_token(antecedent_words[0]))
    {
        return Some(CorefPairRoute::TitleContainment);
    }

    match (current_kind, antecedent_kind) {
        (CorefMentionKind::Pronoun, CorefMentionKind::Named) => {
            Some(CorefPairRoute::PronounToNamed)
        }
        (CorefMentionKind::Pronoun, CorefMentionKind::Nominal) => {
            Some(CorefPairRoute::PronounToNominal)
        }
        (CorefMentionKind::Nominal, CorefMentionKind::Named) => {
            Some(CorefPairRoute::NominalToNamed)
        }
        (CorefMentionKind::Named, CorefMentionKind::Named)
        | (CorefMentionKind::Named, CorefMentionKind::Nominal)
        | (CorefMentionKind::Nominal, CorefMentionKind::Nominal) => {
            Some(CorefPairRoute::OtherCompatible)
        }
        _ => None,
    }
}

fn coref_window_limits(
    mention_kind: CorefMentionKind,
    config: &InvarantV3CorefConfig,
) -> (usize, usize) {
    match mention_kind {
        CorefMentionKind::Named => (config.max_named_antecedents, config.max_named_sent_window),
        CorefMentionKind::Nominal => (
            config.max_nominal_antecedents,
            config.max_nominal_sent_window,
        ),
        CorefMentionKind::Pronoun => (
            config.max_pronoun_antecedents,
            config.max_pronoun_sent_window,
        ),
    }
}

fn coref_candidate_score(
    current: &CorefMentionRow,
    antecedent: &CorefMentionRow,
    route: CorefPairRoute,
    representative: bool,
    surface_repeat: bool,
) -> i32 {
    let mut score = route.base_score();
    let sentence_distance = current
        .sentence_index
        .saturating_sub(antecedent.sentence_index);
    if sentence_distance == 0 {
        score += 80;
    } else if sentence_distance == 1 {
        score += 50;
    }
    if current.chunk_index.is_some() && current.chunk_index == antecedent.chunk_index {
        score += 40;
    }
    if current.has_known_seed && antecedent.has_known_seed {
        score += 140;
    } else if antecedent.has_known_seed {
        score += 180;
    }
    if current.kind.is_some() && current.kind == antecedent.kind {
        score += 80;
    }
    if current.title_like && antecedent.title_like {
        score += 30;
    }
    if surface_repeat {
        score += 60;
    }
    if representative {
        score += 40;
    }
    if matches!(route, CorefPairRoute::NominalToNamed) && antecedent.kind.is_none() {
        score -= 120;
    }
    if sentence_distance > 6 {
        score -= 80;
    }
    score
}

fn update_cluster_representative(cluster: &mut CorefClusterState, rows: &[CorefMentionRow]) {
    if let Some(seed_ix) = cluster.best_seeded_mention_ix {
        cluster.representative_mention_ix = seed_ix;
    } else if let Some(named_ix) = cluster.best_named_mention_ix {
        cluster.representative_mention_ix = named_ix;
    } else {
        cluster.representative_mention_ix = cluster.most_recent_mention_ix;
    }
    let row = &rows[cluster.representative_mention_ix];
    cluster.first_sentence_index = cluster.first_sentence_index.min(row.sentence_index);
    cluster.last_sentence_index = cluster.last_sentence_index.max(row.sentence_index);
}

#[derive(Default)]
struct CorefScratch {
    seen_generation: Vec<u32>,
    current_generation: u32,
    surface_recent: Vec<SmallVec<[usize; 8]>>,
    acronym_recent: Vec<SmallVec<[usize; 8]>>,
    recent_named: SmallVec<[usize; 128]>,
    recent_nominal: SmallVec<[usize; 64]>,
    candidate_pool: SmallVec<[usize; 32]>,
}

fn coref_begin_candidate_pool(scratch: &mut CorefScratch, mention_count: usize) {
    if scratch.seen_generation.len() < mention_count {
        scratch.seen_generation.resize(mention_count, 0);
    }
    if scratch.current_generation == u32::MAX {
        scratch.seen_generation.fill(0);
        scratch.current_generation = 1;
    } else {
        scratch.current_generation += 1;
        if scratch.current_generation == 0 {
            scratch.current_generation = 1;
        }
    }
    scratch.candidate_pool.clear();
}

fn coref_push_candidate_ix(scratch: &mut CorefScratch, candidate_ix: usize, current_ix: usize) {
    if candidate_ix >= current_ix {
        return;
    }
    if scratch.seen_generation[candidate_ix] == scratch.current_generation {
        return;
    }
    scratch.seen_generation[candidate_ix] = scratch.current_generation;
    scratch.candidate_pool.push(candidate_ix);
}

fn push_recent_ix<A>(values: &mut SmallVec<A>, value: usize, max_len: usize)
where
    A: smallvec::Array<Item = usize>,
{
    if values.len() >= max_len {
        values.remove(0);
    }
    values.push(value);
}

fn coref_kind_tag(kind: Option<&EntityKind>) -> u8 {
    match kind {
        Some(EntityKind::Character) => 1,
        Some(EntityKind::Npc) => 2,
        Some(EntityKind::Organization) => 3,
        Some(EntityKind::Faction) => 4,
        Some(EntityKind::Location) => 5,
        Some(EntityKind::Event) => 6,
        Some(EntityKind::Item) => 7,
        Some(EntityKind::Concept) => 8,
        Some(EntityKind::Other) => 9,
        None => 0,
    }
}

fn intern_coref_head_ord(
    normalized: &str,
    head_ord_by_value: &mut FxHashMap<String, u32>,
    head_values: &mut Vec<String>,
) -> Option<u32> {
    let head = normalized.split_whitespace().last()?;
    if head.is_empty() {
        return None;
    }
    if let Some(existing) = head_ord_by_value.get(head).copied() {
        return Some(existing);
    }
    let ord = head_values.len() as u32;
    head_ord_by_value.insert(head.to_owned(), ord);
    head_values.push(head.to_owned());
    Some(ord)
}

fn coref_discourse_compatible(current: &CorefMentionRow, antecedent: &CorefMentionRow) -> bool {
    if let (Some(current_chunk), Some(antecedent_chunk)) =
        (current.chunk_index, antecedent.chunk_index)
    {
        current_chunk.abs_diff(antecedent_chunk) <= 1
    } else {
        current
            .sentence_index
            .saturating_sub(antecedent.sentence_index)
            <= 3
    }
}

fn coref_block_candidates(
    scratch: &mut CorefScratch,
    index: usize,
    rows: &[CorefMentionRow],
    row: &CorefMentionRow,
    max_sent_window: usize,
    recent_named_by_head: &[SmallVec<[usize; 8]>],
    recent_nominal_by_head_kind: &FxHashMap<(u32, u8), SmallVec<[usize; 8]>>,
) {
    if let Some(bucket) = scratch.surface_recent.get(row.surface_ord as usize) {
        let recent = bucket
            .iter()
            .rev()
            .copied()
            .collect::<SmallVec<[usize; 8]>>();
        for candidate_ix in recent {
            coref_push_candidate_ix(scratch, candidate_ix, index);
        }
    }
    if let Some(acronym_ord) = row.acronym_ord {
        if let Some(bucket) = scratch.acronym_recent.get(acronym_ord as usize) {
            let recent = bucket
                .iter()
                .rev()
                .copied()
                .collect::<SmallVec<[usize; 8]>>();
            for candidate_ix in recent {
                coref_push_candidate_ix(scratch, candidate_ix, index);
            }
        }
    }
    match row.mention_kind {
        CorefMentionKind::Pronoun => {
            let recent_named = scratch
                .recent_named
                .iter()
                .rev()
                .take(12)
                .copied()
                .collect::<SmallVec<[usize; 12]>>();
            for candidate_ix in recent_named {
                let antecedent = &rows[candidate_ix];
                if row.sentence_index.saturating_sub(antecedent.sentence_index) <= max_sent_window
                    && coref_discourse_compatible(row, antecedent)
                {
                    coref_push_candidate_ix(scratch, candidate_ix, index);
                }
            }
            let recent_nominal = scratch
                .recent_nominal
                .iter()
                .rev()
                .take(6)
                .copied()
                .collect::<SmallVec<[usize; 6]>>();
            for candidate_ix in recent_nominal {
                let antecedent = &rows[candidate_ix];
                if row.sentence_index.saturating_sub(antecedent.sentence_index) <= max_sent_window
                    && coref_discourse_compatible(row, antecedent)
                {
                    coref_push_candidate_ix(scratch, candidate_ix, index);
                }
            }
        }
        CorefMentionKind::Nominal => {
            if let Some(head_ord) = row.head_ord {
                let kind_tag = coref_kind_tag(row.kind.as_ref());
                for key in [(head_ord, kind_tag), (head_ord, 0)] {
                    if let Some(bucket) = recent_nominal_by_head_kind.get(&key) {
                        let recent = bucket
                            .iter()
                            .rev()
                            .copied()
                            .collect::<SmallVec<[usize; 8]>>();
                        for candidate_ix in recent {
                            let antecedent = &rows[candidate_ix];
                            if row.sentence_index.saturating_sub(antecedent.sentence_index)
                                <= max_sent_window
                            {
                                coref_push_candidate_ix(scratch, candidate_ix, index);
                            }
                        }
                    }
                }
                if let Some(bucket) = recent_named_by_head.get(head_ord as usize) {
                    let recent = bucket
                        .iter()
                        .rev()
                        .copied()
                        .collect::<SmallVec<[usize; 8]>>();
                    for candidate_ix in recent {
                        let antecedent = &rows[candidate_ix];
                        if row.sentence_index.saturating_sub(antecedent.sentence_index)
                            <= max_sent_window
                        {
                            coref_push_candidate_ix(scratch, candidate_ix, index);
                        }
                    }
                }
            }
        }
        CorefMentionKind::Named => {
            if let Some(head_ord) = row.head_ord {
                if let Some(bucket) = recent_named_by_head.get(head_ord as usize) {
                    let recent = bucket
                        .iter()
                        .rev()
                        .copied()
                        .collect::<SmallVec<[usize; 8]>>();
                    for candidate_ix in recent {
                        let antecedent = &rows[candidate_ix];
                        if row.sentence_index.saturating_sub(antecedent.sentence_index)
                            <= max_sent_window
                        {
                            coref_push_candidate_ix(scratch, candidate_ix, index);
                        }
                    }
                }
            }
        }
    }
}

fn union_find_find(parents: &mut [usize], ix: usize) -> usize {
    if parents[ix] != ix {
        let parent = parents[ix];
        parents[ix] = union_find_find(parents, parent);
    }
    parents[ix]
}

fn union_find_union(parents: &mut [usize], ranks: &mut [u8], left: usize, right: usize) {
    let left_root = union_find_find(parents, left);
    let right_root = union_find_find(parents, right);
    if left_root == right_root {
        return;
    }
    if ranks[left_root] < ranks[right_root] {
        parents[left_root] = right_root;
    } else if ranks[left_root] > ranks[right_root] {
        parents[right_root] = left_root;
    } else {
        parents[right_root] = left_root;
        ranks[left_root] = ranks[left_root].saturating_add(1);
    }
}

fn seed_coref_family_edges(scan: &NativeScanRows) -> (Vec<CorefAcceptedEdge>, Vec<bool>) {
    let mut accepted_edges = Vec::<CorefAcceptedEdge>::new();
    let mut family_locked_mentions = vec![false; scan.mentions.len()];

    for family in &scan.mention_families {
        if family.mention_kind != CorefMentionKind::Named
            || family.ambiguous
            || family.resolved_entity_ref.is_none()
            || family.member_indexes.len() < 2
        {
            continue;
        }
        let representative = family.representative_mention_ix;
        for &member_ix in &family.member_indexes {
            if member_ix == representative {
                continue;
            }
            accepted_edges.push(CorefAcceptedEdge {
                left_ix: member_ix,
                right_ix: representative,
                route: CorefPairRoute::ExactSurface,
                score_millis: 1320,
            });
            family_locked_mentions[member_ix] = true;
        }
    }

    (accepted_edges, family_locked_mentions)
}

fn build_coref_rows(
    scan: &NativeScanRows,
    structure: &NativeStructureRows,
    config: &InvarantV3CorefConfig,
) -> NativeCorefRows {
    let mut result = NativeCorefRows::default();
    if !config.enable || scan.mentions.is_empty() {
        return result;
    }

    let mut head_ord_by_value = FxHashMap::<String, u32>::default();
    let mut head_values = Vec::<String>::new();
    let rows = scan
        .mentions
        .iter()
        .enumerate()
        .map(|(mention_ix, mention)| {
            let surface_ord = scan.mention_surface_ords[mention_ix];
            let normalized = scan.surface_atoms[surface_ord as usize].normalized.as_ref();
            CorefMentionRow {
                sentence_index: mention.sentence_index,
                chunk_index: structure
                    .sentence_chunk_indexes
                    .get(mention.sentence_index)
                    .copied()
                    .flatten(),
                mention_kind: scan.mention_coref_kinds[mention_ix],
                has_known_seed: matches!(mention.entity_ref, Some(MentionEntityRef::Known(_))),
                kind: mention.kind.clone(),
                surface_ord,
                acronym_ord: scan.surface_acronym_ords[surface_ord as usize],
                head_ord: intern_coref_head_ord(
                    normalized,
                    &mut head_ord_by_value,
                    &mut head_values,
                ),
                title_like: normalized
                    .split_whitespace()
                    .next()
                    .map(title_token)
                    .unwrap_or(false),
            }
        })
        .collect::<Vec<_>>();
    let coref_active = rows
        .iter()
        .enumerate()
        .map(|(index, row)| match row.mention_kind {
            CorefMentionKind::Pronoun | CorefMentionKind::Nominal => true,
            CorefMentionKind::Named => {
                row.has_known_seed
                    || scan.surface_acronym_ords[scan.mention_surface_ords[index] as usize]
                        .is_some()
                    || scan.surface_counts[scan.mention_surface_ords[index] as usize] > 1
            }
        })
        .collect::<Vec<_>>();
    let mut cluster_by_mention = vec![0u32; rows.len()];
    let mut representative_by_mention = vec![None; rows.len()];
    let mut candidate_links_by_mention = vec![SmallVec::<[usize; 2]>::new(); rows.len()];
    let mut scratch = CorefScratch {
        seen_generation: vec![0; rows.len()],
        current_generation: 0,
        surface_recent: vec![SmallVec::<[usize; 8]>::new(); scan.surface_atoms.len()],
        acronym_recent: vec![SmallVec::<[usize; 8]>::new(); scan.acronym_values.len()],
        recent_named: SmallVec::new(),
        recent_nominal: SmallVec::new(),
        candidate_pool: SmallVec::new(),
    };
    let mut recent_named_by_head = vec![SmallVec::<[usize; 8]>::new(); head_values.len()];
    let mut recent_nominal_by_head_kind = FxHashMap::<(u32, u8), SmallVec<[usize; 8]>>::default();
    let mut clusters = Vec::<CorefClusterState>::new();
    let (mut accepted_edges, family_locked_mentions) = seed_coref_family_edges(scan);
    let mut candidate_link_count = 0usize;

    for (index, row) in rows.iter().enumerate() {
        if !coref_active[index] {
            continue;
        }
        let mention = &scan.mentions[index];
        let surface_ord = row.surface_ord as usize;
        let normalized = scan.surface_atoms[surface_ord].normalized.as_ref();
        let acronym_ord = row.acronym_ord.map(|value| value as usize);
        let (_max_antecedents, max_sent_window) = coref_window_limits(row.mention_kind, config);

        if family_locked_mentions[index] {
            if !normalized.is_empty() {
                push_recent_ix(&mut scratch.surface_recent[surface_ord], index, 8);
            }
            if let Some(acronym_ord) = acronym_ord {
                push_recent_ix(&mut scratch.acronym_recent[acronym_ord], index, 8);
            }
            match row.mention_kind {
                CorefMentionKind::Named => {
                    push_recent_ix(&mut scratch.recent_named, index, 128);
                    if let Some(head_ord) = row.head_ord {
                        if recent_named_by_head.len() <= head_ord as usize {
                            recent_named_by_head
                                .resize(head_ord as usize + 1, SmallVec::<[usize; 8]>::new());
                        }
                        push_recent_ix(&mut recent_named_by_head[head_ord as usize], index, 8);
                    }
                }
                CorefMentionKind::Nominal => {
                    push_recent_ix(&mut scratch.recent_nominal, index, 64);
                }
                CorefMentionKind::Pronoun => {}
            }
            continue;
        }

        coref_begin_candidate_pool(&mut scratch, rows.len());
        coref_block_candidates(
            &mut scratch,
            index,
            &rows,
            row,
            max_sent_window,
            &recent_named_by_head,
            &recent_nominal_by_head_kind,
        );

        let mut best = None::<CorefAntecedentCandidate>;
        let mut runner_up = None::<CorefAntecedentCandidate>;

        for prior_ix in scratch.candidate_pool.iter().copied() {
            if !coref_active[prior_ix] {
                continue;
            }
            let antecedent_row = &rows[prior_ix];
            let sentence_distance = row
                .sentence_index
                .saturating_sub(antecedent_row.sentence_index);
            if sentence_distance > max_sent_window {
                continue;
            }
            let antecedent = &scan.mentions[prior_ix];
            let Some(route) = coref_pair_route(
                mention,
                antecedent,
                normalized,
                scan.surface_atoms[scan.mention_surface_ords[prior_ix] as usize]
                    .normalized
                    .as_ref(),
                row.mention_kind,
                antecedent_row.mention_kind,
                acronym_ord.map(|ord| scan.acronym_values[ord].as_ref()),
                scan.surface_acronym_ords[scan.mention_surface_ords[prior_ix] as usize]
                    .map(|ord| scan.acronym_values[ord as usize].as_ref()),
            ) else {
                continue;
            };
            let representative = representative_by_mention[prior_ix]
                .map(|rep_ix| rep_ix == prior_ix)
                .unwrap_or(false);
            let surface_repeat = !normalized.is_empty()
                && scratch
                    .surface_recent
                    .get(surface_ord)
                    .map(|values| values.iter().any(|existing| *existing < index))
                    .unwrap_or(false);
            let score_millis =
                coref_candidate_score(row, antecedent_row, route, representative, surface_repeat);
            let candidate = CorefAntecedentCandidate {
                mention_ix: prior_ix,
                score_millis,
                route,
            };
            if best
                .as_ref()
                .map(|existing| candidate.score_millis > existing.score_millis)
                .unwrap_or(true)
            {
                runner_up = best;
                best = Some(candidate);
            } else if runner_up
                .as_ref()
                .map(|existing| candidate.score_millis > existing.score_millis)
                .unwrap_or(true)
            {
                runner_up = Some(candidate);
            }
        }

        let (threshold, margin_threshold) = match row.mention_kind {
            CorefMentionKind::Pronoun => (980, 180),
            _ => (900, 140),
        };
        let top_score = best
            .map(|candidate| candidate.score_millis)
            .unwrap_or_default();
        let runner_score = runner_up
            .map(|candidate| candidate.score_millis)
            .unwrap_or_default();
        let margin = top_score.saturating_sub(runner_score);
        let attach = best
            .filter(|candidate| candidate.score_millis >= threshold && margin >= margin_threshold);
        let near_threshold = best.filter(|candidate| {
            candidate.score_millis + 80 >= threshold && candidate.score_millis < threshold
        });

        if let Some(attach) = attach {
            accepted_edges.push(CorefAcceptedEdge {
                left_ix: index,
                right_ix: attach.mention_ix,
                route: attach.route,
                score_millis: attach.score_millis,
            });
        }
        if let Some(candidate) = near_threshold {
            candidate_links_by_mention[index].push(candidate.mention_ix);
            candidate_link_count += 1;
        }
        if !normalized.is_empty() {
            push_recent_ix(&mut scratch.surface_recent[surface_ord], index, 8);
        }
        if let Some(acronym_ord) = acronym_ord {
            push_recent_ix(&mut scratch.acronym_recent[acronym_ord], index, 8);
        }
        match row.mention_kind {
            CorefMentionKind::Named => {
                push_recent_ix(&mut scratch.recent_named, index, 128);
                if let Some(head_ord) = row.head_ord {
                    if recent_named_by_head.len() <= head_ord as usize {
                        recent_named_by_head
                            .resize(head_ord as usize + 1, SmallVec::<[usize; 8]>::new());
                    }
                    push_recent_ix(&mut recent_named_by_head[head_ord as usize], index, 8);
                }
            }
            CorefMentionKind::Nominal => {
                push_recent_ix(&mut scratch.recent_nominal, index, 64);
                if let Some(head_ord) = row.head_ord {
                    let kind_tag = coref_kind_tag(row.kind.as_ref());
                    push_recent_ix(
                        recent_nominal_by_head_kind
                            .entry((head_ord, kind_tag))
                            .or_default(),
                        index,
                        8,
                    );
                    if kind_tag != 0 {
                        push_recent_ix(
                            recent_nominal_by_head_kind
                                .entry((head_ord, 0))
                                .or_default(),
                            index,
                            8,
                        );
                    }
                }
            }
            CorefMentionKind::Pronoun => {}
        }
    }

    let mut parents = (0..rows.len()).collect::<Vec<_>>();
    let mut ranks = vec![0u8; rows.len()];
    for edge in &accepted_edges {
        union_find_union(&mut parents, &mut ranks, edge.left_ix, edge.right_ix);
    }
    let mut cluster_index_by_root = FxHashMap::<usize, usize>::default();
    let mut cluster_meta_by_root = FxHashMap::<usize, (u32, i32, bool)>::default();
    for edge in &accepted_edges {
        let root = union_find_find(&mut parents, edge.left_ix);
        cluster_meta_by_root
            .entry(root)
            .and_modify(|(route_bits, max_score, _)| {
                *route_bits |= edge.route.bit();
                *max_score = (*max_score).max(edge.score_millis);
            })
            .or_insert((edge.route.bit(), edge.score_millis, false));
    }
    for (mention_ix, links) in candidate_links_by_mention.iter().enumerate() {
        if links.is_empty() || !coref_active[mention_ix] {
            continue;
        }
        let root = union_find_find(&mut parents, mention_ix);
        cluster_meta_by_root
            .entry(root)
            .and_modify(|(_, _, ambiguous)| *ambiguous = true)
            .or_insert((0, 0, true));
    }
    for (index, row) in rows.iter().enumerate() {
        if !coref_active[index] {
            continue;
        }
        let root = union_find_find(&mut parents, index);
        let cluster_ix = if let Some(existing) = cluster_index_by_root.get(&root).copied() {
            existing
        } else {
            let cluster_ix = clusters.len();
            clusters.push(CorefClusterState {
                member_indexes: Vec::new(),
                representative_mention_ix: index,
                most_recent_mention_ix: index,
                best_named_mention_ix: None,
                best_seeded_mention_ix: None,
                first_sentence_index: row.sentence_index,
                last_sentence_index: row.sentence_index,
                chunk_indexes: SmallVec::new(),
                named_count: 0,
                nominal_count: 0,
                pronoun_count: 0,
                route_mix_bits: 0,
                max_score_millis: 0,
                ambiguous: false,
            });
            cluster_index_by_root.insert(root, cluster_ix);
            cluster_ix
        };
        cluster_by_mention[index] = cluster_ix as u32;
        let cluster = &mut clusters[cluster_ix];
        cluster.member_indexes.push(index);
        cluster.most_recent_mention_ix = cluster.most_recent_mention_ix.max(index);
        cluster.first_sentence_index = cluster.first_sentence_index.min(row.sentence_index);
        cluster.last_sentence_index = cluster.last_sentence_index.max(row.sentence_index);
        if let Some(chunk_index) = row.chunk_index {
            if cluster.chunk_indexes.len() < config.persist_chunk_cap
                && !cluster
                    .chunk_indexes
                    .iter()
                    .any(|existing| *existing == chunk_index)
            {
                cluster.chunk_indexes.push(chunk_index);
            }
        }
        match row.mention_kind {
            CorefMentionKind::Named => {
                cluster.named_count += 1;
                cluster.best_named_mention_ix = Some(
                    cluster
                        .best_named_mention_ix
                        .map(|existing| existing.min(index))
                        .unwrap_or(index),
                );
            }
            CorefMentionKind::Nominal => cluster.nominal_count += 1,
            CorefMentionKind::Pronoun => cluster.pronoun_count += 1,
        }
        if row.has_known_seed {
            cluster.best_seeded_mention_ix = Some(
                cluster
                    .best_seeded_mention_ix
                    .map(|existing| existing.min(index))
                    .unwrap_or(index),
            );
        }
    }
    for (root, cluster_ix) in cluster_index_by_root {
        if let Some(cluster) = clusters.get_mut(cluster_ix) {
            if let Some((route_bits, max_score, ambiguous)) =
                cluster_meta_by_root.get(&root).copied()
            {
                cluster.route_mix_bits = route_bits;
                cluster.max_score_millis = max_score;
                cluster.ambiguous = ambiguous;
            }
            update_cluster_representative(cluster, &rows);
            for member_ix in &cluster.member_indexes {
                representative_by_mention[*member_ix] = Some(cluster.representative_mention_ix);
            }
        }
    }

    result.rows = rows;
    result.cluster_by_mention = cluster_by_mention;
    result.representative_by_mention = representative_by_mention;
    result.candidate_links_by_mention = candidate_links_by_mention;
    result.summary = NativeCorefSummary {
        cluster_count: clusters.len(),
        attached_mention_count: accepted_edges.len(),
        candidate_link_count,
        conflict_cluster_count: 0,
    };
    result.clusters = clusters;
    result
}

fn build_native_entity_memory(snapshot: Option<&KernelGraphSnapshot>) -> NativeEntityMemory {
    let mut memory = NativeEntityMemory::default();
    let Some(snapshot) = snapshot else {
        return memory;
    };
    let _ = memory.kernel.rebuild_from_kernel_batches(
        vec![
            KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Full,
                recorded_at: None,
                vertices: snapshot.vertices.clone(),
                edges: snapshot.asserted_edges.clone(),
            },
            KernelMutationBatch {
                layer: KernelGraphLayer::Candidate,
                scope: KernelMutationScope::Full,
                recorded_at: None,
                vertices: Vec::new(),
                edges: snapshot.candidate_edges.clone(),
            },
        ],
        None,
    );
    memory.entity_index = memory.kernel.entity_sidecar();
    for vertex in &snapshot.vertices {
        if let Some(facet) = vertex.entity_facet.as_ref() {
            if let Some(entity_id) = facet
                .canonical_entity_id
                .clone()
                .or_else(|| vertex.entity_id.clone())
            {
                if let Some(kind) = facet.entity_kind.clone() {
                    memory.entity_kinds.insert(entity_id, kind);
                }
            }
        }
    }
    for edge in &snapshot.asserted_edges {
        if edge.edge_type.0 != "alias_of" {
            continue;
        }
        let alias_surface = snapshot
            .vertices
            .iter()
            .find(|vertex| vertex.id.0 == edge.source_id.0)
            .and_then(|vertex| {
                vertex
                    .entity_facet
                    .as_ref()
                    .and_then(|facet| facet.surface.clone())
                    .or_else(|| {
                        vertex
                            .value
                            .get("name")
                            .and_then(|value| value.as_str())
                            .map(str::to_owned)
                    })
            });
        let entity_id = snapshot
            .vertices
            .iter()
            .find(|vertex| vertex.id.0 == edge.target_id.0)
            .and_then(|vertex| {
                vertex
                    .entity_facet
                    .as_ref()
                    .and_then(|facet| facet.canonical_entity_id.clone())
                    .or_else(|| vertex.entity_id.clone())
            });
        if let (Some(alias_surface), Some(entity_id)) = (alias_surface, entity_id) {
            memory
                .known_aliases
                .insert((entity_id, normalize_surface(&alias_surface)));
        }
    }
    memory
}

fn mention_id(document: &IngestDocument, mention_ix: usize) -> phoenix_semantic_v2::MentionId {
    phoenix_semantic_v2::MentionId(format!("mention::{}:{mention_ix}", document.document_id.0))
}

fn candidate_evidence(kind: &str, detail: impl Into<String>) -> CandidateEvidence {
    CandidateEvidence {
        kind: kind.to_owned(),
        detail: detail.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateEvidenceKind {
    Seed,
    Kernel,
    Coref,
    ResolverLink,
    LocalSurface,
    SurfaceCluster,
    LinkedMention,
    KindPenalty,
    NewSpeculative,
}

impl CandidateEvidenceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Seed => "seed",
            Self::Kernel => "kernel",
            Self::Coref => "coref",
            Self::ResolverLink => "resolver_link",
            Self::LocalSurface => "local_surface",
            Self::SurfaceCluster => "surface_cluster",
            Self::LinkedMention => "linked_mention",
            Self::KindPenalty => "kind_penalty",
            Self::NewSpeculative => "new_speculative",
        }
    }

    const fn bit(self) -> u16 {
        match self {
            Self::Seed => 1 << 0,
            Self::Kernel => 1 << 1,
            Self::Coref => 1 << 2,
            Self::ResolverLink => 1 << 3,
            Self::LocalSurface => 1 << 4,
            Self::SurfaceCluster => 1 << 5,
            Self::LinkedMention => 1 << 6,
            Self::KindPenalty => 1 << 7,
            Self::NewSpeculative => 1 << 8,
        }
    }
}

fn stable_slot_cmp(left: &CandidateSlot, right: &CandidateSlot, entity_ids: &[String]) -> Ordering {
    right.score_millis.cmp(&left.score_millis).then_with(|| {
        entity_ids[left.entity_ord as usize].cmp(&entity_ids[right.entity_ord as usize])
    })
}

fn sort_candidate_slots(candidates: &mut SmallVec<[CandidateSlot; 6]>, entity_ids: &[String]) {
    candidates.sort_by(|left, right| stable_slot_cmp(left, right, entity_ids));
}

fn merge_candidate_slot(
    candidates: &mut SmallVec<[CandidateSlot; 6]>,
    entity_ord: u32,
    source: CandidateSourceKind,
    score_millis: i32,
    evidence_kind: CandidateEvidenceKind,
    mode: ResolveDetailMode,
    evidence_detail: impl FnOnce() -> String,
) {
    const MAX_CANDIDATE_SLOTS: usize = 8;
    let evidence_bits = evidence_kind.bit();

    if let Some(entry) = candidates
        .iter_mut()
        .find(|candidate| candidate.entity_ord == entity_ord)
    {
        if score_millis > entry.score_millis {
            entry.score_millis = score_millis;
            entry.source = source;
        }
        entry.evidence_bits |= evidence_bits;
        if mode == ResolveDetailMode::Detailed {
            let evidence = candidate_evidence(evidence_kind.as_str(), evidence_detail());
            if !entry.evidence.iter().any(|existing| existing == &evidence) {
                entry.evidence.push(evidence);
            }
        }
        return;
    }

    let mut evidence = SmallVec::<[CandidateEvidence; 4]>::new();
    if mode == ResolveDetailMode::Detailed {
        evidence.push(candidate_evidence(
            evidence_kind.as_str(),
            evidence_detail(),
        ));
    }

    if candidates.len() < MAX_CANDIDATE_SLOTS {
        candidates.push(CandidateSlot {
            entity_ord,
            source,
            score_millis,
            evidence_bits,
            evidence,
        });
        return;
    }

    let Some((weakest_ix, weakest_score)) = candidates
        .iter()
        .enumerate()
        .min_by_key(|(_, candidate)| candidate.score_millis)
        .map(|(index, candidate)| (index, candidate.score_millis))
    else {
        return;
    };

    if score_millis <= weakest_score {
        return;
    }

    candidates[weakest_ix] = CandidateSlot {
        entity_ord,
        source,
        score_millis,
        evidence_bits,
        evidence,
    };
}

fn best_candidate_summary(
    candidates: &[CandidateSlot],
    entity_ids: &[String],
) -> Option<BestCandidateSummary> {
    candidates
        .iter()
        .max_by(|left, right| {
            left.score_millis.cmp(&right.score_millis).then_with(|| {
                entity_ids[right.entity_ord as usize].cmp(&entity_ids[left.entity_ord as usize])
            })
        })
        .map(|candidate| BestCandidateSummary {
            entity_ord: candidate.entity_ord,
            source: candidate.source,
            score_millis: candidate.score_millis,
        })
}

fn candidate_slot_to_entity(candidate: CandidateSlot, entity_ids: &[String]) -> CandidateEntity {
    CandidateEntity {
        entity_id: entity_ids[candidate.entity_ord as usize].clone(),
        source: candidate.source.as_str().to_owned(),
        score_millis: candidate.score_millis,
        evidence: candidate.evidence.into_vec(),
    }
}

fn candidate_list_from_slots(
    mut candidates: SmallVec<[CandidateSlot; 6]>,
    entity_ids: &[String],
) -> Vec<CandidateEntity> {
    sort_candidate_slots(&mut candidates, entity_ids);
    candidates
        .into_iter()
        .map(|candidate| candidate_slot_to_entity(candidate, entity_ids))
        .collect()
}

fn stable_compact_slot_cmp(
    left: &CompactCandidateSlot,
    right: &CompactCandidateSlot,
    entity_ids: &[String],
) -> Ordering {
    right.score_millis.cmp(&left.score_millis).then_with(|| {
        entity_ids[left.entity_ord as usize].cmp(&entity_ids[right.entity_ord as usize])
    })
}

fn sort_compact_candidate_slots(
    candidates: &mut SmallVec<[CompactCandidateSlot; 6]>,
    entity_ids: &[String],
) {
    candidates.sort_by(|left, right| stable_compact_slot_cmp(left, right, entity_ids));
}

fn merge_compact_candidate_slot(
    candidates: &mut SmallVec<[CompactCandidateSlot; 6]>,
    entity_ord: u32,
    source: CandidateSourceKind,
    score_millis: i32,
    evidence_kind: CandidateEvidenceKind,
) {
    const MAX_CANDIDATE_SLOTS: usize = 8;
    let evidence_bits = evidence_kind.bit();

    if let Some(entry) = candidates
        .iter_mut()
        .find(|candidate| candidate.entity_ord == entity_ord)
    {
        if score_millis > entry.score_millis {
            entry.score_millis = score_millis;
            entry.source = source;
        }
        entry.evidence_bits |= evidence_bits;
        return;
    }

    if candidates.len() < MAX_CANDIDATE_SLOTS {
        candidates.push(CompactCandidateSlot {
            entity_ord,
            source,
            score_millis,
            evidence_bits,
        });
        return;
    }

    let Some((weakest_ix, weakest_score)) = candidates
        .iter()
        .enumerate()
        .min_by_key(|(_, candidate)| candidate.score_millis)
        .map(|(index, candidate)| (index, candidate.score_millis))
    else {
        return;
    };

    if score_millis <= weakest_score {
        return;
    }

    candidates[weakest_ix] = CompactCandidateSlot {
        entity_ord,
        source,
        score_millis,
        evidence_bits,
    };
}

fn best_compact_candidate_summary(
    candidates: &[CompactCandidateSlot],
    entity_ids: &[String],
) -> Option<CompactBestCandidateSummary> {
    candidates
        .iter()
        .max_by(|left, right| {
            left.score_millis.cmp(&right.score_millis).then_with(|| {
                entity_ids[right.entity_ord as usize].cmp(&entity_ids[left.entity_ord as usize])
            })
        })
        .map(|candidate| CompactBestCandidateSummary {
            entity_ord: candidate.entity_ord,
            source: candidate.source,
            score_millis: candidate.score_millis,
        })
}

fn compact_candidate_has_alias_signal(candidate: &CompactCandidateSlot) -> bool {
    matches!(
        candidate.source,
        CandidateSourceKind::KernelAlias
            | CandidateSourceKind::AliasLink
            | CandidateSourceKind::LocalSurface
    ) || candidate.evidence_bits
        & (CandidateEvidenceKind::Kernel.bit()
            | CandidateEvidenceKind::ResolverLink.bit()
            | CandidateEvidenceKind::LocalSurface.bit()
            | CandidateEvidenceKind::SurfaceCluster.bit())
        != 0
}

fn bump_small_count(counts: &mut SmallVec<[(u32, usize); 4]>, entity_ord: u32, delta: usize) {
    if let Some((_, count)) = counts
        .iter_mut()
        .find(|(existing_ord, _)| *existing_ord == entity_ord)
    {
        *count += delta;
    } else {
        counts.push((entity_ord, delta));
    }
}

fn intern_entity_ord(
    entity_id: &str,
    entity_ord_by_id: &mut FxHashMap<String, u32>,
    entity_ids: &mut Vec<String>,
    entity_kinds_by_ord: &mut Vec<Option<String>>,
    entity_memory: &NativeEntityMemory,
) -> u32 {
    if let Some(existing) = entity_ord_by_id.get(entity_id).copied() {
        return existing;
    }
    let ord = entity_ids.len() as u32;
    entity_ord_by_id.insert(entity_id.to_owned(), ord);
    entity_ids.push(entity_id.to_owned());
    entity_kinds_by_ord.push(entity_memory.entity_kinds.get(entity_id).cloned());
    ord
}

fn build_prepared_mentions_native(
    document: &IngestDocument,
    scan: &NativeScanRows,
    chunks: &[ChunkRecord],
) -> Vec<PreparedMention> {
    let mention_by_range = scan
        .mentions
        .iter()
        .enumerate()
        .map(|(index, mention)| ((mention.range.start, mention.range.end), index))
        .collect::<FxHashMap<_, _>>();
    let mut links_by_source = FxHashMap::<usize, SmallVec<[usize; 4]>>::default();
    for link in &scan.resolver_links {
        let Some(source_ix) = mention_by_range
            .get(&(link.source_range.start, link.source_range.end))
            .copied()
        else {
            continue;
        };
        let Some(target_range) = link.target_range else {
            continue;
        };
        let Some(target_ix) = mention_by_range
            .get(&(target_range.start, target_range.end))
            .copied()
        else {
            continue;
        };
        links_by_source
            .entry(source_ix)
            .or_default()
            .push(target_ix);
    }
    let mut entity_links_by_source =
        FxHashMap::<usize, SmallVec<[PreparedResolverEntityLink; 4]>>::default();
    for link in &scan.resolver_links {
        let Some(source_ix) = mention_by_range
            .get(&(link.source_range.start, link.source_range.end))
            .copied()
        else {
            continue;
        };
        let Some(target_entity) = link.target_entity.as_ref() else {
            continue;
        };
        let Some(entity_id) = entity_id_from_ref(document, target_entity) else {
            continue;
        };
        let (source, score_millis) = match link.link_kind {
            Some(ResolverLinkKind::Pronoun) => ("pronoun_link", 1150),
            Some(ResolverLinkKind::AliasCandidate) => ("alias_link", 950),
            None => ("alias_link", 850),
        };
        entity_links_by_source
            .entry(source_ix)
            .or_default()
            .push(PreparedResolverEntityLink {
                entity_id: entity_id.0,
                source,
                score_millis,
                evidence_detail: match link.link_kind {
                    Some(ResolverLinkKind::Pronoun) => "Pronoun",
                    Some(ResolverLinkKind::AliasCandidate) => "AliasCandidate",
                    None => "Unknown",
                },
            });
    }

    let mut prepared = Vec::with_capacity(scan.mentions.len());
    let mut chunk_cursor = 0usize;
    for (mention_ix, mention) in scan.mentions.iter().enumerate() {
        while chunk_cursor < chunks.len() && chunks[chunk_cursor].range.end <= mention.range.start {
            chunk_cursor += 1;
        }
        let chunk_ix = chunks
            .get(chunk_cursor)
            .filter(|chunk| range_contains(chunk.range, mention.range))
            .map(|_| chunk_cursor as u32);
        prepared.push(PreparedMention {
            mention_ix,
            surface_ord: scan.mention_surface_ords[mention_ix],
            chunk_ix,
            linked_mentions: links_by_source.remove(&mention_ix).unwrap_or_default(),
            resolver_entity_links: entity_links_by_source
                .remove(&mention_ix)
                .unwrap_or_default(),
        });
    }
    prepared
}

fn build_prepared_mentions(
    document: &IngestDocument,
    mentions: &[MentionSpan],
    resolver_links: &[ResolverLink],
    chunks: &[ChunkRecord],
) -> (Vec<PreparedMention>, Vec<String>) {
    let mention_by_range = mentions
        .iter()
        .enumerate()
        .map(|(index, mention)| ((mention.range.start, mention.range.end), index))
        .collect::<FxHashMap<_, _>>();
    let mut links_by_source = FxHashMap::<usize, SmallVec<[usize; 4]>>::default();
    for link in resolver_links {
        let Some(source_ix) = mention_by_range
            .get(&(link.source_range.start, link.source_range.end))
            .copied()
        else {
            continue;
        };
        let Some(target_range) = link.target_range else {
            continue;
        };
        let Some(target_ix) = mention_by_range
            .get(&(target_range.start, target_range.end))
            .copied()
        else {
            continue;
        };
        links_by_source
            .entry(source_ix)
            .or_default()
            .push(target_ix);
    }
    let mut entity_links_by_source =
        FxHashMap::<usize, SmallVec<[PreparedResolverEntityLink; 4]>>::default();
    for link in resolver_links {
        let Some(source_ix) = mention_by_range
            .get(&(link.source_range.start, link.source_range.end))
            .copied()
        else {
            continue;
        };
        let Some(target_entity) = link.target_entity.as_ref() else {
            continue;
        };
        let Some(entity_id) = entity_id_from_ref(document, target_entity) else {
            continue;
        };
        let (source, score_millis) = match link.link_kind {
            Some(ResolverLinkKind::Pronoun) => ("pronoun_link", 1150),
            Some(ResolverLinkKind::AliasCandidate) => ("alias_link", 950),
            None => ("alias_link", 850),
        };
        entity_links_by_source
            .entry(source_ix)
            .or_default()
            .push(PreparedResolverEntityLink {
                entity_id: entity_id.0,
                source,
                score_millis,
                evidence_detail: match link.link_kind {
                    Some(ResolverLinkKind::Pronoun) => "Pronoun",
                    Some(ResolverLinkKind::AliasCandidate) => "AliasCandidate",
                    None => "Unknown",
                },
            });
    }

    let mut prepared = Vec::with_capacity(mentions.len());
    let mut surface_ord_by_value = FxHashMap::<String, u32>::default();
    let mut surface_values = Vec::<String>::new();
    let mut chunk_cursor = 0usize;
    for (mention_ix, mention) in mentions.iter().enumerate() {
        while chunk_cursor < chunks.len() && chunks[chunk_cursor].range.end <= mention.range.start {
            chunk_cursor += 1;
        }
        let normalized = normalize_surface(&mention.surface);
        let surface_ord = if let Some(existing) = surface_ord_by_value.get(&normalized).copied() {
            existing
        } else {
            let ord = surface_values.len() as u32;
            surface_ord_by_value.insert(normalized.clone(), ord);
            surface_values.push(normalized);
            ord
        };
        let chunk_ix = chunks
            .get(chunk_cursor)
            .filter(|chunk| range_contains(chunk.range, mention.range))
            .map(|_| chunk_cursor as u32);
        prepared.push(PreparedMention {
            mention_ix,
            surface_ord,
            chunk_ix,
            linked_mentions: links_by_source.remove(&mention_ix).unwrap_or_default(),
            resolver_entity_links: entity_links_by_source
                .remove(&mention_ix)
                .unwrap_or_default(),
        });
    }
    (prepared, surface_values)
}

fn build_kernel_mention_resolution_map(
    document: &IngestDocument,
    mention_count: usize,
    entity_memory: &NativeEntityMemory,
) -> Vec<Option<String>> {
    let mut resolved = vec![None; mention_count];
    let prefix = format!("mention::{}:", document.document_id.0);
    for (mention_id, entity_id) in &entity_memory.entity_index.mention_entities {
        if !mention_id.starts_with(&prefix) {
            continue;
        }
        let Some(index_text) = mention_id[prefix.len()..].parse::<usize>().ok() else {
            continue;
        };
        if let Some(slot) = resolved.get_mut(index_text) {
            *slot = Some(entity_id.clone());
        }
    }
    resolved
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResolveDetailMode {
    Compact,
    Detailed,
}

fn record_resolution_diagnostic(
    mode: ResolveDetailMode,
    diagnostics: &mut Vec<Diagnostic>,
    diagnostic_counts: &mut FxHashMap<&'static str, usize>,
    code: &'static str,
    message: impl FnOnce() -> String,
) {
    match mode {
        ResolveDetailMode::Detailed => diagnostics.push(Diagnostic {
            code: code.to_owned(),
            message: message(),
        }),
        ResolveDetailMode::Compact => {
            diagnostic_counts
                .entry(code)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }
}

fn append_resolution_diagnostic_summaries(
    diagnostics: &mut Vec<Diagnostic>,
    diagnostic_counts: FxHashMap<&'static str, usize>,
) {
    const ORDER: [(&str, &str); 8] = [
        (
            "er_known_seed_match",
            "Mentions resolved from explicit known-entity seeds.",
        ),
        (
            "er_kernel_alias_match",
            "Mentions resolved from kernel alias or prior resolution memory.",
        ),
        (
            "er_pronoun_link_match",
            "Pronouns resolved from explicit resolver links.",
        ),
        (
            "er_alias_link_match",
            "Mentions resolved from alias-style or linked local evidence.",
        ),
        (
            "er_collective_merge",
            "Mentions resolved after collective surface/link reinforcement.",
        ),
        (
            "er_new_speculative_entity",
            "Mentions promoted to new speculative canonical entities.",
        ),
        (
            "er_ambiguous_resolution",
            "Mentions remained ambiguous after collective resolution.",
        ),
        (
            "er_alias_rejected_low_margin",
            "Alias confirmations were rejected because evidence or margin was too weak.",
        ),
    ];

    for (code, summary) in ORDER {
        let Some(count) = diagnostic_counts.get(code).copied() else {
            continue;
        };
        diagnostics.push(Diagnostic {
            code: code.to_owned(),
            message: format!("{} Count={}.", summary, count),
        });
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn resolve_mentions(
    document: &IngestDocument,
    scan: &ScanArtifact,
    _structure: &StructureArtifact,
    chunks: &[ChunkRecord],
    entity_memory: &NativeEntityMemory,
) -> (
    Vec<DetailedMentionResolution>,
    Vec<ResolvedMention>,
    Vec<AliasConfirmation>,
    Vec<Diagnostic>,
) {
    let (_compact, detailed, resolved_mentions, alias_confirmations, _summary, diagnostics) =
        resolve_mentions_with_mode(
            document,
            &scan.mentions,
            &scan.resolver_links,
            chunks,
            entity_memory,
            ResolveDetailMode::Detailed,
        );
    (
        detailed,
        resolved_mentions,
        alias_confirmations,
        diagnostics,
    )
}

fn resolve_mentions_compact_native(
    document: &IngestDocument,
    scan: &NativeScanRows,
    coref: &NativeCorefRows,
    chunks: &[ChunkRecord],
    entity_memory: &NativeEntityMemory,
) -> (
    Vec<CompactResolutionOrd>,
    Vec<AliasConfirmationOrd>,
    NativeErSummary,
    Vec<Diagnostic>,
    Vec<String>,
    FxHashMap<String, u32>,
) {
    let progress = native_progress_enabled();
    let phase_started = Instant::now();
    let prepared = build_prepared_mentions_native(document, scan, chunks);
    let kernel_resolved_entities =
        build_kernel_mention_resolution_map(document, prepared.len(), entity_memory);
    if progress {
        eprintln!(
            "[runtime-ingest] resolve_subphase=prepare_mentions document_id={} wall_ms={} mentions={} surfaces={}",
            document.document_id.0,
            phase_started.elapsed().as_millis(),
            prepared.len(),
            scan.surface_atoms.len(),
        );
    }

    let phase_started = Instant::now();
    let mut diagnostics = Vec::<Diagnostic>::new();
    let mut diagnostic_counts = FxHashMap::<&'static str, usize>::default();
    let mut entity_ord_by_id = FxHashMap::<String, u32>::default();
    let mut entity_ids = Vec::<String>::new();
    let mut entity_kinds_by_ord = Vec::<Option<String>>::new();

    for mention in &scan.mentions {
        if let Some(MentionEntityRef::Known(entity_id)) = mention.entity_ref.as_ref() {
            intern_entity_ord(
                &entity_id.0,
                &mut entity_ord_by_id,
                &mut entity_ids,
                &mut entity_kinds_by_ord,
                entity_memory,
            );
        }
    }
    for binding in &scan.entity_library.surface_bindings {
        match binding.entity_ref.as_ref() {
            Some(MentionEntityRef::Known(entity_id)) => {
                intern_entity_ord(
                    &entity_id.0,
                    &mut entity_ord_by_id,
                    &mut entity_ids,
                    &mut entity_kinds_by_ord,
                    entity_memory,
                );
            }
            Some(MentionEntityRef::Speculative(speculative)) => {
                let speculative_id = format!(
                    "{}::{}",
                    document.document_id.0,
                    speculative.replace(' ', "_")
                );
                intern_entity_ord(
                    &speculative_id,
                    &mut entity_ord_by_id,
                    &mut entity_ids,
                    &mut entity_kinds_by_ord,
                    entity_memory,
                );
            }
            None => {}
        }
    }
    for entity_id in kernel_resolved_entities.iter().flatten() {
        intern_entity_ord(
            entity_id,
            &mut entity_ord_by_id,
            &mut entity_ids,
            &mut entity_kinds_by_ord,
            entity_memory,
        );
    }
    for prepared_mention in &prepared {
        for link in &prepared_mention.resolver_entity_links {
            intern_entity_ord(
                &link.entity_id,
                &mut entity_ord_by_id,
                &mut entity_ids,
                &mut entity_kinds_by_ord,
                entity_memory,
            );
        }
    }
    for surface in &scan.surface_atoms {
        let normalized = surface.normalized.as_ref();
        if let Some(kernel_candidates) = entity_memory.entity_index.alias_candidates.get(normalized)
        {
            for candidate in kernel_candidates {
                intern_entity_ord(
                    &candidate.entity_id,
                    &mut entity_ord_by_id,
                    &mut entity_ids,
                    &mut entity_kinds_by_ord,
                    entity_memory,
                );
            }
        }
        if !normalized.is_empty() && !is_pronoun(normalized) {
            let speculative_id = format!(
                "{}::{}",
                document.document_id.0,
                normalized.replace(' ', "_")
            );
            intern_entity_ord(
                &speculative_id,
                &mut entity_ord_by_id,
                &mut entity_ids,
                &mut entity_kinds_by_ord,
                entity_memory,
            );
        }
    }
    if progress {
        eprintln!(
            "[runtime-ingest] resolve_subphase=intern_entities document_id={} wall_ms={} entities={}",
            document.document_id.0,
            phase_started.elapsed().as_millis(),
            entity_ids.len(),
        );
    }

    let phase_started = Instant::now();
    let surface_count = scan.surface_atoms.len();
    let mut surface_kernel_candidates =
        vec![SmallVec::<[CompactCandidateSlot; 4]>::new(); surface_count];
    let mut surface_speculative_ord = vec![None; surface_count];
    let mut surface_library_entity_ord = vec![None; surface_count];
    let mut surface_pronouns = vec![false; surface_count];
    let mut surface_known_counts = vec![SmallVec::<[(u32, usize); 4]>::new(); surface_count];
    let mut resolver_links_by_mention =
        vec![SmallVec::<[CompactResolverEntityLink; 4]>::new(); prepared.len()];
    let mut kernel_resolved_ord_by_mention = vec![None; prepared.len()];
    let mut coref_rep_seed_ord_by_mention = vec![None; prepared.len()];

    for (surface_ord, surface) in scan.surface_atoms.iter().enumerate() {
        let normalized = surface.normalized.as_ref();
        surface_pronouns[surface_ord] = surface.is_pronoun;
        if let Some(entity_ref) = scan.entity_library.surface_bindings[surface_ord]
            .entity_ref
            .as_ref()
        {
            let entity_id = match entity_ref {
                MentionEntityRef::Known(entity_id) => entity_id.0.clone(),
                MentionEntityRef::Speculative(speculative) => {
                    format!(
                        "{}::{}",
                        document.document_id.0,
                        speculative.replace(' ', "_")
                    )
                }
            };
            surface_library_entity_ord[surface_ord] = entity_ord_by_id.get(&entity_id).copied();
        }
        if let Some(kernel_candidates) = entity_memory.entity_index.alias_candidates.get(normalized)
        {
            for candidate in kernel_candidates {
                let relation = candidate.relation_type.as_deref().unwrap_or("kernel");
                let (source, bonus) = match relation {
                    "alias_of" => (CandidateSourceKind::KernelAlias, 100),
                    "resolved_to" => (CandidateSourceKind::KernelResolved, 120),
                    "candidate_same_as" => (CandidateSourceKind::KernelCandidate, 0),
                    _ => (CandidateSourceKind::KernelAlias, 0),
                };
                let entity_ord = entity_ord_by_id[&candidate.entity_id];
                surface_kernel_candidates[surface_ord].push(CompactCandidateSlot {
                    entity_ord,
                    source,
                    score_millis: ((candidate.score * 1000.0).round() as i32) + 700 + bonus,
                    evidence_bits: CandidateEvidenceKind::Kernel.bit(),
                });
            }
        }
        if surface_library_entity_ord[surface_ord].is_none()
            && !surface_pronouns[surface_ord]
            && !normalized.is_empty()
        {
            let speculative_id = format!(
                "{}::{}",
                document.document_id.0,
                normalized.replace(' ', "_")
            );
            surface_speculative_ord[surface_ord] = entity_ord_by_id.get(&speculative_id).copied();
        }
    }

    for (mention_ix, prepared_mention) in prepared.iter().enumerate() {
        let mention = &scan.mentions[prepared_mention.mention_ix];
        if let Some(MentionEntityRef::Known(entity_id)) = mention.entity_ref.as_ref() {
            let entity_ord = entity_ord_by_id[&entity_id.0];
            bump_small_count(
                &mut surface_known_counts[prepared_mention.surface_ord as usize],
                entity_ord,
                1,
            );
        }
        if let Some(entity_id) = kernel_resolved_entities
            .get(prepared_mention.mention_ix)
            .and_then(|value| value.as_ref())
        {
            kernel_resolved_ord_by_mention[mention_ix] = entity_ord_by_id.get(entity_id).copied();
        }
        if let Some(rep_ix) = coref
            .representative_by_mention
            .get(prepared_mention.mention_ix)
            .copied()
            .flatten()
        {
            if rep_ix != prepared_mention.mention_ix {
                if let Some(MentionEntityRef::Known(entity_id)) =
                    scan.mentions[rep_ix].entity_ref.as_ref()
                {
                    coref_rep_seed_ord_by_mention[mention_ix] =
                        entity_ord_by_id.get(&entity_id.0).copied();
                }
            }
        }
        let mut compact_links = SmallVec::<[CompactResolverEntityLink; 4]>::new();
        for link in &prepared_mention.resolver_entity_links {
            let source = match link.source {
                "pronoun_link" => CandidateSourceKind::PronounLink,
                _ => CandidateSourceKind::AliasLink,
            };
            compact_links.push(CompactResolverEntityLink {
                entity_ord: entity_ord_by_id[&link.entity_id],
                source,
                score_millis: link.score_millis,
            });
        }
        resolver_links_by_mention[mention_ix] = compact_links;
    }
    if progress {
        eprintln!(
            "[runtime-ingest] resolve_subphase=prepare_templates document_id={} wall_ms={} surface_count={}",
            document.document_id.0,
            phase_started.elapsed().as_millis(),
            scan.surface_atoms.len(),
        );
    }

    let phase_started = Instant::now();
    let mut mention_states = vec![SmallVec::<[CompactCandidateSlot; 6]>::new(); prepared.len()];
    let mut base_best = vec![None; prepared.len()];
    let mut surface_support = vec![SmallVec::<[(u32, usize); 4]>::new(); surface_count];

    for (index, prepared_mention) in prepared.iter().enumerate() {
        let mention = &scan.mentions[prepared_mention.mention_ix];
        let normalized = scan.surface_atoms[prepared_mention.surface_ord as usize]
            .normalized
            .as_ref();
        let candidates = &mut mention_states[index];

        if let Some(MentionEntityRef::Known(entity_id)) = mention.entity_ref.as_ref() {
            merge_compact_candidate_slot(
                candidates,
                entity_ord_by_id[&entity_id.0],
                CandidateSourceKind::Seed,
                1800,
                CandidateEvidenceKind::Seed,
            );
        }
        if let Some(entity_ord) = surface_library_entity_ord[prepared_mention.surface_ord as usize]
        {
            let score = match scan.entity_library.surface_bindings
                [prepared_mention.surface_ord as usize]
                .entity_ref
                .as_ref()
            {
                Some(MentionEntityRef::Known(_)) => 1560,
                Some(MentionEntityRef::Speculative(_)) => 760,
                None => 0,
            };
            if score > 0 {
                merge_compact_candidate_slot(
                    candidates,
                    entity_ord,
                    CandidateSourceKind::LocalSurface,
                    score,
                    CandidateEvidenceKind::LocalSurface,
                );
            }
        }
        for candidate in &surface_kernel_candidates[prepared_mention.surface_ord as usize] {
            merge_compact_candidate_slot(
                candidates,
                candidate.entity_ord,
                candidate.source,
                candidate.score_millis,
                CandidateEvidenceKind::Kernel,
            );
        }
        if let Some(entity_ord) = kernel_resolved_ord_by_mention[index] {
            merge_compact_candidate_slot(
                candidates,
                entity_ord,
                CandidateSourceKind::KernelResolved,
                1820,
                CandidateEvidenceKind::Kernel,
            );
        }
        if let Some(entity_ord) = coref_rep_seed_ord_by_mention[index] {
            merge_compact_candidate_slot(
                candidates,
                entity_ord,
                CandidateSourceKind::CorefCluster,
                1260,
                CandidateEvidenceKind::Coref,
            );
        }
        for link in &resolver_links_by_mention[index] {
            merge_compact_candidate_slot(
                candidates,
                link.entity_ord,
                link.source,
                link.score_millis,
                CandidateEvidenceKind::ResolverLink,
            );
        }
        if !normalized.is_empty() {
            for (entity_ord, support) in
                &surface_known_counts[prepared_mention.surface_ord as usize]
            {
                let own_known_match = matches!(
                    mention.entity_ref.as_ref(),
                    Some(MentionEntityRef::Known(known_id))
                        if entity_ord_by_id.get(&known_id.0).copied() == Some(*entity_ord)
                            && *support == 1
                );
                if own_known_match {
                    continue;
                }
                merge_compact_candidate_slot(
                    candidates,
                    *entity_ord,
                    CandidateSourceKind::LocalSurface,
                    900,
                    CandidateEvidenceKind::LocalSurface,
                );
            }
        }
        if let Some(entity_ord) = surface_speculative_ord[prepared_mention.surface_ord as usize] {
            merge_compact_candidate_slot(
                candidates,
                entity_ord,
                CandidateSourceKind::NewSpeculative,
                420,
                CandidateEvidenceKind::NewSpeculative,
            );
        }
        for candidate in candidates.iter() {
            if candidate.source.is_strong() {
                bump_small_count(
                    &mut surface_support[prepared_mention.surface_ord as usize],
                    candidate.entity_ord,
                    1,
                );
            }
        }
        sort_compact_candidate_slots(candidates, &entity_ids);
        base_best[index] = best_compact_candidate_summary(candidates, &entity_ids);
    }
    const MAX_SURFACE_CLUSTER_CANDIDATES: usize = 4;
    for (surface_ord, support_counts) in surface_support.iter_mut().enumerate() {
        if surface_pronouns[surface_ord] {
            support_counts.clear();
            continue;
        }
        support_counts.retain(|(_, support)| *support >= 2);
        support_counts.sort_by(|left, right| {
            right
                .1
                .cmp(&left.1)
                .then_with(|| entity_ids[left.0 as usize].cmp(&entity_ids[right.0 as usize]))
        });
        if support_counts.len() > MAX_SURFACE_CLUSTER_CANDIDATES {
            support_counts.truncate(MAX_SURFACE_CLUSTER_CANDIDATES);
        }
    }
    if progress {
        eprintln!(
            "[runtime-ingest] resolve_subphase=base_pass document_id={} wall_ms={}",
            document.document_id.0,
            phase_started.elapsed().as_millis(),
        );
    }

    let phase_started = Instant::now();
    let entity_normalized = entity_ids
        .iter()
        .map(|entity_id| normalize_surface(entity_id))
        .collect::<Vec<_>>();

    let mut resolutions = Vec::with_capacity(prepared.len());
    let mut alias_confirmations = Vec::<AliasConfirmationOrd>::new();
    let mut er_summary = NativeErSummary::default();

    for (index, prepared_mention) in prepared.iter().enumerate() {
        let mention = &scan.mentions[prepared_mention.mention_ix];
        let normalized = scan.surface_atoms[prepared_mention.surface_ord as usize]
            .normalized
            .as_ref();
        let candidates = &mut mention_states[index];

        for (entity_ord, support) in &surface_support[prepared_mention.surface_ord as usize] {
            merge_compact_candidate_slot(
                candidates,
                *entity_ord,
                CandidateSourceKind::LocalSurface,
                780 + (*support as i32 * 80),
                CandidateEvidenceKind::SurfaceCluster,
            );
        }
        for linked_ix in &prepared_mention.linked_mentions {
            if let Some(candidate) = base_best.get(*linked_ix).and_then(|candidate| *candidate) {
                merge_compact_candidate_slot(
                    candidates,
                    candidate.entity_ord,
                    candidate.source,
                    candidate.score_millis
                        + if candidate.source == CandidateSourceKind::PronounLink {
                            180
                        } else {
                            120
                        },
                    CandidateEvidenceKind::LinkedMention,
                );
            }
        }
        if let Some(rep_ix) = coref
            .representative_by_mention
            .get(prepared_mention.mention_ix)
            .copied()
            .flatten()
        {
            if rep_ix != prepared_mention.mention_ix {
                if let Some(candidate) = base_best.get(rep_ix).and_then(|candidate| *candidate) {
                    merge_compact_candidate_slot(
                        candidates,
                        candidate.entity_ord,
                        CandidateSourceKind::CorefCluster,
                        candidate.score_millis + 120,
                        CandidateEvidenceKind::Coref,
                    );
                }
            }
        }
        for linked_ix in coref
            .candidate_links_by_mention
            .get(prepared_mention.mention_ix)
            .into_iter()
            .flat_map(|values| values.iter())
        {
            if let Some(candidate) = base_best.get(*linked_ix).and_then(|candidate| *candidate) {
                merge_compact_candidate_slot(
                    candidates,
                    candidate.entity_ord,
                    CandidateSourceKind::CorefCluster,
                    candidate.score_millis,
                    CandidateEvidenceKind::Coref,
                );
            }
        }

        for candidate in candidates.iter_mut() {
            if let Some(kind) = mention.kind.as_ref() {
                if let Some(Some(existing_kind)) =
                    entity_kinds_by_ord.get(candidate.entity_ord as usize)
                {
                    if existing_kind != entity_kind_name(kind) {
                        candidate.score_millis -= 220;
                        candidate.evidence_bits |= CandidateEvidenceKind::KindPenalty.bit();
                    }
                }
            }
            if candidate.source == CandidateSourceKind::NewSpeculative && mention.confidence >= 0.8
            {
                candidate.score_millis += 180;
            }
            if prepared_mention.chunk_ix.is_some() {
                candidate.score_millis += 20;
            }
        }

        sort_compact_candidate_slots(candidates, &entity_ids);
        let top = candidates.first().copied();
        let runner_up = candidates.get(1).copied();
        let top_score = top
            .map(|candidate| candidate.score_millis)
            .unwrap_or_default();
        let margin = top_score.saturating_sub(
            runner_up
                .map(|candidate| candidate.score_millis)
                .unwrap_or_default(),
        );

        let pronoun = surface_pronouns[prepared_mention.surface_ord as usize];
        let resolved_ord = top.and_then(|candidate| {
            let threshold = if pronoun { 1100 } else { 900 };
            let margin_threshold = if pronoun { 220 } else { 180 };
            let speculative = candidate.source == CandidateSourceKind::NewSpeculative;
            if candidate.score_millis >= threshold
                && margin >= margin_threshold
                && (!speculative || candidate.score_millis >= 700)
            {
                Some(candidate.entity_ord)
            } else {
                None
            }
        });

        let (kind, diagnostic_code) = if let Some(top_candidate) = top {
            if resolved_ord.is_some() {
                er_summary.resolved_count += 1;
                (
                    CompactResolutionKind::Resolved,
                    match top_candidate.source {
                        CandidateSourceKind::Seed => "er_known_seed_match",
                        CandidateSourceKind::KernelAlias | CandidateSourceKind::KernelResolved => {
                            "er_kernel_alias_match"
                        }
                        CandidateSourceKind::PronounLink => "er_pronoun_link_match",
                        CandidateSourceKind::CorefCluster => "er_collective_merge",
                        CandidateSourceKind::AliasLink | CandidateSourceKind::LocalSurface => {
                            "er_alias_link_match"
                        }
                        CandidateSourceKind::NewSpeculative => "er_new_speculative_entity",
                        _ => "er_collective_merge",
                    },
                )
            } else {
                er_summary.ambiguous_count += 1;
                (CompactResolutionKind::Ambiguous, "er_ambiguous_resolution")
            }
        } else {
            er_summary.unresolved_count += 1;
            (CompactResolutionKind::Unresolved, "")
        };

        if !diagnostic_code.is_empty() {
            record_resolution_diagnostic(
                ResolveDetailMode::Compact,
                &mut diagnostics,
                &mut diagnostic_counts,
                diagnostic_code,
                || String::new(),
            );
        }

        if let (CompactResolutionKind::Resolved, Some(entity_ord), Some(top_candidate)) =
            (kind, resolved_ord, top)
        {
            let entity_id = &entity_ids[entity_ord as usize];
            if !normalized.is_empty()
                && !pronoun
                && normalized != entity_normalized[entity_ord as usize]
                && top_candidate.source != CandidateSourceKind::Seed
                && top_candidate.source != CandidateSourceKind::KernelResolved
                && top_candidate.source != CandidateSourceKind::PronounLink
                && compact_candidate_has_alias_signal(&top_candidate)
                && top_score >= 1000
                && margin >= 260
                && !entity_memory
                    .known_aliases
                    .contains(&(entity_id.clone(), normalized.to_owned()))
            {
                alias_confirmations.push(AliasConfirmationOrd {
                    alias_surface: mention.surface.clone(),
                    normalized: normalized.to_owned(),
                    entity_ord,
                    confidence_millis: top_score.max(0) as u32,
                    mention_index: prepared_mention.mention_ix,
                });
            } else if !normalized.is_empty()
                && normalized != entity_normalized[entity_ord as usize]
                && !entity_memory
                    .known_aliases
                    .contains(&(entity_id.clone(), normalized.to_owned()))
            {
                record_resolution_diagnostic(
                    ResolveDetailMode::Compact,
                    &mut diagnostics,
                    &mut diagnostic_counts,
                    "er_alias_rejected_low_margin",
                    || String::new(),
                );
            }
        } else if top.is_some() && margin < 180 {
            record_resolution_diagnostic(
                ResolveDetailMode::Compact,
                &mut diagnostics,
                &mut diagnostic_counts,
                "er_alias_rejected_low_margin",
                || String::new(),
            );
        }

        resolutions.push(CompactResolutionOrd {
            mention_index: prepared_mention.mention_ix,
            entity_ord: resolved_ord,
            chunk_index: prepared_mention.chunk_ix,
        });
    }

    alias_confirmations.sort_by(|left, right| {
        entity_ids[left.entity_ord as usize]
            .cmp(&entity_ids[right.entity_ord as usize])
            .then_with(|| left.normalized.cmp(&right.normalized))
            .then_with(|| left.mention_index.cmp(&right.mention_index))
    });
    alias_confirmations.dedup_by(|left, right| {
        left.entity_ord == right.entity_ord && left.normalized == right.normalized
    });
    er_summary.alias_confirmation_count = alias_confirmations.len();
    append_resolution_diagnostic_summaries(&mut diagnostics, diagnostic_counts);
    if progress {
        eprintln!(
            "[runtime-ingest] resolve_subphase=decision_pass document_id={} wall_ms={} alias_confirmations={}",
            document.document_id.0,
            phase_started.elapsed().as_millis(),
            er_summary.alias_confirmation_count,
        );
    }

    (
        resolutions,
        alias_confirmations,
        er_summary,
        diagnostics,
        entity_ids,
        entity_ord_by_id,
    )
}

fn resolve_mentions_with_mode(
    document: &IngestDocument,
    mentions: &[MentionSpan],
    resolver_links: &[ResolverLink],
    chunks: &[ChunkRecord],
    entity_memory: &NativeEntityMemory,
    mode: ResolveDetailMode,
) -> (
    Vec<CompactResolutionRow>,
    Vec<DetailedMentionResolution>,
    Vec<ResolvedMention>,
    Vec<AliasConfirmation>,
    NativeErSummary,
    Vec<Diagnostic>,
) {
    let (prepared, surface_values) =
        build_prepared_mentions(document, mentions, resolver_links, chunks);
    let kernel_resolved_entities =
        build_kernel_mention_resolution_map(document, prepared.len(), entity_memory);
    let mut diagnostics = Vec::<Diagnostic>::new();
    let mut diagnostic_counts = FxHashMap::<&'static str, usize>::default();
    let mut mention_states = vec![MentionState::default(); prepared.len()];
    let mut base_best = vec![None; prepared.len()];
    let mut surface_support = FxHashMap::<u32, FxHashMap<u32, usize>>::default();
    let mut surface_known_counts = FxHashMap::<u32, FxHashMap<u32, usize>>::default();
    let mut entity_ord_by_id = FxHashMap::<String, u32>::default();
    let mut entity_ids = Vec::<String>::new();
    let mut entity_kinds_by_ord = Vec::<Option<String>>::new();

    for prepared_mention in &prepared {
        let mention = &mentions[prepared_mention.mention_ix];
        if let Some(MentionEntityRef::Known(entity_id)) = mention.entity_ref.as_ref() {
            let entity_ord = intern_entity_ord(
                &entity_id.0,
                &mut entity_ord_by_id,
                &mut entity_ids,
                &mut entity_kinds_by_ord,
                entity_memory,
            );
            surface_known_counts
                .entry(prepared_mention.surface_ord)
                .or_default()
                .entry(entity_ord)
                .and_modify(|count| *count += 1)
                .or_insert(1);
        }
    }

    for (index, prepared_mention) in prepared.iter().enumerate() {
        let mention = &mentions[prepared_mention.mention_ix];
        let normalized = &surface_values[prepared_mention.surface_ord as usize];
        let candidates = &mut mention_states[index].candidates;

        if let Some(MentionEntityRef::Known(entity_id)) = mention.entity_ref.as_ref() {
            let entity_ord = intern_entity_ord(
                &entity_id.0,
                &mut entity_ord_by_id,
                &mut entity_ids,
                &mut entity_kinds_by_ord,
                entity_memory,
            );
            merge_candidate_slot(
                candidates,
                entity_ord,
                CandidateSourceKind::Seed,
                1800,
                CandidateEvidenceKind::Seed,
                mode,
                || mention.surface.clone(),
            );
        }

        if let Some(kernel_candidates) = entity_memory.entity_index.alias_candidates.get(normalized)
        {
            for candidate in kernel_candidates {
                let relation = candidate.relation_type.as_deref().unwrap_or("kernel");
                let (source, bonus) = match relation {
                    "alias_of" => (CandidateSourceKind::KernelAlias, 100),
                    "resolved_to" => (CandidateSourceKind::KernelResolved, 120),
                    "candidate_same_as" => (CandidateSourceKind::KernelCandidate, 0),
                    _ => (CandidateSourceKind::KernelAlias, 0),
                };
                let entity_ord = intern_entity_ord(
                    &candidate.entity_id,
                    &mut entity_ord_by_id,
                    &mut entity_ids,
                    &mut entity_kinds_by_ord,
                    entity_memory,
                );
                merge_candidate_slot(
                    candidates,
                    entity_ord,
                    source,
                    ((candidate.score * 1000.0).round() as i32) + 700 + bonus,
                    CandidateEvidenceKind::Kernel,
                    mode,
                    || relation.to_owned(),
                );
            }
        }

        if let Some(entity_id) = kernel_resolved_entities
            .get(prepared_mention.mention_ix)
            .and_then(|value| value.as_ref())
        {
            let entity_ord = intern_entity_ord(
                entity_id,
                &mut entity_ord_by_id,
                &mut entity_ids,
                &mut entity_kinds_by_ord,
                entity_memory,
            );
            merge_candidate_slot(
                candidates,
                entity_ord,
                CandidateSourceKind::KernelResolved,
                1820,
                CandidateEvidenceKind::Kernel,
                mode,
                || "resolved_to".to_owned(),
            );
        }

        for link in &prepared_mention.resolver_entity_links {
            let entity_ord = intern_entity_ord(
                &link.entity_id,
                &mut entity_ord_by_id,
                &mut entity_ids,
                &mut entity_kinds_by_ord,
                entity_memory,
            );
            let source = match link.source {
                "pronoun_link" => CandidateSourceKind::PronounLink,
                _ => CandidateSourceKind::AliasLink,
            };
            merge_candidate_slot(
                candidates,
                entity_ord,
                source,
                link.score_millis,
                CandidateEvidenceKind::ResolverLink,
                mode,
                || link.evidence_detail.to_owned(),
            );
        }

        if !normalized.is_empty() {
            if let Some(surface_entities) = surface_known_counts.get(&prepared_mention.surface_ord)
            {
                for (entity_ord, support) in surface_entities {
                    let own_known_match = matches!(
                        mention.entity_ref.as_ref(),
                        Some(MentionEntityRef::Known(known_id))
                            if entity_ord_by_id.get(&known_id.0).copied() == Some(*entity_ord)
                                && *support == 1
                    );
                    if own_known_match {
                        continue;
                    }
                    merge_candidate_slot(
                        candidates,
                        *entity_ord,
                        CandidateSourceKind::LocalSurface,
                        900,
                        CandidateEvidenceKind::LocalSurface,
                        mode,
                        || normalized.clone(),
                    );
                }
            }
        }

        if !is_pronoun(normalized) && !normalized.is_empty() {
            let speculative_id = speculative_entity_id(&document.document_id, &mention.surface);
            let entity_ord = intern_entity_ord(
                &speculative_id,
                &mut entity_ord_by_id,
                &mut entity_ids,
                &mut entity_kinds_by_ord,
                entity_memory,
            );
            merge_candidate_slot(
                candidates,
                entity_ord,
                CandidateSourceKind::NewSpeculative,
                420,
                CandidateEvidenceKind::NewSpeculative,
                mode,
                || normalized.clone(),
            );
        }

        for candidate in candidates.iter() {
            if candidate.source.is_strong() {
                surface_support
                    .entry(prepared_mention.surface_ord)
                    .or_default()
                    .entry(candidate.entity_ord)
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }
        sort_candidate_slots(candidates, &entity_ids);
        base_best[index] = best_candidate_summary(candidates, &entity_ids);
    }

    let mut resolutions = Vec::with_capacity(prepared.len());
    let mut detailed_resolutions = Vec::with_capacity(prepared.len());
    let mut resolved_mentions = Vec::with_capacity(prepared.len());
    let mut alias_confirmations = Vec::<AliasConfirmation>::new();
    let mut er_summary = NativeErSummary::default();

    for (index, prepared_mention) in prepared.iter().enumerate() {
        let mention = &mentions[prepared_mention.mention_ix];
        let normalized = &surface_values[prepared_mention.surface_ord as usize];
        let candidates = &mut mention_states[index].candidates;
        if let Some(surface_entities) = surface_support.get(&prepared_mention.surface_ord) {
            for (entity_ord, support) in surface_entities {
                merge_candidate_slot(
                    candidates,
                    *entity_ord,
                    CandidateSourceKind::LocalSurface,
                    780 + (*support as i32 * 80),
                    CandidateEvidenceKind::SurfaceCluster,
                    mode,
                    || support.to_string(),
                );
            }
        }
        for linked_ix in &prepared_mention.linked_mentions {
            if let Some(candidate) = base_best.get(*linked_ix).and_then(|candidate| *candidate) {
                merge_candidate_slot(
                    candidates,
                    candidate.entity_ord,
                    candidate.source,
                    candidate.score_millis
                        + if candidate.source == CandidateSourceKind::PronounLink {
                            180
                        } else {
                            120
                        },
                    CandidateEvidenceKind::LinkedMention,
                    mode,
                    || linked_ix.to_string(),
                );
            }
        }
        for candidate in candidates.iter_mut() {
            if let Some(kind) = mention.kind.as_ref() {
                if let Some(Some(existing_kind)) =
                    entity_kinds_by_ord.get(candidate.entity_ord as usize)
                {
                    if existing_kind != entity_kind_name(kind) {
                        candidate.score_millis -= 220;
                        candidate.evidence_bits |= CandidateEvidenceKind::KindPenalty.bit();
                        if mode == ResolveDetailMode::Detailed {
                            let evidence =
                                candidate_evidence("kind_penalty", existing_kind.clone());
                            if !candidate
                                .evidence
                                .iter()
                                .any(|existing| existing == &evidence)
                            {
                                candidate.evidence.push(evidence);
                            }
                        }
                    }
                }
            }
            if candidate.source == CandidateSourceKind::NewSpeculative && mention.confidence >= 0.8
            {
                candidate.score_millis += 180;
            }
            if prepared_mention.chunk_ix.is_some() {
                candidate.score_millis += 20;
            }
        }

        sort_candidate_slots(candidates, &entity_ids);
        let top = candidates.first().cloned();
        let runner_up = candidates.get(1).cloned();
        let top_score = top
            .as_ref()
            .map(|candidate| candidate.score_millis)
            .unwrap_or_default();
        let margin = top_score.saturating_sub(
            runner_up
                .as_ref()
                .map(|candidate| candidate.score_millis)
                .unwrap_or_default(),
        );

        let pronoun = is_pronoun(normalized);
        let resolved = top.as_ref().and_then(|candidate| {
            let threshold = if pronoun { 1100 } else { 900 };
            let margin_threshold = if pronoun { 220 } else { 180 };
            let speculative = candidate.source == CandidateSourceKind::NewSpeculative;
            if candidate.score_millis >= threshold
                && margin >= margin_threshold
                && (!speculative || candidate.score_millis >= 700)
            {
                Some(EntityId(entity_ids[candidate.entity_ord as usize].clone()))
            } else {
                None
            }
        });

        let decision = if let Some(entity_id) = resolved.clone() {
            er_summary.resolved_count += 1;
            let code = match top.as_ref().map(|candidate| candidate.source) {
                Some(CandidateSourceKind::Seed) => "er_known_seed_match",
                Some(CandidateSourceKind::KernelAlias | CandidateSourceKind::KernelResolved) => {
                    "er_kernel_alias_match"
                }
                Some(CandidateSourceKind::PronounLink) => "er_pronoun_link_match",
                Some(CandidateSourceKind::AliasLink | CandidateSourceKind::LocalSurface) => {
                    "er_alias_link_match"
                }
                Some(CandidateSourceKind::NewSpeculative) => "er_new_speculative_entity",
                _ => "er_collective_merge",
            };
            record_resolution_diagnostic(
                mode,
                &mut diagnostics,
                &mut diagnostic_counts,
                code,
                || {
                    format!(
                        "Resolved mention '{}' to '{}' with score {} and margin {}.",
                        mention.surface, entity_id.0, top_score, margin
                    )
                },
            );
            ResolutionDecisionState {
                kind: ResolvedMentionKind::Resolved,
                entity_id: Some(entity_id),
                confidence_millis: top_score.max(0) as u32,
                margin_millis: margin.max(0) as u32,
            }
        } else if !candidates.is_empty() {
            er_summary.ambiguous_count += 1;
            record_resolution_diagnostic(
                mode,
                &mut diagnostics,
                &mut diagnostic_counts,
                "er_ambiguous_resolution",
                || {
                    format!(
                        "Mention '{}' stayed ambiguous; top candidate score {} margin {}.",
                        mention.surface, top_score, margin
                    )
                },
            );
            ResolutionDecisionState {
                kind: ResolvedMentionKind::Ambiguous,
                entity_id: None,
                confidence_millis: top_score.max(0) as u32,
                margin_millis: margin.max(0) as u32,
            }
        } else {
            er_summary.unresolved_count += 1;
            ResolutionDecisionState {
                kind: ResolvedMentionKind::Unresolved,
                entity_id: None,
                confidence_millis: 0,
                margin_millis: 0,
            }
        };

        if let (ResolvedMentionKind::Resolved, Some(entity_id), Some(top_candidate)) =
            (&decision.kind, decision.entity_id.as_ref(), top.as_ref())
        {
            if !normalized.is_empty()
                && !pronoun
                && normalized.as_str() != normalize_surface(&entity_id.0)
                && top_candidate.source != CandidateSourceKind::Seed
                && top_candidate.source != CandidateSourceKind::KernelResolved
                && top_candidate.source != CandidateSourceKind::PronounLink
                && candidate_slot_has_alias_signal(top_candidate)
                && decision.confidence_millis >= 1000
                && decision.margin_millis >= 260
                && !entity_memory
                    .known_aliases
                    .contains(&(entity_id.0.clone(), normalized.clone()))
            {
                alias_confirmations.push(AliasConfirmation {
                    alias_surface: mention.surface.clone(),
                    normalized: normalized.clone(),
                    entity_id: entity_id.clone(),
                    confidence_millis: decision.confidence_millis,
                    mention_id: mention_id(document, prepared_mention.mention_ix),
                });
            } else if !normalized.is_empty()
                && normalized.as_str() != normalize_surface(&entity_id.0)
                && !entity_memory
                    .known_aliases
                    .contains(&(entity_id.0.clone(), normalized.clone()))
            {
                record_resolution_diagnostic(
                    mode,
                    &mut diagnostics,
                    &mut diagnostic_counts,
                    "er_alias_rejected_low_margin",
                    || {
                        format!(
                            "Alias confirmation for '{}' -> '{}' was rejected because the evidence was not alias-specific enough.",
                            mention.surface, entity_id.0
                        )
                    },
                );
            }
        } else if top.is_some() && margin < 180 {
            record_resolution_diagnostic(
                mode,
                &mut diagnostics,
                &mut diagnostic_counts,
                "er_alias_rejected_low_margin",
                || {
                    format!(
                        "Alias-style resolution for '{}' was rejected because the candidate margin was too small.",
                        mention.surface
                    )
                },
            );
        }

        let candidate_list = match mode {
            ResolveDetailMode::Detailed => {
                candidate_list_from_slots(candidates.clone(), &entity_ids)
            }
            ResolveDetailMode::Compact => Vec::new(),
        };
        resolutions.push(CompactResolutionRow {
            mention_index: prepared_mention.mention_ix,
            entity_id: decision.entity_id.clone(),
            chunk_index: prepared_mention.chunk_ix,
            kind: match decision.kind {
                ResolvedMentionKind::Resolved => CompactResolutionKind::Resolved,
                ResolvedMentionKind::Ambiguous => CompactResolutionKind::Ambiguous,
                ResolvedMentionKind::Unresolved => CompactResolutionKind::Unresolved,
            },
            confidence_millis: decision.confidence_millis,
            margin_millis: decision.margin_millis,
        });
        if mode == ResolveDetailMode::Detailed {
            detailed_resolutions.push(DetailedMentionResolution {
                mention_ix: prepared_mention.mention_ix,
                mention_id: mention_id(document, prepared_mention.mention_ix),
                entity_id: decision.entity_id.clone(),
                candidates: candidate_list.clone(),
                decision: decision.clone(),
            });
            resolved_mentions.push(ResolvedMention {
                mention_id: mention_id(document, prepared_mention.mention_ix),
                mention_index: prepared_mention.mention_ix,
                range: mention.range,
                surface: mention.surface.clone(),
                normalized: normalized.clone(),
                kind: mention.kind.clone(),
                entity_id: decision.entity_id.clone(),
                decision: ResolutionDecision {
                    status: match decision.kind {
                        ResolvedMentionKind::Resolved => "resolved".to_owned(),
                        ResolvedMentionKind::Ambiguous => "ambiguous".to_owned(),
                        ResolvedMentionKind::Unresolved => "unresolved".to_owned(),
                    },
                    confidence_millis: decision.confidence_millis,
                    margin_millis: decision.margin_millis,
                },
                candidates: candidate_list,
            });
        }
    }

    alias_confirmations.sort_by(|left, right| {
        left.entity_id
            .0
            .cmp(&right.entity_id.0)
            .then_with(|| left.normalized.cmp(&right.normalized))
            .then_with(|| left.mention_id.0.cmp(&right.mention_id.0))
    });
    alias_confirmations.dedup_by(|left, right| {
        left.entity_id == right.entity_id && left.normalized == right.normalized
    });
    er_summary.alias_confirmation_count = alias_confirmations.len();

    if mode == ResolveDetailMode::Compact {
        append_resolution_diagnostic_summaries(&mut diagnostics, diagnostic_counts);
    }

    (
        resolutions,
        detailed_resolutions,
        resolved_mentions,
        alias_confirmations,
        er_summary,
        diagnostics,
    )
}

fn build_semantic_records_native(
    document: &IngestDocument,
    scan: &NativeScanRows,
    structure: &NativeStructureRows,
    chunks: &[ChunkRecord],
    resolutions: &[CompactResolutionOrd],
    alias_confirmations: &[AliasConfirmationOrd],
    entity_ids: &[String],
) -> (
    Vec<SemanticEntityRecord>,
    Vec<SemanticRelationRecord>,
    usize,
    Vec<Diagnostic>,
) {
    let mut entities = vec![None::<EntityAccumulatorOrd>; entity_ids.len()];
    let mut unresolved_relation_count = 0usize;
    let mut self_relation_count = 0usize;
    let mut low_quality_relates_count = 0usize;
    let relation_canonical_mentions = build_relation_canonical_mentions(scan);
    let resolved_entity_by_mention_ix = resolutions
        .iter()
        .map(|resolution| resolution.entity_ord)
        .collect::<Vec<_>>();
    let entity_ord_by_id = entity_ids
        .iter()
        .enumerate()
        .map(|(entity_ord, entity_id)| (entity_id.as_str(), entity_ord as u32))
        .collect::<FxHashMap<_, _>>();

    for resolution in resolutions {
        let Some(entity_ord) = resolution.entity_ord else {
            continue;
        };
        let mention = &scan.mentions[resolution.mention_index];
        let entry = entities[entity_ord as usize].get_or_insert_with(|| EntityAccumulatorOrd {
            canonical_name: mention.surface.clone(),
            aliases: SmallVec::new(),
            kind: mention.kind.clone(),
            mention_count: 0,
            chunk_indexes: SmallVec::new(),
        });
        entry.mention_count += 1;
        if let Some(chunk_index) = resolution.chunk_index {
            if !entry
                .chunk_indexes
                .iter()
                .any(|existing| *existing == chunk_index)
            {
                entry.chunk_indexes.push(chunk_index);
            }
        }
    }

    for confirmation in alias_confirmations {
        let entry = entities[confirmation.entity_ord as usize].get_or_insert_with(|| {
            EntityAccumulatorOrd {
                canonical_name: confirmation.alias_surface.clone(),
                aliases: SmallVec::new(),
                kind: None,
                mention_count: 0,
                chunk_indexes: SmallVec::new(),
            }
        });
        if entry.canonical_name != confirmation.alias_surface
            && !entry
                .aliases
                .iter()
                .any(|alias| alias == &confirmation.alias_surface)
        {
            entry.aliases.push(confirmation.alias_surface.clone());
        }
    }

    let mut relation_records = Vec::with_capacity(structure.relation_seeds.len());
    for relation in &structure.relation_seeds {
        let source_endpoint = relation.subject_mention_ix.and_then(|mention_ix| {
            relation_entity_ord_for_mention(
                document,
                scan,
                mention_ix,
                &relation_canonical_mentions,
                &resolved_entity_by_mention_ix,
                &entity_ord_by_id,
            )
        });
        let target_endpoint = relation.object_mention_ix.and_then(|mention_ix| {
            relation_entity_ord_for_mention(
                document,
                scan,
                mention_ix,
                &relation_canonical_mentions,
                &resolved_entity_by_mention_ix,
                &entity_ord_by_id,
            )
        });
        let (Some(source_endpoint), Some(target_endpoint)) = (source_endpoint, target_endpoint)
        else {
            unresolved_relation_count += 1;
            continue;
        };
        if source_endpoint.entity_ord == target_endpoint.entity_ord {
            self_relation_count += 1;
            continue;
        }
        if relation.relation_type == "relates_to"
            && (!relation_endpoint_is_assertable(scan, &source_endpoint)
                || !relation_endpoint_is_assertable(scan, &target_endpoint))
        {
            low_quality_relates_count += 1;
            continue;
        }

        let chunk_index = structure
            .sentence_chunk_indexes
            .get(relation.sentence_index)
            .copied()
            .flatten();
        materialize_relation_endpoint_entity(
            &mut entities,
            &entity_ids,
            scan,
            source_endpoint,
            chunk_index,
        );
        materialize_relation_endpoint_entity(
            &mut entities,
            &entity_ids,
            scan,
            target_endpoint,
            chunk_index,
        );

        relation_records.push(SemanticRelationRecord {
            source_entity_id: EntityId(entity_ids[source_endpoint.entity_ord as usize].clone()),
            target_entity_id: EntityId(entity_ids[target_endpoint.entity_ord as usize].clone()),
            edge_type: relation.relation_type.clone(),
            sentence_index: relation.sentence_index,
            chunk_id: chunk_index
                .and_then(|chunk_ix| chunks.get(chunk_ix as usize))
                .map(|chunk| chunk.chunk_id.0.clone()),
        });
    }

    let mut entity_records = entities
        .into_iter()
        .enumerate()
        .filter_map(|(entity_ord, entity)| {
            let entity = entity?;
            Some(SemanticEntityRecord {
                entity_id: EntityId(entity_ids[entity_ord].clone()),
                canonical_name: entity.canonical_name,
                aliases: entity.aliases.into_vec(),
                kind: entity.kind,
                mention_count: entity.mention_count,
                chunk_ids: entity
                    .chunk_indexes
                    .into_iter()
                    .filter_map(|chunk_index| chunks.get(chunk_index as usize))
                    .map(|chunk| chunk.chunk_id.0.clone())
                    .collect(),
            })
        })
        .collect::<Vec<_>>();
    entity_records.sort_by(|left, right| left.entity_id.0.cmp(&right.entity_id.0));

    let mut diagnostics = Vec::new();
    if unresolved_relation_count > 0 {
        diagnostics.push(Diagnostic {
            code: "er_relation_skipped_unresolved_entity".to_owned(),
            message: format!(
                "Skipped {} asserted relations because one or more arguments stayed unresolved.",
                unresolved_relation_count
            ),
        });
    }
    if self_relation_count > 0 {
        diagnostics.push(Diagnostic {
            code: "er_relation_skipped_same_entity".to_owned(),
            message: format!(
                "Skipped {} asserted relations because both arguments resolved to the same entity.",
                self_relation_count
            ),
        });
    }
    if low_quality_relates_count > 0 {
        diagnostics.push(Diagnostic {
            code: "er_relation_skipped_low_quality_relates_to".to_owned(),
            message: format!(
                "Skipped {} asserted relates_to edges because one or more arguments were too weak to promote into archive truth.",
                low_quality_relates_count
            ),
        });
    }

    (
        entity_records,
        relation_records,
        scan.discovery_count,
        diagnostics,
    )
}

fn relation_entity_ord_for_mention(
    document: &IngestDocument,
    scan: &NativeScanRows,
    mention_ix: usize,
    relation_canonical_mentions: &[usize],
    resolved_entity_by_mention_ix: &[Option<u32>],
    entity_ord_by_id: &FxHashMap<&str, u32>,
) -> Option<RelationEndpointResolution> {
    let canonical_mention_ix = relation_canonical_mentions[mention_ix];
    let canonical_ord = resolved_entity_by_mention_ix
        .get(canonical_mention_ix)
        .copied()
        .flatten();
    if let Some(entity_ord) = canonical_ord {
        return Some(RelationEndpointResolution {
            entity_ord,
            mention_ix: canonical_mention_ix,
            route: RelationEndpointRoute::Resolved,
        });
    }

    if let Some(entity_ord) = relation_entity_ord_from_ref(
        document,
        &scan.mentions[canonical_mention_ix],
        entity_ord_by_id,
    ) {
        return Some(RelationEndpointResolution {
            entity_ord,
            mention_ix: canonical_mention_ix,
            route: relation_endpoint_route(&scan.mentions[canonical_mention_ix], true)?,
        });
    }

    if canonical_mention_ix != mention_ix {
        let entity_ord =
            relation_entity_ord_from_ref(document, &scan.mentions[mention_ix], entity_ord_by_id)?;
        return Some(RelationEndpointResolution {
            entity_ord,
            mention_ix,
            route: relation_endpoint_route(&scan.mentions[mention_ix], false)?,
        });
    }

    None
}

fn relation_endpoint_route(
    mention: &MentionSpan,
    canonical: bool,
) -> Option<RelationEndpointRoute> {
    match mention.entity_ref.as_ref()? {
        MentionEntityRef::Known(_) => Some(if canonical {
            RelationEndpointRoute::CanonicalKnownRef
        } else {
            RelationEndpointRoute::MentionKnownRef
        }),
        MentionEntityRef::Speculative(_) => Some(if canonical {
            RelationEndpointRoute::CanonicalSpeculativeRef
        } else {
            RelationEndpointRoute::MentionSpeculativeRef
        }),
    }
}

fn relation_endpoint_is_assertable(
    scan: &NativeScanRows,
    endpoint: &RelationEndpointResolution,
) -> bool {
    let mention = &scan.mentions[endpoint.mention_ix];
    let surface = mention.surface.trim();
    if surface.is_empty() || is_ascii_pronoun_surface(surface) {
        return false;
    }
    if relation_noise_token(surface) || relation_noise_phrase(surface) {
        return false;
    }
    if mention.kind.is_none()
        && matches!(
            endpoint.route,
            RelationEndpointRoute::CanonicalSpeculativeRef
                | RelationEndpointRoute::MentionSpeculativeRef
        )
    {
        return false;
    }
    if matches!(
        endpoint.route,
        RelationEndpointRoute::CanonicalSpeculativeRef
            | RelationEndpointRoute::MentionSpeculativeRef
    ) && relation_surface_quality_score(mention) <= -200
    {
        return false;
    }
    true
}

fn materialize_relation_endpoint_entity(
    entities: &mut [Option<EntityAccumulatorOrd>],
    entity_ids: &[String],
    scan: &NativeScanRows,
    endpoint: RelationEndpointResolution,
    chunk_index: Option<u32>,
) {
    let mention = &scan.mentions[endpoint.mention_ix];
    let Some(slot) = entities.get_mut(endpoint.entity_ord as usize) else {
        return;
    };
    let entry = slot.get_or_insert_with(|| EntityAccumulatorOrd {
        canonical_name: mention.surface.clone(),
        aliases: SmallVec::new(),
        kind: mention.kind.clone(),
        mention_count: 0,
        chunk_indexes: SmallVec::new(),
    });

    if entry.kind.is_none() && mention.kind.is_some() {
        entry.kind = mention.kind.clone();
    }
    if relation_surface_quality_score(mention)
        > relation_surface_quality_score_name(&entry.canonical_name, entry.kind.is_some(), false)
    {
        if entry.canonical_name != mention.surface
            && !entry
                .aliases
                .iter()
                .any(|alias| alias == &entry.canonical_name)
            && entry.aliases.len() < 8
        {
            entry.aliases.push(entry.canonical_name.clone());
        }
        entry.canonical_name = mention.surface.clone();
    } else if entry.canonical_name != mention.surface
        && !entry.aliases.iter().any(|alias| alias == &mention.surface)
        && entry.aliases.len() < 8
        && mention.surface != entity_ids[endpoint.entity_ord as usize]
    {
        entry.aliases.push(mention.surface.clone());
    }
    if let Some(chunk_index) = chunk_index {
        if !entry
            .chunk_indexes
            .iter()
            .any(|existing| *existing == chunk_index)
        {
            entry.chunk_indexes.push(chunk_index);
        }
    }
}

fn relation_entity_ord_from_ref(
    document: &IngestDocument,
    mention: &MentionSpan,
    entity_ord_by_id: &FxHashMap<&str, u32>,
) -> Option<u32> {
    mention
        .entity_ref
        .as_ref()
        .and_then(|entity_ref| entity_id_from_ref(document, entity_ref))
        .and_then(|entity_id| entity_ord_by_id.get(entity_id.0.as_str()).copied())
}

fn materialize_coref_clusters(
    document: &IngestDocument,
    scan: &NativeScanRows,
    coref: &NativeCorefRows,
    resolutions: &[CompactResolutionOrd],
    entity_ids: &[String],
    chunks: &[ChunkRecord],
    persist_chunk_cap: usize,
) -> (Vec<CorefClusterRecord>, NativeCorefSummary) {
    let resolved_by_mention = resolutions
        .iter()
        .map(|resolution| resolution.entity_ord)
        .collect::<Vec<_>>();
    let mut records = Vec::with_capacity(coref.clusters.len());
    let mut conflict_cluster_count = 0usize;

    for (cluster_ix, cluster) in coref.clusters.iter().enumerate() {
        let representative = scan
            .mentions
            .get(cluster.representative_mention_ix)
            .map(|mention| mention.surface.clone())
            .unwrap_or_default();
        let mut resolved_entity_ids = SmallVec::<[EntityId; 4]>::new();
        for member_ix in &cluster.member_indexes {
            let Some(entity_ord) = resolved_by_mention.get(*member_ix).copied().flatten() else {
                continue;
            };
            let entity_id = EntityId(entity_ids[entity_ord as usize].clone());
            if !resolved_entity_ids
                .iter()
                .any(|existing| existing == &entity_id)
                && resolved_entity_ids.len() < 4
            {
                resolved_entity_ids.push(entity_id);
            }
        }
        if resolved_entity_ids.len() > 1 {
            conflict_cluster_count += 1;
        }
        // Persist only durable coref conflict evidence by default. Non-conflict
        // local clusters are rebuildable from the document and don't justify
        // archive mass on the native hot path.
        if resolved_entity_ids.len() <= 1 {
            continue;
        }
        let chunk_ids = cluster
            .chunk_indexes
            .iter()
            .take(persist_chunk_cap)
            .filter_map(|chunk_index| chunks.get(*chunk_index as usize))
            .map(|chunk| chunk.chunk_id.0.clone())
            .collect::<Vec<_>>();

        records.push(CorefClusterRecord {
            cluster_id: format!("coref::{}:{cluster_ix}", document.document_id.0),
            representative_surface: representative,
            mention_count: cluster.member_indexes.len(),
            first_sentence_index: cluster.first_sentence_index,
            last_sentence_index: cluster.last_sentence_index,
            chunk_ids,
            named_count: cluster.named_count,
            nominal_count: cluster.nominal_count,
            pronoun_count: cluster.pronoun_count,
            resolved_entity_ids: resolved_entity_ids.into_vec(),
            confidence_millis: cluster.max_score_millis.max(0) as u32,
            ambiguous: cluster.ambiguous,
            route_mix_bits: cluster.route_mix_bits,
        });
    }

    let mut summary = coref.summary.clone();
    summary.conflict_cluster_count = conflict_cluster_count;
    (records, summary)
}

fn build_coref_candidate_batch(
    document: &IngestDocument,
    coref_records: &[CorefClusterRecord],
    config: &InvarantV3CorefConfig,
) -> Option<KernelMutationBatch> {
    if !config.emit_phase2_conflict_edges {
        return None;
    }
    let mut edges = Vec::new();
    let scope_key = scope_storage_key(&document.scope);
    for record in coref_records {
        if record.resolved_entity_ids.len() < 2 {
            continue;
        }
        for left_ix in 0..record.resolved_entity_ids.len() {
            for right_ix in left_ix + 1..record.resolved_entity_ids.len() {
                let left = &record.resolved_entity_ids[left_ix];
                let right = &record.resolved_entity_ids[right_ix];
                edges.push(KernelEdge {
                    source_id: KernelVertexId(format!("entity::{}", left.0)),
                    target_id: KernelVertexId(format!("entity::{}", right.0)),
                    edge_type: KernelEdgeType("candidate_corefers_with".to_owned()),
                    weight: record.confidence_millis.max(1) as i64,
                    attributes: json!({
                        "documentId": document.document_id.0,
                        "scopeKey": scope_key,
                        "clusterId": record.cluster_id,
                    }),
                    document_id: Some(document.document_id.0.clone()),
                    provenance: KernelProvenance {
                        resolver: Some("coref_kernel".to_owned()),
                        source: Some("document_cluster_conflict".to_owned()),
                        confidence: Some(record.confidence_millis as f64 / 1000.0),
                        evidence_refs: record.chunk_ids.clone(),
                    },
                    layer: KernelGraphLayer::Candidate,
                    ..KernelEdge::default()
                });
            }
        }
    }
    (!edges.is_empty()).then_some(KernelMutationBatch {
        layer: KernelGraphLayer::Candidate,
        scope: KernelMutationScope::Candidate {
            scope_key: scope_storage_key(&document.scope),
        },
        recorded_at: None,
        vertices: Vec::new(),
        edges,
    })
}

#[cfg(feature = "background-verifier")]
fn expected_kind_from_label(label: &str) -> Option<EntityKind> {
    match label {
        "person" => Some(EntityKind::Character),
        "organization" => Some(EntityKind::Organization),
        "location" => Some(EntityKind::Location),
        "event" => Some(EntityKind::Event),
        "item" => Some(EntityKind::Item),
        "concept" => Some(EntityKind::Concept),
        _ => None,
    }
}

fn verification_window_text(
    text: &str,
    sentences: &[SentenceSpan],
    sentence_index: usize,
    byte_limit: usize,
) -> String {
    if sentences.is_empty() {
        return String::new();
    }
    let start_sentence = sentence_index.saturating_sub(1);
    let end_sentence = (sentence_index + 1).min(sentences.len().saturating_sub(1));
    let start = sentences[start_sentence].range.start as usize;
    let end = sentences[end_sentence].range.end as usize;
    let mut slice = text.get(start..end).unwrap_or_default().trim().to_owned();
    if slice.len() <= byte_limit {
        return slice;
    }
    let mut truncated = byte_limit.min(slice.len());
    while truncated > 0 && !slice.is_char_boundary(truncated) {
        truncated -= 1;
    }
    slice.truncate(truncated);
    slice
}

fn build_verification_tasks(
    document: &IngestDocument,
    scan: &NativeScanRows,
    resolutions: &[CompactResolutionOrd],
    alias_confirmations: &[AliasConfirmationOrd],
    entity_ids: &[String],
    max_windows: usize,
    window_bytes: usize,
) -> Vec<VerificationTask> {
    let mut tasks = Vec::new();
    for confirmation in alias_confirmations {
        if tasks.len() >= max_windows {
            break;
        }
        let Some(mention) = scan.mentions.get(confirmation.mention_index) else {
            continue;
        };
        let window_text = verification_window_text(
            &document.text,
            &scan.sentences,
            mention.sentence_index,
            window_bytes,
        );
        if window_text.is_empty() {
            continue;
        }
        tasks.push(VerificationTask {
            mention_index: confirmation.mention_index,
            normalized_surface: confirmation.normalized.clone(),
            window_text,
            expected_kind: mention.kind.clone(),
        });
    }
    if tasks.len() >= max_windows {
        return tasks;
    }
    for resolution in resolutions {
        if tasks.len() >= max_windows {
            break;
        }
        if resolution.entity_ord.is_some() {
            continue;
        }
        let Some(mention) = scan.mentions.get(resolution.mention_index) else {
            continue;
        };
        let normalized = normalize_surface(&mention.surface);
        if normalized.is_empty() || is_pronoun(&normalized) {
            continue;
        }
        let window_text = verification_window_text(
            &document.text,
            &scan.sentences,
            mention.sentence_index,
            window_bytes,
        );
        if window_text.is_empty() {
            continue;
        }
        let expected_kind = mention.kind.clone().or_else(|| {
            resolution
                .entity_ord
                .and_then(|entity_ord| entity_ids.get(entity_ord as usize))
                .and_then(|_| mention.kind.clone())
        });
        tasks.push(VerificationTask {
            mention_index: resolution.mention_index,
            normalized_surface: normalized,
            window_text,
            expected_kind,
        });
    }
    tasks
}

#[cfg(feature = "background-verifier")]
fn run_background_verification(
    config: &InvarantV3VerificationConfig,
    document: &IngestDocument,
    scan: &NativeScanRows,
    resolutions: &[CompactResolutionOrd],
    alias_confirmations: &[AliasConfirmationOrd],
    entity_ids: &[String],
) -> BackgroundVerificationSummary {
    if !config.enable_background_ner_verifier {
        return BackgroundVerificationSummary::default();
    }
    let tasks = build_verification_tasks(
        document,
        scan,
        resolutions,
        alias_confirmations,
        entity_ids,
        config.max_windows_per_document,
        config.window_bytes,
    );
    let mut summary = BackgroundVerificationSummary {
        task_count: tasks.len(),
        ..BackgroundVerificationSummary::default()
    };
    let (Some(model_path), Some(tokenizer_path)) = (
        config.gliner_model_path.as_deref(),
        config.gliner_tokenizer_path.as_deref(),
    ) else {
        summary.diagnostics.push(Diagnostic {
            code: "er_background_verifier_disabled".to_owned(),
            message:
                "Background verifier is enabled, but local GLiNER model paths are not configured."
                    .to_owned(),
        });
        return summary;
    };
    let verifier = match BackgroundNerVerifier::load(model_path, tokenizer_path) {
        Ok(verifier) => verifier,
        Err(error) => {
            summary.diagnostics.push(Diagnostic {
                code: "er_background_verifier_disabled".to_owned(),
                message: format!("Background verifier could not start: {error}"),
            });
            return summary;
        }
    };
    for task in tasks {
        let extracted = match verifier.extract(&task.window_text) {
            Ok(values) => values,
            Err(error) => {
                summary.diagnostics.push(Diagnostic {
                    code: "er_background_verifier_error".to_owned(),
                    message: format!("Background verifier inference failed: {error}"),
                });
                break;
            }
        };
        let mut matched_alias = false;
        let mut matched_type = false;
        for (surface, label, confidence) in extracted {
            if confidence < 0.75 {
                continue;
            }
            let normalized = normalize_surface(&surface);
            if normalized != task.normalized_surface {
                continue;
            }
            matched_alias = true;
            if let Some(expected_kind) = task.expected_kind.clone() {
                if expected_kind_from_label(&label) == Some(expected_kind) {
                    matched_type = true;
                    break;
                }
            }
        }
        if matched_alias {
            summary.supported_alias_count += 1;
        }
        if matched_type {
            summary.supported_type_count += 1;
        }
    }
    summary
}

#[cfg(not(feature = "background-verifier"))]
fn run_background_verification(
    config: &InvarantV3VerificationConfig,
    document: &IngestDocument,
    scan: &NativeScanRows,
    resolutions: &[CompactResolutionOrd],
    alias_confirmations: &[AliasConfirmationOrd],
    entity_ids: &[String],
) -> BackgroundVerificationSummary {
    let mut summary = BackgroundVerificationSummary::default();
    if !config.enable_background_ner_verifier {
        return summary;
    }
    summary.task_count = build_verification_tasks(
        document,
        scan,
        resolutions,
        alias_confirmations,
        entity_ids,
        config.max_windows_per_document,
        config.window_bytes,
    )
    .len();
    summary.diagnostics.push(Diagnostic {
        code: "er_background_verifier_unavailable".to_owned(),
        message: "Background verifier is configured, but this build was compiled without the `background-verifier` feature.".to_owned(),
    });
    summary
}

fn materialize_alias_confirmations(
    document: &IngestDocument,
    alias_confirmations: Vec<AliasConfirmationOrd>,
    entity_ids: &[String],
) -> Vec<AliasConfirmation> {
    alias_confirmations
        .into_iter()
        .map(|confirmation| AliasConfirmation {
            alias_surface: confirmation.alias_surface,
            normalized: confirmation.normalized,
            entity_id: EntityId(entity_ids[confirmation.entity_ord as usize].clone()),
            confidence_millis: confirmation.confidence_millis,
            mention_id: mention_id(document, confirmation.mention_index),
        })
        .collect()
}

fn build_kernel_batch(
    document: &IngestDocument,
    chunks: &[ChunkRecord],
    entities: &[SemanticEntityRecord],
    relations: &[SemanticRelationRecord],
    alias_confirmations: &[AliasConfirmation],
) -> KernelMutationBatch {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();
    let document_vertex_id = format!("doc::{}", document.document_id.0);
    let scope_key = scope_storage_key(&document.scope);
    vertices.push(KernelVertex {
        id: KernelVertexId(document_vertex_id.clone()),
        kind: "document".to_owned(),
        labels: vec!["document".to_owned()],
        weight: 1,
        value: json!({ "title": document.title }),
        attributes: json!({
            "documentId": document.document_id.0,
            "noteId": document.note_id.as_ref().map(|note_id| note_id.0.clone()),
            "scopeKey": scope_key,
        }),
        document_id: Some(document.document_id.0.clone()),
        chapter_id: Some(0),
        chapters: vec![0],
        ..KernelVertex::default()
    });

    for chunk in chunks {
        vertices.push(KernelVertex {
            id: KernelVertexId(chunk.chunk_id.0.clone()),
            kind: "chunk".to_owned(),
            labels: vec!["chunk".to_owned()],
            weight: 1,
            value: json!({ "text": chunk.text }),
            attributes: json!({
                "documentId": document.document_id.0,
                "chunkId": chunk.chunk_id.0,
                "scopeKey": scope_key,
            }),
            search_chunk_id: Some(chunk.chunk_id.0.clone()),
            document_id: Some(document.document_id.0.clone()),
            chapter_id: Some(chunk.chapter_id),
            chapters: vec![chunk.chapter_id],
            boundary_ordinal: Some(chunk.chapter_id),
            boundary_kind: Some(BoundaryKind::Section),
            boundary_ordinals: vec![chunk.chapter_id],
            ..KernelVertex::default()
        });
        edges.push(KernelEdge {
            source_id: KernelVertexId(document_vertex_id.clone()),
            target_id: KernelVertexId(chunk.chunk_id.0.clone()),
            edge_type: KernelEdgeType("contains".to_owned()),
            weight: 1,
            attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
            document_id: Some(document.document_id.0.clone()),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        });
    }

    for entity in entities {
        let entity_vertex_id = format!("entity::{}", entity.entity_id.0);
        vertices.push(KernelVertex {
            id: KernelVertexId(entity_vertex_id.clone()),
            kind: "entity".to_owned(),
            labels: vec!["entity".to_owned()],
            weight: entity.mention_count.max(1) as i64,
            value: json!({ "name": entity.canonical_name, "aliases": entity.aliases }),
            attributes: json!({
                "documentId": document.document_id.0,
                "entityId": entity.entity_id.0,
                "scopeKey": scope_key,
            }),
            entity_id: Some(entity.entity_id.0.clone()),
            document_id: Some(document.document_id.0.clone()),
            chapter_id: Some(0),
            chapters: vec![0],
            ..KernelVertex::default()
        });
        edges.push(KernelEdge {
            source_id: KernelVertexId(document_vertex_id.clone()),
            target_id: KernelVertexId(entity_vertex_id.clone()),
            edge_type: KernelEdgeType("entity".to_owned()),
            weight: entity.mention_count.max(1) as i64,
            attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
            document_id: Some(document.document_id.0.clone()),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        });
    }

    for confirmation in alias_confirmations {
        let alias_vertex_id = format!(
            "alias::{}::{}",
            confirmation.entity_id.0, confirmation.normalized
        );
        vertices.push(KernelVertex {
            id: KernelVertexId(alias_vertex_id.clone()),
            kind: "alias".to_owned(),
            labels: vec!["alias".to_owned()],
            weight: 1,
            value: json!({ "name": confirmation.alias_surface }),
            attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
            document_id: Some(document.document_id.0.clone()),
            entity_facet: Some(KernelEntityFacet {
                canonical_entity_id: Some(confirmation.entity_id.0.clone()),
                surface: Some(confirmation.alias_surface.clone()),
                entity_kind: Some("alias".to_owned()),
            }),
            ..KernelVertex::default()
        });
        edges.push(KernelEdge {
            source_id: KernelVertexId(alias_vertex_id),
            target_id: KernelVertexId(format!("entity::{}", confirmation.entity_id.0)),
            edge_type: KernelEdgeType("alias_of".to_owned()),
            weight: confirmation.confidence_millis.max(1) as i64,
            attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
            document_id: Some(document.document_id.0.clone()),
            provenance: KernelProvenance {
                resolver: Some("native_collective_er".to_owned()),
                source: Some("confirmed_alias".to_owned()),
                confidence: Some(confirmation.confidence_millis as f64 / 1000.0),
                evidence_refs: vec![confirmation.mention_id.0.clone()],
            },
            resolution_facet: Some(KernelResolutionFacet {
                strategy: Some("collective".to_owned()),
                candidate_rank: Some(0),
                confidence: Some(confirmation.confidence_millis as f64 / 1000.0),
                replaced_edge_key: None,
            }),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        });
    }

    for relation in relations {
        edges.push(KernelEdge {
            source_id: KernelVertexId(format!("entity::{}", relation.source_entity_id.0)),
            target_id: KernelVertexId(format!("entity::{}", relation.target_entity_id.0)),
            edge_type: KernelEdgeType(relation.edge_type.clone()),
            weight: 1,
            attributes: json!({
                "documentId": document.document_id.0,
                "sentenceIndex": relation.sentence_index,
                "scopeKey": scope_key,
            }),
            document_id: Some(document.document_id.0.clone()),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        });
    }

    KernelMutationBatch {
        layer: KernelGraphLayer::Asserted,
        scope: KernelMutationScope::Document {
            document_id: document.document_id.0.clone(),
        },
        recorded_at: None,
        vertices,
        edges,
    }
}

fn locate_sentence(sentences: &[SentenceSpan], range: TextRange) -> Option<usize> {
    let mut left = 0usize;
    let mut right = sentences.len();
    while left < right {
        let middle = left + (right - left) / 2;
        let sentence = &sentences[middle];
        if sentence.range.end < range.end {
            left = middle + 1;
        } else if sentence.range.start > range.start {
            right = middle;
        } else {
            return Some(sentence.index);
        }
    }
    None
}

fn locate_sentence_cursor(
    sentences: &[SentenceSpan],
    cursor: &mut usize,
    range: TextRange,
) -> usize {
    if sentences.is_empty() {
        return 0;
    }
    while *cursor + 1 < sentences.len() && sentences[*cursor].range.end < range.end {
        *cursor += 1;
    }
    sentences
        .get(*cursor)
        .filter(|sentence| sentence.range.start <= range.start && sentence.range.end >= range.end)
        .map(|sentence| sentence.index)
        .or_else(|| locate_sentence(sentences, range))
        .unwrap_or_default()
}

fn extract_boundaries(text: &str) -> Vec<BoundaryRecord> {
    let mut boundaries = Vec::new();
    let mut cursor = 0usize;
    let mut chapter_id = 0u32;
    for line in text.lines() {
        let start = cursor;
        let end = start + line.len();
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let lower = trimmed.to_ascii_lowercase();
            let is_heading = trimmed.starts_with('#')
                || lower.starts_with("chapter ")
                || matches!(
                    lower.as_str(),
                    "prologue" | "epilogue" | "preface" | "introduction"
                );
            if is_heading {
                let label = trimmed.trim_start_matches('#').trim().to_owned();
                boundaries.push(BoundaryRecord {
                    label,
                    range: to_range(start, end),
                    chapter_id,
                    is_chapter: is_heading,
                });
                chapter_id = chapter_id.saturating_add(1);
            }
        }
        cursor = end + 1;
    }
    boundaries
}

fn normalize_surface(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ch.is_whitespace())
        .flat_map(|ch| ch.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn hot_path_stop_words() -> &'static FxHashSet<&'static str> {
    static STOP_WORDS: OnceLock<FxHashSet<&'static str>> = OnceLock::new();
    STOP_WORDS.get_or_init(|| {
        get(LANGUAGE::English)
            .iter()
            .copied()
            .collect::<FxHashSet<_>>()
    })
}

fn hot_path_should_keep_mention(mention: &MentionSpan) -> bool {
    if matches!(mention.entity_ref, Some(MentionEntityRef::Known(_))) {
        return true;
    }
    let normalized = normalize_surface(&mention.surface);
    if normalized.is_empty() {
        return false;
    }
    if is_pronoun(&normalized) {
        return true;
    }
    !normalized
        .split(' ')
        .all(|token| !token.is_empty() && hot_path_stop_words().contains(token))
}

fn normalize_token_surface(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_alphanumeric() {
            normalized.extend(ch.to_lowercase());
        }
    }
    normalized
}

fn candidate_slot_has_alias_signal(candidate: &CandidateSlot) -> bool {
    matches!(
        candidate.source,
        CandidateSourceKind::KernelAlias
            | CandidateSourceKind::AliasLink
            | CandidateSourceKind::LocalSurface
    ) || candidate.evidence_bits
        & (CandidateEvidenceKind::Kernel.bit()
            | CandidateEvidenceKind::ResolverLink.bit()
            | CandidateEvidenceKind::LocalSurface.bit()
            | CandidateEvidenceKind::SurfaceCluster.bit())
        != 0
}

fn entity_kind_name(kind: &EntityKind) -> &'static str {
    match kind {
        EntityKind::Character => "Character",
        EntityKind::Location => "Location",
        EntityKind::Npc => "Npc",
        EntityKind::Item => "Item",
        EntityKind::Faction => "Faction",
        EntityKind::Organization => "Organization",
        EntityKind::Event => "Event",
        EntityKind::Concept => "Concept",
        EntityKind::Other => "Other",
    }
}

fn alias_entries_from_entities(
    document_id: &str,
    entities: &[SemanticEntityRecord],
) -> Vec<AliasEntry> {
    let mut entries = BTreeMap::<String, BTreeMap<(String, String), usize>>::new();
    for entity in entities {
        let forms = std::iter::once(entity.canonical_name.clone()).chain(entity.aliases.clone());
        for form in forms {
            let normalized = normalize_surface(&form);
            if normalized.is_empty() {
                continue;
            }
            entries
                .entry(normalized)
                .or_default()
                .entry((entity.entity_id.0.clone(), document_id.to_owned()))
                .and_modify(|count| *count += entity.mention_count)
                .or_insert(entity.mention_count);
        }
    }

    entries
        .into_iter()
        .map(|(normalized, postings)| AliasEntry {
            normalized,
            postings: postings
                .into_iter()
                .map(|((entity_id, document_id), mention_count)| AliasPosting {
                    entity_id,
                    document_id,
                    mention_count,
                })
                .collect(),
        })
        .collect()
}

fn build_lexical_postings_segment(
    indexed_spans: Vec<IndexedSpan>,
    entities: &[SemanticEntityRecord],
    document_id: &str,
) -> LexicalPostingsSegment {
    LexicalPostingsSegment {
        spans: indexed_spans,
        alias_entries: alias_entries_from_entities(document_id, entities),
    }
}

fn speculative_entity_id(document_id: &DocumentId, surface: &str) -> String {
    format!(
        "{}::{}",
        document_id.0,
        normalize_surface(surface).replace(' ', "_")
    )
}

fn entity_id_from_ref(
    document: &IngestDocument,
    entity_ref: &MentionEntityRef,
) -> Option<EntityId> {
    match entity_ref {
        MentionEntityRef::Known(entity_id) => Some(entity_id.clone()),
        MentionEntityRef::Speculative(speculative) => Some(EntityId(format!(
            "{}::{}",
            document.document_id.0,
            speculative.replace(' ', "_")
        ))),
    }
}

fn is_pronoun(value: &str) -> bool {
    matches!(
        value,
        "he" | "she" | "they" | "them" | "him" | "her" | "we" | "us" | "i" | "you" | "it"
    )
}

fn range_contains(container: TextRange, inner: TextRange) -> bool {
    container.start <= inner.start && container.end >= inner.end
}

fn slice_or_empty(text: &str, range: TextRange) -> &str {
    text.get(range.start as usize..range.end as usize)
        .unwrap_or_default()
}

fn safe_text_slice(text: &str, range: TextRange) -> &str {
    slice_or_empty(
        text,
        TextRange {
            start: range.start.min(text.len() as u32),
            end: range.end.min(text.len() as u32),
        },
    )
}

fn to_range(start: usize, end: usize) -> TextRange {
    TextRange {
        start: start.min(u32::MAX as usize) as u32,
        end: end.min(u32::MAX as usize) as u32,
    }
}

fn is_front_matter_label(label: &str) -> bool {
    matches!(
        label.trim().to_ascii_lowercase().as_str(),
        "prologue" | "preface" | "introduction"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_machine::SurfaceCompiler;
    use phoenix_kernel::KernelVertexClass;
    use phoenix_store_native_core::PhoenixGraphKernelStoreV2;
    use phoenix_store_overgraph::PhoenixOvergraphStore;
    use phoenix_types::{GenderHint, NoteId};
    use std::env;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!(
            "phoenix-ingest-overgraph-test-{name}-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("time")
                .as_nanos()
        ))
    }

    fn native_relation_outputs(
        document: &IngestDocument,
        seeds: &[ResolverEntitySeed],
    ) -> (
        NativeScanRows,
        NativeStructureRows,
        Vec<SemanticRelationRecord>,
        Vec<Diagnostic>,
    ) {
        let engine = PhoenixInvarantV3::default();
        let scan = scan_native_compact(
            &document.text,
            &document.scope,
            seeds,
            &engine.config.extraction,
        );
        let (chunks, _) = build_chunk_records(
            document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );
        let structure = build_native_structure_rows(&document.text, &scan, &chunks);
        let coref = build_coref_rows(&scan, &structure, &engine.config.coref);
        let (resolutions, aliases, _summary, _diagnostics, entity_ids, _entity_ord_by_id) =
            resolve_mentions_compact_native(
                document,
                &scan,
                &coref,
                &chunks,
                &NativeEntityMemory::default(),
            );
        let (_entities, relations, _discovery_count, diagnostics) = build_semantic_records_native(
            document,
            &scan,
            &structure,
            &chunks,
            &resolutions,
            &aliases,
            &entity_ids,
        );
        (scan, structure, relations, diagnostics)
    }

    #[test]
    fn scan_and_structure_capture_mentions_and_relations() {
        let engine = PhoenixInvarantV3::default();
        let scan = engine.scan_parts(
            "Luffy attacked Zoro.",
            &ScopeKey::default(),
            &[
                ResolverEntitySeed {
                    entity_id: EntityId("luffy".to_owned()),
                    canonical_name: "Luffy".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Character),
                    gender: Some(GenderHint::Male),
                    number: None,
                    scope: ScopeKey::default(),
                },
                ResolverEntitySeed {
                    entity_id: EntityId("zoro".to_owned()),
                    canonical_name: "Zoro".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Character),
                    gender: Some(GenderHint::Male),
                    number: None,
                    scope: ScopeKey::default(),
                },
            ],
        );
        assert_eq!(scan.mentions.len(), 2);
        let structure = engine.build_structure_parts("Luffy attacked Zoro.", &scan);
        assert_eq!(structure.sentence_frames.len(), 1);
        assert_eq!(structure.relations.len(), 1);
        assert_eq!(structure.relations[0].relation_type, "attacks");
    }

    #[test]
    fn public_scan_and_structure_match_machine_runtime() {
        let text = "Luffy attacked Zoro in Shells Town.";
        let scope = ScopeKey::default();
        let resolver_seed = vec![
            ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: vec!["Straw Hat".to_owned()],
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: scope.clone(),
            },
            ResolverEntitySeed {
                entity_id: EntityId("zoro".to_owned()),
                canonical_name: "Zoro".to_owned(),
                aliases: vec!["Roronoa Zoro".to_owned()],
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: scope.clone(),
            },
        ];
        let engine = PhoenixInvarantV3::default();
        let compiler = SurfaceCompiler::default();

        let engine_scan = engine.scan_parts(text, &scope, &resolver_seed);
        let engine_structure = engine.build_structure_parts(text, &engine_scan);
        let machine_scan = compiler.compatibility_scan_parts(text, &scope, &resolver_seed);
        let machine_structure = compiler.compatibility_structure_parts(text, &machine_scan);

        assert_eq!(engine_scan, machine_scan);
        assert_eq!(engine_structure, machine_structure);
    }

    #[test]
    #[ignore = "perf smoke"]
    fn public_scan_and_structure_perf_smoke() {
        let text = concat!(
            "Luffy attacked Zoro in Shells Town. ",
            "Nami mapped the harbor while Sanji fed the crew at dawn. ",
            "The Straw Hat crew met Marines near Orange Town and then escaped by sunset. ",
            "Robin reported that Baroque Works had agents in Alubarna. ",
            "Chopper treated Vivi after the battle in the palace square. ",
            "Franky repaired the Thousand Sunny in Water Seven before the next voyage. "
        )
        .repeat(24);
        let scope = ScopeKey::default();
        let resolver_seed = vec![
            ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: vec!["Straw Hat".to_owned()],
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: scope.clone(),
            },
            ResolverEntitySeed {
                entity_id: EntityId("zoro".to_owned()),
                canonical_name: "Zoro".to_owned(),
                aliases: vec!["Roronoa Zoro".to_owned()],
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: scope.clone(),
            },
            ResolverEntitySeed {
                entity_id: EntityId("sunny".to_owned()),
                canonical_name: "Thousand Sunny".to_owned(),
                aliases: vec!["Sunny".to_owned()],
                kind: Some(EntityKind::Item),
                gender: None,
                number: None,
                scope: scope.clone(),
            },
            ResolverEntitySeed {
                entity_id: EntityId("baroque".to_owned()),
                canonical_name: "Baroque Works".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Organization),
                gender: None,
                number: None,
                scope: scope.clone(),
            },
        ];
        let engine = PhoenixInvarantV3::default();
        let compiler = SurfaceCompiler::default();
        let iterations = 128usize;

        let machine_start = Instant::now();
        let mut machine_checksum = 0usize;
        for _ in 0..iterations {
            let scan = compiler.compatibility_scan_parts(&text, &scope, &resolver_seed);
            let structure = compiler.compatibility_structure_parts(&text, &scan);
            machine_checksum += scan.mentions.len() + structure.relations.len();
        }
        let machine_elapsed = machine_start.elapsed();

        let engine_start = Instant::now();
        let mut engine_checksum = 0usize;
        for _ in 0..iterations {
            let scan = engine.scan_parts(&text, &scope, &resolver_seed);
            let structure = engine.build_structure_parts(&text, &scan);
            engine_checksum += scan.mentions.len() + structure.relations.len();
        }
        let engine_elapsed = engine_start.elapsed();

        println!(
            "perf_smoke iterations={iterations} machine_total_ms={} machine_per_iter_ms={:.3} engine_total_ms={} engine_per_iter_ms={:.3} checksum={}/{}",
            machine_elapsed.as_millis(),
            machine_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            engine_elapsed.as_millis(),
            engine_elapsed.as_secs_f64() * 1000.0 / iterations as f64,
            machine_checksum,
            engine_checksum
        );

        assert_eq!(machine_checksum, engine_checksum);
        assert!(machine_checksum > 0);
    }

    #[test]
    fn hybrid_ee_detects_seeded_lowercase_and_multiword_entities() {
        let engine = PhoenixInvarantV3::default();
        let scan = engine.scan_parts(
            "luffy met Acme Corporation in New York.",
            &ScopeKey::default(),
            &[ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: vec!["straw hat".to_owned()],
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            }],
        );
        let surfaces = scan
            .mentions
            .iter()
            .map(|mention| mention.surface.clone())
            .collect::<Vec<_>>();
        assert!(surfaces.iter().any(|surface| surface == "luffy"));
        assert!(surfaces.iter().any(|surface| surface == "Acme Corporation"));
        assert!(surfaces.iter().any(|surface| surface == "New York"));
    }

    #[test]
    fn relation_seed_prefers_distinct_named_object_over_reflexive_pronoun() {
        let document = IngestDocument {
            document_id: DocumentId("doc-reflexive-object".to_owned()),
            note_id: None,
            title: "Reflexive".to_owned(),
            text: "Ryan introduced himself to Dynamis.".to_owned(),
            scope: ScopeKey::default(),
        };
        let seeds = [
            ResolverEntitySeed {
                entity_id: EntityId("ryan".to_owned()),
                canonical_name: "Ryan".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            },
            ResolverEntitySeed {
                entity_id: EntityId("dynamis".to_owned()),
                canonical_name: "Dynamis".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Organization),
                gender: None,
                number: None,
                scope: ScopeKey::default(),
            },
        ];

        let (scan, structure, _relations, _diagnostics) =
            native_relation_outputs(&document, &seeds);
        assert!(structure.relation_seeds.iter().any(|seed| {
            let Some(subject_mention_ix) = seed.subject_mention_ix else {
                return false;
            };
            let Some(object_mention_ix) = seed.object_mention_ix else {
                return false;
            };
            scan.mentions[subject_mention_ix].surface == "Ryan"
                && scan.mentions[object_mention_ix].surface == "Dynamis"
        }));
    }

    #[test]
    fn semantic_relations_skip_same_entity_assertions() {
        let document = IngestDocument {
            document_id: DocumentId("doc-self-loop".to_owned()),
            note_id: None,
            title: "Self Loop".to_owned(),
            text: "Luffy attacked Luffy.".to_owned(),
            scope: ScopeKey::default(),
        };
        let seeds = [ResolverEntitySeed {
            entity_id: EntityId("luffy".to_owned()),
            canonical_name: "Luffy".to_owned(),
            aliases: Vec::new(),
            kind: Some(EntityKind::Character),
            gender: Some(GenderHint::Male),
            number: None,
            scope: ScopeKey::default(),
        }];

        let (_scan, structure, relations, diagnostics) = native_relation_outputs(&document, &seeds);
        assert!(!structure.relation_seeds.iter().any(|seed| {
            matches!(
                (seed.subject_mention_ix, seed.object_mention_ix),
                (Some(_), Some(_))
            )
        }));
        assert!(relations.is_empty());
        assert!(!diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "er_relation_skipped_same_entity"));
    }

    #[test]
    fn semantic_relations_use_pronoun_antecedent_entities() {
        let document = IngestDocument {
            document_id: DocumentId("doc-pronoun-relation".to_owned()),
            note_id: None,
            title: "Pronoun Relation".to_owned(),
            text: "Luffy waited. He joined Dynamis.".to_owned(),
            scope: ScopeKey::default(),
        };
        let seeds = [
            ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            },
            ResolverEntitySeed {
                entity_id: EntityId("dynamis".to_owned()),
                canonical_name: "Dynamis".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Organization),
                gender: None,
                number: None,
                scope: ScopeKey::default(),
            },
        ];

        let (_scan, _structure, relations, _diagnostics) =
            native_relation_outputs(&document, &seeds);
        assert!(relations.iter().any(|relation| {
            relation.source_entity_id.0 == "luffy"
                && relation.target_entity_id.0 == "dynamis"
                && relation.edge_type == "member_of"
        }));
    }

    #[test]
    fn semantic_relations_recover_preposed_location_arguments() {
        let document = IngestDocument {
            document_id: DocumentId("doc-preposed-location".to_owned()),
            note_id: None,
            title: "Preposed Location".to_owned(),
            text: "In New Rome, Ryan stayed.".to_owned(),
            scope: ScopeKey::default(),
        };
        let seeds = [
            ResolverEntitySeed {
                entity_id: EntityId("ryan".to_owned()),
                canonical_name: "Ryan".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            },
            ResolverEntitySeed {
                entity_id: EntityId("new_rome".to_owned()),
                canonical_name: "New Rome".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Location),
                gender: None,
                number: None,
                scope: ScopeKey::default(),
            },
        ];

        let (_scan, _structure, relations, _diagnostics) =
            native_relation_outputs(&document, &seeds);
        assert!(relations.iter().any(|relation| {
            relation.source_entity_id.0 == "ryan"
                && relation.target_entity_id.0 == "new_rome"
                && relation.edge_type == "located_in"
        }));
    }

    #[test]
    fn semantic_relations_skip_discourse_noise_relates_to_endpoints() {
        let document = IngestDocument {
            document_id: DocumentId("doc-discourse-noise".to_owned()),
            note_id: None,
            title: "Discourse Noise".to_owned(),
            text: "Hey amused Ryan.".to_owned(),
            scope: ScopeKey::default(),
        };
        let seeds = [ResolverEntitySeed {
            entity_id: EntityId("ryan".to_owned()),
            canonical_name: "Ryan".to_owned(),
            aliases: Vec::new(),
            kind: Some(EntityKind::Character),
            gender: Some(GenderHint::Male),
            number: None,
            scope: ScopeKey::default(),
        }];

        let (_scan, _structure, relations, diagnostics) =
            native_relation_outputs(&document, &seeds);
        assert!(relations.is_empty());
        assert!(diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == "er_relation_skipped_low_quality_relates_to" }));
    }

    #[test]
    fn semantic_relations_skip_speculative_participle_relates_to_endpoints() {
        let document = IngestDocument {
            document_id: DocumentId("doc-participle-noise".to_owned()),
            note_id: None,
            title: "Participle Noise".to_owned(),
            text: "Driving startled Ryan.".to_owned(),
            scope: ScopeKey::default(),
        };
        let seeds = [ResolverEntitySeed {
            entity_id: EntityId("ryan".to_owned()),
            canonical_name: "Ryan".to_owned(),
            aliases: Vec::new(),
            kind: Some(EntityKind::Character),
            gender: Some(GenderHint::Male),
            number: None,
            scope: ScopeKey::default(),
        }];

        let (_scan, _structure, relations, diagnostics) =
            native_relation_outputs(&document, &seeds);
        assert!(relations.is_empty());
        assert!(diagnostics
            .iter()
            .any(|diagnostic| { diagnostic.code == "er_relation_skipped_low_quality_relates_to" }));
    }

    #[test]
    fn hybrid_ee_keeps_nominal_mentions_unresolved_at_detection_time() {
        let engine = PhoenixInvarantV3::default();
        let scan = engine.scan_parts("manager waited.", &ScopeKey::default(), &[]);
        let manager = scan
            .mentions
            .iter()
            .find(|mention| mention.surface.to_ascii_lowercase().contains("manager"))
            .unwrap_or_else(|| {
                panic!(
                    "manager mention: {:?}",
                    scan.mentions
                        .iter()
                        .map(|mention| mention.surface.clone())
                        .collect::<Vec<_>>()
                )
            });
        assert!(manager.entity_ref.is_none());
        assert_eq!(manager.source, Some(MentionSource::Discovery));
        assert!(manager.kind.is_none());
    }

    #[test]
    fn ingest_persists_archives_and_sidecars() {
        let store = PhoenixOvergraphStore::open(temp_path("archives")).expect("store");
        store.init_bundle_schema().expect("schema");
        let engine = PhoenixInvarantV3::default();
        let (ingest, _artifacts) = engine
            .ingest_documents(
                &store,
                Some(&SessionId("session-1".to_owned())),
                &[IngestDocument {
                    document_id: DocumentId("doc-1".to_owned()),
                    note_id: Some(NoteId("note-1".to_owned())),
                    title: "Harbor".to_owned(),
                    text: "Ryan met Len at the harbor.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                0,
                10,
            )
            .expect("ingest");
        assert_eq!(ingest.document_count, 1);
        let archives = engine
            .load_latest_document_archives(&store, Some(&ScopeKey::default()))
            .expect("archives");
        assert_eq!(archives.len(), 1);
        let spans = engine
            .load_latest_lex_spans(&store, Some(&ScopeKey::default()))
            .expect("spans");
        assert_eq!(spans.len(), 1);
        let sidecar = engine
            .load_scope_lex_sidecar(&store, &ScopeKey::default())
            .expect("sidecar")
            .expect("present");
        assert_eq!(sidecar.spans.len(), 1);
        assert_eq!(archives[0].manifest.mention_count, 2);
        assert!(archives[0].resolved_mentions.is_empty());
    }

    #[test]
    fn ingest_native_round_trips_overgraph_store() {
        let store = PhoenixOvergraphStore::open(temp_path("native-archives")).expect("store");
        store.init_archive_schema().expect("archive schema");
        store.init_graph_kernel_schema().expect("kernel schema");
        store
            .write_kernel_checkpoint(1, "seed", &KernelGraphSnapshot::default())
            .expect("checkpoint");
        assert_eq!(store.kernel_current_generation().expect("generation"), 1);
        assert!(store
            .load_kernel_checkpoint()
            .expect("checkpoint load")
            .is_some());

        let engine = PhoenixInvarantV3::default();
        let session_id = SessionId("session-native".to_owned());
        let document = IngestDocument {
            document_id: DocumentId("doc-native-1".to_owned()),
            note_id: Some(NoteId("note-native-1".to_owned())),
            title: "Native Harbor".to_owned(),
            text: "Ryan met Len at the harbor.".to_owned(),
            scope: ScopeKey::default(),
        };

        let context = store
            .prepare_ingest_context(Some(&session_id), &[document.clone()], 0)
            .expect("prepare context");
        assert!(context.kernel_snapshot.is_some());
        assert_eq!(context.assignments.len(), 1);

        let (ingest, artifacts) = engine
            .ingest_documents_native(&store, Some(&session_id), &[document], 0, 10)
            .expect("native ingest");
        assert_eq!(ingest.document_count, 1);
        assert_eq!(artifacts.document_refs.len(), 1);
        assert_eq!(store.list_dirty_scopes().expect("dirty scopes").len(), 1);

        let manifest = store
            .load_document_manifest(&artifacts.document_refs[0])
            .expect("manifest load")
            .expect("manifest present");
        assert_eq!(manifest.document_id, "doc-native-1");

        let archives = engine
            .load_latest_document_archives_native(&store, Some(&ScopeKey::default()))
            .expect("native archives");
        assert_eq!(archives.len(), 1);
        assert_eq!(archives[0].manifest.document_id, "doc-native-1");

        assert_eq!(store.rebuild_dirty_scope_sidecars(11).expect("rebuild"), 1);
        let spans = engine
            .load_latest_lex_spans_native(&store, Some(&ScopeKey::default()))
            .expect("native spans");
        assert_eq!(spans.len(), 1);

        let graph_vertex_count = artifacts
            .document_manifests
            .iter()
            .map(|manifest| manifest.graph_vertex_count)
            .sum();
        let graph_edge_count = artifacts
            .document_manifests
            .iter()
            .map(|manifest| manifest.graph_edge_count)
            .sum();
        let session_summary = engine.merge_session_summary(
            None,
            session_id.clone(),
            artifacts.session_documents.clone(),
            artifacts.document_refs.clone(),
            artifacts.span_count,
            artifacts.discovery_candidate_count,
            graph_vertex_count,
            graph_edge_count,
            10,
        );
        engine
            .persist_session_summary_native(&store, &session_summary, 1, 10)
            .expect("persist session");
        let loaded_session = engine
            .load_latest_session_summary_native(&store, &session_id)
            .expect("session load")
            .expect("session present");
        assert_eq!(loaded_session.documents.len(), 1);
        assert_eq!(loaded_session.document_refs.len(), 1);
    }

    #[test]
    fn collective_er_resolves_pronouns_to_seeded_entities() {
        let engine = PhoenixInvarantV3::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-pronoun".to_owned()),
            note_id: None,
            title: "Pronoun".to_owned(),
            text: "Luffy waited. He smiled.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = engine.scan_parts(
            &document.text,
            &document.scope,
            &[ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            }],
        );
        let structure = engine.build_structure_parts(&document.text, &scan);
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );
        let (resolutions, resolved_mentions, aliases, diagnostics) = resolve_mentions(
            &document,
            &scan,
            &structure,
            &chunks,
            &NativeEntityMemory::default(),
        );
        assert_eq!(resolutions.len(), 2);
        let pronoun = resolved_mentions
            .iter()
            .find(|mention| mention.surface == "He")
            .expect("pronoun mention");
        assert_eq!(pronoun.decision.status, "resolved");
        assert_eq!(
            pronoun
                .entity_id
                .as_ref()
                .map(|entity_id| entity_id.0.as_str()),
            Some("luffy")
        );
        assert!(!pronoun
            .candidates
            .iter()
            .any(|candidate| candidate.source == "new_speculative"));
        assert!(aliases.is_empty());
        assert!(diagnostics.iter().any(|diagnostic| {
            matches!(
                diagnostic.code.as_str(),
                "er_pronoun_link_match" | "er_collective_merge" | "er_known_seed_match"
            )
        }));
    }

    #[test]
    fn coref_kernel_clusters_named_nominal_and_pronoun_mentions() {
        let engine = PhoenixInvarantV3::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-coref".to_owned()),
            note_id: None,
            title: "Coref".to_owned(),
            text: "Captain Luffy waited. He smiled. The captain laughed.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = scan_native_compact(
            &document.text,
            &document.scope,
            &[ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: vec!["Captain Luffy".to_owned()],
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            }],
            &engine.config.extraction,
        );
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );
        let structure = build_native_structure_rows(&document.text, &scan, &chunks);
        let coref = build_coref_rows(&scan, &structure, &engine.config.coref);

        assert!(coref.summary.cluster_count >= 1);
        assert!(coref.summary.attached_mention_count >= 1);
        assert!(coref
            .clusters
            .iter()
            .any(|cluster| cluster.pronoun_count >= 1 && cluster.named_count >= 1));
    }

    #[test]
    fn ingest_persists_compact_coref_cluster_summaries() {
        let store = PhoenixOvergraphStore::open(temp_path("coref-archive")).expect("store");
        store.init_bundle_schema().expect("schema");
        let engine = PhoenixInvarantV3::default();
        engine
            .ingest_documents(
                &store,
                Some(&SessionId("session-coref".to_owned())),
                &[IngestDocument {
                    document_id: DocumentId("doc-coref-archive".to_owned()),
                    note_id: None,
                    title: "Coref Archive".to_owned(),
                    text: "Luffy waited. Luffy smiled.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                0,
                10,
            )
            .expect("ingest");
        let archives = engine
            .load_latest_document_archives(&store, Some(&ScopeKey::default()))
            .expect("archives");
        assert_eq!(archives.len(), 1);
        assert!(archives[0].coref_summary.cluster_count >= 1);
        assert_eq!(archives[0].coref_clusters.len(), 0);
        assert_eq!(archives[0].manifest.archive_version, 4);
    }

    #[test]
    fn collective_er_does_not_confirm_aliases_from_prior_resolution_history_alone() {
        let engine = PhoenixInvarantV3::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-history-alias".to_owned()),
            note_id: None,
            title: "History".to_owned(),
            text: "Captain waited.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = engine.scan_parts(&document.text, &document.scope, &[]);
        let structure = engine.build_structure_parts(&document.text, &scan);
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );
        let mention_vertex_id = "mention::doc-history-alias:0".to_owned();
        let snapshot = KernelGraphSnapshot {
            vertices: vec![
                KernelVertex {
                    id: KernelVertexId(mention_vertex_id.clone()),
                    kind: "mention".to_owned(),
                    class: KernelVertexClass::Mention,
                    entity_facet: Some(KernelEntityFacet {
                        canonical_entity_id: Some("luffy".to_owned()),
                        surface: Some("Captain".to_owned()),
                        entity_kind: Some("Character".to_owned()),
                    }),
                    ..KernelVertex::default()
                },
                KernelVertex {
                    id: KernelVertexId("entity::luffy".to_owned()),
                    kind: "entity".to_owned(),
                    class: KernelVertexClass::Entity,
                    entity_id: Some("luffy".to_owned()),
                    ..KernelVertex::default()
                },
            ],
            asserted_edges: vec![KernelEdge {
                source_id: KernelVertexId(mention_vertex_id),
                target_id: KernelVertexId("entity::luffy".to_owned()),
                edge_type: KernelEdgeType("resolved_to".to_owned()),
                ..KernelEdge::default()
            }],
            candidate_edges: Vec::new(),
        };
        let memory = build_native_entity_memory(Some(&snapshot));
        let (_resolutions, resolved_mentions, aliases, diagnostics) =
            resolve_mentions(&document, &scan, &structure, &chunks, &memory);
        assert_eq!(resolved_mentions.len(), 1);
        assert_eq!(resolved_mentions[0].decision.status, "resolved");
        assert_eq!(
            resolved_mentions[0]
                .entity_id
                .as_ref()
                .map(|entity_id| entity_id.0.as_str()),
            Some("luffy")
        );
        assert!(aliases.is_empty());
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "er_alias_rejected_low_margin"
                && diagnostic.message.contains("not alias-specific enough")
        }));
    }

    #[test]
    fn collective_er_keeps_ambiguous_kernel_aliases_unresolved() {
        let engine = PhoenixInvarantV3::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-ambiguous".to_owned()),
            note_id: None,
            title: "Ambiguous".to_owned(),
            text: "Ace waited.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = engine.scan_parts(&document.text, &document.scope, &[]);
        let structure = engine.build_structure_parts(&document.text, &scan);
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );
        let snapshot = KernelGraphSnapshot {
            vertices: vec![
                KernelVertex {
                    id: KernelVertexId("alias::luffy::ace".to_owned()),
                    kind: "alias".to_owned(),
                    entity_facet: Some(KernelEntityFacet {
                        canonical_entity_id: Some("luffy".to_owned()),
                        surface: Some("Ace".to_owned()),
                        entity_kind: Some("Character".to_owned()),
                    }),
                    ..KernelVertex::default()
                },
                KernelVertex {
                    id: KernelVertexId("entity::luffy".to_owned()),
                    kind: "entity".to_owned(),
                    entity_id: Some("luffy".to_owned()),
                    ..KernelVertex::default()
                },
                KernelVertex {
                    id: KernelVertexId("alias::sabo::ace".to_owned()),
                    kind: "alias".to_owned(),
                    entity_facet: Some(KernelEntityFacet {
                        canonical_entity_id: Some("sabo".to_owned()),
                        surface: Some("Ace".to_owned()),
                        entity_kind: Some("Character".to_owned()),
                    }),
                    ..KernelVertex::default()
                },
                KernelVertex {
                    id: KernelVertexId("entity::sabo".to_owned()),
                    kind: "entity".to_owned(),
                    entity_id: Some("sabo".to_owned()),
                    ..KernelVertex::default()
                },
            ],
            asserted_edges: vec![
                KernelEdge {
                    source_id: KernelVertexId("alias::luffy::ace".to_owned()),
                    target_id: KernelVertexId("entity::luffy".to_owned()),
                    edge_type: KernelEdgeType("alias_of".to_owned()),
                    ..KernelEdge::default()
                },
                KernelEdge {
                    source_id: KernelVertexId("alias::sabo::ace".to_owned()),
                    target_id: KernelVertexId("entity::sabo".to_owned()),
                    edge_type: KernelEdgeType("alias_of".to_owned()),
                    ..KernelEdge::default()
                },
            ],
            candidate_edges: Vec::new(),
        };
        let memory = build_native_entity_memory(Some(&snapshot));
        let (_resolutions, resolved_mentions, aliases, diagnostics) =
            resolve_mentions(&document, &scan, &structure, &chunks, &memory);
        assert_eq!(resolved_mentions.len(), 1);
        assert_eq!(resolved_mentions[0].decision.status, "ambiguous");
        assert!(resolved_mentions[0].entity_id.is_none());
        assert!(aliases.is_empty());
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "er_ambiguous_resolution"));
    }

    #[test]
    fn collective_er_is_stable_across_identical_reruns() {
        let engine = PhoenixInvarantV3::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-stable".to_owned()),
            note_id: None,
            title: "Stable".to_owned(),
            text: "Luffy waited. He smiled.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = engine.scan_parts(
            &document.text,
            &document.scope,
            &[ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            }],
        );
        let structure = engine.build_structure_parts(&document.text, &scan);
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );

        let run_once = || {
            let (resolutions, resolved_mentions, aliases, diagnostics) = resolve_mentions(
                &document,
                &scan,
                &structure,
                &chunks,
                &NativeEntityMemory::default(),
            );
            (
                resolutions
                    .iter()
                    .map(|resolution| {
                        (
                            resolution.mention_id.0.clone(),
                            resolution
                                .entity_id
                                .as_ref()
                                .map(|entity_id| entity_id.0.clone()),
                            resolution
                                .candidates
                                .iter()
                                .map(|candidate| {
                                    (
                                        candidate.entity_id.clone(),
                                        candidate.source.clone(),
                                        candidate.score_millis,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
                resolved_mentions
                    .iter()
                    .map(|mention| {
                        (
                            mention.mention_id.0.clone(),
                            mention.decision.status.clone(),
                            mention
                                .entity_id
                                .as_ref()
                                .map(|entity_id| entity_id.0.clone()),
                        )
                    })
                    .collect::<Vec<_>>(),
                aliases
                    .iter()
                    .map(|alias| {
                        (
                            alias.entity_id.0.clone(),
                            alias.normalized.clone(),
                            alias.confidence_millis,
                        )
                    })
                    .collect::<Vec<_>>(),
                diagnostics
                    .iter()
                    .map(|diagnostic| (diagnostic.code.clone(), diagnostic.message.clone()))
                    .collect::<Vec<_>>(),
            )
        };

        let first = run_once();
        let second = run_once();
        let third = run_once();
        assert_eq!(first, second);
        assert_eq!(second, third);
    }

    #[test]
    fn background_verifier_configuration_degrades_gracefully_without_feature_or_paths() {
        let mut config = InvarantV3Config::default();
        config.verification.enable_background_ner_verifier = true;
        let engine = PhoenixInvarantV3::new(config);
        let document = IngestDocument {
            document_id: DocumentId("doc-verifier".to_owned()),
            note_id: None,
            title: "Verifier".to_owned(),
            text: "Luffy met Ace.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = scan_native_compact(
            &document.text,
            &document.scope,
            &[ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            }],
            &engine.config.extraction,
        );
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );
        let structure = build_native_structure_rows(&document.text, &scan, &chunks);
        let coref = build_coref_rows(&scan, &structure, &engine.config.coref);
        let (resolutions, alias_confirmation_ords, _summary, _diagnostics, entity_ids, _) =
            resolve_mentions_compact_native(
                &document,
                &scan,
                &coref,
                &chunks,
                &NativeEntityMemory::default(),
            );
        let summary = run_background_verification(
            &engine.config.verification,
            &document,
            &scan,
            &resolutions,
            &alias_confirmation_ords,
            &entity_ids,
        );
        assert!(summary.task_count >= alias_confirmation_ords.len());
        assert!(summary
            .diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code.starts_with("er_background_verifier_")));
    }
}

impl Default for InvarantV3Config {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 64,
            extraction: InvarantV3ExtractionConfig::default(),
            coref: InvarantV3CorefConfig::default(),
            verification: InvarantV3VerificationConfig::default(),
        }
    }
}

pub struct PhoenixInvarantV3 {
    config: InvarantV3Config,
    compiler: SurfaceCompiler,
}

pub type PhoenixIngestNative = PhoenixInvarantV3;

impl Default for PhoenixInvarantV3 {
    fn default() -> Self {
        Self::new(InvarantV3Config::default())
    }
}

#[derive(Clone, Debug, Default)]
pub struct V2IngestArtifacts {
    pub kernel_batches: Vec<KernelMutationBatch>,
    pub session_documents: Vec<SessionDocumentState>,
    pub document_refs: Vec<DocumentRevisionRef>,
    pub document_manifests: Vec<DocumentManifest>,
    pub manifest_namespaces: Vec<String>,
    pub span_count: usize,
    pub discovery_candidate_count: usize,
    pub touched_scopes: Vec<ScopeKey>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeNerMention {
    pub start: u32,
    pub end: u32,
    pub sentence_index: usize,
    pub surface: String,
    pub normalized: String,
    pub label: String,
    pub source: Option<String>,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeNerRaceReport {
    pub scan_ms: u64,
    pub sentence_count: usize,
    pub mention_count: usize,
    pub named_count: usize,
    pub nominal_count: usize,
    pub pronoun_count: usize,
    pub discovery_count: usize,
    pub mentions: Vec<NativeNerMention>,
}

#[derive(Clone, Debug)]
struct BoundaryRecord {
    label: String,
    range: TextRange,
    chapter_id: u32,
    is_chapter: bool,
}

#[derive(Clone, Debug)]
struct PreparedMention {
    mention_ix: usize,
    surface_ord: u32,
    chunk_ix: Option<u32>,
    linked_mentions: SmallVec<[usize; 4]>,
    resolver_entity_links: SmallVec<[PreparedResolverEntityLink; 4]>,
}

#[derive(Clone, Debug)]
struct PreparedResolverEntityLink {
    entity_id: String,
    source: &'static str,
    score_millis: i32,
    evidence_detail: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CorefMentionKind {
    Named,
    Nominal,
    Pronoun,
}

#[derive(Clone, Debug)]
struct SurfaceAtom {
    normalized: Box<str>,
    is_pronoun: bool,
}

#[derive(Clone, Debug, Default)]
struct SurfaceLibraryBinding {
    entity_ref: Option<MentionEntityRef>,
    ambiguous: bool,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug, Default)]
struct AlexEntityLibrary {
    surface_bindings: Vec<SurfaceLibraryBinding>,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug)]
struct MentionFamily {
    mention_kind: CorefMentionKind,
    representative_mention_ix: usize,
    member_indexes: Vec<usize>,
    resolved_entity_ref: Option<MentionEntityRef>,
    ambiguous: bool,
}

#[cfg_attr(not(test), allow(dead_code))]
#[derive(Clone, Debug)]
struct NativeScanRows {
    sentences: Vec<SentenceSpan>,
    mentions: Vec<MentionSpan>,
    resolver_links: Vec<ResolverLink>,
    narrative_hits: Vec<NarrativeVerbHit>,
    mention_families: Vec<MentionFamily>,
    family_ord_by_mention: Vec<Option<u32>>,
    entity_library: AlexEntityLibrary,
    surface_atoms: Vec<SurfaceAtom>,
    surface_counts: Vec<u32>,
    acronym_values: Vec<Box<str>>,
    surface_acronym_ords: Vec<Option<u32>>,
    mention_surface_ords: Vec<u32>,
    mention_coref_kinds: Vec<CorefMentionKind>,
    discovery_count: usize,
    detected_named_count: usize,
    detected_nominal_count: usize,
    detected_pronoun_count: usize,
}

#[derive(Clone, Debug)]
struct NativeRelationSeed {
    sentence_index: usize,
    relation_type: String,
    subject_mention_ix: Option<usize>,
    object_mention_ix: Option<usize>,
}

#[derive(Clone, Debug)]
struct NativeStructureRows {
    relation_seeds: Vec<NativeRelationSeed>,
    sentence_chunk_indexes: Vec<Option<u32>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CorefPairRoute {
    ExactSurface,
    AliasOrAcronym,
    TitleContainment,
    PronounToNamed,
    PronounToNominal,
    NominalToNamed,
    OtherCompatible,
}

impl CorefPairRoute {
    fn bit(self) -> u32 {
        match self {
            Self::ExactSurface => 1 << 0,
            Self::AliasOrAcronym => 1 << 1,
            Self::TitleContainment => 1 << 2,
            Self::PronounToNamed => 1 << 3,
            Self::PronounToNominal => 1 << 4,
            Self::NominalToNamed => 1 << 5,
            Self::OtherCompatible => 1 << 6,
        }
    }

    fn base_score(self) -> i32 {
        match self {
            Self::ExactSurface => 900,
            Self::AliasOrAcronym => 840,
            Self::TitleContainment => 780,
            Self::PronounToNamed => 760,
            Self::PronounToNominal => 700,
            Self::NominalToNamed => 720,
            Self::OtherCompatible => 620,
        }
    }
}

#[derive(Clone, Debug)]
struct CorefMentionRow {
    sentence_index: usize,
    chunk_index: Option<u32>,
    mention_kind: CorefMentionKind,
    has_known_seed: bool,
    kind: Option<EntityKind>,
    surface_ord: u32,
    acronym_ord: Option<u32>,
    head_ord: Option<u32>,
    title_like: bool,
}

#[derive(Clone, Copy, Debug)]
struct CorefAntecedentCandidate {
    mention_ix: usize,
    score_millis: i32,
    route: CorefPairRoute,
}

#[derive(Clone, Copy, Debug)]
struct CorefAcceptedEdge {
    left_ix: usize,
    right_ix: usize,
    route: CorefPairRoute,
    score_millis: i32,
}

#[derive(Clone, Debug, Default)]
struct CorefClusterState {
    member_indexes: Vec<usize>,
    representative_mention_ix: usize,
    most_recent_mention_ix: usize,
    best_named_mention_ix: Option<usize>,
    best_seeded_mention_ix: Option<usize>,
    first_sentence_index: usize,
    last_sentence_index: usize,
    chunk_indexes: SmallVec<[u32; 8]>,
    named_count: usize,
    nominal_count: usize,
    pronoun_count: usize,
    route_mix_bits: u32,
    max_score_millis: i32,
    ambiguous: bool,
}

#[derive(Clone, Debug, Default)]
struct NativeCorefRows {
    rows: Vec<CorefMentionRow>,
    cluster_by_mention: Vec<u32>,
    representative_by_mention: Vec<Option<usize>>,
    candidate_links_by_mention: Vec<SmallVec<[usize; 2]>>,
    clusters: Vec<CorefClusterState>,
    summary: NativeCorefSummary,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CandidateSourceKind {
    Seed,
    KernelAlias,
    KernelResolved,
    KernelCandidate,
    CorefCluster,
    PronounLink,
    AliasLink,
    LocalSurface,
    NewSpeculative,
}

impl CandidateSourceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Seed => "seed",
            Self::KernelAlias => "kernel_alias",
            Self::KernelResolved => "kernel_resolved",
            Self::KernelCandidate => "kernel_candidate",
            Self::CorefCluster => "coref_cluster",
            Self::PronounLink => "pronoun_link",
            Self::AliasLink => "alias_link",
            Self::LocalSurface => "local_surface",
            Self::NewSpeculative => "new_speculative",
        }
    }

    fn is_strong(self) -> bool {
        matches!(
            self,
            Self::Seed
                | Self::KernelAlias
                | Self::KernelResolved
                | Self::CorefCluster
                | Self::PronounLink
                | Self::AliasLink
        )
    }
}

#[derive(Clone, Debug)]
struct CandidateSlot {
    entity_ord: u32,
    source: CandidateSourceKind,
    score_millis: i32,
    evidence_bits: u16,
    evidence: SmallVec<[CandidateEvidence; 4]>,
}

#[derive(Clone, Debug, Default)]
struct MentionState {
    candidates: SmallVec<[CandidateSlot; 6]>,
}

#[derive(Clone, Copy, Debug)]
struct CompactResolverEntityLink {
    entity_ord: u32,
    source: CandidateSourceKind,
    score_millis: i32,
}

#[derive(Clone, Copy, Debug)]
struct CompactCandidateSlot {
    entity_ord: u32,
    source: CandidateSourceKind,
    score_millis: i32,
    evidence_bits: u16,
}

#[derive(Clone, Copy, Debug, Default)]
struct CompactResolutionOrd {
    mention_index: usize,
    entity_ord: Option<u32>,
    chunk_index: Option<u32>,
}

#[derive(Clone, Debug)]
struct AliasConfirmationOrd {
    alias_surface: String,
    normalized: String,
    entity_ord: u32,
    confidence_millis: u32,
    mention_index: usize,
}

#[derive(Clone, Copy, Debug)]
struct BestCandidateSummary {
    entity_ord: u32,
    source: CandidateSourceKind,
    score_millis: i32,
}

#[derive(Clone, Copy, Debug)]
struct CompactBestCandidateSummary {
    entity_ord: u32,
    source: CandidateSourceKind,
    score_millis: i32,
}

#[derive(Default)]
struct NativeEntityMemory {
    kernel: DeterministicKernel,
    entity_index: KernelEntitySidecar,
    entity_kinds: FxHashMap<String, String>,
    known_aliases: FxHashSet<(String, String)>,
}

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "background-verifier"), allow(dead_code))]
struct VerificationTask {
    mention_index: usize,
    normalized_surface: String,
    window_text: String,
    expected_kind: Option<EntityKind>,
}

#[derive(Clone, Debug, Default)]
struct BackgroundVerificationSummary {
    task_count: usize,
    supported_alias_count: usize,
    supported_type_count: usize,
    diagnostics: Vec<Diagnostic>,
}

#[cfg(feature = "background-verifier")]
#[derive(Debug, Error)]
enum BackgroundVerifierError {
    #[error("invalid verifier path: {0}")]
    InvalidPath(String),
    #[error("failed to load verifier model: {0}")]
    ModelLoad(String),
    #[error("verifier inference failed: {0}")]
    Inference(String),
}

#[cfg(feature = "background-verifier")]
struct BackgroundNerVerifier {
    model: GLiNER<SpanMode>,
}

#[cfg(feature = "background-verifier")]
impl BackgroundNerVerifier {
    fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self, BackgroundVerifierError> {
        let params = Parameters::default().with_threshold(0.75);
        let runtime_params = RuntimeParameters::default();
        let model = GLiNER::<SpanMode>::new(
            params,
            runtime_params,
            tokenizer_path.to_str().ok_or_else(|| {
                BackgroundVerifierError::InvalidPath(tokenizer_path.display().to_string())
            })?,
            model_path.to_str().ok_or_else(|| {
                BackgroundVerifierError::InvalidPath(model_path.display().to_string())
            })?,
        )
        .map_err(|error| BackgroundVerifierError::ModelLoad(error.to_string()))?;
        Ok(Self { model })
    }

    fn extract(&self, text: &str) -> Result<Vec<(String, String, f32)>, BackgroundVerifierError> {
        const LABELS: [&str; 6] = [
            "person",
            "organization",
            "location",
            "event",
            "item",
            "concept",
        ];
        let input = TextInput::from_str(&[text], &LABELS)
            .map_err(|error| BackgroundVerifierError::Inference(error.to_string()))?;
        let output = self
            .model
            .inference(input)
            .map_err(|error| BackgroundVerifierError::Inference(error.to_string()))?;
        let mut values = Vec::new();
        for spans in output.spans {
            for span in spans {
                values.push((
                    span.text().to_owned(),
                    span.class().to_owned(),
                    span.probability(),
                ));
            }
        }
        Ok(values)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ResolvedMentionKind {
    Resolved,
    Ambiguous,
    Unresolved,
}

#[derive(Clone, Debug)]
struct ResolutionDecisionState {
    kind: ResolvedMentionKind,
    entity_id: Option<EntityId>,
    confidence_millis: u32,
    margin_millis: u32,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
struct DetailedMentionResolution {
    mention_ix: usize,
    mention_id: phoenix_semantic_v2::MentionId,
    entity_id: Option<EntityId>,
    candidates: Vec<CandidateEntity>,
    decision: ResolutionDecisionState,
}

#[derive(Clone, Debug)]
struct EntityAccumulatorOrd {
    canonical_name: String,
    aliases: SmallVec<[String; 4]>,
    kind: Option<EntityKind>,
    mention_count: usize,
    chunk_indexes: SmallVec<[u32; 4]>,
}

#[derive(Clone, Debug)]
struct NativeScanBundle {
    scan: NativeScanRows,
    boundaries: Vec<BoundaryRecord>,
    chunks: Vec<ChunkRecord>,
    indexed_spans: Vec<IndexedSpan>,
    structure: NativeStructureRows,
    coref: NativeCorefRows,
}

#[derive(Clone, Debug)]
struct NativeResolutionBundle {
    alias_confirmations: Vec<AliasConfirmation>,
    coref_clusters: Vec<CorefClusterRecord>,
    er_summary: NativeErSummary,
    coref_summary: NativeCorefSummary,
    entities: Vec<SemanticEntityRecord>,
    relations: Vec<SemanticRelationRecord>,
    discovery_count: usize,
    diagnostics: Vec<Diagnostic>,
    kernel_batch: KernelMutationBatch,
    candidate_kernel_batch: Option<KernelMutationBatch>,
}

#[derive(Clone, Debug)]
struct PreparedDocumentDraft {
    prepared: PreparedDocument,
    document_summary: IngestDocumentSummary,
    session_document: SessionDocumentState,
    span_count: usize,
    discovery_count: usize,
    diagnostics: Vec<Diagnostic>,
    candidate_kernel_batch: Option<KernelMutationBatch>,
}

struct DocumentOutcome {
    document_summary: IngestDocumentSummary,
    session_document: SessionDocumentState,
    scope: ScopeKey,
    span_count: usize,
    discovery_count: usize,
    diagnostics: Vec<Diagnostic>,
    kernel_batch: KernelMutationBatch,
    candidate_kernel_batch: Option<KernelMutationBatch>,
    archive: DocumentArchive,
}

#[derive(Clone, Debug)]
struct IngestedDocumentOutcome {
    assignment: DocumentOrdinalAssignment,
    document_summary: IngestDocumentSummary,
    session_document: SessionDocumentState,
    span_count: usize,
    discovery_count: usize,
    diagnostics: Vec<Diagnostic>,
    candidate_kernel_batch: Option<KernelMutationBatch>,
}

impl PhoenixInvarantV3 {
    pub fn new(config: InvarantV3Config) -> Self {
        let compiler = machine_compiler_for_extraction(&config.extraction);
        Self { config, compiler }
    }

    pub fn config(&self) -> &InvarantV3Config {
        &self.config
    }

    pub fn benchmark_native_ner(
        &self,
        text: &str,
        scope: &ScopeKey,
        resolver_seed: &[ResolverEntitySeed],
    ) -> NativeNerRaceReport {
        let started = Instant::now();
        let scan = scan_native_compact(text, scope, resolver_seed, &self.config.extraction);
        let scan_ms = started.elapsed().as_millis() as u64;
        NativeNerRaceReport {
            scan_ms,
            sentence_count: scan.sentences.len(),
            mention_count: scan.mentions.len(),
            named_count: scan.detected_named_count,
            nominal_count: scan.detected_nominal_count,
            pronoun_count: scan.detected_pronoun_count,
            discovery_count: scan.discovery_count,
            mentions: scan
                .mentions
                .iter()
                .zip(scan.mention_coref_kinds.iter())
                .map(|(mention, coref_kind)| NativeNerMention {
                    start: mention.range.start,
                    end: mention.range.end,
                    sentence_index: mention.sentence_index,
                    surface: mention.surface.clone(),
                    normalized: normalize_surface(&mention.surface),
                    label: native_ner_label(mention, *coref_kind).to_owned(),
                    source: mention
                        .source
                        .as_ref()
                        .map(native_mention_source_name)
                        .map(str::to_owned),
                    confidence: mention.confidence,
                })
                .collect(),
        }
    }

    pub fn scan_parts(
        &self,
        text: &str,
        scope: &ScopeKey,
        resolver_seed: &[ResolverEntitySeed],
    ) -> ScanArtifact {
        let progress = native_progress_enabled();
        let phase_started = Instant::now();
        let scan = self
            .compiler
            .compatibility_scan_parts(text, scope, resolver_seed);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=compatibility_scan_parts wall_ms={} sentences={} mentions={} resolver_links={} narrative_hits={}",
                phase_started.elapsed().as_millis(),
                scan.sentences.len(),
                scan.mentions.len(),
                scan.resolver_links.len(),
                scan.narrative_hits.len(),
            );
        }
        scan
    }

    pub fn build_structure_parts(&self, text: &str, scan: &ScanArtifact) -> StructureArtifact {
        self.compiler.compatibility_structure_parts(text, scan)
    }

    pub fn ingest_documents(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        session_id: Option<&SessionId>,
        documents: &[IngestDocument],
        revision: u64,
        created_at: i64,
    ) -> Result<(IngestResult, V2IngestArtifacts), StoreError> {
        let entity_memory = NativeEntityMemory::default();
        let assignments = documents
            .iter()
            .map(|document| DocumentOrdinalAssignment {
                document_id: document.document_id.0.clone(),
                scope: document.scope.clone(),
                scope_key: scope_storage_key(&document.scope),
                scope_ord: ScopeOrd(0),
                document_ord: DocumentOrd(0),
                revision: revision + 1,
            })
            .collect::<Vec<_>>();
        let drafts = documents
            .par_iter()
            .enumerate()
            .map(|(index, document)| {
                self.build_document_outcome(
                    document,
                    session_id,
                    &assignments[index],
                    &entity_memory,
                    created_at,
                )
            })
            .collect::<Vec<_>>();
        let mut outcomes = Vec::with_capacity(drafts.len());
        for draft in drafts {
            outcomes.push(draft?);
        }
        outcomes.sort_by(|left, right| {
            left.document_summary
                .document_id
                .0
                .cmp(&right.document_summary.document_id.0)
        });

        let mut touched_scope_keys = FxHashSet::default();
        for outcome in &outcomes {
            self.persist_document_archive(store, &outcome.archive, revision + 1, created_at)?;
            touched_scope_keys.insert(scope_storage_key(&outcome.scope));
        }

        let mut touched_scopes = outcomes
            .iter()
            .map(|outcome| outcome.scope.clone())
            .collect::<Vec<_>>();
        touched_scopes.sort_by_key(scope_storage_key);
        touched_scopes.dedup_by(|left, right| scope_storage_key(left) == scope_storage_key(right));
        for scope in &touched_scopes {
            let sidecar = self.build_scope_lex_sidecar(store, scope, revision + 1, created_at)?;
            self.persist_scope_lex_sidecar(store, &sidecar, revision + 1, created_at)?;
        }

        let total_entities = outcomes
            .iter()
            .map(|outcome| outcome.archive.entities.len())
            .sum();
        let total_mentions = outcomes
            .iter()
            .map(|outcome| outcome.archive.manifest.mention_count)
            .sum();
        let total_discovery = outcomes.iter().map(|outcome| outcome.discovery_count).sum();
        let total_span_count = outcomes.iter().map(|outcome| outcome.span_count).sum();
        let total_leaves = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.leaf_count)
            .sum();
        let total_chapters = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.chapter_count)
            .sum();
        let total_boundaries = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.boundary_count)
            .sum();
        let total_parents = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.parent_count)
            .sum();
        let graph_edge_count: usize = outcomes
            .iter()
            .map(|outcome| outcome.kernel_batch.edges.len())
            .sum();
        let graph_vertex_count: usize = outcomes
            .iter()
            .map(|outcome| outcome.kernel_batch.vertices.len())
            .sum();
        let mut ingest_diagnostics = vec![
            Diagnostic {
                code: "PX_INGEST_INVARANT_V2".to_owned(),
                message: format!(
                    "Invarant V2 ingested {} archives, {} chunks, {} entities, {} graph vertices, and {} graph edges.",
                    outcomes.len(),
                    total_leaves,
                    total_entities,
                    graph_vertex_count,
                    graph_edge_count
                ),
            },
            Diagnostic {
                code: "PX_INGEST_INVARANT_V2_ARCHIVE".to_owned(),
                message: format!(
                    "Native V2 persisted {} document archives, {} scope sidecars, and emitted {} kernel vertices.",
                    outcomes.len(),
                    touched_scope_keys.len(),
                    graph_vertex_count
                ),
            },
        ];
        ingest_diagnostics.extend(
            outcomes
                .iter()
                .flat_map(|outcome| outcome.diagnostics.iter().cloned()),
        );

        Ok((
            IngestResult {
                session_id: session_id.cloned(),
                document_count: outcomes.len(),
                warning_count: 2,
                documents: outcomes
                    .iter()
                    .map(|outcome| outcome.document_summary.clone())
                    .collect(),
                chunk_stats: Some(phoenix_types::ChunkStats {
                    documents: outcomes.len(),
                    total_chapters,
                    total_boundaries,
                    total_parents,
                    total_leaves,
                }),
                graph_summary: Some(phoenix_types::GraphSummary {
                    documents: outcomes.len(),
                    total_chapters,
                    total_boundaries,
                    total_leaves,
                    total_entities,
                    total_mentions,
                    total_edges: graph_edge_count,
                    cross_chapter_links: 0,
                }),
                entity_summary: Some(phoenix_types::EntitySummary {
                    total_entities,
                    total_aliases: outcomes
                        .iter()
                        .flat_map(|outcome| outcome.archive.entities.iter())
                        .map(|entity| entity.aliases.len())
                        .sum(),
                    total_mentions,
                    multi_chapter_entities: 0,
                }),
                discovery_summary: Some(phoenix_types::DiscoverySummary {
                    candidate_count: total_discovery,
                    mention_count: total_mentions,
                    persisted_count: total_entities,
                }),
                retrieval_summary: Some(phoenix_types::RetrievalSummary {
                    qgram_documents: outcomes.len(),
                    gldr_chunks: total_leaves,
                    gldr_entities: total_entities,
                    gldr_edges: graph_edge_count,
                    raptor_documents: outcomes.len(),
                    raptor_leaves: total_leaves,
                    raptor_enabled: true,
                }),
                relation_counts: Vec::new(),
                diagnostics: ingest_diagnostics,
            },
            V2IngestArtifacts {
                kernel_batches: outcomes
                    .iter()
                    .flat_map(|outcome| {
                        std::iter::once(outcome.kernel_batch.clone())
                            .chain(outcome.candidate_kernel_batch.clone())
                    })
                    .collect(),
                session_documents: outcomes
                    .iter()
                    .map(|outcome| outcome.session_document.clone())
                    .collect(),
                document_refs: outcomes
                    .iter()
                    .map(|outcome| DocumentRevisionRef {
                        document_id: outcome.archive.manifest.document_id.clone(),
                        scope: outcome.archive.manifest.scope.clone(),
                        scope_ord: outcome.archive.manifest.scope_ord,
                        document_ord: outcome.archive.manifest.document_ord,
                        revision: outcome.archive.manifest.revision,
                    })
                    .collect(),
                document_manifests: outcomes
                    .iter()
                    .map(|outcome| outcome.archive.manifest.clone())
                    .collect(),
                manifest_namespaces: vec![
                    "invarant-v2.document".to_owned(),
                    "invarant-v2.session".to_owned(),
                    "invarant-v2.scope_lex".to_owned(),
                ],
                span_count: total_span_count,
                discovery_candidate_count: total_discovery,
                touched_scopes,
            },
        ))
    }

    pub fn ingest_documents_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        session_id: Option<&SessionId>,
        documents: &[IngestDocument],
        revision: u64,
        created_at: i64,
    ) -> Result<(IngestResult, V2IngestArtifacts), StoreError> {
        let progress = native_progress_enabled();
        let started = Instant::now();
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=prepare_ingest_context start documents={}",
                documents.len()
            );
        }
        let context = store.prepare_ingest_context(session_id, documents, revision)?;
        let entity_memory = build_native_entity_memory(context.kernel_snapshot.as_ref());
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=prepare_ingest_context finish assignments={} wall_ms={}",
                context.assignments.len(),
                started.elapsed().as_millis()
            );
        }
        let started = Instant::now();
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=build_document_outcome start documents={}",
                documents.len()
            );
        }
        let draft_results = documents
            .par_iter()
            .enumerate()
            .map(|(index, document)| {
                self.build_document_prepared_outcome(
                    document,
                    session_id,
                    &context.assignments[index],
                    &entity_memory,
                    created_at,
                )
            })
            .collect::<Vec<_>>();
        let mut drafts = Vec::with_capacity(draft_results.len());
        for draft in draft_results {
            drafts.push(draft?);
        }
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=build_document_outcome finish outcomes={} wall_ms={}",
                drafts.len(),
                started.elapsed().as_millis()
            );
        }
        drafts.sort_by(|left, right| {
            left.document_summary
                .document_id
                .0
                .cmp(&right.document_summary.document_id.0)
        });

        let mut prepared = Vec::with_capacity(drafts.len());
        let mut outcomes = Vec::with_capacity(drafts.len());
        for draft in drafts {
            let assignment = draft.prepared.assignment.clone();
            prepared.push(draft.prepared);
            outcomes.push(IngestedDocumentOutcome {
                assignment,
                document_summary: draft.document_summary,
                session_document: draft.session_document,
                span_count: draft.span_count,
                discovery_count: draft.discovery_count,
                diagnostics: draft.diagnostics,
                candidate_kernel_batch: draft.candidate_kernel_batch,
            });
        }

        let dirty_scopes = self.build_dirty_scope_records(&outcomes, created_at);
        let started = Instant::now();
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=persist_prepared_documents start documents={} dirty_scopes={}",
                prepared.len(),
                dirty_scopes.len()
            );
        }
        store.persist_prepared_documents(&prepared, None, &dirty_scopes, created_at)?;
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=persist_prepared_documents finish documents={} dirty_scopes={} wall_ms={}",
                prepared.len(),
                dirty_scopes.len(),
                started.elapsed().as_millis()
            );
        }

        let total_entities = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.entity_count)
            .sum();
        let total_mentions: usize = prepared
            .iter()
            .map(|document| document.manifest.mention_count)
            .sum();
        let total_discovery = outcomes.iter().map(|outcome| outcome.discovery_count).sum();
        let total_span_count = outcomes.iter().map(|outcome| outcome.span_count).sum();
        let total_leaves = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.leaf_count)
            .sum();
        let total_chapters = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.chapter_count)
            .sum();
        let total_boundaries = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.boundary_count)
            .sum();
        let total_parents = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.parent_count)
            .sum();
        let graph_edge_count: usize = prepared
            .iter()
            .map(|document| document.kernel_batch.edges.len())
            .sum();
        let graph_vertex_count: usize = prepared
            .iter()
            .map(|document| document.kernel_batch.vertices.len())
            .sum();
        let touched_scopes = dirty_scopes
            .iter()
            .map(|record| record.scope.clone())
            .collect::<Vec<_>>();
        let mut ingest_diagnostics = vec![
            Diagnostic {
                code: "PX_INGEST_INVARANT_V2".to_owned(),
                message: format!(
                    "Invarant V2 ingested {} archives, {} chunks, {} entities, {} graph vertices, and {} graph edges.",
                    outcomes.len(),
                    total_leaves,
                    total_entities,
                    graph_vertex_count,
                    graph_edge_count
                ),
            },
            Diagnostic {
                code: "PX_INGEST_INVARANT_V2_SEGMENTED".to_owned(),
                message: format!(
                    "Native V2 persisted {} manifests, {} prepared segment sets, and marked {} scopes dirty without synchronous sidecar rebuilds.",
                    outcomes.len(),
                    prepared.len(),
                    dirty_scopes.len()
                ),
            },
        ];
        ingest_diagnostics.extend(
            outcomes
                .iter()
                .flat_map(|outcome| outcome.diagnostics.iter().cloned()),
        );

        Ok((
            IngestResult {
                session_id: session_id.cloned(),
                document_count: outcomes.len(),
                warning_count: 2,
                documents: outcomes
                    .iter()
                    .map(|outcome| outcome.document_summary.clone())
                    .collect(),
                chunk_stats: Some(phoenix_types::ChunkStats {
                    documents: outcomes.len(),
                    total_chapters,
                    total_boundaries,
                    total_parents,
                    total_leaves,
                }),
                graph_summary: Some(phoenix_types::GraphSummary {
                    documents: outcomes.len(),
                    total_chapters,
                    total_boundaries,
                    total_leaves,
                    total_entities,
                    total_mentions,
                    total_edges: graph_edge_count,
                    cross_chapter_links: 0,
                }),
                entity_summary: Some(phoenix_types::EntitySummary {
                    total_entities,
                    total_aliases: prepared
                        .iter()
                        .map(|document| document.manifest.alias_count)
                        .sum(),
                    total_mentions,
                    multi_chapter_entities: 0,
                }),
                discovery_summary: Some(phoenix_types::DiscoverySummary {
                    candidate_count: total_discovery,
                    mention_count: total_mentions,
                    persisted_count: total_entities,
                }),
                retrieval_summary: Some(phoenix_types::RetrievalSummary {
                    qgram_documents: outcomes.len(),
                    gldr_chunks: total_leaves,
                    gldr_entities: total_entities,
                    gldr_edges: graph_edge_count,
                    raptor_documents: outcomes.len(),
                    raptor_leaves: total_leaves,
                    raptor_enabled: true,
                }),
                relation_counts: Vec::new(),
                diagnostics: ingest_diagnostics,
            },
            {
                let session_documents = outcomes
                    .iter()
                    .map(|outcome| outcome.session_document.clone())
                    .collect::<Vec<_>>();
                let mut kernel_batches = Vec::new();
                let mut document_refs = Vec::new();
                let mut document_manifests = Vec::new();

                for (document, outcome) in prepared.into_iter().zip(outcomes.into_iter()) {
                    let manifest = document.manifest;
                    document_refs.push(DocumentRevisionRef {
                        document_id: manifest.document_id.clone(),
                        scope: manifest.scope.clone(),
                        scope_ord: manifest.scope_ord,
                        document_ord: manifest.document_ord,
                        revision: manifest.revision,
                    });
                    kernel_batches.push(document.kernel_batch);
                    if let Some(candidate_batch) = outcome.candidate_kernel_batch {
                        kernel_batches.push(candidate_batch);
                    }
                    document_manifests.push(manifest);
                }

                V2IngestArtifacts {
                    kernel_batches,
                    session_documents,
                    document_refs,
                    document_manifests,
                    manifest_namespaces: vec![
                        "invarant-v2.document".to_owned(),
                        "invarant-v2.session".to_owned(),
                        "invarant-v2.scope_lex".to_owned(),
                    ],
                    span_count: total_span_count,
                    discovery_candidate_count: total_discovery,
                    touched_scopes,
                }
            },
        ))
    }

    fn persist_value<T: Serialize>(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        mut header: BundleHeader,
        value: &T,
    ) -> Result<(), StoreError> {
        let payload = encode_archive(value)?;
        header.byte_len = payload.len();
        store.put_bundle(&header, &payload)
    }

    fn load_latest_value<T: DeserializeOwned>(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        kind: BundleKind,
        scope: Option<&str>,
        entity_key: &str,
    ) -> Result<Option<T>, StoreError> {
        let header = store
            .list_bundle_headers(kind, scope)?
            .into_iter()
            .filter(|header| header.key.entity_key == entity_key)
            .max_by_key(|header| header.key.revision);
        let Some(header) = header else {
            return Ok(None);
        };
        let Some(payload) = store.get_bundle(&header.key)? else {
            return Ok(None);
        };
        Ok(Some(decode_archive(&payload)?))
    }

    fn build_scope_lex_sidecar(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        scope: &ScopeKey,
        revision: u64,
        created_at: i64,
    ) -> Result<ScopeLexSidecar, StoreError> {
        let archives = self.load_latest_document_archives(store, Some(scope))?;
        let mut spans = Vec::new();
        let mut document_ids = Vec::with_capacity(archives.len());
        let mut entries = FxHashMap::<String, AliasEntry>::default();
        let mut entity_ids = FxHashSet::<String>::default();

        for archive in archives {
            document_ids.push(archive.manifest.document_id.clone());
            spans.extend(archive.indexed_spans);
            for entity in archive.entities {
                entity_ids.insert(entity.entity_id.0.clone());
                let forms = std::iter::once(entity.canonical_name)
                    .chain(entity.aliases.into_iter())
                    .collect::<SmallVec<[String; 4]>>();
                for form in forms {
                    let normalized = normalize_surface(&form);
                    if normalized.is_empty() {
                        continue;
                    }
                    let entry = entries
                        .entry(normalized.clone())
                        .or_insert_with(|| AliasEntry {
                            normalized,
                            postings: Vec::new(),
                        });
                    if let Some(existing) = entry
                        .postings
                        .iter_mut()
                        .find(|posting| posting.entity_id == entity.entity_id.0)
                    {
                        existing.mention_count += entity.mention_count;
                    } else {
                        entry.postings.push(AliasPosting {
                            entity_id: entity.entity_id.0.clone(),
                            document_id: archive.manifest.document_id.clone(),
                            mention_count: entity.mention_count,
                        });
                    }
                }
            }
        }

        spans.sort_by(|left, right| left.span_id.cmp(&right.span_id));
        document_ids.sort();
        document_ids.dedup();

        let mut alias_entries = entries.into_values().collect::<Vec<_>>();
        alias_entries.sort_by(|left, right| left.normalized.cmp(&right.normalized));
        for entry in &mut alias_entries {
            entry.postings.sort_by(|left, right| {
                left.entity_id
                    .cmp(&right.entity_id)
                    .then_with(|| left.document_id.cmp(&right.document_id))
            });
        }

        Ok(ScopeLexSidecar {
            scope: scope.clone(),
            scope_key: scope_storage_key(scope),
            scope_ord: None,
            spans,
            alias_entries,
            document_ids,
            entity_count: entity_ids.len(),
            generated_at: created_at,
            generation: revision,
        })
    }

    pub fn persist_document_archive(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        archive: &DocumentArchive,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        self.persist_value(
            store,
            document_archive_header(
                &archive.manifest.scope,
                &archive.manifest.document_id,
                revision,
                0,
                created_at,
            ),
            archive,
        )
    }

    pub fn persist_session_summary(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        summary: &SessionArchive,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        self.persist_value(
            store,
            session_archive_header(&summary.session_id, revision, 0, created_at),
            summary,
        )
    }

    pub fn persist_scope_lex_sidecar(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        sidecar: &ScopeLexSidecar,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        self.persist_value(
            store,
            scope_lex_sidecar_header(&sidecar.scope, revision, 0, created_at),
            sidecar,
        )
    }

    pub fn persist_session_summary_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        summary: &SessionArchive,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        store.persist_session_archive(summary, revision, created_at)
    }

    pub fn load_latest_session_summary(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        session_id: &SessionId,
    ) -> Result<Option<SessionArchive>, StoreError> {
        self.load_latest_value(
            store,
            BundleKind::SessionArchive,
            Some(&session_id.0),
            &session_id.0,
        )
    }

    pub fn load_latest_session_summary_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        session_id: &SessionId,
    ) -> Result<Option<SessionArchive>, StoreError> {
        store.load_latest_session_archive(session_id)
    }

    pub fn load_latest_lex_spans(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<IndexedSpan>, StoreError> {
        if let Some(scope) = scope {
            let scope_key = scope_storage_key(scope);
            let sidecar = self.load_latest_value::<ScopeLexSidecar>(
                store,
                BundleKind::ScopeLexSidecar,
                Some(&scope_key),
                &scope_key,
            )?;
            return Ok(sidecar.map(|value| value.spans).unwrap_or_default());
        }

        let headers = store.list_bundle_headers(BundleKind::ScopeLexSidecar, None)?;
        let mut latest = FxHashMap::<String, BundleHeader>::default();
        for header in headers {
            match latest.get(&header.key.entity_key) {
                Some(existing) if existing.key.revision >= header.key.revision => {}
                _ => {
                    latest.insert(header.key.entity_key.clone(), header);
                }
            }
        }
        let mut spans = Vec::new();
        for header in latest.into_values() {
            let Some(payload) = store.get_bundle(&header.key)? else {
                continue;
            };
            let sidecar: ScopeLexSidecar = decode_archive(&payload)?;
            spans.extend(sidecar.spans);
        }
        spans.sort_by(|left, right| left.span_id.cmp(&right.span_id));
        Ok(spans)
    }

    pub fn load_latest_lex_spans_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<IndexedSpan>, StoreError> {
        store.load_lex_spans(scope)
    }

    pub fn load_scope_lex_sidecar(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        scope: &ScopeKey,
    ) -> Result<Option<ScopeLexSidecar>, StoreError> {
        let scope_key = scope_storage_key(scope);
        self.load_latest_value(
            store,
            BundleKind::ScopeLexSidecar,
            Some(&scope_key),
            &scope_key,
        )
    }

    pub fn load_latest_document_archives(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<DocumentArchive>, StoreError> {
        let scope_key = scope.map(scope_storage_key);
        let headers =
            store.list_bundle_headers(BundleKind::DocumentArchive, scope_key.as_deref())?;
        let mut latest = FxHashMap::<String, BundleHeader>::default();
        for header in headers {
            match latest.get(&header.key.entity_key) {
                Some(existing) if existing.key.revision >= header.key.revision => {}
                _ => {
                    latest.insert(header.key.entity_key.clone(), header);
                }
            }
        }
        let mut headers = latest.into_values().collect::<Vec<_>>();
        headers.sort_by(|left, right| left.key.entity_key.cmp(&right.key.entity_key));
        let mut values = Vec::with_capacity(headers.len());
        for header in headers {
            let Some(payload) = store.get_bundle(&header.key)? else {
                continue;
            };
            values.push(decode_archive(&payload)?);
        }
        Ok(values)
    }

    pub fn load_latest_document_archives_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<DocumentArchive>, StoreError> {
        store.load_latest_document_archives(scope)
    }

    pub fn merge_session_summary(
        &self,
        existing: Option<SessionArchive>,
        session_id: SessionId,
        documents: Vec<SessionDocumentState>,
        document_refs: Vec<DocumentRevisionRef>,
        span_count: usize,
        discovery_candidate_count: usize,
        graph_vertex_count: usize,
        graph_edge_count: usize,
        updated_at: i64,
    ) -> SessionArchive {
        let mut by_document = existing
            .as_ref()
            .map(|summary| {
                summary
                    .documents
                    .iter()
                    .cloned()
                    .map(|document| (document.document_id.0.clone(), document))
                    .collect::<FxHashMap<_, _>>()
            })
            .unwrap_or_default();
        for document in documents {
            by_document.insert(document.document_id.0.clone(), document);
        }
        let mut ref_by_document = existing
            .map(|summary| {
                summary
                    .document_refs
                    .into_iter()
                    .map(|document| (document.document_id.clone(), document))
                    .collect::<FxHashMap<_, _>>()
            })
            .unwrap_or_default();
        for document_ref in document_refs {
            ref_by_document.insert(document_ref.document_id.clone(), document_ref);
        }
        let mut merged_documents = by_document.into_values().collect::<Vec<_>>();
        merged_documents.sort_by(|left, right| left.document_id.0.cmp(&right.document_id.0));
        let mut merged_document_refs = ref_by_document.into_values().collect::<Vec<_>>();
        merged_document_refs.sort_by(|left, right| left.document_id.cmp(&right.document_id));
        SessionArchive {
            session_id,
            session_ord: None,
            documents: merged_documents,
            document_refs: merged_document_refs,
            discovery_candidate_count,
            span_count,
            graph_vertex_count,
            graph_edge_count,
            graph_generation: 0,
            lex_generation: 0,
            updated_at,
            archive_version: 2,
        }
    }

    fn scan_document_bundle(
        &self,
        document: &IngestDocument,
    ) -> Result<NativeScanBundle, StoreError> {
        let progress = native_progress_enabled();

        let phase_started = Instant::now();
        let scan = scan_native_compact(
            &document.text,
            &document.scope,
            &[],
            &self.config.extraction,
        );
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=scan_native_compact document_id={} wall_ms={} sentences={} mentions={} resolver_links={} narrative_hits={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                scan.sentences.len(),
                scan.mentions.len(),
                scan.resolver_links.len(),
                scan.narrative_hits.len(),
            );
        }

        let phase_started = Instant::now();
        let boundaries = extract_boundaries(&document.text);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=extract_boundaries document_id={} wall_ms={} boundaries={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                boundaries.len(),
            );
        }

        let phase_started = Instant::now();
        let chunk_ranges = build_chunks(
            &document.text,
            &ChunkerConfig {
                chunk_size: self.config.chunk_size,
                overlap: self.config.overlap,
            },
        );
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_chunks document_id={} wall_ms={} chunk_ranges={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                chunk_ranges.len(),
            );
        }

        let phase_started = Instant::now();
        let (chunks, indexed_spans) = build_chunk_records(document, &boundaries, &chunk_ranges);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_chunk_records document_id={} wall_ms={} chunks={} indexed_spans={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                chunks.len(),
                indexed_spans.len(),
            );
        }

        let phase_started = Instant::now();
        let structure = build_native_structure_rows(&document.text, &scan, &chunks);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_native_structure_rows document_id={} wall_ms={} relation_seeds={} sentence_chunk_indexes={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                structure.relation_seeds.len(),
                structure.sentence_chunk_indexes.len(),
            );
        }

        let phase_started = Instant::now();
        let coref = build_coref_rows(&scan, &structure, &self.config.coref);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_coref_rows document_id={} wall_ms={} clusters={} attached_mentions={} candidate_links={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                coref.summary.cluster_count,
                coref.summary.attached_mention_count,
                coref.summary.candidate_link_count,
            );
        }

        Ok(NativeScanBundle {
            scan,
            boundaries,
            chunks,
            indexed_spans,
            structure,
            coref,
        })
    }

    fn resolve_document_bundle(
        &self,
        document: &IngestDocument,
        scan_bundle: &NativeScanBundle,
        entity_memory: &NativeEntityMemory,
    ) -> Result<NativeResolutionBundle, StoreError> {
        let progress = native_progress_enabled();

        let phase_started = Instant::now();
        let (
            resolutions,
            alias_confirmation_ords,
            mut er_summary,
            mut er_diagnostics,
            entity_ids,
            _entity_ord_by_id,
        ) = resolve_mentions_compact_native(
            document,
            &scan_bundle.scan,
            &scan_bundle.coref,
            &scan_bundle.chunks,
            entity_memory,
        );
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=resolve_mentions document_id={} wall_ms={} resolutions={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                resolutions.len(),
            );
        }

        let phase_started = Instant::now();
        let (entities, relations, discovery_count, mut relation_diagnostics) =
            build_semantic_records_native(
                document,
                &scan_bundle.scan,
                &scan_bundle.structure,
                &scan_bundle.chunks,
                &resolutions,
                &alias_confirmation_ords,
                &entity_ids,
            );
        er_diagnostics.append(&mut relation_diagnostics);
        er_summary.detected_mention_count = scan_bundle.scan.mentions.len();
        er_summary.detected_named_count = scan_bundle.scan.detected_named_count;
        er_summary.detected_nominal_count = scan_bundle.scan.detected_nominal_count;
        er_summary.detected_pronoun_count = scan_bundle.scan.detected_pronoun_count;
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_semantic_records document_id={} wall_ms={} entities={} relations={} discovery_count={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                entities.len(),
                relations.len(),
                discovery_count,
            );
        }

        let verification_summary = run_background_verification(
            &self.config.verification,
            document,
            &scan_bundle.scan,
            &resolutions,
            &alias_confirmation_ords,
            &entity_ids,
        );
        er_summary.verifier_task_count = verification_summary.task_count;
        er_summary.verifier_supported_alias_count = verification_summary.supported_alias_count;
        er_summary.verifier_supported_type_count = verification_summary.supported_type_count;
        er_diagnostics.extend(verification_summary.diagnostics);

        let alias_confirmations =
            materialize_alias_confirmations(document, alias_confirmation_ords, &entity_ids);
        let (coref_clusters, coref_summary) = materialize_coref_clusters(
            document,
            &scan_bundle.scan,
            &scan_bundle.coref,
            &resolutions,
            &entity_ids,
            &scan_bundle.chunks,
            self.config.coref.persist_chunk_cap,
        );

        let phase_started = Instant::now();
        let kernel_batch = build_kernel_batch(
            document,
            &scan_bundle.chunks,
            &entities,
            &relations,
            &alias_confirmations,
        );
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_kernel_batch document_id={} wall_ms={} vertices={} edges={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                kernel_batch.vertices.len(),
                kernel_batch.edges.len(),
            );
        }
        let candidate_kernel_batch =
            build_coref_candidate_batch(document, &coref_clusters, &self.config.coref);

        Ok(NativeResolutionBundle {
            alias_confirmations,
            coref_clusters,
            er_summary,
            coref_summary,
            entities,
            relations,
            discovery_count,
            diagnostics: er_diagnostics,
            kernel_batch,
            candidate_kernel_batch,
        })
    }

    fn build_document_state(
        &self,
        document: &IngestDocument,
        session_id: Option<&SessionId>,
        assignment: &DocumentOrdinalAssignment,
        created_at: i64,
        boundaries: &[BoundaryRecord],
        chunk_count: usize,
        entities: &[SemanticEntityRecord],
        kernel_batch: &KernelMutationBatch,
        discovery_count: usize,
        mention_count: usize,
    ) -> (
        IngestDocumentSummary,
        SessionDocumentState,
        DocumentManifest,
    ) {
        let document_summary = IngestDocumentSummary {
            document_id: document.document_id.clone(),
            note_id: document.note_id.clone(),
            chapter_count: boundaries
                .iter()
                .filter(|boundary| boundary.is_chapter)
                .count(),
            boundary_count: boundaries.len(),
            parent_count: boundaries.len(),
            leaf_count: chunk_count,
            entity_count: entities.len(),
            edge_count: kernel_batch.edges.len(),
            has_front_matter_chapter: boundaries
                .iter()
                .any(|boundary| boundary.is_chapter && is_front_matter_label(&boundary.label)),
            has_front_matter_boundary: boundaries
                .iter()
                .any(|boundary| is_front_matter_label(&boundary.label)),
        };
        let session_document = SessionDocumentState {
            document_id: document.document_id.clone(),
            note_id: document.note_id.clone(),
            chapter_count: document_summary.chapter_count,
            boundary_count: document_summary.boundary_count,
            chapter_titles: boundaries
                .iter()
                .filter(|boundary| boundary.is_chapter)
                .map(|boundary| boundary.label.clone())
                .collect(),
            boundary_labels: boundaries
                .iter()
                .map(|boundary| boundary.label.clone())
                .collect(),
            parent_count: document_summary.parent_count,
            leaf_count: document_summary.leaf_count,
            entity_count: document_summary.entity_count,
            discovery_count,
            has_front_matter_chapter: document_summary.has_front_matter_chapter,
            has_front_matter_boundary: document_summary.has_front_matter_boundary,
            updated_at: created_at,
        };
        let alias_count = entities.iter().map(|entity| entity.aliases.len()).sum();
        let manifest = DocumentManifest {
            document_id: document.document_id.0.clone(),
            document_version_id: DocumentVersionId(format!(
                "{}::{}",
                document.document_id.0, assignment.revision
            )),
            note_id: document.note_id.clone(),
            scope: document.scope.clone(),
            scope_key: assignment.scope_key.clone(),
            scope_ord: assignment.scope_ord,
            document_ord: assignment.document_ord,
            revision: assignment.revision,
            title: document.title.clone(),
            text_len: document.text.len(),
            fingerprint: format!("{}:{}", document.document_id.0, document.text.len()),
            config_hash: "invarant-v2::document-archive".to_owned(),
            session_id: session_id.cloned(),
            document_summary: document_summary.clone(),
            session_document: session_document.clone(),
            discovery_count,
            mention_count,
            span_count: document_summary.leaf_count,
            entity_count: entities.len(),
            alias_count,
            graph_edge_count: kernel_batch.edges.len(),
            graph_vertex_count: kernel_batch.vertices.len(),
            segment_refs: Vec::new(),
            created_at,
            archive_version: 4,
        };

        (document_summary, session_document, manifest)
    }

    fn build_prepared_document(
        &self,
        document: &IngestDocument,
        session_id: Option<&SessionId>,
        assignment: &DocumentOrdinalAssignment,
        created_at: i64,
        scan_bundle: NativeScanBundle,
        resolution_bundle: NativeResolutionBundle,
    ) -> Result<PreparedDocumentDraft, StoreError> {
        let progress = native_progress_enabled();
        let phase_started = Instant::now();
        let mention_count = scan_bundle.scan.mentions.len();
        let (document_summary, session_document, mut manifest) = self.build_document_state(
            document,
            session_id,
            assignment,
            created_at,
            &scan_bundle.boundaries,
            scan_bundle.chunks.len(),
            &resolution_bundle.entities,
            &resolution_bundle.kernel_batch,
            resolution_bundle.discovery_count,
            mention_count,
        );
        let causal_substrate = build_document_causal_substrate(document, &scan_bundle, created_at);
        let temporal_substrate =
            build_document_temporal_substrate(document, &scan_bundle, created_at);
        let event_identity_substrate = build_document_event_identity_substrate(
            document,
            manifest.revision,
            &causal_substrate,
            &temporal_substrate,
        );

        let mut segments = Vec::<PreparedDocumentSegment>::new();
        let mut segment_refs = Vec::<DocumentSegmentRef>::new();

        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::AliasConfirmationTable,
            resolution_bundle.alias_confirmations.len(),
            &resolution_bundle.alias_confirmations,
        )?;
        if !resolution_bundle.coref_clusters.is_empty() {
            self.push_segment(
                &mut segments,
                &mut segment_refs,
                DocumentSegmentKind::CorefClusterTable,
                resolution_bundle.coref_clusters.len(),
                &resolution_bundle.coref_clusters,
            )?;
        }
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::ChunkTable,
            scan_bundle.chunks.len(),
            &scan_bundle.chunks,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::EntityTable,
            resolution_bundle.entities.len(),
            &resolution_bundle.entities,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::RelationTable,
            resolution_bundle.relations.len(),
            &resolution_bundle.relations,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::CausalSubstrateTable,
            causal_substrate.propositions.len()
                + causal_substrate.semantic_events.len()
                + causal_substrate.semantic_states.len()
                + causal_substrate.semantic_claims.len()
                + causal_substrate.semantic_relations.len()
                + causal_substrate.temporal_bindings.len()
                + causal_substrate.causal_candidates.len()
                + causal_substrate.causal_links.len()
                + causal_substrate.causal_diagnostics.len(),
            &causal_substrate,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::TemporalSubstrateTable,
            temporal_substrate.propositions.len()
                + temporal_substrate.semantic_events.len()
                + temporal_substrate.semantic_states.len()
                + temporal_substrate.semantic_claims.len()
                + temporal_substrate.surface_temporal_cues.len()
                + temporal_substrate.timex_records.len()
                + temporal_substrate.anchor_candidates.len()
                + temporal_substrate.axis_records.len()
                + temporal_substrate.reference_timex_edges.len()
                + temporal_substrate.reference_event_edges.len()
                + temporal_substrate.temporal_claims.len()
                + temporal_substrate.temporal_constraints.len()
                + temporal_substrate.temporal_diagnostics.len(),
            &temporal_substrate,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::EventIdentitySubstrateTable,
            event_identity_substrate.mention_seeds.len()
                + event_identity_substrate.diagnostics.len(),
            &event_identity_substrate,
        )?;

        let lexical_started = Instant::now();
        let lexical = build_lexical_postings_segment(
            scan_bundle.indexed_spans,
            &resolution_bundle.entities,
            &manifest.document_id,
        );
        if progress {
            eprintln!(
                "[runtime-ingest] prepare_subphase=build_lexical_postings alias_entries={} spans={} wall_ms={}",
                lexical.alias_entries.len(),
                lexical.spans.len(),
                lexical_started.elapsed().as_millis(),
            );
        }
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::LexicalPostings,
            lexical.spans.len() + lexical.alias_entries.len(),
            &lexical,
        )?;

        manifest.segment_refs = segment_refs;
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_prepared_document document_id={} wall_ms={} segments={} mention_count={} alias_count={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                segments.len(),
                mention_count,
                manifest.alias_count,
            );
        }

        Ok(PreparedDocumentDraft {
            prepared: PreparedDocument {
                assignment: assignment.clone(),
                manifest,
                segments,
                kernel_batch: resolution_bundle.kernel_batch,
            },
            document_summary,
            session_document,
            span_count: scan_bundle.chunks.len(),
            discovery_count: resolution_bundle.discovery_count,
            diagnostics: resolution_bundle.diagnostics,
            candidate_kernel_batch: resolution_bundle.candidate_kernel_batch,
        })
    }

    fn push_segment<T: Serialize>(
        &self,
        segments: &mut Vec<PreparedDocumentSegment>,
        refs: &mut Vec<DocumentSegmentRef>,
        kind: DocumentSegmentKind,
        row_count: usize,
        value: &T,
    ) -> Result<(), StoreError> {
        let started = Instant::now();
        let (payload, uncompressed_len) = encode_segment_payload(value)?;
        let ordinal = segments.len() as u32;
        let header = DocumentSegmentHeader::new(
            kind,
            ordinal,
            row_count as u32,
            uncompressed_len,
            payload.len(),
        );
        refs.push(DocumentSegmentRef {
            kind,
            ordinal,
            row_count: row_count as u32,
            byte_len: payload.len() as u32,
            uncompressed_len: uncompressed_len as u32,
        });
        segments.push(PreparedDocumentSegment { header, payload });
        if native_progress_enabled() {
            eprintln!(
                "[runtime-ingest] prepare_segment kind={kind:?} ordinal={} rows={} wall_ms={} uncompressed_bytes={} compressed_bytes={}",
                ordinal,
                row_count,
                started.elapsed().as_millis(),
                uncompressed_len,
                refs.last().map(|segment_ref| segment_ref.byte_len).unwrap_or_default(),
            );
        }
        Ok(())
    }

    fn build_dirty_scope_records(
        &self,
        outcomes: &[IngestedDocumentOutcome],
        created_at: i64,
    ) -> Vec<DirtyScopeRecord> {
        let mut by_scope = FxHashMap::<String, DirtyScopeRecord>::default();
        for outcome in outcomes {
            let entry = by_scope
                .entry(outcome.assignment.scope_key.clone())
                .or_insert_with(|| DirtyScopeRecord {
                    scope: outcome.assignment.scope.clone(),
                    scope_key: outcome.assignment.scope_key.clone(),
                    scope_ord: outcome.assignment.scope_ord,
                    document_ords: Vec::new(),
                    updated_at: created_at,
                });
            entry.document_ords.push(outcome.assignment.document_ord);
            entry.updated_at = created_at;
        }
        let mut values = by_scope.into_values().collect::<Vec<_>>();
        for value in &mut values {
            value
                .document_ords
                .sort_by_key(|document_ord| document_ord.0);
            value
                .document_ords
                .dedup_by(|left, right| left.0 == right.0);
        }
        values.sort_by(|left, right| left.scope_key.cmp(&right.scope_key));
        values
    }

    fn build_document_prepared_outcome(
        &self,
        document: &IngestDocument,
        session_id: Option<&SessionId>,
        assignment: &DocumentOrdinalAssignment,
        entity_memory: &NativeEntityMemory,
        created_at: i64,
    ) -> Result<PreparedDocumentDraft, StoreError> {
        let scan_bundle = self.scan_document_bundle(document)?;
        let resolution_bundle =
            self.resolve_document_bundle(document, &scan_bundle, entity_memory)?;
        self.build_prepared_document(
            document,
            session_id,
            assignment,
            created_at,
            scan_bundle,
            resolution_bundle,
        )
    }

    fn build_document_outcome(
        &self,
        document: &IngestDocument,
        session_id: Option<&SessionId>,
        assignment: &DocumentOrdinalAssignment,
        entity_memory: &NativeEntityMemory,
        created_at: i64,
    ) -> Result<DocumentOutcome, StoreError> {
        let progress = native_progress_enabled();
        let document_started = Instant::now();
        let scan_bundle = self.scan_document_bundle(document)?;
        let resolution_bundle =
            self.resolve_document_bundle(document, &scan_bundle, entity_memory)?;
        let mention_count = scan_bundle.scan.mentions.len();
        let (document_summary, session_document, manifest) = self.build_document_state(
            document,
            session_id,
            assignment,
            created_at,
            &scan_bundle.boundaries,
            scan_bundle.chunks.len(),
            &resolution_bundle.entities,
            &resolution_bundle.kernel_batch,
            resolution_bundle.discovery_count,
            mention_count,
        );
        let causal_substrate = build_document_causal_substrate(document, &scan_bundle, created_at);
        let temporal_substrate =
            build_document_temporal_substrate(document, &scan_bundle, created_at);
        let event_identity_substrate = build_document_event_identity_substrate(
            document,
            manifest.revision,
            &causal_substrate,
            &temporal_substrate,
        );
        let relation_candidates = build_native_relation_candidates(document, &scan_bundle);
        let phase_started = Instant::now();
        let archive = DocumentArchive {
            manifest,
            tokens: Vec::new(),
            sentences: Vec::new(),
            mentions: Vec::new(),
            resolver_links: Vec::new(),
            resolved_mentions: Vec::new(),
            alias_confirmations: resolution_bundle.alias_confirmations,
            coref_clusters: resolution_bundle.coref_clusters,
            er_summary: resolution_bundle.er_summary,
            coref_summary: resolution_bundle.coref_summary,
            chunks: scan_bundle.chunks,
            indexed_spans: scan_bundle.indexed_spans,
            entities: resolution_bundle.entities,
            relations: resolution_bundle.relations,
            evidence_spans: Vec::new(),
            relation_candidates,
            graph_batch: KernelMutationBatch::default(),
            structure: None,
            causal_substrate: Some(causal_substrate),
            temporal_substrate: Some(temporal_substrate),
            event_identity_substrate: Some(event_identity_substrate),
        };
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=assemble_archive document_id={} wall_ms={} mention_count={} alias_count={} total_wall_ms={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                mention_count,
                archive.manifest.alias_count,
                document_started.elapsed().as_millis(),
            );
        }
        Ok(DocumentOutcome {
            archive,
            kernel_batch: resolution_bundle.kernel_batch,
            candidate_kernel_batch: resolution_bundle.candidate_kernel_batch,
            document_summary: document_summary.clone(),
            session_document,
            scope: document.scope.clone(),
            span_count: document_summary.leaf_count,
            discovery_count: resolution_bundle.discovery_count,
            diagnostics: resolution_bundle.diagnostics,
        })
    }
}

fn build_document_causal_substrate(
    document: &IngestDocument,
    scan_bundle: &NativeScanBundle,
    created_at: i64,
) -> DocumentCausalSubstrate {
    let chunk_spans = scan_bundle
        .chunks
        .iter()
        .map(|chunk| ChunkSpan {
            kind: None,
            range: chunk.range,
            head: chunk.range,
            modifiers: Vec::new(),
            sentence_index: scan_bundle
                .scan
                .sentences
                .iter()
                .position(|sentence| {
                    sentence.range.start <= chunk.range.start
                        && sentence.range.end >= chunk.range.end
                })
                .unwrap_or_default(),
        })
        .collect::<Vec<_>>();
    let artifacts = SurfaceCompileArtifacts {
        scan: ScanArtifact {
            sentences: scan_bundle.scan.sentences.clone(),
            tokens: Vec::new(),
            mentions: scan_bundle.scan.mentions.clone(),
            chunks: chunk_spans,
            resolver_links: scan_bundle.scan.resolver_links.clone(),
            narrative_hits: scan_bundle.scan.narrative_hits.clone(),
            diagnostics: vec![Diagnostic {
                code: "PX_CAUSAL_SUBSTRATE_SCAN".to_owned(),
                message: "Rebuilt compact scan artifact for causal substrate compilation."
                    .to_owned(),
            }],
        },
        structure: build_causal_structure_artifact(document, scan_bundle),
        surface: phoenix_types::SurfaceDocument::default(),
    };
    let propositions = PropositionLowerer::lower(&artifacts);
    let semantics = SemanticLowerer::lower(&propositions);
    let temporal_bindings = propositions
        .iter()
        .map(|proposition| {
            TimeKernel::bind_label(proposition.predicate.predicate.as_str(), Some(created_at))
        })
        .collect::<Vec<_>>();
    let causality = CausalityLowerer::lower(CausalityRequest {
        text: &document.text,
        artifacts: &artifacts,
        propositions: &propositions,
        semantics: &semantics,
        temporal_bindings: &temporal_bindings,
    });
    DocumentCausalSubstrate {
        propositions,
        semantic_events: semantics.events,
        semantic_states: semantics.states,
        semantic_claims: semantics.claims,
        semantic_relations: semantics.relations,
        temporal_bindings: temporal_bindings
            .into_iter()
            .map(|binding| RecordedTemporalBinding {
                anchor: binding.anchor,
                recorded_window: binding.recorded_window,
            })
            .collect(),
        causal_candidates: causality.candidates,
        causal_links: causality.links,
        causal_diagnostics: causality.diagnostics,
    }
}

fn build_native_relation_candidates(
    document: &IngestDocument,
    scan_bundle: &NativeScanBundle,
) -> Vec<RelationCandidate> {
    let mut relations = Vec::with_capacity(scan_bundle.structure.relation_seeds.len());
    for seed in &scan_bundle.structure.relation_seeds {
        let Some(hit) = scan_bundle.scan.narrative_hits.iter().find(|hit| {
            hit.sentence_index == seed.sentence_index && hit.relation_type == seed.relation_type
        }) else {
            continue;
        };
        let Some(sentence) = scan_bundle.scan.sentences.get(seed.sentence_index) else {
            continue;
        };
        let evidence = vec![EvidenceSpan {
            document_id: Some(DocumentId(document.document_id.0.clone())),
            note_id: document.note_id.clone(),
            label: document
                .text
                .get(sentence.range.start as usize..sentence.range.end as usize)
                .unwrap_or_default()
                .trim()
                .to_owned(),
            kind: Some("sentence".to_owned()),
            range: sentence.range,
        }];
        relations.push(RelationCandidate {
            sentence_index: seed.sentence_index,
            verb_range: hit.range,
            lemma: hit.lemma.clone(),
            event_class: hit.event_class.clone(),
            relation_type: hit.relation_type.clone(),
            subject: seed
                .subject_mention_ix
                .and_then(|index| scan_bundle.scan.mentions.get(index))
                .map(frame_slot_from_native_mention),
            object: seed
                .object_mention_ix
                .and_then(|index| scan_bundle.scan.mentions.get(index))
                .map(frame_slot_from_native_mention),
            recipient: None,
            attachments: Vec::new(),
            evidence,
        });
    }
    relations
}

fn build_document_temporal_substrate(
    document: &IngestDocument,
    scan_bundle: &NativeScanBundle,
    created_at: i64,
) -> DocumentTemporalSubstrate {
    let chunk_spans = scan_bundle
        .chunks
        .iter()
        .map(|chunk| ChunkSpan {
            kind: None,
            range: chunk.range,
            head: chunk.range,
            modifiers: Vec::new(),
            sentence_index: scan_bundle
                .scan
                .sentences
                .iter()
                .position(|sentence| {
                    sentence.range.start <= chunk.range.start
                        && sentence.range.end >= chunk.range.end
                })
                .unwrap_or_default(),
        })
        .collect::<Vec<_>>();
    let artifacts = SurfaceCompileArtifacts {
        scan: ScanArtifact {
            sentences: scan_bundle.scan.sentences.clone(),
            tokens: Vec::new(),
            mentions: scan_bundle.scan.mentions.clone(),
            chunks: chunk_spans,
            resolver_links: scan_bundle.scan.resolver_links.clone(),
            narrative_hits: scan_bundle.scan.narrative_hits.clone(),
            diagnostics: vec![Diagnostic {
                code: "PX_TEMPORAL_SUBSTRATE_SCAN".to_owned(),
                message: "Rebuilt compact scan artifact for temporal substrate compilation."
                    .to_owned(),
            }],
        },
        structure: build_causal_structure_artifact(document, scan_bundle),
        surface: phoenix_types::SurfaceDocument::default(),
    };
    let propositions = PropositionLowerer::lower(&artifacts);
    let semantics = SemanticLowerer::lower(&propositions);
    let document_id = document.document_id.0.clone();
    let dct_axis_id = TemporalAxisId("axis:world".to_owned());
    let mut axis_records = vec![TemporalAxisRecord {
        axis_id: dct_axis_id.clone(),
        document_id: document_id.clone(),
        kind: TemporalAxisKind::World,
        label: "world".to_owned(),
        evidence_refs: vec!["document_created_at".to_owned()],
    }];
    let mut axis_by_kind = FxHashMap::<TemporalAxisKind, TemporalAxisId>::default();
    axis_by_kind.insert(TemporalAxisKind::World, dct_axis_id.clone());

    let dct_temporal = temporal_window(Some(created_at), Some(created_at), Some(created_at));
    let dct_timex_id = TemporalTimexId(format!("timex:{document_id}:dct"));
    let mut timex_records = vec![TemporalTimexRecord {
        timex_id: dct_timex_id.clone(),
        document_id: document_id.clone(),
        proposition_id: None,
        sentence_index: 0,
        label: "document_created_at".to_owned(),
        normalized_value: Some(created_at.to_string()),
        range: None,
        axis_id: dct_axis_id.clone(),
        temporal: dct_temporal.clone(),
        confidence_millis: 1000,
        source_class: "document_created_at".to_owned(),
        evidence_refs: vec!["manifest.created_at".to_owned()],
    }];
    let node_id_by_proposition = semantics
        .events
        .iter()
        .filter_map(|event| {
            event
                .event_id
                .as_ref()
                .map(|event_id| (event.proposition_id.to_string(), event_id.0.clone()))
        })
        .chain(semantics.states.iter().filter_map(|state| {
            state
                .state_id
                .as_ref()
                .map(|state_id| (state.proposition_id.to_string(), state_id.0.clone()))
        }))
        .chain(semantics.claims.iter().filter_map(|claim| {
            claim
                .claim_id
                .as_ref()
                .map(|claim_id| (claim.proposition_id.to_string(), claim_id.0.clone()))
        }))
        .collect::<FxHashMap<_, _>>();

    let mut propositions_sorted = propositions.iter().collect::<Vec<_>>();
    propositions_sorted.sort_by(|left, right| {
        left.sentence_index
            .cmp(&right.sentence_index)
            .then_with(|| {
                left.proposition_id
                    .as_str()
                    .cmp(right.proposition_id.as_str())
            })
    });

    let mut surface_temporal_cues = Vec::<SurfaceTemporalCueRecord>::new();
    let mut anchor_candidates = Vec::<TemporalAnchorRecord>::new();
    let mut reference_timex_edges = Vec::<TemporalReferenceEdge>::new();
    let mut reference_event_edges = Vec::<TemporalReferenceEdge>::new();
    let mut temporal_claims = Vec::<TemporalClaimAtom>::new();
    let mut temporal_constraints = Vec::<TemporalConstraintRecord>::new();
    let mut temporal_diagnostics = Vec::<TemporalDiagnosticRecord>::new();
    let mut last_event_by_axis = FxHashMap::<String, (String, usize)>::default();

    for proposition in propositions_sorted {
        let axis_kind = proposition_axis_kind(proposition);
        let axis_id = ensure_temporal_axis(
            &mut axis_records,
            &mut axis_by_kind,
            &document_id,
            axis_kind,
        );
        let snippet = proposition_snippet(document, proposition);
        let snippet_lower = snippet.to_ascii_lowercase();
        let snippet_range = proposition_text_range(proposition);

        let mut cue_specs = temporal_cue_specs(proposition, &snippet_lower);
        if cue_specs.is_empty() {
            cue_specs.push((
                "predicate".to_owned(),
                proposition.predicate.predicate.to_string(),
            ));
        }
        for (index, (cue_kind, label)) in cue_specs.into_iter().enumerate() {
            surface_temporal_cues.push(SurfaceTemporalCueRecord {
                cue_id: format!("cue:{}:{}:{index}", document_id, proposition.proposition_id),
                proposition_id: Some(proposition.proposition_id.to_string()),
                sentence_index: proposition.sentence_index,
                cue_kind,
                label,
                range: snippet_range,
            });
        }

        let explicit_timex = detect_explicit_timex(
            &document_id,
            proposition,
            &snippet_lower,
            snippet_range,
            &axis_id,
            created_at,
        );
        let mut timex_id = dct_timex_id.clone();
        let mut anchor_kind = "document_created_at".to_owned();
        let mut anchor_temporal = dct_temporal.clone();
        let mut anchor_evidence = vec!["manifest.created_at".to_owned()];
        let mut anchor_source = "document_created_at".to_owned();
        if let Some(record) = explicit_timex {
            timex_id = record.timex_id.clone();
            anchor_kind = "explicit_timex".to_owned();
            anchor_temporal = record.temporal.clone();
            anchor_evidence = vec![record.label.clone()];
            anchor_source = record.source_class.clone();
            timex_records.push(record);
        }

        if let Some(event_id) = node_id_by_proposition.get(proposition.proposition_id.as_str()) {
            let event_fingerprint = temporal_event_fingerprint(&document_id, proposition, event_id);
            let anchor_id = TemporalAnchorId(format!("anchor:{event_fingerprint}"));
            anchor_candidates.push(TemporalAnchorRecord {
                anchor_id: anchor_id.clone(),
                document_id: document_id.clone(),
                proposition_id: Some(proposition.proposition_id.to_string()),
                event_id: Some(event_id.clone()),
                canonical_event_id: None,
                timex_id: Some(timex_id.clone()),
                reference_event_id: None,
                canonical_reference_event_id: None,
                axis_id: axis_id.clone(),
                label: anchor_kind.clone(),
                anchor_kind: anchor_kind.clone(),
                temporal: anchor_temporal.clone(),
                confidence_millis: if anchor_kind == "explicit_timex" {
                    900
                } else {
                    520
                },
                source_class: anchor_source.clone(),
                evidence_refs: anchor_evidence.clone(),
            });
            reference_timex_edges.push(TemporalReferenceEdge {
                edge_id: format!("ref-timex:{event_fingerprint}"),
                document_id: document_id.clone(),
                axis_id: axis_id.clone(),
                source_event_id: event_id.clone(),
                canonical_source_event_id: None,
                target_event_id: None,
                canonical_target_event_id: None,
                target_timex_id: Some(timex_id.clone()),
                relation: "anchors_to".to_owned(),
                confidence_millis: if anchor_kind == "explicit_timex" {
                    900
                } else {
                    520
                },
                evidence_refs: anchor_evidence.clone(),
            });
            temporal_claims.push(TemporalClaimAtom {
                claim_id: format!("tclaim:anchor:{event_fingerprint}"),
                document_id: document_id.clone(),
                proposition_id: Some(proposition.proposition_id.to_string()),
                event_id: Some(event_id.clone()),
                canonical_event_id: None,
                axis_id: axis_id.clone(),
                source_kind: anchor_source.clone(),
                label: format!("{event_id} anchored to {}", timex_id.0),
                confidence_millis: if anchor_kind == "explicit_timex" {
                    900
                } else {
                    520
                },
                temporal: anchor_temporal.clone(),
                evidence_refs: anchor_evidence.clone(),
            });
            temporal_constraints.push(TemporalConstraintRecord {
                constraint_id: TemporalConstraintId(format!(
                    "tconstraint:anchor:{event_fingerprint}"
                )),
                document_id: document_id.clone(),
                axis_id: axis_id.clone(),
                source_event_id: Some(event_id.clone()),
                canonical_source_event_id: None,
                target_event_id: None,
                canonical_target_event_id: None,
                target_timex_id: Some(timex_id.clone()),
                kind: TemporalConstraintKind::AnchoredAt,
                confidence_millis: if anchor_kind == "explicit_timex" {
                    900
                } else {
                    520
                },
                hard: anchor_kind == "explicit_timex",
                temporal: anchor_temporal.clone(),
                evidence_refs: anchor_evidence.clone(),
            });

            if let Some((previous_event_id, previous_sentence_index)) =
                last_event_by_axis.get(&axis_id.0).cloned()
            {
                let relation_label = if snippet_lower.contains("before") {
                    "before_previous"
                } else if snippet_lower.contains("after") || snippet_lower.contains("later") {
                    "after_previous"
                } else {
                    "narrative_sequence"
                };
                reference_event_edges.push(TemporalReferenceEdge {
                    edge_id: format!("ref-event:{previous_event_id}:{event_id}"),
                    document_id: document_id.clone(),
                    axis_id: axis_id.clone(),
                    source_event_id: previous_event_id.clone(),
                    canonical_source_event_id: None,
                    target_event_id: Some(event_id.clone()),
                    canonical_target_event_id: None,
                    target_timex_id: None,
                    relation: relation_label.to_owned(),
                    confidence_millis: 640,
                    evidence_refs: vec![relation_label.to_owned()],
                });
                temporal_claims.push(TemporalClaimAtom {
                    claim_id: format!("tclaim:order:{previous_event_id}:{event_id}"),
                    document_id: document_id.clone(),
                    proposition_id: Some(proposition.proposition_id.to_string()),
                    event_id: Some(event_id.clone()),
                    canonical_event_id: None,
                    axis_id: axis_id.clone(),
                    source_kind: "narrative_sequence".to_owned(),
                    label: format!(
                        "{previous_event_id} precedes {event_id} across sentences {previous_sentence_index}->{}",
                        proposition.sentence_index
                    ),
                    confidence_millis: 640,
                    temporal: temporal_window(None, Some(created_at), None),
                    evidence_refs: vec![relation_label.to_owned()],
                });
                temporal_constraints.push(TemporalConstraintRecord {
                    constraint_id: TemporalConstraintId(format!(
                        "tconstraint:order:{previous_event_id}:{event_id}"
                    )),
                    document_id: document_id.clone(),
                    axis_id: axis_id.clone(),
                    source_event_id: Some(previous_event_id.clone()),
                    canonical_source_event_id: None,
                    target_event_id: Some(event_id.clone()),
                    canonical_target_event_id: None,
                    target_timex_id: None,
                    kind: TemporalConstraintKind::EndBeforeStart,
                    confidence_millis: 640,
                    hard: false,
                    temporal: temporal_window(None, Some(created_at), None),
                    evidence_refs: vec![relation_label.to_owned()],
                });
            }

            last_event_by_axis.insert(
                axis_id.0.clone(),
                (event_id.clone(), proposition.sentence_index),
            );
        } else {
            temporal_diagnostics.push(TemporalDiagnosticRecord {
                code: "temporal_missing_event".to_owned(),
                message: format!(
                    "no semantic event id found for proposition {}",
                    proposition.proposition_id
                ),
            });
        }
    }

    if timex_records.len() == 1 {
        temporal_diagnostics.push(TemporalDiagnosticRecord {
            code: "temporal_only_dct_anchor".to_owned(),
            message: "Only document created time was available as a normalized timex.".to_owned(),
        });
    }

    DocumentTemporalSubstrate {
        propositions,
        semantic_events: semantics.events,
        semantic_states: semantics.states,
        semantic_claims: semantics.claims,
        surface_temporal_cues,
        timex_records,
        anchor_candidates,
        axis_records,
        reference_timex_edges,
        reference_event_edges,
        temporal_claims,
        temporal_constraints,
        temporal_diagnostics,
    }
}

fn build_document_event_identity_substrate(
    document: &IngestDocument,
    revision: u64,
    causal_substrate: &DocumentCausalSubstrate,
    temporal_substrate: &DocumentTemporalSubstrate,
) -> DocumentEventIdentitySubstrate {
    let proposition_by_id = temporal_substrate
        .propositions
        .iter()
        .map(|proposition| (proposition.proposition_id.to_string(), proposition))
        .collect::<FxHashMap<_, _>>();
    let anchors_by_event = temporal_substrate
        .anchor_candidates
        .iter()
        .filter_map(|anchor| {
            anchor
                .event_id
                .as_ref()
                .map(|event_id| (event_id.clone(), anchor.clone()))
        })
        .fold(
            FxHashMap::<String, Vec<TemporalAnchorRecord>>::default(),
            |mut rows, (event_id, anchor)| {
                rows.entry(event_id).or_default().push(anchor);
                rows
            },
        );
    let temporal_neighbors = temporal_substrate.reference_event_edges.iter().fold(
        FxHashMap::<String, Vec<String>>::default(),
        |mut rows, edge| {
            rows.entry(edge.source_event_id.clone())
                .or_default()
                .extend(edge.target_event_id.clone());
            if let Some(target) = edge.target_event_id.as_ref() {
                rows.entry(target.clone())
                    .or_default()
                    .push(edge.source_event_id.clone());
            }
            rows
        },
    );
    let mut causal_neighbors = FxHashMap::<String, Vec<String>>::default();
    for edge in &causal_substrate.causal_links {
        let Some(source_id) = semantic_node_id(&edge.source) else {
            continue;
        };
        let Some(target_id) = semantic_node_id(&edge.target) else {
            continue;
        };
        causal_neighbors
            .entry(source_id.clone())
            .or_default()
            .push(target_id.clone());
        causal_neighbors
            .entry(target_id)
            .or_default()
            .push(source_id);
    }
    for edge in &causal_substrate.causal_candidates {
        let Some(source_id) = semantic_node_id(&edge.source) else {
            continue;
        };
        let Some(target_id) = semantic_node_id(&edge.target) else {
            continue;
        };
        causal_neighbors
            .entry(source_id.clone())
            .or_default()
            .push(target_id.clone());
        causal_neighbors
            .entry(target_id)
            .or_default()
            .push(source_id);
    }
    let mut mention_seeds = Vec::<EventMentionPacketSeed>::new();
    let mut diagnostics = Vec::<EventIdentityDiagnosticRecord>::new();

    for (event_id, proposition_id, label, event_type) in temporal_substrate
        .semantic_events
        .iter()
        .filter_map(|event| {
            event.event_id.as_ref().map(|event_id| {
                (
                    event_id.0.clone(),
                    event.proposition_id.to_string(),
                    event.label.to_string(),
                    "event".to_owned(),
                )
            })
        })
        .chain(
            temporal_substrate
                .semantic_states
                .iter()
                .filter_map(|state| {
                    state.state_id.as_ref().map(|state_id| {
                        (
                            state_id.0.clone(),
                            state.proposition_id.to_string(),
                            state.label.to_string(),
                            "state".to_owned(),
                        )
                    })
                }),
        )
    {
        let Some(proposition) = proposition_by_id.get(proposition_id.as_str()) else {
            diagnostics.push(EventIdentityDiagnosticRecord {
                code: "event_identity_missing_proposition".to_owned(),
                message: format!("missing proposition for event node {event_id}"),
            });
            continue;
        };

        let participant_slots = proposition
            .arguments
            .iter()
            .map(|argument| EventParticipantSlot {
                role: argument.role.to_string(),
                entity_id: argument.entity_id.clone(),
                mention_index: argument.mention_index,
                label: argument
                    .range
                    .map(source_range_to_text_range)
                    .map(|range| safe_text_slice(&document.text, range).trim().to_owned())
                    .filter(|label| !label.is_empty()),
                range: argument.range.map(|range| TextRange {
                    start: range.start,
                    end: range.end,
                }),
            })
            .collect::<Vec<_>>();
        let place_labels = proposition
            .arguments
            .iter()
            .filter(|argument| is_place_role(argument.role.as_str()))
            .filter_map(|argument| {
                argument
                    .range
                    .map(source_range_to_text_range)
                    .map(|range| safe_text_slice(&document.text, range).trim().to_owned())
                    .filter(|label| !label.is_empty())
            })
            .collect::<Vec<_>>();
        let source_semantics = proposition_source_semantics(proposition);
        let modality_semantics = proposition_modality_semantics(proposition);
        let event_fingerprint = event_identity_fingerprint(
            proposition,
            event_type.as_str(),
            &participant_slots,
            &place_labels,
        );
        let time_anchor_rows = anchors_by_event.get(&event_id).cloned().unwrap_or_default();
        mention_seeds.push(EventMentionPacketSeed {
            mention_id: EventMentionId(format!(
                "event-mention:{}:{}:{}:{}",
                scope_storage_key(&document.scope),
                document.document_id.0,
                revision,
                proposition_id
            )),
            event_id: event_id.clone(),
            document_id: document.document_id.0.clone(),
            proposition_id: proposition_id.clone(),
            revision,
            label,
            normalized_predicate: proposition
                .predicate
                .predicate
                .to_string()
                .to_ascii_lowercase(),
            event_type,
            participant_slots,
            place_labels,
            explicit_timex_ids: time_anchor_rows
                .iter()
                .filter_map(|anchor| anchor.timex_id.clone())
                .collect(),
            time_anchor_ids: time_anchor_rows
                .iter()
                .map(|anchor| anchor.anchor_id.clone())
                .collect(),
            causal_neighbor_event_ids: dedupe_strings(
                causal_neighbors.get(&event_id).cloned().unwrap_or_default(),
            ),
            temporal_neighbor_event_ids: dedupe_strings(
                temporal_neighbors
                    .get(&event_id)
                    .cloned()
                    .unwrap_or_default(),
            ),
            sentence_index: proposition.sentence_index,
            clause_range: proposition.clause_range.map(|range| TextRange {
                start: range.start,
                end: range.end,
            }),
            polarity_negative: proposition_negative(proposition),
            source_semantics,
            modality_semantics,
            realis: proposition_realis_label(proposition, source_semantics, modality_semantics),
            event_fingerprint,
            evidence_refs: proposition
                .evidence
                .iter()
                .map(|reference| reference.label.to_string())
                .collect(),
        });
    }

    mention_seeds.sort_by(|left, right| {
        (
            left.document_id.as_str(),
            left.revision,
            left.sentence_index,
            left.mention_id.0.as_str(),
        )
            .cmp(&(
                right.document_id.as_str(),
                right.revision,
                right.sentence_index,
                right.mention_id.0.as_str(),
            ))
    });

    DocumentEventIdentitySubstrate {
        mention_seeds,
        diagnostics,
    }
}

fn semantic_node_id(node: &SemanticNodeRef) -> Option<String> {
    match node {
        SemanticNodeRef::Event(event_id) => Some(event_id.0.clone()),
        SemanticNodeRef::State(state_id) => Some(state_id.0.clone()),
        SemanticNodeRef::Claim(claim_id) => Some(claim_id.0.clone()),
    }
}

fn proposition_source_semantics(proposition: &phoenix_types::Proposition) -> EventSourceSemantics {
    if proposition.quote.is_some() {
        EventSourceSemantics::ReportedSpeech
    } else if proposition.attribution.is_some() {
        EventSourceSemantics::AttributedClaim
    } else {
        EventSourceSemantics::WorldAssertion
    }
}

fn proposition_modality_semantics(
    proposition: &phoenix_types::Proposition,
) -> EventModalitySemantics {
    if proposition_negative(proposition) {
        EventModalitySemantics::Negated
    } else if proposition.conditional.is_some() {
        EventModalitySemantics::Conditional
    } else if proposition
        .scope_ops
        .iter()
        .any(|operation| operation.kind.eq_ignore_ascii_case("planned"))
    {
        EventModalitySemantics::Planned
    } else if proposition
        .scope_ops
        .iter()
        .any(|operation| operation.kind.eq_ignore_ascii_case("hypothetical"))
    {
        EventModalitySemantics::Hypothetical
    } else {
        EventModalitySemantics::Asserted
    }
}

fn proposition_negative(proposition: &phoenix_types::Proposition) -> bool {
    proposition.scope_ops.iter().any(|operation| {
        matches!(
            operation.kind.to_ascii_lowercase().as_str(),
            "negated" | "negative" | "not" | "never"
        )
    })
}

fn proposition_realis_label(
    proposition: &phoenix_types::Proposition,
    source: EventSourceSemantics,
    modality: EventModalitySemantics,
) -> String {
    match modality {
        EventModalitySemantics::Conditional => "conditional".to_owned(),
        EventModalitySemantics::Planned => "planned".to_owned(),
        EventModalitySemantics::Hypothetical => "hypothetical".to_owned(),
        EventModalitySemantics::Negated => "negated".to_owned(),
        EventModalitySemantics::Asserted => match source {
            EventSourceSemantics::ReportedSpeech => "reported".to_owned(),
            EventSourceSemantics::AttributedClaim => "attributed".to_owned(),
            EventSourceSemantics::WorldAssertion => {
                if proposition.quote.is_some() {
                    "reported".to_owned()
                } else {
                    "asserted".to_owned()
                }
            }
        },
    }
}

fn is_place_role(role: &str) -> bool {
    let normalized = role.to_ascii_lowercase();
    matches!(
        normalized.as_str(),
        "location" | "place" | "destination" | "origin" | "site" | "where"
    )
}

fn event_identity_fingerprint(
    proposition: &phoenix_types::Proposition,
    event_type: &str,
    participant_slots: &[EventParticipantSlot],
    place_labels: &[String],
) -> String {
    let participant_signature = participant_slots
        .iter()
        .map(|slot| {
            format!(
                "{}:{}:{}",
                slot.role,
                slot.entity_id
                    .as_ref()
                    .map(|entity_id| entity_id.0.as_str())
                    .unwrap_or(""),
                slot.label.as_deref().unwrap_or("")
            )
        })
        .collect::<Vec<_>>()
        .join("|");
    let place_signature = place_labels.join("|");
    format!(
        "{}:{}:{}:{}",
        event_type,
        proposition.predicate.predicate.to_ascii_lowercase(),
        participant_signature,
        place_signature
    )
}

fn dedupe_strings(values: Vec<String>) -> Vec<String> {
    let mut seen = FxHashSet::default();
    let mut deduped = Vec::new();
    for value in values {
        if seen.insert(value.clone()) {
            deduped.push(value);
        }
    }
    deduped
}

fn ensure_temporal_axis(
    axis_records: &mut Vec<TemporalAxisRecord>,
    axis_by_kind: &mut FxHashMap<TemporalAxisKind, TemporalAxisId>,
    document_id: &str,
    kind: TemporalAxisKind,
) -> TemporalAxisId {
    if let Some(existing) = axis_by_kind.get(&kind) {
        return existing.clone();
    }
    let label = match kind {
        TemporalAxisKind::World => "world",
        TemporalAxisKind::Reported => "reported",
        TemporalAxisKind::Conditional => "conditional",
        TemporalAxisKind::Hypothetical => "hypothetical",
        TemporalAxisKind::Planned => "planned",
    };
    let axis_id = TemporalAxisId(format!("axis:{label}"));
    axis_records.push(TemporalAxisRecord {
        axis_id: axis_id.clone(),
        document_id: document_id.to_owned(),
        kind,
        label: label.to_owned(),
        evidence_refs: vec![label.to_owned()],
    });
    axis_by_kind.insert(kind, axis_id.clone());
    axis_id
}

fn proposition_axis_kind(proposition: &phoenix_types::Proposition) -> TemporalAxisKind {
    if proposition.conditional.is_some() {
        return TemporalAxisKind::Conditional;
    }
    if proposition.quote.is_some() || proposition.attribution.is_some() {
        return TemporalAxisKind::Reported;
    }
    for op in &proposition.scope_ops {
        let modality = op
            .modality
            .as_ref()
            .map(|value| value.as_str().to_ascii_lowercase())
            .unwrap_or_default();
        if modality.contains("plan") || modality.contains("future") || modality.contains("intend") {
            return TemporalAxisKind::Planned;
        }
        if modality.contains("hyp") || modality.contains("possible") || modality.contains("maybe") {
            return TemporalAxisKind::Hypothetical;
        }
    }
    TemporalAxisKind::World
}

fn proposition_text_range(proposition: &phoenix_types::Proposition) -> Option<TextRange> {
    proposition
        .clause_range
        .map(source_range_to_text_range)
        .or_else(|| {
            Some(source_range_to_text_range(
                proposition.predicate.trigger_range,
            ))
        })
}

fn source_range_to_text_range(range: phoenix_types::SourceRange) -> TextRange {
    TextRange {
        start: range.start,
        end: range.end,
    }
}

fn proposition_snippet(
    document: &IngestDocument,
    proposition: &phoenix_types::Proposition,
) -> String {
    let range = proposition
        .clause_range
        .unwrap_or(proposition.predicate.trigger_range);
    let bytes = document.text.as_bytes();
    let start = usize::min(range.start as usize, bytes.len());
    let end = usize::min(range.end as usize, bytes.len());
    if start >= end {
        return proposition.predicate.predicate.to_string();
    }
    String::from_utf8_lossy(&bytes[start..end]).into_owned()
}

fn temporal_cue_specs(
    proposition: &phoenix_types::Proposition,
    snippet_lower: &str,
) -> Vec<(String, String)> {
    let mut cues = Vec::new();
    if proposition.quote.is_some() {
        cues.push(("quote".to_owned(), "quoted_context".to_owned()));
    }
    if proposition.attribution.is_some() {
        cues.push(("attribution".to_owned(), "attributed_context".to_owned()));
    }
    if proposition.conditional.is_some() {
        cues.push(("conditional".to_owned(), "conditional_context".to_owned()));
    }
    for token in [
        "before",
        "after",
        "during",
        "while",
        "then",
        "later",
        "earlier",
        "today",
        "yesterday",
        "tomorrow",
    ] {
        if snippet_lower.contains(token) {
            cues.push(("temporal_lexeme".to_owned(), token.to_owned()));
        }
    }
    cues
}

fn detect_explicit_timex(
    document_id: &str,
    proposition: &phoenix_types::Proposition,
    snippet_lower: &str,
    range: Option<TextRange>,
    axis_id: &TemporalAxisId,
    created_at: i64,
) -> Option<TemporalTimexRecord> {
    const DAY_MS: i64 = 86_400_000;
    let (label, normalized_value, valid_from, source_class) = if snippet_lower.contains("yesterday")
    {
        (
            "yesterday".to_owned(),
            Some("yesterday".to_owned()),
            Some(created_at - DAY_MS),
            "deictic_yesterday".to_owned(),
        )
    } else if snippet_lower.contains("tomorrow") {
        (
            "tomorrow".to_owned(),
            Some("tomorrow".to_owned()),
            Some(created_at + DAY_MS),
            "deictic_tomorrow".to_owned(),
        )
    } else if snippet_lower.contains("today")
        || snippet_lower.contains("now")
        || snippet_lower.contains("currently")
    {
        (
            "today".to_owned(),
            Some("today".to_owned()),
            Some(created_at),
            "deictic_today".to_owned(),
        )
    } else if let Some(year) = snippet_lower
        .split(|ch: char| !ch.is_ascii_alphanumeric())
        .find(|token| token.len() == 4 && token.chars().all(|ch| ch.is_ascii_digit()))
    {
        (
            year.to_owned(),
            Some(year.to_owned()),
            None,
            "year_literal".to_owned(),
        )
    } else {
        return None;
    };

    Some(TemporalTimexRecord {
        timex_id: TemporalTimexId(format!(
            "timex:{}:{}:{}",
            document_id, proposition.proposition_id, label
        )),
        document_id: document_id.to_owned(),
        proposition_id: Some(proposition.proposition_id.to_string()),
        sentence_index: proposition.sentence_index,
        label,
        normalized_value,
        range,
        axis_id: axis_id.clone(),
        temporal: temporal_window(valid_from, Some(created_at), None),
        confidence_millis: if valid_from.is_some() { 880 } else { 620 },
        source_class,
        evidence_refs: vec![proposition.proposition_id.to_string()],
    })
}

fn temporal_event_fingerprint(
    document_id: &str,
    proposition: &phoenix_types::Proposition,
    event_id: &str,
) -> String {
    format!(
        "{document_id}:{}:{}:{}",
        proposition.proposition_id, proposition.sentence_index, event_id
    )
}

fn temporal_window(
    valid_from: Option<i64>,
    recorded_from: Option<i64>,
    valid_to: Option<i64>,
) -> BiTemporalWindow {
    BiTemporalWindow {
        valid_from,
        valid_to,
        recorded_from,
        recorded_to: None,
    }
}

fn build_causal_structure_artifact(
    document: &IngestDocument,
    scan_bundle: &NativeScanBundle,
) -> StructureArtifact {
    let mut sentence_frames = scan_bundle
        .scan
        .sentences
        .iter()
        .map(|sentence| SentenceFrame {
            sentence: sentence.clone(),
            mentions: scan_bundle
                .scan
                .mentions
                .iter()
                .filter(|mention| mention.sentence_index == sentence.index)
                .cloned()
                .collect(),
            chunks: scan_bundle
                .chunks
                .iter()
                .filter(|chunk| {
                    sentence.range.start <= chunk.range.start
                        && sentence.range.end >= chunk.range.end
                })
                .map(|chunk| ChunkSpan {
                    kind: None,
                    range: chunk.range,
                    head: chunk.range,
                    modifiers: Vec::new(),
                    sentence_index: sentence.index,
                })
                .collect(),
            verb_frames: Vec::new(),
            clause_ranges: vec![sentence.range],
            diagnostics: Vec::new(),
        })
        .collect::<Vec<_>>();

    let relations = build_native_relation_candidates(document, scan_bundle);
    let mut evidence_spans = Vec::new();
    for relation in &relations {
        let Some(hit) = scan_bundle.scan.narrative_hits.iter().find(|hit| {
            hit.sentence_index == relation.sentence_index
                && hit.relation_type == relation.relation_type
        }) else {
            continue;
        };
        let sentence = match scan_bundle.scan.sentences.get(relation.sentence_index) {
            Some(sentence) => sentence,
            None => continue,
        };
        if let Some(frame) = sentence_frames.get_mut(relation.sentence_index) {
            frame.verb_frames.push(VerbFrame {
                verb_range: relation.verb_range,
                lemma: hit.lemma.clone(),
                event_class: hit.event_class.clone(),
                relation_type: relation.relation_type.clone(),
                transitivity: hit.transitivity.clone(),
                subject_candidates: relation.subject.clone().into_iter().collect(),
                object_candidates: relation.object.clone().into_iter().collect(),
                recipient_candidates: Vec::new(),
                pp_attachments: Vec::new(),
                clause_range: sentence.range,
                evidence: relation.evidence.clone(),
            });
        }
        evidence_spans.extend(relation.evidence.iter().cloned());
    }

    StructureArtifact {
        sentence_frames,
        relations,
        evidence_spans,
        diagnostics: vec![Diagnostic {
            code: "PX_CAUSAL_SUBSTRATE_STRUCTURE".to_owned(),
            message: "Rebuilt compact structure artifact for causal substrate compilation."
                .to_owned(),
        }],
    }
}

fn frame_slot_from_native_mention(mention: &MentionSpan) -> FrameSlot {
    FrameSlot {
        range: mention.range,
        entity_ref: mention.entity_ref.clone(),
        confidence: mention.confidence,
    }
}

fn native_ner_label(mention: &MentionSpan, coref_kind: CorefMentionKind) -> &'static str {
    match coref_kind {
        CorefMentionKind::Pronoun => "pronoun",
        CorefMentionKind::Nominal => "nominal",
        CorefMentionKind::Named => match mention.kind {
            Some(EntityKind::Character | EntityKind::Npc) => "person",
            Some(EntityKind::Organization | EntityKind::Faction) => "organization",
            Some(EntityKind::Location) => "location",
            Some(EntityKind::Event) => "event",
            Some(EntityKind::Item) => "item",
            Some(EntityKind::Concept) => "concept",
            Some(EntityKind::Other) | None => "named",
        },
    }
}

fn native_mention_source_name(source: &MentionSource) -> &'static str {
    match source {
        MentionSource::Known => "known",
        MentionSource::Alias => "alias",
        MentionSource::Fuzzy => "fuzzy",
        MentionSource::Discovery => "discovery",
    }
}

fn native_progress_enabled() -> bool {
    std::env::var_os("PHOENIX_PERF_PROGRESS").is_some()
        || std::env::var_os("PHOENIX_INGEST_PROGRESS").is_some()
}
