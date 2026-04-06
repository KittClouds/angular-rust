use std::collections::BTreeMap;
use std::cmp::Ordering;
use std::path::PathBuf;
#[cfg(feature = "background-verifier")]
use std::path::Path;

use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use memchr::memchr3_iter;
use phoenix_chunker::{build_chunks, ChunkerConfig};
use phoenix_graph_kernel::{
    KernelEdge, KernelEdgeType, KernelEntitySidecar, KernelGraphLayer, KernelGraphSnapshot,
    KernelMutationBatch,
    KernelMutationScope, KernelProvenance, KernelResolutionFacet, KernelVertex, KernelVertexId,
    PhoenixGraphKernel,
};
use phoenix_semantic_v2::{
    scope_storage_key, AliasConfirmation, AliasEntry, AliasPosting, CandidateEntity,
    CandidateEvidence, ChunkId, ChunkRecord, CompactResolutionKind, CompactResolutionRow,
    CorefClusterRecord, DirtyScopeRecord, DocumentArchive, DocumentManifest, DocumentOrd,
    DocumentOrdinalAssignment, DocumentRevisionRef, DocumentSegmentHeader, DocumentSegmentKind,
    DocumentSegmentRef, DocumentVersionId, LexicalPostingsSegment, NativeCorefSummary,
    NativeErSummary, PreparedDocument, PreparedDocumentSegment, ResolutionDecision,
    ResolvedMention, ScopeLexSidecar, ScopeOrd, SemanticEntityRecord, SemanticRelationRecord,
    SessionArchive,
};
use phoenix_store_native_core::{
    BundleHeader, BundleKey, BundleKind, PhoenixArchiveStoreV2, PhoenixBundleStoreV2,
    StoreError,
};
use phoenix_types::{
    BoundaryKind, ChunkKind, ChunkSpan, Diagnostic, DocumentId, EntityId, EntityKind,
    EvidenceSpan, FrameSlot, IndexedSpan, IndexedTextField, IngestDocument, IngestDocumentSummary,
    IngestResult, LexicalField, MentionEntityRef, MentionSource, MentionSpan,
    NarrativeTransitivity, NarrativeVerbHit, PosTag, RelationCandidate, ResolverEntitySeed,
    ResolverLink, ResolverLinkKind, ScopeKey, ScanArtifact, SentenceFrame, SentenceSpan,
    SessionDocumentState, SessionId, StructureArtifact, TextRange, TokenClass, TokenSpan,
    VerbFrame,
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
use std::time::Instant;
#[cfg(feature = "background-verifier")]
use thiserror::Error;

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

fn tokenize(text: &str) -> Vec<TokenSpan> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }
        if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
            let mut end = start + ch.len_utf8();
            while let Some((next_ix, next)) = chars.peek().copied() {
                if next.is_alphanumeric() || next == '\'' || next == '-' {
                    chars.next();
                    end = next_ix + next.len_utf8();
                } else {
                    break;
                }
            }
            let token = &text[start..end];
            let lower = token.to_ascii_lowercase();
            let pos = if is_pronoun(&lower) {
                Some(PosTag::Pronoun)
            } else if is_verb_token(&lower) {
                Some(PosTag::Verb)
            } else if token.chars().next().is_some_and(|value| value.is_uppercase()) {
                Some(PosTag::ProperNoun)
            } else {
                Some(PosTag::Noun)
            };
            tokens.push(TokenSpan {
                range: to_range(start, end),
                token_class: Some(if token.chars().all(|value| value.is_numeric()) {
                    TokenClass::Number
                } else {
                    TokenClass::Word
                }),
                pos,
                masked: false,
                capitalized: token.chars().next().is_some_and(|value| value.is_uppercase()),
            });
        } else {
            tokens.push(TokenSpan {
                range: to_range(start, start + ch.len_utf8()),
                token_class: Some(if ch.is_ascii_punctuation() {
                    TokenClass::Punctuation
                } else {
                    TokenClass::Symbol
                }),
                pos: Some(PosTag::Punctuation),
                masked: false,
                capitalized: false,
            });
        }
    }
    tokens
}

fn sentence_spans(text: &str) -> Vec<SentenceSpan> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut current_start = 0usize;
    for boundary in memchr3_iter(b'.', b'!', b'?', text.as_bytes()) {
        let end = boundary + 1;
        let trimmed_start = trim_start_offset(text, current_start, end);
        let trimmed_end = trim_end_offset(text, trimmed_start, end);
        if trimmed_start < trimmed_end {
            ranges.push((trimmed_start, trimmed_end));
        }
        current_start = end;
    }
    if current_start < text.len() {
        let trimmed_start = trim_start_offset(text, current_start, text.len());
        let trimmed_end = trim_end_offset(text, trimmed_start, text.len());
        if trimmed_start < trimmed_end {
            ranges.push((trimmed_start, trimmed_end));
        }
    }
    ranges
        .into_iter()
        .enumerate()
        .map(|(index, (start, end))| SentenceSpan {
            index,
            range: to_range(start, end),
        })
        .collect()
}

const RULE_NER_MAX_BYTES_WITHOUT_SEEDS: usize = 256 * 1024;
const HOT_PATH_PATTERN_NER_MAX_BYTES: usize = 512 * 1024;

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

#[derive(Clone, Debug)]
struct GazetteerEntry {
    token_forms: SmallVec<[String; 4]>,
    kind: Option<EntityKind>,
    entity_ref: Option<MentionEntityRef>,
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

fn build_seed_gazetteer(
    resolver_seed: &[ResolverEntitySeed],
) -> FxHashMap<String, SmallVec<[GazetteerEntry; 4]>> {
    let mut by_first_token = FxHashMap::<String, SmallVec<[GazetteerEntry; 4]>>::default();
    for seed in resolver_seed {
        let forms = std::iter::once(seed.canonical_name.as_str()).chain(seed.aliases.iter().map(String::as_str));
        for form in forms {
            let tokens = normalize_surface(form)
                .split_whitespace()
                .map(str::to_owned)
                .collect::<SmallVec<[String; 4]>>();
            if tokens.is_empty() {
                continue;
            }
            let entry = GazetteerEntry {
                token_forms: tokens.clone(),
                kind: seed.kind.clone(),
                entity_ref: Some(MentionEntityRef::Known(seed.entity_id.clone())),
            };
            by_first_token
                .entry(tokens[0].clone())
                .or_default()
                .push(entry);
        }
    }
    for entries in by_first_token.values_mut() {
        entries.sort_by(|left, right| right.token_forms.len().cmp(&left.token_forms.len()));
    }
    by_first_token
}

fn seeded_gazetteer_mentions(
    text: &str,
    tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
    resolver_seed: &[ResolverEntitySeed],
) -> Vec<DetectedMention> {
    let gazetteer = build_seed_gazetteer(resolver_seed);
    if gazetteer.is_empty() {
        return Vec::new();
    }
    let normalized_tokens = tokens
        .iter()
        .map(|token| normalize_token_surface(slice_or_empty(text, token.range)))
        .collect::<Vec<_>>();
    let mut mentions = Vec::new();
    let mut sentence_cursor = 0usize;
    let mut index = 0usize;
    while index < tokens.len() {
        let Some(entries) = gazetteer.get(&normalized_tokens[index]) else {
            index += 1;
            continue;
        };
        let mut matched = false;
        for entry in entries {
            let token_len = entry.token_forms.len();
            if index + token_len > tokens.len() {
                continue;
            }
            let window = &normalized_tokens[index..index + token_len];
            if window != entry.token_forms.as_slice() {
                continue;
            }
            let start = tokens[index].range.start;
            let end = tokens[index + token_len - 1].range.end;
            let range = TextRange { start, end };
            let surface = slice_or_empty(text, range).to_owned();
            mentions.push(DetectedMention {
                range,
                surface: surface.clone(),
                normalized: normalize_surface(&surface),
                mention_kind: DetectedMentionKind::Named,
                type_hint: entry.kind.clone(),
                entity_ref: entry.entity_ref.clone(),
                source: DetectedMentionSourceKind::SeedGazetteer,
                confidence: 0.98,
                sentence_index: locate_sentence_cursor(sentences, &mut sentence_cursor, range),
            });
            index += token_len;
            matched = true;
            break;
        }
        if !matched {
            index += 1;
        }
    }
    mentions
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
    matches!(value.to_ascii_lowercase().as_str(), "of" | "the" | "and" | "&")
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
        "mr"
            | "mrs"
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
        if observed_ranges.iter().any(|existing| range_overlaps(*existing, range)) {
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
                    type_hint: infer_seed_kind(&surface, resolver_seed).or(Some(EntityKind::Character)),
                    entity_ref: infer_seed_entity_ref(&surface, resolver_seed)
                        .or_else(|| Some(MentionEntityRef::Speculative(normalize_surface(&surface)))),
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
                type_hint: infer_seed_kind(&surface, resolver_seed).or_else(|| infer_heuristic_kind(&surface)),
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

fn detect_mentions(
    text: &str,
    tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
    resolver_seed: &[ResolverEntitySeed],
    config: &InvarantV3ExtractionConfig,
) -> Vec<MentionSpan> {
    let mut detected = seeded_gazetteer_mentions(text, tokens, sentences, resolver_seed);
    let observed_ranges = detected.iter().map(|mention| mention.range).collect::<Vec<_>>();

    if config.enable_scirs2_rule_ner {
        detected.extend(scirs2_rule_mentions(text, sentences, resolver_seed));
    }
    if config.enable_scirs2_pattern_ner {
        detected.extend(scirs2_pattern_mentions(text, sentences));
    }
    if config.enable_native_refinement {
        detected.extend(native_refinement_mentions(
            text,
            tokens,
            sentences,
            &observed_ranges,
            resolver_seed,
        ));
    }

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
    resolver_seed: &[ResolverEntitySeed],
) -> Vec<MentionSpan> {
    detect_mentions(
        text,
        tokens,
        sentences,
        resolver_seed,
        &InvarantV3ExtractionConfig::default(),
    )
}

fn detect_mentions_hot_path(
    text: &str,
    tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
    resolver_seed: &[ResolverEntitySeed],
    config: &InvarantV3ExtractionConfig,
) -> Vec<MentionSpan> {
    let mut hot_config = config.clone();
    if text.len() > HOT_PATH_PATTERN_NER_MAX_BYTES {
        hot_config.enable_scirs2_pattern_ner = false;
    }
    detect_mentions(text, tokens, sentences, resolver_seed, &hot_config)
}

fn build_resolver_links(mentions: &[MentionSpan]) -> Vec<ResolverLink> {
    let mut links = Vec::new();
    let mut last_entity_by_surface = FxHashMap::<String, usize>::default();
    let mut antecedent = None::<usize>;
    for (index, mention) in mentions.iter().enumerate() {
        let normalized = normalize_surface(&mention.surface);
        if is_pronoun(&normalized) {
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
        if let Some(previous_ix) = last_entity_by_surface.get(&normalized).copied() {
            let previous = &mentions[previous_ix];
            links.push(ResolverLink {
                source_range: mention.range,
                target_range: Some(previous.range),
                target_entity: previous.entity_ref.clone(),
                link_kind: Some(ResolverLinkKind::AliasCandidate),
                confidence: 0.61,
                sentence_index: mention.sentence_index,
            });
        }
        if mention.entity_ref.is_some() {
            antecedent = Some(index);
        }
        last_entity_by_surface.insert(normalized, index);
    }
    links
}

fn discover_narrative_hits(
    text: &str,
    tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
) -> Vec<NarrativeVerbHit> {
    let mut hits = Vec::new();
    let mut sentence_cursor = 0usize;
    for token in tokens {
        if !matches!(token.pos, Some(PosTag::Verb)) {
            continue;
        }
        let normalized = normalize_token_surface(slice_or_empty(text, token.range));
        let (lemma, event_class, relation_type, transitivity) = classify_verb(&normalized);
        hits.push(NarrativeVerbHit {
            range: token.range,
            lemma,
            event_class,
            relation_type,
            transitivity,
            sentence_index: locate_sentence_cursor(sentences, &mut sentence_cursor, token.range),
            confidence: 0.7,
        });
    }
    hits
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
        if let Some(summary) = boundary_label.as_deref().filter(|value| !value.trim().is_empty()) {
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
    _scope: &ScopeKey,
    resolver_seed: &[ResolverEntitySeed],
    extraction: &InvarantV3ExtractionConfig,
) -> NativeScanRows {
    let tokens = tokenize(text);
    let sentences = sentence_spans(text);
    let mentions = detect_mentions_hot_path(text, &tokens, &sentences, resolver_seed, extraction);
    let resolver_links = build_resolver_links(&mentions);
    let narrative_hits = discover_narrative_hits(text, &tokens, &sentences);
    let detected_pronoun_count = mentions
        .iter()
        .filter(|mention| is_pronoun(&normalize_surface(&mention.surface)))
        .count();
    let detected_nominal_count = mentions
        .iter()
        .filter(|mention| {
            mention.entity_ref.is_none()
                && !is_pronoun(&normalize_surface(&mention.surface))
                && mention
                    .surface
                    .chars()
                    .next()
                    .map(|ch| ch.is_lowercase())
                    .unwrap_or(false)
        })
        .count();
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
        let hit_indexes = sentence_hits.get(sentence.index).cloned().unwrap_or_default();
        for hit_ix in hit_indexes {
            let hit = &scan.narrative_hits[hit_ix];
            let mut subject_mention_ix = None;
            let mut object_mention_ix = None;
            for mention_ix in &mention_indexes {
                let mention = &scan.mentions[*mention_ix];
                if mention.range.end <= hit.range.start {
                    subject_mention_ix = Some(*mention_ix);
                } else if mention.range.start >= hit.range.end {
                    object_mention_ix = Some(*mention_ix);
                    break;
                }
            }
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

fn classify_coref_mention(mention: &MentionSpan) -> CorefMentionKind {
    let normalized = normalize_surface(&mention.surface);
    if is_pronoun(&normalized) {
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

fn acronym_of(surface: &str) -> Option<String> {
    let parts = normalize_surface(surface)
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
    if let (
        Some(MentionEntityRef::Known(left)),
        Some(MentionEntityRef::Known(right)),
    ) = (current.entity_ref.as_ref(), antecedent.entity_ref.as_ref())
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
        (CorefMentionKind::Pronoun, CorefMentionKind::Named) => Some(CorefPairRoute::PronounToNamed),
        (CorefMentionKind::Pronoun, CorefMentionKind::Nominal) => {
            Some(CorefPairRoute::PronounToNominal)
        }
        (CorefMentionKind::Nominal, CorefMentionKind::Named) => Some(CorefPairRoute::NominalToNamed),
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
        CorefMentionKind::Nominal => {
            (config.max_nominal_antecedents, config.max_nominal_sent_window)
        }
        CorefMentionKind::Pronoun => {
            (config.max_pronoun_antecedents, config.max_pronoun_sent_window)
        }
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

fn build_coref_rows(
    scan: &NativeScanRows,
    structure: &NativeStructureRows,
    config: &InvarantV3CorefConfig,
) -> NativeCorefRows {
    let mut result = NativeCorefRows::default();
    if !config.enable || scan.mentions.is_empty() {
        return result;
    }

    let rows = scan
        .mentions
        .iter()
        .enumerate()
        .map(|(_mention_ix, mention)| CorefMentionRow {
            sentence_index: mention.sentence_index,
            chunk_index: structure
                .sentence_chunk_indexes
                .get(mention.sentence_index)
                .copied()
                .flatten(),
            mention_kind: classify_coref_mention(mention),
            has_known_seed: matches!(mention.entity_ref, Some(MentionEntityRef::Known(_))),
            kind: mention.kind.clone(),
        })
        .collect::<Vec<_>>();
    let normalized_surfaces = scan
        .mentions
        .iter()
        .map(|mention| normalize_surface(&mention.surface))
        .collect::<Vec<_>>();
    let acronyms = scan
        .mentions
        .iter()
        .map(|mention| acronym_of(&mention.surface))
        .collect::<Vec<_>>();
    let mut cluster_by_mention = vec![0u32; rows.len()];
    let mut representative_by_mention = vec![None; rows.len()];
    let mut candidate_links_by_mention = vec![SmallVec::<[usize; 2]>::new(); rows.len()];
    let mut surface_recent = FxHashMap::<String, SmallVec<[usize; 8]>>::default();
    let mut acronym_recent = FxHashMap::<String, SmallVec<[usize; 8]>>::default();
    let mut recent_named = SmallVec::<[usize; 32]>::new();
    let mut recent_nominal = SmallVec::<[usize; 24]>::new();
    let mut clusters = Vec::<CorefClusterState>::new();
    let mut attached_mention_count = 0usize;
    let mut candidate_link_count = 0usize;

    for (index, row) in rows.iter().enumerate() {
        let mention = &scan.mentions[index];
        let normalized = &normalized_surfaces[index];
        let (_max_antecedents, max_sent_window) = coref_window_limits(row.mention_kind, config);
        let mut candidate_pool = SmallVec::<[usize; 16]>::new();
        let mut push_candidate_ix = |candidate_ix: usize| {
            if candidate_ix >= index || candidate_pool.iter().any(|existing| *existing == candidate_ix) {
                return;
            }
            candidate_pool.push(candidate_ix);
        };
        if !normalized.is_empty() {
            if let Some(bucket) = surface_recent.get(normalized) {
                for candidate_ix in bucket.iter().rev() {
                    push_candidate_ix(*candidate_ix);
                }
            }
        }
        if let Some(acronym) = acronyms[index].as_ref() {
            if let Some(bucket) = acronym_recent.get(acronym) {
                for candidate_ix in bucket.iter().rev() {
                    push_candidate_ix(*candidate_ix);
                }
            }
        }
        match row.mention_kind {
            CorefMentionKind::Pronoun => {
                for candidate_ix in recent_named.iter().rev().take(12) {
                    if row
                        .sentence_index
                        .saturating_sub(rows[*candidate_ix].sentence_index)
                        <= max_sent_window
                    {
                        push_candidate_ix(*candidate_ix);
                    }
                }
                for candidate_ix in recent_nominal.iter().rev().take(6) {
                    if row
                        .sentence_index
                        .saturating_sub(rows[*candidate_ix].sentence_index)
                        <= max_sent_window
                    {
                        push_candidate_ix(*candidate_ix);
                    }
                }
            }
            CorefMentionKind::Nominal => {
                for candidate_ix in recent_named.iter().rev().take(10) {
                    if row
                        .sentence_index
                        .saturating_sub(rows[*candidate_ix].sentence_index)
                        <= max_sent_window
                    {
                        push_candidate_ix(*candidate_ix);
                    }
                }
            }
            CorefMentionKind::Named => {
                for candidate_ix in recent_named.iter().rev().take(6) {
                    if row
                        .sentence_index
                        .saturating_sub(rows[*candidate_ix].sentence_index)
                        <= max_sent_window
                    {
                        push_candidate_ix(*candidate_ix);
                    }
                }
            }
        }

        let mut best = None::<CorefAntecedentCandidate>;
        let mut runner_up = None::<CorefAntecedentCandidate>;

        for prior_ix in candidate_pool {
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
                &normalized_surfaces[prior_ix],
                row.mention_kind,
                antecedent_row.mention_kind,
                acronyms[index].as_deref(),
                acronyms[prior_ix].as_deref(),
            ) else {
                continue;
            };
            let cluster_ix = cluster_by_mention[prior_ix] as usize;
            let representative = clusters
                .get(cluster_ix)
                .map(|cluster| cluster.representative_mention_ix == prior_ix)
                .unwrap_or(false);
            let surface_repeat = !normalized.is_empty()
                && surface_recent
                    .get(normalized)
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
        let top_score = best.map(|candidate| candidate.score_millis).unwrap_or_default();
        let runner_score = runner_up.map(|candidate| candidate.score_millis).unwrap_or_default();
        let margin = top_score.saturating_sub(runner_score);
        let attach = best.filter(|candidate| {
            candidate.score_millis >= threshold && margin >= margin_threshold
        });
        let near_threshold = best.filter(|candidate| {
            candidate.score_millis + 80 >= threshold && candidate.score_millis < threshold
        });

        let cluster_ix = if let Some(attach) = attach {
            let cluster_ix = cluster_by_mention[attach.mention_ix] as usize;
            let cluster = &mut clusters[cluster_ix];
            cluster.member_indexes.push(index);
            cluster.most_recent_mention_ix = index;
            cluster.first_sentence_index = cluster.first_sentence_index.min(row.sentence_index);
            cluster.last_sentence_index = cluster.last_sentence_index.max(row.sentence_index);
            if let Some(chunk_index) = row.chunk_index {
                if cluster.chunk_indexes.len() < config.persist_chunk_cap
                    && !cluster.chunk_indexes.iter().any(|existing| *existing == chunk_index)
                {
                    cluster.chunk_indexes.push(chunk_index);
                }
            }
            match row.mention_kind {
                CorefMentionKind::Named => {
                    cluster.named_count += 1;
                    cluster.best_named_mention_ix = Some(index);
                }
                CorefMentionKind::Nominal => cluster.nominal_count += 1,
                CorefMentionKind::Pronoun => cluster.pronoun_count += 1,
            }
            if row.has_known_seed {
                cluster.best_seeded_mention_ix = Some(index);
            }
            cluster.route_mix_bits |= attach.route.bit();
            cluster.max_score_millis = cluster.max_score_millis.max(attach.score_millis);
            update_cluster_representative(cluster, &rows);
            representative_by_mention[index] = Some(cluster.representative_mention_ix);
            attached_mention_count += 1;
            cluster_ix
        } else {
            let mut cluster = CorefClusterState {
                member_indexes: vec![index],
                representative_mention_ix: index,
                most_recent_mention_ix: index,
                best_named_mention_ix: matches!(row.mention_kind, CorefMentionKind::Named)
                    .then_some(index),
                best_seeded_mention_ix: row.has_known_seed.then_some(index),
                first_sentence_index: row.sentence_index,
                last_sentence_index: row.sentence_index,
                chunk_indexes: SmallVec::new(),
                named_count: usize::from(matches!(row.mention_kind, CorefMentionKind::Named)),
                nominal_count: usize::from(matches!(row.mention_kind, CorefMentionKind::Nominal)),
                pronoun_count: usize::from(matches!(row.mention_kind, CorefMentionKind::Pronoun)),
                route_mix_bits: 0,
                max_score_millis: 0,
                ambiguous: false,
            };
            if let Some(chunk_index) = row.chunk_index {
                cluster.chunk_indexes.push(chunk_index);
            }
            clusters.push(cluster);
            clusters.len() - 1
        };

        if let Some(candidate) = near_threshold {
            candidate_links_by_mention[index].push(candidate.mention_ix);
            candidate_link_count += 1;
            if let Some(cluster) = clusters.get_mut(cluster_ix) {
                cluster.ambiguous = true;
                cluster.route_mix_bits |= candidate.route.bit();
                cluster.max_score_millis = cluster.max_score_millis.max(candidate.score_millis);
            }
        }
        cluster_by_mention[index] = cluster_ix as u32;
        if !normalized.is_empty() {
            let bucket = surface_recent.entry(normalized.clone()).or_default();
            bucket.push(index);
            if bucket.len() > 8 {
                bucket.remove(0);
            }
        }
        if let Some(acronym) = acronyms[index].as_ref() {
            let bucket = acronym_recent.entry(acronym.clone()).or_default();
            bucket.push(index);
            if bucket.len() > 8 {
                bucket.remove(0);
            }
        }
        match row.mention_kind {
            CorefMentionKind::Named => {
                recent_named.push(index);
                if recent_named.len() > 128 {
                    recent_named.remove(0);
                }
            }
            CorefMentionKind::Nominal => {
                recent_nominal.push(index);
                if recent_nominal.len() > 64 {
                    recent_nominal.remove(0);
                }
            }
            CorefMentionKind::Pronoun => {}
        }
    }

    result.rows = rows;
    result.cluster_by_mention = cluster_by_mention;
    result.representative_by_mention = representative_by_mention;
    result.candidate_links_by_mention = candidate_links_by_mention;
    result.summary = NativeCorefSummary {
        cluster_count: clusters.len(),
        attached_mention_count,
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
    let _ = memory.kernel.rebuild_from_kernel_batches(vec![
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
    ]);
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

fn stable_slot_cmp(
    left: &CandidateSlot,
    right: &CandidateSlot,
    entity_ids: &[String],
) -> Ordering {
    right
        .score_millis
        .cmp(&left.score_millis)
        .then_with(|| {
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

    if let Some(entry) = candidates.iter_mut().find(|candidate| candidate.entity_ord == entity_ord) {
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
        evidence.push(candidate_evidence(evidence_kind.as_str(), evidence_detail()));
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

fn best_candidate_summary(candidates: &[CandidateSlot], entity_ids: &[String]) -> Option<BestCandidateSummary> {
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
    right
        .score_millis
        .cmp(&left.score_millis)
        .then_with(|| {
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

    if let Some(entry) = candidates.iter_mut().find(|candidate| candidate.entity_ord == entity_ord) {
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

fn bump_small_count(
    counts: &mut SmallVec<[(u32, usize); 4]>,
    entity_ord: u32,
    delta: usize,
) {
    if let Some((_, count)) = counts.iter_mut().find(|(existing_ord, _)| *existing_ord == entity_ord)
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
        links_by_source.entry(source_ix).or_default().push(target_ix);
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
            resolver_entity_links: entity_links_by_source.remove(&mention_ix).unwrap_or_default(),
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
    (detailed, resolved_mentions, alias_confirmations, diagnostics)
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
    let (prepared, surface_values) =
        build_prepared_mentions(document, &scan.mentions, &scan.resolver_links, chunks);
    let kernel_resolved_entities =
        build_kernel_mention_resolution_map(document, prepared.len(), entity_memory);
    if progress {
        eprintln!(
            "[runtime-ingest] resolve_subphase=prepare_mentions document_id={} wall_ms={} mentions={} surfaces={}",
            document.document_id.0,
            phase_started.elapsed().as_millis(),
            prepared.len(),
            surface_values.len(),
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
    for normalized in &surface_values {
        if let Some(kernel_candidates) = entity_memory.entity_index.alias_candidates.get(normalized) {
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
            let speculative_id =
                format!("{}::{}", document.document_id.0, normalized.replace(' ', "_"));
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
    let surface_count = surface_values.len();
    let mut surface_kernel_candidates =
        vec![SmallVec::<[CompactCandidateSlot; 4]>::new(); surface_count];
    let mut surface_speculative_ord = vec![None; surface_count];
    let mut surface_pronouns = vec![false; surface_count];
    let mut surface_known_counts = vec![SmallVec::<[(u32, usize); 4]>::new(); surface_count];
    let mut resolver_links_by_mention =
        vec![SmallVec::<[CompactResolverEntityLink; 4]>::new(); prepared.len()];
    let mut kernel_resolved_ord_by_mention = vec![None; prepared.len()];
    let mut coref_rep_seed_ord_by_mention = vec![None; prepared.len()];

    for (surface_ord, normalized) in surface_values.iter().enumerate() {
        surface_pronouns[surface_ord] = is_pronoun(normalized);
        if let Some(kernel_candidates) = entity_memory.entity_index.alias_candidates.get(normalized) {
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
        if !surface_pronouns[surface_ord] && !normalized.is_empty() {
            let speculative_id =
                format!("{}::{}", document.document_id.0, normalized.replace(' ', "_"));
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
            surface_count,
        );
    }

    let phase_started = Instant::now();
    let mut mention_states = vec![SmallVec::<[CompactCandidateSlot; 6]>::new(); prepared.len()];
    let mut base_best = vec![None; prepared.len()];
    let mut surface_support = vec![SmallVec::<[(u32, usize); 4]>::new(); surface_count];

    for (index, prepared_mention) in prepared.iter().enumerate() {
        let mention = &scan.mentions[prepared_mention.mention_ix];
        let normalized = &surface_values[prepared_mention.surface_ord as usize];
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
            for (entity_ord, support) in &surface_known_counts[prepared_mention.surface_ord as usize] {
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
        let normalized = &surface_values[prepared_mention.surface_ord as usize];
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
            if candidate.source == CandidateSourceKind::NewSpeculative && mention.confidence >= 0.8 {
                candidate.score_millis += 180;
            }
            if prepared_mention.chunk_ix.is_some() {
                candidate.score_millis += 20;
            }
        }

        sort_compact_candidate_slots(candidates, &entity_ids);
        let top = candidates.first().copied();
        let runner_up = candidates.get(1).copied();
        let top_score = top.map(|candidate| candidate.score_millis).unwrap_or_default();
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
                && normalized.as_str() != entity_normalized[entity_ord as usize]
                && top_candidate.source != CandidateSourceKind::Seed
                && top_candidate.source != CandidateSourceKind::KernelResolved
                && top_candidate.source != CandidateSourceKind::PronounLink
                && compact_candidate_has_alias_signal(&top_candidate)
                && top_score >= 1000
                && margin >= 260
                && !entity_memory
                    .known_aliases
                    .contains(&(entity_id.clone(), normalized.clone()))
            {
                alias_confirmations.push(AliasConfirmationOrd {
                    alias_surface: mention.surface.clone(),
                    normalized: normalized.clone(),
                    entity_ord,
                    confidence_millis: top_score.max(0) as u32,
                    mention_index: prepared_mention.mention_ix,
                });
            } else if !normalized.is_empty()
                && normalized.as_str() != entity_normalized[entity_ord as usize]
                && !entity_memory
                    .known_aliases
                    .contains(&(entity_id.clone(), normalized.clone()))
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

        if let Some(kernel_candidates) = entity_memory
            .entity_index
            .alias_candidates
            .get(normalized)
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
            if let Some(surface_entities) = surface_known_counts.get(&prepared_mention.surface_ord) {
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
                            if !candidate.evidence.iter().any(|existing| existing == &evidence) {
                                candidate.evidence.push(evidence);
                            }
                        }
                    }
                }
            }
            if candidate.source == CandidateSourceKind::NewSpeculative && mention.confidence >= 0.8 {
                candidate.score_millis += 180;
            }
            if prepared_mention.chunk_ix.is_some() {
                candidate.score_millis += 20;
            }
        }

        sort_candidate_slots(candidates, &entity_ids);
        let top = candidates.first().cloned();
        let runner_up = candidates.get(1).cloned();
        let top_score = top.as_ref().map(|candidate| candidate.score_millis).unwrap_or_default();
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

        if let (ResolvedMentionKind::Resolved, Some(entity_id), Some(top_candidate)) = (
            &decision.kind,
            decision.entity_id.as_ref(),
            top.as_ref(),
        ) {
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
            ResolveDetailMode::Detailed => candidate_list_from_slots(candidates.clone(), &entity_ids),
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
    let resolved_entity_by_mention_ix = resolutions
        .iter()
        .map(|resolution| resolution.entity_ord)
        .collect::<Vec<_>>();

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

    let relation_records = structure
        .relation_seeds
        .iter()
        .filter_map(|relation| {
            let source_entity_ord = relation
                .subject_mention_ix
                .and_then(|mention_ix| resolved_entity_by_mention_ix.get(mention_ix))
                .copied()
                .flatten();
            let target_entity_ord = relation
                .object_mention_ix
                .and_then(|mention_ix| resolved_entity_by_mention_ix.get(mention_ix))
                .copied()
                .flatten();
            if source_entity_ord.is_none() || target_entity_ord.is_none() {
                unresolved_relation_count += 1;
            }
            Some(SemanticRelationRecord {
                source_entity_id: EntityId(entity_ids[source_entity_ord? as usize].clone()),
                target_entity_id: EntityId(entity_ids[target_entity_ord? as usize].clone()),
                edge_type: relation.relation_type.clone(),
                sentence_index: relation.sentence_index,
                chunk_id: structure
                    .sentence_chunk_indexes
                    .get(relation.sentence_index)
                    .cloned()
                    .flatten()
                    .and_then(|chunk_ix| chunks.get(chunk_ix as usize))
                    .map(|chunk| chunk.chunk_id.0.clone()),
            })
        })
        .collect::<Vec<_>>();

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

    let diagnostics = if unresolved_relation_count == 0 {
        Vec::new()
    } else {
        vec![Diagnostic {
            code: "er_relation_skipped_unresolved_entity".to_owned(),
            message: format!(
                "Skipped {} asserted relations because one or more arguments stayed unresolved.",
                unresolved_relation_count
            ),
        }]
    };

    (
        entity_records,
        relation_records,
        scan.discovery_count,
        diagnostics,
    )
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
            if !resolved_entity_ids.iter().any(|existing| existing == &entity_id)
                && resolved_entity_ids.len() < 4
            {
                resolved_entity_ids.push(entity_id);
            }
        }
        if resolved_entity_ids.len() > 1 {
            conflict_cluster_count += 1;
        }
        if cluster.member_indexes.len() == 1
            && resolved_entity_ids.len() <= 1
            && !cluster.ambiguous
        {
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
        "person" | "role" => Some(EntityKind::Character),
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
            message: "Background verifier is enabled, but local GLiNER model paths are not configured.".to_owned(),
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
            entity_facet: Some(phoenix_graph_kernel::KernelEntityFacet {
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

fn frame_slot_from_mention(mention: &MentionSpan) -> FrameSlot {
    FrameSlot {
        range: mention.range,
        entity_ref: mention.entity_ref.clone(),
        confidence: mention.confidence,
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
                || matches!(lower.as_str(), "prologue" | "epilogue" | "preface" | "introduction");
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

fn document_alias_entries(document: &DocumentArchive) -> Vec<AliasEntry> {
    let mut entries = BTreeMap::<String, BTreeMap<(String, String), usize>>::new();
    for entity in &document.entities {
        let forms = std::iter::once(entity.canonical_name.clone()).chain(entity.aliases.clone());
        for form in forms {
            let normalized = normalize_surface(&form);
            if normalized.is_empty() {
                continue;
            }
            entries
                .entry(normalized)
                .or_default()
                .entry((entity.entity_id.0.clone(), document.manifest.document_id.clone()))
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

fn speculative_entity_id(document_id: &DocumentId, surface: &str) -> String {
    format!("{}::{}", document_id.0, normalize_surface(surface).replace(' ', "_"))
}

fn entity_id_from_ref(document: &IngestDocument, entity_ref: &MentionEntityRef) -> Option<EntityId> {
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

fn is_verb_token(value: &str) -> bool {
    matches!(
        value,
        "attack" | "attacked" | "attacks" | "met" | "meet" | "meets" | "rose" | "rise"
            | "rises" | "woke" | "wake" | "wakes" | "wrote" | "write" | "writes"
            | "mapped" | "map" | "maps" | "gave" | "give" | "gives" | "waited" | "wait"
            | "waits" | "saw" | "see" | "sees" | "found" | "find" | "finds" | "fought"
            | "fight" | "fights" | "moved" | "move" | "moves" | "crossed" | "cross"
            | "crosses"
    ) || value.ends_with("ed")
}

fn classify_verb(value: &str) -> (String, String, String, Option<NarrativeTransitivity>) {
    match value {
        "attack" | "attacked" | "attacks" | "fight" | "fought" | "fights" => (
            "attack".to_owned(),
            "conflict".to_owned(),
            "attacks".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
        "met" | "meet" | "meets" => (
            "meet".to_owned(),
            "interaction".to_owned(),
            "meets".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
        "gave" | "give" | "gives" => (
            "give".to_owned(),
            "transfer".to_owned(),
            "gives".to_owned(),
            Some(NarrativeTransitivity::Ditransitive),
        ),
        "wrote" | "write" | "writes" | "mapped" | "map" | "maps" => (
            value.trim_end_matches("ed").trim_end_matches('s').to_owned(),
            "creation".to_owned(),
            "writes".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
        other => (
            other.trim_end_matches("ed").trim_end_matches('s').to_owned(),
            "action".to_owned(),
            "relates_to".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
    }
}

fn range_contains(container: TextRange, inner: TextRange) -> bool {
    container.start <= inner.start && container.end >= inner.end
}

fn count_token_words(tokens: &[TokenSpan]) -> usize {
    tokens
        .iter()
        .filter(|token| matches!(token.token_class, Some(TokenClass::Word)))
        .count()
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

fn trim_start_offset(text: &str, start: usize, end: usize) -> usize {
    let slice = &text[start..end];
    start + slice.len().saturating_sub(slice.trim_start().len())
}

fn trim_end_offset(text: &str, start: usize, end: usize) -> usize {
    let slice = &text[start..end];
    start + slice.trim_end().len()
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
    use phoenix_graph_kernel::KernelVertexClass;
    use phoenix_store_lmdb::PhoenixLmdbStore;
    use phoenix_types::{GenderHint, NoteId};
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!(
            "phoenix-invarant-v2-test-{name}-{}",
            SystemTime::now().duration_since(UNIX_EPOCH).expect("time").as_nanos()
        ))
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
        let store = PhoenixLmdbStore::open(temp_path("archives")).expect("store");
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
            pronoun.entity_id.as_ref().map(|entity_id| entity_id.0.as_str()),
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
        let store = PhoenixLmdbStore::open(temp_path("coref-archive")).expect("store");
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
        assert!(!archives[0].coref_clusters.is_empty());
        assert!(archives[0].coref_summary.cluster_count >= 1);
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
                    entity_facet: Some(phoenix_graph_kernel::KernelEntityFacet {
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
        let (_resolutions, resolved_mentions, aliases, diagnostics) = resolve_mentions(
            &document,
            &scan,
            &structure,
            &chunks,
            &memory,
        );
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
                    entity_facet: Some(phoenix_graph_kernel::KernelEntityFacet {
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
                    entity_facet: Some(phoenix_graph_kernel::KernelEntityFacet {
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
        let (_resolutions, resolved_mentions, aliases, diagnostics) = resolve_mentions(
            &document,
            &scan,
            &structure,
            &chunks,
            &memory,
        );
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
                            resolution.entity_id.as_ref().map(|entity_id| entity_id.0.clone()),
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
                            mention.entity_id.as_ref().map(|entity_id| entity_id.0.clone()),
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

#[derive(Default)]
pub struct PhoenixInvarantV3 {
    config: InvarantV3Config,
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

#[derive(Clone, Debug)]
struct NativeScanRows {
    sentences: Vec<SentenceSpan>,
    mentions: Vec<MentionSpan>,
    resolver_links: Vec<ResolverLink>,
    narrative_hits: Vec<NarrativeVerbHit>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CorefMentionKind {
    Named,
    Nominal,
    Pronoun,
}

#[derive(Clone, Debug)]
struct CorefMentionRow {
    sentence_index: usize,
    chunk_index: Option<u32>,
    mention_kind: CorefMentionKind,
    has_known_seed: bool,
    kind: Option<EntityKind>,
}

#[derive(Clone, Copy, Debug)]
struct CorefAntecedentCandidate {
    mention_ix: usize,
    score_millis: i32,
    route: CorefPairRoute,
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
    kernel: PhoenixGraphKernel,
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
    fn load(
        model_path: &Path,
        tokenizer_path: &Path,
    ) -> Result<Self, BackgroundVerifierError> {
        let params = Parameters::default().with_threshold(0.75);
        let runtime_params = RuntimeParameters::default();
        let model = GLiNER::<SpanMode>::new(
            params,
            runtime_params,
            tokenizer_path
                .to_str()
                .ok_or_else(|| {
                    BackgroundVerifierError::InvalidPath(tokenizer_path.display().to_string())
                })?,
            model_path
                .to_str()
                .ok_or_else(|| BackgroundVerifierError::InvalidPath(model_path.display().to_string()))?,
        )
        .map_err(|error| BackgroundVerifierError::ModelLoad(error.to_string()))?;
        Ok(Self { model })
    }

    fn extract(
        &self,
        text: &str,
    ) -> Result<Vec<(String, String, f32)>, BackgroundVerifierError> {
        const LABELS: [&str; 7] = [
            "person",
            "organization",
            "location",
            "event",
            "item",
            "concept",
            "role",
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
struct DocumentOutcome {
    assignment: DocumentOrdinalAssignment,
    archive: DocumentArchive,
    kernel_batch: KernelMutationBatch,
    candidate_kernel_batch: Option<KernelMutationBatch>,
    document_summary: IngestDocumentSummary,
    session_document: SessionDocumentState,
    scope: ScopeKey,
    span_count: usize,
    discovery_count: usize,
    diagnostics: Vec<Diagnostic>,
}

impl PhoenixInvarantV3 {
    pub fn new(config: InvarantV3Config) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &InvarantV3Config {
        &self.config
    }

    pub fn scan_parts(
        &self,
        text: &str,
        _scope: &ScopeKey,
        resolver_seed: &[ResolverEntitySeed],
    ) -> ScanArtifact {
        let progress = native_progress_enabled();
        let phase_started = Instant::now();
        let tokens = tokenize(text);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=tokenize wall_ms={} tokens={}",
                phase_started.elapsed().as_millis(),
                tokens.len(),
            );
        }
        let phase_started = Instant::now();
        let sentences = sentence_spans(text);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=sentence_spans wall_ms={} sentences={}",
                phase_started.elapsed().as_millis(),
                sentences.len(),
            );
        }
        let phase_started = Instant::now();
        let mentions = detect_mentions(
            text,
            &tokens,
            &sentences,
            resolver_seed,
            &self.config.extraction,
        );
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=discover_mentions wall_ms={} mentions={}",
                phase_started.elapsed().as_millis(),
                mentions.len(),
            );
        }
        let phase_started = Instant::now();
        let resolver_links = build_resolver_links(&mentions);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=build_resolver_links wall_ms={} resolver_links={}",
                phase_started.elapsed().as_millis(),
                resolver_links.len(),
            );
        }
        let phase_started = Instant::now();
        let narrative_hits = discover_narrative_hits(text, &tokens, &sentences);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=discover_narrative_hits wall_ms={} narrative_hits={}",
                phase_started.elapsed().as_millis(),
                narrative_hits.len(),
            );
        }
        let chunks = sentences
            .iter()
            .map(|sentence| ChunkSpan {
                kind: Some(ChunkKind::Clause),
                range: sentence.range,
                head: sentence.range,
                modifiers: Vec::new(),
                sentence_index: sentence.index,
            })
            .collect::<Vec<_>>();
        ScanArtifact {
            diagnostics: vec![Diagnostic {
                code: "PX_INVARANT_V2_SCAN".to_owned(),
                message: format!(
                    "Invarant V2 scanned {} tokens, {} sentences, and {} mentions.",
                    count_token_words(&tokens),
                    sentences.len(),
                    mentions.len()
                ),
            }],
            sentences,
            tokens,
            mentions,
            chunks,
            resolver_links,
            narrative_hits,
        }
    }

    pub fn build_structure_parts(&self, text: &str, scan: &ScanArtifact) -> StructureArtifact {
        let mut sentence_mentions = vec![Vec::<MentionSpan>::new(); scan.sentences.len()];
        for mention in &scan.mentions {
            if let Some(bucket) = sentence_mentions.get_mut(mention.sentence_index) {
                bucket.push(mention.clone());
            }
        }
        let mut sentence_chunks = vec![Vec::<ChunkSpan>::new(); scan.sentences.len()];
        for chunk in &scan.chunks {
            if let Some(bucket) = sentence_chunks.get_mut(chunk.sentence_index) {
                bucket.push(chunk.clone());
            }
        }
        let mut sentence_hits = vec![Vec::<NarrativeVerbHit>::new(); scan.sentences.len()];
        for hit in &scan.narrative_hits {
            if let Some(bucket) = sentence_hits.get_mut(hit.sentence_index) {
                bucket.push(hit.clone());
            }
        }

        let mut sentence_frames = Vec::with_capacity(scan.sentences.len());
        let mut relations = Vec::new();
        let mut evidence_spans = Vec::new();

        for sentence in &scan.sentences {
            let index = sentence.index;
            let mentions = sentence_mentions.get(index).cloned().unwrap_or_default();
            let chunks = sentence_chunks.get(index).cloned().unwrap_or_default();
            let hits = sentence_hits.get(index).cloned().unwrap_or_default();
            let mut diagnostics = Vec::new();
            let mut verb_frames = Vec::new();

            for hit in hits {
                let subject = mentions
                    .iter()
                    .filter(|mention| mention.range.end <= hit.range.start)
                    .max_by_key(|mention| mention.range.end)
                    .map(frame_slot_from_mention);
                let trailing = mentions
                    .iter()
                    .filter(|mention| mention.range.start >= hit.range.end)
                    .collect::<Vec<_>>();
                let object = trailing.first().map(|mention| frame_slot_from_mention(mention));
                let recipient = trailing.get(1).map(|mention| frame_slot_from_mention(mention));
                let evidence = vec![EvidenceSpan {
                    document_id: None,
                    note_id: None,
                    label: slice_or_empty(text, sentence.range).trim().to_owned(),
                    kind: Some("sentence".to_owned()),
                    range: sentence.range,
                }];
                evidence_spans.extend(evidence.iter().cloned());
                let attachments = chunks
                    .iter()
                    .filter(|chunk| chunk.range.start >= hit.range.end)
                    .take(1)
                    .map(|chunk| chunk.range)
                    .collect::<Vec<_>>();
                if subject.is_none() {
                    diagnostics.push(Diagnostic {
                        code: "PX_INVARANT_V2_STRUCTURE_SUBJECT_GAP".to_owned(),
                        message: format!(
                            "Invarant V2 inferred relation '{}' without a clear subject.",
                            hit.relation_type
                        ),
                    });
                }
                relations.push(RelationCandidate {
                    sentence_index: index,
                    verb_range: hit.range,
                    lemma: hit.lemma.clone(),
                    event_class: hit.event_class.clone(),
                    relation_type: hit.relation_type.clone(),
                    subject: subject.clone(),
                    object: object.clone(),
                    recipient: recipient.clone(),
                    attachments: attachments.clone(),
                    evidence: evidence.clone(),
                });
                verb_frames.push(VerbFrame {
                    verb_range: hit.range,
                    lemma: hit.lemma,
                    event_class: hit.event_class,
                    relation_type: hit.relation_type,
                    transitivity: hit.transitivity,
                    subject_candidates: subject.into_iter().collect(),
                    object_candidates: object.into_iter().collect(),
                    recipient_candidates: recipient.into_iter().collect(),
                    pp_attachments: attachments,
                    clause_range: sentence.range,
                    evidence,
                });
            }

            sentence_frames.push(SentenceFrame {
                sentence: sentence.clone(),
                mentions,
                chunks,
                verb_frames,
                clause_ranges: vec![sentence.range],
                diagnostics,
            });
        }

        StructureArtifact {
            sentence_frames,
            relations,
            evidence_spans,
            diagnostics: vec![Diagnostic {
                code: "PX_INVARANT_V2_STRUCTURE".to_owned(),
                message: "Invarant V2 built sentence frames and relation candidates.".to_owned(),
            }],
        }
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

        let mut touched_scopes = outcomes.iter().map(|outcome| outcome.scope.clone()).collect::<Vec<_>>();
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
        let drafts = documents
            .par_iter()
            .enumerate()
            .map(|(index, document)| {
                self.build_document_outcome(
                    document,
                    session_id,
                    &context.assignments[index],
                    &entity_memory,
                    created_at,
                )
            })
            .collect::<Vec<_>>();
        let mut outcomes = Vec::with_capacity(drafts.len());
        for draft in drafts {
            outcomes.push(draft?);
        }
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=build_document_outcome finish outcomes={} wall_ms={}",
                outcomes.len(),
                started.elapsed().as_millis()
            );
        }
        outcomes.sort_by(|left, right| {
            left.document_summary
                .document_id
                .0
                .cmp(&right.document_summary.document_id.0)
        });

        let started = Instant::now();
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=prepare_document start documents={}",
                outcomes.len()
            );
        }
        let prepared = outcomes
            .iter()
            .map(|outcome| {
                self.prepare_document(&outcome.archive, &outcome.assignment, &outcome.kernel_batch)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if progress {
            let segment_count = prepared
                .iter()
                .map(|document| document.segments.len())
                .sum::<usize>();
            eprintln!(
                "[runtime-ingest] subphase=prepare_document finish prepared={} segments={} wall_ms={}",
                prepared.len(),
                segment_count,
                started.elapsed().as_millis()
            );
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
        let headers = store.list_bundle_headers(BundleKind::DocumentArchive, scope_key.as_deref())?;
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

    fn prepare_document(
        &self,
        archive: &DocumentArchive,
        assignment: &DocumentOrdinalAssignment,
        kernel_batch: &KernelMutationBatch,
    ) -> Result<PreparedDocument, StoreError> {
        let progress = native_progress_enabled();
        let mut manifest = archive.manifest.clone();
        let mut segments = Vec::<PreparedDocumentSegment>::new();
        let mut segment_refs = Vec::<DocumentSegmentRef>::new();

        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::AliasConfirmationTable,
            archive.alias_confirmations.len(),
            &archive.alias_confirmations,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::CorefClusterTable,
            archive.coref_clusters.len(),
            &archive.coref_clusters,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::ChunkTable,
            archive.chunks.len(),
            &archive.chunks,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::EntityTable,
            archive.entities.len(),
            &archive.entities,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::RelationTable,
            archive.relations.len(),
            &archive.relations,
        )?;
        let lexical_started = Instant::now();
        let lexical = LexicalPostingsSegment {
            spans: archive.indexed_spans.clone(),
            alias_entries: document_alias_entries(archive),
        };
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
        manifest.scope_ord = assignment.scope_ord;
        manifest.document_ord = assignment.document_ord;
        manifest.revision = assignment.revision;

        Ok(PreparedDocument {
            assignment: assignment.clone(),
            manifest,
            segments,
            kernel_batch: kernel_batch.clone(),
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
        outcomes: &[DocumentOutcome],
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
            value.document_ords.sort_by_key(|document_ord| document_ord.0);
            value.document_ords.dedup_by(|left, right| left.0 == right.0);
        }
        values.sort_by(|left, right| left.scope_key.cmp(&right.scope_key));
        values
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
        let document_version_id = DocumentVersionId(format!(
            "{}::{}",
            document.document_id.0, assignment.revision
        ));

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

        let phase_started = Instant::now();
        let (
            resolutions,
            alias_confirmation_ords,
            mut er_summary,
            mut er_diagnostics,
            entity_ids,
            _entity_ord_by_id,
        ) =
            resolve_mentions_compact_native(document, &scan, &coref, &chunks, entity_memory);
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
                &scan,
                &structure,
                &chunks,
                &resolutions,
                &alias_confirmation_ords,
                &entity_ids,
            );
        er_diagnostics.append(&mut relation_diagnostics);
        er_summary.detected_mention_count = scan.mentions.len();
        er_summary.detected_named_count = scan.detected_named_count;
        er_summary.detected_nominal_count = scan.detected_nominal_count;
        er_summary.detected_pronoun_count = scan.detected_pronoun_count;
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
            &scan,
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
            &scan,
            &coref,
            &resolutions,
            &entity_ids,
            &chunks,
            self.config.coref.persist_chunk_cap,
        );

        let phase_started = Instant::now();
        let kernel_batch = build_kernel_batch(
            document,
            &chunks,
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

        let phase_started = Instant::now();
        let document_summary = IngestDocumentSummary {
            document_id: document.document_id.clone(),
            note_id: document.note_id.clone(),
            chapter_count: boundaries.iter().filter(|boundary| boundary.is_chapter).count(),
            boundary_count: boundaries.len(),
            parent_count: boundaries.len(),
            leaf_count: chunks.len(),
            entity_count: entities.len(),
            edge_count: kernel_batch.edges.len(),
            has_front_matter_chapter: boundaries.iter().any(|boundary| {
                boundary.is_chapter && is_front_matter_label(&boundary.label)
            }),
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
            boundary_labels: boundaries.iter().map(|boundary| boundary.label.clone()).collect(),
            parent_count: document_summary.parent_count,
            leaf_count: document_summary.leaf_count,
            entity_count: document_summary.entity_count,
            discovery_count,
            has_front_matter_chapter: document_summary.has_front_matter_chapter,
            has_front_matter_boundary: document_summary.has_front_matter_boundary,
            updated_at: created_at,
        };
        let mention_count = scan.mentions.len();
        let scope_key = assignment.scope_key.clone();
        let manifest = DocumentManifest {
            document_id: document.document_id.0.clone(),
            document_version_id,
            note_id: document.note_id.clone(),
            scope: document.scope.clone(),
            scope_key,
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
            alias_count: entities.iter().map(|entity| entity.aliases.len()).sum(),
            graph_edge_count: kernel_batch.edges.len(),
            graph_vertex_count: kernel_batch.vertices.len(),
            segment_refs: Vec::new(),
            created_at,
            archive_version: 4,
        };
        let archive = DocumentArchive {
            manifest,
            tokens: Vec::new(),
            sentences: Vec::new(),
            mentions: Vec::new(),
            resolver_links: Vec::new(),
            resolved_mentions: Vec::new(),
            alias_confirmations,
            coref_clusters,
            er_summary,
            coref_summary,
            chunks,
            indexed_spans,
            entities,
            relations,
            evidence_spans: Vec::new(),
            relation_candidates: Vec::new(),
            graph_batch: KernelMutationBatch::default(),
            structure: None,
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
            assignment: assignment.clone(),
            archive,
            kernel_batch,
            candidate_kernel_batch,
            document_summary: document_summary.clone(),
            session_document,
            scope: document.scope.clone(),
            span_count: document_summary.leaf_count,
            discovery_count,
            diagnostics: er_diagnostics,
        })
    }
}

fn native_progress_enabled() -> bool {
    std::env::var_os("PHOENIX_PERF_PROGRESS").is_some()
        || std::env::var_os("PHOENIX_INGEST_PROGRESS").is_some()
}
