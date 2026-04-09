use std::cmp::Ordering;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{OnceLock, RwLock};

use compact_str::CompactString;
use memchr::memchr3_iter;
use phoenix_types::{
    Attachment, ChunkKind, ChunkSpan, Clause, Diagnostic, EntityKind, EvidenceSpan, FrameSlot,
    MentionEntityRef, MentionSource, MentionSpan, NarrativeTransitivity, NarrativeVerbHit,
    PhraseKind, PhraseNode, PosTag, QuoteBlock, RelationCandidate, ResolverEntitySeed,
    ResolverLink, ResolverLinkKind, ScanArtifact, ScanRequest, ScopeKey, SentenceFrame,
    SentenceSpan, SourceRange, SpeakerCue, StructureArtifact, StructureRequest, SurfaceDocument,
    SurfaceUnit, SurfaceUnitKind, TextRange, Token, TokenClass, TokenSpan, VerbFrame,
};
use rustc_hash::FxHashMap;
use scirs2_text::information_extraction::{
    Entity as IeEntity, EntityType as IeEntityType, RuleBasedNER,
};
use scirs2_text::named_entity_recognition::{
    extract_entities, NerEntity, NerEntityType as PatternEntityType, NerPatternConfig,
};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

const RULE_NER_MAX_BYTES_WITHOUT_SEEDS: usize = 256 * 1024;
const HOT_PATH_PATTERN_NER_MAX_BYTES: usize = 512 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MachineExtractionConfig {
    pub enable_scirs2_rule_ner: bool,
    pub enable_scirs2_pattern_ner: bool,
    pub enable_native_refinement: bool,
}

impl Default for MachineExtractionConfig {
    fn default() -> Self {
        Self {
            enable_scirs2_rule_ner: true,
            enable_scirs2_pattern_ner: true,
            enable_native_refinement: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MachineConfig {
    #[serde(default)]
    pub extraction: MachineExtractionConfig,
}

impl Default for MachineConfig {
    fn default() -> Self {
        Self {
            extraction: MachineExtractionConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceCompileArtifacts {
    pub scan: ScanArtifact,
    pub structure: StructureArtifact,
    pub surface: SurfaceDocument,
}

pub struct SurfaceCompiler {
    config: MachineConfig,
    seed_cache: RwLock<Option<CachedSeedResources>>,
}

impl Default for SurfaceCompiler {
    fn default() -> Self {
        Self {
            config: MachineConfig::default(),
            seed_cache: RwLock::new(None),
        }
    }
}

impl SurfaceCompiler {
    pub fn new(config: MachineConfig) -> Self {
        Self {
            config,
            seed_cache: RwLock::new(None),
        }
    }

    pub fn config(&self) -> &MachineConfig {
        &self.config
    }

    pub fn compile(
        &self,
        text: &str,
        scope: &ScopeKey,
        resolver_seed: &[ResolverEntitySeed],
    ) -> SurfaceCompileArtifacts {
        let scan = self.scan_parts(text, scope, resolver_seed);
        let structure = self.build_structure_parts(text, &scan);
        let surface = surface_from_artifacts(text, &scan, &structure);
        SurfaceCompileArtifacts {
            scan,
            structure,
            surface,
        }
    }

    pub fn scan_request(&self, request: &ScanRequest) -> SurfaceCompileArtifacts {
        self.compile(&request.text, &request.scope, &request.resolver_seed)
    }

    pub fn scan_parts(
        &self,
        text: &str,
        _scope: &ScopeKey,
        resolver_seed: &[ResolverEntitySeed],
    ) -> ScanArtifact {
        let tokenized = tokenize(text);
        let seed_resources = self.seed_resources(resolver_seed);
        let sentences = sentence_spans(text);
        let detected = detect_mentions_hot_path(
            text,
            &tokenized,
            &sentences,
            resolver_seed,
            &seed_resources,
            &self.config.extraction,
        );
        let mentions = detected.mentions;
        let resolver_links = build_resolver_links(&mentions, &detected.normalized_surfaces);
        let narrative_hits =
            discover_narrative_hits(&tokenized.tokens, &tokenized.normalized_tokens, &sentences);
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
                code: "PX_MACHINE_SCAN".to_owned(),
                message: format!(
                    "Machine scanned {} tokens, {} sentences, and {} mentions.",
                    count_token_words(&tokenized.tokens),
                    sentences.len(),
                    mentions.len()
                ),
            }],
            sentences,
            tokens: tokenized.tokens,
            mentions,
            chunks,
            resolver_links,
            narrative_hits,
        }
    }

    pub fn build_structure_parts(&self, text: &str, scan: &ScanArtifact) -> StructureArtifact {
        let sentence_mention_ranges =
            sentence_item_ranges(scan.sentences.len(), &scan.mentions, |mention| {
                mention.sentence_index
            });
        let sentence_chunk_ranges =
            sentence_item_ranges(scan.sentences.len(), &scan.chunks, |chunk| {
                chunk.sentence_index
            });
        let sentence_hit_ranges =
            sentence_item_ranges(scan.sentences.len(), &scan.narrative_hits, |hit| {
                hit.sentence_index
            });

        let mut sentence_frames = Vec::with_capacity(scan.sentences.len());
        let mut relations = Vec::new();
        let mut evidence_spans = Vec::new();

        for sentence in &scan.sentences {
            let index = sentence.index;
            let mention_slice =
                &scan.mentions[sentence_mention_ranges.get(index).cloned().unwrap_or(0..0)];
            let chunk_slice =
                &scan.chunks[sentence_chunk_ranges.get(index).cloned().unwrap_or(0..0)];
            let hit_slice =
                &scan.narrative_hits[sentence_hit_ranges.get(index).cloned().unwrap_or(0..0)];
            let mut diagnostics = Vec::new();
            let mut verb_frames = Vec::with_capacity(hit_slice.len());

            for hit in hit_slice {
                let subject = mention_slice
                    .iter()
                    .filter(|mention| mention.range.end <= hit.range.start)
                    .max_by_key(|mention| mention.range.end)
                    .map(frame_slot_from_mention);
                let (object_mention, recipient_mention) =
                    first_two_trailing_mentions(mention_slice, hit.range.end);
                let object = object_mention.map(frame_slot_from_mention);
                let recipient = recipient_mention.map(frame_slot_from_mention);
                let evidence = vec![EvidenceSpan {
                    document_id: None,
                    note_id: None,
                    label: slice_or_empty(text, sentence.range).trim().to_owned(),
                    kind: Some("sentence".to_owned()),
                    range: sentence.range,
                }];
                evidence_spans.extend(evidence.iter().cloned());
                let attachments = chunk_slice
                    .iter()
                    .filter(|chunk| chunk.range.start >= hit.range.end)
                    .take(1)
                    .map(|chunk| chunk.range)
                    .collect::<Vec<_>>();
                if subject.is_none() {
                    diagnostics.push(Diagnostic {
                        code: "PX_MACHINE_STRUCTURE_SUBJECT_GAP".to_owned(),
                        message: format!(
                            "Machine inferred relation '{}' without a clear subject.",
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
                    lemma: hit.lemma.clone(),
                    event_class: hit.event_class.clone(),
                    relation_type: hit.relation_type.clone(),
                    transitivity: hit.transitivity.clone(),
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
                mentions: mention_slice.to_vec(),
                chunks: chunk_slice.to_vec(),
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
                code: "PX_MACHINE_STRUCTURE".to_owned(),
                message: "Machine built sentence frames and relation candidates.".to_owned(),
            }],
        }
    }

    pub fn build_structure(&self, text: &str, scan: &ScanArtifact) -> StructureArtifact {
        self.build_structure_parts(text, scan)
    }

    pub fn compatibility_scan(&self, request: &ScanRequest) -> ScanArtifact {
        let mut scan = self.scan_parts(&request.text, &request.scope, &request.resolver_seed);
        scan.diagnostics = vec![Diagnostic {
            code: "PX_INVARANT_V2_SCAN".to_owned(),
            message: format!(
                "Invarant V2 scanned {} tokens, {} sentences, and {} mentions.",
                count_token_words(&scan.tokens),
                scan.sentences.len(),
                scan.mentions.len()
            ),
        }];
        scan
    }

    pub fn compatibility_structure(&self, request: &StructureRequest) -> StructureArtifact {
        let mut structure = self.build_structure_parts(&request.text, &request.scan);
        for frame in &mut structure.sentence_frames {
            for diagnostic in &mut frame.diagnostics {
                if diagnostic.code == "PX_MACHINE_STRUCTURE_SUBJECT_GAP" {
                    diagnostic.code = "PX_INVARANT_V2_STRUCTURE_SUBJECT_GAP".to_owned();
                    diagnostic.message = diagnostic
                        .message
                        .replace("Machine inferred", "Invarant V2 inferred");
                }
            }
        }
        structure.diagnostics = vec![Diagnostic {
            code: "PX_INVARANT_V2_STRUCTURE".to_owned(),
            message: "Invarant V2 built sentence frames and relation candidates.".to_owned(),
        }];
        structure
    }

    fn seed_resources(&self, resolver_seed: &[ResolverEntitySeed]) -> CachedSeedResources {
        let cache_key = seed_cache_key(resolver_seed);
        if let Some(cached) = self
            .seed_cache
            .read()
            .expect("machine seed cache poisoned")
            .as_ref()
            .filter(|cached| cached.cache_key == cache_key)
            .cloned()
        {
            return cached;
        }
        let built = CachedSeedResources {
            cache_key,
            gazetteer: build_seed_gazetteer(resolver_seed),
            rule_seed_sets: build_rule_seed_sets(resolver_seed),
        };
        *self
            .seed_cache
            .write()
            .expect("machine seed cache poisoned") = Some(built.clone());
        built
    }
}

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

#[derive(Clone, Debug, Default)]
struct TokenizedDocument {
    tokens: Vec<TokenSpan>,
    normalized_tokens: Vec<String>,
}

type SeedGazetteer = FxHashMap<String, SmallVec<[GazetteerEntry; 4]>>;

#[derive(Clone, Debug, Default)]
struct CachedRuleSeedSets {
    people: Vec<String>,
    organizations: Vec<String>,
    locations: Vec<String>,
}

#[derive(Clone, Debug)]
struct CachedSeedResources {
    cache_key: u64,
    gazetteer: SeedGazetteer,
    rule_seed_sets: CachedRuleSeedSets,
}

#[derive(Clone, Debug, Default)]
struct MentionDetectionResult {
    mentions: Vec<MentionSpan>,
    normalized_surfaces: Vec<String>,
}

#[derive(Clone, Debug)]
struct GazetteerEntry {
    token_forms: SmallVec<[String; 4]>,
    kind: Option<EntityKind>,
    entity_ref: Option<MentionEntityRef>,
}

fn surface_from_artifacts(
    text: &str,
    scan: &ScanArtifact,
    structure: &StructureArtifact,
) -> SurfaceDocument {
    let tokens = scan
        .tokens
        .iter()
        .map(|token| Token {
            range: SourceRange::from(token.range),
            surface: CompactString::from(slice_or_empty(text, token.range)),
            normalized: CompactString::from(normalize_token_surface(slice_or_empty(
                text,
                token.range,
            ))),
            class: token.token_class.clone(),
            pos: token.pos.clone(),
        })
        .collect::<Vec<_>>();
    let sentences = scan
        .sentences
        .iter()
        .map(|sentence| phoenix_types::Sentence {
            index: sentence.index,
            range: SourceRange::from(sentence.range),
        })
        .collect::<Vec<_>>();
    let clauses = structure
        .sentence_frames
        .iter()
        .flat_map(|frame| {
            frame.clause_ranges.iter().map(|range| Clause {
                sentence_index: frame.sentence.index,
                range: SourceRange::from(*range),
            })
        })
        .collect::<Vec<_>>();
    let phrases = scan
        .chunks
        .iter()
        .map(|chunk| PhraseNode {
            kind: match chunk.kind.clone().unwrap_or(ChunkKind::Clause) {
                ChunkKind::Np => PhraseKind::Np,
                ChunkKind::Vp => PhraseKind::Vp,
                ChunkKind::Pp => PhraseKind::Pp,
                ChunkKind::Clause => PhraseKind::Clause,
                ChunkKind::AdjP => PhraseKind::Ap,
            },
            range: SourceRange::from(chunk.range),
            head: Some(SourceRange::from(chunk.head)),
            modifiers: chunk
                .modifiers
                .iter()
                .copied()
                .map(SourceRange::from)
                .collect(),
            sentence_index: chunk.sentence_index,
        })
        .collect::<Vec<_>>();
    let attachments = structure
        .relations
        .iter()
        .flat_map(|relation| {
            relation.attachments.iter().map(|attachment| Attachment {
                source: SourceRange::from(relation.verb_range),
                target: SourceRange::from(*attachment),
                sentence_index: relation.sentence_index,
                label: CompactString::from(relation.relation_type.as_str()),
            })
        })
        .collect::<Vec<_>>();
    let units = scan
        .sentences
        .iter()
        .map(|sentence| SurfaceUnit {
            kind: SurfaceUnitKind::Sentence,
            key: None,
            range: SourceRange::from(sentence.range),
            sentence_index: sentence.index,
            chunk_id_hint: None,
        })
        .collect::<Vec<_>>();

    SurfaceDocument {
        tokens,
        sentences,
        clauses,
        quote_blocks: Vec::<QuoteBlock>::new(),
        speaker_cues: Vec::<SpeakerCue>::new(),
        phrases,
        attachments,
        units,
    }
}

fn tokenize(text: &str) -> TokenizedDocument {
    let mut tokens = Vec::new();
    let mut normalized_tokens = Vec::new();
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
            let normalized = normalize_token_surface(token);
            let pos = if is_pronoun(&normalized) {
                Some(PosTag::Pronoun)
            } else if is_verb_token(&normalized) {
                Some(PosTag::Verb)
            } else if token
                .chars()
                .next()
                .is_some_and(|value| value.is_uppercase())
            {
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
                capitalized: token
                    .chars()
                    .next()
                    .is_some_and(|value| value.is_uppercase()),
            });
            normalized_tokens.push(normalized);
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
            normalized_tokens.push(String::new());
        }
    }
    TokenizedDocument {
        tokens,
        normalized_tokens,
    }
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

fn build_seed_gazetteer(resolver_seed: &[ResolverEntitySeed]) -> SeedGazetteer {
    let mut by_first_token = FxHashMap::<String, SmallVec<[GazetteerEntry; 4]>>::default();
    for seed in resolver_seed {
        let forms = std::iter::once(seed.canonical_name.as_str())
            .chain(seed.aliases.iter().map(String::as_str));
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

fn build_rule_seed_sets(resolver_seed: &[ResolverEntitySeed]) -> CachedRuleSeedSets {
    let mut sets = CachedRuleSeedSets::default();
    for seed in resolver_seed {
        let target = match seed.kind.as_ref() {
            Some(EntityKind::Character) | Some(EntityKind::Npc) => &mut sets.people,
            Some(EntityKind::Organization) | Some(EntityKind::Faction) => &mut sets.organizations,
            Some(EntityKind::Location) => &mut sets.locations,
            _ => continue,
        };
        target.push(seed.canonical_name.clone());
        target.extend(seed.aliases.iter().cloned());
    }
    sets
}

fn seed_cache_key(resolver_seed: &[ResolverEntitySeed]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for seed in resolver_seed {
        seed.entity_id.0.hash(&mut hasher);
        seed.canonical_name.hash(&mut hasher);
        seed.aliases.hash(&mut hasher);
        match seed.kind.as_ref() {
            Some(kind) => std::mem::discriminant(kind).hash(&mut hasher),
            None => 0u8.hash(&mut hasher),
        }
    }
    hasher.finish()
}

fn seeded_gazetteer_mentions(
    text: &str,
    tokens: &[TokenSpan],
    normalized_tokens: &[String],
    sentences: &[SentenceSpan],
    gazetteer: &SeedGazetteer,
) -> Vec<DetectedMention> {
    if gazetteer.is_empty() {
        return Vec::new();
    }
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

fn build_rule_entities_from_sets(text: &str, seed_sets: &CachedRuleSeedSets) -> Vec<IeEntity> {
    if seed_sets.people.is_empty()
        && seed_sets.organizations.is_empty()
        && seed_sets.locations.is_empty()
        && text.len() > RULE_NER_MAX_BYTES_WITHOUT_SEEDS
    {
        return Vec::new();
    }
    let mut ner = RuleBasedNER::with_basic_knowledge();
    if !seed_sets.people.is_empty() {
        ner.add_person_names(seed_sets.people.clone());
    }
    if !seed_sets.organizations.is_empty() {
        ner.add_organizations(seed_sets.organizations.clone());
    }
    if !seed_sets.locations.is_empty() {
        ner.add_locations(seed_sets.locations.clone());
    }
    ner.extract_entities(text).unwrap_or_default()
}

fn scirs2_rule_mentions(
    text: &str,
    sentences: &[SentenceSpan],
    resolver_seed: &[ResolverEntitySeed],
    seed_sets: &CachedRuleSeedSets,
) -> Vec<DetectedMention> {
    let mut sentence_cursor = 0usize;
    build_rule_entities_from_sets(text, seed_sets)
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
    let mut sentence_cursor = 0usize;
    extract_entities(text, pattern_ner_config())
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
    normalized_tokens: &[String],
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
        let normalized_token = normalized_tokens
            .get(index)
            .map(String::as_str)
            .unwrap_or_default();
        let sentence_index = locate_sentence_cursor(sentences, &mut sentence_cursor, token.range);

        if matches!(token.pos, Some(PosTag::Pronoun)) {
            mentions.push(DetectedMention {
                range,
                surface: token_text.to_owned(),
                normalized: normalized_token.to_owned(),
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

fn extend_detected_mentions_without_overlaps(
    detected: &mut Vec<DetectedMention>,
    observed_ranges: &mut Vec<TextRange>,
    additions: Vec<DetectedMention>,
) {
    for mention in additions {
        if observed_ranges
            .iter()
            .any(|existing| range_overlaps(*existing, mention.range))
        {
            continue;
        }
        observed_ranges.push(mention.range);
        detected.push(mention);
    }
}

fn range_overlaps(left: TextRange, right: TextRange) -> bool {
    left.start < right.end && right.start < left.end
}

fn sentence_item_ranges<T, F>(
    sentence_count: usize,
    items: &[T],
    sentence_index: F,
) -> Vec<std::ops::Range<usize>>
where
    F: Fn(&T) -> usize,
{
    let mut ranges = Vec::with_capacity(sentence_count);
    let mut cursor = 0usize;
    for sentence_ix in 0..sentence_count {
        let start = cursor;
        while cursor < items.len() && sentence_index(&items[cursor]) == sentence_ix {
            cursor += 1;
        }
        ranges.push(start..cursor);
    }
    ranges
}

fn first_two_trailing_mentions<'a>(
    mentions: &'a [MentionSpan],
    after_end: u32,
) -> (Option<&'a MentionSpan>, Option<&'a MentionSpan>) {
    let mut first = None;
    let mut second = None;
    for mention in mentions {
        if mention.range.start < after_end {
            continue;
        }
        if first.is_none() {
            first = Some(mention);
        } else {
            second = Some(mention);
            break;
        }
    }
    (first, second)
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
    tokenized: &TokenizedDocument,
    sentences: &[SentenceSpan],
    resolver_seed: &[ResolverEntitySeed],
    seed_resources: &CachedSeedResources,
    config: &MachineExtractionConfig,
) -> MentionDetectionResult {
    let mut detected = seeded_gazetteer_mentions(
        text,
        &tokenized.tokens,
        &tokenized.normalized_tokens,
        sentences,
        &seed_resources.gazetteer,
    );
    let mut observed_ranges = detected
        .iter()
        .map(|mention| mention.range)
        .collect::<Vec<_>>();

    if config.enable_scirs2_rule_ner {
        extend_detected_mentions_without_overlaps(
            &mut detected,
            &mut observed_ranges,
            scirs2_rule_mentions(
                text,
                sentences,
                resolver_seed,
                &seed_resources.rule_seed_sets,
            ),
        );
    }
    if config.enable_scirs2_pattern_ner {
        extend_detected_mentions_without_overlaps(
            &mut detected,
            &mut observed_ranges,
            scirs2_pattern_mentions(text, sentences),
        );
    }
    if config.enable_native_refinement {
        let observed_snapshot = observed_ranges.clone();
        extend_detected_mentions_without_overlaps(
            &mut detected,
            &mut observed_ranges,
            native_refinement_mentions(
                text,
                &tokenized.tokens,
                &tokenized.normalized_tokens,
                sentences,
                &observed_snapshot,
                resolver_seed,
            ),
        );
    }

    let deduped = dedupe_detected_mentions(detected);
    let normalized_surfaces = deduped
        .iter()
        .map(|mention| mention.normalized.clone())
        .collect::<Vec<_>>();
    let mentions = deduped
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
        .collect();
    MentionDetectionResult {
        mentions,
        normalized_surfaces,
    }
}

fn detect_mentions_hot_path(
    text: &str,
    tokenized: &TokenizedDocument,
    sentences: &[SentenceSpan],
    resolver_seed: &[ResolverEntitySeed],
    seed_resources: &CachedSeedResources,
    config: &MachineExtractionConfig,
) -> MentionDetectionResult {
    let mut hot_config = config.clone();
    if text.len() > HOT_PATH_PATTERN_NER_MAX_BYTES {
        hot_config.enable_scirs2_pattern_ner = false;
    }
    detect_mentions(
        text,
        tokenized,
        sentences,
        resolver_seed,
        seed_resources,
        &hot_config,
    )
}

fn build_resolver_links(
    mentions: &[MentionSpan],
    normalized_surfaces: &[String],
) -> Vec<ResolverLink> {
    let mut links = Vec::new();
    let mut last_entity_by_surface = FxHashMap::<&str, usize>::default();
    let mut antecedent = None::<usize>;
    for (index, mention) in mentions.iter().enumerate() {
        let normalized = normalized_surfaces
            .get(index)
            .map(String::as_str)
            .unwrap_or_default();
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
    tokens: &[TokenSpan],
    normalized_tokens: &[String],
    sentences: &[SentenceSpan],
) -> Vec<NarrativeVerbHit> {
    let mut hits = Vec::new();
    let mut sentence_cursor = 0usize;
    for (index, token) in tokens.iter().enumerate() {
        if !matches!(token.pos, Some(PosTag::Verb)) {
            continue;
        }
        let normalized = normalized_tokens
            .get(index)
            .map(String::as_str)
            .unwrap_or_default();
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

fn normalize_surface(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    let mut pending_space = false;
    for ch in value.chars() {
        if ch.is_alphanumeric() {
            if pending_space && !normalized.is_empty() {
                normalized.push(' ');
            }
            pending_space = false;
            for lowered in ch.to_lowercase() {
                normalized.push(lowered);
            }
        } else if ch.is_whitespace() {
            pending_space = true;
        }
    }
    normalized
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

fn is_pronoun(value: &str) -> bool {
    matches!(
        value,
        "he" | "she" | "they" | "them" | "him" | "her" | "we" | "us" | "i" | "you" | "it"
    )
}

fn is_verb_token(value: &str) -> bool {
    matches!(
        value,
        "attack"
            | "attacked"
            | "attacks"
            | "met"
            | "meet"
            | "meets"
            | "rose"
            | "rise"
            | "rises"
            | "woke"
            | "wake"
            | "wakes"
            | "wrote"
            | "write"
            | "writes"
            | "mapped"
            | "map"
            | "maps"
            | "gave"
            | "give"
            | "gives"
            | "waited"
            | "wait"
            | "waits"
            | "saw"
            | "see"
            | "sees"
            | "found"
            | "find"
            | "finds"
            | "fought"
            | "fight"
            | "fights"
            | "moved"
            | "move"
            | "moves"
            | "crossed"
            | "cross"
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
            value
                .trim_end_matches("ed")
                .trim_end_matches('s')
                .to_owned(),
            "creation".to_owned(),
            "writes".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
        other => (
            other
                .trim_end_matches("ed")
                .trim_end_matches('s')
                .to_owned(),
            "action".to_owned(),
            "relates_to".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
    }
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

fn pattern_ner_config() -> &'static NerPatternConfig {
    static CONFIG: OnceLock<NerPatternConfig> = OnceLock::new();
    CONFIG.get_or_init(|| {
        let mut config = NerPatternConfig::none();
        config.heuristic_entities = true;
        config
    })
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
