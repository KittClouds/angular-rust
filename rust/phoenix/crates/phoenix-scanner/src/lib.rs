use std::cell::RefCell;
use std::collections::{hash_map::DefaultHasher, BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};

use fst::{Map, MapBuilder};
use phoenix_alex::{
    is_stop_word_with_profile, normalize_raw, normalized_has_meaningful_token,
    split_sentence_ranges, strip_possessive, Lexicon,
};
use phoenix_types::{
    ChunkKind, ChunkSpan, DiscoveryThresholds, EntityKind, FuzzyMode, GenderHint, KnownMatch,
    KnownMatchSource, LexiconEntry, MentionEntityRef, MentionSource, MentionSpan, NarrativeRule,
    NarrativeTransitivity, NarrativeVerbHit, PosTag, ResolverEntitySeed, ResolverLink,
    ResolverLinkKind, ScanArtifact, ScanRequest, ScannerConfig, ScopeKey, SentenceSpan, SessionId,
    TextRange, TokenClass, TokenSpan,
};

pub struct PhoenixScanner {
    config: ScannerConfig,
    sessions: RefCell<HashMap<String, ScannerSessionState>>,
    narrative: NarrativeMatcher,
}

impl PhoenixScanner {
    pub fn new(config: ScannerConfig) -> Self {
        let narrative = NarrativeMatcher::new(&config.narrative_overlay)
            .expect("narrative dictionary should build");
        Self {
            config,
            sessions: RefCell::new(HashMap::new()),
            narrative,
        }
    }

    pub fn scan(&self, request: &ScanRequest) -> ScanArtifact {
        self.scan_parts(
            &request.text,
            &request.scope,
            request.session_id.as_ref(),
            &request.resolver_seed,
        )
    }

    pub fn scan_parts(
        &self,
        text: &str,
        scope: &ScopeKey,
        session_id: Option<&SessionId>,
        resolver_seed: &[ResolverEntitySeed],
    ) -> ScanArtifact {
        if let Some(session_id) = session_id {
            let mut sessions = self.sessions.borrow_mut();
            let session = sessions.entry(session_id.0.clone()).or_default();
            self.scan_with_session(session, text, scope, resolver_seed)
        } else {
            let mut session = ScannerSessionState::default();
            self.scan_with_session(&mut session, text, scope, resolver_seed)
        }
    }

    fn scan_with_session(
        &self,
        session: &mut ScannerSessionState,
        text: &str,
        scope: &ScopeKey,
        resolver_seed: &[ResolverEntitySeed],
    ) -> ScanArtifact {
        let ScannerSessionState {
            seed_hash,
            lexicon,
            resolver,
            discovery,
        } = session;
        resolver.seed(resolver_seed);
        ensure_lexicon(seed_hash, lexicon, resolver_seed);
        let lexicon = lexicon
            .as_ref()
            .expect("lexicon should exist after ensure_lexicon");

        let sentences = split_sentences(text);
        let exact_candidates = lexicon
            .scan(text, scope)
            .into_iter()
            .map(known_match_to_candidate)
            .collect::<Vec<_>>();

        let base_tokens = tokenize(text, &[]);
        let fuzzy_candidates = if self.config.fuzzy_mode == FuzzyMode::Off {
            Vec::new()
        } else {
            build_fuzzy_candidates(
                text,
                &base_tokens,
                scope,
                lexicon,
                &self.config.stopword_profile,
            )
        };

        let mut mention_candidates = exact_candidates;
        mention_candidates.extend(fuzzy_candidates);
        let exact_and_fuzzy = select_mentions(mention_candidates, &sentences);

        let first_pass = build_pass_artifacts(text, &sentences, &exact_and_fuzzy, &self.narrative);
        let discovery_mentions = build_discovery_mentions(
            text,
            &sentences,
            &first_pass.tokens,
            &first_pass.chunks,
            &first_pass.narrative_hits,
            &exact_and_fuzzy,
            scope,
            &self.config.discovery_thresholds,
            &self.config.stopword_profile,
            discovery,
            lexicon,
        );

        let mut final_candidates = exact_and_fuzzy
            .iter()
            .cloned()
            .map(mention_to_candidate)
            .collect::<Vec<_>>();
        final_candidates.extend(discovery_mentions.into_iter().map(mention_to_candidate));
        let mentions = select_mentions(final_candidates, &sentences);
        let final_pass = build_pass_artifacts(text, &sentences, &mentions, &self.narrative);
        let resolver_links =
            resolve_links(text, &final_pass.tokens, &mentions, &sentences, resolver);

        ScanArtifact {
            sentences,
            tokens: final_pass.tokens,
            mentions,
            chunks: final_pass.chunks,
            resolver_links,
            narrative_hits: final_pass.narrative_hits,
            diagnostics: Vec::new(),
        }
    }
}

impl Default for PhoenixScanner {
    fn default() -> Self {
        Self::new(ScannerConfig::default())
    }
}

#[derive(Default)]
struct ScannerSessionState {
    seed_hash: Option<u64>,
    lexicon: Option<Lexicon>,
    resolver: ResolverState,
    discovery: HashMap<String, DiscoveryLedger>,
}

fn ensure_lexicon(
    seed_hash_slot: &mut Option<u64>,
    lexicon_slot: &mut Option<Lexicon>,
    resolver_seed: &[ResolverEntitySeed],
) {
    let seed_hash = hash_seed(resolver_seed);
    let needs_rebuild = *seed_hash_slot != Some(seed_hash);
    if needs_rebuild {
        let entries = resolver_seed
            .iter()
            .map(seed_to_lexicon_entry)
            .collect::<Vec<_>>();
        *lexicon_slot = Some(Lexicon::from_entries(&entries).expect("lexicon from seed"));
        *seed_hash_slot = Some(seed_hash);
    }
    if lexicon_slot.is_none() {
        *lexicon_slot =
            Some(Lexicon::from_entries(&[]).expect("empty lexicon should always build"));
    }
}

fn hash_seed(seed: &[ResolverEntitySeed]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for entry in seed {
        entry.entity_id.0.hash(&mut hasher);
        entry.canonical_name.hash(&mut hasher);
        entry.aliases.hash(&mut hasher);
        entry.scope.world_id.hash(&mut hasher);
        entry.scope.narrative_id.hash(&mut hasher);
    }
    hasher.finish()
}

fn seed_to_lexicon_entry(seed: &ResolverEntitySeed) -> LexiconEntry {
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

#[derive(Clone)]
struct MentionCandidate {
    range: TextRange,
    surface: String,
    kind: Option<EntityKind>,
    entity_ref: Option<MentionEntityRef>,
    source: MentionSource,
    confidence: f32,
    priority: u8,
}

fn known_match_to_candidate(matched: KnownMatch) -> MentionCandidate {
    let surface = matched.surface.clone();
    let source = match matched.source {
        Some(KnownMatchSource::ExactAlias | KnownMatchSource::ExactAutoAlias) => {
            MentionSource::Alias
        }
        Some(KnownMatchSource::FuzzyAnchor) => MentionSource::Fuzzy,
        _ => MentionSource::Known,
    };
    let entity_ref = if matched.entries.len() == 1 {
        Some(MentionEntityRef::Known(
            matched.entries[0].entity_id.clone(),
        ))
    } else {
        None
    };
    MentionCandidate {
        range: matched.range,
        surface,
        kind: matched.entries.first().and_then(|entry| entry.kind.clone()),
        entity_ref,
        source: source.clone(),
        confidence: matched.confidence,
        priority: match source {
            MentionSource::Known | MentionSource::Alias => {
                if token_count(&matched.surface) > 1 {
                    4
                } else {
                    3
                }
            }
            MentionSource::Fuzzy => 2,
            MentionSource::Discovery => 1,
        },
    }
}

fn mention_to_candidate(mention: MentionSpan) -> MentionCandidate {
    MentionCandidate {
        range: mention.range,
        surface: mention.surface,
        kind: mention.kind,
        entity_ref: mention.entity_ref,
        source: mention.source.unwrap_or(MentionSource::Discovery),
        confidence: mention.confidence,
        priority: 1,
    }
}

#[derive(Clone)]
struct ScanToken {
    span: TokenSpan,
}

#[derive(Default)]
struct PassArtifacts {
    tokens: Vec<TokenSpan>,
    chunks: Vec<ChunkSpan>,
    narrative_hits: Vec<NarrativeVerbHit>,
}

fn build_pass_artifacts(
    text: &str,
    sentences: &[SentenceSpan],
    mentions: &[MentionSpan],
    narrative: &NarrativeMatcher,
) -> PassArtifacts {
    let tokens = tokenize(text, mentions);
    let chunks = chunk(text, &tokens, sentences, narrative);
    let narrative_hits = extract_narrative_hits(text, &chunks, sentences, narrative);
    PassArtifacts {
        tokens: tokens.into_iter().map(|token| token.span).collect(),
        chunks,
        narrative_hits,
    }
}

fn split_sentences(text: &str) -> Vec<SentenceSpan> {
    let mut sentences = split_sentence_ranges(text)
        .into_iter()
        .enumerate()
        .map(|(index, (start, end))| SentenceSpan {
            index,
            range: TextRange {
                start: start as u32,
                end: end as u32,
            },
        })
        .collect::<Vec<_>>();

    if sentences.is_empty() {
        sentences.push(SentenceSpan {
            index: 0,
            range: TextRange {
                start: 0,
                end: text.len() as u32,
            },
        });
    }

    sentences
}

fn tokenize(text: &str, mentions: &[MentionSpan]) -> Vec<ScanToken> {
    let mut tokens = Vec::new();
    let masks = mentions
        .iter()
        .map(|mention| mention.range)
        .collect::<Vec<_>>();
    let mut index = 0usize;

    while index < text.len() {
        if let Some(mask) = mask_at(index, &masks) {
            let range = *mask;
            let surface = slice(text, range);
            tokens.push(ScanToken {
                span: TokenSpan {
                    range,
                    token_class: Some(TokenClass::Word),
                    pos: Some(PosTag::ProperNoun),
                    masked: true,
                    capitalized: surface
                        .chars()
                        .next()
                        .map(|ch| ch.is_uppercase())
                        .unwrap_or(false),
                },
            });
            index = range.end as usize;
            continue;
        }

        let Some(ch) = text[index..].chars().next() else {
            break;
        };
        if ch.is_whitespace() {
            index += ch.len_utf8();
            continue;
        }

        let start = index;
        if ch.is_ascii_alphanumeric() || ch == '\'' || ch == '-' {
            index += ch.len_utf8();
            while index < text.len() {
                let Some(next) = text[index..].chars().next() else {
                    break;
                };
                let can_extend = next.is_ascii_alphanumeric()
                    || next == '\''
                    || next == '-'
                    || (next == '.'
                        && text[index + next.len_utf8()..]
                            .chars()
                            .next()
                            .map(|following| following.is_ascii_alphanumeric())
                            .unwrap_or(false));
                if can_extend {
                    index += next.len_utf8();
                } else {
                    break;
                }
            }
        } else {
            index += ch.len_utf8();
        }

        let range = TextRange {
            start: start as u32,
            end: index as u32,
        };
        let surface = slice(text, range);
        let token_class = if surface.chars().all(|c| c.is_ascii_digit()) {
            TokenClass::Number
        } else if surface
            .chars()
            .all(|c| c.is_ascii_punctuation() && c != '\'' && c != '-')
        {
            TokenClass::Punctuation
        } else if surface
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '\'' || c == '-')
        {
            TokenClass::Word
        } else {
            TokenClass::Symbol
        };
        tokens.push(ScanToken {
            span: TokenSpan {
                range,
                token_class: Some(token_class),
                pos: Some(guess_pos(surface, false)),
                masked: false,
                capitalized: surface
                    .chars()
                    .next()
                    .map(|ch| ch.is_uppercase())
                    .unwrap_or(false),
            },
        });
    }

    retag_with_context(text, &mut tokens);
    tokens
}

fn retag_with_context(text: &str, tokens: &mut [ScanToken]) {
    for index in 0..tokens.len() {
        if tokens[index].span.masked {
            continue;
        }
        let current = tokens[index].span.pos.clone().unwrap_or(PosTag::Other);
        let previous = index
            .checked_sub(1)
            .and_then(|value| tokens.get(value))
            .and_then(|token| token.span.pos.clone())
            .unwrap_or(PosTag::Other);

        if matches!(previous, PosTag::Determiner | PosTag::Adjective) && is_verbal(&current) {
            tokens[index].span.pos = Some(PosTag::Noun);
        }
        if previous == PosTag::Modal && is_nominal(&current) {
            tokens[index].span.pos = Some(PosTag::Verb);
        }
        if current == PosTag::Other {
            let surface = slice(text, tokens[index].span.range);
            if surface.ends_with("ly") {
                tokens[index].span.pos = Some(PosTag::Adverb);
            }
        }
    }
}

fn guess_pos(token: &str, masked: bool) -> PosTag {
    if masked {
        return PosTag::ProperNoun;
    }
    let lower = token.to_ascii_lowercase();
    if lower.is_empty() {
        return PosTag::Other;
    }
    if [".", ",", "!", "?", "(", ")", ":", ";"].contains(&token) {
        return PosTag::Punctuation;
    }
    if matches!(
        lower.as_str(),
        "he" | "him"
            | "his"
            | "she"
            | "her"
            | "hers"
            | "it"
            | "its"
            | "they"
            | "them"
            | "their"
            | "we"
            | "us"
            | "i"
            | "me"
            | "you"
    ) {
        return PosTag::Pronoun;
    }
    if matches!(lower.as_str(), "who" | "that" | "which" | "whom") {
        return PosTag::RelativePronoun;
    }
    if matches!(
        lower.as_str(),
        "the" | "a" | "an" | "this" | "that" | "these" | "those"
    ) {
        return PosTag::Determiner;
    }
    if matches!(
        lower.as_str(),
        "in" | "on"
            | "at"
            | "to"
            | "from"
            | "with"
            | "into"
            | "over"
            | "under"
            | "around"
            | "through"
            | "after"
            | "before"
            | "for"
            | "of"
            | "by"
    ) {
        return PosTag::Preposition;
    }
    if matches!(lower.as_str(), "and" | "or" | "but" | "nor" | "yet") {
        return PosTag::Conjunction;
    }
    if matches!(
        lower.as_str(),
        "can" | "could" | "should" | "would" | "may" | "might" | "will"
    ) {
        return PosTag::Modal;
    }
    if matches!(
        lower.as_str(),
        "is" | "are" | "was" | "were" | "be" | "been" | "being" | "have" | "has" | "had"
    ) {
        return PosTag::Auxiliary;
    }
    if lower.ends_with("ly") {
        return PosTag::Adverb;
    }
    if [
        "ous", "ful", "ive", "al", "less", "able", "ic", "ish", "ant", "ent",
    ]
    .iter()
    .any(|suffix| lower.ends_with(suffix))
    {
        return PosTag::Adjective;
    }
    if lower.ends_with("ed") || lower.ends_with("ing") {
        return PosTag::Verb;
    }
    if token
        .chars()
        .next()
        .map(|ch| ch.is_uppercase())
        .unwrap_or(false)
    {
        return PosTag::ProperNoun;
    }
    PosTag::Noun
}

fn build_fuzzy_candidates(
    text: &str,
    tokens: &[ScanToken],
    scope: &ScopeKey,
    lexicon: &Lexicon,
    stopword_profile: &str,
) -> Vec<MentionCandidate> {
    let mut candidates = Vec::new();
    for token in tokens {
        if token.span.masked
            || token.span.token_class != Some(TokenClass::Word)
            || !token.span.capitalized
        {
            continue;
        }
        let surface = slice(text, token.span.range);
        let normalized = normalize_raw(surface);
        if normalized.is_empty()
            || !normalized_has_meaningful_token(&normalized, stopword_profile)
            || is_stop_word_with_profile(strip_possessive(&normalized), stopword_profile)
        {
            continue;
        }
        if let Some(mut matched) = lexicon.fuzzy_anchor(surface, scope) {
            matched.range = token.span.range;
            candidates.push(known_match_to_candidate(matched));
        }
    }
    candidates
}

fn chunk(
    text: &str,
    tokens: &[ScanToken],
    sentences: &[SentenceSpan],
    narrative: &NarrativeMatcher,
) -> Vec<ChunkSpan> {
    let mut chunks = Vec::new();
    let mut index = 0usize;
    while index < tokens.len() {
        if tokens[index].span.pos == Some(PosTag::Punctuation) {
            index += 1;
            continue;
        }
        if let Some((chunk, consumed)) = try_preposition_phrase(tokens, index, sentences) {
            chunks.push(chunk);
            index += consumed;
            continue;
        }
        if let Some((chunk, consumed)) = try_verb_phrase(text, tokens, index, sentences, narrative)
        {
            chunks.push(chunk);
            index += consumed;
            continue;
        }
        if let Some((chunk, consumed)) = try_noun_phrase(tokens, index, sentences) {
            chunks.push(chunk);
            index += consumed;
            continue;
        }
        if let Some((chunk, consumed)) = try_adj_phrase(tokens, index, sentences) {
            chunks.push(chunk);
            index += consumed;
            continue;
        }
        if let Some((chunk, consumed)) = try_clause(text, tokens, index, sentences, narrative) {
            chunks.push(chunk);
            index += consumed;
            continue;
        }
        index += 1;
    }
    chunks
}

fn try_noun_phrase(
    tokens: &[ScanToken],
    start: usize,
    sentences: &[SentenceSpan],
) -> Option<(ChunkSpan, usize)> {
    let mut index = start;
    let mut modifiers = Vec::new();
    if tokens.get(index)?.span.pos == Some(PosTag::Determiner) {
        modifiers.push(tokens[index].span.range);
        index += 1;
    }
    while index < tokens.len() && tokens[index].span.pos == Some(PosTag::Adjective) {
        modifiers.push(tokens[index].span.range);
        index += 1;
    }
    let noun_start = index;
    while index < tokens.len()
        && is_nominal(&tokens[index].span.pos.clone().unwrap_or(PosTag::Other))
    {
        index += 1;
    }
    if index > noun_start {
        let start_range = tokens[start].span.range;
        let end_range = tokens[index - 1].span.range;
        Some((
            ChunkSpan {
                kind: Some(ChunkKind::Np),
                range: TextRange {
                    start: start_range.start,
                    end: end_range.end,
                },
                head: end_range,
                modifiers,
                sentence_index: sentence_index_for(start_range.start as usize, sentences),
            },
            index - start,
        ))
    } else {
        None
    }
}

fn try_verb_phrase(
    text: &str,
    tokens: &[ScanToken],
    start: usize,
    sentences: &[SentenceSpan],
    narrative: &NarrativeMatcher,
) -> Option<(ChunkSpan, usize)> {
    let mut index = start;
    let mut modifiers = Vec::new();
    if matches!(
        tokens.get(index)?.span.pos,
        Some(PosTag::Auxiliary | PosTag::Modal)
    ) {
        modifiers.push(tokens[index].span.range);
        index += 1;
    }
    while index < tokens.len() && tokens[index].span.pos == Some(PosTag::Adverb) {
        modifiers.push(tokens[index].span.range);
        index += 1;
    }

    let mut head_index = None;
    if index < tokens.len()
        && matches!(
            tokens[index].span.pos,
            Some(PosTag::Verb | PosTag::Auxiliary | PosTag::Modal)
        )
    {
        head_index = Some(index);
        index += 1;
    } else if let Some(token) = tokens.get(index) {
        if narrative.lookup(slice(text, token.span.range)).is_some() {
            head_index = Some(index);
            index += 1;
        }
    }

    while index < tokens.len() && tokens[index].span.pos == Some(PosTag::Adverb) {
        modifiers.push(tokens[index].span.range);
        index += 1;
    }
    let head_index = head_index?;
    let head = tokens[head_index].span.range;
    let start_range = tokens[start].span.range;
    let end_range = tokens[index - 1].span.range;
    Some((
        ChunkSpan {
            kind: Some(ChunkKind::Vp),
            range: TextRange {
                start: start_range.start,
                end: end_range.end,
            },
            head,
            modifiers,
            sentence_index: sentence_index_for(start_range.start as usize, sentences),
        },
        index - start,
    ))
}

fn try_preposition_phrase(
    tokens: &[ScanToken],
    start: usize,
    sentences: &[SentenceSpan],
) -> Option<(ChunkSpan, usize)> {
    if tokens.get(start)?.span.pos != Some(PosTag::Preposition) {
        return None;
    }
    let (np, consumed) = try_noun_phrase(tokens, start + 1, sentences)?;
    let prep = tokens[start].span.range;
    Some((
        ChunkSpan {
            kind: Some(ChunkKind::Pp),
            range: TextRange {
                start: prep.start,
                end: np.range.end,
            },
            head: prep,
            modifiers: std::iter::once(np.head)
                .chain(np.modifiers.iter().copied())
                .collect(),
            sentence_index: sentence_index_for(prep.start as usize, sentences),
        },
        consumed + 1,
    ))
}

fn try_adj_phrase(
    tokens: &[ScanToken],
    start: usize,
    sentences: &[SentenceSpan],
) -> Option<(ChunkSpan, usize)> {
    let mut index = start;
    let mut modifiers = Vec::new();
    while index < tokens.len() && tokens[index].span.pos == Some(PosTag::Adverb) {
        modifiers.push(tokens[index].span.range);
        index += 1;
    }
    if index >= tokens.len()
        || tokens[index].span.pos != Some(PosTag::Adjective)
        || modifiers.is_empty()
    {
        return None;
    }
    let head = tokens[index].span.range;
    let start_range = tokens[start].span.range;
    Some((
        ChunkSpan {
            kind: Some(ChunkKind::AdjP),
            range: TextRange {
                start: start_range.start,
                end: head.end,
            },
            head,
            modifiers,
            sentence_index: sentence_index_for(start_range.start as usize, sentences),
        },
        index - start + 1,
    ))
}

fn try_clause(
    text: &str,
    tokens: &[ScanToken],
    start: usize,
    sentences: &[SentenceSpan],
    narrative: &NarrativeMatcher,
) -> Option<(ChunkSpan, usize)> {
    if tokens.get(start)?.span.pos != Some(PosTag::RelativePronoun) {
        return None;
    }
    let (vp, consumed) = try_verb_phrase(text, tokens, start + 1, sentences, narrative)?;
    let mut end = vp.range.end;
    let mut total = consumed + 1;
    if let Some((np, noun_consumed)) = try_noun_phrase(tokens, start + 1 + consumed, sentences) {
        end = np.range.end;
        total += noun_consumed;
    }
    Some((
        ChunkSpan {
            kind: Some(ChunkKind::Clause),
            range: TextRange {
                start: tokens[start].span.range.start,
                end,
            },
            head: vp.head,
            modifiers: vec![tokens[start].span.range],
            sentence_index: sentence_index_for(tokens[start].span.range.start as usize, sentences),
        },
        total,
    ))
}

fn extract_narrative_hits(
    text: &str,
    chunks: &[ChunkSpan],
    sentences: &[SentenceSpan],
    narrative: &NarrativeMatcher,
) -> Vec<NarrativeVerbHit> {
    chunks
        .iter()
        .filter(|chunk| chunk.kind == Some(ChunkKind::Vp))
        .filter_map(|chunk| {
            let surface = slice(text, chunk.head);
            let rule = narrative.lookup(surface)?;
            Some(NarrativeVerbHit {
                range: chunk.head,
                lemma: rule.lemma,
                event_class: rule.event_class,
                relation_type: rule.relation_type,
                transitivity: Some(rule.transitivity),
                sentence_index: sentence_index_for(chunk.range.start as usize, sentences),
                confidence: 0.9,
            })
        })
        .collect()
}

fn build_discovery_mentions(
    text: &str,
    sentences: &[SentenceSpan],
    tokens: &[TokenSpan],
    chunks: &[ChunkSpan],
    narrative_hits: &[NarrativeVerbHit],
    mentions: &[MentionSpan],
    scope: &ScopeKey,
    thresholds: &DiscoveryThresholds,
    stopword_profile: &str,
    ledger: &mut HashMap<String, DiscoveryLedger>,
    lexicon: &Lexicon,
) -> Vec<MentionSpan> {
    let mut emitted = Vec::new();
    let mention_ranges = mentions
        .iter()
        .map(|mention| mention.range)
        .collect::<Vec<_>>();
    let np_heads = chunks
        .iter()
        .filter(|chunk| chunk.kind == Some(ChunkKind::Np))
        .map(|chunk| chunk.head)
        .collect::<HashSet<_>>();
    let narrative_sentences = narrative_hits
        .iter()
        .map(|hit| hit.sentence_index)
        .collect::<HashSet<_>>();

    // Precompute: sentence-start byte offsets (first non-whitespace in each sentence)
    let sentence_starts: HashSet<u32> = sentences
        .iter()
        .map(|s| {
            let mut offset = s.range.start as usize;
            let bytes = text.as_bytes();
            while offset < bytes.len() && bytes[offset].is_ascii_whitespace() {
                offset += 1;
            }
            offset as u32
        })
        .collect();

    // Precompute: set of all lowercase word surfaces for the lowercase-alias check.
    // A single pass over tokens; only words that are NOT capitalized get inserted.
    let lowercase_surfaces: HashSet<String> = tokens
        .iter()
        .filter(|t| !t.capitalized && t.token_class == Some(TokenClass::Word) && !t.masked)
        .map(|t| normalize_raw(slice(text, t.range)))
        .filter(|n| !n.is_empty())
        .collect();

    let mut index = 0usize;
    while index < tokens.len() {
        if tokens[index].masked || overlaps_any(tokens[index].range, &mention_ranges) {
            index += 1;
            continue;
        }
        if !tokens[index].capitalized || tokens[index].token_class != Some(TokenClass::Word) {
            index += 1;
            continue;
        }

        // Stop-word prefix skip: if the first word is a stop-word, skip it so the next
        // capitalized word can be properly picked up as a standalone entity.
        let first_surface = slice(text, tokens[index].range);
        let first_normalized = normalize_raw(first_surface);
        if is_stop_word_with_profile(strip_possessive(&first_normalized), stopword_profile) {
            index += 1;
            continue;
        }

        let start = tokens[index].range.start;
        let mut end = tokens[index].range.end;
        let mut cursor = index + 1;
        while cursor < tokens.len()
            && !tokens[cursor].masked
            && tokens[cursor].capitalized
            && tokens[cursor].token_class == Some(TokenClass::Word)
        {
            end = tokens[cursor].range.end;
            cursor += 1;
        }

        // POS-tag gate: only for single-word candidates (multiword phrases are almost always entities)
        let is_single_word = cursor == index + 1;
        if is_single_word && !is_discovery_eligible_pos(&tokens[index].pos) {
            index = cursor;
            continue;
        }
        // Dialogue-lead hard skip: single word right after opening quote is never an entity
        if is_single_word && start > 0 {
            let prev = text.as_bytes()[start as usize - 1];
            if prev == b'"' || prev == 0x9C {
                index = cursor;
                continue;
            }
        }
        let range = TextRange { start, end };
        let surface = slice(text, range)
            .trim_matches(|ch: char| ch.is_ascii_punctuation())
            .to_owned();
        let normalized = normalize_raw(&surface);
        if normalized.is_empty()
            || !normalized_has_meaningful_token(&normalized, stopword_profile)
            || is_stop_word_with_profile(strip_possessive(&normalized), stopword_profile)
        {
            index = cursor;
            continue;
        }
        if !lexicon.lookup(&surface, scope).is_empty() {
            index = cursor;
            continue;
        }

        let sentence_index = sentence_index_for(range.start as usize, sentences);
        let entry = ledger.entry(normalized.clone()).or_default();
        entry.count += 1;
        entry.score += 1.0;
        if narrative_sentences.contains(&sentence_index) {
            entry.score += thresholds.narrative_bonus;
        }
        if np_heads.contains(&range) {
            entry.score += thresholds.np_head_bonus;
        }
        if surface
            .chars()
            .next()
            .map(|ch| ch.is_uppercase())
            .unwrap_or(false)
        {
            entry.score += thresholds.capitalized_bonus;
        }

        // Sentence-start penalty: single-word at beginning of sentence needs more evidence
        if is_single_word && sentence_starts.contains(&start) {
            entry.score -= thresholds.sentence_start_penalty;
        }

        // Lowercase-alias penalty: if the same word appears lowercase elsewhere, it's common
        if lowercase_surfaces.contains(&normalized) {
            entry.score -= thresholds.lowercase_alias_penalty;
        }

        entry.last_surface = surface.clone();

        if entry.count >= thresholds.min_occurrences && entry.score >= thresholds.min_score {
            emitted.push(MentionSpan {
                range,
                surface,
                kind: Some(EntityKind::Other),
                entity_ref: Some(MentionEntityRef::Speculative(normalized)),
                source: Some(MentionSource::Discovery),
                confidence: (entry.score / (entry.count as f32 + 1.0)).min(0.9),
                sentence_index,
            });
        }
        index = cursor;
    }
    emitted
}

/// Only Noun, ProperNoun, Other, or untagged tokens are eligible for discovery.
/// Adjectives, adverbs, verbs, auxiliaries, modals, conjunctions, prepositions,
/// determiners, and pronouns are rejected — they are never entity names.
#[inline]
fn is_discovery_eligible_pos(pos: &Option<PosTag>) -> bool {
    matches!(
        pos,
        None | Some(PosTag::Noun) | Some(PosTag::ProperNoun) | Some(PosTag::Other)
    )
}

fn select_mentions(
    candidates: Vec<MentionCandidate>,
    sentences: &[SentenceSpan],
) -> Vec<MentionSpan> {
    let mut candidates = candidates;
    candidates.sort_by(|left, right| {
        left.range
            .start
            .cmp(&right.range.start)
            .then(right.priority.cmp(&left.priority))
            .then((right.range.end - right.range.start).cmp(&(left.range.end - left.range.start)))
    });

    let mut selected = Vec::new();
    let mut occupied = Vec::<TextRange>::new();
    for candidate in candidates {
        if overlaps_any(candidate.range, &occupied) {
            continue;
        }
        occupied.push(candidate.range);
        selected.push(MentionSpan {
            range: candidate.range,
            surface: candidate.surface,
            kind: candidate.kind,
            entity_ref: candidate.entity_ref,
            source: Some(candidate.source),
            confidence: candidate.confidence,
            sentence_index: sentence_index_for(candidate.range.start as usize, sentences),
        });
    }
    selected
}

fn resolve_links(
    text: &str,
    tokens: &[TokenSpan],
    mentions: &[MentionSpan],
    sentences: &[SentenceSpan],
    resolver: &mut ResolverState,
) -> Vec<ResolverLink> {
    let mut links = Vec::new();
    let mention_by_start = mentions
        .iter()
        .cloned()
        .map(|mention| (mention.range.start, mention))
        .collect::<BTreeMap<_, _>>();

    for token in tokens {
        if let Some(mention) = mention_by_start.get(&token.range.start) {
            resolver.observe_mention(mention);
            continue;
        }
        if token.pos != Some(PosTag::Pronoun) {
            continue;
        }
        let surface = slice(text, token.range);
        if let Some(entity_ref) = resolver.resolve_pronoun(surface) {
            links.push(ResolverLink {
                source_range: token.range,
                target_range: resolver.last_range(entity_ref.clone()),
                target_entity: Some(entity_ref),
                link_kind: Some(ResolverLinkKind::Pronoun),
                confidence: 0.88,
                sentence_index: sentence_index_for(token.range.start as usize, sentences),
            });
        }
    }

    for sentence in sentences {
        let local_mentions = mentions
            .iter()
            .filter(|mention| contains(sentence.range, mention.range))
            .cloned()
            .collect::<Vec<_>>();
        links.extend(alias_links_for_sentence(
            text,
            &local_mentions,
            sentence.index,
        ));
    }

    links
}

fn alias_links_for_sentence(
    text: &str,
    mentions: &[MentionSpan],
    sentence_index: usize,
) -> Vec<ResolverLink> {
    let mut links = Vec::new();
    for left_index in 0..mentions.len() {
        for right_index in left_index + 1..mentions.len() {
            let left = &mentions[left_index];
            let right = &mentions[right_index];
            if right.range.start.saturating_sub(left.range.end) > 48 {
                break;
            }
            let between = slice(
                text,
                TextRange {
                    start: left.range.end,
                    end: right.range.start,
                },
            )
            .to_lowercase();
            let pattern_hit = between.contains("aka")
                || between.contains("also known as")
                || between.contains("called")
                || between.trim() == ","
                || between.contains('(');
            if !pattern_hit {
                continue;
            }
            let left_known = left.entity_ref.clone();
            let right_known = right.entity_ref.clone();
            let (source, target_range, target_entity) =
                if matches!(left_known, Some(MentionEntityRef::Known(_))) {
                    (right.range, Some(left.range), left_known)
                } else if matches!(right_known, Some(MentionEntityRef::Known(_))) {
                    (left.range, Some(right.range), right_known)
                } else {
                    (
                        right.range,
                        Some(left.range),
                        left.entity_ref.clone().or(right.entity_ref.clone()),
                    )
                };
            if target_entity.is_some() {
                links.push(ResolverLink {
                    source_range: source,
                    target_range,
                    target_entity,
                    link_kind: Some(ResolverLinkKind::AliasCandidate),
                    confidence: 0.91,
                    sentence_index,
                });
            }
        }
    }
    links
}

#[derive(Default)]
struct DiscoveryLedger {
    count: u32,
    score: f32,
    last_surface: String,
}

#[derive(Default)]
struct ResolverState {
    registry: HashMap<String, ResolverEntity>,
    history: Vec<String>,
}

impl ResolverState {
    fn seed(&mut self, seeds: &[ResolverEntitySeed]) {
        for seed in seeds {
            self.registry.insert(
                seed.entity_id.0.clone(),
                ResolverEntity {
                    entity_ref: MentionEntityRef::Known(seed.entity_id.clone()),
                    names: std::iter::once(seed.canonical_name.clone())
                        .chain(seed.aliases.iter().cloned())
                        .collect(),
                    gender: seed.gender.clone().unwrap_or(GenderHint::Unknown),
                    last_range: None,
                },
            );
        }
    }

    fn observe_mention(&mut self, mention: &MentionSpan) {
        let key = resolver_key(mention.entity_ref.clone(), &mention.surface);
        let entity = self
            .registry
            .entry(key.clone())
            .or_insert_with(|| ResolverEntity {
                entity_ref: mention.entity_ref.clone().unwrap_or_else(|| {
                    MentionEntityRef::Speculative(normalize_raw(&mention.surface))
                }),
                names: vec![mention.surface.clone()],
                gender: GenderHint::Unknown,
                last_range: None,
            });
        if !entity.names.contains(&mention.surface) {
            entity.names.push(mention.surface.clone());
        }
        entity.last_range = Some(mention.range);
        self.history.retain(|existing| existing != &key);
        self.history.insert(0, key);
        self.history.truncate(12);
    }

    fn resolve_pronoun(&self, pronoun: &str) -> Option<MentionEntityRef> {
        let desired = pronoun_gender(pronoun);
        for key in &self.history {
            let entity = self.registry.get(key)?;
            if genders_compatible(&entity.gender, &desired) {
                return Some(entity.entity_ref.clone());
            }
        }
        None
    }

    fn last_range(&self, entity_ref: MentionEntityRef) -> Option<TextRange> {
        self.registry
            .values()
            .find(|entity| entity.entity_ref == entity_ref)
            .and_then(|entity| entity.last_range)
    }
}

#[derive(Clone)]
struct ResolverEntity {
    entity_ref: MentionEntityRef,
    names: Vec<String>,
    gender: GenderHint,
    last_range: Option<TextRange>,
}

fn resolver_key(entity_ref: Option<MentionEntityRef>, surface: &str) -> String {
    match entity_ref {
        Some(MentionEntityRef::Known(entity_id)) => entity_id.0,
        Some(MentionEntityRef::Speculative(key)) => format!("spec:{key}"),
        None => format!("surf:{}", normalize_raw(surface)),
    }
}

fn pronoun_gender(pronoun: &str) -> GenderHint {
    match normalize_raw(pronoun).as_str() {
        "he" | "him" | "his" => GenderHint::Male,
        "she" | "her" | "hers" => GenderHint::Female,
        "they" | "them" | "their" => GenderHint::Plural,
        "it" | "its" => GenderHint::Neutral,
        _ => GenderHint::Unknown,
    }
}

fn genders_compatible(entity: &GenderHint, desired: &GenderHint) -> bool {
    entity == desired
        || matches!(entity, GenderHint::Unknown)
        || matches!(desired, GenderHint::Unknown)
}

#[derive(Clone)]
struct NarrativeRuleCompiled {
    lemma: String,
    event_class: String,
    relation_type: String,
    transitivity: NarrativeTransitivity,
}

struct NarrativeMatcher {
    map: Map<Vec<u8>>,
    rules: Vec<NarrativeRuleCompiled>,
}

impl NarrativeMatcher {
    fn new(overlay: &[NarrativeRule]) -> Result<Self, String> {
        let mut rules = default_narrative_rules();
        rules.extend(overlay.iter().cloned().map(|rule| NarrativeRuleCompiled {
            lemma: stem(&rule.lemma),
            event_class: rule.event_class,
            relation_type: rule.relation_type,
            transitivity: rule.transitivity,
        }));
        rules.sort_by(|left, right| left.lemma.cmp(&right.lemma));
        rules.dedup_by(|left, right| left.lemma == right.lemma);

        let mut builder = MapBuilder::memory();
        for (index, rule) in rules.iter().enumerate() {
            builder
                .insert(rule.lemma.as_str(), index as u64)
                .map_err(|_| "narrative fst insert failed".to_owned())?;
        }
        let bytes = builder
            .into_inner()
            .map_err(|_| "narrative fst finish failed".to_owned())?;
        let map = Map::new(bytes).map_err(|_| "narrative fst load failed".to_owned())?;
        Ok(Self { map, rules })
    }

    fn lookup(&self, surface: &str) -> Option<NarrativeRuleCompiled> {
        let lemma = stem(surface);
        let index = self.map.get(lemma.as_str())? as usize;
        self.rules.get(index).cloned()
    }
}

fn default_narrative_rules() -> Vec<NarrativeRuleCompiled> {
    [
        (
            "attack",
            "battle",
            "attacks",
            NarrativeTransitivity::Transitive,
        ),
        (
            "fight",
            "battle",
            "fights",
            NarrativeTransitivity::Transitive,
        ),
        (
            "fought",
            "battle",
            "fights",
            NarrativeTransitivity::Transitive,
        ),
        (
            "parked",
            "travel",
            "arrives",
            NarrativeTransitivity::Intransitive,
        ),
        (
            "stared",
            "discovery",
            "observes",
            NarrativeTransitivity::Transitive,
        ),
        (
            "stopped",
            "prevent",
            "prevents",
            NarrativeTransitivity::Transitive,
        ),
        (
            "pulled",
            "acquire",
            "takes",
            NarrativeTransitivity::Transitive,
        ),
        (
            "admitted",
            "dialogue",
            "mentions",
            NarrativeTransitivity::Transitive,
        ),
        (
            "memorized",
            "discovery",
            "discovers",
            NarrativeTransitivity::Transitive,
        ),
        (
            "studied",
            "discovery",
            "observes",
            NarrativeTransitivity::Transitive,
        ),
        (
            "chatted",
            "dialogue",
            "speaksTo",
            NarrativeTransitivity::Intransitive,
        ),
        (
            "wanted",
            "desire",
            "wants",
            NarrativeTransitivity::Transitive,
        ),
        (
            "ignored",
            "dialogue",
            "ignores",
            NarrativeTransitivity::Transitive,
        ),
        (
            "escaped",
            "travel",
            "departs",
            NarrativeTransitivity::Intransitive,
        ),
        (
            "arrive",
            "travel",
            "arrives",
            NarrativeTransitivity::Intransitive,
        ),
        (
            "approach",
            "travel",
            "arrives",
            NarrativeTransitivity::Intransitive,
        ),
        (
            "discover",
            "discovery",
            "discovers",
            NarrativeTransitivity::Transitive,
        ),
        (
            "find",
            "discovery",
            "finds",
            NarrativeTransitivity::Transitive,
        ),
        (
            "see",
            "discovery",
            "observes",
            NarrativeTransitivity::Transitive,
        ),
        (
            "saw",
            "discovery",
            "observes",
            NarrativeTransitivity::Transitive,
        ),
        (
            "say",
            "dialogue",
            "speaksTo",
            NarrativeTransitivity::Ditransitive,
        ),
        (
            "said",
            "dialogue",
            "speaksTo",
            NarrativeTransitivity::Ditransitive,
        ),
        (
            "speak",
            "dialogue",
            "speaksTo",
            NarrativeTransitivity::Intransitive,
        ),
        (
            "spoke",
            "dialogue",
            "speaksTo",
            NarrativeTransitivity::Intransitive,
        ),
        (
            "tell",
            "dialogue",
            "speaksTo",
            NarrativeTransitivity::Ditransitive,
        ),
        (
            "become",
            "transform",
            "becomes",
            NarrativeTransitivity::Transitive,
        ),
        (
            "became",
            "transform",
            "becomes",
            NarrativeTransitivity::Transitive,
        ),
        ("is", "state", "is", NarrativeTransitivity::Transitive),
        ("was", "state", "is", NarrativeTransitivity::Transitive),
    ]
    .into_iter()
    .map(
        |(lemma, event_class, relation_type, transitivity)| NarrativeRuleCompiled {
            lemma: lemma.to_owned(),
            event_class: event_class.to_owned(),
            relation_type: relation_type.to_owned(),
            transitivity,
        },
    )
    .collect()
}

fn stem(surface: &str) -> String {
    let lower = normalize_raw(surface);
    for suffix in ["ing", "ed", "es", "s", "er", "tion", "ness"] {
        if lower.ends_with(suffix) && lower.len() > suffix.len() + 2 {
            return lower[..lower.len() - suffix.len()].to_owned();
        }
    }
    lower
}

fn sentence_index_for(offset: usize, sentences: &[SentenceSpan]) -> usize {
    sentences
        .iter()
        .find(|sentence| {
            offset >= sentence.range.start as usize && offset < sentence.range.end as usize
        })
        .map(|sentence| sentence.index)
        .unwrap_or(0)
}

fn slice(text: &str, range: TextRange) -> &str {
    text.get(range.start as usize..range.end as usize)
        .unwrap_or_default()
}

fn mask_at(position: usize, masks: &[TextRange]) -> Option<&TextRange> {
    masks
        .iter()
        .find(|mask| position >= mask.start as usize && position < mask.end as usize)
}

fn overlaps_any(range: TextRange, others: &[TextRange]) -> bool {
    others.iter().any(|other| overlaps(range, *other))
}

fn overlaps(left: TextRange, right: TextRange) -> bool {
    left.start < right.end && right.start < left.end
}

fn contains(outer: TextRange, inner: TextRange) -> bool {
    outer.start <= inner.start && outer.end >= inner.end
}

fn token_count(surface: &str) -> usize {
    surface.split_whitespace().count().max(1)
}

fn is_nominal(tag: &PosTag) -> bool {
    matches!(tag, PosTag::Noun | PosTag::Pronoun | PosTag::ProperNoun)
}

fn is_verbal(tag: &PosTag) -> bool {
    matches!(tag, PosTag::Verb | PosTag::Auxiliary | PosTag::Modal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seed(
        id: &str,
        name: &str,
        aliases: &[&str],
        kind: EntityKind,
        gender: GenderHint,
    ) -> ResolverEntitySeed {
        ResolverEntitySeed {
            entity_id: phoenix_types::EntityId(id.to_owned()),
            canonical_name: name.to_owned(),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
            kind: Some(kind),
            gender: Some(gender),
            number: None,
            scope: ScopeKey::default(),
        }
    }

    #[test]
    fn entity_first_masking_keeps_multiword_mentions_atomic() {
        let scanner = PhoenixScanner::default();
        let text = "Monkey D. Luffy attacked the gate.";
        let artifact = scanner.scan(&ScanRequest {
            text: text.to_owned(),
            scope: ScopeKey::default(),
            session_id: None,
            resolver_seed: vec![seed(
                "luffy",
                "Monkey D. Luffy",
                &[],
                EntityKind::Character,
                GenderHint::Male,
            )],
        });

        assert_eq!(artifact.mentions.len(), 1);
        assert_eq!(artifact.mentions[0].surface, "Monkey D. Luffy");
        assert!(artifact
            .tokens
            .iter()
            .any(|token| token.masked && slice(text, token.range) == "Monkey D. Luffy"));
    }

    #[test]
    fn stopword_suppression_blocks_false_discovery() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Beautiful chaos shimmered in the hall.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-a".to_owned())),
            resolver_seed: Vec::new(),
        });

        assert!(artifact
            .mentions
            .iter()
            .all(|mention| mention.source != Some(MentionSource::Discovery)));
    }

    #[test]
    fn stopword_profile_blocks_sentence_openers_and_pronouns() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "He waited. Then he moved. What changed? What stayed?".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-stopwords".to_owned())),
            resolver_seed: Vec::new(),
        });

        assert!(artifact
            .mentions
            .iter()
            .all(|mention| mention.source != Some(MentionSource::Discovery)));
    }

    #[test]
    fn discovery_noise_overlay_blocks_editorial_terms() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Image flashed. Gesture landed. Image blurred. Gesture held.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-noise".to_owned())),
            resolver_seed: Vec::new(),
        });

        assert!(artifact
            .mentions
            .iter()
            .all(|mention| mention.source != Some(MentionSource::Discovery)));
    }

    #[test]
    fn overlap_resolution_prefers_exact_multiword_mentions() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Monkey D. Luffy smiled.".to_owned(),
            scope: ScopeKey::default(),
            session_id: None,
            resolver_seed: vec![seed(
                "luffy",
                "Monkey D. Luffy",
                &["Luffy"],
                EntityKind::Character,
                GenderHint::Male,
            )],
        });

        assert_eq!(artifact.mentions.len(), 1);
        assert_eq!(artifact.mentions[0].surface, "Monkey D. Luffy");
    }

    #[test]
    fn pronoun_resolution_tracks_recency_and_gender() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Luffy arrived. He smiled.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("resolver-a".to_owned())),
            resolver_seed: vec![seed(
                "luffy",
                "Luffy",
                &[],
                EntityKind::Character,
                GenderHint::Male,
            )],
        });

        assert!(artifact.resolver_links.iter().any(|link| link.link_kind
            == Some(ResolverLinkKind::Pronoun)
            && link.target_entity
                == Some(MentionEntityRef::Known(phoenix_types::EntityId(
                    "luffy".to_owned()
                )))));
    }

    #[test]
    fn alias_candidate_detection_is_sentence_local() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Monkey D. Luffy, aka Straw Hat, arrived. Luffy or Zoro fought.".to_owned(),
            scope: ScopeKey::default(),
            session_id: None,
            resolver_seed: vec![
                seed(
                    "luffy",
                    "Monkey D. Luffy",
                    &["Straw Hat"],
                    EntityKind::Character,
                    GenderHint::Male,
                ),
                seed("zoro", "Zoro", &[], EntityKind::Character, GenderHint::Male),
            ],
        });

        let alias_links = artifact
            .resolver_links
            .iter()
            .filter(|link| link.link_kind == Some(ResolverLinkKind::AliasCandidate))
            .collect::<Vec<_>>();
        assert!(!alias_links.is_empty());
        assert!(alias_links.iter().any(|link| {
            link.target_entity
                == Some(MentionEntityRef::Known(phoenix_types::EntityId(
                    "luffy".to_owned(),
                )))
        }));
        assert!(alias_links.iter().all(|link| {
            link.target_entity
                != Some(MentionEntityRef::Known(phoenix_types::EntityId(
                    "zoro".to_owned(),
                )))
        }));
    }

    #[test]
    fn narrative_matcher_handles_irregular_forms() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Luffy attacked and spoke.".to_owned(),
            scope: ScopeKey::default(),
            session_id: None,
            resolver_seed: vec![seed(
                "luffy",
                "Luffy",
                &[],
                EntityKind::Character,
                GenderHint::Male,
            )],
        });

        assert!(artifact
            .narrative_hits
            .iter()
            .any(|hit| hit.relation_type == "attacks"));
        assert!(artifact
            .narrative_hits
            .iter()
            .any(|hit| hit.relation_type == "speaksTo"));
    }

    #[test]
    fn discovery_requires_repeated_evidence() {
        let scanner = PhoenixScanner::default();
        let request = ScanRequest {
            text: "Captain Ember arrived.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-b".to_owned())),
            resolver_seed: Vec::new(),
        };

        let first = scanner.scan(&request);
        assert!(first
            .mentions
            .iter()
            .all(|mention| mention.source != Some(MentionSource::Discovery)));

        let second = scanner.scan(&request);
        assert!(second
            .mentions
            .iter()
            .any(|mention| mention.source == Some(MentionSource::Discovery)));
    }

    #[test]
    fn repeated_real_names_still_emit_discovery_mentions() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Fiora answered. Kamaria waited. Fiora moved. Kamaria smiled.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-real-names".to_owned())),
            resolver_seed: Vec::new(),
        });

        let discovery_surfaces = artifact
            .mentions
            .iter()
            .filter(|mention| mention.source == Some(MentionSource::Discovery))
            .map(|mention| mention.surface.as_str())
            .collect::<Vec<_>>();
        assert!(discovery_surfaces.contains(&"Fiora"));
        assert!(discovery_surfaces.contains(&"Kamaria"));
    }

    #[test]
    fn multiword_phrases_with_signal_survive_stopword_filtering() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "The Ember Gate opened. The Ember Gate cracked.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-multiword".to_owned())),
            resolver_seed: Vec::new(),
        });

        assert!(
            artifact.mentions.iter().any(|mention| {
                mention.source == Some(MentionSource::Discovery) && mention.surface == "Ember Gate"
            }),
            "The leading 'The' should be stripped, discovering 'Ember Gate'"
        );
    }

    #[test]
    fn off_stopword_profile_disables_discovery_suppression() {
        let scanner = PhoenixScanner::new(ScannerConfig {
            stopword_profile: "off".to_owned(),
            ..ScannerConfig::default()
        });
        // Use a capitalized noun (not pronoun) that the default stopword list would block
        let artifact = scanner.scan(&ScanRequest {
            text: "North called. North answered.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-off-profile".to_owned())),
            resolver_seed: Vec::new(),
        });

        assert!(artifact.mentions.iter().any(|mention| {
            mention.source == Some(MentionSource::Discovery) && mention.surface == "North"
        }));
    }

    #[test]
    fn shared_sentence_splitter_preserves_guard_behavior() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Dr. Luffy ran. Mr. Zoro stayed! Wow?".to_owned(),
            scope: ScopeKey::default(),
            session_id: None,
            resolver_seed: Vec::new(),
        });

        assert_eq!(artifact.sentences.len(), 3);
        assert_eq!(
            slice(
                "Dr. Luffy ran. Mr. Zoro stayed! Wow?",
                artifact.sentences[0].range
            ),
            "Dr. Luffy ran."
        );
        assert_eq!(
            slice(
                "Dr. Luffy ran. Mr. Zoro stayed! Wow?",
                artifact.sentences[1].range
            ),
            "Mr. Zoro stayed!"
        );
        assert_eq!(
            slice(
                "Dr. Luffy ran. Mr. Zoro stayed! Wow?",
                artifact.sentences[2].range
            ),
            "Wow?"
        );
    }

    #[test]
    fn pos_tag_gate_blocks_non_noun_discoveries() {
        // The POS tagger gives ProperNoun to capitalized words, so POS gate primarily
        // catches continuations. But let's verify a verb-form at sentence start where
        // the word also appears lowercase (double penalty) gets suppressed.
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Stayed behind. She stayed close. Stayed firm.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-pos-verb".to_owned())),
            resolver_seed: Vec::new(),
        });

        // "Stayed" appears capitalized at sentence start but lowercase too
        // sentence-start penalty + lowercase-alias penalty should suppress it
        assert!(
            artifact
                .mentions
                .iter()
                .all(|m| m.source != Some(MentionSource::Discovery) || m.surface != "Stayed"),
            "Verb 'Stayed' at sentence start with lowercase alias should be suppressed"
        );
    }

    #[test]
    fn prefix_stop_words_are_skipped() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Then Isolde stayed. Then Isolde left. The Circle opened. The Circle closed."
                .to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-stop-prefix".to_owned())),
            resolver_seed: Vec::new(),
        });

        assert!(
            artifact
                .mentions
                .iter()
                .any(|m| m.source == Some(MentionSource::Discovery) && m.surface == "Isolde"),
            "Isolde should be discovered without the 'Then' prefix"
        );
        assert!(
            artifact
                .mentions
                .iter()
                .any(|m| m.source == Some(MentionSource::Discovery) && m.surface == "Circle"),
            "Circle should be discovered without the 'The' prefix"
        );
        assert!(
            artifact
                .mentions
                .iter()
                .all(|m| m.source != Some(MentionSource::Discovery)
                    || !m.surface.starts_with("Then ") && !m.surface.starts_with("The ")),
            "Entities prefixed with stop-words should have been stripped"
        );
    }

    #[test]
    fn lowercase_alias_suppresses_common_words() {
        let scanner = PhoenixScanner::default();
        // "Ice" appears capitalized AND "ice" appears lowercase → penalty kills it
        let artifact = scanner.scan(&ScanRequest {
            text: "Ice formed quickly. The ice fell from the edge. Ice shattered.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-lowercase".to_owned())),
            resolver_seed: Vec::new(),
        });

        assert!(
            artifact
                .mentions
                .iter()
                .all(|m| m.source != Some(MentionSource::Discovery) || m.surface != "Ice"),
            "Common word 'Ice' that appears lowercase should be suppressed"
        );
    }

    #[test]
    fn dialogue_lead_penalty_suppresses_interjections() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: r#""Nah. Not really." He paused. "Nah. Never mind.""#.to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-dialogue".to_owned())),
            resolver_seed: Vec::new(),
        });

        assert!(
            artifact
                .mentions
                .iter()
                .all(|m| m.source != Some(MentionSource::Discovery) || m.surface != "Nah"),
            "Dialogue-lead word 'Nah' after quote should be suppressed"
        );
    }

    #[test]
    fn real_names_survive_all_new_filters() {
        let scanner = PhoenixScanner::default();
        let artifact = scanner.scan(&ScanRequest {
            text: "Fiora laughed. Kamaria waited. Fiora moved. Kamaria smiled.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-survive".to_owned())),
            resolver_seed: Vec::new(),
        });

        let discovery_surfaces: Vec<&str> = artifact
            .mentions
            .iter()
            .filter(|m| m.source == Some(MentionSource::Discovery))
            .map(|m| m.surface.as_str())
            .collect();
        assert!(
            discovery_surfaces.contains(&"Fiora"),
            "Real name 'Fiora' must survive all filters"
        );
        assert!(
            discovery_surfaces.contains(&"Kamaria"),
            "Real name 'Kamaria' must survive all filters"
        );
    }

    #[test]
    fn sentence_start_common_word_suppressed() {
        let scanner = PhoenixScanner::default();
        // "Hold" only appears at sentence start and also appears lowercase
        let artifact = scanner.scan(&ScanRequest {
            text: "Hold steady. She tried to hold on. Hold firm.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("disc-sentence-start".to_owned())),
            resolver_seed: Vec::new(),
        });

        assert!(
            artifact
                .mentions
                .iter()
                .all(|m| m.source != Some(MentionSource::Discovery) || m.surface != "Hold"),
            "Common word 'Hold' at sentence start with lowercase alias should be suppressed"
        );
    }
}
