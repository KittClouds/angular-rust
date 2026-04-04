use phoenix_alex::{
    is_stop_word_with_profile, normalize_raw, normalized_has_meaningful_token, split_sentence_ranges,
    strip_possessive,
};
use phoenix_types::{
    ChunkKind, ChunkSpan, Diagnostic, EntityId, EntityKind, MentionEntityRef, MentionSource,
    MentionSpan, NarrativeTransitivity, NarrativeVerbHit, PosTag, ResolverEntitySeed,
    ResolverLink, ResolverLinkKind, ScanArtifact, ScopeKey, SentenceSpan, TextRange, TokenClass,
    TokenSpan,
};
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

#[derive(Default)]
pub struct NativeObservedScanner;

impl NativeObservedScanner {
    pub fn scan_parts(
        &self,
        text: &str,
        scope: &ScopeKey,
        resolver_seed: &[ResolverEntitySeed],
    ) -> ScanArtifact {
        let _ = scope;
        let sentences = split_sentences(text);
        let tokens = tokenize(text);
        let sentence_bounds = sentence_token_bounds(&tokens, &sentences);
        let seed_index = SeedIndex::new(resolver_seed);
        let exact_mentions =
            exact_seed_mentions(text, &tokens, &sentences, &sentence_bounds, &seed_index);
        let exact_ranges_by_sentence =
            candidate_ranges_by_sentence(&exact_mentions, sentences.len());
        let mentions = select_mentions(
            exact_mentions.clone(),
            discovery_mentions(
                text,
                &tokens,
                &sentences,
                &sentence_bounds,
                &seed_index,
                &exact_ranges_by_sentence,
            ),
        );
        let chunks = build_chunks(&tokens, &sentences, &sentence_bounds, &mentions);
        let narrative_hits = narrative_hits(text, &tokens, &sentences, &sentence_bounds, &chunks);
        let resolver_links =
            resolve_links_fast(text, &tokens, &sentences, &sentence_bounds, &mentions);
        ScanArtifact {
            sentences,
            tokens: tokens.into_iter().map(|token| token.span).collect(),
            mentions,
            chunks,
            resolver_links,
            narrative_hits,
            diagnostics: vec![Diagnostic {
                code: "PX_INVARANT_NATIVE_SCAN".to_owned(),
                message:
                    "Invarant used the native single-pass observed scanner instead of the legacy multi-pass scanner."
                        .to_owned(),
            }],
        }
    }
}

#[derive(Clone, Debug)]
struct ObservedCandidate {
    range: TextRange,
    surface: String,
    kind: Option<EntityKind>,
    entity_ref: Option<MentionEntityRef>,
    source: MentionSource,
    confidence: f32,
    priority: u8,
    sentence_index: usize,
}

#[derive(Clone, Debug)]
struct TokenInfo {
    span: TokenSpan,
    normalized: String,
}

#[derive(Clone, Debug)]
struct SeedEntry {
    entity_id: EntityId,
    kind: Option<EntityKind>,
    source: MentionSource,
}

#[derive(Default)]
struct SeedIndex {
    exact: FxHashMap<String, SmallVec<[SeedEntry; 1]>>,
    max_phrase_len: usize,
}

impl SeedIndex {
    fn new(seeds: &[ResolverEntitySeed]) -> Self {
        let mut exact = FxHashMap::<String, SmallVec<[SeedEntry; 1]>>::default();
        let mut max_phrase_len = 1usize;
        for seed in seeds {
            for (surface, source) in std::iter::once((&seed.canonical_name, MentionSource::Known))
                .chain(seed.aliases.iter().map(|alias| (alias, MentionSource::Alias)))
            {
                let normalized = normalize_raw(surface);
                if normalized.is_empty() {
                    continue;
                }
                max_phrase_len = max_phrase_len.max(normalized.split_whitespace().count());
                exact.entry(normalized).or_default().push(SeedEntry {
                    entity_id: seed.entity_id.clone(),
                    kind: seed.kind.clone(),
                    source: source.clone(),
                });
            }
        }
        Self {
            exact,
            max_phrase_len: max_phrase_len.max(1),
        }
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
    if sentences.is_empty() && !text.is_empty() {
        sentences.push(SentenceSpan {
            index: 0,
            range: TextRange {
                start: 0,
                end: text.len().min(u32::MAX as usize) as u32,
            },
        });
    }
    sentences
}

fn tokenize(text: &str) -> Vec<TokenInfo> {
    let mut tokens = tokenize_internal(text);
    retag_with_context(text, &mut tokens);
    tokens
}

fn tokenize_internal(text: &str) -> Vec<TokenInfo> {
    let mut tokens = Vec::new();
    let mut index = 0usize;
    while index < text.len() {
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
            start: start.min(u32::MAX as usize) as u32,
            end: index.min(u32::MAX as usize) as u32,
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
        tokens.push(TokenInfo {
            normalized: normalize_raw(surface),
            span: TokenSpan {
                range,
                token_class: Some(token_class),
                pos: Some(guess_pos(surface)),
                masked: false,
                capitalized: surface
                    .chars()
                    .next()
                    .map(|value| value.is_uppercase())
                    .unwrap_or(false),
            },
        });
    }
    tokens
}

fn retag_with_context(text: &str, tokens: &mut [TokenInfo]) {
    for index in 0..tokens.len() {
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

fn guess_pos(token: &str) -> PosTag {
    let lower = token.to_ascii_lowercase();
    if lower.is_empty() {
        return PosTag::Other;
    }
    if [".", ",", "!", "?", "(", ")", ":", ";"].contains(&token) {
        return PosTag::Punctuation;
    }
    if matches!(
        lower.as_str(),
        "he"
            | "him"
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
            | "near"
            | "inside"
            | "outside"
            | "across"
            | "during"
    ) {
        return PosTag::Preposition;
    }
    if matches!(
        lower.as_str(),
        "and" | "or" | "but" | "nor" | "so" | "yet"
    ) {
        return PosTag::Conjunction;
    }
    if matches!(
        lower.as_str(),
        "is" | "are" | "was" | "were" | "be" | "been" | "being"
    ) {
        return PosTag::Auxiliary;
    }
    if matches!(
        lower.as_str(),
        "can" | "could" | "may" | "might" | "must" | "shall" | "should" | "will" | "would"
    ) {
        return PosTag::Modal;
    }
    if lower.ends_with("ing") || lower.ends_with("ed") {
        return PosTag::Verb;
    }
    if lower.ends_with("ly") {
        return PosTag::Adverb;
    }
    if lower.ends_with("ous")
        || lower.ends_with("ful")
        || lower.ends_with("ive")
        || lower.ends_with("less")
        || lower.ends_with("able")
    {
        return PosTag::Adjective;
    }
    if token.chars().next().map(|ch| ch.is_uppercase()).unwrap_or(false) {
        return PosTag::ProperNoun;
    }
    PosTag::Noun
}

fn exact_seed_mentions(
    text: &str,
    tokens: &[TokenInfo],
    sentences: &[SentenceSpan],
    sentence_bounds: &[(usize, usize)],
    seed_index: &SeedIndex,
) -> Vec<ObservedCandidate> {
    if seed_index.exact.is_empty() {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    for (sentence, &(sentence_start, sentence_end)) in
        sentences.iter().zip(sentence_bounds.iter())
    {
        let mut cursor = 0usize;
        while sentence_start + cursor < sentence_end {
            let token_index = sentence_start + cursor;
            if tokens[token_index].span.token_class != Some(TokenClass::Word) {
                cursor += 1;
                continue;
            }
            let mut best_match: Option<(usize, &SmallVec<[SeedEntry; 1]>, String)> = None;
            let max_len = seed_index.max_phrase_len.min(sentence_end - token_index);
            let mut phrase = String::new();
            for len in 0..max_len {
                let idx = token_index + len;
                if tokens[idx].span.token_class != Some(TokenClass::Word) {
                    break;
                }
                let normalized = tokens[idx].normalized.as_str();
                if normalized.is_empty() {
                    break;
                }
                if !phrase.is_empty() {
                    phrase.push(' ');
                }
                phrase.push_str(normalized);
                if let Some(entries) = seed_index.exact.get(&phrase) {
                    let end_index = token_index + len;
                    let surface = slice(
                        text,
                        TextRange {
                            start: tokens[token_index].span.range.start,
                            end: tokens[end_index].span.range.end,
                        },
                    )
                    .to_owned();
                    best_match = Some((len + 1, entries, surface));
                }
            }
            if let Some((consumed, entries, surface)) = best_match {
                let start = tokens[token_index].span.range.start;
                let end = tokens[token_index + consumed - 1].span.range.end;
                let unique_entity = if entries.len() == 1 {
                    Some(entries[0].entity_id.clone())
                } else {
                    None
                };
                let source = entries
                    .first()
                    .map(|entry| entry.source.clone())
                    .unwrap_or(MentionSource::Known);
                candidates.push(ObservedCandidate {
                    range: TextRange { start, end },
                    surface,
                    kind: entries.first().and_then(|entry| entry.kind.clone()),
                    entity_ref: unique_entity.map(MentionEntityRef::Known),
                    source: source.clone(),
                    confidence: if matches!(source, MentionSource::Alias) {
                        0.93
                    } else {
                        0.97
                    },
                    priority: if consumed > 1 { 5 } else { 4 },
                    sentence_index: sentence.index,
                });
                cursor += consumed;
                continue;
            }
            cursor += 1;
        }
    }
    candidates
}

fn discovery_mentions(
    text: &str,
    tokens: &[TokenInfo],
    sentences: &[SentenceSpan],
    sentence_bounds: &[(usize, usize)],
    seed_index: &SeedIndex,
    exact_ranges_by_sentence: &[Vec<TextRange>],
) -> Vec<ObservedCandidate> {
    let mut candidates = Vec::new();
    for (sentence, &(sentence_start, sentence_end)) in
        sentences.iter().zip(sentence_bounds.iter())
    {
        let known_ranges = exact_ranges_by_sentence
            .get(sentence.index)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let mut cursor = 0usize;
        while sentence_start + cursor < sentence_end {
            let token = &tokens[sentence_start + cursor];
            if token.span.token_class != Some(TokenClass::Word)
                || !token.span.capitalized
                || overlaps_any(token.span.range, known_ranges)
            {
                cursor += 1;
                continue;
            }
            let first_normalized = token.normalized.as_str();
            if first_normalized.is_empty()
                || is_stop_word_with_profile(strip_possessive(&first_normalized), "default")
            {
                cursor += 1;
                continue;
            }
            let start = cursor;
            let mut end = cursor + 1;
            while sentence_start + end < sentence_end {
                let next = &tokens[sentence_start + end];
                if next.span.token_class != Some(TokenClass::Word) {
                    break;
                }
                let surface = slice(text, next.span.range);
                if next.span.capitalized || connective_token(surface) {
                    end += 1;
                } else {
                    break;
                }
            }
            let start_range = tokens[sentence_start + start].span.range;
            let end_range = tokens[sentence_start + end - 1].span.range;
            let surface = slice(
                text,
                TextRange {
                    start: start_range.start,
                    end: end_range.end,
                },
            )
            .trim_matches(|ch: char| ch.is_ascii_punctuation())
            .to_owned();
            let normalized = normalize_raw(&surface);
            if !normalized.is_empty()
                && normalized_has_meaningful_token(&normalized, "default")
                && !is_stop_word_with_profile(strip_possessive(&normalized), "default")
                && !seed_index.exact.contains_key(&normalized)
            {
                candidates.push(ObservedCandidate {
                    range: TextRange {
                        start: start_range.start,
                        end: end_range.end,
                    },
                    surface,
                    kind: Some(EntityKind::Other),
                    entity_ref: Some(MentionEntityRef::Speculative(normalized)),
                    source: MentionSource::Discovery,
                    confidence: if end - start > 1 { 0.78 } else { 0.62 },
                    priority: if end - start > 1 { 3 } else { 2 },
                    sentence_index: sentence.index,
                });
            }
            cursor = end;
        }
    }
    candidates
}

fn select_mentions(
    mut exact: Vec<ObservedCandidate>,
    mut discovery: Vec<ObservedCandidate>,
) -> Vec<MentionSpan> {
    exact.append(&mut discovery);
    exact.sort_by(|left, right| {
        left.range
            .start
            .cmp(&right.range.start)
            .then(right.priority.cmp(&left.priority))
            .then(
                (right.range.end.saturating_sub(right.range.start))
                    .cmp(&left.range.end.saturating_sub(left.range.start)),
            )
    });
    let mut last_end = 0u32;
    let mut mentions = Vec::new();
    for candidate in exact {
        if candidate.range.start < last_end {
            continue;
        }
        last_end = candidate.range.end;
        mentions.push(MentionSpan {
            range: candidate.range,
            surface: candidate.surface,
            kind: candidate.kind,
            entity_ref: candidate.entity_ref,
            source: Some(candidate.source),
            confidence: candidate.confidence,
            sentence_index: candidate.sentence_index,
        });
    }
    mentions
}

fn build_chunks(
    tokens: &[TokenInfo],
    sentences: &[SentenceSpan],
    sentence_bounds: &[(usize, usize)],
    mentions: &[MentionSpan],
) -> Vec<ChunkSpan> {
    let mut chunks = Vec::new();
    let mut seen = FxHashSet::<(u8, u32, u32)>::default();
    let mentions_by_sentence = mentions_by_sentence(mentions, sentences.len());
    for (sentence, &(sentence_start, sentence_end)) in
        sentences.iter().zip(sentence_bounds.iter())
    {
        let sentence_mentions = mentions_by_sentence
            .get(sentence.index)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        if !sentence_mentions.is_empty() {
            for mention in sentence_mentions {
                insert_chunk(
                    &mut chunks,
                    &mut seen,
                    ChunkKind::Np,
                    mention.range,
                    mention.range,
                    SmallVec::new(),
                    sentence.index,
                );
            }
        }
        let head = tokens[sentence_start..sentence_end]
            .iter()
            .find_map(|token| {
                matches!(
                    token.span.pos,
                    Some(PosTag::Verb | PosTag::Auxiliary | PosTag::Modal)
                )
                .then_some(token.span.range)
            })
            .unwrap_or(sentence.range);
        insert_chunk(
            &mut chunks,
            &mut seen,
            ChunkKind::Clause,
            sentence.range,
            head,
            SmallVec::new(),
            sentence.index,
        );

        let mention_ranges = sentence_mentions
            .iter()
            .map(|mention| mention.range)
            .collect::<Vec<_>>();
        for token in &tokens[sentence_start..sentence_end] {
            if token.span.pos == Some(PosTag::Pronoun)
                && !overlaps_any(token.span.range, &mention_ranges)
            {
                insert_chunk(
                    &mut chunks,
                    &mut seen,
                    ChunkKind::Np,
                    token.span.range,
                    token.span.range,
                    SmallVec::new(),
                    sentence.index,
                );
            }
        }
        for (offset, token) in tokens[sentence_start..sentence_end].iter().enumerate() {
            if token.span.pos != Some(PosTag::Preposition) {
                continue;
            }
            let next_np = sentence_mentions
                .iter()
                .find(|mention| mention.range.start >= token.span.range.end)
                .map(|mention| mention.range)
                .or_else(|| {
                    tokens[sentence_start..sentence_end]
                        .iter()
                        .skip(offset + 1)
                        .find(|candidate| {
                            matches!(
                                candidate.span.pos,
                                Some(PosTag::Noun | PosTag::ProperNoun | PosTag::Pronoun)
                            )
                        })
                        .map(|candidate| candidate.span.range)
                });
            if let Some(target_range) = next_np {
                let mut modifiers = SmallVec::<[TextRange; 2]>::new();
                modifiers.push(target_range);
                insert_chunk(
                    &mut chunks,
                    &mut seen,
                    ChunkKind::Pp,
                    TextRange {
                        start: token.span.range.start,
                        end: target_range.end,
                    },
                    token.span.range,
                    modifiers,
                    sentence.index,
                );
            }
        }
    }
    chunks.sort_by_key(|chunk| (chunk.sentence_index, chunk.range.start, chunk.range.end));
    chunks
}

fn narrative_hits(
    text: &str,
    tokens: &[TokenInfo],
    sentences: &[SentenceSpan],
    sentence_bounds: &[(usize, usize)],
    chunks: &[ChunkSpan],
) -> Vec<NarrativeVerbHit> {
    let mut hits = Vec::new();
    let sentence_np_counts = chunks
        .iter()
        .filter(|chunk| chunk.kind == Some(ChunkKind::Np))
        .fold(FxHashMap::<usize, usize>::default(), |mut acc, chunk| {
            *acc.entry(chunk.sentence_index).or_insert(0) += 1;
            acc
        });
    for (sentence, &(sentence_start, sentence_end)) in
        sentences.iter().zip(sentence_bounds.iter())
    {
        for token in &tokens[sentence_start..sentence_end] {
            if !matches!(
                token.span.pos,
                Some(PosTag::Verb | PosTag::Auxiliary | PosTag::Modal)
            ) {
                continue;
            }
            let surface = slice(text, token.span.range);
            let lower = surface.to_ascii_lowercase();
            if matches!(
                lower.as_str(),
                "is" | "are" | "was" | "were" | "be" | "been" | "being" | "have" | "has" | "had"
            ) {
                continue;
            }
            let np_count = sentence_np_counts.get(&sentence.index).copied().unwrap_or(0);
            let transitivity = if np_count >= 3 {
                NarrativeTransitivity::Ditransitive
            } else if np_count >= 2 {
                NarrativeTransitivity::Transitive
            } else {
                NarrativeTransitivity::Intransitive
            };
            hits.push(NarrativeVerbHit {
                range: token.span.range,
                lemma: normalize_lemma(surface),
                event_class: "action".to_owned(),
                relation_type: normalize_lemma(surface),
                transitivity: Some(transitivity),
                sentence_index: sentence.index,
                confidence: 0.78,
            });
        }
    }
    hits
}

fn resolve_links_fast(
    text: &str,
    tokens: &[TokenInfo],
    sentences: &[SentenceSpan],
    sentence_bounds: &[(usize, usize)],
    mentions: &[MentionSpan],
) -> Vec<ResolverLink> {
    let mut links = Vec::new();
    let mut recent_entities = SmallVec::<[(TextRange, MentionEntityRef); 8]>::new();
    let mentions_by_sentence = mentions_by_sentence(mentions, sentences.len());
    let mention_ranges = mentions
        .iter()
        .map(|mention| (mention.range.start, mention.range.end))
        .collect::<FxHashSet<_>>();
    for (sentence, &(sentence_start, sentence_end)) in
        sentences.iter().zip(sentence_bounds.iter())
    {
        if let Some(local_mentions) = mentions_by_sentence.get(sentence.index) {
            for mention in local_mentions {
                if let Some(entity_ref) = mention.entity_ref.clone() {
                    recent_entities.push((mention.range, entity_ref));
                    if recent_entities.len() > 8 {
                        recent_entities.remove(0);
                    }
                }
            }
        }
        for token in &tokens[sentence_start..sentence_end] {
            if token.span.pos != Some(PosTag::Pronoun) {
                continue;
            }
            if mention_ranges.contains(&(token.span.range.start, token.span.range.end)) {
                continue;
            }
            if let Some((target_range, target_entity)) = recent_entities.last().cloned() {
                links.push(ResolverLink {
                    source_range: token.span.range,
                    target_range: Some(target_range),
                    target_entity: Some(target_entity),
                    link_kind: Some(ResolverLinkKind::Pronoun),
                    confidence: 0.82,
                    sentence_index: sentence.index,
                });
            }
        }
        if let Some(local_mentions) = mentions_by_sentence.get(sentence.index) {
            links.extend(alias_links_for_sentence(text, local_mentions, sentence.index));
        }
    }
    links
}

fn alias_links_for_sentence(
    text: &str,
    mentions: &[&MentionSpan],
    sentence_index: usize,
) -> Vec<ResolverLink> {
    if mentions.len() < 2 {
        return Vec::new();
    }
    let mut links = Vec::new();
    for pair in mentions.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if right.range.start.saturating_sub(left.range.end) > 48 {
            continue;
        }
        let between = slice(
            text,
            TextRange {
                start: left.range.end,
                end: right.range.start,
            },
        )
        .to_ascii_lowercase();
        let pattern_hit = between.contains("aka")
            || between.contains("also known as")
            || between.contains("called")
            || between.trim() == ","
            || between.contains('(');
        if !pattern_hit {
            continue;
        }
        let (source, target_range, target_entity) =
            if matches!(left.entity_ref, Some(MentionEntityRef::Known(_))) {
                (right.range, Some(left.range), left.entity_ref.clone())
            } else if matches!(right.entity_ref, Some(MentionEntityRef::Known(_))) {
                (left.range, Some(right.range), right.entity_ref.clone())
            } else if normalize_raw(&left.surface) == normalize_raw(&right.surface) {
                (right.range, Some(left.range), left.entity_ref.clone())
            } else {
                continue;
            };
        links.push(ResolverLink {
            source_range: source,
            target_range,
            target_entity,
            link_kind: Some(ResolverLinkKind::AliasCandidate),
            confidence: 0.89,
            sentence_index,
        });
    }
    links
}

fn sentence_token_bounds(tokens: &[TokenInfo], sentences: &[SentenceSpan]) -> Vec<(usize, usize)> {
    let mut bounds = Vec::with_capacity(sentences.len());
    let mut token_cursor = 0usize;
    for sentence in sentences {
        while token_cursor < tokens.len()
            && tokens[token_cursor].span.range.end <= sentence.range.start
        {
            token_cursor += 1;
        }
        let start = token_cursor;
        while token_cursor < tokens.len()
            && tokens[token_cursor].span.range.start < sentence.range.end
        {
            token_cursor += 1;
        }
        bounds.push((start, token_cursor));
    }
    bounds
}

fn candidate_ranges_by_sentence(
    candidates: &[ObservedCandidate],
    sentence_count: usize,
) -> Vec<Vec<TextRange>> {
    let mut grouped = vec![Vec::new(); sentence_count];
    for candidate in candidates {
        if let Some(bucket) = grouped.get_mut(candidate.sentence_index) {
            bucket.push(candidate.range);
        }
    }
    grouped
}

fn mentions_by_sentence<'a>(
    mentions: &'a [MentionSpan],
    sentence_count: usize,
) -> Vec<Vec<&'a MentionSpan>> {
    let mut grouped = vec![Vec::new(); sentence_count];
    for mention in mentions {
        if let Some(bucket) = grouped.get_mut(mention.sentence_index) {
            bucket.push(mention);
        }
    }
    grouped
}

fn insert_chunk(
    chunks: &mut Vec<ChunkSpan>,
    seen: &mut FxHashSet<(u8, u32, u32)>,
    kind: ChunkKind,
    range: TextRange,
    head: TextRange,
    modifiers: SmallVec<[TextRange; 2]>,
    sentence_index: usize,
) {
    let kind_tag = match kind {
        ChunkKind::Np => 1,
        ChunkKind::Vp => 2,
        ChunkKind::Pp => 3,
        ChunkKind::Clause => 4,
        ChunkKind::AdjP => 5,
    };
    if !seen.insert((kind_tag, range.start, range.end)) {
        return;
    }
    chunks.push(ChunkSpan {
        kind: Some(kind),
        range,
        head,
        modifiers: modifiers.into_vec(),
        sentence_index,
    });
}

fn overlaps_any(range: TextRange, occupied: &[TextRange]) -> bool {
    occupied
        .iter()
        .any(|other| range.start < other.end && other.start < range.end)
}

fn contains(outer: TextRange, inner: TextRange) -> bool {
    outer.start <= inner.start && outer.end >= inner.end
}

fn is_nominal(pos: &PosTag) -> bool {
    matches!(pos, PosTag::Noun | PosTag::ProperNoun | PosTag::Pronoun)
}

fn is_verbal(pos: &PosTag) -> bool {
    matches!(pos, PosTag::Verb | PosTag::Auxiliary | PosTag::Modal)
}

fn connective_token(token: &str) -> bool {
    matches!(
        token.to_ascii_lowercase().as_str(),
        "of" | "the" | "and" | "de" | "da" | "del" | "van" | "von" | "la" | "le"
    )
}

fn normalize_lemma(surface: &str) -> String {
    let lower = surface.to_ascii_lowercase();
    if lower.ends_with("ing") && lower.len() > 4 {
        return lower.trim_end_matches("ing").to_owned();
    }
    if lower.ends_with("ed") && lower.len() > 3 {
        return lower.trim_end_matches("ed").to_owned();
    }
    if lower.ends_with('s') && lower.len() > 3 {
        return lower.trim_end_matches('s').to_owned();
    }
    lower
}

fn slice(text: &str, range: TextRange) -> &str {
    let start = (range.start as usize).min(text.len());
    let end = (range.end as usize).min(text.len());
    if start >= end {
        ""
    } else {
        text.get(start..end).unwrap_or("")
    }
}

#[cfg(test)]
mod tests {
    use phoenix_types::ScopeKey;

    use super::NativeObservedScanner;

    #[test]
    fn native_scanner_emits_single_pass_artifacts() {
        let scanner = NativeObservedScanner;
        let artifact = scanner.scan_parts(
            "Ryan met Len at the harbor. He called Len the ferryman.",
            &ScopeKey::default(),
            &[],
        );
        assert!(!artifact.sentences.is_empty());
        assert!(!artifact.tokens.is_empty());
        assert!(!artifact.chunks.is_empty());
        assert!(!artifact.mentions.is_empty());
        assert!(artifact
            .diagnostics
            .iter()
            .any(|diag| diag.code == "PX_INVARANT_NATIVE_SCAN"));
    }
}
