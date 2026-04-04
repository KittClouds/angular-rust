use std::collections::{BTreeMap, BTreeSet};

use phoenix_types::{EntityKind, ResolverEntitySeed, SentenceSpan, TextRange, TokenSpan};
use scirs2_text::cleansing::{
    expand_contractions, normalize_currencies, normalize_numbers, normalize_percentages,
    normalize_unicode, normalize_whitespace, remove_accents, replace_emails, replace_urls,
    strip_html_tags,
};
use scirs2_text::information_extraction::{
    CoreferenceResolver, Entity as IeEntity, EntityType as IeEntityType, RuleBasedNER,
    TemporalExtractor,
};
use scirs2_text::named_entity_recognition::{
    extract_entities, NerEntityType as PatternEntityType, NerPatternConfig,
};

const PATTERN_NER_MAX_BYTES: usize = 512 * 1024;
const RULE_NER_MAX_BYTES_WITHOUT_SEEDS: usize = 256 * 1024;
const COREFERENCE_MAX_BYTES_WITHOUT_SEEDS: usize = 256 * 1024;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct NormalizedTextRecord {
    pub normalized_text: String,
    pub folded_text: String,
    pub provider: String,
    pub provider_version: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderMention {
    pub surface: String,
    pub normalized_surface: String,
    pub kind: Option<EntityKind>,
    pub range: TextRange,
    pub sentence_index: usize,
    pub confidence: f32,
    pub provider: String,
    pub provider_version: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderTime {
    pub surface: String,
    pub normalized: Option<String>,
    pub range: TextRange,
    pub sentence_index: usize,
    pub confidence: f32,
    pub provider: String,
    pub provider_version: String,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderCoreferenceMention {
    pub surface: String,
    pub canonical_surface: String,
    pub range: TextRange,
    pub sentence_index: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProviderCoreferenceChain {
    pub canonical: String,
    pub mentions: Vec<ProviderCoreferenceMention>,
    pub confidence: f32,
    pub provider: String,
    pub provider_version: String,
}

pub trait TextNormalizationProvider {
    fn normalize(&self, text: &str) -> NormalizedTextRecord;
}

pub trait NerProvider {
    fn extract_mentions(
        &self,
        text: &str,
        sentences: &[SentenceSpan],
        tokens: &[TokenSpan],
        observed_mentions: &[(TextRange, Option<EntityKind>)],
        seeds: &[ResolverEntitySeed],
    ) -> Vec<ProviderMention>;

    fn extract_time_candidates(&self, text: &str, sentences: &[SentenceSpan]) -> Vec<ProviderTime>;
}

pub trait CoreferenceProvider {
    fn resolve(
        &self,
        text: &str,
        sentences: &[SentenceSpan],
        seeds: &[ResolverEntitySeed],
    ) -> Vec<ProviderCoreferenceChain>;
}

#[derive(Clone, Debug, Default)]
pub struct Scirs2TextNormalizer;

impl TextNormalizationProvider for Scirs2TextNormalizer {
    fn normalize(&self, text: &str) -> NormalizedTextRecord {
        let stripped = strip_html_tags(text);
        let normalized_urls = replace_urls(&stripped, "<url>");
        let normalized_emails = replace_emails(&normalized_urls, "<email>");
        let expanded = expand_contractions(&normalized_emails);
        let unicode = normalize_unicode(&expanded).unwrap_or(expanded);
        let accentless = remove_accents(&unicode);
        let normalized_numbers = normalize_numbers(&accentless, "<num>");
        let normalized_currency = normalize_currencies(&normalized_numbers, "<currency>");
        let normalized_percentages = normalize_percentages(&normalized_currency, "<percent>");
        let normalized_text = normalize_whitespace(&normalized_percentages);
        NormalizedTextRecord {
            folded_text: normalized_text.to_lowercase(),
            normalized_text,
            provider: "scirs2-text-preprocess".to_owned(),
            provider_version: "0.4.1".to_owned(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Scirs2TextNerProvider;

impl NerProvider for Scirs2TextNerProvider {
    fn extract_mentions(
        &self,
        text: &str,
        sentences: &[SentenceSpan],
        tokens: &[TokenSpan],
        observed_mentions: &[(TextRange, Option<EntityKind>)],
        seeds: &[ResolverEntitySeed],
    ) -> Vec<ProviderMention> {
        let rule_entities = build_rule_entities(text, seeds);
        let pattern_entities = if text.len() <= PATTERN_NER_MAX_BYTES {
            extract_entities(text, &NerPatternConfig::all()).unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut mentions = Vec::new();

        for entity in &rule_entities {
            if let Some(kind) = map_ie_entity_kind(&entity.entity_type) {
                mentions.push(ProviderMention {
                    surface: entity.text.clone(),
                    normalized_surface: normalize_surface(&entity.text),
                    kind: Some(kind),
                    range: TextRange {
                        start: entity.start.min(u32::MAX as usize) as u32,
                        end: entity.end.min(u32::MAX as usize) as u32,
                    },
                    sentence_index: sentence_index_for_range(
                        sentences,
                        entity.start.min(u32::MAX as usize) as u32,
                        entity.end.min(u32::MAX as usize) as u32,
                    ),
                    confidence: (entity.confidence as f32).max(0.55),
                    provider: "scirs2-text-rule-ner".to_owned(),
                    provider_version: "0.4.1".to_owned(),
                });
            }
        }
        for entity in &pattern_entities {
            if let Some(kind) = map_pattern_entity_kind(&entity.entity_type) {
                mentions.push(ProviderMention {
                    surface: entity.text.clone(),
                    normalized_surface: normalize_surface(&entity.text),
                    kind: Some(kind),
                    range: TextRange {
                        start: entity.start.min(u32::MAX as usize) as u32,
                        end: entity.end.min(u32::MAX as usize) as u32,
                    },
                    sentence_index: sentence_index_for_range(
                        sentences,
                        entity.start.min(u32::MAX as usize) as u32,
                        entity.end.min(u32::MAX as usize) as u32,
                    ),
                    confidence: entity.confidence as f32,
                    provider: "scirs2-text-pattern-ner".to_owned(),
                    provider_version: "0.4.1".to_owned(),
                });
            }
        }

        mentions.extend(sequence_refinement_mentions(
            text,
            sentences,
            tokens,
            observed_mentions,
            &rule_entities,
            seeds,
        ));
        dedupe_mentions(mentions)
    }

    fn extract_time_candidates(&self, text: &str, sentences: &[SentenceSpan]) -> Vec<ProviderTime> {
        let mut seen = BTreeSet::new();
        let mut times = Vec::new();
        for expression in TemporalExtractor::new().extract(text).unwrap_or_default() {
            let key = (
                expression.start,
                expression.end,
                expression.text.clone(),
            );
            if !seen.insert(key) {
                continue;
            }
            let normalized_surface = normalize_surface(&expression.text);
            times.push(ProviderTime {
                surface: expression.text,
                normalized: Some(normalized_surface),
                range: TextRange {
                    start: expression.start.min(u32::MAX as usize) as u32,
                    end: expression.end.min(u32::MAX as usize) as u32,
                },
                sentence_index: sentence_index_for_range(
                    sentences,
                    expression.start.min(u32::MAX as usize) as u32,
                    expression.end.min(u32::MAX as usize) as u32,
                ),
                confidence: expression.confidence as f32,
                provider: "scirs2-text-temporal".to_owned(),
                provider_version: "0.4.1".to_owned(),
            });
        }
        times
    }
}

#[derive(Clone, Debug, Default)]
pub struct Scirs2TextCoreferenceProvider;

impl CoreferenceProvider for Scirs2TextCoreferenceProvider {
    fn resolve(
        &self,
        text: &str,
        sentences: &[SentenceSpan],
        seeds: &[ResolverEntitySeed],
    ) -> Vec<ProviderCoreferenceChain> {
        if seeds.is_empty() && text.len() > COREFERENCE_MAX_BYTES_WITHOUT_SEEDS {
            return Vec::new();
        }
        let entities = build_rule_entities(text, seeds);
        CoreferenceResolver::new()
            .resolve(text, &entities)
            .unwrap_or_default()
            .into_iter()
            .filter(|chain| chain.mentions.len() >= 2)
            .map(|chain| {
                let canonical = chain
                    .mentions
                    .first()
                    .map(|mention| mention.text.clone())
                    .unwrap_or_default();
                ProviderCoreferenceChain {
                    canonical: canonical.clone(),
                    mentions: chain
                        .mentions
                        .into_iter()
                        .map(|mention| ProviderCoreferenceMention {
                            surface: mention.text,
                            canonical_surface: canonical.clone(),
                            range: TextRange {
                                start: mention.start.min(u32::MAX as usize) as u32,
                                end: mention.end.min(u32::MAX as usize) as u32,
                            },
                            sentence_index: sentence_index_for_range(
                                sentences,
                                mention.start.min(u32::MAX as usize) as u32,
                                mention.end.min(u32::MAX as usize) as u32,
                            ),
                        })
                        .collect(),
                    confidence: chain.confidence as f32,
                    provider: "scirs2-text-coref".to_owned(),
                    provider_version: "0.4.1".to_owned(),
                }
            })
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct InvarantNlpPipeline {
    pub normalizer: Scirs2TextNormalizer,
    pub ner: Scirs2TextNerProvider,
    pub coreference: Scirs2TextCoreferenceProvider,
}

fn dedupe_mentions(mentions: Vec<ProviderMention>) -> Vec<ProviderMention> {
    let mut best = BTreeMap::<(u32, u32, String, String), ProviderMention>::new();
    for mention in mentions {
        let key = (
            mention.range.start,
            mention.range.end,
            mention.normalized_surface.clone(),
            mention.provider.clone(),
        );
        match best.entry(key) {
            std::collections::btree_map::Entry::Occupied(mut entry) => {
                if mention.confidence > entry.get().confidence {
                    entry.insert(mention);
                }
            }
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(mention);
            }
        }
    }
    best.into_values().collect()
}

fn sequence_refinement_mentions(
    text: &str,
    sentences: &[SentenceSpan],
    tokens: &[TokenSpan],
    observed_mentions: &[(TextRange, Option<EntityKind>)],
    rule_entities: &[IeEntity],
    seeds: &[ResolverEntitySeed],
) -> Vec<ProviderMention> {
    let sentence_tokens = build_sentence_tokens(text, sentences, tokens);
    let observed_ranges_by_sentence = observed_ranges_by_sentence(sentences, observed_mentions);
    let rule_ranges_by_sentence = rule_ranges_by_sentence(sentences, rule_entities);
    let mut mentions = Vec::new();
    let mut seen = BTreeSet::new();
    for sentence in sentence_tokens {
        if sentence.tokens.is_empty() {
            continue;
        }
        let occupied = occupied_token_ranges(
            &sentence.tokens,
            observed_ranges_by_sentence
                .get(sentence.sentence_index)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            rule_ranges_by_sentence
                .get(sentence.sentence_index)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        );
        let mut index = 0usize;
        while index < sentence.tokens.len() {
            if occupied.contains(&index) || !looks_like_entity_token(&sentence.tokens[index].text) {
                index += 1;
                continue;
            }
            let start_index = index;
            let mut end_index = index + 1;
            while end_index < sentence.tokens.len()
                && (looks_like_entity_token(&sentence.tokens[end_index].text)
                    || connective_token(&sentence.tokens[end_index].text))
            {
                end_index += 1;
            }
            let absolute_start = sentence.tokens[start_index].range.start;
            let absolute_end = sentence.tokens[end_index - 1].range.end;
            let surface = safe_slice(text, absolute_start as usize, absolute_end as usize);
            let normalized_surface = normalize_surface(&surface);
            if normalized_surface.len() >= 3 {
                let kind = infer_seed_kind(&surface, seeds).or(Some(EntityKind::Character));
                let key = (
                    absolute_start,
                    absolute_end,
                    kind.clone().map(kind_tag).unwrap_or("OTHER").to_owned(),
                );
                if seen.insert(key) {
                    mentions.push(ProviderMention {
                        normalized_surface,
                        surface,
                        kind,
                        range: TextRange {
                            start: absolute_start,
                            end: absolute_end,
                        },
                        sentence_index: sentence.sentence_index,
                        confidence: 0.58,
                        provider: "invarant-sequence-ner".to_owned(),
                        provider_version: "v1".to_owned(),
                    });
                }
            }
            index = end_index;
        }
    }
    mentions
}

#[derive(Clone, Debug)]
struct SentenceToken {
    sentence_index: usize,
    tokens: Vec<TokenFragment>,
}

#[derive(Clone, Debug)]
struct TokenFragment {
    text: String,
    range: TextRange,
}

fn build_sentence_tokens(
    text: &str,
    sentences: &[SentenceSpan],
    tokens: &[TokenSpan],
) -> Vec<SentenceToken> {
    let mut results = Vec::with_capacity(sentences.len());
    let mut token_cursor = 0usize;
    for sentence in sentences {
        let sentence_start = sentence.range.start as usize;
        let sentence_end = sentence.range.end as usize;
        while token_cursor < tokens.len() && tokens[token_cursor].range.end <= sentence.range.start {
            token_cursor += 1;
        }
        let mut lookahead = token_cursor;
        let mut collected = Vec::new();
        while lookahead < tokens.len() && tokens[lookahead].range.start < sentence.range.end {
            let token = &tokens[lookahead];
            if token.range.start >= sentence.range.start && token.range.end <= sentence.range.end {
                let surface = safe_slice(text, token.range.start as usize, token.range.end as usize);
                if !surface.trim().is_empty() {
                    collected.push(TokenFragment {
                        text: surface,
                        range: token.range,
                    });
                }
            }
            lookahead += 1;
        }
        if collected.is_empty() {
            collected = fallback_tokens(text, sentence_start, sentence_end);
        }
        results.push(SentenceToken {
            sentence_index: sentence.index,
            tokens: collected,
        });
    }
    results
}

fn fallback_tokens(text: &str, start: usize, end: usize) -> Vec<TokenFragment> {
    let (start, end) = normalized_bounds(text.len(), start, end);
    let sentence = &text[start..end];
    let mut tokens = Vec::new();
    let mut cursor = 0usize;
    for part in sentence.split_whitespace() {
        if let Some(relative) = sentence[cursor..].find(part) {
            let token_start = start + cursor + relative;
            let token_end = token_start + part.len();
            tokens.push(TokenFragment {
                text: part.to_owned(),
                range: TextRange {
                    start: token_start.min(u32::MAX as usize) as u32,
                    end: token_end.min(u32::MAX as usize) as u32,
                },
            });
            cursor = token_end.saturating_sub(start);
        }
    }
    tokens
}

fn token_indices_for_range(tokens: &[TokenFragment], start: u32, end: u32) -> Option<(usize, usize)> {
    let start_index = tokens
        .iter()
        .position(|token| token.range.end > start && token.range.start <= start)
        .or_else(|| {
            tokens
                .iter()
                .position(|token| token.range.start < end && token.range.end > start)
        })?;
    let end_index = tokens
        .iter()
        .rposition(|token| token.range.start < end && token.range.end > start)
        .map(|index| index + 1)?;
    (end_index > start_index).then_some((start_index, end_index))
}

fn observed_ranges_by_sentence(
    sentences: &[SentenceSpan],
    observed_mentions: &[(TextRange, Option<EntityKind>)],
) -> Vec<Vec<TextRange>> {
    let mut grouped = vec![Vec::new(); sentences.len()];
    for (range, _) in observed_mentions {
        if let Some(bucket) =
            grouped.get_mut(sentence_index_for_range(sentences, range.start, range.end))
        {
            bucket.push(range.clone());
        }
    }
    grouped
}

fn rule_ranges_by_sentence(
    sentences: &[SentenceSpan],
    rule_entities: &[IeEntity],
) -> Vec<Vec<TextRange>> {
    let mut grouped = vec![Vec::new(); sentences.len()];
    for entity in rule_entities {
        let range = TextRange {
            start: entity.start.min(u32::MAX as usize) as u32,
            end: entity.end.min(u32::MAX as usize) as u32,
        };
        if let Some(bucket) =
            grouped.get_mut(sentence_index_for_range(sentences, range.start, range.end))
        {
            bucket.push(range);
        }
    }
    grouped
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

fn occupied_token_ranges(
    tokens: &[TokenFragment],
    observed_ranges: &[TextRange],
    rule_ranges: &[TextRange],
) -> BTreeSet<usize> {
    let mut occupied = BTreeSet::new();
    for range in observed_ranges {
        if let Some((start, end)) = token_indices_for_range(tokens, range.start, range.end) {
            occupied.extend(start..end);
        }
    }
    for range in rule_ranges {
        if let Some((start, end)) = token_indices_for_range(tokens, range.start, range.end) {
            occupied.extend(start..end);
        }
    }
    occupied
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

fn looks_like_entity_token(value: &str) -> bool {
    value
        .chars()
        .next()
        .map(|ch| ch.is_uppercase())
        .unwrap_or(false)
        && value.chars().any(|ch| ch.is_alphabetic())
}

fn connective_token(value: &str) -> bool {
    matches!(value.to_ascii_lowercase().as_str(), "of" | "the" | "and" | "&")
}

fn kind_tag(kind: EntityKind) -> &'static str {
    match kind {
        EntityKind::Character | EntityKind::Npc => "PER",
        EntityKind::Organization | EntityKind::Faction => "ORG",
        EntityKind::Location => "LOC",
        EntityKind::Event | EntityKind::Item | EntityKind::Concept | EntityKind::Other => "OTHER",
    }
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

fn sentence_index_for_range(sentences: &[SentenceSpan], start: u32, end: u32) -> usize {
    let idx = sentences.partition_point(|sentence| sentence.range.start <= start);
    if idx > 0 {
        let candidate = &sentences[idx - 1];
        if candidate.range.start <= start && candidate.range.end >= end {
            return candidate.index;
        }
    }
    sentences
        .get(idx)
        .filter(|sentence| sentence.range.end > start && sentence.range.start < end)
        .map(|sentence| sentence.index)
        .or_else(|| {
            idx.checked_sub(1).and_then(|previous| {
                sentences
                    .get(previous)
                    .filter(|sentence| sentence.range.end > start && sentence.range.start < end)
                    .map(|sentence| sentence.index)
            })
        })
        .unwrap_or_default()
}

fn safe_slice(text: &str, start: usize, end: usize) -> String {
    let (start, end) = normalized_bounds(text.len(), start, end);
    if start >= end {
        return String::new();
    }
    text[start..end].trim().to_owned()
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
