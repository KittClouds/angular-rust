use std::cmp::{max, min};
use std::sync::OnceLock;

use memchr::memchr;
use phoenix_alex::split_sentence_ranges;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SentenceLengthDistribution {
    #[serde(rename = "1")]
    pub count1: i32,
    #[serde(rename = "2-6")]
    pub count2_to6: i32,
    #[serde(rename = "7-15")]
    pub count7_to15: i32,
    #[serde(rename = "16-25")]
    pub count16_to25: i32,
    #[serde(rename = "26-39")]
    pub count26_to39: i32,
    #[serde(rename = "40+")]
    pub count40_plus: i32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FlowInsights {
    pub consecutive_patterns: i32,
    pub dominant_range: String,
    pub variety_score: i32,
    pub has_monotony: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KeywordDensity {
    pub word: String,
    pub count: i32,
    pub percentage: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyticsHighlightRange {
    pub from: i32,
    pub to: i32,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhraseEchoItem {
    pub id: String,
    pub phrase: String,
    pub occurrence_count: i32,
    pub severity: String,
    pub snippets: Vec<String>,
    pub highlight_ranges: Vec<AnalyticsHighlightRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RepetitionAnalysis {
    pub items: Vec<PhraseEchoItem>,
    pub total_flags: i32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProximityConflictItem {
    pub id: String,
    pub root: String,
    pub surface_forms: Vec<String>,
    pub part_of_speech: String,
    pub min_word_distance: i32,
    pub severity: String,
    pub snippets: Vec<String>,
    pub highlight_ranges: Vec<AnalyticsHighlightRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProximityAnalysis {
    pub items: Vec<ProximityConflictItem>,
    pub total_flags: i32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadenceSentence {
    pub id: String,
    pub paragraph_index: i32,
    pub sentence_index: i32,
    pub from: i32,
    pub to: i32,
    pub word_count: i32,
    pub bucket: String,
    pub snippet: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadenceHotspot {
    pub id: String,
    pub r#type: String,
    pub label: String,
    pub severity: String,
    pub explanation: String,
    pub sentence_ids: Vec<String>,
    pub highlight_ranges: Vec<AnalyticsHighlightRange>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CadenceAnalysis {
    pub sentences: Vec<CadenceSentence>,
    pub hotspots: Vec<CadenceHotspot>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TextAnalytics {
    pub word_count: i32,
    pub character_count: i32,
    pub character_count_no_spaces: i32,
    pub sentence_count: i32,
    pub paragraph_count: i32,
    pub reading_level: String,
    pub reading_time_minutes: i32,
    pub reading_time_seconds: i32,
    pub speaking_time_minutes: i32,
    pub speaking_time_seconds: i32,
    pub average_sentence_length: f64,
    pub sentence_length_variation: f64,
    pub flow_score: i32,
    pub sentence_length_distribution: SentenceLengthDistribution,
    pub flow_insights: FlowInsights,
    pub keyword_density: Vec<KeywordDensity>,
    pub repetition: RepetitionAnalysis,
    pub proximity: ProximityAnalysis,
    pub cadence: CadenceAnalysis,
}

#[derive(Clone, Copy, Debug)]
struct TokenMatch {
    normalized_id: u32,
    root_id: u32,
    from: usize,
    to: usize,
    index: i32,
}

#[derive(Clone, Debug)]
struct SentenceMatch {
    from: usize,
    to: usize,
    paragraph_index: i32,
    word_count: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SimpleRange {
    from: usize,
    to: usize,
}

#[derive(Clone, Debug)]
struct PhraseAccumulator {
    count: u32,
    last_start: i32,
    ranges: Vec<SimpleRange>,
    token_starts: Vec<i32>,
}

#[derive(Clone, Copy, Debug)]
struct WordSpan {
    from: usize,
    to: usize,
}

#[derive(Clone, Debug, Default)]
struct StringInterner {
    values: Vec<String>,
    indices: FxHashMap<String, u32>,
}

#[derive(Clone, Debug, Default)]
struct ScanStats {
    word_spans: Vec<WordSpan>,
    tokens: Vec<TokenMatch>,
    word_count: i32,
    syllable_count: i32,
    character_count_no_spaces: i32,
    keyword_frequencies: FxHashMap<u32, i32>,
    interner: StringInterner,
}

pub fn analyze_text(text: &str) -> TextAnalytics {
    if text.is_empty() {
        return get_empty_analytics();
    }

    let stats = scan_text(text);
    let paragraph_starts = paragraph_start_offsets(text);
    let paragraph_count = paragraph_starts.len() as i32;
    let sentences = extract_sentence_matches(text, &stats.word_spans, &paragraph_starts);
    let sentence_lengths = sentences
        .iter()
        .map(|sentence| sentence.word_count)
        .collect::<Vec<_>>();
    let word_count = stats.word_count;
    let character_count = text.len() as i32;
    let sentence_count = sentences.len() as i32;
    let reading_level = calculate_reading_level(word_count, sentence_count, stats.syllable_count);
    let reading_time_total = ((word_count as f64 / 225.0) * 60.0).ceil() as i32;
    let speaking_time_total = ((word_count as f64 / 150.0) * 60.0).ceil() as i32;
    let average_sentence_length = if sentence_count > 0 {
        ((word_count as f64 / sentence_count as f64) * 10.0).round() / 10.0
    } else {
        0.0
    };
    let sentence_length_variation = calculate_standard_deviation(&sentence_lengths);
    let distribution = categorize_sentence_lengths(&sentence_lengths);
    let flow_insights = analyze_flow_insights(&distribution, &sentence_lengths);
    let flow_score = if sentence_count > 0 {
        let var_score = (sentence_length_variation / 8.0 * 100.0).min(100.0);
        ((var_score * 0.6) + (flow_insights.variety_score as f64 * 0.4)).round() as i32
    } else {
        0
    };

    TextAnalytics {
        word_count,
        character_count,
        character_count_no_spaces: stats.character_count_no_spaces,
        sentence_count,
        paragraph_count,
        reading_level,
        reading_time_minutes: reading_time_total / 60,
        reading_time_seconds: reading_time_total % 60,
        speaking_time_minutes: speaking_time_total / 60,
        speaking_time_seconds: speaking_time_total % 60,
        average_sentence_length,
        sentence_length_variation,
        flow_score,
        sentence_length_distribution: distribution,
        flow_insights,
        keyword_density: calculate_keyword_density(
            &stats.interner,
            stats.keyword_frequencies,
            word_count,
        ),
        repetition: analyze_repetition(text, &stats.tokens, &stats.interner),
        proximity: analyze_proximity(text, &stats.tokens, &stats.interner),
        cadence: analyze_cadence(text, &sentences),
    }
}

pub fn get_empty_analytics() -> TextAnalytics {
    TextAnalytics {
        word_count: 0,
        character_count: 0,
        character_count_no_spaces: 0,
        sentence_count: 0,
        paragraph_count: 0,
        reading_level: "N/A".to_owned(),
        reading_time_minutes: 0,
        reading_time_seconds: 0,
        speaking_time_minutes: 0,
        speaking_time_seconds: 0,
        average_sentence_length: 0.0,
        sentence_length_variation: 0.0,
        flow_score: 0,
        sentence_length_distribution: SentenceLengthDistribution::default(),
        flow_insights: FlowInsights {
            dominant_range: "7-15".to_owned(),
            ..FlowInsights::default()
        },
        keyword_density: Vec::new(),
        repetition: RepetitionAnalysis::default(),
        proximity: ProximityAnalysis::default(),
        cadence: CadenceAnalysis::default(),
    }
}

impl StringInterner {
    fn intern(&mut self, value: &str) -> u32 {
        if let Some(&existing) = self.indices.get(value) {
            return existing;
        }
        let id = self.values.len() as u32;
        let owned = value.to_owned();
        self.indices.insert(owned.clone(), id);
        self.values.push(owned);
        id
    }

    fn get(&self, id: u32) -> &str {
        self.values
            .get(id as usize)
            .map(|value| value.as_str())
            .unwrap_or_default()
    }
}

fn stop_words() -> &'static FxHashSet<&'static str> {
    static STOP_WORDS: OnceLock<FxHashSet<&'static str>> = OnceLock::new();
    STOP_WORDS.get_or_init(|| {
        [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "is", "was", "are", "were", "been", "be", "have", "has", "had",
            "do", "does", "did", "will", "would", "could", "should", "may", "might", "must",
            "shall", "can", "need", "dare", "ought", "used", "it", "its", "this", "that", "these",
            "those", "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "our", "their", "mine", "yours", "hers", "ours", "theirs", "what",
            "which", "who", "whom", "whose", "where", "when", "why", "how", "all", "each", "every",
            "both", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "just", "also", "now", "here", "there",
            "then", "once", "if", "into", "through", "during", "before", "after", "above", "below",
            "up", "down", "out", "off", "over", "under", "again", "further", "any", "about",
        ]
        .into_iter()
        .collect()
    })
}

fn scan_text(text: &str) -> ScanStats {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    let mut word_index = 0i32;
    let mut stats = ScanStats::default();

    while index < bytes.len() {
        if is_utf8_lead_byte(bytes[index]) {
            let ch = text[index..].chars().next().unwrap_or_default();
            if !ch.is_whitespace() {
                stats.character_count_no_spaces += 1;
            }
        }

        if is_word_start(bytes, index) {
            let start = index;
            index += 1;
            while index < bytes.len() && is_word_continue(bytes[index]) {
                index += 1;
            }
            let end = index;
            let raw = &text[start..end];
            stats.word_spans.push(WordSpan {
                from: start,
                to: end,
            });
            stats.word_count += 1;
            stats.syllable_count += count_syllables(raw);

            let normalized = normalize_lexeme(raw);
            if !normalized.is_empty() {
                if normalized.len() >= 4
                    && !stop_words().contains(normalized.as_str())
                    && !normalized.bytes().any(|byte| byte.is_ascii_digit())
                {
                    let keyword_id = stats.interner.intern(&normalized);
                    *stats.keyword_frequencies.entry(keyword_id).or_default() += 1;
                }

                if bytes[start].is_ascii_alphabetic() && raw.bytes().all(is_alpha_token_byte) {
                    let normalized_id = stats.interner.intern(&normalized);
                    let root_id = stats.interner.intern(&stem_word(&normalized));
                    stats.tokens.push(TokenMatch {
                        normalized_id,
                        root_id,
                        from: start,
                        to: end,
                        index: word_index,
                    });
                }
            }

            word_index += 1;
            continue;
        }

        index += 1;
    }

    stats
}

fn is_utf8_lead_byte(byte: u8) -> bool {
    byte & 0b1100_0000 != 0b1000_0000
}

fn is_word_start(bytes: &[u8], index: usize) -> bool {
    bytes[index].is_ascii_alphanumeric() || bytes[index] == b'_'
}

fn is_word_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'\'' | b'-')
}

fn is_alpha_token_byte(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || matches!(byte, b'\'' | b'-')
}

fn count_syllables(word: &str) -> i32 {
    let mut letters = SmallVec::<[u8; 32]>::new();
    for &byte in word.as_bytes() {
        if byte.is_ascii_alphabetic() {
            letters.push(byte.to_ascii_lowercase());
        }
    }
    if letters.len() <= 3 {
        return 1;
    }

    let start = usize::from(letters.first() == Some(&b'y'));
    let mut end = letters.len();
    if let Some(stripped_end) = strip_syllable_suffix(&letters[start..end]) {
        end = start + stripped_end;
    }
    if start >= end {
        return 1;
    }

    let mut count = 0i32;
    let slice = &letters[start..end];
    let mut index = 0usize;
    while index < slice.len() {
        if is_vowel(slice[index]) {
            let run_start = index;
            while index < slice.len() && is_vowel(slice[index]) {
                index += 1;
            }
            count += ((index - run_start) as i32 + 1) / 2;
            continue;
        }
        index += 1;
    }

    count.max(1)
}

fn strip_syllable_suffix(bytes: &[u8]) -> Option<usize> {
    if bytes.len() >= 3
        && bytes.ends_with(b"es")
        && !matches!(
            bytes[bytes.len() - 3],
            b'l' | b'a' | b'e' | b'i' | b'o' | b'u' | b'y'
        )
    {
        return Some(bytes.len() - 2);
    }
    if bytes.len() >= 2 && bytes.ends_with(b"ed") {
        return Some(bytes.len() - 2);
    }
    if bytes.len() >= 2
        && bytes[bytes.len() - 1] == b'e'
        && !matches!(
            bytes[bytes.len() - 2],
            b'l' | b'a' | b'e' | b'i' | b'o' | b'u' | b'y'
        )
    {
        return Some(bytes.len() - 1);
    }
    None
}

fn is_vowel(byte: u8) -> bool {
    matches!(byte, b'a' | b'e' | b'i' | b'o' | b'u' | b'y')
}

fn paragraph_start_offsets(text: &str) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut starts = Vec::new();
    let mut index = 0usize;
    let mut blank_break = true;

    while index < bytes.len() {
        let line_end = memchr(b'\n', &bytes[index..]).map_or(bytes.len(), |found| index + found);
        let line = &bytes[index..line_end];
        let mut trimmed_start = 0usize;
        while trimmed_start < line.len() && line[trimmed_start].is_ascii_whitespace() {
            trimmed_start += 1;
        }
        let has_content = line[trimmed_start..]
            .iter()
            .any(|byte| !byte.is_ascii_whitespace());

        if has_content && blank_break {
            starts.push(index + trimmed_start);
        }
        blank_break = !has_content;
        index = line_end.saturating_add(1);
    }

    starts
}

fn categorize_sentence_lengths(sentence_lengths: &[i32]) -> SentenceLengthDistribution {
    let mut dist = SentenceLengthDistribution::default();
    for &length in sentence_lengths {
        match get_sentence_bucket(length) {
            "1" => dist.count1 += 1,
            "2-6" => dist.count2_to6 += 1,
            "7-15" => dist.count7_to15 += 1,
            "16-25" => dist.count16_to25 += 1,
            "26-39" => dist.count26_to39 += 1,
            _ => dist.count40_plus += 1,
        }
    }
    dist
}

fn get_sentence_bucket(count: i32) -> &'static str {
    if count <= 1 {
        "1"
    } else if count <= 6 {
        "2-6"
    } else if count <= 15 {
        "7-15"
    } else if count <= 25 {
        "16-25"
    } else if count <= 39 {
        "26-39"
    } else {
        "40+"
    }
}

fn detect_consecutive_patterns(lengths: &[i32]) -> i32 {
    let mut pattern_count = 0;
    let mut consecutive_count = 1;
    for index in 1..lengths.len() {
        if (lengths[index] - lengths[index - 1]).abs() <= 3 {
            consecutive_count += 1;
            if consecutive_count >= 3 {
                pattern_count += 1;
            }
        } else {
            consecutive_count = 1;
        }
    }
    pattern_count
}

fn calculate_variety_score(dist: &SentenceLengthDistribution, total_sentences: i32) -> i32 {
    if total_sentences == 0 {
        return 0;
    }
    let probabilities = [
        dist.count1,
        dist.count2_to6,
        dist.count7_to15,
        dist.count16_to25,
        dist.count26_to39,
        dist.count40_plus,
    ]
    .into_iter()
    .filter(|value| *value > 0)
    .map(|value| value as f64 / total_sentences as f64)
    .collect::<Vec<_>>();
    if probabilities.len() <= 1 {
        return 0;
    }
    let entropy = -probabilities
        .iter()
        .map(|probability| probability * probability.log2())
        .sum::<f64>();
    let max_entropy = (probabilities.len() as f64).log2();
    if max_entropy == 0.0 {
        return 0;
    }
    ((entropy / max_entropy) * 100.0).round() as i32
}

fn analyze_flow_insights(
    dist: &SentenceLengthDistribution,
    sentence_lengths: &[i32],
) -> FlowInsights {
    let total = dist.count1
        + dist.count2_to6
        + dist.count7_to15
        + dist.count16_to25
        + dist.count26_to39
        + dist.count40_plus;
    let variety_score = calculate_variety_score(dist, total);
    let consecutive_patterns = detect_consecutive_patterns(sentence_lengths);
    let mut dominant_range = "7-15";
    let mut dominant_value = -1;
    for (label, value) in [
        ("1", dist.count1),
        ("2-6", dist.count2_to6),
        ("7-15", dist.count7_to15),
        ("16-25", dist.count16_to25),
        ("26-39", dist.count26_to39),
        ("40+", dist.count40_plus),
    ] {
        if value > dominant_value {
            dominant_range = label;
            dominant_value = value;
        }
    }

    let mut max_consecutive = 1;
    let mut current_consecutive = 1;
    for index in 1..sentence_lengths.len() {
        if (sentence_lengths[index] - sentence_lengths[index - 1]).abs() <= 3 {
            current_consecutive += 1;
            max_consecutive = max(max_consecutive, current_consecutive);
        } else {
            current_consecutive = 1;
        }
    }

    FlowInsights {
        consecutive_patterns,
        dominant_range: dominant_range.to_owned(),
        variety_score,
        has_monotony: max_consecutive >= 5,
    }
}

fn calculate_reading_level(word_count: i32, sentence_count: i32, syllable_count: i32) -> String {
    if word_count == 0 || sentence_count == 0 {
        return "N/A".to_owned();
    }
    let avg_words_per_sentence = word_count as f64 / sentence_count as f64;
    let avg_syllables_per_word = syllable_count as f64 / word_count as f64;
    let grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59;
    if grade < 1.0 {
        "Kindergarten"
    } else if grade < 6.0 {
        "1st-5th Grade"
    } else if grade < 9.0 {
        "6th-8th Grade"
    } else if grade < 13.0 {
        "9th-12th Grade"
    } else if grade < 17.0 {
        "College Level"
    } else {
        "Graduate Level"
    }
    .to_owned()
}

fn calculate_standard_deviation(numbers: &[i32]) -> f64 {
    if numbers.is_empty() {
        return 0.0;
    }
    let mean = numbers.iter().map(|value| *value as f64).sum::<f64>() / numbers.len() as f64;
    let variance = numbers
        .iter()
        .map(|value| {
            let diff = *value as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / numbers.len() as f64;
    variance.sqrt()
}

fn normalize_lexeme(word: &str) -> String {
    let mut normalized = String::with_capacity(word.len());
    for &byte in word.as_bytes() {
        if byte.is_ascii_alphabetic() {
            normalized.push(byte.to_ascii_lowercase() as char);
        } else if matches!(byte, b'\'' | b'-') {
            normalized.push(byte as char);
        }
    }
    normalized
}

fn stem_word(normalized: &str) -> String {
    let mut stem = normalized.to_owned();
    if stem.len() <= 4 {
        return stem;
    }

    let suffixes = [
        "ingly", "edly", "ment", "ness", "tion", "sion", "able", "ible", "less", "ously", "ing",
        "ers", "ies", "ied", "est", "ism", "ist", "ous", "ive", "ful", "ly", "ed", "es", "er", "s",
    ];
    for suffix in suffixes {
        if stem.ends_with(suffix) && stem.len().saturating_sub(suffix.len()) >= 3 {
            stem.truncate(stem.len() - suffix.len());
            break;
        }
    }

    if stem.ends_with('i') && stem.len() > 3 {
        stem.pop();
        stem.push('y');
    }
    if stem.len() >= 3 {
        let bytes = stem.as_bytes();
        let last = bytes[bytes.len() - 1];
        let prev = bytes[bytes.len() - 2];
        if last == prev && is_ascii_consonant(last) {
            stem.pop();
        }
    }

    stem
}

fn is_ascii_consonant(byte: u8) -> bool {
    byte.is_ascii_lowercase() && !is_vowel(byte)
}

fn calculate_keyword_density(
    interner: &StringInterner,
    frequencies: FxHashMap<u32, i32>,
    total_words: i32,
) -> Vec<KeywordDensity> {
    let mut pairs = frequencies.into_iter().collect::<Vec<_>>();
    pairs.sort_by(|left, right| {
        if left.1 == right.1 {
            interner.get(left.0).cmp(interner.get(right.0))
        } else {
            right.1.cmp(&left.1)
        }
    });
    if pairs.len() > 100 {
        pairs.truncate(100);
    }

    let denominator = (total_words as f64).max(1.0);
    pairs
        .into_iter()
        .map(|(word_id, count)| KeywordDensity {
            word: interner.get(word_id).to_owned(),
            count,
            percentage: ((count as f64 / denominator) * 1000.0).round() / 10.0,
        })
        .collect()
}

fn build_snippet(text: &str, from: usize, to: usize, radius: usize) -> String {
    let start = previous_char_boundary(text, from.saturating_sub(radius));
    let end = next_char_boundary(text, min(to + radius, text.len()));
    let prefix = if start > 0 { "..." } else { "" };
    let suffix = if end < text.len() { "..." } else { "" };
    format!("{prefix}{}{suffix}", compact_whitespace(&text[start..end]))
}

fn compact_whitespace(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut index = 0usize;
    let mut saw_space = false;

    while index < bytes.len() {
        let byte = bytes[index];
        if byte.is_ascii_whitespace() {
            if !saw_space && !out.is_empty() {
                out.push(' ');
            }
            saw_space = true;
            index += 1;
            continue;
        }
        if byte.is_ascii() {
            out.push(byte as char);
            saw_space = false;
            index += 1;
            continue;
        }
        let ch = text[index..].chars().next().unwrap_or_default();
        if ch.is_whitespace() {
            if !saw_space && !out.is_empty() {
                out.push(' ');
            }
            saw_space = true;
        } else {
            out.push(ch);
            saw_space = false;
        }
        index += ch.len_utf8();
    }

    if out.ends_with(' ') {
        out.pop();
    }
    out
}

fn previous_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn next_char_boundary(text: &str, mut index: usize) -> usize {
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

fn severity_from_score(score: i32) -> String {
    if score >= 4 {
        "high"
    } else if score >= 2 {
        "medium"
    } else {
        "low"
    }
    .to_owned()
}

fn analyze_repetition(
    text: &str,
    tokens: &[TokenMatch],
    interner: &StringInterner,
) -> RepetitionAnalysis {
    // Single-pass: count occurrences AND collect ranges simultaneously
    let mut phrase_map = FxHashMap::<SmallVec<[u32; 5]>, PhraseAccumulator>::default();
    for size in 2..=5 {
        if tokens.len() < size {
            continue;
        }
        for index in 0..=tokens.len() - size {
            let slice = &tokens[index..index + size];
            let mut key = SmallVec::<[u32; 5]>::with_capacity(size);
            let mut content_count = 0;
            for token in slice {
                key.push(token.normalized_id);
                let normalized = interner.get(token.normalized_id);
                if normalized.len() >= 4 && !stop_words().contains(normalized) {
                    content_count += 1;
                }
            }
            if content_count < 2 {
                continue;
            }

            let acc = phrase_map.entry(key).or_insert(PhraseAccumulator {
                count: 0,
                last_start: i32::MIN / 2,
                ranges: Vec::new(),
                token_starts: Vec::new(),
            });
            // Use last_start check for counting (preserves original scoring)
            if acc.last_start >= 0 && (acc.last_start - index as i32).abs() < size as i32 {
                continue;
            }
            acc.count += 1;
            acc.last_start = index as i32;
            // Collect ranges inline using accurate overlap check
            let range_overlaps = acc
                .token_starts
                .iter()
                .any(|start| (start - index as i32).abs() < size as i32);
            if !range_overlaps {
                acc.token_starts.push(index as i32);
                acc.ranges.push(SimpleRange {
                    from: slice[0].from,
                    to: slice[slice.len() - 1].to,
                });
            }
        }
    }

    let mut scored = phrase_map
        .into_iter()
        .filter(|(_, acc)| acc.count >= 2)
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        if left.1.count == right.1.count {
            right.0.len().cmp(&left.0.len())
        } else {
            right.1.count.cmp(&left.1.count)
        }
    });
    if scored.len() > 12 {
        scored.truncate(12);
    }

    let items = scored
        .into_iter()
        .map(|(ids, acc)| {
            let highlights = ranges_to_highlights(text, &acc.ranges);
            let snippets = acc
                .ranges
                .iter()
                .take(3)
                .map(|range| build_snippet(text, range.from, range.to, 28))
                .collect::<Vec<_>>();
            let phrase = join_interned(&ids, interner);
            let score = acc.count as i32 + max(0, ids.len() as i32 - 2);
            PhraseEchoItem {
                id: format!("echo:{}", phrase.replace(' ', "-")),
                phrase,
                occurrence_count: acc.count as i32,
                severity: severity_from_score(score),
                snippets,
                highlight_ranges: highlights,
            }
        })
        .collect::<Vec<_>>();

    RepetitionAnalysis {
        total_flags: items.len() as i32,
        items,
    }
}

fn analyze_proximity(
    text: &str,
    tokens: &[TokenMatch],
    interner: &StringInterner,
) -> ProximityAnalysis {
    let mut by_root = FxHashMap::<u32, Vec<usize>>::default();
    for (index, token) in tokens.iter().enumerate() {
        let normalized = interner.get(token.normalized_id);
        let root = interner.get(token.root_id);
        if normalized.len() < 4 || stop_words().contains(normalized) || root.len() < 3 {
            continue;
        }
        by_root.entry(token.root_id).or_default().push(index);
    }

    let mut items = Vec::new();
    let mut seen = FxHashSet::<u32>::default();
    for (root_id, group) in by_root {
        if group.len() < 2 {
            continue;
        }

        let mut highlights = Vec::<SimpleRange>::new();
        let mut min_distance = i32::MAX;
        let mut best_pair: Option<(usize, usize)> = None;
        for index in 1..group.len() {
            let prev = tokens[group[index - 1]];
            let current = tokens[group[index]];
            let distance = current.index - prev.index;
            if distance > 26 {
                continue;
            }
            if distance < min_distance {
                min_distance = distance;
                best_pair = Some((group[index - 1], group[index]));
            }
            push_unique_range(
                &mut highlights,
                SimpleRange {
                    from: prev.from,
                    to: prev.to,
                },
            );
            push_unique_range(
                &mut highlights,
                SimpleRange {
                    from: current.from,
                    to: current.to,
                },
            );
        }

        let Some((left_index, right_index)) = best_pair else {
            continue;
        };
        let left = tokens[left_index];
        let right = tokens[right_index];

        seen.clear();
        let mut surface_forms = Vec::new();
        for token_index in &group {
            let token = tokens[*token_index];
            if seen.insert(token.normalized_id) {
                surface_forms.push(interner.get(token.normalized_id).to_owned());
                if surface_forms.len() == 4 {
                    break;
                }
            }
        }

        highlights.sort_by_key(|highlight| highlight.from);
        let score = max(1, 6 - min(min_distance, 5)) + max(0, highlights.len() as i32 - 2);
        items.push(ProximityConflictItem {
            id: format!("prox:{}", interner.get(root_id)),
            root: interner.get(root_id).to_owned(),
            surface_forms,
            part_of_speech: "root-family".to_owned(),
            min_word_distance: min_distance,
            severity: severity_from_score(score),
            snippets: vec![
                build_snippet(text, left.from, left.to, 28),
                build_snippet(text, right.from, right.to, 28),
            ],
            highlight_ranges: ranges_to_highlights(text, &highlights),
        });
    }

    items.sort_by(|left, right| {
        if left.min_word_distance == right.min_word_distance {
            right
                .highlight_ranges
                .len()
                .cmp(&left.highlight_ranges.len())
        } else {
            left.min_word_distance.cmp(&right.min_word_distance)
        }
    });
    if items.len() > 12 {
        items.truncate(12);
    }

    ProximityAnalysis {
        total_flags: items.len() as i32,
        items,
    }
}

fn push_unique_range(ranges: &mut Vec<SimpleRange>, next: SimpleRange) {
    if ranges
        .iter()
        .any(|range| range.from == next.from && range.to == next.to)
    {
        return;
    }
    ranges.push(next);
}

fn ranges_to_highlights(text: &str, ranges: &[SimpleRange]) -> Vec<AnalyticsHighlightRange> {
    ranges
        .iter()
        .map(|range| AnalyticsHighlightRange {
            from: range.from as i32,
            to: range.to as i32,
            text: text[range.from..range.to].to_owned(),
        })
        .collect()
}

fn join_interned(ids: &[u32], interner: &StringInterner) -> String {
    let mut out = String::new();
    for (index, id) in ids.iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(interner.get(*id));
    }
    out
}

fn extract_sentence_matches(
    text: &str,
    word_spans: &[WordSpan],
    paragraph_starts: &[usize],
) -> Vec<SentenceMatch> {
    let ranges = split_sentence_ranges(text);
    let mut result = Vec::with_capacity(ranges.len());
    let mut paragraph_index = 0usize;
    let mut word_cursor = 0usize;

    for (from, to) in ranges {
        while paragraph_index + 1 < paragraph_starts.len()
            && paragraph_starts[paragraph_index + 1] <= from
        {
            paragraph_index += 1;
        }
        while word_cursor < word_spans.len() && word_spans[word_cursor].to <= from {
            word_cursor += 1;
        }
        let mut sentence_word_count = 0i32;
        let mut count_cursor = word_cursor;
        while count_cursor < word_spans.len() && word_spans[count_cursor].from < to {
            if word_spans[count_cursor].to > from {
                sentence_word_count += 1;
            }
            count_cursor += 1;
        }
        result.push(SentenceMatch {
            from,
            to,
            paragraph_index: paragraph_index as i32,
            word_count: sentence_word_count,
        });
    }

    result
}

fn analyze_cadence(text: &str, matches: &[SentenceMatch]) -> CadenceAnalysis {
    let sentences = matches
        .iter()
        .enumerate()
        .map(|(index, sentence)| CadenceSentence {
            id: format!("sentence:{index}"),
            paragraph_index: sentence.paragraph_index,
            sentence_index: index as i32,
            from: sentence.from as i32,
            to: sentence.to as i32,
            word_count: sentence.word_count,
            bucket: get_sentence_bucket(sentence.word_count).to_owned(),
            snippet: build_snippet(text, sentence.from, sentence.to, 18),
        })
        .collect::<Vec<_>>();

    let mut hotspots = Vec::new();
    let mut run_start = 0usize;
    while run_start < sentences.len() {
        let mut run_end = run_start;
        while run_end + 1 < sentences.len()
            && (sentences[run_end + 1].word_count - sentences[run_end].word_count).abs() <= 3
        {
            run_end += 1;
        }
        if run_end - run_start + 1 >= 5 {
            let run = &sentences[run_start..=run_end];
            hotspots.push(CadenceHotspot {
                id: format!("cadence:monotony:{run_start}"),
                r#type: "monotony".to_owned(),
                label: format!("{} similar-length sentences", run.len()),
                severity: severity_from_score(run.len() as i32 - 1),
                explanation: "A long run of similarly sized sentences can flatten the rhythm."
                    .to_owned(),
                sentence_ids: run.iter().map(|sentence| sentence.id.clone()).collect(),
                highlight_ranges: run
                    .iter()
                    .map(|sentence| AnalyticsHighlightRange {
                        from: sentence.from,
                        to: sentence.to,
                        text: text[sentence.from as usize..sentence.to as usize].to_owned(),
                    })
                    .collect(),
            });
        }
        run_start = run_end + 1;
    }

    for index in 1..sentences.len() {
        let prev = &sentences[index - 1];
        let current = &sentences[index];
        let diff = (current.word_count - prev.word_count).abs();
        if diff < 12 {
            continue;
        }
        hotspots.push(CadenceHotspot {
            id: format!("cadence:whiplash:{index}"),
            r#type: "whiplash".to_owned(),
            label: format!("{} -> {} words", prev.word_count, current.word_count),
            severity: if diff >= 20 { "high" } else { "medium" }.to_owned(),
            explanation: "A sharp sentence-length jump creates a noticeable pacing snap."
                .to_owned(),
            sentence_ids: vec![prev.id.clone(), current.id.clone()],
            highlight_ranges: vec![
                AnalyticsHighlightRange {
                    from: prev.from,
                    to: prev.to,
                    text: text[prev.from as usize..prev.to as usize].to_owned(),
                },
                AnalyticsHighlightRange {
                    from: current.from,
                    to: current.to,
                    text: text[current.from as usize..current.to as usize].to_owned(),
                },
            ],
        });
    }

    if hotspots.len() > 16 {
        hotspots.truncate(16);
    }
    CadenceAnalysis {
        sentences,
        hotspots,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("..")
            .canonicalize()
            .expect("repo root")
    }

    fn stable_hash(text: &str) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for byte in text.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash
    }

    #[test]
    fn analyze_text_includes_repetition_proximity_and_cadence() {
        let text = "The iron gate slammed shut. The iron gate rattled again. The iron gate shook against the wall. \
Bright embers glowed beside the ember-lit grate. Bright embers hissed in the ash. \
Short beat. Tiny pause. Quick breath. Small shrug. Brief glance. \
This sentence suddenly stretches outward with a much more elaborate rhythm than the clipped run that came before it.";
        let result = analyze_text(text);
        assert!(!result.repetition.items.is_empty());
        assert!(result.repetition.items[0].occurrence_count >= 2);
        assert!(!result.proximity.items.is_empty());
        assert!(result.proximity.items[0].min_word_distance > 0);
        assert!(!result.cadence.sentences.is_empty());
        assert!(!result.cadence.hotspots.is_empty());
    }

    #[test]
    fn empty_analytics_has_stable_collections() {
        let result = get_empty_analytics();
        assert!(result.repetition.items.is_empty());
        assert!(result.proximity.items.is_empty());
        assert!(result.cadence.sentences.is_empty());
        assert!(result.cadence.hotspots.is_empty());
    }

    #[test]
    fn analytics_serializes_bucket_keys_like_gokitt() {
        let json = serde_json::to_string(&get_empty_analytics()).expect("serialize");
        assert!(json.contains("\"1\""));
        assert!(json.contains("\"2-6\""));
        assert!(json.contains("\"40+\""));
    }

    #[test]
    fn shortrun_analytics_golden_is_stable() {
        let text = std::fs::read_to_string(repo_root().join("docs").join("shortrun.md"))
            .expect("shortrun text");
        let analytics = analyze_text(&text);
        let json = serde_json::to_string(&analytics).expect("analytics json");
        assert_eq!(stable_hash(&json), 12411509196144750878);
    }

    #[test]
    fn perfect_run_excerpt_analytics_golden_is_stable() {
        let text = std::fs::read_to_string(repo_root().join("docs").join("perfect_run.md"))
            .expect("perfect run text");
        let excerpt_end = text
            .char_indices()
            .nth(48_000)
            .map(|(offset, _)| offset)
            .unwrap_or(text.len());
        let analytics = analyze_text(&text[..excerpt_end]);
        let json = serde_json::to_string(&analytics).expect("analytics json");
        assert_eq!(stable_hash(&json), 17804511521777736245);
    }
}
