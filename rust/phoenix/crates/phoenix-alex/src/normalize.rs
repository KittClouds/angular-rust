use std::collections::HashSet;
use std::sync::OnceLock;

use phoenix_types::{EntityKind, ScopeKey};
use stop_words::{get, LANGUAGE};

pub const MAX_PHRASE_TOKENS: usize = 4;
pub const TOK_SEP: char = '\u{0001}';

const LEGACY_STOP_WORDS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sir", "lady", "lord", "king", "queen", "the", "of", "and",
    "a", "an", "to", "in", "on", "for", "at", "by", "is", "it", "as", "be", "was", "are", "been",
    "with", "from", "into", "that", "this", "has", "have", "had", "his", "her", "its", "their",
    "chapter", "section", "profile", "profiles", "summary", "height", "species", "visuals", "vibe",
    "notes",
];

const DISCOVERY_NOISE_WORDS: &[&str] = &[
    "chapter", "gesture", "image", "images", "note", "notes", "profile", "profiles", "scene",
    "scenes", "section", "summary", "visual", "visuals",
];

const SENTENCE_GUARDS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
];

fn is_joiner(ch: char) -> bool {
    matches!(
        ch,
        '\'' | '\u{2019}'
            | '\u{2018}'
            | '-'
            | '\u{2013}'
            | '\u{2014}'
            | '\u{00B7}'
            | '.'
            | '_'
            | '/'
            | '#'
            | '&'
    )
}

fn lower_char(ch: char) -> char {
    match ch {
        '\u{2019}' | '\u{2018}' => '\'',
        '\u{2013}' | '\u{2014}' => '-',
        _ => ch.to_ascii_lowercase(),
    }
}

pub fn normalize_raw(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut last_was_space = true;

    for ch in text.chars() {
        let mapped = lower_char(ch);
        if mapped.is_ascii_alphanumeric() || mapped.is_whitespace() || is_joiner(mapped) {
            if mapped.is_whitespace() {
                if !last_was_space {
                    output.push(' ');
                    last_was_space = true;
                }
            } else {
                output.push(mapped);
                last_was_space = false;
            }
        } else if !last_was_space {
            output.push(' ');
            last_was_space = true;
        }
    }

    while output.ends_with(' ') {
        output.pop();
    }

    output
}

pub fn canonicalize_with_offsets(text: &str) -> (String, Vec<usize>) {
    let mut output = String::with_capacity(text.len());
    let mut offsets = Vec::with_capacity(text.len() + 1);
    let mut last_was_space = true;

    for (byte_offset, ch) in text.char_indices() {
        let mapped = lower_char(ch);
        if mapped.is_ascii_alphanumeric() || mapped.is_whitespace() || is_joiner(mapped) {
            if mapped.is_whitespace() {
                if !last_was_space {
                    output.push(' ');
                    offsets.push(byte_offset);
                    last_was_space = true;
                }
            } else {
                output.push(mapped);
                for _ in 0..mapped.len_utf8() {
                    offsets.push(byte_offset);
                }
                last_was_space = false;
            }
        } else if !last_was_space {
            output.push(' ');
            offsets.push(byte_offset);
            last_was_space = true;
        }
    }

    while output.ends_with(' ') {
        output.pop();
        offsets.pop();
    }

    offsets.push(text.len());
    (output, offsets)
}

pub fn is_stop_word(token: &str) -> bool {
    let token = strip_possessive(token.trim());
    !token.is_empty() && LEGACY_STOP_WORDS.contains(&token)
}

pub fn is_stop_word_with_profile(token: &str, profile: &str) -> bool {
    let token = strip_possessive(token.trim());
    if token.is_empty() {
        return false;
    }
    match normalize_stopword_profile(profile) {
        StopwordProfile::Off => false,
        StopwordProfile::Default => default_stop_words().contains(token),
    }
}

pub fn normalized_has_meaningful_token(normalized: &str, profile: &str) -> bool {
    normalized
        .split_whitespace()
        .map(strip_possessive)
        .filter(|token| !token.is_empty())
        .any(|token| !is_stop_word_with_profile(token, profile))
}

pub fn is_sentence_guard(token: &str) -> bool {
    SENTENCE_GUARDS.contains(&token)
}

pub fn strip_possessive(token: &str) -> &str {
    token
        .strip_suffix("'s")
        .or_else(|| token.strip_suffix("s'"))
        .unwrap_or(token)
}

pub fn tokens_from_normalized(normalized: &str) -> Vec<&str> {
    normalized
        .split_whitespace()
        .filter(|token| !token.is_empty() && !is_stop_word(token))
        .collect()
}

pub fn tokenize_norm(text: &str) -> Vec<String> {
    let normalized = normalize_raw(text);
    tokens_from_normalized(&normalized)
        .into_iter()
        .map(str::to_owned)
        .collect()
}

pub fn phrase_key(surface: &str) -> Option<String> {
    let normalized = normalize_raw(surface);
    if normalized.is_empty() {
        return None;
    }

    let tokens = tokens_from_normalized(&normalized);
    if tokens.is_empty() || tokens.len() > MAX_PHRASE_TOKENS {
        return None;
    }

    Some(tokens.join(&TOK_SEP.to_string()))
}

pub fn generate_auto_aliases(label: &str, kind: Option<&EntityKind>) -> Vec<String> {
    let kind = kind.cloned().unwrap_or(EntityKind::Other);
    let tokens = tokenize_norm(label);
    if tokens.len() <= 1 {
        return Vec::new();
    }

    let first = tokens[0].clone();
    let last = tokens[tokens.len() - 1].clone();
    let mut aliases = Vec::new();

    if matches!(kind, EntityKind::Character | EntityKind::Npc) {
        if last.len() >= 3 {
            aliases.push(last.clone());
        }
        if tokens.len() >= 3 && first != last {
            aliases.push(format!("{first} {last}"));
        }
        if first.len() >= 4 && first != last {
            aliases.push(first.clone());
        }
    }

    if matches!(kind, EntityKind::Faction | EntityKind::Organization) {
        let acronym = tokens
            .iter()
            .filter_map(|token| token.chars().next())
            .collect::<String>();
        if (2..=5).contains(&acronym.len()) {
            aliases.push(acronym);
        }
        if last.len() >= 4 {
            aliases.push(last);
        }
    }

    if matches!(kind, EntityKind::Location) && first.len() >= 4 {
        aliases.push(first);
    }

    aliases.sort();
    aliases.dedup();
    aliases
}

pub fn scope_matches(entry_scope: &ScopeKey, request_scope: &ScopeKey) -> bool {
    field_matches(
        entry_scope.world_id.as_ref(),
        request_scope.world_id.as_ref(),
    ) && field_matches(
        entry_scope.narrative_id.as_ref(),
        request_scope.narrative_id.as_ref(),
    ) && field_matches(
        entry_scope.folder_id.as_ref(),
        request_scope.folder_id.as_ref(),
    ) && field_matches(
        entry_scope.folder_path.as_ref(),
        request_scope.folder_path.as_ref(),
    )
}

fn field_matches(entry_value: Option<&String>, request_value: Option<&String>) -> bool {
    match entry_value {
        None => true,
        Some(entry_value) => request_value == Some(entry_value),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StopwordProfile {
    Default,
    Off,
}

fn normalize_stopword_profile(profile: &str) -> StopwordProfile {
    match profile.trim().to_ascii_lowercase().as_str() {
        "off" => StopwordProfile::Off,
        _ => StopwordProfile::Default,
    }
}

fn default_stop_words() -> &'static HashSet<&'static str> {
    static STOP_WORDS: OnceLock<HashSet<&'static str>> = OnceLock::new();
    STOP_WORDS.get_or_init(|| {
        let mut words = get(LANGUAGE::English)
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        words.extend(LEGACY_STOP_WORDS.iter().copied());
        words.extend(DISCOVERY_NOISE_WORDS.iter().copied());
        words
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_keeps_joiners_and_collapses_spaces() {
        assert_eq!(normalize_raw("Monkey D. Luffy"), "monkey d. luffy");
        assert_eq!(normalize_raw("Jean-Luc  Picard"), "jean-luc picard");
    }

    #[test]
    fn phrase_key_filters_stop_words() {
        assert_eq!(
            phrase_key("The Lord of the Rings"),
            Some("rings".to_owned())
        );
    }

    #[test]
    fn default_profile_uses_stop_words_crate_and_overlay() {
        assert!(is_stop_word_with_profile("he", "default"));
        assert!(is_stop_word_with_profile("then", "default"));
        assert!(is_stop_word_with_profile("what", "default"));
        assert!(is_stop_word_with_profile("image", "default"));
        assert!(is_stop_word_with_profile("gesture", "default"));
    }

    #[test]
    fn off_profile_disables_stop_words() {
        assert!(!is_stop_word_with_profile("he", "off"));
        assert!(normalized_has_meaningful_token("he", "off"));
    }

    #[test]
    fn meaningful_token_check_keeps_phrases_with_signal() {
        assert!(normalized_has_meaningful_token("the ember gate", "default"));
        assert!(!normalized_has_meaningful_token("the and of", "default"));
    }

    #[test]
    fn auto_aliases_cover_character_faction_and_location_cases() {
        let aliases = generate_auto_aliases("Monkey D. Luffy", Some(&EntityKind::Character));
        assert!(aliases.contains(&"luffy".to_owned()));
        assert!(aliases.contains(&"monkey luffy".to_owned()));

        let faction = generate_auto_aliases("Straw Hat Pirates", Some(&EntityKind::Faction));
        assert!(faction.contains(&"shp".to_owned()));

        let location = generate_auto_aliases("Grand Line", Some(&EntityKind::Location));
        assert!(location.contains(&"grand".to_owned()));
    }
}
