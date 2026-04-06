use memchr::memchr3_iter;

use crate::{is_sentence_guard, normalize_raw};

pub fn split_sentence_ranges(text: &str) -> Vec<(usize, usize)> {
    if text.is_empty() {
        return Vec::new();
    }

    let bytes = text.as_bytes();
    let mut ranges = Vec::new();
    let mut start = skip_ascii_whitespace(bytes, 0);

    for punctuation in memchr3_iter(b'.', b'!', b'?', bytes) {
        if punctuation < start {
            continue;
        }

        let mut token_start = punctuation;
        while token_start > start {
            let previous = bytes[token_start - 1];
            if previous.is_ascii_alphanumeric() || previous == b'\'' || previous == b'-' {
                token_start -= 1;
            } else {
                break;
            }
        }

        let mut end = punctuation + 1;
        while end < bytes.len() && matches!(bytes[end], b'.' | b'!' | b'?') {
            end += 1;
        }

        let guard = normalize_raw(text.get(token_start..end).unwrap_or_default());
        let trimmed_guard = guard.trim_end_matches('.');
        if (guard.len() <= 3 && is_sentence_guard(trimmed_guard)) || trimmed_guard.len() <= 1 {
            continue;
        }

        if end > start {
            ranges.push((start, end));
        }
        start = skip_ascii_whitespace(bytes, end);
    }

    if start < text.len() {
        ranges.push((start, text.len()));
    }

    ranges
}

fn skip_ascii_whitespace(bytes: &[u8], mut index: usize) -> usize {
    while index < bytes.len() && bytes[index].is_ascii_whitespace() {
        index += 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_sentence_ranges_respects_short_guards() {
        let text = "Dr. Luffy ran. Mr. Zoro stayed. Wow!";
        let ranges = split_sentence_ranges(text);
        assert_eq!(ranges.len(), 3);
        assert_eq!(&text[ranges[0].0..ranges[0].1], "Dr. Luffy ran.");
        assert_eq!(&text[ranges[1].0..ranges[1].1], "Mr. Zoro stayed.");
        assert_eq!(&text[ranges[2].0..ranges[2].1], "Wow!");
    }

    #[test]
    fn split_sentence_ranges_handles_empty_input() {
        assert!(split_sentence_ranges("").is_empty());
    }
}
