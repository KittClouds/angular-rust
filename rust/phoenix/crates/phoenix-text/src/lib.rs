use bstr::ByteSlice;
use compact_str::CompactString;
use phoenix_types::{Sentence, SourceRange};

pub use phoenix_alex::{
    canonicalize_with_offsets, is_sentence_guard, normalize_raw, phrase_key, split_sentence_ranges,
    strip_possessive, tokenize_norm, tokens_from_normalized,
};

pub fn sentence_ranges(text: &str) -> Vec<SourceRange> {
    split_sentence_ranges(text)
        .into_iter()
        .map(|(start, end)| SourceRange::new(start as u32, end as u32))
        .collect()
}

pub fn sentence_rows(text: &str) -> Vec<Sentence> {
    sentence_ranges(text)
        .into_iter()
        .enumerate()
        .map(|(index, range)| Sentence { index, range })
        .collect()
}

pub fn paragraph_ranges(text: &str) -> Vec<SourceRange> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'\n' {
            let mut next = index;
            while next < bytes.len() && bytes[next] == b'\n' {
                next += 1;
            }
            if next - index >= 2 {
                if start < index {
                    ranges.push(SourceRange::new(start as u32, index as u32));
                }
                start = next;
                index = next;
                continue;
            }
        }
        index += 1;
    }
    if start < bytes.len() {
        ranges.push(SourceRange::new(start as u32, bytes.len() as u32));
    }
    ranges
}

pub fn bounded_excerpt(text: &str, range: SourceRange, max_bytes: usize) -> CompactString {
    let bytes = &text.as_bytes()[range.start as usize..range.end as usize];
    let excerpt = if bytes.len() <= max_bytes {
        bytes
    } else {
        &bytes[..max_bytes]
    };
    CompactString::from(excerpt.to_str_lossy().as_ref())
}
