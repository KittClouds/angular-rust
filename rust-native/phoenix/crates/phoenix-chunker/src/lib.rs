mod normalize;
mod sentence;

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

pub use normalize::{is_sentence_guard, normalize_raw};
pub use sentence::split_sentence_ranges;

#[cfg(target_arch = "wasm32")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

/// A chunk produced by the sentence-aware sliding window chunker.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Chunk {
    pub start: usize,
    pub end: usize,
}

/// Configuration for the sliding window chunker.
#[derive(Clone, Debug)]
pub struct ChunkerConfig {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self {
            chunk_size: 500,
            overlap: 100,
        }
    }
}

/// Split text into sentence-aligned byte ranges, then pack those sentences
/// into leaf chunks using a sliding window with configurable overlap.
///
/// This is a pure function — no store, no graph, no allocations beyond
/// the output vec. Identical algorithm to Graptor's `build_leaf_chunks`.
pub fn build_chunks(text: &str, config: &ChunkerConfig) -> Vec<Chunk> {
    let sentences = split_sentence_ranges(text);
    if sentences.is_empty() {
        if text.trim().is_empty() {
            return Vec::new();
        }
        return vec![Chunk {
            start: 0,
            end: text.len(),
        }];
    }

    let mut chunks = Vec::new();
    let mut window: Vec<(usize, usize)> = Vec::new();
    let mut current_len = 0usize;

    let emit = |window: &[(usize, usize)], chunks: &mut Vec<Chunk>| {
        if window.is_empty() {
            return;
        }
        chunks.push(Chunk {
            start: window[0].0,
            end: window[window.len() - 1].1,
        });
    };

    for (start, end) in sentences {
        let sentence_len = end - start;
        if current_len > 0 && current_len + sentence_len > config.chunk_size {
            emit(&window, &mut chunks);
            let mut overlap_len = 0usize;
            let mut new_window = Vec::new();
            for &(s, e) in window.iter().rev() {
                let span_len = e - s;
                if overlap_len + span_len > config.overlap {
                    break;
                }
                overlap_len += span_len;
                new_window.push((s, e));
            }
            new_window.reverse();
            window = new_window;
            current_len = overlap_len;
        }
        window.push((start, end));
        current_len += sentence_len;
    }
    emit(&window, &mut chunks);
    chunks
}

// ─── wasm-bindgen entry points ───────────────────────────────────────────────

/// WASM entry point: takes text, chunk_size and overlap,
/// returns JSON-serialized array of `{ start, end }` byte ranges.
#[wasm_bindgen]
pub fn chunk_text(text: &str, chunk_size: usize, overlap: usize) -> String {
    let config = ChunkerConfig {
        chunk_size,
        overlap,
    };
    let chunks = build_chunks(text, &config);
    serde_json::to_string(&chunks).unwrap_or_else(|_| "[]".to_owned())
}

/// WASM entry point: returns just the sentence byte ranges (no windowing).
#[wasm_bindgen]
pub fn sentence_ranges(text: &str) -> String {
    let ranges: Vec<Chunk> = split_sentence_ranges(text)
        .into_iter()
        .map(|(start, end)| Chunk { start, end })
        .collect();
    serde_json::to_string(&ranges).unwrap_or_else(|_| "[]".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_chunks_produces_sentence_aligned_windows() {
        let text = "The sun rose over the mountains. Birds began to sing. A river flowed through the valley. \
                     The village woke slowly. Smoke curled from chimneys.";
        let config = ChunkerConfig {
            chunk_size: 80,
            overlap: 30,
        };
        let chunks = build_chunks(text, &config);

        assert!(chunks.len() >= 2, "should produce multiple chunks");

        for chunk in &chunks {
            let slice = &text[chunk.start..chunk.end];
            assert!(!slice.is_empty());
            // Each chunk should end at a sentence boundary (period)
            let trimmed = slice.trim_end();
            assert!(
                trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?'),
                "chunk should end at sentence boundary: '{trimmed}'"
            );
        }
    }

    #[test]
    fn build_chunks_handles_empty_text() {
        let chunks = build_chunks("", &ChunkerConfig::default());
        assert!(chunks.is_empty());
    }

    #[test]
    fn build_chunks_handles_single_sentence() {
        let text = "Hello world.";
        let chunks = build_chunks(text, &ChunkerConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(&text[chunks[0].start..chunks[0].end], "Hello world.");
    }

    #[test]
    fn build_chunks_handles_text_without_punctuation() {
        let text = "No punctuation here just flowing text";
        let chunks = build_chunks(text, &ChunkerConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(&text[chunks[0].start..chunks[0].end], text);
    }

    #[test]
    fn wasm_chunk_text_returns_valid_json() {
        let result = chunk_text("Hello world. Goodbye moon.", 500, 100);
        let parsed: Vec<Chunk> = serde_json::from_str(&result).expect("valid JSON");
        assert_eq!(parsed.len(), 1);
    }

    #[test]
    fn wasm_sentence_ranges_returns_valid_json() {
        let result = sentence_ranges("Dr. Luffy ran. Mr. Zoro stayed.");
        let parsed: Vec<Chunk> = serde_json::from_str(&result).expect("valid JSON");
        assert_eq!(parsed.len(), 2);
    }

    #[test]
    fn overlap_produces_shared_content_between_chunks() {
        let text = "First sentence here. Second sentence follows. Third sentence now. Fourth sentence end.";
        let config = ChunkerConfig {
            chunk_size: 45,
            overlap: 25,
        };
        let chunks = build_chunks(text, &config);

        if chunks.len() >= 2 {
            let first_end = chunks[0].end;
            let second_start = chunks[1].start;
            assert!(
                second_start < first_end,
                "overlap should cause second chunk to start before first ends"
            );
        }
    }
}
