//! Stable public entrypoints for sentence splitting and chunk windowing.
//!
//! This stage owns only text segmentation. It expects raw text and returns
//! sentence or chunk ranges. It persists nothing and should be the preferred
//! entrypoint when callers want chunking without pulling in the rest of the
//! pipeline.

use crate::{build_chunks, split_sentence_ranges, Chunk, ChunkerConfig};

pub fn sentence_ranges(text: &str) -> Vec<(usize, usize)> {
    split_sentence_ranges(text)
}

pub fn chunk_ranges(text: &str, config: &ChunkerConfig) -> Vec<Chunk> {
    build_chunks(text, config)
}

pub fn default_chunk_ranges(text: &str) -> Vec<Chunk> {
    build_chunks(text, &ChunkerConfig::default())
}
