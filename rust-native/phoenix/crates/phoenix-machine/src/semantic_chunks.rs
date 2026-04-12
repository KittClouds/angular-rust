use compact_str::CompactString;
use phoenix_chunker::split_sentence_ranges;
use phoenix_embed::{OrtTextEmbedError, OrtTextEmbedder};
use phoenix_types::{SentenceSpan, SurfaceUnit, SurfaceUnitKind};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct MachineSemanticChunkConfig {
    pub min_sentences: usize,
    pub max_sentences: usize,
    pub min_split_score: f32,
    pub split_deviation_weight: f32,
}

impl Default for MachineSemanticChunkConfig {
    fn default() -> Self {
        Self {
            min_sentences: 2,
            max_sentences: 8,
            min_split_score: 0.24,
            split_deviation_weight: 0.45,
        }
    }
}

#[derive(Debug, Error)]
pub enum SemanticChunkError {
    #[error(transparent)]
    Embed(#[from] OrtTextEmbedError),
    #[error("semantic chunk embedding count mismatch: expected {expected}, got {actual}")]
    EmbeddingCountMismatch { expected: usize, actual: usize },
}

pub fn build_semantic_units(
    text: &str,
    embedder: &OrtTextEmbedder,
    config: &MachineSemanticChunkConfig,
) -> Result<Vec<SurfaceUnit>, SemanticChunkError> {
    let sentences = sentence_spans(text);
    if sentences.is_empty() {
        return Ok(Vec::new());
    }

    let mut slices = Vec::with_capacity(sentences.len());
    for sentence in &sentences {
        slices.push(
            text.get(sentence.range.start as usize..sentence.range.end as usize)
                .unwrap_or_default()
                .trim(),
        );
    }
    let embeddings = embedder.embed_slices(&slices)?;
    build_semantic_units_from_embeddings(&sentences, &embeddings, config)
}

pub fn build_semantic_units_from_embeddings(
    sentences: &[SentenceSpan],
    embeddings: &[Vec<f32>],
    config: &MachineSemanticChunkConfig,
) -> Result<Vec<SurfaceUnit>, SemanticChunkError> {
    if sentences.len() != embeddings.len() {
        return Err(SemanticChunkError::EmbeddingCountMismatch {
            expected: sentences.len(),
            actual: embeddings.len(),
        });
    }
    if sentences.is_empty() {
        return Ok(Vec::new());
    }
    if sentences.len() == 1 {
        return Ok(vec![unit_for_group(sentences, 0, 1, 0)]);
    }

    let config = normalized_config(config);
    let boundaries = smoothed_boundary_scores(embeddings);
    let threshold = split_threshold(&boundaries, &config);
    let mut groups = initial_groups(sentences.len(), &boundaries, threshold, &config);
    merge_short_groups(&mut groups, &boundaries, &config);
    split_large_groups(&mut groups, &boundaries, &config);

    let mut units = Vec::with_capacity(groups.len());
    for (ordinal, (start, end)) in groups.into_iter().enumerate() {
        units.push(unit_for_group(sentences, start, end, ordinal));
    }
    Ok(units)
}

fn sentence_spans(text: &str) -> Vec<SentenceSpan> {
    let mut spans = Vec::new();
    for (index, (start, end)) in split_sentence_ranges(text).into_iter().enumerate() {
        spans.push(SentenceSpan {
            index,
            range: super::to_range(start, end),
        });
    }
    if spans.is_empty() && !text.trim().is_empty() {
        let start = text.len().saturating_sub(text.trim_start().len());
        let end = text.trim_end().len();
        spans.push(SentenceSpan {
            index: 0,
            range: super::to_range(start, end),
        });
    }
    spans
}

fn normalized_config(config: &MachineSemanticChunkConfig) -> MachineSemanticChunkConfig {
    let min_sentences = config.min_sentences.max(1);
    let max_sentences = config.max_sentences.max(min_sentences);
    MachineSemanticChunkConfig {
        min_sentences,
        max_sentences,
        min_split_score: config.min_split_score.max(0.0),
        split_deviation_weight: config.split_deviation_weight.max(0.0),
    }
}

fn smoothed_boundary_scores(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.len() < 2 {
        return Vec::new();
    }

    let mut raw = Vec::with_capacity(embeddings.len() - 1);
    for pair in embeddings.windows(2) {
        raw.push(1.0 - cosine_similarity(&pair[0], &pair[1]));
    }

    let mut smoothed = Vec::with_capacity(raw.len());
    for index in 0..raw.len() {
        let mut total = raw[index] * 2.0;
        let mut weight = 2.0;
        if index > 0 {
            total += raw[index - 1];
            weight += 1.0;
        }
        if index + 1 < raw.len() {
            total += raw[index + 1];
            weight += 1.0;
        }
        smoothed.push(total / weight);
    }
    smoothed
}

fn split_threshold(boundaries: &[f32], config: &MachineSemanticChunkConfig) -> f32 {
    if boundaries.is_empty() {
        return config.min_split_score;
    }
    let mean = boundaries.iter().copied().sum::<f32>() / boundaries.len() as f32;
    let variance = boundaries
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f32>()
        / boundaries.len() as f32;
    config
        .min_split_score
        .max(mean + variance.sqrt() * config.split_deviation_weight)
}

fn initial_groups(
    sentence_count: usize,
    boundaries: &[f32],
    threshold: f32,
    config: &MachineSemanticChunkConfig,
) -> Vec<(usize, usize)> {
    let mut groups = Vec::new();
    let mut start = 0usize;

    for boundary_index in 0..boundaries.len() {
        let current_len = boundary_index + 1 - start;
        let hard_split = current_len >= config.max_sentences;
        let semantic_split = current_len >= config.min_sentences
            && boundaries[boundary_index] >= threshold
            && is_local_peak(boundaries, boundary_index);
        if hard_split || semantic_split {
            groups.push((start, boundary_index + 1));
            start = boundary_index + 1;
        }
    }
    if start < sentence_count {
        groups.push((start, sentence_count));
    }
    groups
}

fn merge_short_groups(
    groups: &mut Vec<(usize, usize)>,
    boundaries: &[f32],
    config: &MachineSemanticChunkConfig,
) {
    while let Some(index) = groups
        .iter()
        .position(|(start, end)| end.saturating_sub(*start) < config.min_sentences)
    {
        if groups.len() == 1 {
            break;
        }

        let (start, end) = groups[index];
        let len = end - start;
        let prev = index.checked_sub(1);
        let next = (index + 1 < groups.len()).then_some(index + 1);
        let prev_cost = prev.map(|_| boundaries[start.saturating_sub(1)]);
        let next_cost = next.map(|_| boundaries[end.saturating_sub(1)]);

        let can_merge_prev = prev
            .map(|slot| groups[slot].1 - groups[slot].0 + len <= config.max_sentences)
            .unwrap_or(false);
        let can_merge_next = next
            .map(|slot| groups[slot].1 - groups[slot].0 + len <= config.max_sentences)
            .unwrap_or(false);

        match choose_merge_side(prev_cost, next_cost, can_merge_prev, can_merge_next) {
            MergeSide::Previous => {
                let slot = prev.expect("previous group");
                groups[slot].1 = end;
                groups.remove(index);
            }
            MergeSide::Next => {
                let slot = next.expect("next group");
                groups[slot].0 = start;
                groups.remove(index);
            }
        }
    }
}

fn split_large_groups(
    groups: &mut Vec<(usize, usize)>,
    boundaries: &[f32],
    config: &MachineSemanticChunkConfig,
) {
    let mut rebuilt = Vec::with_capacity(groups.len());
    for &(mut start, end) in groups.iter() {
        while end - start > config.max_sentences {
            let min_split = start + config.min_sentences - 1;
            let max_split = (start + config.max_sentences - 1).min(end - 1);
            let split_after = (min_split..=max_split)
                .max_by(|left, right| boundaries[*left].total_cmp(&boundaries[*right]))
                .unwrap_or(max_split);
            rebuilt.push((start, split_after + 1));
            start = split_after + 1;
        }
        rebuilt.push((start, end));
    }
    *groups = rebuilt;
}

fn choose_merge_side(
    prev_cost: Option<f32>,
    next_cost: Option<f32>,
    can_merge_prev: bool,
    can_merge_next: bool,
) -> MergeSide {
    match (can_merge_prev, can_merge_next, prev_cost, next_cost) {
        (true, true, Some(prev), Some(next)) => {
            if prev <= next {
                MergeSide::Previous
            } else {
                MergeSide::Next
            }
        }
        (true, _, _, _) => MergeSide::Previous,
        (_, true, _, _) => MergeSide::Next,
        (_, _, Some(prev), Some(next)) => {
            if prev <= next {
                MergeSide::Previous
            } else {
                MergeSide::Next
            }
        }
        (_, _, Some(_), None) => MergeSide::Previous,
        _ => MergeSide::Next,
    }
}

fn is_local_peak(boundaries: &[f32], index: usize) -> bool {
    let current = boundaries[index];
    let left = index
        .checked_sub(1)
        .and_then(|slot| boundaries.get(slot))
        .copied()
        .unwrap_or(f32::MIN);
    let right = boundaries.get(index + 1).copied().unwrap_or(f32::MIN);
    current >= left && current >= right
}

fn unit_for_group(
    sentences: &[SentenceSpan],
    start: usize,
    end: usize,
    ordinal: usize,
) -> SurfaceUnit {
    let first = &sentences[start];
    let last = &sentences[end - 1];
    SurfaceUnit {
        kind: SurfaceUnitKind::Paragraph,
        key: None,
        range: phoenix_types::SourceRange {
            start: first.range.start,
            end: last.range.end,
        },
        sentence_index: first.index,
        chunk_id_hint: Some(CompactString::from(format!("semantic:{ordinal:04}"))),
    }
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (lhs, rhs) in left.iter().zip(right.iter()) {
        dot += lhs * rhs;
        left_norm += lhs * lhs;
        right_norm += rhs * rhs;
    }
    if left_norm <= f32::EPSILON || right_norm <= f32::EPSILON {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

enum MergeSide {
    Previous,
    Next,
}

#[cfg(test)]
mod tests {
    use super::{
        build_semantic_units_from_embeddings, MachineSemanticChunkConfig, SemanticChunkError,
    };
    use phoenix_types::SentenceSpan;

    fn test_sentences(count: usize) -> Vec<SentenceSpan> {
        (0..count)
            .map(|index| SentenceSpan {
                index,
                range: phoenix_types::TextRange {
                    start: (index as u32) * 10,
                    end: (index as u32) * 10 + 9,
                },
            })
            .collect()
    }

    #[test]
    fn semantic_chunker_splits_on_large_topic_shift() {
        let sentences = test_sentences(4);
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.98, 0.02],
            vec![0.05, 0.95],
            vec![0.0, 1.0],
        ];

        let units = build_semantic_units_from_embeddings(
            &sentences,
            &embeddings,
            &MachineSemanticChunkConfig::default(),
        )
        .expect("semantic units");

        assert_eq!(units.len(), 2);
        assert_eq!(units[0].sentence_index, 0);
        assert_eq!(units[1].sentence_index, 2);
    }

    #[test]
    fn semantic_chunker_merges_single_sentence_island() {
        let sentences = test_sentences(5);
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.01],
            vec![0.0, 1.0],
            vec![0.99, 0.01],
            vec![0.98, 0.02],
        ];

        let units = build_semantic_units_from_embeddings(
            &sentences,
            &embeddings,
            &MachineSemanticChunkConfig {
                min_sentences: 2,
                max_sentences: 5,
                min_split_score: 0.18,
                split_deviation_weight: 0.15,
            },
        )
        .expect("semantic units");

        assert!(units.iter().all(|unit| {
            let span = unit.range.end.saturating_sub(unit.range.start);
            span >= 19
        }));
        assert!(units.len() <= 2);
    }

    #[test]
    fn semantic_chunker_rejects_embedding_count_mismatch() {
        let error = build_semantic_units_from_embeddings(
            &test_sentences(3),
            &vec![vec![1.0, 0.0], vec![1.0, 0.0]],
            &MachineSemanticChunkConfig::default(),
        )
        .expect_err("embedding mismatch");

        assert!(matches!(
            error,
            SemanticChunkError::EmbeddingCountMismatch {
                expected: 3,
                actual: 2
            }
        ));
    }
}
