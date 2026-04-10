use rustc_hash::FxHashSet;
use tokenizers::Encoding;

use crate::gliner_relex::{GlinerRelexEntity, GlinerRelexLabel, GlinerRelexRelation};

#[derive(Debug, Clone)]
pub struct WordSpan {
    pub start: usize,
    pub end: usize,
}

pub fn split_words(text: &str) -> Vec<WordSpan> {
    let mut words = Vec::with_capacity((text.len() / 5).max(1));
    let mut start = None::<usize>;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() {
            if let Some(word_start) = start.take() {
                words.push(WordSpan {
                    start: word_start,
                    end: index,
                });
            }
        } else if ch.is_ascii_punctuation() {
            if let Some(word_start) = start.take() {
                words.push(WordSpan {
                    start: word_start,
                    end: index,
                });
            }
            words.push(WordSpan {
                start: index,
                end: index + ch.len_utf8(),
            });
        } else if start.is_none() {
            start = Some(index);
        }
    }
    if let Some(word_start) = start {
        words.push(WordSpan {
            start: word_start,
            end: text.len(),
        });
    }
    words
}

pub fn build_prompt_tokens(
    entity_labels: &[GlinerRelexLabel],
    relation_labels: &[GlinerRelexLabel],
) -> Vec<String> {
    const ENT_TOKEN: &str = "<<ENT>>";
    const REL_TOKEN: &str = "<<REL>>";
    const SEP_TOKEN: &str = "<<SEP>>";
    let mut prompt = Vec::with_capacity(entity_labels.len() * 2 + relation_labels.len() * 2 + 2);
    for label in entity_labels {
        prompt.push(ENT_TOKEN.to_owned());
        prompt.push(render_label(label));
    }
    prompt.push(SEP_TOKEN.to_owned());
    for label in relation_labels {
        prompt.push(REL_TOKEN.to_owned());
        prompt.push(render_label(label));
    }
    prompt.push(SEP_TOKEN.to_owned());
    prompt
}

pub fn build_words_mask(encoding: &Encoding, prompt_word_count: usize) -> Vec<i64> {
    let mut mask = vec![0i64; encoding.len()];
    let mut previous_word = None::<u32>;
    let mut seen_words = 0usize;
    for (token_idx, word_id) in encoding.get_word_ids().iter().enumerate() {
        let Some(word_id) = word_id else {
            continue;
        };
        if previous_word != Some(*word_id) {
            seen_words += 1;
        }
        if seen_words > prompt_word_count && previous_word != Some(*word_id) {
            mask[token_idx] = (seen_words - prompt_word_count) as i64;
        }
        previous_word = Some(*word_id);
    }
    mask
}

fn render_label(label: &GlinerRelexLabel) -> String {
    let mut out = label.label.clone();
    if let Some(description) = label.description.as_deref().map(str::trim) {
        if !description.is_empty() {
            out.push_str(": ");
            out.push_str(description);
        }
    }
    out
}

fn trim_span_bounds(text: &str, mut start: usize, mut end: usize) -> (usize, usize) {
    while start < end {
        let ch = text[start..end].chars().next().unwrap_or_default();
        if ch.is_whitespace() || ch.is_ascii_punctuation() {
            start += ch.len_utf8();
        } else {
            break;
        }
    }
    while end > start {
        let ch = text[start..end].chars().next_back().unwrap_or_default();
        if ch.is_whitespace() || ch.is_ascii_punctuation() {
            end -= ch.len_utf8();
        } else {
            break;
        }
    }
    (start, end)
}

pub fn decode_entities(
    text: &str,
    words: &[WordSpan],
    labels: &[GlinerRelexLabel],
    threshold: f32,
    flat_ner: bool,
    multi_label: bool,
    shape: [usize; 4],
    values: &[f32],
) -> Vec<GlinerRelexEntity> {
    if shape[0] != 1 || shape[3] != 3 {
        return Vec::new();
    }
    let word_count = shape[1].min(words.len());
    let class_count = shape[2].min(labels.len());
    let raw_threshold = threshold_to_logit(threshold);
    let mut spans = Vec::with_capacity(class_count);
    let mut starts = Vec::with_capacity(word_count);
    let mut ends = Vec::with_capacity(word_count);
    for class_idx in 0..class_count {
        starts.clear();
        ends.clear();
        for word_idx in 0..word_count {
            if logits_at(values, shape, word_idx, class_idx, 0) >= raw_threshold {
                starts.push(word_idx);
            }
            if logits_at(values, shape, word_idx, class_idx, 1) >= raw_threshold {
                ends.push(word_idx);
            }
        }
        for &start in &starts {
            for &end in &ends {
                if end < start {
                    continue;
                }
                let start_logit = logits_at(values, shape, start, class_idx, 0);
                let end_logit = logits_at(values, shape, end, class_idx, 1);
                let mut score = sigmoid(start_logit).min(sigmoid(end_logit));
                let mut valid = true;
                for word_idx in start..=end {
                    let inside_logit = logits_at(values, shape, word_idx, class_idx, 2);
                    if inside_logit < raw_threshold {
                        valid = false;
                        break;
                    }
                    score = score.min(sigmoid(inside_logit));
                }
                if !valid {
                    continue;
                }
                let (trimmed_start, trimmed_end) =
                    trim_span_bounds(text, words[start].start, words[end].end);
                if trimmed_end <= trimmed_start {
                    continue;
                }
                spans.push(GlinerRelexEntity {
                    text: text[trimmed_start..trimmed_end].to_owned(),
                    label: labels[class_idx].label.clone(),
                    score,
                    start: trimmed_start,
                    end: trimmed_end,
                });
            }
        }
    }
    greedy_filter_entities(spans, flat_ner, multi_label)
}

pub fn decode_relations(
    entities: &[GlinerRelexEntity],
    labels: &[GlinerRelexLabel],
    threshold: f32,
    rel_idx_shape: [usize; 3],
    rel_idx_values: &[i64],
    rel_logits_shape: [usize; 3],
    rel_logits_values: &[f32],
    rel_mask_values: &[bool],
) -> Vec<GlinerRelexRelation> {
    if rel_idx_shape[0] != 1 || rel_logits_shape[0] != 1 {
        return Vec::new();
    }
    let pair_count = rel_idx_shape[1].min(rel_logits_shape[1]);
    let class_count = rel_logits_shape[2].min(labels.len());
    let raw_threshold = threshold_to_logit(threshold);
    let mut rows = Vec::with_capacity(pair_count.min(class_count));
    let mut seen = FxHashSet::default();
    for pair_idx in 0..pair_count {
        if !rel_mask_values.get(pair_idx).copied().unwrap_or(false) {
            continue;
        }
        let head_idx = rel_idx_values.get(pair_idx * 2).copied().unwrap_or(-1);
        let tail_idx = rel_idx_values.get(pair_idx * 2 + 1).copied().unwrap_or(-1);
        if head_idx < 0 || tail_idx < 0 {
            continue;
        }
        let head_idx = head_idx as usize;
        let tail_idx = tail_idx as usize;
        let (Some(head), Some(tail)) = (entities.get(head_idx), entities.get(tail_idx)) else {
            continue;
        };
        if head_idx == tail_idx {
            continue;
        }
        for class_idx in 0..class_count {
            let raw_score = rel_logits_values[pair_idx * rel_logits_shape[2] + class_idx];
            if raw_score < raw_threshold {
                continue;
            }
            let row = GlinerRelexRelation {
                head: head.text.clone(),
                label: labels[class_idx].label.clone(),
                tail: tail.text.clone(),
                score: sigmoid(raw_score),
                head_idx,
                tail_idx,
            };
            if seen.insert((row.head_idx, row.tail_idx, class_idx)) {
                rows.push(row);
            }
        }
    }
    rows.sort_by(cmp_relation);
    rows
}

fn logits_at(
    values: &[f32],
    shape: [usize; 4],
    word_idx: usize,
    class_idx: usize,
    slot_idx: usize,
) -> f32 {
    values[((word_idx * shape[2] + class_idx) * shape[3]) + slot_idx]
}

fn greedy_filter_entities(
    mut spans: Vec<GlinerRelexEntity>,
    flat_ner: bool,
    multi_label: bool,
) -> Vec<GlinerRelexEntity> {
    spans.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.start.cmp(&right.start))
            .then_with(|| left.end.cmp(&right.end))
    });
    let mut selected = Vec::<GlinerRelexEntity>::new();
    for span in spans {
        if !multi_label
            && selected
                .iter()
                .any(|row| row.start == span.start && row.end == span.end)
        {
            continue;
        }
        if flat_ner
            && selected
                .iter()
                .any(|row| span.start < row.end && span.end > row.start)
        {
            continue;
        }
        selected.push(span);
    }
    selected.sort_by(cmp_entity);
    selected
}

fn cmp_entity(left: &GlinerRelexEntity, right: &GlinerRelexEntity) -> std::cmp::Ordering {
    left.start
        .cmp(&right.start)
        .then_with(|| left.end.cmp(&right.end))
        .then_with(|| left.label.cmp(&right.label))
}

fn cmp_relation(left: &GlinerRelexRelation, right: &GlinerRelexRelation) -> std::cmp::Ordering {
    left.head_idx
        .cmp(&right.head_idx)
        .then_with(|| left.tail_idx.cmp(&right.tail_idx))
        .then_with(|| left.label.cmp(&right.label))
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn threshold_to_logit(threshold: f32) -> f32 {
    if threshold <= 0.0 {
        f32::NEG_INFINITY
    } else if threshold >= 1.0 {
        f32::INFINITY
    } else {
        (threshold / (1.0 - threshold)).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_supports_descriptions() {
        let prompt = build_prompt_tokens(
            &[GlinerRelexLabel {
                label: "person".to_owned(),
                description: Some("A human individual".to_owned()),
            }],
            &[GlinerRelexLabel {
                label: "works for".to_owned(),
                description: None,
            }],
        );
        assert_eq!(
            prompt,
            vec![
                "<<ENT>>".to_owned(),
                "person: A human individual".to_owned(),
                "<<SEP>>".to_owned(),
                "<<REL>>".to_owned(),
                "works for".to_owned(),
                "<<SEP>>".to_owned()
            ]
        );
    }

    #[test]
    fn split_words_matches_gliner_punctuation_boundaries() {
        let text = "Paris, France.";
        let spans = split_words(text);
        let pieces = spans
            .iter()
            .map(|span| &text[span.start..span.end])
            .collect::<Vec<_>>();
        assert_eq!(pieces, vec!["Paris", ",", "France", "."]);
    }

    #[test]
    fn decode_relations_dedupes_repeat_pairs() {
        let entities = vec![
            GlinerRelexEntity {
                text: "A".to_owned(),
                label: "person".to_owned(),
                score: 0.9,
                start: 0,
                end: 1,
            },
            GlinerRelexEntity {
                text: "B".to_owned(),
                label: "org".to_owned(),
                score: 0.8,
                start: 2,
                end: 3,
            },
        ];
        let labels = vec![GlinerRelexLabel {
            label: "works for".to_owned(),
            description: None,
        }];
        let relations = decode_relations(
            &entities,
            &labels,
            0.5,
            [1, 2, 2],
            &[0, 1, 0, 1],
            [1, 2, 1],
            &[2.0, 2.0],
            &[true, true],
        );
        assert_eq!(relations.len(), 1);
    }

    #[test]
    fn trim_span_bounds_drops_punctuation_edges() {
        assert_eq!(trim_span_bounds(" Paris, ", 0, 8), (1, 6));
    }
}
