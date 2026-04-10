use std::cmp::Ordering;

use half::f16;
use ort::value::DynValue;
use tokenizers::Tokenizer;

use crate::gliner_bi::{
    GlinerBiError, GlinerBiLabelSet, GlinerBiOverlapPolicy, GlinerBiPrediction,
};

#[derive(Clone, Debug)]
pub(super) struct WordSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug)]
pub(super) struct GlinerBiTextTensors {
    pub words: Vec<WordSpan>,
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub words_mask: Vec<i64>,
}

impl GlinerBiTextTensors {
    pub fn seq_len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }
}

pub(super) fn words_splitter(text: &str) -> Vec<WordSpan> {
    let mut words = Vec::new();
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut i = 0;

    while i < chars.len() {
        let (pos, ch) = chars[i];
        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        if ch.is_alphanumeric() || ch == '_' {
            let start = pos;
            let mut end = pos + ch.len_utf8();
            i += 1;

            while i < chars.len() {
                let (next_pos, next_ch) = chars[i];
                if next_ch.is_alphanumeric() || next_ch == '_' {
                    end = next_pos + next_ch.len_utf8();
                    i += 1;
                } else if next_ch == '-' && i + 1 < chars.len() {
                    let (_, after_hyphen) = chars[i + 1];
                    if after_hyphen.is_alphanumeric() || after_hyphen == '_' {
                        end = chars[i + 1].0 + after_hyphen.len_utf8();
                        i += 2;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            words.push(WordSpan { start, end });
        } else {
            words.push(WordSpan {
                start: pos,
                end: pos + ch.len_utf8(),
            });
            i += 1;
        }
    }
    words
}

pub(super) fn build_text_tensors(
    text: &str,
    tokenizer: &Tokenizer,
    cls_id: i64,
    sep_id: i64,
) -> Result<Option<GlinerBiTextTensors>, GlinerBiError> {
    let words = words_splitter(text);
    if words.is_empty() {
        return Ok(None);
    }

    let encoding = tokenizer
        .encode(
            words
                .iter()
                .map(|word| &text[word.start..word.end])
                .collect::<Vec<_>>(),
            false,
        )
        .map_err(|error| GlinerBiError::Inference(format!("tokenize text failed: {error}")))?;
    let token_ids = encoding.get_ids();
    let mut input_ids = Vec::with_capacity(token_ids.len() + 2);
    input_ids.push(cls_id);
    input_ids.extend(token_ids.iter().map(|&id| i64::from(id)));
    input_ids.push(sep_id);

    let attention_mask = vec![1_i64; input_ids.len()];
    let mut words_mask = Vec::with_capacity(input_ids.len());
    words_mask.push(0);
    words_mask.extend(build_words_mask_from_word_ids(encoding.get_word_ids()));
    words_mask.push(0);
    Ok(Some(GlinerBiTextTensors {
        words,
        input_ids,
        attention_mask,
        words_mask,
    }))
}

fn build_words_mask_from_word_ids(word_ids: &[Option<u32>]) -> Vec<i64> {
    let mut out = Vec::with_capacity(word_ids.len());
    let mut previous = None::<u32>;
    for &word_id in word_ids {
        let marker = match word_id {
            Some(current) if previous != Some(current) => i64::from(current) + 1,
            _ => 0,
        };
        out.push(marker);
        previous = word_id;
    }
    out
}

pub(super) fn build_label_set(
    labels: &[String],
    tokenizer: &Tokenizer,
    cls_id: i64,
    sep_id: i64,
) -> Result<GlinerBiLabelSet, GlinerBiError> {
    if labels.is_empty() {
        return Err(GlinerBiError::InvalidInput(
            "at least one label is required".to_owned(),
        ));
    }

    let mut max_label_len = 0;
    let mut label_encodings = Vec::with_capacity(labels.len());
    for label in labels {
        let encoding = tokenizer
            .encode(label.as_str(), false)
            .map_err(|error| GlinerBiError::Inference(format!("tokenize label failed: {error}")))?;
        max_label_len = max_label_len.max(encoding.get_ids().len() + 2);
        label_encodings.push(encoding.get_ids().to_vec());
    }

    let mut input_ids = Vec::with_capacity(labels.len() * max_label_len);
    let mut attention_mask = Vec::with_capacity(labels.len() * max_label_len);
    for ids in label_encodings {
        input_ids.push(cls_id);
        attention_mask.push(1);
        for &id in &ids {
            input_ids.push(i64::from(id));
            attention_mask.push(1);
        }
        input_ids.push(sep_id);
        attention_mask.push(1);

        let pad_len = max_label_len - (ids.len() + 2);
        input_ids.extend(std::iter::repeat_n(0, pad_len));
        attention_mask.extend(std::iter::repeat_n(0, pad_len));
    }

    Ok(GlinerBiLabelSet {
        labels: labels.to_vec(),
        input_ids,
        attention_mask,
        max_label_len,
    })
}

pub(super) fn build_span_tensors(num_words: usize, max_width: usize) -> (Vec<i64>, Vec<bool>) {
    let num_spans = num_words * max_width;
    let mut span_idx = vec![0; num_spans * 2];
    let mut span_mask = vec![false; num_spans];

    for start_idx in 0..num_words {
        let actual_max_width = max_width.min(num_words - start_idx);
        for width in 0..actual_max_width {
            let dim = start_idx * max_width + width;
            span_idx[dim * 2] = start_idx as i64;
            span_idx[dim * 2 + 1] = (start_idx + width) as i64;
            span_mask[dim] = true;
        }
    }
    (span_idx, span_mask)
}

pub(super) fn extract_logits(value: &DynValue) -> Result<Vec<f32>, GlinerBiError> {
    if let Ok(view) = value.try_extract_tensor::<f32>() {
        return view
            .as_slice()
            .map(ToOwned::to_owned)
            .ok_or_else(|| GlinerBiError::Inference("logits f32 not contiguous".to_owned()));
    }
    let view = value
        .try_extract_tensor::<f16>()
        .map_err(|_| GlinerBiError::Inference("could not extract logits".to_owned()))?;
    let slice = view
        .as_slice()
        .ok_or_else(|| GlinerBiError::Inference("logits fp16 not contiguous".to_owned()))?;
    Ok(slice.iter().map(|value| value.to_f32()).collect())
}

pub(super) fn decode_predictions(
    text: &str,
    words: &[WordSpan],
    label_set: &GlinerBiLabelSet,
    max_width: usize,
    threshold: f32,
    overlap_policy: GlinerBiOverlapPolicy,
    logits: &[f32],
) -> Result<Vec<GlinerBiPrediction>, GlinerBiError> {
    let num_words = words.len();
    let num_labels = label_set.labels.len();
    let expected_len = num_words * max_width * num_labels;
    if logits.len() < expected_len {
        return Err(GlinerBiError::Inference(format!(
            "logits too short: expected {expected_len}, got {}",
            logits.len()
        )));
    }

    let mut predictions = Vec::new();
    for start_idx in 0..num_words {
        let actual_max_width = max_width.min(num_words - start_idx);
        for width in 0..actual_max_width {
            for label_idx in 0..num_labels {
                let offset = (start_idx * max_width + width) * num_labels + label_idx;
                let score = sigmoid(logits[offset]);
                if score < threshold {
                    continue;
                }
                let end_idx = start_idx + width;
                let start_char = words[start_idx].start;
                let end_char = words[end_idx].end;
                predictions.push(GlinerBiPrediction {
                    text: text[start_char..end_char].to_owned(),
                    label: label_set.labels[label_idx].clone(),
                    span_start: start_char,
                    span_end: end_char,
                    score,
                });
            }
        }
    }
    Ok(apply_overlap_policy(predictions, overlap_policy))
}

fn apply_overlap_policy(
    mut predictions: Vec<GlinerBiPrediction>,
    policy: GlinerBiOverlapPolicy,
) -> Vec<GlinerBiPrediction> {
    if policy == GlinerBiOverlapPolicy::KeepAll {
        predictions.sort_by(cmp_score_desc);
        return predictions;
    }

    if policy == GlinerBiOverlapPolicy::LongestThenScore {
        predictions.sort_by(|left, right| {
            span_len(right)
                .cmp(&span_len(left))
                .then_with(|| cmp_score_desc(left, right))
        });
    } else {
        predictions.sort_by(cmp_score_desc);
    }

    let mut filtered = Vec::new();
    for prediction in predictions {
        let overlap = filtered.iter().any(|row: &GlinerBiPrediction| {
            prediction.span_start < row.span_end && prediction.span_end > row.span_start
        });
        if !overlap {
            filtered.push(prediction);
        }
    }
    filtered.sort_by(cmp_score_desc);
    filtered
}

fn cmp_score_desc(left: &GlinerBiPrediction, right: &GlinerBiPrediction) -> Ordering {
    right
        .score
        .partial_cmp(&left.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.span_start.cmp(&right.span_start))
        .then_with(|| left.span_end.cmp(&right.span_end))
        .then_with(|| left.label.cmp(&right.label))
}

fn span_len(row: &GlinerBiPrediction) -> usize {
    row.span_end.saturating_sub(row.span_start)
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_keeps_hyphenated_words_and_punctuation_offsets() {
        let text = "AI-native graph, phase-5.";
        let words = words_splitter(text);
        let pieces = words
            .iter()
            .map(|word| &text[word.start..word.end])
            .collect::<Vec<_>>();
        assert_eq!(pieces, vec!["AI-native", "graph", ",", "phase-5", "."]);
    }

    #[test]
    fn overlap_policy_can_prefer_longest_span() {
        let predictions = vec![
            pred("Cristiano Ronaldo", 0, 18, 0.93),
            pred("Cristiano Ronaldo dos Santos Aveiro", 0, 36, 0.71),
        ];
        let filtered = apply_overlap_policy(predictions, GlinerBiOverlapPolicy::LongestThenScore);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].text, "Cristiano Ronaldo dos Santos Aveiro");
    }

    #[test]
    fn overlap_policy_highest_score_keeps_stronger_span() {
        let predictions = vec![pred("full name", 0, 9, 0.71), pred("name", 5, 9, 0.93)];
        let filtered = apply_overlap_policy(predictions, GlinerBiOverlapPolicy::HighestScore);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].text, "name");
    }

    #[test]
    fn words_mask_marks_first_subtoken_for_each_word() {
        let words_mask = build_words_mask_from_word_ids(&[
            Some(0),
            Some(0),
            Some(1),
            Some(1),
            Some(2),
            None,
            Some(3),
        ]);
        assert_eq!(words_mask, vec![1, 0, 2, 0, 3, 0, 4]);
    }

    fn pred(text: &str, span_start: usize, span_end: usize, score: f32) -> GlinerBiPrediction {
        GlinerBiPrediction {
            text: text.to_owned(),
            label: "person".to_owned(),
            span_start,
            span_end,
            score,
        }
    }
}
