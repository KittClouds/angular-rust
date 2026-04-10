use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use half::f16;
use ort::value::DynValue;
use serde::Deserialize;

use crate::gliclass::{GliclassClassificationType, GliclassLabelScore, GliclassPrediction};
use crate::gliclass_instruct::GliclassInstructError;
use crate::gliclass_instruct_format::GliclassInstructLabel;

pub(super) fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let payload = fs::read_to_string(path).ok()?;
    serde_json::from_str::<T>(&payload).ok()
}

pub(super) fn load_prompt_first(model_dir: &Path) -> bool {
    load_json::<serde_json::Value>(&model_dir.join("config.json"))
        .and_then(|value| {
            value
                .get("prompt_first")
                .and_then(serde_json::Value::as_bool)
        })
        .unwrap_or(true)
}

pub(super) fn load_max_length(model_dir: &Path) -> usize {
    load_json::<serde_json::Value>(&model_dir.join("tokenizer_config.json"))
        .and_then(|value| value.get("max_length").and_then(serde_json::Value::as_u64))
        .or_else(|| {
            load_json::<serde_json::Value>(&model_dir.join("tokenizer_config.json")).and_then(
                |value| {
                    value
                        .get("model_max_length")
                        .and_then(serde_json::Value::as_u64)
                },
            )
        })
        .map(|value| value as usize)
        .unwrap_or(4096)
        .clamp(32, 32768)
}

pub(super) fn validate_input(
    text: &str,
    labels: &[GliclassInstructLabel],
) -> Result<(), GliclassInstructError> {
    if text.trim().is_empty() {
        return Err(GliclassInstructError::InvalidInput(
            "text cannot be empty".to_owned(),
        ));
    }
    if labels.is_empty() {
        return Err(GliclassInstructError::InvalidInput(
            "at least one label is required".to_owned(),
        ));
    }
    Ok(())
}

pub(super) fn find_positions(values: &[i64], target: i64) -> Vec<usize> {
    values
        .iter()
        .enumerate()
        .filter_map(|(index, &value)| (value == target).then_some(index))
        .collect()
}

pub(super) fn find_text_start(encoding: &tokenizers::Encoding, text_token_index: u32) -> usize {
    encoding
        .get_ids()
        .iter()
        .position(|&value| value == text_token_index)
        .unwrap_or(encoding.len())
}

pub(super) fn build_segment_ids(sequence_len: usize, text_start: usize) -> Vec<i64> {
    (0..sequence_len)
        .map(|index| (index >= text_start) as i64)
        .collect()
}

pub(super) fn build_class_embeddings(
    hidden_states: &[f32],
    class_positions: &[usize],
    label_count: usize,
    hidden_size: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; label_count * hidden_size];
    for (index, &position) in class_positions.iter().take(label_count).enumerate() {
        let start = position * hidden_size;
        let end = start + hidden_size;
        out[index * hidden_size..(index + 1) * hidden_size]
            .copy_from_slice(&hidden_states[start..end]);
    }
    out
}

pub(super) fn extract_tensor_f32(value: &DynValue) -> Result<Vec<f32>, GliclassInstructError> {
    if let Ok(view) = value.try_extract_tensor::<f32>() {
        return view
            .as_slice()
            .map(ToOwned::to_owned)
            .ok_or_else(|| GliclassInstructError::Inference("tensor not contiguous".to_owned()));
    }
    let view = value
        .try_extract_tensor::<f16>()
        .map_err(|error| GliclassInstructError::Inference(format!("extract tensor: {error}")))?;
    let slice = view
        .as_slice()
        .ok_or_else(|| GliclassInstructError::Inference("fp16 tensor not contiguous".to_owned()))?;
    Ok(slice.iter().map(|value| value.to_f32()).collect())
}

pub(super) fn build_prediction(
    label_names: &[String],
    logits: &[f32],
    classification_type: GliclassClassificationType,
    threshold: f32,
) -> GliclassPrediction {
    let mut all_scores = match classification_type {
        GliclassClassificationType::SingleLabel => softmax_scores(label_names, logits),
        GliclassClassificationType::MultiLabel => sigmoid_scores(label_names, logits),
    };
    sort_scores_desc(&mut all_scores);
    let selected = match classification_type {
        GliclassClassificationType::SingleLabel => {
            all_scores.first().cloned().into_iter().collect()
        }
        GliclassClassificationType::MultiLabel => all_scores
            .iter()
            .filter(|row| row.score >= threshold)
            .cloned()
            .collect(),
    };
    GliclassPrediction {
        classification_type,
        threshold: (classification_type == GliclassClassificationType::MultiLabel)
            .then_some(threshold),
        selected,
        all_scores,
    }
}

fn sigmoid_scores(labels: &[String], logits: &[f32]) -> Vec<GliclassLabelScore> {
    labels
        .iter()
        .zip(logits.iter().copied())
        .map(|(label, logit)| GliclassLabelScore {
            label: label.clone(),
            logit,
            score: sigmoid(logit),
        })
        .collect()
}

fn softmax_scores(labels: &[String], logits: &[f32]) -> Vec<GliclassLabelScore> {
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = logits
        .iter()
        .map(|value| (*value - max_logit).exp())
        .collect::<Vec<_>>();
    let denom = exp.iter().sum::<f32>().max(f32::EPSILON);
    labels
        .iter()
        .zip(logits.iter().copied().zip(exp))
        .map(|(label, (logit, value))| GliclassLabelScore {
            label: label.clone(),
            logit,
            score: value / denom,
        })
        .collect()
}

fn sort_scores_desc(rows: &mut [GliclassLabelScore]) {
    rows.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.label.cmp(&right.label))
    });
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_segment_ids_flips_after_text_token() {
        assert_eq!(build_segment_ids(5, 3), vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn structured_predict_validation_rejects_empty_text() {
        let error = validate_input(
            "",
            &[GliclassInstructLabel {
                label: "space".to_owned(),
                description: None,
            }],
        )
        .expect_err("empty text should fail");
        assert!(matches!(error, GliclassInstructError::InvalidInput(_)));
    }

    #[test]
    fn build_prediction_thresholds_multilabel_scores() {
        let labels = vec!["low".to_owned(), "high".to_owned()];
        let prediction = build_prediction(
            &labels,
            &[-4.0, 4.0],
            GliclassClassificationType::MultiLabel,
            0.8,
        );
        assert_eq!(prediction.selected.len(), 1);
        assert_eq!(prediction.selected[0].label, "high");
    }
}
