use std::fs;
use std::path::{Path, PathBuf};

use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use tokenizers::{tokenizer::TruncationDirection, EncodeInput, Tokenizer};

#[derive(Debug, thiserror::Error)]
pub enum NliError {
    #[error("failed to load NLI model: {0}")]
    ModelLoad(String),
    #[error("NLI inference failed: {0}")]
    Inference(String),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NliScores {
    pub contradiction: f32,
    pub entailment: f32,
    pub neutral: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NliPairJudgment {
    pub forward: NliScores,
    pub reverse: Option<NliScores>,
    pub used_reverse: bool,
    pub best_hypothesis: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct NliExportMetadata {
    #[serde(default)]
    contradiction_idx: Option<usize>,
    #[serde(default)]
    entailment_idx: Option<usize>,
    #[serde(default)]
    neutral_idx: Option<usize>,
    #[serde(default)]
    max_length: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct NliLabelMap {
    contradiction_idx: usize,
    entailment_idx: usize,
    neutral_idx: usize,
}

impl Default for NliLabelMap {
    fn default() -> Self {
        Self {
            contradiction_idx: 0,
            entailment_idx: 1,
            neutral_idx: 2,
        }
    }
}

pub struct NliModel {
    session: Session,
    tokenizer: Tokenizer,
    labels: NliLabelMap,
    max_length: usize,
}

impl NliModel {
    pub fn load(model_dir: &Path) -> Result<Self, NliError> {
        let model_path = find_existing_path(model_dir, &["model.onnx", "onnx\\model.onnx"])?;
        let tokenizer_path =
            find_existing_path(model_dir, &["tokenizer.json", "onnx\\tokenizer.json"])?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|error| NliError::ModelLoad(error.to_string()))?;
        let session = Session::builder()
            .map_err(|error| NliError::ModelLoad(error.to_string()))?
            .with_intra_threads(1)
            .map_err(|error| NliError::ModelLoad(error.to_string()))?
            .commit_from_file(model_path)
            .map_err(|error| NliError::ModelLoad(error.to_string()))?;
        let metadata = load_metadata(model_dir);
        let labels = NliLabelMap {
            contradiction_idx: metadata
                .as_ref()
                .and_then(|value| value.contradiction_idx)
                .unwrap_or(0),
            entailment_idx: metadata
                .as_ref()
                .and_then(|value| value.entailment_idx)
                .unwrap_or(1),
            neutral_idx: metadata
                .as_ref()
                .and_then(|value| value.neutral_idx)
                .unwrap_or(2),
        };
        let max_length = metadata
            .and_then(|value| value.max_length)
            .unwrap_or(256)
            .clamp(32, 512);
        Ok(Self {
            session,
            tokenizer,
            labels,
            max_length,
        })
    }

    pub fn score(&self, premise: &str, hypothesis: &str) -> Result<NliScores, NliError> {
        let mut encoding = self
            .tokenizer
            .encode(EncodeInput::Dual(premise.into(), hypothesis.into()), true)
            .map_err(|error| NliError::Inference(error.to_string()))?;
        if encoding.len() > self.max_length {
            encoding.truncate(self.max_length, 0, TruncationDirection::Right);
        }
        let input_ids = encoding
            .get_ids()
            .iter()
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>();
        let attention_mask = encoding
            .get_attention_mask()
            .iter()
            .map(|&value| i64::from(value))
            .collect::<Vec<_>>();
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(NliError::Inference("empty tokenized sequence".to_owned()));
        }
        let ids_tensor = Tensor::from_array(([1, seq_len], input_ids))
            .map_err(|error| NliError::Inference(error.to_string()))?;
        let mask_tensor = Tensor::from_array(([1, seq_len], attention_mask))
            .map_err(|error| NliError::Inference(error.to_string()))?;
        let inputs = ort::inputs![ids_tensor, mask_tensor]
            .map_err(|error| NliError::Inference(error.to_string()))?;
        let outputs = self
            .session
            .run(inputs)
            .map_err(|error| NliError::Inference(error.to_string()))?;
        let logits = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| NliError::Inference(error.to_string()))?;
        let values = logits
            .as_slice()
            .ok_or_else(|| NliError::Inference("non-contiguous logits".to_owned()))?;
        if values.len() < 3 {
            return Err(NliError::Inference(format!(
                "expected at least 3 logits, got {}",
                values.len()
            )));
        }
        let max_logit = values[..3]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut exp = [0.0f32; 3];
        let mut sum = 0.0f32;
        for (index, value) in values[..3].iter().enumerate() {
            exp[index] = (*value - max_logit).exp();
            sum += exp[index];
        }
        if sum <= f32::EPSILON {
            return Err(NliError::Inference("softmax sum was zero".to_owned()));
        }
        let probs = [exp[0] / sum, exp[1] / sum, exp[2] / sum];
        Ok(NliScores {
            contradiction: probs[self.labels.contradiction_idx],
            entailment: probs[self.labels.entailment_idx],
            neutral: probs[self.labels.neutral_idx],
        })
    }

    pub fn judge_relation(
        &self,
        premise: &str,
        forward_hypotheses: &[String],
        reverse_hypotheses: &[String],
    ) -> Result<NliPairJudgment, NliError> {
        let (forward, forward_hypothesis) = self.best_scores(premise, forward_hypotheses)?;
        let reverse = if reverse_hypotheses.is_empty() {
            None
        } else {
            Some(self.best_scores(premise, reverse_hypotheses)?)
        };
        let use_reverse = reverse
            .as_ref()
            .map(|(scores, _)| {
                scores.entailment > forward.entailment
                    && scores.entailment - forward.entailment >= 0.005
                    && scores.contradiction <= forward.contradiction + 0.02
            })
            .unwrap_or(false);
        let best_hypothesis = if use_reverse {
            reverse
                .as_ref()
                .map(|(_, hypothesis)| hypothesis.clone())
                .unwrap_or_else(|| forward_hypothesis.clone())
        } else {
            forward_hypothesis.clone()
        };
        Ok(NliPairJudgment {
            forward,
            reverse: reverse.map(|value| value.0),
            used_reverse: use_reverse,
            best_hypothesis,
        })
    }

    fn best_scores(
        &self,
        premise: &str,
        hypotheses: &[String],
    ) -> Result<(NliScores, String), NliError> {
        let mut best_scores = None::<NliScores>;
        let mut best_hypothesis = None::<String>;
        for hypothesis in hypotheses {
            let scores = self.score(premise, hypothesis)?;
            let replace = match best_scores {
                Some(current) => {
                    scores.entailment > current.entailment
                        || (scores.entailment == current.entailment
                            && scores.contradiction < current.contradiction)
                }
                None => true,
            };
            if replace {
                best_scores = Some(scores);
                best_hypothesis = Some(hypothesis.clone());
            }
        }
        match (best_scores, best_hypothesis) {
            (Some(scores), Some(hypothesis)) => Ok((scores, hypothesis)),
            _ => Err(NliError::Inference(
                "no hypotheses were provided for NLI scoring".to_owned(),
            )),
        }
    }
}

fn find_existing_path(root: &Path, candidates: &[&str]) -> Result<PathBuf, NliError> {
    for candidate in candidates {
        let path = root.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(NliError::ModelLoad(format!(
        "missing required asset under {}",
        root.display()
    )))
}

fn load_metadata(root: &Path) -> Option<NliExportMetadata> {
    for candidate in [
        root.join("export_metadata.json"),
        root.join("onnx").join("export_metadata.json"),
    ] {
        let Ok(payload) = fs::read_to_string(&candidate) else {
            continue;
        };
        if let Ok(metadata) = serde_json::from_str::<NliExportMetadata>(&payload) {
            return Some(metadata);
        }
    }
    None
}
