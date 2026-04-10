use std::cmp::Ordering;
use std::fs;
use std::path::{Path, PathBuf};

use half::f16;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::{tokenizer::TruncationDirection, EncodeInput, Tokenizer};

const LABEL_TOKEN: &str = "<<LABEL>>";
const SEP_TOKEN: &str = "<<SEP>>";

#[derive(Debug, Error)]
pub enum GliclassError {
    #[error("failed to load GLiClass model: {0}")]
    ModelLoad(String),
    #[error("GLiClass inference failed: {0}")]
    Inference(String),
    #[error("invalid GLiClass input: {0}")]
    InvalidInput(String),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GliclassClassificationType {
    SingleLabel,
    MultiLabel,
}

impl Default for GliclassClassificationType {
    fn default() -> Self {
        Self::MultiLabel
    }
}

impl GliclassClassificationType {
    pub fn parse(value: &str) -> Result<Self, GliclassError> {
        match value.trim().to_ascii_lowercase().as_str() {
            "single" | "single-label" | "single_label" => Ok(Self::SingleLabel),
            "multi" | "multi-label" | "multi_label" => Ok(Self::MultiLabel),
            other => Err(GliclassError::InvalidInput(format!(
                "unsupported classification type '{other}'"
            ))),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GliclassLabelScore {
    pub label: String,
    pub logit: f32,
    pub score: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GliclassPrediction {
    pub classification_type: GliclassClassificationType,
    pub threshold: Option<f32>,
    pub selected: Vec<GliclassLabelScore>,
    pub all_scores: Vec<GliclassLabelScore>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GliclassModelMetadata {
    pub architecture_type: String,
    pub prompt_first: bool,
    pub max_num_classes: usize,
    pub max_length: usize,
    pub model_path: String,
    pub tokenizer_path: String,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub primary_output_name: String,
}

#[derive(Clone, Debug, Default)]
pub struct GliclassPredictOptions {
    pub classification_type: GliclassClassificationType,
    pub threshold: f32,
    pub prompt: Option<String>,
    pub label_chunk_size: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct GliclassRootConfig {
    #[serde(default)]
    architecture_type: String,
    #[serde(default)]
    prompt_first: bool,
    #[serde(default)]
    max_num_classes: Option<usize>,
    #[serde(default)]
    encoder_config: Option<GliclassEncoderConfig>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct GliclassEncoderConfig {
    #[serde(default)]
    max_position_embeddings: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct GliclassTokenizerConfig {
    #[serde(default)]
    model_max_length: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct GliclassOnnxConfig {
    #[serde(default)]
    architecture_type: String,
    #[serde(default)]
    prompt_first: bool,
}

pub struct GliclassModel {
    session: Session,
    tokenizer: Tokenizer,
    prompt_first: bool,
    max_num_classes: usize,
    max_length: usize,
    primary_output_name: String,
    metadata: GliclassModelMetadata,
}

impl GliclassModel {
    pub fn load(model_dir: &Path) -> Result<Self, GliclassError> {
        let model_path = find_existing_path(
            model_dir,
            &[
                "model_quantized.onnx",
                "onnx\\model_quantized.onnx",
                "model.onnx",
                "onnx\\model.onnx",
                "onnx\\model-int8-quantized.onnx",
                "onnx\\model-uint8-quantized.onnx",
                "onnx\\model-int4-quantized.onnx",
                "onnx\\model-uint4-quantized.onnx",
            ],
        )?;
        let tokenizer_path =
            find_existing_path(model_dir, &["tokenizer.json", "onnx\\tokenizer.json"])?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| GliclassError::ModelLoad(format!("tokenizer: {error}")))?;
        let session = Session::builder()
            .and_then(|builder| builder.with_intra_threads(1))
            .and_then(|builder| builder.commit_from_file(&model_path))
            .map_err(|error| GliclassError::ModelLoad(format!("session: {error}")))?;

        let root_config =
            load_json::<GliclassRootConfig>(&model_dir.join("config.json")).unwrap_or_default();
        let onnx_config =
            load_json::<GliclassOnnxConfig>(&model_dir.join("onnx").join("config.json"))
                .unwrap_or_default();
        let tokenizer_config =
            load_json::<GliclassTokenizerConfig>(&model_dir.join("tokenizer_config.json"))
                .or_else(|| {
                    load_json::<GliclassTokenizerConfig>(
                        &model_dir.join("onnx").join("tokenizer_config.json"),
                    )
                })
                .unwrap_or_default();

        let architecture_type = if !root_config.architecture_type.is_empty() {
            root_config.architecture_type.clone()
        } else if !onnx_config.architecture_type.is_empty() {
            onnx_config.architecture_type.clone()
        } else {
            "uni-encoder".to_owned()
        };
        if architecture_type != "uni-encoder" {
            return Err(GliclassError::ModelLoad(format!(
                "unsupported GLiClass architecture '{architecture_type}', only uni-encoder is supported"
            )));
        }

        let prompt_first = if root_config.prompt_first {
            true
        } else {
            onnx_config.prompt_first
        };
        let max_num_classes = root_config.max_num_classes.unwrap_or(25).clamp(1, 256);
        let max_length = tokenizer_config
            .model_max_length
            .or_else(|| {
                root_config
                    .encoder_config
                    .as_ref()
                    .and_then(|config| config.max_position_embeddings)
            })
            .unwrap_or(8192)
            .clamp(32, 32768);
        let input_names = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect::<Vec<_>>();
        let output_names = session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect::<Vec<_>>();
        let primary_output_name = output_names
            .iter()
            .find(|name| name.as_str() == "logits")
            .cloned()
            .or_else(|| output_names.first().cloned())
            .ok_or_else(|| GliclassError::ModelLoad("model reported no outputs".to_owned()))?;

        let metadata = GliclassModelMetadata {
            architecture_type,
            prompt_first,
            max_num_classes,
            max_length,
            model_path: model_path.display().to_string(),
            tokenizer_path: tokenizer_path.display().to_string(),
            input_names,
            output_names,
            primary_output_name: primary_output_name.clone(),
        };

        Ok(Self {
            session,
            tokenizer,
            prompt_first,
            max_num_classes,
            max_length,
            primary_output_name,
            metadata,
        })
    }

    pub fn metadata(&self) -> &GliclassModelMetadata {
        &self.metadata
    }

    pub fn predict(
        &self,
        text: &str,
        labels: &[String],
        options: &GliclassPredictOptions,
    ) -> Result<GliclassPrediction, GliclassError> {
        if text.trim().is_empty() {
            return Err(GliclassError::InvalidInput(
                "text cannot be empty".to_owned(),
            ));
        }
        if labels.is_empty() {
            return Err(GliclassError::InvalidInput(
                "at least one label is required".to_owned(),
            ));
        }

        let chunk_size = options
            .label_chunk_size
            .unwrap_or(self.max_num_classes)
            .clamp(1, self.max_num_classes);
        let mut logits = Vec::with_capacity(labels.len());
        for chunk in labels.chunks(chunk_size) {
            let input = self.prepare_input(text, chunk, options.prompt.as_deref());
            logits.extend(self.run_logits(&input, chunk.len())?);
        }

        if logits.len() != labels.len() {
            return Err(GliclassError::Inference(format!(
                "expected {} logits, got {}",
                labels.len(),
                logits.len()
            )));
        }

        let mut all_scores = match options.classification_type {
            GliclassClassificationType::SingleLabel => softmax_scores(labels, &logits),
            GliclassClassificationType::MultiLabel => sigmoid_scores(labels, &logits),
        };
        sort_scores_desc(&mut all_scores);

        let selected = match options.classification_type {
            GliclassClassificationType::SingleLabel => {
                all_scores.first().cloned().into_iter().collect()
            }
            GliclassClassificationType::MultiLabel => all_scores
                .iter()
                .filter(|row| row.score >= options.threshold)
                .cloned()
                .collect(),
        };

        Ok(GliclassPrediction {
            classification_type: options.classification_type,
            threshold: match options.classification_type {
                GliclassClassificationType::SingleLabel => None,
                GliclassClassificationType::MultiLabel => Some(options.threshold),
            },
            selected,
            all_scores,
        })
    }

    fn prepare_input(&self, text: &str, labels: &[String], prompt: Option<&str>) -> String {
        build_input(self.prompt_first, text, labels, prompt)
    }

    fn run_logits(&self, input: &str, label_count: usize) -> Result<Vec<f32>, GliclassError> {
        let mut encoding = self
            .tokenizer
            .encode(EncodeInput::Single(input.into()), true)
            .map_err(|error| GliclassError::Inference(format!("tokenize input: {error}")))?;
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
        let sequence_len = input_ids.len();
        if sequence_len == 0 {
            return Err(GliclassError::Inference(
                "tokenizer returned an empty sequence".to_owned(),
            ));
        }

        let ids_tensor = Tensor::from_array(([1, sequence_len], input_ids))
            .map_err(|error| GliclassError::Inference(format!("input_ids tensor: {error}")))?;
        let mask_tensor = Tensor::from_array(([1, sequence_len], attention_mask))
            .map_err(|error| GliclassError::Inference(format!("attention_mask tensor: {error}")))?;
        let inputs = ort::inputs! {
            "input_ids" => ids_tensor,
            "attention_mask" => mask_tensor,
        }
        .map_err(|error| GliclassError::Inference(format!("session inputs: {error}")))?;
        let outputs = self
            .session
            .run(inputs)
            .map_err(|error| GliclassError::Inference(format!("session run: {error}")))?;
        let has_outputs = outputs.keys().next().is_some();
        let output = outputs
            .get(&self.primary_output_name)
            .or_else(|| outputs.get("logits"))
            .or_else(|| has_outputs.then(|| &outputs[0]))
            .ok_or_else(|| GliclassError::Inference("model returned no outputs".to_owned()))?;

        let values = if let Ok(view) = output.try_extract_tensor::<f32>() {
            view.as_slice().map(ToOwned::to_owned).ok_or_else(|| {
                GliclassError::Inference("logits tensor not contiguous".to_owned())
            })?
        } else {
            let view = output.try_extract_tensor::<f16>().map_err(|error| {
                GliclassError::Inference(format!("extract logits tensor: {error}"))
            })?;
            let slice = view.as_slice().ok_or_else(|| {
                GliclassError::Inference("fp16 logits tensor not contiguous".to_owned())
            })?;
            slice.iter().map(|value| value.to_f32()).collect::<Vec<_>>()
        };

        if values.len() < label_count {
            return Err(GliclassError::Inference(format!(
                "expected at least {label_count} logits, got {}",
                values.len()
            )));
        }
        Ok(values.into_iter().take(label_count).collect())
    }
}

fn find_existing_path(root: &Path, candidates: &[&str]) -> Result<PathBuf, GliclassError> {
    for candidate in candidates {
        let path = root.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(GliclassError::ModelLoad(format!(
        "missing required asset under {}",
        root.display()
    )))
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let payload = fs::read_to_string(path).ok()?;
    serde_json::from_str::<T>(&payload).ok()
}

fn build_input(prompt_first: bool, text: &str, labels: &[String], prompt: Option<&str>) -> String {
    let prompt = prompt
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_default();
    let label_bytes = labels
        .iter()
        .map(|label| LABEL_TOKEN.len() + label.len())
        .sum::<usize>();
    let prompt_bytes = if prompt.is_empty() {
        0
    } else {
        prompt.len() + 1
    };
    let suffix_capacity = label_bytes + SEP_TOKEN.len() + prompt_bytes;
    let mut suffix = String::with_capacity(suffix_capacity);
    for label in labels {
        suffix.push_str(LABEL_TOKEN);
        suffix.push_str(label);
    }
    suffix.push_str(SEP_TOKEN);
    if !prompt.is_empty() {
        suffix.push_str(prompt);
        suffix.push(' ');
    }

    let mut input = String::with_capacity(text.len() + suffix.len());
    if prompt_first {
        input.push_str(&suffix);
        input.push_str(text);
    } else {
        input.push_str(text);
        input.push_str(&suffix);
    }
    input
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
    let mut exp = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for logit in logits {
        let value = (*logit - max_logit).exp();
        exp.push(value);
        sum += value;
    }
    let denom = if sum <= f32::EPSILON { 1.0 } else { sum };
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
    use std::fs;
    use std::path::PathBuf;

    #[test]
    fn parse_classification_type_accepts_common_aliases() {
        assert_eq!(
            GliclassClassificationType::parse("multi-label").expect("multi"),
            GliclassClassificationType::MultiLabel
        );
        assert_eq!(
            GliclassClassificationType::parse("single").expect("single"),
            GliclassClassificationType::SingleLabel
        );
    }

    #[test]
    fn prepare_input_matches_prompt_first_contract() {
        let labels = vec!["topic.product".to_owned(), "sentiment.positive".to_owned()];
        let input = build_input(
            true,
            "The product quality is amazing.",
            &labels,
            Some("Classify this review:"),
        );
        assert_eq!(
            input,
            "<<LABEL>>topic.product<<LABEL>>sentiment.positive<<SEP>>Classify this review: The product quality is amazing."
        );
    }

    #[test]
    fn softmax_scores_normalize_all_logits() {
        let labels = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let rows = softmax_scores(&labels, &[1.0, 3.0, 2.0]);
        let total = rows.iter().map(|row| row.score).sum::<f32>();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sigmoid_scores_follow_logit_order() {
        let labels = vec!["low".to_owned(), "high".to_owned()];
        let rows = sigmoid_scores(&labels, &[-3.0, 3.0]);
        assert!(rows[0].score < rows[1].score);
    }

    #[test]
    fn find_existing_path_prefers_root_quantized_model() {
        let root = unique_test_dir("gliclass-path-preference");
        fs::create_dir_all(root.join("onnx")).expect("create test tree");
        fs::write(root.join("model_quantized.onnx"), b"root-quantized").expect("root model");
        fs::write(root.join("onnx").join("model.onnx"), b"nested-model").expect("nested model");
        let resolved = find_existing_path(&root, &["model_quantized.onnx", "onnx\\model.onnx"])
            .expect("resolve model path");
        assert_eq!(resolved, root.join("model_quantized.onnx"));
        let _ = fs::remove_dir_all(root);
    }

    fn unique_test_dir(label: &str) -> PathBuf {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("unix time")
            .as_nanos();
        std::env::temp_dir().join(format!("phoenix-rel-post-{label}-{stamp}"))
    }
}
