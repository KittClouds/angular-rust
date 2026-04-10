use std::path::Path;

use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::{tokenizer::TruncationDirection, EncodeInput, Tokenizer};

use crate::gliclass::{GliclassClassificationType, GliclassPrediction};
use crate::gliclass_instruct_format::{
    build_input, GliclassInstructExample, GliclassInstructLabel,
};
use crate::gliclass_instruct_runtime::{
    build_class_embeddings, build_prediction, build_segment_ids, extract_tensor_f32,
    find_positions, find_text_start, load_json, load_max_length, load_prompt_first, validate_input,
};
use crate::ort_runtime::load_session;

#[derive(Debug, Error)]
pub enum GliclassInstructError {
    #[error("failed to load GLiClass-instruct model: {0}")]
    ModelLoad(String),
    #[error("GLiClass-instruct inference failed: {0}")]
    Inference(String),
    #[error("invalid GLiClass-instruct input: {0}")]
    InvalidInput(String),
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct GliclassInstructConfig {
    pub class_token_index: u32,
    pub text_token_index: u32,
    #[serde(default)]
    pub use_segment_embeddings: bool,
    pub hidden_size: usize,
    #[serde(default)]
    pub architecture_type: String,
    #[serde(default)]
    pub onnx_files: OnnxFiles,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct OnnxFiles {
    pub encoder: String,
    pub projectors: String,
    pub scorer: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GliclassInstructMetadata {
    pub architecture_type: String,
    pub hidden_size: usize,
    pub class_token_index: u32,
    pub text_token_index: u32,
    pub prompt_first: bool,
    pub max_length: usize,
    pub use_segment_embeddings: bool,
    pub encoder_path: String,
    pub projectors_path: String,
    pub scorer_path: String,
    pub tokenizer_path: String,
    pub encoder_inputs: Vec<String>,
    pub encoder_outputs: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct GliclassInstructPredictOptions {
    pub classification_type: GliclassClassificationType,
    pub threshold: f32,
    pub prompt: Option<String>,
    pub examples: Vec<GliclassInstructExample>,
}

impl Default for GliclassInstructPredictOptions {
    fn default() -> Self {
        Self {
            classification_type: GliclassClassificationType::MultiLabel,
            threshold: 0.5,
            prompt: None,
            examples: Vec::new(),
        }
    }
}

pub struct GliclassInstructModel {
    encoder: Session,
    projectors: Session,
    scorer: Session,
    tokenizer: Tokenizer,
    config: GliclassInstructConfig,
    prompt_first: bool,
    max_length: usize,
    need_segment_ids: bool,
    metadata: GliclassInstructMetadata,
}

impl GliclassInstructModel {
    pub fn load(model_dir: &Path) -> Result<Self, GliclassInstructError> {
        let config = load_json::<GliclassInstructConfig>(&model_dir.join("gliclass_config.json"))
            .ok_or_else(|| {
            GliclassInstructError::ModelLoad("missing gliclass_config.json".to_owned())
        })?;
        let encoder_path = model_dir.join(&config.onnx_files.encoder);
        let projectors_path = model_dir.join(&config.onnx_files.projectors);
        let scorer_path = model_dir.join(&config.onnx_files.scorer);
        let tokenizer_path = model_dir.join("tokenizer.json");
        let encoder = load_session(&encoder_path).map_err(|error| {
            GliclassInstructError::ModelLoad(format!("session {}: {error}", encoder_path.display()))
        })?;
        let projectors = load_session(&projectors_path).map_err(|error| {
            GliclassInstructError::ModelLoad(format!(
                "session {}: {error}",
                projectors_path.display()
            ))
        })?;
        let scorer = load_session(&scorer_path).map_err(|error| {
            GliclassInstructError::ModelLoad(format!("session {}: {error}", scorer_path.display()))
        })?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| GliclassInstructError::ModelLoad(format!("tokenizer: {error}")))?;
        let prompt_first = load_prompt_first(model_dir);
        let max_length = load_max_length(model_dir);
        let need_segment_ids = encoder
            .inputs
            .iter()
            .any(|input| input.name == "segment_ids");
        let metadata = GliclassInstructMetadata {
            architecture_type: config.architecture_type.clone(),
            hidden_size: config.hidden_size,
            class_token_index: config.class_token_index,
            text_token_index: config.text_token_index,
            prompt_first,
            max_length,
            use_segment_embeddings: config.use_segment_embeddings && need_segment_ids,
            encoder_path: encoder_path.display().to_string(),
            projectors_path: projectors_path.display().to_string(),
            scorer_path: scorer_path.display().to_string(),
            tokenizer_path: tokenizer_path.display().to_string(),
            encoder_inputs: encoder
                .inputs
                .iter()
                .map(|input| input.name.clone())
                .collect(),
            encoder_outputs: encoder
                .outputs
                .iter()
                .map(|output| output.name.clone())
                .collect(),
        };
        Ok(Self {
            encoder,
            projectors,
            scorer,
            tokenizer,
            config,
            prompt_first,
            max_length,
            need_segment_ids,
            metadata,
        })
    }

    pub fn metadata(&self) -> &GliclassInstructMetadata {
        &self.metadata
    }

    pub fn predict(
        &self,
        text: &str,
        labels: &[String],
        options: &GliclassInstructPredictOptions,
    ) -> Result<GliclassPrediction, GliclassInstructError> {
        let labels = labels
            .iter()
            .cloned()
            .map(|label| GliclassInstructLabel {
                label,
                description: None,
            })
            .collect::<Vec<_>>();
        self.predict_structured(text, &labels, options)
    }

    pub fn predict_structured(
        &self,
        text: &str,
        labels: &[GliclassInstructLabel],
        options: &GliclassInstructPredictOptions,
    ) -> Result<GliclassPrediction, GliclassInstructError> {
        validate_input(text, labels)?;
        let input = build_input(
            self.prompt_first,
            text,
            labels,
            &options.examples,
            options.prompt.as_deref(),
        );
        let label_names = labels
            .iter()
            .map(|label| label.label.clone())
            .collect::<Vec<_>>();
        let logits = self.run_decomposed(&input, labels.len())?;
        Ok(build_prediction(
            &label_names,
            &logits,
            options.classification_type,
            options.threshold,
        ))
    }

    fn run_decomposed(
        &self,
        input: &str,
        label_count: usize,
    ) -> Result<Vec<f32>, GliclassInstructError> {
        let mut encoding = self
            .tokenizer
            .encode(EncodeInput::Single(input.into()), true)
            .map_err(|error| GliclassInstructError::Inference(format!("tokenize: {error}")))?;
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
            return Err(GliclassInstructError::Inference(
                "tokenizer returned an empty sequence".to_owned(),
            ));
        }
        let class_positions = find_positions(&input_ids, i64::from(self.config.class_token_index));
        if class_positions.len() < label_count {
            return Err(GliclassInstructError::Inference(format!(
                "expected at least {label_count} class tokens, found {}",
                class_positions.len()
            )));
        }
        let ids_tensor = Tensor::from_array(([1, sequence_len], input_ids))
            .map_err(|error| GliclassInstructError::Inference(format!("input_ids: {error}")))?;
        let mask_tensor =
            Tensor::from_array(([1, sequence_len], attention_mask)).map_err(|error| {
                GliclassInstructError::Inference(format!("attention_mask: {error}"))
            })?;
        let encoder_outputs = if self.need_segment_ids {
            let segment_ids = build_segment_ids(
                sequence_len,
                find_text_start(&encoding, self.config.text_token_index),
            );
            let seg_tensor =
                Tensor::from_array(([1, sequence_len], segment_ids)).map_err(|error| {
                    GliclassInstructError::Inference(format!("segment_ids: {error}"))
                })?;
            let inputs = ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
                "segment_ids" => seg_tensor,
            }
            .map_err(|error| {
                GliclassInstructError::Inference(format!("encoder inputs: {error}"))
            })?;
            self.encoder.run(inputs).map_err(|error| {
                GliclassInstructError::Inference(format!("encoder run: {error}"))
            })?
        } else {
            let inputs = ort::inputs! {
                "input_ids" => ids_tensor,
                "attention_mask" => mask_tensor,
            }
            .map_err(|error| {
                GliclassInstructError::Inference(format!("encoder inputs: {error}"))
            })?;
            self.encoder.run(inputs).map_err(|error| {
                GliclassInstructError::Inference(format!("encoder run: {error}"))
            })?
        };
        let hidden_states = extract_tensor_f32(&encoder_outputs[0])?;
        if hidden_states.len() != sequence_len * self.config.hidden_size {
            return Err(GliclassInstructError::Inference(format!(
                "encoder output shape mismatch: expected {} values, got {}",
                sequence_len * self.config.hidden_size,
                hidden_states.len()
            )));
        }
        let text_embedding = hidden_states[..self.config.hidden_size].to_vec();
        let class_embeddings = build_class_embeddings(
            &hidden_states,
            &class_positions,
            label_count,
            self.config.hidden_size,
        );
        let text_tensor = Tensor::from_array(([1, self.config.hidden_size], text_embedding))
            .map_err(|error| {
                GliclassInstructError::Inference(format!("text embedding: {error}"))
            })?;
        let class_tensor =
            Tensor::from_array(([1, label_count, self.config.hidden_size], class_embeddings))
                .map_err(|error| {
                    GliclassInstructError::Inference(format!("class embeddings: {error}"))
                })?;
        let proj_inputs = ort::inputs! {
            "text_embedding" => text_tensor,
            "class_embeddings" => class_tensor,
        }
        .map_err(|error| GliclassInstructError::Inference(format!("projector inputs: {error}")))?;
        let proj_outputs = self.projectors.run(proj_inputs).map_err(|error| {
            GliclassInstructError::Inference(format!("projectors run: {error}"))
        })?;
        let scorer_inputs = vec![
            ("text_projected", proj_outputs[0].view()),
            ("class_projected", proj_outputs[1].view()),
        ];
        let scorer_outputs = self
            .scorer
            .run(scorer_inputs)
            .map_err(|error| GliclassInstructError::Inference(format!("scorer run: {error}")))?;
        let logits = extract_tensor_f32(&scorer_outputs[0])?;
        if logits.len() < label_count {
            return Err(GliclassInstructError::Inference(format!(
                "expected at least {label_count} logits, got {}",
                logits.len()
            )));
        }
        Ok(logits.into_iter().take(label_count).collect())
    }
}
