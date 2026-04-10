use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::Tokenizer;

use crate::gliner_relex_decode::{
    build_prompt_tokens, build_words_mask, decode_entities, decode_relations, split_words,
};
use crate::ort_runtime::{load_session_with_intra_threads, recommended_thread_count};

#[derive(Debug, Error)]
pub enum GlinerRelexError {
    #[error("failed to load GLiNER relex model: {0}")]
    ModelLoad(String),
    #[error("GLiNER relex inference failed: {0}")]
    Inference(String),
    #[error("invalid GLiNER relex input: {0}")]
    InvalidInput(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlinerRelexLabel {
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlinerRelexEntity {
    pub text: String,
    pub label: String,
    pub score: f32,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlinerRelexRelation {
    pub head: String,
    pub label: String,
    pub tail: String,
    pub score: f32,
    pub head_idx: usize,
    pub tail_idx: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlinerRelexPrediction {
    pub active_relation_pairs: usize,
    pub seq_len: usize,
    pub logits_shape: [usize; 4],
    pub rel_idx_shape: [usize; 3],
    pub rel_logits_shape: [usize; 3],
    pub entities: Vec<GlinerRelexEntity>,
    pub relations: Vec<GlinerRelexRelation>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlinerRelexMetadata {
    pub model_path: String,
    pub tokenizer_path: String,
    pub max_len: usize,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct GlinerRelexPredictOptions {
    pub threshold: f32,
    pub relation_threshold: f32,
    pub flat_ner: bool,
    pub multi_label: bool,
}

impl Default for GlinerRelexPredictOptions {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            relation_threshold: 0.5,
            flat_ner: false,
            multi_label: false,
        }
    }
}

#[derive(Default, Deserialize)]
struct GlinerRelexConfig {
    #[serde(default)]
    max_len: Option<usize>,
}

pub struct GlinerRelexModel {
    session: ort::session::Session,
    tokenizer: Tokenizer,
    metadata: GlinerRelexMetadata,
}

impl GlinerRelexModel {
    pub fn load(model_root: &Path) -> Result<Self, GlinerRelexError> {
        let model_path = find_relex_model_asset(model_root)?;
        let tokenizer_path =
            find_existing_asset(model_root, &["tokenizer.json", "onnx\\tokenizer.json"])?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| GlinerRelexError::ModelLoad(format!("tokenizer: {error}")))?;
        let session = load_session_with_intra_threads(&model_path, relex_thread_count())
            .map_err(|error| GlinerRelexError::ModelLoad(format!("session: {error}")))?;
        let max_len = load_json::<GlinerRelexConfig>(&model_root.join("gliner_config.json"))
            .and_then(|config| config.max_len)
            .unwrap_or(1024);
        let metadata = GlinerRelexMetadata {
            model_path: model_path.display().to_string(),
            tokenizer_path: tokenizer_path.display().to_string(),
            max_len,
            input_names: session
                .inputs
                .iter()
                .map(|input| input.name.clone())
                .collect(),
            output_names: session
                .outputs
                .iter()
                .map(|output| output.name.clone())
                .collect(),
        };
        Ok(Self {
            session,
            tokenizer,
            metadata,
        })
    }

    pub fn metadata(&self) -> &GlinerRelexMetadata {
        &self.metadata
    }

    pub fn predict(
        &self,
        text: &str,
        entity_labels: &[GlinerRelexLabel],
        relation_labels: &[GlinerRelexLabel],
        options: &GlinerRelexPredictOptions,
    ) -> Result<GlinerRelexPrediction, GlinerRelexError> {
        if text.trim().is_empty() {
            return Err(GlinerRelexError::InvalidInput(
                "text cannot be empty".to_owned(),
            ));
        }
        if entity_labels.is_empty() {
            return Err(GlinerRelexError::InvalidInput(
                "at least one entity label is required".to_owned(),
            ));
        }
        let words = split_words(text);
        let prompt_tokens = build_prompt_tokens(entity_labels, relation_labels);
        let prompt_word_count = prompt_tokens.len();
        let mut input_tokens = Vec::with_capacity(prompt_word_count + words.len());
        input_tokens.extend(prompt_tokens.iter().map(String::as_str));
        input_tokens.extend(words.iter().map(|word| &text[word.start..word.end]));
        let encoding = self
            .tokenizer
            .encode(input_tokens, true)
            .map_err(|error| GlinerRelexError::Inference(format!("tokenize: {error}")))?;
        let seq_len = encoding.len();
        if seq_len == 0 {
            return Err(GlinerRelexError::Inference(
                "tokenizer returned an empty sequence".to_owned(),
            ));
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
        let words_mask = build_words_mask(&encoding, prompt_word_count);
        let text_lengths = vec![words.len() as i64];
        let input_ids_tensor = Tensor::from_array(([1, seq_len], input_ids))
            .map_err(|error| GlinerRelexError::Inference(format!("input_ids: {error}")))?;
        let attention_mask_tensor = Tensor::from_array(([1, seq_len], attention_mask))
            .map_err(|error| GlinerRelexError::Inference(format!("attention_mask: {error}")))?;
        let words_mask_tensor = Tensor::from_array(([1, seq_len], words_mask))
            .map_err(|error| GlinerRelexError::Inference(format!("words_mask: {error}")))?;
        let text_lengths_tensor = Tensor::from_array(([1, 1], text_lengths))
            .map_err(|error| GlinerRelexError::Inference(format!("text_lengths: {error}")))?;
        let inputs = ort::inputs! {
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "words_mask" => words_mask_tensor,
            "text_lengths" => text_lengths_tensor,
        }
        .map_err(|error| GlinerRelexError::Inference(format!("session inputs: {error}")))?;
        let outputs = self
            .session
            .run(inputs)
            .map_err(|error| GlinerRelexError::Inference(format!("session run: {error}")))?;
        let logits_value = outputs
            .get("logits")
            .ok_or_else(|| GlinerRelexError::Inference("missing 'logits' output".to_owned()))?;
        let logits_view = logits_value.try_extract_tensor::<f32>().map_err(|error| {
            GlinerRelexError::Inference(format!("extract 'logits' fp32 tensor: {error}"))
        })?;
        let logits_shape = to_shape4(logits_view.shape())?;
        let logits_values = logits_view.as_slice().ok_or_else(|| {
            GlinerRelexError::Inference("'logits' tensor is not contiguous".to_owned())
        })?;

        let rel_idx_value = outputs
            .get("rel_idx")
            .ok_or_else(|| GlinerRelexError::Inference("missing 'rel_idx' output".to_owned()))?;
        let rel_idx_view = rel_idx_value.try_extract_tensor::<i64>().map_err(|error| {
            GlinerRelexError::Inference(format!("extract 'rel_idx' i64 tensor: {error}"))
        })?;
        let rel_idx_shape = to_shape3(rel_idx_view.shape())?;
        let rel_idx_values = rel_idx_view.as_slice().ok_or_else(|| {
            GlinerRelexError::Inference("'rel_idx' tensor is not contiguous".to_owned())
        })?;

        let rel_logits_value = outputs
            .get("rel_logits")
            .ok_or_else(|| GlinerRelexError::Inference("missing 'rel_logits' output".to_owned()))?;
        let rel_logits_view = rel_logits_value
            .try_extract_tensor::<f32>()
            .map_err(|error| {
                GlinerRelexError::Inference(format!("extract 'rel_logits' fp32 tensor: {error}"))
            })?;
        let rel_logits_shape = to_shape3(rel_logits_view.shape())?;
        let rel_logits_values = rel_logits_view.as_slice().ok_or_else(|| {
            GlinerRelexError::Inference("'rel_logits' tensor is not contiguous".to_owned())
        })?;

        let rel_mask_value = outputs
            .get("rel_mask")
            .ok_or_else(|| GlinerRelexError::Inference("missing 'rel_mask' output".to_owned()))?;
        let rel_mask_view = rel_mask_value
            .try_extract_tensor::<bool>()
            .map_err(|error| {
                GlinerRelexError::Inference(format!("extract 'rel_mask' bool tensor: {error}"))
            })?;
        let rel_mask_values = rel_mask_view.as_slice().ok_or_else(|| {
            GlinerRelexError::Inference("'rel_mask' tensor is not contiguous".to_owned())
        })?;
        let entities = decode_entities(
            text,
            &words,
            entity_labels,
            options.threshold,
            options.flat_ner,
            options.multi_label,
            logits_shape,
            logits_values,
        );
        let relations = if relation_labels.is_empty() {
            Vec::new()
        } else {
            decode_relations(
                &entities,
                relation_labels,
                options.relation_threshold,
                rel_idx_shape,
                rel_idx_values,
                rel_logits_shape,
                rel_logits_values,
                rel_mask_values,
            )
        };
        Ok(GlinerRelexPrediction {
            active_relation_pairs: rel_mask_values.iter().filter(|&&value| value).count(),
            seq_len,
            logits_shape,
            rel_idx_shape,
            rel_logits_shape,
            entities,
            relations,
        })
    }
}

fn find_relex_model_asset(root: &Path) -> Result<PathBuf, GlinerRelexError> {
    if let Ok(file_name) = env::var("PHOENIX_GLINER_RELEX_ONNX_FILE") {
        let path = root.join(file_name);
        if path.exists() {
            return Ok(path);
        }
        return Err(GlinerRelexError::ModelLoad(format!(
            "PHOENIX_GLINER_RELEX_ONNX_FILE points to missing asset under {}",
            root.display()
        )));
    }
    find_existing_asset(
        root,
        &[
            "model_relex.onnx",
            "onnx\\model_relex.onnx",
            "model.onnx",
            "onnx\\model.onnx",
            "model_quantized.onnx",
            "onnx\\model_quantized.onnx",
        ],
    )
}

fn relex_thread_count() -> usize {
    env::var("PHOENIX_GLINER_RELEX_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(recommended_thread_count)
}

fn load_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let payload = fs::read_to_string(path).ok()?;
    serde_json::from_str(&payload).ok()
}

fn find_existing_asset(root: &Path, candidates: &[&str]) -> Result<PathBuf, GlinerRelexError> {
    for candidate in candidates {
        let path = root.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(GlinerRelexError::ModelLoad(format!(
        "missing required GLiNER asset under {}",
        root.display()
    )))
}

fn to_shape4(shape: &[usize]) -> Result<[usize; 4], GlinerRelexError> {
    shape
        .try_into()
        .map_err(|_| GlinerRelexError::Inference(format!("unexpected logits rank: {shape:?}")))
}

fn to_shape3(shape: &[usize]) -> Result<[usize; 3], GlinerRelexError> {
    shape
        .try_into()
        .map_err(|_| GlinerRelexError::Inference(format!("unexpected relation rank: {shape:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_roundtrips_serde() {
        let value = GlinerRelexMetadata {
            model_path: "model.onnx".to_owned(),
            tokenizer_path: "tokenizer.json".to_owned(),
            max_len: 1024,
            input_names: vec!["input_ids".to_owned()],
            output_names: vec!["logits".to_owned()],
        };
        let json = serde_json::to_string(&value).expect("serialize metadata");
        assert!(json.contains("model.onnx"));
    }

    #[test]
    fn relex_model_asset_prefers_fp32_over_quantized() {
        let prior = env::var("PHOENIX_GLINER_RELEX_ONNX_FILE").ok();
        env::remove_var("PHOENIX_GLINER_RELEX_ONNX_FILE");
        let root =
            std::env::temp_dir().join(format!("phoenix-gliner-relex-asset-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).expect("create temp model root");
        fs::write(root.join("model_quantized.onnx"), []).expect("write quantized sentinel");
        fs::write(root.join("model.onnx"), []).expect("write fp32 sentinel");
        let selected = find_relex_model_asset(&root).expect("select model asset");
        assert_eq!(
            selected.file_name().and_then(|name| name.to_str()),
            Some("model.onnx")
        );
        let _ = fs::remove_dir_all(&root);
        if let Some(value) = prior {
            env::set_var("PHOENIX_GLINER_RELEX_ONNX_FILE", value);
        }
    }
}
