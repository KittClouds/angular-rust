use std::env;
use std::path::{Path, PathBuf};
use std::thread;

use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::TensorRefMut;
use thiserror::Error;
use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection, TruncationParams,
    TruncationStrategy,
};

const PASSAGE_PREFIX: &str = "passage: ";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextEmbeddingProfile {
    Truncate256,
    Native384,
    Native786,
    Native1024,
}

impl Default for TextEmbeddingProfile {
    fn default() -> Self {
        Self::Native384
    }
}

impl TextEmbeddingProfile {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "256" | "256-truncated" | "truncated-256" => Some(Self::Truncate256),
            "384" | "native-384" => Some(Self::Native384),
            "786" | "native-786" => Some(Self::Native786),
            "1024" | "native-1024" => Some(Self::Native1024),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Truncate256 => "256-truncated",
            Self::Native384 => "384",
            Self::Native786 => "786",
            Self::Native1024 => "1024",
        }
    }

    pub fn target_dim(self) -> usize {
        match self {
            Self::Truncate256 => 256,
            Self::Native384 => 384,
            Self::Native786 => 786,
            Self::Native1024 => 1024,
        }
    }

    fn project(self, values: &[f32]) -> Result<Vec<f32>, OrtTextEmbedError> {
        match self {
            Self::Truncate256 => {
                if values.len() < 256 {
                    return Err(OrtTextEmbedError::EmbeddingDimension {
                        expected: 256,
                        actual: values.len(),
                        profile: self.label(),
                    });
                }
                Ok(normalize_embedding(&values[..256]))
            }
            _ => {
                if values.len() != self.target_dim() {
                    return Err(OrtTextEmbedError::EmbeddingDimension {
                        expected: self.target_dim(),
                        actual: values.len(),
                        profile: self.label(),
                    });
                }
                Ok(normalize_embedding(values))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrtTextEmbedConfig {
    pub model_root: PathBuf,
    pub batch_size: usize,
    pub max_length: usize,
    pub profile: TextEmbeddingProfile,
    pub prefix_passage: bool,
}

impl Default for OrtTextEmbedConfig {
    fn default() -> Self {
        Self {
            model_root: default_embedding_model_root(),
            batch_size: 12,
            max_length: 512,
            profile: TextEmbeddingProfile::default(),
            prefix_passage: false,
        }
    }
}

#[derive(Debug, Error)]
pub enum OrtTextEmbedError {
    #[error("failed to load embedding model: {0}")]
    ModelLoad(String),
    #[error("embedding inference failed: {0}")]
    Inference(String),
    #[error(
        "embedding dimension mismatch for profile {profile}: expected {expected}, got {actual}"
    )]
    EmbeddingDimension {
        expected: usize,
        actual: usize,
        profile: &'static str,
    },
}

pub struct OrtTextEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    cpu_memory_info: MemoryInfo,
    batch_size: usize,
    profile: TextEmbeddingProfile,
    prefix_passage: bool,
    need_token_type_ids: bool,
}

#[derive(Default)]
struct TensorScratch {
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    token_type_ids: Vec<i64>,
}

impl TensorScratch {
    fn prepare(&mut self, len: usize, need_token_type_ids: bool) {
        self.input_ids.resize(len, 0);
        self.attention_mask.resize(len, 0);
        if need_token_type_ids {
            self.token_type_ids.resize(len, 0);
        } else {
            self.token_type_ids.clear();
        }
    }
}

impl OrtTextEmbedder {
    pub fn load(config: &OrtTextEmbedConfig) -> Result<Self, OrtTextEmbedError> {
        let model_path =
            find_existing_path(&config.model_root, &["onnx\\model.onnx", "model.onnx"])?;
        let tokenizer_path = find_existing_path(
            &config.model_root,
            &["tokenizer.json", "onnx\\tokenizer.json"],
        )?;
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| OrtTextEmbedError::ModelLoad(format!("tokenizer: {error}")))?;
        configure_tokenizer(&mut tokenizer, config.max_length.max(16))
            .map_err(|error| OrtTextEmbedError::ModelLoad(format!("tokenizer config: {error}")))?;
        let available_threads = thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1);
        let session = Session::builder()
            .and_then(|builder| builder.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|builder| builder.with_parallel_execution(false))
            .and_then(|builder| builder.with_inter_threads(1))
            .and_then(|builder| builder.with_intra_threads(available_threads))
            .and_then(|builder| builder.commit_from_file(&model_path))
            .map_err(|error| OrtTextEmbedError::ModelLoad(format!("session: {error}")))?;
        let need_token_type_ids = session
            .inputs
            .iter()
            .any(|input| input.name == "token_type_ids");
        let cpu_memory_info = MemoryInfo::new(
            AllocationDevice::CPU,
            0,
            AllocatorType::Arena,
            MemoryType::CPUInput,
        )
        .map_err(|error| OrtTextEmbedError::ModelLoad(format!("memory info: {error}")))?;
        Ok(Self {
            session,
            tokenizer,
            cpu_memory_info,
            batch_size: config.batch_size.max(1),
            profile: config.profile,
            prefix_passage: config.prefix_passage,
            need_token_type_ids,
        })
    }

    pub fn profile(&self) -> TextEmbeddingProfile {
        self.profile
    }

    pub fn embed_batched<S: AsRef<str>>(
        &self,
        texts: &[S],
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, OrtTextEmbedError> {
        let batch_size = batch_size.max(1);
        let mut rows = Vec::with_capacity(texts.len());
        let mut scratch = TensorScratch::default();
        for chunk in texts.chunks(batch_size) {
            rows.extend(self.embed_batch(chunk, &mut scratch)?);
        }
        Ok(rows)
    }

    pub fn embed_texts<S: AsRef<str>>(
        &self,
        texts: &[S],
    ) -> Result<Vec<Vec<f32>>, OrtTextEmbedError> {
        self.embed_batched(texts, self.batch_size)
    }

    pub fn embed_slices(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, OrtTextEmbedError> {
        self.embed_texts(texts)
    }

    fn embed_batch<S: AsRef<str>>(
        &self,
        texts: &[S],
        scratch: &mut TensorScratch,
    ) -> Result<Vec<Vec<f32>>, OrtTextEmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = if self.prefix_passage {
            let mut prefixed = Vec::with_capacity(texts.len());
            for text in texts {
                let text = text.as_ref();
                let mut value = String::with_capacity(PASSAGE_PREFIX.len() + text.len());
                value.push_str(PASSAGE_PREFIX);
                value.push_str(text);
                prefixed.push(value);
            }
            self.tokenizer
                .encode_batch_fast(prefixed, true)
                .map_err(|error| OrtTextEmbedError::Inference(format!("encode_batch: {error}")))?
        } else {
            let mut inputs = Vec::with_capacity(texts.len());
            for text in texts {
                inputs.push(text.as_ref());
            }
            self.tokenizer
                .encode_batch_fast(inputs, true)
                .map_err(|error| OrtTextEmbedError::Inference(format!("encode_batch: {error}")))?
        };
        let max_len = encodings
            .first()
            .map(|encoding| encoding.len().max(1))
            .unwrap_or(1);
        let batch_len = encodings.len();
        let flat_len = batch_len * max_len;
        scratch.prepare(flat_len, self.need_token_type_ids);
        for (row, encoding) in encodings.iter().enumerate() {
            let offset = row * max_len;
            copy_ids(
                &mut scratch.input_ids[offset..offset + max_len],
                encoding.get_ids(),
            );
            copy_ids(
                &mut scratch.attention_mask[offset..offset + max_len],
                encoding.get_attention_mask(),
            );
            if self.need_token_type_ids {
                copy_ids(
                    &mut scratch.token_type_ids[offset..offset + max_len],
                    encoding.get_type_ids(),
                );
            }
        }

        let shape = [batch_len as i64, max_len as i64];
        let input_ids = tensor_ref_from_buffer(
            &self.cpu_memory_info,
            &mut scratch.input_ids,
            shape,
            "input_ids",
        )?;
        let attention_mask = tensor_ref_from_buffer(
            &self.cpu_memory_info,
            &mut scratch.attention_mask,
            shape,
            "attention_mask",
        )?;
        let outputs = if self.need_token_type_ids {
            let token_type_ids = tensor_ref_from_buffer(
                &self.cpu_memory_info,
                &mut scratch.token_type_ids,
                shape,
                "token_type_ids",
            )?;
            self.session
                .run([
                    SessionInputValue::from(input_ids),
                    SessionInputValue::from(attention_mask),
                    SessionInputValue::from(token_type_ids),
                ])
                .map_err(|error| OrtTextEmbedError::Inference(format!("run: {error}")))?
        } else {
            self.session
                .run([
                    SessionInputValue::from(input_ids),
                    SessionInputValue::from(attention_mask),
                ])
                .map_err(|error| OrtTextEmbedError::Inference(format!("run: {error}")))?
        };

        let hidden = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|error| OrtTextEmbedError::Inference(format!("extract: {error}")))?;
        let view = hidden.view();
        let shape = view.shape();
        if shape.len() != 3 {
            return Err(OrtTextEmbedError::Inference(format!(
                "expected 3D hidden state, got shape {shape:?}"
            )));
        }
        let hidden_dim = shape[2];
        let values = view.as_slice().ok_or_else(|| {
            OrtTextEmbedError::Inference("hidden state was non-contiguous".to_owned())
        })?;

        let mut rows = Vec::with_capacity(batch_len);
        for row in 0..batch_len {
            let start = row * max_len * hidden_dim;
            let cls = &values[start..start + hidden_dim];
            rows.push(self.profile.project(cls)?);
        }
        Ok(rows)
    }
}

pub fn default_embedding_model_root() -> PathBuf {
    workspace_root()
        .join("rust-native")
        .join("phoenix-hnsw-smoke")
        .join("models")
        .join("snowflake-arctic-embed-xs")
}

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("workspace root")
        .to_path_buf()
}

pub fn default_ort_dylib_path(workspace_root: &Path) -> Option<PathBuf> {
    [
        workspace_root
            .join("node_modules")
            .join("@huggingface")
            .join("transformers")
            .join("node_modules")
            .join("onnxruntime-node")
            .join("bin")
            .join("napi-v6")
            .join("win32")
            .join("x64")
            .join("onnxruntime.dll"),
        workspace_root
            .join("node_modules")
            .join("onnxruntime-node")
            .join("bin")
            .join("napi-v3")
            .join("win32")
            .join("x64")
            .join("onnxruntime.dll"),
    ]
    .into_iter()
    .find(|path| path.exists())
}

fn find_existing_path(root: &Path, candidates: &[&str]) -> Result<PathBuf, OrtTextEmbedError> {
    for candidate in candidates {
        let path = root.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(OrtTextEmbedError::ModelLoad(format!(
        "missing required asset under {}",
        root.display()
    )))
}

fn normalize_embedding(values: &[f32]) -> Vec<f32> {
    let norm = values
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt()
        .max(1e-12);
    values.iter().map(|value| *value / norm).collect()
}

fn configure_tokenizer(
    tokenizer: &mut Tokenizer,
    max_length: usize,
) -> Result<(), tokenizers::Error> {
    let mut padding = tokenizer
        .get_padding()
        .cloned()
        .unwrap_or_else(|| PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        });
    padding.strategy = PaddingStrategy::BatchLongest;
    tokenizer.with_padding(Some(padding));
    tokenizer.with_truncation(Some(TruncationParams {
        max_length,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
        direction: TruncationDirection::Right,
    }))?;
    Ok(())
}

fn copy_ids(target: &mut [i64], values: &[u32]) {
    for (slot, value) in target.iter_mut().zip(values.iter()) {
        *slot = i64::from(*value);
    }
}

fn tensor_ref_from_buffer<'a>(
    memory_info: &MemoryInfo,
    buffer: &'a mut Vec<i64>,
    shape: [i64; 2],
    label: &str,
) -> Result<TensorRefMut<'a, i64>, OrtTextEmbedError> {
    unsafe {
        TensorRefMut::from_raw(
            memory_info.clone(),
            buffer.as_mut_ptr().cast(),
            Vec::from(shape),
        )
        .map_err(|error| OrtTextEmbedError::Inference(format!("{label}: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_parse_expected_labels() {
        assert_eq!(
            TextEmbeddingProfile::parse("256-truncated"),
            Some(TextEmbeddingProfile::Truncate256)
        );
        assert_eq!(
            TextEmbeddingProfile::parse("384"),
            Some(TextEmbeddingProfile::Native384)
        );
        assert_eq!(
            TextEmbeddingProfile::parse("786"),
            Some(TextEmbeddingProfile::Native786)
        );
        assert_eq!(
            TextEmbeddingProfile::parse("1024"),
            Some(TextEmbeddingProfile::Native1024)
        );
    }

    #[test]
    fn truncate_profile_projects_first_256_dims() {
        let values = (0..384).map(|value| value as f32 + 1.0).collect::<Vec<_>>();
        let projected = TextEmbeddingProfile::Truncate256
            .project(&values)
            .expect("truncate profile");
        assert_eq!(projected.len(), 256);
    }

    #[test]
    fn native_profile_rejects_wrong_dimension() {
        let error = TextEmbeddingProfile::Native384
            .project(&vec![1.0; 256])
            .expect_err("dimension mismatch");
        assert!(matches!(
            error,
            OrtTextEmbedError::EmbeddingDimension {
                expected: 384,
                actual: 256,
                ..
            }
        ));
    }
}
