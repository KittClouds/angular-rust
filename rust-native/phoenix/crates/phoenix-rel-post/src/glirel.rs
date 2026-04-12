use std::cmp::Ordering;
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use half::f16;
use ort::session::Session;
use ort::value::Tensor;
use rustc_hash::{FxHashMap, FxHasher};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokenizers::Tokenizer;

const ENCODER_DIM: usize = 1024;
const PROJECTION_DIM: usize = 768;
const CLS_ID: i64 = 1;
const SEP_ID: i64 = 2;
const REL_ID: i64 = 128002;
const MAX_PROMPT_SCHEMA_CACHE_ENTRIES: usize = 256;
const MAX_WORDPIECE_CACHE_ENTRIES: usize = 8_192;
const MAX_SCHEMA_BATCH_WINDOWS: usize = 8;
const MAX_SCHEMA_BATCH_SEQUENCE_TOKENS: usize = 4_096;
const MAX_SCHEMA_BATCH_PAIR_SLOTS: usize = 512;

#[derive(Debug, Error)]
pub enum GlirelError {
    #[error("failed to load GLiREL model: {0}")]
    ModelLoad(String),
    #[error("GLiREL inference failed: {0}")]
    Inference(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlirelEntity {
    pub text: String,
    pub entity_type: String,
    pub span_start: usize,
    pub span_end: usize,
    pub entity_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlirelRelationTypeSpec {
    pub label: String,
    #[serde(default)]
    pub head_types: Vec<String>,
    #[serde(default)]
    pub tail_types: Vec<String>,
    #[serde(default)]
    pub cue_phrases: Vec<String>,
    #[serde(default)]
    pub conflicts_with: Vec<String>,
    #[serde(default)]
    pub priority_millis: i32,
    #[serde(default = "default_accept_threshold_millis")]
    pub accept_threshold_millis: u32,
    #[serde(default = "default_review_threshold_millis")]
    pub review_threshold_millis: u32,
    #[serde(default = "default_max_predictions_per_window")]
    pub max_predictions_per_window: usize,
    #[serde(default = "default_directed")]
    pub directed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlirelSentenceWindow {
    pub index: usize,
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlirelPairSeed {
    pub head_index: usize,
    pub tail_index: usize,
    pub score_millis: i32,
    #[serde(default)]
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlirelProposalConfig {
    pub sentence_window_radius: usize,
    pub max_pair_char_distance: usize,
    pub min_seed_score_millis: i32,
}

impl Default for GlirelProposalConfig {
    fn default() -> Self {
        Self {
            sentence_window_radius: 1,
            max_pair_char_distance: 220,
            min_seed_score_millis: 330,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GlirelRelationPrediction {
    pub head_index: usize,
    pub tail_index: usize,
    pub head: String,
    pub tail: String,
    pub relation: String,
    pub confidence: f32,
    #[serde(default)]
    pub evidence: Vec<String>,
}

pub struct GlirelModel {
    encoder: Session,
    scoring_head: Session,
    tokenizer: Tokenizer,
    projection_weight: Vec<f32>,
    projection_bias: Vec<f32>,
    token_cache: Mutex<GlirelTokenCache>,
}

#[derive(Clone, Debug)]
struct PromptSchemaCacheEntry {
    relation_labels: Vec<String>,
    prompt_word_count: usize,
    input_ids_prefix: Vec<i64>,
    word_ids_prefix: Vec<Option<usize>>,
}

#[derive(Default)]
struct GlirelTokenCache {
    prompt_prefixes: FxHashMap<u64, Vec<PromptSchemaCacheEntry>>,
    wordpiece_ids: FxHashMap<String, Box<[i64]>>,
}

pub(crate) struct GlirelBatchItem<'a> {
    pub text: &'a str,
    pub entities: &'a [GlirelEntity],
}

struct PreparedSchemaBatchRow<'a> {
    output_index: usize,
    text: &'a str,
    entities: &'a [GlirelEntity],
    prompt_word_count: usize,
    text_word_count: usize,
    input_ids: Vec<i64>,
    attention_mask: Vec<i64>,
    word_ids: Vec<Option<usize>>,
    span_index: Vec<(i64, i64)>,
    relation_pairs: Vec<i64>,
    pair_map: Vec<(usize, usize)>,
}

struct PreparedScoringBatchRow<'a> {
    output_index: usize,
    text: &'a str,
    entities: &'a [GlirelEntity],
    text_word_count: usize,
    text_word_representations: Vec<f32>,
    word_mask: Vec<i64>,
    span_index: Vec<(i64, i64)>,
    relation_pairs: Vec<i64>,
    relation_representations: Vec<f32>,
    pair_map: Vec<(usize, usize)>,
}

impl GlirelModel {
    pub fn load(model_dir: &Path) -> Result<Self, GlirelError> {
        let encoder_path = find_existing_path(model_dir, &["encoder.onnx"])?;
        let scoring_path = find_existing_path(model_dir, &["scoring_head.onnx"])?;
        let tokenizer_path = model_dir.join("tokenizer.json");
        let weight_path = model_dir.join("projection_weight.bin");
        let bias_path = model_dir.join("projection_bias.bin");

        for (label, path) in [
            ("tokenizer", &tokenizer_path),
            ("projection_weight", &weight_path),
            ("projection_bias", &bias_path),
        ] {
            if !path.exists() {
                return Err(GlirelError::ModelLoad(format!(
                    "{label} missing at {}",
                    path.display()
                )));
            }
        }

        let encoder = Session::builder()
            .and_then(|builder| builder.with_intra_threads(1))
            .and_then(|builder| builder.commit_from_file(&encoder_path))
            .map_err(|error| GlirelError::ModelLoad(format!("encoder session: {error}")))?;

        let scoring_head = Session::builder()
            .and_then(|builder| builder.with_intra_threads(1))
            .and_then(|builder| builder.commit_from_file(&scoring_path))
            .map_err(|error| GlirelError::ModelLoad(format!("scoring head session: {error}")))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| GlirelError::ModelLoad(format!("tokenizer: {error}")))?;

        let projection_weight = load_f32_bin(&weight_path, PROJECTION_DIM * ENCODER_DIM)?;
        let projection_bias = load_f32_bin(&bias_path, PROJECTION_DIM)?;

        Ok(Self {
            encoder,
            scoring_head,
            tokenizer,
            projection_weight,
            projection_bias,
            token_cache: Mutex::new(GlirelTokenCache::default()),
        })
    }

    pub fn extract(
        &self,
        text: &str,
        entities: &[GlirelEntity],
        relation_labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<GlirelRelationPrediction>, GlirelError> {
        if text.trim().is_empty() || entities.len() < 2 || relation_labels.is_empty() {
            return Ok(Vec::new());
        }

        let text_words: Vec<&str> = text.split_whitespace().collect();
        if text_words.is_empty() {
            return Ok(Vec::new());
        }

        let (input_ids, attention_mask, word_ids) =
            self.tokenize_words(relation_labels, &text_words)?;
        let prompt_word_count = relation_labels.len() * 2 + 1;
        let hidden = self.run_encoder(&input_ids, &attention_mask)?;
        let projected = self.project(&hidden, input_ids.len());
        let word_representations = first_subword_pool(
            &projected,
            &word_ids,
            prompt_word_count + text_words.len(),
            PROJECTION_DIM,
        );

        let relation_representations =
            build_relation_type_representations(&word_representations, relation_labels.len());
        let text_start = prompt_word_count * PROJECTION_DIM;
        let text_word_representations = word_representations
            [text_start..text_start + text_words.len() * PROJECTION_DIM]
            .to_vec();
        let span_index = entities_to_word_spans(text, &text_words, entities);
        let relation_pairs = build_relations_idx(&span_index);
        let pair_map = directed_pair_map(entities.len());
        let word_mask = vec![1_i64; text_words.len()];

        let score_tensor = self.run_scoring_head(
            &text_word_representations,
            &word_mask,
            &span_index,
            &relation_pairs,
            &relation_representations,
            text_words.len(),
            entities.len(),
            pair_map.len(),
            relation_labels.len(),
        )?;

        let mut predictions = decode_predictions(
            entities,
            relation_labels,
            &pair_map,
            &score_tensor,
            threshold,
        )?;
        boost_context_matches(text, &mut predictions);
        dedupe_predictions(&mut predictions);

        Ok(predictions)
    }

    pub fn extract_with_schema(
        &self,
        text: &str,
        entities: &[GlirelEntity],
        relation_specs: &[GlirelRelationTypeSpec],
        threshold: f32,
    ) -> Result<Vec<GlirelRelationPrediction>, GlirelError> {
        let labels = relation_specs
            .iter()
            .map(|spec| spec.label.as_str())
            .collect::<Vec<_>>();
        let predictions = self.extract(text, entities, &labels, threshold)?;

        Ok(finalize_relation_predictions(
            text,
            entities,
            relation_specs,
            predictions,
        ))
    }

    pub(crate) fn extract_many_with_schema<'a>(
        &self,
        items: &[GlirelBatchItem<'a>],
        relation_specs: &[GlirelRelationTypeSpec],
        threshold: f32,
    ) -> Result<Vec<Vec<GlirelRelationPrediction>>, GlirelError> {
        if items.is_empty() {
            return Ok(Vec::new());
        }

        let labels = relation_specs
            .iter()
            .map(|spec| spec.label.as_str())
            .collect::<Vec<_>>();
        let mut outputs = vec![Vec::new(); items.len()];
        let prepared = self.prepare_schema_batch_rows(items, &labels)?;

        let mut offset = 0usize;
        while offset < prepared.len() {
            let batch_len = next_schema_batch_len(&prepared[offset..]);
            let rows = &prepared[offset..offset + batch_len];
            let chunk_outputs =
                match self.run_schema_batch(rows, relation_specs, &labels, threshold) {
                    Ok(value) => value,
                    Err(error) if rows.len() > 1 => rows
                        .iter()
                        .map(|row| {
                            self.extract_with_schema(
                                row.text,
                                row.entities,
                                relation_specs,
                                threshold,
                            )
                            .map(|predictions| (row.output_index, predictions))
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                    Err(error) => return Err(error),
                };
            for (output_index, predictions) in chunk_outputs {
                outputs[output_index] = predictions;
            }
            offset += batch_len;
        }

        Ok(outputs)
    }

    fn tokenize_words(
        &self,
        relation_labels: &[&str],
        text_words: &[&str],
    ) -> Result<(Vec<i64>, Vec<i64>, Vec<Option<usize>>), GlirelError> {
        let prompt_schema = self.prompt_schema_prefix(relation_labels)?;
        let mut input_ids = prompt_schema.input_ids_prefix.clone();
        let mut word_ids = prompt_schema.word_ids_prefix.clone();
        let mut word_index = prompt_schema.prompt_word_count;

        for &word in text_words {
            self.append_tokenized_word(word, word_index, &mut input_ids, &mut word_ids)?;
            word_index += 1;
        }

        input_ids.push(SEP_ID);
        word_ids.push(None);
        let attention_mask = vec![1_i64; input_ids.len()];
        Ok((input_ids, attention_mask, word_ids))
    }

    fn prompt_schema_prefix(
        &self,
        relation_labels: &[&str],
    ) -> Result<PromptSchemaCacheEntry, GlirelError> {
        let signature = relation_label_signature(relation_labels);
        if let Some(entry) = self.lookup_prompt_schema(signature, relation_labels) {
            return Ok(entry);
        }

        let prompt_words = build_prompt_words(relation_labels);
        let prompt_word_count = prompt_words.len();
        let mut input_ids_prefix = vec![CLS_ID];
        let mut word_ids_prefix = vec![None];
        for (word_index, word) in prompt_words.iter().enumerate() {
            self.append_tokenized_word(
                word.as_str(),
                word_index,
                &mut input_ids_prefix,
                &mut word_ids_prefix,
            )?;
        }

        let entry = PromptSchemaCacheEntry {
            relation_labels: relation_labels
                .iter()
                .map(|label| (*label).to_owned())
                .collect(),
            prompt_word_count,
            input_ids_prefix,
            word_ids_prefix,
        };
        self.store_prompt_schema(signature, entry.clone());
        Ok(entry)
    }

    fn lookup_prompt_schema(
        &self,
        signature: u64,
        relation_labels: &[&str],
    ) -> Option<PromptSchemaCacheEntry> {
        let cache = self.token_cache.lock().ok()?;
        let entries = cache.prompt_prefixes.get(&signature)?;
        entries
            .iter()
            .find(|entry| prompt_schema_matches_labels(entry, relation_labels))
            .cloned()
    }

    fn store_prompt_schema(&self, signature: u64, entry: PromptSchemaCacheEntry) {
        let Ok(mut cache) = self.token_cache.lock() else {
            return;
        };
        if cache.prompt_prefixes.len() >= MAX_PROMPT_SCHEMA_CACHE_ENTRIES
            && !cache.prompt_prefixes.contains_key(&signature)
        {
            return;
        }
        cache
            .prompt_prefixes
            .entry(signature)
            .or_default()
            .push(entry);
    }

    fn append_tokenized_word(
        &self,
        word: &str,
        word_index: usize,
        input_ids: &mut Vec<i64>,
        word_ids: &mut Vec<Option<usize>>,
    ) -> Result<(), GlirelError> {
        match word {
            "[REL]" => {
                input_ids.push(REL_ID);
                word_ids.push(Some(word_index));
                return Ok(());
            }
            "[SEP]" => {
                input_ids.push(SEP_ID);
                word_ids.push(Some(word_index));
                return Ok(());
            }
            _ => {}
        }

        if let Ok(cache) = self.token_cache.lock() {
            if let Some(cached) = cache.wordpiece_ids.get(word) {
                append_token_ids(cached.as_ref(), word_index, input_ids, word_ids);
                return Ok(());
            }
        }

        let token_ids = self.tokenize_word_piece_ids(word)?;
        self.store_wordpiece_ids(word, &token_ids);
        append_token_ids(&token_ids, word_index, input_ids, word_ids);
        Ok(())
    }

    fn store_wordpiece_ids(&self, word: &str, token_ids: &[i64]) {
        let Ok(mut cache) = self.token_cache.lock() else {
            return;
        };
        if cache.wordpiece_ids.len() >= MAX_WORDPIECE_CACHE_ENTRIES
            && !cache.wordpiece_ids.contains_key(word)
        {
            return;
        }
        cache
            .wordpiece_ids
            .entry(word.to_owned())
            .or_insert_with(|| token_ids.to_vec().into_boxed_slice());
    }

    fn tokenize_word_piece_ids(&self, word: &str) -> Result<Vec<i64>, GlirelError> {
        let encoding = self
            .tokenizer
            .encode(word, false)
            .map_err(|error| GlirelError::Inference(format!("tokenize '{word}': {error}")))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    fn run_encoder(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
    ) -> Result<Vec<f32>, GlirelError> {
        self.run_encoder_batch(input_ids, attention_mask, 1, input_ids.len())
    }

    fn run_encoder_batch(
        &self,
        input_ids: &[i64],
        attention_mask: &[i64],
        batch_size: usize,
        sequence_len: usize,
    ) -> Result<Vec<f32>, GlirelError> {
        let ids = Tensor::from_array(([batch_size, sequence_len], input_ids.to_vec()))
            .map_err(|error| GlirelError::Inference(format!("input_ids tensor: {error}")))?;
        let mask = Tensor::from_array(([batch_size, sequence_len], attention_mask.to_vec()))
            .map_err(|error| GlirelError::Inference(format!("attention_mask tensor: {error}")))?;

        let inputs = ort::inputs! {
            "input_ids" => ids,
            "attention_mask" => mask,
        }
        .map_err(|error| GlirelError::Inference(format!("encoder inputs: {error}")))?;

        let outputs = self
            .encoder
            .run(inputs)
            .map_err(|error| GlirelError::Inference(format!("encoder run: {error}")))?;

        let hidden_value = outputs
            .get("last_hidden_state")
            .ok_or_else(|| GlirelError::Inference("missing last_hidden_state".to_owned()))?;

        if let Ok(view) = hidden_value.try_extract_tensor::<f32>() {
            return view.as_slice().map(ToOwned::to_owned).ok_or_else(|| {
                GlirelError::Inference("encoder hidden tensor not contiguous".to_owned())
            });
        }

        let view = hidden_value
            .try_extract_tensor::<f16>()
            .map_err(|error| GlirelError::Inference(format!("extract fp16 hidden: {error}")))?;
        let slice = view.as_slice().ok_or_else(|| {
            GlirelError::Inference("encoder fp16 hidden tensor not contiguous".to_owned())
        })?;
        Ok(slice.iter().map(|value| value.to_f32()).collect())
    }

    fn project(&self, hidden: &[f32], token_count: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; token_count * PROJECTION_DIM];
        self.project_into(hidden, &mut output, token_count);
        output
    }

    fn project_batch(&self, hidden: &[f32], batch_size: usize, token_count: usize) -> Vec<f32> {
        let hidden_stride = token_count * ENCODER_DIM;
        let output_stride = token_count * PROJECTION_DIM;
        let mut output = vec![0.0f32; batch_size * output_stride];
        for batch_index in 0..batch_size {
            let src = batch_index * hidden_stride;
            let dst = batch_index * output_stride;
            self.project_into(
                &hidden[src..src + hidden_stride],
                &mut output[dst..dst + output_stride],
                token_count,
            );
        }
        output
    }

    fn project_into(&self, hidden: &[f32], output: &mut [f32], token_count: usize) {
        for token_index in 0..token_count {
            let input_offset = token_index * ENCODER_DIM;
            let output_offset = token_index * PROJECTION_DIM;
            for output_index in 0..PROJECTION_DIM {
                let mut sum = self.projection_bias[output_index];
                let weight_offset = output_index * ENCODER_DIM;
                for input_index in 0..ENCODER_DIM {
                    sum += hidden[input_offset + input_index]
                        * self.projection_weight[weight_offset + input_index];
                }
                output[output_offset + output_index] = sum;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn run_scoring_head(
        &self,
        text_word_representations: &[f32],
        word_mask: &[i64],
        span_index: &[(i64, i64)],
        relation_pairs: &[i64],
        relation_representations: &[f32],
        text_word_count: usize,
        entity_count: usize,
        pair_count: usize,
        relation_count: usize,
    ) -> Result<Vec<f32>, GlirelError> {
        let span_flat = span_index
            .iter()
            .flat_map(|&(start, end)| [start, end])
            .collect::<Vec<_>>();
        self.run_scoring_head_batch(
            text_word_representations,
            word_mask,
            &span_flat,
            relation_pairs,
            relation_representations,
            1,
            text_word_count,
            entity_count,
            pair_count,
            relation_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_scoring_head_batch(
        &self,
        text_word_representations: &[f32],
        word_mask: &[i64],
        span_flat: &[i64],
        relation_pairs: &[i64],
        relation_representations: &[f32],
        batch_size: usize,
        text_word_count: usize,
        entity_count: usize,
        pair_count: usize,
        relation_count: usize,
    ) -> Result<Vec<f32>, GlirelError> {
        let word_rep = Tensor::from_array((
            [batch_size, text_word_count, PROJECTION_DIM],
            text_word_representations.to_vec(),
        ))
        .map_err(|error| GlirelError::Inference(format!("word_rep tensor: {error}")))?;
        let word_mask = Tensor::from_array(([batch_size, text_word_count], word_mask.to_vec()))
            .map_err(|error| GlirelError::Inference(format!("word_mask tensor: {error}")))?;
        let span_idx = Tensor::from_array(([batch_size, entity_count, 2], span_flat.to_vec()))
            .map_err(|error| GlirelError::Inference(format!("span_idx tensor: {error}")))?;
        let relations_idx =
            Tensor::from_array(([batch_size, pair_count, 2, 2], relation_pairs.to_vec())).map_err(
                |error| GlirelError::Inference(format!("relations_idx tensor: {error}")),
            )?;
        let rel_type_rep = Tensor::from_array((
            [batch_size, relation_count, PROJECTION_DIM],
            relation_representations.to_vec(),
        ))
        .map_err(|error| GlirelError::Inference(format!("rel_type_rep tensor: {error}")))?;

        let inputs = ort::inputs! {
            "word_rep" => word_rep,
            "word_mask" => word_mask,
            "span_idx" => span_idx,
            "relations_idx" => relations_idx,
            "rel_type_rep" => rel_type_rep,
        }
        .map_err(|error| GlirelError::Inference(format!("scoring inputs: {error}")))?;

        let outputs = self
            .scoring_head
            .run(inputs)
            .map_err(|error| GlirelError::Inference(format!("scoring head run: {error}")))?;

        let score_value = outputs
            .get("relation_scores")
            .ok_or_else(|| GlirelError::Inference("missing relation_scores".to_owned()))?;
        let view = score_value
            .try_extract_tensor::<f32>()
            .map_err(|error| GlirelError::Inference(format!("extract relation scores: {error}")))?;
        view.as_slice().map(ToOwned::to_owned).ok_or_else(|| {
            GlirelError::Inference("relation score tensor not contiguous".to_owned())
        })
    }

    fn prepare_schema_batch_rows<'a>(
        &self,
        items: &[GlirelBatchItem<'a>],
        relation_labels: &[&str],
    ) -> Result<Vec<PreparedSchemaBatchRow<'a>>, GlirelError> {
        let mut rows = Vec::with_capacity(items.len());
        for (output_index, item) in items.iter().enumerate() {
            if item.text.trim().is_empty() || item.entities.len() < 2 {
                continue;
            }
            let text_words = item.text.split_whitespace().collect::<Vec<_>>();
            if text_words.is_empty() {
                continue;
            }
            let (input_ids, attention_mask, word_ids) =
                self.tokenize_words(relation_labels, &text_words)?;
            let prompt_word_count = relation_labels.len() * 2 + 1;
            let span_index = entities_to_word_spans(item.text, &text_words, item.entities);
            let relation_pairs = build_relations_idx(&span_index);
            let pair_map = directed_pair_map(item.entities.len());
            rows.push(PreparedSchemaBatchRow {
                output_index,
                text: item.text,
                entities: item.entities,
                prompt_word_count,
                text_word_count: text_words.len(),
                input_ids,
                attention_mask,
                word_ids,
                span_index,
                relation_pairs,
                pair_map,
            });
        }
        Ok(rows)
    }

    fn run_schema_batch<'a>(
        &self,
        rows: &[PreparedSchemaBatchRow<'a>],
        relation_specs: &[GlirelRelationTypeSpec],
        relation_labels: &[&str],
        threshold: f32,
    ) -> Result<Vec<(usize, Vec<GlirelRelationPrediction>)>, GlirelError> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = rows.len();
        let relation_count = relation_labels.len();
        let max_sequence_len = rows
            .iter()
            .map(|row| row.input_ids.len())
            .max()
            .unwrap_or_default();
        let mut flat_input_ids = vec![0_i64; batch_size * max_sequence_len];
        let mut flat_attention_mask = vec![0_i64; batch_size * max_sequence_len];
        for (row_index, row) in rows.iter().enumerate() {
            let offset = row_index * max_sequence_len;
            flat_input_ids[offset..offset + row.input_ids.len()].copy_from_slice(&row.input_ids);
            flat_attention_mask[offset..offset + row.attention_mask.len()]
                .copy_from_slice(&row.attention_mask);
        }

        let hidden = self.run_encoder_batch(
            &flat_input_ids,
            &flat_attention_mask,
            batch_size,
            max_sequence_len,
        )?;
        let projected = self.project_batch(&hidden, batch_size, max_sequence_len);

        let mut scoring_rows = Vec::with_capacity(batch_size);
        for (row_index, row) in rows.iter().enumerate() {
            let projected_stride = max_sequence_len * PROJECTION_DIM;
            let projected_offset = row_index * projected_stride;
            let projected_slice = &projected[projected_offset..projected_offset + projected_stride];
            let mut word_ids = row.word_ids.clone();
            word_ids.resize(max_sequence_len, None);
            let word_representations = first_subword_pool(
                projected_slice,
                &word_ids,
                row.prompt_word_count + row.text_word_count,
                PROJECTION_DIM,
            );
            let relation_representations =
                build_relation_type_representations(&word_representations, relation_count);
            let text_start = row.prompt_word_count * PROJECTION_DIM;
            let text_word_representations = word_representations
                [text_start..text_start + row.text_word_count * PROJECTION_DIM]
                .to_vec();
            scoring_rows.push(PreparedScoringBatchRow {
                output_index: row.output_index,
                text: row.text,
                entities: row.entities,
                text_word_count: row.text_word_count,
                text_word_representations,
                word_mask: vec![1_i64; row.text_word_count],
                span_index: row.span_index.clone(),
                relation_pairs: row.relation_pairs.clone(),
                relation_representations,
                pair_map: row.pair_map.clone(),
            });
        }

        let max_text_word_count = scoring_rows
            .iter()
            .map(|row| row.text_word_count)
            .max()
            .unwrap_or_default();
        let max_entity_count = scoring_rows
            .iter()
            .map(|row| row.span_index.len())
            .max()
            .unwrap_or_default();
        let max_pair_count = scoring_rows
            .iter()
            .map(|row| row.pair_map.len())
            .max()
            .unwrap_or_default();

        let mut flat_word_rep = vec![0.0f32; batch_size * max_text_word_count * PROJECTION_DIM];
        let mut flat_word_mask = vec![0_i64; batch_size * max_text_word_count];
        let mut flat_span_idx = vec![0_i64; batch_size * max_entity_count * 2];
        let mut flat_relations_idx = vec![0_i64; batch_size * max_pair_count * 4];
        let mut flat_rel_type_rep = vec![0.0f32; batch_size * relation_count * PROJECTION_DIM];

        for (row_index, row) in scoring_rows.iter().enumerate() {
            let word_rep_offset = row_index * max_text_word_count * PROJECTION_DIM;
            flat_word_rep[word_rep_offset..word_rep_offset + row.text_word_representations.len()]
                .copy_from_slice(&row.text_word_representations);
            let word_mask_offset = row_index * max_text_word_count;
            flat_word_mask[word_mask_offset..word_mask_offset + row.word_mask.len()]
                .copy_from_slice(&row.word_mask);

            let span_offset = row_index * max_entity_count * 2;
            for (entity_index, &(start, end)) in row.span_index.iter().enumerate() {
                let dst = span_offset + entity_index * 2;
                flat_span_idx[dst] = start;
                flat_span_idx[dst + 1] = end;
            }

            let pair_offset = row_index * max_pair_count * 4;
            flat_relations_idx[pair_offset..pair_offset + row.relation_pairs.len()]
                .copy_from_slice(&row.relation_pairs);

            let rel_type_offset = row_index * relation_count * PROJECTION_DIM;
            flat_rel_type_rep
                [rel_type_offset..rel_type_offset + row.relation_representations.len()]
                .copy_from_slice(&row.relation_representations);
        }

        let score_tensor = self.run_scoring_head_batch(
            &flat_word_rep,
            &flat_word_mask,
            &flat_span_idx,
            &flat_relations_idx,
            &flat_rel_type_rep,
            batch_size,
            max_text_word_count,
            max_entity_count,
            max_pair_count,
            relation_count,
        )?;

        let score_stride = max_pair_count * relation_count;
        let mut outputs = Vec::with_capacity(batch_size);
        for (row_index, row) in scoring_rows.iter().enumerate() {
            let score_offset = row_index * score_stride;
            let score_len = row.pair_map.len() * relation_count;
            let mut predictions = decode_predictions(
                row.entities,
                relation_labels,
                &row.pair_map,
                &score_tensor[score_offset..score_offset + score_len],
                threshold,
            )?;
            boost_context_matches(row.text, &mut predictions);
            dedupe_predictions(&mut predictions);
            outputs.push((
                row.output_index,
                finalize_relation_predictions(row.text, row.entities, relation_specs, predictions),
            ));
        }

        Ok(outputs)
    }
}

fn default_directed() -> bool {
    true
}

fn default_max_predictions_per_window() -> usize {
    1
}

fn default_accept_threshold_millis() -> u32 {
    700
}

fn default_review_threshold_millis() -> u32 {
    450
}

pub fn split_sentence_windows(text: &str) -> Vec<GlirelSentenceWindow> {
    let mut windows = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0usize;
    let mut index = 0usize;
    let mut cursor = 0usize;

    while cursor < bytes.len() {
        let byte = bytes[cursor];
        let boundary = match byte {
            b'.' | b'!' | b'?' => true,
            b'\n' => cursor + 1 < bytes.len() && bytes[cursor + 1] == b'\n',
            _ => false,
        };
        if boundary {
            let end = if byte == b'\n' { cursor } else { cursor + 1 };
            let sentence = text[start..end].trim();
            if !sentence.is_empty() {
                windows.push(GlirelSentenceWindow {
                    index,
                    start,
                    end,
                    text: sentence.to_owned(),
                });
                index += 1;
            }
            start = if byte == b'\n' && cursor + 1 < bytes.len() {
                cursor + 2
            } else {
                cursor + 1
            };
        }
        cursor += 1;
    }

    if start < text.len() {
        let sentence = text[start..].trim();
        if !sentence.is_empty() {
            windows.push(GlirelSentenceWindow {
                index,
                start,
                end: text.len(),
                text: sentence.to_owned(),
            });
        }
    }

    windows
}

pub fn seed_relation_pairs(
    text: &str,
    entities: &[GlirelEntity],
    config: &GlirelProposalConfig,
) -> Vec<GlirelPairSeed> {
    let windows = split_sentence_windows(text);
    let mut seeds = Vec::new();
    let mut seen = HashSet::new();

    for (head_index, head) in entities.iter().enumerate() {
        for (tail_index, tail) in entities.iter().enumerate() {
            if head_index == tail_index || head.text == tail.text {
                continue;
            }

            let same_window = relation_window_distance(&windows, head, tail)
                .is_some_and(|distance| distance <= config.sentence_window_radius);
            if !same_window {
                continue;
            }

            let char_distance = entity_char_distance(head, tail);
            if char_distance > config.max_pair_char_distance {
                continue;
            }

            let name_quality =
                (entity_name_quality(&head.text) + entity_name_quality(&tail.text)) / 2.0;
            let proximity = 1.0 / (1.0 + (char_distance as f32 / 80.0));
            let same_type_bonus = if head.entity_type == tail.entity_type {
                0.06
            } else {
                0.0
            };
            let repeated_capitalized_bonus =
                if looks_like_named_entity(&head.text) && looks_like_named_entity(&tail.text) {
                    0.08
                } else {
                    0.0
                };
            let score = 0.28
                + (name_quality * 0.32)
                + (proximity * 0.34)
                + same_type_bonus
                + repeated_capitalized_bonus;
            let score_millis = (score * 1000.0).round() as i32;
            if score_millis < config.min_seed_score_millis {
                continue;
            }

            if seen.insert((head_index, tail_index)) {
                seeds.push(GlirelPairSeed {
                    head_index,
                    tail_index,
                    score_millis,
                    evidence: vec![
                        format!("char_distance:{char_distance}"),
                        format!("name_quality:{name_quality:.2}"),
                        format!("sentence_window_radius:{}", config.sentence_window_radius),
                    ],
                });
            }
        }
    }

    seeds.sort_by(|left, right| right.score_millis.cmp(&left.score_millis));
    seeds
}

pub fn extract_heuristic_relations(
    text: &str,
    entities: &[GlirelEntity],
    relation_specs: &[GlirelRelationTypeSpec],
    config: &GlirelProposalConfig,
) -> Vec<GlirelRelationPrediction> {
    let windows = split_sentence_windows(text);
    let pair_seeds = seed_relation_pairs(text, entities, config);
    if windows.is_empty() || pair_seeds.is_empty() || relation_specs.is_empty() {
        return Vec::new();
    }

    let mut predictions = Vec::new();
    let text_lower = text.to_lowercase();

    for window in &windows {
        let window_entities = pair_seeds
            .iter()
            .filter(|seed| {
                entity_pair_in_window(
                    window,
                    &entities[seed.head_index],
                    &entities[seed.tail_index],
                    config.sentence_window_radius,
                    &windows,
                )
            })
            .collect::<Vec<_>>();
        if window_entities.is_empty() {
            continue;
        }

        let window_lower = window.text.to_lowercase();
        for spec in relation_specs {
            let cue_matches = relation_cue_matches(spec, &window_lower, &text_lower);
            if cue_matches.is_empty() {
                continue;
            }

            let mut ranked = Vec::new();
            for seed in &window_entities {
                let head = &entities[seed.head_index];
                let tail = &entities[seed.tail_index];
                let schema_score = schema_pair_score(spec, head, tail);
                if schema_score <= 0.0 {
                    continue;
                }

                let confidence = (seed.score_millis as f32 / 1000.0) * (0.55 + schema_score * 0.45);
                ranked.push(GlirelRelationPrediction {
                    head_index: seed.head_index,
                    tail_index: seed.tail_index,
                    head: head.text.clone(),
                    tail: tail.text.clone(),
                    relation: spec.label.clone(),
                    confidence,
                    evidence: seed
                        .evidence
                        .iter()
                        .cloned()
                        .chain(cue_matches.iter().map(|cue| format!("cue:{cue}")))
                        .collect(),
                });
            }

            ranked.sort_by(|left, right| {
                right
                    .confidence
                    .partial_cmp(&left.confidence)
                    .unwrap_or(Ordering::Equal)
            });

            let take_n = spec.max_predictions_per_window.max(1);
            let top_confidence = ranked.first().map(|row| row.confidence).unwrap_or(0.0);
            for prediction in ranked.into_iter().take(take_n) {
                if prediction.confidence + 0.0001 < top_confidence * 0.80 {
                    break;
                }
                predictions.push(prediction);
            }
        }
    }

    finalize_relation_predictions(text, entities, relation_specs, predictions)
}

pub fn finalize_relation_predictions(
    text: &str,
    entities: &[GlirelEntity],
    relation_specs: &[GlirelRelationTypeSpec],
    mut predictions: Vec<GlirelRelationPrediction>,
) -> Vec<GlirelRelationPrediction> {
    repair_relation_directions(entities, relation_specs, &mut predictions);
    boost_context_matches(text, &mut predictions);
    dedupe_predictions(&mut predictions);
    suppress_relation_conflicts(relation_specs, &mut predictions);
    predictions
}

pub fn repair_relation_directions(
    entities: &[GlirelEntity],
    relation_specs: &[GlirelRelationTypeSpec],
    predictions: &mut [GlirelRelationPrediction],
) {
    for prediction in predictions {
        let Some(spec) = relation_specs
            .iter()
            .find(|spec| spec.label == prediction.relation)
        else {
            continue;
        };

        let head = &entities[prediction.head_index];
        let tail = &entities[prediction.tail_index];
        if !spec.directed {
            if head.span_start > tail.span_start {
                swap_prediction_direction(prediction);
            }
            continue;
        }

        let head_valid = type_allowed(&spec.head_types, &head.entity_type);
        let tail_valid = type_allowed(&spec.tail_types, &tail.entity_type);
        let flipped_head_valid = type_allowed(&spec.head_types, &tail.entity_type);
        let flipped_tail_valid = type_allowed(&spec.tail_types, &head.entity_type);

        if (!head_valid || !tail_valid) && flipped_head_valid && flipped_tail_valid {
            swap_prediction_direction(prediction);
        } else if head_valid && tail_valid && flipped_head_valid && flipped_tail_valid {
            if head.span_start > tail.span_start {
                swap_prediction_direction(prediction);
            }
        }
    }
}

pub fn suppress_relation_conflicts(
    relation_specs: &[GlirelRelationTypeSpec],
    predictions: &mut Vec<GlirelRelationPrediction>,
) {
    let mut to_remove = HashSet::new();
    for left_index in 0..predictions.len() {
        for right_index in left_index + 1..predictions.len() {
            let left = &predictions[left_index];
            let right = &predictions[right_index];
            let same_pair = (left.head_index == right.head_index
                && left.tail_index == right.tail_index)
                || (left.head_index == right.tail_index && left.tail_index == right.head_index);
            if !same_pair || left.relation == right.relation {
                continue;
            }

            if !relations_conflict(relation_specs, &left.relation, &right.relation) {
                continue;
            }

            let left_priority = relation_priority(relation_specs, &left.relation);
            let right_priority = relation_priority(relation_specs, &right.relation);
            let left_score = (left.confidence * 1000.0).round() as i32 + left_priority;
            let right_score = (right.confidence * 1000.0).round() as i32 + right_priority;
            if left_score >= right_score {
                to_remove.insert(right_index);
            } else {
                to_remove.insert(left_index);
            }
        }
    }

    if !to_remove.is_empty() {
        let mut index = 0usize;
        predictions.retain(|_| {
            let keep = !to_remove.contains(&index);
            index += 1;
            keep
        });
    }
}

fn build_prompt_words(relation_labels: &[&str]) -> Vec<String> {
    let mut words = Vec::with_capacity(relation_labels.len() * 2 + 1);
    for &label in relation_labels {
        words.push("[REL]".to_owned());
        words.push(label.to_owned());
    }
    words.push("[SEP]".to_owned());
    words
}

fn relation_label_signature(relation_labels: &[&str]) -> u64 {
    let mut hasher = FxHasher::default();
    relation_labels.len().hash(&mut hasher);
    for label in relation_labels {
        label.hash(&mut hasher);
        0xff_u8.hash(&mut hasher);
    }
    hasher.finish()
}

fn prompt_schema_matches_labels(entry: &PromptSchemaCacheEntry, relation_labels: &[&str]) -> bool {
    entry.relation_labels.len() == relation_labels.len()
        && entry
            .relation_labels
            .iter()
            .map(|label| label.as_str())
            .eq(relation_labels.iter().copied())
}

fn append_token_ids(
    token_ids: &[i64],
    word_index: usize,
    input_ids: &mut Vec<i64>,
    word_ids: &mut Vec<Option<usize>>,
) {
    input_ids.extend_from_slice(token_ids);
    word_ids.resize(word_ids.len() + token_ids.len(), Some(word_index));
}

fn build_relation_type_representations(
    word_representations: &[f32],
    relation_count: usize,
) -> Vec<f32> {
    let mut output = Vec::with_capacity(relation_count * PROJECTION_DIM);
    for relation_index in 0..relation_count {
        let rel_word = relation_index * 2;
        let label_word = rel_word + 1;
        let rel_offset = rel_word * PROJECTION_DIM;
        let label_offset = label_word * PROJECTION_DIM;
        for dimension in 0..PROJECTION_DIM {
            output.push(
                (word_representations[rel_offset + dimension]
                    + word_representations[label_offset + dimension])
                    / 2.0,
            );
        }
    }
    output
}

fn decode_predictions(
    entities: &[GlirelEntity],
    relation_labels: &[&str],
    pair_map: &[(usize, usize)],
    scores: &[f32],
    threshold: f32,
) -> Result<Vec<GlirelRelationPrediction>, GlirelError> {
    let relation_count = relation_labels.len();
    let expected_min = pair_map.len() * relation_count;
    if scores.len() < expected_min {
        return Err(GlirelError::Inference(format!(
            "relation_scores too short: expected at least {expected_min}, got {}",
            scores.len()
        )));
    }

    let mut predictions = Vec::new();
    for (pair_index, &(head_index, tail_index)) in pair_map.iter().enumerate() {
        for (relation_index, &label) in relation_labels.iter().enumerate() {
            let flat_index = pair_index * relation_count + relation_index;
            let confidence = sigmoid(scores[flat_index]);
            if confidence < threshold {
                continue;
            }
            predictions.push(GlirelRelationPrediction {
                head_index,
                tail_index,
                head: entities[head_index].text.clone(),
                tail: entities[tail_index].text.clone(),
                relation: label.to_owned(),
                confidence,
                evidence: vec![format!("glirel_score:{:.3}", scores[flat_index])],
            });
        }
    }

    predictions.sort_by(|left, right| {
        right
            .confidence
            .partial_cmp(&left.confidence)
            .unwrap_or(Ordering::Equal)
    });
    Ok(predictions)
}

fn next_schema_batch_len(rows: &[PreparedSchemaBatchRow<'_>]) -> usize {
    let mut count = 0usize;
    let mut max_sequence_len = 0usize;
    let mut max_pair_count = 0usize;

    for row in rows {
        let next_count = count + 1;
        let next_max_sequence_len = max_sequence_len.max(row.input_ids.len());
        let next_max_pair_count = max_pair_count.max(row.pair_map.len());
        let exceeds_budget = next_count > MAX_SCHEMA_BATCH_WINDOWS
            || next_max_sequence_len.saturating_mul(next_count) > MAX_SCHEMA_BATCH_SEQUENCE_TOKENS
            || next_max_pair_count.saturating_mul(next_count) > MAX_SCHEMA_BATCH_PAIR_SLOTS;
        if exceeds_budget && count > 0 {
            break;
        }
        count = next_count;
        max_sequence_len = next_max_sequence_len;
        max_pair_count = next_max_pair_count;
    }

    count.max(1)
}

fn boost_context_matches(text: &str, predictions: &mut [GlirelRelationPrediction]) {
    let text_lower = text.to_lowercase();
    let keyword_sets: &[(&str, &[&str])] = &[
        (
            "chose",
            &["chose", "choose", "selected", "picked", "went with"],
        ),
        ("rejected", &["rejected", "ruled out", "passed on"]),
        (
            "replaced",
            &["replaced", "switched from", "migrated from", "moved from"],
        ),
        (
            "depends_on",
            &["depends on", "relies on", "built on", "uses"],
        ),
        (
            "introduced",
            &["introduced", "added", "implemented", "rolled out"],
        ),
        (
            "deprecated",
            &["deprecated", "removed", "sunset", "dropped"],
        ),
        ("caused", &["caused", "resulted in", "led to", "triggered"]),
        ("fixed", &["fixed", "resolved", "patched", "addressed"]),
        (
            "constrained_by",
            &[
                "constrained by",
                "limited by",
                "compliance",
                "requirement",
                "must",
            ],
        ),
    ];

    for prediction in predictions.iter_mut() {
        for &(label, keywords) in keyword_sets {
            if prediction.relation != label {
                continue;
            }
            if keywords.iter().any(|keyword| text_lower.contains(keyword)) {
                prediction.confidence *= 1.3;
                prediction.evidence.push("context_keyword_boost".to_owned());
            }
            break;
        }
    }

    predictions.sort_by(|left, right| {
        right
            .confidence
            .partial_cmp(&left.confidence)
            .unwrap_or(Ordering::Equal)
    });
}

fn dedupe_predictions(predictions: &mut Vec<GlirelRelationPrediction>) {
    let mut seen_undirected = HashSet::new();
    predictions.retain(|prediction| {
        let key = if prediction.head_index < prediction.tail_index {
            (
                prediction.head_index,
                prediction.tail_index,
                prediction.relation.clone(),
            )
        } else {
            (
                prediction.tail_index,
                prediction.head_index,
                prediction.relation.clone(),
            )
        };
        seen_undirected.insert(key)
    });

    let mut seen_directed = HashSet::new();
    predictions.retain(|prediction| {
        seen_directed.insert((
            prediction.head_index,
            prediction.tail_index,
            prediction.relation.clone(),
        ))
    });
}

fn swap_prediction_direction(prediction: &mut GlirelRelationPrediction) {
    std::mem::swap(&mut prediction.head_index, &mut prediction.tail_index);
    std::mem::swap(&mut prediction.head, &mut prediction.tail);
    prediction
        .evidence
        .push("direction_flipped_by_schema".to_owned());
}

fn type_allowed(allowed: &[String], actual: &str) -> bool {
    allowed.is_empty() || allowed.iter().any(|label| label == actual)
}

fn relations_conflict(relation_specs: &[GlirelRelationTypeSpec], left: &str, right: &str) -> bool {
    relation_specs.iter().any(|spec| {
        (spec.label == left && spec.conflicts_with.iter().any(|label| label == right))
            || (spec.label == right && spec.conflicts_with.iter().any(|label| label == left))
    })
}

fn relation_priority(relation_specs: &[GlirelRelationTypeSpec], label: &str) -> i32 {
    relation_specs
        .iter()
        .find(|spec| spec.label == label)
        .map(|spec| spec.priority_millis)
        .unwrap_or_default()
}

fn directed_pair_map(entity_count: usize) -> Vec<(usize, usize)> {
    let mut output =
        Vec::with_capacity(entity_count.saturating_mul(entity_count.saturating_sub(1)));
    for head_index in 0..entity_count {
        for tail_index in 0..entity_count {
            if head_index != tail_index {
                output.push((head_index, tail_index));
            }
        }
    }
    output
}

fn relation_cue_matches(
    spec: &GlirelRelationTypeSpec,
    window_lower: &str,
    text_lower: &str,
) -> Vec<String> {
    let cues = if spec.cue_phrases.is_empty() {
        default_relation_cues(&spec.label)
    } else {
        spec.cue_phrases.clone()
    };

    cues.into_iter()
        .filter(|cue| window_lower.contains(cue) || text_lower.contains(cue))
        .collect()
}

fn default_relation_cues(label: &str) -> Vec<String> {
    match label {
        "supports" => vec![
            "supports".to_owned(),
            "helped".to_owned(),
            "backed".to_owned(),
        ],
        "contradicts" => vec![
            "but".to_owned(),
            "however".to_owned(),
            "contradicted".to_owned(),
        ],
        "same_event_as" => vec!["same moment".to_owned(), "same event".to_owned()],
        "caused" => vec![
            "caused".to_owned(),
            "led to".to_owned(),
            "triggered".to_owned(),
        ],
        "introduced" => vec![
            "introduced".to_owned(),
            "added".to_owned(),
            "brought".to_owned(),
        ],
        "replaced" => vec![
            "replaced".to_owned(),
            "instead of".to_owned(),
            "in place of".to_owned(),
        ],
        "depends_on" => vec![
            "depends on".to_owned(),
            "relies on".to_owned(),
            "uses".to_owned(),
        ],
        other => other
            .split(['_', '-', ' '])
            .map(str::trim)
            .filter(|part| part.len() > 2)
            .map(ToOwned::to_owned)
            .collect(),
    }
}

fn schema_pair_score(
    spec: &GlirelRelationTypeSpec,
    head: &GlirelEntity,
    tail: &GlirelEntity,
) -> f32 {
    let head_valid = type_allowed(&spec.head_types, &head.entity_type);
    let tail_valid = type_allowed(&spec.tail_types, &tail.entity_type);
    let flipped_head_valid = type_allowed(&spec.head_types, &tail.entity_type);
    let flipped_tail_valid = type_allowed(&spec.tail_types, &head.entity_type);
    match (
        head_valid && tail_valid,
        flipped_head_valid && flipped_tail_valid,
    ) {
        (true, true) => 1.0,
        (true, false) => 0.92,
        (false, true) => 0.76,
        (false, false) if spec.head_types.is_empty() && spec.tail_types.is_empty() => 0.72,
        _ => 0.0,
    }
}

fn relation_window_distance(
    windows: &[GlirelSentenceWindow],
    head: &GlirelEntity,
    tail: &GlirelEntity,
) -> Option<usize> {
    let head_index = windows
        .iter()
        .find(|window| entity_in_window(window, head))?
        .index;
    let tail_index = windows
        .iter()
        .find(|window| entity_in_window(window, tail))?
        .index;
    Some(head_index.abs_diff(tail_index))
}

fn entity_pair_in_window(
    window: &GlirelSentenceWindow,
    head: &GlirelEntity,
    tail: &GlirelEntity,
    radius: usize,
    windows: &[GlirelSentenceWindow],
) -> bool {
    let Some(head_distance) = windows
        .iter()
        .find(|candidate| entity_in_window(candidate, head))
        .map(|candidate| candidate.index.abs_diff(window.index))
    else {
        return false;
    };
    let Some(tail_distance) = windows
        .iter()
        .find(|candidate| entity_in_window(candidate, tail))
        .map(|candidate| candidate.index.abs_diff(window.index))
    else {
        return false;
    };
    head_distance <= radius && tail_distance <= radius
}

fn entity_in_window(window: &GlirelSentenceWindow, entity: &GlirelEntity) -> bool {
    entity.span_start < window.end && entity.span_end > window.start
}

fn entity_char_distance(head: &GlirelEntity, tail: &GlirelEntity) -> usize {
    if head.span_end <= tail.span_start {
        tail.span_start - head.span_end
    } else if tail.span_end <= head.span_start {
        head.span_start - tail.span_end
    } else {
        0
    }
}

fn entity_name_quality(name: &str) -> f32 {
    let token_count = name.split_whitespace().count().max(1);
    let capitalized = name.chars().next().is_some_and(|char| char.is_uppercase());
    let mixed_case = name.chars().any(|char| char.is_uppercase())
        && name.chars().any(|char| char.is_lowercase());
    let alphabetic_ratio = {
        let total = name.chars().count().max(1) as f32;
        let alpha = name.chars().filter(|char| char.is_alphabetic()).count() as f32;
        alpha / total
    };

    let mut score = 0.45;
    if capitalized {
        score += 0.18;
    }
    if mixed_case || token_count > 1 {
        score += 0.12;
    }
    score += alphabetic_ratio * 0.15;
    score.clamp(0.2, 1.0)
}

fn looks_like_named_entity(name: &str) -> bool {
    name.split_whitespace()
        .any(|token| token.chars().next().is_some_and(|char| char.is_uppercase()))
}

fn first_subword_pool(
    token_representations: &[f32],
    word_ids: &[Option<usize>],
    word_count: usize,
    dimension: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; word_count * dimension];
    let mut seen = vec![false; word_count];

    for (token_index, word_index) in word_ids.iter().enumerate() {
        if let Some(word_index) = *word_index {
            if word_index >= word_count || seen[word_index] {
                continue;
            }
            seen[word_index] = true;
            let src = token_index * dimension;
            let dst = word_index * dimension;
            output[dst..dst + dimension]
                .copy_from_slice(&token_representations[src..src + dimension]);
        }
    }

    output
}

fn entities_to_word_spans(
    text: &str,
    words: &[&str],
    entities: &[GlirelEntity],
) -> Vec<(i64, i64)> {
    let mut word_char_spans = Vec::with_capacity(words.len());
    let mut byte_position = 0usize;
    for &word in words {
        if let Some(relative) = text[byte_position..].find(word) {
            let absolute = byte_position + relative;
            let char_start = text[..absolute].chars().count();
            let char_end = char_start + word.chars().count();
            word_char_spans.push((char_start, char_end));
            byte_position = absolute + word.len();
        } else {
            let last_end = word_char_spans.last().map(|&(_, end)| end).unwrap_or(0);
            word_char_spans.push((last_end, last_end + word.chars().count()));
        }
    }

    entities
        .iter()
        .map(|entity| {
            let mut start_word = 0i64;
            let mut end_word = 0i64;
            let mut found = false;
            for (word_index, &(word_start, word_end)) in word_char_spans.iter().enumerate() {
                if word_end > entity.span_start && word_start < entity.span_end {
                    if !found {
                        start_word = word_index as i64;
                        found = true;
                    }
                    end_word = word_index as i64;
                }
            }
            (start_word, end_word)
        })
        .collect()
}

fn build_relations_idx(span_index: &[(i64, i64)]) -> Vec<i64> {
    let entity_count = span_index.len();
    let mut output = Vec::with_capacity(entity_count * entity_count.saturating_sub(1) * 4);
    for head_index in 0..entity_count {
        for tail_index in 0..entity_count {
            if head_index == tail_index {
                continue;
            }
            output.push(span_index[head_index].0);
            output.push(span_index[head_index].1);
            output.push(span_index[tail_index].0);
            output.push(span_index[tail_index].1);
        }
    }
    output
}

fn find_existing_path(dir: &Path, candidates: &[&str]) -> Result<PathBuf, GlirelError> {
    for candidate in candidates {
        let path = dir.join(candidate);
        if path.exists() {
            return Ok(path);
        }
    }
    Err(GlirelError::ModelLoad(format!(
        "could not find any of {:?} under {}",
        candidates,
        dir.display()
    )))
}

fn load_f32_bin(path: &Path, expected_len: usize) -> Result<Vec<f32>, GlirelError> {
    let bytes = std::fs::read(path)
        .map_err(|error| GlirelError::ModelLoad(format!("read {}: {error}", path.display())))?;
    if bytes.len() != expected_len * 4 {
        return Err(GlirelError::ModelLoad(format!(
            "{} expected {} f32 values ({} bytes), got {} bytes",
            path.display(),
            expected_len,
            expected_len * 4,
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_behaves() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn first_subword_pool_uses_first_piece() {
        let reps = vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0];
        let word_ids = vec![None, Some(0), Some(0), Some(1), None];
        let pooled = first_subword_pool(&reps, &word_ids, 2, 2);
        assert_eq!(pooled, vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn entities_to_word_spans_handles_multiword_entities() {
        let text = "We chose Apache Kafka for messaging";
        let words = text.split_whitespace().collect::<Vec<_>>();
        let entities = vec![GlirelEntity {
            text: "Apache Kafka".to_owned(),
            entity_type: "Component".to_owned(),
            span_start: 10,
            span_end: 22,
            entity_id: None,
        }];
        let spans = entities_to_word_spans(text, &words, &entities);
        assert_eq!(spans[0], (2, 3));
    }

    #[test]
    fn build_relations_idx_emits_directed_pairs() {
        let spans = vec![(0, 0), (3, 3), (5, 6)];
        let index = build_relations_idx(&spans);
        assert_eq!(index.len(), 24);
        assert_eq!(&index[0..4], &[0, 0, 3, 3]);
        assert_eq!(&index[4..8], &[0, 0, 5, 6]);
    }

    #[test]
    fn decode_predictions_applies_threshold() {
        let entities = vec![
            GlirelEntity {
                text: "Alice".to_owned(),
                entity_type: "Person".to_owned(),
                span_start: 0,
                span_end: 5,
                entity_id: None,
            },
            GlirelEntity {
                text: "PostgreSQL".to_owned(),
                entity_type: "Database".to_owned(),
                span_start: 13,
                span_end: 23,
                entity_id: None,
            },
        ];
        let pair_map = vec![(0, 1), (1, 0)];
        let scores = vec![3.0, -4.0];
        let predictions =
            decode_predictions(&entities, &["chose"], &pair_map, &scores, 0.8).expect("decode");
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0].head, "Alice");
        assert_eq!(predictions[0].tail, "PostgreSQL");
        assert_eq!(predictions[0].relation, "chose");
    }

    #[test]
    fn dedupe_predictions_keeps_highest_confidence_direction() {
        let mut predictions = vec![
            GlirelRelationPrediction {
                head_index: 0,
                tail_index: 1,
                head: "Alice".to_owned(),
                tail: "PostgreSQL".to_owned(),
                relation: "chose".to_owned(),
                confidence: 0.91,
                evidence: Vec::new(),
            },
            GlirelRelationPrediction {
                head_index: 1,
                tail_index: 0,
                head: "PostgreSQL".to_owned(),
                tail: "Alice".to_owned(),
                relation: "chose".to_owned(),
                confidence: 0.89,
                evidence: Vec::new(),
            },
        ];
        dedupe_predictions(&mut predictions);
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0].head, "Alice");
    }

    #[test]
    fn extract_with_schema_can_flip_direction() {
        let mut prediction = GlirelRelationPrediction {
            head_index: 0,
            tail_index: 1,
            head: "PostgreSQL".to_owned(),
            tail: "Alice".to_owned(),
            relation: "chosen_by".to_owned(),
            confidence: 0.9,
            evidence: Vec::new(),
        };
        swap_prediction_direction(&mut prediction);
        assert_eq!(prediction.head, "Alice");
        assert_eq!(prediction.tail, "PostgreSQL");
    }

    #[test]
    fn split_sentence_windows_breaks_text_cleanly() {
        let windows = split_sentence_windows("Alice met Bob. Carol argued back!\n\nDave left?");
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0].text, "Alice met Bob.");
        assert_eq!(windows[1].text, "Carol argued back!");
        assert_eq!(windows[2].text, "Dave left?");
    }

    #[test]
    fn relation_label_signature_tracks_schema_order() {
        let left = relation_label_signature(&["works_for", "member_of"]);
        let right = relation_label_signature(&["works_for", "member_of"]);
        let flipped = relation_label_signature(&["member_of", "works_for"]);
        assert_eq!(left, right);
        assert_ne!(left, flipped);
    }

    #[test]
    fn seed_relation_pairs_prefers_nearby_named_entities() {
        let text = "Alice told Bob the truth. Far away, the weather changed.";
        let entities = vec![
            GlirelEntity {
                text: "Alice".to_owned(),
                entity_type: "Person".to_owned(),
                span_start: 0,
                span_end: 5,
                entity_id: None,
            },
            GlirelEntity {
                text: "Bob".to_owned(),
                entity_type: "Person".to_owned(),
                span_start: 11,
                span_end: 14,
                entity_id: None,
            },
            GlirelEntity {
                text: "weather".to_owned(),
                entity_type: "Concept".to_owned(),
                span_start: 38,
                span_end: 45,
                entity_id: None,
            },
        ];

        let seeds = seed_relation_pairs(text, &entities, &GlirelProposalConfig::default());
        assert!(seeds
            .iter()
            .any(|seed| seed.head_index == 0 && seed.tail_index == 1));
        assert!(seeds.first().is_some_and(|seed| seed.score_millis >= 330));
    }

    #[test]
    fn extract_heuristic_relations_uses_cues_and_schema() {
        let text = "Alice introduced Bob to Carol at the market.";
        let entities = vec![
            GlirelEntity {
                text: "Alice".to_owned(),
                entity_type: "Person".to_owned(),
                span_start: 0,
                span_end: 5,
                entity_id: None,
            },
            GlirelEntity {
                text: "Bob".to_owned(),
                entity_type: "Person".to_owned(),
                span_start: 17,
                span_end: 20,
                entity_id: None,
            },
            GlirelEntity {
                text: "Carol".to_owned(),
                entity_type: "Person".to_owned(),
                span_start: 24,
                span_end: 29,
                entity_id: None,
            },
        ];
        let specs = vec![GlirelRelationTypeSpec {
            label: "introduced".to_owned(),
            head_types: vec!["Person".to_owned()],
            tail_types: vec!["Person".to_owned()],
            cue_phrases: vec!["introduced".to_owned()],
            conflicts_with: Vec::new(),
            priority_millis: 100,
            accept_threshold_millis: 700,
            review_threshold_millis: 450,
            max_predictions_per_window: 1,
            directed: true,
        }];

        let rows =
            extract_heuristic_relations(text, &entities, &specs, &GlirelProposalConfig::default());
        assert!(!rows.is_empty());
        assert_eq!(rows[0].relation, "introduced");
        assert!(rows[0]
            .evidence
            .iter()
            .any(|value| value.starts_with("cue:")));
    }

    #[test]
    fn suppress_relation_conflicts_uses_priority() {
        let specs = vec![
            GlirelRelationTypeSpec {
                label: "introduced".to_owned(),
                head_types: Vec::new(),
                tail_types: Vec::new(),
                cue_phrases: Vec::new(),
                conflicts_with: vec!["depends_on".to_owned()],
                priority_millis: 250,
                accept_threshold_millis: 700,
                review_threshold_millis: 450,
                max_predictions_per_window: 1,
                directed: true,
            },
            GlirelRelationTypeSpec {
                label: "depends_on".to_owned(),
                head_types: Vec::new(),
                tail_types: Vec::new(),
                cue_phrases: Vec::new(),
                conflicts_with: vec!["introduced".to_owned()],
                priority_millis: 0,
                accept_threshold_millis: 700,
                review_threshold_millis: 450,
                max_predictions_per_window: 1,
                directed: true,
            },
        ];
        let mut predictions = vec![
            GlirelRelationPrediction {
                head_index: 0,
                tail_index: 1,
                head: "Alice".to_owned(),
                tail: "Plan".to_owned(),
                relation: "introduced".to_owned(),
                confidence: 0.61,
                evidence: Vec::new(),
            },
            GlirelRelationPrediction {
                head_index: 0,
                tail_index: 1,
                head: "Alice".to_owned(),
                tail: "Plan".to_owned(),
                relation: "depends_on".to_owned(),
                confidence: 0.72,
                evidence: Vec::new(),
            },
        ];
        suppress_relation_conflicts(&specs, &mut predictions);
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0].relation, "introduced");
    }

    #[test]
    fn next_schema_batch_len_respects_token_budget() {
        let row = |output_index| PreparedSchemaBatchRow {
            output_index,
            text: "Alice works for Dynamis.",
            entities: &[],
            prompt_word_count: 3,
            text_word_count: 4,
            input_ids: vec![1_i64; 1_800],
            attention_mask: vec![1_i64; 1_800],
            word_ids: vec![None; 1_800],
            span_index: Vec::new(),
            relation_pairs: Vec::new(),
            pair_map: vec![(0, 1); 8],
        };
        let rows = vec![row(0), row(1), row(2)];
        assert_eq!(next_schema_batch_len(&rows), 2);
    }
}
