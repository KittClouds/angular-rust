use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{Device, IndexOp, Tensor};
use candle_onnx::{read_file, simple_eval};
use phoenix_hyperbolic::{
    HyperbolicDiskHnsw, HyperbolicHnswBuilder, HnswBuildParams, MetricF32, PoincareMetric,
};
use serde::Serialize;
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeNeighbor {
    index: usize,
    score: f32,
    text: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeQueryReport {
    query: String,
    recall_at_k: f32,
    hnsw: Vec<SmokeNeighbor>,
    brute_force: Vec<SmokeNeighbor>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    model: String,
    corpus_id: String,
    chunk_count: usize,
    query_count: usize,
    dims: usize,
    load_ms: u64,
    embed_ms: u64,
    build_ms: u64,
    save_load_ms: u64,
    mean_recall_at_k: f32,
    queries: Vec<SmokeQueryReport>,
}

#[derive(Clone, Debug)]
struct SmokeConfig {
    corpus_id: Option<String>,
    input_path: Option<PathBuf>,
    chunk_limit: usize,
    top_k: usize,
    ef_search: usize,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        Self {
            corpus_id: None,
            input_path: None,
            chunk_limit: 96,
            top_k: 5,
            ef_search: 64,
        }
    }
}

#[derive(Clone, Debug)]
struct SmokeBundle {
    corpus_id: String,
    chunks: Vec<String>,
    queries: Vec<String>,
}

struct CandleEmbedder {
    tokenizer: Tokenizer,
    model: candle_onnx::onnx::ModelProto,
    device: Device,
}

impl CandleEmbedder {
    fn load(model_root: &Path) -> Result<Self, String> {
        let tokenizer_path = model_root.join("tokenizer.json");
        let model_path = model_root.join("onnx").join("model.onnx");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|error| format!("failed to load tokenizer {}: {error}", tokenizer_path.display()))?;
        let model = read_file(&model_path)
            .map_err(|error| format!("failed to load ONNX model {}: {error}", model_path.display()))?;
        Ok(Self {
            tokenizer,
            model,
            device: Device::Cpu,
        })
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|error| format!("failed to encode input: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let max_len = encodings
            .iter()
            .map(|encoding| encoding.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let batch = encodings.len();

        let mut input_ids = vec![0i64; batch * max_len];
        let mut attention_mask = vec![0i64; batch * max_len];
        let mut token_type_ids = vec![0i64; batch * max_len];

        for (row, encoding) in encodings.iter().enumerate() {
            let row_offset = row * max_len;
            for (col, token_id) in encoding.get_ids().iter().enumerate() {
                input_ids[row_offset + col] = *token_id as i64;
            }
            for (col, mask) in encoding.get_attention_mask().iter().enumerate() {
                attention_mask[row_offset + col] = *mask as i64;
            }
            for (col, token_type) in encoding.get_type_ids().iter().enumerate() {
                token_type_ids[row_offset + col] = *token_type as i64;
            }
        }

        let input_shape = (batch, max_len);
        let mut inputs = HashMap::<String, Tensor>::new();
        inputs.insert(
            "input_ids".to_owned(),
            Tensor::from_vec(input_ids, input_shape, &self.device)
                .map_err(|error| format!("failed to build input_ids tensor: {error}"))?,
        );
        inputs.insert(
            "attention_mask".to_owned(),
            Tensor::from_vec(attention_mask, input_shape, &self.device)
                .map_err(|error| format!("failed to build attention_mask tensor: {error}"))?,
        );
        if graph_has_input(&self.model, "token_type_ids") {
            inputs.insert(
                "token_type_ids".to_owned(),
                Tensor::from_vec(token_type_ids, input_shape, &self.device)
                    .map_err(|error| format!("failed to build token_type_ids tensor: {error}"))?,
            );
        }

        let outputs = simple_eval(&self.model, inputs)
            .map_err(|error| format!("failed to evaluate ONNX graph: {error}"))?;
        let hidden = select_hidden_output(&self.model, outputs)?;
        let cls = hidden
            .i((.., 0, ..))
            .map_err(|error| format!("failed to select CLS embedding: {error}"))?;
        let squared = cls
            .sqr()
            .map_err(|error| format!("failed to square embeddings: {error}"))?;
        let summed = squared
            .sum_keepdim(1)
            .map_err(|error| format!("failed to sum embeddings: {error}"))?;
        let norms = summed
            .sqrt()
            .map_err(|error| format!("failed to compute embedding norm: {error}"))?;
        let normalized = cls
            .broadcast_div(&norms)
            .map_err(|error| format!("failed to normalize embeddings: {error}"))?;
        normalized
            .to_vec2::<f32>()
            .map_err(|error| format!("failed to extract embedding rows: {error}"))
    }
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let config = SmokeConfig {
        corpus_id: args
            .windows(2)
            .find_map(|window| (window[0] == "--corpus").then(|| window[1].clone())),
        input_path: args
            .windows(2)
            .find_map(|window| (window[0] == "--input").then(|| PathBuf::from(&window[1]))),
        chunk_limit: parse_usize_arg(&args, "--chunk-limit")
            .unwrap_or_else(|| SmokeConfig::default().chunk_limit)
            .max(1),
        top_k: parse_usize_arg(&args, "--top-k")
            .unwrap_or_else(|| SmokeConfig::default().top_k)
            .max(1),
        ef_search: parse_usize_arg(&args, "--ef-search")
            .unwrap_or_else(|| SmokeConfig::default().ef_search)
            .max(4),
    };

    match run_smoke(&config) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("serialize smoke report")
            );
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run_smoke(config: &SmokeConfig) -> Result<SmokeReport, String> {
    let bundle = load_bundle(config)?;
    let model_name = "Snowflake/snowflake-arctic-embed-xs";
    let model_root = model_root();

    let load_started = Instant::now();
    let model = CandleEmbedder::load(&model_root)?;
    let load_ms = load_started.elapsed().as_millis() as u64;

    let embed_started = Instant::now();
    let doc_embeddings = model.embed(&bundle.chunks)?;
    let query_embeddings = model.embed(&bundle.queries)?;
    let embed_ms = embed_started.elapsed().as_millis() as u64;

    let dims = doc_embeddings
        .first()
        .map(|embedding| embedding.len())
        .ok_or_else(|| "no embeddings were produced".to_owned())?;

    let metric = PoincareMetric { curvature: 1.0 };
    let params = HnswBuildParams {
        m: 16,
        m0: 32,
        ef_construction: config.ef_search.max(64),
        level_mult: 1.0 / (16.0_f32).ln(),
    };

    let build_started = Instant::now();
    let mut builder = HyperbolicHnswBuilder::new(dims, metric, params);
    for embedding in &doc_embeddings {
        builder.insert(embedding.clone());
    }
    let packed = builder.into_packed();
    let build_ms = build_started.elapsed().as_millis() as u64;

    let save_load_started = Instant::now();
    let temp_path = env::temp_dir().join(format!(
        "phoenix-hyperbolic-smoke-{}-{}.bin",
        bundle.corpus_id,
        now_ms()
    ));
    packed
        .write_to_file(&temp_path.to_string_lossy())
        .map_err(|error| format!("failed to write index: {error}"))?;
    let index = HyperbolicDiskHnsw::open(&temp_path.to_string_lossy(), metric)
        .map_err(|error| format!("failed to reopen index: {error}"))?;
    let save_load_ms = save_load_started.elapsed().as_millis() as u64;
    let _ = fs::remove_file(&temp_path);

    let mut queries = Vec::with_capacity(bundle.queries.len());
    let mut recall_sum = 0.0f32;
    for (query, query_embedding) in bundle.queries.iter().zip(query_embeddings.iter()) {
        let brute = brute_force_neighbors(query_embedding, &doc_embeddings, config.top_k, metric);
        let hnsw = index.search(query_embedding, config.top_k, config.ef_search);
        let overlap = hnsw
            .iter()
            .filter(|candidate| brute.iter().any(|(ix, _)| candidate.id == *ix as u32))
            .count();
        let recall = overlap as f32 / config.top_k as f32;
        recall_sum += recall;

        queries.push(SmokeQueryReport {
            query: query.clone(),
            recall_at_k: recall,
            hnsw: hnsw
                .iter()
                .map(|candidate| SmokeNeighbor {
                    index: candidate.id as usize,
                    score: candidate.dist,
                    text: bundle
                        .chunks
                        .get(candidate.id as usize)
                        .cloned()
                        .unwrap_or_default(),
                })
                .collect(),
            brute_force: brute
                .iter()
                .map(|(ix, score)| SmokeNeighbor {
                    index: *ix,
                    score: *score,
                    text: bundle.chunks.get(*ix).cloned().unwrap_or_default(),
                })
                .collect(),
        });
    }

    Ok(SmokeReport {
        model: model_name.to_owned(),
        corpus_id: bundle.corpus_id,
        chunk_count: bundle.chunks.len(),
        query_count: bundle.queries.len(),
        dims,
        load_ms,
        embed_ms,
        build_ms,
        save_load_ms,
        mean_recall_at_k: if queries.is_empty() {
            0.0
        } else {
            recall_sum / queries.len() as f32
        },
        queries,
    })
}

fn graph_has_input(model: &candle_onnx::onnx::ModelProto, name: &str) -> bool {
    model.graph
        .as_ref()
        .map(|graph| graph.input.iter().any(|input| input.name == name))
        .unwrap_or(false)
}

fn select_hidden_output(
    model: &candle_onnx::onnx::ModelProto,
    outputs: HashMap<String, Tensor>,
) -> Result<Tensor, String> {
    for preferred in ["last_hidden_state", "token_embeddings"] {
        if let Some(tensor) = outputs.get(preferred) {
            return Ok(tensor.clone());
        }
    }
    if let Some(graph) = model.graph.as_ref() {
        for output in &graph.output {
            if let Some(tensor) = outputs.get(&output.name) {
                return Ok(tensor.clone());
            }
        }
    }
    outputs
        .into_iter()
        .next()
        .map(|(_, tensor)| tensor)
        .ok_or_else(|| "ONNX model returned no outputs".to_owned())
}

fn load_bundle(config: &SmokeConfig) -> Result<SmokeBundle, String> {
    if let Some(path) = &config.input_path {
        let text = fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let chunks = paragraph_chunks(&text, config.chunk_limit);
        let queries = derive_queries(&chunks);
        return Ok(SmokeBundle {
            corpus_id: path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("input")
                .to_owned(),
            chunks,
            queries,
        });
    }

    if let Some(corpus_id) = &config.corpus_id {
        let path = docs_path(corpus_id)?;
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let chunks = paragraph_chunks(&text, config.chunk_limit);
        let queries = derive_queries(&chunks);
        return Ok(SmokeBundle {
            corpus_id: corpus_id.clone(),
            chunks,
            queries,
        });
    }

    Ok(SmokeBundle {
        corpus_id: "built_in".to_owned(),
        chunks: vec![
            "passage: Luffy gathered the crew before dawn and outlined the escape route from Water Seven.".to_owned(),
            "passage: Nami kept the map dry and warned the crew about the marine patrol near the harbor.".to_owned(),
            "passage: Zoro stayed behind on the roof to watch for snipers while the others moved toward the ship.".to_owned(),
            "passage: Robin studied the ancient inscription and noticed the final line had been carved recently.".to_owned(),
            "passage: Franky reinforced the damaged hull with scavenged steel plates before the storm arrived.".to_owned(),
            "passage: The archivist hid the ledger inside the bell tower after the council announced the seizure.".to_owned(),
            "passage: A burst pipe flooded the lower tunnel and forced the smugglers to abandon the cache.".to_owned(),
            "passage: The captain delayed the launch because the signal lantern from the cliffs never appeared.".to_owned(),
            "passage: A quiet argument in the engine room revealed that the fuel reserve had been siphoned overnight.".to_owned(),
            "passage: The medic treated the burn and ordered everyone away from the cracked reactor casing.".to_owned(),
            "passage: The witness insisted the judge accepted the bribe before the verdict was read aloud.".to_owned(),
            "passage: The rebels used the festival drums to mask the sound of carts carrying supplies into the old district.".to_owned(),
        ],
        queries: vec![
            "query: Which passage is about marines near the harbor?".to_owned(),
            "query: Where was the ledger hidden after the council acted?".to_owned(),
            "query: Which passage mentions a delayed launch caused by a missing signal?".to_owned(),
            "query: Which passage is about tunnel flooding forcing smugglers to retreat?".to_owned(),
        ],
    })
}

fn model_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("snowflake-arctic-embed-xs")
}

fn docs_path(corpus_id: &str) -> Result<PathBuf, String> {
    let filename = match corpus_id {
        "perfect_run" => "perfect_run.md",
        "shortrun" => "shortrun.md",
        other => return Err(format!("unsupported corpus id: {other}")),
    };
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = crate_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| "failed to resolve repository root".to_owned())?;
    Ok(repo_root.join("docs").join(filename))
}

fn paragraph_chunks(text: &str, limit: usize) -> Vec<String> {
    let chunks = text
        .split("\n\n")
        .map(str::trim)
        .filter(|chunk| !chunk.is_empty())
        .map(|chunk| format!("passage: {}", compact_whitespace(chunk)))
        .take(limit)
        .collect::<Vec<_>>();
    if chunks.len() >= 2 {
        return chunks;
    }
    windowed_sentence_chunks(text, limit, 1400)
}

fn windowed_sentence_chunks(text: &str, limit: usize, target_chars: usize) -> Vec<String> {
    let sentences = text
        .split_inclusive(['.', '!', '?'])
        .map(str::trim)
        .filter(|sentence| !sentence.is_empty())
        .collect::<Vec<_>>();
    if sentences.is_empty() {
        return vec![format!("passage: {}", compact_whitespace(text))];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    for sentence in sentences {
        if !current.is_empty() && current.len() + sentence.len() + 1 > target_chars {
            chunks.push(format!("passage: {}", compact_whitespace(&current)));
            if chunks.len() >= limit {
                return chunks;
            }
            current.clear();
        }
        if !current.is_empty() {
            current.push(' ');
        }
        current.push_str(sentence);
    }
    if !current.is_empty() && chunks.len() < limit {
        chunks.push(format!("passage: {}", compact_whitespace(&current)));
    }
    if chunks.is_empty() {
        chunks.push(format!("passage: {}", compact_whitespace(text)));
    }
    chunks
}

fn derive_queries(chunks: &[String]) -> Vec<String> {
    let mut queries = Vec::new();
    for chunk in chunks.iter().take(8) {
        let raw = chunk.strip_prefix("passage: ").unwrap_or(chunk);
        let summary = raw
            .split(['.', '!', '?'])
            .find(|segment| !segment.trim().is_empty())
            .unwrap_or(raw)
            .trim();
        if !summary.is_empty() {
            queries.push(format!("query: {summary}"));
        }
    }
    if queries.is_empty() {
        queries.push("query: summarize the main event".to_owned());
    }
    queries
}

fn compact_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn brute_force_neighbors(
    query: &[f32],
    docs: &[Vec<f32>],
    top_k: usize,
    metric: PoincareMetric,
) -> Vec<(usize, f32)> {
    let mut scored = docs
        .iter()
        .enumerate()
        .map(|(index, doc)| (index, metric.eval(query, doc)))
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(top_k);
    scored
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].parse::<usize>().ok()))
        .flatten()
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}
