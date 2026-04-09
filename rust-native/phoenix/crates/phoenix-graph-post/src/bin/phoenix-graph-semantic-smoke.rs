use std::env;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Instant;

use memmap2::MmapOptions;
use phoenix_graph_post::semantic::{
    build_chunk_node_records, build_chunk_nodes, collect_chunk_neighbors, default_ort_dylib_path,
    embed_chunks, first_half_markdown, semantic_embedder, workspace_root, SemanticChunkConfig,
    SemanticChunkNeighbor, SemanticEmbedConfig,
};
use phoenix_store_native_core::PhoenixSemanticIndexStore;
use phoenix_store_native_core::SEMANTIC_VECTOR_DIM;
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::ScopeKey;
use serde::Serialize;

#[derive(Clone, Debug)]
struct SmokeConfig {
    input_path: PathBuf,
    document_id: String,
    narrative_id: Option<String>,
    store_path: PathBuf,
    embed: SemanticEmbedConfig,
    chunk: SemanticChunkConfig,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        let root = workspace_root();
        Self {
            input_path: root.join("docs").join("shortrun.md"),
            document_id: "shortrun-half-a".to_owned(),
            narrative_id: Some("shortrun".to_owned()),
            store_path: default_store_path(),
            embed: SemanticEmbedConfig::default(),
            chunk: SemanticChunkConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RichNeighborHit {
    source_node_id: String,
    target_node_id: String,
    distance: f64,
    source_chunk_index: usize,
    target_chunk_index: Option<usize>,
    source_excerpt: String,
    target_excerpt: Option<String>,
    #[serde(default)]
    evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    model: &'static str,
    profile: String,
    input_path: String,
    store_path: String,
    ort_dylib_path: Option<String>,
    model_root: String,
    embedding_dim: usize,
    persisted_to_semantic_index: bool,
    store_skip_reason: Option<String>,
    total_bytes: usize,
    used_bytes: usize,
    chunk_count: usize,
    query_count: usize,
    neighbor_limit: usize,
    embed_ms: u64,
    store_ms: u64,
    query_ms: u64,
    neighbors: Vec<RichNeighborHit>,
}

fn main() {
    match run(parse_args(env::args().skip(1).collect())) {
        Ok(report) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&report).expect("serialize semantic smoke report")
            );
        }
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run(config: SmokeConfig) -> Result<SmokeReport, Box<dyn std::error::Error>> {
    let ort_path = ensure_ort_path();

    let file = File::open(&config.input_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let total_bytes = mmap.len();
    let text = first_half_markdown(&mmap[..])?;
    let used_bytes = text.len();
    let chunks = build_chunk_nodes(&config.document_id, text, &config.chunk);

    let embed_started = Instant::now();
    let model = semantic_embedder(&config.embed)?;
    let embeddings = embed_chunks(&model, text, &chunks)?;
    let embed_ms = embed_started.elapsed().as_millis() as u64;
    let embedding_dim = embeddings.first().map(|row| row.len()).unwrap_or(0);

    let mut store_ms = 0_u64;
    let mut query_ms = 0_u64;
    let mut persisted_to_semantic_index = false;
    let mut store_skip_reason = None::<String>;
    let mut hits = Vec::new();
    if embedding_dim == SEMANTIC_VECTOR_DIM {
        let store_started = Instant::now();
        let store = PhoenixOvergraphStore::open(&config.store_path)?;
        let scope = ScopeKey::default();
        let rows = build_chunk_node_records(
            &scope,
            &config.document_id,
            config.narrative_id.as_deref(),
            &chunks,
            &embeddings,
        )?;
        store.upsert_semantic_node_vectors_native(&rows)?;
        store_ms = store_started.elapsed().as_millis() as u64;
        persisted_to_semantic_index = true;

        if config.chunk.query_count > 0 && config.chunk.neighbor_limit > 0 {
            let query_started = Instant::now();
            hits = collect_chunk_neighbors(&store, &scope, &chunks, &embeddings, &config.chunk)?;
            query_ms = query_started.elapsed().as_millis() as u64;
        }
    } else {
        store_skip_reason = Some(format!(
            "semantic index expects {SEMANTIC_VECTOR_DIM} dims, runner emitted {embedding_dim}"
        ));
    }

    Ok(SmokeReport {
        model: "Snowflake/snowflake-arctic-embed-xs",
        profile: "384".to_owned(),
        input_path: config.input_path.display().to_string(),
        store_path: config.store_path.display().to_string(),
        ort_dylib_path: ort_path.map(|path| path.display().to_string()),
        model_root: config.embed.model_root.display().to_string(),
        embedding_dim,
        persisted_to_semantic_index,
        store_skip_reason,
        total_bytes,
        used_bytes,
        chunk_count: chunks.len(),
        query_count: config.chunk.query_count.min(chunks.len()),
        neighbor_limit: config.chunk.neighbor_limit,
        embed_ms,
        store_ms,
        query_ms,
        neighbors: enrich_hits(text, &hits),
    })
}

fn parse_args(args: Vec<String>) -> SmokeConfig {
    let mut config = SmokeConfig::default();
    if let Some(path) = string_arg(&args, "--input") {
        config.input_path = PathBuf::from(path);
    }
    if let Some(path) = string_arg(&args, "--store") {
        config.store_path = PathBuf::from(path);
    }
    if let Some(value) = string_arg(&args, "--document-id") {
        config.document_id = value;
    }
    if let Some(value) = string_arg(&args, "--narrative-id") {
        config.narrative_id = Some(value);
    }
    if let Some(value) = string_arg(&args, "--model-root") {
        config.embed.model_root = PathBuf::from(value);
    }
    if let Some(value) = usize_arg(&args, "--batch-size") {
        config.embed.batch_size = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--max-length") {
        config.embed.max_length = value.max(16);
    }
    if let Some(value) = usize_arg(&args, "--chunk-size") {
        config.chunk.chunk_size = value.max(64);
    }
    if let Some(value) = usize_arg(&args, "--overlap") {
        config.chunk.overlap = value.min(config.chunk.chunk_size.saturating_sub(1));
    }
    if let Some(value) = usize_arg(&args, "--query-count") {
        config.chunk.query_count = value;
    }
    if let Some(value) = usize_arg(&args, "--neighbor-limit") {
        config.chunk.neighbor_limit = value;
    }
    if let Some(value) = usize_arg(&args, "--oversample") {
        config.chunk.oversample = value.max(config.chunk.neighbor_limit);
    }
    config
}

fn ensure_ort_path() -> Option<PathBuf> {
    if let Some(existing) = env::var_os("ORT_DYLIB_PATH") {
        return Some(PathBuf::from(existing));
    }
    let root = workspace_root();
    let path = default_ort_dylib_path(&root)?;
    env::set_var("ORT_DYLIB_PATH", &path);
    Some(path)
}

fn enrich_hits(text: &str, hits: &[SemanticChunkNeighbor]) -> Vec<RichNeighborHit> {
    let mut rows = Vec::with_capacity(hits.len());
    for hit in hits {
        rows.push(RichNeighborHit {
            source_node_id: hit.source_node_id.clone(),
            target_node_id: hit.target_node_id.clone(),
            distance: hit.distance,
            source_chunk_index: hit.source_chunk_index,
            target_chunk_index: hit.target_chunk_index,
            source_excerpt: excerpt(text, hit.source_start, hit.source_end),
            target_excerpt: hit
                .target_start
                .zip(hit.target_end)
                .map(|(start, end)| excerpt(text, start, end)),
            evidence_refs: hit.evidence_refs.clone(),
        });
    }
    rows
}

fn excerpt(text: &str, start: usize, end: usize) -> String {
    let slice = text
        .get(start..end)
        .unwrap_or("")
        .split_whitespace()
        .take(28)
        .collect::<Vec<_>>()
        .join(" ");
    if slice.chars().count() > 180 {
        let mut value = slice.chars().take(180).collect::<String>();
        value.push_str("...");
        value
    } else {
        slice
    }
}

fn string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].clone()))
}

fn usize_arg(args: &[String], flag: &str) -> Option<usize> {
    string_arg(args, flag).and_then(|value| value.parse::<usize>().ok())
}

fn default_store_path() -> PathBuf {
    let root = if Path::new("G:\\").exists() {
        PathBuf::from("G:\\phoenix-temp")
    } else {
        workspace_root().join("target")
    };
    root.join(format!("graph-semantic-smoke-{}", std::process::id()))
}
