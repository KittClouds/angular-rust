use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::time::Instant;

use memmap2::MmapOptions;
use phoenix_embed::{
    default_embedding_model_root, default_ort_dylib_path, workspace_root, OrtTextEmbedConfig,
    OrtTextEmbedder, TextEmbeddingProfile,
};
use phoenix_machine::{MachineSemanticChunkConfig, SurfaceCompiler};
use phoenix_types::{ChunkKind, ScopeKey, SurfaceUnitKind};
use serde::Serialize;

#[derive(Clone, Debug)]
struct SmokeConfig {
    input_path: PathBuf,
    model_root: PathBuf,
    batch_size: usize,
    max_length: usize,
    profile: TextEmbeddingProfile,
    semantic: MachineSemanticChunkConfig,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        Self {
            input_path: workspace_root().join("docs").join("shortrun.md"),
            model_root: default_embedding_model_root(),
            batch_size: 12,
            max_length: 512,
            profile: TextEmbeddingProfile::Native384,
            semantic: MachineSemanticChunkConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SemanticUnitPreview {
    sentence_index: usize,
    start: u32,
    end: u32,
    chunk_id_hint: Option<String>,
    excerpt: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    input_path: String,
    model_root: String,
    ort_dylib_path: Option<String>,
    profile: String,
    batch_size: usize,
    max_length: usize,
    text_bytes: usize,
    scan_ms: u64,
    semantic_ms: u64,
    sentence_count: usize,
    token_count: usize,
    mention_count: usize,
    phrase_count: usize,
    clause_count: usize,
    np_count: usize,
    vp_count: usize,
    pp_count: usize,
    semantic_unit_count: usize,
    semantic_units: Vec<SemanticUnitPreview>,
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
    let ort_dylib_path = ensure_ort_dylib_path();
    let file = File::open(&config.input_path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let text = std::str::from_utf8(&mmap[..])?;

    let compiler = SurfaceCompiler::default();
    let scan_started = Instant::now();
    let artifacts = compiler.analyze_document(text, &ScopeKey::default(), &[]);
    let scan_ms = scan_started.elapsed().as_millis() as u64;

    let embedder = OrtTextEmbedder::load(&OrtTextEmbedConfig {
        model_root: config.model_root.clone(),
        batch_size: config.batch_size,
        max_length: config.max_length,
        profile: config.profile,
        prefix_passage: true,
    })?;
    let semantic_started = Instant::now();
    let semantic_units = compiler.semantic_units(text, &embedder, &config.semantic)?;
    let semantic_ms = semantic_started.elapsed().as_millis() as u64;

    let mut clause_count = 0usize;
    let mut np_count = 0usize;
    let mut vp_count = 0usize;
    let mut pp_count = 0usize;
    for chunk in &artifacts.scan.chunks {
        match chunk.kind {
            Some(ChunkKind::Clause) => clause_count += 1,
            Some(ChunkKind::Np) => np_count += 1,
            Some(ChunkKind::Vp) => vp_count += 1,
            Some(ChunkKind::Pp) => pp_count += 1,
            Some(ChunkKind::AdjP) | None => {}
        }
    }

    Ok(SmokeReport {
        input_path: config.input_path.display().to_string(),
        model_root: config.model_root.display().to_string(),
        ort_dylib_path: ort_dylib_path.map(|path| path.display().to_string()),
        profile: config.profile.label().to_owned(),
        batch_size: config.batch_size,
        max_length: config.max_length,
        text_bytes: text.len(),
        scan_ms,
        semantic_ms,
        sentence_count: artifacts.scan.sentences.len(),
        token_count: artifacts.scan.tokens.len(),
        mention_count: artifacts.scan.mentions.len(),
        phrase_count: artifacts
            .surface
            .units
            .iter()
            .filter(|unit| unit.kind == SurfaceUnitKind::Phrase)
            .count(),
        clause_count,
        np_count,
        vp_count,
        pp_count,
        semantic_unit_count: semantic_units.len(),
        semantic_units: semantic_units
            .iter()
            .take(10)
            .map(|unit| SemanticUnitPreview {
                sentence_index: unit.sentence_index,
                start: unit.range.start,
                end: unit.range.end,
                chunk_id_hint: unit.chunk_id_hint.as_ref().map(ToString::to_string),
                excerpt: excerpt(text, unit.range.start as usize, unit.range.end as usize),
            })
            .collect(),
    })
}

fn parse_args(args: Vec<String>) -> SmokeConfig {
    let mut config = SmokeConfig::default();
    if let Some(value) = string_arg(&args, "--input") {
        config.input_path = PathBuf::from(value);
    }
    if let Some(value) = string_arg(&args, "--model-root") {
        config.model_root = PathBuf::from(value);
    }
    if let Some(value) = usize_arg(&args, "--batch-size") {
        config.batch_size = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--max-length") {
        config.max_length = value.max(16);
    }
    if let Some(value) = string_arg(&args, "--profile") {
        if let Some(profile) = TextEmbeddingProfile::parse(&value) {
            config.profile = profile;
        }
    }
    if let Some(value) = usize_arg(&args, "--min-sentences") {
        config.semantic.min_sentences = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--max-sentences") {
        config.semantic.max_sentences = value.max(config.semantic.min_sentences);
    }
    config
}

fn ensure_ort_dylib_path() -> Option<PathBuf> {
    if let Some(existing) = env::var_os("ORT_DYLIB_PATH") {
        return Some(PathBuf::from(existing));
    }
    let root = workspace_root();
    let path = default_ort_dylib_path(&root)?;
    env::set_var("ORT_DYLIB_PATH", &path);
    Some(path)
}

fn excerpt(text: &str, start: usize, end: usize) -> String {
    let slice = text
        .get(start..end)
        .unwrap_or_default()
        .split_whitespace()
        .take(28)
        .collect::<Vec<_>>()
        .join(" ");
    if slice.len() > 180 {
        format!("{}...", &slice[..180])
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
