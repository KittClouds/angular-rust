use std::path::PathBuf;

use phoenix_rel_post::{
    build_relation_mention_seed_sidecar_from_store, persist_relation_mention_seed_sidecar,
    RelationMentionSeeder, RelationSeedConfig,
};
use phoenix_store_native_core::PhoenixArchiveStoreV2;
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::SessionId;
use serde::Serialize;

#[derive(Debug, Clone)]
struct SeedConfig {
    store_path: PathBuf,
    model_root: PathBuf,
    session_id: Option<SessionId>,
    persist: bool,
    json: bool,
    threshold: f32,
    chunk_size: usize,
    overlap: usize,
    max_chunks_per_archive: usize,
    max_windows_per_chunk: usize,
    max_microchunks_per_archive: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SeedBatchReport {
    scope_key: String,
    archive_count: usize,
    candidate_chunk_count: usize,
    microchunk_count: usize,
    seed_count: usize,
    generation: u64,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    let store = PhoenixOvergraphStore::open(&config.store_path).map_err(|error| {
        format!(
            "failed to open store {}: {error}",
            config.store_path.display()
        )
    })?;
    let session = match config.session_id.as_ref() {
        Some(session_id) => store
            .load_latest_session_archive(session_id)
            .map_err(|error| format!("failed to load session archive: {error}"))?,
        None => None,
    };
    let seeder = RelationMentionSeeder::load(&config.model_root, config.threshold)?;
    let mut dirty = store
        .list_dirty_scopes()
        .map_err(|error| format!("failed to list dirty scopes: {error}"))?;
    dirty.sort_by(|left, right| left.scope_key.cmp(&right.scope_key));

    let seed_config = RelationSeedConfig {
        threshold: config.threshold,
        chunk_size: config.chunk_size,
        overlap: config.overlap,
        max_chunks_per_archive: config.max_chunks_per_archive,
        max_windows_per_chunk: config.max_windows_per_chunk,
        max_microchunks_per_archive: config.max_microchunks_per_archive,
    };
    let mut reports = Vec::new();
    for record in dirty {
        let (sidecar, report) = build_relation_mention_seed_sidecar_from_store(
            &store,
            &record,
            session.as_ref(),
            &seeder,
            &seed_config,
            now_ms(),
        )
        .map_err(|error| format!("failed to build relation mention seeds: {error}"))?;
        if config.persist {
            persist_relation_mention_seed_sidecar(&store, &sidecar)
                .map_err(|error| format!("failed to persist relation mention seeds: {error}"))?;
        }
        reports.push(SeedBatchReport {
            scope_key: report.scope_key,
            archive_count: report.archive_count,
            candidate_chunk_count: report.candidate_chunk_count,
            microchunk_count: report.microchunk_count,
            seed_count: report.seed_count,
            generation: report.generation,
        });
    }

    if config.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&reports)
                .map_err(|error| format!("failed to render seed report: {error}"))?
        );
    } else {
        for report in reports {
            println!("scope: {}", report.scope_key);
            println!("- archives: {}", report.archive_count);
            println!("- candidate chunks: {}", report.candidate_chunk_count);
            println!("- microchunks: {}", report.microchunk_count);
            println!("- seeds: {}", report.seed_count);
            println!("- generation: {}", report.generation);
        }
    }
    Ok(())
}

fn parse_args(args: &[String]) -> Result<SeedConfig, String> {
    let store_path = parse_path_arg(args, "--store-path")
        .ok_or_else(|| "--store-path is required".to_owned())?;
    let model_root = parse_path_arg(args, "--model-root")
        .ok_or_else(|| "--model-root is required".to_owned())?;
    Ok(SeedConfig {
        store_path,
        model_root,
        session_id: parse_string_arg(args, "--session-id").map(SessionId),
        persist: args.iter().any(|arg| arg == "--persist-seeds"),
        json: args.iter().any(|arg| arg == "--json"),
        threshold: parse_f32_arg(args, "--threshold").unwrap_or(0.55),
        chunk_size: parse_usize_arg(args, "--chunk-size").unwrap_or(320),
        overlap: parse_usize_arg(args, "--overlap").unwrap_or(64),
        max_chunks_per_archive: parse_usize_arg(args, "--max-chunks-per-archive").unwrap_or(8),
        max_windows_per_chunk: parse_usize_arg(args, "--max-windows-per-chunk").unwrap_or(4),
        max_microchunks_per_archive: parse_usize_arg(args, "--max-microchunks-per-archive")
            .unwrap_or(24),
    })
}

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].clone())
}

fn parse_path_arg(args: &[String], flag: &str) -> Option<PathBuf> {
    parse_string_arg(args, flag).map(PathBuf::from)
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    parse_string_arg(args, flag).and_then(|value| value.parse::<usize>().ok())
}

fn parse_f32_arg(args: &[String], flag: &str) -> Option<f32> {
    parse_string_arg(args, flag).and_then(|value| value.parse::<f32>().ok())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}
