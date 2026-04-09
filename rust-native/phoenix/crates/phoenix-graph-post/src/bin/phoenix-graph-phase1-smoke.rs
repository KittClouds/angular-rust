use std::env;
use std::path::PathBuf;

use phoenix_graph_post::semantic_graph::{
    derive_semantic_graph_review_batch_from_store, persist_semantic_graph_patch_sidecar,
    SemanticGraphConfig,
};
use phoenix_graph_post::SemanticNliConfig;
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixSemanticGraphPatchStore, StoreError,
};
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::ScopeKey;
use serde::Serialize;

#[derive(Clone, Debug)]
struct SmokeConfig {
    store_path: PathBuf,
    persist: bool,
    edge_limit: usize,
    graph: SemanticGraphConfig,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EdgePreview {
    edge_id: String,
    family: String,
    status: String,
    score_millis: u32,
    source_node_id: String,
    target_node_id: String,
    nli_support_millis: Option<u32>,
    nli_contradiction_millis: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    store_path: String,
    scope_key: String,
    model_id: String,
    embedding_profile: String,
    embedding_dim: usize,
    node_count: usize,
    edge_count: usize,
    edge_family_counts: std::collections::BTreeMap<String, usize>,
    persisted: bool,
    preview: Vec<EdgePreview>,
}

fn main() {
    match run(parse_args(env::args().skip(1).collect())) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize")
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run(config: SmokeConfig) -> Result<SmokeReport, String> {
    let store =
        PhoenixOvergraphStore::open(&config.store_path).map_err(|error| error.to_string())?;
    let scope = discover_scope(&store)?;
    let created_at = now_ms();
    let Some(batch) =
        derive_semantic_graph_review_batch_from_store(&store, &scope, &config.graph, created_at)
            .map_err(|error| error.to_string())?
    else {
        return Err("no semantic graph batch could be derived".to_owned());
    };
    let persisted = if config.persist {
        store
            .init_semantic_graph_patch_schema()
            .map_err(|error| error.to_string())?;
        persist_semantic_graph_patch_sidecar(&store, &batch.sidecar)
            .map_err(|error| error.to_string())?;
        true
    } else {
        false
    };
    Ok(SmokeReport {
        store_path: config.store_path.display().to_string(),
        scope_key: batch.scope_key,
        model_id: batch.sidecar.model_id,
        embedding_profile: batch.sidecar.embedding_profile,
        embedding_dim: batch.sidecar.embedding_dim,
        node_count: batch.sidecar.summary.node_count,
        edge_count: batch.sidecar.summary.edge_count,
        edge_family_counts: batch.sidecar.summary.edge_family_counts,
        persisted,
        preview: batch
            .sidecar
            .candidate_edges
            .iter()
            .filter(|edge| {
                !matches!(
                    edge.candidate_status,
                    phoenix_semantic_v2::SemanticCandidateStatus::Rejected
                )
            })
            .take(config.edge_limit)
            .map(|edge| EdgePreview {
                edge_id: edge.edge_id.clone(),
                family: format!("{:?}", edge.family),
                status: format!("{:?}", edge.candidate_status),
                score_millis: edge.score_millis,
                source_node_id: edge.source_node_id.clone(),
                target_node_id: edge.target_node_id.clone(),
                nli_support_millis: edge.nli_support_millis,
                nli_contradiction_millis: edge.nli_contradiction_millis,
            })
            .collect(),
    })
}

fn discover_scope(store: &PhoenixOvergraphStore) -> Result<ScopeKey, String> {
    let archives = store
        .load_latest_document_archives(None)
        .map_err(|error| error.to_string())?;
    archives
        .first()
        .map(|archive| archive.manifest.scope.clone())
        .ok_or_else(|| "store did not contain any document archives".to_owned())
}

fn parse_args(args: Vec<String>) -> SmokeConfig {
    let mut config = SmokeConfig {
        store_path: PathBuf::new(),
        persist: false,
        edge_limit: 16,
        graph: SemanticGraphConfig::default(),
    };
    if let Some(path) = string_arg(&args, "--store-path") {
        config.store_path = PathBuf::from(path);
    }
    if let Some(value) = usize_arg(&args, "--neighbor-limit") {
        config.graph.neighbor_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--oversample") {
        config.graph.oversample = value.max(config.graph.neighbor_limit);
    }
    if let Some(value) = usize_arg(&args, "--min-score") {
        config.graph.min_score_millis = value.min(1000) as u32;
    }
    if let Some(value) = usize_arg(&args, "--edge-limit") {
        config.edge_limit = value.max(1);
    }
    if let Some(path) = string_arg(&args, "--nli-model-root") {
        config.graph.nli = Some(SemanticNliConfig {
            model_root: PathBuf::from(path),
            support_threshold_millis: 720,
            contradiction_threshold_millis: 740,
            review_threshold_millis: 560,
        });
    }
    if args.iter().any(|arg| arg == "--persist-patches") {
        config.persist = true;
    }
    config
}

fn string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].clone()))
}

fn usize_arg(args: &[String], flag: &str) -> Option<usize> {
    string_arg(args, flag).and_then(|value| value.parse::<usize>().ok())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

#[allow(dead_code)]
fn _store_error(error: StoreError) -> String {
    error.to_string()
}
