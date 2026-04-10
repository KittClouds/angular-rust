use std::env;
use std::path::PathBuf;

use phoenix_api::PhoenixPipelineApi;
use phoenix_memory_post::api as memory_api;
use phoenix_store_overgraph::PhoenixOvergraphStore;
use serde::Serialize;

#[derive(Clone, Debug)]
struct Config {
    store_path: PathBuf,
    created_at: i64,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct SidecarCounts {
    states: usize,
    events: usize,
    claims: usize,
    gaps: usize,
    conflicts: usize,
    cards: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct YieldSmokeReport {
    store_path: String,
    created_at: i64,
    event_identity: phoenix_api::EventIdentityRunReport,
    temporal: phoenix_api::TemporalRunReport,
    causal: phoenix_api::CausalRunReport,
    state_schema: phoenix_api::StateSchemaRunReport,
    memory_scope_count: usize,
    memory_sidecar: SidecarCounts,
    graph: phoenix_api::GraphRunReport,
}

fn main() {
    match run(parse_args(env::args().skip(1).collect())) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize phase2.5 report")
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run(config: Config) -> Result<YieldSmokeReport, String> {
    if config.store_path.as_os_str().is_empty() {
        return Err("missing --store-path".to_owned());
    }

    let store =
        PhoenixOvergraphStore::open(&config.store_path).map_err(|error| error.to_string())?;
    let api = PhoenixPipelineApi::new(store);

    let event_identity = api
        .run_event_identity_scope(None, config.created_at)
        .map_err(|error| error.to_string())?;
    let temporal = api
        .run_temporal_scope(None, config.created_at)
        .map_err(|error| error.to_string())?;
    let causal = api
        .run_causal_scope(None, config.created_at)
        .map_err(|error| error.to_string())?;
    let state_schema = api
        .run_state_schema_scope(None, config.created_at)
        .map_err(|error| error.to_string())?;

    let memory_batches =
        memory_api::derive_batches(api.store(), None).map_err(|error| error.to_string())?;
    let mut memory_sidecar = SidecarCounts::default();
    for batch in &memory_batches {
        let sidecar = memory_api::persist_patch_sidecar(api.store(), batch, config.created_at)
            .map_err(|error| error.to_string())?;
        memory_sidecar.states += sidecar.states.len();
        memory_sidecar.events += sidecar.events.len();
        memory_sidecar.claims += sidecar.claims.len();
        memory_sidecar.gaps += sidecar.gaps.len();
        memory_sidecar.conflicts += sidecar.conflicts.len();
        memory_sidecar.cards += sidecar.entity_cards.len();
    }

    let graph = api
        .run_graph_scope(None, config.created_at)
        .map_err(|error| error.to_string())?;

    Ok(YieldSmokeReport {
        store_path: config.store_path.display().to_string(),
        created_at: config.created_at,
        event_identity,
        temporal,
        causal,
        state_schema,
        memory_scope_count: memory_batches.len(),
        memory_sidecar,
        graph,
    })
}

fn parse_args(args: Vec<String>) -> Config {
    let mut config = Config {
        store_path: PathBuf::new(),
        created_at: now_ms(),
    };
    if let Some(path) = string_arg(&args, "--store-path") {
        config.store_path = PathBuf::from(path);
    }
    if let Some(created_at) = string_arg(&args, "--created-at").and_then(|value| value.parse().ok())
    {
        config.created_at = created_at;
    }
    config
}

fn string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].clone()))
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
