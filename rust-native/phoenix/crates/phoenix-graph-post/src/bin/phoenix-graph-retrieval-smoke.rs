use std::env;
use std::path::PathBuf;

use phoenix_graph_kernel::KernelVertex;
use phoenix_graph_post::api::{
    retrieved_causal_explanation, retrieved_history, retrieved_world_state,
    GraphRetrievedCausalExplanationQueryRequest, GraphRetrievedHistoryQueryRequest,
    GraphRetrievedRegion, GraphRetrievedSeed, GraphRetrievedWorldStateQueryRequest,
};
use phoenix_graph_post::semantic_graph::{
    derive_semantic_graph_review_batch_from_store, persist_semantic_graph_patch_sidecar,
    SemanticGraphConfig,
};
use phoenix_graph_post::{
    derive_scope_review_batch, persist_graph_patch_sidecar, SemanticNliConfig,
};
use phoenix_store_native_core::{
    AnnIndexFamily, PhoenixArchiveStoreV2, PhoenixCausalPatchStore, PhoenixEventIdentityPatchStore,
    PhoenixGraphPatchStore, PhoenixMemoryPatchStore, PhoenixSemanticGraphPatchStore,
    PhoenixTemporalPatchStore,
};
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::ScopeKey;
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Debug)]
struct SmokeConfig {
    store_path: PathBuf,
    refresh_graph: bool,
    refresh_semantic: bool,
    seed_limit: usize,
    oversample: usize,
    expansion_hops: usize,
    region_node_limit: usize,
    history_limit: usize,
    causal_limit: usize,
    graph: SemanticGraphConfig,
}

#[derive(Clone, Debug)]
struct WorldAnchor {
    entity_id: String,
    slot_key: String,
    query_text: String,
}

#[derive(Clone, Debug)]
struct CausalTarget {
    vertex_id: String,
    query_text: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SelectionSummary {
    id: String,
    label: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RetrievalSummary {
    query_text: String,
    abstain: bool,
    abstain_reason: Option<String>,
    selected: Option<SelectionSummary>,
    seed_count: usize,
    top_seeds: Vec<GraphRetrievedSeed>,
    region: GraphRetrievedRegion,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    store_path: String,
    scope_key: String,
    graph_generation: u64,
    semantic_generation: u64,
    document_ann_metric: Option<String>,
    node_ann_metric: Option<String>,
    world_anchor: Option<WorldAnchorReport>,
    causal_target: Option<CausalTargetReport>,
    world_state: Option<RetrievalSummary>,
    world_state_error: Option<String>,
    history: Option<RetrievalSummary>,
    history_error: Option<String>,
    causal_explanation: Option<RetrievalSummary>,
    causal_explanation_error: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct WorldAnchorReport {
    entity_id: String,
    slot_key: String,
    query_text: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CausalTargetReport {
    vertex_id: String,
    query_text: String,
}

fn main() {
    match run(parse_args(env::args().skip(1).collect())) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize retrieval smoke report")
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
    let graph_sidecar = ensure_graph_sidecar(&store, &scope, config.refresh_graph)?;
    let semantic_sidecar =
        ensure_semantic_sidecar(&store, &scope, &config, config.refresh_semantic)?;
    let document_ann_metric = store
        .load_ann_manifest(&scope, AnnIndexFamily::Document, None)
        .map_err(|error| error.to_string())?
        .map(|manifest| manifest.metric);
    let node_ann_metric = store
        .load_ann_manifest(&scope, AnnIndexFamily::NodePrototype, Some("claim"))
        .map_err(|error| error.to_string())?
        .map(|manifest| manifest.metric)
        .or_else(|| {
            store
                .load_ann_manifest(&scope, AnnIndexFamily::NodePrototype, Some("state"))
                .ok()
                .flatten()
                .map(|manifest| manifest.metric)
        })
        .or_else(|| {
            store
                .load_ann_manifest(&scope, AnnIndexFamily::NodePrototype, Some("event"))
                .ok()
                .flatten()
                .map(|manifest| manifest.metric)
        });
    let vertices = graph_sidecar.graph_batch.vertices.as_slice();
    let world_anchor = discover_world_anchor(vertices);
    let causal_target = discover_causal_target(vertices);

    let (world_state, world_state_error) = match world_anchor.as_ref() {
        Some(anchor) => match retrieved_world_state(
            &store,
            &scope,
            &GraphRetrievedWorldStateQueryRequest {
                query_text: anchor.query_text.clone(),
                entity_id: anchor.entity_id.clone(),
                slot_key: anchor.slot_key.clone(),
                valid_at: None,
                recorded_at: None,
                include_candidate_graph: true,
                seed_limit: config.seed_limit,
                oversample: config.oversample,
                expansion_hops: config.expansion_hops,
                region_node_limit: config.region_node_limit,
            },
        ) {
            Ok(Some(answer)) => (
                Some(RetrievalSummary {
                    query_text: answer.query_text,
                    abstain: answer.answer.abstain,
                    abstain_reason: answer.answer.abstain_reason,
                    selected: answer.answer.selected.map(|candidate| SelectionSummary {
                        id: candidate.state.state_vertex_id,
                        label: candidate.state.value,
                    }),
                    seed_count: answer.seeds.len(),
                    top_seeds: answer.seeds.into_iter().take(6).collect(),
                    region: answer.region,
                }),
                None,
            ),
            Ok(None) => (None, Some("projection kernel was unavailable".to_owned())),
            Err(error) => (None, Some(error.to_string())),
        },
        None => (None, None),
    };

    let (history, history_error) = match world_anchor.as_ref() {
        Some(anchor) => match retrieved_history(
            &store,
            &scope,
            &GraphRetrievedHistoryQueryRequest {
                query_text: format!("history of {} for {}", anchor.slot_key, anchor.entity_id),
                entity_id: anchor.entity_id.clone(),
                slot_key: Some(anchor.slot_key.clone()),
                since_valid_at: 0,
                until_valid_at: None,
                recorded_at: None,
                include_candidate_graph: true,
                truth_plane: phoenix_graph_post::api::GraphTruthPlane::WorldState,
                limit: Some(config.history_limit),
                seed_limit: config.seed_limit,
                oversample: config.oversample,
                expansion_hops: config.expansion_hops,
                region_node_limit: config.region_node_limit.max(128),
            },
        ) {
            Ok(Some(answer)) => (
                Some(RetrievalSummary {
                    query_text: answer.query_text,
                    abstain: answer.answer.abstain,
                    abstain_reason: answer.answer.abstain_reason,
                    selected: answer.answer.selected.map(|candidate| SelectionSummary {
                        id: candidate.change.state.state_vertex_id,
                        label: format!(
                            "{:?} {}",
                            candidate.change.change_kind, candidate.change.state.value
                        ),
                    }),
                    seed_count: answer.seeds.len(),
                    top_seeds: answer.seeds.into_iter().take(6).collect(),
                    region: answer.region,
                }),
                None,
            ),
            Ok(None) => (None, Some("projection kernel was unavailable".to_owned())),
            Err(error) => (None, Some(error.to_string())),
        },
        None => (None, None),
    };

    let (causal_explanation, causal_explanation_error) = match causal_target.as_ref() {
        Some(target) => match retrieved_causal_explanation(
            &store,
            &scope,
            &GraphRetrievedCausalExplanationQueryRequest {
                query_text: target.query_text.clone(),
                target_vertex_id: target.vertex_id.clone(),
                valid_at: None,
                recorded_at: None,
                include_candidate_graph: true,
                max_depth: 3,
                limit: Some(config.causal_limit),
                truth_plane: phoenix_graph_post::api::GraphTruthPlane::WorldState,
                seed_limit: config.seed_limit,
                oversample: config.oversample,
                expansion_hops: config.expansion_hops.max(3),
                region_node_limit: config.region_node_limit.max(144),
            },
        ) {
            Ok(Some(answer)) => (
                Some(RetrievalSummary {
                    query_text: answer.query_text,
                    abstain: answer.answer.abstain,
                    abstain_reason: answer.answer.abstain_reason,
                    selected: answer.answer.selected.map(|path| SelectionSummary {
                        id: path.source_vertex_id,
                        label: format!("{} hops", path.hops.len()),
                    }),
                    seed_count: answer.seeds.len(),
                    top_seeds: answer.seeds.into_iter().take(6).collect(),
                    region: answer.region,
                }),
                None,
            ),
            Ok(None) => (None, Some("projection kernel was unavailable".to_owned())),
            Err(error) => (None, Some(error.to_string())),
        },
        None => (None, None),
    };

    Ok(SmokeReport {
        store_path: config.store_path.display().to_string(),
        scope_key: graph_sidecar.scope_key,
        graph_generation: graph_sidecar.generation,
        semantic_generation: semantic_sidecar.generation,
        document_ann_metric,
        node_ann_metric,
        world_anchor: world_anchor.map(|anchor| WorldAnchorReport {
            entity_id: anchor.entity_id,
            slot_key: anchor.slot_key,
            query_text: anchor.query_text,
        }),
        causal_target: causal_target.map(|target| CausalTargetReport {
            vertex_id: target.vertex_id,
            query_text: target.query_text,
        }),
        world_state,
        world_state_error,
        history,
        history_error,
        causal_explanation,
        causal_explanation_error,
    })
}

fn discover_scope(store: &PhoenixOvergraphStore) -> Result<ScopeKey, String> {
    store
        .load_latest_document_archives(None)
        .map_err(|error| error.to_string())?
        .into_iter()
        .next()
        .map(|archive| archive.manifest.scope)
        .ok_or_else(|| "store did not contain any document archives".to_owned())
}

fn ensure_graph_sidecar(
    store: &PhoenixOvergraphStore,
    scope: &ScopeKey,
    refresh: bool,
) -> Result<phoenix_semantic_v2::GraphScopeSidecar, String> {
    if !refresh {
        if let Some(sidecar) = store
            .load_graph_patch_sidecar(scope)
            .map_err(|error| error.to_string())?
        {
            return Ok(sidecar);
        }
    }
    let archives = store
        .load_latest_document_archives(Some(scope))
        .map_err(|error| error.to_string())?;
    let event_identity = store
        .load_event_identity_patch_sidecar(scope)
        .map_err(|error| error.to_string())?;
    let temporal = store
        .load_temporal_patch_sidecar(scope)
        .map_err(|error| error.to_string())?;
    let causal = store
        .load_causal_patch_sidecar(scope)
        .map_err(|error| error.to_string())?;
    let memory = store
        .load_memory_patch_sidecar(scope)
        .map_err(|error| error.to_string())?;
    let batch = derive_scope_review_batch(
        archives.as_slice(),
        None,
        None,
        event_identity.as_ref(),
        temporal.as_ref(),
        causal.as_ref(),
        memory.as_ref(),
    );
    persist_graph_patch_sidecar(store, &batch, now_ms()).map_err(|error| error.to_string())
}

fn ensure_semantic_sidecar(
    store: &PhoenixOvergraphStore,
    scope: &ScopeKey,
    config: &SmokeConfig,
    refresh: bool,
) -> Result<phoenix_semantic_v2::SemanticGraphScopeSidecar, String> {
    if !refresh {
        if let Some(sidecar) = store
            .load_semantic_graph_patch_sidecar(scope)
            .map_err(|error| error.to_string())?
        {
            return Ok(sidecar);
        }
    }
    store
        .init_semantic_graph_patch_schema()
        .map_err(|error| error.to_string())?;
    let Some(batch) =
        derive_semantic_graph_review_batch_from_store(store, scope, &config.graph, now_ms())
            .map_err(|error| error.to_string())?
    else {
        return Err("no semantic graph batch could be derived".to_owned());
    };
    persist_semantic_graph_patch_sidecar(store, &batch.sidecar)
        .map_err(|error| error.to_string())?;
    Ok(batch.sidecar)
}

fn discover_world_anchor(vertices: &[KernelVertex]) -> Option<WorldAnchor> {
    vertices
        .iter()
        .filter(|vertex| matches!(vertex.kind.as_str(), "state" | "claim"))
        .find_map(|vertex| {
            let entity_id = vertex.entity_id.as_ref()?;
            let slot_key = slot_key_of(vertex)?;
            Some(WorldAnchor {
                entity_id: entity_id.clone(),
                slot_key: slot_key.to_owned(),
                query_text: format!("current {} for {}", slot_key, entity_id),
            })
        })
}

fn discover_causal_target(vertices: &[KernelVertex]) -> Option<CausalTarget> {
    vertices
        .iter()
        .find(|vertex| vertex.kind == "event")
        .or_else(|| vertices.iter().find(|vertex| vertex.kind == "claim"))
        .map(|vertex| CausalTarget {
            vertex_id: vertex.id.0.clone(),
            query_text: format!("what led to {}", describe_vertex(vertex)),
        })
}

fn slot_key_of(vertex: &KernelVertex) -> Option<&str> {
    string_attr(&vertex.value, "slotKey").or_else(|| string_attr(&vertex.attributes, "slotKey"))
}

fn describe_vertex(vertex: &KernelVertex) -> String {
    let mut parts = Vec::new();
    if !vertex.labels.is_empty() {
        parts.push(vertex.labels.join(" "));
    }
    for key in [
        "kind",
        "slotKey",
        "value",
        "status",
        "oldValue",
        "newValue",
        "objectValue",
    ] {
        if let Some(value) =
            string_attr(&vertex.value, key).or_else(|| string_attr(&vertex.attributes, key))
        {
            if !value.is_empty() {
                parts.push(value.to_owned());
            }
        }
    }
    if parts.is_empty() {
        vertex.id.0.clone()
    } else {
        parts.join(" ")
    }
}

fn string_attr<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

fn parse_args(args: Vec<String>) -> SmokeConfig {
    let mut config = SmokeConfig {
        store_path: PathBuf::new(),
        refresh_graph: false,
        refresh_semantic: false,
        seed_limit: 8,
        oversample: 20,
        expansion_hops: 3,
        region_node_limit: 160,
        history_limit: 8,
        causal_limit: 6,
        graph: SemanticGraphConfig::default(),
    };
    if let Some(path) = string_arg(&args, "--store-path") {
        config.store_path = PathBuf::from(path);
    }
    if let Some(value) = usize_arg(&args, "--seed-limit") {
        config.seed_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--oversample") {
        config.oversample = value.max(config.seed_limit);
    }
    if let Some(value) = usize_arg(&args, "--expansion-hops") {
        config.expansion_hops = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--region-node-limit") {
        config.region_node_limit = value.max(32);
    }
    if let Some(value) = usize_arg(&args, "--history-limit") {
        config.history_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--causal-limit") {
        config.causal_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--neighbor-limit") {
        config.graph.neighbor_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--min-score") {
        config.graph.min_score_millis = value.min(1000) as u32;
    }
    if let Some(path) = string_arg(&args, "--nli-model-root") {
        config.graph.nli = Some(SemanticNliConfig {
            model_root: PathBuf::from(path),
            support_threshold_millis: 720,
            contradiction_threshold_millis: 740,
            review_threshold_millis: 560,
        });
    }
    if args.iter().any(|arg| arg == "--refresh-graph") {
        config.refresh_graph = true;
    }
    if args.iter().any(|arg| arg == "--refresh-semantic") {
        config.refresh_semantic = true;
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
