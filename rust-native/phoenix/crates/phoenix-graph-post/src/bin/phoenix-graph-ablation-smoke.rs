use std::env;
use std::path::PathBuf;

use phoenix_graph_post::diffusion_eval::{
    default_diffusion_cases, evaluate_causal_diffusion_cases, evaluate_history_diffusion_cases,
    evaluate_world_state_diffusion_cases, GraphDiffusionCaseResult,
};
use phoenix_graph_post::eval::{
    default_ablation_cases, evaluate_causal_cases, evaluate_history_cases,
    evaluate_world_state_cases, GraphAblationCaseResult, GraphSoftFamily,
};
use phoenix_graph_post::semantic_graph::{
    derive_semantic_graph_review_batch_from_store, persist_semantic_graph_patch_sidecar,
    SemanticGraphConfig,
};
use phoenix_graph_post::smoke_support::{
    discover_causal_target, discover_world_anchor, now_ms, string_arg, usize_arg, CausalTarget,
    WorldAnchor,
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
    world_anchor: Option<WorldAnchor>,
    causal_target: Option<CausalTarget>,
    graph: SemanticGraphConfig,
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

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AblationDelta {
    changed_selected: bool,
    abstain_changed: bool,
    candidate_edge_delta: i64,
    vertex_delta: i64,
    selected_score_delta_millis: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AblationCaseReport {
    case_name: String,
    #[serde(default)]
    families: Vec<GraphSoftFamily>,
    metrics: phoenix_graph_post::eval::GraphEvalMetrics,
    vs_hard_only: Option<AblationDelta>,
    vs_full_soft: Option<AblationDelta>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiffusionCaseReport {
    case_name: String,
    diffusion: String,
    metrics: phoenix_graph_post::eval::GraphEvalMetrics,
    vs_ppr: Option<AblationDelta>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AblationReport {
    store_path: String,
    scope_key: String,
    graph_generation: u64,
    semantic_generation: u64,
    document_ann_metric: Option<String>,
    node_ann_metric: Option<String>,
    world_anchor: Option<WorldAnchorReport>,
    causal_target: Option<CausalTargetReport>,
    #[serde(default)]
    world_state_diffusion: Vec<DiffusionCaseReport>,
    #[serde(default)]
    history_diffusion: Vec<DiffusionCaseReport>,
    #[serde(default)]
    causal_explanation_diffusion: Vec<DiffusionCaseReport>,
    #[serde(default)]
    world_state: Vec<AblationCaseReport>,
    #[serde(default)]
    history: Vec<AblationCaseReport>,
    #[serde(default)]
    causal_explanation: Vec<AblationCaseReport>,
}

fn main() {
    match run(parse_args(env::args().skip(1).collect())) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize ablation smoke")
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run(config: SmokeConfig) -> Result<AblationReport, String> {
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
    let cases = default_ablation_cases();
    let diffusion_cases = default_diffusion_cases();
    let world_anchor = config
        .world_anchor
        .clone()
        .or_else(|| discover_world_anchor(&graph_sidecar.graph_batch.vertices));
    let causal_target = config
        .causal_target
        .clone()
        .or_else(|| discover_causal_target(&graph_sidecar.graph_batch.vertices));

    let world_state = match world_anchor.as_ref() {
        Some(anchor) => evaluate_world_state_cases(
            &store,
            &scope,
            &phoenix_graph_post::api::GraphRetrievedWorldStateQueryRequest {
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
            &cases,
        )
        .map_err(|error| error.to_string())?
        .unwrap_or_default(),
        None => Vec::new(),
    };
    let world_state_diffusion = match world_anchor.as_ref() {
        Some(anchor) => evaluate_world_state_diffusion_cases(
            &store,
            &scope,
            &phoenix_graph_post::api::GraphRetrievedWorldStateQueryRequest {
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
            &diffusion_cases,
        )
        .map_err(|error| error.to_string())?
        .unwrap_or_default(),
        None => Vec::new(),
    };
    let history = match world_anchor.as_ref() {
        Some(anchor) => evaluate_history_cases(
            &store,
            &scope,
            &phoenix_graph_post::api::GraphRetrievedHistoryQueryRequest {
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
            &cases,
        )
        .map_err(|error| error.to_string())?
        .unwrap_or_default(),
        None => Vec::new(),
    };
    let history_diffusion = match world_anchor.as_ref() {
        Some(anchor) => evaluate_history_diffusion_cases(
            &store,
            &scope,
            &phoenix_graph_post::api::GraphRetrievedHistoryQueryRequest {
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
            &diffusion_cases,
        )
        .map_err(|error| error.to_string())?
        .unwrap_or_default(),
        None => Vec::new(),
    };
    let causal_explanation = match causal_target.as_ref() {
        Some(target) => evaluate_causal_cases(
            &store,
            &scope,
            &phoenix_graph_post::api::GraphRetrievedCausalExplanationQueryRequest {
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
            &cases,
        )
        .map_err(|error| error.to_string())?
        .unwrap_or_default(),
        None => Vec::new(),
    };
    let causal_explanation_diffusion = match causal_target.as_ref() {
        Some(target) => evaluate_causal_diffusion_cases(
            &store,
            &scope,
            &phoenix_graph_post::api::GraphRetrievedCausalExplanationQueryRequest {
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
            &diffusion_cases,
        )
        .map_err(|error| error.to_string())?
        .unwrap_or_default(),
        None => Vec::new(),
    };

    Ok(AblationReport {
        store_path: config.store_path.display().to_string(),
        scope_key: format!("{scope:?}"),
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
        world_state_diffusion: decorate_diffusion_cases(world_state_diffusion),
        history_diffusion: decorate_diffusion_cases(history_diffusion),
        causal_explanation_diffusion: decorate_diffusion_cases(causal_explanation_diffusion),
        world_state: decorate_cases(world_state),
        history: decorate_cases(history),
        causal_explanation: decorate_cases(causal_explanation),
    })
}

fn decorate_cases(rows: Vec<GraphAblationCaseResult>) -> Vec<AblationCaseReport> {
    let hard_only = rows
        .iter()
        .find(|row| row.case_name == "hard_only")
        .cloned();
    let full_soft = rows
        .iter()
        .find(|row| row.case_name == "full_soft")
        .cloned();
    rows.into_iter()
        .map(|row| AblationCaseReport {
            vs_hard_only: hard_only.as_ref().map(|base| delta_between(base, &row)),
            vs_full_soft: full_soft.as_ref().map(|base| delta_between(base, &row)),
            case_name: row.case_name,
            families: row.families,
            metrics: row.metrics,
        })
        .collect()
}

fn decorate_diffusion_cases(rows: Vec<GraphDiffusionCaseResult>) -> Vec<DiffusionCaseReport> {
    let ppr = rows
        .iter()
        .find(|row| row.case_name == "personalized_pagerank")
        .cloned();
    rows.into_iter()
        .map(|row| DiffusionCaseReport {
            vs_ppr: ppr
                .as_ref()
                .map(|base| delta_between_metrics(&base.metrics, &row.metrics)),
            case_name: row.case_name,
            diffusion: row.diffusion.label().to_owned(),
            metrics: row.metrics,
        })
        .collect()
}

fn delta_between(
    base: &GraphAblationCaseResult,
    current: &GraphAblationCaseResult,
) -> AblationDelta {
    delta_between_metrics(&base.metrics, &current.metrics)
}

fn delta_between_metrics(
    base: &phoenix_graph_post::eval::GraphEvalMetrics,
    current: &phoenix_graph_post::eval::GraphEvalMetrics,
) -> AblationDelta {
    AblationDelta {
        changed_selected: base.selected_id != current.selected_id,
        abstain_changed: base.abstain != current.abstain,
        candidate_edge_delta: current.region.candidate_edge_count as i64
            - base.region.candidate_edge_count as i64,
        vertex_delta: current.region.vertex_count as i64 - base.region.vertex_count as i64,
        selected_score_delta_millis: match (
            base.selected_score_millis,
            current.selected_score_millis,
        ) {
            (Some(left), Some(right)) => Some(right - left),
            _ => None,
        },
    }
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
            .map_err(|e| e.to_string())?
        {
            return Ok(sidecar);
        }
    }
    let archives = store
        .load_latest_document_archives(Some(scope))
        .map_err(|error| error.to_string())?;
    let batch = derive_scope_review_batch(
        archives.as_slice(),
        None,
        None,
        store
            .load_event_identity_patch_sidecar(scope)
            .map_err(|e| e.to_string())?
            .as_ref(),
        store
            .load_temporal_patch_sidecar(scope)
            .map_err(|e| e.to_string())?
            .as_ref(),
        store
            .load_causal_patch_sidecar(scope)
            .map_err(|e| e.to_string())?
            .as_ref(),
        store
            .load_memory_patch_sidecar(scope)
            .map_err(|e| e.to_string())?
            .as_ref(),
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
            .map_err(|e| e.to_string())?
        {
            return Ok(sidecar);
        }
    }
    store
        .init_semantic_graph_patch_schema()
        .map_err(|e| e.to_string())?;
    let Some(batch) =
        derive_semantic_graph_review_batch_from_store(store, scope, &config.graph, now_ms())
            .map_err(|error| error.to_string())?
    else {
        return Err("no semantic graph batch could be derived".to_owned());
    };
    persist_semantic_graph_patch_sidecar(store, &batch.sidecar).map_err(|e| e.to_string())?;
    Ok(batch.sidecar)
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
        world_anchor: None,
        causal_target: None,
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
    if let (Some(entity_id), Some(slot_key)) = (
        string_arg(&args, "--entity-id"),
        string_arg(&args, "--slot-key"),
    ) {
        let query_text = string_arg(&args, "--world-query-text")
            .unwrap_or_else(|| format!("current {} for {}", slot_key, entity_id));
        config.world_anchor = Some(WorldAnchor {
            entity_id,
            slot_key,
            query_text,
        });
    }
    if let Some(vertex_id) = string_arg(&args, "--causal-target-id") {
        let query_text = string_arg(&args, "--causal-query-text")
            .unwrap_or_else(|| format!("what led to {}", vertex_id));
        config.causal_target = Some(CausalTarget {
            vertex_id,
            query_text,
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
