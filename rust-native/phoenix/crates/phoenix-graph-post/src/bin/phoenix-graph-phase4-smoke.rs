use std::env;
use std::path::PathBuf;

use phoenix_graph_post::api::{
    retrieved_causal_explanation, retrieved_history, retrieved_world_state,
    GraphRetrievedCausalExplanationQueryRequest, GraphRetrievedHistoryQueryRequest,
    GraphRetrievedWorldStateQueryRequest, GraphTruthPlane,
};
use phoenix_graph_post::semantic_graph::{
    derive_semantic_graph_review_batch_from_store, persist_semantic_graph_patch_sidecar,
    SemanticGraphConfig,
};
use phoenix_graph_post::smoke_support::{
    discover_causal_target, discover_causal_target_candidates, discover_world_anchor, now_ms,
    string_arg, usize_arg, CausalTarget, CausalTargetCandidate, WorldAnchor,
};
use phoenix_graph_post::{
    derive_scope_review_batch, persist_graph_patch_sidecar, SemanticNliConfig,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixCausalPatchStore, PhoenixEventIdentityPatchStore,
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
    causal_target_limit: usize,
    world_anchor: Option<WorldAnchor>,
    causal_target: Option<CausalTarget>,
    graph: SemanticGraphConfig,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompareSelection {
    id: Option<String>,
    score_millis: Option<i64>,
    abstain: bool,
    abstain_reason: Option<String>,
    rerank_delta_millis: Option<i64>,
    positive_score_millis: Option<i64>,
    negative_score_millis: Option<i64>,
    path_deterministic_rank: Option<usize>,
    path_deterministic_score_millis: Option<i64>,
    path_rerank_delta_millis: Option<i64>,
    event_rerank_delta_millis: Option<i64>,
    event_positive_score_millis: Option<i64>,
    event_negative_score_millis: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CompareReport {
    query_text: String,
    before: CompareSelection,
    after: CompareSelection,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    store_path: String,
    scope_key: String,
    world_anchor: Option<WorldAnchor>,
    causal_target: Option<CausalTarget>,
    #[serde(default)]
    causal_candidates: Vec<CausalTargetCandidate>,
    world_state: Option<CompareReport>,
    history: Option<CompareReport>,
    causal_explanation: Option<CompareReport>,
}

fn main() {
    match run(parse_args(env::args().skip(1).collect())) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize phase4 smoke")
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run(config: SmokeConfig) -> Result<SmokeReport, String> {
    if let Some(model_root) = string_arg(&env::args().skip(1).collect::<Vec<_>>(), "--model-root") {
        env::set_var("PHOENIX_GLICLASS_INSTRUCT_MODEL_ROOT", model_root);
    }
    let store =
        PhoenixOvergraphStore::open(&config.store_path).map_err(|error| error.to_string())?;
    let scope = discover_scope(&store)?;
    let graph_sidecar = ensure_graph_sidecar(&store, &scope, config.refresh_graph)?;
    let _semantic_sidecar =
        ensure_semantic_sidecar(&store, &scope, &config, config.refresh_semantic)?;
    let world_anchor = config
        .world_anchor
        .clone()
        .or_else(|| discover_world_anchor(&graph_sidecar.graph_batch.vertices));
    let causal_candidates = discover_causal_target_candidates(
        &graph_sidecar.graph_batch.vertices,
        &graph_sidecar.graph_batch.edges,
        config.causal_target_limit,
    );
    let causal_target = config.causal_target.clone().or_else(|| {
        causal_candidates
            .first()
            .map(|candidate| CausalTarget {
                vertex_id: candidate.vertex_id.clone(),
                query_text: candidate.query_text.clone(),
            })
            .or_else(|| discover_causal_target(&graph_sidecar.graph_batch.vertices))
    });

    let world_state = world_anchor
        .as_ref()
        .map(|anchor| {
            compare_phase4(anchor.query_text.clone(), || {
                retrieved_world_state(
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
                )
                .map_err(|error| error.to_string())
                .map(|answer| {
                    answer.map(|row| {
                        summarize(
                            row.answer
                                .selected
                                .as_ref()
                                .map(|candidate| candidate.state.state_vertex_id.clone()),
                            row.answer
                                .selected
                                .as_ref()
                                .map(|candidate| candidate.answer_score),
                            row.answer.abstain,
                            row.answer.abstain_reason,
                            row.answer
                                .selected
                                .as_ref()
                                .and_then(|candidate| candidate.query_rerank.as_ref()),
                            None,
                            None,
                        )
                    })
                })
            })
        })
        .transpose()?;

    let history = world_anchor
        .as_ref()
        .map(|anchor| {
            compare_phase4(
                format!("history of {} for {}", anchor.slot_key, anchor.entity_id),
                || {
                    retrieved_history(
                        &store,
                        &scope,
                        &GraphRetrievedHistoryQueryRequest {
                            query_text: format!(
                                "history of {} for {}",
                                anchor.slot_key, anchor.entity_id
                            ),
                            entity_id: anchor.entity_id.clone(),
                            slot_key: Some(anchor.slot_key.clone()),
                            since_valid_at: 0,
                            until_valid_at: None,
                            recorded_at: None,
                            include_candidate_graph: true,
                            truth_plane: GraphTruthPlane::WorldState,
                            limit: Some(config.history_limit),
                            seed_limit: config.seed_limit,
                            oversample: config.oversample,
                            expansion_hops: config.expansion_hops,
                            region_node_limit: config.region_node_limit.max(128),
                        },
                    )
                    .map_err(|error| error.to_string())
                    .map(|answer| {
                        answer.map(|row| {
                            summarize(
                                row.answer.selected.as_ref().map(|candidate| {
                                    candidate.change.state.state_vertex_id.clone()
                                }),
                                row.answer
                                    .selected
                                    .as_ref()
                                    .map(|candidate| candidate.answer_score),
                                row.answer.abstain,
                                row.answer.abstain_reason,
                                row.answer
                                    .selected
                                    .as_ref()
                                    .and_then(|candidate| candidate.query_rerank.as_ref()),
                                None,
                                None,
                            )
                        })
                    })
                },
            )
        })
        .transpose()?;

    let causal_explanation = causal_target
        .as_ref()
        .map(|target| {
            compare_phase4(target.query_text.clone(), || {
                retrieved_causal_explanation(
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
                        truth_plane: GraphTruthPlane::WorldState,
                        seed_limit: config.seed_limit,
                        oversample: config.oversample,
                        expansion_hops: config.expansion_hops.max(3),
                        region_node_limit: config.region_node_limit.max(144),
                    },
                )
                .map_err(|error| error.to_string())
                .map(|answer| {
                    answer.map(|row| {
                        summarize(
                            row.answer
                                .selected
                                .as_ref()
                                .map(|candidate| candidate.source_vertex_id.clone()),
                            row.answer
                                .selected
                                .as_ref()
                                .map(|candidate| candidate.answer_score),
                            row.answer.abstain,
                            row.answer.abstain_reason,
                            row.answer
                                .selected
                                .as_ref()
                                .and_then(|candidate| candidate.query_rerank.as_ref()),
                            row.answer
                                .selected
                                .as_ref()
                                .and_then(|candidate| candidate.path_rerank.as_ref()),
                            row.answer
                                .selected
                                .as_ref()
                                .and_then(|candidate| candidate.event_rerank.as_ref()),
                        )
                    })
                })
            })
        })
        .transpose()?;

    Ok(SmokeReport {
        store_path: config.store_path.display().to_string(),
        scope_key: format!("{scope:?}"),
        world_anchor,
        causal_target,
        causal_candidates,
        world_state: world_state.flatten(),
        history: history.flatten(),
        causal_explanation: causal_explanation.flatten(),
    })
}

fn compare_phase4(
    query_text: String,
    f: impl Fn() -> Result<Option<CompareSelection>, String>,
) -> Result<Option<CompareReport>, String> {
    let previous = env::var("PHOENIX_GRAPH_PHASE4_DISABLED").ok();
    env::set_var("PHOENIX_GRAPH_PHASE4_DISABLED", "1");
    let before = f()?;
    restore_env(previous.as_deref());
    let after = f()?;
    match (before, after) {
        (Some(before), Some(after)) => Ok(Some(CompareReport {
            query_text,
            before,
            after,
        })),
        _ => Ok(None),
    }
}

fn summarize(
    id: Option<String>,
    score: Option<f64>,
    abstain: bool,
    abstain_reason: Option<String>,
    rerank: Option<&phoenix_graph_post::GraphPhase4RerankScore>,
    path_rerank: Option<&phoenix_graph_post::GraphPathRerankScore>,
    event_rerank: Option<&phoenix_graph_post::GraphPhase4RerankScore>,
) -> CompareSelection {
    CompareSelection {
        id,
        score_millis: score.map(to_millis),
        abstain,
        abstain_reason,
        rerank_delta_millis: rerank.map(|row| to_millis(row.applied_delta)),
        positive_score_millis: rerank.map(|row| to_millis(row.positive_score)),
        negative_score_millis: rerank.map(|row| to_millis(row.negative_score)),
        path_deterministic_rank: path_rerank.map(|row| row.deterministic_rank),
        path_deterministic_score_millis: path_rerank.map(|row| to_millis(row.deterministic_score)),
        path_rerank_delta_millis: path_rerank.map(|row| to_millis(row.applied_delta)),
        event_rerank_delta_millis: event_rerank.map(|row| to_millis(row.applied_delta)),
        event_positive_score_millis: event_rerank.map(|row| to_millis(row.positive_score)),
        event_negative_score_millis: event_rerank.map(|row| to_millis(row.negative_score)),
    }
}

fn to_millis(value: f64) -> i64 {
    (value * 1000.0).round() as i64
}

fn restore_env(previous: Option<&str>) {
    match previous {
        Some(value) => env::set_var("PHOENIX_GRAPH_PHASE4_DISABLED", value),
        None => env::remove_var("PHOENIX_GRAPH_PHASE4_DISABLED"),
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
        causal_target_limit: 8,
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
    if let Some(value) = usize_arg(&args, "--causal-target-limit") {
        config.causal_target_limit = value.max(1);
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
