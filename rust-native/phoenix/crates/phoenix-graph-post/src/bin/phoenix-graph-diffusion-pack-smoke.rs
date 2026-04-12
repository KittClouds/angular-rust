use std::env;
use std::path::{Path, PathBuf};

use phoenix_graph_post::diffusion_eval::{
    default_diffusion_cases, evaluate_causal_diffusion_cases, evaluate_history_diffusion_cases,
    evaluate_world_state_diffusion_cases, GraphDiffusionCase, GraphDiffusionCaseResult,
};
use phoenix_graph_post::semantic_graph::{
    derive_semantic_graph_review_batch_from_store, persist_semantic_graph_patch_sidecar,
    SemanticGraphConfig,
};
use phoenix_graph_post::smoke_support::{now_ms, string_arg, usize_arg, CausalTarget, WorldAnchor};
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
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
struct SmokeConfig {
    store_path: PathBuf,
    fixture_path: PathBuf,
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

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct DiffusionFixturePack {
    pack_name: String,
    #[serde(default)]
    cases: Vec<DiffusionFixtureCase>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiffusionFixtureCase {
    case_name: String,
    world_anchor: WorldAnchor,
    causal_target: CausalTarget,
    history_query_text: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiffusionFamilySummary {
    compared_case_count: usize,
    identical_selection_count: usize,
    winner_flip_count: usize,
    abstain_flip_count: usize,
    no_candidate_count: usize,
    ppr_higher_selected_score_count: usize,
    heat_higher_selected_score_count: usize,
    tie_selected_score_count: usize,
    ppr_higher_best_post_score_count: usize,
    heat_higher_best_post_score_count: usize,
    tie_best_post_score_count: usize,
    mean_ppr_candidate_count: f64,
    mean_heat_candidate_count: f64,
    mean_heat_minus_ppr_selected_score_millis: Option<f64>,
    mean_heat_minus_ppr_best_post_score_millis: Option<f64>,
    mean_heat_minus_ppr_structural_delta_millis: Option<f64>,
    mean_heat_minus_ppr_proximity_millis: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiffusionPackCaseReport {
    case_name: String,
    world_anchor: WorldAnchor,
    causal_target: CausalTarget,
    world_state: Vec<GraphDiffusionCaseResult>,
    history: Vec<GraphDiffusionCaseResult>,
    causal_explanation: Vec<GraphDiffusionCaseResult>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct DiffusionPackReport {
    pack_name: String,
    store_path: String,
    fixture_path: String,
    scope_key: String,
    graph_generation: u64,
    semantic_generation: u64,
    case_count: usize,
    world_state_summary: DiffusionFamilySummary,
    history_summary: DiffusionFamilySummary,
    causal_explanation_summary: DiffusionFamilySummary,
    #[serde(default)]
    cases: Vec<DiffusionPackCaseReport>,
}

fn main() {
    match run(parse_args(env::args().skip(1).collect())) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize diffusion pack report")
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run(config: SmokeConfig) -> Result<DiffusionPackReport, String> {
    let fixture = load_fixture_pack(&config.fixture_path)?;
    let store =
        PhoenixOvergraphStore::open(&config.store_path).map_err(|error| error.to_string())?;
    let scope = discover_scope(&store)?;
    let graph_sidecar = ensure_graph_sidecar(&store, &scope, config.refresh_graph)?;
    let semantic_sidecar =
        ensure_semantic_sidecar(&store, &scope, &config, config.refresh_semantic)?;
    let diffusion_cases = default_diffusion_cases();
    let mut reports = Vec::with_capacity(fixture.cases.len());
    for case in &fixture.cases {
        let world_request = phoenix_graph_post::api::GraphRetrievedWorldStateQueryRequest {
            query_text: case.world_anchor.query_text.clone(),
            entity_id: case.world_anchor.entity_id.clone(),
            slot_key: case.world_anchor.slot_key.clone(),
            valid_at: None,
            recorded_at: None,
            include_candidate_graph: true,
            seed_limit: config.seed_limit,
            oversample: config.oversample,
            expansion_hops: config.expansion_hops,
            region_node_limit: config.region_node_limit,
        };
        let history_request = phoenix_graph_post::api::GraphRetrievedHistoryQueryRequest {
            query_text: case.history_query_text.clone().unwrap_or_else(|| {
                format!(
                    "history of {} for {}",
                    case.world_anchor.slot_key, case.world_anchor.entity_id
                )
            }),
            entity_id: case.world_anchor.entity_id.clone(),
            slot_key: Some(case.world_anchor.slot_key.clone()),
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
        };
        let causal_request = phoenix_graph_post::api::GraphRetrievedCausalExplanationQueryRequest {
            query_text: case.causal_target.query_text.clone(),
            target_vertex_id: case.causal_target.vertex_id.clone(),
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
        };
        reports.push(DiffusionPackCaseReport {
            case_name: case.case_name.clone(),
            world_anchor: case.world_anchor.clone(),
            causal_target: case.causal_target.clone(),
            world_state: evaluate_world_state_diffusion_cases(
                &store,
                &scope,
                &world_request,
                &diffusion_cases,
            )
            .map_err(|error| error.to_string())?
            .unwrap_or_default(),
            history: evaluate_history_diffusion_cases(
                &store,
                &scope,
                &history_request,
                &diffusion_cases,
            )
            .map_err(|error| error.to_string())?
            .unwrap_or_default(),
            causal_explanation: evaluate_causal_diffusion_cases(
                &store,
                &scope,
                &causal_request,
                &diffusion_cases,
            )
            .map_err(|error| error.to_string())?
            .unwrap_or_default(),
        });
    }

    Ok(DiffusionPackReport {
        pack_name: fixture.pack_name,
        store_path: config.store_path.display().to_string(),
        fixture_path: config.fixture_path.display().to_string(),
        scope_key: format!("{scope:?}"),
        graph_generation: graph_sidecar.generation,
        semantic_generation: semantic_sidecar.generation,
        case_count: reports.len(),
        world_state_summary: summarize_family(&reports, |case| &case.world_state),
        history_summary: summarize_family(&reports, |case| &case.history),
        causal_explanation_summary: summarize_family(&reports, |case| &case.causal_explanation),
        cases: reports,
    })
}

fn summarize_family(
    reports: &[DiffusionPackCaseReport],
    select: impl Fn(&DiffusionPackCaseReport) -> &[GraphDiffusionCaseResult],
) -> DiffusionFamilySummary {
    let mut summary = SummaryAccumulator::default();
    for report in reports {
        let rows = select(report);
        let Some(ppr) = rows
            .iter()
            .find(|row| row.diffusion == GraphDiffusionCase::PersonalizedPagerank)
        else {
            continue;
        };
        let Some(heat) = rows
            .iter()
            .find(|row| row.diffusion == GraphDiffusionCase::HeatKernel)
        else {
            continue;
        };
        summary.compared_case_count += 1;
        if ppr.metrics.selected_id == heat.metrics.selected_id {
            summary.identical_selection_count += 1;
        } else {
            summary.winner_flip_count += 1;
        }
        if ppr.metrics.abstain != heat.metrics.abstain {
            summary.abstain_flip_count += 1;
        }
        if ppr.metrics.candidate_count == 0 && heat.metrics.candidate_count == 0 {
            summary.no_candidate_count += 1;
        }
        summary.ppr_candidate_sum += ppr.metrics.candidate_count as f64;
        summary.heat_candidate_sum += heat.metrics.candidate_count as f64;
        score_cmp(
            ppr.metrics.selected_score_millis,
            heat.metrics.selected_score_millis,
            &mut summary.ppr_higher_selected_score_count,
            &mut summary.heat_higher_selected_score_count,
            &mut summary.tie_selected_score_count,
            &mut summary.selected_score_diffs,
        );
        score_cmp(
            ppr.metrics.best_post_structural_score_millis,
            heat.metrics.best_post_structural_score_millis,
            &mut summary.ppr_higher_best_post_score_count,
            &mut summary.heat_higher_best_post_score_count,
            &mut summary.tie_best_post_score_count,
            &mut summary.best_post_score_diffs,
        );
        diff_opt(
            ppr.metrics.selected_structural_delta_millis.map(i64::from),
            heat.metrics.selected_structural_delta_millis.map(i64::from),
            &mut summary.structural_delta_diffs,
        );
        diff_opt(
            ppr.metrics
                .selected_structural_proximity_millis
                .map(i64::from),
            heat.metrics
                .selected_structural_proximity_millis
                .map(i64::from),
            &mut summary.proximity_diffs,
        );
    }
    summary.finish()
}

#[derive(Default)]
struct SummaryAccumulator {
    compared_case_count: usize,
    identical_selection_count: usize,
    winner_flip_count: usize,
    abstain_flip_count: usize,
    no_candidate_count: usize,
    ppr_higher_selected_score_count: usize,
    heat_higher_selected_score_count: usize,
    tie_selected_score_count: usize,
    ppr_higher_best_post_score_count: usize,
    heat_higher_best_post_score_count: usize,
    tie_best_post_score_count: usize,
    ppr_candidate_sum: f64,
    heat_candidate_sum: f64,
    selected_score_diffs: Vec<f64>,
    best_post_score_diffs: Vec<f64>,
    structural_delta_diffs: Vec<f64>,
    proximity_diffs: Vec<f64>,
}

impl SummaryAccumulator {
    fn finish(self) -> DiffusionFamilySummary {
        let denom = self.compared_case_count.max(1) as f64;
        DiffusionFamilySummary {
            compared_case_count: self.compared_case_count,
            identical_selection_count: self.identical_selection_count,
            winner_flip_count: self.winner_flip_count,
            abstain_flip_count: self.abstain_flip_count,
            no_candidate_count: self.no_candidate_count,
            ppr_higher_selected_score_count: self.ppr_higher_selected_score_count,
            heat_higher_selected_score_count: self.heat_higher_selected_score_count,
            tie_selected_score_count: self.tie_selected_score_count,
            ppr_higher_best_post_score_count: self.ppr_higher_best_post_score_count,
            heat_higher_best_post_score_count: self.heat_higher_best_post_score_count,
            tie_best_post_score_count: self.tie_best_post_score_count,
            mean_ppr_candidate_count: self.ppr_candidate_sum / denom,
            mean_heat_candidate_count: self.heat_candidate_sum / denom,
            mean_heat_minus_ppr_selected_score_millis: mean_or_none(&self.selected_score_diffs),
            mean_heat_minus_ppr_best_post_score_millis: mean_or_none(&self.best_post_score_diffs),
            mean_heat_minus_ppr_structural_delta_millis: mean_or_none(&self.structural_delta_diffs),
            mean_heat_minus_ppr_proximity_millis: mean_or_none(&self.proximity_diffs),
        }
    }
}

fn score_cmp(
    ppr: Option<i64>,
    heat: Option<i64>,
    ppr_higher: &mut usize,
    heat_higher: &mut usize,
    tie: &mut usize,
    diffs: &mut Vec<f64>,
) {
    if let (Some(ppr), Some(heat)) = (ppr, heat) {
        if heat > ppr {
            *heat_higher += 1;
        } else if ppr > heat {
            *ppr_higher += 1;
        } else {
            *tie += 1;
        }
        diffs.push((heat - ppr) as f64);
    }
}

fn diff_opt(ppr: Option<i64>, heat: Option<i64>, diffs: &mut Vec<f64>) {
    if let (Some(ppr), Some(heat)) = (ppr, heat) {
        diffs.push((heat - ppr) as f64);
    }
}

fn mean_or_none(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn load_fixture_pack(path: &Path) -> Result<DiffusionFixturePack, String> {
    let raw = std::fs::read_to_string(path).map_err(|error| error.to_string())?;
    serde_json::from_str(&raw).map_err(|error| error.to_string())
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
    let default_fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("fixtures")
        .join("graph-local-diffusion-shortrun.json");
    let mut config = SmokeConfig {
        store_path: PathBuf::new(),
        fixture_path: default_fixture,
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
    if let Some(path) = string_arg(&args, "--fixture-path") {
        config.fixture_path = PathBuf::from(path);
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
