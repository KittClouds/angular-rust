use std::collections::BTreeMap;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use crate::PhoenixPipelineApi;
use phoenix_causal_post::{
    normalize_causal_inputs_with_sidecars, CausalEventProfile, CausalReviewCase,
    CausalSourceClaimTraceSummary,
};
use phoenix_graph_kernel::{
    causal_path_candidate_views_from_snapshot, KernelEdge, KernelGraphSnapshot, KernelVertex,
    KernelViewRequest, PhoenixGraphKernel,
};
use phoenix_graph_post::api::{
    open_scope_query_session, rank_causal_explanation_answer, retrieved_causal_explanation,
    retrieved_causal_explanation_with_session, retrieved_history, retrieved_history_with_session,
    retrieved_world_state, retrieved_world_state_with_session, GraphCausalExplanationQueryRequest,
    GraphRetrievedCausalExplanationAnswer, GraphRetrievedCausalExplanationQueryRequest,
    GraphRetrievedHistoryQueryRequest, GraphRetrievedWorldStateQueryRequest, GraphTruthPlane,
    ScopeQuerySession,
};
use phoenix_graph_post::semantic_graph::{
    derive_semantic_graph_review_batch_from_store, persist_semantic_graph_patch_sidecar,
    SemanticGraphConfig,
};
use phoenix_graph_post::smoke_support::{
    describe_vertex, discover_causal_target_candidates, now_ms, string_attr, vertex_entity_id,
    vertex_slot_key, CausalTargetCandidate, WorldAnchor,
};
use phoenix_graph_post::{
    clear_graph_thread_local_caches, reset_graph_runtime_telemetry,
    snapshot_graph_runtime_telemetry, GraphRuntimeTelemetrySnapshot,
};
use phoenix_graph_post::{derive_scope_review_batch, persist_graph_patch_sidecar};
use phoenix_semantic_v2::{
    CausalCompilerSummary, DocumentArchive, EventIdentityCompilerSummary, GraphCompilerSummary,
    MemoryCompilerSummary, SemanticGraphCompilerSummary, StateSchemaCompilerSummary,
    TemporalCompilerSummary, TemporalScopeSidecar,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixCausalPatchStore, PhoenixErPatchStore,
    PhoenixEventIdentityPatchStore, PhoenixGraphPatchStore, PhoenixMemoryPatchStore,
    PhoenixSemanticGraphPatchStore, PhoenixStateSchemaPatchStore, PhoenixTemporalPatchStore,
};
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::{ScopeKey, SemanticNodeRef};
use serde::Serialize;

#[derive(Clone, Debug)]
pub struct DepthAuditConfig {
    pub store_path: PathBuf,
    pub refresh_pipeline: bool,
    pub refresh_graph: bool,
    pub refresh_semantic: bool,
    pub probe_limit: usize,
    pub seed_limit: usize,
    pub oversample: usize,
    pub expansion_hops: usize,
    pub region_node_limit: usize,
    pub history_limit: usize,
    pub causal_limit: usize,
    pub graph: SemanticGraphConfig,
}

impl Default for DepthAuditConfig {
    fn default() -> Self {
        Self {
            store_path: PathBuf::new(),
            refresh_pipeline: false,
            refresh_graph: false,
            refresh_semantic: false,
            probe_limit: 4,
            seed_limit: 8,
            oversample: 20,
            expansion_hops: 3,
            region_node_limit: 160,
            history_limit: 8,
            causal_limit: 6,
            graph: SemanticGraphConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArchiveSummary {
    pub document_count: usize,
    pub text_len: usize,
    pub token_count: usize,
    pub sentence_count: usize,
    pub mention_count: usize,
    pub chunk_count: usize,
    pub entity_count: usize,
    pub relation_count: usize,
    pub alias_count: usize,
    pub discovery_count: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PersistedStage<T> {
    pub generation: u64,
    pub summary: T,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeSidecarSummary {
    pub event_identity: Option<PersistedStage<EventIdentityCompilerSummary>>,
    pub temporal: Option<PersistedStage<TemporalCompilerSummary>>,
    pub causal: Option<PersistedStage<CausalCompilerSummary>>,
    pub memory: Option<PersistedStage<MemoryCompilerSummary>>,
    pub state_schema: Option<PersistedStage<StateSchemaCompilerSummary>>,
    pub state_schema_diagnostics: Option<BTreeMap<String, usize>>,
    pub graph: Option<PersistedStage<GraphCompilerSummary>>,
    pub semantic_graph: Option<PersistedStage<SemanticGraphCompilerSummary>>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeLossSummary {
    pub world_anchor_count: usize,
    pub path_bearing_causal_target_count: usize,
    pub graph_state_per_memory_state_millis: Option<u32>,
    pub graph_event_per_memory_event_millis: Option<u32>,
    pub graph_event_per_canonical_event_millis: Option<u32>,
    pub temporal_interval_per_canonical_event_millis: Option<u32>,
    pub causal_edge_per_graph_event_millis: Option<u32>,
    pub semantic_candidate_edge_per_graph_edge_millis: Option<u32>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeCase {
    pub probe_id: String,
    pub query_text: String,
    pub selected_id: Option<String>,
    pub selected_label: Option<String>,
    pub candidate_count: usize,
    pub seed_count: usize,
    pub region_vertex_count: usize,
    pub asserted_edge_count: usize,
    pub candidate_edge_count: usize,
    pub truncated: bool,
    pub abstain: bool,
    pub abstain_reason: Option<String>,
    pub error: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProbeFamilySummary {
    pub total: usize,
    pub answered: usize,
    pub abstained: usize,
    pub errors: usize,
    pub candidate_count_total: usize,
    pub region_vertex_count_total: usize,
    pub reason_counts: BTreeMap<String, usize>,
    pub samples: Vec<ProbeCase>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeProbeSummary {
    pub world_state: ProbeFamilySummary,
    pub history: ProbeFamilySummary,
    pub causal_explanation: ProbeFamilySummary,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalPathLedgerSummary {
    pub target_count: usize,
    pub raw_path_target_count: usize,
    pub visible_path_target_count: usize,
    pub ranked_path_target_count: usize,
    pub plane_allowed_target_count: usize,
    pub query_answered_count: usize,
    pub query_abstained_count: usize,
    pub raw_path_candidate_total: usize,
    pub visible_path_candidate_total: usize,
    pub ranked_candidate_total: usize,
    pub raw_missing_endpoint_edge_total: usize,
    pub visible_missing_endpoint_edge_total: usize,
    pub raw_incoming_missing_source_total: usize,
    pub visible_incoming_missing_source_total: usize,
    pub endpoint_shape_counts: BTreeMap<String, usize>,
    pub source_resolution_tier_counts: BTreeMap<String, usize>,
    pub target_resolution_tier_counts: BTreeMap<String, usize>,
    pub fallback_source_edge_total: usize,
    pub fallback_target_edge_total: usize,
    pub promoted_source_edge_total: usize,
    pub promoted_target_edge_total: usize,
    pub reason_counts: BTreeMap<String, usize>,
    pub samples: Vec<CausalTargetLedger>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalTargetLedger {
    pub target_vertex_id: String,
    pub target_kind: Option<String>,
    pub target_label: String,
    pub target_visible: bool,
    pub raw_incoming_causal_edges: usize,
    pub raw_outgoing_causal_edges: usize,
    pub visible_incoming_causal_edges: usize,
    pub visible_outgoing_causal_edges: usize,
    pub raw_missing_endpoint_edges: usize,
    pub visible_missing_endpoint_edges: usize,
    pub raw_incoming_missing_source_edges: usize,
    pub visible_incoming_missing_source_edges: usize,
    pub endpoint_shape_counts: BTreeMap<String, usize>,
    pub source_resolution_tier_counts: BTreeMap<String, usize>,
    pub target_resolution_tier_counts: BTreeMap<String, usize>,
    pub fallback_source_edges: usize,
    pub fallback_target_edges: usize,
    pub promoted_source_edges: usize,
    pub promoted_target_edges: usize,
    pub raw_path_candidate_count: usize,
    pub visible_path_candidate_count: usize,
    pub ranked_candidate_count: usize,
    pub plane_allowed_candidate_count: usize,
    pub query_candidate_count: usize,
    pub query_abstain: bool,
    pub query_abstain_reason: Option<String>,
    pub best_ranked_score_millis: Option<i64>,
    pub best_ranked_depth: Option<usize>,
    pub raw_incoming_samples: Vec<CausalEdgeSample>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalEdgeSample {
    pub source_id: String,
    pub source_present: bool,
    pub source_semantic_kind: Option<String>,
    pub source_semantic_id: Option<String>,
    pub source_semantic_label: Option<String>,
    pub source_resolution_tier: Option<String>,
    pub source_endpoint_fallback: bool,
    pub source_endpoint_promoted: bool,
    pub source_kind: Option<String>,
    pub source_label: String,
    pub target_id: String,
    pub target_present: bool,
    pub target_semantic_kind: Option<String>,
    pub target_semantic_id: Option<String>,
    pub target_semantic_label: Option<String>,
    pub target_resolution_tier: Option<String>,
    pub target_endpoint_fallback: bool,
    pub target_endpoint_promoted: bool,
    pub target_kind: Option<String>,
    pub target_label: String,
    pub edge_type: String,
    pub layer: String,
    pub relation_kind: Option<String>,
    pub status: Option<String>,
    pub confidence_millis: Option<i64>,
    pub valid_from: Option<i64>,
    pub valid_to: Option<i64>,
    pub recorded_at: Option<i64>,
    pub evidence_ref_count: usize,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AuditTotals {
    pub archives: ArchiveSummary,
    pub missing_sidecar_counts: BTreeMap<String, usize>,
    pub probes: ScopeProbeSummary,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TimingSummary {
    pub count: u64,
    pub total_us: u64,
    pub mean_us: f64,
    pub max_us: u64,
}

impl TimingSummary {
    fn record(&mut self, elapsed_us: u64) {
        self.count += 1;
        self.total_us += elapsed_us;
        self.max_us = self.max_us.max(elapsed_us);
        self.mean_us = self.total_us as f64 / self.count as f64;
    }
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DepthAuditInstrumentation {
    pub refresh_pipeline: TimingSummary,
    pub refresh_event_identity: TimingSummary,
    pub refresh_temporal: TimingSummary,
    pub refresh_causal: TimingSummary,
    pub refresh_late_sidecars: TimingSummary,
    pub load_scope_archives: TimingSummary,
    pub audit_scope: TimingSummary,
    pub release_graph_thread_locals: TimingSummary,
    pub release_retained_runtime: TimingSummary,
    pub store_close: TimingSummary,
    pub ensure_graph_sidecar: TimingSummary,
    pub ensure_semantic_sidecar: TimingSummary,
    pub scope_probes: TimingSummary,
    pub world_probe_query: TimingSummary,
    pub history_probe_query: TimingSummary,
    pub causal_probe_query: TimingSummary,
    pub summarize_causal_ledger: TimingSummary,
    pub raw_causal_snapshot: TimingSummary,
    pub visible_causal_snapshot: TimingSummary,
    pub causal_ledger_query: TimingSummary,
    pub graph_runtime: GraphRuntimeTelemetrySnapshot,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeDepthAudit {
    pub scope_key: String,
    pub archive: ArchiveSummary,
    pub sidecars: ScopeSidecarSummary,
    pub losses: ScopeLossSummary,
    pub probes: ScopeProbeSummary,
    pub causal_ledger: CausalPathLedgerSummary,
    pub source_claim_trace: CausalSourceClaimTraceSummary,
    pub temporal_source_claim_support: TemporalSourceClaimSupportSummary,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalSourceClaimSupportSummary {
    pub temporal_sidecar_present: bool,
    pub total_source_claim_case_count: usize,
    pub unique_source_claim_count: usize,
    pub with_any_support_count: usize,
    pub without_any_support_count: usize,
    pub with_anchor_count: usize,
    pub with_interval_count: usize,
    pub with_gap_count: usize,
    pub with_memory_card_count: usize,
    pub with_claim_atom_count: usize,
    pub with_reference_edge_count: usize,
    pub with_constraint_count: usize,
    pub anchor_source_counts: BTreeMap<String, usize>,
    pub gap_reason_counts: BTreeMap<String, usize>,
    pub samples: Vec<TemporalSourceClaimSupportSample>,
}

impl Default for TemporalSourceClaimSupportSummary {
    fn default() -> Self {
        Self {
            temporal_sidecar_present: false,
            total_source_claim_case_count: 0,
            unique_source_claim_count: 0,
            with_any_support_count: 0,
            without_any_support_count: 0,
            with_anchor_count: 0,
            with_interval_count: 0,
            with_gap_count: 0,
            with_memory_card_count: 0,
            with_claim_atom_count: 0,
            with_reference_edge_count: 0,
            with_constraint_count: 0,
            anchor_source_counts: BTreeMap::new(),
            gap_reason_counts: BTreeMap::new(),
            samples: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalSourceClaimSupportSample {
    pub document_id: String,
    pub claim_id: String,
    pub claim_label: String,
    pub proposition_id: String,
    pub proposition_predicate: String,
    pub anchor_count: usize,
    pub interval_count: usize,
    pub gap_count: usize,
    pub memory_card_count: usize,
    pub claim_atom_count: usize,
    pub reference_edge_count: usize,
    pub constraint_count: usize,
    pub has_any_support: bool,
    pub anchor_sources: Vec<String>,
    pub gap_reasons: Vec<String>,
    pub strongest_valid_from: Option<i64>,
    pub strongest_recorded_from: Option<i64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DepthAuditReport {
    pub store_path: String,
    pub refreshed_pipeline: bool,
    pub refreshed_graph: bool,
    pub refreshed_semantic: bool,
    pub scope_count: usize,
    pub totals: AuditTotals,
    pub instrumentation: DepthAuditInstrumentation,
    pub scopes: Vec<ScopeDepthAudit>,
}

type CachedCausalQueryResult = Result<Option<GraphRetrievedCausalExplanationAnswer>, String>;

pub fn run_depth_audit(config: DepthAuditConfig) -> Result<DepthAuditReport, String> {
    if config.store_path.as_os_str().is_empty() {
        return Err("missing --store-path".to_owned());
    }
    reset_graph_runtime_telemetry();
    let mut instrumentation = DepthAuditInstrumentation::default();
    let mut store =
        PhoenixOvergraphStore::open(&config.store_path).map_err(|error| error.to_string())?;
    if config.refresh_pipeline {
        let started = Instant::now();
        let api = PhoenixPipelineApi::new(store);
        refresh_pipeline_sidecars(&api, &mut instrumentation)?;
        store = api.into_store();
        instrumentation.refresh_pipeline.record(elapsed_us(started));
    }
    let load_scope_archives_started = Instant::now();
    let scope_archives = load_scope_archives(&store)?;
    instrumentation
        .load_scope_archives
        .record(elapsed_us(load_scope_archives_started));
    let mut scopes = Vec::with_capacity(scope_archives.len());
    let mut totals = AuditTotals::default();
    for (scope_key, scope, archives) in scope_archives {
        let audit_scope_started = Instant::now();
        let audit = audit_scope(
            &store,
            &config,
            scope_key,
            scope,
            archives,
            &mut instrumentation,
        )?;
        instrumentation
            .audit_scope
            .record(elapsed_us(audit_scope_started));
        merge_archive(&mut totals.archives, &audit.archive);
        accumulate_missing_sidecars(&mut totals.missing_sidecar_counts, &audit.sidecars);
        accumulate_probe_summary(&mut totals.probes, &audit.probes);
        scopes.push(audit);
    }
    instrumentation.graph_runtime = snapshot_graph_runtime_telemetry();
    let release_graph_thread_locals_started = Instant::now();
    clear_graph_thread_local_caches();
    instrumentation
        .release_graph_thread_locals
        .record(elapsed_us(release_graph_thread_locals_started));
    let release_retained_runtime_started = Instant::now();
    store.clear_retained_runtime_state();
    instrumentation
        .release_retained_runtime
        .record(elapsed_us(release_retained_runtime_started));
    let store_close_started = Instant::now();
    drop(store);
    instrumentation
        .store_close
        .record(elapsed_us(store_close_started));
    Ok(DepthAuditReport {
        store_path: config.store_path.display().to_string(),
        refreshed_pipeline: config.refresh_pipeline,
        refreshed_graph: config.refresh_graph,
        refreshed_semantic: config.refresh_semantic,
        scope_count: scopes.len(),
        totals,
        instrumentation,
        scopes,
    })
}

fn audit_scope(
    store: &PhoenixOvergraphStore,
    config: &DepthAuditConfig,
    scope_key: String,
    scope: ScopeKey,
    archives: Vec<DocumentArchive>,
    instrumentation: &mut DepthAuditInstrumentation,
) -> Result<ScopeDepthAudit, String> {
    let archive = summarize_archives(&archives);
    let mut graph_sidecar = if config.refresh_graph {
        let started = Instant::now();
        let sidecar = ensure_graph_sidecar(store, &scope)?;
        instrumentation
            .ensure_graph_sidecar
            .record(elapsed_us(started));
        Some(sidecar)
    } else {
        store
            .load_graph_patch_sidecar(&scope)
            .map_err(|error| error.to_string())?
    };
    if graph_sidecar.is_none() && config.refresh_semantic {
        let started = Instant::now();
        graph_sidecar = Some(ensure_graph_sidecar(store, &scope)?);
        instrumentation
            .ensure_graph_sidecar
            .record(elapsed_us(started));
    }
    let semantic_sidecar = if config.refresh_semantic {
        let started = Instant::now();
        let sidecar = ensure_semantic_sidecar(store, &scope, config)?;
        instrumentation
            .ensure_semantic_sidecar
            .record(elapsed_us(started));
        Some(sidecar)
    } else {
        store
            .load_semantic_graph_patch_sidecar(&scope)
            .map_err(|error| error.to_string())?
    };
    let event_identity = store
        .load_event_identity_patch_sidecar(&scope)
        .map_err(|error| error.to_string())?;
    let temporal = store
        .load_temporal_patch_sidecar(&scope)
        .map_err(|error| error.to_string())?;
    let causal = store
        .load_causal_patch_sidecar(&scope)
        .map_err(|error| error.to_string())?;
    let memory = store
        .load_memory_patch_sidecar(&scope)
        .map_err(|error| error.to_string())?;
    let er = store
        .load_er_patch_sidecar(&scope)
        .map_err(|error| error.to_string())?;
    let state_schema = store
        .load_state_schema_patch_sidecar(&scope)
        .map_err(|error| error.to_string())?;
    let sidecars = ScopeSidecarSummary {
        event_identity: event_identity.as_ref().map(|sidecar| PersistedStage {
            generation: sidecar.generation,
            summary: sidecar.summary.clone(),
        }),
        temporal: temporal.as_ref().map(|sidecar| PersistedStage {
            generation: sidecar.generation,
            summary: sidecar.summary.clone(),
        }),
        causal: causal.as_ref().map(|sidecar| PersistedStage {
            generation: sidecar.generation,
            summary: sidecar.summary.clone(),
        }),
        memory: memory.as_ref().map(|sidecar| PersistedStage {
            generation: sidecar.generation,
            summary: sidecar.summary.clone(),
        }),
        state_schema: state_schema.as_ref().map(|sidecar| PersistedStage {
            generation: sidecar.generation,
            summary: sidecar.summary.clone(),
        }),
        state_schema_diagnostics: state_schema
            .as_ref()
            .map(|sidecar| sidecar.diagnostics.clone()),
        graph: graph_sidecar.as_ref().map(|sidecar| PersistedStage {
            generation: sidecar.generation,
            summary: sidecar.summary.clone(),
        }),
        semantic_graph: semantic_sidecar.as_ref().map(|sidecar| PersistedStage {
            generation: sidecar.generation,
            summary: sidecar.summary.clone(),
        }),
    };
    let query_session = if graph_sidecar.is_some() {
        open_scope_query_session(store, &scope).map_err(|error| error.to_string())?
    } else {
        None
    };
    let causal_targets = graph_sidecar
        .as_ref()
        .map(|sidecar| {
            discover_causal_target_candidates(
                &sidecar.graph_batch.vertices,
                &sidecar.graph_batch.edges,
                config.probe_limit.max(1),
            )
        })
        .unwrap_or_default();
    let mut causal_query_cache = BTreeMap::<String, CachedCausalQueryResult>::new();
    let (losses, probes) = match graph_sidecar.as_ref() {
        Some(sidecar) => {
            let losses = summarize_losses(
                &sidecars,
                &sidecar.graph_batch.vertices,
                &sidecar.graph_batch.edges,
            );
            let started = Instant::now();
            let probes = run_scope_probes(
                store,
                &scope,
                query_session.as_ref(),
                causal_targets.as_slice(),
                &mut causal_query_cache,
                config,
                sidecar,
                instrumentation,
            )?;
            instrumentation.scope_probes.record(elapsed_us(started));
            (losses, probes)
        }
        None => (ScopeLossSummary::default(), ScopeProbeSummary::default()),
    };
    let causal_ledger = match graph_sidecar.as_ref() {
        Some(sidecar) => {
            let started = Instant::now();
            let summary = summarize_causal_ledger(
                store,
                &scope,
                query_session.as_ref(),
                causal_targets.as_slice(),
                &causal_query_cache,
                config,
                sidecar,
                semantic_sidecar.as_ref(),
                instrumentation,
            )?;
            instrumentation
                .summarize_causal_ledger
                .record(elapsed_us(started));
            summary
        }
        None => CausalPathLedgerSummary::default(),
    };
    let normalized =
        normalize_causal_inputs_with_sidecars(archives.as_slice(), er.as_ref(), temporal.as_ref());
    let source_claim_trace = normalized.source_claim_trace.clone();
    let temporal_source_claim_support = summarize_temporal_source_claim_support(
        &normalized.review_cases,
        &normalized.event_profiles,
        temporal.as_ref(),
    );
    Ok(ScopeDepthAudit {
        scope_key,
        archive,
        sidecars,
        losses,
        probes,
        causal_ledger,
        source_claim_trace,
        temporal_source_claim_support,
    })
}

fn summarize_temporal_source_claim_support(
    review_cases: &[CausalReviewCase],
    event_profiles: &[CausalEventProfile],
    temporal_sidecar: Option<&TemporalScopeSidecar>,
) -> TemporalSourceClaimSupportSummary {
    let mut summary = TemporalSourceClaimSupportSummary {
        temporal_sidecar_present: temporal_sidecar.is_some(),
        ..TemporalSourceClaimSupportSummary::default()
    };
    let mut unique_claims = HashSet::<String>::new();
    let profile_by_key = event_profiles
        .iter()
        .filter_map(|profile| match &profile.node {
            SemanticNodeRef::Claim(claim_id) => Some((
                temporal_claim_key(profile.document_id.as_str(), claim_id.0.as_str()),
                profile,
            )),
            _ => None,
        })
        .collect::<HashMap<_, _>>();

    let mut anchors_by_key =
        HashMap::<String, Vec<&phoenix_semantic_v2::TemporalAnchorRecord>>::new();
    let mut intervals_by_key =
        HashMap::<String, Vec<&phoenix_semantic_v2::TemporalIntervalRecord>>::new();
    let mut gaps_by_key = HashMap::<String, Vec<&phoenix_semantic_v2::TemporalGapRecord>>::new();
    let mut memory_cards_by_key =
        HashMap::<String, Vec<&phoenix_semantic_v2::TemporalMemoryCard>>::new();
    let mut claim_atoms_by_key =
        HashMap::<String, Vec<&phoenix_semantic_v2::TemporalClaimAtom>>::new();
    let mut reference_edges_by_key =
        HashMap::<String, Vec<&phoenix_semantic_v2::TemporalReferenceEdge>>::new();
    let mut constraints_by_key =
        HashMap::<String, Vec<&phoenix_semantic_v2::TemporalConstraintRecord>>::new();

    if let Some(sidecar) = temporal_sidecar {
        for anchor in &sidecar.anchors {
            let Some(event_id) = anchor.event_id.as_deref() else {
                continue;
            };
            anchors_by_key
                .entry(temporal_claim_key(anchor.document_id.as_str(), event_id))
                .or_default()
                .push(anchor);
        }
        for interval in &sidecar.intervals {
            intervals_by_key
                .entry(temporal_claim_key(
                    interval.document_id.as_str(),
                    interval.event_id.as_str(),
                ))
                .or_default()
                .push(interval);
        }
        for gap in &sidecar.gaps {
            let Some(event_id) = gap.event_id.as_deref() else {
                continue;
            };
            gaps_by_key
                .entry(temporal_claim_key(gap.document_id.as_str(), event_id))
                .or_default()
                .push(gap);
        }
        for card in &sidecar.memory_cards {
            memory_cards_by_key
                .entry(temporal_claim_key(
                    card.document_id.as_str(),
                    card.event_id.as_str(),
                ))
                .or_default()
                .push(card);
        }
        for atom in &sidecar.claim_atoms {
            let Some(event_id) = atom.event_id.as_deref() else {
                continue;
            };
            claim_atoms_by_key
                .entry(temporal_claim_key(atom.document_id.as_str(), event_id))
                .or_default()
                .push(atom);
        }
        for edge in &sidecar.reference_edges {
            reference_edges_by_key
                .entry(temporal_claim_key(
                    edge.document_id.as_str(),
                    edge.source_event_id.as_str(),
                ))
                .or_default()
                .push(edge);
            if let Some(target_event_id) = edge.target_event_id.as_deref() {
                reference_edges_by_key
                    .entry(temporal_claim_key(
                        edge.document_id.as_str(),
                        target_event_id,
                    ))
                    .or_default()
                    .push(edge);
            }
        }
        for constraint in &sidecar.constraints {
            if let Some(source_event_id) = constraint.source_event_id.as_deref() {
                constraints_by_key
                    .entry(temporal_claim_key(
                        constraint.document_id.as_str(),
                        source_event_id,
                    ))
                    .or_default()
                    .push(constraint);
            }
            if let Some(target_event_id) = constraint.target_event_id.as_deref() {
                constraints_by_key
                    .entry(temporal_claim_key(
                        constraint.document_id.as_str(),
                        target_event_id,
                    ))
                    .or_default()
                    .push(constraint);
            }
        }
    }

    for case in review_cases {
        let SemanticNodeRef::Claim(claim_id) = &case.source else {
            continue;
        };
        let key = temporal_claim_key(case.document_id.as_str(), claim_id.0.as_str());
        summary.total_source_claim_case_count += 1;
        if unique_claims.insert(key.clone()) {
            summary.unique_source_claim_count += 1;
        }

        let anchor_rows = anchors_by_key.get(&key).cloned().unwrap_or_default();
        let interval_rows = intervals_by_key.get(&key).cloned().unwrap_or_default();
        let gap_rows = gaps_by_key.get(&key).cloned().unwrap_or_default();
        let memory_rows = memory_cards_by_key.get(&key).cloned().unwrap_or_default();
        let claim_atom_rows = claim_atoms_by_key.get(&key).cloned().unwrap_or_default();
        let reference_rows = reference_edges_by_key
            .get(&key)
            .cloned()
            .unwrap_or_default();
        let constraint_rows = constraints_by_key.get(&key).cloned().unwrap_or_default();
        let has_any_support = !anchor_rows.is_empty()
            || !interval_rows.is_empty()
            || !gap_rows.is_empty()
            || !memory_rows.is_empty()
            || !claim_atom_rows.is_empty()
            || !reference_rows.is_empty()
            || !constraint_rows.is_empty();
        summary.with_any_support_count += has_any_support as usize;
        summary.without_any_support_count += (!has_any_support) as usize;
        summary.with_anchor_count += (!anchor_rows.is_empty()) as usize;
        summary.with_interval_count += (!interval_rows.is_empty()) as usize;
        summary.with_gap_count += (!gap_rows.is_empty()) as usize;
        summary.with_memory_card_count += (!memory_rows.is_empty()) as usize;
        summary.with_claim_atom_count += (!claim_atom_rows.is_empty()) as usize;
        summary.with_reference_edge_count += (!reference_rows.is_empty()) as usize;
        summary.with_constraint_count += (!constraint_rows.is_empty()) as usize;

        let anchor_sources = unique_sorted_strings(
            &anchor_rows
                .iter()
                .map(|row| row.source_class.as_str())
                .collect::<Vec<_>>(),
        );
        let gap_reasons = unique_sorted_strings(
            &gap_rows
                .iter()
                .map(|row| row.reason.as_str())
                .collect::<Vec<_>>(),
        );
        for source_class in &anchor_sources {
            *summary
                .anchor_source_counts
                .entry(source_class.clone())
                .or_default() += 1;
        }
        for reason in &gap_reasons {
            *summary.gap_reason_counts.entry(reason.clone()).or_default() += 1;
        }

        if summary.samples.len() >= 16 {
            continue;
        }
        let strongest = interval_rows
            .first()
            .map(|row| &row.temporal)
            .or_else(|| {
                memory_rows
                    .iter()
                    .find_map(|row| row.strongest_interval.as_ref())
            })
            .or_else(|| anchor_rows.first().map(|row| &row.temporal));
        let profile = profile_by_key.get(&key).copied();
        summary.samples.push(TemporalSourceClaimSupportSample {
            document_id: case.document_id.clone(),
            claim_id: claim_id.0.clone(),
            claim_label: profile.map(|row| row.label.clone()).unwrap_or_default(),
            proposition_id: profile
                .map(|row| row.proposition_id.clone())
                .unwrap_or_default(),
            proposition_predicate: profile
                .map(|row| row.normalized_predicate.clone())
                .unwrap_or_default(),
            anchor_count: anchor_rows.len(),
            interval_count: interval_rows.len(),
            gap_count: gap_rows.len(),
            memory_card_count: memory_rows.len(),
            claim_atom_count: claim_atom_rows.len(),
            reference_edge_count: reference_rows.len(),
            constraint_count: constraint_rows.len(),
            has_any_support: has_any_support,
            anchor_sources,
            gap_reasons,
            strongest_valid_from: strongest.and_then(|row| row.valid_from),
            strongest_recorded_from: strongest.and_then(|row| row.recorded_from),
        });
    }

    summary
}

fn temporal_claim_key(document_id: &str, event_id: &str) -> String {
    format!("{document_id}::{event_id}")
}

fn unique_sorted_strings(values: &[&str]) -> Vec<String> {
    let mut rows = values
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    rows.sort();
    rows.dedup();
    rows
}

fn causal_query_request(
    target: &CausalTargetCandidate,
    config: &DepthAuditConfig,
) -> GraphRetrievedCausalExplanationQueryRequest {
    GraphRetrievedCausalExplanationQueryRequest {
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
    }
}

fn execute_causal_query(
    store: &PhoenixOvergraphStore,
    scope: &ScopeKey,
    query_session: Option<&ScopeQuerySession>,
    config: &DepthAuditConfig,
    target: &CausalTargetCandidate,
) -> CachedCausalQueryResult {
    let request = causal_query_request(target, config);
    match query_session {
        Some(session) => retrieved_causal_explanation_with_session(store, session, &request)
            .map_err(|error| error.to_string()),
        None => {
            retrieved_causal_explanation(store, scope, &request).map_err(|error| error.to_string())
        }
    }
}

fn run_scope_probes(
    store: &PhoenixOvergraphStore,
    scope: &ScopeKey,
    query_session: Option<&ScopeQuerySession>,
    causal_targets: &[CausalTargetCandidate],
    causal_query_cache: &mut BTreeMap<String, CachedCausalQueryResult>,
    config: &DepthAuditConfig,
    graph_sidecar: &phoenix_semantic_v2::GraphScopeSidecar,
    instrumentation: &mut DepthAuditInstrumentation,
) -> Result<ScopeProbeSummary, String> {
    let world_anchors =
        collect_world_anchors(&graph_sidecar.graph_batch.vertices, config.probe_limit);
    let mut probes = ScopeProbeSummary::default();
    for anchor in &world_anchors {
        let world_request = GraphRetrievedWorldStateQueryRequest {
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
        };
        let world_started = Instant::now();
        let world = match query_session {
            Some(session) => retrieved_world_state_with_session(store, session, &world_request),
            None => retrieved_world_state(store, scope, &world_request),
        };
        instrumentation
            .world_probe_query
            .record(elapsed_us(world_started));
        push_probe_case(&mut probes.world_state, world_probe_case(anchor, world));
        let history_request = GraphRetrievedHistoryQueryRequest {
            query_text: format!("history of {} for {}", anchor.slot_key, anchor.entity_id),
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
        };
        let history_started = Instant::now();
        let history = match query_session {
            Some(session) => retrieved_history_with_session(store, session, &history_request),
            None => retrieved_history(store, scope, &history_request),
        };
        instrumentation
            .history_probe_query
            .record(elapsed_us(history_started));
        push_probe_case(&mut probes.history, history_probe_case(anchor, history));
    }
    for target in causal_targets {
        if !causal_query_cache.contains_key(target.vertex_id.as_str()) {
            let causal_started = Instant::now();
            let causal = execute_causal_query(store, scope, query_session, config, target);
            instrumentation
                .causal_probe_query
                .record(elapsed_us(causal_started));
            causal_query_cache.insert(target.vertex_id.clone(), causal);
        }
        if let Some(causal) = causal_query_cache.get(target.vertex_id.as_str()) {
            push_probe_case(
                &mut probes.causal_explanation,
                causal_probe_case(target, causal),
            );
        }
    }
    Ok(probes)
}

fn summarize_causal_ledger(
    store: &PhoenixOvergraphStore,
    scope: &ScopeKey,
    query_session: Option<&ScopeQuerySession>,
    causal_targets: &[CausalTargetCandidate],
    causal_query_cache: &BTreeMap<String, CachedCausalQueryResult>,
    config: &DepthAuditConfig,
    graph_sidecar: &phoenix_semantic_v2::GraphScopeSidecar,
    semantic_sidecar: Option<&phoenix_semantic_v2::SemanticGraphScopeSidecar>,
    instrumentation: &mut DepthAuditInstrumentation,
) -> Result<CausalPathLedgerSummary, String> {
    let raw_started = Instant::now();
    let raw_snapshot = raw_causal_snapshot(graph_sidecar, semantic_sidecar);
    instrumentation
        .raw_causal_snapshot
        .record(elapsed_us(raw_started));
    let visible_started = Instant::now();
    let visible_snapshot = visible_causal_snapshot(graph_sidecar, semantic_sidecar, query_session)?;
    instrumentation
        .visible_causal_snapshot
        .record(elapsed_us(visible_started));
    let mut summary = CausalPathLedgerSummary {
        target_count: causal_targets.len(),
        ..CausalPathLedgerSummary::default()
    };
    for target in causal_targets {
        let ledger = causal_target_ledger(
            store,
            scope,
            query_session,
            causal_query_cache.get(target.vertex_id.as_str()),
            config,
            target,
            &raw_snapshot,
            &visible_snapshot,
            instrumentation,
        );
        summary.raw_path_candidate_total += ledger.raw_path_candidate_count;
        summary.visible_path_candidate_total += ledger.visible_path_candidate_count;
        summary.ranked_candidate_total += ledger.ranked_candidate_count;
        summary.raw_path_target_count += (ledger.raw_path_candidate_count > 0) as usize;
        summary.visible_path_target_count += (ledger.visible_path_candidate_count > 0) as usize;
        summary.ranked_path_target_count += (ledger.ranked_candidate_count > 0) as usize;
        summary.plane_allowed_target_count += (ledger.plane_allowed_candidate_count > 0) as usize;
        summary.raw_missing_endpoint_edge_total += ledger.raw_missing_endpoint_edges;
        summary.visible_missing_endpoint_edge_total += ledger.visible_missing_endpoint_edges;
        summary.raw_incoming_missing_source_total += ledger.raw_incoming_missing_source_edges;
        summary.visible_incoming_missing_source_total +=
            ledger.visible_incoming_missing_source_edges;
        merge_counts(
            &mut summary.endpoint_shape_counts,
            &ledger.endpoint_shape_counts,
        );
        merge_counts(
            &mut summary.source_resolution_tier_counts,
            &ledger.source_resolution_tier_counts,
        );
        merge_counts(
            &mut summary.target_resolution_tier_counts,
            &ledger.target_resolution_tier_counts,
        );
        summary.fallback_source_edge_total += ledger.fallback_source_edges;
        summary.fallback_target_edge_total += ledger.fallback_target_edges;
        summary.promoted_source_edge_total += ledger.promoted_source_edges;
        summary.promoted_target_edge_total += ledger.promoted_target_edges;
        if ledger.query_abstain {
            summary.query_abstained_count += 1;
        } else {
            summary.query_answered_count += 1;
        }
        if let Some(reason) = ledger.query_abstain_reason.as_deref() {
            *summary.reason_counts.entry(reason.to_owned()).or_default() += 1;
        }
        summary.samples.push(ledger);
    }
    Ok(summary)
}

fn causal_target_ledger(
    store: &PhoenixOvergraphStore,
    scope: &ScopeKey,
    query_session: Option<&ScopeQuerySession>,
    cached_query: Option<&CachedCausalQueryResult>,
    config: &DepthAuditConfig,
    target: &CausalTargetCandidate,
    raw_snapshot: &KernelGraphSnapshot,
    visible_snapshot: &KernelGraphSnapshot,
    instrumentation: &mut DepthAuditInstrumentation,
) -> CausalTargetLedger {
    let raw_vertices = vertex_map(&raw_snapshot.vertices);
    let visible_vertices = vertex_map(&visible_snapshot.vertices);
    let raw_paths = causal_path_candidate_views_from_snapshot(
        raw_snapshot,
        target.vertex_id.as_str(),
        3,
        config.causal_limit.saturating_mul(4).max(12),
    );
    let visible_paths = causal_path_candidate_views_from_snapshot(
        visible_snapshot,
        target.vertex_id.as_str(),
        3,
        config.causal_limit.saturating_mul(4).max(12),
    );
    let rank_request = GraphCausalExplanationQueryRequest {
        target_vertex_id: target.vertex_id.clone(),
        valid_at: None,
        recorded_at: None,
        include_candidate_graph: true,
        max_depth: 3,
        limit: Some(config.causal_limit),
        truth_plane: GraphTruthPlane::WorldState,
    };
    let ranked = rank_causal_explanation_answer(&rank_request, visible_snapshot);
    let owned_query;
    let query = match cached_query {
        Some(query) => query,
        None => {
            let query_started = Instant::now();
            owned_query = execute_causal_query(store, scope, query_session, config, target);
            instrumentation
                .causal_ledger_query
                .record(elapsed_us(query_started));
            &owned_query
        }
    };
    let (query_candidate_count, query_abstain, query_abstain_reason) = match query {
        Ok(Some(answer)) => (
            answer.answer.candidates.len(),
            answer.answer.abstain,
            answer.answer.abstain_reason.clone(),
        ),
        Ok(None) => (
            0,
            true,
            Some("projection kernel was unavailable".to_owned()),
        ),
        Err(error) => (0, true, Some(error.clone())),
    };
    let raw_target = raw_vertices.get(target.vertex_id.as_str()).copied();
    let visible_target = visible_vertices.get(target.vertex_id.as_str()).copied();
    let raw_gaps = causal_endpoint_gaps(raw_snapshot, target.vertex_id.as_str());
    let visible_gaps = causal_endpoint_gaps(visible_snapshot, target.vertex_id.as_str());
    let endpoint_stats = causal_endpoint_stats(raw_snapshot, target.vertex_id.as_str());
    CausalTargetLedger {
        target_vertex_id: target.vertex_id.clone(),
        target_kind: raw_target.map(|vertex| vertex.kind.clone()),
        target_label: raw_target
            .map(describe_vertex)
            .unwrap_or_else(|| target.description.clone()),
        target_visible: visible_target.is_some(),
        raw_incoming_causal_edges: causal_edge_count_to(raw_snapshot, target.vertex_id.as_str()),
        raw_outgoing_causal_edges: causal_edge_count_from(raw_snapshot, target.vertex_id.as_str()),
        visible_incoming_causal_edges: causal_edge_count_to(
            visible_snapshot,
            target.vertex_id.as_str(),
        ),
        visible_outgoing_causal_edges: causal_edge_count_from(
            visible_snapshot,
            target.vertex_id.as_str(),
        ),
        raw_missing_endpoint_edges: raw_gaps.missing_endpoint_edges,
        visible_missing_endpoint_edges: visible_gaps.missing_endpoint_edges,
        raw_incoming_missing_source_edges: raw_gaps.incoming_missing_source_edges,
        visible_incoming_missing_source_edges: visible_gaps.incoming_missing_source_edges,
        endpoint_shape_counts: endpoint_stats.endpoint_shape_counts,
        source_resolution_tier_counts: endpoint_stats.source_resolution_tier_counts,
        target_resolution_tier_counts: endpoint_stats.target_resolution_tier_counts,
        fallback_source_edges: endpoint_stats.fallback_source_edges,
        fallback_target_edges: endpoint_stats.fallback_target_edges,
        promoted_source_edges: endpoint_stats.promoted_source_edges,
        promoted_target_edges: endpoint_stats.promoted_target_edges,
        raw_path_candidate_count: raw_paths.len(),
        visible_path_candidate_count: visible_paths.len(),
        ranked_candidate_count: ranked.candidates.len(),
        plane_allowed_candidate_count: ranked
            .candidates
            .iter()
            .filter(|candidate| candidate.plane_allowed)
            .count(),
        query_candidate_count,
        query_abstain,
        query_abstain_reason,
        best_ranked_score_millis: ranked
            .candidates
            .first()
            .map(|candidate| (candidate.answer_score * 1000.0).round() as i64),
        best_ranked_depth: ranked
            .candidates
            .first()
            .map(|candidate| candidate.hops.len()),
        raw_incoming_samples: causal_edge_samples(raw_snapshot, target.vertex_id.as_str(), 4),
    }
}

fn raw_causal_snapshot(
    graph_sidecar: &phoenix_semantic_v2::GraphScopeSidecar,
    semantic_sidecar: Option<&phoenix_semantic_v2::SemanticGraphScopeSidecar>,
) -> KernelGraphSnapshot {
    KernelGraphSnapshot {
        vertices: graph_sidecar.graph_batch.vertices.clone(),
        asserted_edges: graph_sidecar.graph_batch.edges.clone(),
        candidate_edges: semantic_sidecar
            .map(|sidecar| sidecar.candidate_graph_batch.edges.clone())
            .unwrap_or_default(),
    }
}

fn visible_causal_snapshot(
    graph_sidecar: &phoenix_semantic_v2::GraphScopeSidecar,
    semantic_sidecar: Option<&phoenix_semantic_v2::SemanticGraphScopeSidecar>,
    query_session: Option<&ScopeQuerySession>,
) -> Result<KernelGraphSnapshot, String> {
    if let Some(session) = query_session {
        return Ok(session.view_as_of(KernelViewRequest {
            valid_at: None,
            recorded_at: None,
            include_candidate_graph: true,
        }));
    }
    let mut kernel = PhoenixGraphKernel::new();
    kernel
        .apply_kernel_batch(graph_sidecar.graph_batch.clone())
        .map_err(|error| error.to_string())?;
    if let Some(sidecar) = semantic_sidecar {
        kernel
            .apply_kernel_batch(sidecar.candidate_graph_batch.clone())
            .map_err(|error| error.to_string())?;
    }
    Ok(kernel.view_as_of(KernelViewRequest {
        valid_at: None,
        recorded_at: None,
        include_candidate_graph: true,
    }))
}

fn vertex_map(vertices: &[KernelVertex]) -> BTreeMap<&str, &KernelVertex> {
    vertices
        .iter()
        .map(|vertex| (vertex.id.0.as_str(), vertex))
        .collect()
}

fn merge_counts(target: &mut BTreeMap<String, usize>, source: &BTreeMap<String, usize>) {
    for (key, count) in source {
        *target.entry(key.clone()).or_default() += count;
    }
}

fn causal_edge_count_to(snapshot: &KernelGraphSnapshot, target_id: &str) -> usize {
    snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
        .filter(|edge| edge.edge_type.0 == "causal_link" && edge.target_id.0 == target_id)
        .count()
}

fn causal_edge_count_from(snapshot: &KernelGraphSnapshot, source_id: &str) -> usize {
    snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
        .filter(|edge| edge.edge_type.0 == "causal_link" && edge.source_id.0 == source_id)
        .count()
}

#[derive(Clone, Copy, Debug, Default)]
struct CausalEndpointGaps {
    missing_endpoint_edges: usize,
    incoming_missing_source_edges: usize,
}

#[derive(Clone, Debug, Default)]
struct CausalEndpointStats {
    endpoint_shape_counts: BTreeMap<String, usize>,
    source_resolution_tier_counts: BTreeMap<String, usize>,
    target_resolution_tier_counts: BTreeMap<String, usize>,
    fallback_source_edges: usize,
    fallback_target_edges: usize,
    promoted_source_edges: usize,
    promoted_target_edges: usize,
}

fn causal_endpoint_gaps(snapshot: &KernelGraphSnapshot, target_id: &str) -> CausalEndpointGaps {
    let vertices = vertex_map(&snapshot.vertices);
    let mut gaps = CausalEndpointGaps::default();
    for edge in snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
        .filter(|edge| edge.edge_type.0 == "causal_link" && edge.target_id.0 == target_id)
    {
        let source_missing = !vertices.contains_key(edge.source_id.0.as_str());
        let target_missing = !vertices.contains_key(edge.target_id.0.as_str());
        if source_missing || target_missing {
            gaps.missing_endpoint_edges += 1;
        }
        if source_missing {
            gaps.incoming_missing_source_edges += 1;
        }
    }
    gaps
}

fn causal_endpoint_stats(snapshot: &KernelGraphSnapshot, target_id: &str) -> CausalEndpointStats {
    let vertices = vertex_map(&snapshot.vertices);
    let mut stats = CausalEndpointStats::default();
    for edge in snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
        .filter(|edge| edge.edge_type.0 == "causal_link" && edge.target_id.0 == target_id)
    {
        let source_kind = endpoint_kind(edge, &vertices, true);
        let target_kind = endpoint_kind(edge, &vertices, false);
        let shape = format!("{source_kind}->{target_kind}");
        *stats.endpoint_shape_counts.entry(shape).or_default() += 1;
        let source_tier =
            string_attr(&edge.attributes, "sourceEndpointResolutionTier").unwrap_or("unknown");
        let target_tier =
            string_attr(&edge.attributes, "targetEndpointResolutionTier").unwrap_or("unknown");
        *stats
            .source_resolution_tier_counts
            .entry(source_tier.to_owned())
            .or_default() += 1;
        *stats
            .target_resolution_tier_counts
            .entry(target_tier.to_owned())
            .or_default() += 1;
        stats.fallback_source_edges +=
            bool_attr(&edge.attributes, "sourceEndpointFallback") as usize;
        stats.fallback_target_edges +=
            bool_attr(&edge.attributes, "targetEndpointFallback") as usize;
        stats.promoted_source_edges +=
            bool_attr(&edge.attributes, "sourceEndpointPromoted") as usize;
        stats.promoted_target_edges +=
            bool_attr(&edge.attributes, "targetEndpointPromoted") as usize;
    }
    stats
}

fn endpoint_kind(
    edge: &KernelEdge,
    vertices: &BTreeMap<&str, &KernelVertex>,
    source: bool,
) -> String {
    let vertex_id = if source {
        edge.source_id.0.as_str()
    } else {
        edge.target_id.0.as_str()
    };
    vertices
        .get(vertex_id)
        .map(|vertex| vertex.kind.clone())
        .or_else(|| {
            let key = if source {
                "sourceSemanticNodeKind"
            } else {
                "targetSemanticNodeKind"
            };
            string_attr(&edge.attributes, key).map(str::to_owned)
        })
        .unwrap_or_else(|| "missing".to_owned())
}

fn bool_attr(value: &serde_json::Value, key: &str) -> bool {
    value
        .get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false)
}

fn causal_edge_samples(
    snapshot: &KernelGraphSnapshot,
    target_id: &str,
    limit: usize,
) -> Vec<CausalEdgeSample> {
    let vertices = vertex_map(&snapshot.vertices);
    let mut edges = snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
        .filter(|edge| edge.edge_type.0 == "causal_link" && edge.target_id.0 == target_id)
        .collect::<Vec<_>>();
    edges.sort_by(|left, right| {
        right
            .provenance
            .confidence
            .partial_cmp(&left.provenance.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.source_id.0.cmp(&right.source_id.0))
    });
    edges
        .into_iter()
        .take(limit.max(1))
        .map(|edge| causal_edge_sample(edge, &vertices))
        .collect()
}

fn causal_edge_sample(
    edge: &KernelEdge,
    vertices: &BTreeMap<&str, &KernelVertex>,
) -> CausalEdgeSample {
    let source = vertices.get(edge.source_id.0.as_str()).copied();
    let target = vertices.get(edge.target_id.0.as_str()).copied();
    CausalEdgeSample {
        source_id: edge.source_id.0.clone(),
        source_present: source.is_some(),
        source_semantic_kind: string_attr(&edge.attributes, "sourceSemanticNodeKind")
            .map(str::to_owned),
        source_semantic_id: string_attr(&edge.attributes, "sourceSemanticNodeId")
            .map(str::to_owned),
        source_semantic_label: string_attr(&edge.attributes, "sourceSemanticLabel")
            .map(str::to_owned),
        source_resolution_tier: string_attr(&edge.attributes, "sourceEndpointResolutionTier")
            .map(str::to_owned),
        source_endpoint_fallback: bool_attr(&edge.attributes, "sourceEndpointFallback"),
        source_endpoint_promoted: bool_attr(&edge.attributes, "sourceEndpointPromoted"),
        source_kind: source.map(|vertex| vertex.kind.clone()),
        source_label: source
            .map(describe_vertex)
            .unwrap_or_else(|| edge.source_id.0.clone()),
        target_id: edge.target_id.0.clone(),
        target_present: target.is_some(),
        target_semantic_kind: string_attr(&edge.attributes, "targetSemanticNodeKind")
            .map(str::to_owned),
        target_semantic_id: string_attr(&edge.attributes, "targetSemanticNodeId")
            .map(str::to_owned),
        target_semantic_label: string_attr(&edge.attributes, "targetSemanticLabel")
            .map(str::to_owned),
        target_resolution_tier: string_attr(&edge.attributes, "targetEndpointResolutionTier")
            .map(str::to_owned),
        target_endpoint_fallback: bool_attr(&edge.attributes, "targetEndpointFallback"),
        target_endpoint_promoted: bool_attr(&edge.attributes, "targetEndpointPromoted"),
        target_kind: target.map(|vertex| vertex.kind.clone()),
        target_label: target
            .map(describe_vertex)
            .unwrap_or_else(|| edge.target_id.0.clone()),
        edge_type: edge.edge_type.0.clone(),
        layer: format!("{:?}", edge.layer),
        relation_kind: string_attr(&edge.attributes, "relationKind").map(str::to_owned),
        status: string_attr(&edge.attributes, "status").map(str::to_owned),
        confidence_millis: edge
            .provenance
            .confidence
            .map(|confidence| (confidence * 1000.0).round() as i64),
        valid_from: edge.temporal.valid_from,
        valid_to: edge.temporal.valid_to,
        recorded_at: edge.temporal.recorded_at,
        evidence_ref_count: edge.provenance.evidence_refs.len(),
    }
}

fn world_probe_case(
    anchor: &WorldAnchor,
    result: Result<
        Option<phoenix_graph_post::api::GraphRetrievedWorldStateAnswer>,
        phoenix_graph_post::api::GraphQueryError,
    >,
) -> ProbeCase {
    match result {
        Ok(Some(answer)) => ProbeCase {
            probe_id: format!("{}::{}", anchor.entity_id, anchor.slot_key),
            query_text: answer.query_text,
            selected_id: answer
                .answer
                .selected
                .as_ref()
                .map(|candidate| candidate.state.state_vertex_id.clone()),
            selected_label: answer
                .answer
                .selected
                .as_ref()
                .map(|candidate| candidate.state.value.clone()),
            candidate_count: answer.answer.candidates.len(),
            seed_count: answer.seeds.len(),
            region_vertex_count: answer.region.vertex_count,
            asserted_edge_count: answer.region.asserted_edge_count,
            candidate_edge_count: answer.region.candidate_edge_count,
            truncated: answer.region.truncated,
            abstain: answer.answer.abstain,
            abstain_reason: answer.answer.abstain_reason.clone(),
            error: None,
        },
        Ok(None) => error_probe_case(
            &format!("{}::{}", anchor.entity_id, anchor.slot_key),
            &anchor.query_text,
            "projection kernel was unavailable",
        ),
        Err(error) => error_probe_case(
            &format!("{}::{}", anchor.entity_id, anchor.slot_key),
            &anchor.query_text,
            &error.to_string(),
        ),
    }
}

fn history_probe_case(
    anchor: &WorldAnchor,
    result: Result<
        Option<phoenix_graph_post::api::GraphRetrievedHistoryAnswer>,
        phoenix_graph_post::api::GraphQueryError,
    >,
) -> ProbeCase {
    match result {
        Ok(Some(answer)) => ProbeCase {
            probe_id: format!("{}::{}", anchor.entity_id, anchor.slot_key),
            query_text: answer.query_text,
            selected_id: answer
                .answer
                .selected
                .as_ref()
                .map(|candidate| candidate.change.state.state_vertex_id.clone()),
            selected_label: answer.answer.selected.as_ref().map(|candidate| {
                format!(
                    "{:?} {}",
                    candidate.change.change_kind, candidate.change.state.value
                )
            }),
            candidate_count: answer.answer.candidates.len(),
            seed_count: answer.seeds.len(),
            region_vertex_count: answer.region.vertex_count,
            asserted_edge_count: answer.region.asserted_edge_count,
            candidate_edge_count: answer.region.candidate_edge_count,
            truncated: answer.region.truncated,
            abstain: answer.answer.abstain,
            abstain_reason: answer.answer.abstain_reason.clone(),
            error: None,
        },
        Ok(None) => error_probe_case(
            &format!("{}::{}", anchor.entity_id, anchor.slot_key),
            &anchor.query_text,
            "projection kernel was unavailable",
        ),
        Err(error) => error_probe_case(
            &format!("{}::{}", anchor.entity_id, anchor.slot_key),
            &anchor.query_text,
            &error.to_string(),
        ),
    }
}

fn causal_probe_case(
    target: &CausalTargetCandidate,
    result: &CachedCausalQueryResult,
) -> ProbeCase {
    match result {
        Ok(Some(answer)) => ProbeCase {
            probe_id: target.vertex_id.clone(),
            query_text: answer.query_text.clone(),
            selected_id: answer
                .answer
                .selected
                .as_ref()
                .map(|path| path.source_vertex_id.clone()),
            selected_label: answer
                .answer
                .selected
                .as_ref()
                .map(|path| format!("{} hops", path.hops.len())),
            candidate_count: answer.answer.candidates.len(),
            seed_count: answer.seeds.len(),
            region_vertex_count: answer.region.vertex_count,
            asserted_edge_count: answer.region.asserted_edge_count,
            candidate_edge_count: answer.region.candidate_edge_count,
            truncated: answer.region.truncated,
            abstain: answer.answer.abstain,
            abstain_reason: answer.answer.abstain_reason.clone(),
            error: None,
        },
        Ok(None) => error_probe_case(
            &target.vertex_id,
            &target.query_text,
            "projection kernel was unavailable",
        ),
        Err(error) => error_probe_case(&target.vertex_id, &target.query_text, error),
    }
}

fn push_probe_case(summary: &mut ProbeFamilySummary, case: ProbeCase) {
    summary.total += 1;
    summary.candidate_count_total += case.candidate_count;
    summary.region_vertex_count_total += case.region_vertex_count;
    if case.error.is_some() {
        summary.errors += 1;
    } else if case.abstain {
        summary.abstained += 1;
    } else {
        summary.answered += 1;
    }
    if let Some(reason) = case.abstain_reason.as_deref().or(case.error.as_deref()) {
        *summary.reason_counts.entry(reason.to_owned()).or_default() += 1;
    }
    summary.samples.push(case);
}

fn summarize_losses(
    sidecars: &ScopeSidecarSummary,
    vertices: &[KernelVertex],
    edges: &[KernelEdge],
) -> ScopeLossSummary {
    let anchors = collect_world_anchors(vertices, usize::MAX / 2);
    let causal_targets = discover_causal_target_candidates(vertices, edges, vertices.len().max(1));
    ScopeLossSummary {
        world_anchor_count: anchors.len(),
        path_bearing_causal_target_count: causal_targets
            .iter()
            .filter(|candidate| candidate.path_bearing)
            .count(),
        graph_state_per_memory_state_millis: ratio_millis(
            sidecars
                .graph
                .as_ref()
                .map(|stage| stage.summary.state_node_count),
            sidecars
                .memory
                .as_ref()
                .map(|stage| stage.summary.state_count),
        ),
        graph_event_per_memory_event_millis: ratio_millis(
            sidecars
                .graph
                .as_ref()
                .map(|stage| stage.summary.event_node_count),
            sidecars
                .memory
                .as_ref()
                .map(|stage| stage.summary.event_count),
        ),
        graph_event_per_canonical_event_millis: ratio_millis(
            sidecars
                .graph
                .as_ref()
                .map(|stage| stage.summary.event_node_count),
            sidecars
                .event_identity
                .as_ref()
                .map(|stage| stage.summary.canonical_event_count),
        ),
        temporal_interval_per_canonical_event_millis: ratio_millis(
            sidecars
                .temporal
                .as_ref()
                .map(|stage| stage.summary.interval_count),
            sidecars
                .event_identity
                .as_ref()
                .map(|stage| stage.summary.canonical_event_count),
        ),
        causal_edge_per_graph_event_millis: ratio_millis(
            sidecars
                .causal
                .as_ref()
                .map(|stage| stage.summary.committed_edge_count),
            sidecars
                .graph
                .as_ref()
                .map(|stage| stage.summary.event_node_count),
        ),
        semantic_candidate_edge_per_graph_edge_millis: ratio_millis(
            sidecars
                .semantic_graph
                .as_ref()
                .map(|stage| stage.summary.edge_count),
            sidecars
                .graph
                .as_ref()
                .map(|stage| stage.summary.projection_edge_count),
        ),
    }
}

fn collect_world_anchors(vertices: &[KernelVertex], limit: usize) -> Vec<WorldAnchor> {
    let mut ranked = vertices
        .iter()
        .filter_map(|vertex| {
            let entity_id = vertex_entity_id(vertex)?;
            let slot_key = vertex_slot_key(vertex)?;
            Some((
                world_anchor_score(vertex, slot_key),
                entity_id.to_owned(),
                slot_key.to_owned(),
            ))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.cmp(left));
    let mut seen = BTreeMap::<(String, String), ()>::new();
    let mut anchors = Vec::new();
    for (_, entity_id, slot_key) in ranked {
        if seen
            .insert((entity_id.clone(), slot_key.clone()), ())
            .is_some()
        {
            continue;
        }
        anchors.push(WorldAnchor {
            query_text: format!("current {} for {}", slot_key, entity_id),
            entity_id,
            slot_key,
        });
        if anchors.len() >= limit {
            break;
        }
    }
    anchors
}

fn load_scope_archives(
    store: &PhoenixOvergraphStore,
) -> Result<Vec<(String, ScopeKey, Vec<DocumentArchive>)>, String> {
    let archives = store
        .load_latest_document_archives(None)
        .map_err(|error| error.to_string())?;
    let mut grouped = BTreeMap::<String, (ScopeKey, Vec<DocumentArchive>)>::new();
    for archive in archives {
        let entry = grouped
            .entry(archive.manifest.scope_key.clone())
            .or_insert_with(|| (archive.manifest.scope.clone(), Vec::new()));
        entry.1.push(archive);
    }
    Ok(grouped
        .into_iter()
        .map(|(scope_key, (scope, archives))| (scope_key, scope, archives))
        .collect())
}

fn summarize_archives(archives: &[DocumentArchive]) -> ArchiveSummary {
    let mut summary = ArchiveSummary::default();
    for archive in archives {
        summary.document_count += 1;
        summary.text_len += archive.manifest.text_len;
        summary.token_count += archive.tokens.len();
        summary.sentence_count += archive.sentences.len();
        summary.mention_count += archive.mentions.len();
        summary.chunk_count += archive.chunks.len();
        summary.entity_count += archive.entities.len();
        summary.relation_count += archive.relations.len();
        summary.alias_count += archive.manifest.alias_count;
        summary.discovery_count += archive.manifest.discovery_count;
    }
    summary
}

fn ensure_graph_sidecar(
    store: &PhoenixOvergraphStore,
    scope: &ScopeKey,
) -> Result<phoenix_semantic_v2::GraphScopeSidecar, String> {
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

fn refresh_pipeline_sidecars(
    api: &PhoenixPipelineApi<PhoenixOvergraphStore>,
    instrumentation: &mut DepthAuditInstrumentation,
) -> Result<(), String> {
    let created_at = now_ms();
    let event_identity_started = Instant::now();
    api.run_event_identity_scope(None, created_at)
        .map_err(|error| error.to_string())?;
    instrumentation
        .refresh_event_identity
        .record(elapsed_us(event_identity_started));
    let temporal_started = Instant::now();
    api.run_temporal_scope(None, created_at)
        .map_err(|error| error.to_string())?;
    instrumentation
        .refresh_temporal
        .record(elapsed_us(temporal_started));
    let causal_started = Instant::now();
    api.run_causal_scope(None, created_at)
        .map_err(|error| error.to_string())?;
    instrumentation
        .refresh_causal
        .record(elapsed_us(causal_started));
    let late_sidecars_started = Instant::now();
    api.run_late_sidecar_scope(None, created_at)
        .map_err(|error| error.to_string())?;
    instrumentation
        .refresh_late_sidecars
        .record(elapsed_us(late_sidecars_started));
    Ok(())
}

fn ensure_semantic_sidecar(
    store: &PhoenixOvergraphStore,
    scope: &ScopeKey,
    config: &DepthAuditConfig,
) -> Result<phoenix_semantic_v2::SemanticGraphScopeSidecar, String> {
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

fn accumulate_missing_sidecars(
    counts: &mut BTreeMap<String, usize>,
    sidecars: &ScopeSidecarSummary,
) {
    for (name, present) in [
        ("eventIdentity", sidecars.event_identity.is_some()),
        ("temporal", sidecars.temporal.is_some()),
        ("causal", sidecars.causal.is_some()),
        ("memory", sidecars.memory.is_some()),
        ("stateSchema", sidecars.state_schema.is_some()),
        ("graph", sidecars.graph.is_some()),
        ("semanticGraph", sidecars.semantic_graph.is_some()),
    ] {
        if !present {
            *counts.entry(name.to_owned()).or_default() += 1;
        }
    }
}

fn accumulate_probe_summary(total: &mut ScopeProbeSummary, scope: &ScopeProbeSummary) {
    accumulate_family(&mut total.world_state, &scope.world_state);
    accumulate_family(&mut total.history, &scope.history);
    accumulate_family(&mut total.causal_explanation, &scope.causal_explanation);
}

fn accumulate_family(total: &mut ProbeFamilySummary, scope: &ProbeFamilySummary) {
    total.total += scope.total;
    total.answered += scope.answered;
    total.abstained += scope.abstained;
    total.errors += scope.errors;
    total.candidate_count_total += scope.candidate_count_total;
    total.region_vertex_count_total += scope.region_vertex_count_total;
    for (reason, count) in &scope.reason_counts {
        *total.reason_counts.entry(reason.clone()).or_default() += count;
    }
}

fn merge_archive(total: &mut ArchiveSummary, scope: &ArchiveSummary) {
    total.document_count += scope.document_count;
    total.text_len += scope.text_len;
    total.token_count += scope.token_count;
    total.sentence_count += scope.sentence_count;
    total.mention_count += scope.mention_count;
    total.chunk_count += scope.chunk_count;
    total.entity_count += scope.entity_count;
    total.relation_count += scope.relation_count;
    total.alias_count += scope.alias_count;
    total.discovery_count += scope.discovery_count;
}

fn error_probe_case(probe_id: &str, query_text: &str, error: &str) -> ProbeCase {
    ProbeCase {
        probe_id: probe_id.to_owned(),
        query_text: query_text.to_owned(),
        error: Some(error.to_owned()),
        ..ProbeCase::default()
    }
}

fn elapsed_us(started: Instant) -> u64 {
    started.elapsed().as_micros() as u64
}

fn ratio_millis(numer: Option<usize>, denom: Option<usize>) -> Option<u32> {
    let numer = numer?;
    let denom = denom?;
    if denom == 0 {
        None
    } else {
        Some(((numer as u64 * 1000) / denom as u64) as u32)
    }
}

fn world_anchor_score(vertex: &KernelVertex, slot_key: &str) -> i32 {
    let mut score = match vertex.kind.as_str() {
        "state" => 240,
        "claim" => 140,
        _ => 0,
    };
    score += if slot_key.starts_with("entity.") {
        180
    } else {
        0
    };
    score += if string_attr(&vertex.value, "status")
        .or_else(|| string_attr(&vertex.attributes, "status"))
        == Some("active")
    {
        50
    } else {
        0
    };
    score += match slot_key {
        "entity.location" => 140,
        "entity.membership" => 130,
        "entity.employer" => 120,
        "entity.role" => 110,
        "entity.status" => 100,
        key if key.starts_with("entity.") => 90,
        key if key.starts_with("relation.") => -260,
        _ => 0,
    };
    if vertex.id.0.contains("conflict") || vertex.id.0.contains("gap") {
        score -= 120;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::{collect_world_anchors, ratio_millis};
    use phoenix_graph_kernel::{KernelVertex, KernelVertexClass, KernelVertexId};
    use serde_json::json;

    #[test]
    fn ratio_millis_requires_nonzero_denominator() {
        assert_eq!(ratio_millis(Some(3), Some(2)), Some(1500));
        assert_eq!(ratio_millis(Some(3), Some(0)), None);
    }

    #[test]
    fn collect_world_anchors_prefers_stable_entity_slots() {
        let anchors = collect_world_anchors(
            &[
                vertex("graph::state::state:test::a:entity.status", "entity.status"),
                vertex(
                    "graph::state::state:test::a:entity.location",
                    "entity.location",
                ),
            ],
            2,
        );
        assert_eq!(anchors[0].slot_key, "entity.location");
    }

    fn vertex(id: &str, slot_key: &str) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: "state".to_owned(),
            class: KernelVertexClass::State,
            entity_id: Some("test::a".to_owned()),
            value: json!({ "slotKey": slot_key, "status": "active" }),
            ..KernelVertex::default()
        }
    }
}
