use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use phoenix_graph_kernel::KernelGraphSnapshot;
use phoenix_ingest_overgraph::{InvarantV3Config, PhoenixInvarantV3};
use phoenix_memory_post::api as memory_api;
use phoenix_rel_post::{
    default_relation_type_specs, draft_relation_decisions, run_primary_relation_lane,
    GlirelRelationTypeSpec,
};
use phoenix_scope_analysis::ScopeAnalysisContext;
use phoenix_state_schema_post::api as state_schema_api;
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixEventIdentityPatchStore,
    PhoenixGraphKernelStoreV2, PhoenixMemoryPatchStore, PhoenixRelationPatchStore,
    PhoenixScopeRuntimeStore, PhoenixStateSchemaPatchStore, ScopeImageSpec, ScopeRuntimeImage,
};
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::{DocumentId, IngestDocument, ScopeKey};

mod report;

use report::{build_deltas, build_projections, flow_phase_stats, PostIngestBenchRun};
pub use report::{mean_ms, BenchReport, CaseReport, FlowCaseReport};

#[derive(Debug, Clone)]
pub struct Config {
    pub warmups: usize,
    pub iterations: usize,
    pub chapter: usize,
    pub json: bool,
    pub output_path: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            warmups: 1,
            iterations: 5,
            chapter: 1,
            json: false,
            output_path: None,
        }
    }
}

#[derive(Debug, Clone)]
struct CaseInput {
    case_id: String,
    title: String,
    text: String,
}

pub fn run(config: &Config) -> Result<BenchReport, String> {
    if config.iterations == 0 {
        return Err("--iterations must be greater than 0".to_owned());
    }

    let root = workspace_root();
    let full_text = fs::read_to_string(root.join("docs").join("shortrun.md"))
        .map_err(|error| format!("failed to read docs/shortrun.md: {error}"))?;
    let full_case = CaseInput {
        case_id: "shortrun-full".to_owned(),
        title: "Shortrun Full".to_owned(),
        text: full_text.clone(),
    };
    let chapter_case = extract_chapter_case(&full_text, config.chapter)?;

    let ingest = PhoenixInvarantV3::new(InvarantV3Config::default());
    let relation_specs = default_relation_type_specs();
    let cases = vec![
        run_case(&root, &ingest, &full_case, config, &relation_specs)?,
        run_case(&root, &ingest, &chapter_case, config, &relation_specs)?,
    ];

    Ok(BenchReport {
        corpus: "shortrun".to_owned(),
        warmups: config.warmups,
        iterations: config.iterations,
        projections: build_projections(&cases),
        deltas: build_deltas(&cases),
        cases,
    })
}

pub fn default_output_path() -> PathBuf {
    workspace_root()
        .join("rust-native")
        .join("phoenix")
        .join("reports")
        .join("post-ingest-bench-shortrun.json")
}

fn run_case(
    root: &Path,
    ingest: &PhoenixInvarantV3,
    input: &CaseInput,
    config: &Config,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Result<CaseReport, String> {
    for warmup in 0..config.warmups {
        let (store, store_path) = seed_store(root, ingest, input, "legacy-warmup", warmup)?;
        let result = run_legacy_post_ingest(&store, relation_specs)
            .map_err(|error| format!("legacy warmup failed for {}: {error}", input.case_id));
        drop(store);
        cleanup_seed_store(&store_path);
        let _ = result?;
    }

    let mut legacy_runs = Vec::with_capacity(config.iterations);
    for iteration in 0..config.iterations {
        let (store, store_path) = seed_store(root, ingest, input, "legacy", iteration)?;
        let result = run_legacy_post_ingest(&store, relation_specs)
            .map_err(|error| format!("legacy benchmark failed for {}: {error}", input.case_id));
        drop(store);
        cleanup_seed_store(&store_path);
        legacy_runs.push(result?);
    }

    for warmup in 0..config.warmups {
        let (store, store_path) = seed_store(root, ingest, input, "shared-warmup", warmup)?;
        let result = run_shared_post_ingest(&store, relation_specs)
            .map_err(|error| format!("shared warmup failed for {}: {error}", input.case_id));
        drop(store);
        cleanup_seed_store(&store_path);
        let _ = result?;
    }

    let mut shared_runs = Vec::with_capacity(config.iterations);
    for iteration in 0..config.iterations {
        let (store, store_path) = seed_store(root, ingest, input, "shared", iteration)?;
        let result = run_shared_post_ingest(&store, relation_specs)
            .map_err(|error| format!("shared benchmark failed for {}: {error}", input.case_id));
        drop(store);
        cleanup_seed_store(&store_path);
        shared_runs.push(result?);
    }

    let legacy_last = legacy_runs
        .last()
        .cloned()
        .ok_or_else(|| format!("no legacy runs captured for {}", input.case_id))?;
    let shared_last = shared_runs
        .last()
        .cloned()
        .ok_or_else(|| format!("no shared runs captured for {}", input.case_id))?;

    Ok(CaseReport {
        case_id: input.case_id.clone(),
        title: input.title.clone(),
        text_bytes: input.text.len(),
        legacy: FlowCaseReport {
            counts: legacy_last.counts,
            phases: flow_phase_stats(&legacy_runs),
        },
        shared: FlowCaseReport {
            counts: shared_last.counts,
            phases: flow_phase_stats(&shared_runs),
        },
    })
}

fn seed_store(
    root: &Path,
    ingest: &PhoenixInvarantV3,
    input: &CaseInput,
    lane: &str,
    iteration: usize,
) -> Result<(PhoenixOvergraphStore, PathBuf), String> {
    let store_path = root
        .join("rust-native")
        .join("phoenix")
        .join("reports")
        .join("bench-stores")
        .join(format!(
            "{lane}-{}-{}-{iteration}",
            input.case_id,
            std::process::id()
        ));
    let _ = fs::remove_dir_all(&store_path);

    let store = PhoenixOvergraphStore::open(&store_path)
        .map_err(|error| format!("failed to open {}: {error}", store_path.display()))?;
    store
        .init_archive_schema()
        .map_err(|error| format!("failed to init archive schema: {error}"))?;
    store
        .init_graph_kernel_schema()
        .map_err(|error| format!("failed to init kernel schema: {error}"))?;
    store
        .init_er_patch_schema()
        .map_err(|error| format!("failed to init er patch schema: {error}"))?;
    store
        .init_event_identity_patch_schema()
        .map_err(|error| format!("failed to init event identity schema: {error}"))?;
    store
        .init_relation_patch_schema()
        .map_err(|error| format!("failed to init relation patch schema: {error}"))?;
    store
        .init_state_schema_patch_schema()
        .map_err(|error| format!("failed to init state schema schema: {error}"))?;
    store
        .init_memory_patch_schema()
        .map_err(|error| format!("failed to init memory patch schema: {error}"))?;
    store
        .write_kernel_checkpoint(1, "bench-seed", &KernelGraphSnapshot::default())
        .map_err(|error| format!("failed to seed kernel checkpoint: {error}"))?;

    let document = IngestDocument {
        document_id: DocumentId(input.case_id.clone()),
        note_id: None,
        title: input.title.clone(),
        text: input.text.clone(),
        scope: ScopeKey::default(),
    };
    ingest
        .ingest_documents_native(&store, None, &[document], 0, benchmark_created_at())
        .map_err(|error| format!("failed to seed store for {}: {error}", input.case_id))?;
    Ok((store, store_path))
}

fn run_legacy_post_ingest(
    store: &PhoenixOvergraphStore,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Result<PostIngestBenchRun, String> {
    let mut report = PostIngestBenchRun::default();
    report.counts.dirty_scope_count = store
        .list_dirty_scopes()
        .map_err(|error| format!("failed to list dirty scopes: {error}"))?
        .len();

    let total_started = Instant::now();

    let relation_started = Instant::now();
    let mut relation_batches = phoenix_rel_post::api::derive_batches(store, None)
        .map_err(|error| format!("failed to derive relation batches: {error}"))?;
    for batch in &mut relation_batches {
        run_primary_relation_lane(batch, None, relation_specs)
            .map_err(|error| format!("failed to run relation lane: {error}"))?;
        let decisions = draft_relation_decisions(batch, relation_specs);
        let sidecar = phoenix_rel_post::api::persist_patch_sidecar(
            store,
            batch,
            &decisions,
            benchmark_created_at() + 10,
        )
        .map_err(|error| format!("failed to persist relation sidecar: {error}"))?;
        report.counts.relation_scope_count += 1;
        report.counts.relation_case_count += batch.review_cases.len();
        report.counts.persisted_relation_edge_count += sidecar.edge_additions.len();
    }
    report.relation_stage_us = elapsed_us(relation_started);

    let state_started = Instant::now();
    let mut state_batches = state_schema_api::derive_batches(store, None)
        .map_err(|error| format!("failed to derive state schema batches: {error}"))?;
    for batch in &mut state_batches {
        state_schema_api::run_batch(batch, benchmark_created_at() + 11);
        let sidecar =
            state_schema_api::persist_patch_sidecar(store, batch, benchmark_created_at() + 11)
                .map_err(|error| format!("failed to persist state schema sidecar: {error}"))?;
        report.counts.state_schema_scope_count += 1;
        report.counts.state_schema_slot_family_count += sidecar.slot_families.len();
        report.counts.state_schema_slot_definition_count += sidecar.slot_definitions.len();
        report.counts.state_schema_active_definition_count += sidecar
            .slot_definitions
            .iter()
            .filter(|definition| {
                matches!(
                    definition.lifecycle,
                    phoenix_semantic_v2::StateSlotLifecycle::Active
                        | phoenix_semantic_v2::StateSlotLifecycle::Stable
                )
            })
            .count();
        report.counts.state_schema_candidate_count += sidecar.slot_candidates.len();
        report.counts.state_schema_write_proposal_count += sidecar.write_proposals.len();
    }
    report.state_schema_stage_us = elapsed_us(state_started);

    let memory_started = Instant::now();
    let memory_batches = memory_api::derive_batches(store, None)
        .map_err(|error| format!("failed to derive memory batches: {error}"))?;
    for batch in &memory_batches {
        let sidecar = memory_api::persist_patch_sidecar(store, batch, benchmark_created_at() + 12)
            .map_err(|error| format!("failed to persist memory sidecar: {error}"))?;
        report.counts.memory_scope_count += 1;
        report.counts.memory_state_count += sidecar.states.len();
        report.counts.memory_card_count += sidecar.entity_cards.len();
    }
    report.memory_stage_us = elapsed_us(memory_started);
    report.total_us = elapsed_us(total_started);

    Ok(report)
}

fn run_shared_post_ingest(
    store: &PhoenixOvergraphStore,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Result<PostIngestBenchRun, String> {
    let mut report = PostIngestBenchRun::default();
    let total_started = Instant::now();

    let dirty_started = Instant::now();
    let mut dirty = store
        .list_dirty_scopes()
        .map_err(|error| format!("failed to list dirty scopes: {error}"))?;
    dirty.sort_by(|left, right| left.scope_key.cmp(&right.scope_key));
    report.dirty_list_us = elapsed_us(dirty_started);
    report.counts.dirty_scope_count = dirty.len();

    for scope in &dirty {
        let scope_load_started = Instant::now();
        let runtime = store
            .load_scope_runtime_image(scope, ScopeImageSpec::post_ingest())
            .map_err(|error| format!("failed to load runtime image: {error}"))?;
        report.scope_load_us += elapsed_us(scope_load_started);

        run_shared_scope(&mut report, store, &runtime, relation_specs)?;
    }

    report.total_us = elapsed_us(total_started);
    Ok(report)
}

#[allow(clippy::too_many_arguments)]
fn run_shared_scope(
    report: &mut PostIngestBenchRun,
    store: &PhoenixOvergraphStore,
    runtime: &ScopeRuntimeImage,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Result<(), String> {
    let analysis = ScopeAnalysisContext::from_runtime_image(runtime.clone(), None);
    let relation_sidecar = analysis.runtime.sidecars.relation.as_ref();
    let event_identity_sidecar = analysis.runtime.sidecars.event_identity.as_ref();
    let memory_sidecar = analysis.runtime.sidecars.memory.as_ref();

    let relation_started = Instant::now();
    let mut batch = phoenix_rel_post::api::derive_batch_from_analysis(&analysis, relation_sidecar)
        .map_err(|error| format!("failed to derive relation batch: {error}"))?;
    run_primary_relation_lane(&mut batch, None, relation_specs)
        .map_err(|error| format!("failed to run relation lane: {error}"))?;
    let decisions = draft_relation_decisions(&batch, relation_specs);
    let relation_sidecar = phoenix_rel_post::api::persist_patch_sidecar(
        store,
        &batch,
        &decisions,
        benchmark_created_at() + 20,
    )
    .map_err(|error| format!("failed to persist relation sidecar: {error}"))?;
    report.relation_stage_us += elapsed_us(relation_started);
    report.counts.relation_scope_count += 1;
    report.counts.relation_case_count += batch.review_cases.len();
    report.counts.persisted_relation_edge_count += relation_sidecar.edge_additions.len();

    let state_started = Instant::now();
    let mut state_batch =
        state_schema_api::derive_batch_from_analysis(&analysis, Some(&relation_sidecar));
    state_schema_api::run_batch(&mut state_batch, benchmark_created_at() + 21);
    let state_schema_sidecar =
        state_schema_api::persist_patch_sidecar(store, &state_batch, benchmark_created_at() + 21)
            .map_err(|error| format!("failed to persist state schema sidecar: {error}"))?;
    report.state_schema_stage_us += elapsed_us(state_started);
    report.counts.state_schema_scope_count += 1;
    report.counts.state_schema_slot_family_count += state_schema_sidecar.slot_families.len();
    report.counts.state_schema_slot_definition_count += state_schema_sidecar.slot_definitions.len();
    report.counts.state_schema_active_definition_count += state_schema_sidecar
        .slot_definitions
        .iter()
        .filter(|definition| {
            matches!(
                definition.lifecycle,
                phoenix_semantic_v2::StateSlotLifecycle::Active
                    | phoenix_semantic_v2::StateSlotLifecycle::Stable
            )
        })
        .count();
    report.counts.state_schema_candidate_count += state_schema_sidecar.slot_candidates.len();
    report.counts.state_schema_write_proposal_count += state_schema_sidecar.write_proposals.len();

    let memory_started = Instant::now();
    let memory_batch = memory_api::derive_batch_from_analysis(
        &analysis,
        Some(&relation_sidecar),
        Some(&state_schema_sidecar),
        event_identity_sidecar,
        memory_sidecar,
    );
    let memory_sidecar =
        memory_api::persist_patch_sidecar(store, &memory_batch, benchmark_created_at() + 22)
            .map_err(|error| format!("failed to persist memory sidecar: {error}"))?;
    report.memory_stage_us += elapsed_us(memory_started);
    report.counts.memory_scope_count += 1;
    report.counts.memory_state_count += memory_sidecar.states.len();
    report.counts.memory_card_count += memory_sidecar.entity_cards.len();

    Ok(())
}

fn elapsed_us(started: Instant) -> u64 {
    started.elapsed().as_micros() as u64
}

fn cleanup_seed_store(path: &Path) {
    let _ = fs::remove_dir_all(path);
}

fn extract_chapter_case(text: &str, chapter: usize) -> Result<CaseInput, String> {
    let mut headings = text
        .match_indices("## Chapter ")
        .map(|(offset, _)| offset)
        .collect::<Vec<_>>();
    if headings.is_empty() {
        return Err("no chapter headings found in shortrun.md".to_owned());
    }
    headings.sort_unstable();
    let chapter_index = chapter.saturating_sub(1);
    let Some(&start) = headings.get(chapter_index) else {
        return Err(format!("chapter {chapter} not found"));
    };
    let end = headings
        .get(chapter_index + 1)
        .copied()
        .unwrap_or(text.len());
    let slice = text[start..end].trim().to_owned();
    let title_line = slice
        .lines()
        .next()
        .unwrap_or("Chapter Slice")
        .trim()
        .to_owned();
    Ok(CaseInput {
        case_id: format!("shortrun-chapter-{chapter}"),
        title: title_line,
        text: slice,
    })
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("workspace root")
        .to_path_buf()
}

fn benchmark_created_at() -> i64 {
    1_700_000_000_000
}
