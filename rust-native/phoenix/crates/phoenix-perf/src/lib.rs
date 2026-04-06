use std::alloc::{GlobalAlloc, Layout, System};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use phoenix_runtime::PhoenixRuntime;
use phoenix_types::{
    CommitRequest, CreateSessionRequest, DocumentId, GraphDeltaRequest, IngestDocument,
    IngestRequest, QueryRequest, QueryTarget, RuntimeConfig, ScopeKey, SessionId, TemporalMarker,
};
use serde::{Deserialize, Serialize};

#[global_allocator]
static GLOBAL_ALLOCATOR: TrackingAllocator = TrackingAllocator;

static CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static PHASE_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

const EXCERPT_BYTES: usize = 32 * 1024;
const BYTES_PER_PAGE: usize = 64 * 1024;

pub struct TrackingAllocator;

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            record_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        record_dealloc(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            if new_size >= layout.size() {
                record_alloc(new_size - layout.size());
            } else {
                record_dealloc(layout.size() - new_size);
            }
        }
        new_ptr
    }
}

fn record_alloc(size: usize) {
    if size == 0 {
        return;
    }
    let current = CURRENT_BYTES.fetch_add(size, Ordering::SeqCst) + size;
    update_peak(&PEAK_BYTES, current);
    update_peak(&PHASE_PEAK_BYTES, current);
}

fn record_dealloc(size: usize) {
    if size == 0 {
        return;
    }
    let _ = CURRENT_BYTES.fetch_update(Ordering::SeqCst, Ordering::SeqCst, |current| {
        Some(current.saturating_sub(size))
    });
}

fn update_peak(slot: &AtomicUsize, candidate: usize) {
    let mut observed = slot.load(Ordering::Relaxed);
    while candidate > observed {
        match slot.compare_exchange(observed, candidate, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => break,
            Err(next) => observed = next,
        }
    }
}

fn current_bytes() -> usize {
    CURRENT_BYTES.load(Ordering::SeqCst)
}

fn reset_phase_peak() -> usize {
    let current = current_bytes();
    PHASE_PEAK_BYTES.store(current, Ordering::SeqCst);
    current
}

fn phase_peak_bytes() -> usize {
    PHASE_PEAK_BYTES.load(Ordering::SeqCst)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusDocument {
    pub id: String,
    pub title: String,
    pub path: String,
    pub bytes: usize,
    pub text: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuerySelection {
    pub lexical: Vec<String>,
    pub graph: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseReport {
    pub name: String,
    pub wall_ms: u64,
    pub heap_before_bytes: usize,
    pub heap_after_bytes: usize,
    pub heap_peak_bytes: usize,
    pub heap_peak_delta_bytes: usize,
    pub output_bytes: usize,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryRunReport {
    pub query: String,
    pub chunk_hits: usize,
    pub node_hits: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QueryBatchReport {
    pub phase: PhaseReport,
    pub runs: Vec<QueryRunReport>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkConfig {
    pub iterations: usize,
    pub warmup_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 1,
            warmup_iterations: 0,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkSample {
    pub wall_ms: u64,
    pub heap_peak_delta_bytes: usize,
    pub output_bytes: usize,
    pub chunk_hits: usize,
    pub node_hits: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkSummary {
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub min_wall_ms: u64,
    pub p50_wall_ms: u64,
    pub p95_wall_ms: u64,
    pub max_wall_ms: u64,
    pub mean_wall_ms: f64,
    pub max_peak_delta_bytes: usize,
    pub total_output_bytes: usize,
    pub total_chunk_hits: usize,
    pub total_node_hits: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SteadyStateBenchmarkReport {
    pub name: String,
    pub summary: BenchmarkSummary,
    pub samples: Vec<BenchmarkSample>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusStatsSnapshot {
    pub document_count: usize,
    pub chapter_count: usize,
    pub parent_count: usize,
    pub leaf_count: usize,
    pub entity_count: usize,
    pub discovery_candidate_count: usize,
    pub graph_vertex_count: usize,
    pub graph_edge_count: usize,
    pub span_count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusBudget {
    pub max_analyze_text_ms: u64,
    pub max_excerpt_scan_ms: u64,
    pub max_excerpt_structure_ms: u64,
    pub max_ingest_ms: u64,
    pub max_commit_ms: u64,
    pub max_rebuild_ms: u64,
    pub max_graph_delta_ms: u64,
    pub max_lexical_query_batch_ms: u64,
    pub max_graph_query_batch_ms: u64,
    pub max_snapshot_export_ms: u64,
    pub max_snapshot_import_ms: u64,
    pub max_restore_query_ms: u64,
    pub max_ingest_peak_delta_bytes: usize,
    pub max_graph_delta_peak_delta_bytes: usize,
    pub max_snapshot_import_peak_delta_bytes: usize,
    pub max_snapshot_bytes: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BudgetCheck {
    pub passed: bool,
    pub failures: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CorpusPerfReport {
    pub corpus_id: String,
    pub title: String,
    pub path: String,
    pub input_bytes: usize,
    pub input_chars: usize,
    pub estimated_input_pages: usize,
    pub excerpt_bytes: usize,
    pub selected_queries: QuerySelection,
    pub relation_counts: Vec<phoenix_types::RelationCount>,
    pub ingest_summary: Option<phoenix_types::IngestResult>,
    pub session_stats: Option<CorpusStatsSnapshot>,
    pub phases: Vec<PhaseReport>,
    pub lexical_query_batch: QueryBatchReport,
    pub graph_query_batch: QueryBatchReport,
    pub steady_state: Vec<SteadyStateBenchmarkReport>,
    pub snapshot_bytes: usize,
    pub snapshot_relation_count: usize,
    pub budget: CorpusBudget,
    pub budget_check: BudgetCheck,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PerfSuiteReport {
    pub generated_at: i64,
    pub benchmark_config: BenchmarkConfig,
    pub corpora: Vec<CorpusPerfReport>,
    pub total_failures: usize,
}

pub fn load_standard_corpus() -> Result<Vec<CorpusDocument>, String> {
    let docs_dir = repo_root().join("docs");
    let mut corpus = Vec::new();
    for (id, title, filename) in [
        ("shortrun", "Shortrun", "shortrun.md"),
        ("perfect_run", "Perfect Run", "perfect_run.md"),
    ] {
        let path = docs_dir.join(filename);
        let text = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        corpus.push(CorpusDocument {
            id: id.to_owned(),
            title: title.to_owned(),
            path: path.to_string_lossy().to_string(),
            bytes: text.len(),
            text,
        });
    }
    Ok(corpus)
}

pub fn run_perf_suite() -> Result<PerfSuiteReport, String> {
    run_perf_suite_filtered(None)
}

pub fn run_perf_suite_filtered(corpus_filter: Option<&str>) -> Result<PerfSuiteReport, String> {
    run_perf_suite_filtered_with_config(corpus_filter, &BenchmarkConfig::default())
}

pub fn run_perf_suite_filtered_with_config(
    corpus_filter: Option<&str>,
    benchmark_config: &BenchmarkConfig,
) -> Result<PerfSuiteReport, String> {
    let mut corpora = Vec::new();
    for corpus in load_standard_corpus()? {
        if let Some(filter) = corpus_filter {
            if corpus.id != filter {
                continue;
            }
        }
        corpora.push(run_corpus_suite(&corpus, benchmark_config)?);
    }
    if corpora.is_empty() {
        return Err(match corpus_filter {
            Some(filter) => format!("no corpus matched filter: {filter}"),
            None => "no corpora available".to_owned(),
        });
    }
    let total_failures = corpora
        .iter()
        .map(|report| report.budget_check.failures.len())
        .sum();
    Ok(PerfSuiteReport {
        generated_at: now_ms(),
        benchmark_config: benchmark_config.clone(),
        corpora,
        total_failures,
    })
}

pub fn write_suite_report(
    report: &PerfSuiteReport,
    out_dir: impl AsRef<Path>,
) -> Result<(PathBuf, PathBuf), String> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)
        .map_err(|error| format!("failed to create {}: {error}", out_dir.display()))?;
    let json_path = out_dir.join("latest-native.json");
    let md_path = out_dir.join("latest-native.md");
    let json = serde_json::to_string_pretty(report).map_err(|error| error.to_string())?;
    let markdown = render_markdown(report);
    fs::write(&json_path, json)
        .map_err(|error| format!("failed to write {}: {error}", json_path.display()))?;
    fs::write(&md_path, markdown)
        .map_err(|error| format!("failed to write {}: {error}", md_path.display()))?;
    Ok((json_path, md_path))
}

pub fn render_markdown(report: &PerfSuiteReport) -> String {
    let mut lines = vec![
        "# Phoenix Native Performance Report".to_owned(),
        String::new(),
        format!("Generated at: `{}`", report.generated_at),
        format!(
            "Benchmark iterations: `{}` with `{}` warmup iteration(s)",
            report.benchmark_config.iterations, report.benchmark_config.warmup_iterations
        ),
        format!("Budget failures: `{}`", report.total_failures),
        String::new(),
    ];

    for corpus in &report.corpora {
        lines.push(format!("## {}", corpus.title));
        lines.push(String::new());
        lines.push(format!("- Path: `{}`", corpus.path));
        lines.push(format!("- Input bytes: `{}`", corpus.input_bytes));
        lines.push(format!("- Snapshot bytes: `{}`", corpus.snapshot_bytes));
        lines.push(format!(
            "- Budget status: `{}`",
            if corpus.budget_check.passed {
                "passing"
            } else {
                "failing"
            }
        ));
        for failure in &corpus.budget_check.failures {
            lines.push(format!("- Failure: {}", failure));
        }
        lines.push(String::new());
        lines.push("| Phase | Wall ms | Peak delta MiB | Output bytes |".to_owned());
        lines.push("| --- | ---: | ---: | ---: |".to_owned());
        for phase in &corpus.phases {
            lines.push(format!(
                "| {} | {} | {:.2} | {} |",
                phase.name,
                phase.wall_ms,
                bytes_to_mib(phase.heap_peak_delta_bytes),
                phase.output_bytes
            ));
        }
        lines.push(format!(
            "| lexical_query_batch | {} | {:.2} | {} |",
            corpus.lexical_query_batch.phase.wall_ms,
            bytes_to_mib(corpus.lexical_query_batch.phase.heap_peak_delta_bytes),
            corpus.lexical_query_batch.phase.output_bytes
        ));
        lines.push(format!(
            "| graph_query_batch | {} | {:.2} | {} |",
            corpus.graph_query_batch.phase.wall_ms,
            bytes_to_mib(corpus.graph_query_batch.phase.heap_peak_delta_bytes),
            corpus.graph_query_batch.phase.output_bytes
        ));
        if !corpus.steady_state.is_empty() {
            lines.push(String::new());
            lines.push("| Benchmark | Iter | Min ms | P50 ms | P95 ms | Max ms | Peak delta MiB |".to_owned());
            lines.push("| --- | ---: | ---: | ---: | ---: | ---: | ---: |".to_owned());
            for benchmark in &corpus.steady_state {
                lines.push(format!(
                    "| {} | {} | {} | {} | {} | {} | {:.2} |",
                    benchmark.name,
                    benchmark.summary.iterations,
                    benchmark.summary.min_wall_ms,
                    benchmark.summary.p50_wall_ms,
                    benchmark.summary.p95_wall_ms,
                    benchmark.summary.max_wall_ms,
                    bytes_to_mib(benchmark.summary.max_peak_delta_bytes)
                ));
            }
        }
        lines.push(String::new());
    }

    lines.join("\n")
}

pub fn strict_check(report: &PerfSuiteReport) -> Result<(), String> {
    if report.total_failures == 0 {
        return Ok(());
    }
    let mut lines = vec!["Phoenix performance budgets failed:".to_owned()];
    for corpus in &report.corpora {
        for failure in &corpus.budget_check.failures {
            lines.push(format!("{}: {}", corpus.corpus_id, failure));
        }
    }
    Err(lines.join("\n"))
}

fn run_corpus_suite(
    corpus: &CorpusDocument,
    benchmark_config: &BenchmarkConfig,
) -> Result<CorpusPerfReport, String> {
    let runtime =
        PhoenixRuntime::new(RuntimeConfig::default()).map_err(|error| error.to_string())?;
    let mut phases = Vec::new();

    let (_, init_phase) = measure_phase("init_runtime", || {
        runtime.init().map_err(|error| error.to_string())
    })?;
    phases.push(init_phase);

    let (session, create_session_phase) = measure_phase("create_session", || {
        runtime
            .create_session(CreateSessionRequest {
                session_id: Some(SessionId(format!("perf-{}", corpus.id))),
                label: format!("Phoenix Perf {}", corpus.title),
                scope: ScopeKey::default(),
            })
            .map_err(|error| error.to_string())
    })?;
    phases.push(create_session_phase);

    let excerpt = excerpt_text(&corpus.text, EXCERPT_BYTES);
    let (analytics, analyze_phase) = measure_phase("analyze_text", || {
        Ok::<_, String>(runtime.analyze_text(&corpus.text))
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&analytics)
            .map_err(|error| error.to_string())?
            .len(),
        ..analyze_phase
    });

    let (scan, excerpt_scan_phase) = measure_phase("excerpt_scan", || {
        Ok::<_, String>(runtime.scan_text(phoenix_types::ScanRequest {
            text: excerpt.clone(),
            scope: ScopeKey::default(),
            session_id: Some(session.session_id.clone()),
            resolver_seed: Vec::new(),
        }))
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&scan)
            .map_err(|error| error.to_string())?
            .len(),
        ..excerpt_scan_phase
    });

    let (structure, excerpt_structure_phase) = measure_phase("excerpt_structure", || {
        Ok::<_, String>(runtime.build_structure(phoenix_types::StructureRequest {
            text: excerpt.clone(),
            scan: scan.clone(),
        }))
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&structure)
            .map_err(|error| error.to_string())?
            .len(),
        ..excerpt_structure_phase
    });

    let doc_id = DocumentId(format!("perf-doc-{}", corpus.id));
    let (ingest, ingest_phase) = measure_phase("ingest_document", || {
        runtime
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![IngestDocument {
                    document_id: doc_id.clone(),
                    note_id: None,
                    title: corpus.title.clone(),
                    text: corpus.text.clone(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .map_err(|error| error.to_string())
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&ingest)
            .map_err(|error| error.to_string())?
            .len(),
        ..ingest_phase
    });

    let (commit, commit_phase) = measure_phase("commit_session", || {
        runtime
            .commit(CommitRequest {
                session_id: session.session_id.clone(),
                reason: Some("perf-suite".to_owned()),
            })
            .map_err(|error| error.to_string())
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&commit)
            .map_err(|error| error.to_string())?
            .len(),
        ..commit_phase
    });

    let (rebuild, rebuild_phase) = measure_phase("rebuild_lex", || {
        runtime
            .rebuild(phoenix_types::RebuildRequest {
                session_id: Some(session.session_id.clone()),
                reason: Some("perf-suite".to_owned()),
            })
            .map_err(|error| error.to_string())
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&rebuild)
            .map_err(|error| error.to_string())?
            .len(),
        ..rebuild_phase
    });

    let (state, session_state_phase) = measure_phase("session_state", || {
        runtime
            .session_state(&session.session_id)
            .map_err(|error| error.to_string())
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&state)
            .map_err(|error| error.to_string())?
            .len(),
        ..session_state_phase
    });

    let (stats, session_stats_phase) = measure_phase("session_stats", || {
        runtime
            .session_stats(&session.session_id)
            .map_err(|error| error.to_string())
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&stats)
            .map_err(|error| error.to_string())?
            .len(),
        ..session_stats_phase
    });

    let (graph_delta, graph_delta_phase) = measure_phase("graph_delta", || {
        runtime
            .graph_delta(GraphDeltaRequest {
                session_id: session.session_id.clone(),
                scope: ScopeKey::default(),
                changed_documents: vec![doc_id.clone()],
                limit: None,
                since_commit: Some(commit.commit_id.clone()),
                include_candidate_graph: false,
            })
            .map_err(|error| error.to_string())
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&graph_delta)
            .map_err(|error| error.to_string())?
            .len(),
        ..graph_delta_phase
    });

    let selected_queries = select_queries(&corpus.text);
    let lexical_query_batch = run_query_batch(
        &runtime,
        &session.session_id,
        &selected_queries.lexical,
        vec![QueryTarget::Chunks],
    )?;
    let graph_query_batch = run_query_batch(
        &runtime,
        &session.session_id,
        &selected_queries.graph,
        vec![QueryTarget::Graph, QueryTarget::Nodes],
    )?;

    let (snapshot_bytes, snapshot_export_phase) = measure_phase("snapshot_export", || {
        runtime.export_snapshot().map_err(|error| error.to_string())
    })?;
    phases.push(PhaseReport {
        output_bytes: snapshot_bytes.len(),
        ..snapshot_export_phase
    });

    let restore_runtime =
        PhoenixRuntime::new(RuntimeConfig::default()).map_err(|error| error.to_string())?;
    let _ = restore_runtime.init().map_err(|error| error.to_string())?;
    let (_, snapshot_import_phase) = measure_phase("snapshot_import", || {
        restore_runtime
            .import_snapshot(&snapshot_bytes)
            .map_err(|error| error.to_string())
    })?;
    phases.push(PhaseReport {
        output_bytes: snapshot_bytes.len(),
        ..snapshot_import_phase
    });

    let (restore_query, restore_query_phase) = measure_phase("restore_query", || {
        restore_runtime
            .query(QueryRequest {
                session_id: Some(session.session_id.clone()),
                query: selected_queries
                    .lexical
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "Ryan".to_owned()),
                scope: ScopeKey::default(),
                targets: vec![QueryTarget::Chunks],
                limit: Some(5),
                temporal: None::<TemporalMarker>,
                semantic_query_vector: None,
                include_candidate_graph: false,
            })
            .map_err(|error| error.to_string())
    })?;
    phases.push(PhaseReport {
        output_bytes: serde_json::to_vec(&restore_query)
            .map_err(|error| error.to_string())?
            .len(),
        ..restore_query_phase
    });

    let steady_state = run_steady_state_benchmarks(
        &runtime,
        &restore_runtime,
        benchmark_config,
        &session.session_id,
        &doc_id,
        &commit.commit_id,
        &excerpt,
        &scan,
        &selected_queries,
    )?;

    let relation_counts = runtime.relation_counts().map_err(|error| error.to_string())?;
    let session_stats = CorpusStatsSnapshot {
        document_count: stats.document_count,
        chapter_count: stats.chapter_count,
        parent_count: stats.parent_count,
        leaf_count: stats.leaf_count,
        entity_count: stats.entity_count,
        discovery_candidate_count: stats.discovery_candidate_count,
        graph_vertex_count: stats.graph_vertex_count,
        graph_edge_count: stats.graph_edge_count,
        span_count: stats.span_count,
    };
    let budget = default_budget_for(corpus);
    let budget_check = check_budget(
        &budget,
        &phases,
        &lexical_query_batch.phase,
        &graph_query_batch.phase,
        snapshot_bytes.len(),
    );

    Ok(CorpusPerfReport {
        corpus_id: corpus.id.clone(),
        title: corpus.title.clone(),
        path: corpus.path.clone(),
        input_bytes: corpus.bytes,
        input_chars: corpus.text.chars().count(),
        estimated_input_pages: corpus.bytes.div_ceil(BYTES_PER_PAGE),
        excerpt_bytes: excerpt.len(),
        selected_queries,
        relation_counts,
        ingest_summary: Some(ingest),
        session_stats: Some(session_stats),
        phases,
        lexical_query_batch,
        graph_query_batch,
        steady_state,
        snapshot_bytes: snapshot_bytes.len(),
        snapshot_relation_count: runtime
            .relation_name_count()
            .map_err(|error| error.to_string())?,
        budget,
        budget_check,
    })
}

#[derive(Clone, Copy, Debug, Default)]
struct BenchmarkObservation {
    output_bytes: usize,
    chunk_hits: usize,
    node_hits: usize,
}

fn run_steady_state_benchmarks(
    runtime: &PhoenixRuntime,
    restore_runtime: &PhoenixRuntime,
    benchmark_config: &BenchmarkConfig,
    session_id: &SessionId,
    doc_id: &DocumentId,
    commit_id: &phoenix_types::CommitId,
    excerpt: &str,
    scan: &phoenix_types::ScanArtifact,
    selected_queries: &QuerySelection,
) -> Result<Vec<SteadyStateBenchmarkReport>, String> {
    let lexical_targets = vec![QueryTarget::Chunks];
    let graph_targets = vec![QueryTarget::Graph, QueryTarget::Nodes];
    let restore_query = selected_queries
        .lexical
        .first()
        .cloned()
        .unwrap_or_else(|| "Ryan".to_owned());

    let mut benchmarks = Vec::new();

    benchmarks.push(run_benchmark(
        "scan_excerpt_steady",
        benchmark_config,
        || {
            let result = runtime.scan_text(phoenix_types::ScanRequest {
                text: excerpt.to_owned(),
                scope: ScopeKey::default(),
                session_id: Some(session_id.clone()),
                resolver_seed: Vec::new(),
            });
            Ok(BenchmarkObservation {
                output_bytes: serde_json::to_vec(&result)
                    .map_err(|error| error.to_string())?
                    .len(),
                ..BenchmarkObservation::default()
            })
        },
    )?);

    benchmarks.push(run_benchmark(
        "structure_excerpt_steady",
        benchmark_config,
        || {
            let result = runtime.build_structure(phoenix_types::StructureRequest {
                text: excerpt.to_owned(),
                scan: scan.clone(),
            });
            Ok(BenchmarkObservation {
                output_bytes: serde_json::to_vec(&result)
                    .map_err(|error| error.to_string())?
                    .len(),
                ..BenchmarkObservation::default()
            })
        },
    )?);

    benchmarks.push(run_benchmark(
        "session_stats_steady",
        benchmark_config,
        || {
            let result = runtime
                .session_stats(session_id)
                .map_err(|error| error.to_string())?;
            Ok(BenchmarkObservation {
                output_bytes: serde_json::to_vec(&result)
                    .map_err(|error| error.to_string())?
                    .len(),
                ..BenchmarkObservation::default()
            })
        },
    )?);

    benchmarks.push(run_benchmark(
        "graph_delta_steady",
        benchmark_config,
        || {
            let result = runtime
                .graph_delta(GraphDeltaRequest {
                    session_id: session_id.clone(),
                    scope: ScopeKey::default(),
                    changed_documents: vec![doc_id.clone()],
                    limit: None,
                    since_commit: Some(commit_id.clone()),
                    include_candidate_graph: false,
                })
                .map_err(|error| error.to_string())?;
            Ok(BenchmarkObservation {
                output_bytes: serde_json::to_vec(&result)
                    .map_err(|error| error.to_string())?
                    .len(),
                chunk_hits: result.chunks.len(),
                node_hits: result.nodes.len(),
            })
        },
    )?);

    benchmarks.push(run_benchmark(
        "lexical_query_steady",
        benchmark_config,
        || run_query_workload(runtime, session_id, &selected_queries.lexical, &lexical_targets),
    )?);

    benchmarks.push(run_benchmark(
        "graph_query_steady",
        benchmark_config,
        || run_query_workload(runtime, session_id, &selected_queries.graph, &graph_targets),
    )?);

    benchmarks.push(run_benchmark(
        "restore_query_steady",
        benchmark_config,
        || {
            let result = restore_runtime
                .query(QueryRequest {
                    session_id: Some(session_id.clone()),
                    query: restore_query.clone(),
                    scope: ScopeKey::default(),
                    targets: lexical_targets.clone(),
                    limit: Some(5),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                })
                .map_err(|error| error.to_string())?;
            Ok(BenchmarkObservation {
                output_bytes: serde_json::to_vec(&result)
                    .map_err(|error| error.to_string())?
                    .len(),
                chunk_hits: result.chunk_hits.len(),
                node_hits: result.node_hits.len(),
            })
        },
    )?);

    Ok(benchmarks)
}

fn run_query_workload(
    runtime: &PhoenixRuntime,
    session_id: &SessionId,
    queries: &[String],
    targets: &[QueryTarget],
) -> Result<BenchmarkObservation, String> {
    let mut output_bytes = 0;
    let mut chunk_hits = 0;
    let mut node_hits = 0;
    for query in queries {
        let result = runtime
            .query(QueryRequest {
                session_id: Some(session_id.clone()),
                query: query.clone(),
                scope: ScopeKey::default(),
                targets: targets.to_vec(),
                limit: Some(8),
                temporal: None,
                semantic_query_vector: None,
                include_candidate_graph: false,
            })
            .map_err(|error| error.to_string())?;
        output_bytes += serde_json::to_vec(&result)
            .map_err(|error| error.to_string())?
            .len();
        chunk_hits += result.chunk_hits.len();
        node_hits += result.node_hits.len();
    }
    Ok(BenchmarkObservation {
        output_bytes,
        chunk_hits,
        node_hits,
    })
}

fn run_benchmark(
    name: &str,
    config: &BenchmarkConfig,
    mut op: impl FnMut() -> Result<BenchmarkObservation, String>,
) -> Result<SteadyStateBenchmarkReport, String> {
    let iterations = config.iterations.max(1);
    for _ in 0..config.warmup_iterations {
        let _ = op()?;
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let (observation, phase) = measure_phase(name, &mut op)?;
        samples.push(BenchmarkSample {
            wall_ms: phase.wall_ms,
            heap_peak_delta_bytes: phase.heap_peak_delta_bytes,
            output_bytes: observation.output_bytes,
            chunk_hits: observation.chunk_hits,
            node_hits: observation.node_hits,
        });
    }
    Ok(SteadyStateBenchmarkReport {
        name: name.to_owned(),
        summary: summarize_benchmark(config, &samples),
        samples,
        note: None,
    })
}

fn summarize_benchmark(config: &BenchmarkConfig, samples: &[BenchmarkSample]) -> BenchmarkSummary {
    if samples.is_empty() {
        return BenchmarkSummary::default();
    }
    let mut walls = samples.iter().map(|sample| sample.wall_ms).collect::<Vec<_>>();
    walls.sort_unstable();
    let total_wall = walls.iter().copied().sum::<u64>();
    BenchmarkSummary {
        iterations: samples.len(),
        warmup_iterations: config.warmup_iterations,
        min_wall_ms: *walls.first().unwrap_or(&0),
        p50_wall_ms: percentile(&walls, 0.50),
        p95_wall_ms: percentile(&walls, 0.95),
        max_wall_ms: *walls.last().unwrap_or(&0),
        mean_wall_ms: total_wall as f64 / samples.len() as f64,
        max_peak_delta_bytes: samples
            .iter()
            .map(|sample| sample.heap_peak_delta_bytes)
            .max()
            .unwrap_or_default(),
        total_output_bytes: samples.iter().map(|sample| sample.output_bytes).sum(),
        total_chunk_hits: samples.iter().map(|sample| sample.chunk_hits).sum(),
        total_node_hits: samples.iter().map(|sample| sample.node_hits).sum(),
    }
}

fn percentile(values: &[u64], percentile: f64) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let rank = ((values.len() - 1) as f64 * percentile).round() as usize;
    values[rank.min(values.len() - 1)]
}

fn default_budget_for(corpus: &CorpusDocument) -> CorpusBudget {
    match corpus.id.as_str() {
        "shortrun" => CorpusBudget {
            max_analyze_text_ms: 2_000,
            max_excerpt_scan_ms: 1_500,
            max_excerpt_structure_ms: 1_500,
            max_ingest_ms: 25_000,
            max_commit_ms: 1_500,
            max_rebuild_ms: 5_000,
            max_graph_delta_ms: 4_000,
            max_lexical_query_batch_ms: 2_000,
            max_graph_query_batch_ms: 4_000,
            max_snapshot_export_ms: 4_000,
            max_snapshot_import_ms: 6_000,
            max_restore_query_ms: 1_500,
            max_ingest_peak_delta_bytes: 256 * 1024 * 1024,
            max_graph_delta_peak_delta_bytes: 64 * 1024 * 1024,
            max_snapshot_import_peak_delta_bytes: 256 * 1024 * 1024,
            max_snapshot_bytes: 64 * 1024 * 1024,
        },
        _ => CorpusBudget {
            max_analyze_text_ms: 10_000,
            max_excerpt_scan_ms: 3_000,
            max_excerpt_structure_ms: 3_000,
            max_ingest_ms: 180_000,
            max_commit_ms: 3_000,
            max_rebuild_ms: 20_000,
            max_graph_delta_ms: 20_000,
            max_lexical_query_batch_ms: 6_000,
            max_graph_query_batch_ms: 20_000,
            max_snapshot_export_ms: 20_000,
            max_snapshot_import_ms: 40_000,
            max_restore_query_ms: 4_000,
            max_ingest_peak_delta_bytes: 1024 * 1024 * 1024,
            max_graph_delta_peak_delta_bytes: 256 * 1024 * 1024,
            max_snapshot_import_peak_delta_bytes: 1024 * 1024 * 1024,
            max_snapshot_bytes: 256 * 1024 * 1024,
        },
    }
}

fn check_budget(
    budget: &CorpusBudget,
    phases: &[PhaseReport],
    lexical_phase: &PhaseReport,
    graph_phase: &PhaseReport,
    snapshot_bytes: usize,
) -> BudgetCheck {
    let mut failures = Vec::new();

    let find = |name: &str| phases.iter().find(|phase| phase.name == name);

    check_phase_wall(
        &mut failures,
        "analyze_text",
        find("analyze_text"),
        budget.max_analyze_text_ms,
    );
    check_phase_wall(
        &mut failures,
        "excerpt_scan",
        find("excerpt_scan"),
        budget.max_excerpt_scan_ms,
    );
    check_phase_wall(
        &mut failures,
        "excerpt_structure",
        find("excerpt_structure"),
        budget.max_excerpt_structure_ms,
    );
    check_phase_wall(
        &mut failures,
        "ingest_document",
        find("ingest_document"),
        budget.max_ingest_ms,
    );
    check_phase_wall(
        &mut failures,
        "commit_session",
        find("commit_session"),
        budget.max_commit_ms,
    );
    check_phase_wall(
        &mut failures,
        "rebuild_lex",
        find("rebuild_lex"),
        budget.max_rebuild_ms,
    );
    check_phase_wall(
        &mut failures,
        "graph_delta",
        find("graph_delta"),
        budget.max_graph_delta_ms,
    );
    check_phase_wall(
        &mut failures,
        "snapshot_export",
        find("snapshot_export"),
        budget.max_snapshot_export_ms,
    );
    check_phase_wall(
        &mut failures,
        "snapshot_import",
        find("snapshot_import"),
        budget.max_snapshot_import_ms,
    );
    check_phase_wall(
        &mut failures,
        "restore_query",
        find("restore_query"),
        budget.max_restore_query_ms,
    );

    if lexical_phase.wall_ms > budget.max_lexical_query_batch_ms {
        failures.push(format!(
            "lexical_query_batch exceeded budget: {}ms > {}ms",
            lexical_phase.wall_ms, budget.max_lexical_query_batch_ms
        ));
    }
    if graph_phase.wall_ms > budget.max_graph_query_batch_ms {
        failures.push(format!(
            "graph_query_batch exceeded budget: {}ms > {}ms",
            graph_phase.wall_ms, budget.max_graph_query_batch_ms
        ));
    }

    check_peak_delta(
        &mut failures,
        "ingest_document",
        find("ingest_document"),
        budget.max_ingest_peak_delta_bytes,
    );
    check_peak_delta(
        &mut failures,
        "graph_delta",
        find("graph_delta"),
        budget.max_graph_delta_peak_delta_bytes,
    );
    check_peak_delta(
        &mut failures,
        "snapshot_import",
        find("snapshot_import"),
        budget.max_snapshot_import_peak_delta_bytes,
    );

    if snapshot_bytes > budget.max_snapshot_bytes {
        failures.push(format!(
            "snapshot exceeded budget: {} bytes > {} bytes",
            snapshot_bytes, budget.max_snapshot_bytes
        ));
    }

    BudgetCheck {
        passed: failures.is_empty(),
        failures,
    }
}

fn check_phase_wall(
    failures: &mut Vec<String>,
    name: &str,
    phase: Option<&PhaseReport>,
    max_ms: u64,
) {
    if let Some(phase) = phase {
        if phase.wall_ms > max_ms {
            failures.push(format!(
                "{name} exceeded budget: {}ms > {}ms",
                phase.wall_ms, max_ms
            ));
        }
    }
}

fn check_peak_delta(
    failures: &mut Vec<String>,
    name: &str,
    phase: Option<&PhaseReport>,
    max_bytes: usize,
) {
    if let Some(phase) = phase {
        if phase.heap_peak_delta_bytes > max_bytes {
            failures.push(format!(
                "{name} peak delta exceeded budget: {} bytes > {} bytes",
                phase.heap_peak_delta_bytes, max_bytes
            ));
        }
    }
}

fn run_query_batch(
    runtime: &PhoenixRuntime,
    session_id: &SessionId,
    queries: &[String],
    targets: Vec<QueryTarget>,
) -> Result<QueryBatchReport, String> {
    let phase_name = if targets
        .iter()
        .any(|target| matches!(target, QueryTarget::Graph | QueryTarget::Nodes))
    {
        "graph_query_batch"
    } else {
        "lexical_query_batch"
    };

    let (runs, phase) = measure_phase(phase_name, || {
        let mut runs = Vec::new();
        for query in queries {
            let result = runtime
                .query(QueryRequest {
                    session_id: Some(session_id.clone()),
                    query: query.clone(),
                    scope: ScopeKey::default(),
                    targets: targets.clone(),
                    limit: Some(8),
                    temporal: None,
                    semantic_query_vector: None,
                    include_candidate_graph: false,
                })
                .map_err(|error| error.to_string())?;
            runs.push(QueryRunReport {
                query: query.clone(),
                chunk_hits: result.chunk_hits.len(),
                node_hits: result.node_hits.len(),
            });
        }
        Ok::<_, String>(runs)
    })?;

    Ok(QueryBatchReport {
        phase: PhaseReport {
            output_bytes: serde_json::to_vec(&runs)
                .map_err(|error| error.to_string())?
                .len(),
            ..phase
        },
        runs,
    })
}

fn measure_phase<T>(
    name: &str,
    op: impl FnOnce() -> Result<T, String>,
) -> Result<(T, PhaseReport), String> {
    if perf_progress_enabled() {
        eprintln!("[phoenix-perf] starting phase: {name}");
    }
    let heap_before = reset_phase_peak();
    let started = Instant::now();
    let value = op()?;
    let wall_ms = started.elapsed().as_millis() as u64;
    let heap_after = current_bytes();
    let heap_peak = phase_peak_bytes();
    let phase = PhaseReport {
        name: name.to_owned(),
        wall_ms,
        heap_before_bytes: heap_before,
        heap_after_bytes: heap_after,
        heap_peak_bytes: heap_peak,
        heap_peak_delta_bytes: heap_peak.saturating_sub(heap_before),
        output_bytes: 0,
        note: None,
    };
    if perf_progress_enabled() {
        eprintln!(
            "[phoenix-perf] finished phase: {} in {}ms (peak delta {:.2} MiB)",
            phase.name,
            phase.wall_ms,
            bytes_to_mib(phase.heap_peak_delta_bytes)
        );
    }
    Ok((
        value,
        phase,
    ))
}

fn excerpt_text(text: &str, target_bytes: usize) -> String {
    let mut end = target_bytes.min(text.len());
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_owned()
}

fn select_queries(text: &str) -> QuerySelection {
    let lower = text.to_ascii_lowercase();
    let mut names = Vec::new();
    for candidate in ["Ryan", "Len", "Ghoul", "Augusti", "Zanbato", "Wyvern"] {
        if lower.contains(&candidate.to_ascii_lowercase()) {
            names.push(candidate.to_owned());
        }
    }
    if names.is_empty() {
        names.push("Ryan".to_owned());
    }

    let mut lexical = vec![names[0].clone()];
    if lower.contains("new rome") {
        lexical.push("\"New Rome\"".to_owned());
    }
    if names.len() > 1 {
        lexical.push(names[1].clone());
    }
    lexical.sort();
    lexical.dedup();

    let mut graph = Vec::new();
    if names.len() >= 2 {
        graph.push(format!("{} {}", names[0], names[1]));
    } else {
        graph.push(names[0].clone());
    }
    if names.len() >= 3 {
        graph.push(format!("{} {}", names[0], names[2]));
    } else if lower.contains("meta-gang") {
        graph.push("\"Meta-Gang\"".to_owned());
    } else if lower.contains("new rome") {
        graph.push("\"New Rome\"".to_owned());
    }
    graph.sort();
    graph.dedup();

    QuerySelection { lexical, graph }
}

fn bytes_to_mib(bytes: usize) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn now_ms() -> i64 {
    #[cfg(target_arch = "wasm32")]
    {
        0
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_millis() as i64
    }
}

pub fn default_out_dir() -> PathBuf {
    phoenix_runtime::workspace_root()
        .join("reports")
        .join("perf")
}

fn repo_root() -> PathBuf {
    phoenix_runtime::workspace_root()
        .join("..")
        .join("..")
        .canonicalize()
        .expect("repo root")
}

fn perf_progress_enabled() -> bool {
    matches!(
        env::var("PHOENIX_PERF_PROGRESS").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_corpus_files_exist() {
        let corpus = load_standard_corpus().expect("corpus");
        assert_eq!(corpus.len(), 2);
        assert!(corpus.iter().all(|doc| !doc.text.is_empty()));
    }

    #[test]
    fn query_selection_prefers_real_corpus_terms() {
        let selection = select_queries("Ryan met Len in New Rome.");
        assert!(selection.lexical.iter().any(|query| query.contains("Ryan")));
        assert!(selection.graph.iter().any(|query| query.contains("Ryan")));
    }

    #[test]
    fn markdown_report_renders_sections() {
        let markdown = render_markdown(&PerfSuiteReport {
            generated_at: 1,
            benchmark_config: BenchmarkConfig::default(),
            corpora: vec![CorpusPerfReport {
                corpus_id: "shortrun".to_owned(),
                title: "Shortrun".to_owned(),
                path: "docs/shortrun.md".to_owned(),
                input_bytes: 10,
                input_chars: 10,
                estimated_input_pages: 1,
                excerpt_bytes: 10,
                selected_queries: QuerySelection::default(),
                relation_counts: Vec::new(),
                ingest_summary: None,
                session_stats: None,
                phases: vec![PhaseReport {
                    name: "ingest_document".to_owned(),
                    wall_ms: 10,
                    heap_before_bytes: 0,
                    heap_after_bytes: 0,
                    heap_peak_bytes: 0,
                    heap_peak_delta_bytes: 0,
                    output_bytes: 10,
                    note: None,
                }],
                lexical_query_batch: QueryBatchReport::default(),
                graph_query_batch: QueryBatchReport::default(),
                steady_state: vec![SteadyStateBenchmarkReport {
                    name: "graph_query_steady".to_owned(),
                    summary: BenchmarkSummary {
                        iterations: 5,
                        warmup_iterations: 1,
                        p50_wall_ms: 8,
                        ..BenchmarkSummary::default()
                    },
                    samples: Vec::new(),
                    note: None,
                }],
                snapshot_bytes: 10,
                snapshot_relation_count: 1,
                budget: CorpusBudget::default(),
                budget_check: BudgetCheck::default(),
            }],
            total_failures: 0,
        });
        assert!(markdown.contains("Phoenix Native Performance Report"));
        assert!(markdown.contains("Shortrun"));
        assert!(markdown.contains("graph_query_steady"));
        assert!(markdown.contains("Benchmark iterations"));
    }

    #[test]
    fn benchmark_summary_reports_percentiles() {
        let config = BenchmarkConfig {
            iterations: 5,
            warmup_iterations: 1,
        };
        let summary = summarize_benchmark(
            &config,
            &[
                BenchmarkSample {
                    wall_ms: 9,
                    ..BenchmarkSample::default()
                },
                BenchmarkSample {
                    wall_ms: 5,
                    ..BenchmarkSample::default()
                },
                BenchmarkSample {
                    wall_ms: 7,
                    ..BenchmarkSample::default()
                },
                BenchmarkSample {
                    wall_ms: 12,
                    ..BenchmarkSample::default()
                },
                BenchmarkSample {
                    wall_ms: 6,
                    ..BenchmarkSample::default()
                },
            ],
        );
        assert_eq!(summary.iterations, 5);
        assert_eq!(summary.warmup_iterations, 1);
        assert_eq!(summary.min_wall_ms, 5);
        assert_eq!(summary.p50_wall_ms, 7);
        assert_eq!(summary.p95_wall_ms, 12);
        assert_eq!(summary.max_wall_ms, 12);
    }

    #[test]
    #[ignore = "native deterministic perf smoke benchmark over perfect_run.md"]
    fn perfect_run_native_perf_smoke_benchmark() {
        let config = BenchmarkConfig::default();
        let report =
            run_perf_suite_filtered_with_config(Some("perfect_run"), &config).expect("perf report");
        let (json_path, md_path) =
            write_suite_report(&report, default_out_dir()).expect("write perf report");
        println!("{}", serde_json::to_string_pretty(&report).expect("json"));
        println!("JSON report: {}", json_path.display());
        println!("Markdown report: {}", md_path.display());
        let corpus = report
            .corpora
            .iter()
            .find(|corpus| corpus.corpus_id == "perfect_run")
            .expect("perfect_run corpus");
        assert!(corpus.budget_check.passed, "{:?}", corpus.budget_check.failures);
    }
}
