use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use phoenix_er_post::{
    compute_lexical_metrics, compute_retrieval_comparison, decision_kind_name,
    default_embedding_model_root, derive_dirty_scope_review_batches,
    derive_dirty_scope_review_batches_with_replay, draft_review_decisions,
    generate_embedding_candidates, generate_fused_candidates, generate_lexical_candidates,
    persist_er_patch_sidecar, summarize_review_cases, ErCaseSmokeSummary, ErDecision,
    ErEmbeddingCandidateSummary, ErEmbeddingConfig, ErEmbeddingModel, ErFusedCandidateSummary,
    ErLexicalMetrics, ErRetrievalComparison, TextEmbeddingProfile,
};
use phoenix_ingest_overgraph::PhoenixInvarantV3;
use phoenix_kernel::KernelGraphSnapshot;
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixGraphKernelStoreV2,
};
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::{DocumentId, IngestDocument, ScopeKey, SessionId};
use serde::Serialize;

#[derive(Debug, Clone)]
struct SmokeConfig {
    store_path: Option<PathBuf>,
    corpus_id: Option<String>,
    session_id: Option<SessionId>,
    candidate_limit: usize,
    embedding_limit: usize,
    case_limit: usize,
    json: bool,
    embed: bool,
    embedding_model_root: Option<PathBuf>,
    embedding_profile: Option<TextEmbeddingProfile>,
    embedding_max_length: Option<usize>,
    persist_patches: bool,
    replay_patches: bool,
    keep_store: bool,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        Self {
            store_path: None,
            corpus_id: Some("shortrun".to_owned()),
            session_id: None,
            candidate_limit: 5,
            embedding_limit: 5,
            case_limit: 32,
            json: false,
            embed: false,
            embedding_model_root: None,
            embedding_profile: None,
            embedding_max_length: None,
            persist_patches: false,
            replay_patches: false,
            keep_store: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CorpusDocument {
    id: String,
    title: String,
    path: String,
    bytes: usize,
    text: String,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct SeedReport {
    seeded: bool,
    corpus_id: Option<String>,
    store_path: Option<String>,
    dirty_scope_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ScopeSmokeReport {
    scope_key: String,
    session_id: Option<String>,
    document_ref_count: usize,
    review_case_count: usize,
    entity_profile_count: usize,
    lexical_summary: phoenix_er_post::ErLexicalCandidateSummary,
    embedding_summary: Option<ErEmbeddingCandidateSummary>,
    fused_summary: ErFusedCandidateSummary,
    metrics: ErLexicalMetrics,
    retrieval: ErRetrievalComparison,
    decision_counts: BTreeMap<String, usize>,
    decisions: Vec<ErDecision>,
    case_summaries: Vec<ErCaseSmokeSummary>,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct AggregateMetrics {
    scope_count: usize,
    case_count: usize,
    cases_with_candidates: usize,
    total_candidate_count: usize,
    average_candidates_per_case: f32,
    embedding_cases_with_candidates: usize,
    embedding_total_candidate_count: usize,
    embedding_average_candidates_per_case: f32,
    fused_cases_with_candidates: usize,
    fused_total_candidate_count: usize,
    fused_average_candidates_per_case: f32,
    additional_cases_covered: isize,
    additional_candidates: isize,
    alias_case_count: usize,
    alias_exact_hit_count: usize,
    alias_exact_hit_rate: f32,
    type_disagreement_case_count: usize,
    type_disagreement_rescue_count: usize,
    type_disagreement_rescue_rate: f32,
    unresolved_case_count: usize,
    ambiguous_case_count: usize,
    unresolved_or_ambiguous_case_count: usize,
    unresolved_or_ambiguous_covered_count: usize,
    unresolved_or_ambiguous_coverage_rate: f32,
    decision_counts: BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    store_path: String,
    session_id: Option<String>,
    seed: SeedReport,
    aggregate: AggregateMetrics,
    scopes: Vec<ScopeSmokeReport>,
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let config = parse_config(&args);
    let temp_store = config.store_path.is_none();

    let report = match run_smoke(&config) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    if config.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize smoke report")
        );
    } else {
        println!("{}", render_markdown(&report, config.case_limit));
    }

    if temp_store && !config.keep_store {
        let _ = fs::remove_dir_all(&report.store_path);
    }
}

fn parse_config(args: &[String]) -> SmokeConfig {
    let mut config = SmokeConfig::default();
    config.store_path = parse_path_arg(args, "--store-path");
    config.corpus_id = parse_string_arg(args, "--corpus").or_else(|| {
        config
            .store_path
            .is_none()
            .then(|| SmokeConfig::default().corpus_id.unwrap())
    });
    config.session_id = parse_string_arg(args, "--session-id").map(SessionId);
    config.candidate_limit = parse_usize_arg(args, "--limit")
        .unwrap_or(config.candidate_limit)
        .max(1);
    config.embedding_limit = parse_usize_arg(args, "--embedding-limit")
        .unwrap_or(config.embedding_limit)
        .max(1);
    config.case_limit = parse_usize_arg(args, "--case-limit")
        .unwrap_or(config.case_limit)
        .max(1);
    config.json = args.iter().any(|arg| arg == "--json");
    config.embed = args.iter().any(|arg| arg == "--embed");
    config.embedding_model_root = parse_path_arg(args, "--embedding-model-root");
    config.embedding_profile = parse_string_arg(args, "--embedding-profile")
        .and_then(|value| TextEmbeddingProfile::parse(&value));
    config.embedding_max_length = parse_usize_arg(args, "--embedding-max-length");
    config.persist_patches = args.iter().any(|arg| arg == "--persist-patches");
    config.replay_patches = args.iter().any(|arg| arg == "--replay-patches");
    config.keep_store = args.iter().any(|arg| arg == "--keep-store");
    config
}

fn run_smoke(config: &SmokeConfig) -> Result<SmokeReport, String> {
    let mut seed = SeedReport::default();
    let store_path = if let Some(path) = config.store_path.clone() {
        path
    } else {
        let corpus = select_corpus(config.corpus_id.as_deref())?;
        let path = unique_smoke_store_path(&corpus.id);
        seed = seed_store(&path, &corpus)?;
        path
    };

    let store = PhoenixOvergraphStore::open(&store_path).map_err(|error| error.to_string())?;
    if config.persist_patches || config.replay_patches {
        store
            .init_er_patch_schema()
            .map_err(|error| error.to_string())?;
    }
    let embedding_config = config.embed.then(|| ErEmbeddingConfig {
        model_root: config
            .embedding_model_root
            .clone()
            .unwrap_or_else(default_embedding_model_root),
        max_length: config.embedding_max_length.unwrap_or(512),
        profile: config
            .embedding_profile
            .unwrap_or(TextEmbeddingProfile::Native384),
        ..Default::default()
    });
    let embedding_model = if let Some(embedding_config) = embedding_config.as_ref() {
        Some(ErEmbeddingModel::load_with_config(embedding_config)?)
    } else {
        None
    };
    let mut batches = if config.replay_patches {
        derive_dirty_scope_review_batches_with_replay(&store, config.session_id.as_ref())
    } else {
        derive_dirty_scope_review_batches(&store, config.session_id.as_ref())
    }
    .map_err(|error| error.to_string())?;
    let mut scopes = Vec::with_capacity(batches.len());
    let mut aggregate = AggregateMetrics {
        scope_count: batches.len(),
        ..Default::default()
    };

    for batch in &mut batches {
        let lexical_summary = generate_lexical_candidates(batch, config.candidate_limit);
        let embedding_summary = if let (Some(model), Some(embedding_config)) =
            (embedding_model.as_ref(), embedding_config.as_ref())
        {
            Some(generate_embedding_candidates(
                batch,
                model,
                config.embedding_limit,
                embedding_config,
            )?)
        } else {
            None
        };
        let fused_summary =
            generate_fused_candidates(batch, config.candidate_limit.max(config.embedding_limit));
        let metrics = compute_lexical_metrics(batch);
        let retrieval = compute_retrieval_comparison(batch);
        let decisions = draft_review_decisions(batch);
        if config.persist_patches {
            persist_er_patch_sidecar(&store, batch, &decisions, now_ms())
                .map_err(|error| error.to_string())?;
        }
        let mut decision_counts = BTreeMap::<String, usize>::new();
        for decision in &decisions {
            let key = decision_kind_name(&decision.kind).to_owned();
            *decision_counts.entry(key.clone()).or_default() += 1;
            *aggregate.decision_counts.entry(key).or_default() += 1;
        }
        aggregate.case_count += metrics.case_count;
        aggregate.cases_with_candidates += metrics.cases_with_candidates;
        aggregate.total_candidate_count += metrics.total_candidate_count;
        if let Some(summary) = embedding_summary.as_ref() {
            aggregate.embedding_cases_with_candidates += summary.matched_case_count;
            aggregate.embedding_total_candidate_count += summary.total_candidate_count;
        }
        aggregate.fused_cases_with_candidates += fused_summary.matched_case_count;
        aggregate.fused_total_candidate_count += fused_summary.total_candidate_count;
        aggregate.additional_cases_covered += retrieval.additional_cases_covered;
        aggregate.additional_candidates += retrieval.additional_candidates;
        aggregate.alias_case_count += metrics.alias_case_count;
        aggregate.alias_exact_hit_count += metrics.alias_exact_hit_count;
        aggregate.type_disagreement_case_count += metrics.type_disagreement_case_count;
        aggregate.type_disagreement_rescue_count += metrics.type_disagreement_rescue_count;
        aggregate.unresolved_case_count += metrics.unresolved_case_count;
        aggregate.ambiguous_case_count += metrics.ambiguous_case_count;
        aggregate.unresolved_or_ambiguous_case_count += metrics.unresolved_or_ambiguous_case_count;
        aggregate.unresolved_or_ambiguous_covered_count +=
            metrics.unresolved_or_ambiguous_covered_count;

        scopes.push(ScopeSmokeReport {
            scope_key: batch.scope_key.clone(),
            session_id: batch.session_id.as_ref().map(|value| value.0.clone()),
            document_ref_count: batch.document_refs.len(),
            review_case_count: batch.review_cases.len(),
            entity_profile_count: batch.entity_profiles.len(),
            lexical_summary,
            embedding_summary,
            fused_summary,
            metrics,
            retrieval,
            decision_counts,
            decisions,
            case_summaries: summarize_review_cases(batch),
        });
    }

    aggregate.average_candidates_per_case =
        ratio(aggregate.total_candidate_count, aggregate.case_count);
    aggregate.embedding_average_candidates_per_case = ratio(
        aggregate.embedding_total_candidate_count,
        aggregate.case_count,
    );
    aggregate.fused_average_candidates_per_case =
        ratio(aggregate.fused_total_candidate_count, aggregate.case_count);
    aggregate.alias_exact_hit_rate =
        ratio(aggregate.alias_exact_hit_count, aggregate.alias_case_count);
    aggregate.type_disagreement_rescue_rate = ratio(
        aggregate.type_disagreement_rescue_count,
        aggregate.type_disagreement_case_count,
    );
    aggregate.unresolved_or_ambiguous_coverage_rate = ratio(
        aggregate.unresolved_or_ambiguous_covered_count,
        aggregate.unresolved_or_ambiguous_case_count,
    );

    if seed.seeded {
        seed.dirty_scope_count = store
            .list_dirty_scopes()
            .map_err(|error| error.to_string())?
            .len();
        seed.store_path = Some(store_path.to_string_lossy().to_string());
    }

    Ok(SmokeReport {
        store_path: store_path.to_string_lossy().to_string(),
        session_id: config.session_id.as_ref().map(|value| value.0.clone()),
        seed,
        aggregate,
        scopes,
    })
}

fn seed_store(path: &Path, corpus: &CorpusDocument) -> Result<SeedReport, String> {
    let store = PhoenixOvergraphStore::open(path).map_err(|error| error.to_string())?;
    store
        .init_archive_schema()
        .map_err(|error| error.to_string())?;
    store
        .init_graph_kernel_schema()
        .map_err(|error| error.to_string())?;
    store
        .write_kernel_checkpoint(1, "seed", &KernelGraphSnapshot::default())
        .map_err(|error| error.to_string())?;

    let document = IngestDocument {
        document_id: DocumentId(format!("er-post-{}", corpus.id)),
        note_id: None,
        title: corpus.title.clone(),
        text: corpus.text.clone(),
        scope: ScopeKey::default(),
    };
    let session_id = SessionId(format!("er-post-session-{}", corpus.id));
    PhoenixInvarantV3::new(Default::default())
        .ingest_documents_native(&store, Some(&session_id), &[document], 0, now_ms())
        .map_err(|error| error.to_string())?;

    Ok(SeedReport {
        seeded: true,
        corpus_id: Some(corpus.id.clone()),
        store_path: Some(path.to_string_lossy().to_string()),
        dirty_scope_count: 0,
    })
}

fn render_markdown(report: &SmokeReport, case_limit: usize) -> String {
    let mut lines = vec![
        "# Phoenix ER Post Smoke".to_owned(),
        String::new(),
        format!("Store: `{}`", report.store_path),
    ];
    if let Some(corpus_id) = report.seed.corpus_id.as_deref() {
        lines.push(format!("Seeded corpus: `{corpus_id}`"));
    }
    lines.push(format!("Scopes: `{}`", report.aggregate.scope_count));
    lines.push(format!("Cases: `{}`", report.aggregate.case_count));
    lines.push(String::new());
    lines.push("## Aggregate".to_owned());
    lines.push(String::new());
    lines.push(format!(
        "- cases with lexical candidates: `{}` / `{}`",
        report.aggregate.cases_with_candidates, report.aggregate.case_count
    ));
    lines.push(format!(
        "- average candidates per case: `{:.2}`",
        report.aggregate.average_candidates_per_case
    ));
    lines.push(format!(
        "- fused cases with candidates: `{}` / `{}`",
        report.aggregate.fused_cases_with_candidates, report.aggregate.case_count
    ));
    lines.push(format!(
        "- average fused candidates per case: `{:.2}`",
        report.aggregate.fused_average_candidates_per_case
    ));
    lines.push(format!(
        "- union delta vs lexical-only: cases `{:+}`, candidates `{:+}`",
        report.aggregate.additional_cases_covered, report.aggregate.additional_candidates
    ));
    if report.aggregate.embedding_total_candidate_count > 0 {
        lines.push(format!(
            "- cases with embedding candidates: `{}` / `{}`",
            report.aggregate.embedding_cases_with_candidates, report.aggregate.case_count
        ));
        lines.push(format!(
            "- average embedding candidates per case: `{:.2}`",
            report.aggregate.embedding_average_candidates_per_case
        ));
    }
    lines.push(format!(
        "- exact alias hit rate: `{}` / `{}` = `{:.3}`",
        report.aggregate.alias_exact_hit_count,
        report.aggregate.alias_case_count,
        report.aggregate.alias_exact_hit_rate
    ));
    lines.push(format!(
        "- type disagreement rescue rate: `{}` / `{}` = `{:.3}`",
        report.aggregate.type_disagreement_rescue_count,
        report.aggregate.type_disagreement_case_count,
        report.aggregate.type_disagreement_rescue_rate
    ));
    lines.push(format!(
        "- unresolved/ambiguous coverage: `{}` / `{}` = `{:.3}`",
        report.aggregate.unresolved_or_ambiguous_covered_count,
        report.aggregate.unresolved_or_ambiguous_case_count,
        report.aggregate.unresolved_or_ambiguous_coverage_rate
    ));

    for scope in &report.scopes {
        lines.push(String::new());
        lines.push(format!("## Scope `{}`", scope.scope_key));
        lines.push(String::new());
        lines.push(format!(
            "- review cases: `{}`; entity profiles: `{}`; docs: `{}`",
            scope.review_case_count, scope.entity_profile_count, scope.document_ref_count
        ));
        lines.push(format!(
            "- lexical coverage: `{}` / `{}`",
            scope.metrics.cases_with_candidates, scope.metrics.case_count
        ));
        lines.push(format!(
            "- avg candidates per case: `{:.2}`",
            scope.metrics.average_candidates_per_case
        ));
        lines.push(format!(
            "- fused coverage: `{}` / `{}`",
            scope.fused_summary.matched_case_count, scope.fused_summary.case_count
        ));
        lines.push(format!(
            "- union vs lexical-only: cases `{:+}`, candidates `{:+}`",
            scope.retrieval.additional_cases_covered, scope.retrieval.additional_candidates
        ));
        if let Some(embedding_summary) = scope.embedding_summary.as_ref() {
            lines.push(format!(
                "- embedding coverage: `{}` / `{}`",
                embedding_summary.matched_case_count, embedding_summary.case_count
            ));
        }
        if !scope.decision_counts.is_empty() {
            lines.push(format!(
                "- decisions: `{}`",
                scope
                    .decision_counts
                    .iter()
                    .map(|(kind, count)| format!("{kind}={count}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        lines.push(String::new());
        lines.push(
            "| Case | Kind | Surface | Status | Lex | Emb | Union | Top lexical | Top embedding | Top fused |"
                .to_owned(),
        );
        lines.push("| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |".to_owned());
        for case in scope.case_summaries.iter().take(case_limit) {
            let top_lexical = case
                .top_candidate
                .as_ref()
                .map(|candidate| {
                    format!(
                        "{} ({}, {})",
                        candidate.entity_id.0,
                        candidate.score_millis,
                        candidate.evidence.join("; ")
                    )
                })
                .unwrap_or_else(|| "-".to_owned());
            let top_embedding = case
                .top_embedding_candidate
                .as_ref()
                .map(|candidate| {
                    format!(
                        "{} ({}, {})",
                        candidate.entity_id.0,
                        candidate.score_millis,
                        candidate.evidence.join("; ")
                    )
                })
                .unwrap_or_else(|| "-".to_owned());
            let top_fused = case
                .top_fused_candidate
                .as_ref()
                .map(|candidate| {
                    format!(
                        "{} ({}, {})",
                        candidate.entity_id.0,
                        candidate.score_millis,
                        candidate.evidence.join("; ")
                    )
                })
                .unwrap_or_else(|| "-".to_owned());
            lines.push(format!(
                "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | {} | {} | {} |",
                case.case_id,
                case.kind,
                escape_md(&case.surface),
                case.decision_status,
                case.lexical_candidate_count,
                case.embedding_candidate_count,
                case.fused_candidate_count,
                escape_md(&top_lexical),
                escape_md(&top_embedding),
                escape_md(&top_fused)
            ));
        }
        if !scope.decisions.is_empty() {
            lines.push(String::new());
            lines.push("| Decision | Case | Entity | Score | Rationale |".to_owned());
            lines.push("| --- | --- | --- | ---: | --- |".to_owned());
            for decision in scope.decisions.iter().take(case_limit) {
                lines.push(format!(
                    "| `{}` | `{}` | `{}` | `{}` | {} |",
                    decision_kind_name(&decision.kind),
                    decision.case_id,
                    decision
                        .entity_id
                        .as_ref()
                        .map(|value| value.0.as_str())
                        .unwrap_or("-"),
                    decision.score_millis,
                    escape_md(&decision.rationale)
                ));
            }
        }
    }

    lines.join("\n")
}

fn escape_md(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

fn load_standard_corpus() -> Result<Vec<CorpusDocument>, String> {
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

fn select_corpus(corpus_filter: Option<&str>) -> Result<CorpusDocument, String> {
    let target = corpus_filter.unwrap_or("shortrun");
    load_standard_corpus()?
        .into_iter()
        .find(|corpus| corpus.id == target)
        .ok_or_else(|| format!("no corpus matched filter: {target}"))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("workspace root")
        .to_path_buf()
}

fn unique_smoke_store_path(corpus_id: &str) -> PathBuf {
    env::temp_dir().join(format!("phoenix-er-post-smoke-{corpus_id}-{}", now_ms()))
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].parse::<usize>().ok()))
        .flatten()
}

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].clone()))
}

fn parse_path_arg(args: &[String], flag: &str) -> Option<PathBuf> {
    parse_string_arg(args, flag).map(PathBuf::from)
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
