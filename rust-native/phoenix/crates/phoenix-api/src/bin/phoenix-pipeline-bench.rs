use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use phoenix_ingest_overgraph::{InvarantV3Config, PhoenixInvarantV3};
use phoenix_rel_post::{
    benchmark_scope_review_pipeline, default_relation_type_specs, GlirelModel,
    RelationBenchmarkCounts, RelationBenchmarkReport, RelationWindowBuildStats,
};
use phoenix_types::{DocumentId, IngestDocument, ScopeKey};
use serde::Serialize;

#[derive(Debug, Clone)]
struct Config {
    warmups: usize,
    iterations: usize,
    chapter: usize,
    json: bool,
    output_path: Option<PathBuf>,
    glirel_model_root: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            warmups: 1,
            iterations: 5,
            chapter: 1,
            json: false,
            output_path: None,
            glirel_model_root: None,
        }
    }
}

#[derive(Debug, Clone)]
struct CaseInput {
    case_id: String,
    title: String,
    text: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct PhaseStats {
    runs_us: Vec<u64>,
    min_us: u64,
    mean_us: f64,
    median_us: f64,
    p95_us: u64,
    max_us: u64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct IngestCaseReport {
    counts: phoenix_ingest_overgraph::IngestBenchmarkCounts,
    phases: BTreeMap<String, PhaseStats>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RelationCaseReport {
    counts: RelationBenchmarkCounts,
    used_model: bool,
    window_build_stats: RelationWindowBuildStats,
    phases: BTreeMap<String, PhaseStats>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct CaseReport {
    case_id: String,
    title: String,
    text_bytes: usize,
    ingest: IngestCaseReport,
    relation: RelationCaseReport,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ProjectionReport {
    phase: String,
    slice_case_id: String,
    full_case_id: String,
    slice_mean_us: f64,
    projected_full_linear_us: f64,
    actual_full_mean_us: f64,
    superlinear_factor: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchReport {
    corpus: String,
    warmups: usize,
    iterations: usize,
    glirel_model_root: Option<String>,
    glirel_model_load_us: Option<u64>,
    cases: Vec<CaseReport>,
    projections: Vec<ProjectionReport>,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
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

    let mut model_load_us = None;
    let mut maybe_model = None;
    if let Some(model_root) = &config.glirel_model_root {
        let started = std::time::Instant::now();
        let model = GlirelModel::load(model_root).map_err(|error| {
            format!(
                "failed to load GLiREL model {}: {error}",
                model_root.display()
            )
        })?;
        model_load_us = Some(started.elapsed().as_micros() as u64);
        maybe_model = Some(model);
    }
    let model_ref = maybe_model.as_ref();
    let relation_specs = default_relation_type_specs();

    let ingest = PhoenixInvarantV3::new(InvarantV3Config::default());
    let cases = vec![
        run_case(&ingest, &full_case, &config, model_ref, &relation_specs)?,
        run_case(&ingest, &chapter_case, &config, model_ref, &relation_specs)?,
    ];
    let projections = build_projections(&cases);

    let report = BenchReport {
        corpus: "shortrun".to_owned(),
        warmups: config.warmups,
        iterations: config.iterations,
        glirel_model_root: config
            .glirel_model_root
            .as_ref()
            .map(|path| path.display().to_string()),
        glirel_model_load_us: model_load_us,
        cases,
        projections,
    };

    let output_path = config
        .output_path
        .clone()
        .unwrap_or_else(|| default_output_path(&root));
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| format!("failed to serialize report: {error}"))?;
    fs::write(&output_path, &json)
        .map_err(|error| format!("failed to write {}: {error}", output_path.display()))?;

    if config.json {
        println!("{json}");
    } else {
        println!("report: {}", output_path.display());
        for case in &report.cases {
            println!(
                "{} bytes={} ingest.meanMs={:.3} relation.meanMs={:.3}",
                case.case_id,
                case.text_bytes,
                mean_ms(case.ingest.phases.get("document_total_us")),
                mean_ms(case.relation.phases.get("review_batch_total_us")),
            );
        }
    }

    Ok(())
}

fn run_case(
    ingest: &PhoenixInvarantV3,
    input: &CaseInput,
    config: &Config,
    model: Option<&GlirelModel>,
    relation_specs: &[phoenix_rel_post::GlirelRelationTypeSpec],
) -> Result<CaseReport, String> {
    let document = IngestDocument {
        document_id: DocumentId(input.case_id.clone()),
        note_id: None,
        title: input.title.clone(),
        text: input.text.clone(),
        scope: ScopeKey::default(),
    };

    for _ in 0..config.warmups {
        let _ = ingest
            .benchmark_document_pipeline(&document, None, benchmark_created_at())
            .map_err(|error| format!("ingest warmup failed for {}: {error}", input.case_id))?;
    }

    let archive = ingest
        .build_archive_for_benchmark(&document, None, benchmark_created_at())
        .map_err(|error| {
            format!(
                "failed to build benchmark archive for {}: {error}",
                input.case_id
            )
        })?;

    let mut ingest_runs = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        ingest_runs.push(
            ingest
                .benchmark_document_pipeline(&document, None, benchmark_created_at())
                .map_err(|error| {
                    format!("ingest benchmark failed for {}: {error}", input.case_id)
                })?,
        );
    }

    for _ in 0..config.warmups {
        let _ = benchmark_scope_review_pipeline(
            std::slice::from_ref(&archive),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            model,
            relation_specs,
        )
        .map_err(|error| format!("relation warmup failed for {}: {error}", input.case_id))?;
    }

    let mut relation_runs = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        relation_runs.push(
            benchmark_scope_review_pipeline(
                std::slice::from_ref(&archive),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                model,
                relation_specs,
            )
            .map_err(|error| format!("relation benchmark failed for {}: {error}", input.case_id))?,
        );
    }

    let ingest_last = ingest_runs
        .last()
        .cloned()
        .ok_or_else(|| format!("no ingest runs captured for {}", input.case_id))?;
    let relation_last = relation_runs
        .last()
        .cloned()
        .ok_or_else(|| format!("no relation runs captured for {}", input.case_id))?;

    Ok(CaseReport {
        case_id: input.case_id.clone(),
        title: input.title.clone(),
        text_bytes: input.text.len(),
        ingest: IngestCaseReport {
            counts: ingest_last.counts,
            phases: ingest_phase_stats(&ingest_runs),
        },
        relation: RelationCaseReport {
            counts: relation_last.counts,
            used_model: relation_last.used_model,
            window_build_stats: relation_last.window_build_stats,
            phases: relation_phase_stats(&relation_runs),
        },
    })
}

fn ingest_phase_stats(
    runs: &[phoenix_ingest_overgraph::IngestBenchmarkReport],
) -> BTreeMap<String, PhaseStats> {
    BTreeMap::from([
        phase_entry(
            "document_total_us",
            runs.iter().map(|run| run.document_total_us).collect(),
        ),
        phase_entry(
            "scan_bundle_us",
            runs.iter().map(|run| run.scan_bundle_us).collect(),
        ),
        phase_entry(
            "resolve_us",
            runs.iter().map(|run| run.resolve_us).collect(),
        ),
        phase_entry(
            "post_resolve_total_us",
            runs.iter().map(|run| run.post_resolve_total_us).collect(),
        ),
        phase_entry(
            "causal_substrate_us",
            runs.iter().map(|run| run.causal_substrate_us).collect(),
        ),
        phase_entry(
            "temporal_substrate_us",
            runs.iter().map(|run| run.temporal_substrate_us).collect(),
        ),
        phase_entry(
            "event_identity_substrate_us",
            runs.iter()
                .map(|run| run.event_identity_substrate_us)
                .collect(),
        ),
        phase_entry(
            "lexical_postings_us",
            runs.iter().map(|run| run.lexical_postings_us).collect(),
        ),
        phase_entry(
            "segment_encode_us",
            runs.iter().map(|run| run.segment_encode_us).collect(),
        ),
    ])
}

fn relation_phase_stats(runs: &[RelationBenchmarkReport]) -> BTreeMap<String, PhaseStats> {
    BTreeMap::from([
        phase_entry(
            "review_batch_total_us",
            runs.iter().map(|run| run.review_batch_total_us).collect(),
        ),
        phase_entry(
            "persisted_relations_us",
            runs.iter().map(|run| run.persisted_relations_us).collect(),
        ),
        phase_entry(
            "entity_profiles_us",
            runs.iter().map(|run| run.entity_profiles_us).collect(),
        ),
        phase_entry(
            "windows_us",
            runs.iter().map(|run| run.windows_us).collect(),
        ),
        phase_entry(
            "review_cases_us",
            runs.iter().map(|run| run.review_cases_us).collect(),
        ),
        phase_entry(
            "patch_merge_us",
            runs.iter().map(|run| run.patch_merge_us).collect(),
        ),
        phase_entry(
            "primary_lane_us",
            runs.iter().map(|run| run.primary_lane_us).collect(),
        ),
    ])
}

fn build_projections(cases: &[CaseReport]) -> Vec<ProjectionReport> {
    let Some(full) = cases.iter().find(|case| case.case_id == "shortrun-full") else {
        return Vec::new();
    };
    let Some(slice) = cases
        .iter()
        .find(|case| case.case_id.starts_with("shortrun-chapter-"))
    else {
        return Vec::new();
    };
    let scale = full.text_bytes as f64 / slice.text_bytes.max(1) as f64;
    let phases = [
        (
            "ingest.document_total_us",
            &slice.ingest.phases,
            &full.ingest.phases,
            "document_total_us",
        ),
        (
            "relation.review_batch_total_us",
            &slice.relation.phases,
            &full.relation.phases,
            "review_batch_total_us",
        ),
        (
            "relation.primary_lane_us",
            &slice.relation.phases,
            &full.relation.phases,
            "primary_lane_us",
        ),
    ];
    phases
        .into_iter()
        .filter_map(|(phase, slice_map, full_map, key)| {
            let slice_mean = slice_map.get(key)?.mean_us;
            let full_mean = full_map.get(key)?.mean_us;
            let projected = slice_mean * scale;
            Some(ProjectionReport {
                phase: phase.to_owned(),
                slice_case_id: slice.case_id.clone(),
                full_case_id: full.case_id.clone(),
                slice_mean_us: slice_mean,
                projected_full_linear_us: projected,
                actual_full_mean_us: full_mean,
                superlinear_factor: if projected <= f64::EPSILON {
                    0.0
                } else {
                    full_mean / projected
                },
            })
        })
        .collect()
}

fn phase_entry(name: &str, runs_us: Vec<u64>) -> (String, PhaseStats) {
    (name.to_owned(), compute_phase_stats(runs_us))
}

fn compute_phase_stats(mut runs_us: Vec<u64>) -> PhaseStats {
    runs_us.sort_unstable();
    let sum = runs_us.iter().copied().sum::<u64>() as f64;
    let len = runs_us.len().max(1);
    PhaseStats {
        min_us: *runs_us.first().unwrap_or(&0),
        mean_us: sum / len as f64,
        median_us: percentile(&runs_us, 0.5),
        p95_us: percentile(&runs_us, 0.95).round() as u64,
        max_us: *runs_us.last().unwrap_or(&0),
        runs_us,
    }
}

fn percentile(sorted: &[u64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[pos.min(sorted.len() - 1)] as f64
}

fn mean_ms(stats: Option<&PhaseStats>) -> f64 {
    stats
        .map(|value| value.mean_us / 1000.0)
        .unwrap_or_default()
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

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut config = Config::default();
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--warmups" => {
                index += 1;
                config.warmups = parse_usize_arg(args.get(index), "--warmups")?;
            }
            "--iterations" => {
                index += 1;
                config.iterations = parse_usize_arg(args.get(index), "--iterations")?;
            }
            "--chapter" => {
                index += 1;
                config.chapter = parse_usize_arg(args.get(index), "--chapter")?;
            }
            "--output" => {
                index += 1;
                let value = args.get(index).ok_or("--output requires a value")?;
                config.output_path = Some(PathBuf::from(value));
            }
            "--glirel-model-root" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or("--glirel-model-root requires a value")?;
                config.glirel_model_root = Some(PathBuf::from(value));
            }
            "--json" => config.json = true,
            flag => return Err(format!("unknown argument: {flag}")),
        }
        index += 1;
    }
    Ok(config)
}

fn parse_usize_arg(value: Option<&String>, flag: &str) -> Result<usize, String> {
    let value = value.ok_or_else(|| format!("{flag} requires a value"))?;
    value
        .parse::<usize>()
        .map_err(|error| format!("invalid {flag} value '{value}': {error}"))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("workspace root")
        .to_path_buf()
}

fn default_output_path(root: &Path) -> PathBuf {
    root.join("rust-native")
        .join("phoenix")
        .join("reports")
        .join("pipeline-bench-shortrun.json")
}

fn benchmark_created_at() -> i64 {
    1_700_000_000_000
}
