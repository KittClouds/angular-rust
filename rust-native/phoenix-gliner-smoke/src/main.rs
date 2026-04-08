use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use composable::Composable;
use gliner::model::input::relation::schema::RelationSchema;
use gliner::model::input::text::TextInput;
use gliner::model::output::decoded::SpanOutput;
use gliner::model::output::relation::RelationOutput;
use gliner::model::params::Parameters;
use gliner::model::pipeline::relation::RelationPipeline;
use gliner::model::pipeline::span::SpanMode;
use gliner::model::pipeline::token::TokenPipeline;
use gliner::model::GLiNER;
use gliner::text::span::Span;
use orp::model::Model;
use orp::params::RuntimeParameters;
use orp::pipeline::Pipeline;
use phoenix_ingest_native::{NativeNerMention, PhoenixInvarantV3};
use phoenix_types::{ResolverEntitySeed, ScopeKey};
use serde::Serialize;

#[derive(Debug, Clone)]
struct SmokeConfig {
    text_path: Option<PathBuf>,
    corpus_id: Option<String>,
    threshold: f32,
    relation_threshold: f32,
    warm_runs: usize,
    corpus_mode: CorpusMode,
    audit: bool,
    multitask_relations: bool,
    audit_window_bytes: usize,
    audit_overlap_bytes: usize,
    audit_max_windows: usize,
    audit_batch_size: usize,
    audit_start_ratio: f32,
    audit_end_ratio: f32,
    audit_summary_only: bool,
    audit_surface_limit: usize,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        Self {
            text_path: None,
            corpus_id: None,
            threshold: 0.45,
            relation_threshold: 0.45,
            warm_runs: 3,
            corpus_mode: CorpusMode::FullDocument,
            audit: false,
            multitask_relations: false,
            audit_window_bytes: 1600,
            audit_overlap_bytes: 240,
            audit_max_windows: 0,
            audit_batch_size: 16,
            audit_start_ratio: 0.0,
            audit_end_ratio: 1.0,
            audit_summary_only: false,
            audit_surface_limit: 200,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum CorpusMode {
    FullDocument,
    Excerpt,
}

impl CorpusMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::FullDocument => "full-document",
            Self::Excerpt => "excerpt",
        }
    }
}

#[derive(Debug, Clone)]
struct SmokeBundle {
    texts: Vec<String>,
    labels: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RaceSpan {
    sequence: usize,
    text: String,
    normalized: String,
    label: String,
    probability: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct LaneReport {
    name: String,
    mode: String,
    load_ms: u64,
    warm_runs: usize,
    warm_ms_runs: Vec<u64>,
    warm_ms_avg: f32,
    text_count: usize,
    mention_count: usize,
    label_counts: BTreeMap<String, usize>,
    named_count: Option<usize>,
    nominal_count: Option<usize>,
    pronoun_count: Option<usize>,
    discovery_count: Option<usize>,
    spans: Vec<RaceSpan>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct OverlapReport {
    unlabeled_overlap: usize,
    exact_label_overlap: usize,
    native_only: usize,
    challenger_only: usize,
    native_recall_vs_challenger: f32,
    exact_label_agreement: f32,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct DragRaceReport {
    threshold: f32,
    warm_runs: usize,
    text_count: usize,
    input_mode: String,
    labels: Vec<String>,
    native: LaneReport,
    gliner_x_small: LaneReport,
    gliner_multitask: LaneReport,
    native_vs_x_small: OverlapReport,
    native_vs_multitask: OverlapReport,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct RelationSequenceReport {
    sequence: usize,
    text_preview: String,
    entity_count: usize,
    relation_count: usize,
    entities: Vec<String>,
    relations: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct MultitaskRelationReport {
    threshold: f32,
    relation_threshold: f32,
    warm_runs: usize,
    input_mode: String,
    seed_model: String,
    labels: Vec<String>,
    relation_labels: Vec<String>,
    load_ms: u64,
    warm_ms_runs: Vec<u64>,
    warm_ms_avg: f32,
    text_count: usize,
    entity_count: usize,
    relation_count: usize,
    entity_label_counts: BTreeMap<String, usize>,
    relation_label_counts: BTreeMap<String, usize>,
    sequences: Vec<RelationSequenceReport>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AuditSpan {
    window_index: usize,
    text: String,
    normalized: String,
    label: String,
    probability: f32,
    repeated_count: usize,
    context: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AuditWindowReport {
    index: usize,
    start: usize,
    end: usize,
    byte_len: usize,
    scan_ms: u64,
    preview: String,
    mention_count: usize,
    label_counts: BTreeMap<String, usize>,
    spans: Vec<AuditSpan>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AuditSurfaceReport {
    normalized: String,
    surface: String,
    count: usize,
    avg_probability: f32,
    labels: Vec<String>,
    label_counts: BTreeMap<String, usize>,
    windows: Vec<usize>,
    suggested_label: Option<String>,
    flags: Vec<String>,
    sample_contexts: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct AuditReport {
    threshold: f32,
    load_ms: u64,
    warm_runs: usize,
    scan_ms_runs: Vec<u64>,
    scan_ms_avg: f32,
    input_mode: String,
    labels: Vec<String>,
    window_bytes: usize,
    overlap_bytes: usize,
    batch_size: usize,
    slice_start_ratio: f32,
    slice_end_ratio: f32,
    total_bytes: usize,
    covered_bytes: usize,
    coverage_ratio: f32,
    window_count: usize,
    mention_count: usize,
    repeated_surface_count: usize,
    repeated_surfaces: Vec<AuditSurfaceReport>,
    label_conflict_count: usize,
    label_conflicts: Vec<AuditSurfaceReport>,
    codename_candidate_count: usize,
    codename_candidates: Vec<AuditSurfaceReport>,
    organization_candidate_count: usize,
    organization_candidates: Vec<AuditSurfaceReport>,
    demotion_candidate_count: usize,
    demotion_candidates: Vec<AuditSurfaceReport>,
    windows: Vec<AuditWindowReport>,
}

#[derive(Debug, Clone)]
struct TextWindow {
    index: usize,
    start: usize,
    end: usize,
    text: String,
}

#[derive(Debug, Clone, Default)]
struct SurfaceAccumulator {
    surface: String,
    count: usize,
    total_probability: f32,
    label_counts: BTreeMap<String, usize>,
    windows: BTreeSet<usize>,
    sample_contexts: Vec<String>,
}

#[derive(Debug, Clone)]
struct RelationHit {
    subject: String,
    relation: String,
    object: String,
    probability: f32,
}

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let config = SmokeConfig {
        text_path: args
            .windows(2)
            .find_map(|window| (window[0] == "--input").then(|| PathBuf::from(&window[1]))),
        corpus_id: args
            .windows(2)
            .find_map(|window| (window[0] == "--corpus").then(|| window[1].clone())),
        threshold: parse_f32_arg(&args, "--threshold").unwrap_or(0.45),
        relation_threshold: parse_f32_arg(&args, "--relation-threshold").unwrap_or(0.45),
        warm_runs: parse_usize_arg(&args, "--warm-runs").unwrap_or(3),
        corpus_mode: parse_corpus_mode_arg(&args, "--input-mode")
            .unwrap_or(CorpusMode::FullDocument),
        audit: has_flag(&args, "--audit"),
        multitask_relations: has_flag(&args, "--multitask-relations"),
        audit_window_bytes: parse_usize_arg(&args, "--audit-window-bytes").unwrap_or(1600),
        audit_overlap_bytes: parse_usize_arg(&args, "--audit-overlap-bytes").unwrap_or(240),
        audit_max_windows: parse_usize_arg(&args, "--audit-max-windows").unwrap_or(0),
        audit_batch_size: parse_usize_arg(&args, "--audit-batch-size").unwrap_or(16),
        audit_start_ratio: parse_f32_arg(&args, "--audit-start-ratio").unwrap_or(0.0),
        audit_end_ratio: parse_f32_arg(&args, "--audit-end-ratio").unwrap_or(1.0),
        audit_summary_only: has_flag(&args, "--audit-summary-only"),
        audit_surface_limit: parse_usize_arg(&args, "--audit-surface-limit").unwrap_or(200),
    };

    if config.multitask_relations {
        match run_multitask_relation_smoke(&config) {
            Ok(report) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("serialize relation report")
                );
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
    } else if config.audit {
        match run_gliner_audit(&config) {
            Ok(report) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("serialize audit report")
                );
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
    } else {
        match run_drag_race(&config) {
            Ok(report) => {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).expect("serialize drag race report")
                );
            }
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
    }
}

fn run_drag_race(config: &SmokeConfig) -> Result<DragRaceReport, Box<dyn Error + Send + Sync>> {
    let bundle = load_input(config)?;
    let native = run_native_lane(&bundle, config)?;
    let gliner_x_small = run_gliner_x_small_lane(&bundle, config)?;
    let gliner_multitask = run_gliner_multitask_lane(&bundle, config)?;

    Ok(DragRaceReport {
        threshold: config.threshold,
        warm_runs: config.warm_runs,
        text_count: bundle.texts.len(),
        input_mode: config.corpus_mode.as_str().to_owned(),
        labels: bundle.labels.clone(),
        native_vs_x_small: overlap_report(&native.spans, &gliner_x_small.spans),
        native_vs_multitask: overlap_report(&native.spans, &gliner_multitask.spans),
        native,
        gliner_x_small,
        gliner_multitask,
    })
}

fn run_native_lane(
    bundle: &SmokeBundle,
    config: &SmokeConfig,
) -> Result<LaneReport, Box<dyn Error + Send + Sync>> {
    let detector = PhoenixInvarantV3::default();
    let scope = ScopeKey::default();
    let resolver_seed: &[ResolverEntitySeed] = &[];
    let mut warm_ms_runs = Vec::with_capacity(config.warm_runs);
    let mut final_reports = Vec::new();

    for run_ix in 0..config.warm_runs {
        let mut total_ms = 0u64;
        let mut reports = Vec::with_capacity(bundle.texts.len());
        for text in &bundle.texts {
            let report = detector.benchmark_native_ner(text, &scope, resolver_seed);
            total_ms += report.scan_ms;
            reports.push(report);
        }
        warm_ms_runs.push(total_ms);
        if run_ix + 1 == config.warm_runs {
            final_reports = reports;
        }
    }

    let mut spans = Vec::new();
    let mut label_counts = BTreeMap::<String, usize>::new();
    let mut mention_count = 0usize;
    let mut named_count = 0usize;
    let mut nominal_count = 0usize;
    let mut pronoun_count = 0usize;
    let mut discovery_count = 0usize;

    for (sequence, report) in final_reports.iter().enumerate() {
        mention_count += report.mention_count;
        named_count += report.named_count;
        nominal_count += report.nominal_count;
        pronoun_count += report.pronoun_count;
        discovery_count += report.discovery_count;
        for mention in &report.mentions {
            *label_counts.entry(mention.label.clone()).or_default() += 1;
            spans.push(native_span(sequence, mention));
        }
    }

    Ok(LaneReport {
        name: "native".to_owned(),
        mode: "deterministic-hot-path".to_owned(),
        load_ms: 0,
        warm_runs: config.warm_runs,
        warm_ms_avg: average_ms(&warm_ms_runs),
        warm_ms_runs,
        text_count: bundle.texts.len(),
        mention_count,
        label_counts,
        named_count: Some(named_count),
        nominal_count: Some(nominal_count),
        pronoun_count: Some(pronoun_count),
        discovery_count: Some(discovery_count),
        spans,
    })
}

fn run_gliner_x_small_lane(
    bundle: &SmokeBundle,
    config: &SmokeConfig,
) -> Result<LaneReport, Box<dyn Error + Send + Sync>> {
    let tokenizer_path = gliner_x_small_root().join("tokenizer.json");
    let model_path = gliner_x_small_root().join("onnx").join("model.onnx");
    let runtime_params = RuntimeParameters::default();
    let text_refs = bundle.texts.iter().map(String::as_str).collect::<Vec<_>>();
    let label_refs = bundle.labels.iter().map(String::as_str).collect::<Vec<_>>();

    let load_started = Instant::now();
    let model = GLiNER::<SpanMode>::new(
        Parameters::default().with_threshold(config.threshold),
        runtime_params,
        tokenizer_path
            .to_str()
            .ok_or("x-small tokenizer path contains invalid utf-8")?,
        model_path
            .to_str()
            .ok_or("x-small model path contains invalid utf-8")?,
    )?;
    let load_ms = load_started.elapsed().as_millis() as u64;

    let mut warm_ms_runs = Vec::with_capacity(config.warm_runs);
    let mut final_spans = Vec::new();
    for run_ix in 0..config.warm_runs {
        let input = TextInput::from_str(&text_refs, &label_refs)?;
        let started = Instant::now();
        let output = model.inference(input)?;
        let elapsed = started.elapsed().as_millis() as u64;
        warm_ms_runs.push(elapsed);
        if run_ix + 1 == config.warm_runs {
            final_spans = output
                .spans
                .iter()
                .flat_map(|seq| seq.iter())
                .filter(|span| span.probability() >= config.threshold)
                .map(extracted_span)
                .collect();
        }
    }

    Ok(build_model_lane(
        "gliner-x-small",
        "span-ner",
        load_ms,
        config.warm_runs,
        warm_ms_runs,
        bundle.texts.len(),
        final_spans,
    ))
}

fn run_gliner_multitask_lane(
    bundle: &SmokeBundle,
    config: &SmokeConfig,
) -> Result<LaneReport, Box<dyn Error + Send + Sync>> {
    let tokenizer_path = gliner_multitask_root().join("tokenizer.json");
    let model_path = gliner_multitask_root()
        .join("onnx")
        .join("model_quantized.onnx");
    let params = Parameters::default().with_threshold(config.threshold);
    let runtime_params = RuntimeParameters::default();
    let text_refs = bundle.texts.iter().map(String::as_str).collect::<Vec<_>>();
    let label_refs = bundle.labels.iter().map(String::as_str).collect::<Vec<_>>();

    let load_started = Instant::now();
    let model = Model::new(
        model_path
            .to_str()
            .ok_or("multitask model path contains invalid utf-8")?,
        runtime_params,
    )?;
    let token_pipeline = TokenPipeline::new(
        tokenizer_path
            .to_str()
            .ok_or("multitask tokenizer path contains invalid utf-8")?,
    )?;
    let token_composable = token_pipeline.to_composable(&model, &params);
    let load_ms = load_started.elapsed().as_millis() as u64;

    let mut warm_ms_runs = Vec::with_capacity(config.warm_runs);
    let mut final_spans = Vec::new();
    for run_ix in 0..config.warm_runs {
        let input = TextInput::from_str(&text_refs, &label_refs)?;
        let started = Instant::now();
        let output: SpanOutput = token_composable.apply(input)?;
        let elapsed = started.elapsed().as_millis() as u64;
        warm_ms_runs.push(elapsed);
        if run_ix + 1 == config.warm_runs {
            final_spans = output
                .spans
                .iter()
                .flat_map(|seq| seq.iter())
                .filter(|span| span.probability() >= config.threshold)
                .map(extracted_span)
                .collect();
        }
    }

    Ok(build_model_lane(
        "gliner-multitask-large-v0.5",
        "token-ner",
        load_ms,
        config.warm_runs,
        warm_ms_runs,
        bundle.texts.len(),
        final_spans,
    ))
}

fn run_multitask_relation_smoke(
    config: &SmokeConfig,
) -> Result<MultitaskRelationReport, Box<dyn Error + Send + Sync>> {
    let bundle = load_relation_input(config)?;
    let x_small_tokenizer_path = gliner_x_small_root().join("tokenizer.json");
    let x_small_model_path = gliner_x_small_root().join("onnx").join("model.onnx");
    let tokenizer_path = gliner_multitask_root().join("tokenizer.json");
    let model_path = gliner_multitask_root()
        .join("onnx")
        .join("model_quantized.onnx");
    let params = Parameters::default().with_threshold(config.threshold);
    let runtime_params = RuntimeParameters::default();
    let relation_schema = default_relation_schema();
    let relation_labels = relation_schema
        .relations()
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    let text_refs = bundle.texts.iter().map(String::as_str).collect::<Vec<_>>();
    let label_refs = bundle.labels.iter().map(String::as_str).collect::<Vec<_>>();

    let load_started = Instant::now();
    let x_small_model = GLiNER::<SpanMode>::new(
        Parameters::default().with_threshold(config.threshold),
        RuntimeParameters::default(),
        x_small_tokenizer_path
            .to_str()
            .ok_or("x-small tokenizer path contains invalid utf-8")?,
        x_small_model_path
            .to_str()
            .ok_or("x-small model path contains invalid utf-8")?,
    )?;
    let relation_model = Model::new(
        model_path
            .to_str()
            .ok_or("multitask model path contains invalid utf-8")?,
        runtime_params,
    )?;
    let relation_composable = RelationPipeline::default(&tokenizer_path, &relation_schema)?
        .to_composable(&relation_model, &params);
    let load_ms = load_started.elapsed().as_millis() as u64;

    let mut warm_ms_runs = Vec::with_capacity(config.warm_runs);
    let mut final_sequences = Vec::new();
    let mut final_entity_label_counts = BTreeMap::<String, usize>::new();
    let mut final_relation_label_counts = BTreeMap::<String, usize>::new();
    let mut final_entity_count = 0usize;
    let mut final_relation_count = 0usize;

    for run_ix in 0..config.warm_runs {
        let input = TextInput::from_str(&text_refs, &label_refs)?;
        let started = Instant::now();
        let seeded_spans = filter_relation_seed_output(x_small_model.inference(input)?, config.threshold);
        let seed_sequences = seeded_spans
            .spans
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let entity_sequences = seed_sequences
            .iter()
            .map(|seq| seq.iter().map(extracted_span).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let relations: RelationOutput = relation_composable.apply(seeded_spans)?;
        let elapsed = started.elapsed().as_millis() as u64;
        warm_ms_runs.push(elapsed);

        if run_ix + 1 == config.warm_runs {
            final_sequences = bundle
                .texts
                .iter()
                .enumerate()
                .map(|(sequence, text)| {
                    let entities = entity_sequences.get(sequence).cloned().unwrap_or_default();
                    let filtered_relations = filter_relation_hits(
                        relations
                            .relations
                            .get(sequence)
                            .map(Vec::as_slice)
                            .unwrap_or(&[]),
                        seed_sequences
                            .get(sequence)
                            .map(Vec::as_slice)
                            .unwrap_or(&[]),
                        text,
                        config.relation_threshold,
                    );

                    for entity in &entities {
                        *final_entity_label_counts
                            .entry(entity.label.clone())
                            .or_default() += 1;
                    }
                    for relation in &filtered_relations {
                        *final_relation_label_counts
                            .entry(relation.relation.clone())
                            .or_default() += 1;
                    }
                    final_entity_count += entities.len();
                    final_relation_count += filtered_relations.len();

                    RelationSequenceReport {
                        sequence,
                        text_preview: preview_text(text, 180),
                        entity_count: entities.len(),
                        relation_count: filtered_relations.len(),
                        entities: entities
                            .iter()
                            .map(|entity| {
                                format!(
                                    "{} [{}] ({:.3})",
                                    entity.text, entity.label, entity.probability
                                )
                            })
                            .collect(),
                        relations: filtered_relations
                            .iter()
                            .map(|relation| {
                                format!(
                                    "{} --{}--> {} ({:.3})",
                                    relation.subject,
                                    relation.relation,
                                    relation.object,
                                    relation.probability
                                )
                            })
                            .collect(),
                    }
                })
                .collect::<Vec<_>>();
        }
    }

    Ok(MultitaskRelationReport {
        threshold: config.threshold,
        relation_threshold: config.relation_threshold,
        warm_runs: config.warm_runs,
        input_mode: config.corpus_mode.as_str().to_owned(),
        seed_model: "gliner-x-small".to_owned(),
        labels: bundle.labels,
        relation_labels,
        load_ms,
        warm_ms_runs: warm_ms_runs.clone(),
        warm_ms_avg: average_ms(&warm_ms_runs),
        text_count: bundle.texts.len(),
        entity_count: final_entity_count,
        relation_count: final_relation_count,
        entity_label_counts: final_entity_label_counts,
        relation_label_counts: final_relation_label_counts,
        sequences: final_sequences,
    })
}

fn load_relation_input(
    config: &SmokeConfig,
) -> Result<SmokeBundle, Box<dyn Error + Send + Sync>> {
    if let Some(path) = &config.text_path {
        let text = fs::read_to_string(path)?;
        return Ok(SmokeBundle {
            texts: relation_texts_from_source(&text, config),
            labels: default_labels(),
        });
    }

    if let Some(corpus_id) = &config.corpus_id {
        let text = fs::read_to_string(docs_path(corpus_id)?)?;
        return Ok(SmokeBundle {
            texts: relation_texts_from_source(&text, config),
            labels: default_labels(),
        });
    }

    Ok(SmokeBundle {
        texts: vec![
            "Ryan joined Dynamis in New Rome. Augustus commanded Vulcan in Rust Town."
                .to_owned(),
            "Wyvern caused chaos in Campania while Renesco worked for Il Migliore."
                .to_owned(),
        ],
        labels: default_labels(),
    })
}

fn relation_texts_from_source(text: &str, config: &SmokeConfig) -> Vec<String> {
    let texts = match config.corpus_mode {
        CorpusMode::Excerpt => excerpt_chunks(text),
        CorpusMode::FullDocument => slice_text_windows(
            build_text_windows(
                text,
                config.audit_window_bytes,
                config.audit_overlap_bytes,
                config.audit_max_windows,
            ),
            config.audit_start_ratio,
            config.audit_end_ratio,
        )
        .into_iter()
        .map(|window| window.text)
        .collect(),
    };
    let filtered = texts
        .into_iter()
        .map(|value| compact_whitespace(&value))
        .filter(|value| !value.is_empty())
        .filter(|value| !looks_like_structural_window(value))
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        vec![compact_whitespace(text)]
    } else {
        filtered
    }
}

fn filter_relation_seed_output(output: SpanOutput, threshold: f32) -> SpanOutput {
    let texts = output.texts;
    let entities = output.entities;
    let spans = output
        .spans
        .into_iter()
        .map(|sequence| filter_relation_seed_sequence(sequence, threshold))
        .collect::<Vec<_>>();
    SpanOutput::new(texts, entities, spans)
}

fn filter_relation_seed_sequence(sequence: Vec<Span>, threshold: f32) -> Vec<Span> {
    let mut best_by_surface = BTreeMap::<String, Span>::new();
    for span in sequence {
        if !relation_seed_span_allowed(&span, threshold) {
            continue;
        }
        let key = normalize_key(span.text());
        match best_by_surface.get(&key) {
            Some(current) if !prefer_relation_seed_span(&span, current) => {}
            _ => {
                best_by_surface.insert(key, span);
            }
        }
    }
    best_by_surface.into_values().collect()
}

fn relation_seed_span_allowed(span: &Span, threshold: f32) -> bool {
    if span.probability() < threshold {
        return false;
    }
    let surface = span.text().trim();
    if surface.is_empty() || is_structural_surface(surface) || is_generic_relation_surface(surface) {
        return false;
    }
    if !surface.chars().any(|value| value.is_alphabetic()) {
        return false;
    }
    match span.class() {
        "person" | "organization" => true,
        "location" => {
            surface.split_whitespace().count() >= 2 || span.probability() >= threshold.max(0.8)
        }
        _ => false,
    }
}

fn prefer_relation_seed_span(candidate: &Span, current: &Span) -> bool {
    let probability_gap = candidate.probability() - current.probability();
    if probability_gap.abs() >= 0.05 {
        return probability_gap > 0.0;
    }
    let candidate_priority = relation_label_priority(candidate.class());
    let current_priority = relation_label_priority(current.class());
    if candidate_priority != current_priority {
        return candidate_priority > current_priority;
    }
    if candidate.text().len() != current.text().len() {
        return candidate.text().len() > current.text().len();
    }
    candidate.probability() > current.probability()
}

fn relation_label_priority(label: &str) -> i32 {
    match label {
        "person" => 6,
        "organization" => 5,
        "location" => 4,
        "event" => 3,
        "item" => 2,
        "concept" => 1,
        _ => 0,
    }
}

fn filter_relation_hits(
    relations: &[gliner::model::output::relation::Relation],
    seed_spans: &[Span],
    text: &str,
    threshold: f32,
) -> Vec<RelationHit> {
    let label_map = build_seed_label_map(seed_spans);
    let mut best_by_key = BTreeMap::<String, RelationHit>::new();
    for relation in relations {
        let family_floor = relation_family_threshold(relation.class(), threshold);
        if relation.probability() < family_floor {
            continue;
        }
        let subject = compact_whitespace(relation.subject());
        let object = compact_whitespace(relation.object());
        if subject.is_empty()
            || object.is_empty()
            || normalize_key(&subject) == normalize_key(&object)
            || is_structural_surface(&subject)
            || is_structural_surface(&object)
            || is_generic_relation_surface(&subject)
            || is_generic_relation_surface(&object)
        {
            continue;
        }
        if !relation_labels_supported(&label_map, relation.class(), &subject, &object) {
            continue;
        }
        if !relation_supported_by_text(text, relation.class(), &subject, &object) {
            continue;
        }
        let hit = RelationHit {
            subject: subject.clone(),
            relation: relation.class().to_owned(),
            object: object.clone(),
            probability: relation.probability(),
        };
        let key = format!(
            "{}::{}::{}",
            normalize_key(&subject),
            hit.relation,
            normalize_key(&object)
        );
        match best_by_key.get(&key) {
            Some(existing) if existing.probability >= hit.probability => {}
            _ => {
                best_by_key.insert(key, hit);
            }
        }
    }
    best_by_key.into_values().collect()
}

fn build_seed_label_map(seed_spans: &[Span]) -> BTreeMap<String, BTreeSet<String>> {
    let mut label_map = BTreeMap::<String, BTreeSet<String>>::new();
    for span in seed_spans {
        label_map
            .entry(normalize_key(span.text()))
            .or_default()
            .insert(span.class().to_owned());
    }
    label_map
}

fn relation_family_threshold(relation: &str, threshold: f32) -> f32 {
    match relation {
        "works_for" => threshold.max(0.45),
        "located_in" => threshold.max(0.45),
        _ => threshold,
    }
}

fn relation_labels_supported(
    label_map: &BTreeMap<String, BTreeSet<String>>,
    relation: &str,
    subject: &str,
    object: &str,
) -> bool {
    let subject_labels = label_map.get(&normalize_key(subject));
    let object_labels = label_map.get(&normalize_key(object));
    match relation {
        "works_for" => has_label(subject_labels, "person") && has_label(object_labels, "organization"),
        "located_in" => {
            (has_label(subject_labels, "person") || has_label(subject_labels, "organization"))
                && has_label(object_labels, "location")
        }
        "member_of" => has_label(subject_labels, "person") && has_label(object_labels, "organization"),
        _ => true,
    }
}

fn has_label(labels: Option<&BTreeSet<String>>, label: &str) -> bool {
    labels.is_some_and(|values| values.contains(label))
}

fn relation_supported_by_text(text: &str, relation: &str, subject: &str, object: &str) -> bool {
    match relation {
        "works_for" => text_between_supports(
            text,
            subject,
            object,
            &[
                " works for ",
                " worked for ",
                " working for ",
                " joins ",
                " joined ",
                " serves ",
                " served ",
                " serving ",
                " employed by ",
            ],
            false,
            96,
        ),
        "located_in" => text_between_supports(
            text,
            subject,
            object,
            &[" in ", " at ", " near ", " from ", " inside ", " within "],
            true,
            72,
        ),
        _ => true,
    }
}

fn text_between_supports(
    text: &str,
    subject: &str,
    object: &str,
    cues: &[&str],
    allow_reverse: bool,
    max_between_chars: usize,
) -> bool {
    let normalized_text = compact_whitespace(text).to_ascii_lowercase();
    let subject = compact_whitespace(subject).to_ascii_lowercase();
    let object = compact_whitespace(object).to_ascii_lowercase();
    text_between_contains_cue(&normalized_text, &subject, &object, cues, max_between_chars)
        || (allow_reverse
            && text_between_contains_cue(
                &normalized_text,
                &object,
                &subject,
                cues,
                max_between_chars,
            ))
}

fn text_between_contains_cue(
    text: &str,
    left: &str,
    right: &str,
    cues: &[&str],
    max_between_chars: usize,
) -> bool {
    let Some(left_start) = text.find(left) else {
        return false;
    };
    let search_start = left_start + left.len();
    let Some(relative_right_start) = text[search_start..].find(right) else {
        return false;
    };
    let right_start = search_start + relative_right_start;
    let between = &text[search_start..right_start];
    between.len() <= max_between_chars && cues.iter().any(|cue| between.contains(cue))
}

fn run_gliner_audit(config: &SmokeConfig) -> Result<AuditReport, Box<dyn Error + Send + Sync>> {
    let (raw_text, labels) = load_audit_input(config)?;
    let windows = slice_text_windows(
        build_text_windows(
        &raw_text,
        config.audit_window_bytes,
        config.audit_overlap_bytes,
        config.audit_max_windows,
        ),
        config.audit_start_ratio,
        config.audit_end_ratio,
    );
    let tokenizer_path = gliner_x_small_root().join("tokenizer.json");
    let model_path = gliner_x_small_root().join("onnx").join("model.onnx");
    let load_started = Instant::now();
    let model = GLiNER::<SpanMode>::new(
        Parameters::default().with_threshold(config.threshold),
        RuntimeParameters::default(),
        tokenizer_path
            .to_str()
            .ok_or("x-small tokenizer path contains invalid utf-8")?,
        model_path
            .to_str()
            .ok_or("x-small model path contains invalid utf-8")?,
    )?;
    let load_ms = load_started.elapsed().as_millis() as u64;

    let label_refs = labels.iter().map(String::as_str).collect::<Vec<_>>();
    let run_count = config.warm_runs.max(1);
    let batch_size = config.audit_batch_size.max(1);
    let mut scan_ms_runs = Vec::with_capacity(run_count);
    let mut final_window_reports = Vec::new();
    let mut final_surface_map = BTreeMap::<String, SurfaceAccumulator>::new();
    let mut final_mention_count = 0usize;

    for run_ix in 0..run_count {
        let started = Instant::now();
        let mut window_reports = Vec::with_capacity(windows.len());
        let mut surface_map = BTreeMap::<String, SurfaceAccumulator>::new();
        let mut mention_count = 0usize;

        for batch in windows.chunks(batch_size) {
            let text_refs = batch.iter().map(|window| window.text.as_str()).collect::<Vec<_>>();
            let batch_started = Instant::now();
            let output = model.inference(TextInput::from_str(&text_refs, &label_refs)?)?;
            let batch_scan_ms = batch_started.elapsed().as_millis() as u64;
            let per_window_scan_ms = if batch.is_empty() {
                0
            } else {
                batch_scan_ms / batch.len() as u64
            };

            for (window, sequence) in batch.iter().zip(output.spans.iter()) {
                let mut spans = Vec::new();
                let mut label_counts = BTreeMap::<String, usize>::new();
                for span in sequence
                    .iter()
                    .filter(|span| span.probability() >= config.threshold)
                {
                    let normalized = normalize_key(span.text());
                    let context = extract_span_context(&window.text, span.text());
                    mention_count += 1;
                    *label_counts.entry(span.class().to_owned()).or_default() += 1;
                    accumulate_surface(
                        &mut surface_map,
                        &normalized,
                        span.text(),
                        span.class(),
                        span.probability(),
                        window.index,
                        &context,
                    );
                    spans.push(AuditSpan {
                        window_index: window.index,
                        text: span.text().to_owned(),
                        normalized,
                        label: span.class().to_owned(),
                        probability: span.probability(),
                        repeated_count: 0,
                        context,
                    });
                }
                window_reports.push(AuditWindowReport {
                    index: window.index,
                    start: window.start,
                    end: window.end,
                    byte_len: window.end.saturating_sub(window.start),
                    scan_ms: per_window_scan_ms,
                    preview: preview_text(&window.text, 140),
                    mention_count: spans.len(),
                    label_counts,
                    spans,
                });
            }
        }

        scan_ms_runs.push(started.elapsed().as_millis() as u64);
        if run_ix + 1 == run_count {
            final_window_reports = window_reports;
            final_surface_map = surface_map;
            final_mention_count = mention_count;
        }
    }

    for window in &mut final_window_reports {
        for span in &mut window.spans {
            span.repeated_count = final_surface_map
                .get(&span.normalized)
                .map(|surface| surface.count)
                .unwrap_or_default();
        }
    }

    let mut repeated_surfaces =
        build_surface_reports(&final_surface_map, |surface| surface.count > 1);
    let mut label_conflicts =
        build_surface_reports(&final_surface_map, |surface| surface.label_counts.len() > 1);
    let mut codename_candidates =
        build_surface_reports(&final_surface_map, is_codename_candidate);
    let mut organization_candidates =
        build_surface_reports(&final_surface_map, is_organization_candidate);
    let mut demotion_candidates =
        build_surface_reports(&final_surface_map, is_demotion_candidate);
    truncate_reports(&mut repeated_surfaces, config.audit_surface_limit);
    truncate_reports(&mut label_conflicts, config.audit_surface_limit);
    truncate_reports(&mut codename_candidates, config.audit_surface_limit);
    truncate_reports(&mut organization_candidates, config.audit_surface_limit);
    truncate_reports(&mut demotion_candidates, config.audit_surface_limit);
    let covered_bytes = covered_byte_count(&windows).min(raw_text.len());

    Ok(AuditReport {
        threshold: config.threshold,
        load_ms,
        warm_runs: run_count,
        scan_ms_avg: average_ms(&scan_ms_runs),
        scan_ms_runs,
        input_mode: config.corpus_mode.as_str().to_owned(),
        labels,
        window_bytes: config.audit_window_bytes,
        overlap_bytes: config.audit_overlap_bytes,
        batch_size,
        slice_start_ratio: config.audit_start_ratio,
        slice_end_ratio: config.audit_end_ratio,
        total_bytes: raw_text.len(),
        covered_bytes,
        coverage_ratio: ratio(covered_bytes, raw_text.len()),
        window_count: final_window_reports.len(),
        mention_count: final_mention_count,
        repeated_surface_count: repeated_surfaces.len(),
        repeated_surfaces,
        label_conflict_count: label_conflicts.len(),
        label_conflicts,
        codename_candidate_count: codename_candidates.len(),
        codename_candidates,
        organization_candidate_count: organization_candidates.len(),
        organization_candidates,
        demotion_candidate_count: demotion_candidates.len(),
        demotion_candidates,
        windows: if config.audit_summary_only {
            Vec::new()
        } else {
            final_window_reports
        },
    })
}

fn build_model_lane(
    name: &str,
    mode: &str,
    load_ms: u64,
    warm_runs: usize,
    warm_ms_runs: Vec<u64>,
    text_count: usize,
    spans: Vec<RaceSpan>,
) -> LaneReport {
    let mut label_counts = BTreeMap::<String, usize>::new();
    for span in &spans {
        *label_counts.entry(span.label.clone()).or_default() += 1;
    }
    LaneReport {
        name: name.to_owned(),
        mode: mode.to_owned(),
        load_ms,
        warm_runs,
        warm_ms_avg: average_ms(&warm_ms_runs),
        warm_ms_runs,
        text_count,
        mention_count: spans.len(),
        label_counts,
        named_count: None,
        nominal_count: None,
        pronoun_count: None,
        discovery_count: None,
        spans,
    }
}

fn overlap_report(native_spans: &[RaceSpan], challenger_spans: &[RaceSpan]) -> OverlapReport {
    let native_unlabeled = native_spans
        .iter()
        .map(unlabeled_overlap_key)
        .collect::<BTreeSet<_>>();
    let challenger_unlabeled = challenger_spans
        .iter()
        .map(unlabeled_overlap_key)
        .collect::<BTreeSet<_>>();
    let native_labeled = native_spans
        .iter()
        .map(labeled_overlap_key)
        .collect::<BTreeSet<_>>();
    let challenger_labeled = challenger_spans
        .iter()
        .map(labeled_overlap_key)
        .collect::<BTreeSet<_>>();

    let unlabeled_overlap = native_unlabeled.intersection(&challenger_unlabeled).count();
    let exact_label_overlap = native_labeled.intersection(&challenger_labeled).count();
    let native_only = native_unlabeled.len().saturating_sub(unlabeled_overlap);
    let challenger_only = challenger_unlabeled.len().saturating_sub(unlabeled_overlap);
    let native_recall_vs_challenger = ratio(unlabeled_overlap, challenger_unlabeled.len());
    let exact_label_agreement = ratio(exact_label_overlap, challenger_labeled.len());

    OverlapReport {
        unlabeled_overlap,
        exact_label_overlap,
        native_only,
        challenger_only,
        native_recall_vs_challenger,
        exact_label_agreement,
    }
}

fn native_span(sequence: usize, mention: &NativeNerMention) -> RaceSpan {
    RaceSpan {
        sequence,
        text: mention.surface.clone(),
        normalized: mention.normalized.clone(),
        label: mention.label.clone(),
        probability: mention.confidence,
    }
}

fn extracted_span(span: &Span) -> RaceSpan {
    RaceSpan {
        sequence: span.sequence(),
        text: span.text().to_owned(),
        normalized: normalize_key(span.text()),
        label: span.class().to_owned(),
        probability: span.probability(),
    }
}

fn unlabeled_overlap_key(span: &RaceSpan) -> String {
    format!("{}::{}", span.sequence, span.normalized)
}

fn labeled_overlap_key(span: &RaceSpan) -> String {
    format!("{}::{}::{}", span.sequence, span.normalized, span.label)
}

fn average_ms(values: &[u64]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().copied().sum::<u64>() as f32 / values.len() as f32
    }
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn load_input(config: &SmokeConfig) -> Result<SmokeBundle, Box<dyn Error + Send + Sync>> {
    if let Some(path) = &config.text_path {
        let text = fs::read_to_string(path)?;
        return Ok(SmokeBundle {
            texts: vec![compact_whitespace(&text)],
            labels: default_labels(),
        });
    }

    if let Some(corpus_id) = &config.corpus_id {
        let path = docs_path(corpus_id)?;
        let text = fs::read_to_string(path)?;
        let texts = match config.corpus_mode {
            CorpusMode::FullDocument => vec![compact_whitespace(&text)],
            CorpusMode::Excerpt => excerpt_chunks(&text),
        };
        return Ok(SmokeBundle {
            texts: if texts.is_empty() {
                vec![compact_whitespace(&text)]
            } else {
                texts
            },
            labels: default_labels(),
        });
    }

    Ok(SmokeBundle {
        texts: vec![
            "Bill Gates co-founded Microsoft in Albuquerque before the company moved to Washington."
                .to_owned(),
            "Steve Jobs founded Apple and led the company from Cupertino.".to_owned(),
            "Satya Nadella works for Microsoft in Redmond.".to_owned(),
        ],
        labels: default_labels(),
    })
}

fn load_audit_input(
    config: &SmokeConfig,
) -> Result<(String, Vec<String>), Box<dyn Error + Send + Sync>> {
    if let Some(path) = &config.text_path {
        return Ok((fs::read_to_string(path)?, default_labels()));
    }
    if let Some(corpus_id) = &config.corpus_id {
        return Ok((fs::read_to_string(docs_path(corpus_id)?)?, default_labels()));
    }
    Err("audit mode requires --input or --corpus".into())
}

fn accumulate_surface(
    surface_map: &mut BTreeMap<String, SurfaceAccumulator>,
    normalized: &str,
    surface: &str,
    label: &str,
    probability: f32,
    window_index: usize,
    context: &str,
) {
    let entry = surface_map
        .entry(normalized.to_owned())
        .or_insert_with(|| SurfaceAccumulator {
            surface: surface.to_owned(),
            ..SurfaceAccumulator::default()
        });
    entry.count += 1;
    entry.total_probability += probability;
    *entry.label_counts.entry(label.to_owned()).or_default() += 1;
    entry.windows.insert(window_index);
    if entry.sample_contexts.len() < 3 && !entry.sample_contexts.iter().any(|value| value == context)
    {
        entry.sample_contexts.push(context.to_owned());
    }
}

fn build_surface_reports<F>(
    surface_map: &BTreeMap<String, SurfaceAccumulator>,
    predicate: F,
) -> Vec<AuditSurfaceReport>
where
    F: Fn(&SurfaceAccumulator) -> bool,
{
    let mut reports = surface_map
        .iter()
        .filter_map(|(normalized, surface)| {
            predicate(surface).then(|| build_surface_report(normalized, surface))
        })
        .collect::<Vec<_>>();
    reports.sort_by(|left, right| {
        right
            .count
            .cmp(&left.count)
            .then_with(|| left.surface.cmp(&right.surface))
    });
    reports
}

fn truncate_reports(reports: &mut Vec<AuditSurfaceReport>, limit: usize) {
    if limit > 0 && reports.len() > limit {
        reports.truncate(limit);
    }
}

fn build_surface_report(normalized: &str, surface: &SurfaceAccumulator) -> AuditSurfaceReport {
    let flags = surface_flags(surface);
    AuditSurfaceReport {
        normalized: normalized.to_owned(),
        surface: surface.surface.clone(),
        count: surface.count,
        avg_probability: if surface.count == 0 {
            0.0
        } else {
            surface.total_probability / surface.count as f32
        },
        labels: sorted_labels(&surface.label_counts),
        label_counts: surface.label_counts.clone(),
        windows: surface.windows.iter().copied().collect(),
        suggested_label: suggested_label(surface, &flags),
        flags,
        sample_contexts: surface.sample_contexts.clone(),
    }
}

fn sorted_labels(label_counts: &BTreeMap<String, usize>) -> Vec<String> {
    let mut labels = label_counts.iter().collect::<Vec<_>>();
    labels.sort_by(|left, right| right.1.cmp(left.1).then_with(|| left.0.cmp(right.0)));
    labels.into_iter().map(|(label, _)| label.clone()).collect()
}

fn surface_flags(surface: &SurfaceAccumulator) -> Vec<String> {
    let mut flags = Vec::new();
    if surface.label_counts.len() > 1 {
        flags.push("label-conflict".to_owned());
    }
    if is_codename_candidate(surface) {
        flags.push("codename-candidate".to_owned());
    }
    if is_organization_candidate(surface) {
        flags.push("organization-candidate".to_owned());
    }
    if is_demotion_candidate(surface) {
        flags.push("demotion-candidate".to_owned());
    }
    flags
}

fn suggested_label(surface: &SurfaceAccumulator, flags: &[String]) -> Option<String> {
    if is_structural_surface(&surface.surface) {
        return None;
    }
    if flags.iter().any(|flag| flag == "organization-candidate") {
        return Some("organization".to_owned());
    }
    if flags.iter().any(|flag| flag == "codename-candidate") {
        return Some("person".to_owned());
    }
    if flags.iter().any(|flag| flag == "demotion-candidate") {
        return None;
    }
    dominant_label(surface).and_then(|(label, _, confidence)| (confidence >= 0.6).then(|| label))
}

fn is_codename_candidate(surface: &SurfaceAccumulator) -> bool {
    if surface.count < 2 {
        return false;
    }
    if is_structural_surface(&surface.surface)
        || is_organization_candidate(surface)
        || is_demotion_candidate(surface)
    {
        return false;
    }
    if dominant_label_name(surface) == Some("location") && label_count(surface, "person") == 0 {
        return false;
    }
    has_person_like_surface_shape(&surface.surface) && has_direct_identity_cue(surface)
}

fn is_organization_candidate(surface: &SurfaceAccumulator) -> bool {
    if is_structural_surface(&surface.surface) {
        return false;
    }
    let normalized = normalize_key(&surface.surface);
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() || has_generic_organization_lead(&normalized) {
        return false;
    }
    let org_markers = [
        "gang",
        "security",
        "corporation",
        "corporations",
        "group",
        "groups",
        "team",
        "teams",
        "family",
        "division",
        "divisions",
        "syndicate",
        "hq",
        "council",
        "league",
        "committee",
    ];
    let has_marker_suffix = tokens.len() >= 2
        && tokens
            .last()
            .is_some_and(|tail| org_markers.contains(tail));
    let direct_cue = has_direct_organization_cue(surface);
    if direct_cue {
        return true;
    }
    if has_marker_suffix && starts_like_named_surface(&surface.surface) {
        return true;
    }
    dominant_label(surface).is_some_and(|(label, count, confidence)| {
        label == "organization"
            && count >= 2
            && confidence >= 0.75
            && starts_like_named_surface(&surface.surface)
    })
}

fn is_demotion_candidate(surface: &SurfaceAccumulator) -> bool {
    if is_structural_surface(&surface.surface) {
        return false;
    }
    if surface.count < 2 {
        return false;
    }
    if dominant_label_name(surface) == Some("person") {
        return false;
    }
    let normalized = normalize_key(&surface.surface);
    let generic_heads = [
        "hero",
        "heroes",
        "villain",
        "villains",
        "monster",
        "monsters",
        "criminal",
        "criminals",
        "psycho",
        "psychos",
        "genius",
        "geniuses",
    ];
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();
    let head = tokens.last().copied().unwrap_or_default();
    generic_heads.contains(&normalized.as_str()) || generic_heads.contains(&head)
}

fn dominant_label(surface: &SurfaceAccumulator) -> Option<(String, usize, f32)> {
    let (label, count) = surface
        .label_counts
        .iter()
        .max_by(|left, right| left.1.cmp(right.1).then_with(|| right.0.cmp(left.0)))?;
    Some((
        label.clone(),
        *count,
        (*count as f32) / (surface.count.max(1) as f32),
    ))
}

fn dominant_label_name(surface: &SurfaceAccumulator) -> Option<&str> {
    surface
        .label_counts
        .iter()
        .max_by(|left, right| left.1.cmp(right.1).then_with(|| right.0.cmp(left.0)))
        .map(|(label, _)| label.as_str())
}

fn label_count(surface: &SurfaceAccumulator, label: &str) -> usize {
    surface.label_counts.get(label).copied().unwrap_or_default()
}

fn is_structural_surface(surface: &str) -> bool {
    let normalized = normalize_key(surface);
    normalized == "table of contents"
        || normalized.starts_with("chapter ")
        || normalized.starts_with("part ")
        || normalized.starts_with("appendix ")
        || normalized.starts_with("section ")
}

fn looks_like_structural_window(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("# ")
        || trimmed.starts_with("## ")
        || is_structural_surface(trimmed)
        || normalize_key(trimmed).starts_with("table of contents")
}

fn is_generic_relation_surface(surface: &str) -> bool {
    let normalized = normalize_key(surface);
    let generic = [
        "hero",
        "heroes",
        "villain",
        "villains",
        "monster",
        "monsters",
        "criminal",
        "criminals",
        "psycho",
        "psychos",
        "genius",
        "geniuses",
        "team",
        "group",
        "member",
        "members",
        "person",
        "people",
        "man",
        "woman",
        "city",
        "town",
    ];
    generic.contains(&normalized.as_str())
}

fn has_generic_organization_lead(normalized: &str) -> bool {
    let generic_leads = [
        "company",
        "organization",
        "group",
        "groups",
        "division",
        "divisions",
        "team",
        "teams",
        "family",
        "league",
        "committee",
        "council",
        "security",
        "courier",
        "member",
        "members",
        "player",
        "players",
    ];
    normalized
        .split_whitespace()
        .next()
        .is_some_and(|lead| generic_leads.contains(&lead))
}

fn has_person_like_surface_shape(surface: &str) -> bool {
    let tokens = surface.split_whitespace().collect::<Vec<_>>();
    if tokens.is_empty() || tokens.len() > 4 {
        return false;
    }
    if surface.starts_with("The ") {
        return false;
    }
    tokens.iter().all(|token| {
        token.chars().any(|value| value.is_alphabetic())
            && token
                .chars()
                .next()
                .is_some_and(|value| value.is_uppercase())
    })
}

fn starts_like_named_surface(surface: &str) -> bool {
    surface
        .chars()
        .find(|value| value.is_alphabetic())
        .is_some_and(|value| value.is_uppercase())
}

fn has_direct_identity_cue(surface: &SurfaceAccumulator) -> bool {
    let lowered_surface = surface.surface.to_ascii_lowercase();
    let direct_patterns = [
        format!("i'm {lowered_surface}"),
        format!("i am {lowered_surface}"),
        format!("name is {lowered_surface}"),
        format!("this is {lowered_surface}"),
        format!("alias {lowered_surface}"),
        format!("call me {lowered_surface}"),
        format!("{lowered_surface} costume"),
    ];
    surface.sample_contexts.iter().any(|context| {
        let lowered = context.to_ascii_lowercase();
        direct_patterns
            .iter()
            .any(|pattern| lowered.contains(pattern))
    })
}

fn has_direct_organization_cue(surface: &SurfaceAccumulator) -> bool {
    let lowered_surface = surface.surface.to_ascii_lowercase();
    let direct_patterns = [
        format!("group called {lowered_surface}"),
        format!("organization called {lowered_surface}"),
        format!("members of {lowered_surface}"),
        format!("part of {lowered_surface}"),
        format!("represent the {lowered_surface}"),
        format!("{lowered_surface} organization"),
        format!("{lowered_surface} corporation"),
        format!("{lowered_surface} division"),
        format!("{lowered_surface} payroll"),
    ];
    surface.sample_contexts.iter().any(|context| {
        let lowered = context.to_ascii_lowercase();
        direct_patterns
            .iter()
            .any(|pattern| lowered.contains(pattern))
    })
}

fn gliner_x_small_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("gliner-x-small")
}

fn gliner_multitask_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("models")
        .join("gliner-multitask-large-v0.5")
}

fn docs_path(corpus_id: &str) -> Result<PathBuf, Box<dyn Error + Send + Sync>> {
    let filename = match corpus_id {
        "perfect_run" => "perfect_run.md",
        "shortrun" => "shortrun.md",
        other => return Err(format!("unsupported corpus id: {other}").into()),
    };
    let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .ok_or("failed to resolve rust-native root")?
        .parent()
        .ok_or("failed to resolve repository root")?;
    Ok(repo_root.join("docs").join(filename))
}

fn parse_f32_arg(args: &[String], flag: &str) -> Option<f32> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].parse::<f32>().ok()))
        .flatten()
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].parse::<usize>().ok()))
        .flatten()
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn parse_corpus_mode_arg(args: &[String], flag: &str) -> Option<CorpusMode> {
    args.windows(2).find_map(|window| {
        if window[0] != flag {
            return None;
        }
        match window[1].as_str() {
            "full" | "full-document" => Some(CorpusMode::FullDocument),
            "excerpt" => Some(CorpusMode::Excerpt),
            _ => None,
        }
    })
}

fn excerpt_chunks(text: &str) -> Vec<String> {
    // Skip title/front-matter fragments so excerpt mode samples narrative text.
    let chunks = split_paragraphs(text)
        .into_iter()
        .filter(|chunk| is_meaningful_excerpt_chunk(chunk))
        .take(2)
        .map(|chunk| compact_whitespace(&chunk))
        .collect::<Vec<_>>();
    if chunks.is_empty() {
        split_paragraphs(text)
            .into_iter()
            .take(2)
            .map(|chunk| compact_whitespace(&chunk))
            .collect()
    } else {
        chunks
    }
}

fn split_paragraphs(text: &str) -> Vec<String> {
    text.replace("\r\n", "\n")
        .replace('\r', "\n")
        .split("\n\n")
        .map(str::trim)
        .filter(|chunk| !chunk.is_empty())
        .map(str::to_owned)
        .collect()
}

fn build_text_windows(
    text: &str,
    window_bytes: usize,
    overlap_bytes: usize,
    max_windows: usize,
) -> Vec<TextWindow> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    if normalized.is_empty() {
        return Vec::new();
    }
    let max_windows = if max_windows == 0 {
        usize::MAX
    } else {
        max_windows
    };

    let mut windows = Vec::new();
    let mut start = 0usize;
    let min_window_bytes = overlap_bytes.saturating_add(128).max(512);
    while start < normalized.len() && windows.len() < max_windows {
        let tentative_end = (start + window_bytes).min(normalized.len());
        let end = adjust_window_end(&normalized, start, tentative_end, min_window_bytes);
        let end = if end <= start {
            normalized.len()
        } else {
            end
        };
        let window_text = compact_whitespace(&normalized[start..end]);
        windows.push(TextWindow {
            index: windows.len(),
            start,
            end,
            text: window_text,
        });
        if end >= normalized.len() {
            break;
        }
        let width = end.saturating_sub(start);
        let effective_overlap = overlap_bytes.min(width.saturating_sub(1));
        let next_start = end.saturating_sub(effective_overlap);
        start = align_forward_to_char_boundary(&normalized, next_start);
    }
    windows
}

fn covered_byte_count(windows: &[TextWindow]) -> usize {
    if windows.is_empty() {
        return 0;
    }
    let mut ranges = windows
        .iter()
        .map(|window| (window.start, window.end))
        .collect::<Vec<_>>();
    ranges.sort_unstable_by_key(|range| range.0);
    let mut covered = 0usize;
    let mut current_start = ranges[0].0;
    let mut current_end = ranges[0].1;
    for (start, end) in ranges.into_iter().skip(1) {
        if start <= current_end {
            current_end = current_end.max(end);
        } else {
            covered += current_end.saturating_sub(current_start);
            current_start = start;
            current_end = end;
        }
    }
    covered + current_end.saturating_sub(current_start)
}

fn slice_text_windows(
    windows: Vec<TextWindow>,
    start_ratio: f32,
    end_ratio: f32,
) -> Vec<TextWindow> {
    if windows.is_empty() {
        return windows;
    }
    let len = windows.len();
    let start = ((len as f32) * start_ratio.clamp(0.0, 1.0)).floor() as usize;
    let end = ((len as f32) * end_ratio.clamp(0.0, 1.0)).ceil() as usize;
    let start = start.min(len);
    let end = end.max(start).min(len);
    windows.into_iter().skip(start).take(end - start).collect()
}

fn adjust_window_end(
    text: &str,
    start: usize,
    tentative_end: usize,
    min_window_bytes: usize,
) -> usize {
    let aligned_end = align_backward_to_char_boundary(text, tentative_end);
    if aligned_end >= text.len() {
        return text.len();
    }
    let slice = &text[start..aligned_end];
    let paragraph_break = slice.rfind("\n\n").map(|index| start + index);
    if let Some(index) = paragraph_break.filter(|index| *index >= start + min_window_bytes) {
        return index;
    }
    let sentence_break = slice
        .char_indices()
        .rev()
        .find_map(|(index, ch)| matches!(ch, '.' | '!' | '?').then_some(start + index + ch.len_utf8()));
    if let Some(index) = sentence_break.filter(|index| *index >= start + min_window_bytes) {
        return index;
    }
    aligned_end
}

fn align_forward_to_char_boundary(text: &str, mut index: usize) -> usize {
    while index < text.len() && !text.is_char_boundary(index) {
        index += 1;
    }
    index
}

fn align_backward_to_char_boundary(text: &str, mut index: usize) -> usize {
    while index > 0 && !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn extract_span_context(text: &str, span_text: &str) -> String {
    let compact = compact_whitespace(text);
    if let Some(start) = compact.find(span_text) {
        let start_char = compact[..start].chars().count();
        let end_char = compact[..start + span_text.len()].chars().count();
        let context_start = start_char.saturating_sub(72);
        let context_end = end_char + 72;
        return compact
            .chars()
            .skip(context_start)
            .take(context_end.saturating_sub(context_start))
            .collect::<String>()
            .trim()
            .to_owned();
    }
    preview_text(&compact, 144)
}

fn preview_text(text: &str, max_chars: usize) -> String {
    let compact = compact_whitespace(text);
    let preview = compact.chars().take(max_chars).collect::<String>();
    if compact.chars().count() > max_chars {
        format!("{preview}...")
    } else {
        preview
    }
}

fn is_meaningful_excerpt_chunk(chunk: &str) -> bool {
    let compact = compact_whitespace(chunk);
    compact.len() >= 64 || compact.contains(". ") || compact.contains(": ")
}

fn compact_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn normalize_key(text: &str) -> String {
    let mut value = String::with_capacity(text.len());
    let mut prev_space = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lower in ch.to_lowercase() {
                value.push(lower);
            }
            prev_space = false;
        } else if !prev_space && !value.is_empty() {
            value.push(' ');
            prev_space = true;
        }
    }
    while value.ends_with(' ') {
        value.pop();
    }
    value
}

fn default_labels() -> Vec<String> {
    vec![
        "person".to_owned(),
        "organization".to_owned(),
        "location".to_owned(),
        "event".to_owned(),
        "item".to_owned(),
        "concept".to_owned(),
    ]
}

fn default_relation_schema() -> RelationSchema {
    let mut schema = RelationSchema::new();
    schema.push_with_allowed_labels("member_of", &["person", "organization"], &["organization"]);
    schema.push_with_allowed_labels("works_for", &["person"], &["organization"]);
    schema.push_with_allowed_labels("serves", &["person", "organization"], &["organization"]);
    schema.push_with_allowed_labels(
        "commands",
        &["person", "organization"],
        &["person", "organization"],
    );
    schema.push_with_allowed_labels(
        "protects",
        &["person", "organization"],
        &["person", "organization", "location"],
    );
    schema.push_with_allowed_labels(
        "located_in",
        &["person", "organization"],
        &["location"],
    );
    schema
}
