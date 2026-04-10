use std::path::PathBuf;
use std::time::Instant;

use phoenix_rel_post::{GlinerRelexLabel, GlinerRelexModel, GlinerRelexPredictOptions};
use serde::Serialize;

#[derive(Debug)]
struct BenchConfig {
    model_root: PathBuf,
    text: String,
    entity_labels: Vec<GlinerRelexLabel>,
    relation_labels: Vec<GlinerRelexLabel>,
    options: GlinerRelexPredictOptions,
    warmups: usize,
    iterations: usize,
    json: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchReport {
    ok: bool,
    model_path: String,
    seq_len: usize,
    entity_count: usize,
    relation_count: usize,
    active_relation_pairs: usize,
    warmups: usize,
    iterations: usize,
    load_ms: u64,
    min_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    runs_ms: Vec<f64>,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    let load_started = Instant::now();
    let model = GlinerRelexModel::load(&config.model_root)
        .map_err(|error| format!("failed to load model: {error}"))?;
    let load_ms = load_started.elapsed().as_millis() as u64;

    let mut last_prediction = None;
    for _ in 0..config.warmups {
        last_prediction = Some(run_once(&model, &config)?);
    }

    let mut runs_ms = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let started = Instant::now();
        last_prediction = Some(run_once(&model, &config)?);
        runs_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let prediction = last_prediction.ok_or_else(|| "no benchmark iterations ran".to_owned())?;
    let stats = stats(&runs_ms)?;
    let report = BenchReport {
        ok: true,
        model_path: model.metadata().model_path.clone(),
        seq_len: prediction.seq_len,
        entity_count: prediction.entities.len(),
        relation_count: prediction.relations.len(),
        active_relation_pairs: prediction.active_relation_pairs,
        warmups: config.warmups,
        iterations: config.iterations,
        load_ms,
        min_ms: stats.min,
        mean_ms: stats.mean,
        median_ms: stats.median,
        p95_ms: stats.p95,
        max_ms: stats.max,
        runs_ms,
    };

    if config.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|error| format!("failed to render JSON: {error}"))?
        );
    } else {
        println!("model: {}", report.model_path);
        println!("loadMs: {}", report.load_ms);
        println!(
            "runs: n={} warmups={} min={:.3} mean={:.3} median={:.3} p95={:.3} max={:.3}",
            report.iterations,
            report.warmups,
            report.min_ms,
            report.mean_ms,
            report.median_ms,
            report.p95_ms,
            report.max_ms
        );
        println!(
            "seqLen={} entities={} relations={} activePairs={}",
            report.seq_len,
            report.entity_count,
            report.relation_count,
            report.active_relation_pairs
        );
    }
    Ok(())
}

fn run_once(
    model: &GlinerRelexModel,
    config: &BenchConfig,
) -> Result<phoenix_rel_post::GlinerRelexPrediction, String> {
    model
        .predict(
            &config.text,
            &config.entity_labels,
            &config.relation_labels,
            &config.options,
        )
        .map_err(|error| format!("inference failed: {error}"))
}

#[derive(Debug)]
struct Stats {
    min: f64,
    mean: f64,
    median: f64,
    p95: f64,
    max: f64,
}

fn stats(values: &[f64]) -> Result<Stats, String> {
    if values.is_empty() {
        return Err("iterations must be greater than zero".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    Ok(Stats {
        min: sorted[0],
        mean,
        median: percentile(&sorted, 0.50),
        p95: percentile(&sorted, 0.95),
        max: *sorted.last().unwrap_or(&sorted[0]),
    })
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn parse_args(args: &[String]) -> Result<BenchConfig, String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Err(usage());
    }
    let model_root = parse_required_path(args, "--model-root")?;
    let text = parse_string_arg(args, "--text").unwrap_or_else(default_text);
    let entity_labels = parse_labels(
        parse_string_arg(args, "--entity-labels"),
        vec!["location", "person", "date", "structure"],
    );
    let relation_labels = parse_labels(
        parse_string_arg(args, "--relation-labels"),
        vec!["located in", "designed by", "completed in"],
    );
    let iterations = parse_usize_arg(args, "--iterations").unwrap_or(20);
    if iterations == 0 {
        return Err("--iterations must be greater than zero".to_owned());
    }
    Ok(BenchConfig {
        model_root,
        text,
        entity_labels,
        relation_labels,
        options: GlinerRelexPredictOptions {
            threshold: parse_f32_arg(args, "--threshold").unwrap_or(0.3),
            relation_threshold: parse_f32_arg(args, "--relation-threshold").unwrap_or(0.5),
            flat_ner: args.iter().any(|arg| arg == "--flat-ner"),
            multi_label: args.iter().any(|arg| arg == "--multi-label"),
        },
        warmups: parse_usize_arg(args, "--warmups").unwrap_or(3),
        iterations,
        json: args.iter().any(|arg| arg == "--json"),
    })
}

fn parse_labels(inline: Option<String>, fallback: Vec<&str>) -> Vec<GlinerRelexLabel> {
    inline
        .map(|value| {
            value
                .split(',')
                .filter_map(parse_label_line)
                .collect::<Vec<_>>()
        })
        .filter(|labels| !labels.is_empty())
        .unwrap_or_else(|| {
            fallback
                .into_iter()
                .map(|label| GlinerRelexLabel {
                    label: label.to_owned(),
                    description: None,
                })
                .collect()
        })
}

fn parse_label_line(value: &str) -> Option<GlinerRelexLabel> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| GlinerRelexLabel {
        label: trimmed.to_owned(),
        description: None,
    })
}

fn parse_required_path(args: &[String], flag: &str) -> Result<PathBuf, String> {
    parse_string_arg(args, flag)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} is required\n\n{}", usage()))
}

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].clone())
}

fn parse_f32_arg(args: &[String], flag: &str) -> Option<f32> {
    parse_string_arg(args, flag).and_then(|value| value.parse::<f32>().ok())
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    parse_string_arg(args, flag).and_then(|value| value.parse::<usize>().ok())
}

fn default_text() -> String {
    "The Eiffel Tower, located in Paris, France, was designed by engineer Gustave Eiffel and completed in 1889.".to_owned()
}

fn usage() -> String {
    [
        "Usage:",
        "  phoenix-gliner-relex-bench --model-root <DIR> [options]",
        "",
        "Options:",
        "  --text <TEXT>",
        "  --entity-labels <A,B,C>",
        "  --relation-labels <A,B,C>",
        "  --threshold <FLOAT>",
        "  --relation-threshold <FLOAT>",
        "  --warmups <N>       Default: 3",
        "  --iterations <N>    Default: 20",
        "  --json",
    ]
    .join("\n")
}
