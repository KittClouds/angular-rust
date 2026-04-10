use std::cmp::Ordering;
use std::path::PathBuf;
use std::time::Instant;

use phoenix_rel_post::{
    GlinerBiModel, GlinerBiOverlapPolicy, GlinerBiPredictOptions, GlinerBiPrediction,
};
use serde::Serialize;

#[derive(Debug)]
struct BenchConfig {
    model_root: PathBuf,
    text: String,
    labels: Vec<String>,
    options: GlinerBiPredictOptions,
    warmups: usize,
    iterations: usize,
    json: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchReport {
    ok: bool,
    model_path: String,
    label_count: usize,
    warmups: usize,
    iterations: usize,
    load_ms: u64,
    label_prep_ms: u64,
    prepared: BenchVariantReport,
    cached: BenchVariantReport,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BenchVariantReport {
    name: String,
    prediction_count: usize,
    min_ms: f64,
    mean_ms: f64,
    median_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    runs_ms: Vec<f64>,
}

#[derive(Debug)]
struct Stats {
    min: f64,
    mean: f64,
    median: f64,
    p95: f64,
    max: f64,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    let load_started = Instant::now();
    let model = GlinerBiModel::load(&config.model_root)
        .map_err(|error| format!("failed to load model: {error}"))?;
    let load_ms = load_started.elapsed().as_millis() as u64;

    let prep_started = Instant::now();
    let label_set = model
        .prepare_labels(&config.labels)
        .map_err(|error| format!("failed to prepare labels: {error}"))?;
    let label_prep_ms = prep_started.elapsed().as_millis() as u64;

    let prepared = bench_variant("prepared", config.warmups, config.iterations, || {
        model.predict_with_label_set(&config.text, &label_set, &config.options)
    })?;
    let cached = bench_variant("cached", config.warmups, config.iterations, || {
        model.predict_with_options(&config.text, &config.labels, &config.options)
    })?;

    let report = BenchReport {
        ok: true,
        model_path: model.metadata().model_path.clone(),
        label_count: config.labels.len(),
        warmups: config.warmups,
        iterations: config.iterations,
        load_ms,
        label_prep_ms,
        prepared,
        cached,
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
        println!("labelPrepMs: {}", report.label_prep_ms);
        println!("labels: {}", report.label_count);
        println!(
            "prepared: n={} warmups={} predictions={} min={:.3} mean={:.3} median={:.3} p95={:.3} max={:.3}",
            report.iterations,
            report.warmups,
            report.prepared.prediction_count,
            report.prepared.min_ms,
            report.prepared.mean_ms,
            report.prepared.median_ms,
            report.prepared.p95_ms,
            report.prepared.max_ms
        );
        println!(
            "cached:   n={} warmups={} predictions={} min={:.3} mean={:.3} median={:.3} p95={:.3} max={:.3}",
            report.iterations,
            report.warmups,
            report.cached.prediction_count,
            report.cached.min_ms,
            report.cached.mean_ms,
            report.cached.median_ms,
            report.cached.p95_ms,
            report.cached.max_ms
        );
    }
    Ok(())
}

fn bench_variant<F>(
    name: &str,
    warmups: usize,
    iterations: usize,
    mut run: F,
) -> Result<BenchVariantReport, String>
where
    F: FnMut() -> Result<Vec<GlinerBiPrediction>, phoenix_rel_post::GlinerBiError>,
{
    let mut last_prediction = None;
    for _ in 0..warmups {
        last_prediction = Some(run().map_err(|error| format!("warmup failed: {error}"))?);
    }

    let mut runs_ms = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let started = Instant::now();
        last_prediction = Some(run().map_err(|error| format!("inference failed: {error}"))?);
        runs_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let prediction_count = last_prediction.unwrap_or_default().len();
    let stats = stats(&runs_ms)?;
    Ok(BenchVariantReport {
        name: name.to_owned(),
        prediction_count,
        min_ms: stats.min,
        mean_ms: stats.mean,
        median_ms: stats.median,
        p95_ms: stats.p95,
        max_ms: stats.max,
        runs_ms,
    })
}

fn stats(values: &[f64]) -> Result<Stats, String> {
    if values.is_empty() {
        return Err("iterations must be greater than zero".to_owned());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
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

    let mut model_root = None::<PathBuf>;
    let mut text = None::<String>;
    let mut labels = Vec::new();
    let mut threshold = 0.5f32;
    let mut overlap_policy = GlinerBiOverlapPolicy::HighestScore;
    let mut warmups = 5usize;
    let mut iterations = 25usize;
    let mut json = false;

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--model-root" | "--model" => {
                model_root = Some(PathBuf::from(next_arg(args, &mut index, "--model-root")?));
            }
            "--text" => text = Some(next_arg(args, &mut index, "--text")?.to_owned()),
            "--labels" | "--label" => {
                push_csv(&mut labels, next_arg(args, &mut index, "--labels")?)
            }
            "--threshold" => {
                let value = next_arg(args, &mut index, "--threshold")?;
                threshold = value
                    .parse::<f32>()
                    .map_err(|error| format!("invalid --threshold '{value}': {error}"))?;
            }
            "--overlap-policy" => {
                overlap_policy =
                    GlinerBiOverlapPolicy::parse(next_arg(args, &mut index, "--overlap-policy")?)?;
            }
            "--warmups" => {
                let value = next_arg(args, &mut index, "--warmups")?;
                warmups = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --warmups '{value}': {error}"))?;
            }
            "--iterations" => {
                let value = next_arg(args, &mut index, "--iterations")?;
                iterations = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --iterations '{value}': {error}"))?;
            }
            "--json" => json = true,
            unknown => return Err(format!("unrecognized argument '{unknown}'\n\n{}", usage())),
        }
        index += 1;
    }

    if iterations == 0 {
        return Err("--iterations must be greater than zero".to_owned());
    }

    Ok(BenchConfig {
        model_root: model_root.unwrap_or_else(|| PathBuf::from("gliner-bi-onnx")),
        text: text.unwrap_or_else(default_text),
        labels: if labels.is_empty() {
            default_labels()
        } else {
            labels
        },
        options: GlinerBiPredictOptions {
            threshold,
            overlap_policy,
        },
        warmups,
        iterations,
        json,
    })
}

fn next_arg<'a>(args: &'a [String], index: &mut usize, flag: &str) -> Result<&'a str, String> {
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn push_csv(out: &mut Vec<String>, value: &str) {
    for label in value.split(',') {
        let trimmed = label.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_owned());
        }
    }
}

fn default_text() -> String {
    "Microsoft was founded by Bill Gates and Paul Allen in Albuquerque before moving to Redmond."
        .to_owned()
}

fn default_labels() -> Vec<String> {
    vec![
        "Person".to_owned(),
        "Organization".to_owned(),
        "Location".to_owned(),
        "Date".to_owned(),
    ]
}

fn usage() -> String {
    [
        "Usage:",
        "  phoenix-gliner-bi-bench --model-root <DIR> [options]",
        "",
        "Options:",
        "  --text <TEXT>",
        "  --labels <A,B,C>",
        "  --threshold <FLOAT>           Default: 0.5",
        "  --overlap-policy <POLICY>     keep-all | highest-score | longest-then-score",
        "  --warmups <N>                 Default: 5",
        "  --iterations <N>              Default: 25",
        "  --json",
    ]
    .join("\n")
}
