use std::cmp::Ordering;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use phoenix_rel_post::{
    flatten_gliclass_instruct_hierarchical_labels, GliclassClassificationType,
    GliclassInstructExample, GliclassInstructLabel, GliclassInstructModel,
    GliclassInstructPredictOptions,
};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone)]
struct BenchConfig {
    model_root: PathBuf,
    text: String,
    labels: Vec<GliclassInstructLabel>,
    examples: Vec<GliclassInstructExample>,
    prompt: Option<String>,
    classification_type: GliclassClassificationType,
    threshold: f32,
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
    selected_count: usize,
    top_label: Option<String>,
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
    let model = GliclassInstructModel::load(&config.model_root)
        .map_err(|error| format!("failed to load model: {error}"))?;
    let load_ms = load_started.elapsed().as_millis() as u64;

    let options = GliclassInstructPredictOptions {
        classification_type: config.classification_type,
        threshold: config.threshold,
        prompt: config.prompt.clone(),
        examples: config.examples.clone(),
    };

    let mut last_prediction = None;
    for _ in 0..config.warmups {
        last_prediction = Some(
            model
                .predict_structured(&config.text, &config.labels, &options)
                .map_err(|error| format!("warmup failed: {error}"))?,
        );
    }

    let mut runs_ms = Vec::with_capacity(config.iterations);
    for _ in 0..config.iterations {
        let started = Instant::now();
        last_prediction = Some(
            model
                .predict_structured(&config.text, &config.labels, &options)
                .map_err(|error| format!("inference failed: {error}"))?,
        );
        runs_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    let prediction = last_prediction.ok_or_else(|| "no benchmark iterations ran".to_owned())?;
    let stats = stats(&runs_ms)?;
    let report = BenchReport {
        ok: true,
        model_path: model.metadata().encoder_path.clone(),
        label_count: config.labels.len(),
        selected_count: prediction.selected.len(),
        top_label: prediction.all_scores.first().map(|row| row.label.clone()),
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
        println!("labels: {}", report.label_count);
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
            "selected={} topLabel={}",
            report.selected_count,
            report.top_label.as_deref().unwrap_or("<none>")
        );
    }
    Ok(())
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
    let mut labels = Vec::<String>::new();
    let mut label_descriptions = Vec::<(String, String)>::new();
    let mut examples = Vec::<GliclassInstructExample>::new();
    let mut prompt = None::<String>;
    let mut classification_type = GliclassClassificationType::MultiLabel;
    let mut threshold = 0.5f32;
    let mut warmups = 5usize;
    let mut iterations = 25usize;
    let mut json = false;

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--model-root" => {
                model_root = Some(PathBuf::from(next_arg(args, &mut index, "--model-root")?));
            }
            "--text" => text = Some(next_arg(args, &mut index, "--text")?.to_owned()),
            "--labels" | "--label" => {
                push_csv(&mut labels, next_arg(args, &mut index, "--labels")?)
            }
            "--label-file" => {
                let payload = fs::read_to_string(next_arg(args, &mut index, "--label-file")?)
                    .map_err(|error| format!("failed to read label file: {error}"))?;
                for line in payload.lines() {
                    push_csv(&mut labels, line);
                }
            }
            "--label-json-file" => {
                let payload = fs::read_to_string(next_arg(args, &mut index, "--label-json-file")?)
                    .map_err(|error| format!("failed to read label JSON file: {error}"))?;
                let value: Value = serde_json::from_str(&payload)
                    .map_err(|error| format!("invalid label JSON: {error}"))?;
                labels.extend(flatten_labels(&value)?);
            }
            "--label-description" => {
                let (label, description) =
                    parse_assignment(next_arg(args, &mut index, "--label-description")?)?;
                label_descriptions.push((label, description));
            }
            "--examples-file" => {
                let payload = fs::read_to_string(next_arg(args, &mut index, "--examples-file")?)
                    .map_err(|error| format!("failed to read examples JSON: {error}"))?;
                examples = serde_json::from_str::<Vec<GliclassInstructExample>>(&payload)
                    .map_err(|error| format!("invalid examples JSON: {error}"))?;
            }
            "--prompt" => prompt = Some(next_arg(args, &mut index, "--prompt")?.to_owned()),
            "--classification-type" => {
                classification_type = GliclassClassificationType::parse(next_arg(
                    args,
                    &mut index,
                    "--classification-type",
                )?)
                .map_err(|error| error.to_string())?;
            }
            "--threshold" => {
                let value = next_arg(args, &mut index, "--threshold")?;
                threshold = value
                    .parse::<f32>()
                    .map_err(|error| format!("invalid --threshold '{value}': {error}"))?;
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

    let labels = if labels.is_empty() {
        default_labels()
    } else {
        labels
            .into_iter()
            .map(|label| GliclassInstructLabel {
                description: label_descriptions.iter().find_map(|(name, description)| {
                    (name == &label).then_some(description.clone())
                }),
                label,
            })
            .collect()
    };

    Ok(BenchConfig {
        model_root: model_root.unwrap_or_else(|| PathBuf::from("gliclass-instruct-onnx-v2")),
        text: text.unwrap_or_else(default_text),
        labels,
        examples,
        prompt,
        classification_type,
        threshold,
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

fn parse_assignment(value: &str) -> Result<(String, String), String> {
    let mut parts = value.splitn(2, '=');
    let label = parts.next().unwrap_or_default().trim();
    let description = parts.next().unwrap_or_default().trim();
    if label.is_empty() || description.is_empty() {
        return Err(format!(
            "expected LABEL=DESCRIPTION for --label-description, got '{value}'"
        ));
    }
    Ok((label.to_owned(), description.to_owned()))
}

fn flatten_labels(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::Array(items) if items.iter().all(|item| item.is_string()) => Ok(items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()),
        _ => flatten_gliclass_instruct_hierarchical_labels(value, "."),
    }
}

fn default_text() -> String {
    "The transit authority suspended commuter rail service after severe storms flooded multiple lines overnight."
        .to_owned()
}

fn default_labels() -> Vec<GliclassInstructLabel> {
    [
        ("world_state", "current state or condition"),
        ("history", "chronological prior events"),
        ("causal_explanation", "cause and effect explanation"),
        ("forecast", "future expectation or prediction"),
    ]
    .into_iter()
    .map(|(label, description)| GliclassInstructLabel {
        label: label.to_owned(),
        description: Some(description.to_owned()),
    })
    .collect()
}

fn usage() -> String {
    [
        "Usage:",
        "  phoenix-gliclass-instruct-bench --model-root <DIR> [options]",
        "",
        "Labels:",
        "  --labels <A,B,C>",
        "  --label-file <PATH>",
        "  --label-json-file <PATH>",
        "  --label-description <A=B>",
        "",
        "Options:",
        "  --text <TEXT>",
        "  --examples-file <PATH>",
        "  --prompt <TEXT>",
        "  --classification-type <TYPE>  multi-label | single-label",
        "  --threshold <FLOAT>           Default: 0.5",
        "  --warmups <N>                 Default: 5",
        "  --iterations <N>              Default: 25",
        "  --json",
    ]
    .join("\n")
}
