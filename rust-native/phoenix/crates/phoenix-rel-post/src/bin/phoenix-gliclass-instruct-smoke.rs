use std::fs;
use std::path::PathBuf;

use phoenix_rel_post::{
    build_gliclass_instruct_hierarchical_scores, flatten_gliclass_instruct_hierarchical_labels,
    GliclassClassificationType, GliclassInstructExample, GliclassInstructLabel,
    GliclassInstructMetadata, GliclassInstructModel, GliclassInstructPredictOptions,
    GliclassLabelScore, GliclassPrediction,
};
use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    ok: bool,
    label_count: usize,
    model: GliclassInstructMetadata,
    prediction: GliclassPrediction,
    top_scores: Vec<GliclassLabelScore>,
    #[serde(skip_serializing_if = "Option::is_none")]
    hierarchical_scores: Option<Value>,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    let model = GliclassInstructModel::load(&config.model_root).map_err(|error| {
        format!(
            "failed to load GLiClass-instruct model {}: {error}",
            config.model_root.display()
        )
    })?;
    let prediction = model
        .predict_structured(
            &config.text,
            &config.labels,
            &GliclassInstructPredictOptions {
                classification_type: config.classification_type,
                threshold: config.threshold,
                prompt: config.prompt.clone(),
                examples: config.examples.clone(),
            },
        )
        .map_err(|error| format!("GLiClass-instruct inference failed: {error}"))?;
    let top_scores = prediction
        .all_scores
        .iter()
        .take(config.max_results)
        .cloned()
        .collect::<Vec<_>>();
    let hierarchical_scores = config
        .hierarchical_labels
        .as_ref()
        .map(|labels| {
            build_gliclass_instruct_hierarchical_scores(labels, &prediction.all_scores, ".")
                .map_err(|error| format!("failed to build hierarchical scores: {error}"))
        })
        .transpose()?;
    let report = SmokeReport {
        ok: true,
        label_count: config.labels.len(),
        model: model.metadata().clone(),
        prediction,
        top_scores,
        hierarchical_scores,
    };
    if config.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|error| format!("failed to render JSON: {error}"))?
        );
    } else {
        println!("encoder:    {}", report.model.encoder_path);
        println!("projectors: {}", report.model.projectors_path);
        println!("scorer:     {}", report.model.scorer_path);
        println!("tokenizer:  {}", report.model.tokenizer_path);
        println!("architecture: {}", report.model.architecture_type);
        println!("promptFirst: {}", report.model.prompt_first);
        println!("maxLength: {}", report.model.max_length);
        println!("labelCount: {}", report.label_count);
        println!(
            "classificationType: {:?}",
            report.prediction.classification_type
        );
        if let Some(threshold) = report.prediction.threshold {
            println!("threshold: {:.3}", threshold);
        }
        if report.prediction.selected.is_empty() {
            println!("selected: none");
        } else {
            println!("selected:");
            for row in &report.prediction.selected {
                println!(
                    "- {} :: score={:.6} logit={:.6}",
                    row.label, row.score, row.logit
                );
            }
        }
        println!("topScores:");
        for row in &report.top_scores {
            println!(
                "- {} :: score={:.6} logit={:.6}",
                row.label, row.score, row.logit
            );
        }
        if let Some(hierarchical_scores) = &report.hierarchical_scores {
            println!("hierarchicalScores:");
            println!(
                "{}",
                serde_json::to_string_pretty(hierarchical_scores)
                    .map_err(|error| format!("failed to render hierarchical JSON: {error}"))?
            );
        }
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct SmokeConfig {
    model_root: PathBuf,
    text: String,
    labels: Vec<GliclassInstructLabel>,
    hierarchical_labels: Option<Value>,
    examples: Vec<GliclassInstructExample>,
    prompt: Option<String>,
    classification_type: GliclassClassificationType,
    threshold: f32,
    max_results: usize,
    json: bool,
}

fn parse_args(args: &[String]) -> Result<SmokeConfig, String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Err(usage());
    }
    let mut model_root = None::<PathBuf>;
    let mut text = None::<String>;
    let mut labels = Vec::<String>::new();
    let mut label_descriptions = Vec::<(String, String)>::new();
    let mut hierarchical_labels = None::<Value>;
    let mut examples = Vec::<GliclassInstructExample>::new();
    let mut prompt = None::<String>;
    let mut classification_type = GliclassClassificationType::MultiLabel;
    let mut threshold = 0.5f32;
    let mut max_results = 8usize;
    let mut json = false;

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--model-root" => {
                model_root = Some(PathBuf::from(next_arg(args, &mut index, "--model-root")?))
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
                hierarchical_labels =
                    matches!(value, Value::Object(_) | Value::Array(_)).then_some(value.clone());
                labels.extend(flatten_json_labels(&value)?);
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
            "--max-results" => {
                let value = next_arg(args, &mut index, "--max-results")?;
                max_results = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --max-results '{value}': {error}"))?;
            }
            "--json" => json = true,
            unknown => return Err(format!("unrecognized argument '{unknown}'\n\n{}", usage())),
        }
        index += 1;
    }

    let model_root =
        model_root.ok_or_else(|| format!("--model-root is required\n\n{}", usage()))?;
    let text = text.ok_or_else(|| format!("--text is required\n\n{}", usage()))?;
    if labels.is_empty() {
        return Err(format!("at least one label is required\n\n{}", usage()));
    }
    let labels = labels
        .into_iter()
        .map(|label| GliclassInstructLabel {
            description: label_descriptions
                .iter()
                .find_map(|(name, description)| (name == &label).then_some(description.clone())),
            label,
        })
        .collect();

    Ok(SmokeConfig {
        model_root,
        text,
        labels,
        hierarchical_labels,
        examples,
        prompt,
        classification_type,
        threshold,
        max_results,
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

fn flatten_json_labels(value: &Value) -> Result<Vec<String>, String> {
    match value {
        Value::Array(items) if items.iter().all(|item| item.is_object()) => items
            .iter()
            .map(|item| {
                item.get("label")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .ok_or_else(|| "label objects must contain a string 'label' field".to_owned())
            })
            .collect(),
        Value::Object(items)
            if items
                .values()
                .all(|item| item.is_object() && item.get("description").is_some()) =>
        {
            Ok(items.keys().cloned().collect())
        }
        Value::Array(items) if items.iter().all(|item| item.is_string()) => Ok(items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect()),
        _ => flatten_gliclass_instruct_hierarchical_labels(value, "."),
    }
}

fn usage() -> String {
    [
        "Usage:",
        "  phoenix-gliclass-instruct-smoke --model-root <DIR> --text <TEXT> [labels] [options]",
        "",
        "Labels:",
        "  --labels <A,B,C>              Flat labels",
        "  --label-file <PATH>           One label per line",
        "  --label-json-file <PATH>      JSON labels: flat list, objects, or hierarchical tree",
        "  --label-description <A=B>     Add a label description (repeatable)",
        "",
        "Options:",
        "  --examples-file <PATH>        JSON array of {text, labels}",
        "  --prompt <TEXT>               Task prompt",
        "  --classification-type <TYPE>  multi-label | single-label",
        "  --threshold <FLOAT>           Multi-label score threshold (default: 0.5)",
        "  --max-results <N>             Limit printed top scores (default: 8)",
        "  --json                        Print JSON output",
    ]
    .join("\n")
}
