use std::fs;
use std::path::PathBuf;

use phoenix_rel_post::{
    GliclassClassificationType, GliclassLabelScore, GliclassModel, GliclassPredictOptions,
    GliclassPrediction,
};
use serde::Serialize;

#[derive(Debug, Clone)]
struct SmokeConfig {
    model_root: PathBuf,
    text: String,
    labels: Vec<String>,
    prompt: Option<String>,
    classification_type: GliclassClassificationType,
    threshold: f32,
    max_results: usize,
    label_chunk_size: Option<usize>,
    json: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    ok: bool,
    label_count: usize,
    model: phoenix_rel_post::GliclassModelMetadata,
    prediction: GliclassPrediction,
    top_scores: Vec<GliclassLabelScore>,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    let model = GliclassModel::load(&config.model_root).map_err(|error| {
        format!(
            "failed to load GLiClass model {}: {error}",
            config.model_root.display()
        )
    })?;
    let prediction = model
        .predict(
            &config.text,
            &config.labels,
            &GliclassPredictOptions {
                classification_type: config.classification_type,
                threshold: config.threshold,
                prompt: config.prompt.clone(),
                label_chunk_size: config.label_chunk_size,
            },
        )
        .map_err(|error| format!("GLiClass inference failed: {error}"))?;
    let top_scores = prediction
        .all_scores
        .iter()
        .take(config.max_results)
        .cloned()
        .collect::<Vec<_>>();
    let report = SmokeReport {
        ok: true,
        label_count: config.labels.len(),
        model: model.metadata().clone(),
        prediction,
        top_scores,
    };

    if config.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|error| format!("failed to render JSON: {error}"))?
        );
    } else {
        println!("model: {}", report.model.model_path);
        println!("tokenizer: {}", report.model.tokenizer_path);
        println!("architecture: {}", report.model.architecture_type);
        println!("promptFirst: {}", report.model.prompt_first);
        println!("maxNumClasses: {}", report.model.max_num_classes);
        println!("maxLength: {}", report.model.max_length);
        println!("primaryOutput: {}", report.model.primary_output_name);
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
    }
    Ok(())
}

fn parse_args(args: &[String]) -> Result<SmokeConfig, String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Err(usage());
    }

    let mut model_root = None::<PathBuf>;
    let mut text = None::<String>;
    let mut text_file = None::<PathBuf>;
    let mut labels = Vec::<String>::new();
    let mut label_file = None::<PathBuf>;
    let mut prompt = None::<String>;
    let mut classification_type = GliclassClassificationType::MultiLabel;
    let mut threshold = 0.5f32;
    let mut max_results = 8usize;
    let mut label_chunk_size = None::<usize>;
    let mut json = false;

    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--model-root" => {
                index += 1;
                let value = args.get(index).ok_or("--model-root requires a value")?;
                model_root = Some(PathBuf::from(value));
            }
            "--text" => {
                index += 1;
                let value = args.get(index).ok_or("--text requires a value")?;
                text = Some(value.clone());
            }
            "--text-file" => {
                index += 1;
                let value = args.get(index).ok_or("--text-file requires a value")?;
                text_file = Some(PathBuf::from(value));
            }
            "--label" => {
                index += 1;
                let value = args.get(index).ok_or("--label requires a value")?;
                push_csv_labels(&mut labels, value);
            }
            "--labels" => {
                index += 1;
                let value = args.get(index).ok_or("--labels requires a value")?;
                push_csv_labels(&mut labels, value);
            }
            "--label-file" => {
                index += 1;
                let value = args.get(index).ok_or("--label-file requires a value")?;
                label_file = Some(PathBuf::from(value));
            }
            "--prompt" => {
                index += 1;
                let value = args.get(index).ok_or("--prompt requires a value")?;
                prompt = Some(value.clone());
            }
            "--classification-type" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or("--classification-type requires a value")?;
                classification_type =
                    GliclassClassificationType::parse(value).map_err(|error| error.to_string())?;
            }
            "--threshold" => {
                index += 1;
                let value = args.get(index).ok_or("--threshold requires a value")?;
                threshold = value
                    .parse::<f32>()
                    .map_err(|error| format!("invalid --threshold '{value}': {error}"))?;
            }
            "--max-results" => {
                index += 1;
                let value = args.get(index).ok_or("--max-results requires a value")?;
                max_results = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --max-results '{value}': {error}"))?;
            }
            "--label-chunk-size" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or("--label-chunk-size requires a value")?;
                label_chunk_size =
                    Some(value.parse::<usize>().map_err(|error| {
                        format!("invalid --label-chunk-size '{value}': {error}")
                    })?);
            }
            "--json" => {
                json = true;
            }
            unknown => {
                return Err(format!("unrecognized argument '{unknown}'\n\n{}", usage()));
            }
        }
        index += 1;
    }

    let model_root = model_root.ok_or_else(usage_model_root)?;
    let text = match (text, text_file) {
        (Some(value), _) => value,
        (None, Some(path)) => fs::read_to_string(&path)
            .map_err(|error| format!("failed to read text file {}: {error}", path.display()))?,
        (None, None) => return Err(usage_text()),
    };

    if let Some(path) = label_file {
        let payload = fs::read_to_string(&path)
            .map_err(|error| format!("failed to read label file {}: {error}", path.display()))?;
        for line in payload.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                labels.push(trimmed.to_owned());
            }
        }
    }
    labels.retain(|label| !label.trim().is_empty());
    if labels.is_empty() {
        return Err(usage_labels());
    }

    Ok(SmokeConfig {
        model_root,
        text,
        labels,
        prompt,
        classification_type,
        threshold,
        max_results,
        label_chunk_size,
        json,
    })
}

fn push_csv_labels(out: &mut Vec<String>, value: &str) {
    for label in value.split(',') {
        let trimmed = label.trim();
        if !trimmed.is_empty() {
            out.push(trimmed.to_owned());
        }
    }
}

fn usage() -> String {
    [
        "Usage:",
        "  phoenix-gliclass-smoke --model-root <DIR> --text <TEXT> --labels <A,B,C> [options]",
        "",
        "Options:",
        "  --text-file <PATH>            Read text from a file",
        "  --label <LABEL>               Add one label (repeatable)",
        "  --label-file <PATH>           Read one label per line",
        "  --prompt <TEXT>               Add a zero-shot task prompt",
        "  --classification-type <TYPE>  multi-label | single-label",
        "  --threshold <FLOAT>           Multi-label score threshold (default: 0.5)",
        "  --label-chunk-size <N>        Chunk labels before inference",
        "  --max-results <N>             Limit printed top scores (default: 8)",
        "  --json                        Print JSON output",
    ]
    .join("\n")
}

fn usage_model_root() -> String {
    format!("--model-root is required\n\n{}", usage())
}

fn usage_text() -> String {
    format!("either --text or --text-file is required\n\n{}", usage())
}

fn usage_labels() -> String {
    format!("at least one label is required\n\n{}", usage())
}
