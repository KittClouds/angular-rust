use std::path::PathBuf;

use phoenix_rel_post::{
    GlinerBiModel, GlinerBiModelMetadata, GlinerBiOverlapPolicy, GlinerBiPredictOptions,
    GlinerBiPrediction,
};
use serde::Serialize;

#[derive(Debug, Clone)]
struct SmokeConfig {
    model_root: PathBuf,
    text: String,
    labels: Vec<String>,
    threshold: f32,
    overlap_policy: GlinerBiOverlapPolicy,
    json: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    ok: bool,
    label_count: usize,
    overlap_policy: GlinerBiOverlapPolicy,
    model: GlinerBiModelMetadata,
    predictions: Vec<GlinerBiPrediction>,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;

    let model = GlinerBiModel::load(&config.model_root).map_err(|error| {
        format!(
            "failed to load GLiNER Bi-Encoder model {}: {error}",
            config.model_root.display()
        )
    })?;

    let label_set = model
        .prepare_labels(&config.labels)
        .map_err(|error| format!("GLiNER label preparation failed: {error}"))?;
    let predictions = model
        .predict_with_label_set(
            &config.text,
            &label_set,
            &GlinerBiPredictOptions {
                threshold: config.threshold,
                overlap_policy: config.overlap_policy,
            },
        )
        .map_err(|error| format!("GLiNER inference failed: {error}"))?;

    let report = SmokeReport {
        ok: true,
        label_count: config.labels.len(),
        overlap_policy: config.overlap_policy,
        model: model.metadata().clone(),
        predictions,
    };

    if config.json {
        println!("{}", serde_json::to_string_pretty(&report).unwrap());
    } else {
        println!("\n=== GLiNER Bi-Encoder Smoke Test ===");
        println!("Model: {}", report.model.model_path);
        println!("Text: {:?}", config.text);
        println!("Labels: {:?}", config.labels);
        println!(
            "Predictions (Threshold = {}, Overlap = {:?}):",
            config.threshold, config.overlap_policy
        );
        for (i, p) in report.predictions.iter().enumerate() {
            println!(
                "  [{}] {:?} ({}) - {:.1}%",
                i + 1,
                p.text,
                p.label,
                p.score * 100.0,
            );
        }
        if report.predictions.is_empty() {
            println!("  (No entities found)");
        }
        println!();
    }

    Ok(())
}

fn parse_args(args: &[String]) -> Result<SmokeConfig, String> {
    let mut model_root = None;
    let mut text = None;
    let mut labels = Vec::new();
    let mut threshold = 0.5;
    let mut overlap_policy = None::<GlinerBiOverlapPolicy>;
    let mut flat_ner = true;
    let mut json = false;
    let mut i = 1;

    let default_text = "Microsoft was founded by Bill Gates and Paul Allen.".to_string();
    let default_labels = vec!["Person".to_owned(), "Organization".to_owned()];

    while i < args.len() {
        let arg = &args[i];
        if arg == "--model" && i + 1 < args.len() {
            model_root = Some(PathBuf::from(&args[i + 1]));
            i += 2;
        } else if arg == "--text" && i + 1 < args.len() {
            text = Some(args[i + 1].clone());
            i += 2;
        } else if arg == "--label" && i + 1 < args.len() {
            labels.push(args[i + 1].clone());
            i += 2;
        } else if arg == "--threshold" && i + 1 < args.len() {
            threshold = args[i + 1].parse().map_err(|_| "invalid threshold")?;
            i += 2;
        } else if arg == "--no-flat" {
            flat_ner = false;
            i += 1;
        } else if arg == "--overlap-policy" && i + 1 < args.len() {
            overlap_policy = Some(GlinerBiOverlapPolicy::parse(&args[i + 1])?);
            i += 2;
        } else if arg == "--json" {
            json = true;
            i += 1;
        } else {
            return Err(format!("unknown argument: {arg}"));
        }
    }

    let model_root = model_root.unwrap_or_else(|| PathBuf::from("gliner-bi-onnx"));
    let text = text.unwrap_or(default_text);
    if labels.is_empty() {
        labels = default_labels;
    }
    let overlap_policy = overlap_policy.unwrap_or(if flat_ner {
        GlinerBiOverlapPolicy::HighestScore
    } else {
        GlinerBiOverlapPolicy::KeepAll
    });

    Ok(SmokeConfig {
        model_root,
        text,
        labels,
        threshold,
        overlap_policy,
        json,
    })
}
