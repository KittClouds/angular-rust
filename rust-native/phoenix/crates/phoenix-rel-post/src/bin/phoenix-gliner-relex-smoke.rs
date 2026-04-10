use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use phoenix_rel_post::{
    GlinerRelexLabel, GlinerRelexMetadata, GlinerRelexModel, GlinerRelexPredictOptions,
    GlinerRelexPrediction,
};
use serde::Serialize;

#[derive(Debug, Clone)]
struct SmokeConfig {
    model_root: PathBuf,
    text: String,
    entity_labels: Vec<GlinerRelexLabel>,
    relation_labels: Vec<GlinerRelexLabel>,
    options: GlinerRelexPredictOptions,
    json: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SmokeReport {
    ok: bool,
    load_ms: u64,
    run_ms: u64,
    model: GlinerRelexMetadata,
    prediction: GlinerRelexPrediction,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    let load_started = Instant::now();
    let model = GlinerRelexModel::load(&config.model_root).map_err(|error| {
        format!(
            "failed to load GLiNER relex model {}: {error}",
            config.model_root.display()
        )
    })?;
    let load_ms = load_started.elapsed().as_millis() as u64;
    let run_started = Instant::now();
    let prediction = model
        .predict(
            &config.text,
            &config.entity_labels,
            &config.relation_labels,
            &config.options,
        )
        .map_err(|error| format!("GLiNER relex inference failed: {error}"))?;
    let run_ms = run_started.elapsed().as_millis() as u64;
    let report = SmokeReport {
        ok: true,
        load_ms,
        run_ms,
        model: model.metadata().clone(),
        prediction,
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
        println!("loadMs: {}", report.load_ms);
        println!("runMs: {}", report.run_ms);
        println!("maxLen: {}", report.model.max_len);
        println!(
            "logitsShape: [{}, {}, {}, {}]",
            report.prediction.logits_shape[0],
            report.prediction.logits_shape[1],
            report.prediction.logits_shape[2],
            report.prediction.logits_shape[3]
        );
        println!(
            "relIdxShape: [{}, {}, {}]",
            report.prediction.rel_idx_shape[0],
            report.prediction.rel_idx_shape[1],
            report.prediction.rel_idx_shape[2]
        );
        println!(
            "relLogitsShape: [{}, {}, {}]",
            report.prediction.rel_logits_shape[0],
            report.prediction.rel_logits_shape[1],
            report.prediction.rel_logits_shape[2]
        );
        println!(
            "activeRelationPairs: {}",
            report.prediction.active_relation_pairs
        );
        println!("entities: {}", report.prediction.entities.len());
        for entity in &report.prediction.entities {
            println!(
                "- {} :: {} :: score={:.4} [{}..{}]",
                entity.label, entity.text, entity.score, entity.start, entity.end
            );
        }
        println!("relations: {}", report.prediction.relations.len());
        for relation in &report.prediction.relations {
            println!(
                "- {} --{}--> {} :: score={:.4}",
                relation.head, relation.label, relation.tail, relation.score
            );
        }
    }
    Ok(())
}

fn parse_args(args: &[String]) -> Result<SmokeConfig, String> {
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        return Err(usage());
    }
    let model_root = parse_required_path(args, "--model-root")?;
    let text = parse_required_string(args, "--text")?;
    let entity_labels = parse_labels(
        parse_string_arg(args, "--entity-labels"),
        parse_string_arg(args, "--entity-label-file"),
        default_entity_labels(),
    )?;
    let relation_labels = parse_labels(
        parse_string_arg(args, "--relation-labels"),
        parse_string_arg(args, "--relation-label-file"),
        default_relation_labels(),
    )?;
    Ok(SmokeConfig {
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
        json: args.iter().any(|arg| arg == "--json"),
    })
}

fn parse_labels(
    inline: Option<String>,
    file: Option<String>,
    fallback: Vec<GlinerRelexLabel>,
) -> Result<Vec<GlinerRelexLabel>, String> {
    if let Some(path) = file {
        let payload = fs::read_to_string(path)
            .map_err(|error| format!("failed to read label file: {error}"))?;
        return Ok(payload.lines().filter_map(parse_label_line).collect());
    }
    if let Some(value) = inline {
        let labels = value
            .split(',')
            .filter_map(parse_label_line)
            .collect::<Vec<_>>();
        if !labels.is_empty() {
            return Ok(labels);
        }
    }
    Ok(fallback)
}

fn parse_label_line(value: &str) -> Option<GlinerRelexLabel> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut parts = trimmed.splitn(2, '=');
    let label = parts.next()?.trim();
    let description = parts
        .next()
        .map(str::trim)
        .filter(|value| !value.is_empty());
    Some(GlinerRelexLabel {
        label: label.to_owned(),
        description: description.map(str::to_owned),
    })
}

fn parse_required_path(args: &[String], flag: &str) -> Result<PathBuf, String> {
    parse_string_arg(args, flag)
        .map(PathBuf::from)
        .ok_or_else(|| format!("{flag} is required\n\n{}", usage()))
}

fn parse_required_string(args: &[String], flag: &str) -> Result<String, String> {
    parse_string_arg(args, flag).ok_or_else(|| format!("{flag} is required\n\n{}", usage()))
}

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].clone())
}

fn parse_f32_arg(args: &[String], flag: &str) -> Option<f32> {
    parse_string_arg(args, flag).and_then(|value| value.parse::<f32>().ok())
}

fn default_entity_labels() -> Vec<GlinerRelexLabel> {
    vec![
        "location".to_owned(),
        "person".to_owned(),
        "date".to_owned(),
        "structure".to_owned(),
    ]
    .into_iter()
    .map(|label| GlinerRelexLabel {
        label,
        description: None,
    })
    .collect()
}

fn default_relation_labels() -> Vec<GlinerRelexLabel> {
    vec![
        "located in".to_owned(),
        "designed by".to_owned(),
        "completed in".to_owned(),
    ]
    .into_iter()
    .map(|label| GlinerRelexLabel {
        label,
        description: None,
    })
    .collect()
}

fn usage() -> String {
    [
        "Usage:",
        "  phoenix-gliner-relex-smoke --model-root <DIR> --text <TEXT> [options]",
        "",
        "Options:",
        "  --entity-labels <A,B,C>      Entity labels (use label=description for descriptions)",
        "  --entity-label-file <PATH>   One entity label per line",
        "  --relation-labels <A,B,C>    Relation labels (use label=description for descriptions)",
        "  --relation-label-file <PATH> One relation label per line",
        "  --threshold <FLOAT>          Entity threshold (default: 0.3)",
        "  --relation-threshold <FLOAT> Relation threshold (default: 0.5)",
        "  --flat-ner                   Enforce non-overlapping entities",
        "  --multi-label                Allow multiple labels per span",
        "  --json                       Print JSON output",
    ]
    .join("\n")
}
