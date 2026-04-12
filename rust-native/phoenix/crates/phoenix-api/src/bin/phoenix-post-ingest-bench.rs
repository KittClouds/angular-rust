use std::fs;
use std::path::PathBuf;

#[path = "../post_ingest_bench/mod.rs"]
mod post_ingest_bench;

use post_ingest_bench::{default_output_path, mean_ms, Config};

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    let report = post_ingest_bench::run(&config)?;
    let output_path = config
        .output_path
        .clone()
        .unwrap_or_else(default_output_path);

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
                "{} bytes={} legacy.meanMs={:.3} shared.meanMs={:.3}",
                case.case_id,
                case.text_bytes,
                mean_ms(case.legacy.phases.get("total_us")),
                mean_ms(case.shared.phases.get("total_us")),
            );
        }
    }

    Ok(())
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
