use std::env;
use std::path::PathBuf;

use phoenix_api::depth_audit::{run_depth_audit, DepthAuditConfig};
use phoenix_graph_post::smoke_support::{string_arg, usize_arg};

fn main() {
    match run_depth_audit(parse_args(env::args().skip(1).collect())) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize depth audit report")
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn parse_args(args: Vec<String>) -> DepthAuditConfig {
    let mut config = DepthAuditConfig::default();
    if let Some(path) = string_arg(&args, "--store-path") {
        config.store_path = PathBuf::from(path);
    }
    if let Some(value) = usize_arg(&args, "--probe-limit") {
        config.probe_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--seed-limit") {
        config.seed_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--oversample") {
        config.oversample = value.max(config.seed_limit);
    }
    if let Some(value) = usize_arg(&args, "--expansion-hops") {
        config.expansion_hops = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--region-node-limit") {
        config.region_node_limit = value.max(32);
    }
    if let Some(value) = usize_arg(&args, "--history-limit") {
        config.history_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--causal-limit") {
        config.causal_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--neighbor-limit") {
        config.graph.neighbor_limit = value.max(1);
    }
    if let Some(value) = usize_arg(&args, "--min-score") {
        config.graph.min_score_millis = value.min(1000) as u32;
    }
    if args.iter().any(|arg| arg == "--refresh-graph") {
        config.refresh_graph = true;
    }
    if args.iter().any(|arg| arg == "--refresh-pipeline") {
        config.refresh_pipeline = true;
    }
    if args.iter().any(|arg| arg == "--refresh-semantic") {
        config.refresh_semantic = true;
    }
    config
}
