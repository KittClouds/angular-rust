use std::env;
use std::path::PathBuf;

use phoenix_perf::{
    default_out_dir, render_markdown, run_perf_suite_filtered_with_config, strict_check,
    write_suite_report, BenchmarkConfig,
};

fn main() {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let strict = args.iter().any(|arg| arg == "--strict");
    let json_only = args.iter().any(|arg| arg == "--json");
    let progress = args.iter().any(|arg| arg == "--progress");
    let corpus_filter = args
        .windows(2)
        .find_map(|window| (window[0] == "--corpus").then(|| window[1].clone()));
    let out_dir = args
        .windows(2)
        .find_map(|window| (window[0] == "--out-dir").then(|| PathBuf::from(&window[1])))
        .unwrap_or_else(default_out_dir);
    let benchmark_config = BenchmarkConfig {
        iterations: parse_usize_arg(&args, "--iterations")
            .unwrap_or_else(|| BenchmarkConfig::default().iterations)
            .max(1),
        warmup_iterations: parse_usize_arg(&args, "--warmup")
            .unwrap_or_else(|| BenchmarkConfig::default().warmup_iterations),
    };

    if progress {
        unsafe {
            env::set_var("PHOENIX_PERF_PROGRESS", "1");
        }
    }

    let report = match run_perf_suite_filtered_with_config(corpus_filter.as_deref(), &benchmark_config) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    let (json_path, md_path) = match write_suite_report(&report, &out_dir) {
        Ok(paths) => paths,
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    };

    if json_only {
        println!("{}", serde_json::to_string_pretty(&report).expect("json"));
    } else {
        println!("{}", render_markdown(&report));
        println!();
        println!("JSON report: {}", json_path.display());
        println!("Markdown report: {}", md_path.display());
    }

    if strict {
        if let Err(error) = strict_check(&report) {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].parse::<usize>().ok()))
        .flatten()
}
