use std::collections::BTreeMap;
use std::path::PathBuf;

use phoenix_state_schema_post::{
    derive_dirty_scope_review_batches, persist_state_schema_patch_sidecar, run_state_schema_scope,
};
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::SessionId;
use serde::Serialize;

#[derive(Debug, Clone)]
struct SmokeConfig {
    store_path: PathBuf,
    session_id: Option<SessionId>,
    json: bool,
    persist_patches: bool,
    candidate_limit: usize,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        Self {
            store_path: PathBuf::new(),
            session_id: None,
            json: false,
            persist_patches: false,
            candidate_limit: 16,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchReport {
    scope_key: String,
    family_count: usize,
    definition_count: usize,
    active_definition_count: usize,
    candidate_definition_count: usize,
    candidate_count: usize,
    promotion_decision_count: usize,
    write_proposal_count: usize,
    lifecycle_counts: BTreeMap<String, usize>,
    diagnostics: BTreeMap<String, usize>,
    candidates: Vec<CandidateReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CandidateReport {
    slot_key: String,
    support_count: usize,
    document_count: usize,
    relation_families: Vec<String>,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    if config.store_path.as_os_str().is_empty() {
        return Err("--store-path is required".to_owned());
    }

    let store = PhoenixOvergraphStore::open(&config.store_path).map_err(|error| {
        format!(
            "failed to open store {}: {error}",
            config.store_path.display()
        )
    })?;
    let mut batches = derive_dirty_scope_review_batches(&store, config.session_id.as_ref())
        .map_err(|error| format!("failed to derive state schema review batches: {error}"))?;
    let mut reports = Vec::new();

    for batch in &mut batches {
        run_state_schema_scope(batch, now_ms());
        if config.persist_patches {
            let sidecar = persist_state_schema_patch_sidecar(&store, batch, now_ms())
                .map_err(|error| format!("failed to persist state schema sidecar: {error}"))?;
            phoenix_state_schema_post::apply_state_schema_patch_sidecar(batch, &sidecar);
        }
        reports.push(BatchReport {
            scope_key: batch.scope_key.clone(),
            family_count: batch.summary.family_count,
            definition_count: batch.summary.definition_count,
            active_definition_count: batch.summary.active_definition_count,
            candidate_definition_count: batch.summary.candidate_definition_count,
            candidate_count: batch.summary.candidate_count,
            promotion_decision_count: batch.summary.promotion_decision_count,
            write_proposal_count: batch.summary.write_proposal_count,
            lifecycle_counts: batch.summary.lifecycle_counts.clone(),
            diagnostics: batch.diagnostics.clone(),
            candidates: batch
                .slot_candidates
                .iter()
                .take(config.candidate_limit)
                .map(|candidate| CandidateReport {
                    slot_key: candidate.slot_key.clone(),
                    support_count: candidate.support_count,
                    document_count: candidate.document_count,
                    relation_families: candidate.relation_families.clone(),
                })
                .collect(),
        });
    }

    if config.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&reports)
                .map_err(|error| format!("failed to render json: {error}"))?
        );
    } else {
        for report in reports {
            println!("scope: {}", report.scope_key);
            println!("- families: {}", report.family_count);
            println!("- definitions: {}", report.definition_count);
            println!("- active definitions: {}", report.active_definition_count);
            println!(
                "- candidate definitions: {}",
                report.candidate_definition_count
            );
            println!("- candidates: {}", report.candidate_count);
            println!("- promotion decisions: {}", report.promotion_decision_count);
            println!("- write proposals: {}", report.write_proposal_count);
            for (key, value) in report.lifecycle_counts {
                println!("- lifecycle {key}: {value}");
            }
            for (key, value) in report.diagnostics {
                println!("- diagnostic {key}: {value}");
            }
            for candidate in report.candidates {
                println!(
                    "- candidate {} supports={} docs={} relations={}",
                    candidate.slot_key,
                    candidate.support_count,
                    candidate.document_count,
                    candidate.relation_families.join(",")
                );
            }
        }
    }

    Ok(())
}

fn parse_args(args: &[String]) -> Result<SmokeConfig, String> {
    let mut config = SmokeConfig::default();
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--store-path" => {
                index += 1;
                let value = args.get(index).ok_or("--store-path requires a value")?;
                config.store_path = PathBuf::from(value);
            }
            "--session-id" => {
                index += 1;
                let value = args.get(index).ok_or("--session-id requires a value")?;
                config.session_id = Some(SessionId(value.clone()));
            }
            "--json" => config.json = true,
            "--persist-patches" => config.persist_patches = true,
            "--candidate-limit" => {
                index += 1;
                let value = args
                    .get(index)
                    .ok_or("--candidate-limit requires a value")?;
                config.candidate_limit = value.parse::<usize>().map_err(|error| {
                    format!("invalid --candidate-limit value '{value}': {error}")
                })?;
            }
            flag => return Err(format!("unknown argument: {flag}")),
        }
        index += 1;
    }
    Ok(config)
}

fn now_ms() -> i64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}
