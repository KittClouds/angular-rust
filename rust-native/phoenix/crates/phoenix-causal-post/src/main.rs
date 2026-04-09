use std::collections::BTreeMap;
use std::path::PathBuf;

use phoenix_causal_post::{
    derive_dirty_scope_review_batches, persist_causal_patch_sidecar, run_causal_scope,
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
    card_limit: usize,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        Self {
            store_path: PathBuf::new(),
            session_id: None,
            json: false,
            persist_patches: false,
            card_limit: 16,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchReport {
    scope_key: String,
    event_profile_count: usize,
    review_case_count: usize,
    edge_record_count: usize,
    committed_edge_count: usize,
    accepted_edge_count: usize,
    supported_edge_count: usize,
    deferred_edge_count: usize,
    rejected_edge_count: usize,
    contradicted_edge_count: usize,
    chain_count: usize,
    counterfactual_review_count: usize,
    memory_card_count: usize,
    kind_counts: BTreeMap<String, usize>,
    outcome_counts: BTreeMap<String, usize>,
    diagnostics: BTreeMap<String, usize>,
    cards: Vec<CardReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CardReport {
    document_id: String,
    label: String,
    sentence_index: usize,
    incoming_edge_count: usize,
    outgoing_edge_count: usize,
    chain_count: usize,
    counterfactual_review_count: usize,
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
        .map_err(|error| format!("failed to derive causal review batches: {error}"))?;
    let mut reports = Vec::new();
    for batch in &mut batches {
        run_causal_scope(batch, now_ms());
        if config.persist_patches {
            let sidecar = persist_causal_patch_sidecar(&store, batch, now_ms())
                .map_err(|error| format!("failed to persist causal sidecar: {error}"))?;
            phoenix_causal_post::apply_causal_patch_sidecar(batch, &sidecar);
        }

        let cards = batch
            .memory_cards
            .iter()
            .take(config.card_limit)
            .map(|card| CardReport {
                document_id: card.document_id.clone(),
                label: card.label.clone(),
                sentence_index: card.sentence_index,
                incoming_edge_count: card.incoming_edge_ids.len(),
                outgoing_edge_count: card.outgoing_edge_ids.len(),
                chain_count: card.chain_ids.len(),
                counterfactual_review_count: card.counterfactual_review_ids.len(),
            })
            .collect::<Vec<_>>();

        reports.push(BatchReport {
            scope_key: batch.scope_key.clone(),
            event_profile_count: batch.event_profiles.len(),
            review_case_count: batch.summary.review_case_count,
            edge_record_count: batch.summary.edge_record_count,
            committed_edge_count: batch.summary.committed_edge_count,
            accepted_edge_count: batch.summary.accepted_edge_count,
            supported_edge_count: batch.summary.supported_edge_count,
            deferred_edge_count: batch.summary.deferred_edge_count,
            rejected_edge_count: batch.summary.rejected_edge_count,
            contradicted_edge_count: batch.summary.contradicted_edge_count,
            chain_count: batch.summary.chain_count,
            counterfactual_review_count: batch.summary.counterfactual_review_count,
            memory_card_count: batch.summary.memory_card_count,
            kind_counts: batch.summary.kind_counts.clone(),
            outcome_counts: batch.summary.outcome_counts.clone(),
            diagnostics: batch.diagnostics.clone(),
            cards,
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
            println!("- event profiles: {}", report.event_profile_count);
            println!("- review cases: {}", report.review_case_count);
            println!("- edge records: {}", report.edge_record_count);
            println!("- committed edges: {}", report.committed_edge_count);
            println!("- accepted edges: {}", report.accepted_edge_count);
            println!("- supported edges: {}", report.supported_edge_count);
            println!("- deferred edges: {}", report.deferred_edge_count);
            println!("- rejected edges: {}", report.rejected_edge_count);
            println!("- contradicted edges: {}", report.contradicted_edge_count);
            println!("- chains: {}", report.chain_count);
            println!(
                "- counterfactual reviews: {}",
                report.counterfactual_review_count
            );
            println!("- memory cards: {}", report.memory_card_count);
            for (kind, count) in report.kind_counts {
                println!("- kind {kind}: {count}");
            }
            for (outcome, count) in report.outcome_counts {
                println!("- outcome {outcome}: {count}");
            }
            for (code, count) in report.diagnostics {
                println!("- diagnostic {code}: {count}");
            }
            for card in report.cards {
                println!(
                    "- card {} @ sentence {} :: in={} out={} chains={} reviews={}",
                    card.label,
                    card.sentence_index,
                    card.incoming_edge_count,
                    card.outgoing_edge_count,
                    card.chain_count,
                    card.counterfactual_review_count
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
            "--card-limit" => {
                index += 1;
                let value = args.get(index).ok_or("--card-limit requires a value")?;
                config.card_limit = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --card-limit value '{value}': {error}"))?;
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
