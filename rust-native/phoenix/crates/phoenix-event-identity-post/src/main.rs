use std::collections::BTreeMap;
use std::path::PathBuf;

use phoenix_event_identity_post::{
    derive_dirty_scope_review_batches, persist_event_identity_patch_sidecar,
    run_event_identity_scope,
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
    mention_packet_count: usize,
    hypothesis_count: usize,
    canonical_event_count: usize,
    membership_count: usize,
    decision_count: usize,
    invalidation_count: usize,
    split_count: usize,
    card_count: usize,
    relation_counts: BTreeMap<String, usize>,
    diagnostics: BTreeMap<String, usize>,
    cards: Vec<CardReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CardReport {
    canonical_event_id: String,
    label: String,
    mention_count: usize,
    document_count: usize,
    dispute_count: usize,
    incompatible_count: usize,
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
        .map_err(|error| format!("failed to derive event identity review batches: {error}"))?;
    let mut reports = Vec::new();

    for batch in &mut batches {
        run_event_identity_scope(batch, now_ms());
        if config.persist_patches {
            let sidecar = persist_event_identity_patch_sidecar(&store, batch, now_ms())
                .map_err(|error| format!("failed to persist event identity sidecar: {error}"))?;
            phoenix_event_identity_post::apply_event_identity_patch_sidecar(batch, &sidecar);
        }

        let cards = batch
            .canonical_event_cards
            .iter()
            .take(config.card_limit)
            .map(|card| CardReport {
                canonical_event_id: card.canonical_event_id.0.clone(),
                label: card.canonical_label.clone(),
                mention_count: card.mention_ids.len(),
                document_count: card.document_ids.len(),
                dispute_count: card.open_dispute_ids.len(),
                incompatible_count: card.incompatible_hypothesis_ids.len(),
            })
            .collect::<Vec<_>>();

        reports.push(BatchReport {
            scope_key: batch.scope_key.clone(),
            mention_packet_count: batch.summary.mention_packet_count,
            hypothesis_count: batch.summary.hypothesis_count,
            canonical_event_count: batch.summary.canonical_event_count,
            membership_count: batch.summary.membership_count,
            decision_count: batch.summary.decision_count,
            invalidation_count: batch.summary.invalidation_count,
            split_count: batch.summary.split_count,
            card_count: batch.summary.card_count,
            relation_counts: batch.summary.relation_counts.clone(),
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
            println!("- mention packets: {}", report.mention_packet_count);
            println!("- hypotheses: {}", report.hypothesis_count);
            println!("- canonical events: {}", report.canonical_event_count);
            println!("- memberships: {}", report.membership_count);
            println!("- decisions: {}", report.decision_count);
            println!("- invalidations: {}", report.invalidation_count);
            println!("- splits: {}", report.split_count);
            println!("- cards: {}", report.card_count);
            for (relation, count) in report.relation_counts {
                println!("- relation {relation}: {count}");
            }
            for (code, count) in report.diagnostics {
                println!("- diagnostic {code}: {count}");
            }
            for card in report.cards {
                println!(
                    "- card {} ({}) mentions={} documents={} disputes={} incompatible={}",
                    card.label,
                    card.canonical_event_id,
                    card.mention_count,
                    card.document_count,
                    card.dispute_count,
                    card.incompatible_count
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
