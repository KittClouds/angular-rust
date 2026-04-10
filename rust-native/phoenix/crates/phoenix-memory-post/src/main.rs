use std::collections::BTreeMap;
use std::path::PathBuf;

use phoenix_memory_post::{derive_dirty_scope_review_batches, persist_memory_patch_sidecar};
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
    claim_count: usize,
    event_count: usize,
    state_count: usize,
    delta_count: usize,
    conflict_count: usize,
    gap_count: usize,
    entity_card_count: usize,
    relationship_ledger_count: usize,
    active_slot_counts: BTreeMap<String, usize>,
    slot_claim_counts: BTreeMap<String, usize>,
    relation_family_counts: BTreeMap<String, usize>,
    unresolved_gap_counts: BTreeMap<String, usize>,
    source_class_counts: BTreeMap<String, usize>,
    status_counts: BTreeMap<String, usize>,
    cards: Vec<CardReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CardReport {
    entity_id: String,
    canonical_name: String,
    aliases: Vec<String>,
    effective_kind: Option<String>,
    current_state: BTreeMap<String, String>,
    recent_delta_count: usize,
    active_relationship_count: usize,
    active_conflict_count: usize,
    open_gap_count: usize,
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
        .map_err(|error| format!("failed to derive memory review batches: {error}"))?;

    let mut reports = Vec::new();
    for batch in &mut batches {
        if config.persist_patches {
            let sidecar = persist_memory_patch_sidecar(&store, batch, now_ms())
                .map_err(|error| format!("failed to persist memory sidecar: {error}"))?;
            phoenix_memory_post::apply_memory_patch_sidecar(batch, &sidecar);
        }

        let cards = batch
            .entity_cards
            .iter()
            .take(config.card_limit)
            .map(|card| CardReport {
                entity_id: card.entity_id.0.clone(),
                canonical_name: card.identity.canonical_name.clone(),
                aliases: card.identity.aliases.clone(),
                effective_kind: card
                    .identity
                    .effective_kind
                    .as_ref()
                    .map(|kind| format!("{kind:?}")),
                current_state: card
                    .current_state
                    .iter()
                    .map(|state| (state.slot_key.clone(), state.value.clone()))
                    .collect(),
                recent_delta_count: card.recent_deltas.len(),
                active_relationship_count: card.active_relationships.len(),
                active_conflict_count: card.active_conflicts.len(),
                open_gap_count: card.open_gaps.len(),
            })
            .collect::<Vec<_>>();

        reports.push(BatchReport {
            scope_key: batch.scope_key.clone(),
            claim_count: batch.summary.claim_count,
            event_count: batch.summary.event_count,
            state_count: batch.summary.state_count,
            delta_count: batch.summary.delta_count,
            conflict_count: batch.summary.conflict_count,
            gap_count: batch.summary.gap_count,
            entity_card_count: batch.summary.entity_card_count,
            relationship_ledger_count: batch.summary.relationship_ledger_count,
            active_slot_counts: batch.summary.active_slot_counts.clone(),
            slot_claim_counts: count_slot_claims(&batch.claims),
            relation_family_counts: count_relation_families(&batch.claims),
            unresolved_gap_counts: batch.summary.unresolved_gap_counts.clone(),
            source_class_counts: batch.summary.source_class_counts.clone(),
            status_counts: batch.summary.status_counts.clone(),
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
            println!("- claims: {}", report.claim_count);
            println!("- events: {}", report.event_count);
            println!("- states: {}", report.state_count);
            println!("- deltas: {}", report.delta_count);
            println!("- conflicts: {}", report.conflict_count);
            println!("- gaps: {}", report.gap_count);
            println!("- entity cards: {}", report.entity_card_count);
            println!(
                "- relationship ledgers: {}",
                report.relationship_ledger_count
            );
            for (slot, count) in report.active_slot_counts {
                println!("- active slot {slot}: {count}");
            }
            for (slot, count) in report.slot_claim_counts {
                println!("- slot claims {slot}: {count}");
            }
            for (relation_family, count) in report.relation_family_counts {
                println!("- relation family {relation_family}: {count}");
            }
            for card in report.cards {
                println!(
                    "- card {} ({}) :: states={} deltas={} conflicts={} gaps={}",
                    card.entity_id,
                    card.canonical_name,
                    card.current_state.len(),
                    card.recent_delta_count,
                    card.active_conflict_count,
                    card.open_gap_count
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

fn count_slot_claims(claims: &[phoenix_semantic_v2::MemoryClaimAtom]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::<String, usize>::new();
    for claim in claims {
        *counts.entry(claim.slot_key.clone()).or_default() += 1;
    }
    counts
}

fn count_relation_families(
    claims: &[phoenix_semantic_v2::MemoryClaimAtom],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::<String, usize>::new();
    for claim in claims {
        if let Some(relation_family) = claim.relation_family.as_ref() {
            *counts.entry(relation_family.clone()).or_default() += 1;
        }
    }
    counts
}
