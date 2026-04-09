use std::collections::BTreeMap;
use std::path::PathBuf;

use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_temporal_post::{
    derive_dirty_scope_review_batches, persist_temporal_patch_sidecar, run_temporal_scope,
};
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
    timex_count: usize,
    anchor_count: usize,
    claim_count: usize,
    constraint_count: usize,
    review_case_count: usize,
    interval_count: usize,
    segment_count: usize,
    conflict_count: usize,
    gap_count: usize,
    memory_card_count: usize,
    axis_counts: BTreeMap<String, usize>,
    source_class_counts: BTreeMap<String, usize>,
    diagnostics: BTreeMap<String, usize>,
    cards: Vec<CardReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CardReport {
    document_id: String,
    event_id: String,
    label: String,
    sentence_index: usize,
    axis_kind: String,
    before_count: usize,
    after_count: usize,
    conflict_count: usize,
    gap_count: usize,
    has_interval: bool,
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
        .map_err(|error| format!("failed to derive temporal review batches: {error}"))?;
    let mut reports = Vec::new();

    for batch in &mut batches {
        run_temporal_scope(batch, now_ms());
        if config.persist_patches {
            let sidecar = persist_temporal_patch_sidecar(&store, batch, now_ms())
                .map_err(|error| format!("failed to persist temporal sidecar: {error}"))?;
            phoenix_temporal_post::apply_temporal_patch_sidecar(batch, &sidecar);
        }

        let cards = batch
            .memory_cards
            .iter()
            .take(config.card_limit)
            .map(|card| CardReport {
                document_id: card.document_id.clone(),
                event_id: card.event_id.clone(),
                label: card.label.clone(),
                sentence_index: card.sentence_index,
                axis_kind: format!("{:?}", card.axis_kind).to_lowercase(),
                before_count: card.before_event_ids.len(),
                after_count: card.after_event_ids.len(),
                conflict_count: card.open_conflict_ids.len(),
                gap_count: card.open_gap_ids.len(),
                has_interval: card.strongest_interval.is_some(),
            })
            .collect::<Vec<_>>();

        reports.push(BatchReport {
            scope_key: batch.scope_key.clone(),
            timex_count: batch.summary.timex_count,
            anchor_count: batch.summary.anchor_count,
            claim_count: batch.summary.claim_count,
            constraint_count: batch.summary.constraint_count,
            review_case_count: batch.summary.review_case_count,
            interval_count: batch.summary.interval_count,
            segment_count: batch.summary.segment_count,
            conflict_count: batch.summary.conflict_count,
            gap_count: batch.summary.gap_count,
            memory_card_count: batch.summary.memory_card_count,
            axis_counts: batch.summary.axis_counts.clone(),
            source_class_counts: batch.summary.source_class_counts.clone(),
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
            println!("- timex: {}", report.timex_count);
            println!("- anchors: {}", report.anchor_count);
            println!("- claims: {}", report.claim_count);
            println!("- constraints: {}", report.constraint_count);
            println!("- review cases: {}", report.review_case_count);
            println!("- intervals: {}", report.interval_count);
            println!("- segments: {}", report.segment_count);
            println!("- conflicts: {}", report.conflict_count);
            println!("- gaps: {}", report.gap_count);
            println!("- cards: {}", report.memory_card_count);
            for (axis, count) in report.axis_counts {
                println!("- axis {axis}: {count}");
            }
            for (source_class, count) in report.source_class_counts {
                println!("- source {source_class}: {count}");
            }
            for (code, count) in report.diagnostics {
                println!("- diagnostic {code}: {count}");
            }
            for card in report.cards {
                println!(
                    "- card {} ({}) axis={} interval={} before={} after={} conflicts={} gaps={}",
                    card.label,
                    card.event_id,
                    card.axis_kind,
                    card.has_interval,
                    card.before_count,
                    card.after_count,
                    card.conflict_count,
                    card.gap_count
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
