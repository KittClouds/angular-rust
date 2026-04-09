use std::collections::BTreeMap;
use std::path::PathBuf;

use phoenix_rel_post::{
    adjudicate_relation_decisions_with_nli, apply_relation_patch_sidecar,
    build_relation_hypotheses, default_relation_type_specs, derive_dirty_scope_review_batches,
    draft_relation_decisions, persist_relation_patch_sidecar, run_primary_relation_lane,
    GlirelModel, NliModel,
};
use phoenix_store_native_core::PhoenixArchiveStoreV2;
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::SessionId;
use serde::Serialize;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
struct SmokeConfig {
    store_path: PathBuf,
    session_id: Option<SessionId>,
    json: bool,
    model_root: Option<PathBuf>,
    nli_model_root: Option<PathBuf>,
    nli_smoke_text: Option<String>,
    nli_smoke_source: Option<String>,
    nli_smoke_target: Option<String>,
    nli_smoke_edge_type: Option<String>,
    persist_patches: bool,
    case_limit: usize,
}

impl Default for SmokeConfig {
    fn default() -> Self {
        Self {
            store_path: PathBuf::new(),
            session_id: None,
            json: false,
            model_root: None,
            nli_model_root: None,
            nli_smoke_text: None,
            nli_smoke_source: None,
            nli_smoke_target: None,
            nli_smoke_edge_type: None,
            persist_patches: false,
            case_limit: 24,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BatchReport {
    scope_key: String,
    review_case_count: usize,
    window_count: usize,
    seeded_pair_count: usize,
    persisted_relation_count: usize,
    entity_profile_count: usize,
    native_relation_type_counts: BTreeMap<String, usize>,
    proposed_relation_counts: BTreeMap<String, usize>,
    decision_counts: BTreeMap<String, usize>,
    support_judgment_count: usize,
    contradiction_judgment_count: usize,
    window_source_counts: BTreeMap<String, usize>,
    anchor_evidence_counts: BTreeMap<String, usize>,
    families_per_window: BTreeMap<String, Vec<String>>,
    rejected_window_reason_counts: BTreeMap<String, usize>,
    archives: Vec<ArchiveReport>,
    windows: Vec<WindowReport>,
    cases: Vec<CaseReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArchiveReport {
    document_id: String,
    sentence_count: usize,
    chunk_count: usize,
    entity_count: usize,
    resolved_mention_count: usize,
    relation_candidate_count: usize,
    persisted_relation_count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct WindowReport {
    window_id: String,
    document_id: String,
    candidate_relation_types: Vec<String>,
    evidence_labels: Vec<String>,
    entities: Vec<String>,
    text_preview: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct CaseReport {
    case_id: String,
    document_id: String,
    window_id: String,
    source: String,
    target: String,
    seed_score_millis: i32,
    decision_status: String,
    seed_evidence: Vec<String>,
    top_relation: Option<String>,
    top_confidence: Option<f32>,
    support_confidence_millis: Option<u32>,
    contradiction_confidence_millis: Option<u32>,
}

fn main() -> Result<(), String> {
    let config = parse_args(&std::env::args().collect::<Vec<_>>())?;
    if let Some(text) = config.nli_smoke_text.as_deref() {
        return run_nli_smoke(&config, text);
    }
    let specs = default_relation_type_specs();

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
        .map_err(|error| format!("failed to derive relation review batches: {error}"))?;
    let model = if let Some(model_root) = config.model_root.as_ref() {
        Some(GlirelModel::load(model_root).map_err(|error| {
            format!(
                "failed to load glirel model {}: {error}",
                model_root.display()
            )
        })?)
    } else {
        None
    };
    let nli = if let Some(model_root) = config.nli_model_root.as_ref() {
        Some(NliModel::load(model_root).map_err(|error| {
            format!("failed to load nli model {}: {error}", model_root.display())
        })?)
    } else {
        None
    };

    let mut reports = Vec::new();
    for batch in &mut batches {
        trim_batch_for_smoke(batch, config.case_limit);
        run_primary_relation_lane(batch, model.as_ref(), &specs)
            .map_err(|error| format!("relation worker failed for {}: {error}", batch.scope_key))?;
        let mut decisions = draft_relation_decisions(batch, &specs);
        if let Some(nli) = nli.as_ref() {
            decisions = adjudicate_relation_decisions_with_nli(batch, &decisions, &specs, nli)
                .map_err(|error| {
                    format!("nli adjudication failed for {}: {error}", batch.scope_key)
                })?;
        }
        if config.persist_patches {
            let sidecar = persist_relation_patch_sidecar(&store, batch, &decisions, now_ms())
                .map_err(|error| format!("failed to persist relation sidecar: {error}"))?;
            apply_relation_patch_sidecar(batch, &sidecar);
        }

        let mut decision_counts = BTreeMap::<String, usize>::new();
        let mut native_relation_type_counts = BTreeMap::<String, usize>::new();
        let mut proposed_relation_counts = BTreeMap::<String, usize>::new();
        let mut support_judgment_count = 0usize;
        let mut contradiction_judgment_count = 0usize;
        let archives = store
            .load_latest_document_archives(Some(&batch.scope))
            .map_err(|error| {
                format!(
                    "failed to inspect archives for {}: {error}",
                    batch.scope_key
                )
            })?
            .into_iter()
            .map(|archive| ArchiveReport {
                document_id: archive.manifest.document_id,
                sentence_count: archive.sentences.len(),
                chunk_count: archive.chunks.len(),
                entity_count: archive.entities.len(),
                resolved_mention_count: archive.resolved_mentions.len(),
                relation_candidate_count: archive.relation_candidates.len(),
                persisted_relation_count: archive.relations.len(),
            })
            .collect::<Vec<_>>();
        for decision in &decisions {
            *decision_counts
                .entry(format!("{:?}", decision.kind).to_lowercase())
                .or_default() += 1;
            if decision.support_confidence_millis.is_some() {
                support_judgment_count += 1;
            }
            if decision.contradiction_confidence_millis.is_some() {
                contradiction_judgment_count += 1;
            }
        }
        for relation in &batch.persisted_relations {
            *native_relation_type_counts
                .entry(relation.edge_type.clone())
                .or_default() += 1;
        }
        for window in &batch.windows {
            for relation_type in &window.candidate_relation_types {
                *native_relation_type_counts
                    .entry(format!("window::{relation_type}"))
                    .or_default() += 1;
            }
        }
        for case in &batch.review_cases {
            if let Some(top) = case.glirel_predictions.first() {
                *proposed_relation_counts
                    .entry(top.relation.clone())
                    .or_default() += 1;
            }
        }
        let windows = batch
            .windows
            .iter()
            .take(config.case_limit)
            .map(|window| WindowReport {
                window_id: window.window_id.clone(),
                document_id: window.document_id.clone(),
                candidate_relation_types: window.candidate_relation_types.clone(),
                evidence_labels: window.evidence_labels.clone(),
                entities: window
                    .entities
                    .iter()
                    .map(|entity| format!("{}:{}", entity.entity_id.0, entity.surface))
                    .collect(),
                text_preview: preview_text(&window.text, 180),
            })
            .collect::<Vec<_>>();
        let cases = batch
            .review_cases
            .iter()
            .take(config.case_limit)
            .map(|case| CaseReport {
                case_id: case.case_id.clone(),
                document_id: case.document_id.clone(),
                window_id: case.window_id.clone(),
                source: case.source_name.clone(),
                target: case.target_name.clone(),
                seed_score_millis: case.seed_score_millis,
                decision_status: case.decision_status.clone(),
                seed_evidence: case.seed_evidence.clone(),
                top_relation: case
                    .glirel_predictions
                    .first()
                    .map(|row| row.relation.clone()),
                top_confidence: case.glirel_predictions.first().map(|row| row.confidence),
                support_confidence_millis: decisions
                    .iter()
                    .find(|decision| decision.case_id == case.case_id)
                    .and_then(|decision| decision.support_confidence_millis),
                contradiction_confidence_millis: decisions
                    .iter()
                    .find(|decision| decision.case_id == case.case_id)
                    .and_then(|decision| decision.contradiction_confidence_millis),
            })
            .collect::<Vec<_>>();
        reports.push(BatchReport {
            scope_key: batch.scope_key.clone(),
            review_case_count: batch.review_cases.len(),
            window_count: batch.windows.len(),
            seeded_pair_count: batch.window_build_stats.seeded_pair_count,
            persisted_relation_count: batch.persisted_relations.len(),
            entity_profile_count: batch.entity_profiles.len(),
            native_relation_type_counts,
            proposed_relation_counts,
            decision_counts,
            support_judgment_count,
            contradiction_judgment_count,
            window_source_counts: batch.window_build_stats.window_source_counts.clone(),
            anchor_evidence_counts: batch.window_build_stats.anchor_evidence_counts.clone(),
            families_per_window: batch.window_build_stats.families_per_window.clone(),
            rejected_window_reason_counts: batch
                .window_build_stats
                .rejected_window_reason_counts
                .clone(),
            archives,
            windows,
            cases,
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
            println!("- windows: {}", report.window_count);
            println!("- review cases: {}", report.review_case_count);
            println!("- seeded pairs: {}", report.seeded_pair_count);
            println!("- persisted relations: {}", report.persisted_relation_count);
            println!("- entity profiles: {}", report.entity_profile_count);
            for archive in report.archives {
                println!(
                    "- archive {} :: sentences={} chunks={} entities={} resolved_mentions={} relation_candidates={} relations={}",
                    archive.document_id,
                    archive.sentence_count,
                    archive.chunk_count,
                    archive.entity_count,
                    archive.resolved_mention_count,
                    archive.relation_candidate_count,
                    archive.persisted_relation_count
                );
            }
            for (kind, count) in report.native_relation_type_counts {
                println!("- native relation type {kind}: {count}");
            }
            for (kind, count) in report.window_source_counts {
                println!("- window source {kind}: {count}");
            }
            for (kind, count) in report.anchor_evidence_counts {
                println!("- anchor evidence {kind}: {count}");
            }
            for (kind, count) in report.rejected_window_reason_counts {
                println!("- rejected window {kind}: {count}");
            }
            for (kind, count) in report.proposed_relation_counts {
                println!("- proposed relation {kind}: {count}");
            }
            for (kind, count) in report.decision_counts {
                println!("- {kind}: {count}");
            }
            println!("- support judgments: {}", report.support_judgment_count);
            println!(
                "- contradiction judgments: {}",
                report.contradiction_judgment_count
            );
            for window in report.windows {
                println!(
                    "- window {} :: {:?} :: {:?} :: {}",
                    window.window_id,
                    window.candidate_relation_types,
                    window.evidence_labels,
                    window.text_preview
                );
            }
            for case in report.cases {
                println!(
                    "- {} :: {} -> {} :: seed={} :: top={} ({:?}) :: {} :: {:?}",
                    case.document_id,
                    case.source,
                    case.target,
                    case.seed_score_millis,
                    case.top_relation.unwrap_or_else(|| "none".to_owned()),
                    case.top_confidence,
                    case.decision_status,
                    case.seed_evidence
                );
                if case.support_confidence_millis.is_some()
                    || case.contradiction_confidence_millis.is_some()
                {
                    println!(
                        "  nli :: support={:?} contradiction={:?}",
                        case.support_confidence_millis, case.contradiction_confidence_millis
                    );
                }
            }
        }
    }

    Ok(())
}

fn parse_args(args: &[String]) -> Result<SmokeConfig, String> {
    let mut config = SmokeConfig::default();
    config.store_path = parse_path_arg(args, "--store-path").unwrap_or_default();
    config.session_id = parse_string_arg(args, "--session-id").map(SessionId);
    config.model_root = parse_path_arg(args, "--model-root");
    config.nli_model_root = parse_path_arg(args, "--nli-model-root");
    config.nli_smoke_text = parse_string_arg(args, "--nli-smoke-text");
    config.nli_smoke_source = parse_string_arg(args, "--nli-smoke-source");
    config.nli_smoke_target = parse_string_arg(args, "--nli-smoke-target");
    config.nli_smoke_edge_type = parse_string_arg(args, "--nli-smoke-edge-type");
    config.json = args.iter().any(|arg| arg == "--json");
    config.persist_patches = args.iter().any(|arg| arg == "--persist-patches");
    config.case_limit = parse_usize_arg(args, "--case-limit").unwrap_or(config.case_limit);
    Ok(config)
}

fn run_nli_smoke(config: &SmokeConfig, text: &str) -> Result<(), String> {
    let model_root = config
        .nli_model_root
        .as_ref()
        .ok_or_else(|| "--nli-model-root is required for NLI smoke".to_owned())?;
    let source = config
        .nli_smoke_source
        .as_deref()
        .ok_or_else(|| "--nli-smoke-source is required".to_owned())?;
    let target = config
        .nli_smoke_target
        .as_deref()
        .ok_or_else(|| "--nli-smoke-target is required".to_owned())?;
    let edge_type = config
        .nli_smoke_edge_type
        .as_deref()
        .ok_or_else(|| "--nli-smoke-edge-type is required".to_owned())?;
    let model = NliModel::load(model_root)
        .map_err(|error| format!("failed to load nli model {}: {error}", model_root.display()))?;
    let forward = build_relation_hypotheses(edge_type, source, target);
    let reverse = build_relation_hypotheses(edge_type, target, source);
    let judgment = model
        .judge_relation(text, &forward, &reverse)
        .map_err(|error| format!("nli smoke failed: {error}"))?;
    if config.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&judgment)
                .map_err(|error| format!("failed to render nli smoke json: {error}"))?
        );
    } else {
        println!("edgeType: {edge_type}");
        println!("bestHypothesis: {}", judgment.best_hypothesis);
        println!("usedReverse: {}", judgment.used_reverse);
        println!(
            "forward: contradiction={:.3} entailment={:.3} neutral={:.3}",
            judgment.forward.contradiction, judgment.forward.entailment, judgment.forward.neutral
        );
        if let Some(reverse) = judgment.reverse {
            println!(
                "reverse: contradiction={:.3} entailment={:.3} neutral={:.3}",
                reverse.contradiction, reverse.entailment, reverse.neutral
            );
        }
    }
    Ok(())
}

fn parse_string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].clone())
}

fn parse_path_arg(args: &[String], flag: &str) -> Option<PathBuf> {
    parse_string_arg(args, flag).map(PathBuf::from)
}

fn parse_usize_arg(args: &[String], flag: &str) -> Option<usize> {
    parse_string_arg(args, flag).and_then(|value| value.parse::<usize>().ok())
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

fn preview_text(text: &str, limit: usize) -> String {
    let trimmed = text.trim();
    if trimmed.chars().count() <= limit {
        return trimmed.to_owned();
    }
    let mut preview = trimmed.chars().take(limit).collect::<String>();
    preview.push_str("...");
    preview
}

fn trim_batch_for_smoke(batch: &mut phoenix_rel_post::RelationScopeReviewBatch, case_limit: usize) {
    if case_limit == 0 || batch.review_cases.len() <= case_limit {
        return;
    }
    let mut keep_windows = BTreeSet::new();
    let mut kept_cases = Vec::new();
    for window in &batch.windows {
        let mut window_cases = batch
            .review_cases
            .iter()
            .filter(|case| case.window_id == window.window_id)
            .cloned()
            .collect::<Vec<_>>();
        if window_cases.is_empty() {
            continue;
        }
        let remaining = case_limit.saturating_sub(kept_cases.len());
        if remaining == 0 {
            break;
        }
        window_cases.truncate(remaining);
        keep_windows.insert(window.window_id.clone());
        kept_cases.extend(window_cases);
        if kept_cases.len() >= case_limit {
            break;
        }
    }
    batch.review_cases = kept_cases;
    batch
        .windows
        .retain(|window| keep_windows.contains(&window.window_id));
}
