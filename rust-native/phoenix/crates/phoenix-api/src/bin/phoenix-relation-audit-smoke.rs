use std::collections::BTreeMap;
use std::path::PathBuf;

use phoenix_store_native_core::PhoenixArchiveStoreV2;
use phoenix_store_overgraph::PhoenixOvergraphStore;
use phoenix_types::{EntityKind, ScopeKey};
use serde::Serialize;

#[derive(Debug, Clone)]
struct Config {
    store_path: PathBuf,
    relation_type: String,
    limit: usize,
    json: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            store_path: PathBuf::new(),
            relation_type: "relates_to".to_owned(),
            limit: 24,
            json: false,
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct AuditReport {
    store_path: String,
    relation_type: String,
    total_matches: usize,
    document_counts: BTreeMap<String, usize>,
    source_kind_counts: BTreeMap<String, usize>,
    target_kind_counts: BTreeMap<String, usize>,
    pronounish_endpoint_count: usize,
    self_loop_count: usize,
    null_kind_endpoint_count: usize,
    samples: Vec<RelationSample>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RelationSample {
    document_id: String,
    sentence_index: usize,
    chunk_id: Option<String>,
    source_entity_id: String,
    source_label: String,
    source_kind: Option<String>,
    target_entity_id: String,
    target_label: String,
    target_kind: Option<String>,
    chunk_preview: String,
    pronounish_endpoint: bool,
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
    let archives = store
        .load_latest_document_archives(Some(&ScopeKey::default()))
        .map_err(|error| format!("failed to load archives: {error}"))?;

    let mut document_counts = BTreeMap::<String, usize>::new();
    let mut source_kind_counts = BTreeMap::<String, usize>::new();
    let mut target_kind_counts = BTreeMap::<String, usize>::new();
    let mut pronounish_endpoint_count = 0usize;
    let mut self_loop_count = 0usize;
    let mut null_kind_endpoint_count = 0usize;
    let mut samples = Vec::new();
    let mut total_matches = 0usize;

    for archive in &archives {
        let entity_by_id = archive
            .entities
            .iter()
            .map(|entity| (entity.entity_id.0.as_str(), entity))
            .collect::<BTreeMap<_, _>>();
        let chunk_text_by_id = archive
            .chunks
            .iter()
            .map(|chunk| (chunk.chunk_id.0.as_str(), chunk.text.as_str()))
            .collect::<BTreeMap<_, _>>();

        for relation in &archive.relations {
            if relation.edge_type != config.relation_type {
                continue;
            }
            total_matches += 1;
            *document_counts
                .entry(archive.manifest.document_id.clone())
                .or_default() += 1;

            let source = entity_by_id.get(relation.source_entity_id.0.as_str());
            let target = entity_by_id.get(relation.target_entity_id.0.as_str());
            let source_kind = source.and_then(|entity| entity.kind.as_ref());
            let target_kind = target.and_then(|entity| entity.kind.as_ref());
            *source_kind_counts
                .entry(kind_label(source_kind).to_owned())
                .or_default() += 1;
            *target_kind_counts
                .entry(kind_label(target_kind).to_owned())
                .or_default() += 1;

            let source_label = source
                .map(|entity| entity.canonical_name.as_str())
                .unwrap_or(relation.source_entity_id.0.as_str());
            let target_label = target
                .map(|entity| entity.canonical_name.as_str())
                .unwrap_or(relation.target_entity_id.0.as_str());
            self_loop_count += (relation.source_entity_id == relation.target_entity_id) as usize;
            null_kind_endpoint_count += (source_kind.is_none() || target_kind.is_none()) as usize;
            let pronounish_endpoint = is_pronounish(source_label) || is_pronounish(target_label);
            pronounish_endpoint_count += pronounish_endpoint as usize;

            if samples.len() < config.limit {
                let chunk_preview = relation
                    .chunk_id
                    .as_deref()
                    .and_then(|chunk_id| chunk_text_by_id.get(chunk_id))
                    .copied()
                    .map(trim_preview)
                    .unwrap_or_default();
                samples.push(RelationSample {
                    document_id: archive.manifest.document_id.clone(),
                    sentence_index: relation.sentence_index,
                    chunk_id: relation.chunk_id.clone(),
                    source_entity_id: relation.source_entity_id.0.clone(),
                    source_label: source_label.to_owned(),
                    source_kind: source_kind.map(|kind| format!("{kind:?}")),
                    target_entity_id: relation.target_entity_id.0.clone(),
                    target_label: target_label.to_owned(),
                    target_kind: target_kind.map(|kind| format!("{kind:?}")),
                    chunk_preview,
                    pronounish_endpoint,
                });
            }
        }
    }

    let report = AuditReport {
        store_path: config.store_path.display().to_string(),
        relation_type: config.relation_type,
        total_matches,
        document_counts,
        source_kind_counts,
        target_kind_counts,
        pronounish_endpoint_count,
        self_loop_count,
        null_kind_endpoint_count,
        samples,
    };

    if config.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|error| format!("failed to render json: {error}"))?
        );
    } else {
        println!("store: {}", report.store_path);
        println!("relation: {}", report.relation_type);
        println!("total matches: {}", report.total_matches);
        println!("pronounish endpoints: {}", report.pronounish_endpoint_count);
        println!("self loops: {}", report.self_loop_count);
        println!("null-kind endpoints: {}", report.null_kind_endpoint_count);
        for (document_id, count) in &report.document_counts {
            println!("- document {document_id}: {count}");
        }
        for sample in &report.samples {
            println!(
                "- {} [{}] {} ({}) -> {} ({}) pronounish={} chunk={}",
                sample.document_id,
                sample.sentence_index,
                sample.source_label,
                sample.source_kind.as_deref().unwrap_or("unknown"),
                sample.target_label,
                sample.target_kind.as_deref().unwrap_or("unknown"),
                sample.pronounish_endpoint,
                sample.chunk_id.as_deref().unwrap_or("")
            );
            if !sample.chunk_preview.is_empty() {
                println!("  {}", sample.chunk_preview);
            }
        }
    }

    Ok(())
}

fn parse_args(args: &[String]) -> Result<Config, String> {
    let mut config = Config::default();
    let mut index = 1usize;
    while index < args.len() {
        match args[index].as_str() {
            "--store-path" => {
                index += 1;
                let value = args.get(index).ok_or("--store-path requires a value")?;
                config.store_path = PathBuf::from(value);
            }
            "--relation-type" => {
                index += 1;
                let value = args.get(index).ok_or("--relation-type requires a value")?;
                config.relation_type = value.clone();
            }
            "--limit" => {
                index += 1;
                let value = args.get(index).ok_or("--limit requires a value")?;
                config.limit = value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid --limit value '{value}': {error}"))?;
            }
            "--json" => config.json = true,
            flag => return Err(format!("unknown argument: {flag}")),
        }
        index += 1;
    }
    Ok(config)
}

fn kind_label(kind: Option<&EntityKind>) -> &'static str {
    match kind {
        Some(EntityKind::Character) => "Character",
        Some(EntityKind::Location) => "Location",
        Some(EntityKind::Npc) => "Npc",
        Some(EntityKind::Item) => "Item",
        Some(EntityKind::Faction) => "Faction",
        Some(EntityKind::Organization) => "Organization",
        Some(EntityKind::Event) => "Event",
        Some(EntityKind::Concept) => "Concept",
        Some(EntityKind::Other) => "Other",
        None => "None",
    }
}

fn is_pronounish(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "he" | "she"
            | "they"
            | "them"
            | "him"
            | "her"
            | "we"
            | "us"
            | "i"
            | "you"
            | "it"
            | "his"
            | "hers"
            | "their"
            | "our"
            | "my"
            | "your"
            | "who"
            | "whom"
    )
}

fn trim_preview(value: &str) -> String {
    let trimmed = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if trimmed.len() <= 220 {
        return trimmed;
    }
    let mut end = 220usize;
    while end > 0 && !trimmed.is_char_boundary(end) {
        end -= 1;
    }
    let mut owned = trimmed[..end].to_owned();
    owned.push_str("...");
    owned
}
