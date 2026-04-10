use std::env;
use std::path::PathBuf;

use phoenix_hyperbolic::AnnMetric;
use phoenix_store_native_core::{
    AnnIndexFamily, PhoenixArchiveStoreV2, PhoenixEventIdentityPatchStore, PhoenixGraphPatchStore,
    PhoenixMemoryPatchStore,
};
use phoenix_store_overgraph::PhoenixOvergraphStore;
use serde::Serialize;

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct Report {
    store_path: String,
    archive_count: usize,
    scope_key: Option<String>,
    graph_sidecar: String,
    memory_sidecar: String,
    event_identity_sidecar: String,
    chunk_ann_manifest: String,
    claim_ann_manifest: String,
    state_ann_manifest: String,
    event_ann_manifest: String,
}

fn main() {
    match run(parse_store_path()) {
        Ok(report) => println!(
            "{}",
            serde_json::to_string_pretty(&report).expect("serialize report")
        ),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

fn run(store_path: PathBuf) -> Result<Report, String> {
    let store = PhoenixOvergraphStore::open(&store_path).map_err(|error| error.to_string())?;
    let archives = store
        .load_latest_document_archives(None)
        .map_err(|error| format!("load_latest_document_archives failed: {error}"))?;
    let scope = archives
        .first()
        .map(|archive| archive.manifest.scope.clone());
    let scope_key = archives
        .first()
        .map(|archive| archive.manifest.scope_key.clone());

    let graph_sidecar = load_optional(
        || {
            let scope = scope.as_ref().ok_or_else(|| "missing scope".to_owned())?;
            store
                .load_graph_patch_sidecar(scope)
                .map(|value| value.map(|sidecar| format!("graph:g{}", sidecar.generation)))
                .map_err(|error| error.to_string())
        },
        "graph",
    );
    let memory_sidecar = load_optional(
        || {
            let scope = scope.as_ref().ok_or_else(|| "missing scope".to_owned())?;
            store
                .load_memory_patch_sidecar(scope)
                .map(|value| value.map(|sidecar| format!("memory:g{}", sidecar.generation)))
                .map_err(|error| error.to_string())
        },
        "memory",
    );
    let event_identity_sidecar = load_optional(
        || {
            let scope = scope.as_ref().ok_or_else(|| "missing scope".to_owned())?;
            store
                .load_event_identity_patch_sidecar(scope)
                .map(|value| value.map(|sidecar| format!("event-identity:g{}", sidecar.generation)))
                .map_err(|error| error.to_string())
        },
        "event-identity",
    );

    let chunk_ann_manifest = load_manifest(&store, scope.as_ref(), "chunk");
    let claim_ann_manifest = load_manifest(&store, scope.as_ref(), "claim");
    let state_ann_manifest = load_manifest(&store, scope.as_ref(), "state");
    let event_ann_manifest = load_manifest(&store, scope.as_ref(), "event");

    Ok(Report {
        store_path: store_path.display().to_string(),
        archive_count: archives.len(),
        scope_key,
        graph_sidecar,
        memory_sidecar,
        event_identity_sidecar,
        chunk_ann_manifest,
        claim_ann_manifest,
        state_ann_manifest,
        event_ann_manifest,
    })
}

fn load_optional(loader: impl FnOnce() -> Result<Option<String>, String>, label: &str) -> String {
    match loader() {
        Ok(Some(value)) => value,
        Ok(None) => format!("{label}:none"),
        Err(error) => format!("{label}:error:{error}"),
    }
}

fn load_manifest(
    store: &PhoenixOvergraphStore,
    scope: Option<&phoenix_types::ScopeKey>,
    kind: &str,
) -> String {
    let Some(scope) = scope else {
        return format!("{kind}:missing-scope");
    };
    match store.load_ann_manifest(scope, AnnIndexFamily::NodePrototype, Some(kind)) {
        Ok(Some(manifest)) => format!(
            "{kind}:g{}:{}:{}",
            manifest.generation_id.0,
            manifest.count,
            AnnMetric::from_label_or_default(&manifest.metric).label()
        ),
        Ok(None) => format!("{kind}:none"),
        Err(error) => format!("{kind}:error:{error}"),
    }
}

fn parse_store_path() -> PathBuf {
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--store-path" {
            if let Some(value) = args.next() {
                return PathBuf::from(value);
            }
        }
    }
    PathBuf::new()
}
