use std::sync::{Arc, Mutex};

use phoenix_native::{runtime_banner, PhoenixNativeHost, SnapshotPartition};
use phoenix_types::{
    AnalyzeTextRequest, CommitRequest, CreateSessionRequest, GraphDeltaRequest, IngestRequest,
    QueryRequest, RebuildRequest, RuntimeConfig, RuntimeInitRequest, RuntimeInitResult,
    RuntimeTarget, ScanRequest, SessionStateRequest, SessionStatsRequest, SnapshotPolicy,
    StorageMode, StoreCommandRequest,
};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

#[derive(Default)]
struct PhoenixDesktopState {
    host: PhoenixNativeHost,
    last_init: Option<RuntimeInitResult>,
}

#[derive(Clone, Default)]
pub struct PhoenixApiImpl {
    state: Arc<Mutex<PhoenixDesktopState>>,
}

#[taurpc::ipc_type]
#[serde(rename_all = "camelCase")]
pub struct DesktopFeatureFlags {
    pub scanner: bool,
    pub structure: bool,
    pub graptor: bool,
    pub gldr: bool,
    pub semantic: bool,
    pub candidate_graph: bool,
}

#[taurpc::ipc_type]
#[serde(rename_all = "camelCase")]
pub struct DesktopInitRequest {
    pub force_reset: bool,
    pub storage_path: Option<String>,
    pub storage: Option<String>,
}

#[taurpc::ipc_type]
#[serde(rename_all = "camelCase")]
pub struct DesktopRelationCount {
    pub relation: String,
    pub rows: usize,
}

#[taurpc::ipc_type]
#[serde(rename_all = "camelCase")]
pub struct DesktopDiagnostic {
    pub code: String,
    pub message: String,
}

#[taurpc::ipc_type]
#[serde(rename_all = "camelCase")]
pub struct DesktopRuntimeInfo {
    pub banner: String,
    pub target: String,
    pub ready: bool,
    pub storage: String,
    pub storage_path: Option<String>,
    pub feature_flags: DesktopFeatureFlags,
    pub schema_version: String,
    pub relation_count: usize,
    pub relation_counts: Vec<DesktopRelationCount>,
    pub diagnostics: Vec<DesktopDiagnostic>,
}

#[taurpc::ipc_type]
#[serde(rename_all = "camelCase")]
pub struct DesktopSnapshotImportResult {
    pub schema_version: String,
    pub relation_count: usize,
    pub created_at: i64,
    pub relation_names: Vec<String>,
    pub checksum: Option<String>,
}

#[taurpc::procedures(path = "phoenix", export_to = "../src/app/generated/phoenix-taurpc.ts")]
pub trait PhoenixApi {
    async fn runtime_info() -> DesktopRuntimeInfo;
    async fn init_runtime(request: DesktopInitRequest) -> Result<DesktopRuntimeInfo, String>;
    async fn close_runtime() -> bool;
    async fn create_session_json(request_json: String) -> Result<String, String>;
    async fn ingest_json(request_json: String) -> Result<String, String>;
    async fn query_json(request_json: String) -> Result<String, String>;
    async fn commit_json(request_json: String) -> Result<String, String>;
    async fn rebuild_json(request_json: String) -> Result<String, String>;
    async fn scan_json(request_json: String) -> Result<String, String>;
    async fn build_structure_json(request_json: String) -> Result<String, String>;
    async fn analyze_text_json(request_json: String) -> Result<String, String>;
    async fn graph_delta_json(request_json: String) -> Result<String, String>;
    async fn session_state_json(request_json: String) -> Result<String, String>;
    async fn session_stats_json(request_json: String) -> Result<String, String>;
    async fn export_snapshot(partition: String) -> Result<Vec<u8>, String>;
    async fn import_snapshot(bytes: Vec<u8>) -> Result<DesktopSnapshotImportResult, String>;
    async fn store_command(command: String, payload_json: String) -> Result<String, String>;
}

#[taurpc::resolvers]
impl PhoenixApi for PhoenixApiImpl {
    async fn runtime_info(self) -> DesktopRuntimeInfo {
        let guard = match self.state.lock() {
            Ok(guard) => guard,
            Err(_) => return desktop_runtime_info(None, None),
        };
        desktop_runtime_info(guard.host.config(), guard.last_init.as_ref())
    }

    async fn init_runtime(self, request: DesktopInitRequest) -> Result<DesktopRuntimeInfo, String> {
        let mut guard = self.lock_state()?;
        if request.force_reset {
            let _ = guard.host.close();
        }

        let init_request = build_init_request(&request);
        let result = guard
            .host
            .open(init_request)
            .map_err(|error| error.to_string())?;
        guard.last_init = Some(result.clone());
        Ok(desktop_runtime_info(guard.host.config(), guard.last_init.as_ref()))
    }

    async fn close_runtime(self) -> bool {
        match self.state.lock() {
            Ok(mut guard) => {
                guard.last_init = None;
                guard.host.close()
            }
            Err(_) => false,
        }
    }

    async fn create_session_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<CreateSessionRequest, _, _>(request_json, |host, request| {
            host.create_session(request)
        })
    }

    async fn ingest_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<IngestRequest, _, _>(request_json, |host, request| host.ingest(request))
    }

    async fn query_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<QueryRequest, _, _>(request_json, |host, request| host.query(request))
    }

    async fn commit_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<CommitRequest, _, _>(request_json, |host, request| host.commit(request))
    }

    async fn rebuild_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<RebuildRequest, _, _>(request_json, |host, request| {
            host.rebuild(request)
        })
    }

    async fn scan_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<ScanRequest, _, _>(request_json, |host, request| host.scan(request))
    }

    async fn build_structure_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<phoenix_types::StructureRequest, _, _>(request_json, |host, request| {
            host.build_structure(request)
        })
    }

    async fn analyze_text_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<AnalyzeTextRequest, _, _>(request_json, |host, request| {
            host.analyze_text(request)
        })
    }

    async fn graph_delta_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<GraphDeltaRequest, _, _>(request_json, |host, request| {
            host.graph_delta(request)
        })
    }

    async fn session_state_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<SessionStateRequest, _, _>(request_json, |host, request| {
            host.session_state(request)
        })
    }

    async fn session_stats_json(self, request_json: String) -> Result<String, String> {
        self.with_host_json::<SessionStatsRequest, _, _>(request_json, |host, request| {
            host.session_stats(request)
        })
    }

    async fn export_snapshot(self, partition: String) -> Result<Vec<u8>, String> {
        let guard = self.lock_state()?;
        guard
            .host
            .export_snapshot_partition(parse_snapshot_partition(&partition)?)
            .map_err(|error| error.to_string())
    }

    async fn import_snapshot(self, bytes: Vec<u8>) -> Result<DesktopSnapshotImportResult, String> {
        let mut guard = self.lock_state()?;
        let envelope = guard
            .host
            .import_snapshot(&bytes)
            .map_err(|error| error.to_string())?;
        let refreshed = guard
            .host
            .runtime()
            .and_then(|runtime| {
                runtime
                    .init()
                    .map_err(phoenix_native::PhoenixNativeError::from)
            })
            .map_err(|error: phoenix_native::PhoenixNativeError| error.to_string())?;
        guard.last_init = Some(refreshed);
        let relation_names = envelope.relations.keys().cloned().collect::<Vec<_>>();
        Ok(DesktopSnapshotImportResult {
            schema_version: envelope.schema_version,
            relation_count: envelope.relation_count,
            created_at: envelope.created_at,
            relation_names,
            checksum: envelope.checksum,
        })
    }

    async fn store_command(self, command: String, payload_json: String) -> Result<String, String> {
        let payload = if payload_json.trim().is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&payload_json)
                .map_err(|error| format!("invalid store command payload JSON: {error}"))?
        };

        let guard = self.lock_state()?;
        let result = guard
            .host
            .store_command(StoreCommandRequest { command, payload })
            .map_err(|error| error.to_string())?;
        serialize_json(&result)
    }
}

impl PhoenixApiImpl {
    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, PhoenixDesktopState>, String> {
        self.state
            .lock()
            .map_err(|_| "phoenix desktop state lock poisoned".to_owned())
    }

    fn with_host_json<Request, Response, F>(
        &self,
        request_json: String,
        op: F,
    ) -> Result<String, String>
    where
        Request: DeserializeOwned,
        Response: Serialize,
        F: FnOnce(&PhoenixNativeHost, Request) -> Result<Response, phoenix_native::PhoenixNativeError>,
    {
        let request = parse_json::<Request>(&request_json)?;
        let guard = self.lock_state()?;
        let response = op(&guard.host, request).map_err(|error| error.to_string())?;
        serialize_json(&response)
    }
}

fn build_init_request(request: &DesktopInitRequest) -> RuntimeInitRequest {
    let storage = request
        .storage
        .as_deref()
        .and_then(parse_storage_mode)
        .unwrap_or_else(|| {
            if request.storage_path.is_some() {
                StorageMode::CozoSqlite
            } else {
                StorageMode::CozoMem
            }
        });

    RuntimeInitRequest {
        config: RuntimeConfig {
            target: RuntimeTarget::Native,
            storage,
            snapshot_policy: SnapshotPolicy::Manual,
            feature_flags: phoenix_types::FeatureFlags {
                scanner: true,
                structure: true,
                graptor: true,
                gldr: true,
                semantic: true,
                candidate_graph: true,
            },
        },
        storage_path: request.storage_path.clone(),
        force_reset: request.force_reset,
    }
}

fn desktop_runtime_info(
    config: Option<&phoenix_native::PhoenixNativeConfig>,
    init_result: Option<&phoenix_types::RuntimeInitResult>,
) -> DesktopRuntimeInfo {
    let runtime = config
        .map(|config| config.runtime.clone())
        .unwrap_or_else(default_runtime_config);
    DesktopRuntimeInfo {
        banner: runtime_banner().to_owned(),
        target: "native".to_owned(),
        ready: init_result.map(|result| result.ready).unwrap_or(false),
        storage: storage_mode_name(runtime.storage).to_owned(),
        storage_path: config
            .and_then(|config| config.storage_path.as_ref())
            .map(|path| path.to_string_lossy().into_owned()),
        feature_flags: DesktopFeatureFlags {
            scanner: runtime.feature_flags.scanner,
            structure: runtime.feature_flags.structure,
            graptor: runtime.feature_flags.graptor,
            gldr: runtime.feature_flags.gldr,
            semantic: runtime.feature_flags.semantic,
            candidate_graph: runtime.feature_flags.candidate_graph,
        },
        schema_version: init_result
            .map(|result| result.schema_version.clone())
            .unwrap_or_default(),
        relation_count: init_result.map(|result| result.relation_count).unwrap_or(0),
        relation_counts: init_result
            .map(|result| {
                result
                    .relation_counts
                    .iter()
                    .map(|relation| DesktopRelationCount {
                        relation: relation.relation.clone(),
                        rows: relation.rows,
                    })
                    .collect()
            })
            .unwrap_or_default(),
        diagnostics: init_result
            .map(|result| {
                result
                    .diagnostics
                    .iter()
                    .map(|diagnostic| DesktopDiagnostic {
                        code: diagnostic.code.clone(),
                        message: diagnostic.message.clone(),
                    })
                    .collect()
            })
            .unwrap_or_default(),
    }
}

fn default_runtime_config() -> RuntimeConfig {
    RuntimeConfig {
        target: RuntimeTarget::Native,
        storage: StorageMode::CozoMem,
        snapshot_policy: SnapshotPolicy::Manual,
        feature_flags: phoenix_types::FeatureFlags {
            scanner: true,
            structure: true,
            graptor: true,
            gldr: true,
            semantic: true,
            candidate_graph: true,
        },
    }
}

fn parse_storage_mode(value: &str) -> Option<StorageMode> {
    match value {
        "cozoMem" | "cozo_mem" | "mem" => Some(StorageMode::CozoMem),
        "cozoSqlite" | "cozo_sqlite" | "sqlite" => Some(StorageMode::CozoSqlite),
        _ => None,
    }
}

fn storage_mode_name(mode: StorageMode) -> &'static str {
    match mode {
        StorageMode::CozoMem => "cozoMem",
        StorageMode::CozoSqlite => "cozoSqlite",
    }
}

fn parse_snapshot_partition(value: &str) -> Result<SnapshotPartition, String> {
    match value {
        "all" => Ok(SnapshotPartition::All),
        "content" => Ok(SnapshotPartition::Content),
        "derived" => Ok(SnapshotPartition::Derived),
        other => Err(format!("unknown snapshot partition: {other}")),
    }
}

fn parse_json<T: DeserializeOwned>(json: &str) -> Result<T, String> {
    serde_json::from_str(json).map_err(|error| format!("invalid Phoenix JSON payload: {error}"))
}

fn serialize_json<T: Serialize>(value: &T) -> Result<String, String> {
    serde_json::to_string(value).map_err(|error| format!("failed to serialize Phoenix result: {error}"))
}
