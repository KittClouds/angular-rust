use std::path::PathBuf;

use phoenix_analytics::TextAnalytics;
use phoenix_runtime::PhoenixRuntime;
pub use phoenix_runtime::SnapshotPartition;
use phoenix_store_cozo::{SnapshotEnvelope, StoreError};
use phoenix_types::{
    AnalyzeTextRequest, CommitRequest, CommitResult, CreateSessionRequest, GraphDeltaRequest,
    GraphDeltaResult, IngestRequest, IngestResult, QueryRequest, QueryResult, RebuildRequest,
    RebuildResult, RuntimeConfig, RuntimeInitRequest, RuntimeInitResult, ScanArtifact, ScanRequest,
    SessionRecord, SessionState, SessionStateRequest, SessionStats, SessionStatsRequest,
    StoreCommandRequest, StoreCommandResult, StructureArtifact, StructureRequest,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum PhoenixNativeError {
    #[error("native runtime is not open")]
    RuntimeNotOpen,
    #[error(transparent)]
    Store(#[from] StoreError),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhoenixNativeConfig {
    pub runtime: RuntimeConfig,
    pub storage_path: Option<PathBuf>,
}

impl Default for PhoenixNativeConfig {
    fn default() -> Self {
        Self {
            runtime: RuntimeConfig::default(),
            storage_path: None,
        }
    }
}

impl PhoenixNativeConfig {
    pub fn from_init_request(request: &RuntimeInitRequest) -> Self {
        Self {
            runtime: request.config.clone(),
            storage_path: request.storage_path.clone().map(PathBuf::from),
        }
    }

    pub fn to_init_request(&self, force_reset: bool) -> RuntimeInitRequest {
        RuntimeInitRequest {
            config: self.runtime.clone(),
            storage_path: self
                .storage_path
                .as_ref()
                .map(|path| path.to_string_lossy().into_owned()),
            force_reset,
        }
    }
}

pub struct PhoenixNativeRuntime {
    config: PhoenixNativeConfig,
    runtime: PhoenixRuntime,
}

impl PhoenixNativeRuntime {
    pub fn open(config: PhoenixNativeConfig) -> Result<Self, StoreError> {
        let runtime = PhoenixRuntime::open(config.runtime.clone(), config.storage_path.clone())?;
        Ok(Self { config, runtime })
    }

    pub fn from_init_request(request: RuntimeInitRequest) -> Result<Self, StoreError> {
        Self::open(PhoenixNativeConfig::from_init_request(&request))
    }

    pub fn config(&self) -> &PhoenixNativeConfig {
        &self.config
    }

    pub fn runtime(&self) -> &PhoenixRuntime {
        &self.runtime
    }

    pub fn init(&self) -> Result<RuntimeInitResult, StoreError> {
        self.runtime.init()
    }

    pub fn create_session(
        &self,
        request: CreateSessionRequest,
    ) -> Result<SessionRecord, StoreError> {
        self.runtime.create_session(request)
    }

    pub fn ingest(&self, request: IngestRequest) -> Result<IngestResult, StoreError> {
        self.runtime.ingest(request)
    }

    pub fn query(&self, request: QueryRequest) -> Result<QueryResult, StoreError> {
        self.runtime.query(request)
    }

    pub fn commit(&self, request: CommitRequest) -> Result<CommitResult, StoreError> {
        self.runtime.commit(request)
    }

    pub fn rebuild(&self, request: RebuildRequest) -> Result<RebuildResult, StoreError> {
        self.runtime.rebuild(request)
    }

    pub fn scan(&self, request: ScanRequest) -> ScanArtifact {
        self.runtime.scan_text(request)
    }

    pub fn build_structure(&self, request: StructureRequest) -> StructureArtifact {
        self.runtime.build_structure(request)
    }

    pub fn analyze_text(&self, request: AnalyzeTextRequest) -> TextAnalytics {
        self.runtime.analyze_text(&request.text)
    }

    pub fn graph_delta(&self, request: GraphDeltaRequest) -> Result<GraphDeltaResult, StoreError> {
        self.runtime.graph_delta(request)
    }

    pub fn session_state(&self, request: SessionStateRequest) -> Result<SessionState, StoreError> {
        self.runtime.session_state(&request.session_id)
    }

    pub fn session_stats(&self, request: SessionStatsRequest) -> Result<SessionStats, StoreError> {
        self.runtime.session_stats(&request.session_id)
    }

    pub fn export_snapshot(&self) -> Result<Vec<u8>, StoreError> {
        self.runtime.export_snapshot()
    }

    pub fn export_snapshot_partition(
        &self,
        partition: SnapshotPartition,
    ) -> Result<Vec<u8>, StoreError> {
        self.runtime.export_snapshot_partition(partition)
    }

    pub fn import_snapshot(&self, bytes: &[u8]) -> Result<SnapshotEnvelope, StoreError> {
        self.runtime.import_snapshot(bytes)
    }

    pub fn store_command(
        &self,
        request: StoreCommandRequest,
    ) -> Result<StoreCommandResult, StoreError> {
        self.runtime.store_command(request)
    }
}

#[derive(Default)]
pub struct PhoenixNativeHost {
    runtime: Option<PhoenixNativeRuntime>,
}

impl PhoenixNativeHost {
    pub fn is_open(&self) -> bool {
        self.runtime.is_some()
    }

    pub fn config(&self) -> Option<&PhoenixNativeConfig> {
        self.runtime.as_ref().map(PhoenixNativeRuntime::config)
    }

    pub fn open(
        &mut self,
        request: RuntimeInitRequest,
    ) -> Result<RuntimeInitResult, PhoenixNativeError> {
        let runtime = PhoenixNativeRuntime::from_init_request(request)?;
        let result = runtime.init()?;
        self.runtime = Some(runtime);
        Ok(result)
    }

    pub fn open_default(&mut self) -> Result<RuntimeInitResult, PhoenixNativeError> {
        self.open(RuntimeInitRequest {
            config: RuntimeConfig::default(),
            storage_path: None,
            force_reset: false,
        })
    }

    pub fn close(&mut self) -> bool {
        self.runtime.take().is_some()
    }

    pub fn runtime(&self) -> Result<&PhoenixNativeRuntime, PhoenixNativeError> {
        self.runtime
            .as_ref()
            .ok_or(PhoenixNativeError::RuntimeNotOpen)
    }

    pub fn create_session(
        &self,
        request: CreateSessionRequest,
    ) -> Result<SessionRecord, PhoenixNativeError> {
        Ok(self.runtime()?.create_session(request)?)
    }

    pub fn ingest(&self, request: IngestRequest) -> Result<IngestResult, PhoenixNativeError> {
        Ok(self.runtime()?.ingest(request)?)
    }

    pub fn query(&self, request: QueryRequest) -> Result<QueryResult, PhoenixNativeError> {
        Ok(self.runtime()?.query(request)?)
    }

    pub fn commit(&self, request: CommitRequest) -> Result<CommitResult, PhoenixNativeError> {
        Ok(self.runtime()?.commit(request)?)
    }

    pub fn rebuild(&self, request: RebuildRequest) -> Result<RebuildResult, PhoenixNativeError> {
        Ok(self.runtime()?.rebuild(request)?)
    }

    pub fn scan(&self, request: ScanRequest) -> Result<ScanArtifact, PhoenixNativeError> {
        Ok(self.runtime()?.scan(request))
    }

    pub fn build_structure(
        &self,
        request: StructureRequest,
    ) -> Result<StructureArtifact, PhoenixNativeError> {
        Ok(self.runtime()?.build_structure(request))
    }

    pub fn analyze_text(
        &self,
        request: AnalyzeTextRequest,
    ) -> Result<TextAnalytics, PhoenixNativeError> {
        Ok(self.runtime()?.analyze_text(request))
    }

    pub fn graph_delta(
        &self,
        request: GraphDeltaRequest,
    ) -> Result<GraphDeltaResult, PhoenixNativeError> {
        Ok(self.runtime()?.graph_delta(request)?)
    }

    pub fn session_state(
        &self,
        request: SessionStateRequest,
    ) -> Result<SessionState, PhoenixNativeError> {
        Ok(self.runtime()?.session_state(request)?)
    }

    pub fn session_stats(
        &self,
        request: SessionStatsRequest,
    ) -> Result<SessionStats, PhoenixNativeError> {
        Ok(self.runtime()?.session_stats(request)?)
    }

    pub fn export_snapshot(&self) -> Result<Vec<u8>, PhoenixNativeError> {
        Ok(self.runtime()?.export_snapshot()?)
    }

    pub fn export_snapshot_partition(
        &self,
        partition: SnapshotPartition,
    ) -> Result<Vec<u8>, PhoenixNativeError> {
        Ok(self.runtime()?.export_snapshot_partition(partition)?)
    }

    pub fn import_snapshot(&self, bytes: &[u8]) -> Result<SnapshotEnvelope, PhoenixNativeError> {
        Ok(self.runtime()?.import_snapshot(bytes)?)
    }

    pub fn store_command(
        &self,
        request: StoreCommandRequest,
    ) -> Result<StoreCommandResult, PhoenixNativeError> {
        Ok(self.runtime()?.store_command(request)?)
    }
}

pub fn runtime_banner() -> &'static str {
    "phoenix-native foundation ready"
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_types::{
        CreateSessionRequest, IngestDocument, IngestRequest, RuntimeTarget, ScopeKey, SessionId,
        StorageMode,
    };

    #[test]
    fn native_banner_is_stable() {
        assert_eq!(runtime_banner(), "phoenix-native foundation ready");
    }

    #[test]
    fn host_opens_default_runtime() {
        let mut host = PhoenixNativeHost::default();
        let result = host.open_default().expect("open default runtime");

        assert!(result.ready);
        assert!(host.is_open());
        assert_eq!(
            host.config().expect("config").runtime.target,
            RuntimeTarget::Native
        );
    }

    #[test]
    fn host_can_ingest_after_open() {
        let mut host = PhoenixNativeHost::default();
        host.open(RuntimeInitRequest {
            config: RuntimeConfig {
                target: RuntimeTarget::Native,
                storage: StorageMode::CozoMem,
                ..RuntimeConfig::default()
            },
            storage_path: None,
            force_reset: false,
        })
        .expect("open runtime");

        let session = host
            .create_session(CreateSessionRequest {
                session_id: Some(SessionId("native-test-session".to_owned())),
                label: "Native host test".to_owned(),
                scope: ScopeKey::default(),
            })
            .expect("create session");

        let ingest = host
            .ingest(IngestRequest {
                session_id: Some(session.session_id.clone()),
                documents: vec![IngestDocument {
                    document_id: "doc-1".into(),
                    note_id: Some("note-1".into()),
                    title: "Doc".to_owned(),
                    text: "Ryan met Bakuto in the hall.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                commit: false,
            })
            .expect("ingest");

        assert_eq!(ingest.document_count, 1);
    }
}
