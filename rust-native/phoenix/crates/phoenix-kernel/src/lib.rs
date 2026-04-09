use arc_swap::ArcSwap;
use phoenix_graph::{
    GraphBackendError, GraphEdgeRecord, GraphMutationBatch, GraptorGraph, PhoenixGraphBackend,
};
use phoenix_graph_kernel::PhoenixGraphKernel;
use std::sync::{Arc, Mutex};

pub use phoenix_graph_kernel::{
    KernelBiTemporal, KernelCalendarFacet, KernelCheckpointData, KernelCheckpointMeta, KernelEdge,
    KernelEdgeType, KernelEntityCandidate, KernelEntityFacet, KernelEntityResolveRequest,
    KernelEntitySidecar, KernelGraphLayer, KernelGraphSnapshot, KernelJournalEntry,
    KernelMutationBatch, KernelMutationScope, KernelProvenance, KernelRelationClass,
    KernelResolutionFacet, KernelVertex, KernelVertexClass, KernelVertexId, KernelViewRequest,
};

pub struct DeterministicKernel {
    writer: Mutex<PhoenixGraphKernel>,
    active_snapshot: ArcSwap<KernelGraphSnapshot>,
}

impl Default for DeterministicKernel {
    fn default() -> Self {
        let kernel = PhoenixGraphKernel::default();
        let snapshot = kernel.snapshot_kernel();
        Self {
            writer: Mutex::new(kernel),
            active_snapshot: ArcSwap::from_pointee(snapshot),
        }
    }
}

impl DeterministicKernel {
    pub fn apply_batch(&self, batch: KernelMutationBatch) -> Result<(), GraphBackendError> {
        let mut writer = self.writer.lock().expect("kernel writer poisoned");
        writer.apply_kernel_batch(batch)?;
        self.active_snapshot
            .store(Arc::new(writer.snapshot_kernel()));
        Ok(())
    }

    pub fn apply_compat_batch(&self, batch: GraphMutationBatch) -> Result<(), GraphBackendError> {
        self.apply_batch(KernelMutationBatch::from(batch))
    }

    pub fn snapshot(&self) -> Arc<KernelGraphSnapshot> {
        self.active_snapshot.load_full()
    }

    pub fn snapshot_legacy(
        &self,
        include_candidate_graph: bool,
    ) -> Result<GraptorGraph, GraphBackendError> {
        let writer = self.writer.lock().expect("kernel writer poisoned");
        PhoenixGraphBackend::snapshot(&*writer, include_candidate_graph)
    }

    pub fn view_as_of(&self, request: KernelViewRequest) -> KernelGraphSnapshot {
        self.writer
            .lock()
            .expect("kernel writer poisoned")
            .view_as_of(request)
    }

    pub fn candidate_edge_records(&self) -> Result<Vec<GraphEdgeRecord>, GraphBackendError> {
        let writer = self.writer.lock().expect("kernel writer poisoned");
        PhoenixGraphBackend::candidate_edges(&*writer)
    }

    pub fn entity_candidates(
        &self,
        request: KernelEntityResolveRequest,
    ) -> Vec<KernelEntityCandidate> {
        self.writer
            .lock()
            .expect("kernel writer poisoned")
            .entity_candidates(request)
    }

    pub fn entity_sidecar(&self) -> KernelEntitySidecar {
        self.writer
            .lock()
            .expect("kernel writer poisoned")
            .entity_sidecar()
    }

    pub fn rebuild_from_kernel_batches(
        &self,
        batches: Vec<KernelMutationBatch>,
        rebuild_token: Option<String>,
    ) -> Result<(), GraphBackendError> {
        let mut writer = self.writer.lock().expect("kernel writer poisoned");
        writer.rebuild_from_kernel_batches(batches)?;
        writer.set_rebuild_token(rebuild_token);
        self.active_snapshot
            .store(Arc::new(writer.snapshot_kernel()));
        Ok(())
    }

    pub fn install_snapshot(&self, snapshot: KernelGraphSnapshot, rebuild_token: Option<String>) {
        let mut writer = self.writer.lock().expect("kernel writer poisoned");
        *writer = PhoenixGraphKernel::from_snapshot(snapshot.clone(), rebuild_token);
        self.active_snapshot.store(Arc::new(snapshot));
    }

    pub fn invalidate(&self) {
        let mut writer = self.writer.lock().expect("kernel writer poisoned");
        writer.invalidate();
    }

    pub fn rebuild_token(&self) -> Option<String> {
        let writer = self.writer.lock().expect("kernel writer poisoned");
        writer.rebuild_token().map(str::to_owned)
    }

    pub fn set_rebuild_token(&self, token: Option<String>) {
        let mut writer = self.writer.lock().expect("kernel writer poisoned");
        writer.set_rebuild_token(token);
    }
}
