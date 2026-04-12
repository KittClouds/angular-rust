use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

use serde::Serialize;

#[derive(Clone, Copy, Debug)]
pub(crate) enum GraphRuntimeMetric {
    LoadProjectionKernel,
    RankedWorldState,
    RankedHistory,
    RankedCausalExplanation,
    RetrievedWorldState,
    RetrievedHistory,
    RetrievedCausalExplanation,
    RetrieveQuerySeeds,
    BuildRegionFromSnapshot,
    BuildRegionFromView,
    EmbedQuery,
    QueryEmbedderLoad,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRuntimeTiming {
    pub count: u64,
    pub total_us: u64,
    pub mean_us: f64,
    pub max_us: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRuntimeTelemetrySnapshot {
    pub load_projection_kernel: GraphRuntimeTiming,
    pub ranked_world_state: GraphRuntimeTiming,
    pub ranked_history: GraphRuntimeTiming,
    pub ranked_causal_explanation: GraphRuntimeTiming,
    pub retrieved_world_state: GraphRuntimeTiming,
    pub retrieved_history: GraphRuntimeTiming,
    pub retrieved_causal_explanation: GraphRuntimeTiming,
    pub retrieve_query_seeds: GraphRuntimeTiming,
    pub build_region_from_snapshot: GraphRuntimeTiming,
    pub build_region_from_view: GraphRuntimeTiming,
    pub embed_query: GraphRuntimeTiming,
    pub query_embedder_load: GraphRuntimeTiming,
    pub loaded_asserted_vertex_total: u64,
    pub loaded_asserted_edge_total: u64,
    pub loaded_candidate_edge_total: u64,
    pub built_region_vertex_total: u64,
    pub built_region_asserted_edge_total: u64,
    pub built_region_candidate_edge_total: u64,
}

#[derive(Default)]
struct AtomicTiming {
    count: AtomicU64,
    total_us: AtomicU64,
    max_us: AtomicU64,
}

impl AtomicTiming {
    fn record(&self, elapsed_us: u64) {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.total_us.fetch_add(elapsed_us, Ordering::Relaxed);
        self.max_us.fetch_max(elapsed_us, Ordering::Relaxed);
    }

    fn snapshot(&self) -> GraphRuntimeTiming {
        let count = self.count.load(Ordering::Relaxed);
        let total_us = self.total_us.load(Ordering::Relaxed);
        GraphRuntimeTiming {
            count,
            total_us,
            mean_us: if count == 0 {
                0.0
            } else {
                total_us as f64 / count as f64
            },
            max_us: self.max_us.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.total_us.store(0, Ordering::Relaxed);
        self.max_us.store(0, Ordering::Relaxed);
    }
}

#[derive(Default)]
struct GraphRuntimeTelemetryState {
    load_projection_kernel: AtomicTiming,
    ranked_world_state: AtomicTiming,
    ranked_history: AtomicTiming,
    ranked_causal_explanation: AtomicTiming,
    retrieved_world_state: AtomicTiming,
    retrieved_history: AtomicTiming,
    retrieved_causal_explanation: AtomicTiming,
    retrieve_query_seeds: AtomicTiming,
    build_region_from_snapshot: AtomicTiming,
    build_region_from_view: AtomicTiming,
    embed_query: AtomicTiming,
    query_embedder_load: AtomicTiming,
    loaded_asserted_vertex_total: AtomicU64,
    loaded_asserted_edge_total: AtomicU64,
    loaded_candidate_edge_total: AtomicU64,
    built_region_vertex_total: AtomicU64,
    built_region_asserted_edge_total: AtomicU64,
    built_region_candidate_edge_total: AtomicU64,
}

impl GraphRuntimeTelemetryState {
    fn timing(&self, metric: GraphRuntimeMetric) -> &AtomicTiming {
        match metric {
            GraphRuntimeMetric::LoadProjectionKernel => &self.load_projection_kernel,
            GraphRuntimeMetric::RankedWorldState => &self.ranked_world_state,
            GraphRuntimeMetric::RankedHistory => &self.ranked_history,
            GraphRuntimeMetric::RankedCausalExplanation => &self.ranked_causal_explanation,
            GraphRuntimeMetric::RetrievedWorldState => &self.retrieved_world_state,
            GraphRuntimeMetric::RetrievedHistory => &self.retrieved_history,
            GraphRuntimeMetric::RetrievedCausalExplanation => &self.retrieved_causal_explanation,
            GraphRuntimeMetric::RetrieveQuerySeeds => &self.retrieve_query_seeds,
            GraphRuntimeMetric::BuildRegionFromSnapshot => &self.build_region_from_snapshot,
            GraphRuntimeMetric::BuildRegionFromView => &self.build_region_from_view,
            GraphRuntimeMetric::EmbedQuery => &self.embed_query,
            GraphRuntimeMetric::QueryEmbedderLoad => &self.query_embedder_load,
        }
    }

    fn snapshot(&self) -> GraphRuntimeTelemetrySnapshot {
        GraphRuntimeTelemetrySnapshot {
            load_projection_kernel: self.load_projection_kernel.snapshot(),
            ranked_world_state: self.ranked_world_state.snapshot(),
            ranked_history: self.ranked_history.snapshot(),
            ranked_causal_explanation: self.ranked_causal_explanation.snapshot(),
            retrieved_world_state: self.retrieved_world_state.snapshot(),
            retrieved_history: self.retrieved_history.snapshot(),
            retrieved_causal_explanation: self.retrieved_causal_explanation.snapshot(),
            retrieve_query_seeds: self.retrieve_query_seeds.snapshot(),
            build_region_from_snapshot: self.build_region_from_snapshot.snapshot(),
            build_region_from_view: self.build_region_from_view.snapshot(),
            embed_query: self.embed_query.snapshot(),
            query_embedder_load: self.query_embedder_load.snapshot(),
            loaded_asserted_vertex_total: self.loaded_asserted_vertex_total.load(Ordering::Relaxed),
            loaded_asserted_edge_total: self.loaded_asserted_edge_total.load(Ordering::Relaxed),
            loaded_candidate_edge_total: self.loaded_candidate_edge_total.load(Ordering::Relaxed),
            built_region_vertex_total: self.built_region_vertex_total.load(Ordering::Relaxed),
            built_region_asserted_edge_total: self
                .built_region_asserted_edge_total
                .load(Ordering::Relaxed),
            built_region_candidate_edge_total: self
                .built_region_candidate_edge_total
                .load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.load_projection_kernel.reset();
        self.ranked_world_state.reset();
        self.ranked_history.reset();
        self.ranked_causal_explanation.reset();
        self.retrieved_world_state.reset();
        self.retrieved_history.reset();
        self.retrieved_causal_explanation.reset();
        self.retrieve_query_seeds.reset();
        self.build_region_from_snapshot.reset();
        self.build_region_from_view.reset();
        self.embed_query.reset();
        self.query_embedder_load.reset();
        self.loaded_asserted_vertex_total
            .store(0, Ordering::Relaxed);
        self.loaded_asserted_edge_total.store(0, Ordering::Relaxed);
        self.loaded_candidate_edge_total.store(0, Ordering::Relaxed);
        self.built_region_vertex_total.store(0, Ordering::Relaxed);
        self.built_region_asserted_edge_total
            .store(0, Ordering::Relaxed);
        self.built_region_candidate_edge_total
            .store(0, Ordering::Relaxed);
    }
}

fn telemetry() -> &'static GraphRuntimeTelemetryState {
    static TELEMETRY: OnceLock<GraphRuntimeTelemetryState> = OnceLock::new();
    TELEMETRY.get_or_init(GraphRuntimeTelemetryState::default)
}

pub(crate) struct GraphRuntimeMeasure {
    metric: GraphRuntimeMetric,
    started_at: Instant,
}

impl Drop for GraphRuntimeMeasure {
    fn drop(&mut self) {
        let elapsed_us = self.started_at.elapsed().as_micros() as u64;
        telemetry().timing(self.metric).record(elapsed_us);
    }
}

pub(crate) fn measure_graph_runtime(metric: GraphRuntimeMetric) -> GraphRuntimeMeasure {
    GraphRuntimeMeasure {
        metric,
        started_at: Instant::now(),
    }
}

pub(crate) fn record_projection_kernel_load(
    asserted_vertices: usize,
    asserted_edges: usize,
    candidate_edges: usize,
) {
    let telemetry = telemetry();
    telemetry
        .loaded_asserted_vertex_total
        .fetch_add(asserted_vertices as u64, Ordering::Relaxed);
    telemetry
        .loaded_asserted_edge_total
        .fetch_add(asserted_edges as u64, Ordering::Relaxed);
    telemetry
        .loaded_candidate_edge_total
        .fetch_add(candidate_edges as u64, Ordering::Relaxed);
}

pub(crate) fn record_region_build(
    vertex_count: usize,
    asserted_edge_count: usize,
    candidate_edge_count: usize,
) {
    let telemetry = telemetry();
    telemetry
        .built_region_vertex_total
        .fetch_add(vertex_count as u64, Ordering::Relaxed);
    telemetry
        .built_region_asserted_edge_total
        .fetch_add(asserted_edge_count as u64, Ordering::Relaxed);
    telemetry
        .built_region_candidate_edge_total
        .fetch_add(candidate_edge_count as u64, Ordering::Relaxed);
}

pub fn reset_graph_runtime_telemetry() {
    telemetry().reset();
}

pub fn snapshot_graph_runtime_telemetry() -> GraphRuntimeTelemetrySnapshot {
    telemetry().snapshot()
}

#[cfg(test)]
mod tests {
    use super::{
        measure_graph_runtime, record_projection_kernel_load, record_region_build,
        reset_graph_runtime_telemetry, snapshot_graph_runtime_telemetry, GraphRuntimeMetric,
    };

    #[test]
    fn telemetry_snapshot_tracks_counts_and_volume() {
        reset_graph_runtime_telemetry();
        {
            let _timer = measure_graph_runtime(GraphRuntimeMetric::LoadProjectionKernel);
        }
        record_projection_kernel_load(3, 5, 7);
        record_region_build(11, 13, 17);

        let snapshot = snapshot_graph_runtime_telemetry();
        assert_eq!(snapshot.load_projection_kernel.count, 1);
        assert_eq!(snapshot.loaded_asserted_vertex_total, 3);
        assert_eq!(snapshot.loaded_asserted_edge_total, 5);
        assert_eq!(snapshot.loaded_candidate_edge_total, 7);
        assert_eq!(snapshot.built_region_vertex_total, 11);
        assert_eq!(snapshot.built_region_asserted_edge_total, 13);
        assert_eq!(snapshot.built_region_candidate_edge_total, 17);
    }
}
