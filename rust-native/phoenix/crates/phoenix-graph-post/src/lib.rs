pub mod api;
pub mod diffusion_eval;
mod diffusion_metrics;
pub mod eval;
mod phase4_contract;
mod phase4_event_scoring;
mod phase4_graph_scoring;
mod phase4_scoring;
mod phase4_scoring_support;
mod phase4_scoring_text;
mod phase5_path_rerank;
mod query_session;
mod retrieval;
mod retrieval_causal;
mod retrieval_common;
mod retrieval_history;
mod retrieval_world;
mod runtime_telemetry;
pub mod semantic;
pub mod semantic_graph;
pub mod smoke_support;

mod compile;
#[cfg(test)]
mod eval_tests;
#[cfg(test)]
mod phase4_graph_tests;
#[cfg(test)]
mod phase4_tests;
#[cfg(test)]
mod phase5_path_rerank_tests;
#[cfg(test)]
mod retrieval_soft_tests;
#[cfg(test)]
mod retrieval_tests;
mod semantic_graph_causal_gap;
mod semantic_graph_contradiction;
mod semantic_graph_contradiction_ledger;
mod semantic_graph_event;
mod semantic_graph_nli;
mod semantic_graph_process;
mod semantic_graph_soft;
mod semantic_graph_support;
#[cfg(test)]
mod semantic_graph_tests;
mod semantic_graph_workspace;
pub mod worker;

pub use compile::{compile_graph_projection, CompiledGraphProjection};
pub use phase4_contract::{
    GraphPathRerankScore, GraphPhase4RerankScore, GraphStructuralRerankScore,
};
pub use runtime_telemetry::{
    reset_graph_runtime_telemetry, snapshot_graph_runtime_telemetry, GraphRuntimeTelemetrySnapshot,
};
pub use semantic_graph_nli::SemanticNliConfig;
pub use worker::{
    apply_graph_patch_sidecar, build_graph_patch_sidecar, derive_dirty_scope_review_batches,
    derive_scope_review_batch, derive_scope_review_batch_from_store, persist_graph_patch_sidecar,
    GraphScopeReviewBatch,
};

pub fn clear_graph_thread_local_caches() {
    retrieval_common::clear_query_embedder_cache();
    phase4_scoring_support::clear_phase4_scorer_cache();
}

#[cfg(test)]
mod tests;
