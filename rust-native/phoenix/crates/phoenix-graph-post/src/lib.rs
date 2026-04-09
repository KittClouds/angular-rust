pub mod api;
mod retrieval;
mod retrieval_causal;
mod retrieval_common;
mod retrieval_history;
mod retrieval_world;
pub mod semantic;
pub mod semantic_graph;

mod compile;
#[cfg(test)]
mod retrieval_tests;
mod semantic_graph_nli;
mod semantic_graph_support;
#[cfg(test)]
mod semantic_graph_tests;
pub mod worker;

pub use compile::{compile_graph_projection, CompiledGraphProjection};
pub use semantic_graph_nli::SemanticNliConfig;
pub use worker::{
    apply_graph_patch_sidecar, build_graph_patch_sidecar, derive_dirty_scope_review_batches,
    derive_scope_review_batch, derive_scope_review_batch_from_store, persist_graph_patch_sidecar,
    GraphScopeReviewBatch,
};

#[cfg(test)]
mod tests;
