pub mod api;

mod graph;
mod normalize;
#[cfg(test)]
mod tests;
mod validate;
mod views;
pub mod worker;

pub use graph::{build_chain_records, build_counterfactual_reviews, CausalGraphStats};
pub use normalize::{
    normalize_causal_inputs, normalize_causal_inputs_with_sidecars, CausalEventProfile,
    CausalNormalizedInputs, CausalReviewCase, CausalSourceClaimTrace,
    CausalSourceClaimTraceSummary,
};
pub use validate::{draft_causal_decisions, CausalDecision, CausalDecisionKind};
pub use views::build_causal_memory_cards;
pub use worker::{
    apply_causal_patch_sidecar, build_causal_patch_sidecar, derive_dirty_scope_review_batches,
    derive_scope_review_batch, derive_scope_review_batch_from_store, persist_causal_patch_sidecar,
    run_causal_scope, CausalScopeReviewBatch,
};
