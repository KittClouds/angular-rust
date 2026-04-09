pub mod api;

mod graph;
mod normalize;
mod resolve;
#[cfg(test)]
mod tests;
mod views;
pub mod worker;

pub use graph::{build_identity_hypotheses, EventIdentityGraphStats};
pub use normalize::{normalize_event_identity_inputs, EventIdentityNormalizedInputs};
pub use resolve::resolve_canonical_events;
pub use views::build_canonical_event_cards;
pub use worker::{
    apply_event_identity_patch_sidecar, build_event_identity_patch_sidecar,
    derive_dirty_scope_review_batches, derive_scope_review_batch,
    derive_scope_review_batch_from_store, persist_event_identity_patch_sidecar,
    run_event_identity_scope, EventIdentityScopeReviewBatch,
};
