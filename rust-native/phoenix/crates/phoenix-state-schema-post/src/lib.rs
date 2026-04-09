pub mod api;

mod mine;
mod normalize;
mod promote;
#[cfg(test)]
mod tests;
pub mod worker;

pub use mine::mine_slot_candidates;
pub use normalize::{
    normalize_state_schema_inputs, StateSchemaEvidenceRow, StateSchemaNormalizedInputs,
};
pub use promote::{promote_slot_definitions, PromotionOutput};
pub use worker::{
    apply_state_schema_patch_sidecar, build_state_schema_patch_sidecar,
    derive_dirty_scope_review_batches, derive_scope_review_batch,
    derive_scope_review_batch_from_store, persist_state_schema_patch_sidecar,
    run_state_schema_scope, StateSchemaScopeReviewBatch,
};
