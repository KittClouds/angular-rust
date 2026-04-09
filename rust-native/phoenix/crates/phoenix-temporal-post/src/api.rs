//! Stable public entrypoints for the post-ingest temporal compiler.
//!
//! This stage owns temporal batch derivation, anchor-first normalization,
//! constraint solving, sidecar persistence, replay, and temporal memory-card
//! materialization. It expects archives with persisted temporal substrate and
//! writes `TemporalScopeSidecar` records.

use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixEventIdentityPatchStore, PhoenixTemporalPatchStore, StoreError,
};
use phoenix_types::SessionId;

use crate::{
    build_temporal_patch_sidecar, derive_dirty_scope_review_batches, derive_scope_review_batch,
    persist_temporal_patch_sidecar, run_temporal_scope, TemporalScopeReviewBatch,
};

pub fn derive_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<TemporalScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixTemporalPatchStore + PhoenixEventIdentityPatchStore,
{
    derive_dirty_scope_review_batches(store, session_id)
}

pub fn derive_batch(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    session: Option<&phoenix_semantic_v2::SessionArchive>,
    dirty: Option<&phoenix_semantic_v2::DirtyScopeRecord>,
) -> TemporalScopeReviewBatch {
    derive_scope_review_batch(archives, session, dirty)
}

pub fn run_batch(batch: &mut TemporalScopeReviewBatch, created_at: i64) {
    run_temporal_scope(batch, created_at);
}

pub fn build_patch_sidecar(
    batch: &TemporalScopeReviewBatch,
    created_at: i64,
) -> phoenix_semantic_v2::TemporalScopeSidecar {
    build_temporal_patch_sidecar(batch, created_at)
}

pub fn persist_patch_sidecar<S>(
    store: &S,
    batch: &TemporalScopeReviewBatch,
    created_at: i64,
) -> Result<phoenix_semantic_v2::TemporalScopeSidecar, StoreError>
where
    S: PhoenixTemporalPatchStore,
{
    persist_temporal_patch_sidecar(store, batch, created_at)
}
