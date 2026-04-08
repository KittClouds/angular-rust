//! Stable public entrypoints for the post-ingest causal compiler.
//!
//! This stage owns causal batch derivation, deterministic causal validation,
//! sidecar persistence, replay, and causal memory-card materialization. It
//! expects archives with persisted causal substrate plus optional ER replay and
//! writes `CausalScopeSidecar` records. Prefer these functions over reaching
//! into the worker internals from orchestration code.

use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixCausalPatchStore, PhoenixErPatchStore, StoreError,
};
use phoenix_types::SessionId;

use crate::{
    build_causal_patch_sidecar, derive_dirty_scope_review_batches, derive_scope_review_batch,
    persist_causal_patch_sidecar, run_causal_scope, CausalScopeReviewBatch,
};

pub fn derive_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<CausalScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixCausalPatchStore,
{
    derive_dirty_scope_review_batches(store, session_id)
}

pub fn derive_batch(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    session: Option<&phoenix_semantic_v2::SessionArchive>,
    dirty: Option<&phoenix_semantic_v2::DirtyScopeRecord>,
    er_sidecar: Option<&phoenix_semantic_v2::ErScopePatchSidecar>,
) -> CausalScopeReviewBatch {
    derive_scope_review_batch(archives, session, dirty, er_sidecar)
}

pub fn run_batch(batch: &mut CausalScopeReviewBatch, created_at: i64) {
    run_causal_scope(batch, created_at)
}

pub fn build_patch_sidecar(
    batch: &CausalScopeReviewBatch,
    created_at: i64,
) -> phoenix_semantic_v2::CausalScopeSidecar {
    build_causal_patch_sidecar(batch, created_at)
}

pub fn persist_patch_sidecar<S>(
    store: &S,
    batch: &CausalScopeReviewBatch,
    created_at: i64,
) -> Result<phoenix_semantic_v2::CausalScopeSidecar, StoreError>
where
    S: PhoenixCausalPatchStore,
{
    persist_causal_patch_sidecar(store, batch, created_at)
}
