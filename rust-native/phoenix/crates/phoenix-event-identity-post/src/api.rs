//! Stable public entrypoints for the post-ingest event identity compiler.
//!
//! This stage owns scope-wide event mention packet normalization, hypothesis
//! graph construction, canonical event resolution, sidecar persistence, and
//! canonical event card materialization. It expects archives with persisted
//! `DocumentEventIdentitySubstrate` and optional ER replay sidecars, and it
//! writes `EventIdentityScopeSidecar` records.

use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixEventIdentityPatchStore, StoreError,
};
use phoenix_types::SessionId;

use crate::{
    build_event_identity_patch_sidecar, derive_dirty_scope_review_batches,
    derive_scope_review_batch, persist_event_identity_patch_sidecar, run_event_identity_scope,
    EventIdentityScopeReviewBatch,
};

pub fn derive_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<EventIdentityScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixEventIdentityPatchStore,
{
    derive_dirty_scope_review_batches(store, session_id)
}

pub fn derive_batch(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    session: Option<&phoenix_semantic_v2::SessionArchive>,
    dirty: Option<&phoenix_semantic_v2::DirtyScopeRecord>,
    er_sidecar: Option<&phoenix_semantic_v2::ErScopePatchSidecar>,
) -> EventIdentityScopeReviewBatch {
    derive_scope_review_batch(archives, session, dirty, er_sidecar)
}

pub fn run_batch(batch: &mut EventIdentityScopeReviewBatch, created_at: i64) {
    run_event_identity_scope(batch, created_at);
}

pub fn build_patch_sidecar(
    batch: &EventIdentityScopeReviewBatch,
    created_at: i64,
) -> phoenix_semantic_v2::EventIdentityScopeSidecar {
    build_event_identity_patch_sidecar(batch, created_at)
}

pub fn persist_patch_sidecar<S>(
    store: &S,
    batch: &EventIdentityScopeReviewBatch,
    created_at: i64,
) -> Result<phoenix_semantic_v2::EventIdentityScopeSidecar, StoreError>
where
    S: PhoenixEventIdentityPatchStore,
{
    persist_event_identity_patch_sidecar(store, batch, created_at)
}
