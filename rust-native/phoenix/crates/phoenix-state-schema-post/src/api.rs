//! Stable public entrypoints for the governed slot-schema compiler.
//!
//! This stage owns slot-family mining, governance, promotion, and write
//! proposal drafting. It expects archive inputs plus relation sidecars and
//! writes `StateSchemaScopeSidecar` records. Prefer these functions when
//! driving schema growth ahead of memory compilation.

use phoenix_scope_analysis::ScopeAnalysisContext;
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixRelationPatchStore, PhoenixStateSchemaPatchStore, StoreError,
};
use phoenix_types::SessionId;

use crate::{
    build_state_schema_patch_sidecar, derive_dirty_scope_review_batches, derive_scope_review_batch,
    normalize_state_schema_inputs, persist_state_schema_patch_sidecar, run_state_schema_scope,
    StateSchemaNormalizedInputs, StateSchemaScopeReviewBatch,
};

pub fn derive_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<StateSchemaScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixRelationPatchStore + PhoenixStateSchemaPatchStore,
{
    derive_dirty_scope_review_batches(store, session_id)
}

pub fn derive_batch(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    session: Option<&phoenix_semantic_v2::SessionArchive>,
    dirty: Option<&phoenix_semantic_v2::DirtyScopeRecord>,
    relation_sidecar: Option<&phoenix_semantic_v2::RelationScopePatchSidecar>,
) -> StateSchemaScopeReviewBatch {
    derive_scope_review_batch(archives, session, dirty, relation_sidecar)
}

pub fn normalize_inputs(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    relation_sidecar: Option<&phoenix_semantic_v2::RelationScopePatchSidecar>,
) -> StateSchemaNormalizedInputs {
    normalize_state_schema_inputs(archives, relation_sidecar)
}

pub fn derive_batch_from_analysis(
    analysis: &ScopeAnalysisContext,
    relation_sidecar: Option<&phoenix_semantic_v2::RelationScopePatchSidecar>,
) -> StateSchemaScopeReviewBatch {
    crate::derive_scope_review_batch_from_analysis(analysis, relation_sidecar)
}

pub fn run_batch(batch: &mut StateSchemaScopeReviewBatch, created_at: i64) {
    run_state_schema_scope(batch, created_at)
}

pub fn build_patch_sidecar(
    batch: &StateSchemaScopeReviewBatch,
    created_at: i64,
) -> phoenix_semantic_v2::StateSchemaScopeSidecar {
    build_state_schema_patch_sidecar(batch, created_at)
}

pub fn persist_patch_sidecar<S>(
    store: &S,
    batch: &StateSchemaScopeReviewBatch,
    created_at: i64,
) -> Result<phoenix_semantic_v2::StateSchemaScopeSidecar, StoreError>
where
    S: PhoenixStateSchemaPatchStore,
{
    persist_state_schema_patch_sidecar(store, batch, created_at)
}
