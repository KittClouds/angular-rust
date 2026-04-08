//! Stable public entrypoints for the temporal memory compiler.
//!
//! This stage owns memory-batch derivation and compilation of claims, states,
//! deltas, conflicts, gaps, and retrieval cards. It expects archive/store
//! inputs plus persisted ER/RE sidecars and writes `MemoryScopeSidecar`
//! records. Prefer these functions when driving memory compilation from
//! orchestration code.

use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixMemoryPatchStore, PhoenixRelationPatchStore,
    StoreError,
};
use phoenix_types::SessionId;

use crate::{
    build_memory_patch_sidecar, compile_memory, derive_dirty_scope_review_batches,
    derive_scope_review_batch, normalize_memory_inputs, persist_memory_patch_sidecar,
    CompiledMemory, MemoryScopeReviewBatch,
};

pub fn derive_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<MemoryScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixRelationPatchStore + PhoenixMemoryPatchStore,
{
    derive_dirty_scope_review_batches(store, session_id)
}

pub fn derive_batch(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    session: Option<&phoenix_semantic_v2::SessionArchive>,
    dirty: Option<&phoenix_semantic_v2::DirtyScopeRecord>,
    lexical: Option<&phoenix_semantic_v2::ScopeLexSidecar>,
    er_sidecar: Option<&phoenix_semantic_v2::ErScopePatchSidecar>,
    relation_sidecar: Option<&phoenix_semantic_v2::RelationScopePatchSidecar>,
) -> MemoryScopeReviewBatch {
    derive_scope_review_batch(archives, session, dirty, lexical, er_sidecar, relation_sidecar)
}

pub fn compile_from_inputs(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    session: Option<&phoenix_semantic_v2::SessionArchive>,
    lexical: Option<&phoenix_semantic_v2::ScopeLexSidecar>,
    er_sidecar: Option<&phoenix_semantic_v2::ErScopePatchSidecar>,
    relation_sidecar: Option<&phoenix_semantic_v2::RelationScopePatchSidecar>,
) -> CompiledMemory {
    let normalized = normalize_memory_inputs(archives, session, lexical, er_sidecar, relation_sidecar);
    compile_memory(&normalized)
}

pub fn build_patch_sidecar(
    batch: &MemoryScopeReviewBatch,
    created_at: i64,
) -> phoenix_semantic_v2::MemoryScopeSidecar {
    build_memory_patch_sidecar(batch, created_at)
}

pub fn persist_patch_sidecar<S>(
    store: &S,
    batch: &MemoryScopeReviewBatch,
    created_at: i64,
) -> Result<phoenix_semantic_v2::MemoryScopeSidecar, StoreError>
where
    S: PhoenixMemoryPatchStore,
{
    persist_memory_patch_sidecar(store, batch, created_at)
}
