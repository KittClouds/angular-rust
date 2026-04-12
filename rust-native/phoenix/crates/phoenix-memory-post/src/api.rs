//! Stable public entrypoints for the temporal memory compiler.
//!
//! This stage owns memory-batch derivation and compilation of claims, states,
//! deltas, conflicts, gaps, and retrieval cards. It expects archive/store
//! inputs plus persisted ER/RE sidecars and writes `MemoryScopeSidecar`
//! records. Prefer these functions when driving memory compilation from
//! orchestration code.

use phoenix_scope_analysis::ScopeAnalysisContext;
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixEventIdentityPatchStore,
    PhoenixMemoryPatchStore, PhoenixRelationPatchStore, PhoenixStateSchemaPatchStore, StoreError,
};
use phoenix_types::SessionId;

use crate::{
    apply_memory_patch_sidecar, build_memory_patch_sidecar, compile_memory,
    derive_dirty_scope_review_batches, derive_scope_review_batch, normalize_memory_inputs,
    persist_memory_patch_sidecar, CompiledMemory, MemoryScopeReviewBatch,
};

pub fn derive_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<MemoryScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2
        + PhoenixErPatchStore
        + PhoenixRelationPatchStore
        + PhoenixMemoryPatchStore
        + PhoenixEventIdentityPatchStore
        + PhoenixStateSchemaPatchStore,
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
    state_schema_sidecar: Option<&phoenix_semantic_v2::StateSchemaScopeSidecar>,
) -> MemoryScopeReviewBatch {
    derive_scope_review_batch(
        archives,
        session,
        dirty,
        lexical,
        er_sidecar,
        relation_sidecar,
        state_schema_sidecar,
    )
}

pub fn derive_batch_with_runtime_sidecars(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    session: Option<&phoenix_semantic_v2::SessionArchive>,
    dirty: Option<&phoenix_semantic_v2::DirtyScopeRecord>,
    lexical: Option<&phoenix_semantic_v2::ScopeLexSidecar>,
    er_sidecar: Option<&phoenix_semantic_v2::ErScopePatchSidecar>,
    relation_sidecar: Option<&phoenix_semantic_v2::RelationScopePatchSidecar>,
    state_schema_sidecar: Option<&phoenix_semantic_v2::StateSchemaScopeSidecar>,
    event_identity_sidecar: Option<&phoenix_semantic_v2::EventIdentityScopeSidecar>,
    memory_sidecar: Option<&phoenix_semantic_v2::MemoryScopeSidecar>,
) -> MemoryScopeReviewBatch {
    let mut batch = derive_scope_review_batch(
        archives,
        session,
        dirty,
        lexical,
        er_sidecar,
        relation_sidecar,
        state_schema_sidecar,
    );
    if let Some(sidecar) = event_identity_sidecar {
        crate::worker::annotate_memory_batch_with_event_identity(&mut batch, sidecar);
    }
    if let Some(sidecar) = memory_sidecar {
        batch.memory_generation = Some(sidecar.generation);
        if batch.claims.is_empty() && batch.states.is_empty() && batch.events.is_empty() {
            apply_memory_patch_sidecar(&mut batch, sidecar);
            if let Some(event_identity_sidecar) = event_identity_sidecar {
                crate::worker::annotate_memory_batch_with_event_identity(
                    &mut batch,
                    event_identity_sidecar,
                );
            }
        }
    }
    batch
}

pub fn derive_batch_from_analysis(
    analysis: &ScopeAnalysisContext,
    relation_sidecar: Option<&phoenix_semantic_v2::RelationScopePatchSidecar>,
    state_schema_sidecar: Option<&phoenix_semantic_v2::StateSchemaScopeSidecar>,
    event_identity_sidecar: Option<&phoenix_semantic_v2::EventIdentityScopeSidecar>,
    memory_sidecar: Option<&phoenix_semantic_v2::MemoryScopeSidecar>,
) -> MemoryScopeReviewBatch {
    let mut batch = crate::derive_scope_review_batch_from_analysis(
        analysis,
        relation_sidecar,
        state_schema_sidecar,
    );
    if let Some(sidecar) = event_identity_sidecar {
        crate::worker::annotate_memory_batch_with_event_identity(&mut batch, sidecar);
    }
    if let Some(sidecar) = memory_sidecar {
        batch.memory_generation = Some(sidecar.generation);
        if batch.claims.is_empty() && batch.states.is_empty() && batch.events.is_empty() {
            apply_memory_patch_sidecar(&mut batch, sidecar);
            if let Some(event_identity_sidecar) = event_identity_sidecar {
                crate::worker::annotate_memory_batch_with_event_identity(
                    &mut batch,
                    event_identity_sidecar,
                );
            }
        }
    }
    batch
}

pub fn compile_from_inputs(
    archives: &[phoenix_semantic_v2::DocumentArchive],
    session: Option<&phoenix_semantic_v2::SessionArchive>,
    lexical: Option<&phoenix_semantic_v2::ScopeLexSidecar>,
    er_sidecar: Option<&phoenix_semantic_v2::ErScopePatchSidecar>,
    relation_sidecar: Option<&phoenix_semantic_v2::RelationScopePatchSidecar>,
    state_schema_sidecar: Option<&phoenix_semantic_v2::StateSchemaScopeSidecar>,
) -> CompiledMemory {
    let normalized = normalize_memory_inputs(
        archives,
        session,
        lexical,
        er_sidecar,
        relation_sidecar,
        state_schema_sidecar,
    );
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
