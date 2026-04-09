//! Stable public entrypoints for the post-ingest entity-resolution worker.
//!
//! This stage owns ER review-batch derivation, candidate generation, decision
//! drafting, and ER sidecar persistence. It expects archive/store inputs and
//! emits `ErScopeReviewBatch` plus optional persisted `ErScopePatchSidecar`
//! records. Prefer these functions when invoking ER from orchestration code.

use phoenix_store_native_core::{PhoenixArchiveStoreV2, PhoenixErPatchStore, StoreError};
use phoenix_types::SessionId;

use crate::{
    build_er_patch_sidecar, derive_dirty_scope_review_batches,
    derive_dirty_scope_review_batches_with_replay, draft_review_decisions,
    generate_embedding_candidates, generate_fused_candidates, generate_lexical_candidates,
    persist_er_patch_sidecar, ErDecision, ErEmbeddingCandidateSummary, ErEmbeddingConfig,
    ErEmbeddingModel, ErFusedCandidateSummary, ErLexicalCandidateSummary, ErScopeReviewBatch,
};

pub fn derive_batches<S: PhoenixArchiveStoreV2>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<ErScopeReviewBatch>, StoreError> {
    derive_dirty_scope_review_batches(store, session_id)
}

pub fn derive_batches_with_replay<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<ErScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore,
{
    derive_dirty_scope_review_batches_with_replay(store, session_id)
}

pub fn run_lexical_retrieval(
    batch: &mut ErScopeReviewBatch,
    limit: usize,
) -> ErLexicalCandidateSummary {
    generate_lexical_candidates(batch, limit)
}

pub fn run_embedding_retrieval(
    batch: &mut ErScopeReviewBatch,
    model: &ErEmbeddingModel,
    limit: usize,
    config: &ErEmbeddingConfig,
) -> Result<ErEmbeddingCandidateSummary, String> {
    generate_embedding_candidates(batch, model, limit, config)
}

pub fn run_fused_ranking(batch: &mut ErScopeReviewBatch, limit: usize) -> ErFusedCandidateSummary {
    generate_fused_candidates(batch, limit)
}

pub fn draft_decisions(batch: &ErScopeReviewBatch) -> Vec<ErDecision> {
    draft_review_decisions(batch)
}

pub fn build_patch_sidecar(
    batch: &ErScopeReviewBatch,
    decisions: &[ErDecision],
    created_at: i64,
) -> phoenix_semantic_v2::ErScopePatchSidecar {
    build_er_patch_sidecar(batch, decisions, created_at)
}

pub fn persist_patch_sidecar<S>(
    store: &S,
    batch: &ErScopeReviewBatch,
    decisions: &[ErDecision],
    created_at: i64,
) -> Result<phoenix_semantic_v2::ErScopePatchSidecar, StoreError>
where
    S: PhoenixErPatchStore,
{
    persist_er_patch_sidecar(store, batch, decisions, created_at)
}
