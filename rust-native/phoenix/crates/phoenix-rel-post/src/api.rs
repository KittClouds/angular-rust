//! Stable public entrypoints for the post-ingest relation worker.
//!
//! This stage owns Alex-first relation batch derivation, GLiREL execution,
//! draft decision generation, optional NLI adjudication, relation sidecar
//! persistence, and explicit fallback mention seeding. It expects persisted
//! archives plus ER/lexical sidecars and emits `RelationScopeReviewBatch`
//! and `RelationScopePatchSidecar` records.

use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixRelationMentionSeedStore,
    PhoenixRelationPatchStore, StoreError,
};
use phoenix_types::SessionId;

use crate::{
    adjudicate_relation_decisions_with_nli, build_relation_hypotheses,
    build_relation_patch_sidecar, default_relation_type_specs, derive_dirty_scope_review_batches,
    derive_dirty_scope_review_batches_with_seeder, draft_relation_decisions,
    persist_relation_patch_sidecar, run_glirel_over_batch, GlirelModel, GlirelRelationTypeSpec,
    GlirelWorkerError, NliModel, RelationDecision, RelationMentionSeeder, RelationScopeReviewBatch,
};

/// Canonical Alex-first relation batch derivation. This path does not invoke
/// fallback mention seeding.
pub fn derive_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<RelationScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixRelationPatchStore,
{
    derive_dirty_scope_review_batches(store, session_id)
}

/// Explicit fallback derivation that may consume persisted or live mention
/// seeds. This is auxiliary and not the preferred path.
pub fn derive_batches_with_seed_fallback<S>(
    store: &S,
    session_id: Option<&SessionId>,
    mention_seeder: Option<&RelationMentionSeeder>,
) -> Result<Vec<RelationScopeReviewBatch>, GlirelWorkerError>
where
    S: PhoenixArchiveStoreV2
        + PhoenixErPatchStore
        + PhoenixRelationMentionSeedStore
        + PhoenixRelationPatchStore,
{
    derive_dirty_scope_review_batches_with_seeder(store, session_id, mention_seeder)
}

pub fn relation_specs() -> Vec<GlirelRelationTypeSpec> {
    default_relation_type_specs()
}

pub fn run_glirel(
    batch: &mut RelationScopeReviewBatch,
    model: &GlirelModel,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Result<(), GlirelWorkerError> {
    run_glirel_over_batch(batch, model, relation_specs)
}

pub fn draft_decisions(
    batch: &RelationScopeReviewBatch,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Vec<RelationDecision> {
    draft_relation_decisions(batch, relation_specs)
}

pub fn build_hypotheses(edge_type: &str, head: &str, tail: &str) -> Vec<String> {
    build_relation_hypotheses(edge_type, head, tail)
}

pub fn adjudicate_with_nli(
    batch: &RelationScopeReviewBatch,
    decisions: &[RelationDecision],
    relation_specs: &[GlirelRelationTypeSpec],
    nli: &NliModel,
) -> Result<Vec<RelationDecision>, GlirelWorkerError> {
    adjudicate_relation_decisions_with_nli(batch, decisions, relation_specs, nli)
}

pub fn build_patch_sidecar(
    batch: &RelationScopeReviewBatch,
    decisions: &[RelationDecision],
    created_at: i64,
) -> phoenix_semantic_v2::RelationScopePatchSidecar {
    build_relation_patch_sidecar(batch, decisions, created_at)
}

pub fn persist_patch_sidecar<S>(
    store: &S,
    batch: &RelationScopeReviewBatch,
    decisions: &[RelationDecision],
    created_at: i64,
) -> Result<phoenix_semantic_v2::RelationScopePatchSidecar, StoreError>
where
    S: PhoenixRelationPatchStore,
{
    persist_relation_patch_sidecar(store, batch, decisions, created_at)
}
