pub mod api;

mod glirel;
mod gliner_seed;
mod nli;
mod seed_worker;
#[cfg(test)]
mod tests;
mod worker;

pub use glirel::{
    extract_heuristic_relations, finalize_relation_predictions, repair_relation_directions,
    seed_relation_pairs, split_sentence_windows, suppress_relation_conflicts, GlirelEntity,
    GlirelError, GlirelModel, GlirelPairSeed, GlirelProposalConfig, GlirelRelationPrediction,
    GlirelRelationTypeSpec, GlirelSentenceWindow,
};
pub use gliner_seed::{RelationMentionSeeder, RelationSeededSpan};
pub use nli::{NliError, NliModel, NliPairJudgment, NliScores};
pub use seed_worker::{
    build_relation_mention_seed_sidecar, build_relation_mention_seed_sidecar_from_store,
    persist_relation_mention_seed_sidecar, RelationSeedConfig, RelationSeedReport,
};
pub use worker::{
    adjudicate_relation_decisions_with_nli, apply_relation_patch_sidecar,
    build_relation_hypotheses, build_relation_patch_sidecar, default_relation_type_specs,
    derive_dirty_scope_review_batches, derive_dirty_scope_review_batches_with_seeder,
    derive_relation_entity_profiles,
    derive_scope_review_batch, derive_scope_review_batch_from_store,
    derive_scope_review_batch_from_store_with_seeder, derive_scope_review_batch_with_seeder,
    draft_relation_decisions, persist_relation_patch_sidecar, run_glirel_over_batch,
    run_primary_relation_lane, GlirelWorkerError, RelationDecision, RelationDecisionKind,
    RelationEntityProfile, RelationReviewCase, RelationScopeReviewBatch, RelationWindowEntity,
    RelationWindowRecord,
};
