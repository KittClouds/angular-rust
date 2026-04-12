use std::time::Instant;

use phoenix_semantic_v2::{
    scope_storage_key, DirtyScopeRecord, DocumentArchive, ErScopePatchSidecar,
    RelationMentionSeedScopeSidecar, RelationScopePatchSidecar, ScopeLexSidecar, SessionArchive,
};
use serde::Serialize;

use super::*;
use crate::worker::{
    build_entity_profiles, build_persisted_relations, build_review_cases, build_windows,
    continuity_relation_map, entity_profile_by_id, select_window_relation_specs,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationBenchmarkCounts {
    pub archive_count: usize,
    pub persisted_relation_count: usize,
    pub entity_profile_count: usize,
    pub window_count: usize,
    pub review_case_count: usize,
    pub total_candidate_relation_type_count: usize,
    pub total_execution_relation_spec_count: usize,
    pub max_execution_relation_spec_count: usize,
    pub total_window_text_bytes: usize,
    pub total_case_window_text_bytes: usize,
    pub total_case_serialized_bytes: usize,
    pub total_window_entity_count: usize,
    pub total_case_prediction_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationBenchmarkReport {
    pub scope_key: String,
    pub review_batch_total_us: u64,
    pub persisted_relations_us: u64,
    pub entity_profiles_us: u64,
    pub windows_us: u64,
    pub review_cases_us: u64,
    pub patch_merge_us: u64,
    pub primary_lane_us: u64,
    pub used_model: bool,
    pub counts: RelationBenchmarkCounts,
    pub window_build_stats: RelationWindowBuildStats,
}

pub fn benchmark_scope_review_pipeline(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    sidecar: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
    relation_seed_sidecar: Option<&RelationMentionSeedScopeSidecar>,
    mention_seeder: Option<&RelationMentionSeeder>,
    model: Option<&GlirelModel>,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Result<RelationBenchmarkReport, GlirelWorkerError> {
    let scope = archives
        .first()
        .map(|archive| archive.manifest.scope.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope.clone()))
        .or_else(|| sidecar.as_ref().map(|value| value.scope.clone()))
        .unwrap_or_default();
    let scope_key = archives
        .first()
        .map(|archive| archive.manifest.scope_key.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope_key.clone()))
        .or_else(|| sidecar.as_ref().map(|value| value.scope_key.clone()))
        .unwrap_or_default();
    let scope_ord = archives
        .first()
        .map(|archive| archive.manifest.scope_ord)
        .or_else(|| dirty.as_ref().map(|record| record.scope_ord))
        .or_else(|| sidecar.as_ref().and_then(|value| value.scope_ord))
        .unwrap_or_default();
    let session_id = archives
        .iter()
        .find_map(|archive| archive.manifest.session_id.clone())
        .or_else(|| session.map(|value| value.session_id.clone()));
    let document_refs = session
        .map(|value| {
            value
                .document_refs
                .iter()
                .filter(|reference| scope_storage_key(&reference.scope) == scope_key)
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let review_batch_started = Instant::now();

    let started = Instant::now();
    let persisted_relations = build_persisted_relations(archives);
    let persisted_relations_us = elapsed_us(started);

    let started = Instant::now();
    let entity_profiles = build_entity_profiles(archives, sidecar, er_sidecar, session);
    let entity_profiles_us = elapsed_us(started);

    let started = Instant::now();
    let profile_by_entity = entity_profile_by_id(&entity_profiles);
    let continuity_hints = continuity_relation_map(&persisted_relations);
    let (windows, mut window_build_stats) = build_windows(
        archives,
        er_sidecar,
        &persisted_relations,
        &entity_profiles,
        &profile_by_entity,
        &continuity_hints,
        relation_seed_sidecar,
        mention_seeder,
    )?;
    let windows_us = elapsed_us(started);

    let started = Instant::now();
    let review_cases = build_review_cases(
        &scope,
        &scope_key,
        scope_ord,
        session_id.clone(),
        &windows,
        &profile_by_entity,
    );
    let review_cases_us = elapsed_us(started);
    window_build_stats.seeded_pair_count = review_cases.len();

    let started = Instant::now();
    let mut batch = RelationScopeReviewBatch {
        scope,
        scope_key: scope_key.clone(),
        scope_ord,
        session_id,
        dirty: dirty.cloned(),
        document_refs,
        windows,
        review_cases,
        entity_profiles,
        persisted_relations,
        lexical_generation: sidecar.map(|value| value.generation),
        er_generation: er_sidecar.map(|value| value.generation),
        relation_generation: relation_sidecar.map(|value| value.generation),
        window_build_stats: window_build_stats.clone(),
    };
    if let Some(sidecar) = relation_sidecar {
        apply_relation_patch_sidecar(&mut batch, sidecar);
    }
    let patch_merge_us = elapsed_us(started);

    let started = Instant::now();
    run_primary_relation_lane(&mut batch, model, relation_specs)?;
    let primary_lane_us = elapsed_us(started);

    let total_window_text_bytes = batch.windows.iter().map(|window| window.text.len()).sum();
    let total_candidate_relation_type_count = batch
        .windows
        .iter()
        .map(|window| window.candidate_relation_types.len())
        .sum();
    let execution_relation_spec_counts = batch
        .windows
        .iter()
        .map(|window| select_window_relation_specs(window, relation_specs).len())
        .collect::<Vec<_>>();
    let total_execution_relation_spec_count = execution_relation_spec_counts
        .iter()
        .copied()
        .sum::<usize>();
    let max_execution_relation_spec_count = execution_relation_spec_counts
        .iter()
        .copied()
        .max()
        .unwrap_or_default();
    let total_case_window_text_bytes = batch
        .review_cases
        .iter()
        .map(|case| case.window_text.len())
        .sum();
    let total_case_serialized_bytes = batch
        .review_cases
        .iter()
        .map(|case| case.serialized.len())
        .sum();
    let total_window_entity_count = batch
        .windows
        .iter()
        .map(|window| window.entities.len())
        .sum();
    let total_case_prediction_count = batch
        .review_cases
        .iter()
        .map(|case| case.glirel_predictions.len())
        .sum();

    Ok(RelationBenchmarkReport {
        scope_key,
        review_batch_total_us: elapsed_us(review_batch_started),
        persisted_relations_us,
        entity_profiles_us,
        windows_us,
        review_cases_us,
        patch_merge_us,
        primary_lane_us,
        used_model: model.is_some(),
        counts: RelationBenchmarkCounts {
            archive_count: archives.len(),
            persisted_relation_count: batch.persisted_relations.len(),
            entity_profile_count: batch.entity_profiles.len(),
            window_count: batch.windows.len(),
            review_case_count: batch.review_cases.len(),
            total_candidate_relation_type_count,
            total_execution_relation_spec_count,
            max_execution_relation_spec_count,
            total_window_text_bytes,
            total_case_window_text_bytes,
            total_case_serialized_bytes,
            total_window_entity_count,
            total_case_prediction_count,
        },
        window_build_stats,
    })
}

fn elapsed_us(started: Instant) -> u64 {
    started.elapsed().as_micros() as u64
}
