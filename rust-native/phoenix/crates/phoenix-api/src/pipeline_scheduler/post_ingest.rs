use phoenix_memory_post::api as memory_api;
use phoenix_rel_post::api as rel_api;
use phoenix_semantic_v2::{MemoryScopeSidecar, RelationScopePatchSidecar, StateSchemaScopeSidecar};
use phoenix_state_schema_post::api as state_schema_api;
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixEventIdentityPatchStore,
    PhoenixMemoryPatchStore, PhoenixRelationPatchStore, PhoenixScopeRuntimeStore,
    PhoenixStateSchemaPatchStore, ScopeImageSpec,
};

use crate::{LateSidecarRunReport, PipelineApiError, PostIngestRunReport, StateSchemaRunReport};

use super::context::PipelineGenerationContext;
use super::types::{PipelineRunRequest, PipelineStage, ScopeGenerationKey, StageProductEnvelope};

pub fn run_post_ingest_pipeline<S>(
    store: &S,
    request: PipelineRunRequest,
    glirel_model: &phoenix_rel_post::GlirelModel,
    relation_specs: &[phoenix_rel_post::GlirelRelationTypeSpec],
    relation_created_at: i64,
    memory_created_at: i64,
) -> Result<PostIngestRunReport, PipelineApiError>
where
    S: PhoenixArchiveStoreV2
        + PhoenixErPatchStore
        + PhoenixEventIdentityPatchStore
        + PhoenixRelationPatchStore
        + PhoenixMemoryPatchStore
        + PhoenixScopeRuntimeStore
        + PhoenixStateSchemaPatchStore,
{
    let mut context = PipelineGenerationContext::new(store, request)?;
    let mut report = PostIngestRunReport::default();
    let scope_keys = context.scope_keys().to_vec();

    for scope in scope_keys {
        while let Some(stage) = context.next_ready_stage_for_scope(&scope) {
            context.mark_stage_running(&scope, stage);
            match stage {
                PipelineStage::Relation => {
                    let relation_sidecar = run_relation_stage(
                        store,
                        &mut context,
                        &scope,
                        glirel_model,
                        relation_specs,
                        relation_created_at,
                    )?;
                    report.relation_scope_count += 1;
                    report.relation_case_count += relation_sidecar.1;
                    report.persisted_relation_edge_count +=
                        relation_sidecar.0.payload.edge_additions.len();
                    context.remember_relation_product(relation_sidecar.0);
                }
                PipelineStage::StateSchema => {
                    let analysis = context.analysis_for(&scope, ScopeImageSpec::post_ingest())?;
                    let relation_sidecar = context.relation_product(&scope);
                    let state_schema_sidecar = run_state_schema_stage(
                        store,
                        &scope,
                        &analysis,
                        relation_sidecar
                            .as_ref()
                            .map(|product| &product.payload)
                            .or(analysis.runtime.sidecars.relation.as_ref()),
                        relation_created_at,
                    )?;
                    report.state_schema_scope_count += 1;
                    report.state_schema_slot_family_count +=
                        state_schema_sidecar.payload.slot_families.len();
                    report.state_schema_slot_definition_count +=
                        state_schema_sidecar.payload.slot_definitions.len();
                    report.state_schema_active_definition_count += state_schema_sidecar
                        .payload
                        .slot_definitions
                        .iter()
                        .filter(|definition| {
                            matches!(
                                definition.lifecycle,
                                phoenix_semantic_v2::StateSlotLifecycle::Active
                                    | phoenix_semantic_v2::StateSlotLifecycle::Stable
                            )
                        })
                        .count();
                    report.state_schema_candidate_count +=
                        state_schema_sidecar.payload.slot_candidates.len();
                    report.state_schema_write_proposal_count +=
                        state_schema_sidecar.payload.write_proposals.len();
                    context.remember_state_schema_product(state_schema_sidecar);
                }
                PipelineStage::Memory => {
                    let analysis = context.analysis_for(&scope, ScopeImageSpec::post_ingest())?;
                    let relation_sidecar = context.relation_product(&scope);
                    let state_schema_sidecar = context.state_schema_product(&scope);
                    let memory_sidecar = run_memory_stage(
                        store,
                        &scope,
                        &analysis,
                        relation_sidecar
                            .as_ref()
                            .map(|product| &product.payload)
                            .or(analysis.runtime.sidecars.relation.as_ref()),
                        state_schema_sidecar
                            .as_ref()
                            .map(|product| &product.payload)
                            .or(analysis.runtime.sidecars.state_schema.as_ref()),
                        analysis.runtime.sidecars.event_identity.as_ref(),
                        analysis.runtime.sidecars.memory.as_ref(),
                        memory_created_at,
                    )?;
                    report.memory_scope_count += 1;
                    report.memory_state_count += memory_sidecar.payload.states.len();
                    report.memory_card_count += memory_sidecar.payload.entity_cards.len();
                    context.remember_memory_product(memory_sidecar);
                }
            }
            context.mark_stage_complete(&scope, stage);
        }
    }

    report.scheduler = context.metrics().clone();
    Ok(report)
}

pub fn run_late_sidecar_pipeline<S>(
    store: &S,
    request: PipelineRunRequest,
    created_at: i64,
) -> Result<LateSidecarRunReport, PipelineApiError>
where
    S: PhoenixArchiveStoreV2
        + PhoenixErPatchStore
        + PhoenixEventIdentityPatchStore
        + PhoenixRelationPatchStore
        + PhoenixMemoryPatchStore
        + PhoenixScopeRuntimeStore
        + PhoenixStateSchemaPatchStore,
{
    let mut context = PipelineGenerationContext::new(store, request)?;
    let mut report = LateSidecarRunReport::default();
    let scope_keys = context.scope_keys().to_vec();

    for scope in scope_keys {
        while let Some(stage) = context.next_ready_stage_for_scope(&scope) {
            context.mark_stage_running(&scope, stage);
            match stage {
                PipelineStage::Relation => {
                    unreachable!("late sidecars should not schedule relation")
                }
                PipelineStage::StateSchema => {
                    let analysis = context.analysis_for(&scope, ScopeImageSpec::late_sidecars())?;
                    let state_schema_sidecar = run_state_schema_stage(
                        store,
                        &scope,
                        &analysis,
                        analysis.runtime.sidecars.relation.as_ref(),
                        created_at,
                    )?;
                    accumulate_state_schema_report(
                        &mut report.state_schema,
                        &state_schema_sidecar.payload,
                    );
                    context.remember_state_schema_product(state_schema_sidecar);
                }
                PipelineStage::Memory => {
                    let analysis = context.analysis_for(&scope, ScopeImageSpec::late_sidecars())?;
                    let state_schema_sidecar = context.state_schema_product(&scope);
                    let memory_sidecar = run_memory_stage(
                        store,
                        &scope,
                        &analysis,
                        analysis.runtime.sidecars.relation.as_ref(),
                        state_schema_sidecar
                            .as_ref()
                            .map(|product| &product.payload)
                            .or(analysis.runtime.sidecars.state_schema.as_ref()),
                        analysis.runtime.sidecars.event_identity.as_ref(),
                        analysis.runtime.sidecars.memory.as_ref(),
                        created_at,
                    )?;
                    report.memory_scope_count += 1;
                    report.memory_state_count += memory_sidecar.payload.states.len();
                    report.memory_event_count += memory_sidecar.payload.events.len();
                    report.memory_claim_count += memory_sidecar.payload.claims.len();
                    report.memory_gap_count += memory_sidecar.payload.gaps.len();
                    report.memory_conflict_count += memory_sidecar.payload.conflicts.len();
                    report.memory_card_count += memory_sidecar.payload.entity_cards.len();
                    context.remember_memory_product(memory_sidecar);
                }
            }
            context.mark_stage_complete(&scope, stage);
        }
    }

    report.scheduler = context.metrics().clone();
    Ok(report)
}

fn run_relation_stage<S>(
    store: &S,
    context: &mut PipelineGenerationContext<'_, S>,
    scope: &ScopeGenerationKey,
    glirel_model: &phoenix_rel_post::GlirelModel,
    relation_specs: &[phoenix_rel_post::GlirelRelationTypeSpec],
    relation_created_at: i64,
) -> Result<(StageProductEnvelope<RelationScopePatchSidecar>, usize), PipelineApiError>
where
    S: PhoenixArchiveStoreV2 + PhoenixRelationPatchStore + PhoenixScopeRuntimeStore,
{
    let prepared = context.prepared_relation_input_for(
        scope,
        ScopeImageSpec::post_ingest(),
        relation_specs,
    )?;
    let mut batch = prepared.batch.clone();
    for job in &prepared.model_jobs {
        context.record_relation_model_job(job);
        rel_api::run_glirel_job_with_input(&mut batch, &prepared, glirel_model, job)?;
    }
    let decisions = rel_api::draft_decisions(&batch, relation_specs);
    let review_case_count = batch.review_cases.len();
    let sidecar = rel_api::persist_patch_sidecar(store, &batch, &decisions, relation_created_at)?;
    Ok((
        StageProductEnvelope {
            key: scope.clone(),
            stage: PipelineStage::Relation,
            created_at: relation_created_at,
            input_fingerprint: scope.generation,
            payload: sidecar,
        },
        review_case_count,
    ))
}

fn run_state_schema_stage<S>(
    store: &S,
    scope: &ScopeGenerationKey,
    analysis: &phoenix_scope_analysis::ScopeAnalysisContext,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
    created_at: i64,
) -> Result<StageProductEnvelope<StateSchemaScopeSidecar>, PipelineApiError>
where
    S: PhoenixStateSchemaPatchStore,
{
    let mut batch = state_schema_api::derive_batch_from_analysis(analysis, relation_sidecar);
    state_schema_api::run_batch(&mut batch, created_at);
    let sidecar = state_schema_api::persist_patch_sidecar(store, &batch, created_at)?;
    Ok(StageProductEnvelope {
        key: scope.clone(),
        stage: PipelineStage::StateSchema,
        created_at,
        input_fingerprint: scope.generation,
        payload: sidecar,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_memory_stage<S>(
    store: &S,
    scope: &ScopeGenerationKey,
    analysis: &phoenix_scope_analysis::ScopeAnalysisContext,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
    state_schema_sidecar: Option<&StateSchemaScopeSidecar>,
    event_identity_sidecar: Option<&phoenix_semantic_v2::EventIdentityScopeSidecar>,
    memory_sidecar: Option<&MemoryScopeSidecar>,
    created_at: i64,
) -> Result<StageProductEnvelope<MemoryScopeSidecar>, PipelineApiError>
where
    S: PhoenixMemoryPatchStore,
{
    let batch = memory_api::derive_batch_from_analysis(
        analysis,
        relation_sidecar,
        state_schema_sidecar,
        event_identity_sidecar,
        memory_sidecar,
    );
    let sidecar = memory_api::persist_patch_sidecar(store, &batch, created_at)?;
    Ok(StageProductEnvelope {
        key: scope.clone(),
        stage: PipelineStage::Memory,
        created_at,
        input_fingerprint: scope.generation,
        payload: sidecar,
    })
}

fn accumulate_state_schema_report(
    report: &mut StateSchemaRunReport,
    sidecar: &StateSchemaScopeSidecar,
) {
    report.state_schema_scope_count += 1;
    report.slot_family_count += sidecar.slot_families.len();
    report.slot_definition_count += sidecar.slot_definitions.len();
    report.active_definition_count += sidecar
        .slot_definitions
        .iter()
        .filter(|definition| {
            matches!(
                definition.lifecycle,
                phoenix_semantic_v2::StateSlotLifecycle::Active
                    | phoenix_semantic_v2::StateSlotLifecycle::Stable
            )
        })
        .count();
    report.candidate_count += sidecar.slot_candidates.len();
    report.write_proposal_count += sidecar.write_proposals.len();
}
