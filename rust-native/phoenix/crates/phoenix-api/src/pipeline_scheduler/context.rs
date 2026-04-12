use std::collections::HashMap;
use std::sync::Arc;

use phoenix_rel_post::{
    api as rel_api, relation_spec_signature, GlirelRelationTypeSpec, RelationModelJob,
    RelationPreparedStageInput,
};
use phoenix_scope_analysis::ScopeAnalysisContext;
use phoenix_semantic_v2::{
    DirtyScopeRecord, MemoryScopeSidecar, RelationScopePatchSidecar, SessionArchive,
    StateSchemaScopeSidecar,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixScopeRuntimeStore, ScopeImageSpec, ScopeRuntimeImage, StoreError,
};

use super::types::{
    stage_dependencies, PipelineRunMetrics, PipelineRunRequest, PipelineStage, PipelineStageStatus,
    ScopeGenerationKey, StageProductEnvelope,
};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RuntimeCacheKey {
    scope: ScopeGenerationKey,
    spec: ScopeImageSpec,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct RelationPreparedInputCacheKey {
    scope: ScopeGenerationKey,
    spec: ScopeImageSpec,
    relation_spec_signature: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct StageRunKey {
    scope: ScopeGenerationKey,
    stage: PipelineStage,
}

pub struct PipelineGenerationContext<'a, S> {
    store: &'a S,
    request: PipelineRunRequest,
    session: Option<Arc<SessionArchive>>,
    scope_keys: Vec<ScopeGenerationKey>,
    dirty_scopes: HashMap<ScopeGenerationKey, DirtyScopeRecord>,
    runtime_cache: HashMap<RuntimeCacheKey, Arc<ScopeRuntimeImage>>,
    analysis_cache: HashMap<RuntimeCacheKey, Arc<ScopeAnalysisContext>>,
    relation_prepared_input_cache:
        HashMap<RelationPreparedInputCacheKey, Arc<RelationPreparedStageInput>>,
    stage_statuses: HashMap<StageRunKey, PipelineStageStatus>,
    relation_products:
        HashMap<ScopeGenerationKey, Arc<StageProductEnvelope<RelationScopePatchSidecar>>>,
    state_schema_products:
        HashMap<ScopeGenerationKey, Arc<StageProductEnvelope<StateSchemaScopeSidecar>>>,
    memory_products: HashMap<ScopeGenerationKey, Arc<StageProductEnvelope<MemoryScopeSidecar>>>,
    metrics: PipelineRunMetrics,
}

impl<'a, S> PipelineGenerationContext<'a, S>
where
    S: PhoenixArchiveStoreV2 + PhoenixScopeRuntimeStore,
{
    pub fn new(store: &'a S, request: PipelineRunRequest) -> Result<Self, StoreError> {
        let session = match request.session_id.as_ref() {
            Some(session_id) => store.load_latest_session_archive(session_id)?.map(Arc::new),
            None => None,
        };
        let mut dirty = store.list_dirty_scopes()?;
        dirty.sort_by(|left, right| left.scope_key.cmp(&right.scope_key));

        let scope_keys = dirty
            .iter()
            .map(ScopeGenerationKey::from_dirty_scope)
            .collect::<Vec<_>>();
        let dirty_scopes = scope_keys
            .iter()
            .cloned()
            .zip(dirty)
            .collect::<HashMap<_, _>>();
        let mut stage_statuses = HashMap::new();
        for scope in &scope_keys {
            for &stage in &request.requested_stages {
                let status = if stage_dependencies(stage).is_empty() {
                    PipelineStageStatus::Ready
                } else {
                    PipelineStageStatus::Blocked
                };
                stage_statuses.insert(
                    StageRunKey {
                        scope: scope.clone(),
                        stage,
                    },
                    status,
                );
            }
        }

        let mut metrics = PipelineRunMetrics {
            scope_count: scope_keys.len(),
            requested_stage_count: scope_keys.len() * request.requested_stages.len(),
            ..Default::default()
        };
        metrics.stage_ready_count = stage_statuses
            .values()
            .filter(|status| **status == PipelineStageStatus::Ready)
            .count();

        Ok(Self {
            store,
            request,
            session,
            scope_keys,
            dirty_scopes,
            runtime_cache: HashMap::new(),
            analysis_cache: HashMap::new(),
            relation_prepared_input_cache: HashMap::new(),
            stage_statuses,
            relation_products: HashMap::new(),
            state_schema_products: HashMap::new(),
            memory_products: HashMap::new(),
            metrics,
        })
    }

    pub fn scope_keys(&self) -> &[ScopeGenerationKey] {
        &self.scope_keys
    }

    pub fn metrics(&self) -> &PipelineRunMetrics {
        &self.metrics
    }

    pub fn next_ready_stage_for_scope(&self, scope: &ScopeGenerationKey) -> Option<PipelineStage> {
        self.request
            .requested_stages
            .iter()
            .copied()
            .find(|stage| self.stage_status(scope, *stage) == PipelineStageStatus::Ready)
    }

    pub fn mark_stage_running(&mut self, scope: &ScopeGenerationKey, stage: PipelineStage) {
        self.stage_statuses.insert(
            StageRunKey {
                scope: scope.clone(),
                stage,
            },
            PipelineStageStatus::Running,
        );
        self.metrics.stage_run_count += 1;
    }

    pub fn mark_stage_complete(&mut self, scope: &ScopeGenerationKey, stage: PipelineStage) {
        self.stage_statuses.insert(
            StageRunKey {
                scope: scope.clone(),
                stage,
            },
            PipelineStageStatus::Complete,
        );
        self.metrics.stage_complete_count += 1;
        self.promote_blocked_dependents(scope);
    }

    pub fn analysis_for(
        &mut self,
        scope: &ScopeGenerationKey,
        spec: ScopeImageSpec,
    ) -> Result<Arc<ScopeAnalysisContext>, StoreError> {
        let cache_key = RuntimeCacheKey {
            scope: scope.clone(),
            spec,
        };
        if let Some(analysis) = self.analysis_cache.get(&cache_key) {
            self.metrics.analysis_cache_hits += 1;
            return Ok(analysis.clone());
        }
        self.metrics.analysis_cache_misses += 1;
        let runtime = self.runtime_image_for(scope, spec)?;
        let analysis = Arc::new(ScopeAnalysisContext::from_runtime_image(
            (*runtime).clone(),
            self.session.as_deref(),
        ));
        self.analysis_cache.insert(cache_key, analysis.clone());
        Ok(analysis)
    }

    pub fn relation_product(
        &self,
        scope: &ScopeGenerationKey,
    ) -> Option<Arc<StageProductEnvelope<RelationScopePatchSidecar>>> {
        self.relation_products.get(scope).cloned()
    }

    pub fn state_schema_product(
        &self,
        scope: &ScopeGenerationKey,
    ) -> Option<Arc<StageProductEnvelope<StateSchemaScopeSidecar>>> {
        self.state_schema_products.get(scope).cloned()
    }

    pub fn remember_relation_product(
        &mut self,
        product: StageProductEnvelope<RelationScopePatchSidecar>,
    ) {
        self.relation_products
            .insert(product.key.clone(), Arc::new(product));
        self.metrics.stage_product_count += 1;
    }

    pub fn remember_state_schema_product(
        &mut self,
        product: StageProductEnvelope<StateSchemaScopeSidecar>,
    ) {
        self.state_schema_products
            .insert(product.key.clone(), Arc::new(product));
        self.metrics.stage_product_count += 1;
    }

    pub fn remember_memory_product(&mut self, product: StageProductEnvelope<MemoryScopeSidecar>) {
        self.memory_products
            .insert(product.key.clone(), Arc::new(product));
        self.metrics.stage_product_count += 1;
    }

    pub fn prepared_relation_input_for(
        &mut self,
        scope: &ScopeGenerationKey,
        spec: ScopeImageSpec,
        relation_specs: &[GlirelRelationTypeSpec],
    ) -> Result<Arc<RelationPreparedStageInput>, crate::PipelineApiError> {
        let relation_spec_signature = relation_spec_signature(relation_specs);
        let cache_key = RelationPreparedInputCacheKey {
            scope: scope.clone(),
            spec,
            relation_spec_signature,
        };
        if let Some(prepared) = self.relation_prepared_input_cache.get(&cache_key) {
            self.metrics.relation_prepared_input_cache_hits += 1;
            return Ok(prepared.clone());
        }
        self.metrics.relation_prepared_input_cache_misses += 1;
        let analysis = self.analysis_for(scope, spec)?;
        let prepared = Arc::new(rel_api::prepare_stage_input_from_analysis(
            &analysis,
            relation_specs,
        )?);
        self.metrics.relation_schema_group_count += prepared.plan.schema_groups.len();
        self.relation_prepared_input_cache
            .insert(cache_key, prepared.clone());
        Ok(prepared)
    }

    pub fn record_relation_model_job(&mut self, job: &RelationModelJob) {
        self.metrics.relation_model_job_count += 1;
        self.metrics.relation_model_job_window_count += job.window_count;
        self.metrics.relation_model_job_pair_slots += job.estimated_pair_slots;
    }

    fn runtime_image_for(
        &mut self,
        scope: &ScopeGenerationKey,
        spec: ScopeImageSpec,
    ) -> Result<Arc<ScopeRuntimeImage>, StoreError> {
        let cache_key = RuntimeCacheKey {
            scope: scope.clone(),
            spec,
        };
        if let Some(runtime) = self.runtime_cache.get(&cache_key) {
            self.metrics.runtime_image_cache_hits += 1;
            return Ok(runtime.clone());
        }
        self.metrics.runtime_image_cache_misses += 1;
        let dirty = self
            .dirty_scopes
            .get(scope)
            .expect("dirty scope key should exist");
        let runtime = Arc::new(self.store.load_scope_runtime_image(dirty, spec)?);
        self.runtime_cache.insert(cache_key, runtime.clone());
        Ok(runtime)
    }

    fn stage_status(
        &self,
        scope: &ScopeGenerationKey,
        stage: PipelineStage,
    ) -> PipelineStageStatus {
        self.stage_statuses
            .get(&StageRunKey {
                scope: scope.clone(),
                stage,
            })
            .copied()
            .unwrap_or(PipelineStageStatus::NotRequested)
    }

    fn promote_blocked_dependents(&mut self, scope: &ScopeGenerationKey) {
        for &stage in &self.request.requested_stages {
            if self.stage_status(scope, stage) != PipelineStageStatus::Blocked {
                continue;
            }
            if stage_dependencies(stage).iter().all(|dependency| {
                self.stage_status(scope, *dependency) == PipelineStageStatus::Complete
            }) {
                self.stage_statuses.insert(
                    StageRunKey {
                        scope: scope.clone(),
                        stage,
                    },
                    PipelineStageStatus::Ready,
                );
                self.metrics.stage_ready_count += 1;
            }
        }
    }
}
