//! Canonical discovery and orchestration façade for Phoenix post-ingest
//! pipeline stages.

use serde::{Deserialize, Serialize};

use phoenix_alex::{api as alex_api, AlexError, Lexicon};
use phoenix_causal_post::api as causal_api;
use phoenix_er_post::api as er_api;
use phoenix_event_identity_post::api as event_identity_api;
use phoenix_graph_post::api as graph_api;
use phoenix_memory_post::api as memory_api;
use phoenix_rel_post::api as rel_api;
use phoenix_state_schema_post::api as state_schema_api;
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixCausalPatchStore, PhoenixErPatchStore,
    PhoenixEventIdentityPatchStore, PhoenixGraphPatchStore, PhoenixMemoryPatchStore,
    PhoenixRelationPatchStore, PhoenixSemanticGraphPatchStore, PhoenixSemanticIndexStore,
    PhoenixStateSchemaPatchStore, PhoenixTemporalPatchStore, StoreError,
};
use phoenix_temporal_post::api as temporal_api;
use phoenix_types::{LexiconEntry, ScopeKey, SessionId};

#[derive(Debug, thiserror::Error)]
pub enum PipelineApiError {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error(transparent)]
    Alex(#[from] AlexError),
    #[error(transparent)]
    Relation(#[from] phoenix_rel_post::GlirelWorkerError),
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PostIngestRunReport {
    pub relation_scope_count: usize,
    pub relation_case_count: usize,
    pub persisted_relation_edge_count: usize,
    pub state_schema_scope_count: usize,
    pub state_schema_active_definition_count: usize,
    pub state_schema_candidate_count: usize,
    pub memory_scope_count: usize,
    pub memory_state_count: usize,
    pub memory_card_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSchemaRunReport {
    pub state_schema_scope_count: usize,
    pub slot_family_count: usize,
    pub slot_definition_count: usize,
    pub active_definition_count: usize,
    pub candidate_count: usize,
    pub write_proposal_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityRunReport {
    pub event_identity_scope_count: usize,
    pub mention_packet_count: usize,
    pub hypothesis_count: usize,
    pub canonical_event_count: usize,
    pub canonical_card_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalRunReport {
    pub causal_scope_count: usize,
    pub causal_review_case_count: usize,
    pub causal_edge_count: usize,
    pub causal_chain_count: usize,
    pub causal_card_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalRunReport {
    pub temporal_scope_count: usize,
    pub temporal_review_case_count: usize,
    pub temporal_interval_count: usize,
    pub temporal_segment_count: usize,
    pub temporal_gap_count: usize,
    pub temporal_card_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphRunReport {
    pub graph_scope_count: usize,
    pub graph_projection_vertex_count: usize,
    pub graph_projection_edge_count: usize,
    pub graph_claim_node_count: usize,
    pub graph_event_node_count: usize,
    pub graph_state_node_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContinuityRunReport {
    pub event_identity: EventIdentityRunReport,
    pub temporal: TemporalRunReport,
    pub causal: CausalRunReport,
    pub state_schema: StateSchemaRunReport,
    pub post_ingest: PostIngestRunReport,
}

pub struct PhoenixPipelineApi<S> {
    store: S,
}

impl<S> PhoenixPipelineApi<S> {
    pub fn new(store: S) -> Self {
        Self { store }
    }

    pub fn store(&self) -> &S {
        &self.store
    }

    pub fn chunker(&self) -> ChunkerStageApi {
        ChunkerStageApi
    }

    pub fn alex(&self) -> AlexStageApi {
        AlexStageApi
    }

    pub fn er(&self) -> ErStageApi<'_, S> {
        ErStageApi { store: &self.store }
    }

    pub fn event_identity(&self) -> EventIdentityStageApi<'_, S> {
        EventIdentityStageApi { store: &self.store }
    }

    pub fn causal(&self) -> CausalStageApi<'_, S> {
        CausalStageApi { store: &self.store }
    }

    pub fn state_schema(&self) -> StateSchemaStageApi<'_, S> {
        StateSchemaStageApi { store: &self.store }
    }

    pub fn temporal(&self) -> TemporalStageApi<'_, S> {
        TemporalStageApi { store: &self.store }
    }

    pub fn graph(&self) -> GraphStageApi<'_, S> {
        GraphStageApi { store: &self.store }
    }

    pub fn rel(&self) -> RelStageApi<'_, S> {
        RelStageApi { store: &self.store }
    }

    pub fn memory(&self) -> MemoryStageApi<'_, S> {
        MemoryStageApi { store: &self.store }
    }
}

impl<S> PhoenixPipelineApi<S>
where
    S: PhoenixArchiveStoreV2
        + PhoenixErPatchStore
        + PhoenixEventIdentityPatchStore
        + PhoenixRelationPatchStore
        + PhoenixMemoryPatchStore
        + PhoenixCausalPatchStore
        + PhoenixStateSchemaPatchStore
        + PhoenixTemporalPatchStore,
{
    pub fn run_post_ingest_scope(
        &self,
        session_id: Option<&SessionId>,
        glirel_model: &phoenix_rel_post::GlirelModel,
        relation_specs: &[phoenix_rel_post::GlirelRelationTypeSpec],
        relation_created_at: i64,
        memory_created_at: i64,
    ) -> Result<PostIngestRunReport, PipelineApiError> {
        let mut relation_batches = rel_api::derive_batches(&self.store, session_id)?;
        let mut persisted_relation_edge_count = 0usize;
        let mut relation_case_count = 0usize;
        for batch in &mut relation_batches {
            rel_api::run_glirel(batch, glirel_model, relation_specs)?;
            let decisions = rel_api::draft_decisions(batch, relation_specs);
            let sidecar = rel_api::persist_patch_sidecar(
                &self.store,
                batch,
                &decisions,
                relation_created_at,
            )?;
            relation_case_count += batch.review_cases.len();
            persisted_relation_edge_count += sidecar.edge_additions.len();
        }

        let mut state_schema_batches = state_schema_api::derive_batches(&self.store, session_id)?;
        let mut state_schema_active_definition_count = 0usize;
        let mut state_schema_candidate_count = 0usize;
        for batch in &mut state_schema_batches {
            state_schema_api::run_batch(batch, relation_created_at);
            let sidecar =
                state_schema_api::persist_patch_sidecar(&self.store, batch, relation_created_at)?;
            state_schema_active_definition_count += sidecar
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
            state_schema_candidate_count += sidecar.slot_candidates.len();
        }

        let memory_batches = memory_api::derive_batches(&self.store, session_id)?;
        let mut memory_state_count = 0usize;
        let mut memory_card_count = 0usize;
        for batch in &memory_batches {
            let sidecar = memory_api::persist_patch_sidecar(&self.store, batch, memory_created_at)?;
            memory_state_count += sidecar.states.len();
            memory_card_count += sidecar.entity_cards.len();
        }

        Ok(PostIngestRunReport {
            relation_scope_count: relation_batches.len(),
            relation_case_count,
            persisted_relation_edge_count,
            state_schema_scope_count: state_schema_batches.len(),
            state_schema_active_definition_count,
            state_schema_candidate_count,
            memory_scope_count: memory_batches.len(),
            memory_state_count,
            memory_card_count,
        })
    }

    pub fn run_causal_scope(
        &self,
        session_id: Option<&SessionId>,
        created_at: i64,
    ) -> Result<CausalRunReport, PipelineApiError> {
        let mut batches = causal_api::derive_batches(&self.store, session_id)?;
        let mut causal_review_case_count = 0usize;
        let mut causal_edge_count = 0usize;
        let mut causal_chain_count = 0usize;
        let mut causal_card_count = 0usize;
        for batch in &mut batches {
            causal_api::run_batch(batch, created_at);
            let sidecar = causal_api::persist_patch_sidecar(&self.store, batch, created_at)?;
            causal_review_case_count += batch.review_cases.len();
            causal_edge_count += sidecar.edge_additions.len();
            causal_chain_count += sidecar.chains.len();
            causal_card_count += sidecar.memory_cards.len();
        }
        Ok(CausalRunReport {
            causal_scope_count: batches.len(),
            causal_review_case_count,
            causal_edge_count,
            causal_chain_count,
            causal_card_count,
        })
    }

    pub fn run_event_identity_scope(
        &self,
        session_id: Option<&SessionId>,
        created_at: i64,
    ) -> Result<EventIdentityRunReport, PipelineApiError> {
        let mut batches = event_identity_api::derive_batches(&self.store, session_id)?;
        let mut mention_packet_count = 0usize;
        let mut hypothesis_count = 0usize;
        let mut canonical_event_count = 0usize;
        let mut canonical_card_count = 0usize;
        for batch in &mut batches {
            event_identity_api::run_batch(batch, created_at);
            let sidecar =
                event_identity_api::persist_patch_sidecar(&self.store, batch, created_at)?;
            mention_packet_count += sidecar.mention_packets.len();
            hypothesis_count += sidecar.identity_hypotheses.len();
            canonical_event_count += sidecar.canonical_events.len();
            canonical_card_count += sidecar.canonical_event_cards.len();
        }
        Ok(EventIdentityRunReport {
            event_identity_scope_count: batches.len(),
            mention_packet_count,
            hypothesis_count,
            canonical_event_count,
            canonical_card_count,
        })
    }

    pub fn run_temporal_scope(
        &self,
        session_id: Option<&SessionId>,
        created_at: i64,
    ) -> Result<TemporalRunReport, PipelineApiError> {
        let mut batches = temporal_api::derive_batches(&self.store, session_id)?;
        let mut temporal_review_case_count = 0usize;
        let mut temporal_interval_count = 0usize;
        let mut temporal_segment_count = 0usize;
        let mut temporal_gap_count = 0usize;
        let mut temporal_card_count = 0usize;
        for batch in &mut batches {
            temporal_api::run_batch(batch, created_at);
            let sidecar = temporal_api::persist_patch_sidecar(&self.store, batch, created_at)?;
            temporal_review_case_count += batch.review_cases.len();
            temporal_interval_count += sidecar.intervals.len();
            temporal_segment_count += sidecar.timeline_segments.len();
            temporal_gap_count += sidecar.gaps.len();
            temporal_card_count += sidecar.memory_cards.len();
        }
        Ok(TemporalRunReport {
            temporal_scope_count: batches.len(),
            temporal_review_case_count,
            temporal_interval_count,
            temporal_segment_count,
            temporal_gap_count,
            temporal_card_count,
        })
    }

    pub fn run_graph_scope(
        &self,
        session_id: Option<&SessionId>,
        created_at: i64,
    ) -> Result<GraphRunReport, PipelineApiError>
    where
        S: PhoenixGraphPatchStore + PhoenixSemanticGraphPatchStore,
    {
        let batches = graph_api::derive_batches(&self.store, session_id)?;
        let mut graph_projection_vertex_count = 0usize;
        let mut graph_projection_edge_count = 0usize;
        let mut graph_claim_node_count = 0usize;
        let mut graph_event_node_count = 0usize;
        let mut graph_state_node_count = 0usize;
        for batch in &batches {
            let sidecar = graph_api::persist_patch_sidecar(&self.store, batch, created_at)?;
            graph_projection_vertex_count += sidecar.summary.projection_vertex_count;
            graph_projection_edge_count += sidecar.summary.projection_edge_count;
            graph_claim_node_count += sidecar.summary.claim_node_count;
            graph_event_node_count += sidecar.summary.event_node_count;
            graph_state_node_count += sidecar.summary.state_node_count;
        }
        Ok(GraphRunReport {
            graph_scope_count: batches.len(),
            graph_projection_vertex_count,
            graph_projection_edge_count,
            graph_claim_node_count,
            graph_event_node_count,
            graph_state_node_count,
        })
    }

    pub fn run_state_schema_scope(
        &self,
        session_id: Option<&SessionId>,
        created_at: i64,
    ) -> Result<StateSchemaRunReport, PipelineApiError> {
        let mut batches = state_schema_api::derive_batches(&self.store, session_id)?;
        let mut slot_family_count = 0usize;
        let mut slot_definition_count = 0usize;
        let mut active_definition_count = 0usize;
        let mut candidate_count = 0usize;
        let mut write_proposal_count = 0usize;
        for batch in &mut batches {
            state_schema_api::run_batch(batch, created_at);
            let sidecar = state_schema_api::persist_patch_sidecar(&self.store, batch, created_at)?;
            slot_family_count += sidecar.slot_families.len();
            slot_definition_count += sidecar.slot_definitions.len();
            active_definition_count += sidecar
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
            candidate_count += sidecar.slot_candidates.len();
            write_proposal_count += sidecar.write_proposals.len();
        }
        Ok(StateSchemaRunReport {
            state_schema_scope_count: batches.len(),
            slot_family_count,
            slot_definition_count,
            active_definition_count,
            candidate_count,
            write_proposal_count,
        })
    }

    pub fn run_continuity_scope(
        &self,
        session_id: Option<&SessionId>,
        event_identity_created_at: i64,
        temporal_created_at: i64,
        causal_created_at: i64,
        glirel_model: &phoenix_rel_post::GlirelModel,
        relation_specs: &[phoenix_rel_post::GlirelRelationTypeSpec],
        relation_created_at: i64,
        memory_created_at: i64,
    ) -> Result<ContinuityRunReport, PipelineApiError> {
        let event_identity =
            self.run_event_identity_scope(session_id, event_identity_created_at)?;
        let temporal = self.run_temporal_scope(session_id, temporal_created_at)?;
        let causal = self.run_causal_scope(session_id, causal_created_at)?;
        let state_schema = self.run_state_schema_scope(session_id, relation_created_at)?;
        let post_ingest = self.run_post_ingest_scope(
            session_id,
            glirel_model,
            relation_specs,
            relation_created_at,
            memory_created_at,
        )?;

        Ok(ContinuityRunReport {
            event_identity,
            temporal,
            causal,
            state_schema,
            post_ingest,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ChunkerStageApi;

impl ChunkerStageApi {
    pub fn sentence_ranges(&self, text: &str) -> Vec<(usize, usize)> {
        phoenix_chunker::api::sentence_ranges(text)
    }

    pub fn build_chunks(
        &self,
        text: &str,
        config: &phoenix_chunker::ChunkerConfig,
    ) -> Vec<phoenix_chunker::Chunk> {
        phoenix_chunker::api::chunk_ranges(text, config)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AlexStageApi;

impl AlexStageApi {
    pub fn build_lexicon(&self, entries: &[LexiconEntry]) -> Result<Lexicon, AlexError> {
        alex_api::build_lexicon(entries)
    }

    pub fn build_snapshot(
        &self,
        entries: &[LexiconEntry],
    ) -> Result<phoenix_types::LexiconSnapshot, AlexError> {
        alex_api::build_snapshot(entries)
    }

    pub fn scan_text(
        &self,
        lexicon: &Lexicon,
        text: &str,
        scope: &ScopeKey,
    ) -> Vec<phoenix_types::KnownMatch> {
        alex_api::scan_text(lexicon, text, scope)
    }
}

pub struct ErStageApi<'a, S> {
    store: &'a S,
}

impl<'a, S> ErStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2,
{
    pub fn derive_batches(
        &self,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<phoenix_er_post::ErScopeReviewBatch>, StoreError> {
        er_api::derive_batches(self.store, session_id)
    }
}

impl<'a, S> ErStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore,
{
    pub fn derive_batches_with_replay(
        &self,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<phoenix_er_post::ErScopeReviewBatch>, StoreError> {
        er_api::derive_batches_with_replay(self.store, session_id)
    }
}

pub struct EventIdentityStageApi<'a, S> {
    store: &'a S,
}

impl<'a, S> EventIdentityStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixEventIdentityPatchStore,
{
    pub fn derive_batches(
        &self,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<phoenix_event_identity_post::EventIdentityScopeReviewBatch>, StoreError> {
        event_identity_api::derive_batches(self.store, session_id)
    }

    pub fn run_scope(
        &self,
        batch: &mut phoenix_event_identity_post::EventIdentityScopeReviewBatch,
        created_at: i64,
    ) {
        event_identity_api::run_batch(batch, created_at);
    }
}

pub struct CausalStageApi<'a, S> {
    store: &'a S,
}

impl<'a, S> CausalStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2
        + PhoenixErPatchStore
        + PhoenixCausalPatchStore
        + PhoenixEventIdentityPatchStore,
{
    pub fn derive_batches(
        &self,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<phoenix_causal_post::CausalScopeReviewBatch>, StoreError> {
        causal_api::derive_batches(self.store, session_id)
    }

    pub fn run_scope(
        &self,
        batch: &mut phoenix_causal_post::CausalScopeReviewBatch,
        created_at: i64,
    ) {
        causal_api::run_batch(batch, created_at);
    }
}

pub struct StateSchemaStageApi<'a, S> {
    store: &'a S,
}

impl<'a, S> StateSchemaStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2 + PhoenixRelationPatchStore + PhoenixStateSchemaPatchStore,
{
    pub fn derive_batches(
        &self,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<phoenix_state_schema_post::StateSchemaScopeReviewBatch>, StoreError> {
        state_schema_api::derive_batches(self.store, session_id)
    }

    pub fn run_scope(
        &self,
        batch: &mut phoenix_state_schema_post::StateSchemaScopeReviewBatch,
        created_at: i64,
    ) {
        state_schema_api::run_batch(batch, created_at);
    }
}

pub struct TemporalStageApi<'a, S> {
    store: &'a S,
}

impl<'a, S> TemporalStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2 + PhoenixTemporalPatchStore + PhoenixEventIdentityPatchStore,
{
    pub fn derive_batches(
        &self,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<phoenix_temporal_post::TemporalScopeReviewBatch>, StoreError> {
        temporal_api::derive_batches(self.store, session_id)
    }

    pub fn run_scope(
        &self,
        batch: &mut phoenix_temporal_post::TemporalScopeReviewBatch,
        created_at: i64,
    ) {
        temporal_api::run_batch(batch, created_at);
    }
}

pub struct GraphStageApi<'a, S> {
    store: &'a S,
}

impl<'a, S> GraphStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2
        + PhoenixGraphPatchStore
        + PhoenixSemanticGraphPatchStore
        + PhoenixSemanticIndexStore
        + PhoenixEventIdentityPatchStore
        + PhoenixTemporalPatchStore
        + PhoenixCausalPatchStore
        + PhoenixMemoryPatchStore,
{
    pub fn derive_batches(
        &self,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<phoenix_graph_post::GraphScopeReviewBatch>, StoreError> {
        graph_api::derive_batches(self.store, session_id)
    }

    pub fn current_slot(
        &self,
        scope: &ScopeKey,
        entity_id: &str,
        slot_key: &str,
        recorded_at: Option<i64>,
    ) -> Result<Option<graph_api::GraphRankedSlotAnswer>, graph_api::GraphQueryError> {
        graph_api::current_slot(self.store, scope, entity_id, slot_key, recorded_at)
    }

    pub fn slot_at(
        &self,
        scope: &ScopeKey,
        request: &graph_api::GraphWorldStateQueryRequest,
    ) -> Result<Option<graph_api::GraphRankedSlotAnswer>, graph_api::GraphQueryError> {
        graph_api::slot_at(self.store, scope, request)
    }

    pub fn what_is_unresolved(
        &self,
        scope: &ScopeKey,
        request: &phoenix_graph_kernel::KernelUnresolvedQueryRequest,
    ) -> Result<Option<Vec<phoenix_graph_kernel::KernelStateIssue>>, graph_api::GraphQueryError>
    {
        graph_api::what_is_unresolved(self.store, scope, request)
    }

    pub fn what_changed(
        &self,
        scope: &ScopeKey,
        request: &phoenix_graph_kernel::KernelWhatChangedRequest,
    ) -> Result<Option<Vec<phoenix_graph_kernel::KernelStateChange>>, graph_api::GraphQueryError>
    {
        graph_api::what_changed(self.store, scope, request)
    }

    pub fn history(
        &self,
        scope: &ScopeKey,
        request: &graph_api::GraphHistoryQueryRequest,
    ) -> Result<Option<graph_api::GraphRankedHistoryAnswer>, graph_api::GraphQueryError> {
        graph_api::history(self.store, scope, request)
    }

    pub fn causal_explanation(
        &self,
        scope: &ScopeKey,
        request: &graph_api::GraphCausalExplanationQueryRequest,
    ) -> Result<Option<graph_api::GraphRankedCausalExplanationAnswer>, graph_api::GraphQueryError>
    {
        graph_api::causal_explanation(self.store, scope, request)
    }

    pub fn ranked_query(
        &self,
        scope: &ScopeKey,
        request: &graph_api::GraphRankedQueryRequest,
    ) -> Result<Option<graph_api::GraphRankedQueryAnswer>, graph_api::GraphQueryError> {
        graph_api::ranked_query(self.store, scope, request)
    }

    pub fn retrieved_world_state(
        &self,
        scope: &ScopeKey,
        request: &graph_api::GraphRetrievedWorldStateQueryRequest,
    ) -> Result<Option<graph_api::GraphRetrievedWorldStateAnswer>, graph_api::GraphQueryError> {
        graph_api::retrieved_world_state(self.store, scope, request)
    }

    pub fn retrieved_history(
        &self,
        scope: &ScopeKey,
        request: &graph_api::GraphRetrievedHistoryQueryRequest,
    ) -> Result<Option<graph_api::GraphRetrievedHistoryAnswer>, graph_api::GraphQueryError> {
        graph_api::retrieved_history(self.store, scope, request)
    }

    pub fn retrieved_causal_explanation(
        &self,
        scope: &ScopeKey,
        request: &graph_api::GraphRetrievedCausalExplanationQueryRequest,
    ) -> Result<Option<graph_api::GraphRetrievedCausalExplanationAnswer>, graph_api::GraphQueryError>
    {
        graph_api::retrieved_causal_explanation(self.store, scope, request)
    }

    pub fn retrieved_query(
        &self,
        scope: &ScopeKey,
        request: &graph_api::GraphRetrievedQueryRequest,
    ) -> Result<Option<graph_api::GraphRetrievedQueryAnswer>, graph_api::GraphQueryError> {
        graph_api::retrieved_query(self.store, scope, request)
    }
}

pub struct RelStageApi<'a, S> {
    store: &'a S,
}

impl<'a, S> RelStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixRelationPatchStore,
{
    pub fn derive_batches(
        &self,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<phoenix_rel_post::RelationScopeReviewBatch>, StoreError> {
        rel_api::derive_batches(self.store, session_id)
    }
}

pub struct MemoryStageApi<'a, S> {
    store: &'a S,
}

impl<'a, S> MemoryStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2
        + PhoenixErPatchStore
        + PhoenixRelationPatchStore
        + PhoenixMemoryPatchStore
        + PhoenixEventIdentityPatchStore
        + PhoenixStateSchemaPatchStore,
{
    pub fn derive_batches(
        &self,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<phoenix_memory_post::MemoryScopeReviewBatch>, StoreError> {
        memory_api::derive_batches(self.store, session_id)
    }
}

#[cfg(test)]
mod tests {
    use super::PhoenixPipelineApi;
    use phoenix_types::{EntityId, LexiconEntry, ScopeKey};

    #[test]
    fn chunker_and_alex_stage_api_smoke() {
        let api = PhoenixPipelineApi::new(());
        let chunker = api.chunker();
        let alex = api.alex();
        let _causal = api.causal();
        let _temporal = api.temporal();
        let text = "Alice works for Dynamis. Dynamis is in New Rome.";
        let sentences = chunker.sentence_ranges(text);
        assert_eq!(sentences.len(), 2);

        let entries = vec![
            LexiconEntry {
                entity_id: EntityId("e1".to_owned()),
                label: "Alice".to_owned(),
                aliases: Vec::new(),
                kind: Some(phoenix_types::EntityKind::Character),
                gender: None,
                number: None,
                scope: ScopeKey::default(),
            },
            LexiconEntry {
                entity_id: EntityId("e2".to_owned()),
                label: "Dynamis".to_owned(),
                aliases: Vec::new(),
                kind: Some(phoenix_types::EntityKind::Organization),
                gender: None,
                number: None,
                scope: ScopeKey::default(),
            },
        ];
        let lexicon = alex.build_lexicon(&entries).expect("build lexicon");
        let matches = alex.scan_text(&lexicon, text, &ScopeKey::default());
        assert!(!matches.is_empty());
    }
}
