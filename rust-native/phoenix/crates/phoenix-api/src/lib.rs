//! Canonical discovery and orchestration façade for Phoenix post-ingest
//! pipeline stages.

use serde::{Deserialize, Serialize};

use phoenix_alex::{api as alex_api, AlexError, Lexicon};
use phoenix_causal_post::api as causal_api;
use phoenix_er_post::api as er_api;
use phoenix_memory_post::api as memory_api;
use phoenix_rel_post::api as rel_api;
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixCausalPatchStore, PhoenixErPatchStore, PhoenixMemoryPatchStore,
    PhoenixRelationPatchStore, StoreError,
};
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
    pub memory_scope_count: usize,
    pub memory_state_count: usize,
    pub memory_card_count: usize,
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

    pub fn causal(&self) -> CausalStageApi<'_, S> {
        CausalStageApi { store: &self.store }
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
        + PhoenixRelationPatchStore
        + PhoenixMemoryPatchStore
        + PhoenixCausalPatchStore,
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
            let sidecar =
                rel_api::persist_patch_sidecar(&self.store, batch, &decisions, relation_created_at)?;
            relation_case_count += batch.review_cases.len();
            persisted_relation_edge_count += sidecar.edge_additions.len();
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

pub struct CausalStageApi<'a, S> {
    store: &'a S,
}

impl<'a, S> CausalStageApi<'a, S>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixCausalPatchStore,
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
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixRelationPatchStore + PhoenixMemoryPatchStore,
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
