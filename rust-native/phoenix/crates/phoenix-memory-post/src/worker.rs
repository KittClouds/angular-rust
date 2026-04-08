use phoenix_semantic_v2::{
    scope_storage_key, DirtyScopeRecord, DocumentArchive, DocumentRevisionRef, EntityMemoryCard,
    MemoryClaimAtom, MemoryCompilerSummary, MemoryConflictRecord, MemoryContinuityGapRecord,
    MemoryDeltaRecord, MemoryEventRecord, MemoryScopeSidecar, MemoryStateRecord,
    RelationScopePatchSidecar, RelationshipMemoryLedger, ScopeLexSidecar, ScopeOrd,
    SessionArchive, ErScopePatchSidecar,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixMemoryPatchStore, PhoenixRelationPatchStore,
    StoreError,
};
use phoenix_types::{ScopeKey, SessionId};
use serde::{Deserialize, Serialize};

use crate::compile::compile_memory;
use crate::normalize::{normalize_memory_inputs, MemoryEntityProfile};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MemoryScopeReviewBatch {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub dirty: Option<DirtyScopeRecord>,
    #[serde(default)]
    pub document_refs: Vec<DocumentRevisionRef>,
    #[serde(default)]
    pub entity_profiles: Vec<MemoryEntityProfile>,
    #[serde(default)]
    pub claims: Vec<MemoryClaimAtom>,
    #[serde(default)]
    pub events: Vec<MemoryEventRecord>,
    #[serde(default)]
    pub states: Vec<MemoryStateRecord>,
    #[serde(default)]
    pub deltas: Vec<MemoryDeltaRecord>,
    #[serde(default)]
    pub conflicts: Vec<MemoryConflictRecord>,
    #[serde(default)]
    pub gaps: Vec<MemoryContinuityGapRecord>,
    #[serde(default)]
    pub entity_cards: Vec<EntityMemoryCard>,
    #[serde(default)]
    pub relationship_ledgers: Vec<RelationshipMemoryLedger>,
    pub lexical_generation: Option<u64>,
    pub er_generation: Option<u64>,
    pub relation_generation: Option<u64>,
    pub memory_generation: Option<u64>,
    pub summary: MemoryCompilerSummary,
}

pub fn derive_scope_review_batch(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    lexical: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
) -> MemoryScopeReviewBatch {
    let scope = archives
        .first()
        .map(|archive| archive.manifest.scope.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope.clone()))
        .or_else(|| lexical.as_ref().map(|value| value.scope.clone()))
        .unwrap_or_default();
    let scope_key = archives
        .first()
        .map(|archive| archive.manifest.scope_key.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope_key.clone()))
        .or_else(|| lexical.as_ref().map(|value| value.scope_key.clone()))
        .unwrap_or_default();
    let scope_ord = archives
        .first()
        .map(|archive| archive.manifest.scope_ord)
        .or_else(|| dirty.as_ref().map(|record| record.scope_ord))
        .or_else(|| lexical.as_ref().and_then(|value| value.scope_ord))
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

    let normalized = normalize_memory_inputs(archives, session, lexical, er_sidecar, relation_sidecar);
    let compiled = compile_memory(&normalized);

    MemoryScopeReviewBatch {
        scope,
        scope_key,
        scope_ord,
        session_id,
        dirty: dirty.cloned(),
        document_refs,
        entity_profiles: normalized.entity_profiles,
        claims: compiled.claims,
        events: compiled.events,
        states: compiled.states,
        deltas: compiled.deltas,
        conflicts: compiled.conflicts,
        gaps: compiled.gaps,
        entity_cards: compiled.entity_cards,
        relationship_ledgers: compiled.relationship_ledgers,
        lexical_generation: lexical.map(|value| value.generation),
        er_generation: er_sidecar.map(|value| value.generation),
        relation_generation: relation_sidecar.map(|value| value.generation),
        memory_generation: None,
        summary: compiled.summary,
    }
}

pub fn derive_scope_review_batch_from_store<S>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
) -> Result<MemoryScopeReviewBatch, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixRelationPatchStore + PhoenixMemoryPatchStore,
{
    let archives = store.load_latest_document_archives(Some(&dirty.scope))?;
    let lexical = store.load_scope_sidecar(&dirty.scope)?;
    let er_sidecar = store.load_er_patch_sidecar(&dirty.scope)?;
    let relation_sidecar = store.load_relation_patch_sidecar(&dirty.scope)?;
    let mut batch = derive_scope_review_batch(
        &archives,
        session,
        Some(dirty),
        lexical.as_ref(),
        er_sidecar.as_ref(),
        relation_sidecar.as_ref(),
    );
    if let Some(memory_sidecar) = store.load_memory_patch_sidecar(&dirty.scope)? {
        apply_memory_patch_sidecar(&mut batch, &memory_sidecar);
    }
    Ok(batch)
}

pub fn derive_dirty_scope_review_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<MemoryScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixRelationPatchStore + PhoenixMemoryPatchStore,
{
    let session = match session_id {
        Some(value) => store.load_latest_session_archive(value)?,
        None => None,
    };
    let mut dirty = store.list_dirty_scopes()?;
    dirty.sort_by(|left, right| left.scope_key.cmp(&right.scope_key));
    dirty
        .into_iter()
        .map(|record| derive_scope_review_batch_from_store(store, &record, session.as_ref()))
        .collect()
}

pub fn build_memory_patch_sidecar(
    batch: &MemoryScopeReviewBatch,
    created_at: i64,
) -> MemoryScopeSidecar {
    MemoryScopeSidecar {
        scope: batch.scope.clone(),
        scope_key: batch.scope_key.clone(),
        scope_ord: Some(batch.scope_ord),
        session_id: batch.session_id.clone(),
        updated_at: created_at,
        generation: created_at as u64,
        claims: batch.claims.clone(),
        events: batch.events.clone(),
        states: batch.states.clone(),
        deltas: batch.deltas.clone(),
        conflicts: batch.conflicts.clone(),
        gaps: batch.gaps.clone(),
        entity_cards: batch.entity_cards.clone(),
        relationship_ledgers: batch.relationship_ledgers.clone(),
        summary: batch.summary.clone(),
    }
}

pub fn persist_memory_patch_sidecar<S>(
    store: &S,
    batch: &MemoryScopeReviewBatch,
    created_at: i64,
) -> Result<MemoryScopeSidecar, StoreError>
where
    S: PhoenixMemoryPatchStore,
{
    let updates = build_memory_patch_sidecar(batch, created_at);
    let merged = match store.load_memory_patch_sidecar(&batch.scope)? {
        Some(existing) => merge_memory_patch_sidecars(existing, updates),
        None => updates,
    };
    store.persist_memory_patch_sidecar(&merged)?;
    Ok(merged)
}

pub fn apply_memory_patch_sidecar(batch: &mut MemoryScopeReviewBatch, sidecar: &MemoryScopeSidecar) {
    batch.claims = sidecar.claims.clone();
    batch.events = sidecar.events.clone();
    batch.states = sidecar.states.clone();
    batch.deltas = sidecar.deltas.clone();
    batch.conflicts = sidecar.conflicts.clone();
    batch.gaps = sidecar.gaps.clone();
    batch.entity_cards = sidecar.entity_cards.clone();
    batch.relationship_ledgers = sidecar.relationship_ledgers.clone();
    batch.memory_generation = Some(sidecar.generation);
    batch.summary = sidecar.summary.clone();
}

fn merge_memory_patch_sidecars(
    mut existing: MemoryScopeSidecar,
    updates: MemoryScopeSidecar,
) -> MemoryScopeSidecar {
    existing.updated_at = existing.updated_at.max(updates.updated_at);
    existing.generation = existing.generation.max(updates.generation);
    existing.claims = updates.claims;
    existing.events = updates.events;
    existing.states = updates.states;
    existing.deltas = updates.deltas;
    existing.conflicts = updates.conflicts;
    existing.gaps = updates.gaps;
    existing.entity_cards = updates.entity_cards;
    existing.relationship_ledgers = updates.relationship_ledgers;
    existing.summary = updates.summary;
    existing
}
