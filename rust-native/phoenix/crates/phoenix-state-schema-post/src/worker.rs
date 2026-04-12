use std::collections::BTreeMap;

use phoenix_scope_analysis::ScopeAnalysisContext;
use phoenix_semantic_v2::{
    scope_storage_key, DirtyScopeRecord, DocumentArchive, DocumentRevisionRef,
    RelationScopePatchSidecar, ScopeOrd, SessionArchive, StateSchemaCompilerSummary,
    StateSchemaScopeSidecar, StateSlotCandidateRecord, StateSlotDefinitionRecord,
    StateSlotFamilyRecord, StateSlotLifecycle, StateSlotPromotionDecisionRecord,
    StateWriteProposal,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixRelationPatchStore, PhoenixStateSchemaPatchStore, StoreError,
};
use phoenix_types::{ScopeKey, SessionId};
use serde::{Deserialize, Serialize};

use crate::{
    mine_slot_candidates, normalize_state_schema_inputs,
    normalize_state_schema_inputs_from_analysis, promote_slot_definitions, StateSchemaEvidenceRow,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateSchemaScopeReviewBatch {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub dirty: Option<DirtyScopeRecord>,
    #[serde(default)]
    pub document_refs: Vec<DocumentRevisionRef>,
    #[serde(default)]
    pub slot_families: Vec<StateSlotFamilyRecord>,
    #[serde(default)]
    pub slot_definitions: Vec<StateSlotDefinitionRecord>,
    #[serde(default)]
    pub slot_candidates: Vec<StateSlotCandidateRecord>,
    #[serde(default)]
    pub promotion_decisions: Vec<StateSlotPromotionDecisionRecord>,
    #[serde(default)]
    pub write_proposals: Vec<StateWriteProposal>,
    pub relation_generation: Option<u64>,
    pub state_schema_generation: Option<u64>,
    pub summary: StateSchemaCompilerSummary,
    #[serde(default)]
    pub diagnostics: BTreeMap<String, usize>,
    #[serde(skip)]
    pub(crate) base_slot_definitions: Vec<StateSlotDefinitionRecord>,
    #[serde(skip)]
    pub(crate) evidence_rows: Vec<StateSchemaEvidenceRow>,
}

pub fn derive_scope_review_batch(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
) -> StateSchemaScopeReviewBatch {
    let scope = archives
        .first()
        .map(|archive| archive.manifest.scope.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope.clone()))
        .unwrap_or_default();
    let scope_key = archives
        .first()
        .map(|archive| archive.manifest.scope_key.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope_key.clone()))
        .unwrap_or_default();
    let scope_ord = archives
        .first()
        .map(|archive| archive.manifest.scope_ord)
        .or_else(|| dirty.as_ref().map(|record| record.scope_ord))
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
    let normalized = normalize_state_schema_inputs(archives, relation_sidecar);
    let base_slot_definitions = normalized.seed_slot_definitions.clone();

    StateSchemaScopeReviewBatch {
        scope,
        scope_key,
        scope_ord,
        session_id,
        dirty: dirty.cloned(),
        document_refs,
        slot_families: normalized.slot_families,
        slot_definitions: normalized.seed_slot_definitions,
        slot_candidates: Vec::new(),
        promotion_decisions: Vec::new(),
        write_proposals: Vec::new(),
        relation_generation: relation_sidecar.map(|value| value.generation),
        state_schema_generation: None,
        summary: StateSchemaCompilerSummary::default(),
        diagnostics: normalized.diagnostics,
        base_slot_definitions,
        evidence_rows: normalized.evidence_rows,
    }
}

pub fn derive_scope_review_batch_from_analysis(
    analysis: &ScopeAnalysisContext,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
) -> StateSchemaScopeReviewBatch {
    let normalized = normalize_state_schema_inputs_from_analysis(analysis, relation_sidecar);
    let base_slot_definitions = normalized.seed_slot_definitions.clone();

    StateSchemaScopeReviewBatch {
        scope: analysis.scope.clone(),
        scope_key: analysis.scope_key.clone(),
        scope_ord: analysis.dirty.scope_ord,
        session_id: analysis.session_id.clone(),
        dirty: Some(analysis.dirty.clone()),
        document_refs: analysis.document_refs.as_ref().to_vec(),
        slot_families: normalized.slot_families,
        slot_definitions: normalized.seed_slot_definitions,
        slot_candidates: Vec::new(),
        promotion_decisions: Vec::new(),
        write_proposals: Vec::new(),
        relation_generation: relation_sidecar.map(|value| value.generation),
        state_schema_generation: analysis
            .runtime
            .sidecars
            .state_schema
            .as_ref()
            .map(|value| value.generation),
        summary: StateSchemaCompilerSummary::default(),
        diagnostics: normalized.diagnostics,
        base_slot_definitions,
        evidence_rows: normalized.evidence_rows,
    }
}

pub fn derive_scope_review_batch_from_store<S>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
) -> Result<StateSchemaScopeReviewBatch, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixRelationPatchStore + PhoenixStateSchemaPatchStore,
{
    let archives = store.load_latest_document_archives(Some(&dirty.scope))?;
    let relation_sidecar = store.load_relation_patch_sidecar(&dirty.scope)?;
    let mut batch =
        derive_scope_review_batch(&archives, session, Some(dirty), relation_sidecar.as_ref());
    if let Some(sidecar) = store.load_state_schema_patch_sidecar(&dirty.scope)? {
        apply_state_schema_patch_sidecar(&mut batch, &sidecar);
    }
    Ok(batch)
}

pub fn derive_dirty_scope_review_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<StateSchemaScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixRelationPatchStore + PhoenixStateSchemaPatchStore,
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

pub fn run_state_schema_scope(batch: &mut StateSchemaScopeReviewBatch, created_at: i64) {
    let slot_candidates = mine_slot_candidates(&batch.evidence_rows);
    let promoted = promote_slot_definitions(
        &batch.base_slot_definitions,
        &slot_candidates,
        &batch.evidence_rows,
        created_at,
    );
    batch.slot_candidates = slot_candidates;
    batch.slot_definitions = promoted.slot_definitions;
    batch.promotion_decisions = promoted.promotion_decisions;
    batch.write_proposals = promoted.write_proposals;
    batch.summary = build_summary(batch);
    batch.diagnostics.insert(
        "write_proposal_count".to_owned(),
        batch.write_proposals.len(),
    );
}

pub fn build_state_schema_patch_sidecar(
    batch: &StateSchemaScopeReviewBatch,
    created_at: i64,
) -> StateSchemaScopeSidecar {
    StateSchemaScopeSidecar {
        scope: batch.scope.clone(),
        scope_key: batch.scope_key.clone(),
        scope_ord: Some(batch.scope_ord),
        session_id: batch.session_id.clone(),
        updated_at: created_at,
        generation: next_generation(batch.state_schema_generation),
        slot_families: batch.slot_families.clone(),
        slot_definitions: batch.slot_definitions.clone(),
        slot_candidates: batch.slot_candidates.clone(),
        promotion_decisions: batch.promotion_decisions.clone(),
        write_proposals: batch.write_proposals.clone(),
        summary: batch.summary.clone(),
        diagnostics: batch.diagnostics.clone(),
    }
}

pub fn persist_state_schema_patch_sidecar<S>(
    store: &S,
    batch: &StateSchemaScopeReviewBatch,
    created_at: i64,
) -> Result<StateSchemaScopeSidecar, StoreError>
where
    S: PhoenixStateSchemaPatchStore,
{
    let updates = build_state_schema_patch_sidecar(batch, created_at);
    let merged = match store.load_state_schema_patch_sidecar(&batch.scope)? {
        Some(existing) => merge_state_schema_patch_sidecars(existing, updates),
        None => updates,
    };
    store.persist_state_schema_patch_sidecar(&merged)?;
    Ok(merged)
}

pub fn apply_state_schema_patch_sidecar(
    batch: &mut StateSchemaScopeReviewBatch,
    sidecar: &StateSchemaScopeSidecar,
) {
    batch.slot_families = sidecar.slot_families.clone();
    batch.slot_definitions = sidecar.slot_definitions.clone();
    batch.slot_candidates = sidecar.slot_candidates.clone();
    batch.promotion_decisions = sidecar.promotion_decisions.clone();
    batch.write_proposals = sidecar.write_proposals.clone();
    batch.state_schema_generation = Some(sidecar.generation);
    batch.summary = sidecar.summary.clone();
    batch.diagnostics = sidecar.diagnostics.clone();
}

fn merge_state_schema_patch_sidecars(
    mut existing: StateSchemaScopeSidecar,
    updates: StateSchemaScopeSidecar,
) -> StateSchemaScopeSidecar {
    existing.updated_at = existing.updated_at.max(updates.updated_at);
    existing.generation = existing.generation.max(updates.generation);
    existing.slot_families = updates.slot_families;
    existing.slot_definitions = updates.slot_definitions;
    existing.slot_candidates = updates.slot_candidates;
    merge_promotion_history(
        &mut existing.promotion_decisions,
        &updates.promotion_decisions,
    );
    existing.write_proposals = updates.write_proposals;
    existing.summary = updates.summary;
    existing.diagnostics = updates.diagnostics;
    existing
}

fn merge_promotion_history(
    existing: &mut Vec<StateSlotPromotionDecisionRecord>,
    updates: &[StateSlotPromotionDecisionRecord],
) {
    existing.extend_from_slice(updates);
    existing.sort_by(|left, right| left.decision_id.0.cmp(&right.decision_id.0));
    existing.dedup_by(|left, right| left.decision_id == right.decision_id);
}

fn build_summary(batch: &StateSchemaScopeReviewBatch) -> StateSchemaCompilerSummary {
    let mut family_counts = BTreeMap::<String, usize>::new();
    let mut lifecycle_counts = BTreeMap::<String, usize>::new();
    let mut owner_type_counts = BTreeMap::<String, usize>::new();
    for definition in &batch.slot_definitions {
        *family_counts
            .entry(
                definition
                    .family_id
                    .0
                    .strip_prefix("family:")
                    .unwrap_or("unknown")
                    .to_owned(),
            )
            .or_default() += 1;
        *lifecycle_counts
            .entry(format!("{:?}", definition.lifecycle).to_lowercase())
            .or_default() += 1;
        *owner_type_counts
            .entry(format!("{:?}", definition.owner_type).to_lowercase())
            .or_default() += 1;
    }
    StateSchemaCompilerSummary {
        family_count: batch.slot_families.len(),
        definition_count: batch.slot_definitions.len(),
        active_definition_count: batch
            .slot_definitions
            .iter()
            .filter(|definition| definition.lifecycle == StateSlotLifecycle::Active)
            .count(),
        stable_definition_count: batch
            .slot_definitions
            .iter()
            .filter(|definition| definition.lifecycle == StateSlotLifecycle::Stable)
            .count(),
        candidate_definition_count: batch
            .slot_definitions
            .iter()
            .filter(|definition| definition.lifecycle == StateSlotLifecycle::Candidate)
            .count(),
        candidate_count: batch.slot_candidates.len(),
        promotion_decision_count: batch.promotion_decisions.len(),
        write_proposal_count: batch.write_proposals.len(),
        family_counts,
        lifecycle_counts,
        owner_type_counts,
    }
}

fn next_generation(current: Option<u64>) -> u64 {
    current.unwrap_or_default() + 1
}
