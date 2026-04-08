use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    scope_storage_key, CausalChainRecord, CausalClaimAtom, CausalCompilerSummary,
    CausalDecisionRecord, CausalEdgeAddition, CausalEdgeAliasRecord, CausalInvalidationRecord,
    CausalMemoryCard, CausalMetricsSnapshot, CausalReviewQueueItem, CausalScopeSidecar,
    CounterfactualReviewRecord, DirtyScopeRecord, DocumentArchive, DocumentRevisionRef,
    ErScopePatchSidecar, ScopeOrd, SessionArchive,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixCausalPatchStore, PhoenixErPatchStore, StoreError,
};
use phoenix_types::{ScopeKey, SessionId};
use serde::{Deserialize, Serialize};

use crate::{
    build_causal_memory_cards, build_chain_records, build_counterfactual_reviews,
    draft_causal_decisions, normalize_causal_inputs, CausalDecision, CausalEventProfile,
    CausalReviewCase,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalScopeReviewBatch {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub dirty: Option<DirtyScopeRecord>,
    #[serde(default)]
    pub document_refs: Vec<DocumentRevisionRef>,
    #[serde(default)]
    pub event_profiles: Vec<CausalEventProfile>,
    #[serde(default)]
    pub review_cases: Vec<CausalReviewCase>,
    #[serde(default)]
    pub claim_atoms: Vec<CausalClaimAtom>,
    #[serde(default)]
    pub decisions: Vec<CausalDecision>,
    #[serde(default)]
    pub edge_records: Vec<CausalEdgeAddition>,
    #[serde(default)]
    pub edge_additions: Vec<CausalEdgeAddition>,
    #[serde(default)]
    pub decision_records: Vec<CausalDecisionRecord>,
    #[serde(default)]
    pub decision_history: Vec<CausalDecisionRecord>,
    #[serde(default)]
    pub invalidations: Vec<CausalInvalidationRecord>,
    #[serde(default)]
    pub edge_aliases: Vec<CausalEdgeAliasRecord>,
    #[serde(default)]
    pub review_queue: Vec<CausalReviewQueueItem>,
    #[serde(default)]
    pub chains: Vec<CausalChainRecord>,
    #[serde(default)]
    pub counterfactual_reviews: Vec<CounterfactualReviewRecord>,
    #[serde(default)]
    pub memory_cards: Vec<CausalMemoryCard>,
    pub metrics_snapshot: CausalMetricsSnapshot,
    pub er_generation: Option<u64>,
    pub causal_generation: Option<u64>,
    #[serde(default)]
    pub summary: CausalCompilerSummary,
    #[serde(default)]
    pub diagnostics: BTreeMap<String, usize>,
}

pub fn derive_scope_review_batch(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    er_sidecar: Option<&ErScopePatchSidecar>,
) -> CausalScopeReviewBatch {
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
    let normalized = normalize_causal_inputs(archives, er_sidecar);

    CausalScopeReviewBatch {
        scope,
        scope_key,
        scope_ord,
        session_id,
        dirty: dirty.cloned(),
        document_refs,
        event_profiles: normalized.event_profiles,
        review_cases: normalized.review_cases,
        claim_atoms: normalized.claim_atoms,
        decisions: Vec::new(),
        edge_records: Vec::new(),
        edge_additions: Vec::new(),
        decision_records: Vec::new(),
        decision_history: Vec::new(),
        invalidations: Vec::new(),
        edge_aliases: Vec::new(),
        review_queue: Vec::new(),
        chains: Vec::new(),
        counterfactual_reviews: Vec::new(),
        memory_cards: Vec::new(),
        metrics_snapshot: CausalMetricsSnapshot::default(),
        er_generation: er_sidecar.map(|value| value.generation),
        causal_generation: None,
        summary: CausalCompilerSummary::default(),
        diagnostics: normalized.diagnostics,
    }
}

pub fn derive_scope_review_batch_from_store<S>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
) -> Result<CausalScopeReviewBatch, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixCausalPatchStore,
{
    let archives = store.load_latest_document_archives(Some(&dirty.scope))?;
    let er_sidecar = store.load_er_patch_sidecar(&dirty.scope)?;
    let mut batch = derive_scope_review_batch(&archives, session, Some(dirty), er_sidecar.as_ref());
    if let Some(causal_sidecar) = store.load_causal_patch_sidecar(&dirty.scope)? {
        apply_causal_patch_sidecar(&mut batch, &causal_sidecar);
    }
    Ok(batch)
}

pub fn derive_dirty_scope_review_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<CausalScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixCausalPatchStore,
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

pub fn run_causal_scope(batch: &mut CausalScopeReviewBatch, created_at: i64) {
    let drafts = draft_causal_decisions(&batch.review_cases, &batch.claim_atoms, created_at);
    let decisions = drafts.decisions.clone();
    let edge_records = drafts.edge_records.clone();
    let committed_edges = committed_edge_additions(&edge_records);
    let reviewable_edges = reviewable_edge_records(&edge_records);
    let graph_stats = crate::graph::build_graph_stats(&committed_edges);
    let chains = build_chain_records(&committed_edges, created_at);
    let counterfactual_reviews =
        build_counterfactual_reviews(&reviewable_edges, &chains, created_at);
    let memory_cards = build_causal_memory_cards(
        &batch.event_profiles,
        &edge_records,
        &committed_edges,
        &chains,
        &counterfactual_reviews,
    );

    let mut kind_counts = BTreeMap::<String, usize>::new();
    for edge in &edge_records {
        *kind_counts.entry(format!("{:?}", edge.kind).to_lowercase()).or_default() += 1;
    }
    if !graph_stats.incoming_by_target.is_empty() {
        batch.diagnostics.insert(
            "active_causal_targets".to_owned(),
            graph_stats.incoming_by_target.len(),
        );
    }

    batch.decisions = decisions;
    batch.edge_records = edge_records.clone();
    batch.edge_additions = committed_edges.clone();
    batch.decision_records = drafts.decision_records.clone();
    batch.decision_history = drafts.decision_records.clone();
    batch.invalidations = drafts.invalidations.clone();
    batch.edge_aliases = drafts.edge_aliases.clone();
    batch.review_queue = drafts.review_queue.clone();
    batch.chains = chains;
    batch.counterfactual_reviews = counterfactual_reviews;
    batch.memory_cards = memory_cards;
    batch.metrics_snapshot = drafts.metrics_snapshot.clone();
    batch.summary = CausalCompilerSummary {
        claim_atom_count: batch.claim_atoms.len(),
        review_case_count: batch.review_cases.len(),
        edge_record_count: batch.edge_records.len(),
        committed_edge_count: batch.edge_additions.len(),
        accepted_edge_count: batch.edge_additions.len(),
        supported_edge_count: count_edges_with_status(&batch.edge_additions, phoenix_semantic_v2::CausalClaimStatus::Supported),
        deferred_edge_count: count_edges_with_status(&batch.edge_records, phoenix_semantic_v2::CausalClaimStatus::Deferred),
        rejected_edge_count: count_edges_with_status(&batch.edge_records, phoenix_semantic_v2::CausalClaimStatus::Rejected),
        contradicted_edge_count: count_edges_with_status(&batch.edge_records, phoenix_semantic_v2::CausalClaimStatus::Contradicted),
        chain_count: batch.chains.len(),
        counterfactual_review_count: batch.counterfactual_reviews.len(),
        memory_card_count: batch.memory_cards.len(),
        invalidation_count: batch.invalidations.len(),
        review_queue_count: batch.review_queue.len(),
        kind_counts,
        outcome_counts: drafts.outcome_counts.clone(),
    };
}

pub fn build_causal_patch_sidecar(
    batch: &CausalScopeReviewBatch,
    created_at: i64,
) -> CausalScopeSidecar {
    CausalScopeSidecar {
        scope: batch.scope.clone(),
        scope_key: batch.scope_key.clone(),
        scope_ord: Some(batch.scope_ord),
        session_id: batch.session_id.clone(),
        updated_at: created_at,
        generation: created_at as u64,
        claim_atoms: batch.claim_atoms.clone(),
        edge_records: batch.edge_records.clone(),
        edge_additions: batch.edge_additions.clone(),
        chains: batch.chains.clone(),
        counterfactual_reviews: batch.counterfactual_reviews.clone(),
        decisions: batch.decision_records.clone(),
        decision_history: batch.decision_history.clone(),
        invalidations: batch.invalidations.clone(),
        edge_aliases: batch.edge_aliases.clone(),
        review_queue: batch.review_queue.clone(),
        memory_cards: batch.memory_cards.clone(),
        metrics_snapshot: batch.metrics_snapshot.clone(),
        summary: batch.summary.clone(),
    }
}

pub fn persist_causal_patch_sidecar<S>(
    store: &S,
    batch: &CausalScopeReviewBatch,
    created_at: i64,
) -> Result<CausalScopeSidecar, StoreError>
where
    S: PhoenixCausalPatchStore,
{
    let updates = build_causal_patch_sidecar(batch, created_at);
    let merged = match store.load_causal_patch_sidecar(&batch.scope)? {
        Some(existing) => merge_causal_patch_sidecars(existing, updates),
        None => updates,
    };
    store.persist_causal_patch_sidecar(&merged)?;
    Ok(merged)
}

pub fn apply_causal_patch_sidecar(batch: &mut CausalScopeReviewBatch, sidecar: &CausalScopeSidecar) {
    batch.claim_atoms = sidecar.claim_atoms.clone();
    batch.edge_records = sidecar.edge_records.clone();
    batch.edge_additions = sidecar.edge_additions.clone();
    batch.chains = sidecar.chains.clone();
    batch.counterfactual_reviews = sidecar.counterfactual_reviews.clone();
    batch.decision_records = sidecar.decisions.clone();
    batch.decision_history = sidecar.decision_history.clone();
    batch.invalidations = sidecar.invalidations.clone();
    batch.edge_aliases = sidecar.edge_aliases.clone();
    batch.review_queue = sidecar.review_queue.clone();
    batch.memory_cards = sidecar.memory_cards.clone();
    batch.metrics_snapshot = sidecar.metrics_snapshot.clone();
    batch.causal_generation = Some(sidecar.generation);
    batch.summary = sidecar.summary.clone();
}

fn merge_causal_patch_sidecars(
    mut existing: CausalScopeSidecar,
    mut updates: CausalScopeSidecar,
) -> CausalScopeSidecar {
    annotate_supersedes(&existing.decision_history, &mut updates.decisions);
    annotate_supersedes(&existing.decision_history, &mut updates.decision_history);
    existing.updated_at = existing.updated_at.max(updates.updated_at);
    existing.generation = existing.generation.max(updates.generation);
    existing.claim_atoms = updates.claim_atoms;
    existing.edge_records = updates.edge_records;
    existing.edge_additions = updates.edge_additions;
    existing.chains = updates.chains;
    existing.counterfactual_reviews = updates.counterfactual_reviews;
    existing.decisions = updates.decisions;
    merge_history(&mut existing.decision_history, &updates.decision_history);
    existing.invalidations = updates.invalidations;
    existing.edge_aliases = updates.edge_aliases;
    existing.review_queue = updates.review_queue;
    existing.memory_cards = updates.memory_cards;
    existing.metrics_snapshot = updates.metrics_snapshot;
    existing.summary = updates.summary;
    existing
}

fn committed_edge_additions(edges: &[CausalEdgeAddition]) -> Vec<CausalEdgeAddition> {
    edges
        .iter()
        .filter(|edge| {
            matches!(
                edge.status,
                phoenix_semantic_v2::CausalClaimStatus::Active
                    | phoenix_semantic_v2::CausalClaimStatus::Supported
            )
        })
        .cloned()
        .collect()
}

fn reviewable_edge_records(edges: &[CausalEdgeAddition]) -> Vec<CausalEdgeAddition> {
    edges
        .iter()
        .filter(|edge| {
            !matches!(
                edge.status,
                phoenix_semantic_v2::CausalClaimStatus::Rejected
                    | phoenix_semantic_v2::CausalClaimStatus::Invalidated
            )
        })
        .cloned()
        .collect()
}

fn count_edges_with_status(
    edges: &[CausalEdgeAddition],
    status: phoenix_semantic_v2::CausalClaimStatus,
) -> usize {
    edges.iter().filter(|edge| edge.status == status).count()
}

fn merge_history(existing: &mut Vec<CausalDecisionRecord>, updates: &[CausalDecisionRecord]) {
    existing.extend_from_slice(updates);
    existing.sort_by(|left, right| left.decision_id.0.cmp(&right.decision_id.0));
    existing.dedup_by(|left, right| left.decision_id == right.decision_id);
}

fn annotate_supersedes(
    history: &[CausalDecisionRecord],
    updates: &mut [CausalDecisionRecord],
) {
    let mut latest_by_edge = BTreeMap::<String, phoenix_semantic_v2::CausalDecisionId>::new();
    for record in history {
        latest_by_edge.insert(record.edge_id.0.clone(), record.decision_id.clone());
    }
    for record in updates {
        if record.supersedes.is_none() {
            record.supersedes = latest_by_edge.get(&record.edge_id.0).cloned();
        }
        latest_by_edge.insert(record.edge_id.0.clone(), record.decision_id.clone());
    }
}
