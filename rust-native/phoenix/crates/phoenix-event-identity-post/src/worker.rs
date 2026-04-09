use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    scope_storage_key, CanonicalEventCard, CanonicalEventRecord, DirtyScopeRecord, DocumentArchive,
    DocumentRevisionRef, ErScopePatchSidecar, EventIdentityCompilerSummary,
    EventIdentityHypothesis, EventIdentityInvalidationRecord, EventIdentityLedgerRecord,
    EventIdentityMembershipRecord, EventIdentityScopeSidecar, EventIdentitySplitRecord,
    EventMentionPacket, ScopeOrd, SessionArchive,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixEventIdentityPatchStore, StoreError,
};
use phoenix_types::{ScopeKey, SessionId};
use serde::{Deserialize, Serialize};

use crate::{
    build_canonical_event_cards, build_identity_hypotheses, normalize_event_identity_inputs,
    resolve_canonical_events,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityScopeReviewBatch {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub dirty: Option<DirtyScopeRecord>,
    #[serde(default)]
    pub document_refs: Vec<DocumentRevisionRef>,
    #[serde(default)]
    pub mention_packets: Vec<EventMentionPacket>,
    #[serde(default)]
    pub identity_hypotheses: Vec<EventIdentityHypothesis>,
    #[serde(default)]
    pub canonical_events: Vec<CanonicalEventRecord>,
    #[serde(default)]
    pub memberships: Vec<EventIdentityMembershipRecord>,
    #[serde(default)]
    pub decisions: Vec<EventIdentityLedgerRecord>,
    #[serde(default)]
    pub decision_history: Vec<EventIdentityLedgerRecord>,
    #[serde(default)]
    pub invalidations: Vec<EventIdentityInvalidationRecord>,
    #[serde(default)]
    pub splits: Vec<EventIdentitySplitRecord>,
    #[serde(default)]
    pub canonical_event_cards: Vec<CanonicalEventCard>,
    pub er_generation: Option<u64>,
    pub event_identity_generation: Option<u64>,
    pub summary: EventIdentityCompilerSummary,
    #[serde(default)]
    pub diagnostics: BTreeMap<String, usize>,
}

pub fn derive_scope_review_batch(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    er_sidecar: Option<&ErScopePatchSidecar>,
) -> EventIdentityScopeReviewBatch {
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
    let normalized = normalize_event_identity_inputs(archives, er_sidecar);

    EventIdentityScopeReviewBatch {
        scope,
        scope_key,
        scope_ord,
        session_id,
        dirty: dirty.cloned(),
        document_refs,
        mention_packets: normalized.mention_packets,
        identity_hypotheses: Vec::new(),
        canonical_events: Vec::new(),
        memberships: Vec::new(),
        decisions: Vec::new(),
        decision_history: Vec::new(),
        invalidations: Vec::new(),
        splits: Vec::new(),
        canonical_event_cards: Vec::new(),
        er_generation: er_sidecar.map(|value| value.generation),
        event_identity_generation: None,
        summary: EventIdentityCompilerSummary::default(),
        diagnostics: normalized.diagnostics,
    }
}

pub fn derive_scope_review_batch_from_store<S>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
) -> Result<EventIdentityScopeReviewBatch, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixEventIdentityPatchStore,
{
    let archives = store.load_latest_document_archives(Some(&dirty.scope))?;
    let er_sidecar = store.load_er_patch_sidecar(&dirty.scope)?;
    let mut batch = derive_scope_review_batch(&archives, session, Some(dirty), er_sidecar.as_ref());
    if let Some(sidecar) = store.load_event_identity_patch_sidecar(&dirty.scope)? {
        apply_event_identity_patch_sidecar(&mut batch, &sidecar);
    }
    Ok(batch)
}

pub fn derive_dirty_scope_review_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<EventIdentityScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixEventIdentityPatchStore,
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

pub fn run_event_identity_scope(batch: &mut EventIdentityScopeReviewBatch, created_at: i64) {
    let (identity_hypotheses, graph_stats, graph_diagnostics) =
        build_identity_hypotheses(&batch.scope_key, &batch.mention_packets);
    let resolved = resolve_canonical_events(
        &batch.scope_key,
        &batch.mention_packets,
        &identity_hypotheses,
        created_at,
    );
    let canonical_event_cards = build_canonical_event_cards(
        &resolved.canonical_events,
        &batch.mention_packets,
        &identity_hypotheses,
    );

    merge_counts(&mut batch.diagnostics, graph_diagnostics);
    merge_counts(&mut batch.diagnostics, resolved.diagnostics);
    for (relation, count) in &graph_stats.by_relation {
        batch
            .diagnostics
            .insert(format!("relation_count:{relation}"), *count);
    }

    batch.identity_hypotheses = identity_hypotheses;
    batch.canonical_events = resolved.canonical_events;
    batch.memberships = resolved.memberships;
    batch.decisions = resolved.decisions.clone();
    batch.decision_history = resolved.decision_history;
    batch.invalidations = resolved.invalidations;
    batch.splits = resolved.splits;
    batch.canonical_event_cards = canonical_event_cards;
    batch.summary = build_summary(batch);
}

pub fn build_event_identity_patch_sidecar(
    batch: &EventIdentityScopeReviewBatch,
    created_at: i64,
) -> EventIdentityScopeSidecar {
    EventIdentityScopeSidecar {
        scope: batch.scope.clone(),
        scope_key: batch.scope_key.clone(),
        scope_ord: Some(batch.scope_ord),
        session_id: batch.session_id.clone(),
        updated_at: created_at,
        generation: next_generation(batch.event_identity_generation),
        mention_packets: batch.mention_packets.clone(),
        identity_hypotheses: batch.identity_hypotheses.clone(),
        canonical_events: batch.canonical_events.clone(),
        memberships: batch.memberships.clone(),
        decisions: batch.decisions.clone(),
        decision_history: batch.decision_history.clone(),
        invalidations: batch.invalidations.clone(),
        splits: batch.splits.clone(),
        canonical_event_cards: batch.canonical_event_cards.clone(),
        summary: batch.summary.clone(),
    }
}

pub fn persist_event_identity_patch_sidecar<S>(
    store: &S,
    batch: &EventIdentityScopeReviewBatch,
    created_at: i64,
) -> Result<EventIdentityScopeSidecar, StoreError>
where
    S: PhoenixEventIdentityPatchStore,
{
    let updates = build_event_identity_patch_sidecar(batch, created_at);
    let merged = match store.load_event_identity_patch_sidecar(&batch.scope)? {
        Some(existing) => merge_event_identity_patch_sidecars(existing, updates),
        None => updates,
    };
    store.persist_event_identity_patch_sidecar(&merged)?;
    Ok(merged)
}

pub fn apply_event_identity_patch_sidecar(
    batch: &mut EventIdentityScopeReviewBatch,
    sidecar: &EventIdentityScopeSidecar,
) {
    batch.mention_packets = sidecar.mention_packets.clone();
    batch.identity_hypotheses = sidecar.identity_hypotheses.clone();
    batch.canonical_events = sidecar.canonical_events.clone();
    batch.memberships = sidecar.memberships.clone();
    batch.decisions = sidecar.decisions.clone();
    batch.decision_history = sidecar.decision_history.clone();
    batch.invalidations = sidecar.invalidations.clone();
    batch.splits = sidecar.splits.clone();
    batch.canonical_event_cards = sidecar.canonical_event_cards.clone();
    batch.event_identity_generation = Some(sidecar.generation);
    batch.summary = sidecar.summary.clone();
}

fn merge_event_identity_patch_sidecars(
    mut existing: EventIdentityScopeSidecar,
    updates: EventIdentityScopeSidecar,
) -> EventIdentityScopeSidecar {
    existing.updated_at = existing.updated_at.max(updates.updated_at);
    existing.generation = existing.generation.max(updates.generation);
    existing.mention_packets = updates.mention_packets;
    existing.identity_hypotheses = updates.identity_hypotheses;
    existing.canonical_events = updates.canonical_events;
    existing.memberships = updates.memberships;
    existing.decisions = updates.decisions;
    existing.invalidations = updates.invalidations;
    existing.splits = updates.splits;
    existing.canonical_event_cards = updates.canonical_event_cards;
    merge_history(&mut existing.decision_history, &updates.decision_history);
    existing.summary = updates.summary;
    existing
}

fn merge_history(
    existing: &mut Vec<EventIdentityLedgerRecord>,
    updates: &[EventIdentityLedgerRecord],
) {
    existing.extend_from_slice(updates);
    existing.sort_by(|left, right| left.decision_id.0.cmp(&right.decision_id.0));
    existing.dedup_by(|left, right| left.decision_id == right.decision_id);
}

fn build_summary(batch: &EventIdentityScopeReviewBatch) -> EventIdentityCompilerSummary {
    let mut relation_counts = BTreeMap::<String, usize>::new();
    for hypothesis in &batch.identity_hypotheses {
        *relation_counts
            .entry(relation_key(hypothesis.relation).to_owned())
            .or_default() += 1;
    }
    EventIdentityCompilerSummary {
        mention_packet_count: batch.mention_packets.len(),
        hypothesis_count: batch.identity_hypotheses.len(),
        canonical_event_count: batch.canonical_events.len(),
        membership_count: batch.memberships.len(),
        decision_count: batch.decisions.len(),
        invalidation_count: batch.invalidations.len(),
        split_count: batch.splits.len(),
        card_count: batch.canonical_event_cards.len(),
        relation_counts,
    }
}

fn relation_key(relation: phoenix_semantic_v2::EventIdentityState) -> &'static str {
    match relation {
        phoenix_semantic_v2::EventIdentityState::FullIdentity => "full_identity",
        phoenix_semantic_v2::EventIdentityState::QuasiIdentity => "quasi_identity",
        phoenix_semantic_v2::EventIdentityState::MemberOfCollection => "member_of_collection",
        phoenix_semantic_v2::EventIdentityState::SubeventOf => "subevent_of",
        phoenix_semantic_v2::EventIdentityState::VersionOf => "version_of",
        phoenix_semantic_v2::EventIdentityState::ReportsOn => "reports_on",
        phoenix_semantic_v2::EventIdentityState::Incompatible => "incompatible",
    }
}

fn merge_counts(target: &mut BTreeMap<String, usize>, updates: BTreeMap<String, usize>) {
    for (key, value) in updates {
        *target.entry(key).or_default() += value;
    }
}

fn next_generation(current: Option<u64>) -> u64 {
    current.unwrap_or_default() + 1
}
