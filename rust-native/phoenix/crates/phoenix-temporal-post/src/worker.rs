use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    scope_storage_key, CanonicalEventId, DirtyScopeRecord, DocumentArchive, DocumentRevisionRef,
    EventIdentityScopeSidecar, ScopeOrd, SessionArchive, TemporalAnchorRecord, TemporalAxisRecord,
    TemporalClaimAtom, TemporalCompilerSummary, TemporalConflictRecord, TemporalConstraintRecord,
    TemporalGapRecord, TemporalIntervalRecord, TemporalMemoryCard, TemporalReferenceEdge,
    TemporalScopeSidecar, TemporalTimexRecord, TimelineSegmentRecord,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixEventIdentityPatchStore, PhoenixTemporalPatchStore, StoreError,
};
use phoenix_types::{ScopeKey, SessionId};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    build_temporal_memory_cards, normalize_temporal_inputs, solve_temporal_inputs,
    SolvedTemporalBatch, TemporalEventProfile, TemporalReviewCase, TemporalTimexProfile,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalScopeReviewBatch {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub dirty: Option<DirtyScopeRecord>,
    #[serde(default)]
    pub document_refs: Vec<DocumentRevisionRef>,
    #[serde(default)]
    pub axes: Vec<TemporalAxisRecord>,
    #[serde(default)]
    pub event_profiles: Vec<TemporalEventProfile>,
    #[serde(default)]
    pub timex_profiles: Vec<TemporalTimexProfile>,
    #[serde(default)]
    pub review_cases: Vec<TemporalReviewCase>,
    #[serde(default)]
    pub claim_atoms: Vec<TemporalClaimAtom>,
    #[serde(default)]
    pub anchors: Vec<TemporalAnchorRecord>,
    #[serde(default)]
    pub reference_edges: Vec<TemporalReferenceEdge>,
    #[serde(default)]
    pub constraints: Vec<TemporalConstraintRecord>,
    #[serde(default)]
    pub intervals: Vec<TemporalIntervalRecord>,
    #[serde(default)]
    pub timeline_segments: Vec<TimelineSegmentRecord>,
    #[serde(default)]
    pub conflicts: Vec<TemporalConflictRecord>,
    #[serde(default)]
    pub gaps: Vec<TemporalGapRecord>,
    #[serde(default)]
    pub memory_cards: Vec<TemporalMemoryCard>,
    pub temporal_generation: Option<u64>,
    pub summary: TemporalCompilerSummary,
    #[serde(default)]
    pub diagnostics: BTreeMap<String, usize>,
}

pub fn derive_scope_review_batch(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
) -> TemporalScopeReviewBatch {
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
    let normalized = normalize_temporal_inputs(archives);

    TemporalScopeReviewBatch {
        scope,
        scope_key,
        scope_ord,
        session_id,
        dirty: dirty.cloned(),
        document_refs,
        axes: normalized.axes,
        event_profiles: normalized.event_profiles,
        timex_profiles: normalized.timex_profiles,
        review_cases: normalized.review_cases,
        claim_atoms: normalized.claim_atoms,
        anchors: normalized.anchors,
        reference_edges: normalized.reference_edges,
        constraints: normalized.constraints,
        intervals: Vec::new(),
        timeline_segments: Vec::new(),
        conflicts: Vec::new(),
        gaps: Vec::new(),
        memory_cards: Vec::new(),
        temporal_generation: None,
        summary: TemporalCompilerSummary::default(),
        diagnostics: normalized.diagnostics,
    }
}

pub fn derive_scope_review_batch_from_store<S>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
) -> Result<TemporalScopeReviewBatch, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixTemporalPatchStore + PhoenixEventIdentityPatchStore,
{
    let archives = store.load_latest_document_archives(Some(&dirty.scope))?;
    let mut batch = derive_scope_review_batch(&archives, session, Some(dirty));
    let event_identity_sidecar = store.load_event_identity_patch_sidecar(&dirty.scope)?;
    if let Some(sidecar) = event_identity_sidecar.as_ref() {
        annotate_temporal_batch_with_event_identity(&mut batch, sidecar);
    }
    if let Some(sidecar) = store.load_temporal_patch_sidecar(&dirty.scope)? {
        apply_temporal_patch_sidecar(&mut batch, &sidecar);
        if let Some(event_identity_sidecar) = event_identity_sidecar.as_ref() {
            annotate_temporal_batch_with_event_identity(&mut batch, event_identity_sidecar);
        }
    }
    Ok(batch)
}

pub fn derive_dirty_scope_review_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<TemporalScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixTemporalPatchStore + PhoenixEventIdentityPatchStore,
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

pub fn run_temporal_scope(batch: &mut TemporalScopeReviewBatch, created_at: i64) {
    let inputs = crate::TemporalNormalizedInputs {
        axes: batch.axes.clone(),
        event_profiles: batch.event_profiles.clone(),
        timex_profiles: batch.timex_profiles.clone(),
        review_cases: batch.review_cases.clone(),
        claim_atoms: batch.claim_atoms.clone(),
        anchors: batch.anchors.clone(),
        reference_edges: batch.reference_edges.clone(),
        constraints: batch.constraints.clone(),
        diagnostics: batch.diagnostics.clone(),
    };
    let solved = solve_temporal_inputs(&inputs, created_at);
    let memory_cards = build_temporal_memory_cards(
        &batch.event_profiles,
        &solved.intervals,
        &solved.conflicts,
        &solved.gaps,
        &solved.graph_stats,
    );

    batch.intervals = solved.intervals.clone();
    batch.timeline_segments = solved.timeline_segments.clone();
    batch.conflicts = solved.conflicts.clone();
    batch.gaps = solved.gaps.clone();
    batch.memory_cards = memory_cards;
    batch.diagnostics = solved.diagnostics.clone();
    batch.summary = build_summary(batch, &solved);
}

pub fn build_temporal_patch_sidecar(
    batch: &TemporalScopeReviewBatch,
    created_at: i64,
) -> TemporalScopeSidecar {
    TemporalScopeSidecar {
        scope: batch.scope.clone(),
        scope_key: batch.scope_key.clone(),
        scope_ord: Some(batch.scope_ord),
        session_id: batch.session_id.clone(),
        updated_at: created_at,
        generation: next_generation(batch.temporal_generation),
        timex_records: batch
            .timex_profiles
            .iter()
            .map(|profile| TemporalTimexRecord {
                timex_id: phoenix_semantic_v2::TemporalTimexId(profile.timex_id.clone()),
                document_id: profile.document_id.clone(),
                proposition_id: profile.proposition_id.clone(),
                sentence_index: profile.sentence_index,
                label: profile.label.clone(),
                normalized_value: profile.normalized_value.clone(),
                range: None,
                axis_id: profile.axis_id.clone(),
                temporal: profile.temporal.clone(),
                confidence_millis: 0,
                source_class: profile.source_class.clone(),
                evidence_refs: Vec::new(),
            })
            .collect(),
        anchors: batch.anchors.clone(),
        axes: batch.axes.clone(),
        reference_edges: batch.reference_edges.clone(),
        claim_atoms: batch.claim_atoms.clone(),
        constraints: batch.constraints.clone(),
        intervals: batch.intervals.clone(),
        timeline_segments: batch.timeline_segments.clone(),
        conflicts: batch.conflicts.clone(),
        gaps: batch.gaps.clone(),
        memory_cards: batch.memory_cards.clone(),
        summary: batch.summary.clone(),
    }
}

pub fn persist_temporal_patch_sidecar<S>(
    store: &S,
    batch: &TemporalScopeReviewBatch,
    created_at: i64,
) -> Result<TemporalScopeSidecar, StoreError>
where
    S: PhoenixTemporalPatchStore,
{
    let sidecar = build_temporal_patch_sidecar(batch, created_at);
    store.persist_temporal_patch_sidecar(&sidecar)?;
    Ok(sidecar)
}

pub fn apply_temporal_patch_sidecar(
    batch: &mut TemporalScopeReviewBatch,
    sidecar: &TemporalScopeSidecar,
) {
    batch.temporal_generation = Some(sidecar.generation);
    batch.axes = sidecar.axes.clone();
    batch.reference_edges = sidecar.reference_edges.clone();
    batch.claim_atoms = sidecar.claim_atoms.clone();
    batch.constraints = sidecar.constraints.clone();
    batch.intervals = sidecar.intervals.clone();
    batch.timeline_segments = sidecar.timeline_segments.clone();
    batch.conflicts = sidecar.conflicts.clone();
    batch.gaps = sidecar.gaps.clone();
    batch.memory_cards = sidecar.memory_cards.clone();
    batch.summary = sidecar.summary.clone();
}

pub(crate) fn annotate_temporal_batch_with_event_identity(
    batch: &mut TemporalScopeReviewBatch,
    sidecar: &EventIdentityScopeSidecar,
) {
    let canonical_by_event = canonical_event_ids_by_event(sidecar);

    for profile in &mut batch.event_profiles {
        profile.canonical_event_id = canonical_for(
            &canonical_by_event,
            profile.document_id.as_str(),
            profile.event_id.as_str(),
        );
    }

    for case in &mut batch.review_cases {
        case.canonical_event_id = canonical_for(
            &canonical_by_event,
            case.document_id.as_str(),
            case.event_id.as_str(),
        );
    }

    for anchor in &mut batch.anchors {
        anchor.canonical_event_id = anchor.event_id.as_deref().and_then(|event_id| {
            canonical_for(&canonical_by_event, anchor.document_id.as_str(), event_id)
        });
        anchor.canonical_reference_event_id =
            anchor.reference_event_id.as_deref().and_then(|event_id| {
                canonical_for(&canonical_by_event, anchor.document_id.as_str(), event_id)
            });
    }

    for edge in &mut batch.reference_edges {
        edge.canonical_source_event_id = canonical_for(
            &canonical_by_event,
            edge.document_id.as_str(),
            edge.source_event_id.as_str(),
        );
        edge.canonical_target_event_id = edge.target_event_id.as_deref().and_then(|event_id| {
            canonical_for(&canonical_by_event, edge.document_id.as_str(), event_id)
        });
    }

    for claim in &mut batch.claim_atoms {
        claim.canonical_event_id = claim.event_id.as_deref().and_then(|event_id| {
            canonical_for(&canonical_by_event, claim.document_id.as_str(), event_id)
        });
    }

    for constraint in &mut batch.constraints {
        constraint.canonical_source_event_id =
            constraint.source_event_id.as_deref().and_then(|event_id| {
                canonical_for(
                    &canonical_by_event,
                    constraint.document_id.as_str(),
                    event_id,
                )
            });
        constraint.canonical_target_event_id =
            constraint.target_event_id.as_deref().and_then(|event_id| {
                canonical_for(
                    &canonical_by_event,
                    constraint.document_id.as_str(),
                    event_id,
                )
            });
    }

    for interval in &mut batch.intervals {
        interval.canonical_event_id = canonical_for(
            &canonical_by_event,
            interval.document_id.as_str(),
            interval.event_id.as_str(),
        );
    }

    for conflict in &mut batch.conflicts {
        conflict.canonical_event_id = conflict.event_id.as_deref().and_then(|event_id| {
            canonical_for(&canonical_by_event, conflict.document_id.as_str(), event_id)
        });
    }

    for gap in &mut batch.gaps {
        gap.canonical_event_id = gap.event_id.as_deref().and_then(|event_id| {
            canonical_for(&canonical_by_event, gap.document_id.as_str(), event_id)
        });
    }

    for segment in &mut batch.timeline_segments {
        segment.canonical_event_ids = segment
            .event_ids
            .iter()
            .filter_map(|event_id| {
                canonical_for(&canonical_by_event, segment.document_id.as_str(), event_id)
            })
            .collect();
        segment.indeterminate_canonical_event_ids = segment
            .indeterminate_event_ids
            .iter()
            .filter_map(|event_id| {
                canonical_for(&canonical_by_event, segment.document_id.as_str(), event_id)
            })
            .collect();
    }

    for card in &mut batch.memory_cards {
        card.canonical_event_id = canonical_for(
            &canonical_by_event,
            card.document_id.as_str(),
            card.event_id.as_str(),
        );
        card.before_canonical_event_ids = card
            .before_event_ids
            .iter()
            .filter_map(|event_id| {
                canonical_for(&canonical_by_event, card.document_id.as_str(), event_id)
            })
            .collect();
        card.after_canonical_event_ids = card
            .after_event_ids
            .iter()
            .filter_map(|event_id| {
                canonical_for(&canonical_by_event, card.document_id.as_str(), event_id)
            })
            .collect();
    }
}

fn canonical_event_ids_by_event(
    sidecar: &EventIdentityScopeSidecar,
) -> FxHashMap<(String, String), CanonicalEventId> {
    let mention_by_id = sidecar
        .mention_packets
        .iter()
        .map(|packet| (packet.mention_id.0.clone(), packet))
        .collect::<FxHashMap<_, _>>();
    let mut rows = FxHashMap::<(String, String), CanonicalEventId>::default();
    for membership in &sidecar.memberships {
        if let Some(packet) = mention_by_id.get(membership.mention_id.0.as_str()) {
            rows.entry((packet.document_id.clone(), packet.event_id.clone()))
                .or_insert_with(|| membership.canonical_event_id.clone());
        }
    }
    rows
}

fn canonical_for(
    mapping: &FxHashMap<(String, String), CanonicalEventId>,
    document_id: &str,
    event_id: &str,
) -> Option<CanonicalEventId> {
    mapping
        .get(&(document_id.to_owned(), event_id.to_owned()))
        .cloned()
}

fn next_generation(current: Option<u64>) -> u64 {
    current.unwrap_or_default() + 1
}

fn build_summary(
    batch: &TemporalScopeReviewBatch,
    solved: &SolvedTemporalBatch,
) -> TemporalCompilerSummary {
    let mut axis_counts = BTreeMap::<String, usize>::new();
    for axis in &batch.axes {
        *axis_counts
            .entry(format!("{:?}", axis.kind).to_lowercase())
            .or_default() += 1;
    }
    let mut source_class_counts = BTreeMap::<String, usize>::new();
    for timex in &batch.timex_profiles {
        *source_class_counts
            .entry(timex.source_class.clone())
            .or_default() += 1;
    }
    for interval in &solved.intervals {
        *source_class_counts
            .entry(interval.source_class.clone())
            .or_default() += 1;
    }

    TemporalCompilerSummary {
        timex_count: batch.timex_profiles.len(),
        anchor_count: batch.anchors.len(),
        claim_count: batch.claim_atoms.len(),
        constraint_count: batch.constraints.len(),
        review_case_count: batch.review_cases.len(),
        interval_count: solved.intervals.len(),
        segment_count: solved.timeline_segments.len(),
        conflict_count: solved.conflicts.len(),
        gap_count: solved.gaps.len(),
        memory_card_count: batch.memory_cards.len(),
        axis_counts,
        source_class_counts,
    }
}
