use phoenix_semantic_v2::{
    TemporalConflictRecord, TemporalGapRecord, TemporalIntervalRecord, TemporalMemoryCard,
};
use rustc_hash::FxHashMap;

use crate::{TemporalEventProfile, TemporalGraphStats};

pub fn build_temporal_memory_cards(
    event_profiles: &[TemporalEventProfile],
    intervals: &[TemporalIntervalRecord],
    conflicts: &[TemporalConflictRecord],
    gaps: &[TemporalGapRecord],
    graph_stats: &TemporalGraphStats,
) -> Vec<TemporalMemoryCard> {
    let interval_by_event = intervals
        .iter()
        .map(|interval| (interval.event_id.clone(), interval))
        .collect::<FxHashMap<_, _>>();
    let canonical_by_event = event_profiles
        .iter()
        .filter_map(|profile| {
            profile
                .canonical_event_id
                .clone()
                .map(|canonical_event_id| (profile.event_id.clone(), canonical_event_id))
        })
        .collect::<FxHashMap<_, _>>();
    let conflict_ids_by_event = index_conflicts(conflicts);
    let gap_ids_by_event = index_gaps(gaps);

    event_profiles
        .iter()
        .map(|profile| {
            let interval = interval_by_event.get(profile.event_id.as_str()).copied();
            TemporalMemoryCard {
                card_id: format!("tcard:{}", profile.event_id),
                document_id: profile.document_id.clone(),
                event_id: profile.event_id.clone(),
                canonical_event_id: profile.canonical_event_id.clone(),
                label: profile.label.clone(),
                sentence_index: profile.sentence_index,
                axis_kind: profile.axis_kind,
                strongest_interval: interval.map(|row| row.temporal.clone()),
                anchor_source: interval.map(|row| row.source_class.clone()),
                before_event_ids: graph_stats
                    .before_by_event
                    .get(profile.event_id.as_str())
                    .cloned()
                    .unwrap_or_default(),
                before_canonical_event_ids: graph_stats
                    .before_by_event
                    .get(profile.event_id.as_str())
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|event_id| canonical_by_event.get(event_id.as_str()).cloned())
                    .collect(),
                after_event_ids: graph_stats
                    .after_by_event
                    .get(profile.event_id.as_str())
                    .cloned()
                    .unwrap_or_default(),
                after_canonical_event_ids: graph_stats
                    .after_by_event
                    .get(profile.event_id.as_str())
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .filter_map(|event_id| canonical_by_event.get(event_id.as_str()).cloned())
                    .collect(),
                open_conflict_ids: conflict_ids_by_event
                    .get(profile.event_id.as_str())
                    .cloned()
                    .unwrap_or_default(),
                open_gap_ids: gap_ids_by_event
                    .get(profile.event_id.as_str())
                    .cloned()
                    .unwrap_or_default(),
                evidence_refs: profile.evidence_refs.clone(),
            }
        })
        .collect()
}

fn index_conflicts(conflicts: &[TemporalConflictRecord]) -> FxHashMap<String, Vec<String>> {
    let mut rows = FxHashMap::<String, Vec<String>>::default();
    for conflict in conflicts {
        if let Some(event_id) = conflict.event_id.clone() {
            rows.entry(event_id)
                .or_default()
                .push(conflict.conflict_id.clone());
        }
    }
    rows
}

fn index_gaps(gaps: &[TemporalGapRecord]) -> FxHashMap<String, Vec<String>> {
    let mut rows = FxHashMap::<String, Vec<String>>::default();
    for gap in gaps {
        if let Some(event_id) = gap.event_id.clone() {
            rows.entry(event_id).or_default().push(gap.gap_id.clone());
        }
    }
    rows
}
