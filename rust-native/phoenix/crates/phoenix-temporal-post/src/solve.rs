use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    TemporalAnchorRecord, TemporalAxisKind, TemporalConflictKind, TemporalConflictRecord,
    TemporalGapKind, TemporalGapRecord, TemporalIntervalRecord, TimelineSegmentId,
    TimelineSegmentKind, TimelineSegmentRecord,
};
use phoenix_types::BiTemporalWindow;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::{
    build_temporal_graph_stats, choose_best_anchor, TemporalGraphStats, TemporalNormalizedInputs,
    TemporalReviewCase,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SolvedTemporalBatch {
    #[serde(default)]
    pub intervals: Vec<TemporalIntervalRecord>,
    #[serde(default)]
    pub timeline_segments: Vec<TimelineSegmentRecord>,
    #[serde(default)]
    pub conflicts: Vec<TemporalConflictRecord>,
    #[serde(default)]
    pub gaps: Vec<TemporalGapRecord>,
    #[serde(default)]
    pub diagnostics: BTreeMap<String, usize>,
    pub graph_stats: TemporalGraphStats,
}

pub fn solve_temporal_inputs(
    inputs: &TemporalNormalizedInputs,
    created_at: i64,
) -> SolvedTemporalBatch {
    let graph_stats = build_temporal_graph_stats(inputs);
    let anchors_by_event = build_anchor_map(&inputs.anchors);
    let axis_kind_by_id = inputs
        .axes
        .iter()
        .map(|axis| (axis.axis_id.0.clone(), axis.kind))
        .collect::<FxHashMap<_, _>>();
    let mut intervals = Vec::<TemporalIntervalRecord>::new();
    let mut conflicts = Vec::<TemporalConflictRecord>::new();
    let mut gaps = Vec::<TemporalGapRecord>::new();
    let diagnostics = inputs.diagnostics.clone();

    for case in &inputs.review_cases {
        let axis_kind = axis_kind_by_id
            .get(case.axis_id.0.as_str())
            .copied()
            .unwrap_or(case.axis_kind);
        let event_anchors = anchors_by_event
            .get(&case.event_id)
            .cloned()
            .unwrap_or_default();
        let explicit_timex_ids = event_anchors
            .iter()
            .filter(|anchor| anchor.anchor_kind == "explicit_timex")
            .filter_map(|anchor| anchor.timex_id.as_ref().map(|id| id.0.clone()))
            .collect::<Vec<_>>();
        let mut distinct_timex_ids = explicit_timex_ids.clone();
        distinct_timex_ids.sort();
        distinct_timex_ids.dedup();
        if distinct_timex_ids.len() > 1 {
            conflicts.push(TemporalConflictRecord {
                conflict_id: format!("tconflict:{}:anchors", case.event_id),
                document_id: case.document_id.clone(),
                axis_id: case.axis_id.clone(),
                kind: TemporalConflictKind::IncompatibleAnchors,
                event_id: Some(case.event_id.clone()),
                canonical_event_id: case.canonical_event_id.clone(),
                constraint_ids: Vec::new(),
                reason: "multiple explicit timex anchors remained unresolved".to_owned(),
            });
            gaps.push(gap(
                case,
                TemporalGapKind::ConflictingAnchors,
                "conflicting_explicit_anchors",
            ));
            continue;
        }

        if let Some(anchor) = choose_best_anchor(case, &anchors_by_event) {
            if supports_interval(&anchor) {
                intervals.push(interval(case, &anchor));
            } else if anchor.anchor_kind == "document_created_at"
                && axis_kind == TemporalAxisKind::World
            {
                gaps.push(gap(
                    case,
                    TemporalGapKind::UnderspecifiedInterval,
                    "document_created_at_fallback",
                ));
            } else {
                gaps.push(gap(
                    case,
                    TemporalGapKind::MissingAnchor,
                    "non_world_axis_without_explicit_anchor",
                ));
            }
        } else {
            gaps.push(gap(case, TemporalGapKind::MissingAnchor, "missing_anchor"));
        }

        if graph_stats
            .before_by_event
            .get(&case.event_id)
            .map(|rows| rows.is_empty())
            .unwrap_or(true)
            && graph_stats
                .after_by_event
                .get(&case.event_id)
                .map(|rows| rows.is_empty())
                .unwrap_or(true)
        {
            gaps.push(gap(
                case,
                TemporalGapKind::UnresolvedOrder,
                "no_order_neighbors",
            ));
        }
    }

    let timeline_segments = build_segments(
        &inputs.review_cases,
        &intervals,
        &gaps,
        created_at,
        &axis_kind_by_id,
    );

    SolvedTemporalBatch {
        intervals,
        timeline_segments,
        conflicts,
        gaps,
        diagnostics,
        graph_stats,
    }
}

fn build_anchor_map(
    anchors: &[TemporalAnchorRecord],
) -> FxHashMap<String, Vec<TemporalAnchorRecord>> {
    let mut rows = FxHashMap::<String, Vec<TemporalAnchorRecord>>::default();
    for anchor in anchors {
        if let Some(event_id) = anchor.event_id.clone() {
            rows.entry(event_id).or_default().push(anchor.clone());
        }
    }
    rows
}

fn supports_interval(anchor: &TemporalAnchorRecord) -> bool {
    anchor.anchor_kind == "explicit_timex"
}

fn interval(case: &TemporalReviewCase, anchor: &TemporalAnchorRecord) -> TemporalIntervalRecord {
    TemporalIntervalRecord {
        interval_id: format!("interval:{}", case.event_id),
        document_id: case.document_id.clone(),
        event_id: case.event_id.clone(),
        canonical_event_id: case.canonical_event_id.clone(),
        axis_id: case.axis_id.clone(),
        anchor_id: Some(anchor.anchor_id.clone()),
        temporal: anchor.temporal.clone(),
        confidence_millis: anchor.confidence_millis,
        source_class: anchor.source_class.clone(),
        evidence_refs: anchor.evidence_refs.clone(),
    }
}

fn gap(case: &TemporalReviewCase, kind: TemporalGapKind, reason: &str) -> TemporalGapRecord {
    TemporalGapRecord {
        gap_id: format!("gap:{}:{reason}", case.event_id),
        document_id: case.document_id.clone(),
        axis_id: case.axis_id.clone(),
        event_id: Some(case.event_id.clone()),
        canonical_event_id: case.canonical_event_id.clone(),
        kind,
        reason: reason.to_owned(),
        evidence_refs: case.anchor_candidate_ids.clone(),
    }
}

fn build_segments(
    review_cases: &[TemporalReviewCase],
    intervals: &[TemporalIntervalRecord],
    gaps: &[TemporalGapRecord],
    created_at: i64,
    axis_kind_by_id: &FxHashMap<String, TemporalAxisKind>,
) -> Vec<TimelineSegmentRecord> {
    let interval_by_event = intervals
        .iter()
        .map(|interval| (interval.event_id.clone(), interval))
        .collect::<FxHashMap<_, _>>();
    let mut groups = BTreeMap::<(String, String), Vec<&TemporalReviewCase>>::new();
    for case in review_cases {
        groups
            .entry((case.document_id.clone(), case.axis_id.0.clone()))
            .or_default()
            .push(case);
    }

    groups
        .into_iter()
        .map(|((document_id, axis_id_value), mut cases)| {
            cases.sort_by(|left, right| {
                (
                    left.sentence_index,
                    interval_order_key(interval_by_event.get(&left.event_id).copied()),
                    left.event_id.as_str(),
                )
                    .cmp(&(
                        right.sentence_index,
                        interval_order_key(interval_by_event.get(&right.event_id).copied()),
                        right.event_id.as_str(),
                    ))
            });
            let event_ids = cases
                .iter()
                .map(|case| case.event_id.clone())
                .collect::<Vec<_>>();
            let canonical_event_ids = cases
                .iter()
                .filter_map(|case| case.canonical_event_id.clone())
                .collect::<Vec<_>>();
            let indeterminate_event_ids = gaps
                .iter()
                .filter(|gap| gap.document_id == document_id && gap.axis_id.0 == axis_id_value)
                .filter_map(|gap| gap.event_id.clone())
                .collect::<Vec<_>>();
            let indeterminate_canonical_event_ids = gaps
                .iter()
                .filter(|gap| gap.document_id == document_id && gap.axis_id.0 == axis_id_value)
                .filter_map(|gap| gap.canonical_event_id.clone())
                .collect::<Vec<_>>();
            let anchored_count = cases
                .iter()
                .filter(|case| interval_by_event.contains_key(case.event_id.as_str()))
                .count();
            let anchor_coverage_millis = if cases.is_empty() {
                0
            } else {
                ((anchored_count * 1000) / cases.len()) as u32
            };
            let temporal = merge_temporal(
                cases.iter().filter_map(|case| {
                    interval_by_event
                        .get(case.event_id.as_str())
                        .map(|row| row.temporal.clone())
                }),
                created_at,
            );
            let axis_kind = axis_kind_by_id
                .get(axis_id_value.as_str())
                .copied()
                .unwrap_or(TemporalAxisKind::World);
            TimelineSegmentRecord {
                segment_id: TimelineSegmentId(format!("segment:{document_id}:{axis_id_value}")),
                document_id,
                axis_id: phoenix_semantic_v2::TemporalAxisId(axis_id_value),
                segment_kind: if axis_kind == TemporalAxisKind::World {
                    TimelineSegmentKind::Main
                } else {
                    TimelineSegmentKind::Branch
                },
                event_ids,
                canonical_event_ids,
                anchor_coverage_millis,
                indeterminate_event_ids,
                indeterminate_canonical_event_ids,
                temporal,
            }
        })
        .collect()
}

fn interval_order_key(interval: Option<&TemporalIntervalRecord>) -> (i64, i64) {
    let temporal = interval
        .map(|row| row.temporal.clone())
        .unwrap_or(BiTemporalWindow {
            valid_from: None,
            valid_to: None,
            recorded_from: None,
            recorded_to: None,
        });
    (
        temporal.valid_from.unwrap_or(i64::MAX),
        temporal.recorded_from.unwrap_or(i64::MAX),
    )
}

fn merge_temporal<I>(windows: I, created_at: i64) -> BiTemporalWindow
where
    I: Iterator<Item = BiTemporalWindow>,
{
    let mut min_valid = None::<i64>;
    let mut max_valid = None::<i64>;
    let mut min_recorded = None::<i64>;
    for window in windows {
        if let Some(valid_from) = window.valid_from {
            min_valid = Some(
                min_valid
                    .map(|current| current.min(valid_from))
                    .unwrap_or(valid_from),
            );
        }
        if let Some(valid_to) = window.valid_to {
            max_valid = Some(
                max_valid
                    .map(|current| current.max(valid_to))
                    .unwrap_or(valid_to),
            );
        }
        if let Some(recorded_from) = window.recorded_from {
            min_recorded = Some(
                min_recorded
                    .map(|current| current.min(recorded_from))
                    .unwrap_or(recorded_from),
            );
        }
    }
    BiTemporalWindow {
        valid_from: min_valid,
        valid_to: max_valid,
        recorded_from: min_recorded.or(Some(created_at)),
        recorded_to: None,
    }
}
