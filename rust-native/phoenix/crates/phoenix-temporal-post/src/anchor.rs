use phoenix_semantic_v2::{TemporalAnchorRecord, TemporalAxisKind};
use rustc_hash::FxHashMap;

use crate::TemporalReviewCase;

pub fn choose_best_anchor(
    case: &TemporalReviewCase,
    anchors_by_event: &FxHashMap<String, Vec<TemporalAnchorRecord>>,
) -> Option<TemporalAnchorRecord> {
    let mut candidates = anchors_by_event.get(&case.event_id)?.clone();
    candidates.sort_by(|left, right| {
        anchor_rank(right, case.axis_kind)
            .cmp(&anchor_rank(left, case.axis_kind))
            .then_with(|| right.confidence_millis.cmp(&left.confidence_millis))
            .then_with(|| left.anchor_id.0.cmp(&right.anchor_id.0))
    });
    candidates.into_iter().next()
}

pub fn has_world_anchor_support(
    case: &TemporalReviewCase,
    anchors_by_event: &FxHashMap<String, Vec<TemporalAnchorRecord>>,
) -> bool {
    anchors_by_event
        .get(&case.event_id)
        .map(|rows| {
            rows.iter().any(|row| {
                row.anchor_kind == "explicit_timex"
                    || (case.axis_kind == TemporalAxisKind::World
                        && row.anchor_kind == "document_created_at")
            })
        })
        .unwrap_or(false)
}

fn anchor_rank(anchor: &TemporalAnchorRecord, axis_kind: TemporalAxisKind) -> u8 {
    match anchor.anchor_kind.as_str() {
        "explicit_timex" => 4,
        "boundary_marker" => 3,
        "reference_event" => 2,
        "document_created_at" if axis_kind == TemporalAxisKind::World => 1,
        _ => 0,
    }
}
