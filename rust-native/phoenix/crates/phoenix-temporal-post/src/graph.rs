use std::collections::BTreeMap;

use phoenix_semantic_v2::TemporalConstraintKind;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::TemporalNormalizedInputs;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalGraphStats {
    #[serde(default)]
    pub before_by_event: FxHashMap<String, Vec<String>>,
    #[serde(default)]
    pub after_by_event: FxHashMap<String, Vec<String>>,
    #[serde(default)]
    pub axis_event_counts: BTreeMap<String, usize>,
}

pub fn build_temporal_graph_stats(inputs: &TemporalNormalizedInputs) -> TemporalGraphStats {
    let mut stats = TemporalGraphStats::default();
    for case in &inputs.review_cases {
        *stats
            .axis_event_counts
            .entry(case.axis_id.0.clone())
            .or_default() += 1;
    }
    for constraint in &inputs.constraints {
        if constraint.kind != TemporalConstraintKind::EndBeforeStart {
            continue;
        }
        let Some(source) = constraint.source_event_id.clone() else {
            continue;
        };
        let Some(target) = constraint.target_event_id.clone() else {
            continue;
        };
        stats
            .before_by_event
            .entry(source.clone())
            .or_default()
            .push(target.clone());
        stats.after_by_event.entry(target).or_default().push(source);
    }
    for values in stats.before_by_event.values_mut() {
        values.sort();
        values.dedup();
    }
    for values in stats.after_by_event.values_mut() {
        values.sort();
        values.dedup();
    }
    stats
}
