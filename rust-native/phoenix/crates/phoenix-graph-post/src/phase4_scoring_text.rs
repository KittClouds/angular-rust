use phoenix_graph_kernel::KernelVertex;

use crate::api::{GraphRankedCausalPath, GraphRankedHistoryCandidate, GraphRankedStateCandidate};

pub(crate) fn build_state_candidate_text(
    query_text: &str,
    candidate: &GraphRankedStateCandidate,
) -> String {
    let state = &candidate.state;
    trim_text(format!(
        "User query: {query_text}\nCandidate kind: state\nEntity: {}\nSlot: {}\nValue: {}\nStatus: {}\nTruth plane: {:?}\nConfidence: {:.3}\nConflicts: {}\nGaps: {}\nContradiction regions: {}\nSource classes: {}\nModalities: {}",
        state.entity_id,
        state.slot_key,
        state.value,
        state.status.clone().unwrap_or_else(|| "unknown".to_owned()),
        candidate.truth_plane,
        state.confidence.unwrap_or(0.0),
        candidate.relevant_conflict_count,
        candidate.relevant_gap_count,
        candidate.contradiction_region_count,
        candidate.supporting_source_classes.join(", "),
        candidate.supporting_modalities.join(", "),
    ))
}

pub(crate) fn build_history_candidate_text(
    query_text: &str,
    candidate: &GraphRankedHistoryCandidate,
) -> String {
    let change = &candidate.change;
    trim_text(format!(
        "User query: {query_text}\nCandidate kind: history_change\nChange kind: {:?}\nEntity: {}\nSlot: {}\nValue: {}\nStatus: {}\nTruth plane: {:?}\nConfidence: {:.3}\nTemporal fitness: {:.3}\nRecency score: {:.3}\nConflicts: {}\nGaps: {}\nContradiction regions: {}\nSource classes: {}\nModalities: {}",
        change.change_kind,
        change.state.entity_id,
        change.state.slot_key,
        change.state.value,
        change.state.status.clone().unwrap_or_else(|| "unknown".to_owned()),
        candidate.truth_plane,
        change.state.confidence.unwrap_or(0.0),
        candidate.temporal_fitness,
        candidate.recency_score,
        candidate.relevant_conflict_count,
        candidate.relevant_gap_count,
        candidate.contradiction_region_count,
        candidate.supporting_source_classes.join(", "),
        candidate.supporting_modalities.join(", "),
    ))
}

pub(crate) fn build_causal_candidate_text(
    query_text: &str,
    candidate: &GraphRankedCausalPath,
) -> String {
    let relations = candidate
        .hops
        .iter()
        .map(|hop| {
            hop.relation_kind
                .clone()
                .unwrap_or_else(|| "causal_link".to_owned())
        })
        .collect::<Vec<_>>()
        .join(" -> ");
    trim_text(format!(
        "User query: {query_text}\nCandidate kind: causal_path\nSource vertex: {}\nTarget vertex: {}\nPath depth: {}\nRelations: {}\nTruth plane: {:?}\nPath stability: {:.3}\nSupport strength: {:.3}\nTemporal fitness: {:.3}\nEvidence refs: {}",
        candidate.source_vertex_id,
        candidate.target_vertex_id,
        candidate.hops.len(),
        relations,
        candidate.truth_plane,
        candidate.path_stability,
        candidate.support_strength,
        candidate.temporal_fitness,
        candidate.evidence_refs.len(),
    ))
}

pub(crate) fn build_event_candidate_text(query_text: &str, vertex: &KernelVertex) -> String {
    let labels = if vertex.labels.is_empty() {
        String::new()
    } else {
        vertex.labels.join(", ")
    };
    let status = string_value(vertex, &["status", "kind"]);
    let value = string_value(vertex, &["value", "objectValue", "newValue", "oldValue"]);
    let slot_key = string_value(vertex, &["slotKey"]);
    trim_text(format!(
        "User query: {query_text}\nCandidate kind: event\nEvent id: {}\nEvent labels: {}\nEvent status: {}\nEvent value: {}\nEvent slot: {}\nEntity: {}\nEvidence refs: {}\nBoundary kind: {:?}",
        vertex.id.0,
        labels,
        status,
        value,
        slot_key,
        vertex.entity_id.clone().unwrap_or_default(),
        vertex.provenance.evidence_refs.len(),
        vertex.boundary_kind,
    ))
}

fn string_value(vertex: &KernelVertex, keys: &[&str]) -> String {
    keys.iter()
        .find_map(|key| {
            vertex
                .value
                .get(*key)
                .and_then(serde_json::Value::as_str)
                .or_else(|| {
                    vertex
                        .attributes
                        .get(*key)
                        .and_then(serde_json::Value::as_str)
                })
        })
        .unwrap_or_default()
        .to_owned()
}

fn trim_text(mut text: String) -> String {
    const LIMIT: usize = 1400;
    if text.len() <= LIMIT {
        return text;
    }
    text.truncate(LIMIT);
    text
}
