use crate::{
    build_slot_state, classify_state_change, collect_slot_states, collect_state_issues,
    compare_slot_states, compare_state_issues, now_ms, vertex_slot_key, KernelEdge, KernelEdgeKey,
    KernelGraphSnapshot, KernelSlotAnswer, KernelSlotQueryRequest, KernelStateChange,
    KernelStateIssue, KernelUnresolvedQueryRequest, KernelWhatChangedRequest,
};
use rustc_hash::FxHashSet;

pub fn slot_at_snapshot(
    snapshot: &KernelGraphSnapshot,
    request: &KernelSlotQueryRequest,
) -> KernelSlotAnswer {
    let mut states = collect_slot_states(
        &snapshot.vertices,
        &snapshot.asserted_edges,
        &request.entity_id,
        &request.slot_key,
    );
    states.sort_by(compare_slot_states);
    let mut conflicts = collect_state_issues(
        &snapshot.vertices,
        &snapshot.asserted_edges,
        "conflict",
        &request.entity_id,
        Some(request.slot_key.as_str()),
        false,
    );
    let mut gaps = collect_state_issues(
        &snapshot.vertices,
        &snapshot.asserted_edges,
        "gap",
        &request.entity_id,
        Some(request.slot_key.as_str()),
        false,
    );
    conflicts.sort_by(compare_state_issues);
    gaps.sort_by(compare_state_issues);

    let active_state = states.first().cloned();
    let competing_states = states.into_iter().skip(1).collect::<Vec<_>>();
    KernelSlotAnswer {
        entity_id: request.entity_id.clone(),
        slot_key: request.slot_key.clone(),
        active_state,
        competing_states,
        conflicts,
        gaps,
    }
}

pub fn unresolved_from_snapshot(
    snapshot: &KernelGraphSnapshot,
    request: &KernelUnresolvedQueryRequest,
) -> Vec<KernelStateIssue> {
    let mut issues = collect_state_issues(
        &snapshot.vertices,
        &snapshot.asserted_edges,
        "conflict",
        &request.entity_id,
        request.slot_key.as_deref(),
        true,
    );
    issues.extend(collect_state_issues(
        &snapshot.vertices,
        &snapshot.asserted_edges,
        "gap",
        &request.entity_id,
        request.slot_key.as_deref(),
        true,
    ));
    issues.sort_by(compare_state_issues);
    issues
}

pub fn entity_timeline_from_snapshot(
    snapshot: &KernelGraphSnapshot,
    entity_id: &str,
    window: Option<(i64, i64)>,
    _tx_point: Option<i64>,
) -> KernelGraphSnapshot {
    let (start, end) = window
        .map(|(start, end)| (Some(start), Some(end)))
        .unwrap_or((None, None));
    let mut vertices = snapshot
        .vertices
        .iter()
        .filter(|vertex| {
            (vertex.id.0 == entity_id
                || vertex.entity_id.as_deref() == Some(entity_id)
                || vertex
                    .entity_facet
                    .as_ref()
                    .and_then(|facet| facet.canonical_entity_id.as_deref())
                    == Some(entity_id))
                && vertex.temporal.overlaps_valid_window(start, end)
        })
        .cloned()
        .collect::<Vec<_>>();
    let vertex_ids = vertices
        .iter()
        .map(|vertex| vertex.id.0.clone())
        .collect::<FxHashSet<_>>();
    let mut asserted_edges = timeline_edges(&snapshot.asserted_edges, &vertex_ids, start, end);
    let mut candidate_edges = timeline_edges(&snapshot.candidate_edges, &vertex_ids, start, end);
    vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
    asserted_edges.sort_by(|left, right| {
        KernelEdgeKey::from_edge(left)
            .storage_key()
            .cmp(&KernelEdgeKey::from_edge(right).storage_key())
    });
    candidate_edges.sort_by(|left, right| {
        KernelEdgeKey::from_edge(left)
            .storage_key()
            .cmp(&KernelEdgeKey::from_edge(right).storage_key())
    });
    KernelGraphSnapshot {
        vertices,
        asserted_edges,
        candidate_edges,
    }
}

pub fn what_changed_from_snapshot(
    timeline: &KernelGraphSnapshot,
    request: &KernelWhatChangedRequest,
) -> Vec<KernelStateChange> {
    let until_valid_at = request.until_valid_at.unwrap_or_else(now_ms);
    let mut changes = timeline
        .vertices
        .iter()
        .filter(|vertex| {
            vertex.kind == "state"
                && vertex.entity_id.as_deref() == Some(request.entity_id.as_str())
                && request
                    .slot_key
                    .as_deref()
                    .map(|slot_key| vertex_slot_key(vertex) == Some(slot_key))
                    .unwrap_or(true)
        })
        .filter_map(|vertex| {
            classify_state_change(&vertex.temporal, request.since_valid_at, until_valid_at).map(
                |change_kind| KernelStateChange {
                    change_kind,
                    state: build_slot_state(vertex, &timeline.asserted_edges, &request.entity_id),
                },
            )
        })
        .collect::<Vec<_>>();
    changes.sort_by(|left, right| {
        left.state
            .temporal
            .valid_from
            .cmp(&right.state.temporal.valid_from)
            .then_with(|| left.state.state_vertex_id.cmp(&right.state.state_vertex_id))
    });
    changes
}

fn timeline_edges(
    edges: &[KernelEdge],
    vertex_ids: &FxHashSet<String>,
    start: Option<i64>,
    end: Option<i64>,
) -> Vec<KernelEdge> {
    edges
        .iter()
        .filter(|edge| {
            (vertex_ids.contains(&edge.source_id.0) || vertex_ids.contains(&edge.target_id.0))
                && edge.temporal.overlaps_valid_window(start, end)
        })
        .cloned()
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::{
        entity_timeline_from_snapshot, slot_at_snapshot, unresolved_from_snapshot,
        what_changed_from_snapshot,
    };
    use crate::{
        KernelBiTemporal, KernelEdge, KernelEdgeType, KernelGraphLayer, KernelGraphSnapshot,
        KernelSlotQueryRequest, KernelUnresolvedQueryRequest, KernelVertex, KernelVertexId,
        KernelViewRequest, KernelWhatChangedRequest, PhoenixGraphKernel,
    };
    use serde_json::json;

    #[test]
    fn slot_snapshot_matches_kernel_slot_query() {
        let snapshot = sample_snapshot();
        let kernel = PhoenixGraphKernel::from_snapshot(snapshot.clone(), None);
        let visible = kernel.view_as_of(KernelViewRequest {
            valid_at: Some(150),
            recorded_at: Some(150),
            include_candidate_graph: false,
        });
        let request = KernelSlotQueryRequest {
            entity_id: "alice".to_owned(),
            slot_key: "entity.location".to_owned(),
            valid_at: Some(150),
            recorded_at: Some(150),
            include_candidate_graph: false,
        };

        let direct = slot_at_snapshot(&visible, &request);
        let through_kernel = kernel.slot_at(request);

        assert_eq!(direct, through_kernel);
    }

    #[test]
    fn unresolved_snapshot_matches_kernel_unresolved_query() {
        let snapshot = sample_snapshot();
        let kernel = PhoenixGraphKernel::from_snapshot(snapshot.clone(), None);
        let visible = kernel.view_as_of(KernelViewRequest {
            valid_at: Some(150),
            recorded_at: Some(150),
            include_candidate_graph: false,
        });
        let request = KernelUnresolvedQueryRequest {
            entity_id: "alice".to_owned(),
            slot_key: Some("entity.location".to_owned()),
            valid_at: Some(150),
            recorded_at: Some(150),
            include_candidate_graph: false,
        };

        let direct = unresolved_from_snapshot(&visible, &request);
        let through_kernel = kernel.what_is_unresolved(request);

        assert_eq!(direct, through_kernel);
    }

    #[test]
    fn what_changed_snapshot_matches_kernel_history_query() {
        let snapshot = sample_snapshot();
        let kernel = PhoenixGraphKernel::from_snapshot(snapshot.clone(), None);
        let request = KernelWhatChangedRequest {
            entity_id: "alice".to_owned(),
            slot_key: Some("entity.location".to_owned()),
            since_valid_at: 50,
            until_valid_at: Some(260),
            recorded_at: Some(260),
            include_candidate_graph: false,
        };

        let timeline = entity_timeline_from_snapshot(
            &snapshot,
            &request.entity_id,
            Some((
                request.since_valid_at,
                request.until_valid_at.unwrap_or(260),
            )),
            request.recorded_at,
        );
        let direct = what_changed_from_snapshot(&timeline, &request);
        let through_kernel = kernel.what_changed(request);

        assert_eq!(direct, through_kernel);
    }

    fn sample_snapshot() -> KernelGraphSnapshot {
        KernelGraphSnapshot {
            vertices: vec![
                state_vertex(
                    "graph::state::alice:old",
                    "alice",
                    "entity.location",
                    "Old",
                    100,
                    200,
                ),
                state_vertex(
                    "graph::state::alice:new",
                    "alice",
                    "entity.location",
                    "New",
                    200,
                    400,
                ),
                issue_vertex(
                    "graph::conflict::alice",
                    "conflict",
                    "alice",
                    "entity.location",
                    210,
                ),
                issue_vertex("graph::gap::alice", "gap", "alice", "entity.location", 220),
                claim_vertex("graph::claim::claim-old"),
                claim_vertex("graph::claim::claim-new"),
            ],
            asserted_edges: vec![
                support_edge("graph::state::alice:old", "graph::claim::claim-old"),
                support_edge("graph::state::alice:new", "graph::claim::claim-new"),
            ],
            candidate_edges: Vec::new(),
        }
    }

    fn state_vertex(
        id: &str,
        entity_id: &str,
        slot_key: &str,
        value: &str,
        valid_from: i64,
        valid_to: i64,
    ) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: "state".to_owned(),
            entity_id: Some(entity_id.to_owned()),
            value: json!({
                "slotKey": slot_key,
                "value": value,
                "status": "active",
            }),
            temporal: KernelBiTemporal {
                valid_from: Some(valid_from),
                valid_to: Some(valid_to),
                recorded_at: Some(valid_from),
                expired_at: None,
            },
            ..KernelVertex::default()
        }
    }

    fn issue_vertex(
        id: &str,
        kind: &str,
        entity_id: &str,
        slot_key: &str,
        valid_from: i64,
    ) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: kind.to_owned(),
            entity_id: Some(entity_id.to_owned()),
            value: json!({
                "slotKey": slot_key,
                "status": "open",
                "kind": kind,
            }),
            temporal: KernelBiTemporal {
                valid_from: Some(valid_from),
                valid_to: None,
                recorded_at: Some(valid_from),
                expired_at: None,
            },
            ..KernelVertex::default()
        }
    }

    fn claim_vertex(id: &str) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: "claim".to_owned(),
            ..KernelVertex::default()
        }
    }

    fn support_edge(source_id: &str, target_id: &str) -> KernelEdge {
        KernelEdge {
            source_id: KernelVertexId(source_id.to_owned()),
            target_id: KernelVertexId(target_id.to_owned()),
            edge_type: KernelEdgeType("supported_by".to_owned()),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        }
    }
}
