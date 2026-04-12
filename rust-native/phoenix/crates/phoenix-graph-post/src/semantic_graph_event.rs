use hashbrown::{HashMap, HashSet};
use phoenix_semantic_v2::{
    SemanticCandidateStatus, SemanticEdgeFamily, SemanticGraphEdgeCandidate, SemanticGraphNodeKind,
};

use crate::semantic_graph_support::{truth_planes_compatible, Prototype, EVENT_KIND};
use crate::semantic_graph_workspace::SemanticNeighborWorkspace;

pub(crate) fn collect_related_event_edges(
    workspace: &mut SemanticNeighborWorkspace<'_>,
    prototypes: &[Prototype],
    neighbor_limit: usize,
    oversample: usize,
    min_score_millis: u32,
) -> Vec<SemanticGraphEdgeCandidate> {
    let prototype_by_id = prototypes
        .iter()
        .map(|prototype| (prototype.node_id.as_str(), prototype))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut edges = Vec::new();
    for (source_index, source) in prototypes.iter().enumerate() {
        if source.node_kind != SemanticGraphNodeKind::Event {
            continue;
        }
        let hits = workspace.query_semantic_node_neighbors(
            source_index,
            EVENT_KIND,
            neighbor_limit,
            oversample,
        );
        for hit in hits {
            let Some(target) = prototype_by_id.get(hit.node_id.as_str()) else {
                continue;
            };
            if !related_event_compatible(source, target) {
                continue;
            }
            let score_millis = related_event_score_millis(hit.distance, source, target);
            if score_millis < min_score_millis {
                continue;
            }
            let Some((left, right)) = ordered_pair(source, target) else {
                continue;
            };
            if !seen.insert((left.node_id.as_str(), right.node_id.as_str())) {
                continue;
            }
            edges.push(SemanticGraphEdgeCandidate {
                edge_id: format!("semantic:related_event:{}:{}", left.node_id, right.node_id),
                family: SemanticEdgeFamily::RelatedEvent,
                source_node_id: left.node_id.clone(),
                source_kind: left.node_kind,
                target_node_id: right.node_id.clone(),
                target_kind: right.node_kind,
                score_millis,
                distance_millis: (hit.distance.max(0.0) * 1000.0).round() as u32,
                candidate_status: SemanticCandidateStatus::Generated,
                evidence_refs: merge_refs(&left.evidence_refs, &right.evidence_refs),
                model_evidence: related_event_evidence(hit.distance, left, right),
                nli_support_millis: None,
                nli_contradiction_millis: None,
            });
        }
    }
    edges.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    edges
}

fn related_event_compatible(source: &Prototype, target: &Prototype) -> bool {
    source.node_kind == SemanticGraphNodeKind::Event
        && target.node_kind == SemanticGraphNodeKind::Event
        && source.node_id != target.node_id
        && truth_planes_compatible(source.truth_plane.as_deref(), target.truth_plane.as_deref())
        && shares_scope(source, target)
        && !same_process_like(source, target)
}

fn same_process_like(source: &Prototype, target: &Prototype) -> bool {
    shares_entity(source, target) || shares_slot(source, target)
}

fn shares_scope(source: &Prototype, target: &Prototype) -> bool {
    same_value(source.document_id.as_deref(), target.document_id.as_deref())
        || same_value(
            source.narrative_id.as_deref(),
            target.narrative_id.as_deref(),
        )
}

fn shares_entity(source: &Prototype, target: &Prototype) -> bool {
    entity_matches(
        source.primary_entity_id.as_deref(),
        target.primary_entity_id.as_deref(),
    ) || entity_matches(
        source.primary_entity_id.as_deref(),
        target.secondary_entity_id.as_deref(),
    ) || entity_matches(
        source.secondary_entity_id.as_deref(),
        target.primary_entity_id.as_deref(),
    ) || entity_matches(
        source.secondary_entity_id.as_deref(),
        target.secondary_entity_id.as_deref(),
    )
}

fn entity_matches(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

fn shares_slot(source: &Prototype, target: &Prototype) -> bool {
    same_value(source.slot_key.as_deref(), target.slot_key.as_deref())
}

fn same_value(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

fn shares_evidence(source: &Prototype, target: &Prototype) -> bool {
    source
        .evidence_refs
        .iter()
        .any(|left| target.evidence_refs.iter().any(|right| right == left))
}

fn ordered_pair<'a>(
    left: &'a Prototype,
    right: &'a Prototype,
) -> Option<(&'a Prototype, &'a Prototype)> {
    if left.node_id == right.node_id {
        None
    } else if left.node_id <= right.node_id {
        Some((left, right))
    } else {
        Some((right, left))
    }
}

fn related_event_score_millis(distance: f64, source: &Prototype, target: &Prototype) -> u32 {
    let mut score = neighbor_score_millis(distance);
    if same_value(source.document_id.as_deref(), target.document_id.as_deref()) {
        score = score.saturating_add(70);
    }
    if same_value(
        source.narrative_id.as_deref(),
        target.narrative_id.as_deref(),
    ) {
        score = score.saturating_add(90);
    }
    if shares_evidence(source, target) {
        score = score.saturating_add(80);
    }
    score.min(900)
}

fn related_event_evidence(distance: f64, source: &Prototype, target: &Prototype) -> Vec<String> {
    let mut evidence = vec![format!("ann:distance={distance:.4}")];
    if same_value(source.document_id.as_deref(), target.document_id.as_deref()) {
        evidence.push("shared-document".to_owned());
    }
    if same_value(
        source.narrative_id.as_deref(),
        target.narrative_id.as_deref(),
    ) {
        evidence.push("shared-narrative".to_owned());
    }
    if shares_evidence(source, target) {
        evidence.push("shared-evidence".to_owned());
    }
    evidence
}

fn neighbor_score_millis(distance: f64) -> u32 {
    ((1.0 / (1.0 + distance.max(0.0))) * 1000.0)
        .round()
        .clamp(0.0, 1000.0) as u32
}

fn merge_refs(left: &[String], right: &[String]) -> Vec<String> {
    let mut merged = left.iter().chain(right.iter()).cloned().collect::<Vec<_>>();
    merged.sort();
    merged.dedup();
    merged
}

#[cfg(test)]
mod tests {
    use super::{related_event_compatible, related_event_score_millis};
    use phoenix_semantic_v2::{SemanticGraphNodeKind, SemanticGraphNodeRecord};

    use crate::semantic_graph_support::{Prototype, EVENT_KIND};

    fn event_prototype(node_id: &str) -> Prototype {
        Prototype {
            node_id: node_id.to_owned(),
            ann_kind: EVENT_KIND,
            node_kind: SemanticGraphNodeKind::Event,
            text_key: node_id.to_owned(),
            text: node_id.to_owned(),
            truth_plane: Some("world".to_owned()),
            document_id: Some("doc-1".to_owned()),
            narrative_id: Some("nar-1".to_owned()),
            evidence_refs: vec!["event://shared".to_owned()],
            semantic_node: SemanticGraphNodeRecord {
                node_id: node_id.to_owned(),
                node_kind: SemanticGraphNodeKind::Event,
                document_id: Some("doc-1".to_owned()),
                narrative_id: Some("nar-1".to_owned()),
                text_key: node_id.to_owned(),
                text_hash: 1,
                truth_plane: Some("world".to_owned()),
                evidence_refs: Vec::new(),
            },
            slot_key: Some("entity.location".to_owned()),
            value_key: Some("value".to_owned()),
            primary_entity_id: None,
            secondary_entity_id: None,
        }
    }

    #[test]
    fn related_event_accepts_same_scope_without_same_process_signals() {
        let mut left = event_prototype("graph::event::1");
        let mut right = event_prototype("graph::event::2");
        left.slot_key = Some("entity.location".to_owned());
        right.slot_key = Some("entity.employer".to_owned());

        assert!(related_event_compatible(&left, &right));
        assert!(related_event_score_millis(0.28, &left, &right) >= 800);
    }

    #[test]
    fn related_event_rejects_same_process_pairs() {
        let mut left = event_prototype("graph::event::1");
        let mut right = event_prototype("graph::event::2");
        left.primary_entity_id = Some("alice".to_owned());
        right.primary_entity_id = Some("alice".to_owned());

        assert!(!related_event_compatible(&left, &right));
    }

    #[test]
    fn related_event_rejects_cross_scope_noise() {
        let left = event_prototype("graph::event::1");
        let mut right = event_prototype("graph::event::2");
        right.document_id = Some("doc-2".to_owned());
        right.narrative_id = Some("nar-2".to_owned());

        assert!(!related_event_compatible(&left, &right));
    }
}
