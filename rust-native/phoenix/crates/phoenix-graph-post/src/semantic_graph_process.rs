use hashbrown::{HashMap, HashSet};
use phoenix_semantic_v2::{
    SemanticCandidateStatus, SemanticEdgeFamily, SemanticGraphEdgeCandidate, SemanticGraphNodeKind,
};
use phoenix_store_native_core::PhoenixSemanticIndexStore;
use phoenix_types::ScopeKey;

use crate::semantic_graph::SemanticGraphError;
use crate::semantic_graph_support::{
    truth_planes_compatible, Prototype, CLAIM_KIND, EVENT_KIND, STATE_KIND,
};

pub(crate) fn collect_same_process_edges_from_store<S>(
    store: &S,
    scope: &ScopeKey,
    prototypes: &[Prototype],
    embeddings: &[Vec<f32>],
    neighbor_limit: usize,
    oversample: usize,
    min_score_millis: u32,
) -> Result<Vec<SemanticGraphEdgeCandidate>, SemanticGraphError>
where
    S: PhoenixSemanticIndexStore,
{
    let prototype_by_id = prototypes
        .iter()
        .map(|prototype| (prototype.node_id.as_str(), prototype))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut edges = Vec::new();
    for (source, embedding) in prototypes.iter().zip(embeddings.iter()) {
        let Some(target_kind) = same_process_target_kind(source.node_kind) else {
            continue;
        };
        let hits = store.query_semantic_node_neighbors(
            embedding,
            scope,
            target_kind,
            Some(&source.node_id),
            neighbor_limit,
            oversample,
        )?;
        for hit in hits {
            let Some(target) = prototype_by_id.get(hit.node_id.as_str()) else {
                continue;
            };
            if !same_process_compatible(source, target) {
                continue;
            }
            let score_millis = same_process_score_millis(hit.distance, source, target);
            if score_millis < min_score_millis {
                continue;
            }
            let Some((left, right)) = ordered_pair(source, target) else {
                continue;
            };
            if !seen.insert((left.node_id.as_str(), right.node_id.as_str())) {
                continue;
            }
            let evidence_refs = merge_refs(&left.evidence_refs, &right.evidence_refs);
            edges.push(SemanticGraphEdgeCandidate {
                edge_id: format!("semantic:same_process:{}:{}", left.node_id, right.node_id),
                family: SemanticEdgeFamily::SameProcess,
                source_node_id: left.node_id.clone(),
                source_kind: left.node_kind,
                target_node_id: right.node_id.clone(),
                target_kind: right.node_kind,
                score_millis,
                distance_millis: (hit.distance.max(0.0) * 1000.0).round() as u32,
                candidate_status: SemanticCandidateStatus::Generated,
                evidence_refs,
                model_evidence: same_process_evidence(hit.distance, left, right),
                nli_support_millis: None,
                nli_contradiction_millis: None,
            });
        }
    }
    edges.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    Ok(edges)
}

fn same_process_target_kind(kind: SemanticGraphNodeKind) -> Option<&'static str> {
    match kind {
        SemanticGraphNodeKind::Claim => Some(CLAIM_KIND),
        SemanticGraphNodeKind::State => Some(STATE_KIND),
        SemanticGraphNodeKind::Event => Some(EVENT_KIND),
        _ => None,
    }
}

fn same_process_compatible(source: &Prototype, target: &Prototype) -> bool {
    source.node_kind == target.node_kind
        && source.node_id != target.node_id
        && truth_planes_compatible(source.truth_plane.as_deref(), target.truth_plane.as_deref())
        && shares_scope(source, target)
        && (shares_entity(source, target) || shares_slot(source, target))
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

fn same_process_score_millis(distance: f64, source: &Prototype, target: &Prototype) -> u32 {
    let mut score = neighbor_score_millis(distance);
    if shares_entity(source, target) {
        score = score.saturating_add(130);
    }
    if shares_slot(source, target) {
        score = score.saturating_add(80);
    }
    if same_value(source.document_id.as_deref(), target.document_id.as_deref()) {
        score = score.saturating_add(40);
    }
    if same_value(
        source.narrative_id.as_deref(),
        target.narrative_id.as_deref(),
    ) {
        score = score.saturating_add(20);
    }
    score.min(1000)
}

fn same_process_evidence(distance: f64, source: &Prototype, target: &Prototype) -> Vec<String> {
    let mut evidence = vec![format!("ann:distance={distance:.4}")];
    if shares_entity(source, target) {
        evidence.push("shared-entity".to_owned());
    }
    if shares_slot(source, target) {
        evidence.push(format!(
            "shared-slot:{}",
            source.slot_key.as_deref().unwrap_or_default()
        ));
    }
    if same_value(source.document_id.as_deref(), target.document_id.as_deref()) {
        evidence.push("shared-document".to_owned());
    }
    if same_value(
        source.narrative_id.as_deref(),
        target.narrative_id.as_deref(),
    ) {
        evidence.push("shared-narrative".to_owned());
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
    use super::{ordered_pair, same_process_compatible, same_process_score_millis};
    use phoenix_semantic_v2::{SemanticGraphNodeKind, SemanticGraphNodeRecord};

    use crate::semantic_graph_support::{Prototype, EVENT_KIND};

    fn event_prototype(node_id: &str, slot_key: &str, entity_id: &str) -> Prototype {
        Prototype {
            node_id: node_id.to_owned(),
            ann_kind: EVENT_KIND,
            node_kind: SemanticGraphNodeKind::Event,
            text_key: node_id.to_owned(),
            text: node_id.to_owned(),
            truth_plane: Some("world".to_owned()),
            document_id: Some("doc-1".to_owned()),
            narrative_id: Some("nar-1".to_owned()),
            evidence_refs: vec![format!("evidence://{node_id}")],
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
            slot_key: Some(slot_key.to_owned()),
            value_key: Some("value".to_owned()),
            primary_entity_id: Some(entity_id.to_owned()),
            secondary_entity_id: None,
        }
    }

    #[test]
    fn same_process_prefers_shared_participant_and_scope() {
        let left = event_prototype("graph::event::1", "entity.location", "alice");
        let right = event_prototype("graph::event::2", "entity.location", "alice");

        assert!(same_process_compatible(&left, &right));
        assert!(same_process_score_millis(0.31, &left, &right) >= 900);
    }

    #[test]
    fn same_process_rejects_cross_scope_noise() {
        let left = event_prototype("graph::event::1", "entity.location", "alice");
        let mut right = event_prototype("graph::event::2", "entity.location", "alice");
        right.document_id = Some("doc-2".to_owned());
        right.narrative_id = Some("nar-2".to_owned());

        assert!(!same_process_compatible(&left, &right));
    }

    #[test]
    fn ordered_pair_is_stable_for_undirected_edges() {
        let left = event_prototype("graph::event::2", "entity.location", "alice");
        let right = event_prototype("graph::event::1", "entity.location", "alice");
        let (first, second) = ordered_pair(&left, &right).expect("ordered pair");
        assert_eq!(first.node_id, "graph::event::1");
        assert_eq!(second.node_id, "graph::event::2");
    }
}
