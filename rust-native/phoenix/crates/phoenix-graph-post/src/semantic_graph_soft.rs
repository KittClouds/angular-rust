use hashbrown::HashMap;
use phoenix_semantic_v2::{
    SemanticCandidateStatus, SemanticEdgeFamily, SemanticGraphEdgeCandidate, SemanticGraphNodeKind,
};
use phoenix_store_native_core::SemanticNodeNeighbor;

use crate::semantic_graph_support::{truth_planes_compatible, Prototype, STATE_KIND};
use crate::semantic_graph_workspace::SemanticNeighborWorkspace;

pub(crate) fn collect_same_slot_family_edges(
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
    let mut edges = Vec::new();
    for (source_index, source) in prototypes.iter().enumerate() {
        if source.node_kind != SemanticGraphNodeKind::Claim {
            continue;
        }
        let Some(slot_key) = source.slot_key.as_deref() else {
            continue;
        };
        let hits = workspace.query_semantic_node_neighbors(
            source_index,
            STATE_KIND,
            neighbor_limit,
            oversample,
        );
        collect_same_slot_family_edges_from_hits(
            &mut edges,
            source,
            slot_key,
            hits.as_slice(),
            &prototype_by_id,
            min_score_millis,
        );
    }
    edges.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    edges
}

fn collect_same_slot_family_edges_from_hits(
    edges: &mut Vec<SemanticGraphEdgeCandidate>,
    source: &Prototype,
    slot_key: &str,
    hits: &[SemanticNodeNeighbor],
    prototype_by_id: &HashMap<&str, &Prototype>,
    min_score_millis: u32,
) {
    for hit in hits {
        let Some(target) = prototype_by_id.get(hit.node_id.as_str()) else {
            continue;
        };
        if target.node_kind != SemanticGraphNodeKind::State
            || target.slot_key.as_deref() != Some(slot_key)
            || !truth_planes_compatible(
                source.truth_plane.as_deref(),
                target.truth_plane.as_deref(),
            )
        {
            continue;
        }
        let score_millis = same_slot_family_score_millis(hit.distance, source, target);
        if score_millis < min_score_millis {
            continue;
        }
        let mut model_evidence = vec![
            format!("ann:distance={:.4}", hit.distance),
            format!("slot-key:{slot_key}"),
        ];
        if source.value_key == target.value_key {
            model_evidence.push("value-key-match".to_owned());
        }
        edges.push(SemanticGraphEdgeCandidate {
            edge_id: format!(
                "semantic:same_slot_family:{}:{}",
                source.node_id, target.node_id
            ),
            family: SemanticEdgeFamily::SameSlotFamily,
            source_node_id: source.node_id.clone(),
            source_kind: source.node_kind,
            target_node_id: target.node_id.clone(),
            target_kind: target.node_kind,
            score_millis,
            distance_millis: (hit.distance.max(0.0) * 1000.0).round() as u32,
            candidate_status: SemanticCandidateStatus::Generated,
            evidence_refs: merge_refs(&source.evidence_refs, &target.evidence_refs),
            model_evidence,
            nli_support_millis: None,
            nli_contradiction_millis: None,
        });
    }
}

fn same_slot_family_score_millis(distance: f64, source: &Prototype, target: &Prototype) -> u32 {
    let base = neighbor_score_millis(distance);
    let bonus = if source.value_key == target.value_key {
        140
    } else {
        70
    };
    base.saturating_add(bonus).min(1000)
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
    use super::collect_same_slot_family_edges_from_hits;
    use hashbrown::HashMap;
    use phoenix_semantic_v2::{SemanticEdgeFamily, SemanticGraphNodeRecord};
    use phoenix_store_native_core::SemanticNodeNeighbor;

    use crate::semantic_graph_support::{Prototype, CLAIM_KIND, STATE_KIND};

    fn prototype(
        node_id: &str,
        ann_kind: &'static str,
        node_kind: phoenix_semantic_v2::SemanticGraphNodeKind,
        slot_key: &str,
        value_key: &str,
    ) -> Prototype {
        Prototype {
            node_id: node_id.to_owned(),
            ann_kind,
            node_kind,
            text_key: node_id.to_owned(),
            text: node_id.to_owned(),
            truth_plane: Some("world".to_owned()),
            document_id: None,
            narrative_id: None,
            evidence_refs: vec![format!("evidence://{node_id}")],
            semantic_node: SemanticGraphNodeRecord {
                node_id: node_id.to_owned(),
                node_kind,
                document_id: None,
                narrative_id: None,
                text_key: node_id.to_owned(),
                text_hash: 1,
                truth_plane: Some("world".to_owned()),
                evidence_refs: Vec::new(),
            },
            slot_key: Some(slot_key.to_owned()),
            value_key: Some(value_key.to_owned()),
            primary_entity_id: None,
            secondary_entity_id: None,
        }
    }

    #[test]
    fn same_slot_family_edges_bridge_claims_into_states() {
        let source = prototype(
            "graph::claim::1",
            CLAIM_KIND,
            phoenix_semantic_v2::SemanticGraphNodeKind::Claim,
            "entity.employer",
            "acme",
        );
        let target = prototype(
            "graph::state::1",
            STATE_KIND,
            phoenix_semantic_v2::SemanticGraphNodeKind::State,
            "entity.employer",
            "acme",
        );
        let by_id = HashMap::from([
            (source.node_id.as_str(), &source),
            (target.node_id.as_str(), &target),
        ]);
        let hits = vec![SemanticNodeNeighbor {
            node_id: target.node_id.clone(),
            node_kind: STATE_KIND.to_owned(),
            distance: 0.22,
            document_id: None,
            narrative_id: None,
            folder_id: None,
            evidence_refs: vec!["evidence://neighbor".to_owned()],
        }];
        let mut edges = Vec::new();

        collect_same_slot_family_edges_from_hits(
            &mut edges,
            &source,
            "entity.employer",
            hits.as_slice(),
            &by_id,
            540,
        );

        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].family, SemanticEdgeFamily::SameSlotFamily);
        assert_eq!(edges[0].source_node_id, source.node_id);
        assert_eq!(edges[0].target_node_id, target.node_id);
        assert!(edges[0].score_millis >= 900);
    }

    #[test]
    fn same_slot_family_edges_skip_mismatched_slot_targets() {
        let source = prototype(
            "graph::claim::1",
            CLAIM_KIND,
            phoenix_semantic_v2::SemanticGraphNodeKind::Claim,
            "entity.employer",
            "acme",
        );
        let target = prototype(
            "graph::state::1",
            STATE_KIND,
            phoenix_semantic_v2::SemanticGraphNodeKind::State,
            "entity.location",
            "london",
        );
        let by_id = HashMap::from([
            (source.node_id.as_str(), &source),
            (target.node_id.as_str(), &target),
        ]);
        let hits = vec![SemanticNodeNeighbor {
            node_id: target.node_id.clone(),
            node_kind: STATE_KIND.to_owned(),
            distance: 0.14,
            document_id: None,
            narrative_id: None,
            folder_id: None,
            evidence_refs: Vec::new(),
        }];
        let mut edges = Vec::new();

        collect_same_slot_family_edges_from_hits(
            &mut edges,
            &source,
            "entity.employer",
            hits.as_slice(),
            &by_id,
            540,
        );

        assert!(edges.is_empty());
    }
}
