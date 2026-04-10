use hashbrown::{HashMap, HashSet};
use phoenix_graph_kernel::KernelMutationBatch;
use phoenix_graph_kernel::{KernelBiTemporal, KernelEdge};
use phoenix_semantic_v2::{
    GraphScopeSidecar, SemanticCandidateStatus, SemanticEdgeFamily, SemanticGraphEdgeCandidate,
    SemanticGraphNodeKind,
};
use phoenix_store_native_core::PhoenixSemanticIndexStore;
use phoenix_types::ScopeKey;

use crate::semantic_graph::SemanticGraphError;
use crate::semantic_graph_support::{truth_planes_compatible, Prototype, EVENT_KIND};

pub(crate) fn collect_missing_intermediate_cause_edges_from_store<S>(
    store: &S,
    scope: &ScopeKey,
    prototypes: &[Prototype],
    embeddings: &[Vec<f32>],
    graph_sidecar: Option<&GraphScopeSidecar>,
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
    let direct_pairs = graph_sidecar
        .map(|sidecar| direct_causal_pairs(&sidecar.graph_batch))
        .unwrap_or_default();
    let temporal_by_id = graph_sidecar
        .map(|sidecar| temporal_by_vertex_id(&sidecar.graph_batch))
        .unwrap_or_default();
    let mut seen = HashSet::new();
    let mut edges = Vec::new();
    for (source, embedding) in prototypes.iter().zip(embeddings.iter()) {
        if source.node_kind != SemanticGraphNodeKind::Event {
            continue;
        }
        let hits = store.query_semantic_node_neighbors(
            embedding,
            scope,
            EVENT_KIND,
            Some(&source.node_id),
            neighbor_limit,
            oversample,
        )?;
        for hit in hits {
            let Some(target) = prototype_by_id.get(hit.node_id.as_str()) else {
                continue;
            };
            let Some((left, right, temporal_gap)) =
                missing_cause_compatible(source, target, &direct_pairs, &temporal_by_id)
            else {
                continue;
            };
            if !seen.insert((left.node_id.as_str(), right.node_id.as_str())) {
                continue;
            }
            let score_millis = missing_cause_score_millis(hit.distance, left, right, temporal_gap);
            if score_millis < min_score_millis {
                continue;
            }
            edges.push(SemanticGraphEdgeCandidate {
                edge_id: format!(
                    "semantic:missing_intermediate_cause:{}:{}",
                    left.node_id, right.node_id
                ),
                family: SemanticEdgeFamily::MissingIntermediateCause,
                source_node_id: left.node_id.clone(),
                source_kind: left.node_kind,
                target_node_id: right.node_id.clone(),
                target_kind: right.node_kind,
                score_millis,
                distance_millis: (hit.distance.max(0.0) * 1000.0).round() as u32,
                candidate_status: SemanticCandidateStatus::Generated,
                evidence_refs: merge_refs(&left.evidence_refs, &right.evidence_refs),
                model_evidence: vec![
                    format!("ann:distance={:.4}", hit.distance),
                    "causal-gap:no-direct-causal-link".to_owned(),
                    "temporal-order:forward".to_owned(),
                    format!("temporal-gap-ms:{temporal_gap}"),
                ],
                nli_support_millis: None,
                nli_contradiction_millis: None,
            });
        }
    }
    edges.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    Ok(edges)
}

fn missing_cause_compatible<'a>(
    source: &'a Prototype,
    target: &'a Prototype,
    direct_pairs: &HashSet<(String, String)>,
    temporal_by_id: &HashMap<String, KernelBiTemporal>,
) -> Option<(&'a Prototype, &'a Prototype, i64)> {
    if source.node_kind != SemanticGraphNodeKind::Event
        || target.node_kind != SemanticGraphNodeKind::Event
        || source.node_id == target.node_id
        || !truth_planes_compatible(source.truth_plane.as_deref(), target.truth_plane.as_deref())
        || !shares_scope(source, target)
        || !shares_entity(source, target)
    {
        return None;
    }
    let (left, right) = ordered_pair(source, target)?;
    if direct_pairs.contains(&(left.node_id.clone(), right.node_id.clone())) {
        return None;
    }
    let temporal_gap = forward_temporal_gap(left, right, temporal_by_id)?;
    if !(1..=86_400_000).contains(&temporal_gap) {
        return None;
    }
    Some((left, right, temporal_gap))
}

fn direct_causal_pairs(batch: &KernelMutationBatch) -> HashSet<(String, String)> {
    batch
        .edges
        .iter()
        .filter(|edge| edge.edge_type.0 == "causal_link")
        .map(|edge| ordered_ids(edge))
        .collect()
}

fn temporal_by_vertex_id(batch: &KernelMutationBatch) -> HashMap<String, KernelBiTemporal> {
    batch
        .vertices
        .iter()
        .filter(|vertex| vertex.kind == "event")
        .map(|vertex| (vertex.id.0.clone(), vertex.temporal.clone()))
        .collect()
}

fn forward_temporal_gap(
    left: &Prototype,
    right: &Prototype,
    temporal_by_id: &HashMap<String, KernelBiTemporal>,
) -> Option<i64> {
    let left_temporal = temporal_by_id.get(left.node_id.as_str())?;
    let right_temporal = temporal_by_id.get(right.node_id.as_str())?;
    let left_start = left_temporal.valid_from?;
    let right_start = right_temporal.valid_from?;
    (left_start < right_start).then_some(right_start - left_start)
}

fn ordered_ids(edge: &KernelEdge) -> (String, String) {
    if edge.source_id.0 <= edge.target_id.0 {
        (edge.source_id.0.clone(), edge.target_id.0.clone())
    } else {
        (edge.target_id.0.clone(), edge.source_id.0.clone())
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

fn shares_scope(source: &Prototype, target: &Prototype) -> bool {
    same_value(source.document_id.as_deref(), target.document_id.as_deref())
        || same_value(
            source.narrative_id.as_deref(),
            target.narrative_id.as_deref(),
        )
}

fn shares_entity(source: &Prototype, target: &Prototype) -> bool {
    same_value(
        source.primary_entity_id.as_deref(),
        target.primary_entity_id.as_deref(),
    ) || same_value(
        source.primary_entity_id.as_deref(),
        target.secondary_entity_id.as_deref(),
    ) || same_value(
        source.secondary_entity_id.as_deref(),
        target.primary_entity_id.as_deref(),
    ) || same_value(
        source.secondary_entity_id.as_deref(),
        target.secondary_entity_id.as_deref(),
    )
}

fn missing_cause_score_millis(
    distance: f64,
    left: &Prototype,
    right: &Prototype,
    temporal_gap: i64,
) -> u32 {
    let mut score = neighbor_score_millis(distance);
    if same_value(left.narrative_id.as_deref(), right.narrative_id.as_deref()) {
        score = score.saturating_add(100);
    }
    if same_value(left.document_id.as_deref(), right.document_id.as_deref()) {
        score = score.saturating_add(60);
    }
    if shares_entity(left, right) {
        score = score.saturating_add(140);
    }
    if temporal_gap <= 7_200_000 {
        score = score.saturating_add(70);
    } else if temporal_gap <= 21_600_000 {
        score = score.saturating_add(35);
    }
    score.min(880)
}

fn neighbor_score_millis(distance: f64) -> u32 {
    ((1.0 / (1.0 + distance.max(0.0))) * 1000.0)
        .round()
        .clamp(0.0, 1000.0) as u32
}

fn same_value(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

fn merge_refs(left: &[String], right: &[String]) -> Vec<String> {
    let mut merged = left.iter().chain(right.iter()).cloned().collect::<Vec<_>>();
    merged.sort();
    merged.dedup();
    merged
}

#[cfg(test)]
mod tests {
    use hashbrown::{HashMap, HashSet};
    use phoenix_graph_kernel::{
        KernelBiTemporal, KernelEdge, KernelEdgeType, KernelGraphLayer, KernelMutationBatch,
        KernelProvenance, KernelRelationClass, KernelVertex, KernelVertexClass, KernelVertexId,
    };
    use phoenix_semantic_v2::{SemanticGraphNodeKind, SemanticGraphNodeRecord};
    use serde_json::json;

    use super::{direct_causal_pairs, missing_cause_compatible};
    use crate::semantic_graph_support::{Prototype, EVENT_KIND};

    fn event(node_id: &str, entity: &str) -> Prototype {
        Prototype {
            node_id: node_id.to_owned(),
            ann_kind: EVENT_KIND,
            node_kind: SemanticGraphNodeKind::Event,
            text_key: node_id.to_owned(),
            text: node_id.to_owned(),
            truth_plane: Some("world".to_owned()),
            document_id: Some("doc-1".to_owned()),
            narrative_id: Some("nar-1".to_owned()),
            evidence_refs: vec![format!("ev://{node_id}")],
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
            primary_entity_id: Some(entity.to_owned()),
            secondary_entity_id: None,
        }
    }

    #[test]
    fn missing_cause_accepts_temporally_ordered_non_direct_pair() {
        let left = event("graph::event::memory::a", "alice");
        let right = event("graph::event::memory::b", "alice");
        let temporal = HashMap::from([
            (
                left.node_id.clone(),
                KernelBiTemporal {
                    valid_from: Some(10),
                    ..Default::default()
                },
            ),
            (
                right.node_id.clone(),
                KernelBiTemporal {
                    valid_from: Some(30),
                    ..Default::default()
                },
            ),
        ]);

        let result = missing_cause_compatible(&left, &right, &HashSet::new(), &temporal);
        assert!(result.is_some());
    }

    #[test]
    fn missing_cause_rejects_direct_causal_pair() {
        let left = event("graph::event::memory::a", "alice");
        let right = event("graph::event::memory::b", "alice");
        let temporal = HashMap::from([
            (
                left.node_id.clone(),
                KernelBiTemporal {
                    valid_from: Some(10),
                    ..Default::default()
                },
            ),
            (
                right.node_id.clone(),
                KernelBiTemporal {
                    valid_from: Some(20),
                    ..Default::default()
                },
            ),
        ]);
        let direct = HashSet::from([(left.node_id.clone(), right.node_id.clone())]);

        assert!(missing_cause_compatible(&left, &right, &direct, &temporal).is_none());
    }

    #[test]
    fn direct_causal_pairs_extracts_causal_links() {
        let batch = KernelMutationBatch {
            edges: vec![KernelEdge {
                source_id: KernelVertexId("graph::event::memory::a".to_owned()),
                target_id: KernelVertexId("graph::event::memory::b".to_owned()),
                edge_type: KernelEdgeType("causal_link".to_owned()),
                relation_class: KernelRelationClass::Semantic,
                weight: 1,
                attributes: json!({}),
                data: None,
                document_id: None,
                narrative_id: None,
                layer: KernelGraphLayer::Asserted,
                temporal: KernelBiTemporal::default(),
                provenance: KernelProvenance::default(),
                resolution_facet: None,
            }],
            vertices: vec![KernelVertex {
                id: KernelVertexId("graph::event::memory::a".to_owned()),
                kind: "event".to_owned(),
                class: KernelVertexClass::Event,
                labels: Vec::new(),
                weight: 1,
                value: json!({}),
                attributes: json!({}),
                temporal: KernelBiTemporal::default(),
                provenance: KernelProvenance::default(),
                entity_id: None,
                search_chunk_id: None,
                document_id: None,
                chapter_id: None,
                chapters: Vec::new(),
                boundary_id: None,
                boundary_ordinal: None,
                boundary_kind: None,
                boundary_ordinals: Vec::new(),
                entity_facet: None,
                calendar_facet: None,
            }],
            ..Default::default()
        };

        assert!(direct_causal_pairs(&batch).contains(&(
            "graph::event::memory::a".to_owned(),
            "graph::event::memory::b".to_owned()
        )));
    }
}
