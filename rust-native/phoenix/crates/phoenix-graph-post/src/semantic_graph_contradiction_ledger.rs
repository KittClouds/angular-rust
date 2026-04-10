use hashbrown::{HashMap, HashSet};
use phoenix_semantic_v2::{
    MemoryScopeSidecar, SemanticCandidateStatus, SemanticEdgeFamily, SemanticGraphEdgeCandidate,
};

use crate::semantic_graph_support::{truth_planes_compatible, Prototype};

pub(super) fn collect_relationship_ledger_edges(
    edges: &mut Vec<SemanticGraphEdgeCandidate>,
    seen: &mut HashSet<String>,
    memory_sidecar: Option<&MemoryScopeSidecar>,
    prototype_by_id: &HashMap<&str, &Prototype>,
    prototypes: &[Prototype],
    embeddings: &[Vec<f32>],
    min_score_millis: u32,
) {
    let Some(memory_sidecar) = memory_sidecar else {
        return;
    };
    let embedding_by_id = prototypes
        .iter()
        .zip(embeddings.iter())
        .map(|(prototype, embedding)| (prototype.node_id.as_str(), embedding.as_slice()))
        .collect::<HashMap<_, _>>();
    for ledger in &memory_sidecar.relationship_ledgers {
        if ledger.supporting_claim_ids.is_empty() || ledger.contradicting_claim_ids.is_empty() {
            continue;
        }
        for support_id in &ledger.supporting_claim_ids {
            let support_node_id = format!("graph::claim::{support_id}");
            let Some(support) = prototype_by_id.get(support_node_id.as_str()).copied() else {
                continue;
            };
            let Some(support_embedding) = embedding_by_id.get(support.node_id.as_str()) else {
                continue;
            };
            for contradiction_id in &ledger.contradicting_claim_ids {
                let contradiction_node_id = format!("graph::claim::{contradiction_id}");
                let Some(contradiction) =
                    prototype_by_id.get(contradiction_node_id.as_str()).copied()
                else {
                    continue;
                };
                if !ledger_pair_compatible(support, contradiction, &ledger.ledger_id) {
                    continue;
                }
                let Some(contradiction_embedding) =
                    embedding_by_id.get(contradiction.node_id.as_str())
                else {
                    continue;
                };
                let distance = embedding_distance(support_embedding, contradiction_embedding);
                let score_millis = ledger_score_millis(distance, support, contradiction);
                if score_millis < min_score_millis {
                    continue;
                }
                let (left, right) = ordered_pair(support, contradiction);
                let dedupe_key = format!("{}|{}", left.node_id, right.node_id);
                if !seen.insert(dedupe_key) {
                    continue;
                }
                edges.push(SemanticGraphEdgeCandidate {
                    edge_id: format!(
                        "semantic:contradictory_support_region:{}:{}",
                        left.node_id, right.node_id
                    ),
                    family: SemanticEdgeFamily::ContradictorySupportRegion,
                    source_node_id: left.node_id.clone(),
                    source_kind: left.node_kind,
                    target_node_id: right.node_id.clone(),
                    target_kind: right.node_kind,
                    score_millis,
                    distance_millis: (distance.max(0.0) * 1000.0).round() as u32,
                    candidate_status: SemanticCandidateStatus::Generated,
                    evidence_refs: merge_refs(&left.evidence_refs, &right.evidence_refs),
                    model_evidence: vec![
                        format!("ann:distance={distance:.4}"),
                        format!("relationship-ledger:{}", ledger.ledger_id),
                        "support-vs-contradiction".to_owned(),
                        format!(
                            "shared-relation:{}",
                            left.slot_key.as_deref().unwrap_or_default()
                        ),
                    ],
                    nli_support_millis: None,
                    nli_contradiction_millis: None,
                });
            }
        }
    }
}

fn ledger_pair_compatible(left: &Prototype, right: &Prototype, ledger_id: &str) -> bool {
    left.node_id != right.node_id
        && truth_planes_compatible(left.truth_plane.as_deref(), right.truth_plane.as_deref())
        && same_value(left.document_id.as_deref(), right.document_id.as_deref())
        && same_value(
            left.primary_entity_id.as_deref(),
            right.primary_entity_id.as_deref(),
        )
        && same_value(
            left.secondary_entity_id.as_deref(),
            right.secondary_entity_id.as_deref(),
        )
        && same_value(left.slot_key.as_deref(), right.slot_key.as_deref())
        && ledger_id.contains(left.secondary_entity_id.as_deref().unwrap_or_default())
}

fn ordered_pair<'a>(left: &'a Prototype, right: &'a Prototype) -> (&'a Prototype, &'a Prototype) {
    if left.node_id <= right.node_id {
        (left, right)
    } else {
        (right, left)
    }
}

fn ledger_score_millis(distance: f64, left: &Prototype, right: &Prototype) -> u32 {
    let mut score = neighbor_score_millis(distance);
    if same_value(
        left.primary_entity_id.as_deref(),
        right.primary_entity_id.as_deref(),
    ) {
        score = score.saturating_add(170);
    }
    if same_value(
        left.secondary_entity_id.as_deref(),
        right.secondary_entity_id.as_deref(),
    ) {
        score = score.saturating_add(170);
    }
    if same_value(left.slot_key.as_deref(), right.slot_key.as_deref()) {
        score = score.saturating_add(130);
    }
    if same_value(left.narrative_id.as_deref(), right.narrative_id.as_deref()) {
        score = score.saturating_add(40);
    }
    if same_value(left.document_id.as_deref(), right.document_id.as_deref()) {
        score = score.saturating_add(30);
    }
    score.min(1000)
}

fn embedding_distance(left: &[f32], right: &[f32]) -> f64 {
    let len = left.len().min(right.len());
    let mut sum = 0.0f64;
    for index in 0..len {
        let delta = left[index] as f64 - right[index] as f64;
        sum += delta * delta;
    }
    sum.sqrt()
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
    use phoenix_semantic_v2::{
        MemoryClaimStatus, MemoryScopeSidecar, RelationshipMemoryLedger, SemanticGraphNodeKind,
        SemanticGraphNodeRecord,
    };
    use phoenix_types::{BiTemporalWindow, EntityId};

    use super::collect_relationship_ledger_edges;
    use crate::semantic_graph_support::{Prototype, CLAIM_KIND};

    fn claim(node_id: &str, claim_id: &str) -> Prototype {
        Prototype {
            node_id: node_id.to_owned(),
            ann_kind: CLAIM_KIND,
            node_kind: SemanticGraphNodeKind::Claim,
            text_key: node_id.to_owned(),
            text: claim_id.to_owned(),
            truth_plane: Some("world".to_owned()),
            document_id: Some("doc-1".to_owned()),
            narrative_id: Some("nar-1".to_owned()),
            evidence_refs: vec![format!("evidence://{claim_id}")],
            semantic_node: SemanticGraphNodeRecord {
                node_id: node_id.to_owned(),
                node_kind: SemanticGraphNodeKind::Claim,
                document_id: Some("doc-1".to_owned()),
                narrative_id: Some("nar-1".to_owned()),
                text_key: node_id.to_owned(),
                text_hash: 1,
                truth_plane: Some("world".to_owned()),
                evidence_refs: Vec::new(),
            },
            slot_key: Some("relation.commands".to_owned()),
            value_key: Some("bob".to_owned()),
            primary_entity_id: Some("alice".to_owned()),
            secondary_entity_id: Some("bob".to_owned()),
        }
    }

    #[test]
    fn relationship_ledgers_emit_contradiction_region_edges() {
        let prototypes = vec![
            claim("graph::claim::support", "support"),
            claim("graph::claim::contradiction", "contradiction"),
        ];
        let embeddings = vec![vec![0.1, 0.2, 0.3], vec![0.11, 0.2, 0.29]];
        let prototype_by_id = prototypes
            .iter()
            .map(|prototype| (prototype.node_id.as_str(), prototype))
            .collect::<HashMap<_, _>>();
        let memory_sidecar = MemoryScopeSidecar {
            relationship_ledgers: vec![RelationshipMemoryLedger {
                ledger_id: "relationship:commands:alice:bob".to_owned(),
                relation_family: "commands".to_owned(),
                source_entity_id: EntityId("alice".to_owned()),
                target_entity_id: EntityId("bob".to_owned()),
                current_status: MemoryClaimStatus::Deferred,
                temporal: BiTemporalWindow::default(),
                supporting_claim_ids: vec!["support".to_owned()],
                contradicting_claim_ids: vec!["contradiction".to_owned()],
            }],
            ..Default::default()
        };
        let mut edges = Vec::new();
        let mut seen = HashSet::new();

        collect_relationship_ledger_edges(
            &mut edges,
            &mut seen,
            Some(&memory_sidecar),
            &prototype_by_id,
            &prototypes,
            &embeddings,
            500,
        );

        assert_eq!(edges.len(), 1);
        assert!(edges[0]
            .model_evidence
            .iter()
            .any(|value| value.contains("relationship-ledger:relationship:commands:alice:bob")));
    }
}
