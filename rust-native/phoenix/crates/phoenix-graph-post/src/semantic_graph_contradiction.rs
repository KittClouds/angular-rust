use hashbrown::{HashMap, HashSet};
use phoenix_semantic_v2::MemoryScopeSidecar;
use phoenix_semantic_v2::{
    SemanticCandidateStatus, SemanticEdgeFamily, SemanticGraphEdgeCandidate, SemanticGraphNodeKind,
};

use crate::semantic_graph_contradiction_ledger::collect_relationship_ledger_edges;
use crate::semantic_graph_support::{truth_planes_compatible, Prototype};

pub(crate) fn collect_contradictory_support_region_edges(
    prototypes: &[Prototype],
    embeddings: &[Vec<f32>],
    memory_sidecar: Option<&MemoryScopeSidecar>,
    neighbor_limit: usize,
    min_score_millis: u32,
) -> Vec<SemanticGraphEdgeCandidate> {
    let prototype_by_id = prototypes
        .iter()
        .map(|prototype| (prototype.node_id.as_str(), prototype))
        .collect::<HashMap<_, _>>();
    let state_buckets = build_state_buckets(prototypes);
    let mut buckets = HashMap::<String, ContradictionBucket>::new();
    for (index, prototype) in prototypes.iter().enumerate() {
        let Some(bucket_key) = contradiction_bucket_key(prototype) else {
            continue;
        };
        let bucket = buckets.entry(bucket_key).or_default();
        match prototype.node_kind {
            SemanticGraphNodeKind::Claim => bucket.claims.push(index),
            SemanticGraphNodeKind::State => bucket.states.push(index),
            _ => {}
        }
    }
    let mut seen = HashSet::<String>::new();
    let mut edges = Vec::new();
    collect_conflict_guided_edges(
        &mut edges,
        &mut seen,
        memory_sidecar,
        &prototype_by_id,
        &state_buckets,
        prototypes,
        embeddings,
        min_score_millis,
    );
    collect_relationship_ledger_edges(
        &mut edges,
        &mut seen,
        memory_sidecar,
        &prototype_by_id,
        prototypes,
        embeddings,
        min_score_millis,
    );
    for bucket in buckets.values() {
        for &claim_index in &bucket.claims {
            let source = &prototypes[claim_index];
            let mut scored = Vec::new();
            for &state_index in &bucket.states {
                let target = &prototypes[state_index];
                if !contradiction_region_compatible(source, target) {
                    continue;
                }
                let distance =
                    embedding_distance(&embeddings[claim_index], &embeddings[state_index]);
                let score_millis = contradiction_region_score_millis(distance, source, target);
                if score_millis < min_score_millis {
                    continue;
                }
                scored.push((state_index, distance, score_millis));
            }
            scored.sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));
            scored.truncate(neighbor_limit.max(1));
            for (state_index, distance, score_millis) in scored {
                let target = &prototypes[state_index];
                let Some((left, right)) = ordered_pair(source, target) else {
                    continue;
                };
                if !seen.insert(dedupe_pair_key(left, right)) {
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
                    model_evidence: contradiction_region_evidence(distance, left, right),
                    nli_support_millis: None,
                    nli_contradiction_millis: None,
                });
            }
        }
    }
    edges.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    edges
}

#[derive(Default)]
struct ContradictionBucket {
    claims: Vec<usize>,
    states: Vec<usize>,
}

fn collect_conflict_guided_edges(
    edges: &mut Vec<SemanticGraphEdgeCandidate>,
    seen: &mut HashSet<String>,
    memory_sidecar: Option<&MemoryScopeSidecar>,
    prototype_by_id: &HashMap<&str, &Prototype>,
    state_buckets: &HashMap<String, Vec<&Prototype>>,
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
    for conflict in &memory_sidecar.conflicts {
        let bucket_key = format!("{}|{}", conflict.entity_id.0, conflict.slot_key);
        let conflict_claims = conflict
            .claim_ids
            .iter()
            .filter_map(|claim_id| {
                let claim_node_id = format!("graph::claim::{claim_id}");
                prototype_by_id.get(claim_node_id.as_str()).copied()
            })
            .collect::<Vec<_>>();
        for left_index in 0..conflict_claims.len() {
            for right_index in (left_index + 1)..conflict_claims.len() {
                let left_claim = conflict_claims[left_index];
                let right_claim = conflict_claims[right_index];
                if !conflict_claim_pair_compatible(left_claim, right_claim) {
                    continue;
                }
                let Some(left_embedding) = embedding_by_id.get(left_claim.node_id.as_str()) else {
                    continue;
                };
                let Some(right_embedding) = embedding_by_id.get(right_claim.node_id.as_str())
                else {
                    continue;
                };
                let distance = embedding_distance(left_embedding, right_embedding);
                let score_millis =
                    contradiction_region_score_millis(distance, left_claim, right_claim);
                if score_millis < min_score_millis {
                    continue;
                }
                let Some((left, right)) = ordered_pair(left_claim, right_claim) else {
                    continue;
                };
                if !seen.insert(dedupe_pair_key(left, right)) {
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
                    model_evidence: {
                        let mut evidence = contradiction_region_evidence(distance, left, right);
                        evidence.push(format!("conflict:{}", conflict.conflict_id));
                        evidence.push("conflict-claim-pair".to_owned());
                        evidence
                    },
                    nli_support_millis: None,
                    nli_contradiction_millis: None,
                });
            }
        }
        let Some(states) = state_buckets.get(bucket_key.as_str()) else {
            continue;
        };
        for claim_id in &conflict.claim_ids {
            let claim_node_id = format!("graph::claim::{claim_id}");
            let Some(claim) = prototype_by_id.get(claim_node_id.as_str()) else {
                continue;
            };
            let Some(claim_embedding) = embedding_by_id.get(claim.node_id.as_str()) else {
                continue;
            };
            for state in states {
                if !conflict_pair_compatible(claim, state, conflict.preferred_claim_id.as_deref()) {
                    continue;
                }
                let Some(state_embedding) = embedding_by_id.get(state.node_id.as_str()) else {
                    continue;
                };
                let distance = embedding_distance(claim_embedding, state_embedding);
                let score_millis = contradiction_region_score_millis(distance, claim, state);
                if score_millis < min_score_millis {
                    continue;
                }
                let Some((left, right)) = ordered_pair(claim, state) else {
                    continue;
                };
                if !seen.insert(dedupe_pair_key(left, right)) {
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
                    model_evidence: {
                        let mut evidence = contradiction_region_evidence(distance, left, right);
                        evidence.push(format!("conflict:{}", conflict.conflict_id));
                        if let Some(preferred_claim_id) = conflict.preferred_claim_id.as_deref() {
                            evidence.push(format!("preferred-claim:{preferred_claim_id}"));
                        }
                        evidence
                    },
                    nli_support_millis: None,
                    nli_contradiction_millis: None,
                });
            }
        }
    }
}

fn contradiction_region_compatible(source: &Prototype, target: &Prototype) -> bool {
    source.node_id != target.node_id
        && cross_kind_claim_state(source, target)
        && truth_planes_compatible(source.truth_plane.as_deref(), target.truth_plane.as_deref())
        && shares_scope(source, target)
        && shares_primary_entity(source, target)
        && shares_slot(source, target)
        && value_mismatch(source, target)
}

fn contradiction_bucket_key(prototype: &Prototype) -> Option<String> {
    let primary_entity_id = prototype.primary_entity_id.as_deref()?;
    let slot_key = prototype.slot_key.as_deref()?;
    matches!(
        prototype.node_kind,
        SemanticGraphNodeKind::Claim | SemanticGraphNodeKind::State
    )
    .then(|| format!("{primary_entity_id}|{slot_key}"))
}

fn build_state_buckets<'a>(prototypes: &'a [Prototype]) -> HashMap<String, Vec<&'a Prototype>> {
    let mut buckets = HashMap::<String, Vec<&Prototype>>::new();
    for prototype in prototypes {
        if prototype.node_kind != SemanticGraphNodeKind::State {
            continue;
        }
        let Some(bucket_key) = contradiction_bucket_key(prototype) else {
            continue;
        };
        buckets.entry(bucket_key).or_default().push(prototype);
    }
    buckets
}

fn cross_kind_claim_state(source: &Prototype, target: &Prototype) -> bool {
    matches!(
        (source.node_kind, target.node_kind),
        (SemanticGraphNodeKind::Claim, SemanticGraphNodeKind::State)
            | (SemanticGraphNodeKind::State, SemanticGraphNodeKind::Claim)
    )
}

fn shares_scope(source: &Prototype, target: &Prototype) -> bool {
    same_value(source.document_id.as_deref(), target.document_id.as_deref())
        || same_value(
            source.narrative_id.as_deref(),
            target.narrative_id.as_deref(),
        )
}

fn shares_primary_entity(source: &Prototype, target: &Prototype) -> bool {
    same_value(
        source.primary_entity_id.as_deref(),
        target.primary_entity_id.as_deref(),
    )
}

fn shares_slot(source: &Prototype, target: &Prototype) -> bool {
    same_value(source.slot_key.as_deref(), target.slot_key.as_deref())
}

fn value_mismatch(source: &Prototype, target: &Prototype) -> bool {
    match (source.value_key.as_deref(), target.value_key.as_deref()) {
        (Some(left), Some(right)) => left != right,
        _ => false,
    }
}

fn same_value(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

fn conflict_pair_compatible(
    claim: &Prototype,
    state: &Prototype,
    preferred_claim_id: Option<&str>,
) -> bool {
    if !contradiction_region_compatible(claim, state) {
        return false;
    }
    let Some(preferred_claim_id) = preferred_claim_id else {
        return true;
    };
    let Some(claim_id) = claim.node_id.strip_prefix("graph::claim::") else {
        return true;
    };
    claim_id != preferred_claim_id || value_mismatch(claim, state)
}

fn conflict_claim_pair_compatible(left: &Prototype, right: &Prototype) -> bool {
    left.node_id != right.node_id
        && left.node_kind == SemanticGraphNodeKind::Claim
        && right.node_kind == SemanticGraphNodeKind::Claim
        && truth_planes_compatible(left.truth_plane.as_deref(), right.truth_plane.as_deref())
        && shares_scope(left, right)
        && shares_primary_entity(left, right)
        && shares_slot(left, right)
        && value_mismatch(left, right)
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

fn dedupe_pair_key(left: &Prototype, right: &Prototype) -> String {
    format!("{}|{}", left.node_id, right.node_id)
}

fn contradiction_region_score_millis(distance: f64, source: &Prototype, target: &Prototype) -> u32 {
    let mut score = neighbor_score_millis(distance);
    if shares_primary_entity(source, target) {
        score = score.saturating_add(150);
    }
    if shares_slot(source, target) {
        score = score.saturating_add(130);
    }
    if value_mismatch(source, target) {
        score = score.saturating_add(110);
    }
    if same_value(
        source.narrative_id.as_deref(),
        target.narrative_id.as_deref(),
    ) {
        score = score.saturating_add(60);
    }
    if same_value(source.document_id.as_deref(), target.document_id.as_deref()) {
        score = score.saturating_add(30);
    }
    score.min(1000)
}

fn contradiction_region_evidence(
    distance: f64,
    source: &Prototype,
    target: &Prototype,
) -> Vec<String> {
    let mut evidence = vec![
        format!("ann:distance={distance:.4}"),
        format!(
            "pair:{}-{}",
            node_kind_name(source.node_kind),
            node_kind_name(target.node_kind)
        ),
    ];
    if shares_primary_entity(source, target) {
        evidence.push("shared-primary-entity".to_owned());
    }
    if let Some(slot_key) = source.slot_key.as_deref().or(target.slot_key.as_deref()) {
        evidence.push(format!("shared-slot:{slot_key}"));
    }
    if value_mismatch(source, target) {
        evidence.push("value-mismatch".to_owned());
    }
    if same_value(
        source.narrative_id.as_deref(),
        target.narrative_id.as_deref(),
    ) {
        evidence.push("shared-narrative".to_owned());
    }
    if same_value(source.document_id.as_deref(), target.document_id.as_deref()) {
        evidence.push("shared-document".to_owned());
    }
    evidence
}

fn node_kind_name(kind: SemanticGraphNodeKind) -> &'static str {
    match kind {
        SemanticGraphNodeKind::Claim => "claim",
        SemanticGraphNodeKind::State => "state",
        _ => "unknown",
    }
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

fn merge_refs(left: &[String], right: &[String]) -> Vec<String> {
    let mut merged = left.iter().chain(right.iter()).cloned().collect::<Vec<_>>();
    merged.sort();
    merged.dedup();
    merged
}

#[cfg(test)]
mod tests {
    use super::{contradiction_region_compatible, contradiction_region_score_millis};
    use phoenix_semantic_v2::{SemanticGraphNodeKind, SemanticGraphNodeRecord};

    use crate::semantic_graph_support::{Prototype, CLAIM_KIND, STATE_KIND};

    fn prototype(
        node_id: &str,
        ann_kind: &'static str,
        node_kind: SemanticGraphNodeKind,
        value_key: &str,
    ) -> Prototype {
        Prototype {
            node_id: node_id.to_owned(),
            ann_kind,
            node_kind,
            text_key: node_id.to_owned(),
            text: node_id.to_owned(),
            truth_plane: Some("world".to_owned()),
            document_id: Some("doc-1".to_owned()),
            narrative_id: Some("nar-1".to_owned()),
            evidence_refs: vec![format!("evidence://{node_id}")],
            semantic_node: SemanticGraphNodeRecord {
                node_id: node_id.to_owned(),
                node_kind,
                document_id: Some("doc-1".to_owned()),
                narrative_id: Some("nar-1".to_owned()),
                text_key: node_id.to_owned(),
                text_hash: 1,
                truth_plane: Some("world".to_owned()),
                evidence_refs: Vec::new(),
            },
            slot_key: Some("entity.employer".to_owned()),
            value_key: Some(value_key.to_owned()),
            primary_entity_id: Some("alice".to_owned()),
            secondary_entity_id: None,
        }
    }

    #[test]
    fn contradiction_region_accepts_claim_state_value_mismatch() {
        let claim = prototype(
            "graph::claim::1",
            CLAIM_KIND,
            SemanticGraphNodeKind::Claim,
            "acme",
        );
        let state = prototype(
            "graph::state::1",
            STATE_KIND,
            SemanticGraphNodeKind::State,
            "globex",
        );

        assert!(contradiction_region_compatible(&claim, &state));
        assert!(contradiction_region_score_millis(0.26, &claim, &state) >= 900);
    }

    #[test]
    fn contradiction_region_rejects_matching_values() {
        let claim = prototype(
            "graph::claim::1",
            CLAIM_KIND,
            SemanticGraphNodeKind::Claim,
            "acme",
        );
        let state = prototype(
            "graph::state::1",
            STATE_KIND,
            SemanticGraphNodeKind::State,
            "acme",
        );

        assert!(!contradiction_region_compatible(&claim, &state));
    }

    #[test]
    fn contradiction_region_rejects_cross_entity_noise() {
        let claim = prototype(
            "graph::claim::1",
            CLAIM_KIND,
            SemanticGraphNodeKind::Claim,
            "acme",
        );
        let mut state = prototype(
            "graph::state::1",
            STATE_KIND,
            SemanticGraphNodeKind::State,
            "globex",
        );
        state.primary_entity_id = Some("bob".to_owned());

        assert!(!contradiction_region_compatible(&claim, &state));
    }
}
