use crate::{KernelBiTemporal, KernelEdge, KernelGraphLayer, KernelGraphSnapshot, KernelVertex};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

const MAX_INCOMING_PER_VERTEX: usize = 8;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelCausalPathFeatures {
    pub depth: usize,
    pub avg_confidence: f64,
    pub evidence_ref_count: usize,
    pub candidate_edge_count: usize,
    pub missing_intermediate_cause_count: usize,
    pub temporal_consistency_ratio: f64,
    pub path_stability: f64,
    pub support_strength: f64,
    pub pattern_strength: f64,
    pub path_span_ms: i64,
}

#[derive(Clone, Debug)]
pub struct KernelCausalPathCandidateView<'a> {
    pub source_vertex_id: &'a str,
    pub path_vertex_ids: Vec<&'a str>,
    pub path_edges: Vec<&'a KernelEdge>,
    pub supporting_modalities: Vec<String>,
    pub features: KernelCausalPathFeatures,
}

pub fn causal_path_candidate_views_from_snapshot<'a>(
    snapshot: &'a KernelGraphSnapshot,
    target_vertex_id: &str,
    max_depth: usize,
    limit: usize,
) -> Vec<KernelCausalPathCandidateView<'a>> {
    let max_depth = max_depth.clamp(1, 5);
    let max_candidates = limit.saturating_mul(4).clamp(12, 64);
    let vertex_by_id = snapshot
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id.0.as_str(), (index, vertex)))
        .collect::<FxHashMap<_, _>>();
    let Some(&(target_index, _)) = vertex_by_id.get(target_vertex_id) else {
        return Vec::new();
    };

    let all_edges = snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
        .collect::<Vec<_>>();
    let incoming = incoming_causal_index(&all_edges, &vertex_by_id, snapshot.vertices.len());
    let outgoing = outgoing_index(&all_edges, &vertex_by_id, snapshot.vertices.len());

    let mut visited = VisitedBits::new(snapshot.vertices.len());
    visited.insert(target_index);
    let mut stack = vec![TraversalFrame {
        current_index: target_index,
        vertex_path_rev: vec![target_index],
        edge_path_rev: Vec::new(),
        visited,
    }];
    let mut candidates = Vec::new();

    while let Some(frame) = stack.pop() {
        let Some(edges) = incoming.get(frame.current_index) else {
            continue;
        };
        for &edge_index in edges {
            let edge = all_edges[edge_index];
            let Some(&(source_index, _)) = vertex_by_id.get(edge.source_id.0.as_str()) else {
                continue;
            };
            if frame.visited.contains(source_index) {
                continue;
            }

            let mut vertex_path_rev = frame.vertex_path_rev.clone();
            vertex_path_rev.push(source_index);
            let mut edge_path_rev = frame.edge_path_rev.clone();
            edge_path_rev.push(edge_index);

            candidates.push(candidate_view(
                &vertex_path_rev,
                &edge_path_rev,
                &snapshot.vertices,
                &all_edges,
                &outgoing,
                &vertex_by_id,
            ));
            if candidates.len() >= max_candidates {
                break;
            }
            if edge_path_rev.len() < max_depth {
                let mut visited = frame.visited.clone();
                visited.insert(source_index);
                stack.push(TraversalFrame {
                    current_index: source_index,
                    vertex_path_rev,
                    edge_path_rev,
                    visited,
                });
            }
        }
        if candidates.len() >= max_candidates {
            break;
        }
    }

    candidates.sort_by(|left, right| left.source_vertex_id.cmp(right.source_vertex_id));
    candidates.truncate(limit.max(1));
    candidates
}

#[derive(Clone)]
struct TraversalFrame {
    current_index: usize,
    vertex_path_rev: Vec<usize>,
    edge_path_rev: Vec<usize>,
    visited: VisitedBits,
}

fn incoming_causal_index(
    all_edges: &[&KernelEdge],
    vertex_by_id: &FxHashMap<&str, (usize, &KernelVertex)>,
    vertex_count: usize,
) -> Vec<Vec<usize>> {
    let mut incoming = vec![Vec::<usize>::new(); vertex_count];
    for (edge_index, edge) in all_edges.iter().enumerate() {
        if edge.edge_type.0 != "causal_link" {
            continue;
        }
        let Some(&(target_index, _)) = vertex_by_id.get(edge.target_id.0.as_str()) else {
            continue;
        };
        incoming[target_index].push(edge_index);
    }
    for edges in incoming.iter_mut() {
        edges.sort_by(|left, right| {
            let left_edge = all_edges[*left];
            let right_edge = all_edges[*right];
            right_edge
                .provenance
                .confidence
                .partial_cmp(&left_edge.provenance.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left_edge.source_id.0.cmp(&right_edge.source_id.0))
        });
        edges.truncate(MAX_INCOMING_PER_VERTEX);
    }
    incoming
}

fn outgoing_index(
    all_edges: &[&KernelEdge],
    vertex_by_id: &FxHashMap<&str, (usize, &KernelVertex)>,
    vertex_count: usize,
) -> Vec<Vec<usize>> {
    let mut outgoing = vec![Vec::<usize>::new(); vertex_count];
    for (edge_index, edge) in all_edges.iter().enumerate() {
        let Some(&(source_index, _)) = vertex_by_id.get(edge.source_id.0.as_str()) else {
            continue;
        };
        outgoing[source_index].push(edge_index);
    }
    outgoing
}

fn candidate_view<'a>(
    vertex_path_rev: &[usize],
    edge_path_rev: &[usize],
    vertices: &'a [KernelVertex],
    all_edges: &[&'a KernelEdge],
    outgoing: &[Vec<usize>],
    vertex_by_id: &FxHashMap<&str, (usize, &KernelVertex)>,
) -> KernelCausalPathCandidateView<'a> {
    let path_vertex_indices = vertex_path_rev.iter().rev().copied().collect::<Vec<_>>();
    let path_vertex_ids = path_vertex_indices
        .iter()
        .map(|&index| vertices[index].id.0.as_str())
        .collect::<Vec<_>>();
    let source_vertex_id = path_vertex_ids.first().copied().unwrap_or_default();
    let path_edges = edge_path_rev
        .iter()
        .rev()
        .map(|&edge_index| all_edges[edge_index])
        .collect::<Vec<_>>();
    let mut supporting_modalities =
        path_modalities(vertex_path_rev, vertices, all_edges, outgoing, vertex_by_id);
    supporting_modalities.sort();
    supporting_modalities.dedup();
    let features = path_features(
        path_vertex_indices.as_slice(),
        path_edges.as_slice(),
        vertices,
    );
    KernelCausalPathCandidateView {
        source_vertex_id,
        path_vertex_ids,
        path_edges,
        supporting_modalities,
        features,
    }
}

fn path_features(
    path_vertex_indices: &[usize],
    path_edges: &[&KernelEdge],
    vertices: &[KernelVertex],
) -> KernelCausalPathFeatures {
    if path_edges.is_empty() {
        return KernelCausalPathFeatures {
            temporal_consistency_ratio: 1.0,
            ..KernelCausalPathFeatures::default()
        };
    }
    let evidence_ref_count = path_edges
        .iter()
        .map(|edge| edge.provenance.evidence_refs.len())
        .sum::<usize>();
    let avg_confidence = path_edges
        .iter()
        .map(|edge| edge.provenance.confidence.unwrap_or(0.5))
        .sum::<f64>()
        / path_edges.len() as f64;
    let candidate_edge_count = path_edges
        .iter()
        .filter(|edge| matches!(edge.layer, KernelGraphLayer::Candidate))
        .count();
    let missing_intermediate_cause_count = path_edges
        .iter()
        .filter(|edge| edge.edge_type.0 == "semantic::missing_intermediate_cause")
        .count();
    let path_stability =
        path_edges.iter().map(edge_stability).sum::<f64>() / path_edges.len() as f64;
    let support_strength =
        (avg_confidence * 0.8) + ((evidence_ref_count.min(10) as f64 / 10.0) * 0.2);
    let pattern_strength =
        path_edges.iter().map(edge_pattern_strength).sum::<f64>() / path_edges.len() as f64;
    KernelCausalPathFeatures {
        depth: path_edges.len(),
        avg_confidence,
        evidence_ref_count,
        candidate_edge_count,
        missing_intermediate_cause_count,
        temporal_consistency_ratio: temporal_consistency(path_vertex_indices, vertices),
        path_stability,
        support_strength,
        pattern_strength,
        path_span_ms: path_span_ms(path_vertex_indices, vertices),
    }
}

fn edge_pattern_strength(edge: &&KernelEdge) -> f64 {
    let mut score: f64 = match edge.edge_type.0.as_str() {
        "causal_link" => 1.0,
        "semantic::same_process" => 0.82,
        "semantic::related_event" => 0.72,
        "supported_by" | "subject" | "object" => 0.58,
        "semantic::missing_intermediate_cause" => 0.38,
        _ => 0.45,
    };
    if matches!(edge.layer, KernelGraphLayer::Candidate) {
        score *= 0.9;
    }
    if edge
        .attributes
        .get("status")
        .and_then(serde_json::Value::as_str)
        == Some("supported")
    {
        score += 0.08;
    }
    score.clamp(0.0, 1.1)
}

fn edge_stability(edge: &&KernelEdge) -> f64 {
    let status = edge
        .attributes
        .get("status")
        .and_then(serde_json::Value::as_str);
    let mut score = status_prior(status);
    if matches!(edge.layer, KernelGraphLayer::Candidate) {
        score *= 0.8;
    }
    if edge.edge_type.0 == "semantic::missing_intermediate_cause" {
        score *= 0.55;
    }
    score
}

fn temporal_consistency(path_vertex_indices: &[usize], vertices: &[KernelVertex]) -> f64 {
    if path_vertex_indices.len() <= 1 {
        return 1.0;
    }
    let mut consistent = 0usize;
    let mut total = 0usize;
    for pair in path_vertex_indices.windows(2) {
        total += 1;
        if temporal_pair_consistent(&vertices[pair[0]].temporal, &vertices[pair[1]].temporal) {
            consistent += 1;
        }
    }
    if total == 0 {
        0.7
    } else {
        consistent as f64 / total as f64
    }
}

fn path_span_ms(path_vertex_indices: &[usize], vertices: &[KernelVertex]) -> i64 {
    let mut earliest = i64::MAX;
    let mut latest = i64::MIN;
    let mut saw_temporal = false;
    for &index in path_vertex_indices {
        let temporal = &vertices[index].temporal;
        if let Some(start) = temporal.valid_from.or(temporal.valid_to) {
            earliest = earliest.min(start);
            latest = latest.max(start);
            saw_temporal = true;
        }
        if let Some(end) = temporal.valid_to {
            earliest = earliest.min(end);
            latest = latest.max(end);
            saw_temporal = true;
        }
    }
    if saw_temporal {
        latest.saturating_sub(earliest)
    } else {
        0
    }
}

fn temporal_pair_consistent(source: &KernelBiTemporal, target: &KernelBiTemporal) -> bool {
    match (source.valid_from, target.valid_from) {
        (Some(source_start), Some(target_start)) => source_start <= target_start,
        _ => match (source.valid_to, target.valid_from) {
            (Some(source_end), Some(target_start)) => source_end <= target_start,
            _ => true,
        },
    }
}

fn status_prior(status: Option<&str>) -> f64 {
    match status.unwrap_or_default() {
        "supported" => 1.0,
        "active" => 0.95,
        "candidate" => 0.65,
        "deferred" => 0.4,
        "contradicted" => 0.15,
        "superseded" => 0.05,
        "rejected" => 0.0,
        _ => 0.2,
    }
}

fn path_modalities(
    vertex_path_rev: &[usize],
    vertices: &[KernelVertex],
    all_edges: &[&KernelEdge],
    outgoing: &[Vec<usize>],
    vertex_by_id: &FxHashMap<&str, (usize, &KernelVertex)>,
) -> Vec<String> {
    let mut modalities = Vec::new();
    for &vertex_index in vertex_path_rev {
        vertex_modalities(
            vertex_index,
            vertices,
            all_edges,
            outgoing,
            vertex_by_id,
            &mut modalities,
        );
    }
    modalities
}

fn vertex_modalities(
    vertex_index: usize,
    vertices: &[KernelVertex],
    all_edges: &[&KernelEdge],
    outgoing: &[Vec<usize>],
    vertex_by_id: &FxHashMap<&str, (usize, &KernelVertex)>,
    modalities: &mut Vec<String>,
) {
    let vertex = &vertices[vertex_index];
    if vertex.kind == "claim" {
        push_modality(&vertex.value, "modality", modalities);
    }
    for &edge_index in &outgoing[vertex_index] {
        let edge = all_edges[edge_index];
        match edge.edge_type.0.as_str() {
            "supported_by" => {
                if let Some(&(_, claim)) = vertex_by_id.get(edge.target_id.0.as_str()) {
                    push_modality(&claim.value, "modality", modalities);
                }
            }
            "under_view" => {
                if let Some(&(_, view)) = vertex_by_id.get(edge.target_id.0.as_str()) {
                    push_modality(&view.value, "modality", modalities);
                    push_modality(&view.value, "modalitySemantics", modalities);
                    if matches!(
                        view.value
                            .get("sourceSemantics")
                            .and_then(serde_json::Value::as_str),
                        Some("reportedSpeech" | "attributedClaim")
                    ) {
                        modalities.push("reported".to_owned());
                    }
                }
            }
            _ => {}
        }
    }
}

fn push_modality(value: &serde_json::Value, key: &str, modalities: &mut Vec<String>) {
    if let Some(modality) = value
        .get(key)
        .and_then(serde_json::Value::as_str)
        .and_then(normalize_modality_label)
    {
        modalities.push(modality.to_owned());
    }
}

fn normalize_modality_label(label: &str) -> Option<&'static str> {
    match label {
        "asserted" | "observed" | "inferred" | "negated" => Some("asserted"),
        "reported" | "reportedSpeech" | "attributedClaim" => Some("reported"),
        "conditional" => Some("conditional"),
        "hypothetical" => Some("hypothetical"),
        "planned" => Some("planned"),
        _ => None,
    }
}

#[derive(Clone, Default)]
struct VisitedBits {
    words: Box<[u64]>,
}

impl VisitedBits {
    fn new(vertex_count: usize) -> Self {
        let words = (vertex_count.max(1) + 63) / 64;
        Self {
            words: vec![0u64; words].into_boxed_slice(),
        }
    }

    fn insert(&mut self, index: usize) {
        self.words[index / 64] |= 1u64 << (index % 64);
    }

    fn contains(&self, index: usize) -> bool {
        (self.words[index / 64] & (1u64 << (index % 64))) != 0
    }
}
