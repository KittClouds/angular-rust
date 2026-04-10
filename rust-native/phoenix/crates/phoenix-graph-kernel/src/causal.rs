use crate::{KernelEdge, KernelGraphSnapshot, KernelVertex};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

const MAX_INCOMING_PER_VERTEX: usize = 8;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KernelCausalPathCandidate {
    pub source_vertex_id: String,
    #[serde(default)]
    pub path_vertex_ids: Vec<String>,
    #[serde(default)]
    pub path_edges: Vec<KernelEdge>,
    #[serde(default)]
    pub supporting_modalities: Vec<String>,
}

pub fn causal_path_candidates_from_snapshot(
    snapshot: &KernelGraphSnapshot,
    target_vertex_id: &str,
    max_depth: usize,
    limit: usize,
) -> Vec<KernelCausalPathCandidate> {
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
    let incoming = build_incoming_causal_index(&all_edges, &vertex_by_id, snapshot.vertices.len());
    let outgoing = build_outgoing_index(&all_edges, &vertex_by_id, snapshot.vertices.len());

    #[derive(Clone)]
    struct Frame {
        current_index: usize,
        vertex_path_rev: Vec<usize>,
        edge_path_rev: Vec<usize>,
        visited: VisitedBits,
    }

    let mut visited = VisitedBits::new(snapshot.vertices.len());
    visited.insert(target_index);
    let mut stack = vec![Frame {
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

            candidates.push(materialize_candidate(
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
                stack.push(Frame {
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

    candidates.sort_by(|left, right| left.source_vertex_id.cmp(&right.source_vertex_id));
    candidates.truncate(limit.max(1));
    candidates
}

fn build_incoming_causal_index(
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

fn build_outgoing_index(
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

fn materialize_candidate(
    vertex_path_rev: &[usize],
    edge_path_rev: &[usize],
    vertices: &[KernelVertex],
    all_edges: &[&KernelEdge],
    outgoing: &[Vec<usize>],
    vertex_by_id: &FxHashMap<&str, (usize, &KernelVertex)>,
) -> KernelCausalPathCandidate {
    let mut path_vertex_ids = vertex_path_rev
        .iter()
        .rev()
        .map(|&index| vertices[index].id.0.clone())
        .collect::<Vec<_>>();
    let source_vertex_id = path_vertex_ids.first().cloned().unwrap_or_default();
    let path_edges = edge_path_rev
        .iter()
        .rev()
        .map(|&edge_index| all_edges[edge_index].clone())
        .collect::<Vec<_>>();
    let mut supporting_modalities =
        collect_path_modalities(vertex_path_rev, vertices, all_edges, outgoing, vertex_by_id);
    supporting_modalities.sort();
    supporting_modalities.dedup();
    if path_vertex_ids.is_empty() {
        path_vertex_ids.push(source_vertex_id.clone());
    }
    KernelCausalPathCandidate {
        source_vertex_id,
        path_vertex_ids,
        path_edges,
        supporting_modalities,
    }
}

fn collect_path_modalities(
    vertex_path_rev: &[usize],
    vertices: &[KernelVertex],
    all_edges: &[&KernelEdge],
    outgoing: &[Vec<usize>],
    vertex_by_id: &FxHashMap<&str, (usize, &KernelVertex)>,
) -> Vec<String> {
    let mut modalities = Vec::new();
    for &vertex_index in vertex_path_rev {
        collect_vertex_modalities(
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

fn collect_vertex_modalities(
    vertex_index: usize,
    vertices: &[KernelVertex],
    all_edges: &[&KernelEdge],
    outgoing: &[Vec<usize>],
    vertex_by_id: &FxHashMap<&str, (usize, &KernelVertex)>,
    modalities: &mut Vec<String>,
) {
    let vertex = &vertices[vertex_index];
    if vertex.kind == "claim" {
        if let Some(modality) = vertex
            .value
            .get("modality")
            .and_then(serde_json::Value::as_str)
            .and_then(normalize_modality_label)
        {
            modalities.push(modality.to_owned());
        }
    }
    for &edge_index in &outgoing[vertex_index] {
        let edge = all_edges[edge_index];
        match edge.edge_type.0.as_str() {
            "supported_by" => {
                if let Some(&(_, claim)) = vertex_by_id.get(edge.target_id.0.as_str()) {
                    if let Some(modality) = claim
                        .value
                        .get("modality")
                        .and_then(serde_json::Value::as_str)
                        .and_then(normalize_modality_label)
                    {
                        modalities.push(modality.to_owned());
                    }
                }
            }
            "under_view" => {
                if let Some(&(_, view)) = vertex_by_id.get(edge.target_id.0.as_str()) {
                    if let Some(modality) = view
                        .value
                        .get("modality")
                        .and_then(serde_json::Value::as_str)
                        .and_then(normalize_modality_label)
                    {
                        modalities.push(modality.to_owned());
                    }
                    if let Some(modality) = view
                        .value
                        .get("modalitySemantics")
                        .and_then(serde_json::Value::as_str)
                        .and_then(normalize_modality_label)
                    {
                        modalities.push(modality.to_owned());
                    }
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
        let word = index / 64;
        let bit = index % 64;
        self.words[word] |= 1u64 << bit;
    }

    fn contains(&self, index: usize) -> bool {
        let word = index / 64;
        let bit = index % 64;
        (self.words[word] & (1u64 << bit)) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::causal_path_candidates_from_snapshot;
    use crate::{
        KernelBiTemporal, KernelEdge, KernelEdgeType, KernelGraphLayer, KernelGraphSnapshot,
        KernelVertex, KernelVertexId,
    };
    use serde_json::json;

    #[test]
    fn causal_candidates_follow_incoming_links_and_preserve_order() {
        let snapshot = KernelGraphSnapshot {
            vertices: vec![
                vertex("a", "event"),
                vertex("b", "event"),
                vertex("c", "event"),
            ],
            asserted_edges: vec![causal_edge("a", "b", 0.8), causal_edge("b", "c", 0.7)],
            candidate_edges: Vec::new(),
        };

        let candidates = causal_path_candidates_from_snapshot(&snapshot, "c", 3, 8);

        assert_eq!(candidates.len(), 2);
        assert_eq!(candidates[0].source_vertex_id, "a");
        assert_eq!(candidates[0].path_vertex_ids, vec!["a", "b", "c"]);
        assert_eq!(candidates[1].path_vertex_ids, vec!["b", "c"]);
    }

    #[test]
    fn causal_candidates_collect_supported_modalities() {
        let snapshot = KernelGraphSnapshot {
            vertices: vec![
                claim("graph::claim::source", "reported"),
                vertex("event-source", "event"),
                vertex("event-target", "event"),
            ],
            asserted_edges: vec![
                causal_edge("event-source", "event-target", 0.9),
                edge("event-source", "graph::claim::source", "supported_by"),
            ],
            candidate_edges: Vec::new(),
        };

        let candidates = causal_path_candidates_from_snapshot(&snapshot, "event-target", 2, 4);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].supporting_modalities, vec!["reported"]);
    }

    fn vertex(id: &str, kind: &str) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: kind.to_owned(),
            temporal: KernelBiTemporal {
                valid_from: Some(1),
                valid_to: None,
                recorded_at: Some(1),
                expired_at: None,
            },
            ..KernelVertex::default()
        }
    }

    fn claim(id: &str, modality: &str) -> KernelVertex {
        KernelVertex {
            value: json!({ "modality": modality }),
            ..vertex(id, "claim")
        }
    }

    fn causal_edge(source_id: &str, target_id: &str, confidence: f64) -> KernelEdge {
        let mut edge = edge(source_id, target_id, "causal_link");
        edge.provenance.confidence = Some(confidence);
        edge
    }

    fn edge(source_id: &str, target_id: &str, edge_type: &str) -> KernelEdge {
        KernelEdge {
            source_id: KernelVertexId(source_id.to_owned()),
            target_id: KernelVertexId(target_id.to_owned()),
            edge_type: KernelEdgeType(edge_type.to_owned()),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        }
    }
}
