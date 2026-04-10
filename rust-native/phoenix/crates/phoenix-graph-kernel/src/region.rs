use crate::{KernelEdge, KernelGraphSnapshot};
use rustc_hash::FxHashMap;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KernelExpandedRegion {
    pub snapshot: KernelGraphSnapshot,
    pub seed_vertex_ids: Vec<String>,
    pub included_vertex_ids: Vec<String>,
    pub truncated: bool,
}

pub fn expand_snapshot_region(
    snapshot: &KernelGraphSnapshot,
    anchor_vertex_ids: &[String],
    seed_vertex_ids: &[String],
    region_node_limit: usize,
    expansion_hops: usize,
    edge_allowed: fn(&KernelEdge) -> bool,
) -> KernelExpandedRegion {
    let dense = snapshot
        .vertices
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id.0.as_str(), index))
        .collect::<FxHashMap<_, _>>();
    let seed_vertex_ids = seed_vertex_ids
        .iter()
        .filter(|vertex_id| dense.contains_key(vertex_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if snapshot.vertices.is_empty() {
        return KernelExpandedRegion {
            snapshot: KernelGraphSnapshot::default(),
            seed_vertex_ids,
            included_vertex_ids: Vec::new(),
            truncated: false,
        };
    }

    let (offsets, targets) = build_traversal_csr(snapshot, &dense, edge_allowed);
    let node_limit = region_node_limit.clamp(8, 256);
    let max_hops = expansion_hops.clamp(1, 4);
    let mut included = vec![false; snapshot.vertices.len()];
    let mut frontier = Vec::<usize>::new();
    for vertex_id in anchor_vertex_ids.iter().chain(seed_vertex_ids.iter()) {
        if let Some(&index) = dense.get(vertex_id.as_str()) {
            if !included[index] {
                included[index] = true;
                frontier.push(index);
            }
        }
    }

    let mut included_count = frontier.len();
    let mut truncated = false;
    for _ in 0..max_hops {
        if frontier.is_empty() || included_count >= node_limit {
            break;
        }
        let mut next_frontier = Vec::new();
        for vertex_index in frontier.drain(..) {
            let start = offsets[vertex_index];
            let end = offsets[vertex_index + 1];
            for &neighbor in &targets[start..end] {
                if included[neighbor] {
                    continue;
                }
                if included_count >= node_limit {
                    truncated = true;
                    continue;
                }
                included[neighbor] = true;
                included_count += 1;
                next_frontier.push(neighbor);
            }
        }
        frontier = next_frontier;
    }

    let mut vertices = snapshot
        .vertices
        .iter()
        .enumerate()
        .filter(|(index, _)| included[*index])
        .map(|(_, vertex)| vertex.clone())
        .collect::<Vec<_>>();
    let mut asserted_edges =
        filter_edges(&snapshot.asserted_edges, &dense, &included, edge_allowed);
    let mut candidate_edges =
        filter_edges(&snapshot.candidate_edges, &dense, &included, edge_allowed);
    vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
    asserted_edges.sort_by(|left, right| left.source_id.0.cmp(&right.source_id.0));
    candidate_edges.sort_by(|left, right| left.source_id.0.cmp(&right.source_id.0));
    let mut included_vertex_ids = vertices
        .iter()
        .map(|vertex| vertex.id.0.clone())
        .collect::<Vec<_>>();
    included_vertex_ids.sort();

    KernelExpandedRegion {
        snapshot: KernelGraphSnapshot {
            vertices,
            asserted_edges,
            candidate_edges,
        },
        seed_vertex_ids,
        included_vertex_ids,
        truncated,
    }
}

fn build_traversal_csr(
    snapshot: &KernelGraphSnapshot,
    dense: &FxHashMap<&str, usize>,
    edge_allowed: fn(&KernelEdge) -> bool,
) -> (Vec<usize>, Vec<usize>) {
    let mut degrees = vec![0usize; snapshot.vertices.len()];
    for edge in snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
        .filter(|edge| edge_allowed(edge))
    {
        let Some(&source_index) = dense.get(edge.source_id.0.as_str()) else {
            continue;
        };
        let Some(&target_index) = dense.get(edge.target_id.0.as_str()) else {
            continue;
        };
        degrees[source_index] += 1;
        degrees[target_index] += 1;
    }
    let mut offsets = vec![0usize; snapshot.vertices.len() + 1];
    for index in 0..snapshot.vertices.len() {
        offsets[index + 1] = offsets[index] + degrees[index];
    }
    let mut cursors = offsets[..snapshot.vertices.len()].to_vec();
    let mut targets = vec![0usize; offsets[snapshot.vertices.len()]];
    for edge in snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
        .filter(|edge| edge_allowed(edge))
    {
        let Some(&source_index) = dense.get(edge.source_id.0.as_str()) else {
            continue;
        };
        let Some(&target_index) = dense.get(edge.target_id.0.as_str()) else {
            continue;
        };
        targets[cursors[source_index]] = target_index;
        cursors[source_index] += 1;
        targets[cursors[target_index]] = source_index;
        cursors[target_index] += 1;
    }
    (offsets, targets)
}

fn filter_edges(
    edges: &[KernelEdge],
    dense: &FxHashMap<&str, usize>,
    included: &[bool],
    edge_allowed: fn(&KernelEdge) -> bool,
) -> Vec<KernelEdge> {
    edges
        .iter()
        .filter(|edge| edge_allowed(edge))
        .filter(|edge| {
            let Some(&source_index) = dense.get(edge.source_id.0.as_str()) else {
                return false;
            };
            let Some(&target_index) = dense.get(edge.target_id.0.as_str()) else {
                return false;
            };
            included[source_index] && included[target_index]
        })
        .cloned()
        .collect::<Vec<_>>()
}

#[cfg(test)]
mod tests {
    use super::expand_snapshot_region;
    use crate::{
        KernelEdge, KernelEdgeType, KernelGraphLayer, KernelGraphSnapshot, KernelVertex,
        KernelVertexId,
    };

    #[test]
    fn expand_snapshot_region_follows_dense_csr_without_string_adjacency() {
        let snapshot = KernelGraphSnapshot {
            vertices: vec![vertex("a"), vertex("b"), vertex("c"), vertex("d")],
            asserted_edges: vec![edge("a", "b"), edge("b", "c")],
            candidate_edges: vec![candidate_edge("c", "d")],
        };

        let expanded = expand_snapshot_region(
            &snapshot,
            &["a".to_owned()],
            &["d".to_owned()],
            4,
            2,
            |_| true,
        );

        assert_eq!(expanded.snapshot.vertices.len(), 4);
        assert_eq!(expanded.seed_vertex_ids, vec!["d".to_owned()]);
        assert!(expanded
            .included_vertex_ids
            .iter()
            .any(|vertex_id| vertex_id == "c"));
    }

    #[test]
    fn expand_snapshot_region_respects_edge_filter() {
        let snapshot = KernelGraphSnapshot {
            vertices: vec![vertex("a"), vertex("b"), vertex("c")],
            asserted_edges: vec![edge("a", "b")],
            candidate_edges: vec![candidate_edge("b", "c")],
        };

        let expanded = expand_snapshot_region(&snapshot, &["a".to_owned()], &[], 3, 3, |edge| {
            edge.layer == KernelGraphLayer::Asserted
        });

        assert_eq!(expanded.snapshot.vertices.len(), 2);
        assert!(!expanded
            .included_vertex_ids
            .iter()
            .any(|vertex_id| vertex_id == "c"));
    }

    fn vertex(id: &str) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: "generic".to_owned(),
            ..KernelVertex::default()
        }
    }

    fn edge(source_id: &str, target_id: &str) -> KernelEdge {
        KernelEdge {
            source_id: KernelVertexId(source_id.to_owned()),
            target_id: KernelVertexId(target_id.to_owned()),
            edge_type: KernelEdgeType("state_of".to_owned()),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        }
    }

    fn candidate_edge(source_id: &str, target_id: &str) -> KernelEdge {
        KernelEdge {
            layer: KernelGraphLayer::Candidate,
            ..edge(source_id, target_id)
        }
    }
}
