use crate::{
    now_ms, KernelEdge, KernelExpandedRegion, KernelGraphSnapshot, KernelVertex,
    KernelViewRequest, PhoenixGraphKernel,
};
use rustc_hash::{FxHashMap, FxHashSet};
use scirs2_graph::CsrGraph;

pub struct KernelQueryView<'a> {
    vertices: Vec<&'a KernelVertex>,
    asserted_edges: Vec<&'a KernelEdge>,
    candidate_edges: Vec<&'a KernelEdge>,
}

impl<'a> KernelQueryView<'a> {
    pub fn vertices(&self) -> &[&'a KernelVertex] {
        &self.vertices
    }

    pub fn asserted_edges(&self) -> &[&'a KernelEdge] {
        &self.asserted_edges
    }

    pub fn candidate_edges(&self) -> &[&'a KernelEdge] {
        &self.candidate_edges
    }

    pub fn find_vertex(&self, vertex_id: &str) -> Option<&'a KernelVertex> {
        self.vertices
            .iter()
            .copied()
            .find(|vertex| vertex.id.0 == vertex_id)
    }

    pub fn expand_region(
        &self,
        anchor_vertex_ids: &[String],
        seed_vertex_ids: &[String],
        region_node_limit: usize,
        expansion_hops: usize,
        edge_allowed: fn(&KernelEdge) -> bool,
    ) -> KernelExpandedRegion {
        let dense = self
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
        if self.vertices.is_empty() {
            return KernelExpandedRegion {
                snapshot: KernelGraphSnapshot::default(),
                seed_vertex_ids,
                included_vertex_ids: Vec::new(),
                truncated: false,
            };
        }

        let traversal_edges = collect_traversal_edges(
            self.asserted_edges.iter().copied(),
            self.candidate_edges.iter().copied(),
            &dense,
            edge_allowed,
        );
        let graph = build_region_graph(self.vertices.len(), traversal_edges.as_slice());
        let node_limit = region_node_limit.clamp(8, 256);
        let max_hops = expansion_hops.clamp(1, 4);
        let mut included = vec![false; self.vertices.len()];
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
                for (neighbor, _) in graph.neighbors(vertex_index) {
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

        let mut vertices = self
            .vertices
            .iter()
            .enumerate()
            .filter(|(index, _)| included[*index])
            .map(|(_, vertex)| (*vertex).clone())
            .collect::<Vec<_>>();
        let mut asserted_edges =
            materialize_edges(self.asserted_edges.iter().copied(), &dense, &included, edge_allowed);
        let mut candidate_edges = materialize_edges(
            self.candidate_edges.iter().copied(),
            &dense,
            &included,
            edge_allowed,
        );
        vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        asserted_edges.sort_by(|left, right| {
            left.source_id
                .0
                .cmp(&right.source_id.0)
                .then_with(|| left.target_id.0.cmp(&right.target_id.0))
                .then_with(|| left.edge_type.0.cmp(&right.edge_type.0))
        });
        candidate_edges.sort_by(|left, right| {
            left.source_id
                .0
                .cmp(&right.source_id.0)
                .then_with(|| left.target_id.0.cmp(&right.target_id.0))
                .then_with(|| left.edge_type.0.cmp(&right.edge_type.0))
        });
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
}

impl PhoenixGraphKernel {
    pub fn query_view(&self, request: KernelViewRequest) -> KernelQueryView<'_> {
        if request.valid_at.is_none() && request.recorded_at.is_none() {
            let mut vertices = self.vertices.values().collect::<Vec<_>>();
            vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
            let mut asserted_edges = self.asserted_edges.values().collect::<Vec<_>>();
            asserted_edges.sort_by(|left, right| {
                left.source_id
                    .0
                    .cmp(&right.source_id.0)
                    .then_with(|| left.target_id.0.cmp(&right.target_id.0))
                    .then_with(|| left.edge_type.0.cmp(&right.edge_type.0))
            });
            let mut candidate_edges = if request.include_candidate_graph {
                self.candidate_edges.values().collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            candidate_edges.sort_by(|left, right| {
                left.source_id
                    .0
                    .cmp(&right.source_id.0)
                    .then_with(|| left.target_id.0.cmp(&right.target_id.0))
                    .then_with(|| left.edge_type.0.cmp(&right.edge_type.0))
            });
            return KernelQueryView {
                vertices,
                asserted_edges,
                candidate_edges,
            };
        }

        let valid_at = request.valid_at.unwrap_or_else(now_ms);
        let tx_at = request.recorded_at.unwrap_or(valid_at);
        let mut vertices = self
            .vertex_history
            .values()
            .filter_map(|records| {
                records
                    .iter()
                    .rev()
                    .find(|record| record.temporal.is_visible_at(valid_at, tx_at))
            })
            .collect::<Vec<_>>();
        vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
        let visible_vertex_ids = vertices
            .iter()
            .map(|vertex| vertex.id.0.as_str())
            .collect::<FxHashSet<_>>();
        let mut asserted_edges = self
            .asserted_edge_history
            .values()
            .filter_map(|records| {
                records
                    .iter()
                    .rev()
                    .find(|record| record.temporal.is_visible_at(valid_at, tx_at))
            })
            .filter(|edge| {
                visible_vertex_ids.contains(edge.source_id.0.as_str())
                    && visible_vertex_ids.contains(edge.target_id.0.as_str())
            })
            .collect::<Vec<_>>();
        asserted_edges.sort_by(|left, right| {
            left.source_id
                .0
                .cmp(&right.source_id.0)
                .then_with(|| left.target_id.0.cmp(&right.target_id.0))
                .then_with(|| left.edge_type.0.cmp(&right.edge_type.0))
        });
        let mut candidate_edges = if request.include_candidate_graph {
            self.candidate_edge_history
                .values()
                .filter_map(|records| {
                    records
                        .iter()
                        .rev()
                        .find(|record| record.temporal.is_visible_at(valid_at, tx_at))
                })
                .filter(|edge| {
                    visible_vertex_ids.contains(edge.source_id.0.as_str())
                        && visible_vertex_ids.contains(edge.target_id.0.as_str())
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        candidate_edges.sort_by(|left, right| {
            left.source_id
                .0
                .cmp(&right.source_id.0)
                .then_with(|| left.target_id.0.cmp(&right.target_id.0))
                .then_with(|| left.edge_type.0.cmp(&right.edge_type.0))
        });

        KernelQueryView {
            vertices,
            asserted_edges,
            candidate_edges,
        }
    }
}

fn collect_traversal_edges<'a>(
    asserted_edges: impl Iterator<Item = &'a KernelEdge>,
    candidate_edges: impl Iterator<Item = &'a KernelEdge>,
    dense: &FxHashMap<&'a str, usize>,
    edge_allowed: fn(&KernelEdge) -> bool,
) -> Vec<(usize, usize, f64)> {
    let mut seen = FxHashSet::default();
    let mut edges = Vec::new();
    for edge in asserted_edges.chain(candidate_edges).filter(|edge| edge_allowed(edge)) {
        let Some(&source) = dense.get(edge.source_id.0.as_str()) else {
            continue;
        };
        let Some(&target) = dense.get(edge.target_id.0.as_str()) else {
            continue;
        };
        if source == target {
            continue;
        }
        let key = undirected_key(source, target);
        if !seen.insert(key) {
            continue;
        }
        let weight = edge.weight.max(1) as f64;
        edges.push((source, target, weight));
    }
    edges
}

fn materialize_edges<'a>(
    edges: impl Iterator<Item = &'a KernelEdge>,
    dense: &FxHashMap<&'a str, usize>,
    included: &[bool],
    edge_allowed: fn(&KernelEdge) -> bool,
) -> Vec<crate::KernelEdge> {
    edges
        .filter(|edge| edge_allowed(edge))
        .filter(|edge| {
            let Some(&source) = dense.get(edge.source_id.0.as_str()) else {
                return false;
            };
            let Some(&target) = dense.get(edge.target_id.0.as_str()) else {
                return false;
            };
            included[source] && included[target]
        })
        .cloned()
        .collect::<Vec<_>>()
}

fn build_region_graph(num_nodes: usize, edges: &[(usize, usize, f64)]) -> CsrGraph {
    CsrGraph::from_edges_parallel(num_nodes, edges.to_vec(), false)
        .or_else(|_| CsrGraph::from_edges(num_nodes, edges.to_vec(), false))
        .expect("query-view region graph should build")
}

fn undirected_key(left: usize, right: usize) -> u64 {
    let (small, large) = if left <= right {
        (left as u64, right as u64)
    } else {
        (right as u64, left as u64)
    };
    (small << 32) | large
}

#[cfg(test)]
mod tests {
    use super::PhoenixGraphKernel;
    use crate::{
        KernelEdge, KernelEdgeType, KernelGraphLayer, KernelMutationBatch, KernelMutationScope,
        KernelVertex, KernelVertexId, KernelViewRequest,
    };

    #[test]
    fn query_view_expand_region_uses_visible_projection_without_full_snapshot() {
        let mut kernel = PhoenixGraphKernel::new();
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Full,
                recorded_at: None,
                vertices: vec![
                    vertex("entity", "entity"),
                    vertex("state", "state"),
                    vertex("claim", "claim"),
                    vertex("event", "event"),
                ],
                edges: vec![
                    edge("entity", "state", "state_of", KernelGraphLayer::Asserted),
                    edge("state", "claim", "supported_by", KernelGraphLayer::Asserted),
                ],
            })
            .expect("apply batch");
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Candidate,
                scope: KernelMutationScope::Candidate {
                    scope_key: "test-region".to_owned(),
                },
                recorded_at: None,
                vertices: Vec::new(),
                edges: vec![edge(
                    "claim",
                    "event",
                    "semantic::same_process",
                    KernelGraphLayer::Candidate,
                )],
            })
            .expect("apply candidate batch");

        let view = kernel.query_view(KernelViewRequest {
            include_candidate_graph: true,
            ..KernelViewRequest::default()
        });
        let region = view.expand_region(
            &["entity".to_owned()],
            &["event".to_owned()],
            8,
            3,
            |_| true,
        );

        assert_eq!(view.vertices().len(), 4);
        assert_eq!(region.snapshot.vertices.len(), 4);
        assert_eq!(region.snapshot.candidate_edges.len(), 1);
        assert!(region
            .included_vertex_ids
            .iter()
            .any(|vertex_id| vertex_id == "event"));
    }

    fn vertex(id: &str, kind: &str) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: kind.to_owned(),
            ..KernelVertex::default()
        }
    }

    fn edge(source: &str, target: &str, edge_type: &str, layer: KernelGraphLayer) -> KernelEdge {
        KernelEdge {
            source_id: KernelVertexId(source.to_owned()),
            target_id: KernelVertexId(target.to_owned()),
            edge_type: KernelEdgeType(edge_type.to_owned()),
            layer,
            ..KernelEdge::default()
        }
    }
}
