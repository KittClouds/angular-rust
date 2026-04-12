use crate::{
    chrono_region::expand_region_for_view, now_ms, KernelEdge, KernelExpandedRegion,
    KernelRegionProfile, KernelVertex, KernelViewRequest, PhoenixGraphKernel,
};
use rustc_hash::FxHashSet;

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
        self.expand_region_with_profile(
            anchor_vertex_ids,
            seed_vertex_ids,
            region_node_limit,
            expansion_hops,
            edge_allowed,
            KernelRegionProfile::Generic,
        )
    }

    pub fn expand_region_with_profile(
        &self,
        anchor_vertex_ids: &[String],
        seed_vertex_ids: &[String],
        region_node_limit: usize,
        expansion_hops: usize,
        edge_allowed: fn(&KernelEdge) -> bool,
        profile: KernelRegionProfile,
    ) -> KernelExpandedRegion {
        expand_region_for_view(
            self,
            anchor_vertex_ids,
            seed_vertex_ids,
            region_node_limit,
            expansion_hops,
            edge_allowed,
            profile,
        )
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
        let region =
            view.expand_region(&["entity".to_owned()], &["event".to_owned()], 8, 3, |_| {
                true
            });

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
