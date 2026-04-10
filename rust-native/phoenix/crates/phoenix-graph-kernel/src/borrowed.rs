use crate::{KernelCsrSidecar, KernelEdge, KernelEdgeKey, KernelVertex, PhoenixGraphKernel};
use rustc_hash::FxHashMap;
use std::sync::RwLockReadGuard;

pub struct KernelCsrRef<'a> {
    guard: RwLockReadGuard<'a, KernelCsrSidecar>,
}

impl<'a> KernelCsrRef<'a> {
    pub fn sidecar(&self) -> &KernelCsrSidecar {
        &self.guard
    }
}

pub struct KernelGraphRef<'a> {
    pub vertices: &'a FxHashMap<String, KernelVertex>,
    pub asserted_edges: &'a FxHashMap<KernelEdgeKey, KernelEdge>,
    pub candidate_edges: &'a FxHashMap<KernelEdgeKey, KernelEdge>,
    csr: RwLockReadGuard<'a, KernelCsrSidecar>,
}

impl<'a> KernelGraphRef<'a> {
    pub fn csr(&self) -> &KernelCsrSidecar {
        &self.csr
    }
}

impl PhoenixGraphKernel {
    pub fn csr_ref(&self) -> KernelCsrRef<'_> {
        self.ensure_csr_sidecar();
        KernelCsrRef {
            guard: self.csr.read().expect("kernel csr sidecar poisoned"),
        }
    }

    pub fn graph_ref(&self) -> KernelGraphRef<'_> {
        self.ensure_csr_sidecar();
        KernelGraphRef {
            vertices: &self.vertices,
            asserted_edges: &self.asserted_edges,
            candidate_edges: &self.candidate_edges,
            csr: self.csr.read().expect("kernel csr sidecar poisoned"),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        KernelEdge, KernelEdgeType, KernelGraphLayer, KernelMutationBatch, KernelMutationScope,
        KernelVertex, KernelVertexId, PhoenixGraphKernel,
    };

    #[test]
    fn graph_ref_exposes_borrowed_active_graph_and_csr() {
        let mut kernel = PhoenixGraphKernel::new();
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Full,
                recorded_at: None,
                vertices: vec![
                    vertex("graph::entity::alice", "entity"),
                    vertex("graph::state::alice:entity.location", "state"),
                ],
                edges: vec![edge(
                    "graph::state::alice:entity.location",
                    "graph::entity::alice",
                    "state_of",
                )],
            })
            .expect("apply batch");

        let graph = kernel.graph_ref();

        assert_eq!(graph.vertices.len(), 2);
        assert_eq!(graph.asserted_edges.len(), 1);
        assert_eq!(graph.csr().vertex_ids.len(), 2);
        assert!(!graph.csr().targets.is_empty());
    }

    fn vertex(id: &str, kind: &str) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: kind.to_owned(),
            ..KernelVertex::default()
        }
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
