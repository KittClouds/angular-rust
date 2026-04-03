use phoenix_graph::{
    GraphBackendError, GraphCounts, GraphEdgeRecord, GraphLayer, GraphMutationBatch,
    GraphMutationScope, GraptorEdge, GraptorGraph, GraptorVertex, PhoenixGraphBackend,
};
use rustc_hash::{FxHashMap, FxHashSet};
use scirs2_graph::DiGraph;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EdgeKey {
    source_id: String,
    target_id: String,
    edge_type: String,
}

impl EdgeKey {
    fn new(source_id: &str, target_id: &str, edge_type: &str) -> Self {
        Self {
            source_id: source_id.to_owned(),
            target_id: target_id.to_owned(),
            edge_type: edge_type.to_owned(),
        }
    }
}

#[derive(Default)]
pub struct NativePhoenixGraph {
    rebuild_token: Option<String>,
    invalidated: bool,
    vertices: FxHashMap<String, GraptorVertex>,
    asserted_edges: FxHashMap<EdgeKey, GraptorEdge>,
    candidate_edges: FxHashMap<EdgeKey, GraptorEdge>,
    document_scope_vertices: FxHashMap<String, FxHashSet<String>>,
    document_scope_edges: FxHashMap<String, FxHashSet<EdgeKey>>,
    session_scope_vertices: FxHashMap<String, FxHashSet<String>>,
    session_scope_edges: FxHashMap<String, FxHashSet<EdgeKey>>,
    candidate_scope_edges: FxHashMap<String, FxHashSet<EdgeKey>>,
    digraph: DiGraph<String, f64>,
    dense_node_index: FxHashMap<String, usize>,
}

impl NativePhoenixGraph {
    pub fn new() -> Self {
        Self {
            digraph: DiGraph::new(),
            ..Self::default()
        }
    }

    fn ensure_ready(&self) -> Result<(), GraphBackendError> {
        if self.invalidated {
            return Err(GraphBackendError::Invalidated);
        }
        Ok(())
    }

    fn replace_vertices(
        vertices: &mut FxHashMap<String, GraptorVertex>,
        scope_map: &mut FxHashMap<String, FxHashSet<String>>,
        scope_key: String,
        new_vertices: Vec<GraptorVertex>,
    ) {
        if let Some(existing) = scope_map.remove(&scope_key) {
            for vertex_id in existing {
                vertices.remove(&vertex_id);
            }
        }
        let scope_vertices = new_vertices
            .iter()
            .map(|vertex| vertex.id.clone())
            .collect::<FxHashSet<_>>();
        for vertex in new_vertices {
            vertices.insert(vertex.id.clone(), vertex);
        }
        scope_map.insert(scope_key, scope_vertices);
    }

    fn replace_edges(
        edge_map: &mut FxHashMap<EdgeKey, GraptorEdge>,
        scope_map: &mut FxHashMap<String, FxHashSet<EdgeKey>>,
        scope_key: String,
        new_edges: Vec<GraptorEdge>,
    ) {
        if let Some(existing) = scope_map.remove(&scope_key) {
            for edge_key in existing {
                edge_map.remove(&edge_key);
            }
        }
        let scope_edges = new_edges
            .iter()
            .map(|edge| EdgeKey::new(&edge.source_id, &edge.target_id, &edge.edge_type))
            .collect::<FxHashSet<_>>();
        for edge in new_edges {
            let edge_key = EdgeKey::new(&edge.source_id, &edge.target_id, &edge.edge_type);
            edge_map.insert(edge_key, edge);
        }
        scope_map.insert(scope_key, scope_edges);
    }

    fn rebuild_digraph(&mut self) {
        self.digraph = DiGraph::new();
        self.dense_node_index.clear();
        for (index, vertex_id) in self.vertices.keys().enumerate() {
            let _ = self.digraph.add_node(vertex_id.clone());
            self.dense_node_index.insert(vertex_id.clone(), index);
        }
        for edge in self.asserted_edges.values() {
            if self.vertices.contains_key(&edge.source_id)
                && self.vertices.contains_key(&edge.target_id)
            {
                let _ = self.digraph.add_edge(
                    edge.source_id.clone(),
                    edge.target_id.clone(),
                    edge.weight.max(1) as f64,
                );
            }
        }
    }

    fn snapshot_from_edges(&self, include_candidate_graph: bool) -> GraptorGraph {
        let mut graph = GraptorGraph::default();
        for (vertex_id, vertex) in &self.vertices {
            graph.vertices.insert(vertex_id.clone(), vertex.clone());
            if let (Some(document_id), Some(chapter_id), Some(_)) = (
                vertex.document_id.clone(),
                vertex.chapter_id,
                vertex.search_chunk_id.clone(),
            ) {
                graph
                    .chapter_leaves
                    .entry((document_id, chapter_id))
                    .or_default()
                    .push(vertex_id.clone());
            }
        }
        for edge in self.asserted_edges.values() {
            graph
                .outgoing
                .entry(edge.source_id.clone())
                .or_default()
                .push(edge.clone());
            graph
                .incoming
                .entry(edge.target_id.clone())
                .or_default()
                .push(edge.clone());
        }
        if include_candidate_graph {
            for edge in self.candidate_edges.values() {
                if !candidate_edge_is_active(edge) {
                    continue;
                }
                graph
                    .outgoing
                    .entry(edge.source_id.clone())
                    .or_default()
                    .push(edge.clone());
                graph
                    .incoming
                    .entry(edge.target_id.clone())
                    .or_default()
                    .push(edge.clone());
            }
        }
        graph
    }
}

impl PhoenixGraphBackend for NativePhoenixGraph {
    fn apply_batch(&mut self, batch: GraphMutationBatch) -> Result<(), GraphBackendError> {
        self.ensure_ready()?;
        let scope_key = batch.scope.scope_key();
        let vertices = batch
            .vertices
            .into_iter()
            .map(GraptorVertex::from)
            .collect::<Vec<_>>();
        let edges = batch
            .edges
            .into_iter()
            .map(GraptorEdge::from)
            .collect::<Vec<_>>();
        match (&batch.layer, &batch.scope) {
            (GraphLayer::Asserted, GraphMutationScope::Document { .. }) => {
                Self::replace_vertices(
                    &mut self.vertices,
                    &mut self.document_scope_vertices,
                    scope_key.clone(),
                    vertices,
                );
                Self::replace_edges(
                    &mut self.asserted_edges,
                    &mut self.document_scope_edges,
                    scope_key,
                    edges,
                );
            }
            (GraphLayer::Asserted, GraphMutationScope::Session { .. }) => {
                Self::replace_vertices(
                    &mut self.vertices,
                    &mut self.session_scope_vertices,
                    scope_key.clone(),
                    vertices,
                );
                Self::replace_edges(
                    &mut self.asserted_edges,
                    &mut self.session_scope_edges,
                    scope_key,
                    edges,
                );
            }
            (GraphLayer::Candidate, GraphMutationScope::Candidate { .. }) => {
                Self::replace_edges(
                    &mut self.candidate_edges,
                    &mut self.candidate_scope_edges,
                    scope_key,
                    edges,
                );
            }
            (GraphLayer::Asserted, GraphMutationScope::Full) => {
                self.vertices.clear();
                self.asserted_edges.clear();
                self.document_scope_vertices.clear();
                self.document_scope_edges.clear();
                self.session_scope_vertices.clear();
                self.session_scope_edges.clear();
                for vertex in vertices {
                    self.vertices.insert(vertex.id.clone(), vertex);
                }
                for edge in edges {
                    let edge_key = EdgeKey::new(&edge.source_id, &edge.target_id, &edge.edge_type);
                    self.asserted_edges.insert(edge_key, edge);
                }
            }
            (GraphLayer::Candidate, GraphMutationScope::Full) => {
                self.candidate_edges.clear();
                self.candidate_scope_edges.clear();
                for edge in edges {
                    let edge_key = EdgeKey::new(&edge.source_id, &edge.target_id, &edge.edge_type);
                    self.candidate_edges.insert(edge_key, edge);
                }
            }
            _ => {
                return Err(GraphBackendError::Operation(
                    "graph mutation scope and layer were incompatible".to_owned(),
                ));
            }
        }
        self.rebuild_digraph();
        Ok(())
    }

    fn rebuild_from_batches(
        &mut self,
        batches: Vec<GraphMutationBatch>,
    ) -> Result<(), GraphBackendError> {
        self.invalidated = false;
        self.vertices.clear();
        self.asserted_edges.clear();
        self.candidate_edges.clear();
        self.document_scope_vertices.clear();
        self.document_scope_edges.clear();
        self.session_scope_vertices.clear();
        self.session_scope_edges.clear();
        self.candidate_scope_edges.clear();
        for batch in batches {
            self.apply_batch(batch)?;
        }
        self.rebuild_digraph();
        Ok(())
    }

    fn snapshot(&self, include_candidate_graph: bool) -> Result<GraptorGraph, GraphBackendError> {
        self.ensure_ready()?;
        Ok(self.snapshot_from_edges(include_candidate_graph))
    }

    fn counts(&self) -> Result<GraphCounts, GraphBackendError> {
        self.ensure_ready()?;
        Ok(GraphCounts {
            vertex_count: self.vertices.len(),
            asserted_edge_count: self.asserted_edges.len(),
            candidate_edge_count: self
                .candidate_edges
                .values()
                .filter(|edge| candidate_edge_is_active(edge))
                .count(),
        })
    }

    fn candidate_edges(&self) -> Result<Vec<GraphEdgeRecord>, GraphBackendError> {
        self.ensure_ready()?;
        Ok(self
            .candidate_edges
            .values()
            .map(GraphEdgeRecord::from)
            .collect())
    }

    fn invalidate(&mut self) {
        self.invalidated = true;
    }

    fn rebuild_token(&self) -> Option<&str> {
        self.rebuild_token.as_deref()
    }

    fn set_rebuild_token(&mut self, token: Option<String>) {
        self.rebuild_token = token;
    }
}

fn candidate_edge_is_active(edge: &GraptorEdge) -> bool {
    !matches!(
        edge.attributes
            .get("graph")
            .and_then(|graph| graph.get("status"))
            .and_then(|status| status.as_str()),
        Some("candidate_rejected")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_graph::{GraphEdgeRecord, GraphMutationScope, GraphVertexRecord};
    use serde_json::json;

    #[test]
    fn document_batches_replace_prior_scope_rows() {
        let mut graph = NativePhoenixGraph::new();
        graph
            .apply_batch(GraphMutationBatch {
                layer: GraphLayer::Asserted,
                scope: GraphMutationScope::Document {
                    document_id: "doc-1".to_owned(),
                },
                vertices: vec![GraphVertexRecord {
                    id: "doc::doc-1".to_owned(),
                    kind: "document".to_owned(),
                    weight: 1,
                    value: json!({ "kind": "document", "documentId": "doc-1" }),
                    attributes: json!({ "documentId": "doc-1" }),
                    document_id: Some("doc-1".to_owned()),
                    ..GraphVertexRecord::default()
                }],
                edges: vec![GraphEdgeRecord {
                    source_id: "doc::doc-1".to_owned(),
                    target_id: "entity::1".to_owned(),
                    edge_type: "mentions".to_owned(),
                    weight: 1,
                    attributes: json!({ "documentId": "doc-1" }),
                    document_id: Some("doc-1".to_owned()),
                    layer: GraphLayer::Asserted,
                    ..GraphEdgeRecord::default()
                }],
            })
            .expect("first batch");
        graph
            .apply_batch(GraphMutationBatch {
                layer: GraphLayer::Asserted,
                scope: GraphMutationScope::Document {
                    document_id: "doc-1".to_owned(),
                },
                vertices: vec![GraphVertexRecord {
                    id: "doc::doc-1".to_owned(),
                    kind: "document".to_owned(),
                    weight: 1,
                    value: json!({ "kind": "document", "documentId": "doc-1" }),
                    attributes: json!({ "documentId": "doc-1" }),
                    document_id: Some("doc-1".to_owned()),
                    ..GraphVertexRecord::default()
                }],
                edges: Vec::new(),
            })
            .expect("replacement batch");

        let snapshot = graph.snapshot(false).expect("snapshot");
        assert!(snapshot.vertices.contains_key("doc::doc-1"));
        assert!(snapshot.outgoing.is_empty());
    }
}
