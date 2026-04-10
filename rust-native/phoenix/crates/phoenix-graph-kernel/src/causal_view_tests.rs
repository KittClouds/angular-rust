use crate::{
    causal_path_candidate_views_from_snapshot, KernelBiTemporal, KernelEdge, KernelEdgeType,
    KernelGraphLayer, KernelGraphSnapshot, KernelVertex, KernelVertexId,
};
use serde_json::json;

#[test]
fn causal_candidate_views_borrow_edges_and_compute_features() {
    let snapshot = KernelGraphSnapshot {
        vertices: vec![
            vertex("a", "event", 1),
            vertex("b", "event", 2),
            vertex("c", "event", 3),
        ],
        asserted_edges: vec![causal_edge(
            "a",
            "b",
            0.8,
            KernelGraphLayer::Asserted,
            "supported",
            2,
        )],
        candidate_edges: vec![causal_edge(
            "b",
            "c",
            0.4,
            KernelGraphLayer::Candidate,
            "candidate",
            1,
        )],
    };

    let candidates = causal_path_candidate_views_from_snapshot(&snapshot, "c", 3, 8);
    let path = candidates
        .iter()
        .find(|candidate| candidate.source_vertex_id == "a")
        .expect("expected full causal path");

    assert_eq!(path.path_vertex_ids, vec!["a", "b", "c"]);
    assert_eq!(path.path_edges.len(), 2);
    assert_eq!(path.path_edges[0].source_id.0, "a");
    assert_eq!(path.features.depth, 2);
    assert_eq!(path.features.evidence_ref_count, 3);
    assert_eq!(path.features.candidate_edge_count, 1);
    assert_eq!(path.features.missing_intermediate_cause_count, 0);
    assert_eq!(path.features.temporal_consistency_ratio, 1.0);
    assert!(path.features.support_strength > 0.5);
}

#[test]
fn causal_candidate_views_collect_modalities_without_owned_edge_clones() {
    let snapshot = KernelGraphSnapshot {
        vertices: vec![
            claim("graph::claim::source", "reported"),
            vertex("source", "event", 1),
            vertex("target", "event", 2),
        ],
        asserted_edges: vec![
            causal_edge(
                "source",
                "target",
                0.9,
                KernelGraphLayer::Asserted,
                "supported",
                1,
            ),
            edge("source", "graph::claim::source", "supported_by"),
        ],
        candidate_edges: Vec::new(),
    };

    let candidates = causal_path_candidate_views_from_snapshot(&snapshot, "target", 2, 4);

    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].supporting_modalities, vec!["reported"]);
    assert_eq!(candidates[0].features.avg_confidence, 0.9);
}

fn vertex(id: &str, kind: &str, valid_from: i64) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(id.to_owned()),
        kind: kind.to_owned(),
        temporal: KernelBiTemporal {
            valid_from: Some(valid_from),
            valid_to: None,
            recorded_at: Some(valid_from),
            expired_at: None,
        },
        ..KernelVertex::default()
    }
}

fn claim(id: &str, modality: &str) -> KernelVertex {
    KernelVertex {
        value: json!({ "modality": modality }),
        ..vertex(id, "claim", 1)
    }
}

fn causal_edge(
    source_id: &str,
    target_id: &str,
    confidence: f64,
    layer: KernelGraphLayer,
    status: &str,
    evidence_count: usize,
) -> KernelEdge {
    let mut edge = edge(source_id, target_id, "causal_link");
    edge.layer = layer;
    edge.provenance.confidence = Some(confidence);
    edge.provenance.evidence_refs = (0..evidence_count)
        .map(|index| format!("evidence-{index}"))
        .collect();
    edge.attributes = json!({ "status": status });
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
