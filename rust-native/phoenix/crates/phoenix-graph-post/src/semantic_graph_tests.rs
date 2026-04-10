use phoenix_graph_kernel::KernelMutationScope;
use phoenix_semantic_v2::{
    SemanticCandidateStatus, SemanticEdgeFamily, SemanticGraphEdgeCandidate, SemanticGraphNodeKind,
    SemanticGraphNodeRecord,
};

use crate::semantic_graph::{compile_candidate_graph_batch, summarize};
use crate::semantic_graph_support::{Prototype, ENTITY_KIND, EVENT_KIND, STATE_KIND};

fn prototype(node_id: &str, ann_kind: &'static str, node_kind: SemanticGraphNodeKind) -> Prototype {
    Prototype {
        node_id: node_id.to_owned(),
        ann_kind,
        node_kind,
        text_key: node_id.to_owned(),
        text: node_id.to_owned(),
        truth_plane: Some("world".to_owned()),
        document_id: None,
        narrative_id: None,
        evidence_refs: Vec::new(),
        semantic_node: SemanticGraphNodeRecord {
            node_id: node_id.to_owned(),
            node_kind,
            document_id: None,
            narrative_id: None,
            text_key: node_id.to_owned(),
            text_hash: 1,
            truth_plane: Some("world".to_owned()),
            evidence_refs: Vec::new(),
        },
        slot_key: Some("entity.employer".to_owned()),
        value_key: Some("acme".to_owned()),
        primary_entity_id: None,
        secondary_entity_id: None,
    }
}

#[test]
fn compile_candidate_graph_batch_skips_rejected_edges() {
    let prototypes = vec![
        prototype("entity::alice", ENTITY_KIND, SemanticGraphNodeKind::Entity),
        prototype("graph::state::1", STATE_KIND, SemanticGraphNodeKind::State),
        prototype("graph::event::1", EVENT_KIND, SemanticGraphNodeKind::Event),
    ];
    let candidates = vec![
        SemanticGraphEdgeCandidate {
            edge_id: "edge-1".to_owned(),
            family: SemanticEdgeFamily::EntityStateSupport,
            source_node_id: "entity::alice".to_owned(),
            source_kind: SemanticGraphNodeKind::Entity,
            target_node_id: "graph::state::1".to_owned(),
            target_kind: SemanticGraphNodeKind::State,
            score_millis: 870,
            distance_millis: 120,
            candidate_status: SemanticCandidateStatus::ReviewedSupport,
            evidence_refs: Vec::new(),
            model_evidence: Vec::new(),
            nli_support_millis: Some(812),
            nli_contradiction_millis: Some(131),
        },
        SemanticGraphEdgeCandidate {
            edge_id: "edge-2".to_owned(),
            family: SemanticEdgeFamily::EntityEventSupport,
            source_node_id: "entity::alice".to_owned(),
            source_kind: SemanticGraphNodeKind::Entity,
            target_node_id: "graph::event::1".to_owned(),
            target_kind: SemanticGraphNodeKind::Event,
            score_millis: 410,
            distance_millis: 420,
            candidate_status: SemanticCandidateStatus::Rejected,
            evidence_refs: Vec::new(),
            model_evidence: Vec::new(),
            nli_support_millis: None,
            nli_contradiction_millis: None,
        },
    ];

    let batch = compile_candidate_graph_batch("scope-key", &prototypes, &candidates, 42);

    assert_eq!(batch.edges.len(), 1);
    assert!(matches!(
        batch.scope,
        KernelMutationScope::Candidate { ref scope_key } if scope_key == "scope-key"
    ));
    assert_eq!(
        batch.edges[0]
            .attributes
            .get("nliSupportMillis")
            .and_then(serde_json::Value::as_u64),
        Some(812)
    );
}

#[test]
fn summarize_counts_reviewed_and_retained_edges() {
    let prototypes = vec![
        prototype("entity::alice", ENTITY_KIND, SemanticGraphNodeKind::Entity),
        prototype("graph::state::1", STATE_KIND, SemanticGraphNodeKind::State),
    ];
    let candidates = vec![
        SemanticGraphEdgeCandidate {
            edge_id: "edge-1".to_owned(),
            family: SemanticEdgeFamily::StateSupport,
            source_node_id: "graph::state::1".to_owned(),
            source_kind: SemanticGraphNodeKind::State,
            target_node_id: "graph::state::2".to_owned(),
            target_kind: SemanticGraphNodeKind::State,
            score_millis: 820,
            distance_millis: 140,
            candidate_status: SemanticCandidateStatus::ReviewedSupport,
            evidence_refs: Vec::new(),
            model_evidence: Vec::new(),
            nli_support_millis: Some(802),
            nli_contradiction_millis: Some(112),
        },
        SemanticGraphEdgeCandidate {
            edge_id: "edge-2".to_owned(),
            family: SemanticEdgeFamily::StateContradiction,
            source_node_id: "graph::state::1".to_owned(),
            source_kind: SemanticGraphNodeKind::State,
            target_node_id: "graph::state::3".to_owned(),
            target_kind: SemanticGraphNodeKind::State,
            score_millis: 600,
            distance_millis: 300,
            candidate_status: SemanticCandidateStatus::Rejected,
            evidence_refs: Vec::new(),
            model_evidence: Vec::new(),
            nli_support_millis: Some(220),
            nli_contradiction_millis: Some(230),
        },
    ];

    let summary = summarize(&prototypes, &candidates);

    assert_eq!(summary.node_count, 2);
    assert_eq!(summary.edge_count, 1);
    assert_eq!(summary.reviewed_support_count, 1);
    assert_eq!(summary.reviewed_contradiction_count, 0);
}
