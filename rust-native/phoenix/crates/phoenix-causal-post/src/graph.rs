use phoenix_semantic_v2::{
    CausalChainId, CausalChainRecord, CausalClaimStatus, CausalEdgeAddition, CounterfactualReason,
    CounterfactualReviewRecord,
};
use phoenix_types::{BiTemporalWindow, SemanticNodeRef};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::normalize::node_key;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CausalGraphStats {
    pub reverse_support_by_edge: FxHashMap<String, u32>,
    pub incoming_by_target: FxHashMap<String, usize>,
}

pub fn build_graph_stats(edges: &[CausalEdgeAddition]) -> CausalGraphStats {
    let mut reverse_support_by_edge = FxHashMap::default();
    let mut forward_by_signature = FxHashMap::<String, u32>::default();
    let mut incoming_by_target = FxHashMap::<String, usize>::default();

    for edge in edges {
        let signature = edge_signature(&edge.source, &edge.target, edge.relation_kind);
        let reverse = edge_signature(&edge.target, &edge.source, edge.relation_kind);
        forward_by_signature
            .entry(signature)
            .and_modify(|value| *value = (*value).max(edge.confidence_millis))
            .or_insert(edge.confidence_millis);
        *incoming_by_target
            .entry(node_key(&edge.target))
            .or_default() += 1;
        if let Some(reverse_score) = forward_by_signature.get(&reverse).copied() {
            reverse_support_by_edge.insert(edge.edge_id.0.clone(), reverse_score);
        }
    }

    CausalGraphStats {
        reverse_support_by_edge,
        incoming_by_target,
    }
}

pub fn build_chain_records(
    edges: &[CausalEdgeAddition],
    created_at: i64,
) -> Vec<CausalChainRecord> {
    let mut adjacency = FxHashMap::<String, Vec<&CausalEdgeAddition>>::default();
    for edge in edges {
        if !matches!(
            edge.status,
            CausalClaimStatus::Active | CausalClaimStatus::Supported
        ) {
            continue;
        }
        adjacency
            .entry(node_key(&edge.source))
            .or_default()
            .push(edge);
    }

    let mut chains = Vec::new();
    let mut seen = FxHashSet::default();
    for first in edges {
        if !matches!(
            first.status,
            CausalClaimStatus::Active | CausalClaimStatus::Supported
        ) {
            continue;
        }
        let Some(next_edges) = adjacency.get(&node_key(&first.target)) else {
            continue;
        };
        for second in next_edges {
            if first.document_id != second.document_id
                || first.relation_kind != second.relation_kind
            {
                continue;
            }
            if first.target == second.target || first.source == second.target {
                continue;
            }
            let chain_id = CausalChainId(format!(
                "chain:{}:{}:{}",
                first.document_id, first.edge_id.0, second.edge_id.0
            ));
            if !seen.insert(chain_id.0.clone()) {
                continue;
            }
            let weakest_status = if matches!(first.status, CausalClaimStatus::Supported)
                && matches!(second.status, CausalClaimStatus::Supported)
            {
                CausalClaimStatus::Supported
            } else {
                CausalClaimStatus::Active
            };
            let temporal_consistency_millis =
                temporal_consistency(&first.effective_interval, &second.effective_interval);
            let explanatory_strength_millis =
                ((first.confidence_millis + second.confidence_millis) / 2).min(1000);
            let speculative = matches!(
                first.relation_kind,
                phoenix_semantic_v2::CausalRelationKind::HypothesizedCause
                    | phoenix_semantic_v2::CausalRelationKind::MediatedCause
            ) || temporal_consistency_millis < 700;
            chains.push(CausalChainRecord {
                chain_id,
                document_id: first.document_id.clone(),
                kind: first.kind,
                relation_kind: first.relation_kind,
                nodes: vec![
                    first.source.clone(),
                    first.target.clone(),
                    second.target.clone(),
                ],
                canonical_event_ids: vec![
                    first.canonical_cause_event_id.clone(),
                    first.canonical_effect_event_id.clone(),
                    second.canonical_effect_event_id.clone(),
                ]
                .into_iter()
                .flatten()
                .collect(),
                edge_ids: vec![first.edge_id.clone(), second.edge_id.clone()],
                weakest_status,
                confidence_millis: explanatory_strength_millis,
                temporal: merge_temporal(&first.effective_interval, &second.effective_interval),
                temporal_consistency_millis,
                explanatory_strength_millis,
                speculative,
                evidence_refs: union_refs(&first.evidence_refs, &second.evidence_refs),
                created_at,
            });
        }
    }
    chains
}

pub fn build_counterfactual_reviews(
    edges: &[CausalEdgeAddition],
    chains: &[CausalChainRecord],
    created_at: i64,
) -> Vec<CounterfactualReviewRecord> {
    let mut by_target = FxHashMap::<String, Vec<&CausalEdgeAddition>>::default();
    let mut by_source = FxHashMap::<String, Vec<&CausalEdgeAddition>>::default();
    for edge in edges {
        by_target
            .entry(format!(
                "{}:{:?}",
                node_key(&edge.target),
                edge.relation_kind
            ))
            .or_default()
            .push(edge);
        by_source
            .entry(format!(
                "{}:{:?}",
                node_key(&edge.source),
                edge.relation_kind
            ))
            .or_default()
            .push(edge);
    }

    let mut reviews = Vec::new();
    for (_target, mut competing) in by_target {
        if competing.len() <= 1 {
            continue;
        }
        competing.sort_by(|left, right| right.confidence_millis.cmp(&left.confidence_millis));
        let best = competing[0];
        let support_path = chains
            .iter()
            .find(|chain| {
                chain
                    .edge_ids
                    .iter()
                    .any(|edge_id| edge_id == &best.edge_id)
            })
            .map(|chain| chain.chain_id.clone());
        for edge in competing.into_iter().skip(1) {
            reviews.push(CounterfactualReviewRecord {
                review_id: phoenix_semantic_v2::CausalReviewId(format!(
                    "review:{}:{}",
                    edge.edge_id.0, best.edge_id.0
                )),
                case_id: edge.case_id.clone(),
                focal_edge_id: edge.edge_id.clone(),
                document_id: edge.document_id.clone(),
                source: edge.source.clone(),
                canonical_cause_event_id: edge.canonical_cause_event_id.clone(),
                target: edge.target.clone(),
                canonical_effect_event_id: edge.canonical_effect_event_id.clone(),
                kind: edge.kind,
                relation_kind: edge.relation_kind,
                confidence_millis: edge.confidence_millis,
                review_reason: CounterfactualReason::CompetingCause,
                competing_cause_ids: vec![best.edge_id.clone()],
                blocker_events: Vec::new(),
                missing_intermediate_events: Vec::new(),
                only_support_path: support_path.clone(),
                rationale: format!("competing_cause stronger_edge={}", best.edge_id.0),
                evidence_refs: union_refs(&edge.evidence_refs, &best.evidence_refs),
                created_at,
            });
        }
    }

    for (_source, outgoing) in by_source {
        if outgoing.len() == 1 {
            let edge = outgoing[0];
            let only_support_path = chains
                .iter()
                .find(|chain| {
                    chain
                        .edge_ids
                        .iter()
                        .any(|edge_id| edge_id == &edge.edge_id)
                })
                .map(|chain| chain.chain_id.clone());
            if let Some(path_id) = only_support_path {
                reviews.push(CounterfactualReviewRecord {
                    review_id: phoenix_semantic_v2::CausalReviewId(format!(
                        "review:brittle:{}",
                        edge.edge_id.0
                    )),
                    case_id: edge.case_id.clone(),
                    focal_edge_id: edge.edge_id.clone(),
                    document_id: edge.document_id.clone(),
                    source: edge.source.clone(),
                    canonical_cause_event_id: edge.canonical_cause_event_id.clone(),
                    target: edge.target.clone(),
                    canonical_effect_event_id: edge.canonical_effect_event_id.clone(),
                    kind: edge.kind,
                    relation_kind: edge.relation_kind,
                    confidence_millis: edge.confidence_millis,
                    review_reason: CounterfactualReason::BrittleSupportPath,
                    competing_cause_ids: Vec::new(),
                    blocker_events: Vec::new(),
                    missing_intermediate_events: Vec::new(),
                    only_support_path: Some(path_id),
                    rationale: "only surviving support path for downstream effect".to_owned(),
                    evidence_refs: edge.evidence_refs.clone(),
                    created_at,
                });
            }
        }
    }

    reviews
}

fn temporal_consistency(left: &BiTemporalWindow, right: &BiTemporalWindow) -> u32 {
    let left_time = left.valid_from.or(left.recorded_from);
    let right_time = right.valid_from.or(right.recorded_from);
    match (left_time, right_time) {
        (Some(left_time), Some(right_time)) if left_time <= right_time => 1000,
        (Some(_), Some(_)) => 200,
        _ => 500,
    }
}

fn merge_temporal(left: &BiTemporalWindow, right: &BiTemporalWindow) -> BiTemporalWindow {
    BiTemporalWindow {
        valid_from: min_opt(left.valid_from, right.valid_from),
        valid_to: max_opt(left.valid_to, right.valid_to),
        recorded_from: min_opt(left.recorded_from, right.recorded_from),
        recorded_to: max_opt(left.recorded_to, right.recorded_to),
    }
}

fn min_opt(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn max_opt(left: Option<i64>, right: Option<i64>) -> Option<i64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.max(right)),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn union_refs(left: &[String], right: &[String]) -> Vec<String> {
    let mut values = left.iter().chain(right.iter()).cloned().collect::<Vec<_>>();
    values.sort();
    values.dedup();
    values
}

fn edge_signature(
    source: &SemanticNodeRef,
    target: &SemanticNodeRef,
    relation_kind: phoenix_semantic_v2::CausalRelationKind,
) -> String {
    format!(
        "{}:{}:{relation_kind:?}",
        node_key(source),
        node_key(target)
    )
}
