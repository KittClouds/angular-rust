use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    CausalChainRecord, CausalClaimStatus, CausalEdgeAddition, CausalMemoryCard,
    CounterfactualReviewRecord,
};

use crate::normalize::{node_key, CausalEventProfile};

pub fn build_causal_memory_cards(
    profiles: &[CausalEventProfile],
    edge_records: &[CausalEdgeAddition],
    committed_edges: &[CausalEdgeAddition],
    chains: &[CausalChainRecord],
    reviews: &[CounterfactualReviewRecord],
) -> Vec<CausalMemoryCard> {
    let mut cards = profiles
        .iter()
        .map(|profile| {
            (
                node_key(&profile.node),
                CausalMemoryCard {
                    node: profile.node.clone(),
                    canonical_event_id: profile.canonical_event_id.clone(),
                    document_id: profile.document_id.clone(),
                    label: profile.label.clone(),
                    sentence_index: profile.sentence_index,
                    incoming_edge_ids: Vec::new(),
                    outgoing_edge_ids: Vec::new(),
                    chain_ids: Vec::new(),
                    counterfactual_review_ids: Vec::new(),
                    why_this_event_matters: None,
                    strongest_upstream_cause: None,
                    most_fragile_downstream_effect: None,
                    open_disputes: Vec::new(),
                    evidence_refs: profile.evidence_refs.clone(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mut incoming_best = BTreeMap::<String, (&CausalEdgeAddition, u32)>::new();
    let mut outgoing_fragile = BTreeMap::<String, (&CausalEdgeAddition, u32)>::new();

    for edge in committed_edges {
        if let Some(card) = cards.get_mut(&node_key(&edge.source)) {
            card.outgoing_edge_ids.push(edge.edge_id.clone());
            extend_refs(&mut card.evidence_refs, &edge.evidence_refs);
        }
        if let Some(card) = cards.get_mut(&node_key(&edge.target)) {
            card.incoming_edge_ids.push(edge.edge_id.clone());
            extend_refs(&mut card.evidence_refs, &edge.evidence_refs);
        }
        incoming_best
            .entry(node_key(&edge.target))
            .and_modify(|entry| {
                if edge.confidence_millis > entry.1 {
                    *entry = (edge, edge.confidence_millis);
                }
            })
            .or_insert((edge, edge.confidence_millis));
        let fragility = 1000u32.saturating_sub(edge.confidence_millis.min(1000));
        outgoing_fragile
            .entry(node_key(&edge.source))
            .and_modify(|entry| {
                if fragility > entry.1 {
                    *entry = (edge, fragility);
                }
            })
            .or_insert((edge, fragility));
    }

    for edge in edge_records {
        if !matches!(
            edge.status,
            CausalClaimStatus::Deferred | CausalClaimStatus::Contradicted
        ) {
            continue;
        }
        let dispute_id = phoenix_semantic_v2::CausalReviewId(format!("dispute:{}", edge.edge_id.0));
        if let Some(card) = cards.get_mut(&node_key(&edge.source)) {
            card.open_disputes.push(dispute_id.clone());
            extend_refs(&mut card.evidence_refs, &edge.evidence_refs);
        }
        if let Some(card) = cards.get_mut(&node_key(&edge.target)) {
            card.open_disputes.push(dispute_id.clone());
            extend_refs(&mut card.evidence_refs, &edge.evidence_refs);
        }
    }

    for chain in chains {
        for node in &chain.nodes {
            if let Some(card) = cards.get_mut(&node_key(node)) {
                card.chain_ids.push(chain.chain_id.clone());
                extend_refs(&mut card.evidence_refs, &chain.evidence_refs);
            }
        }
    }

    for review in reviews {
        if let Some(card) = cards.get_mut(&node_key(&review.source)) {
            card.counterfactual_review_ids
                .push(review.review_id.clone());
            card.open_disputes.push(review.review_id.clone());
            extend_refs(&mut card.evidence_refs, &review.evidence_refs);
        }
        if let Some(card) = cards.get_mut(&node_key(&review.target)) {
            card.counterfactual_review_ids
                .push(review.review_id.clone());
            card.open_disputes.push(review.review_id.clone());
            extend_refs(&mut card.evidence_refs, &review.evidence_refs);
        }
    }

    for (key, card) in &mut cards {
        if let Some((edge, _)) = incoming_best.get(key) {
            card.strongest_upstream_cause = Some(edge.edge_id.clone());
        }
        if let Some((edge, _)) = outgoing_fragile.get(key) {
            card.most_fragile_downstream_effect = Some(edge.edge_id.clone());
        }
        card.why_this_event_matters = Some(describe_card(card));
        card.open_disputes
            .sort_by(|left, right| left.0.cmp(&right.0));
        card.open_disputes.dedup();
    }

    let mut values = cards.into_values().collect::<Vec<_>>();
    values.sort_by(|left, right| {
        (
            left.document_id.as_str(),
            left.sentence_index,
            left.label.as_str(),
        )
            .cmp(&(
                right.document_id.as_str(),
                right.sentence_index,
                right.label.as_str(),
            ))
    });
    values
}

fn describe_card(card: &CausalMemoryCard) -> String {
    format!(
        "incoming={} outgoing={} chains={} disputes={}",
        card.incoming_edge_ids.len(),
        card.outgoing_edge_ids.len(),
        card.chain_ids.len(),
        card.open_disputes.len()
    )
}

fn extend_refs(target: &mut Vec<String>, incoming: &[String]) {
    target.extend(incoming.iter().cloned());
    target.sort();
    target.dedup();
}
