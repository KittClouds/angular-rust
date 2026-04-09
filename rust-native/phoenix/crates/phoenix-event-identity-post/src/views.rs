use phoenix_semantic_v2::{
    CanonicalEventCard, CanonicalEventRecord, EventIdentityHypothesis, EventIdentityState,
    EventMentionPacket,
};
use rustc_hash::{FxHashMap, FxHashSet};

pub fn build_canonical_event_cards(
    canonical_events: &[CanonicalEventRecord],
    mention_packets: &[EventMentionPacket],
    hypotheses: &[EventIdentityHypothesis],
) -> Vec<CanonicalEventCard> {
    let packet_by_id = mention_packets
        .iter()
        .map(|packet| (packet.mention_id.0.clone(), packet))
        .collect::<FxHashMap<_, _>>();
    let mut cards = Vec::<CanonicalEventCard>::new();

    for canonical in canonical_events {
        let mention_ids = canonical
            .mention_ids
            .iter()
            .map(|mention_id| mention_id.0.clone())
            .collect::<FxHashSet<_>>();
        let relevant_hypotheses = hypotheses
            .iter()
            .filter(|hypothesis| {
                mention_ids.contains(&hypothesis.left_mention_id.0)
                    || mention_ids.contains(&hypothesis.right_mention_id.0)
            })
            .collect::<Vec<_>>();

        let related_temporal_event_ids = canonical
            .mention_ids
            .iter()
            .filter_map(|mention_id| packet_by_id.get(&mention_id.0))
            .flat_map(|packet| packet.temporal_neighbor_event_ids.clone())
            .collect::<FxHashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let related_causal_event_ids = canonical
            .mention_ids
            .iter()
            .filter_map(|mention_id| packet_by_id.get(&mention_id.0))
            .flat_map(|packet| packet.causal_neighbor_event_ids.clone())
            .collect::<FxHashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        cards.push(CanonicalEventCard {
            canonical_event_id: canonical.canonical_event_id.clone(),
            canonical_label: canonical.canonical_label.clone(),
            normalized_predicate: canonical.normalized_predicate.clone(),
            event_type: canonical.event_type.clone(),
            mention_ids: canonical.mention_ids.clone(),
            document_ids: canonical.document_ids.clone(),
            strongest_time_anchor_ids: canonical.time_anchor_ids.clone(),
            strongest_participant_slots: canonical.participant_slots.clone(),
            related_temporal_event_ids,
            related_causal_event_ids,
            open_dispute_ids: relevant_hypotheses
                .iter()
                .filter(|hypothesis| hypothesis.relation == EventIdentityState::QuasiIdentity)
                .map(|hypothesis| hypothesis.hypothesis_id.clone())
                .collect(),
            incompatible_hypothesis_ids: relevant_hypotheses
                .iter()
                .filter(|hypothesis| hypothesis.relation == EventIdentityState::Incompatible)
                .map(|hypothesis| hypothesis.hypothesis_id.clone())
                .collect(),
            revision_start: canonical.first_seen_revision,
            revision_end: canonical.latest_seen_revision,
            evidence_refs: canonical.evidence_refs.clone(),
        });
    }

    cards.sort_by(|left, right| left.canonical_event_id.0.cmp(&right.canonical_event_id.0));
    cards
}
