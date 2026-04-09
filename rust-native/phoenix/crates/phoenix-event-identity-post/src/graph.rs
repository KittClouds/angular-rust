use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    EventIdentityHypothesis, EventIdentityHypothesisId, EventIdentityState, EventMentionPacket,
};
use rustc_hash::FxHashSet;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EventIdentityGraphStats {
    pub by_relation: BTreeMap<String, usize>,
}

pub fn build_identity_hypotheses(
    scope_key: &str,
    packets: &[EventMentionPacket],
) -> (
    Vec<EventIdentityHypothesis>,
    EventIdentityGraphStats,
    BTreeMap<String, usize>,
) {
    let mut hypotheses = Vec::<EventIdentityHypothesis>::new();
    let mut diagnostics = BTreeMap::<String, usize>::new();
    let mut seen = FxHashSet::<(String, String)>::default();

    for (left_index, left) in packets.iter().enumerate() {
        for right in packets.iter().skip(left_index + 1) {
            if !candidate_pair(left, right) {
                continue;
            }
            let pair_key = ordered_pair_key(&left.mention_id.0, &right.mention_id.0);
            if !seen.insert(pair_key) {
                continue;
            }
            let hypothesis = build_hypothesis(scope_key, left, right);
            *diagnostics
                .entry(format!("relation:{}", relation_key(hypothesis.relation)))
                .or_default() += 1;
            if hypothesis.blocked {
                *diagnostics
                    .entry("hard_blocked_hypotheses".to_owned())
                    .or_default() += 1;
            }
            hypotheses.push(hypothesis);
        }
    }

    hypotheses.sort_by(|left, right| {
        (
            left.left_mention_id.0.as_str(),
            left.right_mention_id.0.as_str(),
            relation_key(left.relation),
        )
            .cmp(&(
                right.left_mention_id.0.as_str(),
                right.right_mention_id.0.as_str(),
                relation_key(right.relation),
            ))
    });

    let mut stats = EventIdentityGraphStats::default();
    for hypothesis in &hypotheses {
        *stats
            .by_relation
            .entry(relation_key(hypothesis.relation).to_owned())
            .or_default() += 1;
    }
    (hypotheses, stats, diagnostics)
}

fn candidate_pair(left: &EventMentionPacket, right: &EventMentionPacket) -> bool {
    let shared_participants = overlap_count(
        left.participant_slots
            .iter()
            .filter_map(|slot| slot.entity_id.as_ref().map(|id| id.0.clone()))
            .collect::<Vec<_>>(),
        right
            .participant_slots
            .iter()
            .filter_map(|slot| slot.entity_id.as_ref().map(|id| id.0.clone()))
            .collect::<Vec<_>>(),
    );
    let same_predicate = left.normalized_predicate == right.normalized_predicate;
    let same_fingerprint = left.event_fingerprint == right.event_fingerprint;
    let nearby = left.document_id == right.document_id
        && left.sentence_index.abs_diff(right.sentence_index) <= 3;
    same_predicate || same_fingerprint || shared_participants > 0 || nearby
}

fn build_hypothesis(
    scope_key: &str,
    left: &EventMentionPacket,
    right: &EventMentionPacket,
) -> EventIdentityHypothesis {
    let argument_role_score_millis = argument_role_score(left, right).min(420);
    let time_score_millis = time_score(left, right).min(260);
    let place_score_millis = place_score(left, right).min(180);
    let neighborhood_score_millis = neighborhood_score(left, right).min(220);
    let discourse_score_millis = discourse_score(left, right).min(150);
    let lexical_score_millis = lexical_score(left, right).min(220);
    let blocked = hard_incompatible(left, right);
    let relation = if blocked {
        EventIdentityState::Incompatible
    } else if reports_on(left, right) {
        EventIdentityState::ReportsOn
    } else if subevent_of(left, right) {
        EventIdentityState::SubeventOf
    } else if member_of_collection(left, right) {
        EventIdentityState::MemberOfCollection
    } else {
        let total = total_score(
            argument_role_score_millis,
            time_score_millis,
            place_score_millis,
            neighborhood_score_millis,
            discourse_score_millis,
            lexical_score_millis,
        );
        if total >= 920 && argument_role_score_millis >= 240 {
            EventIdentityState::FullIdentity
        } else if total >= 620 && argument_role_score_millis >= 120 {
            if version_of(left, right) {
                EventIdentityState::VersionOf
            } else {
                EventIdentityState::QuasiIdentity
            }
        } else {
            EventIdentityState::Incompatible
        }
    };
    let score_millis = if blocked {
        -1000
    } else {
        total_score(
            argument_role_score_millis,
            time_score_millis,
            place_score_millis,
            neighborhood_score_millis,
            discourse_score_millis,
            lexical_score_millis,
        ) as i32
    };

    EventIdentityHypothesis {
        hypothesis_id: EventIdentityHypothesisId(format!(
            "event-hypothesis:{}:{}:{}",
            scope_key, left.mention_id.0, right.mention_id.0
        )),
        left_mention_id: left.mention_id.clone(),
        right_mention_id: right.mention_id.clone(),
        relation,
        score_millis,
        argument_role_score_millis,
        time_score_millis,
        place_score_millis,
        neighborhood_score_millis,
        discourse_score_millis,
        lexical_score_millis,
        blocked,
        evidence_refs: vec![left.event_id.clone(), right.event_id.clone()],
    }
}

fn ordered_pair_key(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_owned(), right.to_owned())
    } else {
        (right.to_owned(), left.to_owned())
    }
}

fn relation_key(relation: EventIdentityState) -> &'static str {
    match relation {
        EventIdentityState::FullIdentity => "full_identity",
        EventIdentityState::QuasiIdentity => "quasi_identity",
        EventIdentityState::MemberOfCollection => "member_of_collection",
        EventIdentityState::SubeventOf => "subevent_of",
        EventIdentityState::VersionOf => "version_of",
        EventIdentityState::ReportsOn => "reports_on",
        EventIdentityState::Incompatible => "incompatible",
    }
}

fn total_score(
    argument_role: u32,
    time: u32,
    place: u32,
    neighborhood: u32,
    discourse: u32,
    lexical: u32,
) -> u32 {
    argument_role + time + place + neighborhood + discourse + lexical
}

fn argument_role_score(left: &EventMentionPacket, right: &EventMentionPacket) -> u32 {
    let left_ids = left
        .participant_slots
        .iter()
        .filter_map(|slot| slot.entity_id.as_ref().map(|entity_id| entity_id.0.clone()))
        .collect::<Vec<_>>();
    let right_ids = right
        .participant_slots
        .iter()
        .filter_map(|slot| slot.entity_id.as_ref().map(|entity_id| entity_id.0.clone()))
        .collect::<Vec<_>>();
    let entity_overlap = overlap_count(left_ids, right_ids) as u32;
    let role_overlap = overlap_count(
        left.participant_slots
            .iter()
            .map(|slot| slot.role.clone())
            .collect::<Vec<_>>(),
        right
            .participant_slots
            .iter()
            .map(|slot| slot.role.clone())
            .collect::<Vec<_>>(),
    ) as u32;
    entity_overlap * 180 + role_overlap * 40
}

fn time_score(left: &EventMentionPacket, right: &EventMentionPacket) -> u32 {
    let timex_overlap = overlap_count(
        left.explicit_timex_ids
            .iter()
            .map(|id| id.0.clone())
            .collect::<Vec<_>>(),
        right
            .explicit_timex_ids
            .iter()
            .map(|id| id.0.clone())
            .collect::<Vec<_>>(),
    ) as u32;
    let anchor_overlap = overlap_count(
        left.time_anchor_ids
            .iter()
            .map(|id| id.0.clone())
            .collect::<Vec<_>>(),
        right
            .time_anchor_ids
            .iter()
            .map(|id| id.0.clone())
            .collect::<Vec<_>>(),
    ) as u32;
    timex_overlap * 180 + anchor_overlap * 80
}

fn place_score(left: &EventMentionPacket, right: &EventMentionPacket) -> u32 {
    overlap_count(left.place_labels.clone(), right.place_labels.clone()) as u32 * 160
}

fn neighborhood_score(left: &EventMentionPacket, right: &EventMentionPacket) -> u32 {
    let causal_overlap = overlap_count(
        left.causal_neighbor_event_ids.clone(),
        right.causal_neighbor_event_ids.clone(),
    ) as u32;
    let temporal_overlap = overlap_count(
        left.temporal_neighbor_event_ids.clone(),
        right.temporal_neighbor_event_ids.clone(),
    ) as u32;
    causal_overlap * 100 + temporal_overlap * 80
}

fn discourse_score(left: &EventMentionPacket, right: &EventMentionPacket) -> u32 {
    if left.document_id == right.document_id {
        let distance = left.sentence_index.abs_diff(right.sentence_index) as u32;
        if distance == 0 {
            150
        } else if distance <= 2 {
            100
        } else {
            40
        }
    } else if left.revision == right.revision {
        40
    } else {
        80
    }
}

fn lexical_score(left: &EventMentionPacket, right: &EventMentionPacket) -> u32 {
    if left.normalized_predicate == right.normalized_predicate {
        220
    } else if left.normalized_predicate.split_whitespace().any(|token| {
        right
            .normalized_predicate
            .split_whitespace()
            .any(|other| other == token)
    }) {
        80
    } else {
        0
    }
}

fn hard_incompatible(left: &EventMentionPacket, right: &EventMentionPacket) -> bool {
    if left.polarity_negative != right.polarity_negative
        && left.normalized_predicate == right.normalized_predicate
    {
        return true;
    }
    let left_entities = left
        .participant_slots
        .iter()
        .filter_map(|slot| slot.entity_id.as_ref().map(|entity_id| entity_id.0.clone()))
        .collect::<Vec<_>>();
    let right_entities = right
        .participant_slots
        .iter()
        .filter_map(|slot| slot.entity_id.as_ref().map(|entity_id| entity_id.0.clone()))
        .collect::<Vec<_>>();
    if !left_entities.is_empty()
        && !right_entities.is_empty()
        && overlap_count(left_entities, right_entities) == 0
        && left.normalized_predicate == right.normalized_predicate
    {
        return true;
    }
    !left.place_labels.is_empty()
        && !right.place_labels.is_empty()
        && overlap_count(left.place_labels.clone(), right.place_labels.clone()) == 0
        && left.normalized_predicate == right.normalized_predicate
}

fn reports_on(left: &EventMentionPacket, right: &EventMentionPacket) -> bool {
    let reporting = ["report", "reported", "say", "said", "announce", "tell"];
    let lexical = reporting.iter().any(|token| {
        left.normalized_predicate.contains(token) || right.normalized_predicate.contains(token)
    });
    lexical
        && (left.source_semantics != right.source_semantics
            || left.realis != right.realis
            || overlap_count(
                left.causal_neighbor_event_ids.clone(),
                vec![right.event_id.clone()],
            ) > 0)
}

fn subevent_of(left: &EventMentionPacket, right: &EventMentionPacket) -> bool {
    left.temporal_neighbor_event_ids
        .iter()
        .any(|id| id == &right.event_id)
        || right
            .temporal_neighbor_event_ids
            .iter()
            .any(|id| id == &left.event_id)
        || left
            .causal_neighbor_event_ids
            .iter()
            .any(|id| id == &right.event_id)
        || right
            .causal_neighbor_event_ids
            .iter()
            .any(|id| id == &left.event_id)
}

fn member_of_collection(left: &EventMentionPacket, right: &EventMentionPacket) -> bool {
    let left_entities = left
        .participant_slots
        .iter()
        .filter_map(|slot| slot.entity_id.as_ref().map(|entity_id| entity_id.0.clone()))
        .collect::<Vec<_>>();
    let right_entities = right
        .participant_slots
        .iter()
        .filter_map(|slot| slot.entity_id.as_ref().map(|entity_id| entity_id.0.clone()))
        .collect::<Vec<_>>();
    !left_entities.is_empty()
        && !right_entities.is_empty()
        && (is_subset(&left_entities, &right_entities)
            || is_subset(&right_entities, &left_entities))
        && left.normalized_predicate == right.normalized_predicate
}

fn version_of(left: &EventMentionPacket, right: &EventMentionPacket) -> bool {
    left.event_fingerprint == right.event_fingerprint && left.revision != right.revision
}

fn overlap_count(left: Vec<String>, right: Vec<String>) -> usize {
    if left.is_empty() || right.is_empty() {
        return 0;
    }
    let right_set = right.into_iter().collect::<FxHashSet<_>>();
    left.into_iter()
        .filter(|value| right_set.contains(value))
        .count()
}

fn is_subset(left: &[String], right: &[String]) -> bool {
    if left.is_empty() || right.is_empty() {
        return false;
    }
    let right_set = right.iter().cloned().collect::<FxHashSet<_>>();
    left.iter().all(|value| right_set.contains(value))
}
