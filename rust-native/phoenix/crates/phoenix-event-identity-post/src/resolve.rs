use std::collections::BTreeMap;
use std::hash::{Hash, Hasher};

use phoenix_semantic_v2::{
    CanonicalEventId, CanonicalEventRecord, EventIdentityDecisionId, EventIdentityDecisionKind,
    EventIdentityHypothesis, EventIdentityInvalidationRecord, EventIdentityLedgerRecord,
    EventIdentityMembershipId, EventIdentityMembershipRecord, EventIdentitySplitRecord,
    EventIdentityState, EventMentionPacket, EventParticipantSlot, TemporalAnchorId,
};
use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedEventIdentityBatch {
    pub canonical_events: Vec<CanonicalEventRecord>,
    pub memberships: Vec<EventIdentityMembershipRecord>,
    pub decisions: Vec<EventIdentityLedgerRecord>,
    pub decision_history: Vec<EventIdentityLedgerRecord>,
    pub invalidations: Vec<EventIdentityInvalidationRecord>,
    pub splits: Vec<EventIdentitySplitRecord>,
    pub canonical_by_mention: FxHashMap<String, CanonicalEventId>,
    pub diagnostics: BTreeMap<String, usize>,
}

pub fn resolve_canonical_events(
    scope_key: &str,
    packets: &[EventMentionPacket],
    hypotheses: &[EventIdentityHypothesis],
    created_at: i64,
) -> ResolvedEventIdentityBatch {
    let mut parent = packets
        .iter()
        .map(|packet| (packet.mention_id.0.clone(), packet.mention_id.0.clone()))
        .collect::<FxHashMap<_, _>>();
    for hypothesis in hypotheses {
        if hypothesis.relation == EventIdentityState::FullIdentity {
            union(
                &mut parent,
                &hypothesis.left_mention_id.0,
                &hypothesis.right_mention_id.0,
            );
        }
    }

    let packet_by_id = packets
        .iter()
        .map(|packet| (packet.mention_id.0.clone(), packet))
        .collect::<FxHashMap<_, _>>();
    let mut grouped = BTreeMap::<String, Vec<&EventMentionPacket>>::new();
    for packet in packets {
        let root = find_root(&mut parent, &packet.mention_id.0);
        grouped.entry(root).or_default().push(packet);
    }

    let mut canonical_events = Vec::<CanonicalEventRecord>::new();
    let mut memberships = Vec::<EventIdentityMembershipRecord>::new();
    let mut canonical_by_mention = FxHashMap::<String, CanonicalEventId>::default();
    for mentions in grouped.values_mut() {
        mentions.sort_by(|left, right| {
            (
                left.revision,
                left.document_id.as_str(),
                left.mention_id.0.as_str(),
            )
                .cmp(&(
                    right.revision,
                    right.document_id.as_str(),
                    right.mention_id.0.as_str(),
                ))
        });
        let canonical_id = build_canonical_event_id(scope_key, mentions);
        let canonical = build_canonical_event(&canonical_id, scope_key, mentions);
        for mention in mentions.iter() {
            canonical_by_mention.insert(mention.mention_id.0.clone(), canonical_id.clone());
            memberships.push(EventIdentityMembershipRecord {
                membership_id: EventIdentityMembershipId(format!(
                    "event-membership:{}:{}",
                    canonical_id.0, mention.mention_id.0
                )),
                canonical_event_id: canonical_id.clone(),
                mention_id: mention.mention_id.clone(),
                relation: EventIdentityState::FullIdentity,
                confidence_millis: canonical.confidence_millis,
                created_at,
            });
        }
        canonical_events.push(canonical);
    }

    let mut decisions = Vec::<EventIdentityLedgerRecord>::new();
    let mut invalidations = Vec::<EventIdentityInvalidationRecord>::new();
    let splits = Vec::<EventIdentitySplitRecord>::new();
    let mut diagnostics = BTreeMap::<String, usize>::new();

    for hypothesis in hypotheses {
        let decision_kind = match hypothesis.relation {
            EventIdentityState::FullIdentity => EventIdentityDecisionKind::Merge,
            EventIdentityState::Incompatible => EventIdentityDecisionKind::Invalidate,
            _ => EventIdentityDecisionKind::Link,
        };
        let canonical_event_id = if hypothesis.relation == EventIdentityState::FullIdentity {
            canonical_by_mention
                .get(&hypothesis.left_mention_id.0)
                .cloned()
        } else {
            None
        };
        let decision = EventIdentityLedgerRecord {
            decision_id: EventIdentityDecisionId(format!(
                "event-decision:{}:{}",
                hypothesis.hypothesis_id.0,
                relation_key(hypothesis.relation)
            )),
            hypothesis_id: Some(hypothesis.hypothesis_id.clone()),
            canonical_event_id: canonical_event_id.clone(),
            left_mention_id: Some(hypothesis.left_mention_id.clone()),
            right_mention_id: Some(hypothesis.right_mention_id.clone()),
            relation: hypothesis.relation,
            decision_kind,
            rationale: format!(
                "{} score={} arg={} time={} place={} graph={}",
                relation_key(hypothesis.relation),
                hypothesis.score_millis,
                hypothesis.argument_role_score_millis,
                hypothesis.time_score_millis,
                hypothesis.place_score_millis,
                hypothesis.neighborhood_score_millis
            ),
            evidence_refs: hypothesis.evidence_refs.clone(),
            created_at,
        };
        if hypothesis.relation == EventIdentityState::Incompatible {
            invalidations.push(EventIdentityInvalidationRecord {
                invalidation_id: format!("event-invalidation:{}", hypothesis.hypothesis_id.0),
                decision_id: decision.decision_id.clone(),
                canonical_event_id,
                rationale: "hard_incompatible".to_owned(),
                created_at,
            });
        }
        *diagnostics
            .entry(format!("decision:{}", relation_key(hypothesis.relation)))
            .or_default() += 1;
        decisions.push(decision);
    }

    decisions.sort_by(|left, right| left.decision_id.0.cmp(&right.decision_id.0));
    memberships.sort_by(|left, right| left.membership_id.0.cmp(&right.membership_id.0));
    canonical_events
        .sort_by(|left, right| left.canonical_event_id.0.cmp(&right.canonical_event_id.0));

    let decision_history = decisions.clone();
    let _ = packet_by_id;

    ResolvedEventIdentityBatch {
        canonical_events,
        memberships,
        decisions,
        decision_history,
        invalidations,
        splits,
        canonical_by_mention,
        diagnostics,
    }
}

fn build_canonical_event_id(scope_key: &str, mentions: &[&EventMentionPacket]) -> CanonicalEventId {
    let first = mentions.first().expect("mentions");
    let mut hasher = FxHasher::default();
    scope_key.hash(&mut hasher);
    for mention in mentions {
        mention.document_id.hash(&mut hasher);
        mention.revision.hash(&mut hasher);
        mention.mention_id.0.hash(&mut hasher);
        mention.event_id.hash(&mut hasher);
        mention.event_fingerprint.hash(&mut hasher);
    }
    CanonicalEventId(format!(
        "canonical-event:{}:{}:{:016x}",
        sanitize_token(scope_key),
        sanitize_token(&first.normalized_predicate),
        hasher.finish()
    ))
}

fn build_canonical_event(
    canonical_event_id: &CanonicalEventId,
    scope_key: &str,
    mentions: &[&EventMentionPacket],
) -> CanonicalEventRecord {
    let first = mentions.first().expect("mentions");
    let participant_slots = strongest_participant_slots(mentions);
    let place_labels = mentions
        .iter()
        .flat_map(|mention| mention.place_labels.clone())
        .collect::<FxHashSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let time_anchor_ids = mentions
        .iter()
        .flat_map(|mention| mention.time_anchor_ids.clone())
        .collect::<FxHashSet<TemporalAnchorId>>()
        .into_iter()
        .collect::<Vec<_>>();
    let confidence_millis = if mentions.len() > 1 { 920 } else { 700 };

    CanonicalEventRecord {
        canonical_event_id: canonical_event_id.clone(),
        scope_key: scope_key.to_owned(),
        canonical_label: first.label.clone(),
        normalized_predicate: first.normalized_predicate.clone(),
        event_type: first.event_type.clone(),
        source_semantics: first.source_semantics,
        modality_semantics: first.modality_semantics,
        realis: first.realis.clone(),
        mention_ids: mentions
            .iter()
            .map(|mention| mention.mention_id.clone())
            .collect(),
        document_ids: mentions
            .iter()
            .map(|mention| mention.document_id.clone())
            .collect::<FxHashSet<_>>()
            .into_iter()
            .collect(),
        participant_slots,
        place_labels,
        time_anchor_ids,
        first_seen_revision: mentions
            .iter()
            .map(|mention| mention.revision)
            .min()
            .unwrap_or_default(),
        latest_seen_revision: mentions
            .iter()
            .map(|mention| mention.revision)
            .max()
            .unwrap_or_default(),
        confidence_millis,
        evidence_refs: mentions
            .iter()
            .flat_map(|mention| mention.evidence_refs.clone())
            .collect(),
    }
}

fn strongest_participant_slots(mentions: &[&EventMentionPacket]) -> Vec<EventParticipantSlot> {
    let mut seen = FxHashSet::<String>::default();
    let mut slots = Vec::new();
    for mention in mentions {
        for slot in &mention.participant_slots {
            let key = format!(
                "{}:{}:{}",
                slot.role,
                slot.entity_id
                    .as_ref()
                    .map(|entity_id| entity_id.0.as_str())
                    .unwrap_or(""),
                slot.label.as_deref().unwrap_or("")
            );
            if seen.insert(key) {
                slots.push(slot.clone());
            }
        }
    }
    slots
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

fn sanitize_token(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .take(64)
        .collect()
}

fn find_root(parent: &mut FxHashMap<String, String>, key: &str) -> String {
    let mut current = key.to_owned();
    while let Some(next) = parent.get(&current).cloned() {
        if next == current {
            break;
        }
        current = next;
    }
    current
}

fn union(parent: &mut FxHashMap<String, String>, left: &str, right: &str) {
    let left_root = find_root(parent, left);
    let right_root = find_root(parent, right);
    if left_root == right_root {
        return;
    }
    let (winner, loser) = if left_root <= right_root {
        (left_root, right_root)
    } else {
        (right_root, left_root)
    };
    parent.insert(loser, winner);
}
