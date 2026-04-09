use std::collections::BTreeMap;

use phoenix_semantic_v2::{DocumentArchive, ErScopePatchSidecar, EventMentionPacket};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EventIdentityNormalizedInputs {
    #[serde(default)]
    pub mention_packets: Vec<EventMentionPacket>,
    #[serde(default)]
    pub diagnostics: BTreeMap<String, usize>,
}

pub fn normalize_event_identity_inputs(
    archives: &[DocumentArchive],
    _er_sidecar: Option<&ErScopePatchSidecar>,
) -> EventIdentityNormalizedInputs {
    let mut mention_packets = Vec::<EventMentionPacket>::new();
    let mut diagnostics = BTreeMap::<String, usize>::new();

    for archive in archives {
        let Some(substrate) = archive.event_identity_substrate.as_ref() else {
            *diagnostics
                .entry("missing_event_identity_substrate".to_owned())
                .or_default() += 1;
            continue;
        };
        let entity_labels = archive
            .entities
            .iter()
            .map(|entity| (entity.entity_id.0.clone(), entity.canonical_name.clone()))
            .collect::<FxHashMap<_, _>>();

        for seed in &substrate.mention_seeds {
            let mut packet = EventMentionPacket {
                mention_id: seed.mention_id.clone(),
                event_id: seed.event_id.clone(),
                document_id: seed.document_id.clone(),
                proposition_id: seed.proposition_id.clone(),
                revision: seed.revision,
                label: seed.label.clone(),
                normalized_predicate: seed.normalized_predicate.clone(),
                event_type: seed.event_type.clone(),
                participant_slots: seed.participant_slots.clone(),
                place_labels: seed.place_labels.clone(),
                explicit_timex_ids: seed.explicit_timex_ids.clone(),
                time_anchor_ids: seed.time_anchor_ids.clone(),
                causal_neighbor_event_ids: seed.causal_neighbor_event_ids.clone(),
                temporal_neighbor_event_ids: seed.temporal_neighbor_event_ids.clone(),
                sentence_index: seed.sentence_index,
                clause_range: seed.clause_range,
                polarity_negative: seed.polarity_negative,
                source_semantics: seed.source_semantics,
                modality_semantics: seed.modality_semantics,
                realis: seed.realis.clone(),
                event_fingerprint: seed.event_fingerprint.clone(),
                evidence_refs: seed.evidence_refs.clone(),
            };

            for slot in &mut packet.participant_slots {
                if slot
                    .label
                    .as_ref()
                    .map(|label| label.is_empty())
                    .unwrap_or(true)
                {
                    if let Some(entity_id) = slot.entity_id.as_ref() {
                        if let Some(label) = entity_labels.get(&entity_id.0) {
                            slot.label = Some(label.clone());
                        }
                    }
                }
            }

            mention_packets.push(packet);
        }

        if substrate.mention_seeds.is_empty() {
            *diagnostics
                .entry("empty_event_identity_seeds".to_owned())
                .or_default() += 1;
        }
        for diagnostic in &substrate.diagnostics {
            *diagnostics.entry(diagnostic.code.clone()).or_default() += 1;
        }
    }

    mention_packets.sort_by(|left, right| {
        (
            left.document_id.as_str(),
            left.revision,
            left.sentence_index,
            left.mention_id.0.as_str(),
        )
            .cmp(&(
                right.document_id.as_str(),
                right.revision,
                right.sentence_index,
                right.mention_id.0.as_str(),
            ))
    });

    EventIdentityNormalizedInputs {
        mention_packets,
        diagnostics,
    }
}
