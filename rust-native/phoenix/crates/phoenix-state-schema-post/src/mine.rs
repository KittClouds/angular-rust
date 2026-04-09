use std::collections::{BTreeMap, BTreeSet};

use phoenix_semantic_v2::{StateSlotCandidateId, StateSlotCandidateRecord, StateSlotFamilyId};

use crate::normalize::StateSchemaEvidenceRow;

pub fn mine_slot_candidates(rows: &[StateSchemaEvidenceRow]) -> Vec<StateSlotCandidateRecord> {
    let mut grouped = BTreeMap::<String, Vec<&StateSchemaEvidenceRow>>::new();
    for row in rows {
        grouped.entry(row.slot_key.clone()).or_default().push(row);
    }

    let mut candidates = Vec::new();
    for (slot_key, group) in grouped {
        let family_key = group
            .first()
            .map(|row| row.family_key.clone())
            .unwrap_or_else(|| "discovered".to_owned());
        let owner_type = group.first().map(|row| row.owner_type).unwrap_or_default();
        let value_type = group.first().map(|row| row.value_type).unwrap_or_default();
        let normalized_name = slot_key
            .rsplit('.')
            .next()
            .unwrap_or(slot_key.as_str())
            .to_owned();
        let support_count = group.iter().filter(|row| row.positive).count();
        let conflict_count = group.len().saturating_sub(support_count);
        let document_count = group
            .iter()
            .map(|row| row.source_document_id.clone())
            .collect::<BTreeSet<_>>()
            .len();
        let relation_families = group
            .iter()
            .map(|row| row.relation_family.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let value_samples = group
            .iter()
            .filter(|row| row.positive && !row.target_label.is_empty())
            .map(|row| row.target_label.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .take(8)
            .collect::<Vec<_>>();

        let canonicalization_score_millis = ((support_count.min(5) as u32) * 120
            + if relation_families.len() <= 1 {
                250
            } else {
                100
            }
            + if value_samples.len() <= 3 { 180 } else { 80 })
        .min(1000);
        let utility_score_millis = ((support_count.min(6) as u32) * 110
            + (document_count.min(4) as u32) * 90
            + if family_key == "role_preference" {
                40
            } else {
                180
            })
        .saturating_sub((conflict_count.min(4) as u32) * 120)
        .min(1000);

        candidates.push(StateSlotCandidateRecord {
            candidate_id: StateSlotCandidateId(format!("candidate:slot:{slot_key}")),
            family_id: StateSlotFamilyId(format!("family:{family_key}")),
            slot_key: slot_key.clone(),
            normalized_name,
            source_phrase: relation_families
                .first()
                .cloned()
                .unwrap_or_else(|| slot_key.clone()),
            owner_type,
            value_type,
            support_count,
            document_count,
            canonicalization_score_millis,
            utility_score_millis,
            conflict_count,
            relation_families,
            value_samples,
        });
    }

    candidates.sort_by(|left, right| left.slot_key.cmp(&right.slot_key));
    candidates
}
