use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    CanonicalEventId, DocumentArchive, DocumentTemporalSubstrate, TemporalAnchorRecord,
    TemporalAxisId, TemporalAxisKind, TemporalAxisRecord, TemporalClaimAtom,
    TemporalConstraintRecord, TemporalReferenceEdge,
};
use phoenix_types::BiTemporalWindow;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalEventProfile {
    pub event_id: String,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub document_id: String,
    pub proposition_id: String,
    pub label: String,
    pub sentence_index: usize,
    pub axis_id: TemporalAxisId,
    pub axis_kind: TemporalAxisKind,
    pub normalized_predicate: String,
    pub event_fingerprint: String,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalTimexProfile {
    pub timex_id: String,
    pub document_id: String,
    pub proposition_id: Option<String>,
    pub sentence_index: usize,
    pub label: String,
    pub normalized_value: Option<String>,
    pub axis_id: TemporalAxisId,
    pub axis_kind: TemporalAxisKind,
    pub temporal: BiTemporalWindow,
    pub source_class: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalReviewCase {
    pub case_id: String,
    pub document_id: String,
    pub proposition_id: String,
    pub event_id: String,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub label: String,
    pub sentence_index: usize,
    pub axis_id: TemporalAxisId,
    pub axis_kind: TemporalAxisKind,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub anchor_candidate_ids: Vec<String>,
    #[serde(default)]
    pub explicit_timex_ids: Vec<String>,
    #[serde(default)]
    pub reference_event_ids: Vec<String>,
    #[serde(default)]
    pub source_classes: Vec<String>,
    pub has_explicit_timex: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TemporalNormalizedInputs {
    #[serde(default)]
    pub axes: Vec<TemporalAxisRecord>,
    #[serde(default)]
    pub event_profiles: Vec<TemporalEventProfile>,
    #[serde(default)]
    pub timex_profiles: Vec<TemporalTimexProfile>,
    #[serde(default)]
    pub review_cases: Vec<TemporalReviewCase>,
    #[serde(default)]
    pub claim_atoms: Vec<TemporalClaimAtom>,
    #[serde(default)]
    pub anchors: Vec<TemporalAnchorRecord>,
    #[serde(default)]
    pub reference_edges: Vec<TemporalReferenceEdge>,
    #[serde(default)]
    pub constraints: Vec<TemporalConstraintRecord>,
    #[serde(default)]
    pub diagnostics: BTreeMap<String, usize>,
}

pub fn normalize_temporal_inputs(archives: &[DocumentArchive]) -> TemporalNormalizedInputs {
    let mut axes = Vec::<TemporalAxisRecord>::new();
    let mut event_profiles = Vec::<TemporalEventProfile>::new();
    let mut timex_profiles = Vec::<TemporalTimexProfile>::new();
    let mut review_cases = Vec::<TemporalReviewCase>::new();
    let mut claim_atoms = Vec::<TemporalClaimAtom>::new();
    let mut anchors = Vec::<TemporalAnchorRecord>::new();
    let mut reference_edges = Vec::<TemporalReferenceEdge>::new();
    let mut constraints = Vec::<TemporalConstraintRecord>::new();
    let mut diagnostics = BTreeMap::<String, usize>::new();
    let mut seen_axes = FxHashSet::<String>::default();
    let mut seen_cases = FxHashSet::<String>::default();

    for archive in archives {
        let Some(substrate) = archive.temporal_substrate.as_ref() else {
            *diagnostics
                .entry("missing_temporal_substrate".to_owned())
                .or_default() += 1;
            continue;
        };

        let axis_kind_by_id = substrate
            .axis_records
            .iter()
            .map(|axis| (axis.axis_id.0.clone(), axis.kind))
            .collect::<FxHashMap<_, _>>();

        for axis in &substrate.axis_records {
            if seen_axes.insert(axis.axis_id.0.clone()) {
                axes.push(axis.clone());
            }
        }

        let proposition_by_id = substrate
            .propositions
            .iter()
            .map(|proposition| (proposition.proposition_id.to_string(), proposition))
            .collect::<FxHashMap<_, _>>();
        let anchors_by_event = build_anchor_index(substrate);
        let references_by_event = build_reference_index(substrate);

        for (event_id, proposition_id, label) in substrate
            .semantic_events
            .iter()
            .filter_map(|event| {
                event.event_id.as_ref().map(|id| {
                    (
                        id.0.clone(),
                        event.proposition_id.to_string(),
                        event.label.to_string(),
                    )
                })
            })
            .chain(substrate.semantic_states.iter().filter_map(|state| {
                state.state_id.as_ref().map(|id| {
                    (
                        id.0.clone(),
                        state.proposition_id.to_string(),
                        state.label.to_string(),
                    )
                })
            }))
            .chain(substrate.semantic_claims.iter().filter_map(|claim| {
                claim.claim_id.as_ref().map(|id| {
                    (
                        id.0.clone(),
                        claim.proposition_id.to_string(),
                        claim.label.to_string(),
                    )
                })
            }))
        {
            let Some(proposition) = proposition_by_id.get(proposition_id.as_str()) else {
                *diagnostics
                    .entry("temporal_node_missing_proposition".to_owned())
                    .or_default() += 1;
                continue;
            };
            let axis_id = anchors_by_event
                .get(&event_id)
                .and_then(|rows| rows.first())
                .map(|anchor| anchor.axis_id.clone())
                .unwrap_or_else(|| TemporalAxisId("axis:world".to_owned()));
            let axis_kind = axis_kind_by_id
                .get(axis_id.0.as_str())
                .copied()
                .unwrap_or(TemporalAxisKind::World);
            let temporal = anchors_by_event
                .get(&event_id)
                .and_then(|rows| rows.first())
                .map(|anchor| anchor.temporal.clone())
                .unwrap_or_else(|| recorded_temporal(archive.manifest.created_at));
            let profile = TemporalEventProfile {
                event_id: event_id.clone(),
                canonical_event_id: None,
                document_id: archive.manifest.document_id.clone(),
                proposition_id: proposition.proposition_id.to_string(),
                label: label.clone(),
                sentence_index: proposition.sentence_index,
                axis_id: axis_id.clone(),
                axis_kind,
                normalized_predicate: proposition
                    .predicate
                    .predicate
                    .to_string()
                    .to_ascii_lowercase(),
                event_fingerprint: format!(
                    "{}:{}:{}",
                    archive.manifest.document_id, proposition.proposition_id, event_id
                ),
                temporal: temporal.clone(),
                evidence_refs: vec![proposition.proposition_id.to_string()],
            };
            let anchor_rows = anchors_by_event.get(&event_id).cloned().unwrap_or_default();
            let reference_rows = references_by_event
                .get(&event_id)
                .cloned()
                .unwrap_or_default();
            let explicit_timex_ids = anchor_rows
                .iter()
                .filter(|row| row.anchor_kind == "explicit_timex")
                .filter_map(|row| row.timex_id.as_ref().map(|id| id.0.clone()))
                .collect::<Vec<_>>();
            let case = TemporalReviewCase {
                case_id: format!("tcase:{}:{}", archive.manifest.document_id, event_id),
                document_id: archive.manifest.document_id.clone(),
                proposition_id: proposition.proposition_id.to_string(),
                event_id: event_id.clone(),
                canonical_event_id: None,
                label,
                sentence_index: proposition.sentence_index,
                axis_id,
                axis_kind,
                temporal,
                anchor_candidate_ids: anchor_rows
                    .iter()
                    .map(|row| row.anchor_id.0.clone())
                    .collect(),
                explicit_timex_ids: explicit_timex_ids.clone(),
                reference_event_ids: reference_rows
                    .iter()
                    .filter_map(|edge| edge.target_event_id.clone())
                    .collect(),
                source_classes: anchor_rows
                    .iter()
                    .map(|row| row.source_class.clone())
                    .collect(),
                has_explicit_timex: !explicit_timex_ids.is_empty(),
            };
            if seen_cases.insert(case.case_id.clone()) {
                event_profiles.push(profile);
                review_cases.push(case);
            }
        }

        for timex in &substrate.timex_records {
            let axis_kind = axis_kind_by_id
                .get(timex.axis_id.0.as_str())
                .copied()
                .unwrap_or(TemporalAxisKind::World);
            timex_profiles.push(TemporalTimexProfile {
                timex_id: timex.timex_id.0.clone(),
                document_id: timex.document_id.clone(),
                proposition_id: timex.proposition_id.clone(),
                sentence_index: timex.sentence_index,
                label: timex.label.clone(),
                normalized_value: timex.normalized_value.clone(),
                axis_id: timex.axis_id.clone(),
                axis_kind,
                temporal: timex.temporal.clone(),
                source_class: timex.source_class.clone(),
            });
        }

        claim_atoms.extend(substrate.temporal_claims.clone());
        anchors.extend(substrate.anchor_candidates.clone());
        reference_edges.extend(substrate.reference_timex_edges.clone());
        reference_edges.extend(substrate.reference_event_edges.clone());
        constraints.extend(substrate.temporal_constraints.clone());
        *diagnostics
            .entry("surface_temporal_cue_count".to_owned())
            .or_default() += substrate.surface_temporal_cues.len();
    }

    axes.sort_by(|left, right| left.axis_id.0.cmp(&right.axis_id.0));
    event_profiles.sort_by(|left, right| {
        (
            left.document_id.as_str(),
            left.sentence_index,
            left.event_id.as_str(),
        )
            .cmp(&(
                right.document_id.as_str(),
                right.sentence_index,
                right.event_id.as_str(),
            ))
    });
    timex_profiles.sort_by(|left, right| {
        (
            left.document_id.as_str(),
            left.sentence_index,
            left.timex_id.as_str(),
        )
            .cmp(&(
                right.document_id.as_str(),
                right.sentence_index,
                right.timex_id.as_str(),
            ))
    });
    review_cases.sort_by(|left, right| {
        (
            left.document_id.as_str(),
            left.sentence_index,
            left.event_id.as_str(),
        )
            .cmp(&(
                right.document_id.as_str(),
                right.sentence_index,
                right.event_id.as_str(),
            ))
    });

    TemporalNormalizedInputs {
        axes,
        event_profiles,
        timex_profiles,
        review_cases,
        claim_atoms,
        anchors,
        reference_edges,
        constraints,
        diagnostics,
    }
}

fn build_anchor_index(
    substrate: &DocumentTemporalSubstrate,
) -> FxHashMap<String, Vec<TemporalAnchorRecord>> {
    let mut rows = FxHashMap::<String, Vec<TemporalAnchorRecord>>::default();
    for anchor in &substrate.anchor_candidates {
        if let Some(event_id) = anchor.event_id.clone() {
            rows.entry(event_id).or_default().push(anchor.clone());
        }
    }
    rows
}

fn build_reference_index(
    substrate: &DocumentTemporalSubstrate,
) -> FxHashMap<String, Vec<TemporalReferenceEdge>> {
    let mut rows = FxHashMap::<String, Vec<TemporalReferenceEdge>>::default();
    for edge in substrate
        .reference_timex_edges
        .iter()
        .chain(substrate.reference_event_edges.iter())
    {
        rows.entry(edge.source_event_id.clone())
            .or_default()
            .push(edge.clone());
    }
    rows
}

fn recorded_temporal(recorded_from: i64) -> BiTemporalWindow {
    BiTemporalWindow {
        valid_from: None,
        valid_to: None,
        recorded_from: Some(recorded_from),
        recorded_to: None,
    }
}
