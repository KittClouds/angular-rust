use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    CanonicalEventId, CausalClaimAtom, CausalClaimId, CausalClaimPolarity, CausalClaimSourceKind,
    CausalEdgeId, CausalEvidenceClass, CausalRelationKind, DocumentArchive,
    DocumentCausalSubstrate, ErScopePatchSidecar, TemporalScopeSidecar,
};
use phoenix_types::{
    BiTemporalWindow, CausalCandidate, CausalKind, EntityId, Polarity, Proposition, ProvenanceRef,
    SemanticNodeRef, SourceRange, TruthStatus,
};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

const MAX_SOURCE_CLAIM_TRACE_SAMPLES: usize = 16;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalSourceSemantics {
    #[default]
    WorldAssertion,
    ReportedSpeech,
    AttributedClaim,
}

impl CausalSourceSemantics {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::WorldAssertion => "world_assertion",
            Self::ReportedSpeech => "reported_speech",
            Self::AttributedClaim => "attributed_claim",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalModalitySemantics {
    #[default]
    Asserted,
    Conditional,
    Planned,
    Hypothetical,
    Negated,
}

impl CausalModalitySemantics {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Asserted => "asserted",
            Self::Conditional => "conditional",
            Self::Planned => "planned",
            Self::Hypothetical => "hypothetical",
            Self::Negated => "negated",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalEventProfile {
    pub node: SemanticNodeRef,
    pub canonical_event_id: Option<CanonicalEventId>,
    pub document_id: String,
    pub proposition_id: String,
    pub label: String,
    pub sentence_index: usize,
    pub temporal: BiTemporalWindow,
    #[serde(default)]
    pub participant_entity_ids: Vec<EntityId>,
    pub attributed_to: Option<EntityId>,
    pub quoted: bool,
    pub negative: bool,
    #[serde(default)]
    pub source_semantics: CausalSourceSemantics,
    #[serde(default)]
    pub modality_semantics: CausalModalitySemantics,
    #[serde(default)]
    pub normalized_predicate: String,
    #[serde(default)]
    pub event_fingerprint: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalReviewCase {
    pub case_id: String,
    pub document_id: String,
    pub revision: u64,
    pub source: SemanticNodeRef,
    pub canonical_cause_event_id: Option<CanonicalEventId>,
    pub target: SemanticNodeRef,
    pub canonical_effect_event_id: Option<CanonicalEventId>,
    pub kind: CausalKind,
    pub relation_kind: CausalRelationKind,
    pub base_confidence_millis: u32,
    pub base_status: TruthStatus,
    pub cue: Option<String>,
    pub polarity: Polarity,
    pub attributed_to: Option<EntityId>,
    pub temporal: BiTemporalWindow,
    pub source_sentence_index: usize,
    pub target_sentence_index: usize,
    pub sentence_distance: usize,
    pub temporal_legal: bool,
    pub quoted_evidence: bool,
    pub attributed_evidence: bool,
    pub quoted_or_attributed: bool,
    #[serde(default)]
    pub source_semantics: CausalSourceSemantics,
    #[serde(default)]
    pub modality_semantics: CausalModalitySemantics,
    pub shared_participant_count: usize,
    pub source_degree: usize,
    pub target_degree: usize,
    pub graph_support_count: usize,
    pub centrality_millis: u32,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    pub seed_source: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalNormalizedInputs {
    #[serde(default)]
    pub event_profiles: Vec<CausalEventProfile>,
    #[serde(default)]
    pub review_cases: Vec<CausalReviewCase>,
    #[serde(default)]
    pub claim_atoms: Vec<CausalClaimAtom>,
    #[serde(default)]
    pub shadow_local_pair_cases: Vec<CausalReviewCase>,
    #[serde(default)]
    pub shadow_local_pair_claim_atoms: Vec<CausalClaimAtom>,
    #[serde(default)]
    pub source_claim_trace: CausalSourceClaimTraceSummary,
    #[serde(default)]
    pub diagnostics: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalSourceClaimTraceSummary {
    pub total_source_claim_case_count: usize,
    pub with_event_sibling_count: usize,
    pub with_state_sibling_count: usize,
    pub with_both_siblings_count: usize,
    pub without_richer_sibling_count: usize,
    #[serde(default)]
    pub reason_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub seed_source_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub samples: Vec<CausalSourceClaimTrace>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalSourceClaimTrace {
    pub document_id: String,
    pub seed_source: String,
    pub claim_id: String,
    pub claim_label: String,
    pub proposition_id: String,
    pub proposition_predicate: String,
    pub target_node_kind: String,
    pub target_label: String,
    pub sibling_event_count: usize,
    pub sibling_event_id: Option<String>,
    pub sibling_event_label: Option<String>,
    pub sibling_state_count: usize,
    pub sibling_state_id: Option<String>,
    pub sibling_state_label: Option<String>,
    pub trace_reason: String,
}

pub fn normalize_causal_inputs(
    archives: &[DocumentArchive],
    er_sidecar: Option<&ErScopePatchSidecar>,
) -> CausalNormalizedInputs {
    normalize_causal_inputs_with_sidecars(archives, er_sidecar, None)
}

pub fn normalize_causal_inputs_with_sidecars(
    archives: &[DocumentArchive],
    er_sidecar: Option<&ErScopePatchSidecar>,
    temporal_sidecar: Option<&TemporalScopeSidecar>,
) -> CausalNormalizedInputs {
    let mut event_profiles = Vec::new();
    let mut raw_cases = Vec::new();
    let mut shadow_local_pair_cases = Vec::new();
    let mut source_claim_trace = CausalSourceClaimTraceSummary::default();
    let mut diagnostics = BTreeMap::<String, usize>::new();

    for archive in archives {
        let Some(substrate) = archive.causal_substrate.as_ref() else {
            *diagnostics
                .entry("missing_causal_substrate".to_owned())
                .or_default() += 1;
            continue;
        };

        let profile_map = build_event_profile_map(
            archive,
            substrate,
            er_sidecar,
            temporal_sidecar,
            &mut diagnostics,
        );
        if profile_map.is_empty() {
            *diagnostics
                .entry("empty_causal_profiles".to_owned())
                .or_default() += 1;
            continue;
        }
        event_profiles.extend(profile_map.values().cloned());

        let graph_stats = build_degree_map(substrate);
        let mut seen_case_keys = FxHashSet::default();
        let mut seen_shadow_case_keys = FxHashSet::default();
        let archive_case_start = raw_cases.len();

        for link in &substrate.causal_links {
            if let Some(case) = build_review_case(
                archive,
                &profile_map,
                &graph_stats,
                &link.source,
                &link.target,
                link.kind,
                u32::from(link.confidence_millis),
                link.status,
                link.cue.as_ref().map(ToString::to_string),
                link.polarity,
                link.attributed_to.clone(),
                provenance_refs(link.provenance.iter().map(provenance_ref_label)),
                "link",
            ) {
                let case_key = review_case_key(&case);
                if seen_case_keys.insert(case_key) {
                    raw_cases.push(case);
                }
            } else {
                *diagnostics
                    .entry("link_profile_gap".to_owned())
                    .or_default() += 1;
            }
        }

        for candidate in &substrate.causal_candidates {
            if let Some(case) = build_candidate_case(archive, &profile_map, &graph_stats, candidate)
            {
                let case_key = review_case_key(&case);
                if seen_case_keys.insert(case_key) {
                    raw_cases.push(case);
                }
            } else {
                *diagnostics
                    .entry("candidate_profile_gap".to_owned())
                    .or_default() += 1;
            }
        }

        accumulate_source_claim_trace(
            &mut source_claim_trace,
            archive,
            substrate,
            &raw_cases[archive_case_start..],
            &profile_map,
            &mut diagnostics,
        );

        let local_cases = build_local_fallback_cases(archive, &profile_map, &graph_stats);
        for case in local_cases {
            let case_key = review_case_key(&case);
            if seen_shadow_case_keys.insert(case_key) {
                shadow_local_pair_cases.push(case);
                *diagnostics
                    .entry("shadow_local_pair_case_count".to_owned())
                    .or_default() += 1;
            }
        }
    }

    raw_cases.sort_by(|left, right| {
        (
            left.document_id.as_str(),
            left.revision,
            left.source_sentence_index,
            left.target_sentence_index,
            left.case_id.as_str(),
        )
            .cmp(&(
                right.document_id.as_str(),
                right.revision,
                right.source_sentence_index,
                right.target_sentence_index,
                right.case_id.as_str(),
            ))
    });

    let claim_atoms = build_claim_atoms(&raw_cases);
    let shadow_local_pair_claim_atoms = build_claim_atoms(&shadow_local_pair_cases);

    CausalNormalizedInputs {
        event_profiles,
        review_cases: raw_cases,
        claim_atoms,
        shadow_local_pair_cases,
        shadow_local_pair_claim_atoms,
        source_claim_trace,
        diagnostics,
    }
}

fn build_event_profile_map(
    archive: &DocumentArchive,
    substrate: &DocumentCausalSubstrate,
    er_sidecar: Option<&ErScopePatchSidecar>,
    temporal_sidecar: Option<&TemporalScopeSidecar>,
    diagnostics: &mut BTreeMap<String, usize>,
) -> FxHashMap<String, CausalEventProfile> {
    let proposition_by_id = substrate
        .propositions
        .iter()
        .map(|proposition| (proposition.proposition_id.to_string(), proposition))
        .collect::<FxHashMap<_, _>>();
    let temporal_by_proposition = substrate
        .propositions
        .iter()
        .enumerate()
        .map(|(index, proposition)| {
            let temporal = substrate
                .temporal_bindings
                .get(index)
                .map(|binding| {
                    binding
                        .anchor
                        .as_ref()
                        .map(|anchor| anchor.interval.clone())
                        .unwrap_or_else(|| binding.recorded_window.clone())
                })
                .unwrap_or_else(|| BiTemporalWindow {
                    valid_from: Some(archive.manifest.created_at),
                    valid_to: None,
                    recorded_from: Some(archive.manifest.created_at),
                    recorded_to: None,
                });
            (proposition.proposition_id.to_string(), temporal)
        })
        .collect::<FxHashMap<_, _>>();
    let temporal_by_node = build_temporal_profile_override_map(
        archive.manifest.document_id.as_str(),
        temporal_sidecar,
    );
    let mention_ranges_by_id = archive
        .resolved_mentions
        .iter()
        .map(|mention| {
            (
                mention.mention_id.0.clone(),
                SourceRange::new(mention.range.start, mention.range.end),
            )
        })
        .collect::<FxHashMap<String, SourceRange>>();
    let mention_ranges_by_index = archive
        .mentions
        .iter()
        .enumerate()
        .map(|(index, mention)| {
            (
                index,
                SourceRange::new(mention.range.start, mention.range.end),
            )
        })
        .collect::<FxHashMap<usize, SourceRange>>();
    let mut profiles = FxHashMap::default();

    for event in &substrate.semantic_events {
        if let Some(profile) = build_profile_from_record(
            archive,
            event.event_id.clone().map(SemanticNodeRef::Event),
            event.label.to_string(),
            event.proposition_id.to_string(),
            &proposition_by_id,
            &temporal_by_node,
            &temporal_by_proposition,
            er_sidecar,
            &mention_ranges_by_id,
            &mention_ranges_by_index,
            diagnostics,
        ) {
            profiles.insert(node_key(&profile.node), profile);
        }
    }
    for state in &substrate.semantic_states {
        if let Some(profile) = build_profile_from_record(
            archive,
            state.state_id.clone().map(SemanticNodeRef::State),
            state.label.to_string(),
            state.proposition_id.to_string(),
            &proposition_by_id,
            &temporal_by_node,
            &temporal_by_proposition,
            er_sidecar,
            &mention_ranges_by_id,
            &mention_ranges_by_index,
            diagnostics,
        ) {
            profiles.insert(node_key(&profile.node), profile);
        }
    }
    for claim in &substrate.semantic_claims {
        if let Some(profile) = build_profile_from_record(
            archive,
            claim.claim_id.clone().map(SemanticNodeRef::Claim),
            claim.label.to_string(),
            claim.proposition_id.to_string(),
            &proposition_by_id,
            &temporal_by_node,
            &temporal_by_proposition,
            er_sidecar,
            &mention_ranges_by_id,
            &mention_ranges_by_index,
            diagnostics,
        ) {
            profiles.entry(node_key(&profile.node)).or_insert(profile);
        }
    }

    profiles
}

fn build_temporal_profile_override_map(
    document_id: &str,
    temporal_sidecar: Option<&TemporalScopeSidecar>,
) -> FxHashMap<String, BiTemporalWindow> {
    let Some(sidecar) = temporal_sidecar else {
        return FxHashMap::default();
    };

    let mut rows = FxHashMap::<String, BiTemporalWindow>::default();
    for interval in &sidecar.intervals {
        if interval.document_id == document_id {
            rows.insert(interval.event_id.clone(), interval.temporal.clone());
        }
    }
    for card in &sidecar.memory_cards {
        if card.document_id != document_id {
            continue;
        }
        if let Some(interval) = card.strongest_interval.as_ref() {
            rows.entry(card.event_id.clone())
                .or_insert_with(|| interval.clone());
        }
    }
    for anchor in &sidecar.anchors {
        if anchor.document_id != document_id {
            continue;
        }
        let Some(event_id) = anchor.event_id.as_ref() else {
            continue;
        };
        rows.entry(event_id.clone())
            .or_insert_with(|| anchor.temporal.clone());
    }
    rows
}

fn accumulate_source_claim_trace(
    summary: &mut CausalSourceClaimTraceSummary,
    archive: &DocumentArchive,
    substrate: &DocumentCausalSubstrate,
    cases: &[CausalReviewCase],
    profile_map: &FxHashMap<String, CausalEventProfile>,
    diagnostics: &mut BTreeMap<String, usize>,
) {
    let sibling_events = sibling_event_index(substrate);
    let sibling_states = sibling_state_index(substrate);
    for case in cases {
        let SemanticNodeRef::Claim(claim_id) = &case.source else {
            continue;
        };
        let Some(source_profile) = profile_map.get(&node_key(&case.source)) else {
            *diagnostics
                .entry("source_claim_trace_missing_source_profile".to_owned())
                .or_default() += 1;
            continue;
        };
        let target_profile = profile_map.get(&node_key(&case.target));
        let proposition_id = source_profile.proposition_id.as_str();
        let sibling_event = sibling_events.get(proposition_id);
        let sibling_state = sibling_states.get(proposition_id);
        let trace_reason =
            claim_source_trace_reason(sibling_event.is_some(), sibling_state.is_some()).to_owned();

        summary.total_source_claim_case_count += 1;
        summary.with_event_sibling_count += sibling_event.is_some() as usize;
        summary.with_state_sibling_count += sibling_state.is_some() as usize;
        summary.with_both_siblings_count +=
            (sibling_event.is_some() && sibling_state.is_some()) as usize;
        summary.without_richer_sibling_count +=
            (!sibling_event.is_some() && !sibling_state.is_some()) as usize;
        *summary
            .reason_counts
            .entry(trace_reason.clone())
            .or_default() += 1;
        *summary
            .seed_source_counts
            .entry(case.seed_source.clone())
            .or_default() += 1;
        *diagnostics
            .entry(format!("source_claim_trace:{}", trace_reason))
            .or_default() += 1;

        if summary.samples.len() >= MAX_SOURCE_CLAIM_TRACE_SAMPLES {
            continue;
        }
        summary.samples.push(CausalSourceClaimTrace {
            document_id: archive.manifest.document_id.clone(),
            seed_source: case.seed_source.clone(),
            claim_id: claim_id.0.clone(),
            claim_label: source_profile.label.clone(),
            proposition_id: source_profile.proposition_id.clone(),
            proposition_predicate: source_profile.normalized_predicate.clone(),
            target_node_kind: semantic_node_kind(&case.target).to_owned(),
            target_label: target_profile
                .map(|profile| profile.label.clone())
                .unwrap_or_else(|| semantic_node_id(&case.target).to_owned()),
            sibling_event_count: sibling_event.map(|entry| entry.count).unwrap_or_default(),
            sibling_event_id: sibling_event.and_then(|entry| entry.node_id.clone()),
            sibling_event_label: sibling_event.and_then(|entry| entry.label.clone()),
            sibling_state_count: sibling_state.map(|entry| entry.count).unwrap_or_default(),
            sibling_state_id: sibling_state.and_then(|entry| entry.node_id.clone()),
            sibling_state_label: sibling_state.and_then(|entry| entry.label.clone()),
            trace_reason,
        });
    }
}

#[derive(Clone, Debug, Default)]
struct PropositionSiblingSummary {
    count: usize,
    node_id: Option<String>,
    label: Option<String>,
}

fn sibling_event_index(
    substrate: &DocumentCausalSubstrate,
) -> FxHashMap<String, PropositionSiblingSummary> {
    let mut rows = FxHashMap::<String, PropositionSiblingSummary>::default();
    for event in &substrate.semantic_events {
        let entry = rows.entry(event.proposition_id.to_string()).or_default();
        entry.count += 1;
        if entry.node_id.is_none() {
            entry.node_id = event.event_id.as_ref().map(|value| value.0.clone());
        }
        if entry.label.is_none() && !event.label.is_empty() {
            entry.label = Some(event.label.to_string());
        }
    }
    rows
}

fn sibling_state_index(
    substrate: &DocumentCausalSubstrate,
) -> FxHashMap<String, PropositionSiblingSummary> {
    let mut rows = FxHashMap::<String, PropositionSiblingSummary>::default();
    for state in &substrate.semantic_states {
        let entry = rows.entry(state.proposition_id.to_string()).or_default();
        entry.count += 1;
        if entry.node_id.is_none() {
            entry.node_id = state.state_id.as_ref().map(|value| value.0.clone());
        }
        if entry.label.is_none() && !state.label.is_empty() {
            entry.label = Some(state.label.to_string());
        }
    }
    rows
}

fn claim_source_trace_reason(has_event_sibling: bool, has_state_sibling: bool) -> &'static str {
    match (has_event_sibling, has_state_sibling) {
        (true, true) => "claim_with_event_and_state_sibling",
        (true, false) => "claim_with_event_sibling",
        (false, true) => "claim_with_state_sibling",
        (false, false) => "claim_only",
    }
}

fn build_profile_from_record(
    archive: &DocumentArchive,
    node: Option<SemanticNodeRef>,
    label: String,
    proposition_id: String,
    proposition_by_id: &FxHashMap<String, &Proposition>,
    temporal_by_node: &FxHashMap<String, BiTemporalWindow>,
    temporal_by_proposition: &FxHashMap<String, BiTemporalWindow>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    mention_ranges_by_id: &FxHashMap<String, SourceRange>,
    mention_ranges_by_index: &FxHashMap<usize, SourceRange>,
    diagnostics: &mut BTreeMap<String, usize>,
) -> Option<CausalEventProfile> {
    let node = node?;
    let proposition = proposition_by_id.get(&proposition_id)?;
    let temporal = match temporal_by_node.get(semantic_node_id(&node)) {
        Some(window) => {
            *diagnostics
                .entry("profile_temporal:temporal_sidecar".to_owned())
                .or_default() += 1;
            window.clone()
        }
        None => match temporal_by_proposition.get(&proposition.proposition_id.to_string()) {
            Some(window) => {
                *diagnostics
                    .entry("profile_temporal:archive_binding".to_owned())
                    .or_default() += 1;
                window.clone()
            }
            None => {
                *diagnostics
                    .entry("profile_temporal:recorded_fallback".to_owned())
                    .or_default() += 1;
                BiTemporalWindow {
                    valid_from: Some(archive.manifest.created_at),
                    valid_to: None,
                    recorded_from: Some(archive.manifest.created_at),
                    recorded_to: None,
                }
            }
        },
    };
    let proposition_window = proposition_window(proposition, mention_ranges_by_index);
    let mut participants = proposition
        .arguments
        .iter()
        .filter_map(|argument| {
            let entity_id = argument.entity_id.clone()?;
            *diagnostics
                .entry("participant_source:argument".to_owned())
                .or_default() += 1;
            Some(entity_id)
        })
        .collect::<Vec<_>>();
    if let Some(sidecar) = er_sidecar {
        for link in &sidecar.entity_links {
            if link.document_id != archive.manifest.document_id {
                continue;
            }
            let Some(mention_id) = link.mention_id.as_ref() else {
                *diagnostics
                    .entry("participant_source:er_unscoped_skipped".to_owned())
                    .or_default() += 1;
                continue;
            };
            let Some(mention_range) = mention_ranges_by_id.get(&mention_id.0).copied() else {
                *diagnostics
                    .entry("participant_source:er_missing_mention_range".to_owned())
                    .or_default() += 1;
                continue;
            };
            if ranges_overlap(proposition_window, mention_range) {
                participants.push(link.entity_id.clone());
                *diagnostics
                    .entry("participant_source:er_local_overlap".to_owned())
                    .or_default() += 1;
            } else {
                *diagnostics
                    .entry("participant_source:er_document_spill_rejected".to_owned())
                    .or_default() += 1;
            }
        }
    }
    participants.sort();
    participants.dedup();

    let evidence_refs = provenance_refs(proposition.evidence.iter().map(provenance_ref_label));
    let negative = proposition
        .scope_ops
        .iter()
        .any(|scope| scope.polarity.as_deref() == Some("negative"));
    let source_semantics = source_semantics_for(proposition);
    let modality_semantics = modality_semantics_for(proposition, negative);
    let normalized_predicate = normalize_predicate_label(&proposition.predicate.predicate, &label);
    let event_fingerprint = build_event_fingerprint(
        &normalized_predicate,
        &participants,
        source_semantics,
        modality_semantics,
    );
    *diagnostics
        .entry(format!("profile_source:{}", source_semantics.as_str()))
        .or_default() += 1;
    *diagnostics
        .entry(format!("profile_modality:{}", modality_semantics.as_str()))
        .or_default() += 1;

    Some(CausalEventProfile {
        node,
        canonical_event_id: None,
        document_id: archive.manifest.document_id.clone(),
        proposition_id,
        label,
        sentence_index: proposition.sentence_index,
        temporal,
        participant_entity_ids: participants,
        attributed_to: proposition
            .attribution
            .as_ref()
            .and_then(|frame| frame.source_entity_id.clone()),
        quoted: proposition_is_quoted(proposition),
        negative,
        source_semantics,
        modality_semantics,
        normalized_predicate,
        event_fingerprint,
        evidence_refs,
    })
}

fn build_candidate_case(
    archive: &DocumentArchive,
    profile_map: &FxHashMap<String, CausalEventProfile>,
    graph_stats: &FxHashMap<String, usize>,
    candidate: &CausalCandidate,
) -> Option<CausalReviewCase> {
    build_review_case(
        archive,
        profile_map,
        graph_stats,
        &candidate.source,
        &candidate.target,
        candidate.kind,
        u32::from(candidate.confidence_millis),
        candidate.status,
        candidate.cue.as_ref().map(ToString::to_string),
        candidate.polarity,
        candidate.attributed_to.clone(),
        provenance_refs(candidate.provenance.iter().map(provenance_ref_label)),
        "candidate",
    )
}

fn build_review_case(
    archive: &DocumentArchive,
    profile_map: &FxHashMap<String, CausalEventProfile>,
    graph_stats: &FxHashMap<String, usize>,
    source: &SemanticNodeRef,
    target: &SemanticNodeRef,
    kind: CausalKind,
    base_confidence_millis: u32,
    base_status: TruthStatus,
    cue: Option<String>,
    polarity: Polarity,
    attributed_to: Option<EntityId>,
    evidence_refs: Vec<String>,
    seed_source: &str,
) -> Option<CausalReviewCase> {
    let source_profile = profile_map.get(&node_key(source))?;
    let target_profile = profile_map.get(&node_key(target))?;
    if source_profile.document_id != target_profile.document_id {
        return None;
    }
    let sentence_distance = source_profile
        .sentence_index
        .max(target_profile.sentence_index)
        .saturating_sub(
            source_profile
                .sentence_index
                .min(target_profile.sentence_index),
        );
    if sentence_distance > 1 {
        return None;
    }
    let shared_participant_count = count_shared_participants(
        &source_profile.participant_entity_ids,
        &target_profile.participant_entity_ids,
    );
    let source_degree = *graph_stats.get(&node_key(source)).unwrap_or(&0usize);
    let target_degree = *graph_stats.get(&node_key(target)).unwrap_or(&0usize);
    let graph_support_count = source_degree.min(target_degree);
    let centrality_millis = ((source_degree + target_degree).min(6) as u32) * 110;
    let temporal_legal = temporal_precedes(&source_profile.temporal, &target_profile.temporal);
    let source_semantics = merge_source_semantics(
        source_profile.source_semantics,
        target_profile.source_semantics,
        attributed_to.is_some(),
    );
    let modality_semantics = merge_modality_semantics(
        source_profile.modality_semantics,
        target_profile.modality_semantics,
    );
    let quoted_evidence = matches!(source_semantics, CausalSourceSemantics::ReportedSpeech);
    let attributed_evidence = attributed_to.is_some()
        || matches!(source_semantics, CausalSourceSemantics::AttributedClaim);
    let quoted_or_attributed = quoted_evidence || attributed_evidence;
    let case_id = format!(
        "{}:{}:{}:{}:{:?}:r{}",
        archive.manifest.document_id,
        source_profile.proposition_id,
        target_profile.proposition_id,
        seed_source,
        kind,
        archive.manifest.revision
    );

    Some(CausalReviewCase {
        case_id,
        document_id: archive.manifest.document_id.clone(),
        revision: archive.manifest.revision,
        source: source.clone(),
        canonical_cause_event_id: source_profile.canonical_event_id.clone(),
        target: target.clone(),
        canonical_effect_event_id: target_profile.canonical_event_id.clone(),
        kind,
        relation_kind: map_relation_kind(kind),
        base_confidence_millis,
        base_status,
        cue,
        polarity,
        attributed_to,
        temporal: merge_temporal(&source_profile.temporal, &target_profile.temporal),
        source_sentence_index: source_profile.sentence_index,
        target_sentence_index: target_profile.sentence_index,
        sentence_distance,
        temporal_legal,
        quoted_evidence,
        attributed_evidence,
        quoted_or_attributed,
        source_semantics,
        modality_semantics,
        shared_participant_count,
        source_degree,
        target_degree,
        graph_support_count,
        centrality_millis,
        evidence_refs,
        seed_source: seed_source.to_owned(),
    })
}

fn build_local_fallback_cases(
    archive: &DocumentArchive,
    profile_map: &FxHashMap<String, CausalEventProfile>,
    graph_stats: &FxHashMap<String, usize>,
) -> Vec<CausalReviewCase> {
    let mut profiles = profile_map.values().collect::<Vec<_>>();
    profiles.sort_by_key(|profile| (profile.sentence_index, profile.proposition_id.as_str()));
    let mut cases = Vec::new();
    for left_index in 0..profiles.len() {
        let left = profiles[left_index];
        for right in profiles.iter().skip(left_index + 1).copied() {
            let distance = right.sentence_index.saturating_sub(left.sentence_index);
            if distance > 1 {
                break;
            }
            if left.quoted
                || right.quoted
                || left.attributed_to.is_some()
                || right.attributed_to.is_some()
            {
                continue;
            }
            let shared = count_shared_participants(
                &left.participant_entity_ids,
                &right.participant_entity_ids,
            );
            if shared == 0 && distance > 0 {
                continue;
            }
            if let Some(case) = build_review_case(
                archive,
                profile_map,
                graph_stats,
                &left.node,
                &right.node,
                CausalKind::ResultsIn,
                340,
                TruthStatus::Candidate,
                None,
                Polarity::Positive,
                None,
                provenance_refs(
                    left.evidence_refs
                        .iter()
                        .chain(right.evidence_refs.iter())
                        .cloned(),
                ),
                "local_pair",
            ) {
                cases.push(case);
            }
        }
    }
    cases
}

fn build_degree_map(substrate: &DocumentCausalSubstrate) -> FxHashMap<String, usize> {
    let mut degrees = FxHashMap::<String, usize>::default();
    for link in &substrate.causal_links {
        *degrees.entry(node_key(&link.source)).or_default() += 1;
        *degrees.entry(node_key(&link.target)).or_default() += 1;
    }
    for candidate in &substrate.causal_candidates {
        *degrees.entry(node_key(&candidate.source)).or_default() += 1;
        *degrees.entry(node_key(&candidate.target)).or_default() += 1;
    }
    degrees
}

fn temporal_precedes(source: &BiTemporalWindow, target: &BiTemporalWindow) -> bool {
    let source_time = source.valid_from.or(source.recorded_from);
    let target_time = target.valid_from.or(target.recorded_from);
    !matches!((source_time, target_time), (Some(source_time), Some(target_time)) if source_time > target_time)
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

fn count_shared_participants(left: &[EntityId], right: &[EntityId]) -> usize {
    let right_set = right.iter().collect::<FxHashSet<_>>();
    left.iter()
        .filter(|entity_id| right_set.contains(entity_id))
        .count()
}

fn proposition_is_quoted(proposition: &Proposition) -> bool {
    proposition.quote.is_some()
}

fn provenance_ref_label(value: &ProvenanceRef) -> String {
    format!(
        "{}:{}:{}-{}",
        value
            .document_id
            .as_ref()
            .map(|document_id| document_id.0.as_str())
            .unwrap_or("doc"),
        value.label,
        value.range.start,
        value.range.end
    )
}

fn provenance_refs<I>(values: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut refs = values
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    refs.sort();
    refs.dedup();
    refs
}

fn review_case_key(case: &CausalReviewCase) -> String {
    format!(
        "{}:{}:{}:{:?}:r{}",
        case.document_id,
        node_key(&case.source),
        node_key(&case.target),
        case.kind,
        case.revision
    )
}

pub(crate) fn node_key(node: &SemanticNodeRef) -> String {
    match node {
        SemanticNodeRef::Event(id) => format!("event:{}", id.0),
        SemanticNodeRef::Claim(id) => format!("claim:{}", id.0),
        SemanticNodeRef::State(id) => format!("state:{}", id.0),
    }
}

pub(crate) fn semantic_node_id(node: &SemanticNodeRef) -> &str {
    match node {
        SemanticNodeRef::Event(id) => id.0.as_str(),
        SemanticNodeRef::Claim(id) => id.0.as_str(),
        SemanticNodeRef::State(id) => id.0.as_str(),
    }
}

fn semantic_node_kind(node: &SemanticNodeRef) -> &'static str {
    match node {
        SemanticNodeRef::Event(_) => "event",
        SemanticNodeRef::Claim(_) => "claim",
        SemanticNodeRef::State(_) => "state",
    }
}

fn build_claim_atoms(cases: &[CausalReviewCase]) -> Vec<CausalClaimAtom> {
    let mut atoms = cases
        .iter()
        .map(|case| {
            let edge_id = stable_edge_id(case);
            let claim_id = CausalClaimId(format!("claim:{}:{}", edge_id.0, case.seed_source));
            CausalClaimAtom {
                claim_id,
                edge_id,
                document_id: case.document_id.clone(),
                cause_event: case.source.clone(),
                canonical_cause_event_id: case.canonical_cause_event_id.clone(),
                effect_event: case.target.clone(),
                canonical_effect_event_id: case.canonical_effect_event_id.clone(),
                kind: case.kind,
                relation_kind: case.relation_kind,
                source_kind: source_kind_for(case),
                polarity: polarity_for(case),
                evidence_class: evidence_class_for(case),
                strength_millis: case.base_confidence_millis,
                temporal: case.temporal.clone(),
                evidence_refs: case.evidence_refs.clone(),
                created_at: case
                    .temporal
                    .recorded_from
                    .or(case.temporal.valid_from)
                    .unwrap_or_default(),
            }
        })
        .collect::<Vec<_>>();
    atoms.sort_by(|left, right| left.claim_id.0.cmp(&right.claim_id.0));
    atoms.dedup_by(|left, right| left.claim_id == right.claim_id);
    atoms
}

pub(crate) fn stable_edge_id(case: &CausalReviewCase) -> CausalEdgeId {
    CausalEdgeId(format!(
        "edge:{}:{}:{}:{:?}",
        case.document_id,
        node_key(&case.source),
        node_key(&case.target),
        case.relation_kind
    ))
}

fn map_relation_kind(kind: CausalKind) -> CausalRelationKind {
    match kind {
        CausalKind::Causes => CausalRelationKind::DirectCause,
        CausalKind::ResultsIn => CausalRelationKind::MediatedCause,
        CausalKind::Enables | CausalKind::ConditionFor => CausalRelationKind::EnablingCondition,
        CausalKind::Prevents | CausalKind::Hinders => CausalRelationKind::PreventingFactor,
        CausalKind::TriggerFor => CausalRelationKind::Trigger,
        CausalKind::Explains | CausalKind::Motivates | CausalKind::PurposeOf => {
            CausalRelationKind::HypothesizedCause
        }
    }
}

fn source_kind_for(case: &CausalReviewCase) -> CausalClaimSourceKind {
    match case.seed_source.as_str() {
        "link" => CausalClaimSourceKind::ExplicitLink,
        "candidate" if case.cue.is_some() => CausalClaimSourceKind::CandidateCue,
        "candidate" => CausalClaimSourceKind::GraphSupport,
        "local_pair" => CausalClaimSourceKind::LocalTemporalPair,
        _ => CausalClaimSourceKind::GraphSupport,
    }
}

fn polarity_for(case: &CausalReviewCase) -> CausalClaimPolarity {
    if !case.temporal_legal {
        CausalClaimPolarity::Contradict
    } else if !matches!(case.modality_semantics, CausalModalitySemantics::Asserted) {
        CausalClaimPolarity::Underspecify
    } else if case.seed_source == "local_pair"
        && case.cue.is_none()
        && case.shared_participant_count == 0
    {
        CausalClaimPolarity::Underspecify
    } else {
        CausalClaimPolarity::Support
    }
}

fn evidence_class_for(case: &CausalReviewCase) -> CausalEvidenceClass {
    if case.quoted_evidence {
        CausalEvidenceClass::ReportedSupport
    } else if case.attributed_evidence {
        CausalEvidenceClass::AttributedSupport
    } else {
        CausalEvidenceClass::WorldSupport
    }
}

fn proposition_window(
    proposition: &Proposition,
    mention_ranges_by_index: &FxHashMap<usize, SourceRange>,
) -> SourceRange {
    let mut window = proposition
        .clause_range
        .unwrap_or(proposition.predicate.trigger_range);
    window = extend_range(window, proposition.predicate.trigger_range);
    for argument in &proposition.arguments {
        if let Some(range) = argument.range.or_else(|| {
            argument
                .mention_index
                .and_then(|index| mention_ranges_by_index.get(&index).copied())
        }) {
            window = extend_range(window, range);
        }
    }
    if let Some(quote) = proposition.quote.as_ref() {
        window = extend_range(window, quote.quote_range);
    }
    if let Some(attribution) = proposition
        .attribution
        .as_ref()
        .and_then(|frame| frame.quote_range)
    {
        window = extend_range(window, attribution);
    }
    if let Some(condition) = proposition.conditional.as_ref() {
        if let Some(range) = condition.condition_range {
            window = extend_range(window, range);
        }
        if let Some(range) = condition.consequent_range {
            window = extend_range(window, range);
        }
    }
    window
}

fn extend_range(left: SourceRange, right: SourceRange) -> SourceRange {
    SourceRange::new(left.start.min(right.start), left.end.max(right.end))
}

fn ranges_overlap(left: SourceRange, right: SourceRange) -> bool {
    left.start < right.end && right.start < left.end
}

fn source_semantics_for(proposition: &Proposition) -> CausalSourceSemantics {
    if proposition.quote.is_some() {
        CausalSourceSemantics::ReportedSpeech
    } else if proposition.attribution.is_some() {
        CausalSourceSemantics::AttributedClaim
    } else {
        CausalSourceSemantics::WorldAssertion
    }
}

fn modality_semantics_for(proposition: &Proposition, negative: bool) -> CausalModalitySemantics {
    if negative {
        return CausalModalitySemantics::Negated;
    }
    if proposition.conditional.is_some()
        || proposition
            .scope_ops
            .iter()
            .any(|scope| scope.kind.eq_ignore_ascii_case("conditional"))
    {
        return CausalModalitySemantics::Conditional;
    }
    let modality_labels = proposition
        .scope_ops
        .iter()
        .filter_map(|scope| scope.modality.as_deref())
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();
    if modality_labels.iter().any(|value| {
        matches!(
            value.as_str(),
            "planned" | "plan" | "future" | "intended" | "prospective" | "scheduled"
        )
    }) {
        CausalModalitySemantics::Planned
    } else if modality_labels.iter().any(|value| {
        matches!(
            value.as_str(),
            "hypothetical" | "possible" | "counterfactual" | "imagined" | "speculative"
        )
    }) {
        CausalModalitySemantics::Hypothetical
    } else {
        CausalModalitySemantics::Asserted
    }
}

fn merge_source_semantics(
    left: CausalSourceSemantics,
    right: CausalSourceSemantics,
    explicit_attribution: bool,
) -> CausalSourceSemantics {
    if matches!(left, CausalSourceSemantics::ReportedSpeech)
        || matches!(right, CausalSourceSemantics::ReportedSpeech)
    {
        CausalSourceSemantics::ReportedSpeech
    } else if explicit_attribution
        || matches!(left, CausalSourceSemantics::AttributedClaim)
        || matches!(right, CausalSourceSemantics::AttributedClaim)
    {
        CausalSourceSemantics::AttributedClaim
    } else {
        CausalSourceSemantics::WorldAssertion
    }
}

fn merge_modality_semantics(
    left: CausalModalitySemantics,
    right: CausalModalitySemantics,
) -> CausalModalitySemantics {
    for candidate in [
        CausalModalitySemantics::Negated,
        CausalModalitySemantics::Conditional,
        CausalModalitySemantics::Hypothetical,
        CausalModalitySemantics::Planned,
    ] {
        if left == candidate || right == candidate {
            return candidate;
        }
    }
    CausalModalitySemantics::Asserted
}

fn normalize_predicate_label(predicate: &str, fallback: &str) -> String {
    let raw = if predicate.trim().is_empty() {
        fallback
    } else {
        predicate
    };
    raw.trim().to_ascii_lowercase()
}

fn build_event_fingerprint(
    normalized_predicate: &str,
    participants: &[EntityId],
    source_semantics: CausalSourceSemantics,
    modality_semantics: CausalModalitySemantics,
) -> String {
    let participant_key = participants
        .iter()
        .map(|entity_id| entity_id.0.as_str())
        .collect::<Vec<_>>()
        .join("|");
    format!(
        "{}::{}::{}::{}",
        normalized_predicate,
        participant_key,
        source_semantics.as_str(),
        modality_semantics.as_str()
    )
}
