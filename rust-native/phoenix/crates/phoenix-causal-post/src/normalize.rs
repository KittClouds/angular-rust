use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    CausalClaimAtom, CausalClaimId, CausalClaimPolarity, CausalClaimSourceKind,
    CausalEdgeId, CausalRelationKind, DocumentArchive, DocumentCausalSubstrate, ErScopePatchSidecar,
};
use phoenix_types::{
    BiTemporalWindow, CausalCandidate, CausalKind, EntityId, Polarity, Proposition,
    ProvenanceRef, SemanticNodeRef, TruthStatus,
};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalEventProfile {
    pub node: SemanticNodeRef,
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
    pub evidence_refs: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalReviewCase {
    pub case_id: String,
    pub document_id: String,
    pub revision: u64,
    pub source: SemanticNodeRef,
    pub target: SemanticNodeRef,
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
    pub quoted_or_attributed: bool,
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
    pub diagnostics: BTreeMap<String, usize>,
}

pub fn normalize_causal_inputs(
    archives: &[DocumentArchive],
    er_sidecar: Option<&ErScopePatchSidecar>,
) -> CausalNormalizedInputs {
    let mut event_profiles = Vec::new();
    let mut raw_cases = Vec::new();
    let mut diagnostics = BTreeMap::<String, usize>::new();

    for archive in archives {
        let Some(substrate) = archive.causal_substrate.as_ref() else {
            *diagnostics
                .entry("missing_causal_substrate".to_owned())
                .or_default() += 1;
            continue;
        };

        let profile_map = build_event_profile_map(archive, substrate, er_sidecar);
        if profile_map.is_empty() {
            *diagnostics
                .entry("empty_causal_profiles".to_owned())
                .or_default() += 1;
            continue;
        }
        event_profiles.extend(profile_map.values().cloned());

        let graph_stats = build_degree_map(substrate);
        let mut seen_case_keys = FxHashSet::default();
        let mut explicit_case_count = 0usize;

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
                    explicit_case_count += 1;
                    raw_cases.push(case);
                }
            } else {
                *diagnostics
                    .entry("link_profile_gap".to_owned())
                    .or_default() += 1;
            }
        }

        for candidate in &substrate.causal_candidates {
            if let Some(case) = build_candidate_case(archive, &profile_map, &graph_stats, candidate) {
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

        if explicit_case_count == 0 {
            let local_cases = build_local_fallback_cases(archive, &profile_map, &graph_stats);
            for case in local_cases {
                let case_key = review_case_key(&case);
                if seen_case_keys.insert(case_key) {
                    raw_cases.push(case);
                }
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

    CausalNormalizedInputs {
        event_profiles,
        review_cases: raw_cases,
        claim_atoms,
        diagnostics,
    }
}

fn build_event_profile_map(
    archive: &DocumentArchive,
    substrate: &DocumentCausalSubstrate,
    er_sidecar: Option<&ErScopePatchSidecar>,
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
                .map(|binding| binding.anchor.as_ref().map(|anchor| anchor.interval.clone()).unwrap_or_else(|| binding.recorded_window.clone()))
                .unwrap_or_else(|| BiTemporalWindow {
                    valid_from: Some(archive.manifest.created_at),
                    valid_to: None,
                    recorded_from: Some(archive.manifest.created_at),
                    recorded_to: None,
                });
            (proposition.proposition_id.to_string(), temporal)
        })
        .collect::<FxHashMap<_, _>>();
    let mut profiles = FxHashMap::default();

    for event in &substrate.semantic_events {
        if let Some(profile) = build_profile_from_record(
            archive,
            event.event_id
                .clone()
                .map(SemanticNodeRef::Event),
            event.label.to_string(),
            event.proposition_id.to_string(),
            &proposition_by_id,
            &temporal_by_proposition,
            er_sidecar,
        ) {
            profiles.insert(node_key(&profile.node), profile);
        }
    }
    for state in &substrate.semantic_states {
        if let Some(profile) = build_profile_from_record(
            archive,
            state.state_id
                .clone()
                .map(SemanticNodeRef::State),
            state.label.to_string(),
            state.proposition_id.to_string(),
            &proposition_by_id,
            &temporal_by_proposition,
            er_sidecar,
        ) {
            profiles.insert(node_key(&profile.node), profile);
        }
    }
    for claim in &substrate.semantic_claims {
        if let Some(profile) = build_profile_from_record(
            archive,
            claim.claim_id
                .clone()
                .map(SemanticNodeRef::Claim),
            claim.label.to_string(),
            claim.proposition_id.to_string(),
            &proposition_by_id,
            &temporal_by_proposition,
            er_sidecar,
        ) {
            profiles.entry(node_key(&profile.node)).or_insert(profile);
        }
    }

    profiles
}

fn build_profile_from_record(
    archive: &DocumentArchive,
    node: Option<SemanticNodeRef>,
    label: String,
    proposition_id: String,
    proposition_by_id: &FxHashMap<String, &Proposition>,
    temporal_by_proposition: &FxHashMap<String, BiTemporalWindow>,
    er_sidecar: Option<&ErScopePatchSidecar>,
) -> Option<CausalEventProfile> {
    let node = node?;
    let proposition = proposition_by_id.get(&proposition_id)?;
    let mut participants = proposition
        .arguments
        .iter()
        .filter_map(|argument| argument.entity_id.clone())
        .collect::<Vec<_>>();
    if let Some(sidecar) = er_sidecar {
        for link in &sidecar.entity_links {
            if link.document_id == archive.manifest.document_id {
                participants.push(link.entity_id.clone());
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

    Some(CausalEventProfile {
        node,
        document_id: archive.manifest.document_id.clone(),
        proposition_id,
        label,
        sentence_index: proposition.sentence_index,
        temporal: temporal_by_proposition
            .get(&proposition.proposition_id.to_string())
            .cloned()
            .unwrap_or(BiTemporalWindow {
                valid_from: Some(archive.manifest.created_at),
                valid_to: None,
                recorded_from: Some(archive.manifest.created_at),
                recorded_to: None,
            }),
        participant_entity_ids: participants,
        attributed_to: proposition
            .attribution
            .as_ref()
            .and_then(|frame| frame.source_entity_id.clone()),
        quoted: proposition_is_quoted(proposition),
        negative,
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
        .saturating_sub(source_profile.sentence_index.min(target_profile.sentence_index));
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
    let quoted_or_attributed = review_case_needs_quote_caution(
        seed_source,
        source_profile,
        target_profile,
        attributed_to.as_ref(),
    );
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
        target: target.clone(),
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
        quoted_or_attributed,
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
                provenance_refs(left.evidence_refs.iter().chain(right.evidence_refs.iter()).cloned()),
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
    left.iter().filter(|entity_id| right_set.contains(entity_id)).count()
}

fn proposition_is_quoted(proposition: &Proposition) -> bool {
    proposition.quote.is_some()
        || proposition
            .attribution
            .as_ref()
            .and_then(|frame| frame.quote_range)
            .is_some()
}

fn review_case_needs_quote_caution(
    seed_source: &str,
    source_profile: &CausalEventProfile,
    target_profile: &CausalEventProfile,
    attributed_to: Option<&EntityId>,
) -> bool {
    if attributed_to.is_some() {
        return true;
    }
    match seed_source {
        "link" | "candidate" => source_profile.quoted || target_profile.quoted,
        _ => {
            source_profile.quoted
                || target_profile.quoted
                || source_profile.attributed_to.is_some()
                || target_profile.attributed_to.is_some()
        }
    }
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
    let mut refs = values.into_iter().filter(|value| !value.is_empty()).collect::<Vec<_>>();
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
                effect_event: case.target.clone(),
                kind: case.kind,
                relation_kind: case.relation_kind,
                source_kind: source_kind_for(case),
                polarity: polarity_for(case),
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
    if case.quoted_or_attributed
        && matches!(case.seed_source.as_str(), "local_pair")
    {
        CausalClaimPolarity::Underspecify
    } else if case.attributed_to.is_some() {
        CausalClaimPolarity::Underspecify
    } else if !case.temporal_legal {
        CausalClaimPolarity::Contradict
    } else {
        CausalClaimPolarity::Support
    }
}
