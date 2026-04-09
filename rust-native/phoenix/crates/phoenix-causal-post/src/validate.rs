use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    CausalClaimAtom, CausalClaimPolarity, CausalClaimStatus, CausalDecisionId,
    CausalDecisionOutcome, CausalDecisionRecord, CausalEdgeAddition, CausalEdgeAliasRecord,
    CausalEdgeId, CausalEvidenceClass, CausalInvalidationRecord, CausalMetricsSnapshot,
    CausalReviewQueueItem,
};
use phoenix_types::{CausalKind, Polarity, SemanticNodeRef, TruthStatus};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use crate::normalize::{
    node_key, stable_edge_id, CausalModalitySemantics, CausalReviewCase, CausalSourceSemantics,
};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CausalDecisionKind {
    Accept,
    Support,
    Invalidate,
    #[default]
    Defer,
    Reject,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalDecision {
    pub decision_id: CausalDecisionId,
    pub edge_id: CausalEdgeId,
    pub case_id: String,
    pub document_id: String,
    pub kind: CausalDecisionKind,
    pub edge_type: Option<CausalKind>,
    pub score_millis: i32,
    pub rationale: String,
    #[serde(default)]
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CausalDecisionDrafts {
    pub decisions: Vec<CausalDecision>,
    pub edge_records: Vec<CausalEdgeAddition>,
    pub decision_records: Vec<CausalDecisionRecord>,
    pub invalidations: Vec<CausalInvalidationRecord>,
    pub edge_aliases: Vec<CausalEdgeAliasRecord>,
    pub review_queue: Vec<CausalReviewQueueItem>,
    pub outcome_counts: BTreeMap<String, usize>,
    pub diagnostics: BTreeMap<String, usize>,
    pub metrics_snapshot: CausalMetricsSnapshot,
}

#[derive(Clone, Debug, Default)]
struct PairEvidenceSummary {
    support_atom_count: usize,
    contradict_atom_count: usize,
    underspecified_atom_count: usize,
    world_support_count: usize,
    reported_support_count: usize,
    attributed_support_count: usize,
    unique_evidence_count: usize,
}

impl PairEvidenceSummary {
    fn has_world_support(&self) -> bool {
        self.world_support_count > 0
    }

    fn has_reported_support(&self) -> bool {
        self.reported_support_count > 0
    }

    fn has_attributed_support(&self) -> bool {
        self.attributed_support_count > 0
    }

    fn only_non_world_support(&self) -> bool {
        !self.has_world_support() && (self.has_reported_support() || self.has_attributed_support())
    }
}

#[derive(Clone, Debug)]
struct CausalPairNode<'a> {
    edge_id: CausalEdgeId,
    case: &'a CausalReviewCase,
    claim_atoms: Vec<&'a CausalClaimAtom>,
    evidence: PairEvidenceSummary,
    base_strength_millis: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CausalPairFeatures {
    direct_millis: i32,
    temporal_millis: i32,
    topology_millis: i32,
    discourse_millis: i32,
    review_millis: i32,
    competing_cause_count: usize,
    reverse_direction_present: bool,
    transitive_support_count: usize,
    brittle_support_path: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CausalPairScore {
    total_millis: i32,
    band: &'static str,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct PairAssessment {
    kind: CausalDecisionKind,
    rationale: String,
    score: CausalPairScore,
    features: CausalPairFeatures,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct HypothesisGraphStats {
    outgoing_by_source: FxHashMap<String, usize>,
    incoming_by_target: FxHashMap<String, usize>,
    reverse_strength_by_edge: FxHashMap<String, u32>,
    competing_count_by_edge: FxHashMap<String, usize>,
}

#[derive(Clone, Copy)]
struct KindRule {
    accept_threshold: i32,
    review_threshold: i32,
}

pub fn draft_causal_decisions(
    cases: &[CausalReviewCase],
    claim_atoms: &[CausalClaimAtom],
    created_at: i64,
) -> CausalDecisionDrafts {
    let mut claims_by_edge = FxHashMap::<String, Vec<&CausalClaimAtom>>::default();
    let mut cases_by_edge = FxHashMap::<String, Vec<&CausalReviewCase>>::default();
    for atom in claim_atoms {
        claims_by_edge
            .entry(atom.edge_id.0.clone())
            .or_default()
            .push(atom);
    }
    for case in cases {
        cases_by_edge
            .entry(stable_edge_id(case).0)
            .or_default()
            .push(case);
    }

    let mut edge_keys = cases_by_edge.keys().cloned().collect::<Vec<_>>();
    edge_keys.sort();

    let mut nodes = edge_keys
        .iter()
        .map(|edge_key| {
            let edge_cases = &cases_by_edge[edge_key];
            let representative_case = select_representative_case(edge_cases);
            let claim_refs = claims_by_edge.get(edge_key).cloned().unwrap_or_default();
            let evidence = summarize_evidence(representative_case, &claim_refs);
            CausalPairNode {
                edge_id: CausalEdgeId(edge_key.clone()),
                case: representative_case,
                claim_atoms: claim_refs,
                base_strength_millis: base_strength_millis(representative_case, &evidence),
                evidence,
            }
        })
        .collect::<Vec<_>>();
    nodes.sort_by(|left, right| left.edge_id.0.cmp(&right.edge_id.0));

    let hypothesis = build_hypothesis_graph(&nodes);
    let rules = nodes
        .iter()
        .map(|node| kind_rule(node.case.kind))
        .collect::<Vec<_>>();

    let mut assessments = Vec::with_capacity(nodes.len());
    let mut diagnostics = BTreeMap::<String, usize>::new();
    let mut pass_a_accept_count = 0usize;
    let mut pass_a_defer_count = 0usize;
    let mut pass_a_reject_count = 0usize;

    for (index, node) in nodes.iter().enumerate() {
        count_evidence_classes(&mut diagnostics, &node.evidence);
        let assessment = pass_a_assessment(node, &hypothesis, rules[index]);
        count_pass_outcome(&mut diagnostics, "pass_a", &assessment.kind);
        count_feature_band(&mut diagnostics, assessment.score.band);
        if assessment.features.competing_cause_count > 0 {
            *diagnostics
                .entry("competing_cause_edges".to_owned())
                .or_default() += 1;
        }
        if assessment.features.reverse_direction_present {
            *diagnostics
                .entry("reverse_direction_conflicts".to_owned())
                .or_default() += 1;
        }
        match assessment.kind {
            CausalDecisionKind::Accept | CausalDecisionKind::Support => pass_a_accept_count += 1,
            CausalDecisionKind::Defer => pass_a_defer_count += 1,
            CausalDecisionKind::Reject | CausalDecisionKind::Invalidate => pass_a_reject_count += 1,
        }
        assessments.push(assessment);
    }

    let provisional_accepts = assessments
        .iter()
        .enumerate()
        .filter(|(_, assessment)| {
            matches!(
                assessment.kind,
                CausalDecisionKind::Accept | CausalDecisionKind::Support
            )
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let review_pressure = build_review_pressure(&nodes, &assessments, &provisional_accepts);

    let mut final_assessments = Vec::with_capacity(nodes.len());
    let mut pass_b_demoted_count = 0usize;
    for (index, node) in nodes.iter().enumerate() {
        let final_assessment =
            pass_b_assessment(node, &assessments[index], &review_pressure, rules[index]);
        count_pass_outcome(&mut diagnostics, "pass_b", &final_assessment.kind);
        if is_demoted(&assessments[index].kind, &final_assessment.kind) {
            pass_b_demoted_count += 1;
        }
        final_assessments.push(final_assessment);
    }
    diagnostics.insert("pass_b_demoted_edges".to_owned(), pass_b_demoted_count);

    let mut drafts = CausalDecisionDrafts::default();
    let mut temporal_illegal_count = 0usize;
    let mut cue_only_edge_count = 0usize;

    for (index, node) in nodes.iter().enumerate() {
        let assessment = &final_assessments[index];
        let score = assessment.score.total_millis;
        if !node.case.temporal_legal {
            temporal_illegal_count += 1;
        }
        if node.evidence.support_atom_count > 0
            && node.evidence.contradict_atom_count == 0
            && node.case.cue.is_some()
        {
            cue_only_edge_count += 1;
        }

        let decision_id = CausalDecisionId(format!("decision:{}:{}", node.edge_id.0, created_at));
        let evidence = merged_evidence_refs(node.case, &node.claim_atoms);
        *drafts
            .outcome_counts
            .entry(format!("{:?}", assessment.kind).to_lowercase())
            .or_default() += 1;

        let latest_status = match assessment.kind {
            CausalDecisionKind::Accept => CausalClaimStatus::Active,
            CausalDecisionKind::Support => CausalClaimStatus::Supported,
            CausalDecisionKind::Invalidate => CausalClaimStatus::Invalidated,
            CausalDecisionKind::Defer => CausalClaimStatus::Deferred,
            CausalDecisionKind::Reject => {
                if node.evidence.contradict_atom_count > 0 {
                    CausalClaimStatus::Contradicted
                } else {
                    CausalClaimStatus::Rejected
                }
            }
        };

        drafts.decisions.push(CausalDecision {
            decision_id: decision_id.clone(),
            edge_id: node.edge_id.clone(),
            case_id: node.case.case_id.clone(),
            document_id: node.case.document_id.clone(),
            kind: assessment.kind.clone(),
            edge_type: Some(node.case.kind),
            score_millis: score,
            rationale: assessment.rationale.clone(),
            evidence: evidence.clone(),
        });
        drafts.decision_records.push(CausalDecisionRecord {
            decision_id: decision_id.clone(),
            edge_id: node.edge_id.clone(),
            case_id: node.case.case_id.clone(),
            document_id: node.case.document_id.clone(),
            outcome: match assessment.kind {
                CausalDecisionKind::Accept => CausalDecisionOutcome::Accept,
                CausalDecisionKind::Support => CausalDecisionOutcome::Support,
                CausalDecisionKind::Invalidate => CausalDecisionOutcome::Invalidate,
                CausalDecisionKind::Defer => CausalDecisionOutcome::Defer,
                CausalDecisionKind::Reject => CausalDecisionOutcome::Reject,
            },
            source: Some(node.case.source.clone()),
            target: Some(node.case.target.clone()),
            kind: Some(node.case.kind),
            relation_kind: Some(node.case.relation_kind),
            score_millis: score,
            rationale: assessment.rationale.clone(),
            supersedes: None,
            evidence: evidence.clone(),
            reviewed_at: created_at,
        });

        drafts.edge_records.push(CausalEdgeAddition {
            edge_id: node.edge_id.clone(),
            case_id: node.case.case_id.clone(),
            document_id: node.case.document_id.clone(),
            source: node.case.source.clone(),
            canonical_cause_event_id: node.case.canonical_cause_event_id.clone(),
            target: node.case.target.clone(),
            canonical_effect_event_id: node.case.canonical_effect_event_id.clone(),
            kind: node.case.kind,
            relation_kind: node.case.relation_kind,
            status: latest_status,
            first_seen_revision: node.case.revision,
            latest_decision_id: Some(decision_id.clone()),
            confidence_millis: score.clamp(0, 1000) as u32,
            cue: node.case.cue.clone(),
            attributed_to: node.case.attributed_to.clone(),
            polarity: node.case.polarity,
            claim_atom_ids: node
                .claim_atoms
                .iter()
                .map(|atom| atom.claim_id.clone())
                .collect(),
            evidence_refs: evidence.clone(),
            effective_interval: node.case.temporal.clone(),
            observation_interval: node.case.temporal.clone(),
            temporal_certainty_millis: if node.case.temporal_legal { 900 } else { 200 },
            created_at,
        });

        drafts.edge_aliases.push(CausalEdgeAliasRecord {
            alias_key: format!("case:{}", node.case.case_id),
            edge_id: node.edge_id.clone(),
            document_id: node.case.document_id.clone(),
            created_at,
        });
        drafts.edge_aliases.push(CausalEdgeAliasRecord {
            alias_key: format!(
                "legacy:{}:{}:{:?}",
                node_key(&node.case.source),
                node_key(&node.case.target),
                node.case.kind
            ),
            edge_id: node.edge_id.clone(),
            document_id: node.case.document_id.clone(),
            created_at,
        });

        if matches!(assessment.kind, CausalDecisionKind::Invalidate) {
            drafts.invalidations.push(CausalInvalidationRecord {
                invalidation_id: format!("invalidate:{}:{}", node.edge_id.0, created_at),
                edge_id: node.edge_id.clone(),
                decision_id: decision_id.clone(),
                document_id: node.case.document_id.clone(),
                rationale: assessment.rationale.clone(),
                evidence_refs: evidence.clone(),
                created_at,
            });
        }
        if matches!(
            assessment.kind,
            CausalDecisionKind::Defer | CausalDecisionKind::Reject
        ) {
            drafts.review_queue.push(CausalReviewQueueItem {
                queue_id: format!("queue:{}:{}", node.edge_id.0, created_at),
                edge_id: node.edge_id.clone(),
                latest_decision_id: Some(decision_id),
                document_id: node.case.document_id.clone(),
                priority_millis: review_priority(
                    score,
                    node.evidence.contradict_atom_count,
                    node.evidence.underspecified_atom_count,
                ),
                rationale: assessment.rationale.clone(),
                unresolved: true,
                created_at,
            });
        }
    }

    let edge_record_count = drafts.edge_records.len();
    let accepted_count = drafts
        .edge_records
        .iter()
        .filter(|edge| {
            matches!(
                edge.status,
                CausalClaimStatus::Active | CausalClaimStatus::Supported
            )
        })
        .count();
    let supported_count = drafts
        .edge_records
        .iter()
        .filter(|edge| matches!(edge.status, CausalClaimStatus::Supported))
        .count();
    let deferred_count = drafts
        .outcome_counts
        .get("defer")
        .copied()
        .unwrap_or_default();
    let rejected_count = drafts
        .outcome_counts
        .get("reject")
        .copied()
        .unwrap_or_default();
    let invalidated_count = drafts.invalidations.len();
    let contradicted_count = drafts
        .edge_records
        .iter()
        .filter(|edge| matches!(edge.status, CausalClaimStatus::Contradicted))
        .count();
    drafts.metrics_snapshot = CausalMetricsSnapshot {
        edge_record_count,
        accepted_count,
        supported_count,
        deferred_count,
        rejected_count,
        invalidated_count,
        contradicted_count,
        contradiction_rate_per_1k_events_millis: rate_millis(
            contradicted_count,
            cases.len().max(1),
        ),
        edge_survival_rate_millis: 1000,
        chain_collapse_rate_millis: 0,
        avg_claim_atoms_per_edge_millis: avg_millis(
            claim_atoms.len(),
            drafts.edge_records.len().max(1),
        ),
        cue_only_edge_rate_millis: rate_millis(
            cue_only_edge_count,
            drafts.edge_records.len().max(1),
        ),
        card_open_dispute_rate_millis: rate_millis(
            drafts.review_queue.len(),
            drafts.edge_records.len().max(1),
        ),
        temporal_illegality_rejection_rate_millis: rate_millis(
            temporal_illegal_count,
            cases.len().max(1),
        ),
        pass_a_accept_count,
        pass_a_defer_count,
        pass_a_reject_count,
        pass_b_demoted_count,
        world_support_count: diagnostics
            .get("evidence_class:world_support")
            .copied()
            .unwrap_or_default(),
        reported_support_count: diagnostics
            .get("evidence_class:reported_support")
            .copied()
            .unwrap_or_default(),
        attributed_support_count: diagnostics
            .get("evidence_class:attributed_support")
            .copied()
            .unwrap_or_default(),
        shadow_local_pair_candidate_count: 0,
        shadow_local_pair_committed_count: 0,
        shadow_local_pair_deferred_count: 0,
        shadow_local_pair_overlap_count: 0,
    };
    drafts.diagnostics = diagnostics;
    drafts
}

fn select_representative_case<'a>(cases: &[&'a CausalReviewCase]) -> &'a CausalReviewCase {
    cases
        .iter()
        .copied()
        .max_by_key(|case| {
            (
                evidence_rank(case),
                seed_rank(case.seed_source.as_str()),
                usize::from(case.cue.is_some()),
                usize::from(case.temporal_legal),
                case.base_confidence_millis,
            )
        })
        .expect("representative case requires at least one case")
}

fn evidence_rank(case: &CausalReviewCase) -> usize {
    if case.attributed_evidence {
        0
    } else if case.quoted_evidence {
        1
    } else {
        2
    }
}

fn seed_rank(seed_source: &str) -> usize {
    match seed_source {
        "link" => 2,
        "candidate" => 1,
        "local_pair" => 0,
        _ => 0,
    }
}

fn summarize_evidence(
    case: &CausalReviewCase,
    claim_atoms: &[&CausalClaimAtom],
) -> PairEvidenceSummary {
    let mut summary = PairEvidenceSummary::default();
    let mut evidence_refs = FxHashSet::<String>::default();
    for reference in &case.evidence_refs {
        evidence_refs.insert(reference.clone());
    }
    for atom in claim_atoms {
        match atom.polarity {
            CausalClaimPolarity::Support => summary.support_atom_count += 1,
            CausalClaimPolarity::Contradict => summary.contradict_atom_count += 1,
            CausalClaimPolarity::Underspecify => summary.underspecified_atom_count += 1,
        }
        match atom.evidence_class {
            CausalEvidenceClass::WorldSupport => summary.world_support_count += 1,
            CausalEvidenceClass::ReportedSupport => summary.reported_support_count += 1,
            CausalEvidenceClass::AttributedSupport => summary.attributed_support_count += 1,
        }
        for reference in &atom.evidence_refs {
            evidence_refs.insert(reference.clone());
        }
    }
    if claim_atoms.is_empty() {
        match fallback_evidence_class(case) {
            CausalEvidenceClass::WorldSupport => summary.world_support_count += 1,
            CausalEvidenceClass::ReportedSupport => summary.reported_support_count += 1,
            CausalEvidenceClass::AttributedSupport => summary.attributed_support_count += 1,
        }
        summary.support_atom_count = 1;
    }
    summary.unique_evidence_count = evidence_refs.len();
    summary
}

fn fallback_evidence_class(case: &CausalReviewCase) -> CausalEvidenceClass {
    if case.quoted_evidence {
        CausalEvidenceClass::ReportedSupport
    } else if case.attributed_evidence {
        CausalEvidenceClass::AttributedSupport
    } else {
        CausalEvidenceClass::WorldSupport
    }
}

fn base_strength_millis(case: &CausalReviewCase, evidence: &PairEvidenceSummary) -> u32 {
    let mut strength = case.base_confidence_millis as i32;
    strength += match case.seed_source.as_str() {
        "link" => 110,
        "candidate" => 70,
        "local_pair" => 20,
        _ => 0,
    };
    if case.cue.is_some() {
        strength += 90;
    }
    strength += (evidence.world_support_count.min(2) as i32) * 70;
    strength += (evidence.support_atom_count.min(2) as i32) * 40;
    strength.max(0).min(1000) as u32
}

fn build_hypothesis_graph(nodes: &[CausalPairNode<'_>]) -> HypothesisGraphStats {
    let mut outgoing_by_source = FxHashMap::<String, usize>::default();
    let mut incoming_by_target = FxHashMap::<String, usize>::default();
    let mut strength_by_signature = FxHashMap::<String, u32>::default();
    let mut reverse_strength_by_edge = FxHashMap::<String, u32>::default();
    let mut competing_count_by_edge = FxHashMap::<String, usize>::default();
    let mut competing_counts = FxHashMap::<String, usize>::default();

    for node in nodes {
        *outgoing_by_source
            .entry(node_key(&node.case.source))
            .or_default() += 1;
        *incoming_by_target
            .entry(node_key(&node.case.target))
            .or_default() += 1;
        let signature = edge_signature(
            &node.case.source,
            &node.case.target,
            node.case.relation_kind,
        );
        strength_by_signature
            .entry(signature)
            .and_modify(|value| *value = (*value).max(node.base_strength_millis))
            .or_insert(node.base_strength_millis);
        *competing_counts
            .entry(target_signature(&node.case.target, node.case.relation_kind))
            .or_default() += 1;
    }

    for node in nodes {
        let reverse = edge_signature(
            &node.case.target,
            &node.case.source,
            node.case.relation_kind,
        );
        if let Some(reverse_strength) = strength_by_signature.get(&reverse).copied() {
            reverse_strength_by_edge.insert(node.edge_id.0.clone(), reverse_strength);
        }
        let competing_count = competing_counts
            .get(&target_signature(
                &node.case.target,
                node.case.relation_kind,
            ))
            .copied()
            .unwrap_or_default()
            .saturating_sub(1);
        competing_count_by_edge.insert(node.edge_id.0.clone(), competing_count);
    }

    HypothesisGraphStats {
        outgoing_by_source,
        incoming_by_target,
        reverse_strength_by_edge,
        competing_count_by_edge,
    }
}

fn pass_a_assessment(
    node: &CausalPairNode<'_>,
    hypothesis: &HypothesisGraphStats,
    rule: KindRule,
) -> PairAssessment {
    if let Some((kind, rationale)) = hard_gate(node, hypothesis) {
        return PairAssessment {
            kind,
            rationale,
            score: CausalPairScore {
                total_millis: 0,
                band: "blocked",
            },
            features: CausalPairFeatures {
                reverse_direction_present: hypothesis
                    .reverse_strength_by_edge
                    .contains_key(&node.edge_id.0),
                competing_cause_count: hypothesis
                    .competing_count_by_edge
                    .get(&node.edge_id.0)
                    .copied()
                    .unwrap_or_default(),
                ..CausalPairFeatures::default()
            },
        };
    }

    let features = pass_a_features(node, hypothesis);
    let total = features.direct_millis
        + features.temporal_millis
        + features.topology_millis
        + features.discourse_millis;
    let score = CausalPairScore {
        total_millis: total,
        band: feature_band(total, rule),
    };

    let kind = if node.evidence.contradict_atom_count > node.evidence.support_atom_count
        && score.total_millis < rule.accept_threshold
    {
        CausalDecisionKind::Reject
    } else if score.total_millis >= rule.accept_threshold {
        if node.case.seed_source == "link" {
            CausalDecisionKind::Support
        } else {
            CausalDecisionKind::Accept
        }
    } else if score.total_millis >= rule.review_threshold {
        CausalDecisionKind::Defer
    } else {
        CausalDecisionKind::Reject
    };
    let rationale = match kind {
        CausalDecisionKind::Accept | CausalDecisionKind::Support => {
            "pass_a_plausible_world_causality".to_owned()
        }
        CausalDecisionKind::Defer => "pass_a_needs_review".to_owned(),
        CausalDecisionKind::Reject => "pass_a_insufficient_support".to_owned(),
        CausalDecisionKind::Invalidate => "pass_a_invalidated".to_owned(),
    };
    PairAssessment {
        kind,
        rationale,
        score,
        features,
    }
}

fn hard_gate(
    node: &CausalPairNode<'_>,
    hypothesis: &HypothesisGraphStats,
) -> Option<(CausalDecisionKind, String)> {
    if !node.case.temporal_legal {
        return Some((
            invalidate_or_reject(node.case.seed_source.as_str()),
            "temporal_illegality".to_owned(),
        ));
    }
    if matches!(node.case.polarity, Polarity::Negative)
        && !matches!(node.case.kind, CausalKind::Prevents | CausalKind::Hinders)
    {
        return Some((CausalDecisionKind::Reject, "polarity_impossible".to_owned()));
    }
    if let Some(reverse_strength) = hypothesis.reverse_strength_by_edge.get(&node.edge_id.0) {
        if *reverse_strength > node.base_strength_millis.saturating_add(140) {
            return Some((
                invalidate_or_reject(node.case.seed_source.as_str()),
                "reverse_direction_stronger".to_owned(),
            ));
        }
    }
    if node.evidence.only_non_world_support() {
        return Some((
            CausalDecisionKind::Defer,
            "reported_or_attributed_without_world_support".to_owned(),
        ));
    }
    if !matches!(
        node.case.modality_semantics,
        CausalModalitySemantics::Asserted
    ) && !node.evidence.has_world_support()
    {
        return Some((
            CausalDecisionKind::Defer,
            "non_asserted_causality_without_world_support".to_owned(),
        ));
    }
    if node.case.cue.is_none()
        && node.case.base_confidence_millis < 420
        && node.case.shared_participant_count == 0
        && node.case.graph_support_count == 0
        && node.evidence.world_support_count == 0
    {
        return Some((
            CausalDecisionKind::Defer,
            "structurally_underspecified".to_owned(),
        ));
    }
    None
}

fn pass_a_features(
    node: &CausalPairNode<'_>,
    hypothesis: &HypothesisGraphStats,
) -> CausalPairFeatures {
    let mut direct = (node.case.base_confidence_millis as i32 * 45) / 100;
    direct += match node.case.base_status {
        TruthStatus::Asserted => 100,
        TruthStatus::Candidate => 15,
        TruthStatus::Rejected => -180,
        TruthStatus::Expired => -90,
        TruthStatus::Unknown => -30,
    };
    direct += match node.case.seed_source.as_str() {
        "link" => 120,
        "candidate" => 80,
        "local_pair" => 30,
        _ => 20,
    };
    if node.case.cue.is_some() {
        direct += 90;
    }
    direct += (node.evidence.world_support_count.min(2) as i32) * 70;
    direct += (node.evidence.support_atom_count.min(3) as i32) * 35;
    direct -= (node.evidence.contradict_atom_count.min(2) as i32) * 90;
    direct -= (node.evidence.underspecified_atom_count.min(2) as i32) * 60;
    direct = direct.clamp(-180, 520);

    let mut temporal = 80;
    temporal += match node.case.sentence_distance {
        0 => 70,
        1 => 25,
        _ => -80,
    };
    if node.case.temporal_legal {
        temporal += 40;
    }
    temporal = temporal.clamp(-200, 180);

    let source_degree = hypothesis
        .outgoing_by_source
        .get(&node_key(&node.case.source))
        .copied()
        .unwrap_or_default();
    let target_degree = hypothesis
        .incoming_by_target
        .get(&node_key(&node.case.target))
        .copied()
        .unwrap_or_default();
    let competing_count = hypothesis
        .competing_count_by_edge
        .get(&node.edge_id.0)
        .copied()
        .unwrap_or_default();
    let reverse_present = hypothesis
        .reverse_strength_by_edge
        .contains_key(&node.edge_id.0);

    let mut topology = (node.case.shared_participant_count.min(2) as i32) * 60;
    topology += (source_degree.min(3) as i32) * 20;
    topology += (target_degree.min(3) as i32) * 20;
    topology += (node.case.centrality_millis.min(360) as i32) / 6;
    topology -= (competing_count.min(3) as i32) * 35;
    if reverse_present {
        topology -= 80;
    }
    topology = topology.clamp(-220, 180);

    let mut discourse = 0;
    if node.case.quoted_evidence {
        discourse -= 60;
    }
    if node.case.attributed_evidence {
        discourse -= 100;
    }
    discourse += match node.case.source_semantics {
        CausalSourceSemantics::WorldAssertion => 20,
        CausalSourceSemantics::ReportedSpeech => -40,
        CausalSourceSemantics::AttributedClaim => -70,
    };
    discourse += match node.case.modality_semantics {
        CausalModalitySemantics::Asserted => 20,
        CausalModalitySemantics::Conditional => -90,
        CausalModalitySemantics::Planned => -70,
        CausalModalitySemantics::Hypothetical => -110,
        CausalModalitySemantics::Negated => -120,
    };
    if node.evidence.has_world_support() {
        discourse += 30;
    }
    discourse = discourse.clamp(-180, 80);

    CausalPairFeatures {
        direct_millis: direct,
        temporal_millis: temporal,
        topology_millis: topology,
        discourse_millis: discourse,
        review_millis: 0,
        competing_cause_count: competing_count,
        reverse_direction_present: reverse_present,
        transitive_support_count: 0,
        brittle_support_path: false,
    }
}

fn build_review_pressure(
    nodes: &[CausalPairNode<'_>],
    pass_a: &[PairAssessment],
    provisional_accepts: &[usize],
) -> FxHashMap<String, CausalPairFeatures> {
    let mut outgoing = FxHashMap::<String, Vec<usize>>::default();
    let mut incoming = FxHashMap::<String, Vec<usize>>::default();
    let mut by_target = FxHashMap::<String, Vec<usize>>::default();
    let mut by_signature = FxHashMap::<String, usize>::default();

    for &index in provisional_accepts {
        let node = &nodes[index];
        outgoing
            .entry(node_key(&node.case.source))
            .or_default()
            .push(index);
        incoming
            .entry(node_key(&node.case.target))
            .or_default()
            .push(index);
        by_target
            .entry(target_signature(&node.case.target, node.case.relation_kind))
            .or_default()
            .push(index);
        by_signature.insert(
            edge_signature(
                &node.case.source,
                &node.case.target,
                node.case.relation_kind,
            ),
            index,
        );
    }

    let mut review = FxHashMap::<String, CausalPairFeatures>::default();
    for &index in provisional_accepts {
        let node = &nodes[index];
        let next = outgoing
            .get(&node_key(&node.case.target))
            .map(|values| values.len())
            .unwrap_or_default();
        let prev = incoming
            .get(&node_key(&node.case.source))
            .map(|values| values.len())
            .unwrap_or_default();
        let transitive_support_count = next.saturating_add(prev).min(2);
        let target_group = by_target
            .get(&target_signature(
                &node.case.target,
                node.case.relation_kind,
            ))
            .cloned()
            .unwrap_or_default();
        let top_index = target_group
            .iter()
            .copied()
            .max_by_key(|candidate_index| pass_a[*candidate_index].score.total_millis);
        let competing_cause_count = target_group.len().saturating_sub(1);
        let reverse_index = by_signature.get(&edge_signature(
            &node.case.target,
            &node.case.source,
            node.case.relation_kind,
        ));
        let reverse_direction_present = reverse_index.is_some();
        let brittle_support_path = node.evidence.unique_evidence_count <= 1
            && transitive_support_count == 0
            && node.case.cue.is_none();

        let mut review_millis = 0;
        review_millis += (transitive_support_count as i32) * 40;
        if competing_cause_count > 0 {
            if Some(index) == top_index {
                review_millis -= (competing_cause_count.min(2) as i32) * 20;
            } else {
                review_millis -= (competing_cause_count.min(2) as i32) * 120;
            }
        }
        if let Some(reverse_index) = reverse_index {
            if pass_a[*reverse_index].score.total_millis > pass_a[index].score.total_millis + 80 {
                review_millis -= 220;
            }
        }
        if brittle_support_path {
            review_millis -= 80;
        }
        review_millis = review_millis.clamp(-220, 120);

        review.insert(
            node.edge_id.0.clone(),
            CausalPairFeatures {
                direct_millis: 0,
                temporal_millis: 0,
                topology_millis: 0,
                discourse_millis: 0,
                review_millis,
                competing_cause_count,
                reverse_direction_present,
                transitive_support_count,
                brittle_support_path,
            },
        );
    }
    review
}

fn pass_b_assessment(
    node: &CausalPairNode<'_>,
    pass_a: &PairAssessment,
    review_pressure: &FxHashMap<String, CausalPairFeatures>,
    rule: KindRule,
) -> PairAssessment {
    if !matches!(
        pass_a.kind,
        CausalDecisionKind::Accept | CausalDecisionKind::Support
    ) {
        return pass_a.clone();
    }
    let Some(review_features) = review_pressure.get(&node.edge_id.0) else {
        return pass_a.clone();
    };

    let mut assessment = pass_a.clone();
    assessment.features.review_millis = review_features.review_millis;
    assessment.features.transitive_support_count = review_features.transitive_support_count;
    assessment.features.competing_cause_count = review_features.competing_cause_count;
    assessment.features.reverse_direction_present = review_features.reverse_direction_present;
    assessment.features.brittle_support_path = review_features.brittle_support_path;
    assessment.score.total_millis += review_features.review_millis;
    assessment.score.band = feature_band(assessment.score.total_millis, rule);

    if review_features.reverse_direction_present && review_features.review_millis <= -180 {
        assessment.kind = invalidate_or_reject(node.case.seed_source.as_str());
        assessment.rationale = "pass_b_reverse_direction_pressure".to_owned();
        return assessment;
    }
    if review_features.competing_cause_count > 0 && review_features.review_millis <= -120 {
        assessment.kind = CausalDecisionKind::Defer;
        assessment.rationale = "competing_cause_pressure".to_owned();
        return assessment;
    }
    if review_features.brittle_support_path
        && assessment.score.total_millis < rule.accept_threshold + 60
        && node.evidence.world_support_count < 2
    {
        assessment.kind = CausalDecisionKind::Defer;
        assessment.rationale = "brittle_support_path".to_owned();
        return assessment;
    }
    if assessment.score.total_millis < rule.accept_threshold {
        assessment.kind = CausalDecisionKind::Defer;
        assessment.rationale = "pass_b_review_pressure".to_owned();
    } else {
        assessment.rationale = "pass_b_confirmed_world_causality".to_owned();
    }
    assessment
}

fn kind_rule(kind: CausalKind) -> KindRule {
    match kind {
        CausalKind::Causes
        | CausalKind::ResultsIn
        | CausalKind::TriggerFor
        | CausalKind::Prevents => KindRule {
            accept_threshold: 660,
            review_threshold: 430,
        },
        CausalKind::Enables | CausalKind::ConditionFor | CausalKind::Hinders => KindRule {
            accept_threshold: 620,
            review_threshold: 410,
        },
        CausalKind::Explains | CausalKind::Motivates | CausalKind::PurposeOf => KindRule {
            accept_threshold: 700,
            review_threshold: 460,
        },
    }
}

fn count_pass_outcome(
    diagnostics: &mut BTreeMap<String, usize>,
    prefix: &str,
    kind: &CausalDecisionKind,
) {
    *diagnostics
        .entry(format!("{prefix}:{}", format!("{kind:?}").to_lowercase()))
        .or_default() += 1;
}

fn count_feature_band(diagnostics: &mut BTreeMap<String, usize>, band: &str) {
    *diagnostics
        .entry(format!("feature_band:{band}"))
        .or_default() += 1;
}

fn count_evidence_classes(
    diagnostics: &mut BTreeMap<String, usize>,
    evidence: &PairEvidenceSummary,
) {
    if evidence.world_support_count > 0 {
        *diagnostics
            .entry("evidence_class:world_support".to_owned())
            .or_default() += evidence.world_support_count;
    }
    if evidence.reported_support_count > 0 {
        *diagnostics
            .entry("evidence_class:reported_support".to_owned())
            .or_default() += evidence.reported_support_count;
    }
    if evidence.attributed_support_count > 0 {
        *diagnostics
            .entry("evidence_class:attributed_support".to_owned())
            .or_default() += evidence.attributed_support_count;
    }
}

fn feature_band(score: i32, rule: KindRule) -> &'static str {
    if score >= rule.accept_threshold {
        "high"
    } else if score >= rule.review_threshold {
        "mid"
    } else if score > 0 {
        "low"
    } else {
        "blocked"
    }
}

fn invalidate_or_reject(seed_source: &str) -> CausalDecisionKind {
    if seed_source == "link" {
        CausalDecisionKind::Invalidate
    } else {
        CausalDecisionKind::Reject
    }
}

fn review_priority(
    score: i32,
    contradict_atom_count: usize,
    underspecified_atom_count: usize,
) -> u32 {
    let mut priority = score.max(0) as u32;
    priority += (contradict_atom_count.min(3) as u32) * 90;
    priority += (underspecified_atom_count.min(3) as u32) * 60;
    priority.min(1000)
}

fn rate_millis(numerator: usize, denominator: usize) -> u32 {
    if denominator == 0 {
        0
    } else {
        ((numerator * 1000) / denominator) as u32
    }
}

fn avg_millis(total: usize, count: usize) -> u32 {
    if count == 0 {
        0
    } else {
        ((total * 1000) / count) as u32
    }
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

fn target_signature(
    target: &SemanticNodeRef,
    relation_kind: phoenix_semantic_v2::CausalRelationKind,
) -> String {
    format!("{}:{relation_kind:?}", node_key(target))
}

fn merged_evidence_refs(case: &CausalReviewCase, claim_atoms: &[&CausalClaimAtom]) -> Vec<String> {
    let mut refs = case.evidence_refs.clone();
    for atom in claim_atoms {
        refs.extend(atom.evidence_refs.iter().cloned());
    }
    refs.sort();
    refs.dedup();
    refs
}

fn is_demoted(before: &CausalDecisionKind, after: &CausalDecisionKind) -> bool {
    matches!(
        before,
        CausalDecisionKind::Accept | CausalDecisionKind::Support
    ) && !matches!(
        after,
        CausalDecisionKind::Accept | CausalDecisionKind::Support
    )
}
