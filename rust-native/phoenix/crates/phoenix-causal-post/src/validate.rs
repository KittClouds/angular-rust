use std::collections::BTreeMap;

use phoenix_semantic_v2::{
    CausalClaimAtom, CausalClaimPolarity, CausalDecisionId, CausalDecisionOutcome,
    CausalDecisionRecord, CausalEdgeAddition, CausalEdgeAliasRecord, CausalInvalidationRecord,
    CausalMetricsSnapshot, CausalReviewQueueItem, CausalClaimStatus,
};
use phoenix_types::{CausalKind, Polarity, TruthStatus};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::normalize::{stable_edge_id, CausalReviewCase};

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
    pub edge_id: phoenix_semantic_v2::CausalEdgeId,
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
    pub metrics_snapshot: CausalMetricsSnapshot,
}

pub fn draft_causal_decisions(
    cases: &[CausalReviewCase],
    claim_atoms: &[CausalClaimAtom],
    created_at: i64,
) -> CausalDecisionDrafts {
    let mut reverse_max = FxHashMap::<String, u32>::default();
    let mut best_case_by_edge = FxHashMap::<String, &CausalReviewCase>::default();
    let mut claims_by_edge = FxHashMap::<String, Vec<&CausalClaimAtom>>::default();

    for atom in claim_atoms {
        claims_by_edge.entry(atom.edge_id.0.clone()).or_default().push(atom);
    }
    for case in cases {
        let reverse_key = format!(
            "{}:{}:{:?}",
            crate::normalize::node_key(&case.target),
            crate::normalize::node_key(&case.source),
            case.relation_kind
        );
        reverse_max
            .entry(reverse_key)
            .and_modify(|value| *value = (*value).max(case.base_confidence_millis))
            .or_insert(case.base_confidence_millis);
        let edge_id = stable_edge_id(case).0;
        let replace = best_case_by_edge
            .get(&edge_id)
            .map(|existing| {
                case.base_confidence_millis > existing.base_confidence_millis
                    || (case.base_confidence_millis == existing.base_confidence_millis
                        && case.seed_source < existing.seed_source)
            })
            .unwrap_or(true);
        if replace {
            best_case_by_edge.insert(edge_id, case);
        }
    }

    let mut edge_keys = best_case_by_edge.keys().cloned().collect::<Vec<_>>();
    edge_keys.sort();

    let mut drafts = CausalDecisionDrafts::default();
    let mut temporal_illegal_count = 0usize;
    let mut cue_only_edge_count = 0usize;

    for edge_key in edge_keys {
        let case = best_case_by_edge[&edge_key];
        let edge_id = stable_edge_id(case);
        let reverse_score = reverse_max
            .get(&format!(
                "{}:{}:{:?}",
                crate::normalize::node_key(&case.source),
                crate::normalize::node_key(&case.target),
                case.relation_kind
            ))
            .copied()
            .unwrap_or_default();
        let claim_refs = claims_by_edge
            .get(&edge_key)
            .cloned()
            .unwrap_or_default();
        let support_atom_count = claim_refs
            .iter()
            .filter(|atom| matches!(atom.polarity, CausalClaimPolarity::Support))
            .count();
        let contradict_atom_count = claim_refs
            .iter()
            .filter(|atom| matches!(atom.polarity, CausalClaimPolarity::Contradict))
            .count();
        let underspecified_atom_count = claim_refs
            .iter()
            .filter(|atom| matches!(atom.polarity, CausalClaimPolarity::Underspecify))
            .count();
        let score = score_case(
            case,
            reverse_score,
            support_atom_count,
            contradict_atom_count,
            underspecified_atom_count,
        );
        let (kind, rationale) = classify_case(
            case,
            score,
            reverse_score,
            support_atom_count,
            contradict_atom_count,
        );
        if !case.temporal_legal {
            temporal_illegal_count += 1;
        }
        if support_atom_count > 0 && contradict_atom_count == 0 && case.cue.is_some() {
            cue_only_edge_count += 1;
        }

        let decision_id = CausalDecisionId(format!("decision:{}:{}", edge_id.0, created_at));
        let evidence = case.evidence_refs.clone();
        *drafts
            .outcome_counts
            .entry(format!("{kind:?}").to_lowercase())
            .or_default() += 1;

        let latest_status = match kind {
            CausalDecisionKind::Accept => CausalClaimStatus::Active,
            CausalDecisionKind::Support => CausalClaimStatus::Supported,
            CausalDecisionKind::Invalidate => CausalClaimStatus::Invalidated,
            CausalDecisionKind::Defer => CausalClaimStatus::Deferred,
            CausalDecisionKind::Reject => {
                if contradict_atom_count > 0 {
                    CausalClaimStatus::Contradicted
                } else {
                    CausalClaimStatus::Rejected
                }
            }
        };

        drafts.decisions.push(CausalDecision {
            decision_id: decision_id.clone(),
            edge_id: edge_id.clone(),
            case_id: case.case_id.clone(),
            document_id: case.document_id.clone(),
            kind: kind.clone(),
            edge_type: Some(case.kind),
            score_millis: score,
            rationale: rationale.clone(),
            evidence: evidence.clone(),
        });
        drafts.decision_records.push(CausalDecisionRecord {
            decision_id: decision_id.clone(),
            edge_id: edge_id.clone(),
            case_id: case.case_id.clone(),
            document_id: case.document_id.clone(),
            outcome: match kind {
                CausalDecisionKind::Accept => CausalDecisionOutcome::Accept,
                CausalDecisionKind::Support => CausalDecisionOutcome::Support,
                CausalDecisionKind::Invalidate => CausalDecisionOutcome::Invalidate,
                CausalDecisionKind::Defer => CausalDecisionOutcome::Defer,
                CausalDecisionKind::Reject => CausalDecisionOutcome::Reject,
            },
            source: Some(case.source.clone()),
            target: Some(case.target.clone()),
            kind: Some(case.kind),
            relation_kind: Some(case.relation_kind),
            score_millis: score,
            rationale: rationale.clone(),
            supersedes: None,
            evidence: evidence.clone(),
            reviewed_at: created_at,
        });

        drafts.edge_records.push(CausalEdgeAddition {
            edge_id: edge_id.clone(),
            case_id: case.case_id.clone(),
            document_id: case.document_id.clone(),
            source: case.source.clone(),
            target: case.target.clone(),
            kind: case.kind,
            relation_kind: case.relation_kind,
            status: latest_status,
            first_seen_revision: case.revision,
            latest_decision_id: Some(decision_id.clone()),
            confidence_millis: score.max(0) as u32,
            cue: case.cue.clone(),
            attributed_to: case.attributed_to.clone(),
            polarity: case.polarity,
            claim_atom_ids: claim_refs.iter().map(|atom| atom.claim_id.clone()).collect(),
            evidence_refs: evidence.clone(),
            effective_interval: case.temporal.clone(),
            observation_interval: case.temporal.clone(),
            temporal_certainty_millis: if case.temporal_legal { 900 } else { 200 },
            created_at,
        });

        drafts.edge_aliases.push(CausalEdgeAliasRecord {
            alias_key: format!("case:{}", case.case_id),
            edge_id: edge_id.clone(),
            document_id: case.document_id.clone(),
            created_at,
        });
        drafts.edge_aliases.push(CausalEdgeAliasRecord {
            alias_key: format!(
                "legacy:{}:{}:{:?}",
                crate::normalize::node_key(&case.source),
                crate::normalize::node_key(&case.target),
                case.kind
            ),
            edge_id: edge_id.clone(),
            document_id: case.document_id.clone(),
            created_at,
        });

        if matches!(kind, CausalDecisionKind::Invalidate) {
            drafts.invalidations.push(CausalInvalidationRecord {
                invalidation_id: format!("invalidate:{}:{}", edge_id.0, created_at),
                edge_id: edge_id.clone(),
                decision_id: decision_id.clone(),
                document_id: case.document_id.clone(),
                rationale: rationale.clone(),
                evidence_refs: evidence.clone(),
                created_at,
            });
        }
        if matches!(kind, CausalDecisionKind::Defer | CausalDecisionKind::Reject) {
            drafts.review_queue.push(CausalReviewQueueItem {
                queue_id: format!("queue:{}:{}", edge_id.0, created_at),
                edge_id,
                latest_decision_id: Some(decision_id),
                document_id: case.document_id.clone(),
                priority_millis: review_priority(score, contradict_atom_count, underspecified_atom_count),
                rationale,
                unresolved: true,
                created_at,
            });
        }
    }

    let edge_record_count = drafts.edge_records.len();
    let accepted_count = drafts
        .edge_records
        .iter()
        .filter(|edge| matches!(edge.status, CausalClaimStatus::Active | CausalClaimStatus::Supported))
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
        contradiction_rate_per_1k_events_millis: rate_millis(contradicted_count, cases.len().max(1)),
        edge_survival_rate_millis: 1000,
        chain_collapse_rate_millis: 0,
        avg_claim_atoms_per_edge_millis: avg_millis(claim_atoms.len(), drafts.edge_records.len().max(1)),
        cue_only_edge_rate_millis: rate_millis(cue_only_edge_count, drafts.edge_records.len().max(1)),
        card_open_dispute_rate_millis: rate_millis(drafts.review_queue.len(), drafts.edge_records.len().max(1)),
        temporal_illegality_rejection_rate_millis: rate_millis(temporal_illegal_count, cases.len().max(1)),
    };
    drafts
}

fn score_case(
    case: &CausalReviewCase,
    reverse_score: u32,
    support_atom_count: usize,
    contradict_atom_count: usize,
    underspecified_atom_count: usize,
) -> i32 {
    let mut score = case.base_confidence_millis as i32;
    score += match case.base_status {
        TruthStatus::Asserted => 120,
        TruthStatus::Candidate => 0,
        TruthStatus::Rejected => -260,
        TruthStatus::Expired => -120,
        TruthStatus::Unknown => -20,
    };
    score += (case.shared_participant_count.min(3) as i32) * 85;
    score += (case.graph_support_count.min(4) as i32) * 60;
    score += (case.centrality_millis.min(500) as i32) / 4;
    score += (support_atom_count.min(3) as i32) * 70;
    score -= (contradict_atom_count.min(3) as i32) * 110;
    score -= (underspecified_atom_count.min(2) as i32) * 70;
    score -= (case.sentence_distance.min(2) as i32) * 55;
    if case.quoted_or_attributed {
        score -= 200;
    }
    if !case.temporal_legal {
        score -= 380;
    }
    if reverse_score > case.base_confidence_millis + 120 {
        score -= 220;
    }
    if matches!(case.polarity, Polarity::Negative)
        && !matches!(case.kind, CausalKind::Prevents | CausalKind::Hinders)
    {
        score -= 90;
    }
    score
}

fn classify_case(
    case: &CausalReviewCase,
    score: i32,
    reverse_score: u32,
    support_atom_count: usize,
    contradict_atom_count: usize,
) -> (CausalDecisionKind, String) {
    if !case.temporal_legal {
        return (
            if case.seed_source == "link" {
                CausalDecisionKind::Invalidate
            } else {
                CausalDecisionKind::Reject
            },
            "temporal_illegality".to_owned(),
        );
    }
    if reverse_score > case.base_confidence_millis + 120 {
        return (
            if case.seed_source == "link" {
                CausalDecisionKind::Invalidate
            } else {
                CausalDecisionKind::Reject
            },
            "reverse_direction_stronger".to_owned(),
        );
    }

    let rule = kind_rule(case.kind);
    if contradict_atom_count > support_atom_count && score < rule.accept_threshold {
        return (CausalDecisionKind::Reject, "contradictory_claim_atoms".to_owned());
    }
    if case.quoted_or_attributed && score < rule.accept_threshold {
        return (CausalDecisionKind::Defer, "quoted_or_attributed".to_owned());
    }
    if score >= rule.accept_threshold {
        return (
            if case.seed_source == "link" {
                CausalDecisionKind::Support
            } else {
                CausalDecisionKind::Accept
            },
            "supported_by_claim_atoms_and_graph".to_owned(),
        );
    }
    if score >= rule.review_threshold {
        return (CausalDecisionKind::Defer, "needs_counterfactual_review".to_owned());
    }
    (CausalDecisionKind::Reject, "insufficient_support".to_owned())
}

struct KindRule {
    accept_threshold: i32,
    review_threshold: i32,
}

fn kind_rule(kind: CausalKind) -> KindRule {
    match kind {
        CausalKind::Causes | CausalKind::ResultsIn | CausalKind::TriggerFor | CausalKind::Prevents => {
            KindRule {
                accept_threshold: 720,
                review_threshold: 480,
            }
        }
        CausalKind::Enables | CausalKind::ConditionFor | CausalKind::Hinders => KindRule {
            accept_threshold: 680,
            review_threshold: 460,
        },
        CausalKind::Explains | CausalKind::Motivates | CausalKind::PurposeOf => KindRule {
            accept_threshold: 760,
            review_threshold: 520,
        },
    }
}

fn review_priority(score: i32, contradict_atom_count: usize, underspecified_atom_count: usize) -> u32 {
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
