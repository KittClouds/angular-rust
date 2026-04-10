use std::path::PathBuf;

use hashbrown::HashMap;
use phoenix_rel_post::{NliError, NliModel};
use phoenix_semantic_v2::{
    SemanticCandidateStatus, SemanticEdgeFamily, SemanticGraphEdgeCandidate,
};

use crate::semantic::ensure_ort_dylib_path;
use crate::semantic_graph_support::Prototype;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticNliConfig {
    pub model_root: PathBuf,
    pub support_threshold_millis: u32,
    pub contradiction_threshold_millis: u32,
    pub review_threshold_millis: u32,
}

impl SemanticNliConfig {
    pub(crate) fn load_model(&self) -> Result<NliModel, NliError> {
        let _ = ensure_ort_dylib_path();
        NliModel::load(&self.model_root)
    }
}

pub(crate) fn needs_nli_review(family: SemanticEdgeFamily) -> bool {
    matches!(
        family,
        SemanticEdgeFamily::ClaimSupport
            | SemanticEdgeFamily::ClaimContradiction
            | SemanticEdgeFamily::StateSupport
            | SemanticEdgeFamily::StateContradiction
            | SemanticEdgeFamily::ContradictorySupportRegion
    )
}

pub(crate) fn adjudicate_candidates_with_nli(
    candidates: &mut [SemanticGraphEdgeCandidate],
    prototypes: &[Prototype],
    nli: &NliModel,
    config: &SemanticNliConfig,
) -> Result<(), NliError> {
    let prototype_by_id = prototypes
        .iter()
        .map(|prototype| (prototype.node_id.as_str(), prototype))
        .collect::<HashMap<_, _>>();
    for candidate in candidates {
        if !needs_nli_review(candidate.family) {
            continue;
        }
        let Some(source) = prototype_by_id.get(candidate.source_node_id.as_str()) else {
            continue;
        };
        let Some(target) = prototype_by_id.get(candidate.target_node_id.as_str()) else {
            continue;
        };
        let forward = nli.score(&source.text, &target.text)?;
        let reverse = nli.score(&target.text, &source.text)?;
        let support_millis = (forward.entailment.max(reverse.entailment) * 1000.0).round() as u32;
        let contradiction_millis =
            (forward.contradiction.max(reverse.contradiction) * 1000.0).round() as u32;
        candidate.nli_support_millis = Some(support_millis);
        candidate.nli_contradiction_millis = Some(contradiction_millis);
        candidate.model_evidence.push(format!(
            "nli:entailment={:.3};reverse_entailment={:.3};contradiction={:.3};reverse_contradiction={:.3}",
            forward.entailment,
            reverse.entailment,
            forward.contradiction,
            reverse.contradiction,
        ));
        let (family, status) = adjudicate_family(
            candidate.family,
            support_millis,
            contradiction_millis,
            config,
        );
        candidate.family = family;
        candidate.candidate_status = status;
    }
    Ok(())
}

fn adjudicate_family(
    family: SemanticEdgeFamily,
    support_millis: u32,
    contradiction_millis: u32,
    config: &SemanticNliConfig,
) -> (SemanticEdgeFamily, SemanticCandidateStatus) {
    if family == SemanticEdgeFamily::ContradictorySupportRegion {
        return adjudicate_contradiction_region(support_millis, contradiction_millis, config);
    }
    if contradiction_millis >= config.contradiction_threshold_millis
        && contradiction_millis >= support_millis.saturating_add(60)
    {
        return (
            contradiction_variant(family),
            SemanticCandidateStatus::ReviewedContradiction,
        );
    }
    if support_millis >= config.support_threshold_millis
        && support_millis >= contradiction_millis.saturating_add(20)
    {
        return (
            support_variant(family),
            SemanticCandidateStatus::ReviewedSupport,
        );
    }
    if contradiction_millis >= config.review_threshold_millis
        && contradiction_millis > support_millis
    {
        return (
            contradiction_variant(family),
            SemanticCandidateStatus::Deferred,
        );
    }
    if support_millis >= config.review_threshold_millis {
        return (support_variant(family), SemanticCandidateStatus::Deferred);
    }
    (family, SemanticCandidateStatus::Rejected)
}

fn adjudicate_contradiction_region(
    support_millis: u32,
    contradiction_millis: u32,
    config: &SemanticNliConfig,
) -> (SemanticEdgeFamily, SemanticCandidateStatus) {
    if contradiction_millis >= config.contradiction_threshold_millis
        && contradiction_millis >= support_millis.saturating_add(60)
    {
        return (
            SemanticEdgeFamily::ContradictorySupportRegion,
            SemanticCandidateStatus::ReviewedContradiction,
        );
    }
    if contradiction_millis >= config.review_threshold_millis
        && contradiction_millis > support_millis
    {
        return (
            SemanticEdgeFamily::ContradictorySupportRegion,
            SemanticCandidateStatus::Deferred,
        );
    }
    (
        SemanticEdgeFamily::ContradictorySupportRegion,
        SemanticCandidateStatus::Rejected,
    )
}

fn support_variant(family: SemanticEdgeFamily) -> SemanticEdgeFamily {
    match family {
        SemanticEdgeFamily::ClaimSupport | SemanticEdgeFamily::ClaimContradiction => {
            SemanticEdgeFamily::ClaimSupport
        }
        SemanticEdgeFamily::StateSupport | SemanticEdgeFamily::StateContradiction => {
            SemanticEdgeFamily::StateSupport
        }
        _ => family,
    }
}

fn contradiction_variant(family: SemanticEdgeFamily) -> SemanticEdgeFamily {
    match family {
        SemanticEdgeFamily::ClaimSupport | SemanticEdgeFamily::ClaimContradiction => {
            SemanticEdgeFamily::ClaimContradiction
        }
        SemanticEdgeFamily::StateSupport | SemanticEdgeFamily::StateContradiction => {
            SemanticEdgeFamily::StateContradiction
        }
        _ => family,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        adjudicate_family, contradiction_variant, needs_nli_review, support_variant,
        SemanticNliConfig,
    };
    use phoenix_semantic_v2::{SemanticCandidateStatus, SemanticEdgeFamily};

    fn config() -> SemanticNliConfig {
        SemanticNliConfig {
            model_root: PathBuf::from("G:\\phoenix-models\\nli-deberta-v3-small"),
            support_threshold_millis: 720,
            contradiction_threshold_millis: 740,
            review_threshold_millis: 560,
        }
    }

    use std::path::PathBuf;

    #[test]
    fn support_and_contradiction_variants_flip_family_cleanly() {
        assert_eq!(
            support_variant(SemanticEdgeFamily::ClaimContradiction),
            SemanticEdgeFamily::ClaimSupport
        );
        assert_eq!(
            contradiction_variant(SemanticEdgeFamily::StateSupport),
            SemanticEdgeFamily::StateContradiction
        );
    }

    #[test]
    fn adjudication_prefers_clear_contradiction_signal() {
        let (family, status) =
            adjudicate_family(SemanticEdgeFamily::ClaimSupport, 510, 810, &config());
        assert_eq!(family, SemanticEdgeFamily::ClaimContradiction);
        assert_eq!(status, SemanticCandidateStatus::ReviewedContradiction);
    }

    #[test]
    fn adjudication_prefers_clear_support_signal() {
        let (family, status) =
            adjudicate_family(SemanticEdgeFamily::StateContradiction, 790, 410, &config());
        assert_eq!(family, SemanticEdgeFamily::StateSupport);
        assert_eq!(status, SemanticCandidateStatus::ReviewedSupport);
    }

    #[test]
    fn adjudication_rejects_weak_pairs() {
        let (_, status) = adjudicate_family(SemanticEdgeFamily::StateSupport, 320, 280, &config());
        assert_eq!(status, SemanticCandidateStatus::Rejected);
        assert!(needs_nli_review(SemanticEdgeFamily::ClaimSupport));
        assert!(needs_nli_review(
            SemanticEdgeFamily::ContradictorySupportRegion
        ));
        assert!(!needs_nli_review(SemanticEdgeFamily::EntityEventSupport));
    }

    #[test]
    fn contradiction_region_only_survives_contradiction_signal() {
        let (family, status) = adjudicate_family(
            SemanticEdgeFamily::ContradictorySupportRegion,
            390,
            812,
            &config(),
        );
        assert_eq!(family, SemanticEdgeFamily::ContradictorySupportRegion);
        assert_eq!(status, SemanticCandidateStatus::ReviewedContradiction);

        let (_, weak_status) = adjudicate_family(
            SemanticEdgeFamily::ContradictorySupportRegion,
            780,
            420,
            &config(),
        );
        assert_eq!(weak_status, SemanticCandidateStatus::Rejected);
    }
}
