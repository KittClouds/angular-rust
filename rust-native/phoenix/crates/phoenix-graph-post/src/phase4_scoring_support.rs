use std::cell::RefCell;
use std::env;
use std::path::{Path, PathBuf};

use phoenix_rel_post::{
    GliclassClassificationType, GliclassInstructLabel, GliclassInstructModel,
    GliclassInstructPredictOptions, GliclassPrediction,
};

use crate::api::{GraphRankedCausalPath, GraphRankedHistoryCandidate, GraphRankedStateCandidate};
use crate::phase4_contract::GraphPhase4RerankScore;

thread_local! {
    static SCORER_CACHE: RefCell<ScorerCache> = RefCell::new(ScorerCache::default());
}

#[derive(Default)]
struct ScorerCache {
    attempted: bool,
    scorer: Option<GliclassPhase4Scorer>,
}

pub(crate) fn clear_phase4_scorer_cache() {
    SCORER_CACHE.with(|cell| {
        *cell.borrow_mut() = ScorerCache::default();
    });
}

pub(crate) struct GliclassPhase4Scorer {
    model: GliclassInstructModel,
}

impl GliclassPhase4Scorer {
    pub(crate) fn load_default() -> Option<Self> {
        let model_root = default_model_root()?;
        let model = GliclassInstructModel::load(&model_root).ok()?;
        Some(Self { model })
    }

    pub(crate) fn score_labels(
        &self,
        text: &str,
        prompt: &str,
        labels: &[GliclassInstructLabel],
    ) -> Option<GliclassPrediction> {
        self.model
            .predict_structured(
                text,
                labels,
                &GliclassInstructPredictOptions {
                    classification_type: GliclassClassificationType::MultiLabel,
                    threshold: 0.0,
                    prompt: Some(prompt.to_owned()),
                    examples: Vec::new(),
                },
            )
            .ok()
    }
}

pub(crate) fn with_default_scorer<R>(f: impl FnOnce(Option<&GliclassPhase4Scorer>) -> R) -> R {
    SCORER_CACHE.with(|cell| {
        {
            let mut cache = cell.borrow_mut();
            if !cache.attempted {
                cache.attempted = true;
                cache.scorer = GliclassPhase4Scorer::load_default();
            }
        }
        let cache = cell.borrow();
        f(cache.scorer.as_ref())
    })
}

pub(crate) fn label(name: &str, description: &str) -> GliclassInstructLabel {
    GliclassInstructLabel {
        label: name.to_owned(),
        description: Some(description.to_owned()),
    }
}

pub(crate) fn build_rerank_score(
    prediction: &GliclassPrediction,
    positive_label: &str,
    context_label: &str,
    negative_label: &str,
    positive_weight: f64,
    context_weight: f64,
    negative_weight: f64,
    min_delta: f64,
    max_delta: f64,
) -> GraphPhase4RerankScore {
    let positive_score = label_score(prediction, positive_label);
    let context_score = label_score(prediction, context_label);
    let negative_score = label_score(prediction, negative_label);
    let applied_delta = ((positive_score * positive_weight) + (context_score * context_weight)
        - (negative_score * negative_weight))
        .clamp(min_delta, max_delta);
    GraphPhase4RerankScore {
        model: "gliclass_instruct".to_owned(),
        positive_label: positive_label.to_owned(),
        positive_score,
        context_label: context_label.to_owned(),
        context_score,
        negative_label: negative_label.to_owned(),
        negative_score,
        applied_delta,
    }
}

pub(crate) fn phase4_disabled() -> bool {
    matches!(
        env::var("PHOENIX_GRAPH_PHASE4_DISABLED").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub(crate) fn phase4_event_disabled() -> bool {
    matches!(
        env::var("PHOENIX_GRAPH_PHASE4_EVENT_DISABLED")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub(crate) fn phase4_structural_disabled() -> bool {
    matches!(
        env::var("PHOENIX_GRAPH_PHASE4_STRUCTURAL_DISABLED")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub(crate) fn world_abstain(
    selected: Option<&GraphRankedStateCandidate>,
) -> (bool, Option<String>) {
    match selected {
        None => (
            true,
            Some("no world-state candidate passed the plane gate".to_owned()),
        ),
        Some(candidate) if candidate.answer_score < 1.75 => (
            true,
            Some("top candidate was too weak to answer safely".to_owned()),
        ),
        Some(candidate)
            if (candidate.relevant_conflict_count + candidate.relevant_gap_count) > 0
                && candidate.answer_score < 2.1 =>
        {
            (
                true,
                Some(
                    "top candidate remains unresolved under current conflict/gap pressure"
                        .to_owned(),
                ),
            )
        }
        Some(_) => (false, None),
    }
}

pub(crate) fn history_abstain(
    selected: Option<&GraphRankedHistoryCandidate>,
) -> (bool, Option<String>) {
    match selected {
        None => (
            true,
            Some("no history candidate passed the requested truth plane".to_owned()),
        ),
        Some(candidate) if candidate.answer_score < 1.8 => (
            true,
            Some("top history candidate was too weak to answer safely".to_owned()),
        ),
        Some(candidate)
            if (candidate.relevant_conflict_count + candidate.relevant_gap_count) > 0
                && candidate.answer_score < 2.15 =>
        {
            (
                true,
                Some("history window remains unresolved under conflict or gap pressure".to_owned()),
            )
        }
        Some(_) => (false, None),
    }
}

pub(crate) fn causal_abstain(
    selected: Option<&GraphRankedCausalPath>,
    empty_candidates: bool,
) -> (bool, Option<String>) {
    match selected {
        None if empty_candidates => (
            true,
            Some("no causal path was available for the target vertex".to_owned()),
        ),
        None => (
            true,
            Some("no causal path passed the requested truth plane".to_owned()),
        ),
        Some(candidate) if candidate.answer_score < 1.9 => (
            true,
            Some("top causal path was too weak to answer safely".to_owned()),
        ),
        Some(candidate) if candidate.temporal_fitness < 0.65 && candidate.path_stability < 0.7 => (
            true,
            Some("causal support is too brittle to answer confidently".to_owned()),
        ),
        Some(_) => (false, None),
    }
}

fn label_score(prediction: &GliclassPrediction, label: &str) -> f64 {
    prediction
        .all_scores
        .iter()
        .find(|row| row.label == label)
        .map(|row| row.score as f64)
        .unwrap_or(0.0)
}

fn default_model_root() -> Option<PathBuf> {
    env::var_os("PHOENIX_GLICLASS_INSTRUCT_MODEL_ROOT")
        .map(PathBuf::from)
        .filter(|path| path.exists())
        .or_else(|| {
            let root = project_root();
            [
                root.join("gliclass-instruct-onnx-v2"),
                root.join("gliclass-instruct-onnx"),
            ]
            .into_iter()
            .find(|path| path.exists())
        })
}

fn project_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("project root")
        .to_path_buf()
}
