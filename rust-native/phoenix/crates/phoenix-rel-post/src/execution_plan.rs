use std::collections::BTreeMap;

use rustc_hash::FxHashMap;

use crate::glirel::{
    extract_heuristic_relations, GlirelBatchItem, GlirelEntity, GlirelModel, GlirelProposalConfig,
    GlirelRelationTypeSpec,
};
use crate::worker::{
    build_window_glirel_entities, filter_relation_predictions, merge_relation_prediction_lanes,
    select_window_relation_specs, GlirelWorkerError, RelationReviewCase, RelationScopeReviewBatch,
};

#[derive(Clone, Debug, PartialEq)]
pub struct RelationExecutionWindow {
    pub window_index: usize,
    pub case_indices: Vec<usize>,
    pub schema_group_index: usize,
    pub entities: Vec<GlirelEntity>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RelationExecutionSchemaGroup {
    pub schema_labels: Vec<String>,
    pub selected_specs: Vec<GlirelRelationTypeSpec>,
    pub min_threshold: f32,
    pub execution_indices: Vec<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct RelationExecutionPlan {
    pub executions: Vec<RelationExecutionWindow>,
    pub schema_groups: Vec<RelationExecutionSchemaGroup>,
}

impl RelationExecutionPlan {
    pub fn build(
        batch: &RelationScopeReviewBatch,
        relation_specs: &[GlirelRelationTypeSpec],
    ) -> Self {
        let case_indices_by_window = build_case_indices_by_window(batch);
        let mut executions = Vec::new();
        let mut schema_groups = Vec::new();
        let mut schema_group_by_labels = BTreeMap::<Vec<String>, usize>::new();

        for (window_index, case_indices) in case_indices_by_window.into_iter().enumerate() {
            if case_indices.is_empty() {
                continue;
            }
            let Some(window) = batch.windows.get(window_index) else {
                continue;
            };
            if window.entities.len() < 2 {
                continue;
            }
            let selected_specs = select_window_relation_specs(window, relation_specs);
            if selected_specs.is_empty() {
                continue;
            }
            let schema_labels = selected_specs
                .iter()
                .map(|spec| spec.label.clone())
                .collect::<Vec<_>>();
            let schema_group_index = match schema_group_by_labels.get(&schema_labels).copied() {
                Some(index) => index,
                None => {
                    let index = schema_groups.len();
                    schema_group_by_labels.insert(schema_labels.clone(), index);
                    schema_groups.push(RelationExecutionSchemaGroup {
                        schema_labels,
                        min_threshold: selected_specs
                            .iter()
                            .map(|spec| spec.review_threshold_millis)
                            .min()
                            .unwrap_or(450)
                            .min(300) as f32
                            / 1000.0,
                        selected_specs,
                        execution_indices: Vec::new(),
                    });
                    index
                }
            };
            let execution_index = executions.len();
            schema_groups[schema_group_index]
                .execution_indices
                .push(execution_index);
            executions.push(RelationExecutionWindow {
                window_index,
                case_indices,
                schema_group_index,
                entities: build_window_glirel_entities(window),
            });
        }

        Self {
            executions,
            schema_groups,
        }
    }

    pub fn apply_glirel(
        &self,
        batch: &mut RelationScopeReviewBatch,
        model: &GlirelModel,
    ) -> Result<(), GlirelWorkerError> {
        for schema_group in &self.schema_groups {
            if schema_group.execution_indices.is_empty() {
                continue;
            }
            let mut items = Vec::with_capacity(schema_group.execution_indices.len());
            for &execution_index in &schema_group.execution_indices {
                let execution = &self.executions[execution_index];
                let window = &batch.windows[execution.window_index];
                items.push(GlirelBatchItem {
                    text: window.text.as_str(),
                    entities: &execution.entities,
                });
            }

            let model_predictions = model
                .extract_many_with_schema(
                    &items,
                    &schema_group.selected_specs,
                    schema_group.min_threshold,
                )
                .map_err(GlirelWorkerError::Model)?;

            for (group_index, window_predictions) in model_predictions.into_iter().enumerate() {
                let execution = &self.executions[schema_group.execution_indices[group_index]];
                let window = &batch.windows[execution.window_index];
                let heuristic_predictions = extract_heuristic_relations(
                    &window.text,
                    &execution.entities,
                    &schema_group.selected_specs,
                    &GlirelProposalConfig::default(),
                );
                let filtered_predictions = filter_relation_predictions(
                    &window.text,
                    &window.entities,
                    window.range.start as usize,
                    &schema_group.selected_specs,
                    merge_relation_prediction_lanes(window_predictions, heuristic_predictions),
                );
                assign_window_predictions_to_cases(
                    &mut batch.review_cases,
                    &execution.case_indices,
                    &execution.entities,
                    filtered_predictions,
                );
            }
        }
        Ok(())
    }

    pub fn apply_heuristic(&self, batch: &mut RelationScopeReviewBatch) {
        for execution in &self.executions {
            let window = &batch.windows[execution.window_index];
            let schema_group = &self.schema_groups[execution.schema_group_index];
            let filtered_predictions = filter_relation_predictions(
                &window.text,
                &window.entities,
                window.range.start as usize,
                &schema_group.selected_specs,
                extract_heuristic_relations(
                    &window.text,
                    &execution.entities,
                    &schema_group.selected_specs,
                    &GlirelProposalConfig::default(),
                )
                .into_iter()
                .map(|mut prediction| {
                    prediction
                        .evidence
                        .push("proposal_engine:heuristic".to_owned());
                    prediction
                })
                .collect(),
            );
            assign_window_predictions_to_cases(
                &mut batch.review_cases,
                &execution.case_indices,
                &execution.entities,
                filtered_predictions,
            );
        }
    }
}

fn build_case_indices_by_window(batch: &RelationScopeReviewBatch) -> Vec<Vec<usize>> {
    let window_index_by_id = batch
        .windows
        .iter()
        .enumerate()
        .map(|(window_index, window)| (window.window_id.as_str(), window_index))
        .collect::<FxHashMap<_, _>>();
    let mut case_indices_by_window = vec![Vec::new(); batch.windows.len()];
    for (case_index, case) in batch.review_cases.iter().enumerate() {
        let Some(&window_index) = window_index_by_id.get(case.window_id.as_str()) else {
            continue;
        };
        case_indices_by_window[window_index].push(case_index);
    }
    case_indices_by_window
}

fn assign_window_predictions_to_cases(
    review_cases: &mut [RelationReviewCase],
    case_indices: &[usize],
    entities: &[GlirelEntity],
    predictions: Vec<crate::GlirelRelationPrediction>,
) {
    let predictions_by_pair = predictions.into_iter().fold(
        FxHashMap::<(String, String), Vec<crate::GlirelRelationPrediction>>::default(),
        |mut acc, prediction| {
            let head_id = entities[prediction.head_index]
                .entity_id
                .clone()
                .unwrap_or_default();
            let tail_id = entities[prediction.tail_index]
                .entity_id
                .clone()
                .unwrap_or_default();
            acc.entry((head_id, tail_id)).or_default().push(prediction);
            acc
        },
    );

    for &case_index in case_indices {
        let case = &mut review_cases[case_index];
        let key = (
            case.source_entity_id.0.clone(),
            case.target_entity_id.0.clone(),
        );
        let mut rows = predictions_by_pair.get(&key).cloned().unwrap_or_default();
        rows.sort_by(|left, right| {
            right
                .confidence
                .partial_cmp(&left.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        case.glirel_predictions = rows;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        default_relation_type_specs, RelationReviewCase, RelationScopeReviewBatch,
        RelationWindowEntity, RelationWindowRecord,
    };
    use phoenix_semantic_v2::ScopeOrd;
    use phoenix_types::{EntityId, EntityKind, ScopeKey, TextRange};

    fn test_scope() -> ScopeKey {
        ScopeKey {
            world_id: Some("world".to_owned()),
            narrative_id: Some("narrative".to_owned()),
            folder_id: None,
            folder_path: None,
        }
    }

    fn sample_window(
        window_id: &str,
        text: &str,
        candidate_relation_types: Vec<String>,
        entities: Vec<RelationWindowEntity>,
    ) -> RelationWindowRecord {
        RelationWindowRecord {
            window_id: window_id.to_owned(),
            document_id: "doc-1".to_owned(),
            revision: 1,
            window_index: 0,
            range: TextRange {
                start: 0,
                end: text.len() as u32,
            },
            sentence_indices: vec![0],
            chunk_ids: vec!["chunk-1".to_owned()],
            candidate_relation_types,
            evidence_labels: Vec::new(),
            text: text.to_owned(),
            entities,
        }
    }

    fn sample_entity(
        entity_id: &str,
        surface: &str,
        entity_type: &str,
        kind: Option<EntityKind>,
        span_start: usize,
        span_end: usize,
    ) -> RelationWindowEntity {
        RelationWindowEntity {
            entity_id: EntityId(entity_id.to_owned()),
            surface: surface.to_owned(),
            kind,
            entity_type: entity_type.to_owned(),
            span_start,
            span_end,
            sentence_index: 0,
            mention_index: Some(0),
        }
    }

    fn sample_case(
        scope: &ScopeKey,
        scope_key: &str,
        window_id: &str,
        source_entity_id: &str,
        target_entity_id: &str,
    ) -> RelationReviewCase {
        RelationReviewCase {
            case_id: format!("{window_id}:{source_entity_id}:{target_entity_id}"),
            scope: scope.clone(),
            scope_key: scope_key.to_owned(),
            scope_ord: ScopeOrd(7),
            session_id: None,
            document_id: "doc-1".to_owned(),
            revision: 1,
            window_id: window_id.to_owned(),
            window_index: 0,
            window_range: TextRange { start: 0, end: 32 },
            sentence_indices: vec![0],
            chunk_ids: vec!["chunk-1".to_owned()],
            window_text: String::new(),
            source_entity_id: EntityId(source_entity_id.to_owned()),
            target_entity_id: EntityId(target_entity_id.to_owned()),
            source_name: source_entity_id.to_owned(),
            target_name: target_entity_id.to_owned(),
            source_kind: None,
            target_kind: None,
            seed_score_millis: 500,
            seed_evidence: Vec::new(),
            serialized: String::new(),
            blocking_keys: Vec::new(),
            glirel_predictions: Vec::new(),
            accepted_relations: Vec::new(),
            decision_status: "relation_pending".to_owned(),
        }
    }

    #[test]
    fn relation_execution_plan_groups_windows_by_schema() {
        let scope = test_scope();
        let scope_key = "world::narrative".to_owned();
        let batch = RelationScopeReviewBatch {
            scope: scope.clone(),
            scope_key: scope_key.clone(),
            scope_ord: ScopeOrd(7),
            windows: vec![
                sample_window(
                    "window-1",
                    "Alice joined Dynamis.",
                    vec!["member_of".to_owned()],
                    vec![
                        sample_entity(
                            "e1",
                            "Alice",
                            "Character",
                            Some(EntityKind::Character),
                            0,
                            5,
                        ),
                        sample_entity(
                            "e2",
                            "Dynamis",
                            "Organization",
                            Some(EntityKind::Organization),
                            13,
                            20,
                        ),
                    ],
                ),
                sample_window(
                    "window-2",
                    "Dynamis is in New Rome.",
                    vec!["located_in".to_owned()],
                    vec![
                        sample_entity(
                            "e2",
                            "Dynamis",
                            "Organization",
                            Some(EntityKind::Organization),
                            0,
                            7,
                        ),
                        sample_entity(
                            "e3",
                            "New Rome",
                            "Location",
                            Some(EntityKind::Location),
                            14,
                            22,
                        ),
                    ],
                ),
            ],
            review_cases: vec![
                sample_case(&scope, &scope_key, "window-1", "e1", "e2"),
                sample_case(&scope, &scope_key, "window-1", "e2", "e1"),
                sample_case(&scope, &scope_key, "window-2", "e2", "e3"),
            ],
            ..Default::default()
        };

        let plan = RelationExecutionPlan::build(&batch, &default_relation_type_specs());
        assert_eq!(plan.executions.len(), 2);
        assert_eq!(plan.executions[0].case_indices, vec![0, 1]);
        assert_eq!(plan.executions[1].case_indices, vec![2]);
        assert_eq!(plan.schema_groups.len(), 2);
        assert_eq!(
            plan.schema_groups[plan.executions[0].schema_group_index].schema_labels,
            vec!["works_for".to_owned(), "member_of".to_owned()]
        );
        assert_eq!(
            plan.schema_groups[plan.executions[1].schema_group_index].schema_labels,
            vec!["located_in".to_owned()]
        );
    }
}
