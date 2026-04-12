use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::api as rel_api;
use crate::glirel::{
    extract_heuristic_relations, GlirelBatchItem, GlirelEntity, GlirelModel, GlirelProposalConfig,
    GlirelRelationPrediction, GlirelRelationTypeSpec,
};
use crate::worker::{
    filter_relation_predictions, merge_relation_prediction_lanes, GlirelWorkerError,
    RelationReviewCase, RelationScopeReviewBatch,
};
use crate::RelationExecutionPlan;

const MAX_MODEL_JOB_WINDOWS: usize = 8;
const MAX_MODEL_JOB_WORDS: usize = 600;
const MAX_MODEL_JOB_PAIR_SLOTS: usize = 192;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelationModelJob {
    pub schema_group_index: usize,
    pub execution_indices: Vec<usize>,
    pub window_count: usize,
    pub estimated_word_count: usize,
    pub estimated_pair_slots: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RelationPreparedStageInput {
    pub relation_spec_signature: u64,
    pub batch: RelationScopeReviewBatch,
    pub plan: RelationExecutionPlan,
    pub model_jobs: Vec<RelationModelJob>,
}

impl RelationPreparedStageInput {
    pub fn build(
        batch: RelationScopeReviewBatch,
        relation_specs: &[GlirelRelationTypeSpec],
    ) -> Self {
        let plan = RelationExecutionPlan::build(&batch, relation_specs);
        let model_jobs = build_model_jobs(&batch, &plan);
        Self {
            relation_spec_signature: relation_spec_signature(relation_specs),
            batch,
            plan,
            model_jobs,
        }
    }

    pub fn apply_model_job(
        &self,
        batch: &mut RelationScopeReviewBatch,
        model: &GlirelModel,
        job: &RelationModelJob,
    ) -> Result<(), GlirelWorkerError> {
        let schema_group = &self.plan.schema_groups[job.schema_group_index];
        let mut items = Vec::with_capacity(job.execution_indices.len());
        for &execution_index in &job.execution_indices {
            let execution = &self.plan.executions[execution_index];
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

        for (job_index, window_predictions) in model_predictions.into_iter().enumerate() {
            let execution = &self.plan.executions[job.execution_indices[job_index]];
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

        Ok(())
    }

    pub fn apply_all_model_jobs(
        &self,
        batch: &mut RelationScopeReviewBatch,
        model: &GlirelModel,
    ) -> Result<(), GlirelWorkerError> {
        for job in &self.model_jobs {
            self.apply_model_job(batch, model, job)?;
        }
        Ok(())
    }
}

pub fn prepare_stage_input(
    analysis: &phoenix_scope_analysis::ScopeAnalysisContext,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Result<RelationPreparedStageInput, GlirelWorkerError> {
    let batch =
        rel_api::derive_batch_from_analysis(analysis, analysis.runtime.sidecars.relation.as_ref())?;
    Ok(RelationPreparedStageInput::build(batch, relation_specs))
}

pub fn relation_spec_signature(relation_specs: &[GlirelRelationTypeSpec]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for spec in relation_specs {
        spec.label.hash(&mut hasher);
        spec.head_types.hash(&mut hasher);
        spec.tail_types.hash(&mut hasher);
        spec.cue_phrases.hash(&mut hasher);
        spec.conflicts_with.hash(&mut hasher);
        spec.priority_millis.hash(&mut hasher);
        spec.accept_threshold_millis.hash(&mut hasher);
        spec.review_threshold_millis.hash(&mut hasher);
        spec.max_predictions_per_window.hash(&mut hasher);
        spec.directed.hash(&mut hasher);
    }
    hasher.finish()
}

fn build_model_jobs(
    batch: &RelationScopeReviewBatch,
    plan: &RelationExecutionPlan,
) -> Vec<RelationModelJob> {
    let mut jobs = Vec::new();
    for (schema_group_index, schema_group) in plan.schema_groups.iter().enumerate() {
        let mut current_job = RelationModelJob {
            schema_group_index,
            execution_indices: Vec::new(),
            window_count: 0,
            estimated_word_count: 0,
            estimated_pair_slots: 0,
        };
        for &execution_index in &schema_group.execution_indices {
            let execution = &plan.executions[execution_index];
            let window = &batch.windows[execution.window_index];
            let estimated_word_count = window.text.split_whitespace().count();
            let estimated_pair_slots = execution
                .entities
                .len()
                .saturating_mul(execution.entities.len().saturating_sub(1));
            let next_window_count = current_job.window_count + 1;
            let next_word_count = current_job.estimated_word_count + estimated_word_count;
            let next_pair_slots = current_job.estimated_pair_slots + estimated_pair_slots;
            let exceeds_budget = next_window_count > MAX_MODEL_JOB_WINDOWS
                || next_word_count > MAX_MODEL_JOB_WORDS
                || next_pair_slots > MAX_MODEL_JOB_PAIR_SLOTS;

            if exceeds_budget && !current_job.execution_indices.is_empty() {
                jobs.push(current_job);
                current_job = RelationModelJob {
                    schema_group_index,
                    execution_indices: Vec::new(),
                    window_count: 0,
                    estimated_word_count: 0,
                    estimated_pair_slots: 0,
                };
            }

            current_job.execution_indices.push(execution_index);
            current_job.window_count += 1;
            current_job.estimated_word_count += estimated_word_count;
            current_job.estimated_pair_slots += estimated_pair_slots;
        }
        if !current_job.execution_indices.is_empty() {
            jobs.push(current_job);
        }
    }
    jobs
}

fn assign_window_predictions_to_cases(
    review_cases: &mut [RelationReviewCase],
    case_indices: &[usize],
    entities: &[GlirelEntity],
    predictions: Vec<GlirelRelationPrediction>,
) {
    let predictions_by_pair = predictions.into_iter().fold(
        rustc_hash::FxHashMap::<(String, String), Vec<GlirelRelationPrediction>>::default(),
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
    use crate::default_relation_type_specs;
    use phoenix_semantic_v2::{
        CandidateEntity, ChunkId, ChunkRecord, DocumentArchive, DocumentManifest,
        ResolutionDecision, ResolvedMention, ScopeOrd, SemanticEntityRecord,
    };
    use phoenix_types::{
        EntityId, EntityKind, IngestDocumentSummary, ScopeKey, SessionDocumentState, TextRange,
    };

    fn sample_archive() -> DocumentArchive {
        DocumentArchive {
            manifest: DocumentManifest {
                document_id: "doc-1".to_owned(),
                scope_key: "world::story".to_owned(),
                scope: ScopeKey {
                    world_id: Some("world".to_owned()),
                    narrative_id: Some("story".to_owned()),
                    folder_id: None,
                    folder_path: None,
                },
                scope_ord: ScopeOrd(7),
                revision: 1,
                title: "Test".to_owned(),
                session_document: SessionDocumentState::default(),
                document_summary: IngestDocumentSummary::default(),
                ..Default::default()
            },
            resolved_mentions: vec![
                ResolvedMention {
                    mention_id: phoenix_semantic_v2::MentionId("m1".to_owned()),
                    mention_index: 0,
                    range: TextRange { start: 0, end: 5 },
                    surface: "Alice".to_owned(),
                    normalized: "alice".to_owned(),
                    kind: Some(EntityKind::Character),
                    entity_id: Some(EntityId("e1".to_owned())),
                    decision: ResolutionDecision {
                        status: "resolved".to_owned(),
                        confidence_millis: 900,
                        margin_millis: 200,
                    },
                    candidates: vec![CandidateEntity {
                        entity_id: "e1".to_owned(),
                        source: "native".to_owned(),
                        score_millis: 900,
                        evidence: Vec::new(),
                    }],
                },
                ResolvedMention {
                    mention_id: phoenix_semantic_v2::MentionId("m2".to_owned()),
                    mention_index: 1,
                    range: TextRange { start: 13, end: 20 },
                    surface: "Dynamis".to_owned(),
                    normalized: "dynamis".to_owned(),
                    kind: Some(EntityKind::Organization),
                    entity_id: Some(EntityId("e2".to_owned())),
                    decision: ResolutionDecision {
                        status: "resolved".to_owned(),
                        confidence_millis: 890,
                        margin_millis: 200,
                    },
                    candidates: Vec::new(),
                },
            ],
            entities: vec![
                SemanticEntityRecord {
                    entity_id: EntityId("e1".to_owned()),
                    canonical_name: "Alice".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Character),
                    mention_count: 1,
                    chunk_ids: vec!["chunk-1".to_owned()],
                },
                SemanticEntityRecord {
                    entity_id: EntityId("e2".to_owned()),
                    canonical_name: "Dynamis".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Organization),
                    mention_count: 1,
                    chunk_ids: vec!["chunk-1".to_owned()],
                },
            ],
            chunks: vec![ChunkRecord {
                chunk_id: ChunkId("chunk-1".to_owned()),
                range: TextRange { start: 0, end: 21 },
                chapter_id: 0,
                boundary_label: None,
                text: "Alice joined Dynamis.".to_owned(),
            }],
            ..Default::default()
        }
    }

    fn sample_archive_with_id(document_id: &str, chunk_id: &str) -> DocumentArchive {
        let mut archive = sample_archive();
        archive.manifest.document_id = document_id.to_owned();
        archive.entities.iter_mut().for_each(|entity| {
            entity.chunk_ids = vec![chunk_id.to_owned()];
        });
        archive.chunks[0].chunk_id = ChunkId(chunk_id.to_owned());
        archive
    }

    #[test]
    fn relation_spec_signature_is_stable_for_same_schema() {
        let specs = default_relation_type_specs();
        assert_eq!(
            relation_spec_signature(&specs),
            relation_spec_signature(&specs)
        );
    }

    #[test]
    fn prepared_stage_input_emits_model_jobs() {
        let batch =
            crate::derive_scope_review_batch(&[sample_archive()], None, None, None, None, None);
        let prepared = RelationPreparedStageInput::build(batch, &default_relation_type_specs());
        assert!(!prepared.plan.executions.is_empty());
        assert!(!prepared.model_jobs.is_empty());
        assert!(prepared.model_jobs.iter().all(|job| job.window_count > 0));
    }

    #[test]
    fn prepared_stage_input_splits_jobs_by_window_budget() {
        let archives = (0..(MAX_MODEL_JOB_WINDOWS + 1))
            .map(|index| sample_archive_with_id(&format!("doc-{index}"), &format!("chunk-{index}")))
            .collect::<Vec<_>>();
        let batch = crate::derive_scope_review_batch(&archives, None, None, None, None, None);
        let prepared = RelationPreparedStageInput::build(batch, &default_relation_type_specs());

        assert!(prepared.model_jobs.len() >= 2);
        assert!(prepared
            .model_jobs
            .iter()
            .all(|job| job.window_count <= MAX_MODEL_JOB_WINDOWS));
    }
}
