use std::collections::{BTreeMap, BTreeSet};

use phoenix_alex::{
    api as alex_api, split_sentence_ranges as alex_sentence_ranges, AlexError, Lexicon,
};
use phoenix_semantic_v2::{
    scope_storage_key, DirtyScopeRecord, DocumentArchive, DocumentRevisionRef, ErScopePatchSidecar,
    RelationDecisionOutcome, RelationDecisionRecord, RelationEdgeAddition, RelationJudgmentKind,
    RelationJudgmentRecord, RelationMentionSeedScopeSidecar, RelationScopePatchSidecar,
    ScopeLexSidecar, ScopeOrd, SemanticEntityRecord, SemanticRelationRecord, SessionArchive,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixRelationMentionSeedStore,
    PhoenixRelationPatchStore, StoreError,
};
use phoenix_types::{
    EntityId, EntityKind, KnownMatchSource, LexiconEntry, MentionEntityRef, RelationCandidate,
    ScopeKey, SessionId, TextRange,
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::gliner_seed::RelationMentionSeeder;
use crate::glirel::{
    extract_heuristic_relations, seed_relation_pairs, GlirelEntity, GlirelModel,
    GlirelProposalConfig, GlirelRelationPrediction, GlirelRelationTypeSpec,
};
use crate::nli::NliModel;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationWindowEntity {
    pub entity_id: EntityId,
    pub surface: String,
    pub kind: Option<EntityKind>,
    pub entity_type: String,
    pub span_start: usize,
    pub span_end: usize,
    pub sentence_index: usize,
    pub mention_index: Option<usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationWindowRecord {
    pub window_id: String,
    pub document_id: String,
    pub revision: u64,
    pub window_index: usize,
    pub range: TextRange,
    #[serde(default)]
    pub sentence_indices: Vec<usize>,
    #[serde(default)]
    pub chunk_ids: Vec<String>,
    #[serde(default)]
    pub candidate_relation_types: Vec<String>,
    #[serde(default)]
    pub evidence_labels: Vec<String>,
    pub text: String,
    #[serde(default)]
    pub entities: Vec<RelationWindowEntity>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationEntityProfile {
    pub entity_id: EntityId,
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub canonical_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub kind: Option<EntityKind>,
    pub mention_count: usize,
    #[serde(default)]
    pub document_ids: Vec<String>,
    #[serde(default)]
    pub chunk_ids: Vec<String>,
    pub continuity_score_millis: i32,
    pub serialized: String,
    #[serde(default)]
    pub blocking_keys: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationReviewCase {
    pub case_id: String,
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub document_id: String,
    pub revision: u64,
    pub window_id: String,
    pub window_index: usize,
    pub window_range: TextRange,
    #[serde(default)]
    pub sentence_indices: Vec<usize>,
    #[serde(default)]
    pub chunk_ids: Vec<String>,
    pub window_text: String,
    pub source_entity_id: EntityId,
    pub target_entity_id: EntityId,
    pub source_name: String,
    pub target_name: String,
    pub source_kind: Option<EntityKind>,
    pub target_kind: Option<EntityKind>,
    pub seed_score_millis: i32,
    #[serde(default)]
    pub seed_evidence: Vec<String>,
    pub serialized: String,
    #[serde(default)]
    pub blocking_keys: Vec<String>,
    #[serde(default)]
    pub glirel_predictions: Vec<GlirelRelationPrediction>,
    #[serde(default)]
    pub accepted_relations: Vec<SemanticRelationRecord>,
    pub decision_status: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationScopeReviewBatch {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub dirty: Option<DirtyScopeRecord>,
    #[serde(default)]
    pub document_refs: Vec<DocumentRevisionRef>,
    #[serde(default)]
    pub windows: Vec<RelationWindowRecord>,
    #[serde(default)]
    pub review_cases: Vec<RelationReviewCase>,
    #[serde(default)]
    pub entity_profiles: Vec<RelationEntityProfile>,
    #[serde(default)]
    pub persisted_relations: Vec<SemanticRelationRecord>,
    pub lexical_generation: Option<u64>,
    pub er_generation: Option<u64>,
    pub relation_generation: Option<u64>,
    #[serde(default)]
    pub window_build_stats: RelationWindowBuildStats,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationWindowBuildStats {
    #[serde(default)]
    pub window_source_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub anchor_evidence_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub families_per_window: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    pub rejected_window_reason_counts: BTreeMap<String, usize>,
    pub seeded_pair_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RelationSyntheticSentence {
    pub(crate) index: usize,
    pub(crate) chunk_id: String,
    pub(crate) range: TextRange,
    pub(crate) text: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum RelationDecisionKind {
    Accept,
    #[default]
    Defer,
    Reject,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationDecision {
    pub case_id: String,
    pub kind: RelationDecisionKind,
    pub edge_type: Option<String>,
    pub score_millis: i32,
    pub rationale: String,
    #[serde(default)]
    pub evidence: Vec<String>,
    pub source_entity_id: Option<EntityId>,
    pub target_entity_id: Option<EntityId>,
    pub support_confidence_millis: Option<u32>,
    pub contradiction_confidence_millis: Option<u32>,
}

pub fn default_relation_type_specs() -> Vec<GlirelRelationTypeSpec> {
    vec![
        GlirelRelationTypeSpec {
            label: "located_in".to_owned(),
            head_types: vec![
                "Character".to_owned(),
                "Npc".to_owned(),
                "Organization".to_owned(),
                "Faction".to_owned(),
            ],
            tail_types: vec!["Location".to_owned()],
            cue_phrases: vec![
                " in ".to_owned(),
                " at ".to_owned(),
                "near".to_owned(),
                "inside".to_owned(),
                "within".to_owned(),
            ],
            conflicts_with: Vec::new(),
            priority_millis: 90,
            accept_threshold_millis: 670,
            review_threshold_millis: 450,
            max_predictions_per_window: 1,
            directed: true,
        },
        GlirelRelationTypeSpec {
            label: "works_for".to_owned(),
            head_types: vec!["Character".to_owned(), "Npc".to_owned()],
            tail_types: vec!["Organization".to_owned(), "Faction".to_owned()],
            cue_phrases: vec![
                "works for".to_owned(),
                "worked for".to_owned(),
                "working for".to_owned(),
                "joined".to_owned(),
                "joins".to_owned(),
                "served".to_owned(),
                "serves".to_owned(),
                "employed by".to_owned(),
            ],
            conflicts_with: vec!["member_of".to_owned()],
            priority_millis: 120,
            accept_threshold_millis: 500,
            review_threshold_millis: 420,
            max_predictions_per_window: 1,
            directed: true,
        },
        GlirelRelationTypeSpec {
            label: "member_of".to_owned(),
            head_types: vec!["Character".to_owned(), "Npc".to_owned()],
            tail_types: vec!["Faction".to_owned(), "Organization".to_owned()],
            cue_phrases: vec![
                "member of".to_owned(),
                "part of".to_owned(),
                "belongs to".to_owned(),
                "under".to_owned(),
            ],
            conflicts_with: vec!["works_for".to_owned()],
            priority_millis: 100,
            accept_threshold_millis: 640,
            review_threshold_millis: 440,
            max_predictions_per_window: 1,
            directed: true,
        },
        GlirelRelationTypeSpec {
            label: "commands".to_owned(),
            head_types: vec![
                "Character".to_owned(),
                "Npc".to_owned(),
                "Organization".to_owned(),
                "Faction".to_owned(),
            ],
            tail_types: vec![
                "Character".to_owned(),
                "Npc".to_owned(),
                "Organization".to_owned(),
                "Faction".to_owned(),
            ],
            cue_phrases: vec![
                "commanded".to_owned(),
                "commands".to_owned(),
                "led".to_owned(),
                "orders".to_owned(),
            ],
            conflicts_with: Vec::new(),
            priority_millis: 80,
            accept_threshold_millis: 620,
            review_threshold_millis: 450,
            max_predictions_per_window: 1,
            directed: true,
        },
        GlirelRelationTypeSpec {
            label: "protects".to_owned(),
            head_types: vec![
                "Character".to_owned(),
                "Npc".to_owned(),
                "Organization".to_owned(),
                "Faction".to_owned(),
            ],
            tail_types: vec![
                "Character".to_owned(),
                "Npc".to_owned(),
                "Organization".to_owned(),
                "Faction".to_owned(),
                "Location".to_owned(),
            ],
            cue_phrases: vec![
                "protected".to_owned(),
                "protects".to_owned(),
                "defended".to_owned(),
                "guards".to_owned(),
            ],
            conflicts_with: Vec::new(),
            priority_millis: 70,
            accept_threshold_millis: 640,
            review_threshold_millis: 460,
            max_predictions_per_window: 1,
            directed: true,
        },
    ]
}

#[derive(Debug, thiserror::Error)]
pub enum GlirelWorkerError {
    #[error(transparent)]
    Model(#[from] crate::GlirelError),
    #[error(transparent)]
    Nli(#[from] crate::NliError),
    #[error(transparent)]
    Alex(#[from] AlexError),
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("{0}")]
    Seeder(String),
}

pub fn derive_scope_review_batch(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    sidecar: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
) -> RelationScopeReviewBatch {
    derive_scope_review_batch_internal(
        archives,
        session,
        dirty,
        sidecar,
        er_sidecar,
        relation_sidecar,
        None,
        None,
    )
    .expect("seedless relation batch derivation cannot fail")
}

pub fn derive_scope_review_batch_with_seeder(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    sidecar: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
    relation_seed_sidecar: Option<&RelationMentionSeedScopeSidecar>,
    mention_seeder: Option<&RelationMentionSeeder>,
) -> Result<RelationScopeReviewBatch, GlirelWorkerError> {
    derive_scope_review_batch_internal(
        archives,
        session,
        dirty,
        sidecar,
        er_sidecar,
        relation_sidecar,
        relation_seed_sidecar,
        mention_seeder,
    )
}

fn derive_scope_review_batch_internal(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    sidecar: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    relation_sidecar: Option<&RelationScopePatchSidecar>,
    relation_seed_sidecar: Option<&RelationMentionSeedScopeSidecar>,
    mention_seeder: Option<&RelationMentionSeeder>,
) -> Result<RelationScopeReviewBatch, GlirelWorkerError> {
    let scope = archives
        .first()
        .map(|archive| archive.manifest.scope.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope.clone()))
        .or_else(|| sidecar.as_ref().map(|value| value.scope.clone()))
        .unwrap_or_default();
    let scope_key = archives
        .first()
        .map(|archive| archive.manifest.scope_key.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope_key.clone()))
        .or_else(|| sidecar.as_ref().map(|value| value.scope_key.clone()))
        .unwrap_or_default();
    let scope_ord = archives
        .first()
        .map(|archive| archive.manifest.scope_ord)
        .or_else(|| dirty.as_ref().map(|record| record.scope_ord))
        .or_else(|| sidecar.as_ref().and_then(|value| value.scope_ord))
        .unwrap_or_default();
    let session_id = archives
        .iter()
        .find_map(|archive| archive.manifest.session_id.clone())
        .or_else(|| session.map(|value| value.session_id.clone()));
    let document_refs = session
        .map(|value| {
            value
                .document_refs
                .iter()
                .filter(|reference| scope_storage_key(&reference.scope) == scope_key)
                .cloned()
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    let persisted_relations = build_persisted_relations(archives);
    let entity_profiles = build_entity_profiles(archives, sidecar, er_sidecar, session);
    let (windows, mut window_build_stats) = build_windows(
        archives,
        sidecar,
        er_sidecar,
        &persisted_relations,
        &entity_profiles,
        relation_seed_sidecar,
        mention_seeder,
    )?;
    let review_cases = build_review_cases(
        &scope,
        &scope_key,
        scope_ord,
        session_id.clone(),
        &windows,
        &entity_profiles,
    );
    window_build_stats.seeded_pair_count = review_cases.len();

    let mut batch = RelationScopeReviewBatch {
        scope,
        scope_key,
        scope_ord,
        session_id,
        dirty: dirty.cloned(),
        document_refs,
        windows,
        review_cases,
        entity_profiles,
        persisted_relations,
        lexical_generation: sidecar.map(|value| value.generation),
        er_generation: er_sidecar.map(|value| value.generation),
        relation_generation: relation_sidecar.map(|value| value.generation),
        window_build_stats,
    };
    if let Some(sidecar) = relation_sidecar {
        apply_relation_patch_sidecar(&mut batch, sidecar);
    }
    Ok(batch)
}

pub fn derive_scope_review_batch_from_store<S>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
) -> Result<RelationScopeReviewBatch, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixRelationPatchStore,
{
    let archives = store.load_latest_document_archives(Some(&dirty.scope))?;
    let sidecar = store.load_scope_sidecar(&dirty.scope)?;
    let er_sidecar = store.load_er_patch_sidecar(&dirty.scope)?;
    let relation_sidecar = store.load_relation_patch_sidecar(&dirty.scope)?;
    derive_scope_review_batch_internal(
        &archives,
        session,
        Some(dirty),
        sidecar.as_ref(),
        er_sidecar.as_ref(),
        relation_sidecar.as_ref(),
        None,
        None,
    )
    .map_err(|error| match error {
        GlirelWorkerError::Store(store_error) => store_error,
        other => StoreError::Query(format!("relation batch derivation failed: {other}")),
    })
}

pub fn derive_scope_review_batch_from_store_with_seeder<S>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
    mention_seeder: Option<&RelationMentionSeeder>,
) -> Result<RelationScopeReviewBatch, GlirelWorkerError>
where
    S: PhoenixArchiveStoreV2
        + PhoenixErPatchStore
        + PhoenixRelationMentionSeedStore
        + PhoenixRelationPatchStore,
{
    let archives = store.load_latest_document_archives(Some(&dirty.scope))?;
    let sidecar = store.load_scope_sidecar(&dirty.scope)?;
    let er_sidecar = store.load_er_patch_sidecar(&dirty.scope)?;
    let relation_seed_sidecar = store.load_relation_mention_seed_sidecar(&dirty.scope)?;
    let relation_sidecar = store.load_relation_patch_sidecar(&dirty.scope)?;
    derive_scope_review_batch_with_seeder(
        &archives,
        session,
        Some(dirty),
        sidecar.as_ref(),
        er_sidecar.as_ref(),
        relation_sidecar.as_ref(),
        relation_seed_sidecar.as_ref(),
        mention_seeder,
    )
}

pub fn derive_dirty_scope_review_batches<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<RelationScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixRelationPatchStore,
{
    let session = match session_id {
        Some(value) => store.load_latest_session_archive(value)?,
        None => None,
    };
    let mut dirty = store.list_dirty_scopes()?;
    dirty.sort_by(|left, right| left.scope_key.cmp(&right.scope_key));
    dirty
        .into_iter()
        .map(|record| derive_scope_review_batch_from_store(store, &record, session.as_ref()))
        .collect()
}

pub fn derive_dirty_scope_review_batches_with_seeder<S>(
    store: &S,
    session_id: Option<&SessionId>,
    mention_seeder: Option<&RelationMentionSeeder>,
) -> Result<Vec<RelationScopeReviewBatch>, GlirelWorkerError>
where
    S: PhoenixArchiveStoreV2
        + PhoenixErPatchStore
        + PhoenixRelationMentionSeedStore
        + PhoenixRelationPatchStore,
{
    let session = match session_id {
        Some(value) => store.load_latest_session_archive(value)?,
        None => None,
    };
    let mut dirty = store.list_dirty_scopes()?;
    dirty.sort_by(|left, right| left.scope_key.cmp(&right.scope_key));
    dirty
        .into_iter()
        .map(|record| {
            derive_scope_review_batch_from_store_with_seeder(
                store,
                &record,
                session.as_ref(),
                mention_seeder,
            )
        })
        .collect()
}

pub fn run_glirel_over_batch(
    batch: &mut RelationScopeReviewBatch,
    model: &GlirelModel,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Result<(), GlirelWorkerError> {
    if relation_specs.is_empty() {
        return Ok(());
    }

    let mut case_indices_by_window = BTreeMap::<String, Vec<usize>>::new();
    for (index, case) in batch.review_cases.iter().enumerate() {
        case_indices_by_window
            .entry(case.window_id.clone())
            .or_default()
            .push(index);
    }

    let min_threshold = relation_specs
        .iter()
        .map(|spec| spec.review_threshold_millis)
        .min()
        .unwrap_or(450)
        .min(300) as f32
        / 1000.0;

    let windows_by_id = batch
        .windows
        .iter()
        .map(|window| (window.window_id.as_str(), window))
        .collect::<FxHashMap<_, _>>();

    for (window_id, case_indices) in case_indices_by_window {
        let Some(window) = windows_by_id.get(window_id.as_str()) else {
            continue;
        };
        if window.entities.len() < 2 {
            continue;
        }
        let entities = window
            .entities
            .iter()
            .map(|entity| GlirelEntity {
                text: entity.surface.clone(),
                entity_type: entity.entity_type.clone(),
                span_start: entity
                    .span_start
                    .saturating_sub(window.range.start as usize),
                span_end: entity.span_end.saturating_sub(window.range.start as usize),
                entity_id: Some(entity.entity_id.0.clone()),
            })
            .collect::<Vec<_>>();
        let predictions = model
            .extract_with_schema(&window.text, &entities, relation_specs, min_threshold)
            .map_err(GlirelWorkerError::Model)?;
        let filtered_predictions = filter_relation_predictions(
            &window.text,
            &window.entities,
            window.range.start as usize,
            relation_specs,
            predictions,
        );

        let predictions_by_pair = filtered_predictions.into_iter().fold(
            FxHashMap::<(String, String), Vec<GlirelRelationPrediction>>::default(),
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

        for case_index in case_indices {
            let case = &mut batch.review_cases[case_index];
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

    Ok(())
}

pub fn run_primary_relation_lane(
    batch: &mut RelationScopeReviewBatch,
    model: Option<&GlirelModel>,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Result<(), GlirelWorkerError> {
    if let Some(model) = model {
        return run_glirel_over_batch(batch, model, relation_specs);
    }

    let windows_by_id = batch
        .windows
        .iter()
        .map(|window| (window.window_id.as_str(), window))
        .collect::<FxHashMap<_, _>>();
    let mut case_indices_by_window = BTreeMap::<String, Vec<usize>>::new();
    for (index, case) in batch.review_cases.iter().enumerate() {
        case_indices_by_window
            .entry(case.window_id.clone())
            .or_default()
            .push(index);
    }

    for (window_id, case_indices) in case_indices_by_window {
        let Some(window) = windows_by_id.get(window_id.as_str()) else {
            continue;
        };
        let entities = window
            .entities
            .iter()
            .map(|entity| GlirelEntity {
                text: entity.surface.clone(),
                entity_type: entity.entity_type.clone(),
                span_start: entity
                    .span_start
                    .saturating_sub(window.range.start as usize),
                span_end: entity.span_end.saturating_sub(window.range.start as usize),
                entity_id: Some(entity.entity_id.0.clone()),
            })
            .collect::<Vec<_>>();
        let predictions = extract_heuristic_relations(
            &window.text,
            &entities,
            relation_specs,
            &GlirelProposalConfig::default(),
        );
        let filtered_predictions = filter_relation_predictions(
            &window.text,
            &window.entities,
            window.range.start as usize,
            relation_specs,
            predictions,
        );
        let predictions_by_pair = filtered_predictions.into_iter().fold(
            FxHashMap::<(String, String), Vec<GlirelRelationPrediction>>::default(),
            |mut acc, mut prediction| {
                prediction
                    .evidence
                    .push("proposal_engine:heuristic".to_owned());
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

        for case_index in case_indices {
            let case = &mut batch.review_cases[case_index];
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

    Ok(())
}

pub fn draft_relation_decisions(
    batch: &RelationScopeReviewBatch,
    relation_specs: &[GlirelRelationTypeSpec],
) -> Vec<RelationDecision> {
    let spec_by_label = relation_specs
        .iter()
        .map(|spec| (spec.label.as_str(), spec))
        .collect::<FxHashMap<_, _>>();

    batch.review_cases
        .iter()
        .map(|case| {
            match case
                .glirel_predictions
                .first()
                .and_then(|prediction| spec_by_label.get(prediction.relation.as_str()).map(|spec| (prediction, *spec)))
            {
                Some((prediction, spec)) => {
                    let score_millis = (prediction.confidence * 1000.0).round() as i32;
                    if score_millis >= spec.accept_threshold_millis as i32 {
                        RelationDecision {
                            case_id: case.case_id.clone(),
                            kind: RelationDecisionKind::Accept,
                            edge_type: Some(prediction.relation.clone()),
                            score_millis,
                            rationale: "glirel relation score cleared the family acceptance threshold".to_owned(),
                            evidence: prediction.evidence.clone(),
                            source_entity_id: Some(case.source_entity_id.clone()),
                            target_entity_id: Some(case.target_entity_id.clone()),
                            support_confidence_millis: None,
                            contradiction_confidence_millis: None,
                        }
                    } else if score_millis >= spec.review_threshold_millis as i32 {
                        RelationDecision {
                            case_id: case.case_id.clone(),
                            kind: RelationDecisionKind::Defer,
                            edge_type: Some(prediction.relation.clone()),
                            score_millis,
                            rationale: "glirel relation score was promising but below the family acceptance threshold".to_owned(),
                            evidence: prediction.evidence.clone(),
                            source_entity_id: Some(case.source_entity_id.clone()),
                            target_entity_id: Some(case.target_entity_id.clone()),
                            support_confidence_millis: None,
                            contradiction_confidence_millis: None,
                        }
                    } else {
                        RelationDecision {
                            case_id: case.case_id.clone(),
                            kind: RelationDecisionKind::Reject,
                            edge_type: Some(prediction.relation.clone()),
                            score_millis,
                            rationale: "glirel relation score stayed below the review threshold".to_owned(),
                            evidence: prediction.evidence.clone(),
                            source_entity_id: Some(case.source_entity_id.clone()),
                            target_entity_id: Some(case.target_entity_id.clone()),
                            support_confidence_millis: None,
                            contradiction_confidence_millis: None,
                        }
                    }
                }
                None => RelationDecision {
                    case_id: case.case_id.clone(),
                    kind: RelationDecisionKind::Reject,
                    edge_type: None,
                    score_millis: 0,
                    rationale: "no glirel relation proposal was produced for this windowed entity pair".to_owned(),
                    evidence: Vec::new(),
                    source_entity_id: Some(case.source_entity_id.clone()),
                    target_entity_id: Some(case.target_entity_id.clone()),
                    support_confidence_millis: None,
                    contradiction_confidence_millis: None,
                },
            }
        })
        .collect()
}

pub fn adjudicate_relation_decisions_with_nli(
    batch: &RelationScopeReviewBatch,
    decisions: &[RelationDecision],
    relation_specs: &[GlirelRelationTypeSpec],
    nli: &NliModel,
) -> Result<Vec<RelationDecision>, GlirelWorkerError> {
    let spec_by_label = relation_specs
        .iter()
        .map(|spec| (spec.label.as_str(), spec))
        .collect::<FxHashMap<_, _>>();
    let case_by_id = batch
        .review_cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect::<FxHashMap<_, _>>();
    let mut adjudicated = Vec::with_capacity(decisions.len());
    for decision in decisions {
        let Some(case) = case_by_id.get(decision.case_id.as_str()) else {
            adjudicated.push(decision.clone());
            continue;
        };
        let Some(edge_type) = decision.edge_type.as_deref() else {
            adjudicated.push(decision.clone());
            continue;
        };
        let Some(spec) = spec_by_label.get(edge_type) else {
            adjudicated.push(decision.clone());
            continue;
        };
        if decision.kind == RelationDecisionKind::Reject && decision.score_millis <= 0 {
            adjudicated.push(decision.clone());
            continue;
        }
        let forward_templates =
            build_relation_hypotheses(edge_type, &case.source_name, &case.target_name);
        let reverse_templates = if spec.directed {
            build_relation_hypotheses(edge_type, &case.target_name, &case.source_name)
        } else {
            Vec::new()
        };
        let judgment =
            nli.judge_relation(&case.window_text, &forward_templates, &reverse_templates)?;
        let chosen_scores = if judgment.used_reverse {
            judgment.reverse.unwrap_or(judgment.forward)
        } else {
            judgment.forward
        };
        let support_threshold = nli_support_threshold_millis(spec);
        let contradiction_threshold = nli_contradiction_threshold_millis(spec);
        let support_review_threshold = nli_support_review_threshold_millis(spec);
        let entailment_millis = (chosen_scores.entailment * 1000.0).round() as i32;
        let contradiction_millis = (chosen_scores.contradiction * 1000.0).round() as i32;
        let mut next = decision.clone();
        next.evidence.push(format!(
            "nli:entailment={:.3};contradiction={:.3};neutral={:.3};reverse={}",
            chosen_scores.entailment,
            chosen_scores.contradiction,
            chosen_scores.neutral,
            judgment.used_reverse
        ));
        next.evidence
            .push(format!("nli:hypothesis={}", judgment.best_hypothesis));
        if judgment.used_reverse && spec.directed {
            next.source_entity_id = Some(case.target_entity_id.clone());
            next.target_entity_id = Some(case.source_entity_id.clone());
            next.evidence.push("nli:direction=reverse".to_owned());
        }
        if contradiction_millis >= contradiction_threshold as i32
            && contradiction_millis >= entailment_millis + 60
        {
            next.kind = RelationDecisionKind::Reject;
            next.score_millis = contradiction_millis;
            next.rationale =
                "nli contradiction score overruled the extracted relation candidate".to_owned();
            next.support_confidence_millis = None;
            next.contradiction_confidence_millis = Some(contradiction_millis.max(0) as u32);
            adjudicated.push(next);
            continue;
        }
        if entailment_millis >= support_threshold as i32 {
            next.support_confidence_millis = Some(entailment_millis.max(0) as u32);
            next.contradiction_confidence_millis = None;
            if next.kind != RelationDecisionKind::Accept {
                next.kind = RelationDecisionKind::Accept;
                next.score_millis = next.score_millis.max(entailment_millis);
                next.rationale =
                    "nli support score confirmed the relation strongly enough to accept it"
                        .to_owned();
            } else {
                next.score_millis = next.score_millis.max(entailment_millis);
                next.rationale =
                    "glirel extraction was confirmed by nli support evidence".to_owned();
            }
            adjudicated.push(next);
            continue;
        }
        if entailment_millis >= support_review_threshold as i32 {
            next.kind = RelationDecisionKind::Defer;
            next.score_millis = next.score_millis.max(entailment_millis);
            next.support_confidence_millis = Some(entailment_millis.max(0) as u32);
            next.contradiction_confidence_millis = None;
            next.rationale =
                "nli found supporting evidence, but not enough to finalize the relation yet"
                    .to_owned();
            adjudicated.push(next);
            continue;
        }
        if next.kind == RelationDecisionKind::Accept {
            next.kind = RelationDecisionKind::Defer;
            next.rationale =
                "nli did not find enough support to finalize an extracted relation".to_owned();
        } else if next.kind == RelationDecisionKind::Defer {
            next.rationale =
                "nli kept the relation in review because support stayed below the accept bar"
                    .to_owned();
        }
        next.support_confidence_millis = None;
        next.contradiction_confidence_millis = None;
        adjudicated.push(next);
    }
    Ok(adjudicated)
}

pub fn build_relation_hypotheses(edge_type: &str, head: &str, tail: &str) -> Vec<String> {
    let templates = match edge_type {
        "works_for" => &[
            "{head} works for {tail}.",
            "{head} is employed by {tail}.",
            "{head} serves {tail}.",
        ][..],
        "member_of" => &[
            "{head} is a member of {tail}.",
            "{head} belongs to {tail}.",
            "{head} is part of {tail}.",
        ][..],
        "commands" => &[
            "{head} commands {tail}.",
            "{head} leads {tail}.",
            "{head} is in charge of {tail}.",
        ][..],
        "protects" => &[
            "{head} protects {tail}.",
            "{head} defends {tail}.",
            "{head} guards {tail}.",
        ][..],
        "located_in" => &[
            "{head} is located in {tail}.",
            "{head} is in {tail}.",
            "{head} is based in {tail}.",
        ][..],
        _ => &["{head} is related to {tail}."][..],
    };
    templates
        .iter()
        .map(|template| template.replace("{head}", head).replace("{tail}", tail))
        .collect()
}

fn nli_support_threshold_millis(spec: &GlirelRelationTypeSpec) -> u32 {
    spec.accept_threshold_millis.max(720)
}

fn nli_support_review_threshold_millis(spec: &GlirelRelationTypeSpec) -> u32 {
    spec.review_threshold_millis.max(560)
}

fn nli_contradiction_threshold_millis(spec: &GlirelRelationTypeSpec) -> u32 {
    spec.accept_threshold_millis.max(740)
}

pub fn build_relation_patch_sidecar(
    batch: &RelationScopeReviewBatch,
    decisions: &[RelationDecision],
    created_at: i64,
) -> RelationScopePatchSidecar {
    let case_by_id = batch
        .review_cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect::<FxHashMap<_, _>>();
    let mut sidecar = RelationScopePatchSidecar {
        scope: batch.scope.clone(),
        scope_key: batch.scope_key.clone(),
        scope_ord: Some(batch.scope_ord),
        session_id: batch.session_id.clone(),
        updated_at: created_at,
        generation: created_at as u64,
        ..Default::default()
    };

    for decision in decisions {
        let Some(case) = case_by_id.get(decision.case_id.as_str()) else {
            continue;
        };
        let source_entity_id = decision
            .source_entity_id
            .clone()
            .unwrap_or_else(|| case.source_entity_id.clone());
        let target_entity_id = decision
            .target_entity_id
            .clone()
            .unwrap_or_else(|| case.target_entity_id.clone());
        sidecar.decisions.push(RelationDecisionRecord {
            case_id: case.case_id.clone(),
            document_id: case.document_id.clone(),
            window_id: case.window_id.clone(),
            outcome: relation_outcome_from_kind(&decision.kind),
            source_entity_id: Some(source_entity_id.clone()),
            target_entity_id: Some(target_entity_id.clone()),
            edge_type: decision.edge_type.clone(),
            score_millis: decision.score_millis,
            rationale: decision.rationale.clone(),
            evidence: decision.evidence.clone(),
            reviewed_at: created_at,
        });
        if let (Some(edge_type), Some(confidence_millis)) = (
            decision.edge_type.clone(),
            decision.support_confidence_millis,
        ) {
            sidecar.support_judgments.push(RelationJudgmentRecord {
                case_id: case.case_id.clone(),
                document_id: case.document_id.clone(),
                window_id: case.window_id.clone(),
                source_entity_id: source_entity_id.clone(),
                target_entity_id: target_entity_id.clone(),
                edge_type,
                kind: RelationJudgmentKind::Support,
                confidence_millis,
                evidence_refs: decision.evidence.clone(),
                created_at,
            });
        }
        if let (Some(edge_type), Some(confidence_millis)) = (
            decision.edge_type.clone(),
            decision.contradiction_confidence_millis,
        ) {
            sidecar
                .contradiction_judgments
                .push(RelationJudgmentRecord {
                    case_id: case.case_id.clone(),
                    document_id: case.document_id.clone(),
                    window_id: case.window_id.clone(),
                    source_entity_id: source_entity_id.clone(),
                    target_entity_id: target_entity_id.clone(),
                    edge_type,
                    kind: RelationJudgmentKind::Contradict,
                    confidence_millis,
                    evidence_refs: decision.evidence.clone(),
                    created_at,
                });
        }

        if decision.kind == RelationDecisionKind::Accept {
            if let Some(edge_type) = decision.edge_type.clone() {
                sidecar.edge_additions.push(RelationEdgeAddition {
                    case_id: case.case_id.clone(),
                    document_id: case.document_id.clone(),
                    window_id: case.window_id.clone(),
                    source_entity_id,
                    target_entity_id,
                    edge_type,
                    confidence_millis: decision.score_millis.max(0) as u32,
                    evidence_refs: decision.evidence.clone(),
                    created_at,
                });
            }
        }
    }

    dedupe_relation_patch_sidecar(&mut sidecar);
    sidecar
}

pub fn persist_relation_patch_sidecar<S>(
    store: &S,
    batch: &RelationScopeReviewBatch,
    decisions: &[RelationDecision],
    created_at: i64,
) -> Result<RelationScopePatchSidecar, StoreError>
where
    S: PhoenixRelationPatchStore,
{
    let updates = build_relation_patch_sidecar(batch, decisions, created_at);
    let merged = match store.load_relation_patch_sidecar(&batch.scope)? {
        Some(existing) => merge_relation_patch_sidecars(existing, updates),
        None => updates,
    };
    store.persist_relation_patch_sidecar(&merged)?;
    Ok(merged)
}

pub fn apply_relation_patch_sidecar(
    batch: &mut RelationScopeReviewBatch,
    sidecar: &RelationScopePatchSidecar,
) {
    let support_by_case = sidecar
        .support_judgments
        .iter()
        .map(|judgment| (judgment.case_id.as_str(), judgment))
        .collect::<FxHashMap<_, _>>();
    let contradiction_by_case = sidecar
        .contradiction_judgments
        .iter()
        .map(|judgment| (judgment.case_id.as_str(), judgment))
        .collect::<FxHashMap<_, _>>();
    let decision_by_case = sidecar
        .decisions
        .iter()
        .map(|decision| (decision.case_id.as_str(), decision))
        .collect::<FxHashMap<_, _>>();
    for case in &mut batch.review_cases {
        if let Some(decision) = decision_by_case.get(case.case_id.as_str()) {
            if decision.outcome == RelationDecisionOutcome::Accept {
                case.decision_status = decision
                    .edge_type
                    .as_deref()
                    .map(|edge_type| format!("relation_accept:{edge_type}"))
                    .unwrap_or_else(|| "relation_accept".to_owned());
                continue;
            }
        }
        if contradiction_by_case.contains_key(case.case_id.as_str()) {
            case.decision_status = "relation_contradict".to_owned();
        } else if support_by_case.contains_key(case.case_id.as_str()) {
            case.decision_status = "relation_support".to_owned();
        } else if let Some(decision) = decision_by_case.get(case.case_id.as_str()) {
            case.decision_status = match decision.outcome {
                RelationDecisionOutcome::Accept => decision
                    .edge_type
                    .as_deref()
                    .map(|edge_type| format!("relation_accept:{edge_type}"))
                    .unwrap_or_else(|| "relation_accept".to_owned()),
                RelationDecisionOutcome::Support => "relation_support".to_owned(),
                RelationDecisionOutcome::Contradict => "relation_contradict".to_owned(),
                RelationDecisionOutcome::Defer => "relation_defer".to_owned(),
                RelationDecisionOutcome::Reject => "relation_reject".to_owned(),
            };
        }
    }

    for edge in &sidecar.edge_additions {
        let relation = SemanticRelationRecord {
            source_entity_id: edge.source_entity_id.clone(),
            target_entity_id: edge.target_entity_id.clone(),
            edge_type: edge.edge_type.clone(),
            sentence_index: batch
                .review_cases
                .iter()
                .find(|case| case.case_id == edge.case_id)
                .and_then(|case| case.sentence_indices.first().copied())
                .unwrap_or_default(),
            chunk_id: batch
                .review_cases
                .iter()
                .find(|case| case.case_id == edge.case_id)
                .and_then(|case| case.chunk_ids.first().cloned()),
        };
        if !batch.persisted_relations.iter().any(|existing| {
            existing.source_entity_id == relation.source_entity_id
                && existing.target_entity_id == relation.target_entity_id
                && existing.edge_type == relation.edge_type
                && existing.chunk_id == relation.chunk_id
        }) {
            batch.persisted_relations.push(relation.clone());
        }
        if let Some(case) = batch
            .review_cases
            .iter_mut()
            .find(|case| case.case_id == edge.case_id)
        {
            if !case.accepted_relations.iter().any(|existing| {
                existing.source_entity_id == relation.source_entity_id
                    && existing.target_entity_id == relation.target_entity_id
                    && existing.edge_type == relation.edge_type
            }) {
                case.accepted_relations.push(relation);
            }
        }
    }

    batch.relation_generation = Some(sidecar.generation);
}

fn relation_outcome_from_kind(kind: &RelationDecisionKind) -> RelationDecisionOutcome {
    match kind {
        RelationDecisionKind::Accept => RelationDecisionOutcome::Accept,
        RelationDecisionKind::Defer => RelationDecisionOutcome::Defer,
        RelationDecisionKind::Reject => RelationDecisionOutcome::Reject,
    }
}

fn merge_relation_patch_sidecars(
    mut existing: RelationScopePatchSidecar,
    updates: RelationScopePatchSidecar,
) -> RelationScopePatchSidecar {
    existing.updated_at = existing.updated_at.max(updates.updated_at);
    existing.generation = existing.generation.max(updates.generation);
    existing.edge_additions.extend(updates.edge_additions);
    existing.support_judgments.extend(updates.support_judgments);
    existing
        .contradiction_judgments
        .extend(updates.contradiction_judgments);
    existing.decisions.extend(updates.decisions);
    dedupe_relation_patch_sidecar(&mut existing);
    existing
}

fn dedupe_relation_patch_sidecar(sidecar: &mut RelationScopePatchSidecar) {
    let mut edges = BTreeSet::new();
    sidecar.edge_additions.retain(|edge| {
        edges.insert((
            edge.case_id.clone(),
            edge.source_entity_id.0.clone(),
            edge.target_entity_id.0.clone(),
            edge.edge_type.clone(),
        ))
    });
    let mut support = BTreeSet::new();
    sidecar.support_judgments.retain(|judgment| {
        support.insert((
            judgment.case_id.clone(),
            judgment.source_entity_id.0.clone(),
            judgment.target_entity_id.0.clone(),
            judgment.edge_type.clone(),
            judgment.kind as u8,
        ))
    });
    let mut contradiction = BTreeSet::new();
    sidecar.contradiction_judgments.retain(|judgment| {
        contradiction.insert((
            judgment.case_id.clone(),
            judgment.source_entity_id.0.clone(),
            judgment.target_entity_id.0.clone(),
            judgment.edge_type.clone(),
            judgment.kind as u8,
        ))
    });
    let mut decisions = BTreeSet::new();
    sidecar.decisions.retain(|decision| {
        decisions.insert((
            decision.case_id.clone(),
            decision.outcome as u8,
            decision.edge_type.clone().unwrap_or_default(),
        ))
    });
}

fn build_entity_profiles(
    archives: &[DocumentArchive],
    sidecar: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    session: Option<&SessionArchive>,
) -> Vec<RelationEntityProfile> {
    let mut by_entity = FxHashMap::<String, RelationEntityProfile>::default();
    for archive in archives {
        for entity in &archive.entities {
            let entry = by_entity
                .entry(entity.entity_id.0.clone())
                .or_insert_with(|| entity_profile_from_record(archive, entity, session));
            merge_entity_profile(entry, archive, entity);
        }
    }
    if let Some(sidecar) = sidecar {
        for alias in &sidecar.alias_entries {
            for posting in &alias.postings {
                if let Some(profile) = by_entity.get_mut(&posting.entity_id) {
                    if !profile
                        .aliases
                        .iter()
                        .any(|value| value == &alias.normalized)
                    {
                        profile.aliases.push(alias.normalized.clone());
                    }
                }
            }
        }
    }
    if let Some(er_sidecar) = er_sidecar {
        for alias in &er_sidecar.alias_additions {
            if let Some(profile) = by_entity.get_mut(&alias.entity_id.0) {
                if !profile
                    .aliases
                    .iter()
                    .any(|value| value == &alias.alias_surface)
                {
                    profile.aliases.push(alias.alias_surface.clone());
                }
            }
        }
        for override_row in &er_sidecar.type_overrides {
            if let Some(profile) = by_entity.get_mut(&override_row.entity_id.0) {
                profile.kind = Some(override_row.kind.clone());
            }
        }
    }

    let mut rows = by_entity.into_values().collect::<Vec<_>>();
    rows.sort_by(|left, right| left.entity_id.0.cmp(&right.entity_id.0));
    rows
}

pub fn derive_relation_entity_profiles(
    archives: &[DocumentArchive],
    sidecar: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    session: Option<&SessionArchive>,
) -> Vec<RelationEntityProfile> {
    build_entity_profiles(archives, sidecar, er_sidecar, session)
}

fn entity_profile_from_record(
    archive: &DocumentArchive,
    entity: &SemanticEntityRecord,
    session: Option<&SessionArchive>,
) -> RelationEntityProfile {
    let continuity_score_millis = session
        .map(|session| {
            let seen = session
                .document_refs
                .iter()
                .filter(|reference| reference.document_id == archive.manifest.document_id)
                .count();
            (seen as i32 * 125).min(500)
        })
        .unwrap_or_default()
        + ((entity.mention_count.min(8) as i32) * 40);
    let mut aliases = entity.aliases.clone();
    aliases.sort();
    aliases.dedup();
    RelationEntityProfile {
        entity_id: entity.entity_id.clone(),
        scope: archive.manifest.scope.clone(),
        scope_key: archive.manifest.scope_key.clone(),
        scope_ord: archive.manifest.scope_ord,
        session_id: archive.manifest.session_id.clone(),
        canonical_name: entity.canonical_name.clone(),
        aliases,
        kind: entity.kind.clone(),
        mention_count: entity.mention_count,
        document_ids: vec![archive.manifest.document_id.clone()],
        chunk_ids: entity.chunk_ids.clone(),
        continuity_score_millis,
        serialized: serialize_profile(entity),
        blocking_keys: profile_blocking_keys(entity),
    }
}

fn merge_entity_profile(
    profile: &mut RelationEntityProfile,
    archive: &DocumentArchive,
    entity: &SemanticEntityRecord,
) {
    profile.mention_count = profile.mention_count.max(entity.mention_count);
    if !profile
        .document_ids
        .iter()
        .any(|document_id| document_id == &archive.manifest.document_id)
    {
        profile
            .document_ids
            .push(archive.manifest.document_id.clone());
        profile.continuity_score_millis += 90;
    }
    for alias in &entity.aliases {
        if !profile.aliases.iter().any(|value| value == alias) {
            profile.aliases.push(alias.clone());
        }
    }
    for chunk_id in &entity.chunk_ids {
        if !profile.chunk_ids.iter().any(|value| value == chunk_id) {
            profile.chunk_ids.push(chunk_id.clone());
        }
    }
    if profile.kind.is_none() {
        profile.kind = entity.kind.clone();
    }
    profile.serialized = serialize_profile_like(
        &profile.canonical_name,
        &profile.aliases,
        profile.kind.as_ref(),
        &profile.document_ids,
    );
}

fn build_windows(
    archives: &[DocumentArchive],
    _sidecar: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    persisted_relations: &[SemanticRelationRecord],
    profiles: &[RelationEntityProfile],
    relation_seed_sidecar: Option<&RelationMentionSeedScopeSidecar>,
    mention_seeder: Option<&RelationMentionSeeder>,
) -> Result<(Vec<RelationWindowRecord>, RelationWindowBuildStats), GlirelWorkerError> {
    let profile_by_entity = profiles
        .iter()
        .map(|profile| (profile.entity_id.0.clone(), profile))
        .collect::<FxHashMap<_, _>>();
    let er_view = ErPatchView::from_sidecar(er_sidecar);
    let continuity_hints = continuity_relation_map(persisted_relations);
    let alex_lexicon = build_relation_alex_lexicon(profiles)?;
    let mut windows = Vec::new();
    let mut stats = RelationWindowBuildStats::default();

    for archive in archives {
        let mut mentions = collect_alex_relation_mentions(archive, profiles, &alex_lexicon);
        if mentions.is_empty() && !archive.resolved_mentions.is_empty() {
            mentions = collect_relation_mentions(archive, &er_view);
        }
        let seeded_mentions = collect_relation_seed_mentions(archive, relation_seed_sidecar);
        if !seeded_mentions.is_empty() {
            mentions.extend(seeded_mentions);
            dedupe_relation_mentions(&mut mentions);
        }
        if let Some(seeder) = mention_seeder {
            if !archive.relations.is_empty() && (archive.sentences.is_empty() || mentions.len() < 2)
            {
                let seeded_mentions = collect_gliner_seed_mentions(archive, profiles, seeder)?;
                if !seeded_mentions.is_empty() {
                    for mention in &seeded_mentions {
                        let label = format!(
                            "gliner_seed:{}",
                            mention
                                .kind
                                .as_ref()
                                .map(|kind| relation_seed_kind_label(kind))
                                .unwrap_or("unknown")
                        );
                        record_stat(&mut stats.anchor_evidence_counts, label);
                    }
                    mentions.extend(seeded_mentions);
                    dedupe_relation_mentions(&mut mentions);
                }
            }
        }
        let before = windows.len();
        if archive.sentences.is_empty() {
            append_synthetic_sentence_windows(
                &mut windows,
                &mut stats,
                archive,
                profiles,
                &alex_lexicon,
                persisted_relations,
                &continuity_hints,
            );
            if windows.len() == before {
                append_clipped_chunk_windows(
                    &mut windows,
                    &mut stats,
                    archive,
                    profiles,
                    &alex_lexicon,
                    persisted_relations,
                    &continuity_hints,
                );
            }
            if windows.len() == before {
                append_relation_candidate_windows(
                    &mut windows,
                    &mut stats,
                    archive,
                    &mentions,
                    &profile_by_entity,
                    persisted_relations,
                    &continuity_hints,
                );
            }
            if windows.len() == before {
                append_archive_relation_windows(
                    &mut windows,
                    &mut stats,
                    archive,
                    &mentions,
                    &profile_by_entity,
                    persisted_relations,
                    &continuity_hints,
                );
            }
            continue;
        }

        for sentence_index in 0..archive.sentences.len() {
            let start_index = sentence_index.saturating_sub(1);
            let end_index = (sentence_index + 1).min(archive.sentences.len() - 1);
            let start = archive.sentences[start_index].range.start;
            let end = archive.sentences[end_index].range.end;
            let mut entities = mentions
                .iter()
                .filter(|mention| {
                    mention.sentence_index >= start_index && mention.sentence_index <= end_index
                })
                .cloned()
                .collect::<Vec<_>>();
            dedupe_window_entities(&mut entities, &profile_by_entity);
            if entities.len() < 2 {
                record_stat(
                    &mut stats.rejected_window_reason_counts,
                    "too_few_entities".to_owned(),
                );
                continue;
            }
            push_window(
                &mut windows,
                &mut stats,
                archive,
                format!(
                    "{}::{}::{}",
                    archive.manifest.document_id, archive.manifest.revision, sentence_index
                ),
                sentence_index,
                TextRange { start, end },
                (start_index..=end_index).collect(),
                entities,
                "archive_sentence",
                Vec::new(),
                &continuity_hints,
            );
        }
        if windows.len() == before {
            append_chunk_windows(
                &mut windows,
                &mut stats,
                archive,
                &mentions,
                &profile_by_entity,
                persisted_relations,
                &continuity_hints,
            );
        }
        if windows.len() == before {
            append_relation_candidate_windows(
                &mut windows,
                &mut stats,
                archive,
                &mentions,
                &profile_by_entity,
                persisted_relations,
                &continuity_hints,
            );
        }
        if windows.len() == before {
            append_archive_relation_windows(
                &mut windows,
                &mut stats,
                archive,
                &mentions,
                &profile_by_entity,
                persisted_relations,
                &continuity_hints,
            );
        }
    }

    Ok((windows, stats))
}

fn append_chunk_windows(
    windows: &mut Vec<RelationWindowRecord>,
    stats: &mut RelationWindowBuildStats,
    archive: &DocumentArchive,
    mentions: &[RelationMention],
    profile_by_entity: &FxHashMap<String, &RelationEntityProfile>,
    _persisted_relations: &[SemanticRelationRecord],
    continuity_hints: &FxHashMap<(String, String), BTreeSet<String>>,
) {
    for (index, chunk) in archive.chunks.iter().enumerate() {
        let mut entities = mentions
            .iter()
            .filter(|mention| {
                (mention.span_start as u32) < chunk.range.end
                    && (mention.span_end as u32) > chunk.range.start
            })
            .cloned()
            .collect::<Vec<_>>();
        dedupe_window_entities(&mut entities, profile_by_entity);
        if entities.len() < 2 {
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "too_few_entities".to_owned(),
            );
            continue;
        }
        push_window(
            windows,
            stats,
            archive,
            format!(
                "{}::{}::chunk::{}",
                archive.manifest.document_id, archive.manifest.revision, index
            ),
            index,
            chunk.range,
            Vec::new(),
            entities,
            "archive_chunk",
            Vec::new(),
            continuity_hints,
        );
    }
}

fn push_window(
    windows: &mut Vec<RelationWindowRecord>,
    stats: &mut RelationWindowBuildStats,
    archive: &DocumentArchive,
    window_id: String,
    window_index: usize,
    range: TextRange,
    sentence_indices: Vec<usize>,
    entities: Vec<RelationMention>,
    source: &'static str,
    mut evidence_labels: Vec<String>,
    continuity_hints: &FxHashMap<(String, String), BTreeSet<String>>,
) {
    if entities.len() < 2 {
        record_stat(
            &mut stats.rejected_window_reason_counts,
            "too_few_entities".to_owned(),
        );
        return;
    }
    let chunk_ids = archive
        .chunks
        .iter()
        .filter(|chunk| chunk.range.start < range.end && chunk.range.end > range.start)
        .map(|chunk| chunk.chunk_id.0.clone())
        .collect::<Vec<_>>();
    let window_text = render_archive_window_text(archive, range);
    if window_text.trim().is_empty() {
        record_stat(
            &mut stats.rejected_window_reason_counts,
            "empty_text".to_owned(),
        );
        return;
    }
    if looks_like_structural_window(&window_text) {
        record_stat(
            &mut stats.rejected_window_reason_counts,
            "structural".to_owned(),
        );
        return;
    }
    if !has_type_compatible_pair(&entities) {
        record_stat(
            &mut stats.rejected_window_reason_counts,
            "no_type_compatible_pair".to_owned(),
        );
        return;
    }
    if relation_anchor_density_too_high(&window_text, &entities) {
        record_stat(
            &mut stats.rejected_window_reason_counts,
            "anchor_density".to_owned(),
        );
        return;
    }
    let continuity_relations = continuity_relation_types(&entities, continuity_hints);
    let candidate_relation_types = infer_window_relation_types(
        &window_text,
        &entities,
        range.start as usize,
        &continuity_relations,
    );
    if candidate_relation_types.is_empty() {
        record_stat(
            &mut stats.rejected_window_reason_counts,
            "no_family_support".to_owned(),
        );
        return;
    }
    evidence_labels.push(format!("window_source:{source}"));
    for relation in &continuity_relations {
        evidence_labels.push(format!("continuity_hint:{relation}"));
    }
    for label in &evidence_labels {
        if let Some(kind) = label.strip_prefix("anchor_evidence:") {
            record_stat(&mut stats.anchor_evidence_counts, kind.to_owned());
        }
    }
    record_stat(&mut stats.window_source_counts, source.to_owned());
    stats
        .families_per_window
        .insert(window_id.clone(), candidate_relation_types.clone());
    windows.push(RelationWindowRecord {
        window_id,
        document_id: archive.manifest.document_id.clone(),
        revision: archive.manifest.revision,
        window_index,
        range,
        sentence_indices,
        chunk_ids,
        candidate_relation_types,
        evidence_labels,
        text: window_text,
        entities: entities
            .into_iter()
            .map(|entity| RelationWindowEntity {
                entity_id: entity.entity_id,
                surface: entity.surface,
                kind: entity.kind.clone(),
                entity_type: entity
                    .kind
                    .as_ref()
                    .map(|kind| format!("{kind:?}"))
                    .unwrap_or_else(|| "Unknown".to_owned()),
                span_start: entity.span_start,
                span_end: entity.span_end,
                sentence_index: entity.sentence_index,
                mention_index: entity.mention_index,
            })
            .collect(),
    });
}

fn record_stat(map: &mut BTreeMap<String, usize>, key: String) {
    *map.entry(key).or_default() += 1;
}

fn continuity_relation_map(
    persisted_relations: &[SemanticRelationRecord],
) -> FxHashMap<(String, String), BTreeSet<String>> {
    let mut rows = FxHashMap::<(String, String), BTreeSet<String>>::default();
    for relation in persisted_relations {
        if !is_supported_relation_family(&relation.edge_type) {
            continue;
        }
        rows.entry((
            relation.source_entity_id.0.clone(),
            relation.target_entity_id.0.clone(),
        ))
        .or_default()
        .insert(relation.edge_type.clone());
    }
    rows
}

fn append_synthetic_sentence_windows(
    windows: &mut Vec<RelationWindowRecord>,
    stats: &mut RelationWindowBuildStats,
    archive: &DocumentArchive,
    profiles: &[RelationEntityProfile],
    alex_lexicon: &Lexicon,
    _persisted_relations: &[SemanticRelationRecord],
    continuity_hints: &FxHashMap<(String, String), BTreeSet<String>>,
) {
    let mut sentence_index = 0usize;
    for chunk in &archive.chunks {
        let mut chunk_built_any = false;
        let synthetic = split_chunk_into_synthetic_sentences(chunk, sentence_index);
        sentence_index += synthetic.len();
        let candidate_profiles = candidate_profiles_for_chunk(archive, chunk, profiles);
        let mut sentence_mentions =
            Vec::<(RelationSyntheticSentence, Vec<RelationMention>, Vec<String>)>::new();
        for sentence in synthetic {
            let (mentions, labels) = rebuild_sentence_mentions(
                &sentence,
                &archive.manifest.scope,
                &candidate_profiles,
                alex_lexicon,
            );
            if mentions.len() < 2 {
                record_stat(
                    &mut stats.rejected_window_reason_counts,
                    "too_few_entities".to_owned(),
                );
                continue;
            }
            sentence_mentions.push((sentence, mentions, labels));
        }
        for (sentence, mentions, labels) in &sentence_mentions {
            let before = windows.len();
            push_window(
                windows,
                stats,
                archive,
                format!(
                    "{}::{}::synthetic::{}",
                    archive.manifest.document_id, archive.manifest.revision, sentence.index
                ),
                sentence.index,
                sentence.range,
                vec![sentence.index],
                mentions.clone(),
                "synthetic_sentence",
                labels.clone(),
                continuity_hints,
            );
            chunk_built_any |= windows.len() > before;
        }
        if chunk_built_any {
            continue;
        }
        for pair in sentence_mentions.windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            let mut mentions = left.1.clone();
            mentions.extend(right.1.clone());
            dedupe_relation_mentions_by_entity(&mut mentions);
            let mut labels = left.2.clone();
            labels.extend(right.2.clone());
            labels.sort();
            labels.dedup();
            let before = windows.len();
            push_window(
                windows,
                stats,
                archive,
                format!(
                    "{}::{}::synthetic-merge::{}",
                    archive.manifest.document_id, archive.manifest.revision, left.0.index
                ),
                left.0.index,
                TextRange {
                    start: left.0.range.start,
                    end: right.0.range.end,
                },
                vec![left.0.index, right.0.index],
                mentions,
                "synthetic_merge",
                labels,
                continuity_hints,
            );
            let _ = windows.len() > before;
        }
    }
}

fn append_clipped_chunk_windows(
    windows: &mut Vec<RelationWindowRecord>,
    stats: &mut RelationWindowBuildStats,
    archive: &DocumentArchive,
    profiles: &[RelationEntityProfile],
    alex_lexicon: &Lexicon,
    _persisted_relations: &[SemanticRelationRecord],
    continuity_hints: &FxHashMap<(String, String), BTreeSet<String>>,
) {
    for (index, chunk) in archive.chunks.iter().enumerate() {
        let sentence = RelationSyntheticSentence {
            index,
            chunk_id: chunk.chunk_id.0.clone(),
            range: chunk.range,
            text: chunk.text.clone(),
        };
        let candidate_profiles = candidate_profiles_for_chunk(archive, chunk, profiles);
        let (mut mentions, labels) = rebuild_sentence_mentions(
            &sentence,
            &archive.manifest.scope,
            &candidate_profiles,
            alex_lexicon,
        );
        dedupe_relation_mentions_by_entity(&mut mentions);
        if mentions.len() < 2 {
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "too_few_entities".to_owned(),
            );
            continue;
        }
        let clipped = clip_chunk_range(chunk, &mentions);
        push_window(
            windows,
            stats,
            archive,
            format!(
                "{}::{}::chunk-clipped::{}",
                archive.manifest.document_id, archive.manifest.revision, index
            ),
            index,
            clipped,
            Vec::new(),
            mentions,
            "chunk_clipped",
            labels,
            continuity_hints,
        );
    }
}

fn candidate_profiles_for_chunk<'a>(
    archive: &DocumentArchive,
    chunk: &phoenix_semantic_v2::ChunkRecord,
    profiles: &'a [RelationEntityProfile],
) -> Vec<&'a RelationEntityProfile> {
    profiles
        .iter()
        .filter(|profile| {
            profile
                .chunk_ids
                .iter()
                .any(|value| value == &chunk.chunk_id.0)
                || profile
                    .document_ids
                    .iter()
                    .any(|value| value == &archive.manifest.document_id)
        })
        .collect()
}

fn rebuild_sentence_mentions(
    sentence: &RelationSyntheticSentence,
    scope: &ScopeKey,
    profiles: &[&RelationEntityProfile],
    alex_lexicon: &Lexicon,
) -> (Vec<RelationMention>, Vec<String>) {
    let mut mentions = Vec::new();
    let mut evidence_labels = Vec::new();
    let profile_by_entity = profiles
        .iter()
        .map(|profile| (profile.entity_id.0.as_str(), *profile))
        .collect::<FxHashMap<_, _>>();
    for known_match in alex_api::scan_text(alex_lexicon, &sentence.text, scope) {
        let Some(profile) = choose_relation_match_profile(&known_match.entries, &profile_by_entity)
        else {
            continue;
        };
        if !is_relation_scan_surface(
            &known_match.surface,
            profile.kind.as_ref(),
            known_match.source.as_ref(),
        ) {
            continue;
        }
        mentions.push(RelationMention {
            entity_id: profile.entity_id.clone(),
            surface: known_match.surface.clone(),
            kind: profile.kind.clone(),
            sentence_index: sentence.index,
            span_start: sentence.range.start as usize + known_match.range.start as usize,
            span_end: sentence.range.start as usize + known_match.range.end as usize,
            mention_index: None,
        });
        evidence_labels.push(relation_match_source_label(known_match.source.as_ref()));
    }
    dedupe_relation_mentions_by_entity(&mut mentions);
    evidence_labels.sort();
    evidence_labels.dedup();
    (mentions, evidence_labels)
}

pub(crate) fn split_chunk_into_synthetic_sentences(
    chunk: &phoenix_semantic_v2::ChunkRecord,
    sentence_index_start: usize,
) -> Vec<RelationSyntheticSentence> {
    let text = chunk.text.as_str();
    let spans = alex_sentence_ranges(text);
    let mut index = sentence_index_start;
    let mut rows = Vec::new();
    for (start, end) in spans {
        let slice = &text[start..end];
        let trimmed_start = slice.len().saturating_sub(slice.trim_start().len());
        let trimmed_end = slice.trim_end().len();
        if trimmed_end <= trimmed_start {
            continue;
        }
        let local_start = start + trimmed_start;
        let local_end = start + trimmed_end;
        let sentence_text = text[local_start..local_end].trim().to_owned();
        if sentence_text.is_empty() {
            continue;
        }
        rows.push(RelationSyntheticSentence {
            index,
            chunk_id: chunk.chunk_id.0.clone(),
            range: TextRange {
                start: chunk.range.start + local_start as u32,
                end: chunk.range.start + local_end as u32,
            },
            text: sentence_text,
        });
        index += 1;
    }
    if rows.is_empty() && !text.trim().is_empty() {
        rows.push(RelationSyntheticSentence {
            index,
            chunk_id: chunk.chunk_id.0.clone(),
            range: chunk.range,
            text: text.trim().to_owned(),
        });
    }
    rows
}

fn build_relation_alex_lexicon(
    profiles: &[RelationEntityProfile],
) -> Result<Lexicon, GlirelWorkerError> {
    let mut entries = Vec::with_capacity(profiles.len());
    for profile in profiles {
        let mut aliases = profile
            .aliases
            .iter()
            .filter_map(|alias| {
                let trimmed = alias.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_owned())
            })
            .collect::<Vec<_>>();
        aliases.sort();
        aliases.dedup();
        entries.push(LexiconEntry {
            entity_id: profile.entity_id.clone(),
            label: profile.canonical_name.clone(),
            aliases,
            kind: profile.kind.clone(),
            gender: None,
            number: None,
            scope: profile.scope.clone(),
        });
    }
    Ok(alex_api::build_lexicon(&entries)?)
}

fn collect_alex_relation_mentions(
    archive: &DocumentArchive,
    profiles: &[RelationEntityProfile],
    alex_lexicon: &Lexicon,
) -> Vec<RelationMention> {
    let mut mentions = Vec::new();
    for chunk in &archive.chunks {
        let candidate_profiles = candidate_profiles_for_chunk(archive, chunk, profiles);
        let synthetic = split_chunk_into_synthetic_sentences(chunk, 0);
        for sentence in synthetic {
            let (mut sentence_mentions, _) = rebuild_sentence_mentions(
                &sentence,
                &archive.manifest.scope,
                &candidate_profiles,
                alex_lexicon,
            );
            for mention in &mut sentence_mentions {
                mention.sentence_index = mention_sentence_index(archive, mention.span_start);
            }
            mentions.extend(sentence_mentions);
        }
    }
    dedupe_relation_mentions(&mut mentions);
    mentions
}

fn choose_relation_match_profile<'a>(
    entries: &[LexiconEntry],
    profile_by_entity: &FxHashMap<&str, &'a RelationEntityProfile>,
) -> Option<&'a RelationEntityProfile> {
    entries
        .iter()
        .filter_map(|entry| profile_by_entity.get(entry.entity_id.0.as_str()).copied())
        .max_by(|left, right| {
            left.continuity_score_millis
                .cmp(&right.continuity_score_millis)
                .then_with(|| left.canonical_name.len().cmp(&right.canonical_name.len()))
        })
}

fn relation_match_source_label(source: Option<&KnownMatchSource>) -> String {
    match source {
        Some(KnownMatchSource::ExactCanonical) => "anchor_evidence:alex_exact_canonical".to_owned(),
        Some(KnownMatchSource::ExactAlias) => "anchor_evidence:alex_exact_alias".to_owned(),
        Some(KnownMatchSource::ExactAutoAlias) => "anchor_evidence:alex_auto_alias".to_owned(),
        Some(KnownMatchSource::FuzzyAnchor) => "anchor_evidence:alex_fuzzy_anchor".to_owned(),
        None => "anchor_evidence:alex_match".to_owned(),
    }
}

fn is_relation_scan_surface(
    surface: &str,
    kind: Option<&EntityKind>,
    source: Option<&KnownMatchSource>,
) -> bool {
    let allow_compact_alias = matches!(
        source,
        Some(KnownMatchSource::ExactAlias | KnownMatchSource::ExactAutoAlias)
    );
    is_relation_anchor_surface(surface, kind, allow_compact_alias)
}

fn normalize_relation_surface(surface: &str) -> String {
    surface
        .trim()
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn dedupe_relation_mentions_by_entity(mentions: &mut Vec<RelationMention>) {
    let mut best = FxHashMap::<String, RelationMention>::default();
    for mention in mentions.drain(..) {
        match best.get(&mention.entity_id.0) {
            Some(existing) if existing.surface.len() >= mention.surface.len() => {}
            _ => {
                best.insert(mention.entity_id.0.clone(), mention);
            }
        }
    }
    let mut rows = best.into_values().collect::<Vec<_>>();
    rows.sort_by(|left, right| left.entity_id.0.cmp(&right.entity_id.0));
    *mentions = rows;
}

fn render_archive_window_text(archive: &DocumentArchive, range: TextRange) -> String {
    let mut slices = archive
        .chunks
        .iter()
        .filter_map(|chunk| {
            let start = range.start.max(chunk.range.start);
            let end = range.end.min(chunk.range.end);
            if end <= start {
                return None;
            }
            slice_text_range(&chunk.text, TextRange { start, end }, chunk.range)
        })
        .collect::<Vec<_>>();
    slices.dedup();
    slices.join(" ").trim().to_owned()
}

fn has_type_compatible_pair(entities: &[RelationMention]) -> bool {
    entities.iter().enumerate().any(|(index, left)| {
        entities.iter().skip(index + 1).any(|right| {
            pair_matches_relation_kinds(left.kind.as_ref(), right.kind.as_ref())
                || pair_matches_relation_kinds(right.kind.as_ref(), left.kind.as_ref())
        })
    })
}

fn relation_anchor_density_too_high(text: &str, entities: &[RelationMention]) -> bool {
    let token_count = text.split_whitespace().count().max(1);
    if entities.len() > 8 {
        return true;
    }
    if entities.len() <= 3 {
        return false;
    }
    let anchor_chars = entities
        .iter()
        .map(|entity| entity.surface.chars().count())
        .sum::<usize>();
    let unique_entities = entities
        .iter()
        .map(|entity| entity.entity_id.0.as_str())
        .collect::<BTreeSet<_>>()
        .len();
    unique_entities > 6
        || (entities.len() >= 5 && entities.len().saturating_mul(3) > token_count)
        || anchor_chars.saturating_mul(100) > text.len().max(1).saturating_mul(70)
}

fn clip_chunk_range(
    chunk: &phoenix_semantic_v2::ChunkRecord,
    mentions: &[RelationMention],
) -> TextRange {
    let min_start = mentions
        .iter()
        .map(|mention| mention.span_start)
        .min()
        .unwrap_or(chunk.range.start as usize);
    let max_end = mentions
        .iter()
        .map(|mention| mention.span_end)
        .max()
        .unwrap_or(chunk.range.end as usize);
    let start = min_start.saturating_sub(64).max(chunk.range.start as usize) as u32;
    let end = (max_end + 64).min(chunk.range.end as usize) as u32;
    TextRange { start, end }
}

fn find_word_boundary_match(text: &str, surface: &str) -> Option<(usize, usize)> {
    let needle = surface.trim();
    if needle.is_empty() {
        return None;
    }
    let lower_text = text.to_ascii_lowercase();
    let lower_needle = needle.to_ascii_lowercase();
    let mut offset = 0usize;
    while let Some(found) = lower_text[offset..].find(&lower_needle) {
        let start = offset + found;
        let end = start + lower_needle.len();
        let prev_ok = start == 0
            || !text[..start]
                .chars()
                .next_back()
                .is_some_and(relation_surface_char);
        let next_ok = end >= text.len()
            || !text[end..]
                .chars()
                .next()
                .is_some_and(relation_surface_char);
        if prev_ok && next_ok {
            return Some((start, end));
        }
        offset = end;
    }
    None
}

fn relation_surface_char(value: char) -> bool {
    value.is_alphanumeric() || matches!(value, '_' | '-' | '\'')
}

fn continuity_relation_types(
    entities: &[RelationMention],
    continuity_hints: &FxHashMap<(String, String), BTreeSet<String>>,
) -> Vec<String> {
    let mut relations = BTreeSet::new();
    for (index, left) in entities.iter().enumerate() {
        for right in entities.iter().skip(index + 1) {
            if let Some(values) =
                continuity_hints.get(&(left.entity_id.0.clone(), right.entity_id.0.clone()))
            {
                relations.extend(values.iter().cloned());
            }
            if let Some(values) =
                continuity_hints.get(&(right.entity_id.0.clone(), left.entity_id.0.clone()))
            {
                relations.extend(values.iter().cloned());
            }
        }
    }
    relations.into_iter().collect()
}

fn continuity_relation_types_from_entities(
    entities: &[RelationWindowEntity],
    continuity_hints: &FxHashMap<(String, String), BTreeSet<String>>,
) -> Vec<String> {
    let mut relations = BTreeSet::new();
    for (index, left) in entities.iter().enumerate() {
        for right in entities.iter().skip(index + 1) {
            if let Some(values) =
                continuity_hints.get(&(left.entity_id.0.clone(), right.entity_id.0.clone()))
            {
                relations.extend(values.iter().cloned());
            }
            if let Some(values) =
                continuity_hints.get(&(right.entity_id.0.clone(), left.entity_id.0.clone()))
            {
                relations.extend(values.iter().cloned());
            }
        }
    }
    relations.into_iter().collect()
}

fn append_relation_candidate_windows(
    windows: &mut Vec<RelationWindowRecord>,
    stats: &mut RelationWindowBuildStats,
    archive: &DocumentArchive,
    mentions: &[RelationMention],
    profile_by_entity: &FxHashMap<String, &RelationEntityProfile>,
    _persisted_relations: &[SemanticRelationRecord],
    continuity_hints: &FxHashMap<(String, String), BTreeSet<String>>,
) {
    for (index, candidate) in archive.relation_candidates.iter().enumerate() {
        let Some(mut window) =
            build_relation_candidate_window(archive, candidate, index, mentions, profile_by_entity)
        else {
            continue;
        };
        dedupe_window_entities_like(&mut window.entities, profile_by_entity);
        if window.entities.len() < 2 {
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "too_few_entities".to_owned(),
            );
            continue;
        }
        if looks_like_structural_window(&window.text) {
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "structural".to_owned(),
            );
            continue;
        }
        if relation_anchor_density_too_high(
            &window.text,
            &window
                .entities
                .iter()
                .map(|entity| RelationMention {
                    entity_id: entity.entity_id.clone(),
                    surface: entity.surface.clone(),
                    kind: entity.kind.clone(),
                    sentence_index: entity.sentence_index,
                    span_start: entity.span_start,
                    span_end: entity.span_end,
                    mention_index: entity.mention_index,
                })
                .collect::<Vec<_>>(),
        ) {
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "anchor_density".to_owned(),
            );
            continue;
        }
        window
            .evidence_labels
            .push("window_source:relation_candidate".to_owned());
        for relation in continuity_relation_types_from_entities(&window.entities, continuity_hints)
        {
            if !window
                .candidate_relation_types
                .iter()
                .any(|value| value == &relation)
            {
                window.candidate_relation_types.push(relation.clone());
            }
            window
                .evidence_labels
                .push(format!("continuity_hint:{relation}"));
        }
        record_stat(
            &mut stats.window_source_counts,
            "relation_candidate".to_owned(),
        );
        stats.families_per_window.insert(
            window.window_id.clone(),
            window.candidate_relation_types.clone(),
        );
        windows.push(window);
    }
}

fn append_archive_relation_windows(
    windows: &mut Vec<RelationWindowRecord>,
    stats: &mut RelationWindowBuildStats,
    archive: &DocumentArchive,
    mentions: &[RelationMention],
    profile_by_entity: &FxHashMap<String, &RelationEntityProfile>,
    persisted_relations: &[SemanticRelationRecord],
    continuity_hints: &FxHashMap<(String, String), BTreeSet<String>>,
) {
    for (index, relation) in persisted_relations.iter().enumerate() {
        record_stat(
            &mut stats.rejected_window_reason_counts,
            "archive_relation_attempted".to_owned(),
        );
        let Some(mut window) =
            build_archive_relation_window(archive, relation, index, mentions, profile_by_entity)
        else {
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "archive_relation_no_window".to_owned(),
            );
            continue;
        };
        dedupe_window_entities_like(&mut window.entities, profile_by_entity);
        if window.entities.len() < 2 {
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "too_few_entities".to_owned(),
            );
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "archive_relation_too_few_entities".to_owned(),
            );
            continue;
        }
        if looks_like_structural_window(&window.text) {
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "structural".to_owned(),
            );
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "archive_relation_structural".to_owned(),
            );
            continue;
        }
        if relation_anchor_density_too_high(
            &window.text,
            &window
                .entities
                .iter()
                .map(|entity| RelationMention {
                    entity_id: entity.entity_id.clone(),
                    surface: entity.surface.clone(),
                    kind: entity.kind.clone(),
                    sentence_index: entity.sentence_index,
                    span_start: entity.span_start,
                    span_end: entity.span_end,
                    mention_index: entity.mention_index,
                })
                .collect::<Vec<_>>(),
        ) {
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "anchor_density".to_owned(),
            );
            record_stat(
                &mut stats.rejected_window_reason_counts,
                "archive_relation_anchor_density".to_owned(),
            );
            continue;
        }
        let continuity_relations =
            continuity_relation_types_from_entities(&window.entities, continuity_hints);
        let inferred_relations = infer_window_relation_types_from_window_entities(
            &window.text,
            &window.entities,
            0,
            &continuity_relations,
        );
        if inferred_relations.is_empty() {
            let fallback_relations =
                archive_relation_candidate_families_from_entities(&window.entities);
            if fallback_relations.is_empty() {
                record_stat(
                    &mut stats.rejected_window_reason_counts,
                    "no_family_support".to_owned(),
                );
                record_stat(
                    &mut stats.rejected_window_reason_counts,
                    "archive_relation_no_family_support".to_owned(),
                );
                continue;
            }
            window
                .evidence_labels
                .push("archive_relation_fallback:type_compatible".to_owned());
            window.candidate_relation_types = fallback_relations;
        } else {
            window.candidate_relation_types = inferred_relations;
        }
        window
            .evidence_labels
            .push("window_source:archive_relation".to_owned());
        window
            .evidence_labels
            .push(format!("archive_relation_hint:{}", relation.edge_type));
        for relation in continuity_relations {
            if !window
                .candidate_relation_types
                .iter()
                .any(|value| value == &relation)
            {
                window.candidate_relation_types.push(relation.clone());
            }
            window
                .evidence_labels
                .push(format!("continuity_hint:{relation}"));
        }
        record_stat(
            &mut stats.window_source_counts,
            "archive_relation".to_owned(),
        );
        stats.families_per_window.insert(
            window.window_id.clone(),
            window.candidate_relation_types.clone(),
        );
        windows.push(window);
    }
}

fn build_archive_relation_window(
    archive: &DocumentArchive,
    relation: &SemanticRelationRecord,
    index: usize,
    mentions: &[RelationMention],
    profile_by_entity: &FxHashMap<String, &RelationEntityProfile>,
) -> Option<RelationWindowRecord> {
    let source_profile = profile_by_entity
        .get(&relation.source_entity_id.0)
        .copied()?;
    let target_profile = profile_by_entity
        .get(&relation.target_entity_id.0)
        .copied()?;
    let sentence_index = relation.sentence_index;
    let mut candidate_chunks = Vec::new();
    if let Some(chunk_id) = relation.chunk_id.as_ref() {
        if let Some(chunk) = archive
            .chunks
            .iter()
            .find(|chunk| chunk.chunk_id.0 == *chunk_id)
        {
            candidate_chunks.push(chunk);
        }
    }
    for chunk in &archive.chunks {
        if candidate_chunks
            .iter()
            .any(|existing| existing.chunk_id.0 == chunk.chunk_id.0)
        {
            continue;
        }
        if source_profile
            .chunk_ids
            .iter()
            .any(|chunk_id| chunk_id == &chunk.chunk_id.0)
            || target_profile
                .chunk_ids
                .iter()
                .any(|chunk_id| chunk_id == &chunk.chunk_id.0)
        {
            candidate_chunks.push(chunk);
        }
    }
    for chunk in &archive.chunks {
        if candidate_chunks
            .iter()
            .any(|existing| existing.chunk_id.0 == chunk.chunk_id.0)
        {
            continue;
        }
        candidate_chunks.push(chunk);
    }
    for chunk in candidate_chunks {
        if let Some(window) =
            build_archive_relation_window_from_mentions(archive, relation, index, chunk, mentions)
        {
            return Some(window);
        }
        let Some((clipped_range, source_surface, target_surface)) =
            build_archive_relation_range(chunk, source_profile, target_profile)
        else {
            continue;
        };
        let Some(text) = slice_text_range(&chunk.text, clipped_range, chunk.range) else {
            continue;
        };
        if text.trim().is_empty() {
            continue;
        }
        let mut entities = vec![
            relation_window_entity_from_profile(
                source_profile,
                relation.source_entity_id.clone(),
                sentence_index,
                &text,
                &source_surface,
            )?,
            relation_window_entity_from_profile(
                target_profile,
                relation.target_entity_id.clone(),
                sentence_index,
                &text,
                &target_surface,
            )?,
        ];
        align_relation_candidate_entity_spans(&text, &mut entities);
        return Some(RelationWindowRecord {
            window_id: format!(
                "{}::{}::archive-relation::{}",
                archive.manifest.document_id, archive.manifest.revision, index
            ),
            document_id: archive.manifest.document_id.clone(),
            revision: archive.manifest.revision,
            window_index: index,
            range: TextRange {
                start: 0,
                end: text.len() as u32,
            },
            sentence_indices: vec![sentence_index],
            chunk_ids: vec![chunk.chunk_id.0.clone()],
            candidate_relation_types: Vec::new(),
            evidence_labels: Vec::new(),
            text,
            entities,
        });
    }
    None
}

fn build_archive_relation_window_from_mentions(
    archive: &DocumentArchive,
    relation: &SemanticRelationRecord,
    index: usize,
    chunk: &phoenix_semantic_v2::ChunkRecord,
    mentions: &[RelationMention],
) -> Option<RelationWindowRecord> {
    let source = best_chunk_relation_mention(mentions, &relation.source_entity_id, chunk)?;
    let target = best_chunk_relation_mention(mentions, &relation.target_entity_id, chunk)?;
    let clipped = TextRange {
        start: source
            .span_start
            .min(target.span_start)
            .saturating_sub(96)
            .max(chunk.range.start as usize) as u32,
        end: (source.span_end.max(target.span_end) + 96).min(chunk.range.end as usize) as u32,
    };
    let text = slice_text_range(&chunk.text, clipped, chunk.range)?;
    if text.trim().is_empty() {
        return None;
    }
    let mut entities = vec![
        relation_window_entity_from_mention_like(&source, clipped.start as usize),
        relation_window_entity_from_mention_like(&target, clipped.start as usize),
    ];
    align_relation_candidate_entity_spans(&text, &mut entities);
    Some(RelationWindowRecord {
        window_id: format!(
            "{}::{}::archive-relation::{}",
            archive.manifest.document_id, archive.manifest.revision, index
        ),
        document_id: archive.manifest.document_id.clone(),
        revision: archive.manifest.revision,
        window_index: index,
        range: TextRange {
            start: 0,
            end: text.len() as u32,
        },
        sentence_indices: vec![relation.sentence_index],
        chunk_ids: vec![chunk.chunk_id.0.clone()],
        candidate_relation_types: Vec::new(),
        evidence_labels: Vec::new(),
        text,
        entities,
    })
}

fn best_chunk_relation_mention(
    mentions: &[RelationMention],
    entity_id: &EntityId,
    chunk: &phoenix_semantic_v2::ChunkRecord,
) -> Option<RelationMention> {
    mentions
        .iter()
        .filter(|mention| {
            mention.entity_id == *entity_id
                && mention.span_start < chunk.range.end as usize
                && mention.span_end > chunk.range.start as usize
        })
        .cloned()
        .max_by(|left, right| {
            left.surface.len().cmp(&right.surface.len()).then_with(|| {
                std::cmp::Reverse(left.span_start).cmp(&std::cmp::Reverse(right.span_start))
            })
        })
}

fn build_archive_relation_range(
    chunk: &phoenix_semantic_v2::ChunkRecord,
    source_profile: &RelationEntityProfile,
    target_profile: &RelationEntityProfile,
) -> Option<(TextRange, String, String)> {
    let source_surface = best_profile_anchor_surface_in_text(source_profile, &chunk.text)?;
    let target_surface = best_profile_anchor_surface_in_text(target_profile, &chunk.text)?;
    let source_span = find_word_boundary_match(&chunk.text, &source_surface)?;
    let target_span = find_word_boundary_match(&chunk.text, &target_surface)?;
    let min_start = source_span.0.min(target_span.0);
    let max_end = source_span.1.max(target_span.1);
    let start = min_start.saturating_sub(96).min(chunk.text.len()) as u32 + chunk.range.start;
    let end = (max_end + 96).min(chunk.text.len()) as u32 + chunk.range.start;
    Some((TextRange { start, end }, source_surface, target_surface))
}

fn best_profile_anchor_surface_in_text(
    profile: &RelationEntityProfile,
    text: &str,
) -> Option<String> {
    std::iter::once(profile.canonical_name.as_str())
        .chain(profile.aliases.iter().map(String::as_str))
        .map(str::trim)
        .filter(|surface| is_relation_anchor_surface(surface, profile.kind.as_ref(), true))
        .filter_map(|surface| {
            let span = find_word_boundary_match(text, surface)?;
            let preferred = if surface.eq_ignore_ascii_case(&profile.canonical_name) {
                10_000
            } else {
                0
            };
            Some((preferred + surface.len(), span.0, surface))
        })
        .max_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| std::cmp::Reverse(left.1).cmp(&std::cmp::Reverse(right.1)))
        })
        .map(|(_, _, surface)| surface.to_owned())
}

fn relation_window_entity_from_profile(
    profile: &RelationEntityProfile,
    entity_id: EntityId,
    sentence_index: usize,
    text: &str,
    preferred_surface: &str,
) -> Option<RelationWindowEntity> {
    let surface = preferred_surface.trim();
    if surface.is_empty() {
        return None;
    }
    find_word_boundary_match(text, &surface)?;
    let kind = profile.kind.clone();
    Some(RelationWindowEntity {
        entity_id,
        surface: surface.to_owned(),
        kind: kind.clone(),
        entity_type: kind
            .as_ref()
            .map(|kind| format!("{kind:?}"))
            .unwrap_or_else(|| "Unknown".to_owned()),
        span_start: 0,
        span_end: 0,
        sentence_index,
        mention_index: None,
    })
}

fn build_relation_candidate_window(
    archive: &DocumentArchive,
    candidate: &RelationCandidate,
    index: usize,
    mentions: &[RelationMention],
    profile_by_entity: &FxHashMap<String, &RelationEntityProfile>,
) -> Option<RelationWindowRecord> {
    let mut entities = Vec::new();
    if let Some(entity) = relation_window_entity_from_slot(
        archive,
        candidate,
        candidate.subject.as_ref(),
        candidate.sentence_index,
        mentions,
        profile_by_entity,
    ) {
        entities.push(entity);
    }
    if let Some(entity) = relation_window_entity_from_slot(
        archive,
        candidate,
        candidate.object.as_ref(),
        candidate.sentence_index,
        mentions,
        profile_by_entity,
    ) {
        entities.push(entity);
    }
    if let Some(entity) = relation_window_entity_from_slot(
        archive,
        candidate,
        candidate.recipient.as_ref(),
        candidate.sentence_index,
        mentions,
        profile_by_entity,
    ) {
        entities.push(entity);
    }
    if entities.len() < 2 {
        return None;
    }

    let sentence_indices = vec![candidate.sentence_index];
    let candidate_relation_types = vec![candidate.relation_type.clone()];
    let evidence_labels = candidate
        .evidence
        .iter()
        .map(|evidence| evidence.label.trim().to_owned())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let chunk_ids = archive
        .chunks
        .iter()
        .filter(|chunk| {
            let range = relation_candidate_doc_range(candidate);
            chunk.range.start < range.end && chunk.range.end > range.start
        })
        .map(|chunk| chunk.chunk_id.0.clone())
        .collect::<Vec<_>>();
    let text = render_relation_candidate_text(candidate, &entities);
    if text.trim().is_empty() {
        return None;
    }

    let mut local_entities = entities;
    align_relation_candidate_entity_spans(&text, &mut local_entities);

    Some(RelationWindowRecord {
        window_id: format!(
            "{}::{}::candidate::{}",
            archive.manifest.document_id, archive.manifest.revision, index
        ),
        document_id: archive.manifest.document_id.clone(),
        revision: archive.manifest.revision,
        window_index: index,
        range: TextRange {
            start: 0,
            end: text.len() as u32,
        },
        sentence_indices,
        chunk_ids,
        candidate_relation_types,
        evidence_labels,
        text,
        entities: local_entities,
    })
}

fn relation_window_entity_from_slot(
    archive: &DocumentArchive,
    candidate: &RelationCandidate,
    slot: Option<&phoenix_types::FrameSlot>,
    sentence_index: usize,
    mentions: &[RelationMention],
    profile_by_entity: &FxHashMap<String, &RelationEntityProfile>,
) -> Option<RelationWindowEntity> {
    let slot = slot?;
    let MentionEntityRef::Known(entity_id) = slot.entity_ref.as_ref()? else {
        return None;
    };
    let profile = profile_by_entity.get(&entity_id.0).copied();
    let mention = mentions
        .iter()
        .filter(|mention| mention.entity_id == *entity_id)
        .min_by_key(|mention| {
            let sentence_gap = mention.sentence_index.abs_diff(sentence_index);
            let range_gap = mention.span_start.abs_diff(slot.range.start as usize);
            sentence_gap.saturating_mul(10_000) + range_gap
        });
    let surface = mention
        .map(|mention| mention.surface.clone())
        .or_else(|| profile.map(|profile| profile.canonical_name.clone()))
        .or_else(|| text_from_candidate_slot(candidate, slot).filter(|value| !value.is_empty()))
        .or_else(|| {
            text_from_archive_range(archive, slot.range).filter(|value| !value.is_empty())
        })?;
    let kind = mention
        .and_then(|mention| mention.kind.clone())
        .or_else(|| profile.and_then(|profile| profile.kind.clone()));
    Some(RelationWindowEntity {
        entity_id: entity_id.clone(),
        surface,
        kind: kind.clone(),
        entity_type: kind
            .as_ref()
            .map(|kind| format!("{kind:?}"))
            .unwrap_or_else(|| "Unknown".to_owned()),
        span_start: 0,
        span_end: 0,
        sentence_index,
        mention_index: mention.and_then(|mention| mention.mention_index),
    })
}

fn relation_candidate_doc_range(candidate: &RelationCandidate) -> TextRange {
    let mut start = candidate.verb_range.start;
    let mut end = candidate.verb_range.end;
    for range in candidate
        .subject
        .iter()
        .chain(candidate.object.iter())
        .chain(candidate.recipient.iter())
        .map(|slot| slot.range)
        .chain(candidate.attachments.iter().copied())
        .chain(candidate.evidence.iter().map(|evidence| evidence.range))
    {
        start = start.min(range.start);
        end = end.max(range.end);
    }
    TextRange { start, end }
}

fn text_from_candidate_slot(
    candidate: &RelationCandidate,
    slot: &phoenix_types::FrameSlot,
) -> Option<String> {
    candidate
        .evidence
        .iter()
        .find_map(|evidence| slice_text_range(&evidence.label, slot.range, evidence.range))
}

fn text_from_archive_range(archive: &DocumentArchive, range: TextRange) -> Option<String> {
    archive
        .chunks
        .iter()
        .find_map(|chunk| slice_text_range(&chunk.text, range, chunk.range))
}

fn slice_text_range(text: &str, target: TextRange, container: TextRange) -> Option<String> {
    if target.start < container.start || target.end > container.end || target.end <= target.start {
        return None;
    }
    let local_start = (target.start - container.start) as usize;
    let local_end = (target.end - container.start) as usize;
    text.get(local_start..local_end)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.to_owned())
}

fn render_relation_candidate_text(
    candidate: &RelationCandidate,
    entities: &[RelationWindowEntity],
) -> String {
    let mut text = candidate
        .evidence
        .iter()
        .map(|evidence| evidence.label.trim())
        .find(|value| !value.is_empty())
        .unwrap_or("")
        .to_owned();
    for entity in entities {
        if !contains_case_insensitive(&text, &entity.surface) {
            if !text.is_empty() {
                text.push(' ');
            }
            text.push_str(&entity.surface);
        }
    }
    if !contains_case_insensitive(&text, &candidate.lemma) && !candidate.lemma.trim().is_empty() {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(&candidate.lemma);
    }
    text.trim().to_owned()
}

fn contains_case_insensitive(haystack: &str, needle: &str) -> bool {
    let haystack = haystack.to_ascii_lowercase();
    let needle = needle.to_ascii_lowercase();
    !needle.is_empty() && haystack.contains(&needle)
}

fn align_relation_candidate_entity_spans(text: &str, entities: &mut [RelationWindowEntity]) {
    let lower = text.to_ascii_lowercase();
    let mut cursor = 0usize;
    for entity in entities {
        let needle = entity.surface.to_ascii_lowercase();
        if needle.is_empty() {
            continue;
        }
        if let Some(found) = lower[cursor..].find(&needle) {
            let start = cursor + found;
            entity.span_start = start;
            entity.span_end = start + entity.surface.len();
            cursor = entity.span_end.min(text.len());
        }
    }
}

fn dedupe_window_entities_like(
    entities: &mut Vec<RelationWindowEntity>,
    profile_by_entity: &FxHashMap<String, &RelationEntityProfile>,
) {
    let mut by_entity = FxHashMap::<String, RelationWindowEntity>::default();
    for entity in entities.drain(..) {
        let quality = profile_by_entity
            .get(&entity.entity_id.0)
            .map(|profile| profile.continuity_score_millis)
            .unwrap_or_default();
        match by_entity.get(&entity.entity_id.0) {
            Some(existing) => {
                let existing_quality = profile_by_entity
                    .get(&existing.entity_id.0)
                    .map(|profile| profile.continuity_score_millis)
                    .unwrap_or_default();
                if quality > existing_quality || entity.surface.len() > existing.surface.len() {
                    by_entity.insert(entity.entity_id.0.clone(), entity);
                }
            }
            None => {
                by_entity.insert(entity.entity_id.0.clone(), entity);
            }
        }
    }
    *entities = by_entity.into_values().collect::<Vec<_>>();
    entities.sort_by(|left, right| left.span_start.cmp(&right.span_start));
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RelationMention {
    entity_id: EntityId,
    surface: String,
    kind: Option<EntityKind>,
    sentence_index: usize,
    span_start: usize,
    span_end: usize,
    mention_index: Option<usize>,
}

fn collect_relation_mentions(
    archive: &DocumentArchive,
    er_view: &ErPatchView,
) -> Vec<RelationMention> {
    let mut mentions = Vec::new();
    for resolved in &archive.resolved_mentions {
        let mut entity_id = resolved.entity_id.clone();
        if let Some(override_id) = er_view
            .entity_by_mention
            .get(&resolved.mention_id.0)
            .cloned()
        {
            entity_id = Some(EntityId(override_id));
        }
        let Some(entity_id) = entity_id else {
            continue;
        };
        let kind = er_view
            .kind_by_mention
            .get(&resolved.mention_id.0)
            .cloned()
            .or_else(|| er_view.kind_by_entity.get(&entity_id.0).cloned())
            .or_else(|| resolved.kind.clone());
        if !is_relation_salient_surface(&resolved.surface, kind.as_ref()) {
            continue;
        }
        mentions.push(RelationMention {
            entity_id,
            surface: resolved.surface.clone(),
            kind,
            sentence_index: mention_sentence_index(archive, resolved.range.start as usize),
            span_start: resolved.range.start as usize,
            span_end: resolved.range.end as usize,
            mention_index: Some(resolved.mention_index),
        });
    }
    mentions
}

fn collect_relation_seed_mentions(
    archive: &DocumentArchive,
    sidecar: Option<&RelationMentionSeedScopeSidecar>,
) -> Vec<RelationMention> {
    let Some(sidecar) = sidecar else {
        return Vec::new();
    };
    let mut mentions = sidecar
        .seeds
        .iter()
        .filter(|seed| {
            seed.document_id == archive.manifest.document_id
                && seed.revision == archive.manifest.revision
        })
        .filter(|seed| is_relation_salient_surface(&seed.surface, seed.kind.as_ref()))
        .map(|seed| RelationMention {
            entity_id: seed.entity_id.clone(),
            surface: seed.surface.clone(),
            kind: seed.kind.clone(),
            sentence_index: seed
                .sentence_index
                .unwrap_or_else(|| mention_sentence_index(archive, seed.range.start as usize)),
            span_start: seed.range.start as usize,
            span_end: seed.range.end as usize,
            mention_index: None,
        })
        .collect::<Vec<_>>();
    dedupe_relation_mentions(&mut mentions);
    mentions
}

fn collect_gliner_seed_mentions(
    archive: &DocumentArchive,
    profiles: &[RelationEntityProfile],
    seeder: &RelationMentionSeeder,
) -> Result<Vec<RelationMention>, GlirelWorkerError> {
    let candidate_chunks = relation_seed_candidate_chunks(archive, profiles);
    if candidate_chunks.is_empty() {
        return Ok(Vec::new());
    }
    let profile_surface_index = build_profile_surface_index(profiles);
    let seeded = seeder
        .seed_chunk_mentions(&candidate_chunks)
        .map_err(GlirelWorkerError::Seeder)?;
    let chunk_by_id = archive
        .chunks
        .iter()
        .map(|chunk| (chunk.chunk_id.0.as_str(), chunk))
        .collect::<FxHashMap<_, _>>();
    let mut mentions = Vec::new();
    for span in seeded {
        let Some(chunk) = chunk_by_id.get(span.input_id.as_str()) else {
            continue;
        };
        let normalized = normalize_relation_surface(&span.surface);
        let Some(profiles) = profile_surface_index.get(normalized.as_str()) else {
            continue;
        };
        let Some(profile) = choose_seed_profile(profiles, &span.label) else {
            continue;
        };
        if !is_relation_salient_surface(&span.surface, profile.kind.as_ref()) {
            continue;
        }
        mentions.push(RelationMention {
            entity_id: profile.entity_id.clone(),
            surface: span.surface.clone(),
            kind: profile.kind.clone(),
            sentence_index: mention_sentence_index(
                archive,
                chunk.range.start as usize + span.span_start,
            ),
            span_start: chunk.range.start as usize + span.span_start,
            span_end: chunk.range.start as usize + span.span_end,
            mention_index: None,
        });
    }
    dedupe_relation_mentions(&mut mentions);
    Ok(mentions)
}

fn relation_seed_candidate_chunks(
    archive: &DocumentArchive,
    profiles: &[RelationEntityProfile],
) -> Vec<(String, String)> {
    const MAX_RELATION_SEED_CHUNKS: usize = 12;
    let profile_by_entity = profiles
        .iter()
        .map(|profile| (profile.entity_id.0.as_str(), profile))
        .collect::<FxHashMap<_, _>>();
    let chunk_by_id = archive
        .chunks
        .iter()
        .map(|chunk| (chunk.chunk_id.0.as_str(), chunk))
        .collect::<FxHashMap<_, _>>();
    let mut selected = Vec::new();
    let mut seen = BTreeSet::new();

    for relation in &archive.relations {
        if let Some(chunk_id) = relation.chunk_id.as_deref() {
            if seen.insert(chunk_id.to_owned()) {
                if let Some(chunk) = chunk_by_id.get(chunk_id) {
                    selected.push((chunk.chunk_id.0.clone(), chunk.text.clone()));
                    if selected.len() >= MAX_RELATION_SEED_CHUNKS {
                        return selected;
                    }
                }
            }
        }
        for entity_id in [&relation.source_entity_id.0, &relation.target_entity_id.0] {
            let Some(profile) = profile_by_entity.get(entity_id.as_str()) else {
                continue;
            };
            for chunk_id in &profile.chunk_ids {
                if seen.insert(chunk_id.clone()) {
                    if let Some(chunk) = chunk_by_id.get(chunk_id.as_str()) {
                        selected.push((chunk.chunk_id.0.clone(), chunk.text.clone()));
                        if selected.len() >= MAX_RELATION_SEED_CHUNKS {
                            return selected;
                        }
                    }
                }
            }
        }
    }

    if selected.is_empty() && !archive.relations.is_empty() {
        for chunk in archive.chunks.iter().take(MAX_RELATION_SEED_CHUNKS) {
            selected.push((chunk.chunk_id.0.clone(), chunk.text.clone()));
        }
    }
    selected
}

fn build_profile_surface_index<'a>(
    profiles: &'a [RelationEntityProfile],
) -> FxHashMap<String, Vec<&'a RelationEntityProfile>> {
    let mut by_surface = FxHashMap::<String, Vec<&'a RelationEntityProfile>>::default();
    for profile in profiles {
        let canonical = normalize_relation_surface(&profile.canonical_name);
        if !canonical.is_empty() {
            by_surface.entry(canonical).or_default().push(profile);
        }
        for alias in &profile.aliases {
            let normalized = normalize_relation_surface(alias);
            if normalized.is_empty() {
                continue;
            }
            by_surface.entry(normalized).or_default().push(profile);
        }
    }
    by_surface
}

fn choose_seed_profile<'a>(
    profiles: &[&'a RelationEntityProfile],
    label: &str,
) -> Option<&'a RelationEntityProfile> {
    let mut candidates = profiles
        .iter()
        .copied()
        .filter(|profile| profile_kind_matches_seed_label(profile.kind.as_ref(), label))
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        candidates = profiles.to_vec();
    }
    candidates.into_iter().max_by(|left, right| {
        left.continuity_score_millis
            .cmp(&right.continuity_score_millis)
            .then_with(|| left.mention_count.cmp(&right.mention_count))
            .then_with(|| left.canonical_name.len().cmp(&right.canonical_name.len()))
    })
}

fn profile_kind_matches_seed_label(kind: Option<&EntityKind>, label: &str) -> bool {
    match label {
        "person" => matches!(kind, Some(EntityKind::Character | EntityKind::Npc)),
        "organization" => matches!(kind, Some(EntityKind::Organization | EntityKind::Faction)),
        "location" => matches!(kind, Some(EntityKind::Location)),
        _ => false,
    }
}

fn relation_seed_kind_label(kind: &EntityKind) -> &'static str {
    match kind {
        EntityKind::Character | EntityKind::Npc => "person",
        EntityKind::Organization | EntityKind::Faction => "organization",
        EntityKind::Location => "location",
        _ => "other",
    }
}

fn dedupe_relation_mentions(mentions: &mut Vec<RelationMention>) {
    let mut seen = BTreeSet::new();
    mentions.retain(|mention| {
        seen.insert((
            mention.entity_id.0.clone(),
            mention.sentence_index,
            mention.span_start,
            mention.span_end,
        ))
    });
    mentions.sort_by(|left, right| {
        left.span_start
            .cmp(&right.span_start)
            .then_with(|| left.entity_id.0.cmp(&right.entity_id.0))
    });
}

fn mention_sentence_index(archive: &DocumentArchive, position: usize) -> usize {
    archive
        .sentences
        .iter()
        .find(|sentence| {
            position >= sentence.range.start as usize && position < sentence.range.end as usize
        })
        .map(|sentence| sentence.index)
        .unwrap_or_default()
}

fn is_relation_anchor_surface(
    surface: &str,
    kind: Option<&EntityKind>,
    allow_compact_alias: bool,
) -> bool {
    let trimmed = surface.trim();
    if trimmed.is_empty() {
        return false;
    }
    let normalized = trimmed.to_ascii_lowercase();
    let blocked = [
        "he", "she", "it", "they", "them", "him", "her", "his", "hers", "their", "theirs", "i",
        "me", "my", "mine", "you", "your", "yours", "we", "us", "our", "ours", "this", "that",
        "these", "those",
    ];
    let generic = [
        "hero",
        "heroes",
        "villain",
        "villains",
        "monster",
        "monsters",
        "criminal",
        "criminals",
        "city",
        "town",
        "team",
        "group",
        "member",
        "members",
        "people",
        "person",
        "security",
        "driving",
        "chapter",
        "genius",
        "guard",
        "guards",
        "officer",
        "officers",
        "doctor",
        "chief",
        "teacher",
        "father",
        "mother",
        "adventure",
        "comedy",
        "mystery",
        "tragedy",
        "sci-fi",
        "scifi",
        "finally",
        "alright",
        "understood",
    ];
    if blocked.contains(&normalized.as_str()) {
        return false;
    }
    if generic.contains(&normalized.as_str()) {
        return false;
    }
    if normalized.len() <= 2 && !allow_compact_alias && kind != Some(&EntityKind::Item) {
        return false;
    }
    if kind == Some(&EntityKind::Location)
        && trimmed.split_whitespace().count() < 2
        && !allow_compact_alias
        && !has_strong_named_shape(trimmed)
    {
        return false;
    }
    let has_upper = trimmed.chars().any(|ch| ch.is_uppercase());
    let multi_word = trimmed.split_whitespace().nth(1).is_some();
    let alphabetic = trimmed.chars().filter(|ch| ch.is_alphabetic()).count();
    has_upper || multi_word || alphabetic >= 5 || (allow_compact_alias && alphabetic >= 2)
}

fn is_relation_salient_surface(surface: &str, kind: Option<&EntityKind>) -> bool {
    is_relation_anchor_surface(surface, kind, false)
}

fn dedupe_window_entities(
    entities: &mut Vec<RelationMention>,
    profile_by_entity: &FxHashMap<String, &RelationEntityProfile>,
) {
    let mut by_entity = FxHashMap::<String, RelationMention>::default();
    for entity in entities.drain(..) {
        let quality = profile_by_entity
            .get(&entity.entity_id.0)
            .map(|profile| profile.continuity_score_millis)
            .unwrap_or_default();
        match by_entity.get(&entity.entity_id.0) {
            Some(existing) => {
                let existing_quality = profile_by_entity
                    .get(&existing.entity_id.0)
                    .map(|profile| profile.continuity_score_millis)
                    .unwrap_or_default();
                if quality > existing_quality || entity.surface.len() > existing.surface.len() {
                    by_entity.insert(entity.entity_id.0.clone(), entity);
                }
            }
            None => {
                by_entity.insert(entity.entity_id.0.clone(), entity);
            }
        }
    }
    *entities = by_entity.into_values().collect::<Vec<_>>();
    entities.sort_by(|left, right| left.span_start.cmp(&right.span_start));
}

fn build_persisted_relations(archives: &[DocumentArchive]) -> Vec<SemanticRelationRecord> {
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    for archive in archives {
        for relation in &archive.relations {
            let key = (
                relation.source_entity_id.0.clone(),
                relation.target_entity_id.0.clone(),
                relation.edge_type.clone(),
                relation.chunk_id.clone().unwrap_or_default(),
            );
            if seen.insert(key) {
                rows.push(relation.clone());
            }
        }
    }
    rows
}

fn build_review_cases(
    scope: &ScopeKey,
    scope_key: &str,
    scope_ord: ScopeOrd,
    session_id: Option<SessionId>,
    windows: &[RelationWindowRecord],
    profiles: &[RelationEntityProfile],
) -> Vec<RelationReviewCase> {
    let profile_by_entity = profiles
        .iter()
        .map(|profile| (profile.entity_id.0.clone(), profile))
        .collect::<FxHashMap<_, _>>();
    let config = GlirelProposalConfig::default();
    let mut cases = Vec::new();

    for window in windows {
        if looks_like_structural_window(&window.text) {
            continue;
        }
        let glirel_entities = window
            .entities
            .iter()
            .map(|entity| GlirelEntity {
                text: entity.surface.clone(),
                entity_type: entity.entity_type.clone(),
                span_start: entity
                    .span_start
                    .saturating_sub(window.range.start as usize),
                span_end: entity.span_end.saturating_sub(window.range.start as usize),
                entity_id: Some(entity.entity_id.0.clone()),
            })
            .collect::<Vec<_>>();
        let seeds = seed_relation_pairs(&window.text, &glirel_entities, &config);
        for seed in seeds {
            let Some(source) = window.entities.get(seed.head_index) else {
                continue;
            };
            let Some(target) = window.entities.get(seed.tail_index) else {
                continue;
            };
            if !relation_window_entity_allowed(source) || !relation_window_entity_allowed(target) {
                continue;
            }
            let source_profile = profile_by_entity.get(&source.entity_id.0);
            let target_profile = profile_by_entity.get(&target.entity_id.0);
            let source_name = source_profile
                .map(|profile| profile.canonical_name.clone())
                .unwrap_or_else(|| source.surface.clone());
            let target_name = target_profile
                .map(|profile| profile.canonical_name.clone())
                .unwrap_or_else(|| target.surface.clone());
            if !relation_pair_supported_by_window(
                &window.text,
                source,
                target,
                window.range.start as usize,
                &window.candidate_relation_types,
            ) {
                continue;
            }
            let continuity_bonus = source_profile
                .map(|profile| profile.continuity_score_millis)
                .unwrap_or_default()
                + target_profile
                    .map(|profile| profile.continuity_score_millis)
                    .unwrap_or_default();
            let seed_score_millis = seed.score_millis + continuity_bonus.min(400) / 4;
            cases.push(RelationReviewCase {
                case_id: format!(
                    "{}::{}::{}::{}::{}",
                    window.document_id,
                    window.revision,
                    window.window_index,
                    source.entity_id.0,
                    target.entity_id.0
                ),
                scope: scope.clone(),
                scope_key: scope_key.to_owned(),
                scope_ord,
                session_id: session_id.clone(),
                document_id: window.document_id.clone(),
                revision: window.revision,
                window_id: window.window_id.clone(),
                window_index: window.window_index,
                window_range: window.range,
                sentence_indices: window.sentence_indices.clone(),
                chunk_ids: window.chunk_ids.clone(),
                window_text: window.text.clone(),
                source_entity_id: source.entity_id.clone(),
                target_entity_id: target.entity_id.clone(),
                source_name,
                target_name,
                source_kind: source.kind.clone(),
                target_kind: target.kind.clone(),
                seed_score_millis,
                seed_evidence: seed
                    .evidence
                    .into_iter()
                    .chain(
                        window.candidate_relation_types.iter().map(|relation_type| {
                            format!("candidate_relation_type:{relation_type}")
                        }),
                    )
                    .chain(
                        window
                            .evidence_labels
                            .iter()
                            .map(|label| format!("candidate_evidence:{label}")),
                    )
                    .chain([format!(
                        "continuity_bonus:{}",
                        continuity_bonus.min(400) / 4
                    )])
                    .collect(),
                serialized: serialize_case_like(
                    &window.text,
                    &source.surface,
                    &target.surface,
                    source.kind.as_ref(),
                    target.kind.as_ref(),
                ),
                blocking_keys: vec![
                    format!("source:{}", source.entity_id.0),
                    format!("target:{}", target.entity_id.0),
                ],
                glirel_predictions: Vec::new(),
                accepted_relations: Vec::new(),
                decision_status: "relation_pending".to_owned(),
            });
        }
    }

    cases.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    cases
}

#[derive(Default)]
struct ErPatchView {
    entity_by_mention: FxHashMap<String, String>,
    kind_by_mention: FxHashMap<String, EntityKind>,
    kind_by_entity: FxHashMap<String, EntityKind>,
}

impl ErPatchView {
    fn from_sidecar(sidecar: Option<&ErScopePatchSidecar>) -> Self {
        let mut view = Self::default();
        let Some(sidecar) = sidecar else {
            return view;
        };
        for link in &sidecar.entity_links {
            if let Some(mention_id) = &link.mention_id {
                view.entity_by_mention
                    .insert(mention_id.0.clone(), link.entity_id.0.clone());
            }
        }
        for override_row in &sidecar.type_overrides {
            view.kind_by_entity
                .insert(override_row.entity_id.0.clone(), override_row.kind.clone());
            if let Some(mention_id) = &override_row.mention_id {
                view.kind_by_mention
                    .insert(mention_id.0.clone(), override_row.kind.clone());
            }
        }
        view
    }
}

fn serialize_profile(entity: &SemanticEntityRecord) -> String {
    serialize_profile_like(
        &entity.canonical_name,
        &entity.aliases,
        entity.kind.as_ref(),
        &[],
    )
}

fn serialize_profile_like(
    canonical_name: &str,
    aliases: &[String],
    kind: Option<&EntityKind>,
    document_ids: &[String],
) -> String {
    let mut parts = vec![canonical_name.to_owned()];
    if let Some(kind) = kind {
        parts.push(format!("kind:{kind:?}"));
    }
    if !aliases.is_empty() {
        parts.push(format!("aliases:{}", aliases.join(" | ")));
    }
    if !document_ids.is_empty() {
        parts.push(format!("documents:{}", document_ids.join(",")));
    }
    parts.join(" ")
}

fn serialize_case_like(
    window_text: &str,
    source_surface: &str,
    target_surface: &str,
    source_kind: Option<&EntityKind>,
    target_kind: Option<&EntityKind>,
) -> String {
    let mut parts = vec![
        format!("source:{source_surface}"),
        format!("target:{target_surface}"),
        format!("context:{window_text}"),
    ];
    if let Some(kind) = source_kind {
        parts.push(format!("sourceKind:{kind:?}"));
    }
    if let Some(kind) = target_kind {
        parts.push(format!("targetKind:{kind:?}"));
    }
    parts.join(" ")
}

fn filter_relation_predictions(
    window_text: &str,
    entities: &[RelationWindowEntity],
    window_start: usize,
    relation_specs: &[GlirelRelationTypeSpec],
    predictions: Vec<GlirelRelationPrediction>,
) -> Vec<GlirelRelationPrediction> {
    let spec_by_label = relation_specs
        .iter()
        .map(|spec| (spec.label.as_str(), spec))
        .collect::<FxHashMap<_, _>>();
    let mut best_by_key = BTreeMap::<(usize, usize, String), GlirelRelationPrediction>::new();

    for mut prediction in predictions {
        let Some(spec) = spec_by_label.get(prediction.relation.as_str()) else {
            continue;
        };
        let Some(source) = entities.get(prediction.head_index) else {
            continue;
        };
        let Some(target) = entities.get(prediction.tail_index) else {
            continue;
        };
        if source.entity_id == target.entity_id
            || !relation_window_entity_allowed(source)
            || !relation_window_entity_allowed(target)
        {
            continue;
        }
        let score_millis = (prediction.confidence * 1000.0).round() as i32;
        if score_millis < spec.review_threshold_millis as i32 {
            continue;
        }
        if !relation_prediction_supported(
            window_text,
            prediction.relation.as_str(),
            source,
            target,
            window_start,
        ) {
            continue;
        }
        prediction
            .evidence
            .push(format!("family_threshold:{}", spec.review_threshold_millis));
        let key = (
            prediction.head_index,
            prediction.tail_index,
            prediction.relation.clone(),
        );
        match best_by_key.get(&key) {
            Some(existing) if existing.confidence >= prediction.confidence => {}
            _ => {
                best_by_key.insert(key, prediction);
            }
        }
    }

    best_by_key.into_values().collect()
}

fn relation_window_entity_allowed(entity: &RelationWindowEntity) -> bool {
    if !is_relation_anchor_surface(&entity.surface, entity.kind.as_ref(), true) {
        return false;
    }
    matches!(
        entity.kind,
        Some(
            EntityKind::Character
                | EntityKind::Npc
                | EntityKind::Organization
                | EntityKind::Faction
                | EntityKind::Location
        )
    )
}

fn relation_pair_supported_by_window(
    text: &str,
    source: &RelationWindowEntity,
    target: &RelationWindowEntity,
    window_start: usize,
    candidate_relation_types: &[String],
) -> bool {
    pair_matches_relation_kinds(source.kind.as_ref(), target.kind.as_ref())
        && (candidate_relation_types.iter().any(|relation| {
            relation_prediction_supported(text, relation, source, target, window_start)
        }) || candidate_relation_types.iter().any(|relation| {
            relation_kind_supported(relation, source.kind.as_ref(), target.kind.as_ref())
        }))
}

fn relation_prediction_supported(
    text: &str,
    relation: &str,
    source: &RelationWindowEntity,
    target: &RelationWindowEntity,
    window_start: usize,
) -> bool {
    relation_kind_supported(relation, source.kind.as_ref(), target.kind.as_ref())
        && text_supports_relation(
            text,
            relation,
            &source.surface,
            &target.surface,
            local_entity_span(source, window_start),
            local_entity_span(target, window_start),
        )
}

fn relation_kind_supported(
    relation: &str,
    source_kind: Option<&EntityKind>,
    target_kind: Option<&EntityKind>,
) -> bool {
    match relation {
        "works_for" | "member_of" => {
            matches!(source_kind, Some(EntityKind::Character | EntityKind::Npc))
                && matches!(
                    target_kind,
                    Some(EntityKind::Organization | EntityKind::Faction)
                )
        }
        "located_in" => {
            matches!(
                source_kind,
                Some(
                    EntityKind::Character
                        | EntityKind::Npc
                        | EntityKind::Organization
                        | EntityKind::Faction
                )
            ) && matches!(target_kind, Some(EntityKind::Location))
        }
        "commands" => {
            matches!(
                source_kind,
                Some(
                    EntityKind::Character
                        | EntityKind::Npc
                        | EntityKind::Organization
                        | EntityKind::Faction
                )
            ) && matches!(
                target_kind,
                Some(
                    EntityKind::Character
                        | EntityKind::Npc
                        | EntityKind::Organization
                        | EntityKind::Faction
                )
            )
        }
        "protects" => {
            matches!(
                source_kind,
                Some(
                    EntityKind::Character
                        | EntityKind::Npc
                        | EntityKind::Organization
                        | EntityKind::Faction
                )
            ) && matches!(
                target_kind,
                Some(
                    EntityKind::Character
                        | EntityKind::Npc
                        | EntityKind::Organization
                        | EntityKind::Faction
                        | EntityKind::Location
                )
            )
        }
        _ => false,
    }
}

fn archive_relation_candidate_families_from_entities(
    entities: &[RelationWindowEntity],
) -> Vec<String> {
    let mut labels = BTreeSet::new();
    for (index, left) in entities.iter().enumerate() {
        for right in entities.iter().skip(index + 1) {
            if relation_kind_supported("works_for", left.kind.as_ref(), right.kind.as_ref()) {
                labels.insert("works_for".to_owned());
            }
            if relation_kind_supported("member_of", left.kind.as_ref(), right.kind.as_ref()) {
                labels.insert("member_of".to_owned());
            }
            if relation_kind_supported("located_in", left.kind.as_ref(), right.kind.as_ref()) {
                labels.insert("located_in".to_owned());
            }
            if relation_kind_supported("located_in", right.kind.as_ref(), left.kind.as_ref()) {
                labels.insert("located_in".to_owned());
            }
        }
    }
    labels.into_iter().collect()
}

fn pair_matches_relation_kinds(
    source_kind: Option<&EntityKind>,
    target_kind: Option<&EntityKind>,
) -> bool {
    relation_kind_supported("works_for", source_kind, target_kind)
        || relation_kind_supported("member_of", source_kind, target_kind)
        || relation_kind_supported("located_in", source_kind, target_kind)
        || relation_kind_supported("commands", source_kind, target_kind)
        || relation_kind_supported("protects", source_kind, target_kind)
}

fn text_supports_relation(
    text: &str,
    relation: &str,
    source_surface: &str,
    target_surface: &str,
    source_span: Option<(usize, usize)>,
    target_span: Option<(usize, usize)>,
) -> bool {
    match relation {
        "works_for" => text_between_supports(
            text,
            source_surface,
            target_surface,
            source_span,
            target_span,
            &[
                " works for ",
                " worked for ",
                " working for ",
                " joins ",
                " joined ",
                " serves ",
                " served ",
                " serving ",
                " employed by ",
            ],
            false,
            96,
        ),
        "member_of" => text_between_supports(
            text,
            source_surface,
            target_surface,
            source_span,
            target_span,
            &[" member of ", " part of ", " belongs to ", " under "],
            false,
            96,
        ),
        "located_in" => text_between_supports(
            text,
            source_surface,
            target_surface,
            source_span,
            target_span,
            &[" in ", " at ", " near ", " inside ", " within "],
            true,
            72,
        ),
        "commands" => text_between_supports(
            text,
            source_surface,
            target_surface,
            source_span,
            target_span,
            &[" commands ", " commanded ", " led ", " orders "],
            false,
            96,
        ),
        "protects" => text_between_supports(
            text,
            source_surface,
            target_surface,
            source_span,
            target_span,
            &[
                " protects ",
                " protected ",
                " defends ",
                " defended ",
                " guards ",
            ],
            false,
            96,
        ),
        _ => false,
    }
}

fn text_between_supports(
    text: &str,
    source: &str,
    target: &str,
    source_span: Option<(usize, usize)>,
    target_span: Option<(usize, usize)>,
    cues: &[&str],
    allow_reverse: bool,
    max_between_chars: usize,
) -> bool {
    let normalized_text = text.to_ascii_lowercase();
    let source = source.trim().to_ascii_lowercase();
    let target = target.trim().to_ascii_lowercase();
    text_between_contains_cue(
        &normalized_text,
        &source,
        &target,
        source_span,
        target_span,
        cues,
        max_between_chars,
    ) || (allow_reverse
        && text_between_contains_cue(
            &normalized_text,
            &target,
            &source,
            target_span,
            source_span,
            cues,
            max_between_chars,
        ))
}

fn is_supported_relation_family(relation: &str) -> bool {
    matches!(
        relation,
        "works_for" | "located_in" | "member_of" | "commands" | "protects"
    )
}

fn text_between_contains_cue(
    text: &str,
    left: &str,
    right: &str,
    left_span: Option<(usize, usize)>,
    right_span: Option<(usize, usize)>,
    cues: &[&str],
    max_between_chars: usize,
) -> bool {
    let spans = match (left_span, right_span) {
        (Some((left_start, left_end)), Some((right_start, _)))
            if left_end <= right_start && right_start <= text.len() =>
        {
            Some((left_start, left_end, right_start))
        }
        _ => None,
    };
    let between = if let Some((left_start, left_end, right_start)) = spans {
        let left_matches = text
            .get(left_start..left_end)
            .map(|slice| slice == left)
            .unwrap_or(false);
        let right_end = right_start.saturating_add(right.len());
        let right_matches = text
            .get(right_start..right_end)
            .map(|slice| slice == right)
            .unwrap_or(false);
        if left_matches && right_matches {
            text.get(left_end..right_start)
        } else {
            None
        }
    } else {
        let Some(left_start) = text.find(left) else {
            return false;
        };
        let search_start = left_start + left.len();
        let Some(relative_right_start) = text[search_start..].find(right) else {
            return false;
        };
        let right_start = search_start + relative_right_start;
        text.get(search_start..right_start)
    };
    between
        .filter(|between| between.len() <= max_between_chars)
        .is_some_and(|between| cues.iter().any(|cue| between.contains(cue)))
}

fn local_entity_span(entity: &RelationWindowEntity, window_start: usize) -> Option<(usize, usize)> {
    let start = entity.span_start.checked_sub(window_start)?;
    let end = entity.span_end.checked_sub(window_start)?;
    if end > start {
        Some((start, end))
    } else {
        None
    }
}

fn infer_window_relation_types(
    text: &str,
    entities: &[RelationMention],
    window_start: usize,
    continuity_relations: &[String],
) -> Vec<String> {
    infer_window_relation_types_from_window_entities(
        text,
        &entities
            .iter()
            .map(relation_window_entity_from_mention)
            .collect::<Vec<_>>(),
        window_start,
        continuity_relations,
    )
}

fn infer_window_relation_types_from_window_entities(
    text: &str,
    entities: &[RelationWindowEntity],
    window_start: usize,
    continuity_relations: &[String],
) -> Vec<String> {
    let mut labels = BTreeSet::new();
    for relation in [
        "works_for",
        "located_in",
        "member_of",
        "commands",
        "protects",
    ] {
        let supported = entities.iter().enumerate().any(|(left_index, left)| {
            entities.iter().skip(left_index + 1).any(|right| {
                relation_prediction_supported(text, relation, left, right, window_start)
                    || relation_prediction_supported(text, relation, right, left, window_start)
            })
        });
        if supported {
            labels.insert(relation.to_owned());
        }
    }
    labels.extend(
        continuity_relations
            .iter()
            .filter(|relation| is_supported_relation_family(relation))
            .cloned(),
    );
    labels.into_iter().collect()
}

fn relation_window_entity_from_mention(mention: &RelationMention) -> RelationWindowEntity {
    relation_window_entity_from_mention_like(mention, 0)
}

fn relation_window_entity_from_mention_like(
    mention: &RelationMention,
    window_start: usize,
) -> RelationWindowEntity {
    RelationWindowEntity {
        entity_id: mention.entity_id.clone(),
        surface: mention.surface.clone(),
        kind: mention.kind.clone(),
        entity_type: mention
            .kind
            .as_ref()
            .map(|kind| format!("{kind:?}"))
            .unwrap_or_else(|| "Unknown".to_owned()),
        span_start: mention.span_start.saturating_sub(window_start),
        span_end: mention.span_end.saturating_sub(window_start),
        sentence_index: mention.sentence_index,
        mention_index: mention.mention_index,
    }
}

fn looks_like_structural_window(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("# ")
        || trimmed.starts_with("## ")
        || trimmed
            .to_ascii_lowercase()
            .starts_with("table of contents")
}

fn has_strong_named_shape(surface: &str) -> bool {
    let tokens = surface.split_whitespace().collect::<Vec<_>>();
    !tokens.is_empty()
        && tokens.iter().all(|token| {
            token
                .chars()
                .next()
                .is_some_and(|value| value.is_uppercase())
        })
}

fn profile_blocking_keys(entity: &SemanticEntityRecord) -> Vec<String> {
    let mut keys = vec![format!("entity:{}", entity.entity_id.0)];
    keys.push(format!(
        "canonical:{}",
        entity.canonical_name.to_lowercase().replace(' ', "_")
    ));
    if let Some(kind) = entity.kind.as_ref() {
        keys.push(format!("kind:{kind:?}").to_lowercase());
    }
    keys
}
