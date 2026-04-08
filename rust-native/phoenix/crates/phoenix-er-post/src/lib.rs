pub mod api;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::path::{Path, PathBuf};

use candle_core::{Device, IndexOp, Tensor};
use candle_onnx::{read_file, simple_eval};
use phoenix_semantic_v2::{
    scope_storage_key, CompactResolutionKind, CorefClusterRecord, DirtyScopeRecord,
    DocumentArchive, DocumentRevisionRef, ErAliasAddition, ErDecisionOutcome, ErDecisionRecord,
    ErEntityLinkOverride, ErScopePatchSidecar, ErTypeOverride, MentionId, ResolvedMention,
    ScopeLexSidecar, ScopeOrd, SemanticEntityRecord, SessionArchive,
};
use phoenix_store_native_core::{PhoenixArchiveStoreV2, PhoenixErPatchStore, StoreError};
use phoenix_types::{EntityId, EntityKind, MentionSpan, ScopeKey, SessionId, TextRange};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErReviewCaseKind {
    UnresolvedMention,
    AmbiguousMention,
    TypeDisagreement,
    AliasObservation,
    CorefConflict,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErCandidateRef {
    pub entity_id: EntityId,
    pub source: String,
    pub score_millis: i32,
    #[serde(default)]
    pub evidence: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErReviewCase {
    pub case_id: String,
    pub kind: ErReviewCaseKind,
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub document_id: String,
    pub revision: u64,
    pub mention_id: Option<MentionId>,
    pub mention_index: Option<usize>,
    pub surface: String,
    pub normalized_surface: String,
    pub mention_kind: Option<EntityKind>,
    pub resolved_entity_id: Option<EntityId>,
    pub resolved_entity_kind: Option<EntityKind>,
    pub decision_kind: Option<CompactResolutionKind>,
    pub decision_status: String,
    pub confidence_millis: u32,
    pub margin_millis: u32,
    #[serde(default)]
    pub candidates: Vec<ErCandidateRef>,
    #[serde(default)]
    pub lexical_candidates: Vec<ErCandidateRef>,
    #[serde(default)]
    pub embedding_candidates: Vec<ErCandidateRef>,
    #[serde(default)]
    pub fused_candidates: Vec<ErCandidateRef>,
    pub chunk_id: Option<String>,
    pub context: String,
    pub serialized: String,
    #[serde(default)]
    pub blocking_keys: Vec<String>,
}

impl Default for ErReviewCase {
    fn default() -> Self {
        Self {
            case_id: String::new(),
            kind: ErReviewCaseKind::UnresolvedMention,
            scope: ScopeKey::default(),
            scope_key: String::new(),
            scope_ord: ScopeOrd::default(),
            session_id: None,
            document_id: String::new(),
            revision: 0,
            mention_id: None,
            mention_index: None,
            surface: String::new(),
            normalized_surface: String::new(),
            mention_kind: None,
            resolved_entity_id: None,
            resolved_entity_kind: None,
            decision_kind: None,
            decision_status: String::new(),
            confidence_millis: 0,
            margin_millis: 0,
            candidates: Vec::new(),
            lexical_candidates: Vec::new(),
            embedding_candidates: Vec::new(),
            fused_candidates: Vec::new(),
            chunk_id: None,
            context: String::new(),
            serialized: String::new(),
            blocking_keys: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErEntityProfile {
    pub entity_id: EntityId,
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub canonical_name: String,
    pub normalized_canonical: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub kind: Option<EntityKind>,
    pub mention_count: usize,
    #[serde(default)]
    pub document_ids: Vec<String>,
    #[serde(default)]
    pub chunk_ids: Vec<String>,
    pub serialized: String,
    #[serde(default)]
    pub blocking_keys: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErScopeReviewBatch {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub dirty: Option<DirtyScopeRecord>,
    #[serde(default)]
    pub document_refs: Vec<DocumentRevisionRef>,
    #[serde(default)]
    pub review_cases: Vec<ErReviewCase>,
    #[serde(default)]
    pub entity_profiles: Vec<ErEntityProfile>,
    pub lexical_generation: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErLexicalCandidateSummary {
    pub case_count: usize,
    pub matched_case_count: usize,
    pub total_candidate_count: usize,
    pub exact_match_case_count: usize,
    pub token_overlap_case_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErEmbeddingCandidateSummary {
    pub case_count: usize,
    pub matched_case_count: usize,
    pub total_candidate_count: usize,
    pub exact_match_case_count: usize,
    pub semantic_support_case_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErFusedCandidateSummary {
    pub case_count: usize,
    pub matched_case_count: usize,
    pub total_candidate_count: usize,
    pub exact_match_case_count: usize,
    pub embedding_only_case_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErLexicalMetrics {
    pub case_count: usize,
    pub cases_with_candidates: usize,
    pub total_candidate_count: usize,
    pub average_candidates_per_case: f32,
    pub alias_case_count: usize,
    pub alias_exact_hit_count: usize,
    pub alias_exact_hit_rate: f32,
    pub type_disagreement_case_count: usize,
    pub type_disagreement_rescue_count: usize,
    pub type_disagreement_rescue_rate: f32,
    pub unresolved_case_count: usize,
    pub ambiguous_case_count: usize,
    pub unresolved_or_ambiguous_case_count: usize,
    pub unresolved_or_ambiguous_covered_count: usize,
    pub unresolved_or_ambiguous_coverage_rate: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErRetrievalComparison {
    pub case_count: usize,
    pub lexical_cases_with_candidates: usize,
    pub lexical_total_candidate_count: usize,
    pub lexical_average_candidates_per_case: f32,
    pub lexical_alias_exact_hit_rate: f32,
    pub lexical_type_disagreement_rescue_rate: f32,
    pub lexical_unresolved_or_ambiguous_coverage_rate: f32,
    pub union_cases_with_candidates: usize,
    pub union_total_candidate_count: usize,
    pub union_average_candidates_per_case: f32,
    pub union_alias_exact_hit_rate: f32,
    pub union_type_disagreement_rescue_rate: f32,
    pub union_unresolved_or_ambiguous_coverage_rate: f32,
    pub additional_cases_covered: isize,
    pub additional_candidates: isize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErCaseSmokeSummary {
    pub case_id: String,
    pub kind: String,
    pub surface: String,
    pub normalized_surface: String,
    pub document_id: String,
    pub decision_status: String,
    pub mention_kind: Option<String>,
    pub resolved_entity_id: Option<EntityId>,
    pub lexical_candidate_count: usize,
    pub embedding_candidate_count: usize,
    pub fused_candidate_count: usize,
    pub top_candidate: Option<ErCandidateRef>,
    pub top_embedding_candidate: Option<ErCandidateRef>,
    pub top_fused_candidate: Option<ErCandidateRef>,
    pub top_candidate_exact: bool,
    pub top_candidate_kind_match: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ErDecisionKind {
    Link,
    ConfirmAlias,
    PatchType,
    Defer,
    Reject,
}

impl Default for ErDecisionKind {
    fn default() -> Self {
        Self::Defer
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErDecision {
    pub case_id: String,
    pub kind: ErDecisionKind,
    pub entity_id: Option<EntityId>,
    pub patched_kind: Option<EntityKind>,
    pub score_millis: i32,
    pub rationale: String,
    #[serde(default)]
    pub evidence: Vec<String>,
}

pub fn derive_scope_review_batch(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    sidecar: Option<&ScopeLexSidecar>,
) -> ErScopeReviewBatch {
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

    let entity_profiles = build_entity_profiles(archives);
    let review_cases = build_review_cases(archives, &entity_profiles);

    ErScopeReviewBatch {
        scope,
        scope_key,
        scope_ord,
        session_id,
        dirty: dirty.cloned(),
        document_refs,
        review_cases,
        entity_profiles,
        lexical_generation: sidecar.map(|value| value.generation),
    }
}

pub fn derive_scope_review_batch_from_store<S: PhoenixArchiveStoreV2>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
) -> Result<ErScopeReviewBatch, StoreError> {
    let archives = store.load_latest_document_archives(Some(&dirty.scope))?;
    let sidecar = store.load_scope_sidecar(&dirty.scope)?;
    Ok(derive_scope_review_batch(
        &archives,
        session,
        Some(dirty),
        sidecar.as_ref(),
    ))
}

pub fn derive_dirty_scope_review_batches<S: PhoenixArchiveStoreV2>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<ErScopeReviewBatch>, StoreError> {
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

pub fn derive_scope_review_batch_from_store_with_replay<S>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
) -> Result<ErScopeReviewBatch, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore,
{
    let mut batch = derive_scope_review_batch_from_store(store, dirty, session)?;
    if let Some(sidecar) = store.load_er_patch_sidecar(&dirty.scope)? {
        apply_er_patch_sidecar(&mut batch, &sidecar);
    }
    Ok(batch)
}

pub fn derive_dirty_scope_review_batches_with_replay<S>(
    store: &S,
    session_id: Option<&SessionId>,
) -> Result<Vec<ErScopeReviewBatch>, StoreError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore,
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
            derive_scope_review_batch_from_store_with_replay(store, &record, session.as_ref())
        })
        .collect()
}

pub fn build_er_patch_sidecar(
    batch: &ErScopeReviewBatch,
    decisions: &[ErDecision],
    created_at: i64,
) -> ErScopePatchSidecar {
    let case_by_id = batch
        .review_cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect::<FxHashMap<_, _>>();
    let mut sidecar = ErScopePatchSidecar {
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
        sidecar.decisions.push(ErDecisionRecord {
            case_id: case.case_id.clone(),
            document_id: case.document_id.clone(),
            mention_id: case.mention_id.clone(),
            outcome: decision_outcome_from_kind(&decision.kind),
            entity_id: decision.entity_id.clone(),
            patched_kind: decision.patched_kind.clone(),
            score_millis: decision.score_millis,
            rationale: decision.rationale.clone(),
            evidence: decision.evidence.clone(),
            surface: case.surface.clone(),
            normalized_surface: case.normalized_surface.clone(),
            reviewed_at: created_at,
        });

        match decision.kind {
            ErDecisionKind::ConfirmAlias => {
                if let Some(entity_id) = decision.entity_id.clone() {
                    sidecar.alias_additions.push(ErAliasAddition {
                        case_id: case.case_id.clone(),
                        document_id: case.document_id.clone(),
                        mention_id: case.mention_id.clone(),
                        entity_id,
                        alias_surface: case.surface.clone(),
                        normalized: case.normalized_surface.clone(),
                        confidence_millis: decision.score_millis.max(0) as u32,
                        created_at,
                    });
                }
            }
            ErDecisionKind::PatchType => {
                if let (Some(entity_id), Some(kind)) =
                    (decision.entity_id.clone(), decision.patched_kind.clone())
                {
                    sidecar.type_overrides.push(ErTypeOverride {
                        case_id: case.case_id.clone(),
                        document_id: case.document_id.clone(),
                        mention_id: case.mention_id.clone(),
                        entity_id,
                        kind,
                        confidence_millis: decision.score_millis.max(0) as u32,
                        created_at,
                    });
                }
            }
            ErDecisionKind::Link => {
                if let Some(entity_id) = decision.entity_id.clone() {
                    sidecar.entity_links.push(ErEntityLinkOverride {
                        case_id: case.case_id.clone(),
                        document_id: case.document_id.clone(),
                        mention_id: case.mention_id.clone(),
                        entity_id,
                        confidence_millis: decision.score_millis.max(0) as u32,
                        created_at,
                    });
                }
            }
            ErDecisionKind::Defer | ErDecisionKind::Reject => {}
        }
    }

    dedupe_er_patch_sidecar(&mut sidecar);
    sidecar
}

pub fn persist_er_patch_sidecar<S>(
    store: &S,
    batch: &ErScopeReviewBatch,
    decisions: &[ErDecision],
    created_at: i64,
) -> Result<ErScopePatchSidecar, StoreError>
where
    S: PhoenixErPatchStore,
{
    let updates = build_er_patch_sidecar(batch, decisions, created_at);
    let merged = match store.load_er_patch_sidecar(&batch.scope)? {
        Some(existing) => merge_er_patch_sidecars(existing, updates),
        None => updates,
    };
    store.persist_er_patch_sidecar(&merged)?;
    Ok(merged)
}

pub fn apply_er_patch_sidecar(batch: &mut ErScopeReviewBatch, sidecar: &ErScopePatchSidecar) {
    let mut touched_entities = BTreeSet::<String>::new();
    let mut profile_index = batch
        .entity_profiles
        .iter()
        .enumerate()
        .map(|(index, profile)| (profile.entity_id.0.clone(), index))
        .collect::<FxHashMap<_, _>>();

    for alias in &sidecar.alias_additions {
        if let Some(index) = profile_index.get(&alias.entity_id.0).copied() {
            let profile = &mut batch.entity_profiles[index];
            if !profile
                .aliases
                .iter()
                .any(|value| value == &alias.alias_surface)
                && profile.canonical_name != alias.alias_surface
            {
                profile.aliases.push(alias.alias_surface.clone());
                touched_entities.insert(alias.entity_id.0.clone());
            }
        }
    }

    for patch in &sidecar.type_overrides {
        if let Some(index) = profile_index.get(&patch.entity_id.0).copied() {
            let profile = &mut batch.entity_profiles[index];
            if profile.kind.as_ref() != Some(&patch.kind) {
                profile.kind = Some(patch.kind.clone());
                touched_entities.insert(patch.entity_id.0.clone());
            }
        }
    }

    for entity_id in &touched_entities {
        if let Some(index) = profile_index.get(entity_id).copied() {
            refresh_entity_profile(&mut batch.entity_profiles[index]);
        }
    }

    profile_index = batch
        .entity_profiles
        .iter()
        .enumerate()
        .map(|(index, profile)| (profile.entity_id.0.clone(), index))
        .collect::<FxHashMap<_, _>>();
    let decision_by_case = sidecar
        .decisions
        .iter()
        .map(|decision| (decision.case_id.as_str(), decision))
        .collect::<FxHashMap<_, _>>();
    let link_by_case = sidecar
        .entity_links
        .iter()
        .map(|patch| (patch.case_id.as_str(), patch))
        .collect::<FxHashMap<_, _>>();

    for case in &mut batch.review_cases {
        if let Some(decision) = decision_by_case.get(case.case_id.as_str()) {
            case.decision_status = format!("er_{}", er_outcome_name(decision.outcome));
            if let Some(entity_id) = decision.entity_id.clone() {
                case.resolved_entity_id = Some(entity_id);
                case.decision_kind = Some(CompactResolutionKind::Resolved);
            }
            if let Some(kind) = decision.patched_kind.clone() {
                case.resolved_entity_kind = Some(kind);
            }
        }
        if let Some(link) = link_by_case.get(case.case_id.as_str()) {
            case.resolved_entity_id = Some(link.entity_id.clone());
            case.decision_kind = Some(CompactResolutionKind::Resolved);
            ensure_patch_candidate(
                case,
                link.entity_id.clone(),
                "er_patch_link",
                link.confidence_millis as i32,
            );
        }
        if let Some(entity_id) = case.resolved_entity_id.as_ref() {
            if let Some(index) = profile_index.get(&entity_id.0).copied() {
                let profile = &batch.entity_profiles[index];
                case.resolved_entity_kind = profile.kind.clone();
                case.serialized = serialize_case(
                    &case.surface,
                    &case.normalized_surface,
                    case.mention_kind.clone(),
                    Some(profile.canonical_name.as_str()),
                    &case.context,
                );
            }
        }
        case.lexical_candidates.clear();
        case.embedding_candidates.clear();
        case.fused_candidates.clear();
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ErEmbeddingConfig {
    pub model_root: PathBuf,
    pub batch_size: usize,
    pub min_score_millis: i32,
}

impl Default for ErEmbeddingConfig {
    fn default() -> Self {
        Self {
            model_root: default_embedding_model_root(),
            batch_size: 16,
            min_score_millis: 240,
        }
    }
}

pub struct ErEmbeddingModel {
    tokenizer: Tokenizer,
    model: candle_onnx::onnx::ModelProto,
    device: Device,
}

impl ErEmbeddingModel {
    pub fn load(model_root: &Path) -> Result<Self, String> {
        let tokenizer_path = model_root.join("tokenizer.json");
        let model_path = model_root.join("onnx").join("model.onnx");
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|error| {
            format!(
                "failed to load tokenizer {}: {error}",
                tokenizer_path.display()
            )
        })?;
        let model = read_file(&model_path).map_err(|error| {
            format!(
                "failed to load ONNX model {}: {error}",
                model_path.display()
            )
        })?;
        Ok(Self {
            tokenizer,
            model,
            device: Device::Cpu,
        })
    }

    pub fn embed_batched(
        &self,
        texts: &[String],
        batch_size: usize,
    ) -> Result<Vec<Vec<f32>>, String> {
        let batch_size = batch_size.max(1);
        let mut rows = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(batch_size) {
            rows.extend(self.embed(chunk)?);
        }
        Ok(rows)
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = texts
            .iter()
            .map(|text| {
                self.tokenizer
                    .encode(text.as_str(), true)
                    .map_err(|error| format!("failed to encode input: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let max_len = encodings
            .iter()
            .map(|encoding| encoding.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let batch = encodings.len();

        let mut input_ids = vec![0i64; batch * max_len];
        let mut attention_mask = vec![0i64; batch * max_len];
        let mut token_type_ids = vec![0i64; batch * max_len];

        for (row, encoding) in encodings.iter().enumerate() {
            let row_offset = row * max_len;
            for (col, token_id) in encoding.get_ids().iter().enumerate() {
                input_ids[row_offset + col] = *token_id as i64;
            }
            for (col, mask) in encoding.get_attention_mask().iter().enumerate() {
                attention_mask[row_offset + col] = *mask as i64;
            }
            for (col, token_type) in encoding.get_type_ids().iter().enumerate() {
                token_type_ids[row_offset + col] = *token_type as i64;
            }
        }

        let input_shape = (batch, max_len);
        let mut inputs = HashMap::<String, Tensor>::new();
        inputs.insert(
            "input_ids".to_owned(),
            Tensor::from_vec(input_ids, input_shape, &self.device)
                .map_err(|error| format!("failed to build input_ids tensor: {error}"))?,
        );
        inputs.insert(
            "attention_mask".to_owned(),
            Tensor::from_vec(attention_mask, input_shape, &self.device)
                .map_err(|error| format!("failed to build attention_mask tensor: {error}"))?,
        );
        if graph_has_input(&self.model, "token_type_ids") {
            inputs.insert(
                "token_type_ids".to_owned(),
                Tensor::from_vec(token_type_ids, input_shape, &self.device)
                    .map_err(|error| format!("failed to build token_type_ids tensor: {error}"))?,
            );
        }

        let outputs = simple_eval(&self.model, inputs)
            .map_err(|error| format!("failed to evaluate ONNX graph: {error}"))?;
        let hidden = select_hidden_output(&self.model, outputs)?;
        let cls = hidden
            .i((.., 0, ..))
            .map_err(|error| format!("failed to select CLS embedding: {error}"))?;
        let squared = cls
            .sqr()
            .map_err(|error| format!("failed to square embeddings: {error}"))?;
        let summed = squared
            .sum_keepdim(1)
            .map_err(|error| format!("failed to sum embeddings: {error}"))?;
        let norms = summed
            .sqrt()
            .map_err(|error| format!("failed to compute embedding norm: {error}"))?;
        let normalized = cls
            .broadcast_div(&norms)
            .map_err(|error| format!("failed to normalize embeddings: {error}"))?;
        normalized
            .to_vec2::<f32>()
            .map_err(|error| format!("failed to extract embedding rows: {error}"))
    }
}

pub fn default_embedding_model_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(4)
        .expect("workspace root")
        .join("rust-native")
        .join("phoenix-hnsw-smoke")
        .join("models")
        .join("snowflake-arctic-embed-xs")
}

pub fn generate_lexical_candidates(
    batch: &mut ErScopeReviewBatch,
    limit: usize,
) -> ErLexicalCandidateSummary {
    let limit = limit.max(1);
    let prepared = batch
        .entity_profiles
        .iter()
        .map(PreparedLexicalProfile::from_profile)
        .collect::<Vec<_>>();
    let mut summary = ErLexicalCandidateSummary {
        case_count: batch.review_cases.len(),
        ..Default::default()
    };

    for case in &mut batch.review_cases {
        let matches = lexical_candidates_for_case(case, &prepared, limit);
        if !matches.is_empty() {
            summary.matched_case_count += 1;
            summary.total_candidate_count += matches.len();
            if matches.iter().any(|candidate| {
                candidate
                    .evidence
                    .iter()
                    .any(|value| value.starts_with("exact_"))
            }) {
                summary.exact_match_case_count += 1;
            }
            if matches.iter().any(|candidate| {
                candidate
                    .evidence
                    .iter()
                    .any(|value| value.starts_with("token_overlap:"))
            }) {
                summary.token_overlap_case_count += 1;
            }
        }
        case.lexical_candidates = matches;
    }

    summary
}

pub fn generate_embedding_candidates(
    batch: &mut ErScopeReviewBatch,
    model: &ErEmbeddingModel,
    limit: usize,
    config: &ErEmbeddingConfig,
) -> Result<ErEmbeddingCandidateSummary, String> {
    let limit = limit.max(1);
    let prepared = batch
        .entity_profiles
        .iter()
        .map(PreparedEmbeddingProfile::from_profile)
        .collect::<Vec<_>>();
    if prepared.is_empty() || batch.review_cases.is_empty() {
        for case in &mut batch.review_cases {
            case.embedding_candidates.clear();
        }
        return Ok(ErEmbeddingCandidateSummary {
            case_count: batch.review_cases.len(),
            ..Default::default()
        });
    }

    let profile_rows = model.embed_batched(
        &prepared
            .iter()
            .map(|profile| profile.serialized.clone())
            .collect::<Vec<_>>(),
        config.batch_size,
    )?;
    let case_rows = model.embed_batched(
        &batch
            .review_cases
            .iter()
            .map(|case| case.serialized.clone())
            .collect::<Vec<_>>(),
        config.batch_size,
    )?;

    let mut summary = ErEmbeddingCandidateSummary {
        case_count: batch.review_cases.len(),
        ..Default::default()
    };

    for (case, embedding) in batch.review_cases.iter_mut().zip(case_rows.iter()) {
        let matches = embedding_candidates_for_case(
            case,
            embedding,
            &prepared,
            &profile_rows,
            limit,
            config.min_score_millis,
        );
        if !matches.is_empty() {
            summary.matched_case_count += 1;
            summary.total_candidate_count += matches.len();
            if matches.iter().any(|candidate| {
                candidate
                    .evidence
                    .iter()
                    .any(|value| value.starts_with("exact_"))
            }) {
                summary.exact_match_case_count += 1;
            }
            if matches.iter().any(|candidate| {
                candidate
                    .evidence
                    .iter()
                    .any(|value| value.starts_with("embedding_cosine:"))
            }) {
                summary.semantic_support_case_count += 1;
            }
        }
        case.embedding_candidates = matches;
    }

    Ok(summary)
}

pub fn generate_fused_candidates(
    batch: &mut ErScopeReviewBatch,
    limit: usize,
) -> ErFusedCandidateSummary {
    let limit = limit.max(1);
    let mut summary = ErFusedCandidateSummary {
        case_count: batch.review_cases.len(),
        ..Default::default()
    };

    for case in &mut batch.review_cases {
        let matches = fused_candidates(case);
        let embedding_only = matches.iter().any(|candidate| {
            candidate.source == "embedding"
                || (candidate.source == "fused"
                    && candidate
                        .evidence
                        .iter()
                        .any(|value| value.starts_with("embedding_cosine:"))
                    && !candidate_has_exact_match(candidate)
                    && !candidate_has_native_overlap(candidate)
                    && !candidate_has_faction_support(candidate)
                    && !candidate_has_kind_match(candidate))
        });
        let limited = matches.into_iter().take(limit).collect::<Vec<_>>();
        if !limited.is_empty() {
            summary.matched_case_count += 1;
            summary.total_candidate_count += limited.len();
            if limited.iter().any(candidate_has_exact_match) {
                summary.exact_match_case_count += 1;
            }
            if embedding_only {
                summary.embedding_only_case_count += 1;
            }
        }
        case.fused_candidates = limited;
    }

    summary
}

pub fn compute_lexical_metrics(batch: &ErScopeReviewBatch) -> ErLexicalMetrics {
    let case_count = batch.review_cases.len();
    let cases_with_candidates = batch
        .review_cases
        .iter()
        .filter(|case| !case.lexical_candidates.is_empty())
        .count();
    let total_candidate_count = batch
        .review_cases
        .iter()
        .map(|case| case.lexical_candidates.len())
        .sum::<usize>();
    let alias_cases = batch
        .review_cases
        .iter()
        .filter(|case| case.kind == ErReviewCaseKind::AliasObservation)
        .collect::<Vec<_>>();
    let alias_exact_hit_count = alias_cases
        .iter()
        .filter(|case| {
            case.resolved_entity_id.as_ref().is_some_and(|entity_id| {
                case.lexical_candidates.iter().any(|candidate| {
                    &candidate.entity_id == entity_id
                        && candidate
                            .evidence
                            .iter()
                            .any(|value| value.starts_with("exact_alias:"))
                })
            })
        })
        .count();
    let type_cases = batch
        .review_cases
        .iter()
        .filter(|case| case.kind == ErReviewCaseKind::TypeDisagreement)
        .collect::<Vec<_>>();
    let type_disagreement_rescue_count = type_cases
        .iter()
        .filter(|case| {
            case.lexical_candidates.iter().any(|candidate| {
                candidate
                    .evidence
                    .iter()
                    .any(|value| value.starts_with("kind_match:"))
            })
        })
        .count();
    let unresolved_case_count = batch
        .review_cases
        .iter()
        .filter(|case| case.kind == ErReviewCaseKind::UnresolvedMention)
        .count();
    let ambiguous_case_count = batch
        .review_cases
        .iter()
        .filter(|case| case.kind == ErReviewCaseKind::AmbiguousMention)
        .count();
    let unresolved_or_ambiguous_case_count = unresolved_case_count + ambiguous_case_count;
    let unresolved_or_ambiguous_covered_count = batch
        .review_cases
        .iter()
        .filter(|case| {
            matches!(
                case.kind,
                ErReviewCaseKind::UnresolvedMention | ErReviewCaseKind::AmbiguousMention
            ) && !case.lexical_candidates.is_empty()
        })
        .count();

    ErLexicalMetrics {
        case_count,
        cases_with_candidates,
        total_candidate_count,
        average_candidates_per_case: ratio(total_candidate_count, case_count),
        alias_case_count: alias_cases.len(),
        alias_exact_hit_count,
        alias_exact_hit_rate: ratio(alias_exact_hit_count, alias_cases.len()),
        type_disagreement_case_count: type_cases.len(),
        type_disagreement_rescue_count,
        type_disagreement_rescue_rate: ratio(type_disagreement_rescue_count, type_cases.len()),
        unresolved_case_count,
        ambiguous_case_count,
        unresolved_or_ambiguous_case_count,
        unresolved_or_ambiguous_covered_count,
        unresolved_or_ambiguous_coverage_rate: ratio(
            unresolved_or_ambiguous_covered_count,
            unresolved_or_ambiguous_case_count,
        ),
    }
}

pub fn compute_retrieval_comparison(batch: &ErScopeReviewBatch) -> ErRetrievalComparison {
    let lexical = compute_lexical_metrics(batch);
    let case_count = batch.review_cases.len();
    let union_cases_with_candidates = batch
        .review_cases
        .iter()
        .filter(|case| !case.fused_candidates.is_empty())
        .count();
    let union_total_candidate_count = batch
        .review_cases
        .iter()
        .map(|case| case.fused_candidates.len())
        .sum::<usize>();
    let alias_cases = batch
        .review_cases
        .iter()
        .filter(|case| case.kind == ErReviewCaseKind::AliasObservation)
        .collect::<Vec<_>>();
    let union_alias_exact_hit_count = alias_cases
        .iter()
        .filter(|case| {
            case.resolved_entity_id.as_ref().is_some_and(|entity_id| {
                case.fused_candidates.iter().any(|candidate| {
                    &candidate.entity_id == entity_id
                        && candidate
                            .evidence
                            .iter()
                            .any(|value| value.starts_with("exact_alias:"))
                })
            })
        })
        .count();
    let type_cases = batch
        .review_cases
        .iter()
        .filter(|case| case.kind == ErReviewCaseKind::TypeDisagreement)
        .collect::<Vec<_>>();
    let union_type_disagreement_rescue_count = type_cases
        .iter()
        .filter(|case| case.fused_candidates.iter().any(candidate_has_kind_match))
        .count();
    let unresolved_or_ambiguous_covered_count = batch
        .review_cases
        .iter()
        .filter(|case| {
            matches!(
                case.kind,
                ErReviewCaseKind::UnresolvedMention | ErReviewCaseKind::AmbiguousMention
            ) && !case.fused_candidates.is_empty()
        })
        .count();

    ErRetrievalComparison {
        case_count,
        lexical_cases_with_candidates: lexical.cases_with_candidates,
        lexical_total_candidate_count: lexical.total_candidate_count,
        lexical_average_candidates_per_case: lexical.average_candidates_per_case,
        lexical_alias_exact_hit_rate: lexical.alias_exact_hit_rate,
        lexical_type_disagreement_rescue_rate: lexical.type_disagreement_rescue_rate,
        lexical_unresolved_or_ambiguous_coverage_rate: lexical
            .unresolved_or_ambiguous_coverage_rate,
        union_cases_with_candidates,
        union_total_candidate_count,
        union_average_candidates_per_case: ratio(union_total_candidate_count, case_count),
        union_alias_exact_hit_rate: ratio(union_alias_exact_hit_count, alias_cases.len()),
        union_type_disagreement_rescue_rate: ratio(
            union_type_disagreement_rescue_count,
            type_cases.len(),
        ),
        union_unresolved_or_ambiguous_coverage_rate: ratio(
            unresolved_or_ambiguous_covered_count,
            lexical.unresolved_or_ambiguous_case_count,
        ),
        additional_cases_covered: union_cases_with_candidates as isize
            - lexical.cases_with_candidates as isize,
        additional_candidates: union_total_candidate_count as isize
            - lexical.total_candidate_count as isize,
    }
}

pub fn summarize_review_cases(batch: &ErScopeReviewBatch) -> Vec<ErCaseSmokeSummary> {
    batch
        .review_cases
        .iter()
        .map(|case| {
            let top_candidate = case.lexical_candidates.first().cloned();
            let top_embedding_candidate = case.embedding_candidates.first().cloned();
            let top_fused_candidate = case.fused_candidates.first().cloned();
            ErCaseSmokeSummary {
                case_id: case.case_id.clone(),
                kind: review_case_kind_name(&case.kind).to_owned(),
                surface: case.surface.clone(),
                normalized_surface: case.normalized_surface.clone(),
                document_id: case.document_id.clone(),
                decision_status: case.decision_status.clone(),
                mention_kind: case
                    .mention_kind
                    .as_ref()
                    .map(|kind| kind_name(kind).to_owned()),
                resolved_entity_id: case.resolved_entity_id.clone(),
                lexical_candidate_count: case.lexical_candidates.len(),
                embedding_candidate_count: case.embedding_candidates.len(),
                fused_candidate_count: case.fused_candidates.len(),
                top_candidate_exact: top_candidate.as_ref().is_some_and(|candidate| {
                    candidate.evidence.iter().any(|value| {
                        value.starts_with("exact_alias:") || value.starts_with("exact_canonical:")
                    })
                }),
                top_candidate_kind_match: top_candidate.as_ref().is_some_and(|candidate| {
                    candidate
                        .evidence
                        .iter()
                        .any(|value| value.starts_with("kind_match:"))
                }),
                top_candidate,
                top_embedding_candidate,
                top_fused_candidate,
            }
        })
        .collect()
}

pub fn draft_review_decisions(batch: &ErScopeReviewBatch) -> Vec<ErDecision> {
    let profile_by_entity = batch
        .entity_profiles
        .iter()
        .map(|profile| (profile.entity_id.0.as_str(), profile))
        .collect::<FxHashMap<_, _>>();

    batch.review_cases
        .iter()
        .map(|case| {
            let ranked = if case.fused_candidates.is_empty() {
                fused_candidates(case)
            } else {
                case.fused_candidates.clone()
            };
            let top = ranked.first();
            let second_score = ranked.get(1).map(|candidate| candidate.score_millis).unwrap_or(i32::MIN);
            let margin = top
                .map(|candidate| candidate.score_millis - second_score.max(0))
                .unwrap_or(0);
            let exact = top.is_some_and(candidate_has_exact_match);
            let faction_support = top.is_some_and(candidate_has_faction_support);
            let kind_match = top.is_some_and(candidate_has_kind_match);
            let native_overlap = top.is_some_and(candidate_has_native_overlap);
            let semantic_support = top.is_some_and(candidate_has_embedding_support);
            let profile_kind = top
                .and_then(|candidate| profile_by_entity.get(candidate.entity_id.0.as_str()))
                .and_then(|profile| profile.kind.clone());

            match case.kind {
                ErReviewCaseKind::AliasObservation => {
                    if let Some(candidate) = top {
                        if case.resolved_entity_id.as_ref() == Some(&candidate.entity_id) && exact {
                            return ErDecision {
                                case_id: case.case_id.clone(),
                                kind: ErDecisionKind::ConfirmAlias,
                                entity_id: Some(candidate.entity_id.clone()),
                                patched_kind: None,
                                score_millis: candidate.score_millis,
                                rationale: "exact alias matched existing entity".to_owned(),
                                evidence: candidate.evidence.clone(),
                            };
                        }
                        if exact && margin >= 120 && candidate.score_millis >= 900 {
                            return ErDecision {
                                case_id: case.case_id.clone(),
                                kind: ErDecisionKind::Link,
                                entity_id: Some(candidate.entity_id.clone()),
                                patched_kind: None,
                                score_millis: candidate.score_millis,
                                rationale: "exact alias candidate dominated review ranking".to_owned(),
                                evidence: candidate.evidence.clone(),
                            };
                        }
                        return ErDecision {
                            case_id: case.case_id.clone(),
                            kind: ErDecisionKind::Defer,
                            entity_id: Some(candidate.entity_id.clone()),
                            patched_kind: None,
                            score_millis: candidate.score_millis,
                            rationale: "alias candidate exists but needs stronger adjudication".to_owned(),
                            evidence: candidate.evidence.clone(),
                        };
                    }
                    ErDecision {
                        case_id: case.case_id.clone(),
                        kind: ErDecisionKind::Reject,
                        entity_id: None,
                        patched_kind: None,
                        score_millis: 0,
                        rationale: "alias observation had no viable lexical candidate".to_owned(),
                        evidence: Vec::new(),
                    }
                }
                ErReviewCaseKind::TypeDisagreement => {
                    if let Some(candidate) = top {
                        let resolved_overlap =
                            case.resolved_entity_id.as_ref() == Some(&candidate.entity_id);
                        let disagrees_with_mention = match (&case.mention_kind, &profile_kind) {
                            (Some(mention_kind), Some(profile_kind)) => mention_kind != profile_kind,
                            _ => false,
                        };
                        if profile_kind.is_some()
                            && ((kind_match && margin >= 80)
                                || (resolved_overlap && disagrees_with_mention))
                        {
                            return ErDecision {
                                case_id: case.case_id.clone(),
                                kind: ErDecisionKind::PatchType,
                                entity_id: Some(candidate.entity_id.clone()),
                                patched_kind: profile_kind,
                                score_millis: candidate.score_millis,
                                rationale: if kind_match {
                                    "top lexical candidate carries consistent entity kind".to_owned()
                                } else {
                                    "resolved entity profile disagrees with mention kind and suggests a patch".to_owned()
                                },
                                evidence: candidate.evidence.clone(),
                            };
                        }
                        return ErDecision {
                            case_id: case.case_id.clone(),
                            kind: ErDecisionKind::Defer,
                            entity_id: Some(candidate.entity_id.clone()),
                            patched_kind: profile_kind,
                            score_millis: candidate.score_millis,
                            rationale: "type disagreement still needs stronger evidence".to_owned(),
                            evidence: candidate.evidence.clone(),
                        };
                    }
                    ErDecision {
                        case_id: case.case_id.clone(),
                        kind: ErDecisionKind::Defer,
                        entity_id: None,
                        patched_kind: None,
                        score_millis: 0,
                        rationale: "type disagreement has no lexical rescue candidate".to_owned(),
                        evidence: Vec::new(),
                    }
                }
                ErReviewCaseKind::UnresolvedMention
                | ErReviewCaseKind::AmbiguousMention
                | ErReviewCaseKind::CorefConflict => {
                    if let Some(candidate) = top {
                        if exact && margin >= 120 && candidate.score_millis >= 900 {
                            return ErDecision {
                                case_id: case.case_id.clone(),
                                kind: ErDecisionKind::Link,
                                entity_id: Some(candidate.entity_id.clone()),
                                patched_kind: profile_kind,
                                score_millis: candidate.score_millis,
                                rationale: "exact lexical candidate is clearly dominant".to_owned(),
                                evidence: candidate.evidence.clone(),
                            };
                        }
                        if (faction_support || native_overlap || kind_match)
                            && margin >= 80
                            && candidate.score_millis >= 520
                        {
                            return ErDecision {
                                case_id: case.case_id.clone(),
                                kind: ErDecisionKind::Link,
                                entity_id: Some(candidate.entity_id.clone()),
                                patched_kind: profile_kind,
                                score_millis: candidate.score_millis,
                                rationale: "top lexical candidate has stable structural support".to_owned(),
                                evidence: candidate.evidence.clone(),
                            };
                        }
                        if semantic_support && margin >= 60 && candidate.score_millis >= 420 {
                            return ErDecision {
                                case_id: case.case_id.clone(),
                                kind: ErDecisionKind::Link,
                                entity_id: Some(candidate.entity_id.clone()),
                                patched_kind: profile_kind,
                                score_millis: candidate.score_millis,
                                rationale: "embedding support produced a stable top candidate".to_owned(),
                                evidence: candidate.evidence.clone(),
                            };
                        }
                        return ErDecision {
                            case_id: case.case_id.clone(),
                            kind: ErDecisionKind::Defer,
                            entity_id: Some(candidate.entity_id.clone()),
                            patched_kind: profile_kind,
                            score_millis: candidate.score_millis,
                            rationale: "candidate exists but margin is too weak for linking".to_owned(),
                            evidence: candidate.evidence.clone(),
                        };
                    }
                    ErDecision {
                        case_id: case.case_id.clone(),
                        kind: ErDecisionKind::Defer,
                        entity_id: None,
                        patched_kind: None,
                        score_millis: 0,
                        rationale: "no lexical candidate available".to_owned(),
                        evidence: Vec::new(),
                    }
                }
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
struct PreparedLexicalProfile {
    entity_id: EntityId,
    canonical_name: String,
    normalized_canonical: String,
    normalized_aliases: Vec<String>,
    kind: Option<EntityKind>,
    blocking_keys: BTreeSet<String>,
    strong_tokens: BTreeSet<String>,
    faction_like: bool,
    alias_like: bool,
}

impl PreparedLexicalProfile {
    fn from_profile(profile: &ErEntityProfile) -> Self {
        Self {
            entity_id: profile.entity_id.clone(),
            canonical_name: profile.canonical_name.clone(),
            normalized_canonical: profile.normalized_canonical.clone(),
            normalized_aliases: profile
                .aliases
                .iter()
                .map(|value| normalize_surface(value))
                .collect(),
            kind: profile.kind.clone(),
            blocking_keys: profile.blocking_keys.iter().cloned().collect(),
            strong_tokens: strong_tokens(&profile.serialized).into_iter().collect(),
            faction_like: looks_faction_like(&profile.canonical_name)
                || profile
                    .aliases
                    .iter()
                    .any(|alias| looks_faction_like(alias)),
            alias_like: looks_alias_like(&profile.canonical_name)
                || profile.aliases.iter().any(|alias| looks_alias_like(alias)),
        }
    }
}

#[derive(Clone, Debug)]
struct PreparedEmbeddingProfile {
    entity_id: EntityId,
    normalized_canonical: String,
    normalized_aliases: Vec<String>,
    kind: Option<EntityKind>,
    serialized: String,
}

impl PreparedEmbeddingProfile {
    fn from_profile(profile: &ErEntityProfile) -> Self {
        Self {
            entity_id: profile.entity_id.clone(),
            normalized_canonical: profile.normalized_canonical.clone(),
            normalized_aliases: profile
                .aliases
                .iter()
                .map(|value| normalize_surface(value))
                .collect(),
            kind: profile.kind.clone(),
            serialized: profile.serialized.clone(),
        }
    }
}

fn lexical_candidates_for_case(
    case: &ErReviewCase,
    profiles: &[PreparedLexicalProfile],
    limit: usize,
) -> Vec<ErCandidateRef> {
    let case_tokens = strong_tokens(&case.serialized)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let case_keys = case.blocking_keys.iter().cloned().collect::<BTreeSet<_>>();
    let case_faction_like = looks_faction_like(&case.surface);
    let case_alias_like = looks_alias_like(&case.surface);
    let existing_ids = case
        .candidates
        .iter()
        .map(|candidate| candidate.entity_id.0.clone())
        .collect::<BTreeSet<_>>();

    let mut rows = profiles
        .iter()
        .filter_map(|profile| {
            lexical_match(
                case,
                profile,
                &case_tokens,
                &case_keys,
                case_faction_like,
                case_alias_like,
                &existing_ids,
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .score_millis
            .cmp(&left.score_millis)
            .then_with(|| left.entity_id.0.cmp(&right.entity_id.0))
    });
    rows.truncate(limit);
    rows
}

fn lexical_match(
    case: &ErReviewCase,
    profile: &PreparedLexicalProfile,
    case_tokens: &BTreeSet<String>,
    case_keys: &BTreeSet<String>,
    case_faction_like: bool,
    case_alias_like: bool,
    existing_ids: &BTreeSet<String>,
) -> Option<ErCandidateRef> {
    let mut score = 0i32;
    let mut evidence = Vec::<String>::new();

    if !case.normalized_surface.is_empty()
        && case.normalized_surface == profile.normalized_canonical
    {
        score += 980;
        evidence.push(format!("exact_canonical:{}", profile.canonical_name));
    }
    if !case.normalized_surface.is_empty()
        && profile
            .normalized_aliases
            .iter()
            .any(|alias| alias == &case.normalized_surface)
    {
        score += 1000;
        evidence.push(format!("exact_alias:{}", case.normalized_surface));
    }

    let overlap = case_tokens
        .intersection(&profile.strong_tokens)
        .cloned()
        .collect::<Vec<_>>();
    if !overlap.is_empty() {
        score += 240 + (overlap.len() as i32 * 90);
        evidence.push(format!("token_overlap:{}", overlap.join(",")));
    }

    let shared_block_keys = case_keys
        .intersection(&profile.blocking_keys)
        .cloned()
        .collect::<Vec<_>>();
    if !shared_block_keys.is_empty() {
        score += 40 + (shared_block_keys.len() as i32 * 15);
        evidence.push(format!("blocking_overlap:{}", shared_block_keys.join(",")));
    }

    if case_faction_like && profile.faction_like {
        score += 140;
        evidence.push("faction_shape".to_owned());
    }

    if case_alias_like && profile.alias_like {
        score += 70;
        evidence.push("alias_shape".to_owned());
    }

    if let (Some(case_kind), Some(profile_kind)) =
        (case.mention_kind.as_ref(), profile.kind.as_ref())
    {
        if case_kind == profile_kind {
            score += 80;
            evidence.push(format!("kind_match:{}", kind_name(profile_kind)));
        } else {
            score -= 30;
            evidence.push(format!(
                "kind_conflict:{}!={}",
                kind_name(case_kind),
                kind_name(profile_kind)
            ));
        }
    }

    if existing_ids.contains(&profile.entity_id.0) {
        score += 60;
        evidence.push("native_candidate_overlap".to_owned());
    }

    if case
        .resolved_entity_id
        .as_ref()
        .is_some_and(|entity_id| entity_id == &profile.entity_id)
    {
        score += 20;
        evidence.push("current_resolution_overlap".to_owned());
    }

    (score > 0).then(|| ErCandidateRef {
        entity_id: profile.entity_id.clone(),
        source: "lexical".to_owned(),
        score_millis: score,
        evidence,
    })
}

fn embedding_candidates_for_case(
    case: &ErReviewCase,
    case_embedding: &[f32],
    profiles: &[PreparedEmbeddingProfile],
    profile_embeddings: &[Vec<f32>],
    limit: usize,
    min_score_millis: i32,
) -> Vec<ErCandidateRef> {
    let existing_ids = case
        .candidates
        .iter()
        .map(|candidate| candidate.entity_id.0.clone())
        .collect::<BTreeSet<_>>();
    let mut rows = profiles
        .iter()
        .zip(profile_embeddings.iter())
        .filter_map(|(profile, profile_embedding)| {
            embedding_match(
                case,
                case_embedding,
                profile,
                profile_embedding,
                &existing_ids,
                min_score_millis,
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .score_millis
            .cmp(&left.score_millis)
            .then_with(|| left.entity_id.0.cmp(&right.entity_id.0))
    });
    rows.truncate(limit);
    rows
}

fn embedding_match(
    case: &ErReviewCase,
    case_embedding: &[f32],
    profile: &PreparedEmbeddingProfile,
    profile_embedding: &[f32],
    existing_ids: &BTreeSet<String>,
    min_score_millis: i32,
) -> Option<ErCandidateRef> {
    let cosine = dot(case_embedding, profile_embedding);
    let mut score = (cosine * 1000.0).round() as i32;
    let mut evidence = vec![format!("embedding_cosine:{cosine:.3}")];

    if !case.normalized_surface.is_empty()
        && case.normalized_surface == profile.normalized_canonical
    {
        score += 120;
        evidence.push(format!("exact_canonical:{}", case.normalized_surface));
    }
    if !case.normalized_surface.is_empty()
        && profile
            .normalized_aliases
            .iter()
            .any(|alias| alias == &case.normalized_surface)
    {
        score += 140;
        evidence.push(format!("exact_alias:{}", case.normalized_surface));
    }
    if let (Some(case_kind), Some(profile_kind)) =
        (case.mention_kind.as_ref(), profile.kind.as_ref())
    {
        if case_kind == profile_kind {
            score += 60;
            evidence.push(format!("kind_match:{}", kind_name(profile_kind)));
        } else {
            score -= 20;
            evidence.push(format!(
                "kind_conflict:{}!={}",
                kind_name(case_kind),
                kind_name(profile_kind)
            ));
        }
    }
    if existing_ids.contains(&profile.entity_id.0) {
        score += 45;
        evidence.push("native_candidate_overlap".to_owned());
    }
    if case
        .resolved_entity_id
        .as_ref()
        .is_some_and(|entity_id| entity_id == &profile.entity_id)
    {
        score += 30;
        evidence.push("current_resolution_overlap".to_owned());
    }
    if profile.serialized.contains(&case.surface) && !case.surface.is_empty() {
        score += 15;
        evidence.push("serialized_surface_overlap".to_owned());
    }

    (score >= min_score_millis).then(|| ErCandidateRef {
        entity_id: profile.entity_id.clone(),
        source: "embedding".to_owned(),
        score_millis: score,
        evidence,
    })
}

fn fused_candidates(case: &ErReviewCase) -> Vec<ErCandidateRef> {
    let mut merged = BTreeMap::<String, ErCandidateRef>::new();
    for candidate in case
        .lexical_candidates
        .iter()
        .chain(case.embedding_candidates.iter())
    {
        merged
            .entry(candidate.entity_id.0.clone())
            .and_modify(|existing| {
                existing.score_millis += candidate.score_millis;
                if existing.source != candidate.source && existing.source != "fused" {
                    existing.source = "fused".to_owned();
                }
                for value in &candidate.evidence {
                    if !existing.evidence.contains(value) {
                        existing.evidence.push(value.clone());
                    }
                }
            })
            .or_insert_with(|| candidate.clone());
    }
    let mut rows = merged.into_values().collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .score_millis
            .cmp(&left.score_millis)
            .then_with(|| left.entity_id.0.cmp(&right.entity_id.0))
    });
    rows
}

fn build_entity_profiles(archives: &[DocumentArchive]) -> Vec<ErEntityProfile> {
    let mut by_entity = FxHashMap::<String, ErEntityProfile>::default();
    let mut mention_totals = FxHashMap::<String, usize>::default();

    for archive in archives {
        for entity in &archive.entities {
            let entry = by_entity
                .entry(entity.entity_id.0.clone())
                .or_insert_with(|| entity_profile_from_record(archive, entity));
            merge_entity_profile(entry, archive, entity);
            *mention_totals
                .entry(entity.entity_id.0.clone())
                .or_default() += entity.mention_count;
        }

        for confirmation in &archive.alias_confirmations {
            let entry = by_entity
                .entry(confirmation.entity_id.0.clone())
                .or_insert_with(|| ErEntityProfile {
                    entity_id: confirmation.entity_id.clone(),
                    scope: archive.manifest.scope.clone(),
                    scope_key: archive.manifest.scope_key.clone(),
                    scope_ord: archive.manifest.scope_ord,
                    session_id: archive.manifest.session_id.clone(),
                    canonical_name: confirmation.alias_surface.clone(),
                    normalized_canonical: normalize_surface(&confirmation.alias_surface),
                    aliases: vec![confirmation.alias_surface.clone()],
                    kind: None,
                    mention_count: 0,
                    document_ids: vec![archive.manifest.document_id.clone()],
                    chunk_ids: Vec::new(),
                    serialized: String::new(),
                    blocking_keys: Vec::new(),
                });
            entry.aliases.push(confirmation.alias_surface.clone());
            entry
                .document_ids
                .push(archive.manifest.document_id.clone());
        }
    }

    let mut profiles = by_entity.into_values().collect::<Vec<_>>();
    profiles.sort_by(|left, right| left.entity_id.0.cmp(&right.entity_id.0));
    for profile in &mut profiles {
        profile.mention_count = mention_totals
            .get(&profile.entity_id.0)
            .copied()
            .unwrap_or(profile.mention_count);
        profile.aliases = dedupe_strings(profile.aliases.clone());
        profile.document_ids = dedupe_strings(profile.document_ids.clone());
        profile.chunk_ids = dedupe_strings(profile.chunk_ids.clone());
        profile.serialized = serialize_entity_profile(profile);
        profile.blocking_keys = blocking_keys_for_profile(profile);
    }
    profiles
}

fn entity_profile_from_record(
    archive: &DocumentArchive,
    entity: &SemanticEntityRecord,
) -> ErEntityProfile {
    ErEntityProfile {
        entity_id: entity.entity_id.clone(),
        scope: archive.manifest.scope.clone(),
        scope_key: archive.manifest.scope_key.clone(),
        scope_ord: archive.manifest.scope_ord,
        session_id: archive.manifest.session_id.clone(),
        canonical_name: entity.canonical_name.clone(),
        normalized_canonical: normalize_surface(&entity.canonical_name),
        aliases: entity.aliases.clone(),
        kind: entity.kind.clone(),
        mention_count: entity.mention_count,
        document_ids: vec![archive.manifest.document_id.clone()],
        chunk_ids: entity.chunk_ids.clone(),
        serialized: String::new(),
        blocking_keys: Vec::new(),
    }
}

fn merge_entity_profile(
    profile: &mut ErEntityProfile,
    archive: &DocumentArchive,
    entity: &SemanticEntityRecord,
) {
    if profile.canonical_name.is_empty() && !entity.canonical_name.is_empty() {
        profile.canonical_name = entity.canonical_name.clone();
        profile.normalized_canonical = normalize_surface(&entity.canonical_name);
    }
    if profile.kind.is_none() {
        profile.kind = entity.kind.clone();
    }
    profile.aliases.extend(entity.aliases.clone());
    profile
        .document_ids
        .push(archive.manifest.document_id.clone());
    profile.chunk_ids.extend(entity.chunk_ids.clone());
}

fn build_review_cases(
    archives: &[DocumentArchive],
    entity_profiles: &[ErEntityProfile],
) -> Vec<ErReviewCase> {
    let profile_by_entity = entity_profiles
        .iter()
        .map(|profile| (profile.entity_id.0.clone(), profile))
        .collect::<FxHashMap<_, _>>();
    let mut cases = Vec::<ErReviewCase>::new();

    for archive in archives {
        for (mention_index, resolved) in archive.resolved_mentions.iter().enumerate() {
            let decision_kind = compact_resolution_kind(resolved);
            let profile = resolved
                .entity_id
                .as_ref()
                .and_then(|entity_id| profile_by_entity.get(&entity_id.0).copied());
            let mention = archive.mentions.get(mention_index);
            let case_kind = match decision_kind {
                Some(CompactResolutionKind::Unresolved) => {
                    Some(ErReviewCaseKind::UnresolvedMention)
                }
                Some(CompactResolutionKind::Ambiguous) => Some(ErReviewCaseKind::AmbiguousMention),
                _ if type_disagrees(resolved, profile.and_then(|value| value.kind.clone())) => {
                    Some(ErReviewCaseKind::TypeDisagreement)
                }
                _ => None,
            };
            if let Some(kind) = case_kind {
                cases.push(build_mention_case(
                    archive,
                    mention_index,
                    mention,
                    resolved,
                    profile,
                    kind,
                ));
            }
        }

        for confirmation in &archive.alias_confirmations {
            let mention_index = archive
                .resolved_mentions
                .iter()
                .position(|resolved| resolved.mention_id == confirmation.mention_id);
            let mention = mention_index.and_then(|index| archive.mentions.get(index));
            let profile = profile_by_entity.get(&confirmation.entity_id.0).copied();
            if !should_emit_alias_observation(confirmation, mention, profile) {
                continue;
            }
            cases.push(build_alias_case(
                archive,
                mention_index,
                mention,
                confirmation,
                profile,
            ));
        }

        for (cluster_index, cluster) in archive.coref_clusters.iter().enumerate() {
            if cluster.ambiguous || cluster.resolved_entity_ids.len() > 1 {
                cases.push(build_coref_case(archive, cluster_index, cluster));
            }
        }
    }

    cases.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    cases
}

fn build_mention_case(
    archive: &DocumentArchive,
    mention_index: usize,
    mention: Option<&MentionSpan>,
    resolved: &ResolvedMention,
    profile: Option<&ErEntityProfile>,
    kind: ErReviewCaseKind,
) -> ErReviewCase {
    let range = mention.map(|value| value.range).unwrap_or(resolved.range);
    let (chunk_id, context) = context_for_range(archive, range);
    let mention_kind = mention
        .and_then(|value| value.kind.clone())
        .or_else(|| resolved.kind.clone());
    let surface = if resolved.surface.is_empty() {
        mention
            .map(|value| value.surface.clone())
            .unwrap_or_default()
    } else {
        resolved.surface.clone()
    };
    let normalized = if resolved.normalized.is_empty() {
        normalize_surface(&surface)
    } else {
        resolved.normalized.clone()
    };
    let serialized = serialize_case(
        &surface,
        &normalized,
        mention_kind.clone(),
        profile.map(|value| value.canonical_name.as_str()),
        &context,
    );

    ErReviewCase {
        case_id: format!(
            "{}::{}::{}",
            archive.manifest.document_id,
            review_case_kind_name(&kind),
            mention_index
        ),
        kind,
        scope: archive.manifest.scope.clone(),
        scope_key: archive.manifest.scope_key.clone(),
        scope_ord: archive.manifest.scope_ord,
        session_id: archive.manifest.session_id.clone(),
        document_id: archive.manifest.document_id.clone(),
        revision: archive.manifest.revision,
        mention_id: Some(resolved.mention_id.clone()),
        mention_index: Some(mention_index),
        surface: surface.clone(),
        normalized_surface: normalized,
        mention_kind: mention_kind.clone(),
        resolved_entity_id: resolved.entity_id.clone(),
        resolved_entity_kind: profile.and_then(|value| value.kind.clone()),
        decision_kind: compact_resolution_kind(resolved),
        decision_status: resolved.decision.status.clone(),
        confidence_millis: resolved.decision.confidence_millis,
        margin_millis: resolved.decision.margin_millis,
        candidates: resolved
            .candidates
            .iter()
            .map(|candidate| ErCandidateRef {
                entity_id: EntityId(candidate.entity_id.clone()),
                source: candidate.source.clone(),
                score_millis: candidate.score_millis,
                evidence: candidate
                    .evidence
                    .iter()
                    .map(|value| format!("{}:{}", value.kind, value.detail))
                    .collect(),
            })
            .collect(),
        lexical_candidates: Vec::new(),
        embedding_candidates: Vec::new(),
        fused_candidates: Vec::new(),
        chunk_id,
        context,
        serialized,
        blocking_keys: blocking_keys_for_surface(&surface, mention_kind.as_ref()),
    }
}

fn build_alias_case(
    archive: &DocumentArchive,
    mention_index: Option<usize>,
    mention: Option<&MentionSpan>,
    confirmation: &phoenix_semantic_v2::AliasConfirmation,
    profile: Option<&ErEntityProfile>,
) -> ErReviewCase {
    let range = mention.map(|value| value.range).unwrap_or_default();
    let (chunk_id, context) = context_for_range(archive, range);
    let mention_kind = mention.and_then(|value| value.kind.clone());
    let serialized = serialize_case(
        &confirmation.alias_surface,
        &confirmation.normalized,
        mention_kind.clone(),
        profile.map(|value| value.canonical_name.as_str()),
        &context,
    );

    ErReviewCase {
        case_id: format!(
            "{}::alias::{}",
            archive.manifest.document_id, confirmation.mention_id.0
        ),
        kind: ErReviewCaseKind::AliasObservation,
        scope: archive.manifest.scope.clone(),
        scope_key: archive.manifest.scope_key.clone(),
        scope_ord: archive.manifest.scope_ord,
        session_id: archive.manifest.session_id.clone(),
        document_id: archive.manifest.document_id.clone(),
        revision: archive.manifest.revision,
        mention_id: Some(confirmation.mention_id.clone()),
        mention_index,
        surface: confirmation.alias_surface.clone(),
        normalized_surface: confirmation.normalized.clone(),
        mention_kind: mention_kind.clone(),
        resolved_entity_id: Some(confirmation.entity_id.clone()),
        resolved_entity_kind: profile.and_then(|value| value.kind.clone()),
        decision_kind: Some(CompactResolutionKind::Resolved),
        decision_status: "alias_confirmed".to_owned(),
        confidence_millis: confirmation.confidence_millis,
        margin_millis: confirmation.confidence_millis,
        candidates: vec![ErCandidateRef {
            entity_id: confirmation.entity_id.clone(),
            source: "alias_confirmation".to_owned(),
            score_millis: confirmation.confidence_millis as i32,
            evidence: vec!["confirmed_alias".to_owned()],
        }],
        lexical_candidates: Vec::new(),
        embedding_candidates: Vec::new(),
        fused_candidates: Vec::new(),
        chunk_id,
        context,
        serialized,
        blocking_keys: blocking_keys_for_surface(
            &confirmation.alias_surface,
            mention_kind.as_ref(),
        ),
    }
}

fn should_emit_alias_observation(
    confirmation: &phoenix_semantic_v2::AliasConfirmation,
    mention: Option<&MentionSpan>,
    profile: Option<&ErEntityProfile>,
) -> bool {
    let normalized = confirmation.normalized.trim();
    if normalized.is_empty() {
        return false;
    }
    let token_count = normalized.split_whitespace().count();
    if token_count > 1 {
        return true;
    }
    if GENERIC_ALIAS_OBSERVATION_TOKENS
        .iter()
        .any(|token| *token == normalized)
    {
        return false;
    }
    if mention.and_then(|value| value.kind.as_ref()).is_some() {
        return true;
    }
    if profile.and_then(|value| value.kind.as_ref()).is_some() {
        return true;
    }
    true
}

fn build_coref_case(
    archive: &DocumentArchive,
    cluster_index: usize,
    cluster: &CorefClusterRecord,
) -> ErReviewCase {
    let context = cluster
        .chunk_ids
        .first()
        .and_then(|chunk_id| {
            archive
                .chunks
                .iter()
                .find(|chunk| chunk.chunk_id.0 == *chunk_id)
                .map(|chunk| chunk.text.clone())
        })
        .unwrap_or_default();
    let surface = cluster.representative_surface.clone();
    let normalized = normalize_surface(&surface);
    ErReviewCase {
        case_id: format!("{}::coref::{cluster_index}", archive.manifest.document_id),
        kind: ErReviewCaseKind::CorefConflict,
        scope: archive.manifest.scope.clone(),
        scope_key: archive.manifest.scope_key.clone(),
        scope_ord: archive.manifest.scope_ord,
        session_id: archive.manifest.session_id.clone(),
        document_id: archive.manifest.document_id.clone(),
        revision: archive.manifest.revision,
        mention_id: None,
        mention_index: None,
        surface: surface.clone(),
        normalized_surface: normalized.clone(),
        mention_kind: None,
        resolved_entity_id: cluster.resolved_entity_ids.first().cloned(),
        resolved_entity_kind: None,
        decision_kind: Some(CompactResolutionKind::Ambiguous),
        decision_status: "coref_conflict".to_owned(),
        confidence_millis: cluster.confidence_millis,
        margin_millis: 0,
        candidates: cluster
            .resolved_entity_ids
            .iter()
            .cloned()
            .map(|entity_id| ErCandidateRef {
                entity_id,
                source: "coref_cluster".to_owned(),
                score_millis: cluster.confidence_millis as i32,
                evidence: vec![format!("cluster:{}", cluster.cluster_id)],
            })
            .collect(),
        lexical_candidates: Vec::new(),
        embedding_candidates: Vec::new(),
        fused_candidates: Vec::new(),
        chunk_id: cluster.chunk_ids.first().cloned(),
        context: context.clone(),
        serialized: serialize_case(&surface, &normalized, None, None, &context),
        blocking_keys: blocking_keys_for_surface(&surface, None),
    }
}

fn compact_resolution_kind(resolved: &ResolvedMention) -> Option<CompactResolutionKind> {
    match resolved.decision.status.to_ascii_lowercase().as_str() {
        "resolved" => Some(CompactResolutionKind::Resolved),
        "ambiguous" => Some(CompactResolutionKind::Ambiguous),
        "unresolved" => Some(CompactResolutionKind::Unresolved),
        _ => None,
    }
}

fn type_disagrees(resolved: &ResolvedMention, entity_kind: Option<EntityKind>) -> bool {
    match (&resolved.kind, entity_kind) {
        (Some(left), Some(right)) => *left != right,
        _ => false,
    }
}

fn context_for_range(archive: &DocumentArchive, range: TextRange) -> (Option<String>, String) {
    archive
        .chunks
        .iter()
        .find(|chunk| overlaps(chunk.range, range))
        .map(|chunk| (Some(chunk.chunk_id.0.clone()), chunk.text.clone()))
        .unwrap_or_else(|| (None, String::new()))
}

fn overlaps(left: TextRange, right: TextRange) -> bool {
    left.start < right.end && right.start < left.end
}

fn serialize_entity_profile(profile: &ErEntityProfile) -> String {
    let mut parts = Vec::<String>::new();
    if !profile.canonical_name.is_empty() {
        parts.push(profile.canonical_name.clone());
    }
    if !profile.aliases.is_empty() {
        parts.push(profile.aliases.join(" "));
    }
    if let Some(kind) = profile.kind.as_ref() {
        parts.push(kind_name(kind).to_owned());
    }
    collapse_ws(&parts.join(" "))
}

fn serialize_case(
    surface: &str,
    normalized: &str,
    kind: Option<EntityKind>,
    canonical_name: Option<&str>,
    context: &str,
) -> String {
    let mut parts = Vec::<String>::new();
    if !surface.is_empty() {
        parts.push(surface.to_owned());
    }
    if !normalized.is_empty() && normalized != normalize_surface(surface) {
        parts.push(normalized.to_owned());
    }
    if let Some(kind) = kind.as_ref() {
        parts.push(kind_name(kind).to_owned());
    }
    if let Some(value) = canonical_name.filter(|value| !value.is_empty()) {
        parts.push(value.to_owned());
    }
    if !context.is_empty() {
        parts.push(context.to_owned());
    }
    collapse_ws(&parts.join(" "))
}

fn blocking_keys_for_profile(profile: &ErEntityProfile) -> Vec<String> {
    let mut keys = BTreeSet::<String>::new();
    let canonical = normalize_surface(&profile.canonical_name);
    if !canonical.is_empty() {
        keys.insert(format!("canonical:{canonical}"));
    }
    for alias in &profile.aliases {
        let normalized = normalize_surface(alias);
        if !normalized.is_empty() {
            keys.insert(format!("alias:{normalized}"));
        }
    }
    if let Some(kind) = profile.kind.as_ref() {
        keys.insert(format!("kind:{}", kind_name(kind)));
    }
    if looks_faction_like(&profile.canonical_name)
        || profile
            .aliases
            .iter()
            .any(|alias| looks_faction_like(alias))
    {
        keys.insert("shape:faction".to_owned());
    }
    if looks_alias_like(&profile.canonical_name)
        || profile.aliases.iter().any(|alias| looks_alias_like(alias))
    {
        keys.insert("shape:alias".to_owned());
    }
    for token in strong_tokens(&profile.serialized) {
        keys.insert(format!("token:{token}"));
    }
    keys.into_iter().collect()
}

fn blocking_keys_for_surface(surface: &str, kind: Option<&EntityKind>) -> Vec<String> {
    let mut keys = BTreeSet::<String>::new();
    let normalized = normalize_surface(surface);
    if !normalized.is_empty() {
        keys.insert(format!("surface:{normalized}"));
    }
    if let Some(kind) = kind {
        keys.insert(format!("kind:{}", kind_name(kind)));
    }
    if looks_faction_like(surface) {
        keys.insert("shape:faction".to_owned());
    }
    if looks_alias_like(surface) {
        keys.insert("shape:alias".to_owned());
    }
    for token in strong_tokens(surface) {
        keys.insert(format!("token:{token}"));
    }
    keys.into_iter().collect()
}

fn kind_name(kind: &EntityKind) -> &'static str {
    match kind {
        EntityKind::Character => "character",
        EntityKind::Location => "location",
        EntityKind::Npc => "npc",
        EntityKind::Item => "item",
        EntityKind::Faction => "faction",
        EntityKind::Organization => "organization",
        EntityKind::Event => "event",
        EntityKind::Concept => "concept",
        EntityKind::Other => "other",
    }
}

pub fn review_case_kind_name(kind: &ErReviewCaseKind) -> &'static str {
    match kind {
        ErReviewCaseKind::UnresolvedMention => "unresolved",
        ErReviewCaseKind::AmbiguousMention => "ambiguous",
        ErReviewCaseKind::TypeDisagreement => "type",
        ErReviewCaseKind::AliasObservation => "alias",
        ErReviewCaseKind::CorefConflict => "coref",
    }
}

pub fn decision_kind_name(kind: &ErDecisionKind) -> &'static str {
    match kind {
        ErDecisionKind::Link => "link",
        ErDecisionKind::ConfirmAlias => "confirm_alias",
        ErDecisionKind::PatchType => "patch_type",
        ErDecisionKind::Defer => "defer",
        ErDecisionKind::Reject => "reject",
    }
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f32 / denominator as f32
    }
}

fn candidate_has_exact_match(candidate: &ErCandidateRef) -> bool {
    candidate
        .evidence
        .iter()
        .any(|value| value.starts_with("exact_alias:") || value.starts_with("exact_canonical:"))
}

fn candidate_has_kind_match(candidate: &ErCandidateRef) -> bool {
    candidate
        .evidence
        .iter()
        .any(|value| value.starts_with("kind_match:"))
}

fn candidate_has_faction_support(candidate: &ErCandidateRef) -> bool {
    candidate
        .evidence
        .iter()
        .any(|value| value == "faction_shape")
}

fn candidate_has_native_overlap(candidate: &ErCandidateRef) -> bool {
    candidate
        .evidence
        .iter()
        .any(|value| value == "native_candidate_overlap")
}

fn candidate_has_embedding_support(candidate: &ErCandidateRef) -> bool {
    candidate
        .evidence
        .iter()
        .any(|value| value.starts_with("embedding_cosine:"))
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(lhs, rhs)| lhs * rhs)
        .sum()
}

fn graph_has_input(model: &candle_onnx::onnx::ModelProto, name: &str) -> bool {
    model
        .graph
        .as_ref()
        .map(|graph| graph.input.iter().any(|input| input.name == name))
        .unwrap_or(false)
}

fn select_hidden_output(
    model: &candle_onnx::onnx::ModelProto,
    outputs: HashMap<String, Tensor>,
) -> Result<Tensor, String> {
    let output_names = model
        .graph
        .as_ref()
        .map(|graph| {
            graph
                .output
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    for name in ["last_hidden_state", "token_embeddings"] {
        if let Some(tensor) = outputs.get(name) {
            return Ok(tensor.clone());
        }
    }
    for name in output_names {
        if let Some(tensor) = outputs.get(name) {
            return Ok(tensor.clone());
        }
    }
    outputs
        .into_iter()
        .next()
        .map(|(_, tensor)| tensor)
        .ok_or_else(|| "embedding model returned no outputs".to_owned())
}

fn decision_outcome_from_kind(kind: &ErDecisionKind) -> ErDecisionOutcome {
    match kind {
        ErDecisionKind::Link => ErDecisionOutcome::Link,
        ErDecisionKind::ConfirmAlias => ErDecisionOutcome::ConfirmAlias,
        ErDecisionKind::PatchType => ErDecisionOutcome::PatchType,
        ErDecisionKind::Defer => ErDecisionOutcome::Defer,
        ErDecisionKind::Reject => ErDecisionOutcome::Reject,
    }
}

fn er_outcome_name(outcome: ErDecisionOutcome) -> &'static str {
    match outcome {
        ErDecisionOutcome::Link => "link",
        ErDecisionOutcome::ConfirmAlias => "confirm_alias",
        ErDecisionOutcome::PatchType => "patch_type",
        ErDecisionOutcome::Defer => "defer",
        ErDecisionOutcome::Reject => "reject",
    }
}

fn merge_er_patch_sidecars(
    mut existing: ErScopePatchSidecar,
    updates: ErScopePatchSidecar,
) -> ErScopePatchSidecar {
    existing.updated_at = existing.updated_at.max(updates.updated_at);
    existing.generation = existing.generation.max(updates.generation);
    existing.scope = updates.scope;
    existing.scope_key = updates.scope_key;
    existing.scope_ord = updates.scope_ord.or(existing.scope_ord);
    existing.session_id = updates.session_id.or(existing.session_id);
    existing.alias_additions.extend(updates.alias_additions);
    existing.type_overrides.extend(updates.type_overrides);
    existing.entity_links.extend(updates.entity_links);
    existing.decisions.extend(updates.decisions);
    dedupe_er_patch_sidecar(&mut existing);
    existing
}

fn dedupe_er_patch_sidecar(sidecar: &mut ErScopePatchSidecar) {
    let mut alias_by_key = BTreeMap::<(String, String), ErAliasAddition>::new();
    for alias in sidecar.alias_additions.drain(..) {
        alias_by_key.insert((alias.entity_id.0.clone(), alias.normalized.clone()), alias);
    }
    sidecar.alias_additions = alias_by_key.into_values().collect();

    let mut type_by_entity = BTreeMap::<String, ErTypeOverride>::new();
    for patch in sidecar.type_overrides.drain(..) {
        type_by_entity.insert(patch.entity_id.0.clone(), patch);
    }
    sidecar.type_overrides = type_by_entity.into_values().collect();

    let mut link_by_case = BTreeMap::<String, ErEntityLinkOverride>::new();
    for patch in sidecar.entity_links.drain(..) {
        link_by_case.insert(patch.case_id.clone(), patch);
    }
    sidecar.entity_links = link_by_case.into_values().collect();

    let mut decisions_by_case = BTreeMap::<String, ErDecisionRecord>::new();
    for decision in sidecar.decisions.drain(..) {
        decisions_by_case.insert(decision.case_id.clone(), decision);
    }
    sidecar.decisions = decisions_by_case.into_values().collect();
}

fn refresh_entity_profile(profile: &mut ErEntityProfile) {
    profile.aliases = dedupe_strings(profile.aliases.clone());
    profile.document_ids = dedupe_strings(profile.document_ids.clone());
    profile.chunk_ids = dedupe_strings(profile.chunk_ids.clone());
    profile.serialized = serialize_entity_profile(profile);
    profile.blocking_keys = blocking_keys_for_profile(profile);
}

fn ensure_patch_candidate(
    case: &mut ErReviewCase,
    entity_id: EntityId,
    source: &str,
    score_millis: i32,
) {
    if case
        .candidates
        .iter()
        .any(|candidate| candidate.entity_id == entity_id)
    {
        return;
    }
    case.candidates.push(ErCandidateRef {
        entity_id,
        source: source.to_owned(),
        score_millis,
        evidence: vec!["persisted_patch".to_owned()],
    });
}

fn normalize_surface(text: &str) -> String {
    let mut value = String::with_capacity(text.len());
    let mut prev_space = false;
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            for lowered in ch.to_lowercase() {
                value.push(lowered);
            }
            prev_space = false;
        } else if !prev_space && !value.is_empty() {
            value.push(' ');
            prev_space = true;
        }
    }
    while value.ends_with(' ') {
        value.pop();
    }
    value
}

const GENERIC_ALIAS_OBSERVATION_TOKENS: &[&str] = &[
    "boss",
    "boyfriend",
    "brother",
    "captain",
    "chief",
    "daughter",
    "doctor",
    "father",
    "girlfriend",
    "grandfather",
    "grandma",
    "grandmother",
    "grandpa",
    "husband",
    "king",
    "manager",
    "mom",
    "mother",
    "mrs",
    "ms",
    "nurse",
    "officer",
    "prince",
    "princess",
    "professor",
    "queen",
    "sister",
    "son",
    "teacher",
    "uncle",
    "wife",
];

const WEAK_BLOCKING_TOKENS: &[&str] = &[
    "and",
    "boss",
    "boy",
    "boyfriend",
    "brother",
    "chief",
    "daughter",
    "doctor",
    "father",
    "friend",
    "girl",
    "girlfriend",
    "guy",
    "king",
    "man",
    "manager",
    "mom",
    "mother",
    "mrs",
    "ms",
    "nurse",
    "officer",
    "queen",
    "sister",
    "son",
    "teacher",
    "the",
    "uncle",
    "wife",
    "with",
    "woman",
];

const FACTION_SHAPE_TOKENS: &[&str] = &[
    "agency",
    "alliance",
    "association",
    "bureau",
    "clan",
    "club",
    "committee",
    "company",
    "corporation",
    "corp",
    "council",
    "crew",
    "division",
    "family",
    "faction",
    "gang",
    "group",
    "guild",
    "hq",
    "house",
    "league",
    "order",
    "security",
    "society",
    "squad",
    "syndicate",
    "team",
    "union",
    "unit",
];

fn looks_faction_like(text: &str) -> bool {
    let normalized = normalize_surface(text);
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();
    if tokens.len() < 2 {
        return false;
    }
    tokens
        .iter()
        .any(|token| FACTION_SHAPE_TOKENS.iter().any(|shape| shape == token))
}

fn looks_alias_like(text: &str) -> bool {
    let normalized = normalize_surface(text);
    let tokens = normalized.split_whitespace().collect::<Vec<_>>();
    if tokens.len() != 1 {
        return false;
    }
    let token = tokens[0];
    token.len() >= 4 && !WEAK_BLOCKING_TOKENS.iter().any(|blocked| blocked == &token)
}

fn strong_tokens(text: &str) -> Vec<String> {
    normalize_surface(text)
        .split_whitespace()
        .filter(|token| token.len() >= 3)
        .filter(|token| !WEAK_BLOCKING_TOKENS.iter().any(|blocked| blocked == token))
        .take(6)
        .map(|token| token.to_owned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn collapse_ws(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn dedupe_strings(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        apply_er_patch_sidecar, build_er_patch_sidecar, derive_scope_review_batch,
        draft_review_decisions, generate_lexical_candidates, persist_er_patch_sidecar,
        ErDecisionKind, ErReviewCaseKind,
    };
    use phoenix_graph_kernel::KernelMutationBatch;
    use phoenix_semantic_v2::{
        AliasConfirmation, CandidateEntity, ChunkId, ChunkRecord, CorefClusterRecord,
        DocumentArchive, DocumentManifest, NativeCorefSummary, NativeErSummary, ResolutionDecision,
        ResolvedMention, ScopeOrd, SemanticEntityRecord,
    };
    use phoenix_store_native_core::PhoenixErPatchStore;
    use phoenix_store_overgraph::PhoenixOvergraphStore;
    use phoenix_types::{
        DocumentId, EntityId, EntityKind, IngestDocumentSummary, MentionEntityRef, MentionSource,
        MentionSpan, NoteId, ScopeKey, SessionDocumentState, SessionId, TextRange,
    };

    fn sample_manifest() -> DocumentManifest {
        DocumentManifest {
            document_id: "doc-1".to_owned(),
            note_id: Some(NoteId("note-1".to_owned())),
            scope: ScopeKey {
                world_id: Some("world".to_owned()),
                narrative_id: Some("narr".to_owned()),
                folder_id: Some("folder".to_owned()),
                folder_path: Some("/story".to_owned()),
            },
            scope_key: "world::narr::folder::/story".to_owned(),
            scope_ord: ScopeOrd(7),
            revision: 3,
            title: "Sample".to_owned(),
            session_id: Some(SessionId("session-1".to_owned())),
            document_summary: IngestDocumentSummary::default(),
            session_document: SessionDocumentState {
                document_id: DocumentId("doc-1".to_owned()),
                note_id: Some(NoteId("note-1".to_owned())),
                ..Default::default()
            },
            archive_version: 4,
            ..Default::default()
        }
    }

    #[test]
    fn derive_scope_review_batch_surfaces_er_cases_and_profiles() {
        let archive = DocumentArchive {
            manifest: sample_manifest(),
            mentions: vec![
                MentionSpan {
                    range: TextRange { start: 0, end: 9 },
                    surface: "Quicksave".to_owned(),
                    kind: Some(EntityKind::Character),
                    entity_ref: Some(MentionEntityRef::Known(EntityId("entity-1".to_owned()))),
                    source: Some(MentionSource::Discovery),
                    confidence: 0.9,
                    sentence_index: 0,
                },
                MentionSpan {
                    range: TextRange { start: 10, end: 14 },
                    surface: "Meta".to_owned(),
                    kind: Some(EntityKind::Location),
                    entity_ref: None,
                    source: Some(MentionSource::Discovery),
                    confidence: 0.8,
                    sentence_index: 0,
                },
                MentionSpan {
                    range: TextRange { start: 15, end: 20 },
                    surface: "Ghost".to_owned(),
                    kind: Some(EntityKind::Character),
                    entity_ref: None,
                    source: Some(MentionSource::Discovery),
                    confidence: 0.7,
                    sentence_index: 0,
                },
            ],
            resolved_mentions: vec![
                ResolvedMention {
                    mention_id: phoenix_semantic_v2::MentionId("doc-1::m0".to_owned()),
                    mention_index: 0,
                    range: TextRange { start: 0, end: 9 },
                    surface: "Quicksave".to_owned(),
                    normalized: "quicksave".to_owned(),
                    kind: Some(EntityKind::Character),
                    entity_id: Some(EntityId("entity-1".to_owned())),
                    decision: ResolutionDecision {
                        status: "resolved".to_owned(),
                        confidence_millis: 980,
                        margin_millis: 400,
                    },
                    candidates: vec![CandidateEntity {
                        entity_id: "entity-1".to_owned(),
                        source: "native".to_owned(),
                        score_millis: 980,
                        evidence: Vec::new(),
                    }],
                },
                ResolvedMention {
                    mention_id: phoenix_semantic_v2::MentionId("doc-1::m1".to_owned()),
                    mention_index: 1,
                    range: TextRange { start: 10, end: 14 },
                    surface: "Meta".to_owned(),
                    normalized: "meta".to_owned(),
                    kind: Some(EntityKind::Location),
                    entity_id: Some(EntityId("entity-2".to_owned())),
                    decision: ResolutionDecision {
                        status: "resolved".to_owned(),
                        confidence_millis: 810,
                        margin_millis: 120,
                    },
                    candidates: vec![CandidateEntity {
                        entity_id: "entity-2".to_owned(),
                        source: "native".to_owned(),
                        score_millis: 810,
                        evidence: Vec::new(),
                    }],
                },
                ResolvedMention {
                    mention_id: phoenix_semantic_v2::MentionId("doc-1::m2".to_owned()),
                    mention_index: 2,
                    range: TextRange { start: 15, end: 20 },
                    surface: "Ghost".to_owned(),
                    normalized: "ghost".to_owned(),
                    kind: Some(EntityKind::Character),
                    entity_id: None,
                    decision: ResolutionDecision {
                        status: "unresolved".to_owned(),
                        confidence_millis: 220,
                        margin_millis: 20,
                    },
                    candidates: Vec::new(),
                },
            ],
            alias_confirmations: vec![AliasConfirmation {
                alias_surface: "Quicksave".to_owned(),
                normalized: "quicksave".to_owned(),
                entity_id: EntityId("entity-1".to_owned()),
                confidence_millis: 970,
                mention_id: phoenix_semantic_v2::MentionId("doc-1::m0".to_owned()),
            }],
            coref_clusters: vec![CorefClusterRecord {
                cluster_id: "cluster-1".to_owned(),
                representative_surface: "Quicksave".to_owned(),
                mention_count: 2,
                first_sentence_index: 0,
                last_sentence_index: 0,
                chunk_ids: vec!["chunk-1".to_owned()],
                named_count: 1,
                nominal_count: 1,
                pronoun_count: 0,
                resolved_entity_ids: vec![
                    EntityId("entity-1".to_owned()),
                    EntityId("entity-2".to_owned()),
                ],
                confidence_millis: 640,
                ambiguous: true,
                route_mix_bits: 0,
            }],
            er_summary: NativeErSummary::default(),
            coref_summary: NativeCorefSummary::default(),
            chunks: vec![ChunkRecord {
                chunk_id: ChunkId("chunk-1".to_owned()),
                range: TextRange { start: 0, end: 32 },
                chapter_id: 1,
                boundary_label: Some("Chapter 1".to_owned()),
                text: "Quicksave met Meta while Ghost hid.".to_owned(),
            }],
            entities: vec![
                SemanticEntityRecord {
                    entity_id: EntityId("entity-1".to_owned()),
                    canonical_name: "Ryan Romano".to_owned(),
                    aliases: vec!["Courier".to_owned()],
                    kind: Some(EntityKind::Character),
                    mention_count: 4,
                    chunk_ids: vec!["chunk-1".to_owned()],
                },
                SemanticEntityRecord {
                    entity_id: EntityId("entity-2".to_owned()),
                    canonical_name: "Meta-Gang".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Organization),
                    mention_count: 2,
                    chunk_ids: vec!["chunk-1".to_owned()],
                },
            ],
            graph_batch: KernelMutationBatch::default(),
            ..Default::default()
        };

        let mut batch = derive_scope_review_batch(&[archive], None, None, None);
        let lexical = generate_lexical_candidates(&mut batch, 4);
        assert_eq!(batch.entity_profiles.len(), 2);
        assert_eq!(lexical.case_count, batch.review_cases.len());
        assert!(lexical.matched_case_count >= 2);
        assert!(batch
            .entity_profiles
            .iter()
            .any(|profile| profile.aliases.iter().any(|alias| alias == "Quicksave")));
        assert!(batch
            .review_cases
            .iter()
            .any(|case| case.kind == ErReviewCaseKind::AliasObservation));
        assert!(batch
            .review_cases
            .iter()
            .any(|case| case.kind == ErReviewCaseKind::UnresolvedMention));
        assert!(batch
            .review_cases
            .iter()
            .any(|case| case.kind == ErReviewCaseKind::TypeDisagreement));
        assert!(batch
            .review_cases
            .iter()
            .any(|case| case.kind == ErReviewCaseKind::CorefConflict));
        assert!(batch
            .review_cases
            .iter()
            .any(|case| case.serialized.contains("Meta")));
        assert!(batch.review_cases.iter().any(|case| {
            case.kind == ErReviewCaseKind::AliasObservation
                && case
                    .lexical_candidates
                    .iter()
                    .any(|candidate| candidate.entity_id.0 == "entity-1")
        }));
        assert!(batch.review_cases.iter().any(|case| {
            case.kind == ErReviewCaseKind::TypeDisagreement
                && case
                    .lexical_candidates
                    .iter()
                    .any(|candidate| candidate.entity_id.0 == "entity-2")
        }));
        let decisions = draft_review_decisions(&batch);
        assert!(decisions
            .iter()
            .any(|decision| decision.kind == ErDecisionKind::ConfirmAlias));
        assert!(decisions
            .iter()
            .any(|decision| decision.kind == ErDecisionKind::PatchType));
    }

    #[test]
    fn build_and_apply_er_patch_sidecar_replays_updates() {
        let archive = DocumentArchive {
            manifest: sample_manifest(),
            mentions: vec![
                MentionSpan {
                    range: TextRange { start: 0, end: 9 },
                    surface: "Quicksave".to_owned(),
                    kind: Some(EntityKind::Character),
                    entity_ref: Some(MentionEntityRef::Known(EntityId("entity-1".to_owned()))),
                    source: Some(MentionSource::Discovery),
                    confidence: 0.9,
                    sentence_index: 0,
                },
                MentionSpan {
                    range: TextRange { start: 10, end: 14 },
                    surface: "Meta".to_owned(),
                    kind: Some(EntityKind::Location),
                    entity_ref: None,
                    source: Some(MentionSource::Discovery),
                    confidence: 0.8,
                    sentence_index: 0,
                },
            ],
            resolved_mentions: vec![
                ResolvedMention {
                    mention_id: phoenix_semantic_v2::MentionId("doc-1::m0".to_owned()),
                    mention_index: 0,
                    range: TextRange { start: 0, end: 9 },
                    surface: "Quicksave".to_owned(),
                    normalized: "quicksave".to_owned(),
                    kind: Some(EntityKind::Character),
                    entity_id: Some(EntityId("entity-1".to_owned())),
                    decision: ResolutionDecision {
                        status: "resolved".to_owned(),
                        confidence_millis: 980,
                        margin_millis: 400,
                    },
                    candidates: vec![CandidateEntity {
                        entity_id: "entity-1".to_owned(),
                        source: "native".to_owned(),
                        score_millis: 980,
                        evidence: Vec::new(),
                    }],
                },
                ResolvedMention {
                    mention_id: phoenix_semantic_v2::MentionId("doc-1::m1".to_owned()),
                    mention_index: 1,
                    range: TextRange { start: 10, end: 14 },
                    surface: "Meta".to_owned(),
                    normalized: "meta".to_owned(),
                    kind: Some(EntityKind::Location),
                    entity_id: Some(EntityId("entity-2".to_owned())),
                    decision: ResolutionDecision {
                        status: "resolved".to_owned(),
                        confidence_millis: 810,
                        margin_millis: 120,
                    },
                    candidates: vec![CandidateEntity {
                        entity_id: "entity-2".to_owned(),
                        source: "native".to_owned(),
                        score_millis: 810,
                        evidence: Vec::new(),
                    }],
                },
            ],
            alias_confirmations: vec![AliasConfirmation {
                alias_surface: "Quicksave".to_owned(),
                normalized: "quicksave".to_owned(),
                entity_id: EntityId("entity-1".to_owned()),
                confidence_millis: 970,
                mention_id: phoenix_semantic_v2::MentionId("doc-1::m0".to_owned()),
            }],
            entities: vec![
                SemanticEntityRecord {
                    entity_id: EntityId("entity-1".to_owned()),
                    canonical_name: "Ryan Romano".to_owned(),
                    aliases: vec!["Courier".to_owned()],
                    kind: Some(EntityKind::Character),
                    mention_count: 4,
                    chunk_ids: vec!["chunk-1".to_owned()],
                },
                SemanticEntityRecord {
                    entity_id: EntityId("entity-2".to_owned()),
                    canonical_name: "Meta-Gang".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Organization),
                    mention_count: 2,
                    chunk_ids: vec!["chunk-1".to_owned()],
                },
            ],
            chunks: vec![ChunkRecord {
                chunk_id: ChunkId("chunk-1".to_owned()),
                range: TextRange { start: 0, end: 24 },
                chapter_id: 1,
                boundary_label: Some("Chapter 1".to_owned()),
                text: "Quicksave met Meta.".to_owned(),
            }],
            ..Default::default()
        };

        let mut batch = derive_scope_review_batch(&[archive], None, None, None);
        generate_lexical_candidates(&mut batch, 4);
        let decisions = draft_review_decisions(&batch);
        let sidecar = build_er_patch_sidecar(&batch, &decisions, 1234);

        let mut replayed = batch.clone();
        apply_er_patch_sidecar(&mut replayed, &sidecar);

        assert!(replayed
            .entity_profiles
            .iter()
            .find(|profile| profile.entity_id.0 == "entity-1")
            .is_some_and(|profile| profile.aliases.iter().any(|alias| alias == "Quicksave")));
        assert!(replayed
            .entity_profiles
            .iter()
            .find(|profile| profile.entity_id.0 == "entity-2")
            .is_some_and(|profile| profile.kind == Some(EntityKind::Organization)));
        assert!(replayed.review_cases.iter().any(|case| {
            case.kind == ErReviewCaseKind::AliasObservation
                && case.decision_status == "er_confirm_alias"
                && case
                    .resolved_entity_id
                    .as_ref()
                    .is_some_and(|id| id.0 == "entity-1")
        }));
        assert!(replayed.review_cases.iter().any(|case| {
            case.kind == ErReviewCaseKind::TypeDisagreement
                && case.decision_status == "er_patch_type"
                && case.resolved_entity_kind == Some(EntityKind::Organization)
        }));
    }

    #[test]
    fn persists_er_patch_sidecar_in_overgraph_store() {
        let archive = DocumentArchive {
            manifest: sample_manifest(),
            mentions: vec![MentionSpan {
                range: TextRange { start: 0, end: 9 },
                surface: "Quicksave".to_owned(),
                kind: Some(EntityKind::Character),
                entity_ref: Some(MentionEntityRef::Known(EntityId("entity-1".to_owned()))),
                source: Some(MentionSource::Discovery),
                confidence: 0.9,
                sentence_index: 0,
            }],
            resolved_mentions: vec![ResolvedMention {
                mention_id: phoenix_semantic_v2::MentionId("doc-1::m0".to_owned()),
                mention_index: 0,
                range: TextRange { start: 0, end: 9 },
                surface: "Quicksave".to_owned(),
                normalized: "quicksave".to_owned(),
                kind: Some(EntityKind::Character),
                entity_id: Some(EntityId("entity-1".to_owned())),
                decision: ResolutionDecision {
                    status: "resolved".to_owned(),
                    confidence_millis: 980,
                    margin_millis: 400,
                },
                candidates: vec![CandidateEntity {
                    entity_id: "entity-1".to_owned(),
                    source: "native".to_owned(),
                    score_millis: 980,
                    evidence: Vec::new(),
                }],
            }],
            alias_confirmations: vec![AliasConfirmation {
                alias_surface: "Quicksave".to_owned(),
                normalized: "quicksave".to_owned(),
                entity_id: EntityId("entity-1".to_owned()),
                confidence_millis: 970,
                mention_id: phoenix_semantic_v2::MentionId("doc-1::m0".to_owned()),
            }],
            entities: vec![SemanticEntityRecord {
                entity_id: EntityId("entity-1".to_owned()),
                canonical_name: "Ryan Romano".to_owned(),
                aliases: vec!["Courier".to_owned()],
                kind: Some(EntityKind::Character),
                mention_count: 4,
                chunk_ids: vec!["chunk-1".to_owned()],
            }],
            chunks: vec![ChunkRecord {
                chunk_id: ChunkId("chunk-1".to_owned()),
                range: TextRange { start: 0, end: 12 },
                chapter_id: 1,
                boundary_label: None,
                text: "Quicksave.".to_owned(),
            }],
            ..Default::default()
        };

        let mut batch = derive_scope_review_batch(&[archive], None, None, None);
        generate_lexical_candidates(&mut batch, 4);
        let decisions = draft_review_decisions(&batch);

        let store_path = std::env::temp_dir().join(format!(
            "phoenix-er-post-patch-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
        ));
        let store = PhoenixOvergraphStore::open(&store_path).expect("open store");
        store.init_er_patch_schema().expect("init er patch schema");

        let persisted =
            persist_er_patch_sidecar(&store, &batch, &decisions, 4321).expect("persist sidecar");
        let loaded = store
            .load_er_patch_sidecar(&batch.scope)
            .expect("load sidecar")
            .expect("sidecar exists");

        assert_eq!(loaded.scope_key, batch.scope_key);
        assert_eq!(loaded.alias_additions, persisted.alias_additions);
        assert_eq!(loaded.decisions, persisted.decisions);

        let _ = std::fs::remove_dir_all(store_path);
    }
}
