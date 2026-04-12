use serde::{Deserialize, Serialize};

use phoenix_semantic_v2::{DirtyScopeRecord, ScopeOrd};
use phoenix_types::SessionId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PipelineStage {
    Relation,
    StateSchema,
    Memory,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PipelineRunShape {
    PostIngest,
    LateSidecars,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PipelineStageStatus {
    #[default]
    NotRequested,
    Blocked,
    Ready,
    Running,
    Complete,
    Failed,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineRunMetrics {
    pub scope_count: usize,
    pub requested_stage_count: usize,
    pub runtime_image_cache_hits: usize,
    pub runtime_image_cache_misses: usize,
    pub analysis_cache_hits: usize,
    pub analysis_cache_misses: usize,
    pub relation_prepared_input_cache_hits: usize,
    pub relation_prepared_input_cache_misses: usize,
    pub stage_ready_count: usize,
    pub stage_run_count: usize,
    pub stage_complete_count: usize,
    pub stage_product_count: usize,
    pub relation_model_job_count: usize,
    pub relation_model_job_window_count: usize,
    pub relation_model_job_pair_slots: usize,
    pub relation_schema_group_count: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeGenerationKey {
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub generation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineRunRequest {
    pub shape: PipelineRunShape,
    pub session_id: Option<SessionId>,
    #[serde(default)]
    pub requested_stages: Vec<PipelineStage>,
}

impl PipelineRunRequest {
    pub fn post_ingest(session_id: Option<&SessionId>) -> Self {
        Self {
            shape: PipelineRunShape::PostIngest,
            session_id: session_id.cloned(),
            requested_stages: vec![
                PipelineStage::Relation,
                PipelineStage::StateSchema,
                PipelineStage::Memory,
            ],
        }
    }

    pub fn late_sidecars(session_id: Option<&SessionId>) -> Self {
        Self {
            shape: PipelineRunShape::LateSidecars,
            session_id: session_id.cloned(),
            requested_stages: vec![PipelineStage::StateSchema, PipelineStage::Memory],
        }
    }

    pub fn requests_stage(&self, stage: PipelineStage) -> bool {
        self.requested_stages.contains(&stage)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StageProductEnvelope<T> {
    pub key: ScopeGenerationKey,
    pub stage: PipelineStage,
    pub created_at: i64,
    pub input_fingerprint: u64,
    pub payload: T,
}

impl ScopeGenerationKey {
    pub fn from_dirty_scope(dirty: &DirtyScopeRecord) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        dirty.scope_key.hash(&mut hasher);
        dirty.scope_ord.hash(&mut hasher);
        dirty.updated_at.hash(&mut hasher);
        dirty.document_ords.hash(&mut hasher);
        Self {
            scope_key: dirty.scope_key.clone(),
            scope_ord: dirty.scope_ord,
            generation: hasher.finish(),
        }
    }
}

pub fn stage_dependencies(stage: PipelineStage) -> &'static [PipelineStage] {
    match stage {
        PipelineStage::Relation => &[],
        PipelineStage::StateSchema => &[PipelineStage::Relation],
        PipelineStage::Memory => &[PipelineStage::StateSchema, PipelineStage::Relation],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_semantic_v2::DocumentOrd;
    use phoenix_types::ScopeKey;

    #[test]
    fn post_ingest_request_uses_expected_stage_order() {
        let request = PipelineRunRequest::post_ingest(None);
        assert_eq!(
            request.requested_stages,
            vec![
                PipelineStage::Relation,
                PipelineStage::StateSchema,
                PipelineStage::Memory
            ]
        );
    }

    #[test]
    fn late_sidecars_request_uses_expected_stage_order() {
        let request = PipelineRunRequest::late_sidecars(None);
        assert_eq!(
            request.requested_stages,
            vec![PipelineStage::StateSchema, PipelineStage::Memory]
        );
    }

    #[test]
    fn scope_generation_key_tracks_dirty_scope_shape() {
        let dirty = DirtyScopeRecord {
            scope: ScopeKey {
                world_id: Some("world".to_owned()),
                narrative_id: Some("story".to_owned()),
                folder_id: None,
                folder_path: None,
            },
            scope_key: "world::story".to_owned(),
            scope_ord: ScopeOrd(7),
            document_ords: vec![DocumentOrd(1), DocumentOrd(2)],
            updated_at: 1234,
        };
        let first = ScopeGenerationKey::from_dirty_scope(&dirty);
        let second = ScopeGenerationKey::from_dirty_scope(&dirty);
        assert_eq!(first, second);

        let mut changed = dirty.clone();
        changed.document_ords.push(DocumentOrd(3));
        assert_ne!(
            first.generation,
            ScopeGenerationKey::from_dirty_scope(&changed).generation
        );
    }
}
