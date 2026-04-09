use compact_str::CompactString;
use phoenix_kernel::{DeterministicKernel, KernelEntityResolveRequest, KernelGraphSnapshot};
use phoenix_types::{
    CandidateEntityRef, CorefCluster, EntityId, ResolutionDecisionRecord, TruthStatus,
};
use std::sync::Arc;

#[derive(Default)]
pub struct ReferenceKernel {
    kernel: DeterministicKernel,
}

impl ReferenceKernel {
    pub fn new(kernel: DeterministicKernel) -> Self {
        Self { kernel }
    }

    pub fn snapshot(&self) -> Arc<KernelGraphSnapshot> {
        self.kernel.snapshot()
    }

    pub fn entity_candidates_for_surface(&self, surface: &str) -> Vec<CandidateEntityRef> {
        self.kernel
            .entity_candidates(KernelEntityResolveRequest {
                surface: Some(surface.to_owned()),
                mention_vertex_id: None,
                canonical_entity_id: None,
                limit: None,
                include_candidate_graph: true,
                valid_at: None,
                recorded_at: None,
            })
            .into_iter()
            .map(|candidate| CandidateEntityRef {
                entity_id: EntityId(candidate.entity_id),
                source: CompactString::from(
                    candidate
                        .relation_type
                        .unwrap_or_else(|| "kernel".to_owned()),
                ),
                score_millis: (candidate.score * 1000.0) as i32,
            })
            .collect()
    }

    pub fn unresolved_cluster(cluster_id: &str) -> CorefCluster {
        CorefCluster {
            cluster_id: CompactString::from(cluster_id),
            member_mentions: Default::default(),
            representative_surface: CompactString::default(),
            confidence_millis: 0,
            ambiguous: true,
        }
    }

    pub fn unresolved_decision(mention_index: usize) -> ResolutionDecisionRecord {
        ResolutionDecisionRecord {
            mention_index,
            entity_id: None,
            status: TruthStatus::Candidate,
            confidence_millis: 0,
            margin_millis: 0,
        }
    }

    pub fn kernel(&self) -> &DeterministicKernel {
        &self.kernel
    }
}
