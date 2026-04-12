use std::cmp::Ordering;

use hashbrown::HashMap;
use phoenix_store_native_core::SemanticNodeNeighbor;

use crate::semantic_graph_support::Prototype;

#[derive(Clone, Copy, Debug)]
struct CachedNeighborHit {
    prototype_index: usize,
    distance: f64,
}

#[derive(Clone, Debug, Default)]
struct CachedNeighborList {
    search_limit: usize,
    hits: Vec<CachedNeighborHit>,
}

pub(crate) struct SemanticNeighborWorkspace<'a> {
    folder_id: Option<String>,
    prototypes: &'a [Prototype],
    embeddings: &'a [Vec<f32>],
    indices_by_kind: HashMap<&'static str, Vec<usize>>,
    cache: HashMap<(usize, &'static str), CachedNeighborList>,
}

impl<'a> SemanticNeighborWorkspace<'a> {
    pub(crate) fn new(
        folder_id: Option<String>,
        prototypes: &'a [Prototype],
        embeddings: &'a [Vec<f32>],
    ) -> Self {
        let mut indices_by_kind = HashMap::<&'static str, Vec<usize>>::new();
        for (prototype_index, prototype) in prototypes.iter().enumerate() {
            indices_by_kind
                .entry(prototype.ann_kind)
                .or_default()
                .push(prototype_index);
        }
        Self {
            folder_id,
            prototypes,
            embeddings,
            indices_by_kind,
            cache: HashMap::new(),
        }
    }

    pub(crate) fn query_semantic_node_neighbors(
        &mut self,
        source_index: usize,
        target_kind: &'static str,
        limit: usize,
        oversample: usize,
    ) -> Vec<SemanticNodeNeighbor> {
        if limit == 0 || target_kind.is_empty() {
            return Vec::new();
        }
        let search_limit = oversample.max(limit).max(1);
        let key = (source_index, target_kind);
        let needs_rebuild = self
            .cache
            .get(&key)
            .map(|cached| cached.search_limit < search_limit)
            .unwrap_or(true);
        if needs_rebuild {
            let cached = self.build_cached_neighbors(source_index, target_kind, search_limit);
            self.cache.insert(key, cached);
        }
        self.cache
            .get(&key)
            .map(|cached| {
                cached
                    .hits
                    .iter()
                    .take(limit)
                    .map(|hit| self.neighbor_from_hit(*hit))
                    .collect()
            })
            .unwrap_or_default()
    }

    fn build_cached_neighbors(
        &self,
        source_index: usize,
        target_kind: &'static str,
        search_limit: usize,
    ) -> CachedNeighborList {
        let Some(target_indices) = self.indices_by_kind.get(target_kind) else {
            return CachedNeighborList {
                search_limit,
                hits: Vec::new(),
            };
        };
        let Some(source_embedding) = self.embeddings.get(source_index) else {
            return CachedNeighborList {
                search_limit,
                hits: Vec::new(),
            };
        };
        let mut hits = Vec::<CachedNeighborHit>::with_capacity(target_indices.len());
        for &target_index in target_indices {
            if target_index == source_index {
                continue;
            }
            let Some(target_embedding) = self.embeddings.get(target_index) else {
                continue;
            };
            hits.push(CachedNeighborHit {
                prototype_index: target_index,
                distance: embedding_distance(
                    source_embedding.as_slice(),
                    target_embedding.as_slice(),
                ),
            });
        }
        if hits.len() > search_limit {
            hits.select_nth_unstable_by(search_limit, compare_cached_hit);
            hits.truncate(search_limit);
        }
        hits.sort_unstable_by(compare_cached_hit);
        CachedNeighborList { search_limit, hits }
    }

    fn neighbor_from_hit(&self, hit: CachedNeighborHit) -> SemanticNodeNeighbor {
        let prototype = &self.prototypes[hit.prototype_index];
        SemanticNodeNeighbor {
            node_id: prototype.node_id.clone(),
            node_kind: prototype.ann_kind.to_owned(),
            distance: hit.distance,
            document_id: prototype.document_id.clone(),
            narrative_id: prototype.narrative_id.clone(),
            folder_id: self.folder_id.clone(),
            evidence_refs: prototype.evidence_refs.clone(),
        }
    }
}

fn compare_cached_hit(left: &CachedNeighborHit, right: &CachedNeighborHit) -> Ordering {
    left.distance
        .total_cmp(&right.distance)
        .then_with(|| left.prototype_index.cmp(&right.prototype_index))
}

pub(crate) fn embedding_distance(left: &[f32], right: &[f32]) -> f64 {
    let len = left.len().min(right.len());
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut sum3 = 0.0f64;
    let mut index = 0usize;
    while index + 4 <= len {
        let delta0 = left[index] as f64 - right[index] as f64;
        let delta1 = left[index + 1] as f64 - right[index + 1] as f64;
        let delta2 = left[index + 2] as f64 - right[index + 2] as f64;
        let delta3 = left[index + 3] as f64 - right[index + 3] as f64;
        sum0 += delta0 * delta0;
        sum1 += delta1 * delta1;
        sum2 += delta2 * delta2;
        sum3 += delta3 * delta3;
        index += 4;
    }
    let mut sum = sum0 + sum1 + sum2 + sum3;
    while index < len {
        let delta = left[index] as f64 - right[index] as f64;
        sum += delta * delta;
        index += 1;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use phoenix_semantic_v2::{SemanticGraphNodeKind, SemanticGraphNodeRecord};

    use super::SemanticNeighborWorkspace;
    use crate::semantic_graph_support::{Prototype, CLAIM_KIND, STATE_KIND};

    fn prototype(
        node_id: &str,
        ann_kind: &'static str,
        node_kind: SemanticGraphNodeKind,
    ) -> Prototype {
        Prototype {
            node_id: node_id.to_owned(),
            ann_kind,
            node_kind,
            text_key: node_id.to_owned(),
            text: node_id.to_owned(),
            truth_plane: Some("world".to_owned()),
            document_id: Some("doc-1".to_owned()),
            narrative_id: Some("nar-1".to_owned()),
            evidence_refs: vec![format!("evidence://{node_id}")],
            semantic_node: SemanticGraphNodeRecord {
                node_id: node_id.to_owned(),
                node_kind,
                document_id: Some("doc-1".to_owned()),
                narrative_id: Some("nar-1".to_owned()),
                text_key: node_id.to_owned(),
                text_hash: 1,
                truth_plane: Some("world".to_owned()),
                evidence_refs: Vec::new(),
            },
            slot_key: None,
            value_key: None,
            primary_entity_id: None,
            secondary_entity_id: None,
        }
    }

    #[test]
    fn workspace_skips_self_and_orders_hits_by_distance() {
        let prototypes = vec![
            prototype("graph::claim::1", CLAIM_KIND, SemanticGraphNodeKind::Claim),
            prototype("graph::claim::2", CLAIM_KIND, SemanticGraphNodeKind::Claim),
            prototype("graph::claim::3", CLAIM_KIND, SemanticGraphNodeKind::Claim),
        ];
        let embeddings = vec![vec![0.0f32, 0.0], vec![0.1f32, 0.0], vec![0.9f32, 0.0]];
        let mut workspace =
            SemanticNeighborWorkspace::new(Some("folder-a".to_owned()), &prototypes, &embeddings);

        let hits = workspace.query_semantic_node_neighbors(0, CLAIM_KIND, 2, 2);

        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].node_id, "graph::claim::2");
        assert_eq!(hits[1].node_id, "graph::claim::3");
        assert_eq!(hits[0].folder_id.as_deref(), Some("folder-a"));
    }

    #[test]
    fn workspace_returns_empty_for_unknown_kind() {
        let prototypes = vec![prototype(
            "graph::state::1",
            STATE_KIND,
            SemanticGraphNodeKind::State,
        )];
        let embeddings = vec![vec![0.0f32, 0.0]];
        let mut workspace = SemanticNeighborWorkspace::new(None, &prototypes, &embeddings);

        let hits = workspace.query_semantic_node_neighbors(0, "", 4, 8);

        assert!(hits.is_empty());
    }
}
