//! Sphere-native shard manager.
//!
//! Unlike the hyperbolic shard manager, there is no curvature registry.
//! Default strategy is hash sharding because all normalized vectors have radius 1.

use crate::{
    sphere::SphereMetric, Candidate, HnswBuildParams, HyperbolicDiskError, HyperbolicDiskHnsw,
    HyperbolicHnswBuilder,
};
use std::collections::HashMap;

#[derive(Debug)]
pub struct SphereShard {
    pub id: String,
    pub metric: SphereMetric,
    pub builder: Option<HyperbolicHnswBuilder<SphereMetric>>,
    pub index: Option<HyperbolicDiskHnsw<SphereMetric>>,
    pub centroid: Vec<f32>,
    pub depth_range: (usize, usize),
    pub count: usize,
    pub dim: usize,
    pub params: HnswBuildParams,
}

impl SphereShard {
    pub fn new(id: String, metric: SphereMetric, dim: usize, params: HnswBuildParams) -> Self {
        Self {
            id,
            metric,
            builder: Some(HyperbolicHnswBuilder::new(dim, metric, params)),
            index: None,
            centroid: Vec::new(),
            depth_range: (0, 0),
            count: 0,
            dim,
            params,
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>) -> Result<(), String> {
        if let Some(ref mut builder) = self.builder {
            builder.insert(vector);
            self.count += 1;
            Ok(())
        } else {
            Err("Shard already saved to disk, cannot insert".to_string())
        }
    }

    pub fn save_to_disk(&mut self, path: &str) -> Result<(), String> {
        if let Some(builder) = self.builder.take() {
            builder.save_to_disk(path).map_err(|e| format!("{e:?}"))?;
            self.index = Some(
                HyperbolicDiskHnsw::open(path, self.metric)
                    .map_err(|e: HyperbolicDiskError| format!("{e:?}"))?,
            );
            Ok(())
        } else {
            Err("Shard already saved or not initialized".to_string())
        }
    }

    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<Candidate> {
        if let Some(ref index) = self.index {
            index.search(query, k, ef_search)
        } else {
            vec![]
        }
    }

    pub fn reset(&mut self) {
        self.index = None;
        self.builder = Some(HyperbolicHnswBuilder::new(
            self.dim,
            self.metric,
            self.params,
        ));
        self.count = 0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SphereShardStrategy {
    Hash,
    Depth,
    RoundRobin,
}

impl Default for SphereShardStrategy {
    fn default() -> Self {
        Self::Hash
    }
}

#[derive(Debug)]
pub struct ShardedSphereHnsw {
    pub shards: HashMap<String, SphereShard>,
    pub id_to_shard: Vec<(String, usize)>,
    pub strategy: SphereShardStrategy,
    pub dim: usize,
    pub params: HnswBuildParams,
    pub metric: SphereMetric,
}

impl ShardedSphereHnsw {
    pub fn new(dim: usize, params: HnswBuildParams, metric: SphereMetric) -> Self {
        Self {
            shards: HashMap::new(),
            id_to_shard: Vec::new(),
            strategy: SphereShardStrategy::default(),
            dim,
            params,
            metric,
        }
    }

    pub fn get_or_create_shard(&mut self, shard_id: &str) -> &mut SphereShard {
        self.shards.entry(shard_id.to_string()).or_insert_with(|| {
            SphereShard::new(shard_id.to_string(), self.metric, self.dim, self.params)
        })
    }

    pub fn assign_shard(&self, vector: &[f32], depth: Option<usize>) -> String {
        match self.strategy {
            SphereShardStrategy::Hash => {
                let hash: u64 = vector.iter().fold(0u64, |acc, &v| {
                    acc.wrapping_add((v.to_bits() as u64).wrapping_mul(31))
                });
                let num_shards = self.shards.len().max(1);
                format!("shard_{}", hash % num_shards as u64)
            }
            SphereShardStrategy::Depth => {
                let d = depth.unwrap_or(0);
                format!("depth_{}", d / 10)
            }
            SphereShardStrategy::RoundRobin => {
                let num_shards = self.shards.len().max(1);
                let idx = self.id_to_shard.len() % num_shards;
                self.shards
                    .keys()
                    .nth(idx)
                    .cloned()
                    .unwrap_or_else(|| "default".to_string())
            }
        }
    }

    pub fn insert(&mut self, vector: Vec<f32>, depth: Option<usize>) -> Result<usize, String> {
        let shard_id = self.assign_shard(&vector, depth);

        {
            let shard = self.get_or_create_shard(&shard_id);
            shard.insert(vector)?;
        }

        let shard = self.shards.get(&shard_id).unwrap();
        let global_id = self.id_to_shard.len();
        let local_id = shard.count - 1;
        self.id_to_shard.push((shard_id, local_id));

        Ok(global_id)
    }

    pub fn insert_to_shard(&mut self, shard_id: &str, vector: Vec<f32>) -> Result<usize, String> {
        {
            let shard = self.get_or_create_shard(shard_id);
            shard.insert(vector)?;
        }

        let shard = self.shards.get(shard_id).unwrap();
        let global_id = self.id_to_shard.len();
        let local_id = shard.count - 1;
        self.id_to_shard.push((shard_id.to_string(), local_id));

        Ok(global_id)
    }

    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(usize, Candidate)> {
        let mut all_results: Vec<(usize, Candidate)> = Vec::new();

        for (shard_id, shard) in &self.shards {
            let results = shard.search(query, k, ef_search);
            for result in results {
                for (global_idx, (s_id, local_idx)) in self.id_to_shard.iter().enumerate() {
                    if s_id == shard_id && *local_idx == result.id as usize {
                        all_results.push((global_idx, result));
                        break;
                    }
                }
            }
        }

        all_results.sort_by(|a, b| a.1.dist.total_cmp(&b.1.dist));
        all_results.truncate(k);
        all_results
    }

    pub fn save_all(&mut self, base_path: &str) -> Result<(), String> {
        for (shard_id, shard) in self.shards.iter_mut() {
            let path = format!("{}_{}.bin", base_path, shard_id);
            shard.save_to_disk(&path)?;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.id_to_shard.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_shard.is_empty()
    }

    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sphere::SphereDistance;

    #[test]
    fn sharded_sphere_basic() {
        let dim = 4;
        let params = HnswBuildParams::default();
        let metric = SphereMetric {
            distance: SphereDistance::Geodesic,
        };

        let mut manager = ShardedSphereHnsw::new(dim, params, metric);

        for i in 0..20 {
            let v = vec![0.1 * i as f32 + 0.1, 0.05 * i as f32 + 0.1, 0.2, 0.3];
            manager.insert(v, Some(i / 5)).unwrap();
        }

        assert_eq!(manager.len(), 20);
    }
}
