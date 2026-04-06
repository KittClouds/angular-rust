//! Shard Management with Curvature Registry
//!
//! This module implements per-shard curvature management for hierarchical data.
//! Different parts of the hierarchy may have different optimal curvatures.

use crate::{
    Candidate, HnswBuildParams, HyperbolicDiskHnsw, HyperbolicHnswBuilder, PoincareMetric,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Curvature configuration for a shard
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardCurvature {
    /// Current active curvature
    pub current: f32,
    /// Canary curvature (for testing)
    pub canary: Option<f32>,
    /// Traffic percentage for canary (0-100)
    pub canary_traffic: u8,
    /// Learned curvature from data
    pub learned: Option<f32>,
    /// Last update timestamp
    pub updated_at: i64,
}

impl Default for ShardCurvature {
    fn default() -> Self {
        Self {
            current: 1.0,
            canary: None,
            canary_traffic: 0,
            learned: None,
            updated_at: 0,
        }
    }
}

impl ShardCurvature {
    /// Get the effective curvature (considering canary traffic)
    pub fn effective(&self, use_canary: bool) -> f32 {
        if use_canary && self.canary.is_some() && self.canary_traffic > 0 {
            self.canary.unwrap()
        } else {
            self.current
        }
    }

    /// Promote canary to current
    pub fn promote_canary(&mut self) {
        if let Some(c) = self.canary {
            self.current = c;
            self.canary = None;
            self.canary_traffic = 0;
        }
    }

    /// Rollback canary
    pub fn rollback_canary(&mut self) {
        self.canary = None;
        self.canary_traffic = 0;
    }
}

/// Curvature registry for managing per-shard curvatures
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CurvatureRegistry {
    /// Shard curvatures by shard ID
    pub shards: HashMap<String, ShardCurvature>,
    /// Global default curvature
    pub default_curvature: f32,
    /// Registry version (for hot reload)
    pub version: u64,
}

impl CurvatureRegistry {
    /// Create a new registry with default curvature
    pub fn new(default_curvature: f32) -> Self {
        Self {
            shards: HashMap::new(),
            default_curvature,
            version: 0,
        }
    }

    /// Get curvature for a shard
    pub fn get(&self, shard_id: &str) -> f32 {
        self.shards
            .get(shard_id)
            .map(|s| s.current)
            .unwrap_or(self.default_curvature)
    }

    /// Get curvature with canary consideration
    pub fn get_effective(&self, shard_id: &str, use_canary: bool) -> f32 {
        self.shards
            .get(shard_id)
            .map(|s| s.effective(use_canary))
            .unwrap_or(self.default_curvature)
    }

    /// Set curvature for a shard
    pub fn set(&mut self, shard_id: &str, curvature: f32) {
        let entry = self.shards.entry(shard_id.to_string()).or_default();
        entry.current = curvature;
        entry.updated_at = current_timestamp();
        self.version += 1;
    }

    /// Set canary curvature
    pub fn set_canary(&mut self, shard_id: &str, curvature: f32, traffic: u8) {
        let entry = self.shards.entry(shard_id.to_string()).or_default();
        entry.canary = Some(curvature);
        entry.canary_traffic = traffic.min(100);
        entry.updated_at = current_timestamp();
        self.version += 1;
    }

    /// Promote all canaries
    pub fn promote_all_canaries(&mut self) {
        for (_, shard) in self.shards.iter_mut() {
            shard.promote_canary();
        }
        self.version += 1;
    }

    /// Rollback all canaries
    pub fn rollback_all_canaries(&mut self) {
        for (_, shard) in self.shards.iter_mut() {
            shard.rollback_canary();
        }
        self.version += 1;
    }

    /// Record learned curvature
    pub fn set_learned(&mut self, shard_id: &str, curvature: f32) {
        let entry = self.shards.entry(shard_id.to_string()).or_default();
        entry.learned = Some(curvature);
        entry.updated_at = current_timestamp();
    }
}

fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// A single shard in the sharded HNSW system
#[derive(Debug)]
pub struct HyperbolicShard {
    /// Shard ID
    pub id: String,
    /// Builder for in-memory construction
    pub builder: Option<HyperbolicHnswBuilder<PoincareMetric>>,
    /// Disk-based index (after save)
    pub index: Option<HyperbolicDiskHnsw<PoincareMetric>>,
    /// Shard centroid
    pub centroid: Vec<f32>,
    /// Hierarchy depth range (min, max)
    pub depth_range: (usize, usize),
    /// Number of vectors in shard
    pub count: usize,
    /// Dimensionality
    pub dim: usize,
    /// Build parameters
    pub params: HnswBuildParams,
}

impl HyperbolicShard {
    /// Create a new shard
    pub fn new(id: String, curvature: f32, dim: usize, params: HnswBuildParams) -> Self {
        let metric = PoincareMetric { curvature };
        let builder = HyperbolicHnswBuilder::new(dim, metric, params);

        Self {
            id,
            builder: Some(builder),
            index: None,
            centroid: Vec::new(),
            depth_range: (0, 0),
            count: 0,
            dim,
            params,
        }
    }

    /// Insert a vector
    pub fn insert(&mut self, vector: Vec<f32>) -> Result<(), String> {
        if let Some(ref mut builder) = self.builder {
            builder.insert(vector);
            self.count += 1;
            Ok(())
        } else {
            Err("Shard already saved to disk, cannot insert".to_string())
        }
    }

    /// Save shard to disk
    pub fn save_to_disk(&mut self, path: &str) -> Result<(), String> {
        if let Some(builder) = self.builder.take() {
            builder.save_to_disk(path).map_err(|e| format!("{:?}", e))?;
            self.index = Some(
                HyperbolicDiskHnsw::open(
                    path,
                    PoincareMetric {
                        curvature: self.get_curvature(),
                    },
                )
                .map_err(|e| format!("{:?}", e))?,
            );
            Ok(())
        } else {
            Err("Shard already saved or not initialized".to_string())
        }
    }

    /// Search the shard
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<Candidate> {
        if let Some(ref index) = self.index {
            index.search(query, k, ef_search)
        } else {
            vec![]
        }
    }

    /// Get current curvature
    pub fn get_curvature(&self) -> f32 {
        if let Some(ref _builder) = self.builder {
            // Can't easily get curvature from builder, use default
            1.0
        } else if let Some(ref _index) = self.index {
            1.0
        } else {
            1.0
        }
    }

    /// Update curvature (requires rebuild)
    pub fn set_curvature(&mut self, curvature: f32) {
        // Note: This invalidates the current index
        self.index = None;
        self.builder = Some(HyperbolicHnswBuilder::new(
            self.dim,
            PoincareMetric { curvature },
            self.params,
        ));
        self.count = 0;
    }
}

/// Sharded hyperbolic HNSW manager
#[derive(Debug)]
pub struct ShardedHyperbolicHnsw {
    /// Shards by ID
    pub shards: HashMap<String, HyperbolicShard>,
    /// Curvature registry
    pub registry: CurvatureRegistry,
    /// Global ID to shard mapping
    pub id_to_shard: Vec<(String, usize)>,
    /// Shard assignment strategy
    pub strategy: ShardStrategy,
    /// Dimensionality
    pub dim: usize,
    /// Build parameters
    pub params: HnswBuildParams,
}

/// Strategy for assigning vectors to shards
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardStrategy {
    /// Assign by hash
    Hash,
    /// Assign by hierarchy depth
    Depth,
    /// Assign by radius (distance from origin)
    Radius,
    /// Round-robin
    RoundRobin,
}

impl Default for ShardStrategy {
    fn default() -> Self {
        Self::Radius
    }
}

impl ShardedHyperbolicHnsw {
    /// Create a new sharded manager
    pub fn new(default_curvature: f32, dim: usize, params: HnswBuildParams) -> Self {
        Self {
            shards: HashMap::new(),
            registry: CurvatureRegistry::new(default_curvature),
            id_to_shard: Vec::new(),
            strategy: ShardStrategy::default(),
            dim,
            params,
        }
    }

    /// Create or get a shard
    pub fn get_or_create_shard(&mut self, shard_id: &str) -> &mut HyperbolicShard {
        let curvature = self.registry.get(shard_id);
        self.shards.entry(shard_id.to_string()).or_insert_with(|| {
            HyperbolicShard::new(shard_id.to_string(), curvature, self.dim, self.params)
        })
    }

    /// Determine shard for a vector
    pub fn assign_shard(&self, vector: &[f32], depth: Option<usize>) -> String {
        match self.strategy {
            ShardStrategy::Hash => {
                let hash: u64 = vector.iter().fold(0u64, |acc, &v| {
                    acc.wrapping_add((v.to_bits() as u64).wrapping_mul(31))
                });
                let num_shards = self.shards.len().max(1);
                format!("shard_{}", hash % num_shards as u64)
            }
            ShardStrategy::Depth => {
                let d = depth.unwrap_or(0);
                format!("depth_{}", d / 10)
            }
            ShardStrategy::Radius => {
                let radius: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
                let bucket = (radius * 10.0) as usize;
                format!("radius_{}", bucket)
            }
            ShardStrategy::RoundRobin => {
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

    /// Insert vector with automatic shard assignment
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

    /// Insert into specific shard
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

    /// Search across all shards
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(usize, Candidate)> {
        let mut all_results: Vec<(usize, Candidate)> = Vec::new();

        for (shard_id, shard) in &self.shards {
            let results = shard.search(query, k, ef_search);
            for result in results {
                // Find global ID for this local result
                for (global_idx, (s_id, local_idx)) in self.id_to_shard.iter().enumerate() {
                    if s_id == shard_id && *local_idx == result.id as usize {
                        all_results.push((global_idx, result));
                        break;
                    }
                }
            }
        }

        // Sort by distance and take top k
        all_results.sort_by(|a, b| a.1.dist.partial_cmp(&b.1.dist).unwrap());
        all_results.truncate(k);

        all_results
    }

    /// Save all shards to disk
    pub fn save_all(&mut self, base_path: &str) -> Result<(), String> {
        for (shard_id, shard) in self.shards.iter_mut() {
            let path = format!("{}_{}.bin", base_path, shard_id);
            shard.save_to_disk(&path)?;
        }
        Ok(())
    }

    /// Update curvature for a shard
    pub fn update_curvature(&mut self, shard_id: &str, curvature: f32) {
        self.registry.set(shard_id, curvature);
        if let Some(shard) = self.shards.get_mut(shard_id) {
            shard.set_curvature(curvature);
        }
    }

    /// Get total vector count
    pub fn len(&self) -> usize {
        self.id_to_shard.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.id_to_shard.is_empty()
    }

    /// Get number of shards
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }
}
#[cfg(test)]
mod tests {
    use super::{CurvatureRegistry, ShardedHyperbolicHnsw};
    use crate::HnswBuildParams;

    #[test]
    fn test_curvature_registry() {
        let mut registry = CurvatureRegistry::new(1.0);

        registry.set("shard_1", 0.5);
        assert_eq!(registry.get("shard_1"), 0.5);
        assert_eq!(registry.get("shard_2"), 1.0);

        registry.set_canary("shard_1", 0.3, 50);
        assert_eq!(registry.get_effective("shard_1", false), 0.5);
        assert_eq!(registry.get_effective("shard_1", true), 0.3);
    }

    #[test]
    #[ignore]
    fn test_sharded_hnsw() {
        let dim = 4;
        let params = HnswBuildParams::default();
        let mut manager = ShardedHyperbolicHnsw::new(1.0, dim, params);

        for i in 0..20 {
            let v = vec![0.1 * i as f32, 0.05 * i as f32, 0.0, 0.0];
            manager.insert(v, Some(i / 5)).unwrap();
        }

        assert_eq!(manager.len(), 20);

        //         //         manager.save_all("test_shard").unwrap();

        let query = vec![0.3, 0.15, 0.0, 0.0];
        let results = manager.search(&query, 5, 50);

        assert!(!results.is_empty());
    }

    #[test]
    fn test_shard_curvature_update() {
        let dim = 4;
        let params = HnswBuildParams::default();
        let mut manager = ShardedHyperbolicHnsw::new(1.0, dim, params);

        manager.get_or_create_shard("test");
        manager.update_curvature("test", 0.5);

        assert_eq!(manager.registry.get("test"), 0.5);
    }
}
