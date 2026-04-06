//! Hyperbolic HNSW with DiskANN-style mmap persistence
//!
//! This module adapts HNSW layered graph architecture into a DiskANN-style
//! single-file `mmap` structure. It is designed specifically for hyperbolic
//! vectors (Poincaré ball) using `f32`.

pub mod poincare;
pub mod shard;
pub mod tangent;

use memmap2::{Mmap, MmapMut};
use rand::{prelude::*, thread_rng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use thiserror::Error;

const EPS: f32 = 1e-6;

#[derive(Debug, Error)]
pub enum HyperbolicDiskError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Bincode(#[from] bincode::Error),
    #[error("Index error: {0}")]
    IndexError(String),
}

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub dist: f32,
    pub id: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.dist.to_bits() == other.dist.to_bits()
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            self.dist
                .total_cmp(&other.dist)
                .then_with(|| self.id.cmp(&other.id)),
        )
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

pub trait MetricF32: Send + Sync + Clone + 'static {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32;
    fn project_to_ball(&self, vector: &mut [f32]);
}

#[derive(Clone, Copy, Debug)]
pub struct PoincareMetric {
    pub curvature: f32,
}

impl MetricF32 for PoincareMetric {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        let mut diff_sq = 0.0;
        let mut norm_a_sq = 0.0;
        let mut norm_b_sq = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            let diff = x - y;
            diff_sq += diff * diff;
            norm_a_sq += x * x;
            norm_b_sq += y * y;
        }

        let max_norm = 1.0 - EPS;
        norm_a_sq = norm_a_sq.min(max_norm);
        norm_b_sq = norm_b_sq.min(max_norm);

        let c = self.curvature.abs();
        let num = 2.0 * c * diff_sq;
        let den = (1.0 - c * norm_a_sq) * (1.0 - c * norm_b_sq);
        let delta = num / den.max(EPS);

        (1.0 + delta).acosh() / c.sqrt()
    }

    fn project_to_ball(&self, vector: &mut [f32]) {
        let norm_sq: f32 = vector.iter().map(|v| v * v).sum();
        let c = self.curvature.abs();
        let max_radius = 1.0 / c.sqrt() - EPS;

        if norm_sq > max_radius * max_radius {
            let scale = max_radius / norm_sq.sqrt();
            for v in vector.iter_mut() {
                *v *= scale;
            }
        }
    }
}

#[derive(Debug)]
pub struct BuildNode {
    pub id: u32,
    pub vector: Vec<f32>,
    pub connections: Vec<Vec<u32>>,
    pub level: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct HnswBuildParams {
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub level_mult: f32,
}

impl Default for HnswBuildParams {
    fn default() -> Self {
        Self {
            m: 16,
            m0: 32,
            ef_construction: 200,
            level_mult: 1.0 / (16.0_f32).ln(),
        }
    }
}
#[derive(Debug)]
pub struct HyperbolicHnswBuilder<M: MetricF32> {
    nodes: Vec<BuildNode>,
    entry_point: Option<u32>,
    max_level: u32,
    params: HnswBuildParams,
    metric: M,
    dim: usize,
}

impl<M: MetricF32> HyperbolicHnswBuilder<M> {
    pub fn new(dim: usize, metric: M, params: HnswBuildParams) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            params,
            metric,
            dim,
        }
    }

    fn random_level(&self) -> u32 {
        let r: f32 = thread_rng().gen();
        (-r.ln() * self.params.level_mult) as u32
    }

    pub fn insert(&mut self, mut vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dim);
        self.metric.project_to_ball(&mut vector);

        let id = self.nodes.len() as u32;
        let level = self.random_level();
        let mut new_node = BuildNode {
            id,
            vector,
            connections: vec![Vec::new(); (level + 1) as usize],
            level,
        };

        if let Some(entry_id) = self.entry_point {
            let mut curr = entry_id;
            let mut curr_dist = self
                .metric
                .eval(&new_node.vector, &self.nodes[curr as usize].vector);

            for l in (level + 1..=self.max_level).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    for &nb in &self.nodes[curr as usize].connections[l as usize] {
                        let d = self
                            .metric
                            .eval(&new_node.vector, &self.nodes[nb as usize].vector);
                        if d < curr_dist {
                            curr_dist = d;
                            curr = nb;
                            changed = true;
                        }
                    }
                }
            }

            for l in (0..=level.min(self.max_level)).rev() {
                let max_c = if l == 0 {
                    self.params.m0
                } else {
                    self.params.m
                };

                let candidates =
                    self.search_layer(&new_node.vector, curr, self.params.ef_construction, l);
                let selected = self.select_neighbors_heuristic(&new_node.vector, candidates, max_c);

                new_node.connections[l as usize] = selected.clone();

                for nb_id in selected {
                    self.nodes[nb_id as usize].connections[l as usize].push(id);

                    let nb_conn_len = self.nodes[nb_id as usize].connections[l as usize].len();
                    if nb_conn_len > max_c {
                        let mut nb_cands = Vec::with_capacity(nb_conn_len);
                        for &c_id in &self.nodes[nb_id as usize].connections[l as usize] {
                            if (c_id as usize) < self.nodes.len() {
                                let dist = self.metric.eval(
                                    &self.nodes[nb_id as usize].vector,
                                    &self.nodes[c_id as usize].vector,
                                );
                                nb_cands.push(Candidate { id: c_id, dist });
                            }
                        }

                        let query_vector = self.nodes[nb_id as usize].vector.clone();
                        let new_selected =
                            self.select_neighbors_heuristic(&query_vector, nb_cands, max_c);
                        self.nodes[nb_id as usize].connections[l as usize] = new_selected;
                    }
                }

                curr = new_node.connections[l as usize]
                    .first()
                    .copied()
                    .unwrap_or(curr);
            }
        }

        self.nodes.push(new_node);

        if self.entry_point.is_none() || level > self.max_level {
            self.entry_point = Some(id);
            self.max_level = level;
        }
    }

    fn search_layer(&self, query: &[f32], entry: u32, ef: usize, level: u32) -> Vec<Candidate> {
        let entry_dist = self.metric.eval(query, &self.nodes[entry as usize].vector);

        let mut visited = HashSet::new();
        visited.insert(entry);

        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        candidates.push(std::cmp::Reverse(Candidate {
            id: entry,
            dist: entry_dist,
        }));
        results.push(Candidate {
            id: entry,
            dist: entry_dist,
        });

        while let Some(std::cmp::Reverse(cand)) = candidates.pop() {
            if cand.dist > results.peek().unwrap().dist && results.len() >= ef {
                break;
            }

            for &nb in &self.nodes[cand.id as usize].connections[level as usize] {
                if visited.insert(nb) {
                    let d = self.metric.eval(query, &self.nodes[nb as usize].vector);

                    if results.len() < ef || d < results.peek().unwrap().dist {
                        candidates.push(std::cmp::Reverse(Candidate { id: nb, dist: d }));
                        results.push(Candidate { id: nb, dist: d });

                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut final_cands = results.into_vec();
        final_cands.sort_unstable();
        final_cands
    }

    fn select_neighbors_heuristic(
        &self,
        query: &[f32],
        candidates: Vec<Candidate>,
        max_conn: usize,
    ) -> Vec<u32> {
        if max_conn == 0 || candidates.is_empty() {
            return Vec::new();
        }

        let mut cands = candidates;
        cands.sort_by(|a, b| a.dist.total_cmp(&b.dist).then_with(|| a.id.cmp(&b.id)));
        cands.dedup_by(|a, b| a.id == b.id);

        let mut selected: Vec<u32> = Vec::with_capacity(max_conn);

        for cand in &cands {
            let mut occluded = false;
            for &sel in &selected {
                let d_q_c = cand.dist;
                let d_c_sel = self.metric.eval(
                    &self.nodes[cand.id as usize].vector,
                    &self.nodes[sel as usize].vector,
                );

                if d_c_sel < d_q_c {
                    occluded = true;
                    break;
                }
            }
            if !occluded {
                selected.push(cand.id);
                if selected.len() == max_conn {
                    return selected;
                }
            }
        }

        for cand in &cands {
            if selected.iter().any(|&x| x == cand.id) {
                continue;
            }
            selected.push(cand.id);
            if selected.len() == max_conn {
                break;
            }
        }

        let _ = query;
        selected
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct PackedHnswMetadata {
    dim: usize,
    num_vectors: usize,
    max_level: u32,
    entry_point: u32,
    elem_size: u8,
}

impl PackedHnswMetadata {
    pub const fn new(
        dim: usize,
        num_vectors: usize,
        max_level: u32,
        entry_point: u32,
        elem_size: u8,
    ) -> Self {
        Self {
            dim,
            num_vectors,
            max_level,
            entry_point,
            elem_size,
        }
    }

    pub const fn dim(&self) -> usize {
        self.dim
    }

    pub const fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    pub const fn max_level(&self) -> u32 {
        self.max_level
    }

    pub const fn entry_point(&self) -> u32 {
        self.entry_point
    }

    pub const fn elem_size(&self) -> u8 {
        self.elem_size
    }
}

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct PackedHnswGraph {
    pub metadata: PackedHnswMetadata,
    pub vectors: Vec<u8>,
    pub levels: Vec<u8>,
    pub offsets: Vec<u8>,
    pub adjacency: Vec<u8>,
}

impl PackedHnswGraph {
    pub fn write_to_file(&self, file_path: &str) -> Result<(), HyperbolicDiskError> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .read(true)
            .truncate(true)
            .open(file_path)?;

        let vectors_offset = 1024 * 1024;
        file.seek(SeekFrom::Start(vectors_offset))?;
        if !self.vectors.is_empty() {
            file.write_all(&self.vectors)?;
        }
        let levels_offset = vectors_offset + self.vectors.len() as u64;

        file.seek(SeekFrom::Start(levels_offset))?;
        if !self.levels.is_empty() {
            file.write_all(&self.levels)?;
        }
        let offsets_offset = levels_offset + self.levels.len() as u64;

        file.seek(SeekFrom::Start(offsets_offset))?;
        if !self.offsets.is_empty() {
            file.write_all(&self.offsets)?;
        }
        let adjacency_offset = offsets_offset + self.offsets.len() as u64;

        file.seek(SeekFrom::Start(adjacency_offset))?;
        if !self.adjacency.is_empty() {
            file.write_all(&self.adjacency)?;
        }

        let metadata = DiskMetadata {
            dim: self.metadata.dim,
            num_vectors: self.metadata.num_vectors,
            max_level: self.metadata.max_level,
            entry_point: self.metadata.entry_point,
            vectors_offset,
            levels_offset,
            offsets_offset,
            adjacency_offset,
            elem_size: self.metadata.elem_size,
        };
        let md_bytes = bincode::serialize(&metadata)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(&(md_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&md_bytes)?;

        let file_size = if self.metadata.num_vectors == 0 {
            vectors_offset
        } else if self.adjacency.is_empty() {
            offsets_offset + self.offsets.len() as u64
        } else {
            adjacency_offset + self.adjacency.len() as u64
        };
        file.seek(SeekFrom::Start(file_size - 1))?;
        file.write_all(&[0u8])?;
        file.sync_all()?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct DiskMetadata {
    dim: usize,
    num_vectors: usize,
    max_level: u32,
    entry_point: u32,
    vectors_offset: u64,
    levels_offset: u64,
    offsets_offset: u64,
    adjacency_offset: u64,
    elem_size: u8,
}

impl<M: MetricF32> HyperbolicHnswBuilder<M> {
    pub fn into_packed(self) -> PackedHnswGraph {
        let num_vectors = self.nodes.len();
        let mut flat_vectors: Vec<f32> = Vec::with_capacity(num_vectors * self.dim);
        let mut flat_levels: Vec<u32> = Vec::with_capacity(num_vectors);
        let mut flat_offsets: Vec<u64> = Vec::with_capacity(num_vectors);
        let mut flat_adjacency: Vec<u32> = Vec::new();

        for node in &self.nodes {
            flat_vectors.extend_from_slice(&node.vector);
            flat_levels.push(node.level);
            flat_offsets.push(flat_adjacency.len() as u64 * 4);
            for l in 0..=node.level {
                let neighbors = &node.connections[l as usize];
                flat_adjacency.push(neighbors.len() as u32);
                flat_adjacency.extend_from_slice(neighbors);
            }
        }

        PackedHnswGraph {
            metadata: PackedHnswMetadata {
                dim: self.dim,
                num_vectors,
                max_level: self.max_level,
                entry_point: self.entry_point.unwrap_or(0),
                elem_size: 4,
            },
            vectors: bytemuck::cast_slice::<f32, u8>(&flat_vectors).to_vec(),
            levels: bytemuck::cast_slice::<u32, u8>(&flat_levels).to_vec(),
            offsets: bytemuck::cast_slice::<u64, u8>(&flat_offsets).to_vec(),
            adjacency: bytemuck::cast_slice::<u32, u8>(&flat_adjacency).to_vec(),
        }
    }

    pub fn save_to_disk(self, file_path: &str) -> Result<(), HyperbolicDiskError> {
        self.into_packed().write_to_file(file_path)
    }
}
#[derive(Debug)]
pub struct HyperbolicDiskHnsw<M> {
    num_vectors: usize,
    max_level: u32,
    entry_point: u32,

    levels_offset: u64,
    offsets_offset: u64,
    adjacency_offset: u64,

    mmap: Mmap,
    metric: M,
    _phantom: PhantomData<M>,

    vectors_cache: Vec<Vec<f32>>,
}

impl<M: MetricF32> HyperbolicDiskHnsw<M> {
    pub fn from_packed(packed: PackedHnswGraph, metric: M) -> Result<Self, HyperbolicDiskError> {
        let mut vectors_cache = Vec::with_capacity(packed.metadata.num_vectors);
        if packed.metadata.num_vectors > 0 {
            let expected_vector_bytes = packed.metadata.num_vectors * packed.metadata.dim * 4;
            if packed.vectors.len() != expected_vector_bytes {
                return Err(HyperbolicDiskError::IndexError(format!(
                    "invalid packed vectors length: expected {}, got {}",
                    expected_vector_bytes,
                    packed.vectors.len()
                )));
            }
            for i in 0..packed.metadata.num_vectors {
                let start = i * packed.metadata.dim * 4;
                let end = start + (packed.metadata.dim * 4);
                let bytes = &packed.vectors[start..end];
                let mut vec = Vec::with_capacity(packed.metadata.dim);
                for chunk in bytes.chunks_exact(4) {
                    let bytes_arr: [u8; 4] = chunk.try_into().unwrap();
                    vec.push(f32::from_le_bytes(bytes_arr));
                }
                vectors_cache.push(vec);
            }
        }

        let mut mmap_bytes = Vec::new();
        let vectors_offset = 0_u64;
        let levels_offset = vectors_offset + packed.vectors.len() as u64;
        let offsets_offset = levels_offset + packed.levels.len() as u64;
        let adjacency_offset = offsets_offset + packed.offsets.len() as u64;
        mmap_bytes.extend_from_slice(&packed.vectors);
        mmap_bytes.extend_from_slice(&packed.levels);
        mmap_bytes.extend_from_slice(&packed.offsets);
        mmap_bytes.extend_from_slice(&packed.adjacency);
        let mut writable = MmapMut::map_anon(mmap_bytes.len())?;
        writable[..mmap_bytes.len()].copy_from_slice(&mmap_bytes);
        let mmap = writable.make_read_only()?;

        Ok(Self {
            num_vectors: packed.metadata.num_vectors,
            max_level: packed.metadata.max_level,
            entry_point: packed.metadata.entry_point,
            levels_offset,
            offsets_offset,
            adjacency_offset,
            mmap,
            metric,
            _phantom: PhantomData,
            vectors_cache,
        })
    }

    pub fn open(path: &str, metric: M) -> Result<Self, HyperbolicDiskError> {
        let mut file = OpenOptions::new().read(true).write(false).open(path)?;

        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let md_len = u64::from_le_bytes(buf8);

        let mut md_bytes = vec![0u8; md_len as usize];
        file.read_exact(&mut md_bytes)?;
        let meta: DiskMetadata = bincode::deserialize(&md_bytes)?;

        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        let mut vectors_cache = Vec::with_capacity(meta.num_vectors);
        if meta.num_vectors > 0 {
            for i in 0..meta.num_vectors {
                let start = (meta.vectors_offset + (i as u64 * meta.dim as u64 * 4)) as usize;
                let end = start + (meta.dim * 4);
                let bytes = &mmap[start..end];
                let mut vec = Vec::with_capacity(meta.dim);
                for chunk in bytes.chunks_exact(4) {
                    let bytes_arr: [u8; 4] = chunk.try_into().unwrap();
                    vec.push(f32::from_le_bytes(bytes_arr));
                }
                vectors_cache.push(vec);
            }
        }

        Ok(Self {
            num_vectors: meta.num_vectors,
            max_level: meta.max_level,
            entry_point: meta.entry_point,
            levels_offset: meta.levels_offset,
            offsets_offset: meta.offsets_offset,
            adjacency_offset: meta.adjacency_offset,
            mmap,
            metric,
            _phantom: PhantomData,
            vectors_cache,
        })
    }

    #[inline]
    pub fn get_vector(&self, id: u32) -> &[f32] {
        &self.vectors_cache[id as usize]
    }

    #[inline]
    fn get_node_max_level(&self, id: u32) -> u32 {
        let start = (self.levels_offset + (id as u64 * 4)) as usize;
        let bytes = [
            self.mmap[start],
            self.mmap[start + 1],
            self.mmap[start + 2],
            self.mmap[start + 3],
        ];
        u32::from_le_bytes(bytes)
    }

    #[inline]
    fn get_node_base_offset(&self, id: u32) -> u64 {
        let start = (self.offsets_offset + (id as u64 * 8)) as usize;
        let bytes = [
            self.mmap[start],
            self.mmap[start + 1],
            self.mmap[start + 2],
            self.mmap[start + 3],
            self.mmap[start + 4],
            self.mmap[start + 5],
            self.mmap[start + 6],
            self.mmap[start + 7],
        ];
        u64::from_le_bytes(bytes)
    }

    fn get_neighbors(&self, id: u32, target_level: u32) -> Vec<u32> {
        let max_lvl = self.get_node_max_level(id);
        if target_level > max_lvl {
            return vec![];
        }

        let mut byte_offset = (self.adjacency_offset + self.get_node_base_offset(id)) as usize;

        for l in 0..=target_level {
            let len_bytes = [
                self.mmap[byte_offset],
                self.mmap[byte_offset + 1],
                self.mmap[byte_offset + 2],
                self.mmap[byte_offset + 3],
            ];
            let len = u32::from_le_bytes(len_bytes) as usize;
            byte_offset += 4;

            if l == target_level {
                let mut neighbors = Vec::with_capacity(len);
                for _ in 0..len {
                    let nb_bytes = [
                        self.mmap[byte_offset],
                        self.mmap[byte_offset + 1],
                        self.mmap[byte_offset + 2],
                        self.mmap[byte_offset + 3],
                    ];
                    neighbors.push(u32::from_le_bytes(nb_bytes));
                    byte_offset += 4;
                }
                return neighbors;
            }
            byte_offset += len * 4;
        }
        vec![]
    }

    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<Candidate> {
        if self.num_vectors == 0 {
            return vec![];
        }

        let mut q_proj = query.to_vec();
        self.metric.project_to_ball(&mut q_proj);

        let mut curr = self.entry_point;
        let mut curr_dist = self.metric.eval(&q_proj, self.get_vector(curr));

        for l in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for nb in self.get_neighbors(curr, l) {
                    let d = self.metric.eval(&q_proj, self.get_vector(nb));
                    if d < curr_dist {
                        curr_dist = d;
                        curr = nb;
                        changed = true;
                    }
                }
            }
        }

        let mut visited = HashSet::new();
        visited.insert(curr);

        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        candidates.push(std::cmp::Reverse(Candidate {
            id: curr,
            dist: curr_dist,
        }));
        results.push(Candidate {
            id: curr,
            dist: curr_dist,
        });

        while let Some(std::cmp::Reverse(cand)) = candidates.pop() {
            if cand.dist > results.peek().unwrap().dist && results.len() >= ef_search {
                break;
            }

            for nb in self.get_neighbors(cand.id, 0) {
                if visited.insert(nb) {
                    let d = self.metric.eval(&q_proj, self.get_vector(nb));

                    if results.len() < ef_search || d < results.peek().unwrap().dist {
                        candidates.push(std::cmp::Reverse(Candidate { id: nb, dist: d }));
                        results.push(Candidate { id: nb, dist: d });

                        if results.len() > ef_search {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut final_cands = results.into_vec();
        final_cands.sort_unstable();
        final_cands.truncate(k);
        final_cands
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_metric() {
        let metric = PoincareMetric { curvature: 1.0 };

        let origin = vec![0.0_f32; 2];
        let point = vec![0.5_f32, 0.0];

        let dist = metric.eval(&origin, &point);
        assert!(dist > 0.0);

        let mut large_vector = vec![2.0_f32; 2];
        metric.project_to_ball(&mut large_vector);
        let norm_sq: f32 = large_vector.iter().map(|v| v * v).sum();
        assert!(norm_sq < 1.0);
    }

    #[test]
    fn test_build_and_search() {
        let dim = 4;
        let metric = PoincareMetric { curvature: 1.0 };
        let params = HnswBuildParams {
            m: 8,
            m0: 16,
            ef_construction: 100,
            level_mult: 1.0 / (8.0_f32).ln(),
        };

        let mut builder = HyperbolicHnswBuilder::new(dim, metric, params);

        for i in 0..100 {
            let mut vec = vec![0.0_f32; dim];
            for j in 0..dim {
                vec[j] = (i * dim + j) as f32 * 0.01;
            }
            builder.insert(vec);
        }

        let test_path = "test_hyperbolic_hnsw.bin";
        builder.save_to_disk(test_path).unwrap();

        let metric = PoincareMetric { curvature: 1.0 };
        let index = HyperbolicDiskHnsw::open(test_path, metric).unwrap();

        let query = vec![0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32];
        let results = index.search(&query, 5, 50);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);

        for i in 1..results.len() {
            assert!(results[i].dist >= results[i - 1].dist);
        }

        std::fs::remove_file(test_path).unwrap();
    }

    #[test]
    fn test_empty_index() {
        let dim = 4;
        let metric = PoincareMetric { curvature: 1.0 };
        let params = HnswBuildParams::default();

        let builder = HyperbolicHnswBuilder::new(dim, metric, params);

        let test_path = "test_empty.bin";
        builder.save_to_disk(test_path).unwrap();

        let metric = PoincareMetric { curvature: 1.0 };
        let index = HyperbolicDiskHnsw::open(test_path, metric).unwrap();

        let query = vec![0.1_f32; 4];
        let results = index.search(&query, 5, 50);

        assert!(results.is_empty());

        std::fs::remove_file(test_path).unwrap();
    }

    #[test]
    fn test_single_vector() {
        let dim = 4;
        let metric = PoincareMetric { curvature: 1.0 };
        let params = HnswBuildParams::default();

        let mut builder = HyperbolicHnswBuilder::new(dim, metric, params);
        builder.insert(vec![0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32]);

        let test_path = "test_single.bin";
        builder.save_to_disk(test_path).unwrap();

        let metric = PoincareMetric { curvature: 1.0 };
        let index = HyperbolicDiskHnsw::open(test_path, metric).unwrap();

        let query = vec![0.1_f32, 0.2_f32, 0.3_f32, 0.4_f32];
        let results = index.search(&query, 1, 10);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);

        std::fs::remove_file(test_path).unwrap();
    }

    #[test]
    fn test_metric_symmetry() {
        let metric = PoincareMetric { curvature: 1.0 };
        let a = vec![0.1_f32, 0.2_f32];
        let b = vec![0.3_f32, 0.4_f32];

        let dist_ab = metric.eval(&a, &b);
        let dist_ba = metric.eval(&b, &a);

        assert!((dist_ab - dist_ba).abs() < 1e-5);
    }

    #[test]
    fn test_metric_triangle_inequality() {
        let metric = PoincareMetric { curvature: 1.0 };
        let a = vec![0.1_f32, 0.0_f32];
        let b = vec![0.2_f32, 0.0_f32];
        let c = vec![0.3_f32, 0.0_f32];

        let dist_ab = metric.eval(&a, &b);
        let dist_bc = metric.eval(&b, &c);
        let dist_ac = metric.eval(&a, &c);

        assert!(dist_ac <= dist_ab + dist_bc + 1e-5);
    }
}
