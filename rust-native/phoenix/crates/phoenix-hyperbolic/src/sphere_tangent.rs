//! Tangent space operations for the unit hypersphere.
//!
//! Uses a spherical centroid and sphere log/exp maps.
//! Fast phase: Euclidean distance in tangent space.
//! Exact phase: geodesic distance on the sphere.

use serde::{Deserialize, Serialize};

const EPS: f32 = 1e-6;

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x * y)
        .sum::<f32>()
        .clamp(-1.0, 1.0)
}

#[inline]
fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[inline]
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let n = norm(&v);
    if n <= EPS {
        if !v.is_empty() {
            v.fill(0.0);
            v[0] = 1.0;
        }
        return v;
    }
    let inv = n.recip();
    for x in &mut v {
        *x *= inv;
    }
    v
}

#[inline]
pub fn geodesic_distance(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b).acos()
}

#[inline]
pub fn sphere_log_map(x: &[f32], c: &[f32]) -> Vec<f32> {
    let dc = dot(c, x);
    let theta = dc.acos();

    if theta <= EPS {
        return vec![0.0; c.len()];
    }

    let sin_theta = theta.sin().max(EPS);
    let scale = theta / sin_theta;

    x.iter()
        .zip(c.iter())
        .map(|(&xi, &ci)| scale * (xi - dc * ci))
        .collect()
}

#[inline]
pub fn sphere_exp_map(u: &[f32], c: &[f32]) -> Vec<f32> {
    let u_norm = norm(u);

    if u_norm <= EPS {
        return c.to_vec();
    }

    let cos_un = u_norm.cos();
    let sin_un = u_norm.sin() / u_norm;

    let out: Vec<f32> = c
        .iter()
        .zip(u.iter())
        .map(|(&ci, &ui)| cos_un * ci + sin_un * ui)
        .collect();

    normalize(out)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphereTangentCache {
    pub centroid: Vec<f32>,
    pub tangent_coords: Vec<Vec<f32>>,
    pub point_indices: Vec<usize>,
}

impl SphereTangentCache {
    pub fn new(points: &[Vec<f32>], indices: &[usize]) -> Result<Self, String> {
        if points.is_empty() {
            return Err("Empty collection".to_string());
        }
        if points.len() != indices.len() {
            return Err("points/indices length mismatch".to_string());
        }

        let dim = points[0].len();
        let mut mean = vec![0.0f32; dim];

        for p in points {
            if p.len() != dim {
                return Err("dimension mismatch".to_string());
            }
            for (m, &x) in mean.iter_mut().zip(p.iter()) {
                *m += x;
            }
        }

        let centroid = normalize(mean);
        let tangent_coords = points
            .iter()
            .map(|p| sphere_log_map(p, &centroid))
            .collect();

        Ok(Self {
            centroid,
            tangent_coords,
            point_indices: indices.to_vec(),
        })
    }

    pub fn from_centroid(
        centroid: Vec<f32>,
        points: &[Vec<f32>],
        indices: &[usize],
    ) -> Result<Self, String> {
        if points.len() != indices.len() {
            return Err("points/indices length mismatch".to_string());
        }

        let centroid = normalize(centroid);
        let tangent_coords = points
            .iter()
            .map(|p| sphere_log_map(p, &centroid))
            .collect();

        Ok(Self {
            centroid,
            tangent_coords,
            point_indices: indices.to_vec(),
        })
    }

    #[inline]
    pub fn query_tangent(&self, query: &[f32]) -> Vec<f32> {
        sphere_log_map(query, &self.centroid)
    }

    #[inline]
    pub fn tangent_distance_squared(&self, query_tangent: &[f32], idx: usize) -> f32 {
        if idx >= self.tangent_coords.len() {
            return f32::MAX;
        }

        query_tangent
            .iter()
            .zip(self.tangent_coords[idx].iter())
            .map(|(&a, &b)| {
                let d = a - b;
                d * d
            })
            .sum()
    }

    #[inline]
    pub fn exact_distance(&self, query: &[f32], idx: usize, points: &[Vec<f32>]) -> f32 {
        if idx >= points.len() {
            return f32::MAX;
        }
        geodesic_distance(query, &points[idx])
    }

    pub fn add_point(&mut self, point: &[f32], index: usize) {
        let tangent = sphere_log_map(point, &self.centroid);
        self.tangent_coords.push(tangent);
        self.point_indices.push(index);
    }

    pub fn recompute_centroid(&mut self, points: &[Vec<f32>]) -> Result<(), String> {
        if points.is_empty() {
            return Err("Empty collection".to_string());
        }

        let dim = points[0].len();
        let mut mean = vec![0.0f32; dim];

        for p in points {
            if p.len() != dim {
                return Err("dimension mismatch".to_string());
            }
            for (m, &x) in mean.iter_mut().zip(p.iter()) {
                *m += x;
            }
        }

        self.centroid = normalize(mean);
        self.tangent_coords = points
            .iter()
            .map(|p| sphere_log_map(p, &self.centroid))
            .collect();

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.tangent_coords.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tangent_coords.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.centroid.len()
    }
}

#[derive(Debug, Clone)]
pub struct PrunedCandidate {
    pub index: usize,
    pub tangent_dist: f32,
    pub exact_dist: Option<f32>,
}

pub struct SphereTangentPruner {
    caches: Vec<SphereTangentCache>,
    top_n: usize,
    prune_factor: usize,
}

impl SphereTangentPruner {
    pub fn new(top_n: usize, prune_factor: usize) -> Self {
        Self {
            caches: Vec::new(),
            top_n,
            prune_factor,
        }
    }

    pub fn add_cache(&mut self, cache: SphereTangentCache) {
        self.caches.push(cache);
    }

    pub fn caches(&self) -> &[SphereTangentCache] {
        &self.caches
    }

    pub fn caches_mut(&mut self) -> &mut [SphereTangentCache] {
        &mut self.caches
    }

    pub fn search(&self, query: &[f32], points: &[Vec<f32>]) -> Vec<PrunedCandidate> {
        let num_prune = self.top_n * self.prune_factor;
        let mut candidates: Vec<PrunedCandidate> = Vec::new();

        for cache in &self.caches {
            let query_tangent = cache.query_tangent(query);

            for (local_idx, &global_idx) in cache.point_indices.iter().enumerate() {
                let tangent_dist = cache.tangent_distance_squared(&query_tangent, local_idx);
                candidates.push(PrunedCandidate {
                    index: global_idx,
                    tangent_dist,
                    exact_dist: None,
                });
            }
        }

        candidates.sort_by(|a, b| a.tangent_dist.total_cmp(&b.tangent_dist));
        candidates.truncate(num_prune);

        for candidate in &mut candidates {
            if candidate.index < points.len() {
                candidate.exact_dist = Some(geodesic_distance(query, &points[candidate.index]));
            }
        }

        candidates.sort_by(|a, b| {
            a.exact_dist
                .unwrap_or(f32::MAX)
                .total_cmp(&b.exact_dist.unwrap_or(f32::MAX))
        });
        candidates.truncate(self.top_n);

        candidates
    }
}

pub fn tangent_micro_update(
    point: &[f32],
    delta: &[f32],
    centroid: &[f32],
    max_step: f32,
) -> Vec<f32> {
    let tangent = sphere_log_map(point, centroid);
    let delta_norm = norm(delta);
    let scale = if delta_norm > max_step && delta_norm > EPS {
        max_step / delta_norm
    } else {
        1.0
    };

    let new_tangent: Vec<f32> = tangent
        .iter()
        .zip(delta.iter())
        .map(|(&t, &d)| t + scale * d)
        .collect();

    sphere_exp_map(&new_tangent, centroid)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tangent_cache_creation() {
        let points = vec![
            normalize(vec![0.9, 0.1, 0.0]),
            normalize(vec![0.8, 0.2, 0.0]),
            normalize(vec![0.0, 1.0, 0.0]),
        ];
        let indices: Vec<usize> = (0..points.len()).collect();

        let cache = SphereTangentCache::new(&points, &indices).unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.dim(), 3);
    }

    #[test]
    fn tangent_pruning_returns_sorted_results() {
        let points = vec![
            normalize(vec![1.0, 0.0]),
            normalize(vec![0.9, 0.1]),
            normalize(vec![0.0, 1.0]),
            normalize(vec![-1.0, 0.0]),
        ];

        let indices: Vec<usize> = (0..points.len()).collect();
        let cache = SphereTangentCache::new(&points, &indices).unwrap();

        let mut pruner = SphereTangentPruner::new(2, 2);
        pruner.add_cache(cache);

        let query = normalize(vec![1.0, 0.0]);
        let results = pruner.search(&query, &points);

        assert_eq!(results.len(), 2);
        assert!(results[0].exact_dist.unwrap() <= results[1].exact_dist.unwrap());
    }
}
