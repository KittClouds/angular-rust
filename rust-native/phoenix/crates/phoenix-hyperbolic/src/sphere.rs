//! Hypersphere metric for HNSW / DiskANN-style index.
//!
//! Points are normalized onto the unit sphere S^(d-1).
//! Internal traversal uses a cheap monotone score: 1 - dot(a, b).
//! Exact/public distance can be cosine, chordal, chordal-squared, or geodesic.

use serde::{Deserialize, Serialize};

use crate::MetricF32;

const EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum SphereDistance {
    Cosine,
    Chordal,
    ChordalSquared,
    Geodesic,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct SphereMetric {
    pub distance: SphereDistance,
}

impl Default for SphereMetric {
    fn default() -> Self {
        Self {
            distance: SphereDistance::Geodesic,
        }
    }
}

impl SphereMetric {
    #[inline]
    fn dot(a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let chunks = len / 4;
        let remainder = len % 4;
        let mut i = 0usize;
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        for _ in 0..chunks {
            acc0 = a[i].mul_add(b[i], acc0);
            acc1 = a[i + 1].mul_add(b[i + 1], acc1);
            acc2 = a[i + 2].mul_add(b[i + 2], acc2);
            acc3 = a[i + 3].mul_add(b[i + 3], acc3);
            i += 4;
        }

        let mut sum = (acc0 + acc1) + (acc2 + acc3);
        for offset in 0..remainder {
            sum = a[i + offset].mul_add(b[i + offset], sum);
        }
        sum.clamp(-1.0, 1.0)
    }

    #[inline]
    fn chordal_sq_from_dot(dot: f32) -> f32 {
        (2.0 - 2.0 * dot).max(0.0)
    }

    #[inline]
    fn normalize_in_place(v: &mut [f32]) {
        let mut norm_sq = 0.0f32;
        let mut all_finite = true;
        let chunks = v.len() / 4;
        let remainder = v.len() % 4;
        let mut i = 0usize;

        for _ in 0..chunks {
            let x0 = v[i];
            let x1 = v[i + 1];
            let x2 = v[i + 2];
            let x3 = v[i + 3];
            all_finite &= x0.is_finite() && x1.is_finite() && x2.is_finite() && x3.is_finite();
            norm_sq = x0.mul_add(x0, norm_sq);
            norm_sq = x1.mul_add(x1, norm_sq);
            norm_sq = x2.mul_add(x2, norm_sq);
            norm_sq = x3.mul_add(x3, norm_sq);
            i += 4;
        }

        for offset in 0..remainder {
            let x = v[i + offset];
            all_finite &= x.is_finite();
            norm_sq = x.mul_add(x, norm_sq);
        }

        if !all_finite || norm_sq <= EPS {
            // Deterministic fallback for degenerate vectors:
            // map to north pole instead of panicking.
            if !v.is_empty() {
                v.fill(0.0);
                v[0] = 1.0;
            }
            return;
        }

        if (norm_sq - 1.0).abs() <= 1e-6 {
            return;
        }

        let inv = norm_sq.sqrt().recip();
        let chunks = v.len() / 4;
        let remainder = v.len() % 4;
        let mut i = 0usize;
        for _ in 0..chunks {
            v[i] *= inv;
            v[i + 1] *= inv;
            v[i + 2] *= inv;
            v[i + 3] *= inv;
            i += 4;
        }
        for offset in 0..remainder {
            v[i + offset] *= inv;
        }
    }
}

impl MetricF32 for SphereMetric {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot = Self::dot(a, b);
        match self.distance {
            SphereDistance::Cosine => 1.0 - dot,
            SphereDistance::ChordalSquared => Self::chordal_sq_from_dot(dot),
            SphereDistance::Chordal => Self::chordal_sq_from_dot(dot).sqrt(),
            SphereDistance::Geodesic => dot.acos(),
        }
    }

    #[inline]
    fn rank_eval(&self, a: &[f32], b: &[f32]) -> f32 {
        // Monotone with all of the exact distances above on the unit sphere.
        1.0 - Self::dot(a, b)
    }

    #[inline]
    fn project_to_ball(&self, vector: &mut [f32]) {
        Self::normalize_in_place(vector);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetricF32;

    #[test]
    fn normalizes_to_unit_sphere() {
        let metric = SphereMetric::default();
        let mut v = vec![3.0f32, 4.0];
        metric.project_to_ball(&mut v);

        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn geodesic_order_matches_rank_eval() {
        let metric = SphereMetric {
            distance: SphereDistance::Geodesic,
        };

        let mut q = vec![1.0f32, 0.0, 0.0];
        let mut a = vec![0.9f32, 0.1, 0.0];
        let mut b = vec![0.0f32, 1.0, 0.0];

        metric.project_to_ball(&mut q);
        metric.project_to_ball(&mut a);
        metric.project_to_ball(&mut b);

        assert!(metric.rank_eval(&q, &a) < metric.rank_eval(&q, &b));
        assert!(metric.eval(&q, &a) < metric.eval(&q, &b));
    }

    #[test]
    fn opposite_points_are_far() {
        let metric = SphereMetric {
            distance: SphereDistance::Geodesic,
        };

        let mut a = vec![1.0f32, 0.0];
        let mut b = vec![-1.0f32, 0.0];
        metric.project_to_ball(&mut a);
        metric.project_to_ball(&mut b);

        let d = metric.eval(&a, &b);
        assert!((d - std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn degenerate_vectors_project_to_north_pole() {
        let metric = SphereMetric::default();
        let mut zero = vec![0.0f32, 0.0, 0.0];
        metric.project_to_ball(&mut zero);
        assert_eq!(zero, vec![1.0, 0.0, 0.0]);

        let mut nan = vec![f32::NAN, 2.0, 3.0];
        metric.project_to_ball(&mut nan);
        assert_eq!(nan, vec![1.0, 0.0, 0.0]);
    }

    #[test]
    fn distance_variants_preserve_neighbor_ordering() {
        let mut q = vec![1.0f32, 0.0, 0.0];
        let mut near = vec![0.95f32, 0.2, 0.0];
        let mut far = vec![0.0f32, 1.0, 0.0];
        let metric = SphereMetric::default();
        metric.project_to_ball(&mut q);
        metric.project_to_ball(&mut near);
        metric.project_to_ball(&mut far);

        for distance in [
            SphereDistance::Cosine,
            SphereDistance::Chordal,
            SphereDistance::ChordalSquared,
            SphereDistance::Geodesic,
        ] {
            let metric = SphereMetric { distance };
            assert!(metric.rank_eval(&q, &near) < metric.rank_eval(&q, &far));
            assert!(metric.eval(&q, &near) < metric.eval(&q, &far));
        }
    }
}
