use crate::sphere::{SphereDistance, SphereMetric};
use crate::{MetricF32, PoincareMetric};

#[derive(Clone, Copy, Debug)]
pub enum AnnMetric {
    Poincare(PoincareMetric),
    Sphere(SphereMetric),
}

impl Default for AnnMetric {
    fn default() -> Self {
        Self::sphere_geodesic()
    }
}

impl AnnMetric {
    pub const LABEL_POINCARE: &'static str = "hyperbolic:poincare";
    pub const LABEL_SPHERE_COSINE: &'static str = "sphere:cosine";
    pub const LABEL_SPHERE_CHORDAL: &'static str = "sphere:chordal";
    pub const LABEL_SPHERE_CHORDAL_SQUARED: &'static str = "sphere:chordal_squared";
    pub const LABEL_SPHERE_GEODESIC: &'static str = "sphere:geodesic";

    pub const fn poincare_default() -> Self {
        Self::Poincare(PoincareMetric { curvature: 1.0 })
    }

    pub const fn sphere(distance: SphereDistance) -> Self {
        Self::Sphere(SphereMetric { distance })
    }

    pub const fn sphere_geodesic() -> Self {
        Self::sphere(SphereDistance::Geodesic)
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Poincare(_) => Self::LABEL_POINCARE,
            Self::Sphere(SphereMetric {
                distance: SphereDistance::Cosine,
            }) => Self::LABEL_SPHERE_COSINE,
            Self::Sphere(SphereMetric {
                distance: SphereDistance::Chordal,
            }) => Self::LABEL_SPHERE_CHORDAL,
            Self::Sphere(SphereMetric {
                distance: SphereDistance::ChordalSquared,
            }) => Self::LABEL_SPHERE_CHORDAL_SQUARED,
            Self::Sphere(SphereMetric {
                distance: SphereDistance::Geodesic,
            }) => Self::LABEL_SPHERE_GEODESIC,
        }
    }

    pub fn from_label(label: &str) -> Option<Self> {
        match label {
            Self::LABEL_POINCARE | "poincare" => Some(Self::poincare_default()),
            Self::LABEL_SPHERE_COSINE => Some(Self::sphere(SphereDistance::Cosine)),
            Self::LABEL_SPHERE_CHORDAL => Some(Self::sphere(SphereDistance::Chordal)),
            Self::LABEL_SPHERE_CHORDAL_SQUARED => {
                Some(Self::sphere(SphereDistance::ChordalSquared))
            }
            Self::LABEL_SPHERE_GEODESIC | "hypersphere" | "sphere" => Some(Self::sphere_geodesic()),
            _ => None,
        }
    }

    pub fn from_label_or_default(label: &str) -> Self {
        Self::from_label(label).unwrap_or_default()
    }
}

impl MetricF32 for AnnMetric {
    #[inline]
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Poincare(metric) => metric.eval(a, b),
            Self::Sphere(metric) => metric.eval(a, b),
        }
    }

    #[inline]
    fn rank_eval(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Poincare(metric) => metric.rank_eval(a, b),
            Self::Sphere(metric) => metric.rank_eval(a, b),
        }
    }

    #[inline]
    fn project_to_ball(&self, vector: &mut [f32]) {
        match self {
            Self::Poincare(metric) => metric.project_to_ball(vector),
            Self::Sphere(metric) => metric.project_to_ball(vector),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_metric_labels_with_sphere_as_default() {
        assert_eq!(
            AnnMetric::from_label_or_default("sphere:geodesic").label(),
            AnnMetric::LABEL_SPHERE_GEODESIC
        );
        assert_eq!(
            AnnMetric::from_label_or_default("hyperbolic:poincare").label(),
            AnnMetric::LABEL_POINCARE
        );
        assert_eq!(
            AnnMetric::from_label_or_default("unknown").label(),
            AnnMetric::LABEL_SPHERE_GEODESIC
        );
    }
}
