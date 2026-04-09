use std::time::{SystemTime, UNIX_EPOCH};

use crate::sphere::{SphereDistance, SphereMetric};
use crate::{AnnMetric, HnswBuildParams, HyperbolicDiskHnsw, HyperbolicHnswBuilder, MetricF32};

fn temp_index_path(label: &str) -> String {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir()
        .join(format!("phoenix-hyperbolic-{label}-{stamp}.bin"))
        .display()
        .to_string()
}

fn sphere_points() -> Vec<Vec<f32>> {
    let metric = SphereMetric {
        distance: SphereDistance::Geodesic,
    };
    let mut points = vec![
        vec![1.0f32, 0.0, 0.0, 0.0],
        vec![0.92, 0.28, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];
    for point in &mut points {
        metric.project_to_ball(point);
    }
    points
}

#[test]
fn sphere_hnsw_roundtrip_prefers_exact_neighbor() {
    let metric = SphereMetric {
        distance: SphereDistance::Geodesic,
    };
    let mut builder = HyperbolicHnswBuilder::new(4, metric, HnswBuildParams::default());
    let points = sphere_points();

    for point in &points {
        builder.insert(point.clone());
    }

    let path = temp_index_path("sphere-roundtrip");
    builder.save_to_disk(&path).expect("save sphere index");

    let index = HyperbolicDiskHnsw::open(&path, metric).expect("open sphere index");
    let hits = index.search(&points[0], 2, 16);

    assert_eq!(hits.first().map(|hit| hit.id), Some(0));
    let _ = std::fs::remove_file(path);
}

#[test]
fn ann_metric_sphere_runs_through_generic_hnsw() {
    let metric = AnnMetric::sphere_geodesic();
    let mut builder = HyperbolicHnswBuilder::new(4, metric, HnswBuildParams::default());
    let points = sphere_points();

    for point in &points {
        builder.insert(point.clone());
    }

    let path = temp_index_path("ann-metric-sphere");
    builder.save_to_disk(&path).expect("save ann metric index");

    let index = HyperbolicDiskHnsw::open(&path, metric).expect("open ann metric index");
    let hits = index.search(&points[1], 3, 16);

    assert_eq!(hits.first().map(|hit| hit.id), Some(1));
    assert_eq!(metric.label(), AnnMetric::LABEL_SPHERE_GEODESIC);
    let _ = std::fs::remove_file(path);
}
