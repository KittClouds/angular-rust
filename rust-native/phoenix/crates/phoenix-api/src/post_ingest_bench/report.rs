use std::collections::BTreeMap;

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PhaseStats {
    pub runs_us: Vec<u64>,
    pub min_us: u64,
    pub mean_us: f64,
    pub median_us: f64,
    pub p95_us: u64,
    pub max_us: u64,
}

#[derive(Debug, Clone, Default)]
pub struct PostIngestBenchRun {
    pub total_us: u64,
    pub dirty_list_us: u64,
    pub scope_load_us: u64,
    pub relation_stage_us: u64,
    pub state_schema_stage_us: u64,
    pub memory_stage_us: u64,
    pub counts: PostIngestCounts,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PostIngestCounts {
    pub dirty_scope_count: usize,
    pub relation_scope_count: usize,
    pub relation_case_count: usize,
    pub persisted_relation_edge_count: usize,
    pub state_schema_scope_count: usize,
    pub state_schema_slot_family_count: usize,
    pub state_schema_slot_definition_count: usize,
    pub state_schema_active_definition_count: usize,
    pub state_schema_candidate_count: usize,
    pub state_schema_write_proposal_count: usize,
    pub memory_scope_count: usize,
    pub memory_state_count: usize,
    pub memory_card_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FlowCaseReport {
    pub counts: PostIngestCounts,
    pub phases: BTreeMap<String, PhaseStats>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaseReport {
    pub case_id: String,
    pub title: String,
    pub text_bytes: usize,
    pub legacy: FlowCaseReport,
    pub shared: FlowCaseReport,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProjectionReport {
    pub phase: String,
    pub slice_case_id: String,
    pub full_case_id: String,
    pub slice_mean_us: f64,
    pub projected_full_linear_us: f64,
    pub actual_full_mean_us: f64,
    pub superlinear_factor: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DeltaReport {
    pub case_id: String,
    pub legacy_mean_us: f64,
    pub shared_mean_us: f64,
    pub saved_us: f64,
    pub speedup: f64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchReport {
    pub corpus: String,
    pub warmups: usize,
    pub iterations: usize,
    pub cases: Vec<CaseReport>,
    pub projections: Vec<ProjectionReport>,
    pub deltas: Vec<DeltaReport>,
}

pub fn flow_phase_stats(runs: &[PostIngestBenchRun]) -> BTreeMap<String, PhaseStats> {
    BTreeMap::from([
        phase_entry("total_us", runs.iter().map(|run| run.total_us).collect()),
        phase_entry(
            "dirty_list_us",
            runs.iter().map(|run| run.dirty_list_us).collect(),
        ),
        phase_entry(
            "scope_load_us",
            runs.iter().map(|run| run.scope_load_us).collect(),
        ),
        phase_entry(
            "relation_stage_us",
            runs.iter().map(|run| run.relation_stage_us).collect(),
        ),
        phase_entry(
            "state_schema_stage_us",
            runs.iter().map(|run| run.state_schema_stage_us).collect(),
        ),
        phase_entry(
            "memory_stage_us",
            runs.iter().map(|run| run.memory_stage_us).collect(),
        ),
    ])
}

pub fn build_projections(cases: &[CaseReport]) -> Vec<ProjectionReport> {
    let Some(full) = cases.iter().find(|case| case.case_id == "shortrun-full") else {
        return Vec::new();
    };
    let Some(slice) = cases
        .iter()
        .find(|case| case.case_id.starts_with("shortrun-chapter-"))
    else {
        return Vec::new();
    };
    let scale = full.text_bytes as f64 / slice.text_bytes.max(1) as f64;
    [
        (
            "legacy.total_us",
            &slice.legacy.phases,
            &full.legacy.phases,
            "total_us",
        ),
        (
            "shared.total_us",
            &slice.shared.phases,
            &full.shared.phases,
            "total_us",
        ),
        (
            "legacy.relation_stage_us",
            &slice.legacy.phases,
            &full.legacy.phases,
            "relation_stage_us",
        ),
        (
            "shared.relation_stage_us",
            &slice.shared.phases,
            &full.shared.phases,
            "relation_stage_us",
        ),
    ]
    .into_iter()
    .filter_map(|(phase, slice_map, full_map, key)| {
        let slice_mean = slice_map.get(key)?.mean_us;
        let full_mean = full_map.get(key)?.mean_us;
        let projected = slice_mean * scale;
        Some(ProjectionReport {
            phase: phase.to_owned(),
            slice_case_id: slice.case_id.clone(),
            full_case_id: full.case_id.clone(),
            slice_mean_us: slice_mean,
            projected_full_linear_us: projected,
            actual_full_mean_us: full_mean,
            superlinear_factor: if projected <= f64::EPSILON {
                0.0
            } else {
                full_mean / projected
            },
        })
    })
    .collect()
}

pub fn build_deltas(cases: &[CaseReport]) -> Vec<DeltaReport> {
    cases
        .iter()
        .filter_map(|case| {
            let legacy = case.legacy.phases.get("total_us")?.mean_us;
            let shared = case.shared.phases.get("total_us")?.mean_us;
            Some(DeltaReport {
                case_id: case.case_id.clone(),
                legacy_mean_us: legacy,
                shared_mean_us: shared,
                saved_us: legacy - shared,
                speedup: if shared <= f64::EPSILON {
                    0.0
                } else {
                    legacy / shared
                },
            })
        })
        .collect()
}

pub fn compute_phase_stats(mut runs_us: Vec<u64>) -> PhaseStats {
    runs_us.sort_unstable();
    let sum = runs_us.iter().copied().sum::<u64>() as f64;
    let len = runs_us.len().max(1);
    PhaseStats {
        min_us: *runs_us.first().unwrap_or(&0),
        mean_us: sum / len as f64,
        median_us: percentile(&runs_us, 0.5),
        p95_us: percentile(&runs_us, 0.95).round() as u64,
        max_us: *runs_us.last().unwrap_or(&0),
        runs_us,
    }
}

pub fn mean_ms(stats: Option<&PhaseStats>) -> f64 {
    stats
        .map(|value| value.mean_us / 1000.0)
        .unwrap_or_default()
}

fn phase_entry(name: &str, runs_us: Vec<u64>) -> (String, PhaseStats) {
    (name.to_owned(), compute_phase_stats(runs_us))
}

fn percentile(sorted: &[u64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[pos.min(sorted.len() - 1)] as f64
}
