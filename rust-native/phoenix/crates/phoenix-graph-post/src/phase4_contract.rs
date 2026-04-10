use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphPhase4RerankScore {
    pub model: String,
    pub positive_label: String,
    pub positive_score: f64,
    pub context_label: String,
    pub context_score: f64,
    pub negative_label: String,
    pub negative_score: f64,
    pub applied_delta: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphPathRerankScore {
    pub deterministic_rank: usize,
    pub deterministic_score: f64,
    pub learned: GraphPhase4RerankScore,
    pub applied_delta: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GraphStructuralRerankScore {
    pub model: String,
    pub anchor_component: bool,
    pub proximity_score_millis: u32,
    pub component_size: usize,
    pub applied_delta_millis: i32,
}
