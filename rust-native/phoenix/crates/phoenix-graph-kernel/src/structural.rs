use crate::{KernelEdge, KernelGraphSnapshot};
use rustc_hash::{FxHashMap, FxHashSet};
use scirs2_graph::{
    csr_connected_components, personalized_pagerank, CsrGraph, Graph as ScirsGraph,
};

const STRUCTURAL_DAMPING: f64 = 0.85;
const STRUCTURAL_TOLERANCE: f64 = 1e-6;
const STRUCTURAL_ITERATIONS: usize = 24;
const HEAT_KERNEL_TIME: f64 = 2.0;
const HEAT_KERNEL_TERMS: usize = 6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KernelLocalDiffusionKind {
    #[default]
    PersonalizedPagerank,
    HeatKernel,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KernelStructuralProfile {
    #[default]
    WorldState,
    History,
    Causal,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelStructuralScore {
    pub anchor_component: bool,
    pub proximity_score_millis: u32,
    pub component_size: usize,
    pub applied_delta_millis: i32,
    pub diffusion_kind: KernelLocalDiffusionKind,
}

#[derive(Clone, Debug, Default)]
pub struct KernelStructuralAnalytics {
    dense: FxHashMap<String, usize>,
    component_ids: Vec<usize>,
    component_sizes: Vec<usize>,
    proximity: Vec<f64>,
    anchor_components: FxHashSet<usize>,
    anchor_indices: FxHashSet<usize>,
    active: bool,
    diffusion_kind: KernelLocalDiffusionKind,
}

impl KernelStructuralAnalytics {
    pub fn from_snapshot(snapshot: &KernelGraphSnapshot, anchor_vertex_ids: &[String]) -> Self {
        Self::from_snapshot_with_profile(
            snapshot,
            anchor_vertex_ids,
            KernelStructuralProfile::WorldState,
            KernelLocalDiffusionKind::PersonalizedPagerank,
        )
    }

    pub fn from_snapshot_with_profile(
        snapshot: &KernelGraphSnapshot,
        anchor_vertex_ids: &[String],
        profile: KernelStructuralProfile,
        diffusion_kind: KernelLocalDiffusionKind,
    ) -> Self {
        let vertex_ids = snapshot
            .vertices
            .iter()
            .map(|vertex| vertex.id.0.clone())
            .collect::<Vec<_>>();
        if vertex_ids.is_empty() {
            return Self::default();
        }
        let dense = vertex_ids
            .iter()
            .enumerate()
            .map(|(index, id)| (id.clone(), index))
            .collect::<FxHashMap<_, _>>();
        let anchor_indices = anchor_vertex_ids
            .iter()
            .filter_map(|id| dense.get(id).copied())
            .collect::<Vec<_>>();
        if anchor_indices.is_empty() {
            return Self {
                dense,
                component_ids: vec![0; vertex_ids.len()],
                component_sizes: vec![0; vertex_ids.len()],
                proximity: vec![0.0; vertex_ids.len()],
                anchor_components: FxHashSet::default(),
                anchor_indices: FxHashSet::default(),
                active: false,
                diffusion_kind,
            };
        }

        let edge_list = collect_topology_edges(snapshot, &dense, profile);
        let components = build_component_summary(vertex_ids.len(), edge_list.as_slice());
        let anchor_components = anchor_indices
            .iter()
            .filter_map(|index| components.labels.get(*index).copied())
            .collect::<FxHashSet<_>>();
        let anchor_indices = anchor_indices.into_iter().collect::<FxHashSet<_>>();
        let proximity = if edge_list.is_empty() {
            seed_proximity(vertex_ids.len(), &anchor_indices)
        } else {
            match diffusion_kind {
                KernelLocalDiffusionKind::PersonalizedPagerank => average_personalized_pagerank(
                    vertex_ids.len(),
                    edge_list.as_slice(),
                    &anchor_indices,
                ),
                KernelLocalDiffusionKind::HeatKernel => average_heat_kernel_diffusion(
                    vertex_ids.len(),
                    edge_list.as_slice(),
                    &anchor_indices,
                ),
            }
        };

        Self {
            dense,
            component_ids: components.labels,
            component_sizes: components.component_sizes,
            proximity,
            anchor_components,
            anchor_indices,
            active: true,
            diffusion_kind,
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn score(&self, vertex_ids: &[String]) -> Option<KernelStructuralScore> {
        self.score_with_filter(vertex_ids, false)
    }

    pub fn score_non_anchor(&self, vertex_ids: &[String]) -> Option<KernelStructuralScore> {
        self.score_with_filter(vertex_ids, true)
    }

    fn score_with_filter(
        &self,
        vertex_ids: &[String],
        exclude_anchors: bool,
    ) -> Option<KernelStructuralScore> {
        let mut best_index = None;
        let mut best_score = 0.0f64;
        for vertex_id in vertex_ids {
            let Some(index) = self.dense.get(vertex_id).copied() else {
                continue;
            };
            if exclude_anchors && self.anchor_indices.contains(&index) {
                continue;
            }
            let score = self.proximity.get(index).copied().unwrap_or_default();
            if best_index.is_none() || score > best_score {
                best_index = Some(index);
                best_score = score;
            }
        }

        let index = best_index?;
        let component_id = self.component_ids.get(index).copied().unwrap_or_default();
        let component_size = self.component_sizes.get(index).copied().unwrap_or_default();
        let anchor_component = self.anchor_components.contains(&component_id);
        let proximity_score_millis = proximity_score_millis(best_score);
        let component_bonus = ((component_size.min(24) as i32) * 2).min(48);
        let applied_delta_millis = if anchor_component {
            (((proximity_score_millis as i32) * 22) / 100 + component_bonus).min(240)
        } else {
            -80
        };
        Some(KernelStructuralScore {
            anchor_component,
            proximity_score_millis,
            component_size,
            applied_delta_millis,
            diffusion_kind: self.diffusion_kind,
        })
    }
}

struct ComponentSummary {
    labels: Vec<usize>,
    component_sizes: Vec<usize>,
}

fn collect_topology_edges(
    snapshot: &KernelGraphSnapshot,
    dense: &FxHashMap<String, usize>,
    profile: KernelStructuralProfile,
) -> Vec<(usize, usize, f64)> {
    let mut seen = FxHashSet::default();
    let mut edges = Vec::new();
    for edge in snapshot
        .asserted_edges
        .iter()
        .chain(snapshot.candidate_edges.iter())
    {
        let Some(&source) = dense.get(edge.source_id.0.as_str()) else {
            continue;
        };
        let Some(&target) = dense.get(edge.target_id.0.as_str()) else {
            continue;
        };
        if source == target {
            continue;
        }
        let key = undirected_key(source, target);
        if !seen.insert(key) {
            continue;
        }
        let weight = structural_weight(edge, profile);
        edges.push((source, target, weight));
    }
    edges
}

fn build_component_summary(num_nodes: usize, edges: &[(usize, usize, f64)]) -> ComponentSummary {
    let graph = CsrGraph::from_edges_parallel(num_nodes, edges.to_vec(), false)
        .or_else(|_| CsrGraph::from_edges(num_nodes, edges.to_vec(), false))
        .expect("structural csr graph should build");
    let components = csr_connected_components(&graph);
    ComponentSummary {
        labels: components.labels,
        component_sizes: components.component_sizes,
    }
}

fn average_personalized_pagerank(
    num_nodes: usize,
    edges: &[(usize, usize, f64)],
    anchor_indices: &FxHashSet<usize>,
) -> Vec<f64> {
    let graph = build_personalization_graph(num_nodes, edges);
    let mut proximity = vec![0.0; num_nodes];
    let mut successes = 0usize;
    let mut anchor_list = anchor_indices.iter().copied().collect::<Vec<_>>();
    anchor_list.sort_unstable();
    for anchor in anchor_list {
        let Ok(scores) = personalized_pagerank(
            &graph,
            &anchor,
            STRUCTURAL_DAMPING,
            STRUCTURAL_TOLERANCE,
            STRUCTURAL_ITERATIONS,
        ) else {
            continue;
        };
        for (node, score) in scores {
            if let Some(slot) = proximity.get_mut(node) {
                *slot += score;
            }
        }
        successes += 1;
    }
    if successes == 0 {
        return seed_proximity(num_nodes, anchor_indices);
    }
    for value in &mut proximity {
        *value /= successes as f64;
    }
    proximity
}

fn average_heat_kernel_diffusion(
    num_nodes: usize,
    edges: &[(usize, usize, f64)],
    anchor_indices: &FxHashSet<usize>,
) -> Vec<f64> {
    if num_nodes == 0 || anchor_indices.is_empty() {
        return vec![0.0; num_nodes];
    }
    let adjacency = weighted_adjacency(num_nodes, edges);
    let mut proximity = vec![0.0; num_nodes];
    let mut anchor_list = anchor_indices.iter().copied().collect::<Vec<_>>();
    anchor_list.sort_unstable();
    for anchor in anchor_list {
        let scores = heat_kernel_diffusion_from_seed(adjacency.as_slice(), anchor);
        for (slot, score) in proximity.iter_mut().zip(scores.into_iter()) {
            *slot += score;
        }
    }
    for value in &mut proximity {
        *value /= anchor_indices.len() as f64;
    }
    proximity
}

fn build_personalization_graph(
    num_nodes: usize,
    edges: &[(usize, usize, f64)],
) -> ScirsGraph<usize, f64> {
    let mut graph = ScirsGraph::<usize, f64>::new();
    for node in 0..num_nodes {
        graph.add_node(node);
    }
    for &(source, target, weight) in edges {
        let _ = graph.add_edge(source, target, weight);
    }
    graph
}

fn weighted_adjacency(num_nodes: usize, edges: &[(usize, usize, f64)]) -> Vec<Vec<(usize, f64)>> {
    let mut adjacency = vec![Vec::<(usize, f64)>::new(); num_nodes];
    for &(source, target, weight) in edges {
        adjacency[source].push((target, weight));
        adjacency[target].push((source, weight));
    }
    adjacency
}

fn heat_kernel_diffusion_from_seed(adjacency: &[Vec<(usize, f64)>], seed: usize) -> Vec<f64> {
    let mut result = vec![0.0; adjacency.len()];
    let mut frontier = vec![0.0; adjacency.len()];
    if let Some(slot) = frontier.get_mut(seed) {
        *slot = 1.0;
    } else {
        return result;
    }

    let exp_factor = (-HEAT_KERNEL_TIME).exp();
    let mut coefficient = exp_factor;
    add_scaled(&mut result, &frontier, coefficient);
    for term in 1..=HEAT_KERNEL_TERMS {
        frontier = transition_step(adjacency, &frontier);
        coefficient *= HEAT_KERNEL_TIME / term as f64;
        add_scaled(&mut result, &frontier, coefficient);
    }
    normalize_scores(&mut result);
    result
}

fn transition_step(adjacency: &[Vec<(usize, f64)>], current: &[f64]) -> Vec<f64> {
    let mut next = vec![0.0; adjacency.len()];
    for (source, edges) in adjacency.iter().enumerate() {
        let value = current.get(source).copied().unwrap_or_default();
        if value <= 0.0 || edges.is_empty() {
            continue;
        }
        let degree = edges
            .iter()
            .map(|(_, weight)| *weight)
            .sum::<f64>()
            .max(1e-12);
        for &(target, weight) in edges {
            next[target] += value * (weight / degree);
        }
    }
    next
}

fn add_scaled(target: &mut [f64], source: &[f64], scale: f64) {
    for (left, right) in target.iter_mut().zip(source.iter()) {
        *left += *right * scale;
    }
}

fn normalize_scores(scores: &mut [f64]) {
    let total = scores.iter().sum::<f64>();
    if total > 1e-12 {
        for value in scores {
            *value /= total;
        }
    }
}

fn seed_proximity(num_nodes: usize, anchor_indices: &FxHashSet<usize>) -> Vec<f64> {
    if anchor_indices.is_empty() || num_nodes == 0 {
        return vec![0.0; num_nodes];
    }
    let share = 1.0 / anchor_indices.len() as f64;
    let mut proximity = vec![0.0; num_nodes];
    for &anchor in anchor_indices {
        if let Some(slot) = proximity.get_mut(anchor) {
            *slot = share;
        }
    }
    proximity
}

fn structural_weight(edge: &KernelEdge, profile: KernelStructuralProfile) -> f64 {
    let base = edge.weight.max(1) as f64;
    let layer_weight = match edge.layer {
        crate::KernelGraphLayer::Asserted => 1.0,
        crate::KernelGraphLayer::Candidate => 0.85,
    };
    let profile_weight = match profile {
        KernelStructuralProfile::WorldState => match edge.edge_type.0.as_str() {
            "state_of" | "state_value" => 1.25,
            "supported_by" | "about" => 1.0,
            "semantic::same_slot_family" => 1.1,
            "semantic::contradictory_support_region" => 0.55,
            "semantic::same_process" | "semantic::related_event" => 0.72,
            "semantic::missing_intermediate_cause" => 0.45,
            _ => 0.8,
        },
        KernelStructuralProfile::History => match edge.edge_type.0.as_str() {
            "state_of" | "state_value" | "supported_by" => 1.0,
            "semantic::same_process" | "semantic::related_event" => 1.12,
            "semantic::same_slot_family" => 0.92,
            "semantic::missing_intermediate_cause" => 0.78,
            "semantic::contradictory_support_region" => 0.65,
            _ => 0.82,
        },
        KernelStructuralProfile::Causal => match edge.edge_type.0.as_str() {
            "causal_link" => 1.35,
            "semantic::same_process" => 1.15,
            "semantic::related_event" => 0.95,
            "semantic::missing_intermediate_cause" => 0.82,
            "supported_by" | "subject" | "object" => 0.72,
            "under_view" => 0.4,
            _ => 0.7,
        },
    };
    base * layer_weight * profile_weight
}

fn proximity_score_millis(value: f64) -> u32 {
    ((value.max(0.0).sqrt()) * 1000.0)
        .round()
        .clamp(0.0, 1000.0) as u32
}

fn undirected_key(left: usize, right: usize) -> u64 {
    let (small, large) = if left <= right {
        (left as u64, right as u64)
    } else {
        (right as u64, left as u64)
    };
    (small << 32) | large
}

#[cfg(test)]
mod tests {
    use super::{KernelLocalDiffusionKind, KernelStructuralAnalytics, KernelStructuralProfile};
    use crate::{
        KernelEdge, KernelEdgeType, KernelGraphLayer, KernelGraphSnapshot, KernelVertex,
        KernelVertexId,
    };

    #[test]
    fn structural_analytics_promote_anchor_component_nodes() {
        let snapshot = KernelGraphSnapshot {
            vertices: vec![vertex("anchor"), vertex("near"), vertex("far")],
            asserted_edges: vec![edge("anchor", "near", KernelGraphLayer::Asserted)],
            candidate_edges: vec![edge("near", "far", KernelGraphLayer::Candidate)],
        };

        let analytics = KernelStructuralAnalytics::from_snapshot(&snapshot, &["anchor".to_owned()]);

        let near = analytics.score(&["near".to_owned()]).expect("near score");
        let far = analytics.score(&["far".to_owned()]).expect("far score");
        assert!(analytics.is_active());
        assert!(near.anchor_component);
        assert!(near.proximity_score_millis >= far.proximity_score_millis);
        assert_eq!(
            near.diffusion_kind,
            KernelLocalDiffusionKind::PersonalizedPagerank
        );
    }

    #[test]
    fn heat_kernel_prioritizes_closer_causal_node() {
        let snapshot = KernelGraphSnapshot {
            vertices: vec![
                vertex("anchor"),
                vertex("near"),
                vertex("mid"),
                vertex("far"),
            ],
            asserted_edges: vec![
                edge("anchor", "near", KernelGraphLayer::Asserted),
                edge("near", "mid", KernelGraphLayer::Asserted),
                edge("mid", "far", KernelGraphLayer::Asserted),
            ],
            candidate_edges: Vec::new(),
        };

        let analytics = KernelStructuralAnalytics::from_snapshot_with_profile(
            &snapshot,
            &["anchor".to_owned()],
            KernelStructuralProfile::Causal,
            KernelLocalDiffusionKind::HeatKernel,
        );

        let near = analytics.score(&["near".to_owned()]).expect("near score");
        let far = analytics.score(&["far".to_owned()]).expect("far score");
        assert!(near.proximity_score_millis > far.proximity_score_millis);
        assert_eq!(near.diffusion_kind, KernelLocalDiffusionKind::HeatKernel);
    }

    fn vertex(id: &str) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: "generic".to_owned(),
            ..KernelVertex::default()
        }
    }

    fn edge(source: &str, target: &str, layer: KernelGraphLayer) -> KernelEdge {
        KernelEdge {
            source_id: KernelVertexId(source.to_owned()),
            target_id: KernelVertexId(target.to_owned()),
            edge_type: KernelEdgeType("edge".to_owned()),
            layer,
            ..KernelEdge::default()
        }
    }
}
