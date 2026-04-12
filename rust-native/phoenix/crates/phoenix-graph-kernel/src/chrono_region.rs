use crate::query_view::KernelQueryView;
use crate::{
    pcst_region::compact_region_with_pcst, KernelEdge, KernelExpandedRegion, KernelGraphLayer,
    KernelGraphSnapshot, KernelVertex, KernelVertexClass,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KernelRegionProfile {
    #[default]
    Generic,
    WorldState,
    History,
    Causal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum VertexFamily {
    Event,
    State,
    Claim,
    Entity,
    Temporal,
    Context,
    Generic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum EdgeFamily {
    Causal,
    StateSupport,
    Process,
    Support,
    Temporal,
    Identity,
    Generic,
}

#[derive(Clone, Copy, Debug)]
struct TraversalArc<'a> {
    neighbor: usize,
    edge: &'a KernelEdge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrontierItem {
    score_millis: i32,
    depth: usize,
    insertion_order: usize,
    index: usize,
}

impl Ord for FrontierItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score_millis
            .cmp(&other.score_millis)
            .then_with(|| other.depth.cmp(&self.depth))
            .then_with(|| other.insertion_order.cmp(&self.insertion_order))
            .then_with(|| other.index.cmp(&self.index))
    }
}

impl PartialOrd for FrontierItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
struct NeighborCandidate {
    neighbor: usize,
    include_score_millis: i32,
    expand_score_millis: i32,
    next_depth: usize,
    expandable: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct TemporalWindow {
    start: Option<i64>,
    end: Option<i64>,
}

pub(crate) fn expand_region_for_view(
    view: &KernelQueryView<'_>,
    anchor_vertex_ids: &[String],
    seed_vertex_ids: &[String],
    region_node_limit: usize,
    expansion_hops: usize,
    edge_allowed: fn(&KernelEdge) -> bool,
    profile: KernelRegionProfile,
) -> KernelExpandedRegion {
    let dense = view
        .vertices()
        .iter()
        .enumerate()
        .map(|(index, vertex)| (vertex.id.0.as_str(), index))
        .collect::<FxHashMap<_, _>>();
    let seed_vertex_ids = seed_vertex_ids
        .iter()
        .filter(|vertex_id| dense.contains_key(vertex_id.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if view.vertices().is_empty() {
        return KernelExpandedRegion {
            snapshot: KernelGraphSnapshot::default(),
            seed_vertex_ids,
            included_vertex_ids: Vec::new(),
            truncated: false,
        };
    }

    let adjacency = build_adjacency(view, &dense, edge_allowed);
    let stats = LocalPatternStats::from_view(view, edge_allowed);
    let anchor_indices = anchor_vertex_ids
        .iter()
        .filter_map(|id| dense.get(id.as_str()).copied())
        .collect::<FxHashSet<_>>();
    let seed_indices = seed_vertex_ids
        .iter()
        .filter_map(|id| dense.get(id.as_str()).copied())
        .collect::<FxHashSet<_>>();
    let anchor_entity_ids = collect_anchor_entity_ids(view, &anchor_indices, &seed_indices);
    let anchor_window = collect_anchor_window(view, &anchor_indices, &seed_indices);

    let node_limit = region_node_limit.clamp(8, 256);
    let max_hops = expansion_hops.clamp(1, 4);
    let mut included = vec![false; view.vertices().len()];
    let mut expanded = vec![false; view.vertices().len()];
    let mut frontier = BinaryHeap::<FrontierItem>::new();
    let mut insertion_order = 0usize;

    for &index in anchor_indices.iter().chain(seed_indices.iter()) {
        if included[index] {
            continue;
        }
        included[index] = true;
        if is_expandable(
            profile,
            vertex_family(view.vertices()[index]),
            anchor_indices.contains(&index) || seed_indices.contains(&index),
        ) {
            frontier.push(FrontierItem {
                score_millis: 10_000,
                depth: 0,
                insertion_order,
                index,
            });
            insertion_order += 1;
        }
    }

    let mut included_count = included.iter().filter(|slot| **slot).count();
    let mut truncated = false;

    while let Some(item) = frontier.pop() {
        if item.depth >= max_hops || expanded[item.index] {
            continue;
        }
        expanded[item.index] = true;

        let start = adjacency.offsets[item.index];
        let end = adjacency.offsets[item.index + 1];
        if start == end {
            continue;
        }

        let mut best_by_neighbor = FxHashMap::<usize, NeighborCandidate>::default();
        for arc in &adjacency.arcs[start..end] {
            if arc.neighbor == item.index {
                continue;
            }
            let neighbor = view.vertices()[arc.neighbor];
            let family = vertex_family(neighbor);
            if should_prune_transition(
                profile,
                view.vertices()[item.index],
                neighbor,
                arc.edge,
                family,
                anchor_window,
            ) {
                continue;
            }

            let next_depth = item.depth + 1;
            let score_millis = transition_score_millis(
                profile,
                view.vertices()[item.index],
                neighbor,
                arc.edge,
                family,
                next_depth,
                &stats,
                &anchor_entity_ids,
                anchor_window,
                seed_indices.contains(&arc.neighbor),
            );
            let anchor_or_seed =
                anchor_indices.contains(&arc.neighbor) || seed_indices.contains(&arc.neighbor);
            let expandable =
                next_depth < max_hops && is_expandable(profile, family, anchor_or_seed);
            if !included[arc.neighbor]
                && score_millis < include_threshold_millis(profile, family, arc.edge)
            {
                continue;
            }

            let candidate = NeighborCandidate {
                neighbor: arc.neighbor,
                include_score_millis: score_millis,
                expand_score_millis: score_millis
                    + expansion_bonus_millis(profile, family, arc.edge),
                next_depth,
                expandable,
            };
            match best_by_neighbor.get(&arc.neighbor) {
                Some(existing)
                    if existing.include_score_millis >= candidate.include_score_millis => {}
                _ => {
                    best_by_neighbor.insert(arc.neighbor, candidate);
                }
            }
        }

        let mut candidates = best_by_neighbor.into_values().collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .include_score_millis
                .cmp(&left.include_score_millis)
                .then_with(|| left.next_depth.cmp(&right.next_depth))
                .then_with(|| {
                    view.vertices()[left.neighbor]
                        .id
                        .0
                        .cmp(&view.vertices()[right.neighbor].id.0)
                })
        });

        for candidate in candidates {
            if !included[candidate.neighbor] {
                if included_count >= node_limit {
                    truncated = true;
                    break;
                }
                included[candidate.neighbor] = true;
                included_count += 1;
            }
            if candidate.expandable && !expanded[candidate.neighbor] {
                frontier.push(FrontierItem {
                    score_millis: candidate.expand_score_millis,
                    depth: candidate.next_depth,
                    insertion_order,
                    index: candidate.neighbor,
                });
                insertion_order += 1;
            }
        }

        if included_count >= node_limit {
            truncated = true;
            break;
        }
    }

    let included_count = included.iter().filter(|slot| **slot).count();
    let compacted = if included_count > 8 {
        compact_region_with_pcst(
            view,
            &included,
            &anchor_indices,
            &seed_indices,
            edge_allowed,
            profile,
        )
    } else {
        included
    };
    materialize_region(
        view,
        &dense,
        &compacted,
        seed_vertex_ids,
        truncated,
        edge_allowed,
    )
}

struct TraversalAdjacency<'a> {
    offsets: Vec<usize>,
    arcs: Vec<TraversalArc<'a>>,
}

fn build_adjacency<'a>(
    view: &'a KernelQueryView<'a>,
    dense: &FxHashMap<&'a str, usize>,
    edge_allowed: fn(&KernelEdge) -> bool,
) -> TraversalAdjacency<'a> {
    let mut degrees = vec![0usize; view.vertices().len()];
    for edge in view
        .asserted_edges()
        .iter()
        .chain(view.candidate_edges().iter())
        .filter(|edge| edge_allowed(edge))
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
        degrees[source] += 1;
        degrees[target] += 1;
    }
    let mut offsets = vec![0usize; view.vertices().len() + 1];
    for index in 0..view.vertices().len() {
        offsets[index + 1] = offsets[index] + degrees[index];
    }
    let arc_count = offsets[view.vertices().len()];
    if arc_count == 0 {
        return TraversalAdjacency {
            offsets,
            arcs: Vec::new(),
        };
    }
    let mut cursors = offsets[..view.vertices().len()].to_vec();
    let seed_edge = view
        .asserted_edges()
        .first()
        .copied()
        .or_else(|| view.candidate_edges().first().copied())
        .expect("adjacency seed edge should exist when arc count is non-zero");
    let mut arcs = vec![
        TraversalArc {
            neighbor: 0,
            edge: seed_edge,
        };
        arc_count
    ];
    for edge in view
        .asserted_edges()
        .iter()
        .chain(view.candidate_edges().iter())
        .filter(|edge| edge_allowed(edge))
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
        arcs[cursors[source]] = TraversalArc {
            neighbor: target,
            edge,
        };
        cursors[source] += 1;
        arcs[cursors[target]] = TraversalArc {
            neighbor: source,
            edge,
        };
        cursors[target] += 1;
    }
    TraversalAdjacency { offsets, arcs }
}

#[derive(Default)]
struct LocalPatternStats<'a> {
    edge_type_counts: FxHashMap<&'a str, usize>,
    edge_family_counts: FxHashMap<(&'a str, VertexFamily), usize>,
}

impl<'a> LocalPatternStats<'a> {
    fn from_view(view: &'a KernelQueryView<'a>, edge_allowed: fn(&KernelEdge) -> bool) -> Self {
        let mut stats = Self::default();
        for edge in view
            .asserted_edges()
            .iter()
            .chain(view.candidate_edges().iter())
            .filter(|edge| edge_allowed(edge))
        {
            *stats
                .edge_type_counts
                .entry(edge.edge_type.0.as_str())
                .or_insert(0) += 1;
            if let Some(target) = view.find_vertex(edge.target_id.0.as_str()) {
                *stats
                    .edge_family_counts
                    .entry((edge.edge_type.0.as_str(), vertex_family(target)))
                    .or_insert(0) += 1;
            }
            if let Some(source) = view.find_vertex(edge.source_id.0.as_str()) {
                *stats
                    .edge_family_counts
                    .entry((edge.edge_type.0.as_str(), vertex_family(source)))
                    .or_insert(0) += 1;
            }
        }
        stats
    }
}

fn materialize_region(
    view: &KernelQueryView<'_>,
    dense: &FxHashMap<&str, usize>,
    included: &[bool],
    seed_vertex_ids: Vec<String>,
    truncated: bool,
    edge_allowed: fn(&KernelEdge) -> bool,
) -> KernelExpandedRegion {
    let mut vertices = view
        .vertices()
        .iter()
        .enumerate()
        .filter(|(index, _)| included[*index])
        .map(|(_, vertex)| (*vertex).clone())
        .collect::<Vec<_>>();
    let mut asserted_edges = materialize_edges(
        view.asserted_edges().iter().copied(),
        dense,
        included,
        edge_allowed,
    );
    let mut candidate_edges = materialize_edges(
        view.candidate_edges().iter().copied(),
        dense,
        included,
        edge_allowed,
    );
    vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
    asserted_edges.sort_by(|left, right| left.source_id.0.cmp(&right.source_id.0));
    candidate_edges.sort_by(|left, right| left.source_id.0.cmp(&right.source_id.0));
    let mut included_vertex_ids = vertices
        .iter()
        .map(|vertex| vertex.id.0.clone())
        .collect::<Vec<_>>();
    included_vertex_ids.sort();
    KernelExpandedRegion {
        snapshot: KernelGraphSnapshot {
            vertices,
            asserted_edges,
            candidate_edges,
        },
        seed_vertex_ids,
        included_vertex_ids,
        truncated,
    }
}

fn materialize_edges<'a>(
    edges: impl Iterator<Item = &'a KernelEdge>,
    dense: &FxHashMap<&str, usize>,
    included: &[bool],
    edge_allowed: fn(&KernelEdge) -> bool,
) -> Vec<KernelEdge> {
    edges
        .filter(|edge| edge_allowed(edge))
        .filter(|edge| {
            let Some(&source) = dense.get(edge.source_id.0.as_str()) else {
                return false;
            };
            let Some(&target) = dense.get(edge.target_id.0.as_str()) else {
                return false;
            };
            included[source] && included[target]
        })
        .cloned()
        .collect::<Vec<_>>()
}

fn collect_anchor_entity_ids<'a>(
    view: &'a KernelQueryView<'a>,
    anchor_indices: &FxHashSet<usize>,
    seed_indices: &FxHashSet<usize>,
) -> FxHashSet<&'a str> {
    anchor_indices
        .iter()
        .chain(seed_indices.iter())
        .filter_map(|index| view.vertices()[*index].entity_id.as_deref())
        .collect::<FxHashSet<_>>()
}

fn collect_anchor_window(
    view: &KernelQueryView<'_>,
    anchor_indices: &FxHashSet<usize>,
    seed_indices: &FxHashSet<usize>,
) -> TemporalWindow {
    let mut window = TemporalWindow::default();
    for index in anchor_indices.iter().chain(seed_indices.iter()) {
        let temporal = &view.vertices()[*index].temporal;
        if let Some(start) = temporal.valid_from {
            window.start = Some(window.start.map_or(start, |current| current.min(start)));
        }
        if let Some(end) = temporal.valid_to {
            window.end = Some(window.end.map_or(end, |current| current.max(end)));
        }
    }
    window
}

fn transition_score_millis(
    profile: KernelRegionProfile,
    current: &KernelVertex,
    neighbor: &KernelVertex,
    edge: &KernelEdge,
    family: VertexFamily,
    next_depth: usize,
    stats: &LocalPatternStats<'_>,
    anchor_entity_ids: &FxHashSet<&str>,
    anchor_window: TemporalWindow,
    seed_neighbor: bool,
) -> i32 {
    let edge_family = edge_family(edge);
    let edge_count = stats
        .edge_type_counts
        .get(edge.edge_type.0.as_str())
        .copied()
        .unwrap_or(1);
    let pattern_count = stats
        .edge_family_counts
        .get(&(edge.edge_type.0.as_str(), family))
        .copied()
        .unwrap_or(1);
    let mut score = vertex_bias_millis(profile, family) + edge_bias_millis(profile, edge_family);
    score += ((90.0 / (edge_count as f64).sqrt()).round()) as i32;
    score += ((110.0 / (pattern_count as f64).sqrt()).round()) as i32;
    score += ((edge.provenance.confidence.unwrap_or(0.0).clamp(0.0, 1.0) * 100.0).round()) as i32;
    score += temporal_bias_millis(anchor_window, &neighbor.temporal, edge_family);
    if current.entity_id.as_deref().is_some()
        && current.entity_id.as_deref() == neighbor.entity_id.as_deref()
    {
        score += 70;
    }
    if neighbor
        .entity_id
        .as_deref()
        .map(|entity_id| anchor_entity_ids.contains(entity_id))
        .unwrap_or(false)
    {
        score += 90;
    }
    if seed_neighbor {
        score += 140;
    }
    if edge.layer == KernelGraphLayer::Candidate {
        score -= 25;
    }
    score - (next_depth as i32 * 35)
}

fn expansion_bonus_millis(
    profile: KernelRegionProfile,
    family: VertexFamily,
    edge: &KernelEdge,
) -> i32 {
    let family_bonus = match profile {
        KernelRegionProfile::Generic => 0,
        KernelRegionProfile::WorldState => match family {
            VertexFamily::State | VertexFamily::Claim => 70,
            VertexFamily::Entity => 30,
            VertexFamily::Event => 10,
            _ => -20,
        },
        KernelRegionProfile::History => match family {
            VertexFamily::State | VertexFamily::Event => 80,
            VertexFamily::Claim => 50,
            VertexFamily::Entity => 10,
            _ => -20,
        },
        KernelRegionProfile::Causal => match family {
            VertexFamily::Event => 100,
            VertexFamily::Claim => 40,
            VertexFamily::State => 30,
            VertexFamily::Entity => -10,
            _ => -35,
        },
    };
    family_bonus
        + if edge_family(edge) == EdgeFamily::Causal {
            40
        } else {
            0
        }
}

fn include_threshold_millis(
    profile: KernelRegionProfile,
    family: VertexFamily,
    edge: &KernelEdge,
) -> i32 {
    let base = match profile {
        KernelRegionProfile::Generic => -200,
        KernelRegionProfile::WorldState => match family {
            VertexFamily::State | VertexFamily::Claim | VertexFamily::Entity => -20,
            VertexFamily::Event | VertexFamily::Temporal => 10,
            VertexFamily::Context => 50,
            VertexFamily::Generic => 80,
        },
        KernelRegionProfile::History => match family {
            VertexFamily::State | VertexFamily::Event | VertexFamily::Claim => -10,
            VertexFamily::Entity | VertexFamily::Temporal => 20,
            VertexFamily::Context => 40,
            VertexFamily::Generic => 90,
        },
        KernelRegionProfile::Causal => match family {
            VertexFamily::Event | VertexFamily::Claim | VertexFamily::State => 0,
            VertexFamily::Entity => 35,
            VertexFamily::Temporal => 50,
            VertexFamily::Context => 70,
            VertexFamily::Generic => 100,
        },
    };
    base + if edge.layer == KernelGraphLayer::Candidate {
        10
    } else {
        0
    }
}

fn is_expandable(profile: KernelRegionProfile, family: VertexFamily, anchor_or_seed: bool) -> bool {
    match profile {
        KernelRegionProfile::Generic => true,
        KernelRegionProfile::WorldState => {
            matches!(family, VertexFamily::State | VertexFamily::Claim)
                || (anchor_or_seed && family == VertexFamily::Entity)
        }
        KernelRegionProfile::History => {
            matches!(
                family,
                VertexFamily::State | VertexFamily::Claim | VertexFamily::Event
            ) || (anchor_or_seed && matches!(family, VertexFamily::Entity | VertexFamily::Temporal))
        }
        KernelRegionProfile::Causal => {
            matches!(
                family,
                VertexFamily::Event | VertexFamily::Claim | VertexFamily::State
            ) || (anchor_or_seed && family == VertexFamily::Entity)
        }
    }
}

fn should_prune_transition(
    profile: KernelRegionProfile,
    _current: &KernelVertex,
    neighbor: &KernelVertex,
    edge: &KernelEdge,
    family: VertexFamily,
    anchor_window: TemporalWindow,
) -> bool {
    if profile == KernelRegionProfile::Generic {
        return false;
    }
    if family == VertexFamily::Generic && edge.layer == KernelGraphLayer::Candidate {
        return true;
    }
    if matches!(family, VertexFamily::Event | VertexFamily::State)
        && !temporal_overlap(anchor_window, &neighbor.temporal)
        && matches!(
            edge_family(edge),
            EdgeFamily::Process | EdgeFamily::Temporal
        )
    {
        return true;
    }
    false
}

fn temporal_overlap(window: TemporalWindow, temporal: &crate::KernelBiTemporal) -> bool {
    match (window.start, window.end) {
        (None, None) => true,
        (start, end) => {
            let other_start = temporal.valid_from.unwrap_or(i64::MIN);
            let other_end = temporal.valid_to.unwrap_or(i64::MAX);
            let window_start = start.unwrap_or(i64::MIN);
            let window_end = end.unwrap_or(i64::MAX);
            other_start < window_end && other_end > window_start
        }
    }
}

fn temporal_bias_millis(
    window: TemporalWindow,
    temporal: &crate::KernelBiTemporal,
    edge_family: EdgeFamily,
) -> i32 {
    match (window.start, window.end) {
        (None, None) => 0,
        _ if temporal_overlap(window, temporal) => 60,
        _ if matches!(edge_family, EdgeFamily::Process | EdgeFamily::Temporal) => -60,
        _ => -20,
    }
}

fn vertex_bias_millis(profile: KernelRegionProfile, family: VertexFamily) -> i32 {
    match profile {
        KernelRegionProfile::Generic => match family {
            VertexFamily::Event => 140,
            VertexFamily::State => 120,
            VertexFamily::Claim => 100,
            VertexFamily::Entity => 70,
            VertexFamily::Temporal => 50,
            VertexFamily::Context => 20,
            VertexFamily::Generic => 0,
        },
        KernelRegionProfile::WorldState => match family {
            VertexFamily::State => 230,
            VertexFamily::Claim => 170,
            VertexFamily::Entity => 120,
            VertexFamily::Event => 60,
            VertexFamily::Temporal => 40,
            VertexFamily::Context => -10,
            VertexFamily::Generic => -70,
        },
        KernelRegionProfile::History => match family {
            VertexFamily::State => 210,
            VertexFamily::Event => 190,
            VertexFamily::Claim => 130,
            VertexFamily::Entity => 70,
            VertexFamily::Temporal => 75,
            VertexFamily::Context => 0,
            VertexFamily::Generic => -55,
        },
        KernelRegionProfile::Causal => match family {
            VertexFamily::Event => 250,
            VertexFamily::Claim => 150,
            VertexFamily::State => 120,
            VertexFamily::Entity => 45,
            VertexFamily::Temporal => 20,
            VertexFamily::Context => -25,
            VertexFamily::Generic => -80,
        },
    }
}

fn edge_bias_millis(profile: KernelRegionProfile, family: EdgeFamily) -> i32 {
    match profile {
        KernelRegionProfile::Generic => match family {
            EdgeFamily::Causal => 90,
            EdgeFamily::StateSupport => 90,
            EdgeFamily::Process => 80,
            EdgeFamily::Support => 60,
            EdgeFamily::Temporal => 50,
            EdgeFamily::Identity => 35,
            EdgeFamily::Generic => 0,
        },
        KernelRegionProfile::WorldState => match family {
            EdgeFamily::StateSupport => 220,
            EdgeFamily::Support => 110,
            EdgeFamily::Identity => 60,
            EdgeFamily::Temporal => 45,
            EdgeFamily::Process => 30,
            EdgeFamily::Causal => 20,
            EdgeFamily::Generic => 0,
        },
        KernelRegionProfile::History => match family {
            EdgeFamily::StateSupport => 150,
            EdgeFamily::Process => 140,
            EdgeFamily::Temporal => 130,
            EdgeFamily::Causal => 120,
            EdgeFamily::Support => 80,
            EdgeFamily::Identity => 35,
            EdgeFamily::Generic => 0,
        },
        KernelRegionProfile::Causal => match family {
            EdgeFamily::Causal => 230,
            EdgeFamily::Process => 170,
            EdgeFamily::Support => 90,
            EdgeFamily::StateSupport => 70,
            EdgeFamily::Identity => 45,
            EdgeFamily::Temporal => 40,
            EdgeFamily::Generic => 0,
        },
    }
}

fn vertex_family(vertex: &KernelVertex) -> VertexFamily {
    match vertex.class {
        KernelVertexClass::Event => VertexFamily::Event,
        KernelVertexClass::State => VertexFamily::State,
        KernelVertexClass::Entity => VertexFamily::Entity,
        KernelVertexClass::TimeAnchor | KernelVertexClass::CalendarAnchor => VertexFamily::Temporal,
        KernelVertexClass::Document
        | KernelVertexClass::Chunk
        | KernelVertexClass::Alias
        | KernelVertexClass::Mention
        | KernelVertexClass::Narrative
        | KernelVertexClass::Episode => VertexFamily::Context,
        KernelVertexClass::Generic | KernelVertexClass::Memory | KernelVertexClass::Task => {
            match vertex.kind.as_str() {
                "event" => VertexFamily::Event,
                "state" | "conflict" | "gap" => VertexFamily::State,
                "claim" => VertexFamily::Claim,
                "entity" => VertexFamily::Entity,
                "time_anchor" | "calendar_anchor" => VertexFamily::Temporal,
                "chunk" | "document" | "alias" | "mention" | "narrative" | "episode" => {
                    VertexFamily::Context
                }
                _ => VertexFamily::Generic,
            }
        }
    }
}

fn edge_family(edge: &KernelEdge) -> EdgeFamily {
    match edge.edge_type.0.as_str() {
        "causal_link" | "semantic::missing_intermediate_cause" => EdgeFamily::Causal,
        "state_of" | "state_value" => EdgeFamily::StateSupport,
        "semantic::same_process" | "semantic::related_event" => EdgeFamily::Process,
        "supported_by" | "about" | "under_view" | "subject" | "object" => EdgeFamily::Support,
        "canonicalized_as" => EdgeFamily::Identity,
        edge_type if edge_type.contains("time") || edge_type.contains("date") => {
            EdgeFamily::Temporal
        }
        _ => EdgeFamily::Generic,
    }
}

#[cfg(test)]
mod tests {
    use super::{expand_region_for_view, KernelRegionProfile};
    use crate::{
        KernelEdge, KernelEdgeType, KernelGraphLayer, KernelMutationBatch, KernelMutationScope,
        KernelVertex, KernelVertexClass, KernelVertexId, KernelViewRequest, PhoenixGraphKernel,
    };

    #[test]
    fn causal_profile_prefers_event_neighbors_over_context_when_region_is_tight() {
        let view = traversal_view(vec![
            edge("target", "entity_x", "subject", KernelGraphLayer::Asserted),
            edge("target", "entity_y", "subject", KernelGraphLayer::Asserted),
            edge("target", "entity_z", "subject", KernelGraphLayer::Asserted),
            edge("target", "entity_q", "subject", KernelGraphLayer::Asserted),
            edge("target", "entity_r", "subject", KernelGraphLayer::Asserted),
            edge("target", "entity_s", "subject", KernelGraphLayer::Asserted),
            edge("target", "entity_t", "subject", KernelGraphLayer::Asserted),
            edge(
                "target",
                "event_a",
                "semantic::related_event",
                KernelGraphLayer::Candidate,
            ),
            edge(
                "target",
                "event_b",
                "semantic::same_process",
                KernelGraphLayer::Candidate,
            ),
        ]);
        let region = expand_region_for_view(
            &view,
            &["target".to_owned()],
            &[],
            8,
            1,
            |_| true,
            KernelRegionProfile::Causal,
        );
        assert!(region.included_vertex_ids.iter().any(|id| id == "event_a"));
        assert!(region.included_vertex_ids.iter().any(|id| id == "event_b"));
        let context_count = region
            .included_vertex_ids
            .iter()
            .filter(|id| id.starts_with("entity_"))
            .count();
        assert_eq!(context_count, 5);
    }

    #[test]
    fn world_state_profile_includes_context_without_expanding_through_it() {
        let view = traversal_view(vec![
            edge("entity", "state", "state_of", KernelGraphLayer::Asserted),
            edge("state", "claim", "supported_by", KernelGraphLayer::Asserted),
            edge("claim", "chunk", "under_view", KernelGraphLayer::Asserted),
            edge(
                "chunk",
                "noise_state",
                "under_view",
                KernelGraphLayer::Asserted,
            ),
        ]);
        let region = expand_region_for_view(
            &view,
            &["entity".to_owned()],
            &[],
            8,
            4,
            |_| true,
            KernelRegionProfile::WorldState,
        );
        assert!(region.included_vertex_ids.iter().any(|id| id == "chunk"));
        assert!(!region
            .included_vertex_ids
            .iter()
            .any(|id| id == "noise_state"));
    }

    fn traversal_view(edges: Vec<KernelEdge>) -> crate::KernelQueryView<'static> {
        let mut kernel = PhoenixGraphKernel::new();
        let mut asserted_edges = Vec::new();
        let mut candidate_edges = Vec::new();
        for edge in edges {
            match edge.layer {
                KernelGraphLayer::Asserted => asserted_edges.push(edge),
                KernelGraphLayer::Candidate => candidate_edges.push(edge),
            }
        }
        kernel
            .apply_kernel_batch(KernelMutationBatch {
                layer: KernelGraphLayer::Asserted,
                scope: KernelMutationScope::Full,
                recorded_at: None,
                vertices: vec![
                    vertex("target", "event", KernelVertexClass::Event),
                    vertex("event_a", "event", KernelVertexClass::Event),
                    vertex("event_b", "event", KernelVertexClass::Event),
                    vertex("entity_x", "entity", KernelVertexClass::Entity),
                    vertex("entity_y", "entity", KernelVertexClass::Entity),
                    vertex("entity_z", "entity", KernelVertexClass::Entity),
                    vertex("entity_q", "entity", KernelVertexClass::Entity),
                    vertex("entity_r", "entity", KernelVertexClass::Entity),
                    vertex("entity_s", "entity", KernelVertexClass::Entity),
                    vertex("entity_t", "entity", KernelVertexClass::Entity),
                    vertex("entity", "entity", KernelVertexClass::Entity),
                    vertex("state", "state", KernelVertexClass::State),
                    vertex("claim", "claim", KernelVertexClass::Generic),
                    vertex("chunk", "chunk", KernelVertexClass::Chunk),
                    vertex("noise_state", "state", KernelVertexClass::State),
                ],
                edges: asserted_edges,
            })
            .expect("apply traversal batch");
        if !candidate_edges.is_empty() {
            kernel
                .apply_kernel_batch(KernelMutationBatch {
                    layer: KernelGraphLayer::Candidate,
                    scope: KernelMutationScope::Candidate {
                        scope_key: "chrono-region-test".to_owned(),
                    },
                    recorded_at: None,
                    vertices: Vec::new(),
                    edges: candidate_edges,
                })
                .expect("apply candidate traversal batch");
        }
        let leaked = Box::leak(Box::new(kernel));
        leaked.query_view(KernelViewRequest {
            include_candidate_graph: true,
            ..KernelViewRequest::default()
        })
    }

    fn vertex(id: &str, kind: &str, class: KernelVertexClass) -> KernelVertex {
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: kind.to_owned(),
            class,
            entity_id: if kind == "entity" {
                Some(id.to_owned())
            } else {
                None
            },
            ..KernelVertex::default()
        }
    }

    fn edge(source: &str, target: &str, edge_type: &str, layer: KernelGraphLayer) -> KernelEdge {
        KernelEdge {
            source_id: KernelVertexId(source.to_owned()),
            target_id: KernelVertexId(target.to_owned()),
            edge_type: KernelEdgeType(edge_type.to_owned()),
            layer,
            ..KernelEdge::default()
        }
    }
}
