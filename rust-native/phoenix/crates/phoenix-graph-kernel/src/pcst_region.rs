use crate::query_view::KernelQueryView;
use crate::{KernelEdge, KernelGraphLayer, KernelRegionProfile, KernelVertex, KernelVertexClass};
use rustc_hash::FxHashSet;

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
struct CompactArc {
    neighbor: usize,
    edge_cost_millis: i32,
}

pub(crate) fn compact_region_with_pcst(
    view: &KernelQueryView<'_>,
    included: &[bool],
    anchor_indices: &FxHashSet<usize>,
    seed_indices: &FxHashSet<usize>,
    edge_allowed: fn(&KernelEdge) -> bool,
    profile: KernelRegionProfile,
) -> Vec<bool> {
    if profile == KernelRegionProfile::Generic || included.len() < 2 {
        return included.to_vec();
    }

    let roots = if anchor_indices.is_empty() {
        seed_indices.iter().copied().collect::<Vec<_>>()
    } else {
        anchor_indices.iter().copied().collect::<Vec<_>>()
    };
    if roots.is_empty() {
        return included.to_vec();
    }

    let adjacency = build_adjacency(view, included, edge_allowed, profile);
    let active_count = included.iter().filter(|slot| **slot).count();
    if active_count <= roots.len().max(4) {
        return included.to_vec();
    }
    let prizes = build_prizes(view, included, anchor_indices, seed_indices, profile);
    let mut kept = mark_root_reachable(adjacency.as_slice(), included, roots.as_slice());
    strong_prune_leaves(adjacency.as_slice(), &prizes, roots.as_slice(), &mut kept);

    let kept_count = kept.iter().filter(|slot| **slot).count();
    if kept_count < roots.len().max(2) {
        return included.to_vec();
    }
    kept
}

fn build_adjacency(
    view: &KernelQueryView<'_>,
    included: &[bool],
    edge_allowed: fn(&KernelEdge) -> bool,
    profile: KernelRegionProfile,
) -> Vec<Vec<CompactArc>> {
    let mut adjacency = vec![Vec::<CompactArc>::new(); included.len()];
    for edge in view
        .asserted_edges()
        .iter()
        .chain(view.candidate_edges().iter())
        .filter(|edge| edge_allowed(edge))
    {
        let Some(source) = find_included_index(view, included, edge.source_id.0.as_str()) else {
            continue;
        };
        let Some(target) = find_included_index(view, included, edge.target_id.0.as_str()) else {
            continue;
        };
        if source == target {
            continue;
        }
        let cost = edge_cost_millis(
            profile,
            edge,
            vertex_family(view.vertices()[source]),
            vertex_family(view.vertices()[target]),
        );
        adjacency[source].push(CompactArc {
            neighbor: target,
            edge_cost_millis: cost,
        });
        adjacency[target].push(CompactArc {
            neighbor: source,
            edge_cost_millis: cost,
        });
    }
    adjacency
}

fn find_included_index(
    view: &KernelQueryView<'_>,
    included: &[bool],
    vertex_id: &str,
) -> Option<usize> {
    view.vertices()
        .iter()
        .enumerate()
        .find(|(index, vertex)| included[*index] && vertex.id.0 == vertex_id)
        .map(|(index, _)| index)
}

fn build_prizes(
    view: &KernelQueryView<'_>,
    included: &[bool],
    anchor_indices: &FxHashSet<usize>,
    seed_indices: &FxHashSet<usize>,
    profile: KernelRegionProfile,
) -> Vec<i32> {
    let mut prizes = vec![0i32; included.len()];
    for (index, vertex) in view.vertices().iter().enumerate() {
        if !included[index] {
            continue;
        }
        prizes[index] = node_prize_millis(
            profile,
            vertex,
            anchor_indices.contains(&index),
            seed_indices.contains(&index),
        );
    }
    prizes
}

fn node_prize_millis(
    profile: KernelRegionProfile,
    vertex: &KernelVertex,
    anchor: bool,
    seed: bool,
) -> i32 {
    let family = vertex_family(vertex);
    let mut prize = match profile {
        KernelRegionProfile::Generic => 500,
        KernelRegionProfile::WorldState => match family {
            VertexFamily::State => 1500,
            VertexFamily::Claim => 1150,
            VertexFamily::Entity => 650,
            VertexFamily::Event => 350,
            VertexFamily::Temporal => 220,
            VertexFamily::Context => 120,
            VertexFamily::Generic => 40,
        },
        KernelRegionProfile::History => match family {
            VertexFamily::State => 1350,
            VertexFamily::Event => 1300,
            VertexFamily::Claim => 900,
            VertexFamily::Entity => 520,
            VertexFamily::Temporal => 360,
            VertexFamily::Context => 130,
            VertexFamily::Generic => 50,
        },
        KernelRegionProfile::Causal => match family {
            VertexFamily::Event => 1700,
            VertexFamily::Claim => 950,
            VertexFamily::State => 820,
            VertexFamily::Entity => 320,
            VertexFamily::Temporal => 180,
            VertexFamily::Context => 70,
            VertexFamily::Generic => 30,
        },
    };
    if anchor {
        prize += 6000;
    } else if seed {
        prize += 1800;
    }
    if vertex
        .value
        .get("status")
        .and_then(serde_json::Value::as_str)
        .or_else(|| {
            vertex
                .attributes
                .get("status")
                .and_then(serde_json::Value::as_str)
        })
        == Some("active")
    {
        prize += 160;
    }
    prize += (vertex.provenance.evidence_refs.len().min(6) as i32) * 35;
    prize
}

fn mark_root_reachable(
    adjacency: &[Vec<CompactArc>],
    included: &[bool],
    roots: &[usize],
) -> Vec<bool> {
    let mut kept = vec![false; included.len()];
    let mut stack = roots.to_vec();
    while let Some(index) = stack.pop() {
        if index >= included.len() || kept[index] || !included[index] {
            continue;
        }
        kept[index] = true;
        for arc in &adjacency[index] {
            if included[arc.neighbor] && !kept[arc.neighbor] {
                stack.push(arc.neighbor);
            }
        }
    }
    kept
}

fn strong_prune_leaves(
    adjacency: &[Vec<CompactArc>],
    prizes: &[i32],
    roots: &[usize],
    kept: &mut [bool],
) {
    let root_set = roots.iter().copied().collect::<FxHashSet<_>>();
    loop {
        let mut changed = false;
        let mut degrees = vec![0usize; kept.len()];
        let mut cheapest_edge_cost = vec![i32::MAX; kept.len()];
        for (index, arcs) in adjacency.iter().enumerate() {
            if !kept[index] {
                continue;
            }
            for arc in arcs {
                if kept[arc.neighbor] {
                    degrees[index] += 1;
                    cheapest_edge_cost[index] = cheapest_edge_cost[index].min(arc.edge_cost_millis);
                }
            }
        }

        for index in 0..kept.len() {
            if !kept[index] || root_set.contains(&index) || degrees[index] != 1 {
                continue;
            }
            let edge_cost = cheapest_edge_cost[index];
            if edge_cost == i32::MAX {
                continue;
            }
            if prizes[index] <= edge_cost {
                kept[index] = false;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }
}

fn edge_cost_millis(
    profile: KernelRegionProfile,
    edge: &KernelEdge,
    source_family: VertexFamily,
    target_family: VertexFamily,
) -> i32 {
    let layer_penalty = match edge.layer {
        KernelGraphLayer::Asserted => 70,
        KernelGraphLayer::Candidate => 170,
    };
    let family_cost = match profile {
        KernelRegionProfile::Generic => 140,
        KernelRegionProfile::WorldState => match edge_family(edge) {
            EdgeFamily::StateSupport => 55,
            EdgeFamily::Support => 85,
            EdgeFamily::Identity => 90,
            EdgeFamily::Process => 170,
            EdgeFamily::Causal => 190,
            EdgeFamily::Temporal => 130,
            EdgeFamily::Generic => 200,
        },
        KernelRegionProfile::History => match edge_family(edge) {
            EdgeFamily::StateSupport => 70,
            EdgeFamily::Temporal => 75,
            EdgeFamily::Process => 85,
            EdgeFamily::Causal => 95,
            EdgeFamily::Support => 110,
            EdgeFamily::Identity => 130,
            EdgeFamily::Generic => 205,
        },
        KernelRegionProfile::Causal => match edge_family(edge) {
            EdgeFamily::Causal => 45,
            EdgeFamily::Process => 68,
            EdgeFamily::Support => 105,
            EdgeFamily::StateSupport => 125,
            EdgeFamily::Identity => 145,
            EdgeFamily::Temporal => 155,
            EdgeFamily::Generic => 225,
        },
    };
    let mismatch_penalty = match profile {
        KernelRegionProfile::Generic => 0,
        KernelRegionProfile::WorldState
            if matches!(edge_family(edge), EdgeFamily::Process | EdgeFamily::Causal)
                && matches!(source_family, VertexFamily::Claim | VertexFamily::Context)
                && matches!(target_family, VertexFamily::Claim | VertexFamily::Context) =>
        {
            900
        }
        KernelRegionProfile::WorldState
            if matches!(edge_family(edge), EdgeFamily::Process | EdgeFamily::Causal) =>
        {
            420
        }
        KernelRegionProfile::History
            if matches!(edge_family(edge), EdgeFamily::Support)
                && matches!(source_family, VertexFamily::Context)
                && matches!(target_family, VertexFamily::Context) =>
        {
            240
        }
        KernelRegionProfile::Causal
            if matches!(edge_family(edge), EdgeFamily::Support)
                && matches!(source_family, VertexFamily::Context)
                || matches!(target_family, VertexFamily::Context) =>
        {
            520
        }
        KernelRegionProfile::Causal
            if matches!(edge_family(edge), EdgeFamily::StateSupport)
                && matches!(source_family, VertexFamily::Context | VertexFamily::Entity)
                || matches!(target_family, VertexFamily::Context | VertexFamily::Entity) =>
        {
            200
        }
        _ => 0,
    };
    let confidence_discount =
        (edge.provenance.confidence.unwrap_or(0.0).clamp(0.0, 1.0) * 40.0).round() as i32;
    (family_cost + layer_penalty + mismatch_penalty - confidence_discount).max(20)
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
        _ => match vertex.kind.as_str() {
            "event" => VertexFamily::Event,
            "state" | "conflict" | "gap" => VertexFamily::State,
            "claim" => VertexFamily::Claim,
            "entity" => VertexFamily::Entity,
            "time_anchor" | "calendar_anchor" => VertexFamily::Temporal,
            "chunk" | "document" | "alias" | "mention" | "narrative" | "episode" => {
                VertexFamily::Context
            }
            _ => VertexFamily::Generic,
        },
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
    use super::compact_region_with_pcst;
    use crate::{
        KernelEdge, KernelEdgeType, KernelGraphLayer, KernelMutationBatch, KernelMutationScope,
        KernelRegionProfile, KernelVertex, KernelVertexClass, KernelVertexId, KernelViewRequest,
        PhoenixGraphKernel,
    };
    use rustc_hash::FxHashSet;

    #[test]
    fn pcst_world_state_prunes_noisy_relation_branch() {
        let view = view_with_edges(vec![
            edge("entity", "state", "state_of", KernelGraphLayer::Asserted),
            edge("state", "claim", "supported_by", KernelGraphLayer::Asserted),
            edge(
                "claim",
                "noisy_claim",
                "semantic::same_process",
                KernelGraphLayer::Candidate,
            ),
        ]);
        let included = vec![true; view.vertices().len()];
        let anchors = FxHashSet::from_iter([vertex_index(&view, "entity")]);
        let kept = compact_region_with_pcst(
            &view,
            &included,
            &anchors,
            &FxHashSet::default(),
            |_| true,
            KernelRegionProfile::WorldState,
        );
        assert!(kept[vertex_index(&view, "state")]);
        assert!(kept[vertex_index(&view, "claim")]);
        assert!(!kept[vertex_index(&view, "noisy_claim")]);
    }

    #[test]
    fn pcst_causal_keeps_event_chain_and_drops_context_leaf() {
        let view = view_with_edges(vec![
            edge(
                "target",
                "event_a",
                "causal_link",
                KernelGraphLayer::Asserted,
            ),
            edge(
                "event_a",
                "event_b",
                "semantic::same_process",
                KernelGraphLayer::Candidate,
            ),
            edge("event_b", "chunk", "under_view", KernelGraphLayer::Asserted),
        ]);
        let included = vec![true; view.vertices().len()];
        let anchors = FxHashSet::from_iter([vertex_index(&view, "target")]);
        let kept = compact_region_with_pcst(
            &view,
            &included,
            &anchors,
            &FxHashSet::default(),
            |_| true,
            KernelRegionProfile::Causal,
        );
        assert!(kept[vertex_index(&view, "event_a")]);
        assert!(kept[vertex_index(&view, "event_b")]);
        assert!(!kept[vertex_index(&view, "chunk")]);
    }

    fn view_with_edges(edges: Vec<KernelEdge>) -> crate::KernelQueryView<'static> {
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
                    vertex("entity", "entity", KernelVertexClass::Entity),
                    vertex("state", "state", KernelVertexClass::State),
                    vertex("claim", "claim", KernelVertexClass::Generic),
                    vertex("noisy_claim", "claim", KernelVertexClass::Generic),
                    vertex("target", "event", KernelVertexClass::Event),
                    vertex("event_a", "event", KernelVertexClass::Event),
                    vertex("event_b", "event", KernelVertexClass::Event),
                    vertex("chunk", "chunk", KernelVertexClass::Chunk),
                ],
                edges: asserted_edges,
            })
            .expect("apply asserted");
        if !candidate_edges.is_empty() {
            kernel
                .apply_kernel_batch(KernelMutationBatch {
                    layer: KernelGraphLayer::Candidate,
                    scope: KernelMutationScope::Candidate {
                        scope_key: "pcst-region-test".to_owned(),
                    },
                    recorded_at: None,
                    vertices: Vec::new(),
                    edges: candidate_edges,
                })
                .expect("apply candidate");
        }
        Box::leak(Box::new(kernel)).query_view(KernelViewRequest {
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

    fn vertex_index(view: &crate::KernelQueryView<'_>, id: &str) -> usize {
        view.vertices()
            .iter()
            .position(|vertex| vertex.id.0 == id)
            .expect("vertex index")
    }
}
