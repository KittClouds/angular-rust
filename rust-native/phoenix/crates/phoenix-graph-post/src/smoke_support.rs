use phoenix_graph_kernel::{KernelEdge, KernelVertex};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldAnchor {
    pub entity_id: String,
    pub slot_key: String,
    pub query_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalTarget {
    pub vertex_id: String,
    pub query_text: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CausalTargetCandidate {
    pub vertex_id: String,
    pub query_text: String,
    pub description: String,
    pub incoming_causal_edges: usize,
    pub path_bearing: bool,
}

pub fn discover_world_anchor(vertices: &[KernelVertex]) -> Option<WorldAnchor> {
    vertices
        .iter()
        .filter_map(|vertex| {
            let entity_id = vertex_entity_id(vertex)?;
            let slot_key = vertex_slot_key(vertex)?;
            Some((
                world_anchor_score(vertex, slot_key),
                WorldAnchor {
                    entity_id: entity_id.to_owned(),
                    slot_key: slot_key.to_owned(),
                    query_text: format!("current {} for {}", slot_key, entity_id),
                },
            ))
        })
        .max_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then_with(|| left.1.query_text.cmp(&right.1.query_text))
        })
        .map(|(_, anchor)| anchor)
}

pub fn discover_causal_target(vertices: &[KernelVertex]) -> Option<CausalTarget> {
    vertices
        .iter()
        .filter(|vertex| matches!(vertex.kind.as_str(), "event" | "claim"))
        .max_by(|left, right| {
            causal_target_score(left)
                .cmp(&causal_target_score(right))
                .then_with(|| left.id.0.cmp(&right.id.0))
        })
        .map(|vertex| CausalTarget {
            vertex_id: vertex.id.0.clone(),
            query_text: format!("what led to {}", describe_vertex(vertex)),
        })
}

pub fn discover_causal_target_candidates(
    vertices: &[KernelVertex],
    asserted_edges: &[KernelEdge],
    limit: usize,
) -> Vec<CausalTargetCandidate> {
    let incoming = incoming_causal_counts(asserted_edges);
    let mut candidates = vertices
        .iter()
        .filter(|vertex| matches!(vertex.kind.as_str(), "event" | "claim"))
        .map(|vertex| {
            let count = incoming.get(vertex.id.0.as_str()).copied().unwrap_or(0);
            let description = describe_vertex(vertex);
            let query_text = format!("what led to {}", description);
            let path_bonus = if count > 0 { 360 } else { 0 };
            (
                causal_target_score(vertex) + path_bonus + (count as i32 * 260),
                CausalTargetCandidate {
                    vertex_id: vertex.id.0.clone(),
                    query_text,
                    description,
                    incoming_causal_edges: count,
                    path_bearing: count > 0,
                },
            )
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| {
                right
                    .1
                    .incoming_causal_edges
                    .cmp(&left.1.incoming_causal_edges)
            })
            .then_with(|| left.1.vertex_id.cmp(&right.1.vertex_id))
    });
    candidates
        .into_iter()
        .take(limit.max(1))
        .map(|(_, candidate)| candidate)
        .collect()
}

pub fn slot_key_of(vertex: &KernelVertex) -> Option<&str> {
    string_attr(&vertex.value, "slotKey").or_else(|| string_attr(&vertex.attributes, "slotKey"))
}

pub fn vertex_entity_id(vertex: &KernelVertex) -> Option<&str> {
    vertex
        .entity_id
        .as_deref()
        .or_else(|| parse_state_vertex_id(vertex.id.0.as_str()).map(|(entity_id, _)| entity_id))
}

pub fn vertex_slot_key(vertex: &KernelVertex) -> Option<&str> {
    slot_key_of(vertex)
        .or_else(|| parse_state_vertex_id(vertex.id.0.as_str()).map(|(_, slot_key)| slot_key))
}

pub fn describe_vertex(vertex: &KernelVertex) -> String {
    if let Some(label) = string_attr(&vertex.attributes, "semanticLabel") {
        if !label.is_empty() {
            return label.to_owned();
        }
    }
    let mut parts = Vec::new();
    if !vertex.labels.is_empty() {
        parts.push(vertex.labels.join(" "));
    }
    for key in [
        "kind",
        "slotKey",
        "value",
        "status",
        "oldValue",
        "newValue",
        "objectValue",
    ] {
        if let Some(value) =
            string_attr(&vertex.value, key).or_else(|| string_attr(&vertex.attributes, key))
        {
            if !value.is_empty() {
                parts.push(value.to_owned());
            }
        }
    }
    if parts.is_empty() {
        vertex.id.0.clone()
    } else {
        parts.join(" ")
    }
}

pub fn string_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find_map(|window| (window[0] == flag).then(|| window[1].clone()))
}

pub fn usize_arg(args: &[String], flag: &str) -> Option<usize> {
    string_arg(args, flag).and_then(|value| value.parse::<usize>().ok())
}

pub fn string_attr<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value.get(key).and_then(Value::as_str)
}

pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

fn world_anchor_score(vertex: &KernelVertex, slot_key: &str) -> i32 {
    let mut score = match vertex.kind.as_str() {
        "state" => 240,
        "claim" => 140,
        _ => 0,
    };
    if slot_key.starts_with("entity.") {
        score += 180;
    } else if slot_key.starts_with("relation.") {
        score -= 220;
    }
    if string_attr(&vertex.value, "status").or_else(|| string_attr(&vertex.attributes, "status"))
        == Some("active")
    {
        score += 50;
    }
    score += preferred_slot_score(slot_key);
    if vertex.id.0.contains("conflict") || vertex.id.0.contains("gap") {
        score -= 120;
    }
    if vertex.id.0.contains("archive-relation") && slot_key.starts_with("relation.") {
        score -= 180;
    }
    score
}

fn preferred_slot_score(slot_key: &str) -> i32 {
    match slot_key {
        "entity.location" => 140,
        "entity.membership" => 130,
        "entity.employer" => 120,
        "entity.role" => 110,
        "entity.status" => 100,
        key if key.starts_with("entity.") => 90,
        key if key.starts_with("relation.") => -260,
        _ => 0,
    }
}

fn causal_target_score(vertex: &KernelVertex) -> i32 {
    let mut score = match vertex.kind.as_str() {
        "event" => 220,
        "claim" => 80,
        _ => 0,
    };
    let id = vertex.id.0.as_str();
    if id.contains("state_started") {
        score += 180;
    }
    if id.contains("state_expired") || id.contains("state_changed") {
        score += 140;
    }
    if id.contains("conflict_opened") {
        score -= 180;
    }
    if let Some(slot_key) = vertex_slot_key(vertex) {
        if slot_key.starts_with("entity.") {
            score += 90;
        }
    }
    score
}

fn parse_state_vertex_id(id: &str) -> Option<(&str, &str)> {
    let state_prefix = "graph::state::state:";
    let event_prefix = "graph::event::memory::event:state_started:state:";
    let raw = id
        .strip_prefix(state_prefix)
        .or_else(|| id.strip_prefix(event_prefix))?;
    let split_at = raw.rfind(':')?;
    let (entity_id, slot_part) = raw.split_at(split_at);
    Some((entity_id, slot_part.trim_start_matches(':')))
}

fn incoming_causal_counts(edges: &[KernelEdge]) -> std::collections::BTreeMap<&str, usize> {
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for edge in edges {
        if edge.edge_type.0 == "causal_link" {
            *counts.entry(edge.target_id.0.as_str()).or_default() += 1;
        }
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::{discover_causal_target_candidates, discover_world_anchor};
    use phoenix_graph_kernel::{
        KernelEdge, KernelEdgeType, KernelVertex, KernelVertexClass, KernelVertexId,
    };
    use serde_json::json;

    #[test]
    fn discover_world_anchor_prefers_entity_state_over_relation_claim_noise() {
        let anchor = discover_world_anchor(&[
            vertex(
                "graph::claim::claim:archive-relation:test:abombs:mechron:relates_to",
                "claim",
                Some("test::abombs"),
                "relation.relates_to",
                None,
            ),
            vertex(
                "graph::state::state:test::ryan:entity.location",
                "state",
                Some("test::ryan"),
                "entity.location",
                Some("active"),
            ),
        ])
        .expect("anchor");

        assert_eq!(anchor.entity_id, "test::ryan");
        assert_eq!(anchor.slot_key, "entity.location");
    }

    #[test]
    fn discover_world_anchor_prefers_stable_entity_slots() {
        let anchor = discover_world_anchor(&[
            vertex(
                "graph::state::state:test::wyvern:entity.membership",
                "state",
                Some("test::wyvern"),
                "entity.membership",
                Some("active"),
            ),
            vertex(
                "graph::state::state:test::wyvern:entity.status",
                "state",
                Some("test::wyvern"),
                "entity.status",
                Some("active"),
            ),
        ])
        .expect("anchor");

        assert_eq!(anchor.slot_key, "entity.membership");
    }

    #[test]
    fn discover_causal_targets_prefers_path_bearing_events() {
        let no_path = vertex(
            "graph::event::memory::event:state_started:state:test::a:entity.location",
            "event",
            Some("test::a"),
            "entity.location",
            None,
        );
        let cause = vertex("graph::event::memory::cause", "event", None, "", None);
        let effect = vertex("graph::event::memory::effect", "event", None, "", None);
        let candidates = discover_causal_target_candidates(
            &[no_path, cause, effect],
            &[causal_edge(
                "graph::event::memory::cause",
                "graph::event::memory::effect",
            )],
            2,
        );

        assert_eq!(candidates[0].vertex_id, "graph::event::memory::effect");
        assert!(candidates[0].path_bearing);
    }

    fn vertex(
        id: &str,
        kind: &str,
        entity_id: Option<&str>,
        slot_key: &str,
        status: Option<&str>,
    ) -> KernelVertex {
        let mut value = json!({ "slotKey": slot_key });
        if let Some(status) = status {
            value["status"] = json!(status);
        }
        KernelVertex {
            id: KernelVertexId(id.to_owned()),
            kind: kind.to_owned(),
            class: if kind == "state" {
                KernelVertexClass::State
            } else {
                KernelVertexClass::Generic
            },
            entity_id: entity_id.map(str::to_owned),
            value,
            ..KernelVertex::default()
        }
    }

    fn causal_edge(source_id: &str, target_id: &str) -> KernelEdge {
        KernelEdge {
            source_id: KernelVertexId(source_id.to_owned()),
            target_id: KernelVertexId(target_id.to_owned()),
            edge_type: KernelEdgeType("causal_link".to_owned()),
            ..KernelEdge::default()
        }
    }
}
