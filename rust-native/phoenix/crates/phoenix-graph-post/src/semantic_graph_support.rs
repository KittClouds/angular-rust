use phoenix_semantic_v2::{
    DocumentArchive, EventIdentityScopeSidecar, MemoryClaimAtom, MemoryEventRecord, MemoryModality,
    MemoryScopeSidecar, MemoryStateRecord, SemanticCandidateStatus, SemanticEdgeFamily,
    SemanticGraphNodeKind, SemanticGraphNodeRecord,
};
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

pub(crate) const CHUNK_KIND: &str = "chunk";
pub(crate) const CLAIM_KIND: &str = "claim";
pub(crate) const STATE_KIND: &str = "state";
pub(crate) const EVENT_KIND: &str = "event";
pub(crate) const ENTITY_KIND: &str = "entity";

#[derive(Clone)]
pub(crate) struct Prototype {
    pub(crate) node_id: String,
    pub(crate) ann_kind: &'static str,
    pub(crate) node_kind: SemanticGraphNodeKind,
    pub(crate) text_key: String,
    pub(crate) text: String,
    pub(crate) truth_plane: Option<String>,
    pub(crate) document_id: Option<String>,
    pub(crate) narrative_id: Option<String>,
    pub(crate) evidence_refs: Vec<String>,
    pub(crate) semantic_node: SemanticGraphNodeRecord,
    pub(crate) slot_key: Option<String>,
    pub(crate) value_key: Option<String>,
}

pub(crate) fn build_prototypes(
    archives: &[DocumentArchive],
    event_identity_sidecar: Option<&EventIdentityScopeSidecar>,
    memory_sidecar: Option<&MemoryScopeSidecar>,
) -> Vec<Prototype> {
    let mut rows = Vec::new();
    for archive in archives {
        for chunk in &archive.chunks {
            rows.push(prototype(
                format!(
                    "chunk::{}::{}",
                    archive.manifest.document_id, chunk.chunk_id.0
                ),
                CHUNK_KIND,
                SemanticGraphNodeKind::Chunk,
                format!(
                    "chunk:{}:{}",
                    archive.manifest.document_id, chunk.chunk_id.0
                ),
                chunk.text.clone(),
                None,
                Some(archive.manifest.document_id.clone()),
                archive.manifest.scope.narrative_id.clone(),
                vec![format!(
                    "document:{}#bytes:{}-{}",
                    archive.manifest.document_id, chunk.range.start, chunk.range.end
                )],
                None,
                None,
            ));
        }
    }
    if let Some(memory) = memory_sidecar {
        for claim in &memory.claims {
            rows.push(claim_prototype(claim, &memory.scope.narrative_id));
        }
        for state in &memory.states {
            rows.push(state_prototype(state, &memory.scope.narrative_id));
        }
        for event in &memory.events {
            rows.push(event_prototype(event, &memory.scope.narrative_id));
        }
        for entity in &memory.entity_cards {
            let mut text = entity.identity.canonical_name.clone();
            if !entity.identity.aliases.is_empty() {
                text.push_str(" aliases ");
                text.push_str(&entity.identity.aliases.join(" ; "));
            }
            if !entity.current_state.is_empty() {
                text.push_str(" states ");
                text.push_str(
                    &entity
                        .current_state
                        .iter()
                        .map(|state| format!("{} {}", state.slot_key, state.value))
                        .collect::<Vec<_>>()
                        .join(" ; "),
                );
            }
            rows.push(prototype(
                format!("entity::{}", entity.entity_id.0),
                ENTITY_KIND,
                SemanticGraphNodeKind::Entity,
                format!("entity:{}", entity.entity_id.0),
                text,
                None,
                None,
                memory.scope.narrative_id.clone(),
                entity.identity.continuity_refs.clone(),
                None,
                None,
            ));
        }
    }
    if let Some(events) = event_identity_sidecar {
        for event in &events.canonical_events {
            rows.push(prototype(
                format!("graph::event::canonical::{}", event.canonical_event_id.0),
                EVENT_KIND,
                SemanticGraphNodeKind::Event,
                format!("canonical-event:{}", event.canonical_event_id.0),
                format!(
                    "{} {} {:?}",
                    event.canonical_label, event.normalized_predicate, event.participant_slots
                ),
                Some(modality_plane_label(Some(MemoryModality::Asserted))),
                event.document_ids.first().cloned(),
                events.scope.narrative_id.clone(),
                event.evidence_refs.clone(),
                None,
                None,
            ));
        }
    }
    rows
}

pub(crate) fn neighbor_families(
    kind: SemanticGraphNodeKind,
) -> [(&'static str, SemanticEdgeFamily); 2] {
    match kind {
        SemanticGraphNodeKind::Chunk => [
            (CHUNK_KIND, SemanticEdgeFamily::ChunkNeighbor),
            ("", SemanticEdgeFamily::Unknown),
        ],
        SemanticGraphNodeKind::Claim => [
            (CLAIM_KIND, SemanticEdgeFamily::ClaimSupport),
            ("", SemanticEdgeFamily::Unknown),
        ],
        SemanticGraphNodeKind::State => [
            (STATE_KIND, SemanticEdgeFamily::StateSupport),
            ("", SemanticEdgeFamily::Unknown),
        ],
        SemanticGraphNodeKind::Entity => [
            (STATE_KIND, SemanticEdgeFamily::EntityStateSupport),
            (EVENT_KIND, SemanticEdgeFamily::EntityEventSupport),
        ],
        SemanticGraphNodeKind::Event => [
            (EVENT_KIND, SemanticEdgeFamily::EventNeighbor),
            ("", SemanticEdgeFamily::Unknown),
        ],
        SemanticGraphNodeKind::Unknown => [
            ("", SemanticEdgeFamily::Unknown),
            ("", SemanticEdgeFamily::Unknown),
        ],
    }
}

pub(crate) fn resolve_family(
    base: SemanticEdgeFamily,
    source: &Prototype,
    target: &Prototype,
) -> SemanticEdgeFamily {
    match base {
        SemanticEdgeFamily::ClaimSupport
            if source.slot_key == target.slot_key && source.value_key != target.value_key =>
        {
            SemanticEdgeFamily::ClaimContradiction
        }
        SemanticEdgeFamily::StateSupport
            if source.slot_key == target.slot_key && source.value_key != target.value_key =>
        {
            SemanticEdgeFamily::StateContradiction
        }
        _ => base,
    }
}

pub(crate) fn truth_planes_compatible(left: Option<&str>, right: Option<&str>) -> bool {
    match (left, right) {
        (Some(left), Some(right)) => left == right || left == "world" || right == "world",
        _ => true,
    }
}

pub(crate) fn node_kind_label(kind: SemanticGraphNodeKind) -> &'static str {
    match kind {
        SemanticGraphNodeKind::Chunk => CHUNK_KIND,
        SemanticGraphNodeKind::Claim => CLAIM_KIND,
        SemanticGraphNodeKind::State => STATE_KIND,
        SemanticGraphNodeKind::Event => EVENT_KIND,
        SemanticGraphNodeKind::Entity => ENTITY_KIND,
        SemanticGraphNodeKind::Unknown => "unknown",
    }
}

pub(crate) fn family_label(family: SemanticEdgeFamily) -> &'static str {
    match family {
        SemanticEdgeFamily::ChunkNeighbor => "chunk_neighbor",
        SemanticEdgeFamily::ClaimSupport => "claim_support",
        SemanticEdgeFamily::ClaimContradiction => "claim_contradiction",
        SemanticEdgeFamily::StateSupport => "state_support",
        SemanticEdgeFamily::StateContradiction => "state_contradiction",
        SemanticEdgeFamily::EntityStateSupport => "entity_state_support",
        SemanticEdgeFamily::EntityEventSupport => "entity_event_support",
        SemanticEdgeFamily::EventNeighbor => "event_neighbor",
        SemanticEdgeFamily::EntityRoleNeighbor => "entity_role_neighbor",
        SemanticEdgeFamily::Unknown => "unknown",
    }
}

pub(crate) fn status_label(status: SemanticCandidateStatus) -> &'static str {
    match status {
        SemanticCandidateStatus::Generated => "generated",
        SemanticCandidateStatus::ReviewedSupport => "reviewed_support",
        SemanticCandidateStatus::ReviewedContradiction => "reviewed_contradiction",
        SemanticCandidateStatus::Deferred => "deferred",
        SemanticCandidateStatus::Rejected => "rejected",
    }
}

fn prototype(
    node_id: String,
    ann_kind: &'static str,
    node_kind: SemanticGraphNodeKind,
    text_key: String,
    text: String,
    truth_plane: Option<String>,
    document_id: Option<String>,
    narrative_id: Option<String>,
    evidence_refs: Vec<String>,
    slot_key: Option<String>,
    value_key: Option<String>,
) -> Prototype {
    let text_hash = fx_hash64(&text);
    Prototype {
        semantic_node: SemanticGraphNodeRecord {
            node_id: node_id.clone(),
            node_kind,
            document_id: document_id.clone(),
            narrative_id: narrative_id.clone(),
            text_key: text_key.clone(),
            text_hash,
            truth_plane: truth_plane.clone(),
            evidence_refs: evidence_refs.clone(),
        },
        node_id,
        ann_kind,
        node_kind,
        text_key,
        text,
        truth_plane,
        document_id,
        narrative_id,
        evidence_refs,
        slot_key,
        value_key,
    }
}

fn claim_prototype(claim: &MemoryClaimAtom, narrative_id: &Option<String>) -> Prototype {
    prototype(
        format!("graph::claim::{}", claim.claim_id),
        CLAIM_KIND,
        SemanticGraphNodeKind::Claim,
        format!("claim:{}", claim.claim_id),
        format!(
            "{} {} {} {}",
            claim.subject_label, claim.slot_key, claim.object_label, claim.object_value
        ),
        Some(modality_plane_label(Some(claim.modality))),
        Some(claim.document_id.clone()),
        narrative_id.clone(),
        claim.evidence_refs.clone(),
        Some(claim.slot_key.clone()),
        Some(normalized_key(&claim.object_value)),
    )
}

fn state_prototype(state: &MemoryStateRecord, narrative_id: &Option<String>) -> Prototype {
    prototype(
        format!("graph::state::{}", state.state_id),
        STATE_KIND,
        SemanticGraphNodeKind::State,
        format!("state:{}", state.state_id),
        format!("{} {}", state.slot_key, state.value),
        Some(state.source_class.clone()),
        None,
        narrative_id.clone(),
        state
            .claim_ids
            .iter()
            .map(|claim_id| format!("claim:{claim_id}"))
            .collect(),
        Some(state.slot_key.clone()),
        Some(normalized_key(&state.value)),
    )
}

fn event_prototype(event: &MemoryEventRecord, narrative_id: &Option<String>) -> Prototype {
    prototype(
        format!("graph::event::memory::{}", event.event_id),
        EVENT_KIND,
        SemanticGraphNodeKind::Event,
        format!("event:{}", event.event_id),
        format!(
            "{} {} {} {}",
            event.kind,
            event.slot_key,
            event.old_value.clone().unwrap_or_default(),
            event.new_value.clone().unwrap_or_default()
        ),
        None,
        Some(event.document_id.clone()),
        narrative_id.clone(),
        event.evidence_refs.clone(),
        Some(event.slot_key.clone()),
        Some(normalized_key(
            event.new_value.as_deref().unwrap_or_default(),
        )),
    )
}

fn modality_plane_label(modality: Option<MemoryModality>) -> String {
    match modality.unwrap_or(MemoryModality::Asserted) {
        MemoryModality::Asserted | MemoryModality::Observed | MemoryModality::Inferred => "world",
        MemoryModality::Reported => "reported",
        MemoryModality::Planned => "planned",
        MemoryModality::Conditional => "conditional",
        MemoryModality::Hypothetical => "hypothetical",
    }
    .to_owned()
}

fn normalized_key(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn fx_hash64(value: &str) -> u64 {
    let mut hasher = FxHasher::default();
    value.hash(&mut hasher);
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::{prototype, resolve_family, truth_planes_compatible, CLAIM_KIND, STATE_KIND};
    use phoenix_semantic_v2::{SemanticEdgeFamily, SemanticGraphNodeKind};

    #[test]
    fn resolve_family_promotes_slot_value_mismatches_to_contradiction() {
        let left = prototype(
            "graph::state::1".to_owned(),
            STATE_KIND,
            SemanticGraphNodeKind::State,
            "state:1".to_owned(),
            "entity.employer Acme".to_owned(),
            Some("world".to_owned()),
            None,
            None,
            Vec::new(),
            Some("entity.employer".to_owned()),
            Some("acme".to_owned()),
        );
        let right = prototype(
            "graph::state::2".to_owned(),
            STATE_KIND,
            SemanticGraphNodeKind::State,
            "state:2".to_owned(),
            "entity.employer Globex".to_owned(),
            Some("world".to_owned()),
            None,
            None,
            Vec::new(),
            Some("entity.employer".to_owned()),
            Some("globex".to_owned()),
        );
        assert_eq!(
            resolve_family(SemanticEdgeFamily::StateSupport, &left, &right),
            SemanticEdgeFamily::StateContradiction
        );
        let claim_left = prototype(
            "graph::claim::1".to_owned(),
            CLAIM_KIND,
            SemanticGraphNodeKind::Claim,
            "claim:1".to_owned(),
            "Alice employer Acme".to_owned(),
            Some("world".to_owned()),
            None,
            None,
            Vec::new(),
            Some("entity.employer".to_owned()),
            Some("acme".to_owned()),
        );
        let claim_right = prototype(
            "graph::claim::2".to_owned(),
            CLAIM_KIND,
            SemanticGraphNodeKind::Claim,
            "claim:2".to_owned(),
            "Alice employer Globex".to_owned(),
            Some("world".to_owned()),
            None,
            None,
            Vec::new(),
            Some("entity.employer".to_owned()),
            Some("globex".to_owned()),
        );
        assert_eq!(
            resolve_family(SemanticEdgeFamily::ClaimSupport, &claim_left, &claim_right),
            SemanticEdgeFamily::ClaimContradiction
        );
    }

    #[test]
    fn truth_plane_compatibility_keeps_world_open_but_blocks_cross_branch_leaks() {
        assert!(truth_planes_compatible(Some("world"), Some("reported")));
        assert!(truth_planes_compatible(
            Some("conditional"),
            Some("conditional")
        ));
        assert!(!truth_planes_compatible(Some("reported"), Some("planned")));
    }
}
