use phoenix_graph_kernel::{
    KernelBiTemporal, KernelEdge, KernelEdgeType, KernelGraphLayer, KernelMutationBatch,
    KernelMutationScope, KernelProvenance, KernelRelationClass, KernelVertex, KernelVertexClass,
    KernelVertexId, PhoenixGraphKernel,
};
use phoenix_semantic_v2::{
    CanonicalEventId, CanonicalEventRecord, CausalEdgeAddition, CausalScopeSidecar,
    DocumentArchive, EventIdentityScopeSidecar, GraphCompilerSummary, MemoryClaimAtom,
    MemoryConflictRecord, MemoryContinuityGapRecord, MemoryEventRecord, MemoryScopeSidecar,
    MemoryStateRecord, TemporalAnchorId, TemporalAnchorRecord, TemporalReferenceEdge,
    TemporalScopeSidecar, TemporalTimexId, TemporalTimexRecord,
};
use phoenix_types::{BiTemporalWindow, EntityId, SemanticNodeRef};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompiledGraphProjection {
    pub graph_batch: KernelMutationBatch,
    pub summary: GraphCompilerSummary,
}

#[derive(Default)]
struct GraphProjectionBuilder {
    vertices: FxHashMap<String, KernelVertex>,
    edges: FxHashMap<String, KernelEdge>,
}

#[derive(Clone, Debug)]
struct CausalEndpointResolution {
    vertex_id: String,
    vertex: KernelVertex,
    semantic_kind: &'static str,
    semantic_id: String,
    label: String,
    tier: &'static str,
    fallback: bool,
    promoted: bool,
}

struct CausalEndpointContext<'a> {
    canonical_by_event: &'a FxHashMap<String, CanonicalEventId>,
    canonical_by_semantic_node: &'a FxHashMap<String, CanonicalEventId>,
    raw_event_by_semantic_node: &'a FxHashMap<String, String>,
    label_by_semantic_node: &'a FxHashMap<String, String>,
}

impl GraphProjectionBuilder {
    fn add_vertex(&mut self, vertex: KernelVertex) {
        self.vertices.insert(vertex.id.0.clone(), vertex);
    }

    fn add_vertex_if_missing(&mut self, vertex: KernelVertex) {
        self.vertices.entry(vertex.id.0.clone()).or_insert(vertex);
    }

    fn add_edge(&mut self, edge: KernelEdge) {
        let key = format!(
            "{}|{}|{}",
            edge.source_id.0, edge.target_id.0, edge.edge_type.0
        );
        self.edges.insert(key, edge);
    }

    fn add_interval_artifacts(
        &mut self,
        owner_vertex_id: &KernelVertexId,
        temporal: &KernelBiTemporal,
        document_id: Option<String>,
        recorded_at: Option<i64>,
    ) {
        let (vertices, edges) = PhoenixGraphKernel::build_interval_anchor_links(
            owner_vertex_id,
            temporal,
            KernelGraphLayer::Asserted,
            document_id,
            recorded_at,
        );
        for vertex in vertices {
            self.add_vertex(vertex);
        }
        for edge in edges {
            self.add_edge(edge);
        }
    }
}

pub fn compile_graph_projection(
    scope_key: &str,
    event_identity_sidecar: Option<&EventIdentityScopeSidecar>,
    temporal_sidecar: Option<&TemporalScopeSidecar>,
    causal_sidecar: Option<&CausalScopeSidecar>,
    memory_sidecar: Option<&MemoryScopeSidecar>,
    recorded_at: Option<i64>,
) -> CompiledGraphProjection {
    compile_graph_projection_with_archives(
        scope_key,
        &[],
        event_identity_sidecar,
        temporal_sidecar,
        causal_sidecar,
        memory_sidecar,
        recorded_at,
    )
}

pub(crate) fn compile_graph_projection_with_archives(
    scope_key: &str,
    archives: &[DocumentArchive],
    event_identity_sidecar: Option<&EventIdentityScopeSidecar>,
    temporal_sidecar: Option<&TemporalScopeSidecar>,
    causal_sidecar: Option<&CausalScopeSidecar>,
    memory_sidecar: Option<&MemoryScopeSidecar>,
    recorded_at: Option<i64>,
) -> CompiledGraphProjection {
    let mut builder = GraphProjectionBuilder::default();
    let canonical_by_event = canonical_event_ids_by_event(event_identity_sidecar);
    let raw_event_by_semantic_node = raw_event_ids_by_semantic_node(archives);
    let label_by_semantic_node = labels_by_semantic_node(archives);
    let canonical_by_semantic_node = canonical_event_ids_by_semantic_node(
        causal_sidecar,
        &canonical_by_event,
        &raw_event_by_semantic_node,
    );
    let endpoint_context = CausalEndpointContext {
        canonical_by_event: &canonical_by_event,
        canonical_by_semantic_node: &canonical_by_semantic_node,
        raw_event_by_semantic_node: &raw_event_by_semantic_node,
        label_by_semantic_node: &label_by_semantic_node,
    };

    if let Some(sidecar) = event_identity_sidecar {
        for event in &sidecar.canonical_events {
            let event_id = canonical_event_vertex_id(&event.canonical_event_id);
            builder.add_vertex(canonical_event_vertex(event));
            let view_id = canonical_event_view_id(event);
            builder.add_vertex(view_vertex(
                &view_id,
                json!({
                    "plane": "event",
                    "sourceSemantics": label_of(&event.source_semantics),
                    "modalitySemantics": label_of(&event.modality_semantics),
                    "realis": event.realis,
                }),
            ));
            builder.add_edge(edge(
                &event_id,
                &view_id,
                "under_view",
                KernelRelationClass::Narrative,
                json!({}),
                event.document_ids.first().cloned(),
                None,
                KernelProvenance::default(),
            ));
            for slot in &event.participant_slots {
                if let Some(entity_id) = slot.entity_id.as_ref() {
                    builder.add_edge(edge(
                        &event_id,
                        &entity_vertex_id(entity_id),
                        &role_edge_type(&slot.role),
                        KernelRelationClass::Semantic,
                        json!({"role": slot.role, "label": slot.label}),
                        event.document_ids.first().cloned(),
                        None,
                        KernelProvenance::default(),
                    ));
                }
            }
            for time_anchor_id in &event.time_anchor_ids {
                builder.add_edge(edge(
                    &event_id,
                    &time_anchor_vertex_id(time_anchor_id),
                    "anchored_by",
                    KernelRelationClass::Temporal,
                    json!({}),
                    event.document_ids.first().cloned(),
                    None,
                    KernelProvenance::default(),
                ));
            }
        }
    }

    if let Some(sidecar) = memory_sidecar {
        for claim in &sidecar.claims {
            add_memory_claim(&mut builder, claim);
        }
        for state in &sidecar.states {
            add_memory_state(&mut builder, state, recorded_at);
        }
        for event in &sidecar.events {
            add_memory_event(&mut builder, event, recorded_at);
        }
        for conflict in &sidecar.conflicts {
            add_memory_conflict(&mut builder, conflict);
        }
        for gap in &sidecar.gaps {
            add_memory_gap(&mut builder, gap);
        }
    }

    if let Some(sidecar) = temporal_sidecar {
        for timex in &sidecar.timex_records {
            builder.add_vertex(timex_vertex(timex));
        }
        for anchor in &sidecar.anchors {
            add_time_anchor(&mut builder, anchor, recorded_at);
        }
        for edge_record in &sidecar.reference_edges {
            if let Some(target_id) = temporal_target_vertex_id(edge_record, &canonical_by_event) {
                builder.add_edge(temporal_reference_kernel_edge(
                    edge_record,
                    &canonical_by_event,
                    &target_id,
                ));
            }
        }
    }

    if let Some(sidecar) = causal_sidecar {
        for edge_record in &sidecar.edge_additions {
            let causal_temporal = kernel_temporal(&edge_record.effective_interval);
            let source_endpoint = causal_endpoint_resolution(
                &edge_record.source,
                edge_record.canonical_cause_event_id.as_ref(),
                &endpoint_context,
                edge_record.document_id.as_str(),
                &causal_temporal,
            );
            let target_endpoint = causal_endpoint_resolution(
                &edge_record.target,
                edge_record.canonical_effect_event_id.as_ref(),
                &endpoint_context,
                edge_record.document_id.as_str(),
                &causal_temporal,
            );
            builder.add_vertex_if_missing(source_endpoint.vertex.clone());
            builder.add_vertex_if_missing(target_endpoint.vertex.clone());
            builder.add_edge(causal_kernel_edge(
                edge_record,
                &source_endpoint,
                &target_endpoint,
                &causal_temporal,
            ));
        }
    }

    let mut vertices = builder.vertices.into_values().collect::<Vec<_>>();
    vertices.sort_by(|left, right| left.id.0.cmp(&right.id.0));
    let mut edges = builder.edges.into_values().collect::<Vec<_>>();
    edges.sort_by(|left, right| {
        (
            left.source_id.0.as_str(),
            left.target_id.0.as_str(),
            left.edge_type.0.as_str(),
        )
            .cmp(&(
                right.source_id.0.as_str(),
                right.target_id.0.as_str(),
                right.edge_type.0.as_str(),
            ))
    });
    let summary = summarize_projection(&vertices, &edges);

    CompiledGraphProjection {
        graph_batch: KernelMutationBatch {
            layer: KernelGraphLayer::Asserted,
            scope: KernelMutationScope::Projection {
                scope_key: scope_key.to_owned(),
            },
            recorded_at,
            vertices,
            edges,
        },
        summary,
    }
}

fn add_memory_claim(builder: &mut GraphProjectionBuilder, claim: &MemoryClaimAtom) {
    let claim_id = claim_vertex_id(claim.claim_id.as_str());
    let temporal = kernel_temporal(&claim.temporal);
    builder.add_vertex(claim_vertex(claim));
    let view_id = memory_view_id(claim.source_class.as_str(), &claim.modality);
    builder.add_vertex(view_vertex(
        &view_id,
        json!({
            "plane": "memory",
            "sourceClass": claim.source_class,
            "modality": label_of(&claim.modality),
        }),
    ));
    builder.add_edge(edge(
        &claim_id,
        &view_id,
        "under_view",
        KernelRelationClass::Narrative,
        json!({"status": label_of(&claim.status)}),
        Some(claim.document_id.clone()),
        Some(temporal.clone()),
        KernelProvenance::default(),
    ));
    if let Some(entity_id) = claim.source_entity_id.as_ref() {
        builder.add_edge(edge(
            &claim_id,
            &entity_vertex_id(entity_id),
            "subject",
            KernelRelationClass::Semantic,
            json!({"slotKey": claim.slot_key}),
            Some(claim.document_id.clone()),
            Some(temporal.clone()),
            KernelProvenance::default(),
        ));
    }
    let object_id = if let Some(entity_id) = claim.object_entity_id.as_ref() {
        entity_vertex_id(entity_id)
    } else {
        let value_id = value_vertex_id("claim", claim.claim_id.as_str());
        builder.add_vertex(scalar_value_vertex(
            &value_id,
            claim.object_value.as_str(),
            &claim.temporal,
        ));
        value_id
    };
    builder.add_edge(edge(
        &claim_id,
        &object_id,
        "object",
        KernelRelationClass::Semantic,
        json!({"slotKey": claim.slot_key}),
        Some(claim.document_id.clone()),
        Some(temporal),
        provenance("memory", claim.confidence_millis, &claim.evidence_refs),
    ));
}

fn add_memory_state(
    builder: &mut GraphProjectionBuilder,
    state: &MemoryStateRecord,
    recorded_at: Option<i64>,
) {
    let state_id = state_vertex_id(state.state_id.as_str());
    let temporal = kernel_temporal(&state.temporal);
    builder.add_vertex(state_vertex(state));
    builder.add_edge(edge(
        &state_id,
        &entity_vertex_id(&state.entity_id),
        "state_of",
        KernelRelationClass::Memory,
        json!({"slotKey": state.slot_key}),
        None,
        Some(temporal.clone()),
        KernelProvenance::default(),
    ));
    let value_id = if let Some(entity_id) = state.value_entity_id.as_ref() {
        entity_vertex_id(entity_id)
    } else {
        let value_id = value_vertex_id("state", state.state_id.as_str());
        builder.add_vertex(scalar_value_vertex(
            &value_id,
            state.value.as_str(),
            &state.temporal,
        ));
        value_id
    };
    builder.add_edge(edge(
        &state_id,
        &value_id,
        "state_value",
        KernelRelationClass::Memory,
        json!({"status": label_of(&state.status)}),
        None,
        Some(temporal.clone()),
        provenance("memory", state.confidence_millis, &[]),
    ));
    for claim_id in &state.claim_ids {
        builder.add_edge(edge(
            &state_id,
            &claim_vertex_id(claim_id),
            "supported_by",
            KernelRelationClass::Resolution,
            json!({}),
            None,
            Some(temporal.clone()),
            KernelProvenance::default(),
        ));
    }
    builder.add_interval_artifacts(&KernelVertexId(state_id), &temporal, None, recorded_at);
}

fn add_memory_event(
    builder: &mut GraphProjectionBuilder,
    event: &MemoryEventRecord,
    recorded_at: Option<i64>,
) {
    let event_id = memory_event_vertex_id(event.event_id.as_str());
    let temporal = kernel_temporal(&event.temporal);
    builder.add_vertex(memory_event_vertex(event));
    if let Some(canonical_event_id) = event.canonical_event_id.as_ref() {
        builder.add_edge(edge(
            &event_id,
            &canonical_event_vertex_id(canonical_event_id),
            "canonicalized_as",
            KernelRelationClass::Identity,
            json!({}),
            Some(event.document_id.clone()),
            Some(temporal.clone()),
            KernelProvenance::default(),
        ));
    }
    if let Some(entity_id) = event.subject_entity_id.as_ref() {
        builder.add_edge(edge(
            &event_id,
            &entity_vertex_id(entity_id),
            "subject",
            KernelRelationClass::Semantic,
            json!({"slotKey": event.slot_key}),
            Some(event.document_id.clone()),
            Some(temporal.clone()),
            KernelProvenance::default(),
        ));
    }
    if let Some(entity_id) = event.object_entity_id.as_ref() {
        builder.add_edge(edge(
            &event_id,
            &entity_vertex_id(entity_id),
            "object",
            KernelRelationClass::Semantic,
            json!({"slotKey": event.slot_key}),
            Some(event.document_id.clone()),
            Some(temporal.clone()),
            KernelProvenance::default(),
        ));
    }
    for claim_id in &event.claim_ids {
        builder.add_edge(edge(
            &event_id,
            &claim_vertex_id(claim_id),
            "supported_by",
            KernelRelationClass::Resolution,
            json!({}),
            Some(event.document_id.clone()),
            Some(temporal.clone()),
            KernelProvenance::default(),
        ));
    }
    builder.add_interval_artifacts(
        &KernelVertexId(event_id),
        &temporal,
        Some(event.document_id.clone()),
        recorded_at,
    );
}

fn add_memory_conflict(builder: &mut GraphProjectionBuilder, conflict: &MemoryConflictRecord) {
    let conflict_id = conflict_vertex_id(conflict.conflict_id.as_str());
    let temporal = kernel_temporal(&conflict.temporal);
    builder.add_vertex(conflict_vertex(conflict));
    builder.add_edge(edge(
        &conflict_id,
        &entity_vertex_id(&conflict.entity_id),
        "about",
        KernelRelationClass::Memory,
        json!({"slotKey": conflict.slot_key}),
        None,
        Some(temporal.clone()),
        KernelProvenance::default(),
    ));
    for claim_id in &conflict.claim_ids {
        builder.add_edge(edge(
            &conflict_id,
            &claim_vertex_id(claim_id),
            "supported_by",
            KernelRelationClass::Resolution,
            json!({}),
            None,
            Some(temporal.clone()),
            KernelProvenance::default(),
        ));
    }
}

fn add_memory_gap(builder: &mut GraphProjectionBuilder, gap: &MemoryContinuityGapRecord) {
    let gap_id = gap_vertex_id(gap.gap_id.as_str());
    let temporal = kernel_temporal(&gap.temporal);
    builder.add_vertex(gap_vertex(gap));
    builder.add_edge(edge(
        &gap_id,
        &entity_vertex_id(&gap.entity_id),
        "about",
        KernelRelationClass::Memory,
        json!({"slotKey": gap.slot_key}),
        None,
        Some(temporal.clone()),
        KernelProvenance::default(),
    ));
    for claim_id in &gap.claim_ids {
        builder.add_edge(edge(
            &gap_id,
            &claim_vertex_id(claim_id),
            "supported_by",
            KernelRelationClass::Resolution,
            json!({}),
            None,
            Some(temporal.clone()),
            KernelProvenance::default(),
        ));
    }
}

fn add_time_anchor(
    builder: &mut GraphProjectionBuilder,
    anchor: &TemporalAnchorRecord,
    recorded_at: Option<i64>,
) {
    let anchor_id = time_anchor_vertex_id(&anchor.anchor_id);
    let temporal = kernel_temporal(&anchor.temporal);
    builder.add_vertex(time_anchor_vertex(anchor));
    if let Some(canonical_event_id) = anchor.canonical_event_id.as_ref() {
        builder.add_edge(edge(
            &canonical_event_vertex_id(canonical_event_id),
            &anchor_id,
            "anchored_by",
            KernelRelationClass::Temporal,
            json!({"anchorKind": anchor.anchor_kind}),
            Some(anchor.document_id.clone()),
            Some(temporal.clone()),
            provenance("temporal", anchor.confidence_millis, &anchor.evidence_refs),
        ));
    }
    builder.add_interval_artifacts(
        &KernelVertexId(anchor_id),
        &temporal,
        Some(anchor.document_id.clone()),
        recorded_at,
    );
}

fn summarize_projection(vertices: &[KernelVertex], edges: &[KernelEdge]) -> GraphCompilerSummary {
    let mut summary = GraphCompilerSummary::default();
    summary.projection_vertex_count = vertices.len();
    summary.projection_edge_count = edges.len();
    for vertex in vertices {
        match vertex.kind.as_str() {
            "claim" => summary.claim_node_count += 1,
            "event" => summary.event_node_count += 1,
            "state" => summary.state_node_count += 1,
            "view" => summary.view_node_count += 1,
            "value" => summary.value_node_count += 1,
            "time_anchor" => summary.time_anchor_node_count += 1,
            "conflict" => summary.conflict_node_count += 1,
            "gap" => summary.gap_node_count += 1,
            _ => {}
        }
    }
    for edge in edges {
        if edge.edge_type.0 == "causal_link" {
            summary.causal_edge_count += 1;
        }
        if edge.edge_type.0 == "supported_by" {
            summary.support_edge_count += 1;
        }
        if matches!(edge.relation_class, KernelRelationClass::Temporal)
            || edge.edge_type.0.starts_with("temporal::")
        {
            summary.temporal_edge_count += 1;
        }
    }
    summary
}

fn canonical_event_ids_by_event(
    sidecar: Option<&EventIdentityScopeSidecar>,
) -> FxHashMap<String, CanonicalEventId> {
    let mut rows = FxHashMap::default();
    let Some(sidecar) = sidecar else {
        return rows;
    };
    let mention_by_id = sidecar
        .mention_packets
        .iter()
        .map(|packet| (packet.mention_id.0.clone(), packet.event_id.clone()))
        .collect::<FxHashMap<_, _>>();
    for membership in &sidecar.memberships {
        if let Some(event_id) = mention_by_id.get(membership.mention_id.0.as_str()) {
            rows.entry(event_id.clone())
                .or_insert_with(|| membership.canonical_event_id.clone());
        }
    }
    rows
}

fn raw_event_ids_by_semantic_node(archives: &[DocumentArchive]) -> FxHashMap<String, String> {
    let mut rows = FxHashMap::default();
    for archive in archives {
        let Some(substrate) = archive.causal_substrate.as_ref() else {
            continue;
        };
        let mut event_by_proposition = FxHashMap::<String, String>::default();
        for event in &substrate.semantic_events {
            if let Some(event_id) = event.event_id.as_ref() {
                event_by_proposition
                    .entry(event.proposition_id.to_string())
                    .or_insert_with(|| event_id.0.clone());
            }
        }
        for claim in &substrate.semantic_claims {
            if let (Some(claim_id), Some(event_id)) = (
                claim.claim_id.as_ref(),
                event_by_proposition.get(&claim.proposition_id.to_string()),
            ) {
                rows.entry(claim_id.0.clone())
                    .or_insert_with(|| event_id.clone());
            }
        }
        for state in &substrate.semantic_states {
            if let (Some(state_id), Some(event_id)) = (
                state.state_id.as_ref(),
                event_by_proposition.get(&state.proposition_id.to_string()),
            ) {
                rows.entry(state_id.0.clone())
                    .or_insert_with(|| event_id.clone());
            }
        }
    }
    rows
}

fn labels_by_semantic_node(archives: &[DocumentArchive]) -> FxHashMap<String, String> {
    let mut rows = FxHashMap::default();
    for archive in archives {
        let Some(substrate) = archive.causal_substrate.as_ref() else {
            continue;
        };
        for event in &substrate.semantic_events {
            if let Some(event_id) = event.event_id.as_ref() {
                rows.entry(event_id.0.clone())
                    .or_insert_with(|| event.label.to_string());
            }
        }
        for claim in &substrate.semantic_claims {
            if let Some(claim_id) = claim.claim_id.as_ref() {
                rows.entry(claim_id.0.clone())
                    .or_insert_with(|| claim.label.to_string());
            }
        }
        for state in &substrate.semantic_states {
            if let Some(state_id) = state.state_id.as_ref() {
                rows.entry(state_id.0.clone())
                    .or_insert_with(|| state.label.to_string());
            }
        }
    }
    rows
}

fn canonical_event_ids_by_semantic_node(
    causal_sidecar: Option<&CausalScopeSidecar>,
    canonical_by_event: &FxHashMap<String, CanonicalEventId>,
    raw_event_by_semantic_node: &FxHashMap<String, String>,
) -> FxHashMap<String, CanonicalEventId> {
    let mut rows = FxHashMap::default();
    for (semantic_id, event_id) in raw_event_by_semantic_node {
        if let Some(canonical_event_id) = canonical_by_event.get(event_id.as_str()) {
            rows.insert(semantic_id.clone(), canonical_event_id.clone());
        }
    }
    let Some(sidecar) = causal_sidecar else {
        return rows;
    };
    for atom in &sidecar.claim_atoms {
        insert_canonical_for_node(
            &mut rows,
            &atom.cause_event,
            atom.canonical_cause_event_id.as_ref(),
        );
        insert_canonical_for_node(
            &mut rows,
            &atom.effect_event,
            atom.canonical_effect_event_id.as_ref(),
        );
    }
    for edge in sidecar
        .edge_records
        .iter()
        .chain(sidecar.edge_additions.iter())
    {
        insert_canonical_for_node(
            &mut rows,
            &edge.source,
            edge.canonical_cause_event_id.as_ref(),
        );
        insert_canonical_for_node(
            &mut rows,
            &edge.target,
            edge.canonical_effect_event_id.as_ref(),
        );
    }
    for review in &sidecar.counterfactual_reviews {
        insert_canonical_for_node(
            &mut rows,
            &review.source,
            review.canonical_cause_event_id.as_ref(),
        );
        insert_canonical_for_node(
            &mut rows,
            &review.target,
            review.canonical_effect_event_id.as_ref(),
        );
    }
    rows
}

fn insert_canonical_for_node(
    rows: &mut FxHashMap<String, CanonicalEventId>,
    node: &SemanticNodeRef,
    canonical_event_id: Option<&CanonicalEventId>,
) {
    if let Some(canonical_event_id) = canonical_event_id {
        rows.insert(
            semantic_node_raw_id(node).to_owned(),
            canonical_event_id.clone(),
        );
    }
}

fn canonical_event_vertex(event: &CanonicalEventRecord) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(canonical_event_vertex_id(&event.canonical_event_id)),
        kind: "event".to_owned(),
        class: KernelVertexClass::Event,
        labels: vec![
            event.canonical_label.clone(),
            event.normalized_predicate.clone(),
        ],
        weight: event.confidence_millis as i64,
        value: json!({
            "label": event.canonical_label,
            "predicate": event.normalized_predicate,
            "eventType": event.event_type,
        }),
        attributes: json!({
            "documentIds": event.document_ids,
            "mentionIds": event.mention_ids,
            "placeLabels": event.place_labels,
            "sourceSemantics": label_of(&event.source_semantics),
            "modalitySemantics": label_of(&event.modality_semantics),
            "realis": event.realis,
        }),
        temporal: KernelBiTemporal::default(),
        provenance: provenance(
            "eventIdentity",
            event.confidence_millis,
            &event.evidence_refs,
        ),
        entity_id: None,
        search_chunk_id: None,
        document_id: event.document_ids.first().cloned(),
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: None,
    }
}

fn claim_vertex(claim: &MemoryClaimAtom) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(claim_vertex_id(claim.claim_id.as_str())),
        kind: "claim".to_owned(),
        class: KernelVertexClass::Generic,
        labels: vec![claim.slot_key.clone(), claim.object_value.clone()],
        weight: claim.confidence_millis as i64,
        value: json!({
            "slotKey": claim.slot_key,
            "objectValue": claim.object_value,
            "status": label_of(&claim.status),
            "modality": label_of(&claim.modality),
        }),
        attributes: json!({
            "relationFamily": claim.relation_family,
            "subjectLabel": claim.subject_label,
            "objectLabel": claim.object_label,
            "sourceClass": claim.source_class,
            "provenanceLabel": claim.provenance_label,
        }),
        temporal: kernel_temporal(&claim.temporal),
        provenance: provenance("memory", claim.confidence_millis, &claim.evidence_refs),
        entity_id: claim.source_entity_id.as_ref().map(|value| value.0.clone()),
        search_chunk_id: None,
        document_id: Some(claim.document_id.clone()),
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: None,
    }
}

fn state_vertex(state: &MemoryStateRecord) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(state_vertex_id(state.state_id.as_str())),
        kind: "state".to_owned(),
        class: KernelVertexClass::State,
        labels: vec![state.slot_key.clone(), state.value.clone()],
        weight: state.confidence_millis as i64,
        value: json!({
            "slotKey": state.slot_key,
            "value": state.value,
            "status": label_of(&state.status),
            "sourceClass": state.source_class,
        }),
        attributes: json!({
            "valueEntityId": state.value_entity_id.as_ref().map(|value| value.0.clone()),
            "confidenceMillis": state.confidence_millis,
            "claimIds": state.claim_ids,
        }),
        temporal: kernel_temporal(&state.temporal),
        provenance: provenance("memory", state.confidence_millis, &[]),
        entity_id: Some(state.entity_id.0.clone()),
        search_chunk_id: None,
        document_id: None,
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: None,
    }
}

fn memory_event_vertex(event: &MemoryEventRecord) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(memory_event_vertex_id(event.event_id.as_str())),
        kind: "event".to_owned(),
        class: KernelVertexClass::Event,
        labels: vec![event.kind.clone(), event.slot_key.clone()],
        weight: event.claim_ids.len() as i64,
        value: json!({"kind": event.kind, "slotKey": event.slot_key}),
        attributes: json!({
            "oldValue": event.old_value,
            "newValue": event.new_value,
            "conflictId": event.conflict_id,
        }),
        temporal: kernel_temporal(&event.temporal),
        provenance: provenance("memory", 0, &event.evidence_refs),
        entity_id: event
            .subject_entity_id
            .as_ref()
            .map(|value| value.0.clone()),
        search_chunk_id: None,
        document_id: Some(event.document_id.clone()),
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: None,
    }
}

fn conflict_vertex(conflict: &MemoryConflictRecord) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(conflict_vertex_id(conflict.conflict_id.as_str())),
        kind: "conflict".to_owned(),
        class: KernelVertexClass::Generic,
        labels: vec![conflict.slot_key.clone(), label_of(&conflict.kind)],
        weight: conflict.claim_ids.len() as i64,
        value: json!({
            "slotKey": conflict.slot_key,
            "kind": label_of(&conflict.kind),
            "status": label_of(&conflict.status),
        }),
        attributes: json!({
            "preferredClaimId": conflict.preferred_claim_id,
            "claimIds": conflict.claim_ids,
        }),
        temporal: kernel_temporal(&conflict.temporal),
        provenance: KernelProvenance::default(),
        entity_id: Some(conflict.entity_id.0.clone()),
        search_chunk_id: None,
        document_id: None,
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: None,
    }
}

fn gap_vertex(gap: &MemoryContinuityGapRecord) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(gap_vertex_id(gap.gap_id.as_str())),
        kind: "gap".to_owned(),
        class: KernelVertexClass::Generic,
        labels: vec![gap.slot_key.clone(), label_of(&gap.kind)],
        weight: gap.claim_ids.len() as i64,
        value: json!({
            "slotKey": gap.slot_key,
            "kind": label_of(&gap.kind),
            "detail": gap.detail,
            "status": label_of(&gap.status),
        }),
        attributes: json!({
            "claimIds": gap.claim_ids,
        }),
        temporal: kernel_temporal(&gap.temporal),
        provenance: KernelProvenance::default(),
        entity_id: Some(gap.entity_id.0.clone()),
        search_chunk_id: None,
        document_id: None,
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: None,
    }
}

fn timex_vertex(record: &TemporalTimexRecord) -> KernelVertex {
    simple_vertex(
        &timex_vertex_id(&record.timex_id),
        "time_anchor",
        KernelVertexClass::TimeAnchor,
        json!({"label": record.label, "normalizedValue": record.normalized_value}),
        Some(kernel_temporal(&record.temporal)),
    )
}

fn time_anchor_vertex(record: &TemporalAnchorRecord) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(time_anchor_vertex_id(&record.anchor_id)),
        kind: "time_anchor".to_owned(),
        class: KernelVertexClass::TimeAnchor,
        labels: vec![record.label.clone(), record.anchor_kind.clone()],
        weight: record.confidence_millis as i64,
        value: json!({"label": record.label, "anchorKind": record.anchor_kind}),
        attributes: json!({
            "axisId": record.axis_id,
            "timexId": record.timex_id,
            "referenceEventId": record.reference_event_id,
            "canonicalReferenceEventId": record.canonical_reference_event_id,
        }),
        temporal: kernel_temporal(&record.temporal),
        provenance: provenance("temporal", record.confidence_millis, &record.evidence_refs),
        entity_id: None,
        search_chunk_id: None,
        document_id: Some(record.document_id.clone()),
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: None,
    }
}

fn temporal_reference_kernel_edge(
    record: &TemporalReferenceEdge,
    canonical_by_event: &FxHashMap<String, CanonicalEventId>,
    target_id: &str,
) -> KernelEdge {
    let source_id = canonical_or_raw_event_vertex_id(
        record.canonical_source_event_id.as_ref(),
        Some(record.source_event_id.as_str()),
        canonical_by_event,
    );
    edge(
        &source_id,
        target_id,
        &format!("temporal::{}", slug(record.relation.as_str())),
        KernelRelationClass::Temporal,
        json!({"relation": record.relation, "axisId": record.axis_id}),
        Some(record.document_id.clone()),
        None,
        provenance("temporal", record.confidence_millis, &record.evidence_refs),
    )
}

fn causal_kernel_edge(
    record: &CausalEdgeAddition,
    source: &CausalEndpointResolution,
    target: &CausalEndpointResolution,
    temporal: &KernelBiTemporal,
) -> KernelEdge {
    edge(
        &source.vertex_id,
        &target.vertex_id,
        "causal_link",
        KernelRelationClass::Semantic,
        json!({
            "edgeId": record.edge_id,
            "caseId": record.case_id,
            "kind": label_of(&record.kind),
            "relationKind": label_of(&record.relation_kind),
            "status": label_of(&record.status),
            "cue": record.cue,
            "polarity": label_of(&record.polarity),
            "sourceSemanticNodeKind": source.semantic_kind,
            "sourceSemanticNodeId": source.semantic_id,
            "sourceSemanticLabel": source.label,
            "sourceEndpointResolutionTier": source.tier,
            "sourceEndpointFallback": source.fallback,
            "sourceEndpointPromoted": source.promoted,
            "targetSemanticNodeKind": target.semantic_kind,
            "targetSemanticNodeId": target.semantic_id,
            "targetSemanticLabel": target.label,
            "targetEndpointResolutionTier": target.tier,
            "targetEndpointFallback": target.fallback,
            "targetEndpointPromoted": target.promoted,
        }),
        Some(record.document_id.clone()),
        Some(temporal.clone()),
        provenance("causal", record.confidence_millis, &record.evidence_refs),
    )
}

fn causal_endpoint_resolution(
    node: &SemanticNodeRef,
    canonical_event_id: Option<&CanonicalEventId>,
    context: &CausalEndpointContext<'_>,
    document_id: &str,
    temporal: &KernelBiTemporal,
) -> CausalEndpointResolution {
    let semantic_kind = semantic_node_kind(node);
    let semantic_id = semantic_node_raw_id(node).to_owned();
    let label = context
        .label_by_semantic_node
        .get(semantic_id.as_str())
        .cloned()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| semantic_id.clone());
    let (vertex_id, tier, fallback, promoted) = if let Some(canonical_event_id) = canonical_event_id
    {
        (
            canonical_event_vertex_id(canonical_event_id),
            "canonicalEventField",
            false,
            !matches!(node, SemanticNodeRef::Event(_)),
        )
    } else if let Some(canonical_event_id) =
        context.canonical_by_semantic_node.get(semantic_id.as_str())
    {
        (
            canonical_event_vertex_id(canonical_event_id),
            "semanticNodeCanonical",
            false,
            !matches!(node, SemanticNodeRef::Event(_)),
        )
    } else if let Some(raw_event_id) = context.raw_event_by_semantic_node.get(semantic_id.as_str())
    {
        if let Some(canonical_event_id) = context.canonical_by_event.get(raw_event_id.as_str()) {
            (
                canonical_event_vertex_id(canonical_event_id),
                "semanticSiblingCanonicalEvent",
                false,
                !matches!(node, SemanticNodeRef::Event(_)),
            )
        } else {
            (
                memory_event_vertex_id(raw_event_id.as_str()),
                "semanticSiblingRawEvent",
                false,
                !matches!(node, SemanticNodeRef::Event(_)),
            )
        }
    } else if let SemanticNodeRef::Event(event_id) = node {
        if let Some(canonical_event_id) = context.canonical_by_event.get(event_id.0.as_str()) {
            (
                canonical_event_vertex_id(canonical_event_id),
                "eventIdentityCanonical",
                false,
                false,
            )
        } else {
            (
                memory_event_vertex_id(event_id.0.as_str()),
                "rawEvent",
                false,
                false,
            )
        }
    } else {
        match node {
            SemanticNodeRef::Event(_) => unreachable!("event refs are handled before fallback"),
            SemanticNodeRef::Claim(claim_id) => (
                claim_vertex_id(claim_id.0.as_str()),
                "fallbackSemanticClaim",
                true,
                false,
            ),
            SemanticNodeRef::State(state_id) => (
                state_vertex_id(state_id.0.as_str()),
                "stateRef",
                false,
                false,
            ),
        }
    };
    let vertex = match node {
        SemanticNodeRef::Claim(_) if fallback => {
            causal_endpoint_claim_vertex(&vertex_id, node, &label, document_id, temporal)
        }
        SemanticNodeRef::State(_) if fallback => {
            causal_endpoint_state_vertex(&vertex_id, node, &label, document_id, temporal)
        }
        _ => causal_endpoint_event_vertex(&vertex_id, node, &label, document_id, temporal),
    };
    CausalEndpointResolution {
        vertex_id,
        vertex,
        semantic_kind,
        semantic_id,
        label,
        tier,
        fallback,
        promoted,
    }
}

fn temporal_target_vertex_id(
    record: &TemporalReferenceEdge,
    canonical_by_event: &FxHashMap<String, CanonicalEventId>,
) -> Option<String> {
    record
        .canonical_target_event_id
        .as_ref()
        .map(canonical_event_vertex_id)
        .or_else(|| {
            record.target_event_id.as_deref().map(|event_id| {
                canonical_or_raw_event_vertex_id(None, Some(event_id), canonical_by_event)
            })
        })
        .or_else(|| record.target_timex_id.as_ref().map(timex_vertex_id))
}

fn canonical_or_raw_event_vertex_id(
    canonical_event_id: Option<&CanonicalEventId>,
    raw_event_id: Option<&str>,
    canonical_by_event: &FxHashMap<String, CanonicalEventId>,
) -> String {
    canonical_event_id
        .map(canonical_event_vertex_id)
        .or_else(|| {
            raw_event_id.and_then(|event_id| {
                canonical_by_event
                    .get(event_id)
                    .map(canonical_event_vertex_id)
                    .or_else(|| Some(memory_event_vertex_id(event_id)))
            })
        })
        .unwrap_or_else(|| "graph::event::unknown".to_owned())
}

fn edge(
    source_id: &str,
    target_id: &str,
    edge_type: &str,
    relation_class: KernelRelationClass,
    attributes: Value,
    document_id: Option<String>,
    temporal: Option<KernelBiTemporal>,
    provenance: KernelProvenance,
) -> KernelEdge {
    KernelEdge {
        source_id: KernelVertexId(source_id.to_owned()),
        target_id: KernelVertexId(target_id.to_owned()),
        edge_type: KernelEdgeType(edge_type.to_owned()),
        relation_class,
        weight: 1,
        attributes,
        data: None,
        document_id,
        narrative_id: None,
        layer: KernelGraphLayer::Asserted,
        temporal: temporal.unwrap_or_default(),
        provenance,
        resolution_facet: None,
    }
}

fn simple_vertex(
    id: &str,
    kind: &str,
    class: KernelVertexClass,
    value: Value,
    temporal: Option<KernelBiTemporal>,
) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(id.to_owned()),
        kind: kind.to_owned(),
        class,
        labels: Vec::new(),
        weight: 1,
        value,
        attributes: json!({}),
        temporal: temporal.unwrap_or_default(),
        provenance: KernelProvenance::default(),
        entity_id: None,
        search_chunk_id: None,
        document_id: None,
        chapter_id: None,
        chapters: Vec::new(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: Vec::new(),
        entity_facet: None,
        calendar_facet: None,
    }
}

fn scalar_value_vertex(id: &str, value: &str, temporal: &BiTemporalWindow) -> KernelVertex {
    simple_vertex(
        id,
        "value",
        KernelVertexClass::Generic,
        json!({"value": value}),
        Some(kernel_temporal(temporal)),
    )
}

fn view_vertex(id: &str, value: Value) -> KernelVertex {
    simple_vertex(id, "view", KernelVertexClass::Generic, value, None)
}

fn causal_endpoint_event_vertex(
    id: &str,
    node: &SemanticNodeRef,
    label: &str,
    document_id: &str,
    temporal: &KernelBiTemporal,
) -> KernelVertex {
    let mut vertex = simple_vertex(
        id,
        "event",
        KernelVertexClass::Event,
        json!({
            "label": label,
            "eventType": "causalEndpoint",
        }),
        Some(temporal.clone()),
    );
    vertex.labels = vec![label.to_owned()];
    vertex.attributes = json!({
        "sourceClass": "causal_endpoint",
        "semanticNodeKind": semantic_node_kind(node),
        "semanticNodeId": semantic_node_raw_id(node),
        "semanticLabel": label,
    });
    vertex.document_id = Some(document_id.to_owned());
    vertex.provenance.source = Some("causal".to_owned());
    vertex
}

fn causal_endpoint_claim_vertex(
    id: &str,
    node: &SemanticNodeRef,
    label: &str,
    document_id: &str,
    temporal: &KernelBiTemporal,
) -> KernelVertex {
    let mut vertex = simple_vertex(
        id,
        "claim",
        KernelVertexClass::Generic,
        json!({
            "slotKey": "semantic.claim",
            "objectValue": label,
            "status": "active",
            "modality": "asserted",
        }),
        Some(temporal.clone()),
    );
    vertex.labels = vec!["semantic.claim".to_owned(), label.to_owned()];
    vertex.attributes = json!({
        "sourceClass": "causal_semantic_claim",
        "semanticNodeId": semantic_node_raw_id(node),
        "semanticLabel": label,
    });
    vertex.document_id = Some(document_id.to_owned());
    vertex.provenance.source = Some("causal".to_owned());
    vertex
}

fn causal_endpoint_state_vertex(
    id: &str,
    node: &SemanticNodeRef,
    label: &str,
    document_id: &str,
    temporal: &KernelBiTemporal,
) -> KernelVertex {
    let mut vertex = simple_vertex(
        id,
        "state",
        KernelVertexClass::State,
        json!({
            "slotKey": "semantic.state",
            "value": label,
            "status": "active",
        }),
        Some(temporal.clone()),
    );
    vertex.labels = vec!["semantic.state".to_owned(), label.to_owned()];
    vertex.attributes = json!({
        "sourceClass": "causal_semantic_state",
        "semanticNodeId": semantic_node_raw_id(node),
        "semanticLabel": label,
    });
    vertex.document_id = Some(document_id.to_owned());
    vertex.provenance.source = Some("causal".to_owned());
    vertex
}

fn semantic_node_kind(node: &SemanticNodeRef) -> &'static str {
    match node {
        SemanticNodeRef::Event(_) => "event",
        SemanticNodeRef::Claim(_) => "claim",
        SemanticNodeRef::State(_) => "state",
    }
}

fn semantic_node_raw_id(node: &SemanticNodeRef) -> &str {
    match node {
        SemanticNodeRef::Event(id) => id.0.as_str(),
        SemanticNodeRef::Claim(id) => id.0.as_str(),
        SemanticNodeRef::State(id) => id.0.as_str(),
    }
}

fn kernel_temporal(window: &BiTemporalWindow) -> KernelBiTemporal {
    KernelBiTemporal {
        valid_from: window.valid_from,
        valid_to: window.valid_to,
        recorded_at: window.recorded_from,
        expired_at: window.recorded_to,
    }
}

fn provenance(source: &str, confidence_millis: u32, evidence_refs: &[String]) -> KernelProvenance {
    KernelProvenance {
        resolver: None,
        source: Some(source.to_owned()),
        confidence: (confidence_millis > 0).then_some(confidence_millis as f64 / 1000.0),
        evidence_refs: evidence_refs.to_vec(),
    }
}

fn label_of<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn slug(input: &str) -> String {
    let mut slug = String::with_capacity(input.len());
    let mut last_was_sep = false;
    for ch in input.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            last_was_sep = false;
        } else if !last_was_sep {
            slug.push('_');
            last_was_sep = true;
        }
    }
    slug.trim_matches('_').to_owned()
}

fn role_edge_type(role: &str) -> String {
    format!("role::{}", slug(role))
}

fn entity_vertex_id(entity_id: &EntityId) -> String {
    format!("entity::{}", entity_id.0)
}

fn canonical_event_vertex_id(canonical_event_id: &CanonicalEventId) -> String {
    format!("graph::event::canonical::{}", canonical_event_id.0)
}

fn memory_event_vertex_id(event_id: &str) -> String {
    format!("graph::event::memory::{event_id}")
}

fn claim_vertex_id(claim_id: &str) -> String {
    format!("graph::claim::{claim_id}")
}

fn state_vertex_id(state_id: &str) -> String {
    format!("graph::state::{state_id}")
}

fn conflict_vertex_id(conflict_id: &str) -> String {
    format!("graph::conflict::{conflict_id}")
}

fn gap_vertex_id(gap_id: &str) -> String {
    format!("graph::gap::{gap_id}")
}

fn value_vertex_id(prefix: &str, source_id: &str) -> String {
    format!("graph::value::{prefix}::{source_id}")
}

fn time_anchor_vertex_id(anchor_id: &TemporalAnchorId) -> String {
    format!("graph::time_anchor::anchor::{}", anchor_id.0)
}

fn timex_vertex_id(timex_id: &TemporalTimexId) -> String {
    format!("graph::time_anchor::timex::{}", timex_id.0)
}

fn memory_view_id(source_class: &str, modality: &impl serde::Serialize) -> String {
    format!(
        "graph::view::memory::{}::{}",
        slug(source_class),
        slug(label_of(modality).as_str())
    )
}

fn canonical_event_view_id(event: &CanonicalEventRecord) -> String {
    format!(
        "graph::view::event::{}::{}::{}",
        slug(label_of(&event.source_semantics).as_str()),
        slug(label_of(&event.modality_semantics).as_str()),
        slug(event.realis.as_str())
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_graph_kernel::{causal_path_candidate_views_from_snapshot, KernelGraphSnapshot};
    use phoenix_semantic_v2::{
        CausalClaimStatus, CausalEdgeId, CausalRelationKind, DocumentCausalSubstrate,
    };
    use phoenix_types::{CausalKind, ClaimId, ClaimRecord, EventId, EventRecord, Polarity};

    #[test]
    fn causal_claim_endpoint_is_materialized_for_runtime_paths() {
        let canonical_effect = CanonicalEventId("canonical:effect".to_owned());
        let sidecar = CausalScopeSidecar {
            edge_additions: vec![test_causal_edge(
                SemanticNodeRef::Claim(ClaimId("claim:prop:source".to_owned())),
                None,
                SemanticNodeRef::Event(EventId("event:raw:effect".to_owned())),
                Some(canonical_effect.clone()),
            )],
            ..CausalScopeSidecar::default()
        };

        let projection =
            compile_graph_projection("scope", None, None, Some(&sidecar), None, Some(100));
        let source_id = "graph::claim::claim:prop:source";
        let target_id = canonical_event_vertex_id(&canonical_effect);

        assert!(projection
            .graph_batch
            .vertices
            .iter()
            .any(|vertex| vertex.id.0 == source_id && vertex.kind == "claim"));
        assert!(projection
            .graph_batch
            .vertices
            .iter()
            .any(|vertex| vertex.id.0 == target_id && vertex.kind == "event"));
        let causal_edge = projection
            .graph_batch
            .edges
            .iter()
            .find(|edge| {
                edge.edge_type.0 == "causal_link"
                    && edge.source_id.0 == source_id
                    && edge.target_id.0 == target_id
            })
            .expect("causal edge");
        assert_eq!(
            causal_edge
                .attributes
                .get("sourceEndpointResolutionTier")
                .and_then(serde_json::Value::as_str),
            Some("fallbackSemanticClaim")
        );
        assert_eq!(
            causal_edge
                .attributes
                .get("targetEndpointResolutionTier")
                .and_then(serde_json::Value::as_str),
            Some("canonicalEventField")
        );
        assert_eq!(
            causal_edge
                .attributes
                .get("sourceEndpointFallback")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert!(projection.graph_batch.edges.iter().any(|edge| {
            edge.edge_type.0 == "causal_link"
                && edge.source_id.0 == source_id
                && edge.target_id.0 == target_id
        }));

        let snapshot = KernelGraphSnapshot {
            vertices: projection.graph_batch.vertices,
            asserted_edges: projection.graph_batch.edges,
            candidate_edges: Vec::new(),
        };
        let paths = causal_path_candidate_views_from_snapshot(&snapshot, &target_id, 3, 4);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].source_vertex_id, source_id);
    }

    #[test]
    fn causal_claim_endpoint_promotes_to_canonical_event_when_available() {
        let canonical_cause = CanonicalEventId("canonical:cause".to_owned());
        let canonical_effect = CanonicalEventId("canonical:effect".to_owned());
        let sidecar = CausalScopeSidecar {
            edge_additions: vec![test_causal_edge(
                SemanticNodeRef::Claim(ClaimId("claim:prop:source".to_owned())),
                Some(canonical_cause.clone()),
                SemanticNodeRef::Event(EventId("event:raw:effect".to_owned())),
                Some(canonical_effect.clone()),
            )],
            ..CausalScopeSidecar::default()
        };

        let projection =
            compile_graph_projection("scope", None, None, Some(&sidecar), None, Some(100));
        let source_id = canonical_event_vertex_id(&canonical_cause);
        let target_id = canonical_event_vertex_id(&canonical_effect);
        let causal_edge = projection
            .graph_batch
            .edges
            .iter()
            .find(|edge| edge.edge_type.0 == "causal_link")
            .expect("causal edge");

        assert_eq!(causal_edge.source_id.0, source_id);
        assert_eq!(causal_edge.target_id.0, target_id);
        assert_eq!(
            causal_edge
                .attributes
                .get("sourceEndpointPromoted")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert!(projection
            .graph_batch
            .vertices
            .iter()
            .any(|vertex| vertex.id.0 == source_id && vertex.kind == "event"));
    }

    #[test]
    fn causal_claim_endpoint_promotes_to_sibling_raw_event_from_archive() {
        let archive = DocumentArchive {
            causal_substrate: Some(DocumentCausalSubstrate {
                semantic_events: vec![EventRecord {
                    event_id: Some(EventId("event:source".to_owned())),
                    label: "source".into(),
                    proposition_id: "prop:source".into(),
                    ..EventRecord::default()
                }],
                semantic_claims: vec![ClaimRecord {
                    claim_id: Some(ClaimId("claim:prop:source".to_owned())),
                    label: "source claim".into(),
                    proposition_id: "prop:source".into(),
                    ..ClaimRecord::default()
                }],
                ..DocumentCausalSubstrate::default()
            }),
            ..DocumentArchive::default()
        };
        let sidecar = CausalScopeSidecar {
            edge_additions: vec![test_causal_edge(
                SemanticNodeRef::Claim(ClaimId("claim:prop:source".to_owned())),
                None,
                SemanticNodeRef::Event(EventId("event:effect".to_owned())),
                None,
            )],
            ..CausalScopeSidecar::default()
        };

        let projection = compile_graph_projection_with_archives(
            "scope",
            &[archive],
            None,
            None,
            Some(&sidecar),
            None,
            Some(100),
        );
        let causal_edge = projection
            .graph_batch
            .edges
            .iter()
            .find(|edge| edge.edge_type.0 == "causal_link")
            .expect("causal edge");

        assert_eq!(
            causal_edge.source_id.0,
            "graph::event::memory::event:source"
        );
        assert_eq!(
            causal_edge
                .attributes
                .get("sourceEndpointResolutionTier")
                .and_then(serde_json::Value::as_str),
            Some("semanticSiblingRawEvent")
        );
        assert_eq!(
            causal_edge
                .attributes
                .get("sourceEndpointPromoted")
                .and_then(serde_json::Value::as_bool),
            Some(true)
        );
        assert_eq!(
            causal_edge
                .attributes
                .get("sourceSemanticLabel")
                .and_then(serde_json::Value::as_str),
            Some("source claim")
        );
        assert_eq!(
            causal_edge
                .attributes
                .get("sourceEndpointFallback")
                .and_then(serde_json::Value::as_bool),
            Some(false)
        );
    }

    #[test]
    fn causal_fallback_claim_endpoint_uses_archive_label() {
        let archive = DocumentArchive {
            causal_substrate: Some(DocumentCausalSubstrate {
                semantic_claims: vec![ClaimRecord {
                    claim_id: Some(ClaimId("claim:prop:source".to_owned())),
                    label: "the bridge collapsed".into(),
                    proposition_id: "prop:source".into(),
                    ..ClaimRecord::default()
                }],
                ..DocumentCausalSubstrate::default()
            }),
            ..DocumentArchive::default()
        };
        let sidecar = CausalScopeSidecar {
            edge_additions: vec![test_causal_edge(
                SemanticNodeRef::Claim(ClaimId("claim:prop:source".to_owned())),
                None,
                SemanticNodeRef::Event(EventId("event:effect".to_owned())),
                None,
            )],
            ..CausalScopeSidecar::default()
        };

        let projection = compile_graph_projection_with_archives(
            "scope",
            &[archive],
            None,
            None,
            Some(&sidecar),
            None,
            Some(100),
        );
        let source = projection
            .graph_batch
            .vertices
            .iter()
            .find(|vertex| vertex.id.0 == "graph::claim::claim:prop:source")
            .expect("claim endpoint");

        assert_eq!(source.labels[1], "the bridge collapsed");
        assert_eq!(
            source
                .value
                .get("objectValue")
                .and_then(serde_json::Value::as_str),
            Some("the bridge collapsed")
        );
    }

    fn test_causal_edge(
        source: SemanticNodeRef,
        canonical_cause_event_id: Option<CanonicalEventId>,
        target: SemanticNodeRef,
        canonical_effect_event_id: Option<CanonicalEventId>,
    ) -> CausalEdgeAddition {
        CausalEdgeAddition {
            edge_id: CausalEdgeId("edge:1".to_owned()),
            case_id: "case:1".to_owned(),
            document_id: "doc:1".to_owned(),
            source,
            canonical_cause_event_id,
            target,
            canonical_effect_event_id,
            kind: CausalKind::Causes,
            relation_kind: CausalRelationKind::DirectCause,
            status: CausalClaimStatus::Supported,
            first_seen_revision: 1,
            latest_decision_id: None,
            confidence_millis: 870,
            cue: None,
            attributed_to: None,
            polarity: Polarity::Positive,
            claim_atom_ids: Vec::new(),
            evidence_refs: vec!["evidence:1".to_owned()],
            effective_interval: BiTemporalWindow {
                valid_from: Some(10),
                recorded_from: Some(100),
                ..BiTemporalWindow::default()
            },
            observation_interval: BiTemporalWindow::default(),
            temporal_certainty_millis: 1000,
            created_at: 100,
        }
    }
}
