use hashbrown::{HashMap, HashSet};
use phoenix_graph_kernel::{
    KernelEdge, KernelEdgeType, KernelGraphLayer, KernelMutationBatch, KernelMutationScope,
    KernelProvenance, KernelRelationClass, KernelVertex, KernelVertexClass, KernelVertexId,
};
use phoenix_semantic_v2::{
    scope_storage_key, DocumentArchive, EventIdentityScopeSidecar, GraphScopeSidecar,
    MemoryScopeSidecar, ScopeOrd, SemanticCandidateStatus, SemanticEdgeFamily,
    SemanticGraphCompilerSummary, SemanticGraphEdgeCandidate, SemanticGraphNodeKind,
    SemanticGraphScopeSidecar,
};
use phoenix_store_native_core::{
    NativeSemanticNodeVectorRecord, PhoenixArchiveStoreV2, PhoenixEventIdentityPatchStore,
    PhoenixGraphPatchStore, PhoenixMemoryPatchStore, PhoenixSemanticGraphPatchStore,
    PhoenixSemanticIndexStore, SEMANTIC_VECTOR_DIM,
};
use phoenix_types::{ScopeKey, SessionId};
use serde::{Deserialize, Serialize};
use serde_json::json;
use thiserror::Error;

use crate::semantic::{semantic_embedder, SemanticEmbedConfig};
use crate::semantic_graph_nli::{
    adjudicate_candidates_with_nli, needs_nli_review, SemanticNliConfig,
};
use crate::semantic_graph_support::{
    build_prototypes, family_label, neighbor_families, node_kind_label, resolve_family,
    status_label, truth_planes_compatible, Prototype,
};

const SEMANTIC_MODEL_ID: &str = "Snowflake/snowflake-arctic-embed-xs";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticGraphConfig {
    pub embed: SemanticEmbedConfig,
    pub nli: Option<SemanticNliConfig>,
    pub neighbor_limit: usize,
    pub oversample: usize,
    pub min_score_millis: u32,
}

impl Default for SemanticGraphConfig {
    fn default() -> Self {
        Self {
            embed: SemanticEmbedConfig::default(),
            nli: None,
            neighbor_limit: 3,
            oversample: 12,
            min_score_millis: 540,
        }
    }
}

#[derive(Debug, Error)]
pub enum SemanticGraphError {
    #[error(transparent)]
    Store(#[from] phoenix_store_native_core::StoreError),
    #[error(transparent)]
    Embed(#[from] crate::semantic::SemanticNeighborError),
    #[error(transparent)]
    Ort(#[from] phoenix_embed::OrtTextEmbedError),
    #[error(transparent)]
    Nli(#[from] phoenix_rel_post::NliError),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticGraphReviewBatch {
    pub scope: ScopeKey,
    pub scope_key: String,
    pub scope_ord: ScopeOrd,
    pub session_id: Option<SessionId>,
    pub graph_generation: Option<u64>,
    pub memory_generation: Option<u64>,
    pub event_identity_generation: Option<u64>,
    pub sidecar: SemanticGraphScopeSidecar,
}

pub fn derive_semantic_graph_review_batch_from_store<S>(
    store: &S,
    scope: &ScopeKey,
    config: &SemanticGraphConfig,
    created_at: i64,
) -> Result<Option<SemanticGraphReviewBatch>, SemanticGraphError>
where
    S: PhoenixArchiveStoreV2
        + PhoenixGraphPatchStore
        + PhoenixMemoryPatchStore
        + PhoenixEventIdentityPatchStore
        + PhoenixSemanticIndexStore,
{
    let archives = store.load_latest_document_archives(Some(scope))?;
    let graph_sidecar = store.load_graph_patch_sidecar(scope)?;
    let memory_sidecar = store.load_memory_patch_sidecar(scope)?;
    let event_identity_sidecar = store.load_event_identity_patch_sidecar(scope)?;
    if archives.is_empty() && graph_sidecar.is_none() && memory_sidecar.is_none() {
        return Ok(None);
    }
    let scope_key = archives
        .first()
        .map(|archive| archive.manifest.scope_key.clone())
        .or_else(|| {
            graph_sidecar
                .as_ref()
                .map(|sidecar| sidecar.scope_key.clone())
        })
        .or_else(|| {
            memory_sidecar
                .as_ref()
                .map(|sidecar| sidecar.scope_key.clone())
        })
        .unwrap_or_else(|| scope_storage_key(scope));
    let scope_ord = archives
        .first()
        .map(|archive| archive.manifest.scope_ord)
        .or_else(|| graph_sidecar.as_ref().and_then(|sidecar| sidecar.scope_ord))
        .or_else(|| {
            memory_sidecar
                .as_ref()
                .and_then(|sidecar| sidecar.scope_ord)
        })
        .unwrap_or_default();
    let session_id = archives
        .iter()
        .find_map(|archive| archive.manifest.session_id.clone())
        .or_else(|| {
            graph_sidecar
                .as_ref()
                .and_then(|sidecar| sidecar.session_id.clone())
        })
        .or_else(|| {
            memory_sidecar
                .as_ref()
                .and_then(|sidecar| sidecar.session_id.clone())
        });
    let sidecar = build_semantic_graph_sidecar(
        store,
        scope,
        &scope_key,
        scope_ord,
        session_id.as_ref(),
        &archives,
        graph_sidecar.as_ref(),
        event_identity_sidecar.as_ref(),
        memory_sidecar.as_ref(),
        config,
        created_at,
    )?;
    Ok(Some(SemanticGraphReviewBatch {
        scope: scope.clone(),
        scope_key,
        scope_ord,
        session_id,
        graph_generation: graph_sidecar.as_ref().map(|sidecar| sidecar.generation),
        memory_generation: memory_sidecar.as_ref().map(|sidecar| sidecar.generation),
        event_identity_generation: event_identity_sidecar
            .as_ref()
            .map(|sidecar| sidecar.generation),
        sidecar,
    }))
}

pub fn persist_semantic_graph_patch_sidecar<S>(
    store: &S,
    sidecar: &SemanticGraphScopeSidecar,
) -> Result<(), SemanticGraphError>
where
    S: PhoenixSemanticGraphPatchStore,
{
    store.persist_semantic_graph_patch_sidecar(sidecar)?;
    Ok(())
}

pub fn build_semantic_graph_sidecar<S>(
    store: &S,
    scope: &ScopeKey,
    scope_key: &str,
    scope_ord: ScopeOrd,
    session_id: Option<&SessionId>,
    archives: &[DocumentArchive],
    graph_sidecar: Option<&GraphScopeSidecar>,
    event_identity_sidecar: Option<&EventIdentityScopeSidecar>,
    memory_sidecar: Option<&MemoryScopeSidecar>,
    config: &SemanticGraphConfig,
    created_at: i64,
) -> Result<SemanticGraphScopeSidecar, SemanticGraphError>
where
    S: PhoenixSemanticIndexStore,
{
    let prototypes = build_prototypes(archives, event_identity_sidecar, memory_sidecar);
    let model = semantic_embedder(&config.embed)?;
    let texts = prototypes
        .iter()
        .map(|prototype| prototype.text.as_str())
        .collect::<Vec<_>>();
    let embeddings = model.embed_slices(&texts)?;
    let rows = prototypes
        .iter()
        .zip(embeddings.iter())
        .map(|(prototype, embedding)| NativeSemanticNodeVectorRecord {
            scope: scope.clone(),
            node_id: prototype.node_id.clone(),
            node_kind: prototype.ann_kind.to_owned(),
            document_id: prototype.document_id.clone(),
            narrative_id: prototype.narrative_id.clone(),
            folder_id: scope.folder_id.clone(),
            values: embedding.clone(),
            evidence_refs: prototype.evidence_refs.clone(),
            updated_at: created_at,
        })
        .collect::<Vec<_>>();
    store.upsert_semantic_node_vectors_native(&rows)?;
    let mut candidates = collect_candidate_edges(
        store,
        scope,
        &prototypes,
        &embeddings,
        config.neighbor_limit.max(1),
        config.oversample.max(config.neighbor_limit.max(1)),
        config.min_score_millis,
    )?;
    if candidates
        .iter()
        .any(|candidate| needs_nli_review(candidate.family))
    {
        if let Some(nli_config) = config.nli.as_ref() {
            let nli = nli_config.load_model()?;
            adjudicate_candidates_with_nli(&mut candidates, &prototypes, &nli, nli_config)?;
        }
    }
    let batch = compile_candidate_graph_batch(scope_key, &prototypes, &candidates, created_at);
    let summary = summarize(&prototypes, &candidates);
    Ok(SemanticGraphScopeSidecar {
        scope: scope.clone(),
        scope_key: scope_key.to_owned(),
        scope_ord: Some(scope_ord),
        session_id: session_id.cloned(),
        updated_at: created_at,
        generation: created_at as u64,
        model_id: SEMANTIC_MODEL_ID.to_owned(),
        embedding_profile: config.embed.profile.label().to_owned(),
        embedding_dim: SEMANTIC_VECTOR_DIM,
        candidate_nodes: prototypes
            .iter()
            .map(|prototype| prototype.semantic_node.clone())
            .collect(),
        candidate_edges: candidates,
        candidate_graph_batch: batch,
        graph_generation: graph_sidecar.map(|sidecar| sidecar.generation),
        memory_generation: memory_sidecar.map(|sidecar| sidecar.generation),
        event_identity_generation: event_identity_sidecar.map(|sidecar| sidecar.generation),
        summary,
    })
}

fn collect_candidate_edges<S>(
    store: &S,
    scope: &ScopeKey,
    prototypes: &[Prototype],
    embeddings: &[Vec<f32>],
    neighbor_limit: usize,
    oversample: usize,
    min_score_millis: u32,
) -> Result<Vec<SemanticGraphEdgeCandidate>, SemanticGraphError>
where
    S: PhoenixSemanticIndexStore,
{
    let prototype_by_id = prototypes
        .iter()
        .map(|prototype| (prototype.node_id.as_str(), prototype))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut edges = Vec::new();
    for (prototype, embedding) in prototypes.iter().zip(embeddings.iter()) {
        for (target_kind, family) in neighbor_families(prototype.node_kind) {
            for hit in store.query_semantic_node_neighbors(
                embedding,
                scope,
                target_kind,
                Some(&prototype.node_id),
                neighbor_limit,
                oversample,
            )? {
                let Some(target) = prototype_by_id.get(hit.node_id.as_str()) else {
                    continue;
                };
                if !truth_planes_compatible(
                    prototype.truth_plane.as_deref(),
                    target.truth_plane.as_deref(),
                ) {
                    continue;
                }
                let family = resolve_family(family, prototype, target);
                let score_millis = neighbor_score_millis(hit.distance);
                if score_millis < min_score_millis {
                    continue;
                }
                let dedupe = dedupe_key(family, prototype, target);
                if !seen.insert(dedupe) {
                    continue;
                }
                edges.push(SemanticGraphEdgeCandidate {
                    edge_id: format!(
                        "semantic:{}:{}:{}",
                        family_label(family),
                        prototype.node_id,
                        target.node_id
                    ),
                    family,
                    source_node_id: prototype.node_id.clone(),
                    source_kind: prototype.node_kind,
                    target_node_id: target.node_id.clone(),
                    target_kind: target.node_kind,
                    score_millis,
                    distance_millis: (hit.distance.max(0.0) * 1000.0).round() as u32,
                    candidate_status: SemanticCandidateStatus::Generated,
                    evidence_refs: merge_refs(&prototype.evidence_refs, &hit.evidence_refs),
                    model_evidence: vec![format!("ann:distance={:.4}", hit.distance)],
                    nli_support_millis: None,
                    nli_contradiction_millis: None,
                });
            }
        }
    }
    edges.sort_by(|left, right| left.edge_id.cmp(&right.edge_id));
    Ok(edges)
}

pub(crate) fn compile_candidate_graph_batch(
    scope_key: &str,
    prototypes: &[Prototype],
    candidates: &[SemanticGraphEdgeCandidate],
    created_at: i64,
) -> KernelMutationBatch {
    let prototype_by_id = prototypes
        .iter()
        .map(|prototype| (prototype.node_id.as_str(), prototype))
        .collect::<HashMap<_, _>>();
    let mut vertices = Vec::new();
    let mut seen_vertices = HashSet::new();
    let mut edges = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if candidate.candidate_status == SemanticCandidateStatus::Rejected {
            continue;
        }
        for node_id in [&candidate.source_node_id, &candidate.target_node_id] {
            if seen_vertices.insert(node_id.as_str()) {
                if let Some(prototype) = prototype_by_id.get(node_id.as_str()) {
                    if matches!(
                        prototype.node_kind,
                        SemanticGraphNodeKind::Chunk | SemanticGraphNodeKind::Entity
                    ) {
                        vertices.push(candidate_vertex(prototype, created_at));
                    }
                }
            }
        }
        edges.push(KernelEdge {
            source_id: KernelVertexId(candidate.source_node_id.clone()),
            target_id: KernelVertexId(candidate.target_node_id.clone()),
            edge_type: KernelEdgeType(format!("semantic::{}", family_label(candidate.family))),
            relation_class: KernelRelationClass::Candidate,
            weight: candidate.score_millis as i64,
            attributes: json!({
                "family": family_label(candidate.family),
                "status": status_label(candidate.candidate_status),
                "scoreMillis": candidate.score_millis,
                "distanceMillis": candidate.distance_millis,
                "sourceKind": node_kind_label(candidate.source_kind),
                "targetKind": node_kind_label(candidate.target_kind),
                "nliSupportMillis": candidate.nli_support_millis,
                "nliContradictionMillis": candidate.nli_contradiction_millis,
            }),
            data: None,
            document_id: None,
            narrative_id: None,
            layer: KernelGraphLayer::Candidate,
            temporal: Default::default(),
            provenance: KernelProvenance {
                resolver: Some("semantic-graph".to_owned()),
                source: Some(SEMANTIC_MODEL_ID.to_owned()),
                confidence: Some(candidate.score_millis as f64 / 1000.0),
                evidence_refs: candidate.evidence_refs.clone(),
            },
            resolution_facet: None,
        });
    }
    KernelMutationBatch {
        layer: KernelGraphLayer::Candidate,
        scope: KernelMutationScope::Candidate {
            scope_key: scope_key.to_owned(),
        },
        recorded_at: Some(created_at),
        vertices,
        edges,
    }
}

fn candidate_vertex(prototype: &Prototype, created_at: i64) -> KernelVertex {
    KernelVertex {
        id: KernelVertexId(prototype.node_id.clone()),
        kind: prototype.ann_kind.to_owned(),
        class: match prototype.node_kind {
            SemanticGraphNodeKind::Chunk => KernelVertexClass::Chunk,
            SemanticGraphNodeKind::Entity => KernelVertexClass::Entity,
            _ => KernelVertexClass::Generic,
        },
        labels: vec![prototype.text_key.clone()],
        weight: 1,
        value: json!({"textKey": prototype.text_key, "textHash": prototype.semantic_node.text_hash}),
        attributes: json!({"semanticCandidate": true}),
        temporal: phoenix_graph_kernel::KernelBiTemporal {
            recorded_at: Some(created_at),
            ..Default::default()
        },
        provenance: KernelProvenance {
            resolver: Some("semantic-graph".to_owned()),
            source: Some(SEMANTIC_MODEL_ID.to_owned()),
            confidence: Some(1.0),
            evidence_refs: prototype.evidence_refs.clone(),
        },
        entity_id: (prototype.node_kind == SemanticGraphNodeKind::Entity)
            .then(|| prototype.node_id.trim_start_matches("entity::").to_owned()),
        search_chunk_id: None,
        document_id: prototype.document_id.clone(),
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

pub(crate) fn summarize(
    prototypes: &[Prototype],
    candidates: &[SemanticGraphEdgeCandidate],
) -> SemanticGraphCompilerSummary {
    let mut summary = SemanticGraphCompilerSummary {
        node_count: prototypes.len(),
        ..Default::default()
    };
    for prototype in prototypes {
        *summary
            .node_kind_counts
            .entry(node_kind_label(prototype.node_kind).to_owned())
            .or_default() += 1;
    }
    for candidate in candidates {
        match candidate.candidate_status {
            SemanticCandidateStatus::ReviewedSupport => summary.reviewed_support_count += 1,
            SemanticCandidateStatus::ReviewedContradiction => {
                summary.reviewed_contradiction_count += 1
            }
            _ => {}
        }
        if candidate.candidate_status == SemanticCandidateStatus::Rejected {
            continue;
        }
        summary.edge_count += 1;
        *summary
            .edge_family_counts
            .entry(family_label(candidate.family).to_owned())
            .or_default() += 1;
    }
    summary
}

fn dedupe_key(family: SemanticEdgeFamily, source: &Prototype, target: &Prototype) -> String {
    match family {
        SemanticEdgeFamily::ChunkNeighbor
        | SemanticEdgeFamily::ClaimSupport
        | SemanticEdgeFamily::ClaimContradiction
        | SemanticEdgeFamily::StateSupport
        | SemanticEdgeFamily::StateContradiction
        | SemanticEdgeFamily::EventNeighbor => {
            let (left, right) = if source.node_id <= target.node_id {
                (&source.node_id, &target.node_id)
            } else {
                (&target.node_id, &source.node_id)
            };
            format!("{}|{}|{}", family_label(family), left, right)
        }
        _ => format!(
            "{}|{}|{}",
            family_label(family),
            source.node_id,
            target.node_id
        ),
    }
}

fn neighbor_score_millis(distance: f64) -> u32 {
    ((1.0 / (1.0 + distance.max(0.0))) * 1000.0)
        .round()
        .clamp(0.0, 1000.0) as u32
}

fn merge_refs(left: &[String], right: &[String]) -> Vec<String> {
    let mut merged = left.iter().chain(right.iter()).cloned().collect::<Vec<_>>();
    merged.sort();
    merged.dedup();
    merged
}
