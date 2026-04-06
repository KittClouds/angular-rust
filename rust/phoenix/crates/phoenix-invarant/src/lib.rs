use std::collections::{hash_map::DefaultHasher, BTreeMap};
use std::env;
use std::hash::{Hash, Hasher};
use std::time::Instant;

mod nlp;
mod native_scan;
mod native_project;
mod native_structure;
mod semantic;

use phoenix_graph::{
    GraphEdgeRecord, GraphLayer, GraphMutationBatch, GraphMutationScope, GraphVertexRecord,
};
use phoenix_graptor::BorrowedIngestDocument;
use phoenix_chunker::ChunkerConfig;
use phoenix_graptor::{
    load_session_state as load_legacy_session_state, BorrowedIngestRequest,
    NativeIngestArtifacts, PhoenixGraptor,
};
use phoenix_scanner::PhoenixScanner;
use phoenix_store_cozo::{PhoenixCozoStore, StoreError};
use phoenix_store_native::{PhoenixNativeRowStore, ScopedDefinitionFilter, ScopedDocumentFilter};
use phoenix_store_ruvector::PhoenixRuVectorStore;
use phoenix_structure::PhoenixStructure;
use phoenix_types::{
    BoundaryKind, ChunkStats, Diagnostic, DiscoverySummary, DocumentId, EntityId, EntitySummary,
    EntityKind, GraphSummary, IngestDocumentSummary, IngestResult, MentionEntityRef,
    MentionSource, NoteId, RelationCount, ResolverEntitySeed, ScanArtifact, ScopeKey, SessionId,
    SessionState, SessionStats, StructureArtifact,
};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use native_project::{
    legacy_native_scanner_enabled, project_native_document, relation_counts_from_rows,
    write_relation_rows,
};
use native_structure::{legacy_native_structure_enabled, NativeStructureBuilder};

pub use semantic::{
    AliasCandidate, CanonicalEntity, ChunkId, Claim, CoreferenceChainArtifact,
    DocumentAnalysisProfile, DocumentAnalysisStage, DocumentSemanticBundle, DocumentVersionId,
    Event, EvidenceAnchor, EvidenceId,
    MentionCandidate, NormalizedTextArtifact, ProposedEntityLink, Relation, ResolutionProvenance,
    ResolutionStatus, ResolvedMention, ScannedDocument, SemanticBundle, SemanticChunk,
    SemanticChunkKind, SpanId, StructuralKind, StructuralSpan, UnresolvedMention,
};

pub const LEGACY_DOCUMENT_NAMESPACE: &str = "graptor.documents";
pub const INVARANT_DOCUMENTS_NAMESPACE: &str = "invarant.documents";
pub const INVARANT_MANIFEST_NAMESPACE: &str = "invarant.manifest";
pub const INVARANT_SESSION_NAMESPACE: &str = "invarant.session";
pub const INVARANT_ARTIFACTS_NAMESPACE: &str = "invarant.artifacts";
pub const INVARANT_SEMANTIC_DOCUMENT_NAMESPACE: &str = "invarant.semantic.documents";
pub const INVARANT_SEMANTIC_ENTITY_NAMESPACE: &str = "invarant.semantic.entities";
pub const INVARANT_SEMANTIC_CLAIM_NAMESPACE: &str = "invarant.semantic.claims";
pub const INVARANT_SEMANTIC_EVENT_NAMESPACE: &str = "invarant.semantic.events";
pub const INVARANT_SEMANTIC_RELATION_NAMESPACE: &str = "invarant.semantic.relations";
pub const INVARANT_SEMANTIC_EVIDENCE_NAMESPACE: &str = "invarant.semantic.evidence";
pub const INVARANT_SEMANTIC_PROJECTION_NAMESPACE: &str = "invarant.semantic.projections";
pub const INVARANT_SEMANTIC_ANNOTATION_NAMESPACE: &str = "invarant.semantic.annotations";
pub const INVARANT_SEMANTIC_COREFERENCE_NAMESPACE: &str = "invarant.semantic.coreference";
pub const INVARANT_SEMANTIC_RESOLUTION_NAMESPACE: &str = "invarant.semantic.resolutions";
pub const INVARANT_SESSION_STATE_KIND: &str = "invarant_session_state";
pub const INVARANT_SESSION_STATS_KIND: &str = "invarant_session_stats";

pub trait InvarantStore {
    fn fetch_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError>;
    fn put_row(&self, relation: &str, row: Value) -> Result<(), StoreError>;
    fn put_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError>;
    fn as_legacy_cozo(&self) -> Option<&PhoenixCozoStore> {
        None
    }

    fn fetch_scoped_documents(
        &self,
        filter: ScopedDocumentFilter<'_>,
    ) -> Result<Vec<Value>, StoreError> {
        let mut rows = self.fetch_rows("scoped_documents")?;
        rows.retain(|row| phoenix_store_native::matches_scoped_document_filter(row, &filter));
        Ok(rows)
    }

    fn fetch_scoped_definitions(
        &self,
        filter: ScopedDefinitionFilter<'_>,
    ) -> Result<Vec<Value>, StoreError> {
        let mut rows = self.fetch_rows("scoped_definitions")?;
        rows.retain(|row| phoenix_store_native::matches_scoped_definition_filter(row, &filter));
        Ok(rows)
    }
}

impl InvarantStore for PhoenixCozoStore {
    fn fetch_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        PhoenixCozoStore::fetch_rows(self, relation)
    }

    fn put_row(&self, relation: &str, row: Value) -> Result<(), StoreError> {
        PhoenixCozoStore::put_row(self, relation, row)
    }

    fn put_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError> {
        PhoenixCozoStore::put_rows(self, relation, rows)
    }

    fn as_legacy_cozo(&self) -> Option<&PhoenixCozoStore> {
        Some(self)
    }
}

impl InvarantStore for PhoenixRuVectorStore {
    fn fetch_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        PhoenixNativeRowStore::fetch_rows(self, relation)
    }

    fn put_row(&self, relation: &str, row: Value) -> Result<(), StoreError> {
        PhoenixNativeRowStore::put_row(self, relation, row)
    }

    fn put_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError> {
        PhoenixNativeRowStore::put_rows(self, relation, rows)
    }

    fn fetch_scoped_documents(
        &self,
        filter: ScopedDocumentFilter<'_>,
    ) -> Result<Vec<Value>, StoreError> {
        PhoenixNativeRowStore::fetch_scoped_documents(self, filter)
    }

    fn fetch_scoped_definitions(
        &self,
        filter: ScopedDefinitionFilter<'_>,
    ) -> Result<Vec<Value>, StoreError> {
        PhoenixNativeRowStore::fetch_scoped_definitions(self, filter)
    }
}

impl InvarantStore for dyn PhoenixNativeRowStore + '_ {
    fn fetch_rows(&self, relation: &str) -> Result<Vec<Value>, StoreError> {
        PhoenixNativeRowStore::fetch_rows(self, relation)
    }

    fn put_row(&self, relation: &str, row: Value) -> Result<(), StoreError> {
        PhoenixNativeRowStore::put_row(self, relation, row)
    }

    fn put_rows(&self, relation: &str, rows: &[Value]) -> Result<(), StoreError> {
        PhoenixNativeRowStore::put_rows(self, relation, rows)
    }

    fn fetch_scoped_documents(
        &self,
        filter: ScopedDocumentFilter<'_>,
    ) -> Result<Vec<Value>, StoreError> {
        PhoenixNativeRowStore::fetch_scoped_documents(self, filter)
    }

    fn fetch_scoped_definitions(
        &self,
        filter: ScopedDefinitionFilter<'_>,
    ) -> Result<Vec<Value>, StoreError> {
        PhoenixNativeRowStore::fetch_scoped_definitions(self, filter)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalysisContext {
    pub session_id: Option<SessionId>,
    pub scope: ScopeKey,
    pub document_key: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionScratch {
    pub scan_invocations: u64,
    pub documents: BTreeMap<String, DocumentScratch>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentScratch {
    pub last_scan_mentions: usize,
    pub last_structure_relations: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceArtifact {
    pub document_id: String,
    pub title: String,
    pub text_len: usize,
    pub fingerprint: String,
    pub config_hash: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AnnotationArtifact {
    pub document_id: String,
    pub mention_count: usize,
    pub chunk_count: usize,
    pub resolver_link_count: usize,
    pub narrative_hit_count: usize,
    pub structure_relation_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NativeIngestProfile {
    pub document_count: usize,
    pub total_wall_ms: u64,
    pub stages: Vec<DocumentAnalysisStage>,
    pub documents: Vec<DocumentAnalysisProfile>,
    pub counters: BTreeMap<String, usize>,
}

#[derive(Default)]
struct NativeRelationRows {
    notes: Vec<Value>,
    docid_map: Vec<Value>,
    chunkid_map: Vec<Value>,
    chunks: Vec<Value>,
    document_boundaries: Vec<Value>,
    entities: Vec<Value>,
    spans: Vec<Value>,
    span_mentions: Vec<Value>,
    discovery_candidates: Vec<Value>,
    edges: Vec<Value>,
    scoped_documents: Vec<Value>,
    scoped_definitions: Vec<Value>,
}

impl NativeRelationRows {
    fn append(&mut self, mut other: NativeRelationRows) {
        self.notes.append(&mut other.notes);
        self.docid_map.append(&mut other.docid_map);
        self.chunkid_map.append(&mut other.chunkid_map);
        self.chunks.append(&mut other.chunks);
        self.document_boundaries.append(&mut other.document_boundaries);
        self.entities.append(&mut other.entities);
        self.spans.append(&mut other.spans);
        self.span_mentions.append(&mut other.span_mentions);
        self.discovery_candidates
            .append(&mut other.discovery_candidates);
        self.edges.append(&mut other.edges);
        self.scoped_documents.append(&mut other.scoped_documents);
        self.scoped_definitions.append(&mut other.scoped_definitions);
    }
}

struct NativeDocumentProjection {
    summary: IngestDocumentSummary,
    rows: NativeRelationRows,
    graph_batch: GraphMutationBatch,
    alias_count: usize,
    discovery_count: usize,
    mention_span_count: usize,
}

#[derive(Clone)]
struct NativeBoundary {
    boundary_id: i64,
    kind: &'static str,
    depth: i64,
    label: Option<String>,
    ordinal: i64,
    parent_boundary_id: Option<i64>,
    start: i64,
    end: i64,
}

#[derive(Clone)]
struct NativeChapter {
    chapter_id: u32,
    boundary_id: i64,
    boundary_ordinal: u32,
    title: String,
    start: usize,
    end: usize,
    chunk_id: i64,
}

#[derive(Clone)]
struct NativeLeaf {
    chunk_id: i64,
    search_id: String,
    chapter_id: u32,
    boundary_id: i64,
    boundary_ordinal: u32,
    parent_id: i64,
    start: usize,
    end: usize,
    text: String,
}

#[derive(Clone, Debug)]
pub struct InvarantConfig {
    pub chunk_size: usize,
    pub overlap: usize,
}

impl Default for InvarantConfig {
    fn default() -> Self {
        let chunker = ChunkerConfig::default();
        Self {
            chunk_size: chunker.chunk_size,
            overlap: chunker.overlap,
        }
    }
}

pub struct PhoenixInvarant {
    config: InvarantConfig,
    graptor: PhoenixGraptor,
}

impl PhoenixInvarant {
    pub fn new(config: InvarantConfig) -> Self {
        Self {
            config,
            graptor: PhoenixGraptor::default(),
        }
    }

    pub fn scan_parts(
        &self,
        text: &str,
        scope: &ScopeKey,
        resolver_seed: &[ResolverEntitySeed],
        scratch: Option<&mut SessionScratch>,
    ) -> ScanArtifact {
        let mut artifact = if legacy_native_scanner_enabled() {
            PhoenixScanner::default().scan_parts(text, scope, None, resolver_seed)
        } else {
            native_scan::NativeObservedScanner.scan_parts(text, scope, resolver_seed)
        };
        if let Some(scratch) = scratch {
            scratch.scan_invocations += 1;
        }
        artifact.diagnostics.push(Diagnostic {
            code: "PX_INVARANT_SCAN".to_owned(),
            message: "Invarant scanned this document with an explicit disposable analysis context."
                .to_owned(),
        });
        artifact
    }

    pub fn build_structure_parts(
        &self,
        text: &str,
        scan: &ScanArtifact,
        scratch: Option<&mut SessionScratch>,
        document_key: Option<&str>,
    ) -> StructureArtifact {
        let mut artifact = if legacy_native_structure_enabled() {
            PhoenixStructure::default().build_parts(text, scan)
        } else {
            NativeStructureBuilder.build_parts(text, scan)
        };
        if let (Some(scratch), Some(document_key)) = (scratch, document_key) {
            scratch.documents.insert(
                document_key.to_owned(),
                DocumentScratch {
                    last_scan_mentions: scan.mentions.len(),
                    last_structure_relations: artifact.relations.len(),
                },
            );
        }
        artifact.diagnostics.push(Diagnostic {
            code: "PX_INVARANT_STRUCTURE".to_owned(),
            message: if legacy_native_structure_enabled() {
                "Invarant assembled structure from a scanner artifact without shared session state."
                    .to_owned()
            } else {
                "Invarant assembled structure with the native grouped sentence builder instead of the legacy full-document structure pass."
                    .to_owned()
            },
        });
        artifact
    }

    pub fn ingest_native_view(
        &self,
        store: &dyn InvarantStore,
        request: &BorrowedIngestRequest<'_>,
    ) -> Result<
        (
            phoenix_types::IngestResult,
            NativeIngestArtifacts,
            NativeIngestProfile,
        ),
        StoreError,
    > {
        let ingest_started = Instant::now();
        if legacy_native_ingest_enabled() {
            emit_ingest_progress(
                "legacy_native_ingest_requested_but_disabled_for_non_cozo_native_store".to_owned(),
            );
        }
        let (mut ingest, artifacts, mut profile) = self.project_native_ingest(store, request)?;
        profile.total_wall_ms = ingest_started.elapsed().as_millis() as u64;
        profile.document_count = request.documents.len();
        ingest.diagnostics.push(Diagnostic {
            code: "PX_INVARANT_OK".to_owned(),
            message: format!(
                "Invarant completed native ingest through a single explicit staged analysis pipeline for {} documents.",
                request.documents.len()
            ),
        });
        ingest.diagnostics.push(Diagnostic {
            code: "PX_INVARANT_SEMANTIC_CORE".to_owned(),
            message: "Invarant projected native structural, retrieval, semantic, and graph assertion planes without paying the legacy Graptor ingest path."
                .to_owned(),
        });
        if legacy_native_ingest_enabled() {
            ingest.diagnostics.push(Diagnostic {
                code: "PX_INVARANT_LEGACY_DEBUG".to_owned(),
                message:
                    "Legacy Graptor native ingest ran in debug-fallback mode before the Invarant hot path for verification."
                        .to_owned(),
            });
        }
        ingest
            .diagnostics
            .extend(native_ingest_profile_diagnostics(&profile));
        Ok((ingest, artifacts, profile))
    }

    fn project_native_ingest(
        &self,
        store: &dyn InvarantStore,
        request: &BorrowedIngestRequest<'_>,
    ) -> Result<(IngestResult, NativeIngestArtifacts, NativeIngestProfile), StoreError> {
        let started = Instant::now();
        let mut scratch = SessionScratch::default();
        let config_hash = self.config_hash();
        let now = now_ms();
        let mut rows = NativeRelationRows::default();
        let resolver_seed_started = Instant::now();
        let mut resolver_seed_cache = collect_native_resolver_seeds(store, &ScopeKey::default())?;
        let mut stages = vec![completed_stage(
            "collect_resolver_seeds",
            resolver_seed_started,
            BTreeMap::from([("seedCount".to_owned(), resolver_seed_cache.len())]),
        )?];
        let mut document_profiles = Vec::new();
        let mut document_summaries = Vec::new();
        let mut graph_batches = Vec::new();
        let mut ingest_diagnostics = Vec::new();
        let mut total_aliases = 0usize;
        let mut total_discovery_mentions = 0usize;
        let mut total_mention_spans = 0usize;
        let mut total_graph_edges = 0usize;
        for document in request.documents {
            emit_ingest_progress(format!(
                "document={} stage=scan_prepare bytes={}",
                document.document_id.0,
                document.text.len()
            ));
            let document_started = Instant::now();
            let mut document_stages = Vec::new();
            let context = AnalysisContext {
                session_id: request.session_id.clone(),
                scope: document.scope.clone(),
                document_key: Some(document.document_id.0.clone()),
            };
            let mut resolver_seeds = resolver_seed_cache
                .iter()
                .filter(|seed| scope_matches(&seed.scope, &document.scope))
                .cloned()
                .collect::<Vec<_>>();
            document_stages.push(completed_stage(
                "document_scope_seed_filter",
                document_started,
                BTreeMap::from([("resolverSeedCount".to_owned(), resolver_seeds.len())]),
            )?);
            let scan_started = Instant::now();
            let scan = self.scan_parts(
                document.text,
                &document.scope,
                &resolver_seeds,
                Some(&mut scratch),
            );
            document_stages.push(completed_stage(
                "document_scan",
                scan_started,
                BTreeMap::from([
                    ("mentions".to_owned(), scan.mentions.len()),
                    ("sentences".to_owned(), scan.sentences.len()),
                    ("resolverLinks".to_owned(), scan.resolver_links.len()),
                    ("narrativeVerbHits".to_owned(), scan.narrative_hits.len()),
                ]),
            )?);
            let structure_started = Instant::now();
            let structure = self.build_structure_parts(
                document.text,
                &scan,
                Some(&mut scratch),
                Some(document.document_id.0.as_str()),
            );
            document_stages.push(completed_stage(
                "document_structure",
                structure_started,
                BTreeMap::from([("relations".to_owned(), structure.relations.len())]),
            )?);
            let analyze_started = Instant::now();
            let bundle = semantic::analyze_document(
                document,
                request.session_id.as_ref(),
                context.clone(),
                &self.config,
                &config_hash,
                &scan,
                &structure,
                None,
                &resolver_seeds,
            )?;
            document_stages.push(completed_stage(
                "document_semantic_analysis",
                analyze_started,
                BTreeMap::from([
                    ("structuralSpans".to_owned(), bundle.scanned_document.spans.len()),
                    ("leafChunks".to_owned(), bundle.leaf_chunks.len()),
                    ("windowChunks".to_owned(), bundle.window_chunks.len()),
                    (
                        "mentionCandidates".to_owned(),
                        bundle.annotation.mention_candidates.len(),
                    ),
                    (
                        "coreferenceChains".to_owned(),
                        bundle.annotation.coreference_chains.len(),
                    ),
                    (
                        "canonicalEntities".to_owned(),
                        bundle.resolution.canonical_entities.len(),
                    ),
                    ("claims".to_owned(), bundle.semantics.claims.len()),
                    ("events".to_owned(), bundle.semantics.events.len()),
                    ("relations".to_owned(), bundle.semantics.relations.len()),
                ]),
            )?);
            let row_projection_started = Instant::now();
            let projection = project_native_document(
                document,
                request.session_id.as_ref(),
                &context,
                &config_hash,
                &scan,
                &structure,
                &bundle,
                now,
            )?;
            document_stages.push(completed_stage(
                "document_row_projection",
                row_projection_started,
                BTreeMap::from([
                    (
                        "scopedDocumentsPending".to_owned(),
                        rows.scoped_documents.len() + projection.rows.scoped_documents.len(),
                    ),
                    (
                        "scopedDefinitionsPending".to_owned(),
                        rows.scoped_definitions.len() + projection.rows.scoped_definitions.len(),
                    ),
                    ("graphVertices".to_owned(), projection.graph_batch.vertices.len()),
                    ("graphEdges".to_owned(), projection.graph_batch.edges.len()),
                ]),
            )?);
            total_aliases += projection.alias_count;
            total_discovery_mentions += projection.discovery_count;
            total_mention_spans += projection.mention_span_count;
            total_graph_edges += projection.graph_batch.edges.len();
            document_summaries.push(projection.summary.clone());
            graph_batches.push(projection.graph_batch.clone());
            rows.append(projection.rows);
            ingest_diagnostics.extend(scan.diagnostics.clone());
            ingest_diagnostics.extend(structure.diagnostics.clone());
            ingest_diagnostics.extend(bundle.annotation.diagnostics.clone());
            ingest_diagnostics.extend(bundle.resolution.diagnostics.clone());
            ingest_diagnostics.extend(bundle.semantics.diagnostics.clone());
            resolver_seeds.extend(bundle.resolution.canonical_entities.iter().map(resolver_seed_from_entity));
            resolver_seed_cache.extend(bundle.resolution.canonical_entities.iter().map(resolver_seed_from_entity));
            let mut analysis_profile = bundle.analysis_profile.clone();
            let analysis_stages = std::mem::take(&mut analysis_profile.stages);
            document_stages.extend(analysis_stages);
            let total_wall_ms = document_started.elapsed().as_millis() as u64;
            let document_profile = DocumentAnalysisProfile {
                document_id: document.document_id.0.clone(),
                input_bytes: document.text.len(),
                total_wall_ms,
                stages: document_stages,
                counters: BTreeMap::from([
                    ("resolverSeedCount".to_owned(), resolver_seeds.len()),
                    ("structuralSpans".to_owned(), bundle.scanned_document.spans.len()),
                    ("leafChunks".to_owned(), bundle.leaf_chunks.len()),
                    (
                        "mentionCandidates".to_owned(),
                        bundle.annotation.mention_candidates.len(),
                    ),
                    (
                        "canonicalEntities".to_owned(),
                        bundle.resolution.canonical_entities.len(),
                    ),
                    ("claims".to_owned(), bundle.semantics.claims.len()),
                    ("events".to_owned(), bundle.semantics.events.len()),
                    ("relations".to_owned(), bundle.semantics.relations.len()),
                ]),
            };
            emit_ingest_progress(format!(
                "document={} stage=complete wall_ms={} spans={} leaf_chunks={} mentions={} canonical_entities={} claims={} events={} relations={}",
                document_profile.document_id,
                document_profile.total_wall_ms,
                document_profile.counters.get("structuralSpans").copied().unwrap_or_default(),
                document_profile.counters.get("leafChunks").copied().unwrap_or_default(),
                document_profile.counters.get("mentionCandidates").copied().unwrap_or_default(),
                document_profile.counters.get("canonicalEntities").copied().unwrap_or_default(),
                document_profile.counters.get("claims").copied().unwrap_or_default(),
                document_profile.counters.get("events").copied().unwrap_or_default(),
                document_profile.counters.get("relations").copied().unwrap_or_default(),
            ));
            document_profiles.push(document_profile);
        }
        write_relation_rows(store, "notes", &rows.notes, &mut stages)?;
        write_relation_rows(store, "docid_map", &rows.docid_map, &mut stages)?;
        write_relation_rows(store, "chunkid_map", &rows.chunkid_map, &mut stages)?;
        write_relation_rows(store, "chunks", &rows.chunks, &mut stages)?;
        write_relation_rows(
            store,
            "document_boundaries",
            &rows.document_boundaries,
            &mut stages,
        )?;
        write_relation_rows(store, "entities", &rows.entities, &mut stages)?;
        write_relation_rows(store, "spans", &rows.spans, &mut stages)?;
        write_relation_rows(store, "span_mentions", &rows.span_mentions, &mut stages)?;
        write_relation_rows(
            store,
            "discovery_candidates",
            &rows.discovery_candidates,
            &mut stages,
        )?;
        write_relation_rows(store, "edges", &rows.edges, &mut stages)?;
        write_relation_rows(store, "scoped_documents", &rows.scoped_documents, &mut stages)?;
        write_relation_rows(
            store,
            "scoped_definitions",
            &rows.scoped_definitions,
            &mut stages,
        )?;
        let chunk_stats = ChunkStats {
            documents: request.documents.len(),
            total_chapters: document_summaries.iter().map(|summary| summary.chapter_count).sum(),
            total_boundaries: document_summaries.iter().map(|summary| summary.boundary_count).sum(),
            total_parents: document_summaries.iter().map(|summary| summary.parent_count).sum(),
            total_leaves: document_summaries.iter().map(|summary| summary.leaf_count).sum(),
        };
        let graph_summary = GraphSummary {
            documents: request.documents.len(),
            total_chapters: chunk_stats.total_chapters,
            total_boundaries: chunk_stats.total_boundaries,
            total_leaves: chunk_stats.total_leaves,
            total_entities: document_summaries.iter().map(|summary| summary.entity_count).sum(),
            total_mentions: total_mention_spans,
            total_edges: total_graph_edges,
            cross_chapter_links: 0,
        };
        let entity_summary = EntitySummary {
            total_entities: graph_summary.total_entities,
            total_aliases: total_aliases,
            total_mentions: total_mention_spans,
            multi_chapter_entities: 0,
        };
        let discovery_summary = DiscoverySummary {
            candidate_count: rows.discovery_candidates.len(),
            mention_count: total_discovery_mentions,
            persisted_count: rows.discovery_candidates.len(),
        };
        let relation_counts = relation_counts_from_rows(&rows);
        let profile = NativeIngestProfile {
            document_count: request.documents.len(),
            total_wall_ms: started.elapsed().as_millis() as u64,
            stages,
            documents: document_profiles,
            counters: BTreeMap::from([
                ("notes".to_owned(), rows.notes.len()),
                ("chunkRows".to_owned(), rows.chunks.len()),
                ("definitionRows".to_owned(), rows.scoped_definitions.len()),
                ("graphBatchCount".to_owned(), graph_batches.len()),
            ]),
        };
        Ok((
            IngestResult {
                session_id: request.session_id.clone(),
                document_count: request.documents.len(),
                warning_count: 0,
                documents: document_summaries,
                chunk_stats: Some(chunk_stats),
                graph_summary: Some(graph_summary),
                entity_summary: Some(entity_summary),
                discovery_summary: Some(discovery_summary),
                retrieval_summary: None,
                relation_counts,
                diagnostics: ingest_diagnostics,
            },
            NativeIngestArtifacts { graph_batches },
            profile,
        ))
    }

    pub fn persist_session_materializations(
        &self,
        store: &dyn InvarantStore,
        session_id: &SessionId,
        graph_vertex_count: usize,
        graph_edge_count: usize,
        discovery_candidate_count: usize,
        span_count: usize,
    ) -> Result<(), StoreError> {
        if ingest_progress_enabled() {
            eprintln!("[invarant-session] phase=load_state session_id={}", session_id.0);
        }
        let state = load_session_state(store, session_id)?;
        if ingest_progress_enabled() {
            eprintln!(
                "[invarant-session] phase=build_stats session_id={} document_count={}",
                session_id.0,
                state.documents.len()
            );
        }
        let stats = build_session_stats(
            &state,
            graph_vertex_count,
            graph_edge_count,
            discovery_candidate_count,
            span_count,
        );
        let now = now_ms();
        if ingest_progress_enabled() {
            eprintln!("[invarant-session] phase=write_state session_id={}", session_id.0);
        }
        store.put_row(
            "scoped_definitions",
            session_definition_row(
                session_id,
                "state",
                &serde_json::to_value(&state)
                    .map_err(|error| StoreError::Query(error.to_string()))?,
                now,
            ),
        )?;
        if ingest_progress_enabled() {
            eprintln!("[invarant-session] phase=write_stats session_id={}", session_id.0);
        }
        store.put_row(
            "scoped_definitions",
            session_definition_row(
                session_id,
                "stats",
                &serde_json::to_value(&stats)
                    .map_err(|error| StoreError::Query(error.to_string()))?,
                now,
            ),
        )
    }

    fn config_hash(&self) -> String {
        stable_hex(
            "invarant_config",
            &[
                &self.config.chunk_size.to_string(),
                &self.config.overlap.to_string(),
            ],
        )
    }
}

impl Default for PhoenixInvarant {
    fn default() -> Self {
        Self::new(InvarantConfig::default())
    }
}

fn rewrite_invarant_diagnostics(diagnostics: &mut [Diagnostic]) {
    for diagnostic in diagnostics {
        if diagnostic.code.starts_with("PX_GRAPTOR") {
            diagnostic.code = diagnostic.code.replacen("PX_GRAPTOR", "PX_INVARANT", 1);
        }
        if diagnostic.message.contains("Graptor") {
            diagnostic.message = diagnostic.message.replace("Graptor", "Invarant");
        }
        if diagnostic.message.contains("graptor") {
            diagnostic.message = diagnostic.message.replace("graptor", "invarant");
        }
    }
}

fn scoped_definition_row(
    document: &BorrowedIngestDocument<'_>,
    stage: &str,
    payload: &Value,
    now: i64,
) -> Value {
    json!({
        "id": stable_hex("invarant_definition", &[document.document_id.0.as_str(), stage]),
        "narrative_id": document.scope.narrative_id.clone().unwrap_or_else(|| "__global__".to_owned()),
        "namespace": INVARANT_ARTIFACTS_NAMESPACE,
        "definition_key": format!("{stage}:{}", document.document_id.0),
        "payload": payload,
        "created_at": now,
        "updated_at": now,
    })
}

fn source_payload(
    document: &BorrowedIngestDocument<'_>,
    context: &AnalysisContext,
    bundle: &DocumentSemanticBundle,
) -> Value {
    let artifact = SourceArtifact {
        document_id: document.document_id.0.clone(),
        title: document.title.to_owned(),
        text_len: document.text.len(),
        fingerprint: bundle.source_fingerprint.clone(),
        config_hash: bundle.config_hash.clone(),
    };
    json!({
        "stage": "source",
        "documentVersionId": bundle.document_version_id.0,
        "sessionId": context.session_id.as_ref().map(|value| value.0.clone()),
        "scope": context.scope.clone(),
        "artifact": artifact,
    })
}

fn scan_payload(
    document: &BorrowedIngestDocument<'_>,
    context: &AnalysisContext,
    scan: &ScanArtifact,
    config_hash: &str,
) -> Value {
    json!({
        "stage": "scan",
        "documentId": document.document_id.0.clone(),
        "sessionId": context.session_id.as_ref().map(|value| value.0.clone()),
        "scope": context.scope.clone(),
        "configHash": config_hash,
        "sentences": scan.sentences.len(),
        "tokens": scan.tokens.len(),
        "mentions": scan.mentions.len(),
        "resolverLinks": scan.resolver_links.len(),
        "narrativeHits": scan.narrative_hits.len(),
        "diagnostics": scan.diagnostics.clone(),
    })
}

fn structure_payload(
    document: &BorrowedIngestDocument<'_>,
    context: &AnalysisContext,
    structure: &StructureArtifact,
    bundle: &DocumentSemanticBundle,
) -> Value {
    json!({
        "stage": "structure",
        "documentId": document.document_id.0.clone(),
        "documentVersionId": bundle.document_version_id.0,
        "sessionId": context.session_id.as_ref().map(|value| value.0.clone()),
        "scope": context.scope.clone(),
        "configHash": bundle.config_hash,
        "sentenceFrames": structure.sentence_frames.len(),
        "relations": structure.relations.len(),
        "evidenceSpans": structure.evidence_spans.len(),
        "structuralSpanCount": bundle.scanned_document.spans.len(),
        "diagnostics": structure.diagnostics.clone(),
    })
}

fn segmentation_payload(
    document: &BorrowedIngestDocument<'_>,
    context: &AnalysisContext,
    bundle: &DocumentSemanticBundle,
) -> Value {
    json!({
        "stage": "segmentation",
        "documentId": document.document_id.0.clone(),
        "documentVersionId": bundle.document_version_id.0,
        "sessionId": context.session_id.as_ref().map(|value| value.0.clone()),
        "scope": context.scope.clone(),
        "configHash": bundle.config_hash,
        "leafChunkCount": bundle.leaf_chunks.len(),
        "windowChunkCount": bundle.window_chunks.len(),
        "leafByteSpan": bundle
            .leaf_chunks
            .iter()
            .map(|chunk| chunk.range.end.saturating_sub(chunk.range.start) as usize)
            .sum::<usize>(),
    })
}

fn annotation_payload(
    document: &BorrowedIngestDocument<'_>,
    context: &AnalysisContext,
    bundle: &DocumentSemanticBundle,
) -> Value {
    let artifact = AnnotationArtifact {
        document_id: document.document_id.0.clone(),
        mention_count: bundle.annotation.mention_candidates.len(),
        chunk_count: bundle.leaf_chunks.len(),
        resolver_link_count: bundle.annotation.alias_candidates.len(),
        narrative_hit_count: bundle.annotation.time_candidates.len(),
        structure_relation_count: bundle.annotation.relation_cues.len(),
    };
    json!({
        "stage": "annotation",
        "documentVersionId": bundle.document_version_id.0,
        "sessionId": context.session_id.as_ref().map(|value| value.0.clone()),
        "scope": context.scope.clone(),
        "configHash": bundle.config_hash,
        "artifact": artifact,
        "normalizedText": {
            "provider": bundle.annotation.normalized_text.provider,
            "providerVersion": bundle.annotation.normalized_text.provider_version,
            "configHash": bundle.annotation.normalized_text.config_hash,
            "normalizedLength": bundle.annotation.normalized_text.normalized_text.len(),
            "foldedLength": bundle.annotation.normalized_text.folded_text.len(),
        },
        "keyTermCount": bundle.annotation.key_term_candidates.len(),
        "timeCount": bundle.annotation.time_candidates.len(),
        "coreferenceChainCount": bundle.annotation.coreference_chains.len(),
        "relationCueCount": bundle.annotation.relation_cues.len(),
    })
}

fn resolution_payload(
    document: &BorrowedIngestDocument<'_>,
    context: &AnalysisContext,
    bundle: &DocumentSemanticBundle,
) -> Value {
    json!({
        "stage": "resolution",
        "documentId": document.document_id.0.clone(),
        "documentVersionId": bundle.document_version_id.0,
        "sessionId": context.session_id.as_ref().map(|value| value.0.clone()),
        "scope": context.scope.clone(),
        "configHash": bundle.config_hash,
        "canonicalEntityCount": bundle.resolution.canonical_entities.len(),
        "resolvedMentionCount": bundle.resolution.resolved_mentions.len(),
        "proposedLinkCount": bundle.resolution.proposed_links.len(),
        "unresolvedMentionCount": bundle.resolution.unresolved_mentions.len(),
        "coreferenceChainCount": bundle.annotation.coreference_chains.len(),
    })
}

fn semantic_payload(
    document: &BorrowedIngestDocument<'_>,
    context: &AnalysisContext,
    bundle: &DocumentSemanticBundle,
) -> Value {
    json!({
        "stage": "semantic",
        "documentId": document.document_id.0.clone(),
        "documentVersionId": bundle.document_version_id.0,
        "sessionId": context.session_id.as_ref().map(|value| value.0.clone()),
        "scope": context.scope.clone(),
        "configHash": bundle.config_hash,
        "projection": bundle.semantics.projection,
        "claimCount": bundle.semantics.claims.len(),
        "eventCount": bundle.semantics.events.len(),
        "relationCount": bundle.semantics.relations.len(),
        "evidenceCount": bundle.semantics.evidence_anchors.len(),
    })
}

fn semantic_document_row(
    document: &BorrowedIngestDocument<'_>,
    bundle: &DocumentSemanticBundle,
    now: i64,
) -> Value {
    json!({
        "id": stable_hex("invarant_semantic_document", &[document.document_id.0.as_str()]),
        "scope_folder_id": document.scope.folder_id.clone().unwrap_or_else(|| "__root__".to_owned()),
        "narrative_id": document.scope.narrative_id.clone().unwrap_or_else(|| "__global__".to_owned()),
        "namespace": INVARANT_SEMANTIC_DOCUMENT_NAMESPACE,
        "document_key": document.document_id.0.clone(),
        "payload": {
            "documentId": document.document_id.0.clone(),
            "documentVersionId": bundle.document_version_id.0,
            "noteId": document.note_id.as_ref().map(|value| value.0.clone()),
            "sessionId": bundle.context.session_id.as_ref().map(|value| value.0.clone()),
            "scope": bundle.context.scope,
            "summary": bundle.session_document,
            "semantic": {
                "canonicalEntityCount": bundle.resolution.canonical_entities.len(),
                "claimCount": bundle.semantics.claims.len(),
                "eventCount": bundle.semantics.events.len(),
                "relationCount": bundle.semantics.relations.len(),
                "evidenceCount": bundle.semantics.evidence_anchors.len(),
            },
        },
        "seeded_from_scope_folder_id": document.scope.folder_id.clone(),
        "created_at": now,
        "updated_at": now,
    })
}

fn semantic_object_rows(
    document: &BorrowedIngestDocument<'_>,
    bundle: &DocumentSemanticBundle,
    now: i64,
) -> Vec<Value> {
    let mut rows = Vec::with_capacity(8);
    rows.push(semantic_definition_row(
        document,
        INVARANT_SEMANTIC_ENTITY_NAMESPACE,
        &format!("entity-bundle:{}", bundle.document_version_id.0),
        &json!({
            "documentId": document.document_id.0,
            "documentVersionId": bundle.document_version_id.0,
            "entities": &bundle.resolution.canonical_entities,
        }),
        now,
    ));
    rows.push(semantic_definition_row(
        document,
        INVARANT_SEMANTIC_ANNOTATION_NAMESPACE,
        &format!("annotation-bundle:{}", bundle.document_version_id.0),
        &json!({
            "documentId": document.document_id.0,
            "documentVersionId": bundle.document_version_id.0,
            "mentionCount": bundle.annotation.mention_candidates.len(),
            "aliasCount": bundle.annotation.alias_candidates.len(),
            "keyTermCount": bundle.annotation.key_term_candidates.len(),
            "timeCount": bundle.annotation.time_candidates.len(),
            "coreferenceChainCount": bundle.annotation.coreference_chains.len(),
            "relationCueCount": bundle.annotation.relation_cues.len(),
        }),
        now,
    ));
    rows.push(semantic_definition_row(
        document,
        INVARANT_SEMANTIC_COREFERENCE_NAMESPACE,
        &format!("coref-bundle:{}", bundle.document_version_id.0),
        &json!({
            "documentId": document.document_id.0,
            "documentVersionId": bundle.document_version_id.0,
            "chains": &bundle.annotation.coreference_chains,
        }),
        now,
    ));
    rows.push(semantic_definition_row(
        document,
        INVARANT_SEMANTIC_RESOLUTION_NAMESPACE,
        &format!("resolution-bundle:{}", bundle.document_version_id.0),
        &json!({
            "documentId": document.document_id.0,
            "documentVersionId": bundle.document_version_id.0,
            "resolvedMentions": &bundle.resolution.resolved_mentions,
            "proposedLinks": &bundle.resolution.proposed_links,
            "unresolvedMentions": &bundle.resolution.unresolved_mentions,
        }),
        now,
    ));
    rows.push(semantic_definition_row(
        document,
        INVARANT_SEMANTIC_CLAIM_NAMESPACE,
        &format!("claim-bundle:{}", bundle.document_version_id.0),
        &json!({
            "documentId": document.document_id.0,
            "documentVersionId": bundle.document_version_id.0,
            "claims": &bundle.semantics.claims,
        }),
        now,
    ));
    rows.push(semantic_definition_row(
        document,
        INVARANT_SEMANTIC_EVENT_NAMESPACE,
        &format!("event-bundle:{}", bundle.document_version_id.0),
        &json!({
            "documentId": document.document_id.0,
            "documentVersionId": bundle.document_version_id.0,
            "events": &bundle.semantics.events,
        }),
        now,
    ));
    rows.push(semantic_definition_row(
        document,
        INVARANT_SEMANTIC_RELATION_NAMESPACE,
        &format!("relation-bundle:{}", bundle.document_version_id.0),
        &json!({
            "documentId": document.document_id.0,
            "documentVersionId": bundle.document_version_id.0,
            "relations": &bundle.semantics.relations,
        }),
        now,
    ));
    rows.push(semantic_definition_row(
        document,
        INVARANT_SEMANTIC_EVIDENCE_NAMESPACE,
        &format!("evidence-bundle:{}", bundle.document_version_id.0),
        &json!({
            "documentId": document.document_id.0,
            "documentVersionId": bundle.document_version_id.0,
            "evidenceAnchors": &bundle.semantics.evidence_anchors,
        }),
        now,
    ));
    rows.push(semantic_definition_row(
        document,
        INVARANT_SEMANTIC_PROJECTION_NAMESPACE,
        &format!("projection:{}", bundle.document_version_id.0),
        &serde_json::to_value(&bundle.semantics.projection).unwrap_or(Value::Null),
        now,
    ));
    rows
}

fn semantic_definition_row(
    document: &BorrowedIngestDocument<'_>,
    namespace: &str,
    definition_key: &str,
    payload: &Value,
    now: i64,
) -> Value {
    json!({
        "id": stable_hex("invarant_semantic_definition", &[namespace, definition_key]),
        "narrative_id": document.scope.narrative_id.clone().unwrap_or_else(|| "__global__".to_owned()),
        "namespace": namespace,
        "definition_key": definition_key,
        "payload": payload,
        "created_at": now,
        "updated_at": now,
    })
}

fn session_definition_row(session_id: &SessionId, kind: &str, payload: &Value, now: i64) -> Value {
    json!({
        "id": stable_hex("session_definition", &[session_id.0.as_str(), kind]),
        "narrative_id": "__global__",
        "namespace": INVARANT_SESSION_NAMESPACE,
        "definition_key": format!("{kind}:{}", session_id.0),
        "payload": payload,
        "created_at": now,
        "updated_at": now,
    })
}

pub fn load_session_state(
    store: &dyn InvarantStore,
    session_id: &SessionId,
) -> Result<SessionState, StoreError> {
    let state_key = format!("state:{}", session_id.0);
    let rows = store.fetch_scoped_definitions(ScopedDefinitionFilter {
        namespace: Some(INVARANT_SESSION_NAMESPACE),
        definition_key: Some(state_key.as_str()),
        ..ScopedDefinitionFilter::default()
    })?;
    if let Some(payload) = rows.iter().find_map(|row| {
        (row.get("namespace").and_then(Value::as_str) == Some(INVARANT_SESSION_NAMESPACE)
            && row.get("definition_key").and_then(Value::as_str)
                == Some(state_key.as_str()))
        .then(|| row.get("payload").cloned())
        .flatten()
    }) {
        if let Ok(state) = serde_json::from_value::<SessionState>(payload) {
            return Ok(state);
        }
    }

    let documents = store
        .fetch_scoped_documents(ScopedDocumentFilter {
            namespace: Some(INVARANT_SEMANTIC_DOCUMENT_NAMESPACE),
            ..ScopedDocumentFilter::default()
        })?
        .into_iter()
        .filter_map(|row| row.get("payload").cloned())
        .filter(|payload| {
            payload.get("sessionId").and_then(Value::as_str) == Some(session_id.0.as_str())
        })
        .filter_map(|payload| {
            serde_json::from_value::<phoenix_types::SessionDocumentState>(
                payload.get("summary")?.clone(),
            )
            .ok()
        })
        .collect::<Vec<_>>();

    if documents.is_empty() {
        if let Some(store) = store.as_legacy_cozo() {
            return load_legacy_session_state(store, session_id);
        }
        return Ok(SessionState {
            session_id: session_id.clone(),
            documents: Vec::new(),
            manifest_namespaces: vec![
                INVARANT_SESSION_NAMESPACE.to_owned(),
                INVARANT_SEMANTIC_DOCUMENT_NAMESPACE.to_owned(),
                INVARANT_ARTIFACTS_NAMESPACE.to_owned(),
            ],
            updated_at: now_ms(),
        });
    }

    Ok(SessionState {
        session_id: session_id.clone(),
        documents,
        manifest_namespaces: vec![
            INVARANT_SESSION_NAMESPACE.to_owned(),
            INVARANT_SEMANTIC_DOCUMENT_NAMESPACE.to_owned(),
            INVARANT_ARTIFACTS_NAMESPACE.to_owned(),
        ],
        updated_at: now_ms(),
    })
}

pub fn build_session_stats(
    state: &SessionState,
    graph_vertex_count: usize,
    graph_edge_count: usize,
    discovery_candidate_count: usize,
    span_count: usize,
) -> SessionStats {
    SessionStats {
        session_id: state.session_id.clone(),
        document_count: state.documents.len(),
        chapter_count: state
            .documents
            .iter()
            .map(|document| document.chapter_count)
            .sum(),
        boundary_count: state
            .documents
            .iter()
            .map(|document| document.boundary_count)
            .sum(),
        parent_count: state
            .documents
            .iter()
            .map(|document| document.parent_count)
            .sum(),
        leaf_count: state
            .documents
            .iter()
            .map(|document| document.leaf_count)
            .sum(),
        entity_count: state
            .documents
            .iter()
            .map(|document| document.entity_count)
            .sum(),
        discovery_candidate_count,
        graph_vertex_count,
        graph_edge_count,
        span_count,
        updated_at: now_ms(),
    }
}

fn collect_native_resolver_seeds(
    store: &dyn InvarantStore,
    scope: &ScopeKey,
) -> Result<Vec<ResolverEntitySeed>, StoreError> {
    let mut seeds = Vec::new();
    for row in store.fetch_scoped_definitions(ScopedDefinitionFilter {
        namespace: Some(INVARANT_SEMANTIC_ENTITY_NAMESPACE),
        ..ScopedDefinitionFilter::default()
    })? {
        for entity in deserialize_scoped_payloads::<CanonicalEntity>(
            &row,
            INVARANT_SEMANTIC_ENTITY_NAMESPACE,
            "entities",
        ) {
            if scope_matches(&entity.scope, scope) {
                seeds.push(resolver_seed_from_entity(&entity));
            }
        }
    }
    Ok(seeds)
}

fn resolver_seed_from_entity(entity: &CanonicalEntity) -> ResolverEntitySeed {
    ResolverEntitySeed {
        entity_id: entity.entity_id.clone(),
        canonical_name: entity.label.clone(),
        aliases: entity.aliases.clone(),
        kind: entity.kind.clone(),
        gender: None,
        number: None,
        scope: entity.scope.clone(),
    }
}

fn deserialize_scoped_payloads<T>(row: &Value, namespace: &str, array_field: &str) -> Vec<T>
where
    T: for<'de> Deserialize<'de>,
{
    if row.get("namespace").and_then(Value::as_str) != Some(namespace) {
        return Vec::new();
    }
    let Some(payload) = row.get("payload").cloned() else {
        return Vec::new();
    };
    if let Ok(item) = serde_json::from_value::<T>(payload.clone()) {
        return vec![item];
    }
    payload
        .get(array_field)
        .cloned()
        .and_then(|value| serde_json::from_value::<Vec<T>>(value).ok())
        .unwrap_or_default()
}

fn scope_matches(left: &ScopeKey, right: &ScopeKey) -> bool {
    fn field_matches(left: &Option<String>, right: &Option<String>) -> bool {
        match (left.as_deref(), right.as_deref()) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
    }

    field_matches(&left.world_id, &right.world_id)
        && field_matches(&left.narrative_id, &right.narrative_id)
        && field_matches(&left.folder_id, &right.folder_id)
        && field_matches(&left.folder_path, &right.folder_path)
}

fn stable_hex(kind: &str, parts: &[&str]) -> String {
    let mut hasher = DefaultHasher::new();
    kind.hash(&mut hasher);
    for part in parts {
        part.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

pub(crate) fn completed_stage(
    name: &str,
    started: Instant,
    counters: BTreeMap<String, usize>,
) -> Result<DocumentAnalysisStage, StoreError> {
    let stage = DocumentAnalysisStage {
        name: name.to_owned(),
        wall_ms: started.elapsed().as_millis() as u64,
        counters,
    };
    emit_ingest_progress(format!(
        "stage={} wall_ms={} counters={}",
        stage.name,
        stage.wall_ms,
        serde_json::to_string(&stage.counters).unwrap_or_else(|_| "{}".to_owned())
    ));
    maybe_timeout_ingest_stage("native-ingest", &stage)?;
    Ok(stage)
}

pub(crate) fn maybe_timeout_ingest_stage(
    scope_label: &str,
    stage: &DocumentAnalysisStage,
) -> Result<(), StoreError> {
    let Some(timeout_ms) = ingest_stage_timeout_ms() else {
        return Ok(());
    };
    if stage.wall_ms > timeout_ms {
        return Err(StoreError::Query(format!(
            "native ingest stage timeout in {scope_label}: {} took {}ms > {}ms",
            stage.name, stage.wall_ms, timeout_ms
        )));
    }
    Ok(())
}

pub(crate) fn emit_ingest_progress(message: String) {
    if ingest_progress_enabled() {
        eprintln!("[invarant-ingest] {message}");
    }
}

fn ingest_progress_enabled() -> bool {
    matches!(
        env::var("PHOENIX_INGEST_PROGRESS").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    ) || matches!(
        env::var("PHOENIX_PERF_PROGRESS").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn legacy_native_ingest_enabled() -> bool {
    matches!(
        env::var("PHOENIX_INVARANT_USE_LEGACY_NATIVE_INGEST")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn ingest_stage_timeout_ms() -> Option<u64> {
    env::var("PHOENIX_INGEST_STAGE_TIMEOUT_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .filter(|value| *value > 0)
}

fn native_ingest_profile_diagnostics(profile: &NativeIngestProfile) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    diagnostics.push(Diagnostic {
        code: "PX_INVARANT_PROFILE_TOTAL".to_owned(),
        message: format!(
            "Invarant native semantic persistence took {}ms for {} document(s); outer stages: {}.",
            profile.total_wall_ms,
            profile.document_count,
            summarize_stage_timings(&profile.stages)
        ),
    });
    for document in &profile.documents {
        diagnostics.push(Diagnostic {
            code: "PX_INVARANT_PROFILE_DOCUMENT".to_owned(),
            message: format!(
                "Invarant document {} ({} bytes) completed in {}ms; stages: {}.",
                document.document_id,
                document.input_bytes,
                document.total_wall_ms,
                summarize_stage_timings(&document.stages)
            ),
        });
    }
    diagnostics
}

fn summarize_stage_timings(stages: &[DocumentAnalysisStage]) -> String {
    stages
        .iter()
        .map(|stage| format!("{}={}ms", stage.name, stage.wall_ms))
        .collect::<Vec<_>>()
        .join(", ")
}

fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_store_cozo::StoreConfig;
    use phoenix_types::DocumentId;

    #[test]
    fn invarant_scan_uses_local_state() {
        let mesh = PhoenixInvarant::default();
        let mut scratch = SessionScratch::default();
        let first = mesh.scan_parts(
            "Ryan waited at dawn.",
            &ScopeKey::default(),
            &[],
            Some(&mut scratch),
        );
        let second = mesh.scan_parts(
            "He waited alone.",
            &ScopeKey::default(),
            &[],
            Some(&mut scratch),
        );
        assert!(first
            .diagnostics
            .iter()
            .any(|diag| diag.code == "PX_INVARANT_SCAN"));
        assert!(!second
            .resolver_links
            .iter()
            .any(|link| link.target_entity.is_some()));
        assert_eq!(scratch.scan_invocations, 2);
    }

    #[test]
    fn invarant_native_ingest_persists_semantic_planes() {
        let store = PhoenixCozoStore::open(StoreConfig::default()).expect("store");
        let mesh = PhoenixInvarant::default();
        let documents = vec![BorrowedIngestDocument {
            document_id: DocumentId("doc-semantic".to_owned()),
            note_id: None,
            title: "Harbor",
            text: "# Chapter 1\nRyan crossed the harbor.\nLen answered Ryan.\n",
            scope: ScopeKey::default(),
        }];

        let (ingest, _, profile) = mesh
            .ingest_native_view(
                &store,
                &BorrowedIngestRequest {
                    session_id: Some(SessionId("session-semantic".to_owned())),
                    documents: &documents,
                },
            )
            .expect("ingest");

        assert_eq!(ingest.document_count, 1);
        assert_eq!(profile.document_count, 1);
        assert!(!profile.documents.is_empty());
        let scoped_documents = store
            .fetch_rows("scoped_documents")
            .expect("scoped documents");
        assert!(scoped_documents.iter().any(|row| {
            row.get("namespace").and_then(Value::as_str)
                == Some(INVARANT_SEMANTIC_DOCUMENT_NAMESPACE)
        }));
        let definitions = store
            .fetch_rows("scoped_definitions")
            .expect("scoped definitions");
        assert!(definitions.iter().any(|row| {
            row.get("namespace").and_then(Value::as_str) == Some(INVARANT_SEMANTIC_ENTITY_NAMESPACE)
        }));
        assert!(definitions.iter().any(|row| {
            row.get("namespace").and_then(Value::as_str)
                == Some(INVARANT_SEMANTIC_ANNOTATION_NAMESPACE)
        }));
        assert!(definitions.iter().any(|row| {
            row.get("namespace").and_then(Value::as_str)
                == Some(INVARANT_SEMANTIC_RESOLUTION_NAMESPACE)
        }));
        let state = load_session_state(&store, &SessionId("session-semantic".to_owned()))
            .expect("session state");
        assert_eq!(state.documents.len(), 1);
    }
}
