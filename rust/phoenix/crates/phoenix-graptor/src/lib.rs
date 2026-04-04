use std::borrow::Cow;
use std::cmp::{max, min};
use std::collections::VecDeque;

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use phoenix_alex::{normalize_raw, split_sentence_ranges};
use phoenix_graph::{
    GraphCounts, GraphEdgeRecord, GraphLayer, GraphMutationBatch, GraphMutationScope,
    GraphVertexRecord,
};
pub use phoenix_graph::{GraptorEdge, GraptorGraph, GraptorVertex};
use phoenix_scanner::PhoenixScanner;
use phoenix_store_cozo::{
    CompactRelationBuffer, CompactRow, CompactRowView, PhoenixCozoStore, StoreError,
};
use phoenix_structure::PhoenixStructure;
use phoenix_types::{
    BoundaryDetectionStrategy, BoundaryKind, ChunkStats, Diagnostic, DiscoverySummary, DocumentId,
    EntityId, EntityKind, EntitySummary, EvidenceSpan, FrameSlot, GenderHint, GraphDeltaChunk,
    GraphDeltaEdge, GraphDeltaNode, GraphDeltaRequest, GraphDeltaResult, GraphSummary,
    IngestDocument, IngestDocumentSummary, IngestRequest, IngestResult, MentionEntityRef,
    MentionSource, NoteId, RelationCandidate, ResolverEntitySeed, ResolverLinkKind,
    RetrievalSummary, ScopeKey, SessionDocumentState, SessionId, SessionState, SessionStats,
    TextRange,
};
use rustc_hash::{FxHashMap, FxHashSet};
use serde_json::{json, Map, Value};

const DEFAULT_CHAPTER_KEYWORDS: &[&str] = &[
    "chapter",
    "part",
    "section",
    "introduction",
    "conclusion",
    "summary",
    "appendix",
    "#",
];
const LEGACY_DOCUMENT_NAMESPACE: &str = "graptor.documents";
const INVARANT_DOCUMENT_NAMESPACE: &str = "invarant.documents";
const LEGACY_MANIFEST_NAMESPACE: &str = "graptor.manifest";
const INVARANT_MANIFEST_NAMESPACE: &str = "invarant.manifest";
const LEGACY_SESSION_NAMESPACE: &str = "graptor.session";
const INVARANT_SESSION_NAMESPACE: &str = "invarant.session";
const LEGACY_SESSION_STATE_KIND: &str = "graptor_session_state";
const LEGACY_SESSION_STATS_KIND: &str = "graptor_session_stats";
const INVARANT_SESSION_STATE_KIND: &str = "invarant_session_state";
const INVARANT_SESSION_STATS_KIND: &str = "invarant_session_stats";

#[derive(Clone, Debug)]
pub struct GraptorConfig {
    pub chunk_size: usize,
    pub overlap: usize,
    pub parent_chunk_size: usize,
    pub parent_overlap: usize,
    pub boundary_detection: BoundaryDetectionStrategy,
}

impl Default for GraptorConfig {
    fn default() -> Self {
        Self {
            chunk_size: 500,
            overlap: 100,
            parent_chunk_size: 2_000,
            parent_overlap: 500,
            boundary_detection: BoundaryDetectionStrategy::Both {
                keywords: DEFAULT_CHAPTER_KEYWORDS
                    .iter()
                    .map(|keyword| keyword.to_string())
                    .collect(),
                max_depth: 6,
            },
        }
    }
}

impl GraptorConfig {
    pub fn without_chapter_detection(mut self) -> Self {
        self.boundary_detection = BoundaryDetectionStrategy::Disabled;
        self
    }

    pub fn without_boundary_detection(mut self) -> Self {
        self.boundary_detection = BoundaryDetectionStrategy::Disabled;
        self
    }
}

pub struct PhoenixGraptor {
    config: GraptorConfig,
    boundary_matcher: Option<DoubleArrayAhoCorasick>,
    max_heading_depth: u8,
}

#[derive(Clone, Debug)]
pub struct BorrowedIngestDocument<'a> {
    pub document_id: DocumentId,
    pub note_id: Option<NoteId>,
    pub title: &'a str,
    pub text: &'a str,
    pub scope: ScopeKey,
}

#[derive(Clone, Debug)]
pub struct BorrowedIngestRequest<'a> {
    pub session_id: Option<SessionId>,
    pub documents: &'a [BorrowedIngestDocument<'a>],
}

#[derive(Clone, Debug)]
pub struct BorrowedThreadMessage<'a> {
    pub message_id: &'a str,
    pub role: &'a str,
    pub content: &'a str,
    pub created_at: i64,
}

#[derive(Clone, Debug)]
pub struct BorrowedIngestThread<'a> {
    pub document_id: DocumentId,
    pub title: &'a str,
    pub messages: &'a [BorrowedThreadMessage<'a>],
    pub scope: ScopeKey,
}

impl<'a> From<&'a IngestDocument> for BorrowedIngestDocument<'a> {
    fn from(value: &'a IngestDocument) -> Self {
        Self {
            document_id: value.document_id.clone(),
            note_id: value.note_id.clone(),
            title: &value.title,
            text: &value.text,
            scope: value.scope.clone(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct NativeIngestArtifacts {
    pub graph_batches: Vec<GraphMutationBatch>,
}

impl PhoenixGraptor {
    pub fn new(config: GraptorConfig) -> Self {
        let (keywords, max_heading_depth) = boundary_strategy_parts(&config.boundary_detection);
        let boundary_matcher = if keywords.is_empty() {
            None
        } else {
            DoubleArrayAhoCorasickBuilder::new()
                .match_kind(MatchKind::LeftmostLongest)
                .build_with_values(
                    keywords
                        .iter()
                        .enumerate()
                        .map(|(index, keyword)| (keyword.as_bytes(), index as u32)),
                )
                .ok()
        };
        Self {
            config,
            boundary_matcher,
            max_heading_depth,
        }
    }

    pub fn ingest(
        &self,
        store: &PhoenixCozoStore,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        request: &IngestRequest,
    ) -> Result<IngestResult, StoreError> {
        let documents = request
            .documents
            .iter()
            .map(|document| BorrowedIngestDocument {
                document_id: document.document_id.clone(),
                note_id: document.note_id.clone(),
                title: &document.title,
                text: &document.text,
                scope: document.scope.clone(),
            })
            .collect::<Vec<_>>();
        self.ingest_view(
            store,
            scanner,
            structure,
            &BorrowedIngestRequest {
                session_id: request.session_id.clone(),
                documents: &documents,
            },
        )
    }

    pub fn ingest_view(
        &self,
        store: &PhoenixCozoStore,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        request: &BorrowedIngestRequest<'_>,
    ) -> Result<IngestResult, StoreError> {
        Ok(self
            .ingest_view_internal(store, scanner, structure, request, true)?
            .0)
    }

    pub fn ingest_native_view(
        &self,
        store: &PhoenixCozoStore,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        request: &BorrowedIngestRequest<'_>,
    ) -> Result<(IngestResult, NativeIngestArtifacts), StoreError> {
        self.ingest_view_internal(store, scanner, structure, request, false)
    }

    fn ingest_view_internal(
        &self,
        store: &PhoenixCozoStore,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        request: &BorrowedIngestRequest<'_>,
        persist_graph_relations: bool,
    ) -> Result<(IngestResult, NativeIngestArtifacts), StoreError> {
        let now = now_ms();
        let mut registry = EntityRegistry::from_store(store)?;
        let mut diagnostics = Vec::new();
        let mut total_warning_count = 0usize;
        let mut documents = Vec::new();
        let mut graph_batches = Vec::new();
        let mut total_chapters = 0usize;
        let mut total_boundaries = 0usize;
        let mut total_parents = 0usize;
        let mut total_leaves = 0usize;
        let mut total_mentions = 0usize;
        let mut total_edges = 0usize;
        let mut total_cross_chapter = 0usize;
        let mut total_discovery_candidates = 0usize;

        for document in request.documents {
            let processed = self.process_document_streaming(
                store,
                document,
                request.session_id.as_ref(),
                scanner,
                structure,
                &mut registry,
                persist_graph_relations,
            )?;
            total_chapters += processed.summary.chapter_count;
            total_boundaries += processed.summary.boundary_count;
            total_parents += processed.summary.parent_count;
            total_leaves += processed.summary.leaf_count;
            total_mentions += processed.persist_state.mention_count;
            total_edges +=
                processed.persist_state.edge_count + processed.persist_state.graph_edge_count;
            total_cross_chapter += processed.persist_state.cross_chapter_links;
            total_discovery_candidates += processed.persist_state.discovery_count;
            total_warning_count += processed.warning_count;
            diagnostics.extend(processed.diagnostics);
            if let Some(graph_batch) = processed.graph_batch {
                graph_batches.push(graph_batch);
            }
            documents.push(processed.summary);
        }
        let result = IngestResult {
            session_id: request.session_id.clone(),
            document_count: request.documents.len(),
            warning_count: total_warning_count,
            documents,
            chunk_stats: Some(ChunkStats {
                documents: request.documents.len(),
                total_chapters,
                total_boundaries,
                total_parents,
                total_leaves,
            }),
            graph_summary: Some(GraphSummary {
                documents: request.documents.len(),
                total_chapters,
                total_boundaries,
                total_leaves,
                total_entities: registry.entities.len(),
                total_mentions: total_mentions + registry.initial_mentions as usize,
                total_edges,
                cross_chapter_links: total_cross_chapter,
            }),
            entity_summary: Some(EntitySummary {
                total_entities: registry.entities.len(),
                total_aliases: registry.total_aliases(),
                total_mentions: total_mentions + registry.initial_mentions as usize,
                multi_chapter_entities: registry.multi_chapter_entities(),
            }),
            discovery_summary: Some(DiscoverySummary {
                candidate_count: total_discovery_candidates,
                mention_count: total_discovery_candidates,
                persisted_count: total_discovery_candidates,
            }),
            retrieval_summary: Some(RetrievalSummary {
                qgram_documents: request.documents.len(),
                gldr_chunks: total_leaves,
                gldr_entities: registry.entities.len(),
                gldr_edges: total_edges,
                raptor_documents: 0,
                raptor_leaves: 0,
                raptor_enabled: false,
            }),
            relation_counts: store.relation_counts()?,
            diagnostics,
        };

        if let Some(session_id) = request.session_id.as_ref() {
            self.persist_session_manifests(
                store,
                session_id,
                &result,
                now,
                !persist_graph_relations,
            )?;
        }

        Ok((result, NativeIngestArtifacts { graph_batches }))
    }

    pub fn ingest_message_thread_view(
        &self,
        store: &PhoenixCozoStore,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        thread: &BorrowedIngestThread<'_>,
    ) -> Result<IngestResult, StoreError> {
        let synthetic = build_thread_document(thread);
        let document = BorrowedIngestDocument {
            document_id: thread.document_id.clone(),
            note_id: None,
            title: thread.title,
            text: &synthetic.text,
            scope: thread.scope.clone(),
        };
        let note_id = NoteId(document.document_id.0.clone());
        let boundaries = vec![BoundarySpec {
            boundary_id: 0,
            ordinal: 0,
            kind: BoundaryKind::Chapter,
            depth: 1,
            label: "thread".to_owned(),
            parent_boundary_id: None,
            start: 0,
            end: document.text.len(),
        }];
        let mut chapters = vec![ChapterSpec {
            chunk_id: stable_int(
                "chunk",
                &[
                    document.document_id.0.as_str(),
                    "2",
                    "0",
                    &document.text.len().to_string(),
                ],
            ),
            chapter_id: 0,
            boundary_id: 0,
            boundary_ordinal: 0,
            boundary_kind: BoundaryKind::Chapter,
            boundary_depth: 1,
            start: 0,
            end: document.text.len(),
            title: "thread".to_owned(),
            parents: Vec::new(),
        }];
        let mut leaves = build_thread_leaf_chunks(&document, thread, &synthetic, &self.config);
        assign_leaves_to_boundaries(&boundaries, &mut leaves);
        assign_leaves_to_chapters(&document, &chapters, &mut leaves);
        build_parent_chunks(&document, &self.config, &mut chapters, &mut leaves);

        let policy = FlushPolicy::default();
        let mut diagnostics = DiagnosticCollector::new(policy.diagnostic_cap);
        diagnostics.push(Diagnostic {
            code: "PX_GRAPTOR_THREAD_INGEST".to_owned(),
            message: format!(
                "Thread ingest produced {} parent chunks and {} message-aware leaves.",
                chapters
                    .iter()
                    .map(|chapter| chapter.parents.len())
                    .sum::<usize>(),
                leaves.len()
            ),
        });
        let mut buffers = BufferSet::new();
        let mut persist_state = DocumentPersistState::default();
        let mut chapter_links = FxHashMap::<(u32, u32), FxHashSet<String>>::default();
        let mut resolver_scratch = ResolverSeedScratch::default();
        let mut registry = EntityRegistry::from_store(store)?;
        let scan_session_id = SessionId(format!("thread-{}", document.document_id.0));
        let now = now_ms();

        persist_document_backbone(store, &document, &note_id, now)?;
        for boundary in &boundaries {
            buffers
                .document_boundary_rows
                .insert_value(document_boundary_row(&document, &note_id, boundary))
                .expect("document boundary row");
        }
        insert_graph_vertex(
            &mut buffers,
            json!({
                "id": document_vertex_id(&document.document_id),
                "document_id": document.document_id.0,
                "narrative_id": document.scope.narrative_id,
                "value": {
                    "kind": "thread_document",
                    "documentId": document.document_id.0,
                    "noteId": note_id.0,
                    "title": document.title,
                    "messageCount": thread.messages.len(),
                },
                "weight": 1,
                "attributes": {
                    "scope": document.scope,
                    "messageIds": thread.messages.iter().map(|message| message.message_id).collect::<Vec<_>>(),
                    "roles": thread.messages.iter().map(|message| message.role).collect::<Vec<_>>(),
                },
            }),
        );
        buffers
            .graph_label_rows
            .insert_value(json!({
                "vertex_id": document_vertex_id(&document.document_id),
                "label": document.title,
            }))
            .expect("thread document label row");

        for chapter in &chapters {
            buffers
                .chunk_rows
                .insert_value(chapter_chunk_row(&document, chapter, &document.scope))
                .expect("chapter chunk row");
            buffers
                .chunkid_rows
                .insert_value(chapter_chunkid_row(&document, chapter))
                .expect("chapter chunk id row");
            insert_graph_vertex(&mut buffers, chapter_vertex_row(&document, chapter));
            buffers
                .graph_label_rows
                .insert_value(json!({
                    "vertex_id": chapter_vertex_id(&document.document_id, chapter.chapter_id),
                    "label": chapter.title.clone(),
                }))
                .expect("chapter label row");
            insert_graph_edge(
                &mut buffers,
                &mut persist_state,
                graph_edge_row(
                    &document_vertex_id(&document.document_id),
                    &chapter_vertex_id(&document.document_id, chapter.chapter_id),
                    1,
                    "contains",
                    json!({
                        "kind": "contains",
                        "documentId": document.document_id.0,
                        "boundaryId": chapter.boundary_id,
                        "boundaryOrdinal": chapter.boundary_ordinal,
                        "boundaryKind": boundary_kind_str(&chapter.boundary_kind),
                        "assertionKind": "current",
                    }),
                    None,
                    Some(document.document_id.0.clone()),
                    document.scope.narrative_id.clone(),
                ),
            );
        }

        process_document_chunks(
            store,
            policy,
            &document,
            &note_id,
            &chapters,
            &leaves,
            &scan_session_id,
            scanner,
            structure,
            &mut registry,
            &mut resolver_scratch,
            &mut buffers,
            &mut persist_state,
            &mut chapter_links,
            &mut diagnostics,
        )?;

        buffers.flush_all(store)?;

        let summary = IngestDocumentSummary {
            document_id: document.document_id.clone(),
            note_id: Some(note_id.clone()),
            chapter_count: chapters.len(),
            boundary_count: boundaries.len(),
            parent_count: chapters.iter().map(|chapter| chapter.parents.len()).sum(),
            leaf_count: leaves.len(),
            entity_count: persist_state.entity_ids.len(),
            edge_count: persist_state.edge_count + persist_state.graph_edge_count,
            has_front_matter_chapter: false,
            has_front_matter_boundary: false,
        };
        persist_entity_rows(
            store,
            &document,
            None,
            &note_id,
            &summary,
            &chapters,
            &boundaries,
            &registry,
            &persist_state,
            now,
            None,
        )?;
        let (warning_count, diagnostics) = diagnostics.finish();
        Ok(IngestResult {
            session_id: None,
            document_count: 1,
            warning_count,
            documents: vec![summary],
            chunk_stats: Some(ChunkStats {
                documents: 1,
                total_chapters: chapters.len(),
                total_boundaries: boundaries.len(),
                total_parents: chapters.iter().map(|chapter| chapter.parents.len()).sum(),
                total_leaves: leaves.len(),
            }),
            graph_summary: Some(GraphSummary {
                documents: 1,
                total_chapters: chapters.len(),
                total_boundaries: boundaries.len(),
                total_leaves: leaves.len(),
                total_entities: persist_state.entity_ids.len(),
                total_mentions: persist_state.mention_count,
                total_edges: persist_state.edge_count + persist_state.graph_edge_count,
                cross_chapter_links: persist_state.cross_chapter_links,
            }),
            entity_summary: Some(EntitySummary {
                total_entities: persist_state.entity_ids.len(),
                total_aliases: registry.total_aliases(),
                total_mentions: persist_state.mention_count,
                multi_chapter_entities: registry.multi_chapter_entities(),
            }),
            discovery_summary: Some(DiscoverySummary {
                candidate_count: persist_state.discovery_count,
                mention_count: persist_state.discovery_count,
                persisted_count: persist_state.discovery_count,
            }),
            retrieval_summary: Some(RetrievalSummary {
                qgram_documents: 1,
                gldr_chunks: leaves.len(),
                gldr_entities: persist_state.entity_ids.len(),
                gldr_edges: persist_state.edge_count + persist_state.graph_edge_count,
                raptor_documents: 0,
                raptor_leaves: 0,
                raptor_enabled: false,
            }),
            relation_counts: store.relation_counts()?,
            diagnostics,
        })
    }

    pub fn load_graph(&self, store: &PhoenixCozoStore) -> Result<GraptorGraph, StoreError> {
        load_graph_snapshot(store)
    }

    pub fn session_state(
        &self,
        store: &PhoenixCozoStore,
        session_id: &SessionId,
    ) -> Result<SessionState, StoreError> {
        load_session_state(store, session_id)
    }

    pub fn session_stats(
        &self,
        store: &PhoenixCozoStore,
        session_id: &SessionId,
    ) -> Result<SessionStats, StoreError> {
        load_session_stats(store, session_id)
    }

    pub fn graph_delta(
        &self,
        store: &PhoenixCozoStore,
        request: &GraphDeltaRequest,
    ) -> Result<GraphDeltaResult, StoreError> {
        build_graph_delta(store, request)
    }

    fn process_document_streaming(
        &self,
        store: &PhoenixCozoStore,
        document: &BorrowedIngestDocument<'_>,
        session_id: Option<&SessionId>,
        scanner: &PhoenixScanner,
        structure: &PhoenixStructure,
        registry: &mut EntityRegistry,
        persist_graph_relations: bool,
    ) -> Result<ProcessedDocument, StoreError> {
        let note_id = document
            .note_id
            .clone()
            .unwrap_or_else(|| NoteId(document.document_id.0.clone()));
        let boundary_markers = self.detect_boundaries(document.text);
        let boundaries = build_boundary_specs(document, &boundary_markers);
        let mut chapters = build_chapter_specs(document, &boundaries);
        let mut leaves = build_leaf_chunks(document, &self.config);
        assign_leaves_to_boundaries(&boundaries, &mut leaves);
        assign_leaves_to_chapters(document, &chapters, &mut leaves);
        build_parent_chunks(document, &self.config, &mut chapters, &mut leaves);

        let policy = FlushPolicy::default();
        let mut diagnostics = DiagnosticCollector::new(policy.diagnostic_cap);
        diagnostics.push(Diagnostic {
            code: "PX_GRAPTOR_CHUNKERX2".to_owned(),
            message: format!(
                "ChunkerX2-style ingest produced {} boundaries, {} chapters, {} parents, and {} leaves.",
                boundaries.len(),
                chapters.len(),
                chapters
                    .iter()
                    .map(|chapter| chapter.parents.len())
                    .sum::<usize>(),
                leaves.len()
            ),
        });
        let mut buffers = BufferSet::with_graph_persistence(persist_graph_relations);
        let mut persist_state = DocumentPersistState::default();
        let mut chapter_links = FxHashMap::<(u32, u32), FxHashSet<String>>::default();
        let mut resolver_scratch = ResolverSeedScratch::default();
        let now = now_ms();

        persist_document_backbone(store, document, &note_id, now)?;
        for boundary in &boundaries {
            buffers
                .document_boundary_rows
                .insert_value(document_boundary_row(document, &note_id, boundary))
                .expect("document boundary row");
        }
        insert_graph_vertex(
            &mut buffers,
            json!({
                "id": document_vertex_id(&document.document_id),
                "value": {
                    "kind": "document",
                    "documentId": document.document_id.0,
                    "noteId": note_id.0,
                    "title": document.title,
                },
                "weight": 1,
                "attributes": {
                    "scope": document.scope,
                },
            }),
        );
        buffers
            .graph_label_rows
            .insert_value(json!({
                "vertex_id": document_vertex_id(&document.document_id),
                "label": document.title,
            }))
            .expect("document label row");

        for chapter in &chapters {
            buffers
                .chunk_rows
                .insert_value(chapter_chunk_row(document, chapter, &document.scope))
                .expect("chapter chunk row");
            buffers
                .chunkid_rows
                .insert_value(chapter_chunkid_row(document, chapter))
                .expect("chapter chunk id row");
            insert_graph_vertex(&mut buffers, chapter_vertex_row(document, chapter));
            buffers
                .graph_label_rows
                .insert_value(json!({
                    "vertex_id": chapter_vertex_id(&document.document_id, chapter.chapter_id),
                    "label": chapter.title.clone(),
                }))
                .expect("chapter label row");
            insert_graph_edge(
                &mut buffers,
                &mut persist_state,
                graph_edge_row(
                    &document_vertex_id(&document.document_id),
                    &chapter_vertex_id(&document.document_id, chapter.chapter_id),
                    1,
                    "contains",
                    json!({ "kind": "contains" }),
                    None,
                    Some(document.document_id.0.clone()),
                    document.scope.narrative_id.clone(),
                ),
            );
            buffers.flush_due(store, policy)?;
        }

        let scan_session_id = SessionId(format!(
            "{}::{}::graptor",
            session_id
                .map(|value| value.0.clone())
                .unwrap_or_else(|| "session".to_owned()),
            document.document_id.0
        ));

        process_document_chunks(
            store,
            policy,
            document,
            &note_id,
            &chapters,
            &leaves,
            &scan_session_id,
            scanner,
            structure,
            registry,
            &mut resolver_scratch,
            &mut buffers,
            &mut persist_state,
            &mut chapter_links,
            &mut diagnostics,
        )?;

        let graph_batch = if persist_graph_relations {
            None
        } else {
            buffers.asserted_graph_batch_for_document(&document.document_id.0)?
        };

        buffers.flush_all(store)?;

        let summary = IngestDocumentSummary {
            document_id: document.document_id.clone(),
            note_id: Some(note_id.clone()),
            chapter_count: chapters.len(),
            boundary_count: boundaries.len(),
            parent_count: chapters.iter().map(|chapter| chapter.parents.len()).sum(),
            leaf_count: leaves.len(),
            entity_count: persist_state.entity_ids.len(),
            edge_count: persist_state.edge_count + persist_state.graph_edge_count,
            has_front_matter_chapter: chapters
                .first()
                .map(|chapter| chapter.chapter_id == 0)
                .unwrap_or(false),
            has_front_matter_boundary: boundaries
                .first()
                .map(|boundary| boundary.boundary_id == 0)
                .unwrap_or(false),
        };

        persist_entity_rows(
            store,
            document,
            session_id,
            &note_id,
            &summary,
            &chapters,
            &boundaries,
            registry,
            &persist_state,
            now,
            graph_batch.as_ref(),
        )?;

        let (warning_count, diagnostics) = diagnostics.finish();
        Ok(ProcessedDocument {
            summary,
            persist_state,
            warning_count,
            diagnostics,
            graph_batch,
        })
    }

    fn detect_boundaries(&self, text: &str) -> Vec<BoundaryMarker> {
        let mut boundaries = Vec::new();
        if self.max_heading_depth > 0 {
            boundaries.extend(scan_heading_boundaries(text, self.max_heading_depth));
        }
        let Some(matcher) = &self.boundary_matcher else {
            boundaries.sort_by_key(|boundary| boundary.start);
            boundaries.dedup_by_key(|boundary| boundary.start);
            return boundaries;
        };
        let bytes = text.as_bytes();
        let mut last_line_start = None;
        for matched in matcher.leftmost_find_iter(bytes) {
            let (line_start, line_end) = line_bounds(bytes, matched.start());
            if last_line_start == Some(line_start) {
                continue;
            }
            last_line_start = Some(line_start);
            let line = text.get(line_start..line_end).unwrap_or_default();
            if let Some((kind, depth)) = validate_chapter_line(line, self.max_heading_depth) {
                boundaries.push(BoundaryMarker {
                    start: line_start,
                    kind,
                    depth,
                    label: line.trim().to_owned(),
                });
            }
        }
        let mut line_start = 0usize;
        for (offset, byte) in bytes.iter().enumerate() {
            if *byte != b'\n' && offset + 1 != bytes.len() {
                continue;
            }
            let line_end = if *byte == b'\n' { offset } else { bytes.len() };
            if boundaries
                .iter()
                .any(|boundary| boundary.start == line_start)
            {
                line_start = offset + 1;
                continue;
            }
            let line = text.get(line_start..line_end).unwrap_or_default();
            if let Some((kind, depth)) = validate_chapter_line(line, self.max_heading_depth) {
                boundaries.push(BoundaryMarker {
                    start: line_start,
                    kind,
                    depth,
                    label: line.trim().to_owned(),
                });
            }
            line_start = offset + 1;
        }
        boundaries.sort_by(|left, right| {
            left.start
                .cmp(&right.start)
                .then_with(|| left.depth.cmp(&right.depth))
        });
        boundaries.dedup_by_key(|boundary| boundary.start);
        boundaries
    }

    #[cfg(test)]
    fn detect_chapter_boundaries(&self, text: &str) -> Vec<BoundaryMarker> {
        self.detect_boundaries(text)
    }
}

impl Default for PhoenixGraptor {
    fn default() -> Self {
        Self::new(GraptorConfig::default())
    }
}

impl PhoenixGraptor {
    fn persist_session_manifests(
        &self,
        store: &PhoenixCozoStore,
        session_id: &SessionId,
        _result: &IngestResult,
        now: i64,
        native_pipeline: bool,
    ) -> Result<(), StoreError> {
        let state = load_session_state(store, session_id)?;
        let stats = load_session_stats(store, session_id)?;
        let narrative_id = None::<String>;
        persist_session_workspace_artifact(
            store,
            session_id,
            LEGACY_SESSION_STATE_KIND,
            &serde_json::to_value(&state).expect("session state json"),
            narrative_id.as_deref(),
            None,
            now,
        )?;
        if native_pipeline {
            persist_session_workspace_artifact(
                store,
                session_id,
                INVARANT_SESSION_STATE_KIND,
                &serde_json::to_value(&state).expect("session state json"),
                narrative_id.as_deref(),
                None,
                now,
            )?;
        }
        persist_session_workspace_artifact(
            store,
            session_id,
            LEGACY_SESSION_STATS_KIND,
            &serde_json::to_value(&stats).expect("session stats json"),
            narrative_id.as_deref(),
            None,
            now,
        )?;
        if native_pipeline {
            persist_session_workspace_artifact(
                store,
                session_id,
                INVARANT_SESSION_STATS_KIND,
                &serde_json::to_value(&stats).expect("session stats json"),
                narrative_id.as_deref(),
                None,
                now,
            )?;
        }
        persist_session_definition(
            store,
            session_id,
            "state",
            &serde_json::to_value(&state).expect("session state json"),
            narrative_id.as_deref(),
            now,
            native_pipeline,
        )?;
        persist_session_definition(
            store,
            session_id,
            "stats",
            &serde_json::to_value(&stats).expect("session stats json"),
            narrative_id.as_deref(),
            now,
            native_pipeline,
        )?;
        Ok(())
    }
}

pub fn load_graph_snapshot(store: &PhoenixCozoStore) -> Result<GraptorGraph, StoreError> {
    load_graph_snapshot_with_candidate_graph(store, false)
}

pub fn load_graph_snapshot_with_candidate_graph(
    store: &PhoenixCozoStore,
    include_candidate_graph: bool,
) -> Result<GraptorGraph, StoreError> {
    const VERTEX_COLUMNS: &[&str] = &[
        "id",
        "document_id",
        "narrative_id",
        "weight",
        "value",
        "attributes",
    ];
    const EDGE_COLUMNS: &[&str] = &[
        "source_id",
        "target_id",
        "document_id",
        "narrative_id",
        "edge_type",
        "weight",
        "attributes",
        "data",
    ];

    let mut graph = GraptorGraph::default();
    for row in store.fetch_compact_rows_with_columns("graph_vertices", VERTEX_COLUMNS)? {
        let row = CompactRowView::new(VERTEX_COLUMNS, &row);
        let Some(id) = row.get_str("id") else {
            continue;
        };
        let value = row.get_json("value").unwrap_or(Value::Null);
        let attributes = row.get_json("attributes").unwrap_or(Value::Null);
        let kind = value
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned();
        let entity_id = value
            .get("entityId")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let search_chunk_id = value
            .get("searchChunkId")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                attributes
                    .get("searchChunkId")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            });
        let document_id = row
            .get_str("document_id")
            .map(str::to_owned)
            .or_else(|| {
                attributes
                    .get("documentId")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .or_else(|| {
                value
                    .get("documentId")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            });
        let chapter_id = attributes
            .get("chapterId")
            .and_then(Value::as_u64)
            .map(|value| value as u32);
        let chapters = attributes
            .get("chapters")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|value| value as u32)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let boundary_id = attributes
            .get("boundaryId")
            .and_then(Value::as_u64)
            .map(|value| value as u32);
        let boundary_ordinal = attributes
            .get("boundaryOrdinal")
            .and_then(Value::as_u64)
            .map(|value| value as u32);
        let boundary_kind = attributes
            .get("boundaryKind")
            .and_then(Value::as_str)
            .map(boundary_kind_from_str);
        let boundary_ordinals = attributes
            .get("boundaryOrdinals")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_u64)
                    .map(|value| value as u32)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let vertex = GraptorVertex {
            id: id.to_owned(),
            kind,
            weight: row.get_i64("weight").unwrap_or(1),
            value,
            attributes: attributes.clone(),
            entity_id,
            search_chunk_id: search_chunk_id.clone(),
            document_id: document_id.clone(),
            chapter_id,
            chapters,
            boundary_id,
            boundary_ordinal,
            boundary_kind,
            boundary_ordinals,
        };
        graph.vertices.insert(id.to_owned(), vertex);
        if let (Some(document_id), Some(chapter_id), Some(_)) =
            (document_id, chapter_id, search_chunk_id)
        {
            graph
                .chapter_leaves
                .entry((document_id, chapter_id))
                .or_default()
                .push(id.to_owned());
        }
    }
    append_graph_edges(
        &mut graph,
        &store.fetch_compact_rows_with_columns("graph_edges", EDGE_COLUMNS)?,
        EDGE_COLUMNS,
        false,
    );
    if include_candidate_graph {
        append_graph_edges(
            &mut graph,
            &store.fetch_compact_rows_with_columns("graph_candidate_edges", EDGE_COLUMNS)?,
            EDGE_COLUMNS,
            true,
        );
    }
    Ok(graph)
}

fn append_graph_edges(
    graph: &mut GraptorGraph,
    rows: &[CompactRow],
    columns: &[&str],
    candidate_only_active: bool,
) {
    for row in rows {
        let row = CompactRowView::new(columns, row);
        let Some(source_id) = row.get_str("source_id") else {
            continue;
        };
        let Some(target_id) = row.get_str("target_id") else {
            continue;
        };
        let attributes = {
            let mut attributes = row.get_json("attributes").unwrap_or(Value::Null);
            if candidate_only_active && !candidate_edge_is_active(&attributes) {
                continue;
            }
            if let Some(object) = attributes.as_object_mut() {
                if !object.contains_key("documentId") {
                    if let Some(document_id) = row.get_str("document_id") {
                        object.insert("documentId".to_owned(), json!(document_id));
                    }
                }
                if !object.contains_key("narrativeId") {
                    if let Some(narrative_id) = row.get_str("narrative_id") {
                        object.insert("narrativeId".to_owned(), json!(narrative_id));
                    }
                }
            }
            attributes
        };
        let edge = GraptorEdge {
            source_id: source_id.to_owned(),
            target_id: target_id.to_owned(),
            edge_type: row.get_str("edge_type").unwrap_or("edge").to_owned(),
            weight: row.get_i64("weight").unwrap_or(1),
            attributes,
            data: row.get_json("data").filter(|value| !value.is_null()),
            layer: if candidate_only_active {
                GraphLayer::Candidate
            } else {
                GraphLayer::Asserted
            },
        };
        graph
            .outgoing
            .entry(source_id.to_owned())
            .or_default()
            .push(edge.clone());
        graph
            .incoming
            .entry(target_id.to_owned())
            .or_default()
            .push(edge);
    }
}

fn candidate_edge_is_active(attributes: &Value) -> bool {
    !matches!(
        attributes
            .get("graph")
            .and_then(Value::as_object)
            .and_then(|graph| graph.get("status"))
            .and_then(Value::as_str),
        Some("candidate_rejected")
    )
}

pub fn load_session_state(
    store: &PhoenixCozoStore,
    session_id: &SessionId,
) -> Result<SessionState, StoreError> {
    const ARTIFACT_COLUMNS: &[&str] = &["thread_id", "kind", "payload"];
    const SCOPED_DOCUMENT_COLUMNS: &[&str] = &["namespace", "payload"];

    let artifact = store
        .fetch_compact_rows_with_columns("workspace_artifacts", ARTIFACT_COLUMNS)?
        .into_iter()
        .find(|row| {
            let row = CompactRowView::new(ARTIFACT_COLUMNS, row);
            row.get_str("thread_id") == Some(session_id.0.as_str())
                && matches!(
                    row.get_str("kind"),
                    Some(LEGACY_SESSION_STATE_KIND | INVARANT_SESSION_STATE_KIND)
                )
        });
    if let Some(payload) =
        artifact.and_then(|row| CompactRowView::new(ARTIFACT_COLUMNS, &row).get_json("payload"))
    {
        if let Ok(state) = serde_json::from_value(payload) {
            return Ok(state);
        }
    }

    let mut documents = store
        .fetch_compact_rows_with_columns("scoped_documents", SCOPED_DOCUMENT_COLUMNS)?
        .into_iter()
        .filter(|row| {
            matches!(
                CompactRowView::new(SCOPED_DOCUMENT_COLUMNS, row).get_str("namespace"),
                Some(LEGACY_DOCUMENT_NAMESPACE | INVARANT_DOCUMENT_NAMESPACE)
            )
        })
        .filter(|row| {
            CompactRowView::new(SCOPED_DOCUMENT_COLUMNS, row)
                .get_json("payload")
                .as_ref()
                .and_then(|value| value.get("sessionId"))
                .and_then(Value::as_str)
                == Some(session_id.0.as_str())
        })
        .filter_map(|row| {
            let payload = CompactRowView::new(SCOPED_DOCUMENT_COLUMNS, &row).get_json("payload")?;
            Some(SessionDocumentState {
                document_id: DocumentId(
                    payload
                        .get("documentId")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                ),
                note_id: payload
                    .get("noteId")
                    .and_then(Value::as_str)
                    .map(|value| NoteId(value.to_owned())),
                chapter_count: payload
                    .get("summary")
                    .and_then(|value| value.get("chapterCount"))
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize,
                boundary_count: payload
                    .get("summary")
                    .and_then(|value| value.get("boundaryCount"))
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize,
                chapter_titles: payload
                    .get("chapters")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(|value| value.get("title").and_then(Value::as_str))
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                boundary_labels: payload
                    .get("boundaries")
                    .and_then(Value::as_array)
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(|value| value.get("label").and_then(Value::as_str))
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default(),
                parent_count: payload
                    .get("summary")
                    .and_then(|value| value.get("parentCount"))
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize,
                leaf_count: payload
                    .get("summary")
                    .and_then(|value| value.get("leafCount"))
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize,
                entity_count: payload
                    .get("summary")
                    .and_then(|value| value.get("entityCount"))
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize,
                discovery_count: payload
                    .get("discoveryCount")
                    .and_then(Value::as_u64)
                    .unwrap_or_default() as usize,
                has_front_matter_chapter: payload
                    .get("summary")
                    .and_then(|value| value.get("hasFrontMatterChapter"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                has_front_matter_boundary: payload
                    .get("summary")
                    .and_then(|value| value.get("hasFrontMatterBoundary"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                updated_at: payload
                    .get("updatedAt")
                    .and_then(Value::as_i64)
                    .unwrap_or_default(),
            })
        })
        .collect::<Vec<_>>();
    documents.sort_by(|left, right| left.document_id.cmp(&right.document_id));
    Ok(SessionState {
        session_id: session_id.clone(),
        documents,
        manifest_namespaces: vec![
            INVARANT_DOCUMENT_NAMESPACE.to_owned(),
            INVARANT_MANIFEST_NAMESPACE.to_owned(),
            INVARANT_SESSION_NAMESPACE.to_owned(),
            LEGACY_DOCUMENT_NAMESPACE.to_owned(),
            LEGACY_MANIFEST_NAMESPACE.to_owned(),
            LEGACY_SESSION_NAMESPACE.to_owned(),
        ],
        updated_at: now_ms(),
    })
}

pub fn load_session_stats(
    store: &PhoenixCozoStore,
    session_id: &SessionId,
) -> Result<SessionStats, StoreError> {
    const ARTIFACT_COLUMNS: &[&str] = &["thread_id", "kind", "payload"];
    let artifact = store
        .fetch_compact_rows_with_columns("workspace_artifacts", ARTIFACT_COLUMNS)?
        .into_iter()
        .find(|row| {
            let row = CompactRowView::new(ARTIFACT_COLUMNS, row);
            row.get_str("thread_id") == Some(session_id.0.as_str())
                && matches!(
                    row.get_str("kind"),
                    Some(LEGACY_SESSION_STATS_KIND | INVARANT_SESSION_STATS_KIND)
                )
        });
    if let Some(payload) =
        artifact.and_then(|row| CompactRowView::new(ARTIFACT_COLUMNS, &row).get_json("payload"))
    {
        if let Ok(stats) = serde_json::from_value(payload) {
            return Ok(stats);
        }
    }

    let state = load_session_state(store, session_id)?;
    let relation_counts = store.relation_counts()?;
    let count_for = |name: &str| {
        relation_counts
            .iter()
            .find(|count| count.relation == name)
            .map(|count| count.rows)
            .unwrap_or_default()
    };
    Ok(build_session_stats_from_counts(
        &state,
        session_id,
        &GraphCounts {
            vertex_count: count_for("graph_vertices"),
            asserted_edge_count: count_for("graph_edges"),
            candidate_edge_count: count_for("graph_candidate_edges"),
        },
        count_for("discovery_candidates"),
        count_for("spans"),
    ))
}

pub fn build_session_stats_from_counts(
    state: &SessionState,
    session_id: &SessionId,
    graph_counts: &GraphCounts,
    discovery_candidate_count: usize,
    span_count: usize,
) -> SessionStats {
    SessionStats {
        session_id: session_id.clone(),
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
        graph_vertex_count: graph_counts.vertex_count,
        graph_edge_count: graph_counts.asserted_edge_count,
        span_count,
        updated_at: now_ms(),
    }
}

pub fn build_graph_delta(
    store: &PhoenixCozoStore,
    request: &GraphDeltaRequest,
) -> Result<GraphDeltaResult, StoreError> {
    let graph = load_graph_snapshot_with_candidate_graph(store, request.include_candidate_graph)?;
    let state = load_session_state(store, &request.session_id)?;
    Ok(build_graph_delta_from_snapshot(&graph, &state, request))
}

pub fn build_graph_delta_from_snapshot(
    graph: &GraptorGraph,
    state: &SessionState,
    request: &GraphDeltaRequest,
) -> GraphDeltaResult {
    let mut allowed_documents = state
        .documents
        .iter()
        .map(|document| document.document_id.0.clone())
        .collect::<FxHashSet<_>>();
    if !request.changed_documents.is_empty() {
        let requested = request
            .changed_documents
            .iter()
            .map(|document| document.0.clone())
            .collect::<FxHashSet<_>>();
        allowed_documents.retain(|document_id| requested.contains(document_id));
    }

    let mut diagnostics = Vec::new();
    if let Some(since_commit) = request.since_commit.as_ref() {
        diagnostics.push(Diagnostic {
            code: "PX_GRAPH_DELTA_SNAPSHOT".to_owned(),
            message: format!(
                "Graph delta is snapshot-based in v1; sinceCommit {} was treated as a hint only.",
                since_commit.0
            ),
        });
    }
    if request.include_candidate_graph {
        diagnostics.push(Diagnostic {
            code: "PX_GRAPH_DELTA_CANDIDATE".to_owned(),
            message:
                "Graph delta included candidate embedding edges in addition to asserted graph structure."
                    .to_owned(),
        });
    }

    let mut chunk_ids = graph
        .vertices
        .values()
        .filter(|vertex| vertex.kind == "leaf")
        .filter(|vertex| {
            vertex
                .document_id
                .as_ref()
                .map(|document_id| allowed_documents.contains(document_id))
                .unwrap_or(false)
        })
        .map(|vertex| vertex.id.clone())
        .collect::<Vec<_>>();
    chunk_ids.sort();
    if let Some(limit) = request.limit {
        if chunk_ids.len() > limit {
            chunk_ids.truncate(limit);
            diagnostics.push(Diagnostic {
                code: "PX_GRAPH_DELTA_LIMIT".to_owned(),
                message: format!("Graph delta limited to {limit} leaf chunks for this response."),
            });
        }
    }

    let chunk_id_set = chunk_ids.iter().cloned().collect::<FxHashSet<_>>();
    let mut included_nodes = chunk_id_set.clone();
    for chunk_id in &chunk_ids {
        for edge in graph
            .outgoing_any(chunk_id)
            .chain(graph.incoming_any(chunk_id))
        {
            if let Some(vertex) = graph.vertices.get(if edge.source_id == *chunk_id {
                edge.target_id.as_str()
            } else {
                edge.source_id.as_str()
            }) {
                if matches!(vertex.kind.as_str(), "entity" | "event") {
                    included_nodes.insert(vertex.id.clone());
                }
            }
        }
    }

    let mut traversal_depths = FxHashMap::<String, usize>::default();
    let mut traversal_frontier = VecDeque::new();
    for document_id in &allowed_documents {
        let vertex_id = document_vertex_id(&DocumentId(document_id.clone()));
        if graph.vertices.contains_key(&vertex_id) {
            traversal_depths.insert(vertex_id.clone(), 0);
            traversal_frontier.push_back(vertex_id.clone());
            included_nodes.insert(vertex_id);
        }
    }
    while let Some(vertex_id) = traversal_frontier.pop_front() {
        let depth = *traversal_depths.get(&vertex_id).unwrap_or(&0);
        if depth >= 2 {
            continue;
        }
        for edge in graph
            .outgoing_any(&vertex_id)
            .chain(graph.incoming_any(&vertex_id))
        {
            let neighbor_id = if edge.source_id == vertex_id {
                edge.target_id.as_str()
            } else {
                edge.source_id.as_str()
            };
            let Some(vertex) = graph.vertices.get(neighbor_id) else {
                continue;
            };
            if !graph_delta_vertex_allowed(vertex, &allowed_documents, &chunk_id_set) {
                continue;
            }
            let next_depth = depth + 1;
            let should_visit = traversal_depths
                .get(neighbor_id)
                .map(|current| next_depth < *current)
                .unwrap_or(true);
            if should_visit {
                traversal_depths.insert(neighbor_id.to_owned(), next_depth);
                traversal_frontier.push_back(neighbor_id.to_owned());
            }
            included_nodes.insert(neighbor_id.to_owned());
        }
    }

    let mut extra_node_ids = included_nodes
        .iter()
        .filter(|vertex_id| !chunk_id_set.contains(*vertex_id))
        .cloned()
        .collect::<Vec<_>>();
    extra_node_ids.sort();
    let mut chunks = Vec::new();
    for chunk_id in &chunk_ids {
        let Some(vertex) = graph.vertices.get(chunk_id) else {
            continue;
        };
        let start = vertex
            .attributes
            .get("start")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32;
        let end = vertex
            .attributes
            .get("end")
            .and_then(Value::as_u64)
            .unwrap_or_default() as u32;
        let Some(document_id) = vertex.document_id.as_ref().cloned() else {
            continue;
        };
        chunks.push(GraphDeltaChunk {
            vertex_id: vertex.id.clone(),
            chunk_id: vertex.search_chunk_id.clone().unwrap_or_default(),
            document_id: DocumentId(document_id),
            note_id: vertex
                .attributes
                .get("noteId")
                .and_then(Value::as_str)
                .map(|value| NoteId(value.to_owned())),
            chapter_id: vertex.chapter_id.unwrap_or_default(),
            boundary_id: vertex.boundary_id,
            boundary_ordinal: vertex.boundary_ordinal,
            range: TextRange { start, end },
        });
    }

    let mut nodes = Vec::new();
    for node_id in &extra_node_ids {
        let Some(vertex) = graph.vertices.get(node_id) else {
            continue;
        };
        let label = vertex
            .value
            .get("label")
            .and_then(Value::as_str)
            .or_else(|| vertex.value.get("lemma").and_then(Value::as_str))
            .unwrap_or(vertex.id.as_str())
            .to_owned();
        nodes.push(GraphDeltaNode {
            node_id: vertex.id.clone(),
            kind: vertex.kind.clone(),
            label,
            entity_id: vertex
                .entity_id
                .as_ref()
                .map(|value| EntityId(value.clone())),
            document_id: vertex
                .document_id
                .as_ref()
                .map(|value| DocumentId(value.clone())),
            chapter_id: vertex.chapter_id,
            boundary_id: vertex.boundary_id,
            boundary_ordinal: vertex.boundary_ordinal,
            weight: vertex.weight as i32,
        });
    }

    let mut edges = Vec::new();
    let mut edge_keys = FxHashSet::default();
    for vertex_id in &included_nodes {
        for edge in graph.outgoing_any(vertex_id) {
            if included_nodes.contains(&edge.target_id)
                && edge_keys.insert((
                    edge.source_id.clone(),
                    edge.target_id.clone(),
                    edge.edge_type.clone(),
                ))
            {
                edges.push(GraphDeltaEdge {
                    source_id: edge.source_id.clone(),
                    target_id: edge.target_id.clone(),
                    edge_type: edge.edge_type.clone(),
                    weight: edge.weight as i32,
                });
            }
        }
    }
    edges.sort_by(|left, right| {
        (&left.source_id, &left.target_id, &left.edge_type).cmp(&(
            &right.source_id,
            &right.target_id,
            &right.edge_type,
        ))
    });

    GraphDeltaResult {
        session_id: request.session_id.clone(),
        chunks,
        nodes,
        edges,
        diagnostics,
    }
}

fn graph_delta_vertex_allowed(
    vertex: &GraptorVertex,
    allowed_documents: &FxHashSet<String>,
    chunk_id_set: &FxHashSet<String>,
) -> bool {
    match vertex.kind.as_str() {
        "leaf" => chunk_id_set.contains(&vertex.id),
        "document" | "thread_document" | "chapter" | "parent" | "event" => vertex
            .document_id
            .as_ref()
            .map(|document_id| allowed_documents.contains(document_id))
            .unwrap_or(false),
        "entity" | "turn" | "agent" | "task" | "state" | "time_window" => true,
        _ => false,
    }
}

#[derive(Clone, Debug)]
struct BoundaryMarker {
    start: usize,
    kind: BoundaryKind,
    depth: u8,
    label: String,
}

#[derive(Clone, Debug)]
struct BoundarySpec {
    boundary_id: u32,
    ordinal: u32,
    kind: BoundaryKind,
    depth: u8,
    label: String,
    parent_boundary_id: Option<u32>,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct ChapterSpec {
    chunk_id: i64,
    chapter_id: u32,
    boundary_id: u32,
    boundary_ordinal: u32,
    boundary_kind: BoundaryKind,
    boundary_depth: u8,
    start: usize,
    end: usize,
    title: String,
    parents: Vec<ParentChunk>,
}

#[derive(Clone, Debug)]
struct ParentChunk {
    chunk_id: i64,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug)]
struct LeafChunk {
    chunk_id: i64,
    search_id: String,
    chapter_id: u32,
    boundary_id: u32,
    boundary_ordinal: u32,
    boundary_kind: BoundaryKind,
    parent_id: Option<i64>,
    start: usize,
    end: usize,
    message_meta: Option<LeafMessageMeta>,
}

#[derive(Clone, Debug)]
struct LeafMessageMeta {
    message_ids: Vec<String>,
    roles: Vec<String>,
    start_index: usize,
    end_index: usize,
}

#[derive(Clone, Debug)]
struct ThreadSyntheticDocument {
    text: String,
    messages: Vec<ThreadMessageRange>,
}

#[derive(Clone, Debug)]
struct ThreadMessageRange {
    message_id: String,
    role: String,
    message_index: usize,
    start: usize,
    end: usize,
}

#[derive(Clone, Copy, Debug)]
struct FlushPolicy {
    text_heavy_limit: usize,
    medium_limit: usize,
    lightweight_limit: usize,
    diagnostic_cap: usize,
}

impl Default for FlushPolicy {
    fn default() -> Self {
        if cfg!(target_arch = "wasm32") {
            Self {
                text_heavy_limit: 512,
                medium_limit: 512,
                lightweight_limit: 1024,
                diagnostic_cap: 256,
            }
        } else {
            Self {
                text_heavy_limit: 256,
                medium_limit: 512,
                lightweight_limit: 1024,
                diagnostic_cap: 512,
            }
        }
    }
}

#[derive(Default)]
struct DiagnosticCollector {
    cap: usize,
    total: usize,
    diagnostics: Vec<Diagnostic>,
    truncated: bool,
}

impl DiagnosticCollector {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            total: 0,
            diagnostics: Vec::new(),
            truncated: false,
        }
    }

    fn push(&mut self, diagnostic: Diagnostic) {
        self.total += 1;
        if self.diagnostics.len() < self.cap {
            self.diagnostics.push(diagnostic);
        } else {
            self.truncated = true;
        }
    }

    fn finish(mut self) -> (usize, Vec<Diagnostic>) {
        if self.truncated {
            self.diagnostics.push(Diagnostic {
                code: "PX_GRAPTOR_DIAGNOSTICS_TRUNCATED".to_owned(),
                message: format!(
                    "Ingest emitted {} diagnostics; only the first {} are retained in-memory.",
                    self.total, self.cap
                ),
            });
        }
        (self.total, self.diagnostics)
    }
}

#[derive(Default)]
struct ResolverSeedScratch {
    version: u64,
    seeds: Vec<ResolverEntitySeed>,
}

#[derive(Clone, Debug, Default)]
struct DocumentPersistState {
    entity_ids: FxHashSet<String>,
    mention_count: usize,
    edge_count: usize,
    graph_edge_count: usize,
    cross_chapter_links: usize,
    discovery_count: usize,
}

struct ProcessedDocument {
    summary: IngestDocumentSummary,
    persist_state: DocumentPersistState,
    warning_count: usize,
    diagnostics: Vec<Diagnostic>,
    graph_batch: Option<GraphMutationBatch>,
}

struct BufferSet {
    persist_graph_relations: bool,
    chunk_rows: CompactRelationBuffer,
    chunkid_rows: CompactRelationBuffer,
    document_boundary_rows: CompactRelationBuffer,
    span_rows: CompactRelationBuffer,
    span_mention_rows: CompactRelationBuffer,
    evidence_rows: CompactRelationBuffer,
    discovery_rows: CompactRelationBuffer,
    edge_rows: CompactRelationBuffer,
    graph_vertex_rows: CompactRelationBuffer,
    graph_label_rows: CompactRelationBuffer,
    graph_edge_rows: CompactRelationBuffer,
    graph_property_rows: CompactRelationBuffer,
}

impl BufferSet {
    fn new() -> Self {
        Self::with_graph_persistence(true)
    }

    fn with_graph_persistence(persist_graph_relations: bool) -> Self {
        Self {
            persist_graph_relations,
            chunk_rows: CompactRelationBuffer::new("chunks").expect("chunks relation"),
            chunkid_rows: CompactRelationBuffer::new("chunkid_map").expect("chunkid_map relation"),
            document_boundary_rows: CompactRelationBuffer::new("document_boundaries")
                .expect("document_boundaries relation"),
            span_rows: CompactRelationBuffer::new("spans").expect("spans relation"),
            span_mention_rows: CompactRelationBuffer::new("span_mentions")
                .expect("span_mentions relation"),
            evidence_rows: CompactRelationBuffer::new("spans").expect("spans relation"),
            discovery_rows: CompactRelationBuffer::new("discovery_candidates")
                .expect("discovery_candidates relation"),
            edge_rows: CompactRelationBuffer::new("edges").expect("edges relation"),
            graph_vertex_rows: CompactRelationBuffer::new("graph_vertices")
                .expect("graph_vertices relation"),
            graph_label_rows: CompactRelationBuffer::new("graph_vertex_labels")
                .expect("graph_vertex_labels relation"),
            graph_edge_rows: CompactRelationBuffer::new("graph_edges")
                .expect("graph_edges relation"),
            graph_property_rows: CompactRelationBuffer::new("graph_properties")
                .expect("graph_properties relation"),
        }
    }

    fn flush_due(
        &mut self,
        store: &PhoenixCozoStore,
        policy: FlushPolicy,
    ) -> Result<(), StoreError> {
        flush_relation_if_needed(store, &mut self.chunk_rows, policy.text_heavy_limit)?;
        flush_relation_if_needed(store, &mut self.span_rows, policy.text_heavy_limit)?;
        flush_relation_if_needed(store, &mut self.evidence_rows, policy.text_heavy_limit)?;
        flush_relation_if_needed(store, &mut self.document_boundary_rows, policy.medium_limit)?;
        if self.persist_graph_relations {
            flush_relation_if_needed(store, &mut self.graph_vertex_rows, policy.text_heavy_limit)?;
            flush_relation_if_needed(store, &mut self.graph_edge_rows, policy.text_heavy_limit)?;
            flush_relation_if_needed(
                store,
                &mut self.graph_property_rows,
                policy.text_heavy_limit,
            )?;
        }
        flush_relation_if_needed(store, &mut self.discovery_rows, policy.medium_limit)?;
        flush_relation_if_needed(store, &mut self.edge_rows, policy.medium_limit)?;
        if self.persist_graph_relations {
            flush_relation_if_needed(store, &mut self.graph_label_rows, policy.medium_limit)?;
        }
        flush_relation_if_needed(store, &mut self.span_mention_rows, policy.medium_limit)?;
        flush_relation_if_needed(store, &mut self.chunkid_rows, policy.lightweight_limit)?;
        Ok(())
    }

    fn flush_all(&mut self, store: &PhoenixCozoStore) -> Result<(), StoreError> {
        flush_relation_all(store, &mut self.chunk_rows)?;
        flush_relation_all(store, &mut self.chunkid_rows)?;
        flush_relation_all(store, &mut self.document_boundary_rows)?;
        flush_relation_all(store, &mut self.span_rows)?;
        flush_relation_all(store, &mut self.span_mention_rows)?;
        flush_relation_all(store, &mut self.evidence_rows)?;
        flush_relation_all(store, &mut self.discovery_rows)?;
        flush_relation_all(store, &mut self.edge_rows)?;
        if self.persist_graph_relations {
            flush_relation_all(store, &mut self.graph_vertex_rows)?;
            flush_relation_all(store, &mut self.graph_label_rows)?;
            flush_relation_all(store, &mut self.graph_edge_rows)?;
            flush_relation_all(store, &mut self.graph_property_rows)?;
        }
        Ok(())
    }

    fn asserted_graph_batch_for_document(
        &self,
        document_id: &str,
    ) -> Result<Option<GraphMutationBatch>, StoreError> {
        let vertices = self
            .graph_vertex_rows
            .json_rows()?
            .into_iter()
            .filter_map(|row| graph_vertex_record_from_row(&row))
            .filter(|row| row.document_id.as_deref() == Some(document_id))
            .collect::<Vec<_>>();
        let edges = self
            .graph_edge_rows
            .json_rows()?
            .into_iter()
            .filter_map(|row| graph_edge_record_from_row(&row, GraphLayer::Asserted))
            .filter(|row| row.document_id.as_deref() == Some(document_id))
            .collect::<Vec<_>>();
        if vertices.is_empty() && edges.is_empty() {
            return Ok(None);
        }
        Ok(Some(GraphMutationBatch {
            layer: GraphLayer::Asserted,
            scope: GraphMutationScope::Document {
                document_id: document_id.to_owned(),
            },
            vertices,
            edges,
        }))
    }
}

#[derive(Clone, Debug)]
struct MentionRecord {
    entity_id: EntityId,
    surface: String,
    range: TextRange,
    absolute_range: TextRange,
    confidence: f32,
    span_id: String,
    span_mention_id: String,
}

#[derive(Clone, Debug)]
struct DiscoveryRecord {
    key: String,
    chapter_id: u32,
    boundary_id: u32,
    range: TextRange,
    confidence: f32,
}

#[derive(Clone, Debug)]
struct EntityState {
    id: EntityId,
    label: String,
    kind: EntityKind,
    aliases: FxHashSet<String>,
    chapters: FxHashSet<u32>,
    boundary_ordinals: FxHashSet<u32>,
    total_mentions: u32,
}

#[derive(Default)]
struct EntityRegistry {
    entities: FxHashMap<String, EntityState>,
    surfaces: FxHashMap<String, EntityId>,
    cooccurrence: FxHashMap<(String, String, String), u32>,
    initial_mentions: u32,
    version: u64,
}

impl EntityRegistry {
    fn from_store(store: &PhoenixCozoStore) -> Result<Self, StoreError> {
        const ENTITY_COLUMNS: &[&str] = &["id", "label", "kind", "aliases", "total_mentions"];
        let mut registry = Self::default();
        for row in store.fetch_compact_rows_with_columns("entities", ENTITY_COLUMNS)? {
            let row = CompactRowView::new(ENTITY_COLUMNS, &row);
            let Some(id) = row.get_str("id") else {
                continue;
            };
            let label = row.get_str("label").unwrap_or_default().to_owned();
            let aliases = row
                .get_json("aliases")
                .and_then(|value| value.as_array().cloned())
                .map(|values| {
                    values
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_owned)
                        .collect::<FxHashSet<_>>()
                })
                .unwrap_or_default();
            let total_mentions = row.get_u64("total_mentions").unwrap_or_default() as u32;
            let entity = EntityState {
                id: EntityId(id.to_owned()),
                label: label.clone(),
                kind: kind_from_string(row.get_str("kind").unwrap_or("Other")),
                aliases: aliases.clone(),
                chapters: FxHashSet::default(),
                boundary_ordinals: FxHashSet::default(),
                total_mentions,
            };
            registry.initial_mentions += total_mentions;
            registry
                .surfaces
                .insert(normalize_key(&label), entity.id.clone());
            for alias in &aliases {
                registry
                    .surfaces
                    .insert(normalize_key(alias), entity.id.clone());
            }
            registry.entities.insert(entity.id.0.clone(), entity);
        }
        Ok(registry)
    }

    fn refresh_resolver_seed<'a>(
        &self,
        scope: &ScopeKey,
        scratch: &'a mut ResolverSeedScratch,
    ) -> &'a [ResolverEntitySeed] {
        if scratch.version != self.version {
            scratch.seeds.clear();
            scratch
                .seeds
                .extend(self.entities.values().map(|entity| ResolverEntitySeed {
                    entity_id: entity.id.clone(),
                    canonical_name: entity.label.clone(),
                    aliases: entity.aliases.iter().cloned().collect(),
                    kind: Some(entity.kind.clone()),
                    gender: Some(GenderHint::Unknown),
                    number: None,
                    scope: scope.clone(),
                }));
            scratch.version = self.version;
        }
        &scratch.seeds
    }

    fn resolve_or_register(
        &mut self,
        surface: &str,
        explicit_id: Option<&EntityId>,
        kind: Option<EntityKind>,
        chapter_id: u32,
        boundary_ordinal: u32,
    ) -> EntityId {
        if let Some(explicit_id) = explicit_id {
            let existed = self.entities.contains_key(&explicit_id.0);
            let entry = self
                .entities
                .entry(explicit_id.0.clone())
                .or_insert_with(|| EntityState {
                    id: explicit_id.clone(),
                    label: surface.to_owned(),
                    kind: kind.clone().unwrap_or(EntityKind::Other),
                    aliases: FxHashSet::default(),
                    chapters: FxHashSet::default(),
                    boundary_ordinals: FxHashSet::default(),
                    total_mentions: 0,
                });
            entry.chapters.insert(chapter_id);
            entry.boundary_ordinals.insert(boundary_ordinal);
            self.surfaces
                .insert(normalize_key(surface), explicit_id.clone());
            if !existed {
                self.version = self.version.wrapping_add(1);
            }
            return explicit_id.clone();
        }

        let normalized = normalize_key(surface);
        if let Some(existing) = self.surfaces.get(&normalized) {
            if let Some(entity) = self.entities.get_mut(&existing.0) {
                entity.chapters.insert(chapter_id);
                entity.boundary_ordinals.insert(boundary_ordinal);
            }
            return existing.clone();
        }

        let entity_id = EntityId(format!(
            "entity-{}",
            stable_hex("entity", &[normalized.as_str()])
        ));
        let mut chapters = FxHashSet::default();
        chapters.insert(chapter_id);
        let mut boundary_ordinals = FxHashSet::default();
        boundary_ordinals.insert(boundary_ordinal);
        self.entities.insert(
            entity_id.0.clone(),
            EntityState {
                id: entity_id.clone(),
                label: surface.to_owned(),
                kind: kind.unwrap_or(EntityKind::Other),
                aliases: FxHashSet::default(),
                chapters,
                boundary_ordinals,
                total_mentions: 0,
            },
        );
        self.surfaces.insert(normalized, entity_id.clone());
        self.version = self.version.wrapping_add(1);
        entity_id
    }

    fn add_alias(&mut self, entity_id: &EntityId, alias: &str) {
        if let Some(entity) = self.entities.get_mut(&entity_id.0) {
            if normalize_key(alias) == normalize_key(&entity.label) {
                return;
            }
            entity.aliases.insert(alias.to_owned());
            self.surfaces
                .insert(normalize_key(alias), entity_id.clone());
            self.version = self.version.wrapping_add(1);
        }
    }

    fn record_mention(&mut self, entity_id: &EntityId, chapter_id: u32, boundary_ordinal: u32) {
        if let Some(entity) = self.entities.get_mut(&entity_id.0) {
            entity.total_mentions += 1;
            entity.chapters.insert(chapter_id);
            entity.boundary_ordinals.insert(boundary_ordinal);
        }
    }

    fn record_cooccurrence(
        &mut self,
        document: &BorrowedIngestDocument<'_>,
        entity_ids: Vec<EntityId>,
    ) {
        let unique = entity_ids
            .into_iter()
            .map(|entity_id| entity_id.0)
            .collect::<FxHashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        for index in 0..unique.len() {
            for next in index + 1..unique.len() {
                let (left, right) = if unique[index] <= unique[next] {
                    (unique[index].clone(), unique[next].clone())
                } else {
                    (unique[next].clone(), unique[index].clone())
                };
                *self
                    .cooccurrence
                    .entry((document.document_id.0.clone(), left, right))
                    .or_insert(0) += 1;
            }
        }
    }

    fn cooccurrence_rows(&self, document: &BorrowedIngestDocument<'_>) -> Vec<Value> {
        self.cooccurrence
            .iter()
            .filter_map(|((doc_id, left, right), count)| {
                if doc_id != &document.document_id.0 {
                    return None;
                }
                Some(json!({
                    "id": stable_hex("edge", &[doc_id.as_str(), left.as_str(), right.as_str(), "cooccurs"]),
                    "source_id": left,
                    "target_id": right,
                    "rel_type": "cooccurs",
                    "confidence": *count as f64,
                    "bidirectional": true,
                    "source_note": document.note_id.as_ref().map(|value| value.0.clone()).unwrap_or_else(|| document.document_id.0.clone()),
                    "created_at": now_ms(),
                }))
            })
            .collect()
    }

    fn total_aliases(&self) -> usize {
        self.entities
            .values()
            .map(|entity| entity.aliases.len())
            .sum()
    }

    fn multi_chapter_entities(&self) -> usize {
        self.entities
            .values()
            .filter(|entity| entity.chapters.len() > 1)
            .count()
    }
}

fn build_boundary_specs(
    document: &BorrowedIngestDocument<'_>,
    markers: &[BoundaryMarker],
) -> Vec<BoundarySpec> {
    if markers.is_empty() {
        return vec![BoundarySpec {
            boundary_id: 0,
            ordinal: 0,
            kind: BoundaryKind::Chapter,
            depth: 1,
            label: "document".to_owned(),
            parent_boundary_id: None,
            start: 0,
            end: document.text.len(),
        }];
    }

    let mut boundaries = Vec::with_capacity(markers.len() + 1);
    let mut next_id = 1u32;
    let mut next_ordinal = 1u32;
    if markers[0].start > 0 && document.text[..markers[0].start].trim().len() > 0 {
        boundaries.push(BoundarySpec {
            boundary_id: 0,
            ordinal: 0,
            kind: BoundaryKind::Chapter,
            depth: 1,
            label: "front matter".to_owned(),
            parent_boundary_id: None,
            start: 0,
            end: markers[0].start,
        });
    }
    for (index, marker) in markers.iter().enumerate() {
        let end = markers
            .get(index + 1)
            .map(|next| next.start)
            .unwrap_or(document.text.len());
        let parent_boundary_id = markers[..index]
            .iter()
            .enumerate()
            .rev()
            .find(|(_, prior)| prior.depth < marker.depth && prior.start <= marker.start)
            .map(|(prior_index, _)| {
                if markers[0].start > 0 && document.text[..markers[0].start].trim().len() > 0 {
                    prior_index as u32 + 1
                } else {
                    prior_index as u32 + 1
                }
            });
        boundaries.push(BoundarySpec {
            boundary_id: next_id,
            ordinal: next_ordinal,
            kind: marker.kind.clone(),
            depth: marker.depth,
            label: marker.label.clone(),
            parent_boundary_id,
            start: marker.start,
            end,
        });
        next_id += 1;
        next_ordinal += 1;
    }
    boundaries
}

fn build_chapter_specs(
    document: &BorrowedIngestDocument<'_>,
    boundaries: &[BoundarySpec],
) -> Vec<ChapterSpec> {
    let primary = boundaries
        .iter()
        .filter(|boundary| is_primary_boundary(boundary))
        .collect::<Vec<_>>();
    if primary.is_empty() {
        return vec![ChapterSpec {
            chunk_id: stable_int(
                "chunk",
                &[
                    document.document_id.0.as_str(),
                    "2",
                    "0",
                    &document.text.len().to_string(),
                ],
            ),
            chapter_id: 0,
            boundary_id: 0,
            boundary_ordinal: 0,
            boundary_kind: BoundaryKind::Chapter,
            boundary_depth: 1,
            start: 0,
            end: document.text.len(),
            title: "document".to_owned(),
            parents: Vec::new(),
        }];
    }

    primary
        .into_iter()
        .map(|boundary| ChapterSpec {
            chunk_id: stable_int(
                "chunk",
                &[
                    document.document_id.0.as_str(),
                    "2",
                    &boundary.start.to_string(),
                    &boundary.end.to_string(),
                ],
            ),
            chapter_id: boundary.boundary_id,
            boundary_id: boundary.boundary_id,
            boundary_ordinal: boundary.ordinal,
            boundary_kind: boundary.kind.clone(),
            boundary_depth: boundary.depth,
            start: boundary.start,
            end: boundary.end,
            title: boundary.label.clone(),
            parents: Vec::new(),
        })
        .collect()
}

fn build_leaf_chunks(
    document: &BorrowedIngestDocument<'_>,
    config: &GraptorConfig,
) -> Vec<LeafChunk> {
    let sentences = split_sentences(document.text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let mut leaves = Vec::new();
    let mut window = Vec::<TextRange>::new();
    let mut current_len = 0usize;

    let emit = |window: &[TextRange], leaves: &mut Vec<LeafChunk>| {
        if window.is_empty() {
            return;
        }
        let start = window[0].start as usize;
        let end = window[window.len() - 1].end as usize;
        leaves.push(LeafChunk {
            chunk_id: stable_int(
                "chunk",
                &[
                    document.document_id.0.as_str(),
                    "0",
                    &start.to_string(),
                    &end.to_string(),
                ],
            ),
            search_id: String::new(),
            chapter_id: 0,
            boundary_id: 0,
            boundary_ordinal: 0,
            boundary_kind: BoundaryKind::Chapter,
            parent_id: None,
            start,
            end,
            message_meta: None,
        });
    };

    for sentence in sentences {
        let sentence_len = (sentence.end - sentence.start) as usize;
        if current_len > 0 && current_len + sentence_len > config.chunk_size {
            emit(&window, &mut leaves);
            let mut overlap_len = 0usize;
            let mut new_window = Vec::new();
            for span in window.iter().rev() {
                let span_len = (span.end - span.start) as usize;
                if overlap_len + span_len > config.overlap {
                    break;
                }
                overlap_len += span_len;
                new_window.push(*span);
            }
            new_window.reverse();
            window = new_window;
            current_len = overlap_len;
        }
        window.push(sentence);
        current_len += sentence_len;
    }
    emit(&window, &mut leaves);
    leaves
}

fn build_thread_document(thread: &BorrowedIngestThread<'_>) -> ThreadSyntheticDocument {
    let mut text = String::new();
    let mut messages = Vec::new();

    for (message_index, message) in thread.messages.iter().enumerate() {
        let start = text.len();
        text.push('[');
        text.push_str(message.role.trim());
        text.push_str("] ");
        text.push_str(message.content.trim());
        text.push('\n');
        let end = text.len();
        messages.push(ThreadMessageRange {
            message_id: message.message_id.to_owned(),
            role: message.role.to_owned(),
            message_index,
            start,
            end,
        });
    }

    ThreadSyntheticDocument { text, messages }
}

fn build_thread_leaf_chunks(
    document: &BorrowedIngestDocument<'_>,
    thread: &BorrowedIngestThread<'_>,
    synthetic: &ThreadSyntheticDocument,
    config: &GraptorConfig,
) -> Vec<LeafChunk> {
    if synthetic.messages.is_empty() {
        return Vec::new();
    }

    let mut leaves = Vec::new();
    let mut start_index = 0usize;

    while start_index < synthetic.messages.len() {
        let first = &synthetic.messages[start_index];
        let first_len = first.end.saturating_sub(first.start);
        let mut end_index = start_index;
        let mut current_len = 0usize;

        while end_index < synthetic.messages.len() {
            let candidate = &synthetic.messages[end_index];
            let candidate_len = candidate.end.saturating_sub(candidate.start);
            if current_len > 0 && current_len + candidate_len > config.chunk_size {
                break;
            }
            current_len += candidate_len;
            end_index += 1;
        }

        if end_index == start_index {
            let oversize = &synthetic.messages[start_index];
            leaves.extend(split_oversize_thread_message(
                document, oversize, synthetic, config,
            ));
            start_index += 1;
            continue;
        }
        if end_index == start_index + 1 && first_len > config.chunk_size {
            let oversize = &synthetic.messages[start_index];
            leaves.extend(split_oversize_thread_message(
                document, oversize, synthetic, config,
            ));
            start_index += 1;
            continue;
        }

        let last = &synthetic.messages[end_index - 1];
        leaves.push(thread_leaf_chunk(
            document,
            first.start,
            last.end,
            LeafMessageMeta {
                message_ids: thread.messages[start_index..end_index]
                    .iter()
                    .map(|message| message.message_id.to_owned())
                    .collect(),
                roles: thread.messages[start_index..end_index]
                    .iter()
                    .map(|message| message.role.to_owned())
                    .collect(),
                start_index,
                end_index: end_index - 1,
            },
        ));
        start_index = end_index;
    }

    leaves
}

fn split_oversize_thread_message(
    document: &BorrowedIngestDocument<'_>,
    message: &ThreadMessageRange,
    synthetic: &ThreadSyntheticDocument,
    config: &GraptorConfig,
) -> Vec<LeafChunk> {
    let text = &synthetic.text[message.start..message.end];
    let mut sentences = split_sentences(text);
    if sentences.is_empty() {
        sentences.push(TextRange {
            start: 0,
            end: text.len() as u32,
        });
    }

    let mut leaves = Vec::new();
    let mut window = Vec::<TextRange>::new();
    let mut current_len = 0usize;

    let emit = |window: &[TextRange], leaves: &mut Vec<LeafChunk>| {
        if window.is_empty() {
            return;
        }
        let start = message.start + window[0].start as usize;
        let end = message.start + window[window.len() - 1].end as usize;
        leaves.push(thread_leaf_chunk(
            document,
            start,
            end,
            LeafMessageMeta {
                message_ids: vec![message.message_id.clone()],
                roles: vec![message.role.clone()],
                start_index: message.message_index,
                end_index: message.message_index,
            },
        ));
    };

    for sentence in sentences {
        let sentence_len = (sentence.end - sentence.start) as usize;
        if current_len > 0 && current_len + sentence_len > config.chunk_size {
            emit(&window, &mut leaves);
            let mut overlap_len = 0usize;
            let mut new_window = Vec::new();
            for span in window.iter().rev() {
                let span_len = (span.end - span.start) as usize;
                if overlap_len + span_len > config.overlap {
                    break;
                }
                overlap_len += span_len;
                new_window.push(*span);
            }
            new_window.reverse();
            window = new_window;
            current_len = overlap_len;
        }
        window.push(sentence);
        current_len += sentence_len;
    }
    emit(&window, &mut leaves);
    leaves
}

fn thread_leaf_chunk(
    document: &BorrowedIngestDocument<'_>,
    start: usize,
    end: usize,
    message_meta: LeafMessageMeta,
) -> LeafChunk {
    let chunk_id = stable_int(
        "chunk",
        &[
            document.document_id.0.as_str(),
            "0",
            &start.to_string(),
            &end.to_string(),
        ],
    );
    LeafChunk {
        chunk_id,
        search_id: format!(
            "{}:{}:{}:{}:{}-{}",
            document.document_id.0, 0, 0, chunk_id, start, end
        ),
        chapter_id: 0,
        boundary_id: 0,
        boundary_ordinal: 0,
        boundary_kind: BoundaryKind::Chapter,
        parent_id: None,
        start,
        end,
        message_meta: Some(message_meta),
    }
}

fn assign_leaves_to_boundaries(boundaries: &[BoundarySpec], leaves: &mut [LeafChunk]) {
    for leaf in leaves {
        let mut best_boundary = boundaries
            .first()
            .expect("document should always have a boundary");
        let mut best_overlap = 0usize;
        for boundary in boundaries {
            let overlap = overlap_len(leaf.start, leaf.end, boundary.start, boundary.end);
            if overlap > best_overlap
                || (overlap == best_overlap
                    && boundary.depth >= best_boundary.depth
                    && boundary.start >= best_boundary.start)
            {
                best_overlap = overlap;
                best_boundary = boundary;
            }
        }
        leaf.boundary_id = best_boundary.boundary_id;
        leaf.boundary_ordinal = best_boundary.ordinal;
        leaf.boundary_kind = best_boundary.kind.clone();
    }
}

fn assign_leaves_to_chapters(
    document: &BorrowedIngestDocument<'_>,
    chapters: &[ChapterSpec],
    leaves: &mut [LeafChunk],
) {
    for leaf in leaves {
        let mut best_chapter = chapters
            .first()
            .map(|chapter| chapter.chapter_id)
            .unwrap_or(0);
        let mut best_overlap = 0usize;
        for chapter in chapters {
            let overlap = overlap_len(leaf.start, leaf.end, chapter.start, chapter.end);
            if overlap > best_overlap {
                best_overlap = overlap;
                best_chapter = chapter.chapter_id;
            }
        }
        leaf.chapter_id = best_chapter;
        leaf.search_id = format!(
            "{}:{}:{}:{}:{}-{}",
            document.document_id.0,
            leaf.chapter_id,
            leaf.boundary_id,
            leaf.chunk_id,
            leaf.start,
            leaf.end
        );
    }
}

fn build_parent_chunks(
    document: &BorrowedIngestDocument<'_>,
    config: &GraptorConfig,
    chapters: &mut [ChapterSpec],
    leaves: &mut [LeafChunk],
) {
    for chapter in chapters {
        let indices = leaves
            .iter()
            .enumerate()
            .filter_map(|(index, leaf)| (leaf.chapter_id == chapter.chapter_id).then_some(index))
            .collect::<Vec<_>>();
        if indices.is_empty() {
            continue;
        }
        let mut start_index = 0usize;
        while start_index < indices.len() {
            let first = &leaves[indices[start_index]];
            let mut end_index = start_index;
            let mut current_len = 0usize;
            while end_index < indices.len() {
                let leaf = &leaves[indices[end_index]];
                let leaf_len = leaf.end - leaf.start;
                if current_len > 0 && current_len + leaf_len > config.parent_chunk_size {
                    break;
                }
                current_len += leaf_len;
                end_index += 1;
            }
            let last = &leaves[indices[end_index - 1]];
            let parent_id = stable_int(
                "chunk",
                &[
                    document.document_id.0.as_str(),
                    "1",
                    &first.start.to_string(),
                    &last.end.to_string(),
                ],
            );
            chapter.parents.push(ParentChunk {
                chunk_id: parent_id,
                start: first.start,
                end: last.end,
            });
            for index in &indices[start_index..end_index] {
                leaves[*index].parent_id = Some(parent_id);
            }

            if end_index >= indices.len() {
                break;
            }
            let mut overlap_len = 0usize;
            let mut new_start = end_index;
            for index in (start_index..end_index).rev() {
                let leaf = &leaves[indices[index]];
                let leaf_len = leaf.end - leaf.start;
                if overlap_len + leaf_len > config.parent_overlap {
                    break;
                }
                overlap_len += leaf_len;
                new_start = index;
            }
            start_index = if new_start == start_index {
                end_index
            } else {
                new_start
            };
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn process_document_chunks(
    store: &PhoenixCozoStore,
    policy: FlushPolicy,
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    chapters: &[ChapterSpec],
    leaves: &[LeafChunk],
    scan_session_id: &SessionId,
    scanner: &PhoenixScanner,
    structure: &PhoenixStructure,
    registry: &mut EntityRegistry,
    resolver_scratch: &mut ResolverSeedScratch,
    buffers: &mut BufferSet,
    persist_state: &mut DocumentPersistState,
    chapter_links: &mut FxHashMap<(u32, u32), FxHashSet<String>>,
    diagnostics: &mut DiagnosticCollector,
) -> Result<(), StoreError> {
    for chapter in chapters {
        for parent in &chapter.parents {
            buffers
                .chunk_rows
                .insert_value(parent_chunk_row(document, parent, &document.scope))
                .expect("parent chunk row");
            buffers
                .chunkid_rows
                .insert_value(parent_chunkid_row(document, chapter, parent))
                .expect("parent chunk id row");
            let parent_vertex = parent_vertex_id(parent.chunk_id);
            insert_graph_vertex(buffers, parent_vertex_row(document, chapter, parent));
            insert_graph_edge(
                buffers,
                persist_state,
                graph_edge_row(
                    &chapter_vertex_id(&document.document_id, chapter.chapter_id),
                    &parent_vertex,
                    1,
                    "contains",
                    json!({
                        "kind": "contains",
                        "documentId": document.document_id.0,
                        "boundaryId": chapter.boundary_id,
                        "boundaryOrdinal": chapter.boundary_ordinal,
                        "boundaryKind": boundary_kind_str(&chapter.boundary_kind),
                        "assertionKind": "current",
                    }),
                    None,
                    Some(document.document_id.0.clone()),
                    document.scope.narrative_id.clone(),
                ),
            );
            buffers.flush_due(store, policy)?;
        }
    }

    let mut current_chapter: Option<u32> = None;
    for leaf in leaves {
        buffers
            .chunk_rows
            .insert_value(leaf_chunk_row(document, leaf, &document.scope))
            .expect("leaf chunk row");
        buffers
            .chunkid_rows
            .insert_value(leaf_chunkid_row(document, leaf))
            .expect("leaf chunk id row");
        let chapter = chapters
            .iter()
            .find(|chapter| chapter.chapter_id == leaf.chapter_id)
            .expect("leaf chapter should exist");
        let leaf_vertex = leaf_vertex_id(&leaf.search_id);
        insert_graph_vertex(buffers, leaf_vertex_row(document, note_id, chapter, leaf));
        buffers
            .graph_label_rows
            .insert_value(json!({
                "vertex_id": leaf_vertex.clone(),
                "label": leaf.search_id.clone(),
            }))
            .expect("leaf label row");
        let parent_or_chapter = leaf
            .parent_id
            .map(parent_vertex_id)
            .unwrap_or_else(|| chapter_vertex_id(&document.document_id, leaf.chapter_id));
        insert_graph_edge(
            buffers,
            persist_state,
            graph_edge_row(
                &parent_or_chapter,
                &leaf_vertex,
                1,
                "contains",
                json!({
                    "kind": "contains",
                    "documentId": document.document_id.0,
                    "boundaryId": leaf.boundary_id,
                    "boundaryOrdinal": leaf.boundary_ordinal,
                    "boundaryKind": boundary_kind_str(&leaf.boundary_kind),
                    "assertionKind": "current",
                }),
                None,
                Some(document.document_id.0.clone()),
                document.scope.narrative_id.clone(),
            ),
        );

        let leaf_text = preserve_offsets_slice(&document.text[leaf.start..leaf.end]);
        // Only rebuild the resolver seed (and thus the scanner's lexicon) at chapter boundaries.
        // This avoids rebuilding the FST + Aho-Corasick automaton per-leaf when new entities
        // are discovered, trading slight delay in recognizing new entities for a major
        // reduction in lexicon rebuild overhead.
        let chapter_crossed = current_chapter != Some(leaf.chapter_id);
        if chapter_crossed {
            current_chapter = Some(leaf.chapter_id);
            // Invalidate cached version to force seed rebuild
            resolver_scratch.version = u64::MAX;
        } else {
            // Pin version so intra-chapter entity discoveries don't trigger rebuild
            resolver_scratch.version = registry.version;
        }
        let resolver_seed = registry.refresh_resolver_seed(&document.scope, resolver_scratch);
        let scan = scanner.scan_parts(
            &leaf_text,
            &document.scope,
            Some(scan_session_id),
            resolver_seed,
        );
        let structure_artifact = structure.build_parts(&leaf_text, &scan);

        let (mentions, discoveries) =
            resolve_mentions(document, note_id, leaf, &scan, registry, chapter_links);
        persist_state.mention_count += mentions.len();
        if !mentions.is_empty() {
            diagnostics.push(Diagnostic {
                code: "PX_GRAPTOR_LEAF".to_owned(),
                message: format!(
                    "Leaf {} yielded {} entity mentions and {} relation candidates.",
                    leaf.search_id,
                    mentions.len(),
                    structure_artifact.relations.len()
                ),
            });
        }
        if !discoveries.is_empty() {
            diagnostics.push(Diagnostic {
                code: "PX_GRAPTOR_DISCOVERY".to_owned(),
                message: format!(
                    "Leaf {} produced {} speculative discovery candidates.",
                    leaf.search_id,
                    discoveries.len()
                ),
            });
        }

        for discovery in discoveries {
            buffers
                .discovery_rows
                .insert_value(discovery_row(document, &discovery))
                .expect("discovery row");
            persist_state.discovery_count += 1;
        }

        for mention in &mentions {
            persist_state.entity_ids.insert(mention.entity_id.0.clone());
            buffers
                .span_rows
                .insert_value(mention_span_row(document, note_id, mention))
                .expect("mention span row");
            buffers
                .span_mention_rows
                .insert_value(mention_span_mention_row(mention))
                .expect("span mention row");
            let entity_vertex = entity_vertex_id(&mention.entity_id);
            let entity = registry
                .entities
                .get(&mention.entity_id.0)
                .expect("entity should exist");
            insert_graph_vertex(buffers, entity_vertex_row(entity, document));
            buffers
                .graph_label_rows
                .insert_value(json!({
                    "vertex_id": entity_vertex.clone(),
                    "label": entity.label.clone(),
                }))
                .expect("entity label row");
            for alias in &entity.aliases {
                buffers
                    .graph_label_rows
                    .insert_value(json!({
                        "vertex_id": entity_vertex.clone(),
                        "label": alias.clone(),
                    }))
                    .expect("entity alias label row");
            }
            insert_graph_edge(
                buffers,
                persist_state,
                graph_edge_row(
                    &leaf_vertex,
                    &entity_vertex,
                    max(1, (mention.confidence * 100.0).round() as i64),
                    "mentions",
                    json!({
                        "confidence": mention.confidence,
                        "documentId": document.document_id.0,
                        "boundaryId": leaf.boundary_id,
                        "boundaryOrdinal": leaf.boundary_ordinal,
                        "boundaryKind": boundary_kind_str(&leaf.boundary_kind),
                        "assertionKind": "current",
                    }),
                    None,
                    Some(document.document_id.0.clone()),
                    document.scope.narrative_id.clone(),
                ),
            );
        }

        record_graph_property(
            &mut buffers.graph_property_rows,
            &leaf_vertex,
            "vertex",
            "chunk.text",
            json!(leaf_text),
            now_ms(),
        );

        apply_alias_candidates(&scan, &mentions, registry);
        for evidence in
            build_absolute_evidence(document, note_id, leaf, &structure_artifact.evidence_spans)
        {
            buffers
                .evidence_rows
                .insert_value(evidence_row(document, note_id, &evidence))
                .expect("evidence row");
        }
        materialize_relations(
            document,
            note_id,
            leaf,
            &structure_artifact.relations,
            &mentions,
            buffers,
            persist_state,
        );
        registry.record_cooccurrence(
            document,
            mentions
                .iter()
                .map(|mention| mention.entity_id.clone())
                .collect(),
        );
        buffers.flush_due(store, policy)?;
    }

    for edge in registry.cooccurrence_rows(document) {
        let left = edge["source_id"].as_str().unwrap_or_default().to_owned();
        let right = edge["target_id"].as_str().unwrap_or_default().to_owned();
        let count = edge["confidence"].as_f64().unwrap_or(1.0) as i64;
        buffers
            .edge_rows
            .insert_value(edge)
            .expect("cooccurrence edge row");
        persist_state.edge_count += 1;
        insert_graph_edge(
            buffers,
            persist_state,
            graph_edge_row(
                &entity_vertex_id(&EntityId(left.clone())),
                &entity_vertex_id(&EntityId(right.clone())),
                count,
                "cooccurs",
                json!({ "count": count, "documentId": document.document_id.0 }),
                None,
                Some(document.document_id.0.clone()),
                document.scope.narrative_id.clone(),
            ),
        );
        insert_graph_edge(
            buffers,
            persist_state,
            graph_edge_row(
                &entity_vertex_id(&EntityId(right.clone())),
                &entity_vertex_id(&EntityId(left.clone())),
                count,
                "cooccurs",
                json!({ "count": count, "documentId": document.document_id.0 }),
                None,
                Some(document.document_id.0.clone()),
                document.scope.narrative_id.clone(),
            ),
        );
    }

    for ((left, right), shared_entities) in chapter_links.iter() {
        let edge_id = stable_hex(
            "edge",
            &[
                document.document_id.0.as_str(),
                &left.to_string(),
                &right.to_string(),
                "cross_chapter",
            ],
        );
        buffers
            .edge_rows
            .insert_value(json!({
                "id": edge_id,
                "source_id": chapter_vertex_id(&document.document_id, *left),
                "target_id": chapter_vertex_id(&document.document_id, *right),
                "rel_type": "cross_chapter",
                "confidence": shared_entities.len() as f64,
                "bidirectional": true,
                "source_note": note_id.0,
                "created_at": now_ms(),
            }))
            .expect("cross chapter edge row");
        persist_state.edge_count += 1;
        insert_graph_edge(
            buffers,
            persist_state,
            graph_edge_row(
                &chapter_vertex_id(&document.document_id, *left),
                &chapter_vertex_id(&document.document_id, *right),
                shared_entities.len() as i64,
                "cross_chapter",
                json!({
                    "sharedEntityCount": shared_entities.len(),
                    "documentId": document.document_id.0,
                    "boundaryId": left,
                    "boundaryOrdinal": left,
                    "boundaryKind": "chapter",
                    "assertionKind": "current",
                }),
                Some(json!({ "sharedEntities": shared_entities })),
                Some(document.document_id.0.clone()),
                document.scope.narrative_id.clone(),
            ),
        );
    }
    persist_state.cross_chapter_links = chapter_links.len();
    buffers.flush_due(store, policy)?;
    Ok(())
}

fn build_document_manifest(
    document: &BorrowedIngestDocument<'_>,
    session_id: Option<&SessionId>,
    summary: &IngestDocumentSummary,
    boundaries: &[BoundarySpec],
    chapters: &[ChapterSpec],
    discovery_count: usize,
    now: i64,
    asserted_graph_batch: Option<&GraphMutationBatch>,
) -> Value {
    json!({
        "documentId": document.document_id.0,
        "sessionId": session_id.map(|value| value.0.clone()),
        "noteId": summary.note_id.as_ref().map(|id| id.0.clone()),
        "title": document.title,
        "scope": document.scope,
        "summary": summary,
        "discoveryCount": discovery_count,
        "boundaries": boundaries.iter().map(|boundary| {
            json!({
                "boundaryId": boundary.boundary_id,
                "ordinal": boundary.ordinal,
                "kind": boundary_kind_str(&boundary.kind),
                "depth": boundary.depth,
                "label": boundary.label,
                "parentBoundaryId": boundary.parent_boundary_id,
                "start": boundary.start,
                "end": boundary.end,
            })
        }).collect::<Vec<_>>(),
        "chapters": chapters.iter().map(|chapter| {
            json!({
                "chapterId": chapter.chapter_id,
                "boundaryId": chapter.boundary_id,
                "boundaryOrdinal": chapter.boundary_ordinal,
                "title": chapter.title,
                "start": chapter.start,
                "end": chapter.end,
                "parentCount": chapter.parents.len(),
                "parentIds": chapter.parents.iter().map(|parent| parent.chunk_id).collect::<Vec<_>>(),
            })
        }).collect::<Vec<_>>(),
        "assertedGraphBatch": asserted_graph_batch,
        "updatedAt": now,
    })
}

fn scoped_document_row(
    document: &BorrowedIngestDocument<'_>,
    payload: &Value,
    now: i64,
    native_pipeline: bool,
) -> Value {
    json!({
        "id": stable_hex("scoped_document", &[document.document_id.0.as_str()]),
        "scope_folder_id": document.scope.folder_id.clone().unwrap_or_else(|| "__root__".to_owned()),
        "narrative_id": document.scope.narrative_id.clone().unwrap_or_else(|| "__global__".to_owned()),
        "namespace": if native_pipeline { INVARANT_DOCUMENT_NAMESPACE } else { LEGACY_DOCUMENT_NAMESPACE },
        "document_key": document.document_id.0,
        "payload": payload,
        "seeded_from_scope_folder_id": document.scope.folder_id,
        "created_at": now,
        "updated_at": now,
    })
}

fn scoped_document_definition_row(
    document: &BorrowedIngestDocument<'_>,
    payload: &Value,
    now: i64,
    native_pipeline: bool,
) -> Value {
    json!({
        "id": stable_hex("scoped_definition", &[document.document_id.0.as_str(), "manifest"]),
        "narrative_id": document.scope.narrative_id.clone().unwrap_or_else(|| "__global__".to_owned()),
        "namespace": if native_pipeline { INVARANT_MANIFEST_NAMESPACE } else { LEGACY_MANIFEST_NAMESPACE },
        "definition_key": format!("document:{}", document.document_id.0),
        "payload": payload,
        "created_at": now,
        "updated_at": now,
    })
}

fn scoped_entity_field_row(
    document: &BorrowedIngestDocument<'_>,
    session_id: Option<&SessionId>,
    entity: &EntityState,
    now: i64,
) -> Value {
    json!({
        "id": stable_hex("scoped_entity", &[document.document_id.0.as_str(), entity.id.0.as_str()]),
        "entity_id": entity.id.0,
        "scope_folder_id": document.scope.folder_id.clone().unwrap_or_else(|| "__root__".to_owned()),
        "narrative_id": document.scope.narrative_id.clone().unwrap_or_else(|| "__global__".to_owned()),
        "field_key": "graptor.registry",
        "value_json": {
            "sessionId": session_id.map(|value| value.0.clone()),
            "documentId": document.document_id.0,
            "label": entity.label,
            "kind": kind_to_string(&entity.kind),
            "aliases": entity.aliases.iter().cloned().collect::<Vec<_>>(),
            "chapters": entity.chapters.iter().copied().collect::<Vec<_>>(),
            "boundaryOrdinals": entity.boundary_ordinals.iter().copied().collect::<Vec<_>>(),
            "totalMentions": entity.total_mentions,
        },
        "seeded_from_scope_folder_id": document.scope.folder_id,
        "created_at": now,
        "updated_at": now,
    })
}

fn discovery_row_id(document: &BorrowedIngestDocument<'_>, discovery: &DiscoveryRecord) -> String {
    stable_hex(
        "discovery",
        &[
            document.document_id.0.as_str(),
            discovery.key.as_str(),
            &discovery.chapter_id.to_string(),
            &discovery.boundary_id.to_string(),
            &discovery.range.start.to_string(),
            &discovery.range.end.to_string(),
        ],
    )
}

fn discovery_row(document: &BorrowedIngestDocument<'_>, discovery: &DiscoveryRecord) -> Value {
    json!({
        "token": discovery_row_id(document, discovery),
        "kind": 0,
        "score": discovery.confidence as f64,
        "status": 1,
        "last_seen": now_ms(),
        "first_seen": now_ms(),
        "count": 1,
    })
}

fn record_graph_json_properties(
    rows: &mut CompactRelationBuffer,
    owner_id: &str,
    owner_type: &str,
    prefix: &str,
    value: &Value,
    now: i64,
) {
    // Store as a single JSON blob instead of recursing into sub-keys.
    // Child properties are queryable via the parent blob, avoiding
    // amplification that generated ~6-12 rows per vertex/edge.
    record_graph_property(rows, owner_id, owner_type, prefix, value.clone(), now);
}

fn graph_metadata_json(
    layer: &str,
    resolver: &str,
    confidence: f64,
    evidence_refs: Vec<String>,
) -> Value {
    json!({
        "layer": layer,
        "status": "asserted",
        "resolver": resolver,
        "confidence": confidence,
        "evidence_refs": evidence_refs,
    })
}

fn ensure_graph_metadata(
    attributes: &mut Map<String, Value>,
    layer: &str,
    resolver: &str,
    confidence: f64,
    evidence_refs: Vec<String>,
) {
    if !attributes.contains_key("graph") {
        attributes.insert(
            "graph".to_owned(),
            graph_metadata_json(layer, resolver, confidence, evidence_refs),
        );
    }
}

fn row_document_id(row: &Map<String, Value>) -> Option<String> {
    row.get("document_id")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            row.get("attributes")
                .and_then(Value::as_object)
                .and_then(|attributes| attributes.get("documentId"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .or_else(|| {
            row.get("value")
                .and_then(Value::as_object)
                .and_then(|value| value.get("documentId"))
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

fn row_boundary_ref(row: &Map<String, Value>, document_id: Option<&str>) -> Option<String> {
    let attributes = row.get("attributes").and_then(Value::as_object)?;
    let boundary_id = attributes.get("boundaryId").and_then(Value::as_u64)?;
    let boundary_doc = attributes
        .get("documentId")
        .and_then(Value::as_str)
        .or(document_id)
        .unwrap_or_default();
    Some(format!("boundary:{boundary_doc}:{boundary_id}"))
}

fn collect_vertex_evidence_refs(row: &Map<String, Value>) -> Vec<String> {
    let mut refs = Vec::new();
    if let Some(vertex_id) = row.get("id").and_then(Value::as_str) {
        refs.push(format!("graph_vertex:{vertex_id}"));
    }
    let document_id = row_document_id(row);
    if let Some(document_id) = document_id.as_deref() {
        refs.push(format!("document:{document_id}"));
    }
    if let Some(boundary_ref) = row_boundary_ref(row, document_id.as_deref()) {
        refs.push(boundary_ref);
    }
    if let Some(search_chunk_id) = row
        .get("value")
        .and_then(Value::as_object)
        .and_then(|value| value.get("searchChunkId"))
        .and_then(Value::as_str)
    {
        refs.push(format!("leaf:{search_chunk_id}"));
    }
    if let Some(entity_id) = row
        .get("value")
        .and_then(Value::as_object)
        .and_then(|value| value.get("entityId"))
        .and_then(Value::as_str)
    {
        refs.push(format!("entity:{entity_id}"));
    }
    refs
}

fn collect_edge_evidence_refs(row: &Map<String, Value>) -> Vec<String> {
    let mut refs = Vec::new();
    let source_id = row
        .get("source_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let target_id = row
        .get("target_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let edge_type = row
        .get("edge_type")
        .and_then(Value::as_str)
        .unwrap_or("edge");
    refs.push(format!("graph_edge:{source_id}->{target_id}::{edge_type}"));
    if !source_id.is_empty() {
        refs.push(format!("graph_vertex:{source_id}"));
    }
    if !target_id.is_empty() {
        refs.push(format!("graph_vertex:{target_id}"));
    }
    let document_id = row_document_id(row);
    if let Some(document_id) = document_id.as_deref() {
        refs.push(format!("document:{document_id}"));
    }
    if let Some(boundary_ref) = row_boundary_ref(row, document_id.as_deref()) {
        refs.push(boundary_ref);
    }
    refs
}

fn row_confidence(row: &Map<String, Value>) -> f64 {
    row.get("attributes")
        .and_then(Value::as_object)
        .and_then(|attributes| attributes.get("confidence"))
        .and_then(Value::as_f64)
        .or_else(|| {
            row.get("data")
                .and_then(Value::as_object)
                .and_then(|data| data.get("confidence"))
                .and_then(Value::as_f64)
        })
        .unwrap_or(1.0)
}

fn graph_vertex_record_from_row(row: &Value) -> Option<GraphVertexRecord> {
    let object = row.as_object()?;
    let id = object.get("id")?.as_str()?.to_owned();
    let value = object.get("value").cloned().unwrap_or(Value::Null);
    let attributes = object.get("attributes").cloned().unwrap_or(Value::Null);
    let document_id = row_document_id(object);
    let chapters = attributes
        .get("chapters")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_u64)
                .map(|value| value as u32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let boundary_ordinals = attributes
        .get("boundaryOrdinals")
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_u64)
                .map(|value| value as u32)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    Some(GraphVertexRecord {
        id,
        kind: value
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned(),
        weight: object.get("weight").and_then(Value::as_i64).unwrap_or(1),
        value: value.clone(),
        attributes: attributes.clone(),
        entity_id: value
            .get("entityId")
            .and_then(Value::as_str)
            .map(str::to_owned),
        search_chunk_id: value
            .get("searchChunkId")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                attributes
                    .get("searchChunkId")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }),
        document_id,
        chapter_id: attributes
            .get("chapterId")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        chapters,
        boundary_id: attributes
            .get("boundaryId")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        boundary_ordinal: attributes
            .get("boundaryOrdinal")
            .and_then(Value::as_u64)
            .map(|value| value as u32),
        boundary_kind: attributes
            .get("boundaryKind")
            .and_then(Value::as_str)
            .map(boundary_kind_from_str),
        boundary_ordinals,
    })
}

fn graph_edge_record_from_row(row: &Value, layer: GraphLayer) -> Option<GraphEdgeRecord> {
    let object = row.as_object()?;
    Some(GraphEdgeRecord {
        source_id: object.get("source_id")?.as_str()?.to_owned(),
        target_id: object.get("target_id")?.as_str()?.to_owned(),
        edge_type: object
            .get("edge_type")
            .and_then(Value::as_str)
            .unwrap_or("edge")
            .to_owned(),
        weight: object.get("weight").and_then(Value::as_i64).unwrap_or(1),
        attributes: object.get("attributes").cloned().unwrap_or(Value::Null),
        data: object.get("data").cloned().filter(|value| !value.is_null()),
        document_id: object
            .get("document_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                object
                    .get("attributes")
                    .and_then(Value::as_object)
                    .and_then(|attributes| attributes.get("documentId"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }),
        narrative_id: object
            .get("narrative_id")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .or_else(|| {
                object
                    .get("attributes")
                    .and_then(Value::as_object)
                    .and_then(|attributes| attributes.get("narrativeId"))
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            }),
        layer,
    })
}

fn graph_row_boundary_ordinal(row: &Value) -> Option<i64> {
    row.get("attributes")
        .and_then(Value::as_object)
        .and_then(|attributes| {
            attributes
                .get("boundaryOrdinal")
                .and_then(Value::as_i64)
                .or_else(|| attributes.get("chapterId").and_then(Value::as_i64))
        })
        .or_else(|| {
            row.get("value")
                .and_then(Value::as_object)
                .and_then(|value| {
                    value
                        .get("boundaryOrdinal")
                        .and_then(Value::as_i64)
                        .or_else(|| value.get("chapterId").and_then(Value::as_i64))
                })
        })
}

fn record_graph_property(
    rows: &mut CompactRelationBuffer,
    owner_id: &str,
    owner_type: &str,
    key: &str,
    value: Value,
    now: i64,
) {
    let txn_id = stable_int("graph_property", &[owner_id, owner_type, key]);
    let mut row = serde_json::Map::new();
    row.insert("owner_id".to_owned(), json!(owner_id));
    row.insert("owner_type".to_owned(), json!(owner_type));
    row.insert("key".to_owned(), json!(key));
    row.insert("valid_from".to_owned(), json!(now));
    row.insert("value_type".to_owned(), json!(graph_value_type(&value)));
    row.insert("value_blob".to_owned(), value);
    row.insert("valid_until".to_owned(), Value::Null);
    row.insert("txn_id".to_owned(), json!(txn_id));
    rows.insert_value(Value::Object(row))
        .expect("graph property row");
}

fn graph_value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn flush_relation_if_needed(
    store: &PhoenixCozoStore,
    rows: &mut CompactRelationBuffer,
    limit: usize,
) -> Result<(), StoreError> {
    if let Some(drained) = rows.drain_if_len_ge(limit) {
        store.put_compact_rows_owned(rows.relation(), drained)?;
    }
    Ok(())
}

fn flush_relation_all(
    store: &PhoenixCozoStore,
    rows: &mut CompactRelationBuffer,
) -> Result<(), StoreError> {
    if !rows.is_empty() {
        store.put_compact_rows_owned(rows.relation(), rows.drain_all())?;
    }
    Ok(())
}

fn insert_graph_vertex(buffers: &mut BufferSet, row: Value) {
    let mut row = row;
    if let Some(object) = row.as_object_mut() {
        if !object.contains_key("document_id") {
            if let Some(document_id) = object
                .get("attributes")
                .and_then(Value::as_object)
                .and_then(|attributes| attributes.get("documentId"))
                .cloned()
                .or_else(|| {
                    object
                        .get("value")
                        .and_then(Value::as_object)
                        .and_then(|value| value.get("documentId"))
                        .cloned()
                })
            {
                object.insert("document_id".to_owned(), document_id);
            }
        }
        if !object.contains_key("narrative_id") {
            if let Some(narrative_id) = object
                .get("attributes")
                .and_then(Value::as_object)
                .and_then(|attributes| attributes.get("narrativeId"))
                .cloned()
                .or_else(|| {
                    object
                        .get("attributes")
                        .and_then(Value::as_object)
                        .and_then(|attributes| attributes.get("scope"))
                        .and_then(Value::as_object)
                        .and_then(|scope| scope.get("narrativeId"))
                        .cloned()
                })
            {
                object.insert("narrative_id".to_owned(), narrative_id);
            }
        }
        let confidence = row_confidence(object);
        let evidence_refs = collect_vertex_evidence_refs(object);
        let attributes = object
            .entry("attributes".to_owned())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("graph vertex attributes object");
        ensure_graph_metadata(
            attributes,
            "asserted",
            "phoenix-graptor",
            confidence,
            evidence_refs,
        );
    }
    let now = now_ms();
    let valid_from = graph_row_boundary_ordinal(&row).unwrap_or(now);
    if let Some(vertex_id) = row.get("id").and_then(Value::as_str) {
        if let Some(weight) = row.get("weight").cloned() {
            record_graph_property(
                &mut buffers.graph_property_rows,
                vertex_id,
                "vertex",
                "weight",
                weight,
                valid_from,
            );
        }
        if let Some(value) = row.get("value").filter(|value| !value.is_null()) {
            record_graph_json_properties(
                &mut buffers.graph_property_rows,
                vertex_id,
                "vertex",
                "value",
                value,
                valid_from,
            );
        }
        if let Some(attributes) = row.get("attributes").filter(|value| !value.is_null()) {
            record_graph_json_properties(
                &mut buffers.graph_property_rows,
                vertex_id,
                "vertex",
                "attributes",
                attributes,
                valid_from,
            );
        }
    }
    buffers
        .graph_vertex_rows
        .insert_value(row)
        .expect("graph vertex row");
}

fn insert_graph_edge(
    buffers: &mut BufferSet,
    persist_state: &mut DocumentPersistState,
    row: Value,
) {
    let mut row = row;
    if let Some(object) = row.as_object_mut() {
        if !object.contains_key("document_id") {
            if let Some(document_id) = object
                .get("attributes")
                .and_then(Value::as_object)
                .and_then(|attributes| attributes.get("documentId"))
                .cloned()
            {
                object.insert("document_id".to_owned(), document_id);
            }
        }
        if !object.contains_key("narrative_id") {
            if let Some(narrative_id) = object
                .get("attributes")
                .and_then(Value::as_object)
                .and_then(|attributes| attributes.get("narrativeId"))
                .cloned()
            {
                object.insert("narrative_id".to_owned(), narrative_id);
            }
        }
        let confidence = row_confidence(object);
        let evidence_refs = collect_edge_evidence_refs(object);
        let attributes = object
            .entry("attributes".to_owned())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .expect("graph edge attributes object");
        ensure_graph_metadata(
            attributes,
            "asserted",
            "phoenix-graptor",
            confidence,
            evidence_refs,
        );
    }
    let now = now_ms();
    let valid_from = graph_row_boundary_ordinal(&row).unwrap_or(now);
    let source_id = row
        .get("source_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let target_id = row
        .get("target_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let edge_type = row
        .get("edge_type")
        .and_then(Value::as_str)
        .unwrap_or("edge");
    let owner_id = format!("{source_id}->{target_id}::{edge_type}");
    if let Some(weight) = row.get("weight").cloned() {
        record_graph_property(
            &mut buffers.graph_property_rows,
            &owner_id,
            "edge",
            "weight",
            weight,
            valid_from,
        );
    }
    record_graph_property(
        &mut buffers.graph_property_rows,
        &owner_id,
        "edge",
        "edge_type",
        json!(edge_type),
        valid_from,
    );
    if let Some(attributes) = row.get("attributes").filter(|value| !value.is_null()) {
        record_graph_json_properties(
            &mut buffers.graph_property_rows,
            &owner_id,
            "edge",
            "attributes",
            attributes,
            valid_from,
        );
    }
    if let Some(data) = row.get("data").filter(|value| !value.is_null()) {
        record_graph_json_properties(
            &mut buffers.graph_property_rows,
            &owner_id,
            "edge",
            "data",
            data,
            valid_from,
        );
    }
    buffers
        .graph_edge_rows
        .insert_value(row)
        .expect("graph edge row");
    persist_state.graph_edge_count += 1;
}

fn persist_session_workspace_artifact(
    store: &PhoenixCozoStore,
    session_id: &SessionId,
    kind: &str,
    payload: &Value,
    narrative_id: Option<&str>,
    folder_id: Option<&str>,
    now: i64,
) -> Result<(), StoreError> {
    let producer = if kind.starts_with("invarant_") {
        "invarant"
    } else {
        "graptor"
    };
    store.put_row(
        "workspace_artifacts",
        json!({
            "key": format!("{producer}:{}:{}", session_id.0, kind),
            "thread_id": session_id.0,
            "narrative_id": narrative_id.unwrap_or("__global__"),
            "folder_id": folder_id.unwrap_or("__root__"),
            "kind": kind,
            "payload": payload,
            "pinned": false,
            "produced_by": producer,
            "created_at": now,
            "updated_at": now,
        }),
    )
}

fn persist_session_definition(
    store: &PhoenixCozoStore,
    session_id: &SessionId,
    kind: &str,
    payload: &Value,
    narrative_id: Option<&str>,
    now: i64,
    native_pipeline: bool,
) -> Result<(), StoreError> {
    store.put_row(
        "scoped_definitions",
        json!({
            "id": stable_hex("session_definition", &[session_id.0.as_str(), kind]),
            "narrative_id": narrative_id.unwrap_or("__global__"),
            "namespace": if native_pipeline { INVARANT_SESSION_NAMESPACE } else { LEGACY_SESSION_NAMESPACE },
            "definition_key": format!("{}:{}", kind, session_id.0),
            "payload": payload,
            "created_at": now,
            "updated_at": now,
        }),
    )
}

fn persist_document_backbone(
    store: &PhoenixCozoStore,
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    now: i64,
) -> Result<(), StoreError> {
    store.put_rows(
        "notes",
        &[json!({
            "id": note_id.0,
            "version": 1,
            "world_id": document.scope.world_id.clone().unwrap_or_default(),
            "title": document.title,
            "content": document.text,
            "markdown_content": document.text,
            "folder_id": document.scope.folder_id,
            "entity_kind": null,
            "entity_subtype": null,
            "is_entity": false,
            "is_pinned": false,
            "favorite": false,
            "owner_id": document.document_id.0,
            "narrative_id": document.scope.narrative_id,
            "order": null,
            "created_at": now,
            "updated_at": now,
            "valid_from": now,
            "valid_to": null,
            "is_current": true,
            "change_reason": "phoenix_graptor_ingest",
        })],
    )?;
    store.put_rows(
        "docid_map",
        &[json!({
            "id": stable_int("docid", &[document.document_id.0.as_str()]),
            "docid": document.document_id.0,
            "created_at": now,
        })],
    )?;
    Ok(())
}

fn persist_entity_rows(
    store: &PhoenixCozoStore,
    document: &BorrowedIngestDocument<'_>,
    session_id: Option<&SessionId>,
    note_id: &NoteId,
    summary: &IngestDocumentSummary,
    chapters: &[ChapterSpec],
    boundaries: &[BoundarySpec],
    registry: &EntityRegistry,
    persist_state: &DocumentPersistState,
    now: i64,
    asserted_graph_batch: Option<&GraphMutationBatch>,
) -> Result<(), StoreError> {
    let entity_rows = persist_state
        .entity_ids
        .iter()
        .map(|entity_id| {
            let entity = registry
                .entities
                .get(entity_id)
                .expect("entity should exist");
            entity_row(entity, &document.scope, note_id, now)
        })
        .collect::<Vec<_>>();
    store.put_rows("entities", &entity_rows)?;

    let document_manifest = build_document_manifest(
        document,
        session_id,
        summary,
        boundaries,
        chapters,
        persist_state.discovery_count,
        now,
        asserted_graph_batch,
    );
    let native_pipeline = asserted_graph_batch.is_some();
    store.put_rows(
        "scoped_documents",
        &[scoped_document_row(
            document,
            &document_manifest,
            now,
            native_pipeline,
        )],
    )?;
    store.put_rows(
        "scoped_definitions",
        &[scoped_document_definition_row(
            document,
            &document_manifest,
            now,
            native_pipeline,
        )],
    )?;
    let scoped_entity_rows = persist_state
        .entity_ids
        .iter()
        .map(|entity_id| {
            let entity = registry
                .entities
                .get(entity_id)
                .expect("entity should exist");
            scoped_entity_field_row(document, session_id, entity, now)
        })
        .collect::<Vec<_>>();
    store.put_rows("scoped_entity_fields", &scoped_entity_rows)?;
    Ok(())
}

fn resolve_mentions(
    _document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    leaf: &LeafChunk,
    scan: &phoenix_types::ScanArtifact,
    registry: &mut EntityRegistry,
    chapter_links: &mut FxHashMap<(u32, u32), FxHashSet<String>>,
) -> (Vec<MentionRecord>, Vec<DiscoveryRecord>) {
    let mut mentions = Vec::new();
    let mut discoveries = Vec::new();
    for mention in &scan.mentions {
        if mention.source == Some(MentionSource::Discovery)
            || matches!(mention.entity_ref, Some(MentionEntityRef::Speculative(_)))
        {
            discoveries.push(DiscoveryRecord {
                key: match mention.entity_ref.as_ref() {
                    Some(MentionEntityRef::Speculative(key)) => key.clone(),
                    _ => normalize_key(&mention.surface),
                },
                chapter_id: leaf.chapter_id,
                boundary_id: leaf.boundary_id,
                range: TextRange {
                    start: leaf.start as u32 + mention.range.start,
                    end: leaf.start as u32 + mention.range.end,
                },
                confidence: mention.confidence,
            });
            continue;
        }
        let explicit_id = match mention.entity_ref.as_ref() {
            Some(MentionEntityRef::Known(entity_id)) => Some(entity_id),
            _ => None,
        };
        let entity_id = registry.resolve_or_register(
            &mention.surface,
            explicit_id,
            mention.kind.clone(),
            leaf.chapter_id,
            leaf.boundary_ordinal,
        );
        registry.record_mention(&entity_id, leaf.chapter_id, leaf.boundary_ordinal);
        if let Some(entity) = registry.entities.get(&entity_id.0) {
            for chapter_id in entity
                .chapters
                .iter()
                .copied()
                .filter(|chapter| *chapter != leaf.chapter_id)
            {
                let key = if chapter_id < leaf.chapter_id {
                    (chapter_id, leaf.chapter_id)
                } else {
                    (leaf.chapter_id, chapter_id)
                };
                chapter_links
                    .entry(key)
                    .or_default()
                    .insert(entity.label.clone());
            }
        }
        let absolute_range = TextRange {
            start: leaf.start as u32 + mention.range.start,
            end: leaf.start as u32 + mention.range.end,
        };
        let span_id = stable_hex(
            "span",
            &[
                note_id.0.as_str(),
                entity_id.0.as_str(),
                &leaf.chapter_id.to_string(),
                &absolute_range.start.to_string(),
                &absolute_range.end.to_string(),
                mention.surface.as_str(),
            ],
        );
        mentions.push(MentionRecord {
            entity_id,
            surface: mention.surface.clone(),
            range: mention.range,
            absolute_range,
            confidence: mention.confidence,
            span_id: span_id.clone(),
            span_mention_id: stable_hex(
                "spanmention",
                &[span_id.as_str(), mention.surface.as_str()],
            ),
        });
    }
    (mentions, discoveries)
}

fn apply_alias_candidates(
    scan: &phoenix_types::ScanArtifact,
    mentions: &[MentionRecord],
    registry: &mut EntityRegistry,
) {
    for link in &scan.resolver_links {
        if link.link_kind != Some(ResolverLinkKind::AliasCandidate) {
            continue;
        }
        let Some(source) = mentions
            .iter()
            .find(|mention| overlaps(mention.range, link.source_range))
        else {
            continue;
        };
        let target = match link.target_entity.as_ref() {
            Some(MentionEntityRef::Known(entity_id)) => Some(entity_id.clone()),
            _ => mentions
                .iter()
                .find(|mention| {
                    link.target_range
                        .map(|range| overlaps(mention.range, range))
                        .unwrap_or(false)
                })
                .map(|mention| mention.entity_id.clone()),
        };
        if let Some(target) = target {
            registry.add_alias(&target, &source.surface);
        }
    }
}

fn build_absolute_evidence(
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    leaf: &LeafChunk,
    evidence: &[EvidenceSpan],
) -> Vec<EvidenceSpan> {
    evidence
        .iter()
        .map(|span| EvidenceSpan {
            document_id: Some(document.document_id.clone()),
            note_id: Some(note_id.clone()),
            label: span.label.clone(),
            kind: span.kind.clone(),
            range: TextRange {
                start: leaf.start as u32 + span.range.start,
                end: leaf.start as u32 + span.range.end,
            },
        })
        .collect()
}

fn materialize_relations(
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    leaf: &LeafChunk,
    relations: &[RelationCandidate],
    mentions: &[MentionRecord],
    buffers: &mut BufferSet,
    persist_state: &mut DocumentPersistState,
) {
    for relation in relations {
        let event_id = stable_hex(
            "event",
            &[
                document.document_id.0.as_str(),
                leaf.search_id.as_str(),
                relation.lemma.as_str(),
                &relation.verb_range.start.to_string(),
            ],
        );
        insert_graph_vertex(
            buffers,
            json!({
                "id": event_id.clone(),
                "document_id": document.document_id.0,
                "narrative_id": document.scope.narrative_id,
                "value": {
                    "kind": "event",
                    "lemma": relation.lemma,
                    "eventClass": relation.event_class,
                    "relationType": relation.relation_type,
                },
                "weight": 1,
                "attributes": {
                    "documentId": document.document_id.0,
                    "noteId": note_id.0,
                    "chapterId": leaf.chapter_id,
                    "boundaryId": leaf.boundary_id,
                    "boundaryOrdinal": leaf.boundary_ordinal,
                    "boundaryKind": boundary_kind_str(&leaf.boundary_kind),
                    "searchChunkId": leaf.search_id,
                    "verbRange": relation.verb_range,
                },
            }),
        );
        buffers
            .graph_label_rows
            .insert_value(json!({
                "vertex_id": event_id.clone(),
                "label": relation.lemma.clone(),
            }))
            .expect("event lemma label row");
        buffers
            .graph_label_rows
            .insert_value(json!({
                "vertex_id": event_id.clone(),
                "label": relation.relation_type.clone(),
            }))
            .expect("event relation label row");
        insert_graph_edge(
            buffers,
            persist_state,
            graph_edge_row(
                &leaf_vertex_id(&leaf.search_id),
                &event_id,
                1,
                "has_event",
                json!({
                    "kind": "event",
                    "documentId": document.document_id.0,
                    "boundaryId": leaf.boundary_id,
                    "boundaryOrdinal": leaf.boundary_ordinal,
                    "boundaryKind": boundary_kind_str(&leaf.boundary_kind),
                    "assertionKind": "current",
                }),
                None,
                Some(document.document_id.0.clone()),
                document.scope.narrative_id.clone(),
            ),
        );

        if let Some(subject_id) = resolve_slot_entity(relation.subject.as_ref(), mentions) {
            insert_graph_edge(
                buffers,
                persist_state,
                graph_edge_row(
                    &entity_vertex_id(&subject_id),
                    &event_id,
                    100,
                    "event_subject",
                    json!({
                        "role": "subject",
                        "documentId": document.document_id.0,
                        "boundaryId": leaf.boundary_id,
                        "boundaryOrdinal": leaf.boundary_ordinal,
                        "boundaryKind": boundary_kind_str(&leaf.boundary_kind),
                        "assertionKind": "current",
                    }),
                    None,
                    Some(document.document_id.0.clone()),
                    document.scope.narrative_id.clone(),
                ),
            );
            if let Some(object_id) = resolve_slot_entity(relation.object.as_ref(), mentions) {
                insert_graph_edge(
                    buffers,
                    persist_state,
                    graph_edge_row(
                        &event_id,
                        &entity_vertex_id(&object_id),
                        100,
                        "event_object",
                        json!({
                            "role": "object",
                            "documentId": document.document_id.0,
                            "boundaryId": leaf.boundary_id,
                            "boundaryOrdinal": leaf.boundary_ordinal,
                            "boundaryKind": boundary_kind_str(&leaf.boundary_kind),
                            "assertionKind": "current",
                        }),
                        None,
                        Some(document.document_id.0.clone()),
                        document.scope.narrative_id.clone(),
                    ),
                );
                let edge_id = stable_hex(
                    "edge",
                    &[
                        document.document_id.0.as_str(),
                        subject_id.0.as_str(),
                        object_id.0.as_str(),
                        relation.relation_type.as_str(),
                        "object",
                        event_id.as_str(),
                    ],
                );
                buffers
                    .edge_rows
                    .insert_value(json!({
                        "id": edge_id,
                        "source_id": subject_id.0,
                        "target_id": object_id.0,
                        "rel_type": format!("{}:object", relation.relation_type),
                        "confidence": 0.95,
                        "bidirectional": false,
                        "source_note": note_id.0,
                        "created_at": now_ms(),
                    }))
                    .expect("relation object edge row");
                persist_state.edge_count += 1;
            }
            if let Some(recipient_id) = resolve_slot_entity(relation.recipient.as_ref(), mentions) {
                insert_graph_edge(
                    buffers,
                    persist_state,
                    graph_edge_row(
                        &event_id,
                        &entity_vertex_id(&recipient_id),
                        90,
                        "event_recipient",
                        json!({
                            "role": "recipient",
                            "documentId": document.document_id.0,
                            "boundaryId": leaf.boundary_id,
                            "boundaryOrdinal": leaf.boundary_ordinal,
                            "boundaryKind": boundary_kind_str(&leaf.boundary_kind),
                            "assertionKind": "current",
                        }),
                        None,
                        Some(document.document_id.0.clone()),
                        document.scope.narrative_id.clone(),
                    ),
                );
                let edge_id = stable_hex(
                    "edge",
                    &[
                        document.document_id.0.as_str(),
                        subject_id.0.as_str(),
                        recipient_id.0.as_str(),
                        relation.relation_type.as_str(),
                        "recipient",
                        event_id.as_str(),
                    ],
                );
                buffers
                    .edge_rows
                    .insert_value(json!({
                        "id": edge_id,
                        "source_id": subject_id.0,
                        "target_id": recipient_id.0,
                        "rel_type": format!("{}:recipient", relation.relation_type),
                        "confidence": 0.9,
                        "bidirectional": false,
                        "source_note": note_id.0,
                        "created_at": now_ms(),
                    }))
                    .expect("relation recipient edge row");
                persist_state.edge_count += 1;
            }
        }

        for evidence in &relation.evidence {
            let absolute = EvidenceSpan {
                document_id: Some(document.document_id.clone()),
                note_id: Some(note_id.clone()),
                label: evidence.label.clone(),
                kind: evidence.kind.clone(),
                range: TextRange {
                    start: leaf.start as u32 + evidence.range.start,
                    end: leaf.start as u32 + evidence.range.end,
                },
            };
            buffers
                .evidence_rows
                .insert_value(evidence_row(document, note_id, &absolute))
                .expect("relation evidence row");
        }
    }
}

fn resolve_slot_entity(slot: Option<&FrameSlot>, mentions: &[MentionRecord]) -> Option<EntityId> {
    let slot = slot?;
    match slot.entity_ref.as_ref() {
        Some(MentionEntityRef::Known(entity_id)) => Some(entity_id.clone()),
        _ => mentions
            .iter()
            .find(|mention| overlaps(mention.range, slot.range))
            .map(|mention| mention.entity_id.clone()),
    }
}

fn split_sentences(text: &str) -> Vec<TextRange> {
    let mut sentences = split_sentence_ranges(text)
        .into_iter()
        .map(|(start, end)| TextRange {
            start: start as u32,
            end: end as u32,
        })
        .collect::<Vec<_>>();
    if sentences.is_empty() && !text.is_empty() {
        sentences.push(TextRange {
            start: 0,
            end: text.len() as u32,
        });
    }
    sentences
}

fn boundary_strategy_parts(strategy: &BoundaryDetectionStrategy) -> (Vec<String>, u8) {
    match strategy {
        BoundaryDetectionStrategy::Disabled => (Vec::new(), 0),
        BoundaryDetectionStrategy::Keywords { keywords } => (keywords.clone(), 0),
        BoundaryDetectionStrategy::MarkdownHeadings { max_depth } => (Vec::new(), *max_depth),
        BoundaryDetectionStrategy::Both {
            keywords,
            max_depth,
        } => (keywords.clone(), *max_depth),
    }
}

fn is_primary_boundary(boundary: &BoundarySpec) -> bool {
    matches!(
        boundary.kind,
        BoundaryKind::Chapter | BoundaryKind::Section | BoundaryKind::Act
    ) || matches!(boundary.kind, BoundaryKind::Heading) && boundary.depth <= 1
}

fn scan_heading_boundaries(text: &str, max_depth: u8) -> Vec<BoundaryMarker> {
    if max_depth == 0 {
        return Vec::new();
    }
    let mut boundaries = Vec::new();
    let bytes = text.as_bytes();
    let mut line_start = 0usize;
    for (offset, byte) in bytes.iter().enumerate() {
        if *byte != b'\n' && offset + 1 != bytes.len() {
            continue;
        }
        let line_end = if *byte == b'\n' { offset } else { bytes.len() };
        let line = text.get(line_start..line_end).unwrap_or_default();
        if let Some((kind, depth)) = validate_heading_line(line, max_depth) {
            boundaries.push(BoundaryMarker {
                start: line_start,
                kind,
                depth,
                label: line.trim().to_owned(),
            });
        }
        line_start = offset + 1;
    }
    boundaries
}

fn validate_chapter_line(line: &str, max_heading_depth: u8) -> Option<(BoundaryKind, u8)> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Some((kind, depth)) = validate_heading_line(trimmed, max_heading_depth) {
        return Some((kind, depth));
    }
    let lower = trimmed.to_ascii_lowercase();
    if lower.starts_with("chapter")
        || lower.starts_with("part")
        || lower.starts_with("section")
        || lower.starts_with("introduction")
        || lower.starts_with("conclusion")
        || lower.starts_with("summary")
        || lower.starts_with("appendix")
    {
        return Some((BoundaryKind::Chapter, 1));
    }
    if validate_numbered_section(trimmed) {
        return Some((BoundaryKind::Section, 2));
    }
    None
}

fn validate_heading_line(line: &str, max_heading_depth: u8) -> Option<(BoundaryKind, u8)> {
    if max_heading_depth == 0 {
        return None;
    }
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let depth = trimmed.bytes().take_while(|byte| *byte == b'#').count();
    if depth == 0 || depth > max_heading_depth as usize {
        return None;
    }
    let rest = trimmed[depth..].trim();
    if rest.is_empty() {
        return None;
    }
    Some((BoundaryKind::Heading, depth as u8))
}

fn validate_numbered_section(line: &str) -> bool {
    let trimmed = line.trim_start();
    if trimmed.is_empty() || !trimmed.as_bytes()[0].is_ascii_digit() {
        return false;
    }
    let mut index = 0usize;
    let bytes = trimmed.as_bytes();
    while index < bytes.len() && (bytes[index].is_ascii_digit() || bytes[index] == b'.') {
        index += 1;
    }
    index < bytes.len() && matches!(bytes[index], b' ' | b'\t' | b':' | b')')
}

fn document_boundary_row(
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    boundary: &BoundarySpec,
) -> Value {
    json!({
        "doc_id": document.document_id.0,
        "boundary_id": boundary.boundary_id,
        "kind": boundary_kind_str(&boundary.kind),
        "depth": boundary.depth,
        "label": boundary.label,
        "ordinal": boundary.ordinal,
        "parent_boundary_id": boundary.parent_boundary_id,
        "note_id": note_id.0,
        "start_char": boundary.start,
        "end_char": boundary.end,
        "created_at": now_ms(),
    })
}

fn boundary_kind_str(kind: &BoundaryKind) -> &'static str {
    match kind {
        BoundaryKind::Chapter => "chapter",
        BoundaryKind::Heading => "heading",
        BoundaryKind::Section => "section",
        BoundaryKind::Act => "act",
        BoundaryKind::Other => "other",
    }
}

fn boundary_kind_from_str(kind: &str) -> BoundaryKind {
    match kind {
        "chapter" => BoundaryKind::Chapter,
        "heading" => BoundaryKind::Heading,
        "section" => BoundaryKind::Section,
        "act" => BoundaryKind::Act,
        _ => BoundaryKind::Other,
    }
}

fn chapter_chunk_row(
    document: &BorrowedIngestDocument<'_>,
    chapter: &ChapterSpec,
    scope: &ScopeKey,
) -> Value {
    json!({
        "chunk_id": chapter.chunk_id,
        "doc_id": document.document_id.0,
        "level": 2,
        "start": chapter.start,
        "end": chapter.end,
        "text": chapter.title,
        "parent_id": null,
        "scope_narrative": scope.narrative_id,
        "scope_folder": scope.folder_id,
        "created_at": now_ms(),
    })
}

fn chapter_chunkid_row(document: &BorrowedIngestDocument<'_>, chapter: &ChapterSpec) -> Value {
    json!({
        "id": chapter.chunk_id,
        "chunk_key": format!("{}:{}:chapter", document.document_id.0, chapter.chapter_id),
        "doc_id": document.document_id.0,
        "created_at": now_ms(),
    })
}

fn chapter_vertex_row(document: &BorrowedIngestDocument<'_>, chapter: &ChapterSpec) -> Value {
    json!({
        "id": chapter_vertex_id(&document.document_id, chapter.chapter_id),
        "document_id": document.document_id.0,
        "narrative_id": document.scope.narrative_id,
        "value": {
            "kind": "chapter",
            "chapterId": chapter.chapter_id,
            "boundaryId": chapter.boundary_id,
            "boundaryOrdinal": chapter.boundary_ordinal,
            "boundaryKind": boundary_kind_str(&chapter.boundary_kind),
            "title": chapter.title,
        },
        "weight": 1,
        "attributes": {
            "documentId": document.document_id.0,
            "boundaryId": chapter.boundary_id,
            "boundaryOrdinal": chapter.boundary_ordinal,
            "boundaryKind": boundary_kind_str(&chapter.boundary_kind),
            "boundaryDepth": chapter.boundary_depth,
            "start": chapter.start,
            "end": chapter.end,
        },
    })
}

fn parent_chunk_row(
    document: &BorrowedIngestDocument<'_>,
    parent: &ParentChunk,
    scope: &ScopeKey,
) -> Value {
    json!({
        "chunk_id": parent.chunk_id,
        "doc_id": document.document_id.0,
        "level": 1,
        "start": parent.start,
        "end": parent.end,
        "text": preserve_offsets_slice(&document.text[parent.start..parent.end]),
        "parent_id": null,
        "scope_narrative": scope.narrative_id,
        "scope_folder": scope.folder_id,
        "created_at": now_ms(),
    })
}

fn parent_chunkid_row(
    document: &BorrowedIngestDocument<'_>,
    chapter: &ChapterSpec,
    parent: &ParentChunk,
) -> Value {
    json!({
        "id": parent.chunk_id,
        "chunk_key": format!("{}:{}:parent:{}-{}", document.document_id.0, chapter.chapter_id, parent.start, parent.end),
        "doc_id": document.document_id.0,
        "created_at": now_ms(),
    })
}

fn parent_vertex_row(
    document: &BorrowedIngestDocument<'_>,
    chapter: &ChapterSpec,
    parent: &ParentChunk,
) -> Value {
    json!({
        "id": parent_vertex_id(parent.chunk_id),
        "document_id": document.document_id.0,
        "narrative_id": document.scope.narrative_id,
        "value": {
            "kind": "parent",
            "chunkId": parent.chunk_id,
            "chapterId": chapter.chapter_id,
            "boundaryId": chapter.boundary_id,
            "boundaryOrdinal": chapter.boundary_ordinal,
        },
        "weight": 1,
        "attributes": {
            "documentId": document.document_id.0,
            "boundaryId": chapter.boundary_id,
            "boundaryOrdinal": chapter.boundary_ordinal,
            "boundaryKind": boundary_kind_str(&chapter.boundary_kind),
            "start": parent.start,
            "end": parent.end,
            "chapterTitle": chapter.title,
        },
    })
}

fn leaf_chunk_row(
    document: &BorrowedIngestDocument<'_>,
    leaf: &LeafChunk,
    scope: &ScopeKey,
) -> Value {
    json!({
        "chunk_id": leaf.chunk_id,
        "doc_id": document.document_id.0,
        "level": 0,
        "start": leaf.start,
        "end": leaf.end,
        "text": preserve_offsets_slice(&document.text[leaf.start..leaf.end]),
        "parent_id": leaf.parent_id,
        "scope_narrative": scope.narrative_id,
        "scope_folder": scope.folder_id,
        "created_at": now_ms(),
    })
}

fn leaf_chunkid_row(document: &BorrowedIngestDocument<'_>, leaf: &LeafChunk) -> Value {
    json!({
        "id": leaf.chunk_id,
        "chunk_key": leaf.search_id,
        "doc_id": document.document_id.0,
        "created_at": now_ms(),
    })
}

fn leaf_vertex_row(
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    chapter: &ChapterSpec,
    leaf: &LeafChunk,
) -> Value {
    let mut attributes = Map::new();
    attributes.insert("documentId".to_owned(), json!(document.document_id.0));
    attributes.insert("noteId".to_owned(), json!(note_id.0));
    attributes.insert("chapterId".to_owned(), json!(leaf.chapter_id));
    attributes.insert("boundaryId".to_owned(), json!(leaf.boundary_id));
    attributes.insert("boundaryOrdinal".to_owned(), json!(leaf.boundary_ordinal));
    attributes.insert(
        "boundaryKind".to_owned(),
        json!(boundary_kind_str(&leaf.boundary_kind)),
    );
    attributes.insert("chapterTitle".to_owned(), json!(chapter.title));
    attributes.insert("start".to_owned(), json!(leaf.start));
    attributes.insert("end".to_owned(), json!(leaf.end));
    if let Some(message_meta) = leaf.message_meta.as_ref() {
        attributes.insert("messageIds".to_owned(), json!(message_meta.message_ids));
        attributes.insert("roles".to_owned(), json!(message_meta.roles));
        attributes.insert(
            "messageStartIndex".to_owned(),
            json!(message_meta.start_index),
        );
        attributes.insert("messageEndIndex".to_owned(), json!(message_meta.end_index));
    }
    json!({
        "id": leaf_vertex_id(&leaf.search_id),
        "document_id": document.document_id.0,
        "narrative_id": document.scope.narrative_id,
        "value": {
            "kind": "leaf",
            "searchChunkId": leaf.search_id,
            "chunkId": leaf.chunk_id,
            "boundaryId": leaf.boundary_id,
            "boundaryOrdinal": leaf.boundary_ordinal,
        },
        "weight": 1,
        "attributes": Value::Object(attributes),
    })
}

fn entity_row(entity: &EntityState, scope: &ScopeKey, note_id: &NoteId, now: i64) -> Value {
    json!({
        "id": entity.id.0,
        "label": entity.label,
        "kind": kind_to_string(&entity.kind),
        "subtype": null,
        "aliases": entity.aliases.iter().cloned().collect::<Vec<_>>(),
        "first_note": note_id.0,
        "total_mentions": entity.total_mentions,
        "narrative_id": scope.narrative_id,
        "created_by": "graptor",
        "created_at": now,
        "updated_at": now,
    })
}

fn entity_vertex_row(entity: &EntityState, document: &BorrowedIngestDocument<'_>) -> Value {
    json!({
        "id": entity_vertex_id(&entity.id),
        "document_id": document.document_id.0,
        "narrative_id": document.scope.narrative_id,
        "value": {
            "kind": "entity",
            "entityId": entity.id.0,
            "label": entity.label,
            "entityKind": kind_to_string(&entity.kind),
        },
        "weight": entity.total_mentions,
        "attributes": {
            "documentId": document.document_id.0,
            "aliases": entity.aliases.iter().cloned().collect::<Vec<_>>(),
            "chapters": entity.chapters.iter().copied().collect::<Vec<_>>(),
            "boundaryOrdinals": entity.boundary_ordinals.iter().copied().collect::<Vec<_>>(),
        },
    })
}

fn mention_span_row(
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    mention: &MentionRecord,
) -> Value {
    json!({
        "id": mention.span_id,
        "world_id": document.scope.world_id,
        "note_id": note_id.0,
        "narrative_id": document.scope.narrative_id,
        "start": mention.absolute_range.start as i64,
        "end": mention.absolute_range.end as i64,
        "text": mention.surface,
        "content_hash": stable_hex("spanhash", &[mention.surface.as_str()]),
        "span_kind": "entity_mention",
        "status": "resolved",
        "created_by": "graptor",
        "created_at": now_ms(),
        "updated_at": now_ms(),
    })
}

fn mention_span_mention_row(mention: &MentionRecord) -> Value {
    json!({
        "id": mention.span_mention_id,
        "span_id": mention.span_id,
        "candidate_entity_id": mention.entity_id.0,
        "match_type": "exact",
        "confidence": mention.confidence,
        "ev_frequency": null,
        "ev_capital_ratio": null,
        "ev_context_score": null,
        "ev_cooccurrence": null,
        "status": "resolved",
        "created_at": now_ms(),
        "updated_at": now_ms(),
    })
}

fn evidence_id(note_id: &NoteId, evidence: &EvidenceSpan) -> String {
    stable_hex(
        "span",
        &[
            note_id.0.as_str(),
            evidence.kind.as_deref().unwrap_or("evidence"),
            &evidence.range.start.to_string(),
            &evidence.range.end.to_string(),
            evidence.label.as_str(),
        ],
    )
}

fn evidence_row(
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    evidence: &EvidenceSpan,
) -> Value {
    json!({
        "id": evidence_id(note_id, evidence),
        "world_id": document.scope.world_id,
        "note_id": note_id.0,
        "narrative_id": document.scope.narrative_id,
        "start": evidence.range.start as i64,
        "end": evidence.range.end as i64,
        "text": evidence.label,
        "content_hash": stable_hex("spanhash", &[evidence.label.as_str()]),
        "span_kind": evidence.kind.as_deref().unwrap_or("evidence"),
        "status": "derived",
        "created_by": "graptor",
        "created_at": now_ms(),
        "updated_at": now_ms(),
    })
}

fn graph_edge_row(
    source_id: &str,
    target_id: &str,
    weight: i64,
    edge_type: &str,
    attributes: Value,
    data: Option<Value>,
    document_id: Option<String>,
    narrative_id: Option<String>,
) -> Value {
    let (valid_from_boundary, valid_to_boundary, assertion_kind) = attributes
        .as_object()
        .map(|attributes| {
            (
                attributes.get("boundaryId").and_then(Value::as_u64),
                attributes.get("validToBoundary").and_then(Value::as_u64),
                attributes.get("assertionKind").and_then(Value::as_str),
            )
        })
        .unwrap_or((None, None, None));
    json!({
        "source_id": source_id,
        "target_id": target_id,
        "document_id": document_id,
        "narrative_id": narrative_id,
        "valid_from_doc": attributes.get("documentId").cloned().or_else(|| document_id.clone().map(Value::from)),
        "valid_from_boundary": valid_from_boundary.map(|value| value as i64),
        "valid_to_doc": attributes.get("validToDoc").cloned(),
        "valid_to_boundary": valid_to_boundary.map(|value| value as i64),
        "assertion_kind": assertion_kind,
        "weight": weight,
        "attributes": attributes,
        "data": data,
        "edge_type": edge_type,
    })
}

fn document_vertex_id(document_id: &DocumentId) -> String {
    format!("doc::{}", document_id.0)
}

fn chapter_vertex_id(document_id: &DocumentId, chapter_id: u32) -> String {
    format!("chapter::{}::{}", document_id.0, chapter_id)
}

fn parent_vertex_id(parent_id: i64) -> String {
    format!("parent::{parent_id}")
}

fn leaf_vertex_id(search_id: &str) -> String {
    format!("leaf::{search_id}")
}

fn entity_vertex_id(entity_id: &EntityId) -> String {
    format!("entity::{}", entity_id.0)
}

fn normalize_key(text: &str) -> String {
    normalize_raw(text).trim().to_owned()
}

fn preserve_offsets_slice(text: &str) -> Cow<'_, str> {
    if text.contains('\n') {
        Cow::Owned(
            text.chars()
                .map(|ch| if ch == '\n' { ' ' } else { ch })
                .collect(),
        )
    } else {
        Cow::Borrowed(text)
    }
}

fn overlaps(left: TextRange, right: TextRange) -> bool {
    left.start < right.end && right.start < left.end
}

fn overlap_len(left_start: usize, left_end: usize, right_start: usize, right_end: usize) -> usize {
    min(left_end, right_end).saturating_sub(max(left_start, right_start))
}

fn line_bounds(bytes: &[u8], position: usize) -> (usize, usize) {
    let mut start = position;
    while start > 0 && bytes[start - 1] != b'\n' {
        start -= 1;
    }
    let mut end = position;
    while end < bytes.len() && bytes[end] != b'\n' {
        end += 1;
    }
    (start, end)
}

fn kind_to_string(kind: &EntityKind) -> &'static str {
    match kind {
        EntityKind::Character => "Character",
        EntityKind::Location => "Location",
        EntityKind::Npc => "Npc",
        EntityKind::Item => "Item",
        EntityKind::Faction => "Faction",
        EntityKind::Organization => "Organization",
        EntityKind::Event => "Event",
        EntityKind::Concept => "Concept",
        EntityKind::Other => "Other",
    }
}

fn kind_from_string(value: &str) -> EntityKind {
    match value {
        "Character" => EntityKind::Character,
        "Location" => EntityKind::Location,
        "Npc" => EntityKind::Npc,
        "Item" => EntityKind::Item,
        "Faction" => EntityKind::Faction,
        "Organization" => EntityKind::Organization,
        "Event" => EntityKind::Event,
        "Concept" => EntityKind::Concept,
        _ => EntityKind::Other,
    }
}

fn stable_int(prefix: &str, parts: &[&str]) -> i64 {
    (stable_hash(prefix, parts) & 0x7fff_ffff) as i64
}

fn stable_hex(prefix: &str, parts: &[&str]) -> String {
    format!("{:016x}", stable_hash(prefix, parts))
}

fn stable_hash(prefix: &str, parts: &[&str]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in prefix.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for part in parts {
        hash ^= 0xff;
        hash = hash.wrapping_mul(0x100000001b3);
        for byte in part.as_bytes() {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

fn now_ms() -> i64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now() as i64
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_millis() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_scanner::PhoenixScanner;
    use phoenix_store_cozo::PhoenixCozoStore;
    use phoenix_structure::PhoenixStructure;
    use phoenix_types::IngestRequest;

    fn sample_document(text: &str) -> IngestDocument {
        IngestDocument {
            document_id: DocumentId("doc-1".to_owned()),
            note_id: Some(NoteId("note-1".to_owned())),
            title: "Sample".to_owned(),
            text: text.to_owned(),
            scope: ScopeKey::default(),
        }
    }

    #[test]
    fn chapter_detector_picks_up_headers() {
        let graptor = PhoenixGraptor::default();
        let boundaries =
            graptor.detect_chapter_boundaries("# Prologue\nRyan woke up.\n\nChapter 1\nRyan ran.");
        assert_eq!(boundaries.len(), 2);
    }

    #[test]
    fn chunkerx2_style_chunks_include_parents() {
        let document = sample_document(
            "Chapter 1\nRyan woke up. Ryan sharpened the blade. Ryan left.\n\nChapter 2\nRyan found Len. Len smiled.",
        );
        let borrowed = BorrowedIngestDocument::from(&document);
        let markers = PhoenixGraptor::default().detect_chapter_boundaries(&document.text);
        let boundaries = build_boundary_specs(&borrowed, &markers);
        let mut chapters = build_chapter_specs(&borrowed, &boundaries);
        let mut leaves = build_leaf_chunks(&borrowed, &GraptorConfig::default());
        assign_leaves_to_boundaries(&boundaries, &mut leaves);
        assign_leaves_to_chapters(&borrowed, &chapters, &mut leaves);
        build_parent_chunks(
            &borrowed,
            &GraptorConfig::default(),
            &mut chapters,
            &mut leaves,
        );

        assert!(!leaves.is_empty());
        assert!(chapters.iter().any(|chapter| !chapter.parents.is_empty()));
        assert!(leaves.iter().all(|leaf| !leaf.search_id.is_empty()));
    }

    #[test]
    fn registry_merges_aliases_across_chapters() {
        let mut registry = EntityRegistry::default();
        let id = registry.resolve_or_register("Ryan", None, Some(EntityKind::Character), 1, 1);
        registry.add_alias(&id, "Romano");
        let resolved =
            registry.resolve_or_register("Romano", None, Some(EntityKind::Character), 2, 2);
        registry.record_mention(&id, 1, 1);
        registry.record_mention(&resolved, 2, 2);

        assert_eq!(id, resolved);
        assert_eq!(registry.total_aliases(), 1);
        assert_eq!(registry.multi_chapter_entities(), 1);
    }

    #[test]
    fn discovery_candidates_persist_without_promoting_canonical_entities() {
        let store = PhoenixCozoStore::new().expect("store");
        let scanner = PhoenixScanner::default();
        let structure = PhoenixStructure::default();
        let graptor = PhoenixGraptor::default();

        let result = graptor
            .ingest(
                &store,
                &scanner,
                &structure,
                &IngestRequest {
                    session_id: Some(SessionId("session-discovery".to_owned())),
                    documents: vec![IngestDocument {
                        document_id: DocumentId("doc-discovery".to_owned()),
                        note_id: None,
                        title: "Discovery".to_owned(),
                        text: "Zanthor moved fast. Zanthor returned later.".to_owned(),
                        scope: ScopeKey {
                            world_id: Some("world-1".to_owned()),
                            narrative_id: None,
                            folder_id: None,
                            folder_path: None,
                        },
                    }],
                    commit: false,
                },
            )
            .expect("ingest");

        let discovery_rows = store
            .fetch_rows("discovery_candidates")
            .expect("discovery rows");
        let entity_rows = store.fetch_rows("entities").expect("entity rows");

        assert!(
            !discovery_rows.is_empty(),
            "speculative discovery should be persisted"
        );
        assert_eq!(
            result
                .discovery_summary
                .as_ref()
                .map(|summary| summary.persisted_count),
            Some(discovery_rows.len())
        );
        assert!(
            entity_rows.is_empty(),
            "discovery-only surfaces should not be promoted into canonical entities during ingest"
        );
    }

    #[test]
    fn shared_sentence_splitter_matches_graptor_sentence_ranges() {
        let text = "Dr. Luffy ran. Mr. Zoro stayed! Wow?";
        let ranges = split_sentences(text);

        assert_eq!(ranges.len(), 3);
        assert_eq!(
            &text[ranges[0].start as usize..ranges[0].end as usize],
            "Dr. Luffy ran."
        );
        assert_eq!(
            &text[ranges[1].start as usize..ranges[1].end as usize],
            "Mr. Zoro stayed!"
        );
        assert_eq!(
            &text[ranges[2].start as usize..ranges[2].end as usize],
            "Wow?"
        );
    }

    #[test]
    fn thread_ingest_preserves_message_metadata_and_bypasses_chapters() {
        let store = PhoenixCozoStore::new().expect("store");
        let scanner = PhoenixScanner::default();
        let structure = PhoenixStructure::default();
        let graptor = PhoenixGraptor::new(GraptorConfig::default().without_chapter_detection());
        let messages = vec![
            BorrowedThreadMessage {
                message_id: "msg-1",
                role: "user",
                content: "Chapter 1: Ryan reaches the harbor.",
                created_at: 1,
            },
            BorrowedThreadMessage {
                message_id: "msg-2",
                role: "assistant",
                content: "Len is waiting there with the artifact.",
                created_at: 2,
            },
        ];

        let result = graptor
            .ingest_message_thread_view(
                &store,
                &scanner,
                &structure,
                &BorrowedIngestThread {
                    document_id: DocumentId("thread-doc-1".to_owned()),
                    title: "Thread Window",
                    messages: &messages,
                    scope: ScopeKey {
                        narrative_id: Some("thread-1".to_owned()),
                        ..ScopeKey::default()
                    },
                },
            )
            .expect("thread ingest");

        assert_eq!(result.documents[0].chapter_count, 1);

        let leaf = store
            .fetch_rows("graph_vertices")
            .expect("graph vertices")
            .into_iter()
            .find(|row| {
                row.get("value")
                    .and_then(Value::as_object)
                    .and_then(|value| value.get("kind"))
                    .and_then(Value::as_str)
                    == Some("leaf")
            })
            .expect("leaf vertex");
        let attributes = leaf
            .get("attributes")
            .and_then(Value::as_object)
            .expect("leaf attributes");
        assert_eq!(
            attributes
                .get("messageIds")
                .and_then(Value::as_array)
                .map(|items| items.len()),
            Some(2)
        );
        assert_eq!(
            attributes
                .get("roles")
                .and_then(Value::as_array)
                .map(|items| items.len()),
            Some(2)
        );
    }

    #[test]
    fn thread_ingest_splits_oversized_single_messages() {
        let store = PhoenixCozoStore::new().expect("store");
        let scanner = PhoenixScanner::default();
        let structure = PhoenixStructure::default();
        let graptor = PhoenixGraptor::new(GraptorConfig {
            chunk_size: 40,
            overlap: 10,
            parent_chunk_size: 80,
            parent_overlap: 20,
            ..GraptorConfig::default().without_chapter_detection()
        });
        let content = "Ryan crossed the bridge. Ryan met Len. Ryan found the artifact. Ryan escaped before dawn.";
        let messages = vec![BorrowedThreadMessage {
            message_id: "msg-big",
            role: "user",
            content,
            created_at: 1,
        }];

        let result = graptor
            .ingest_message_thread_view(
                &store,
                &scanner,
                &structure,
                &BorrowedIngestThread {
                    document_id: DocumentId("thread-doc-2".to_owned()),
                    title: "Oversized Thread Window",
                    messages: &messages,
                    scope: ScopeKey {
                        narrative_id: Some("thread-2".to_owned()),
                        ..ScopeKey::default()
                    },
                },
            )
            .expect("thread ingest");

        assert!(result.documents[0].leaf_count > 1);
        let leaf_rows = store.fetch_rows("graph_vertices").expect("graph vertices");
        let thread_leaf_count = leaf_rows
            .iter()
            .filter(|row| row.get("document_id").and_then(Value::as_str) == Some("thread-doc-2"))
            .filter(|row| {
                row.get("value")
                    .and_then(Value::as_object)
                    .and_then(|value| value.get("kind"))
                    .and_then(Value::as_str)
                    == Some("leaf")
            })
            .count();
        assert!(thread_leaf_count > 1);
    }
}
