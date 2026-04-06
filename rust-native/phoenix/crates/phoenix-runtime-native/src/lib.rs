use phoenix_analytics::TextAnalytics;
use phoenix_causality::{CausalityLowerer, CausalityRequest};
use phoenix_ingest_native::{PhoenixIngestNative, V2IngestArtifacts};
use phoenix_kernel::DeterministicKernel;
use phoenix_machine::{SurfaceCompileArtifacts, SurfaceCompiler};
use phoenix_mentions::MentionCompiler;
use phoenix_proposition::PropositionLowerer;
use phoenix_query::QueryPlan;
use phoenix_semantics::{SemanticBundle, SemanticLowerer};
use phoenix_semantic_v2::{DocumentRevisionRef, SessionArchive};
use phoenix_store_native_core::{PhoenixArchiveStoreV2, StoreError};
use phoenix_time::{TemporalBinding, TimeKernel};
use phoenix_types::{
    CausalBundle, Diagnostic, EntityId, IngestDocument, IngestResult, IndexedSpan,
    LexicalSearchResult, NodeHit, PreparedMentionRecord, QueryRequest, QueryResult, ScanArtifact,
    ScanRequest, ScopeKey, SessionDocumentState, SessionId, StructureArtifact, StructureRequest,
};
use rustc_hash::FxHashMap;
use std::collections::BTreeSet;

pub struct PhoenixRuntimeNative {
    pub deterministic_kernel: DeterministicKernel,
    pub surface_compiler: SurfaceCompiler,
    ingest_engine: PhoenixIngestNative,
}

#[derive(Clone, Debug, Default)]
pub struct NativeIngestArtifacts {
    pub kernel_batches: Vec<phoenix_kernel::KernelMutationBatch>,
    pub session_documents: Vec<SessionDocumentState>,
    pub document_refs: Vec<DocumentRevisionRef>,
    pub document_manifests: Vec<phoenix_semantic_v2::DocumentManifest>,
    pub manifest_namespaces: Vec<String>,
    pub span_count: usize,
    pub discovery_candidate_count: usize,
    pub touched_scopes: Vec<ScopeKey>,
}

impl From<V2IngestArtifacts> for NativeIngestArtifacts {
    fn from(value: V2IngestArtifacts) -> Self {
        Self {
            kernel_batches: value.kernel_batches,
            session_documents: value.session_documents,
            document_refs: value.document_refs,
            document_manifests: value.document_manifests,
            manifest_namespaces: value.manifest_namespaces,
            span_count: value.span_count,
            discovery_candidate_count: value.discovery_candidate_count,
            touched_scopes: value.touched_scopes,
        }
    }
}

impl PhoenixRuntimeNative {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn scan_text(&self, request: ScanRequest) -> ScanArtifact {
        self.surface_compiler.compatibility_scan(&request)
    }

    pub fn build_structure(&self, request: StructureRequest) -> StructureArtifact {
        self.surface_compiler.compatibility_structure(&request)
    }

    pub fn analyze_text(&self, text: &str) -> TextAnalytics {
        phoenix_analytics::analyze_text(text)
    }

    pub fn compile_surface(&self, request: ScanRequest) -> SurfaceCompileArtifacts {
        self.surface_compiler.scan_request(&request)
    }

    pub fn ingest_documents_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        session_id: Option<&SessionId>,
        documents: &[IngestDocument],
        revision: u64,
        created_at: i64,
    ) -> Result<(IngestResult, NativeIngestArtifacts), StoreError> {
        self.ingest_engine
            .ingest_documents_native(store, session_id, documents, revision, created_at)
            .map(|(result, artifacts)| (result, artifacts.into()))
            .map_err(|error| StoreError::Query(error.to_string()))
    }

    pub fn prepare_mentions(&self, artifacts: &SurfaceCompileArtifacts) -> Vec<PreparedMentionRecord> {
        MentionCompiler::prepare(artifacts)
    }

    pub fn lower_propositions(&self, artifacts: &SurfaceCompileArtifacts) -> Vec<phoenix_types::Proposition> {
        PropositionLowerer::lower(artifacts)
    }

    pub fn lower_semantics(&self, propositions: &[phoenix_types::Proposition]) -> SemanticBundle {
        SemanticLowerer::lower(propositions)
    }

    pub fn bind_time(&self, label: &str, recorded_at: Option<i64>) -> TemporalBinding {
        TimeKernel::bind_label(label, recorded_at)
    }

    pub fn lower_causality(
        &self,
        text: &str,
        artifacts: &SurfaceCompileArtifacts,
        propositions: &[phoenix_types::Proposition],
        semantics: &SemanticBundle,
        temporal_bindings: &[TemporalBinding],
    ) -> CausalBundle {
        CausalityLowerer::lower(CausalityRequest {
            text,
            artifacts,
            propositions,
            semantics,
            temporal_bindings,
        })
    }

    pub fn plan_query(&self, request: &QueryRequest) -> QueryPlan {
        phoenix_query::DeterministicQuery::plan(request)
    }

    pub fn query_with_lexical(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        lexical: LexicalSearchResult,
        request: &QueryRequest,
    ) -> Result<QueryResult, StoreError> {
        let chunk_hits = lexical
            .span_hits
            .iter()
            .map(|hit| phoenix_types::ChunkHit {
                chunk_id: hit.span_id.clone(),
                score: hit.score,
            })
            .collect::<Vec<_>>();

        let normalized_query = normalize_surface(&request.query);
        let mut entity_scores = FxHashMap::<String, f64>::default();
        let postings = store.lookup_alias_postings(&request.scope, &normalized_query)?;
        let exact_score = term_score(&normalized_query, &normalized_query);
        for posting in postings {
            entity_scores
                .entry(posting.entity_id.clone())
                .and_modify(|existing| {
                    *existing += exact_score + (posting.mention_count as f64).ln_1p() * 0.05
                })
                .or_insert(exact_score + (posting.mention_count as f64).ln_1p() * 0.05);
        }

        if entity_scores.is_empty() && !lexical.span_hits.is_empty() {
            let document_ids = lexical
                .span_hits
                .iter()
                .filter_map(|hit| hit.document_id.as_ref().map(|document_id| document_id.0.clone()))
                .collect::<BTreeSet<_>>();
            if !document_ids.is_empty() {
                let archives = store.load_latest_document_archives(Some(&request.scope))?;
                let mut document_boosts = FxHashMap::<String, f64>::default();
                for hit in &lexical.span_hits {
                    if let Some(document_id) = hit.document_id.as_ref() {
                        document_boosts
                            .entry(document_id.0.clone())
                            .and_modify(|existing| *existing = existing.max(hit.score))
                            .or_insert(hit.score);
                    }
                }
                for archive in archives {
                    if !document_ids.contains(&archive.manifest.document_id) {
                        continue;
                    }
                    let doc_boost = document_boosts
                        .get(&archive.manifest.document_id)
                        .copied()
                        .unwrap_or(0.25);
                    for entity in archive.entities {
                        entity_scores
                            .entry(entity.entity_id.0.clone())
                            .and_modify(|existing| {
                                *existing += doc_boost + (entity.mention_count as f64).ln_1p() * 0.05
                            })
                            .or_insert(doc_boost + (entity.mention_count as f64).ln_1p() * 0.05);
                    }
                }
            }
        }

        let mut node_hits = entity_scores
            .into_iter()
            .map(|(entity_id, score)| NodeHit {
                entity_id: Some(EntityId(entity_id)),
                score,
            })
            .collect::<Vec<_>>();
        if node_hits.is_empty() {
            node_hits.extend(chunk_hits.iter().take(request.limit.unwrap_or(5)).map(|hit| NodeHit {
                entity_id: None,
                score: hit.score * 0.25,
            }));
        }
        node_hits.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.entity_id.cmp(&right.entity_id))
        });
        node_hits.truncate(request.limit.unwrap_or(5));

        let mut diagnostics = lexical.diagnostics;
        diagnostics.push(Diagnostic {
            code: "PX_TRIVERSE_V2_KERNEL".to_owned(),
            message:
                "Native query assembled entity hits from persisted alias postings and compact semantic archives."
                    .to_owned(),
        });
        diagnostics.push(Diagnostic {
            code: "PX_RUNTIME_NATIVE_QUERY".to_owned(),
            message:
                "Native runtime query used deterministic lexical recall plus persisted native semantic state."
                    .to_owned(),
        });

        Ok(QueryResult {
            session_id: request.session_id.clone(),
            chunk_hits,
            node_hits,
            diagnostics,
        })
    }

    pub fn load_latest_session_summary(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        session_id: &SessionId,
    ) -> Result<Option<SessionArchive>, StoreError> {
        store.load_latest_session_archive(session_id)
    }

    pub fn persist_session_summary(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        summary: &SessionArchive,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        store.persist_session_archive(summary, revision, created_at)
    }

    pub fn load_latest_lex_spans(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<IndexedSpan>, StoreError> {
        store.load_lex_spans(scope)
    }

    pub fn merge_session_summary(
        &self,
        existing: Option<SessionArchive>,
        session_id: SessionId,
        documents: Vec<SessionDocumentState>,
        document_refs: Vec<DocumentRevisionRef>,
        discovery_candidate_count: usize,
        span_count: usize,
        graph_vertex_count: usize,
        graph_edge_count: usize,
        updated_at: i64,
    ) -> SessionArchive {
        let mut by_document = existing
            .as_ref()
            .map(|summary| {
                summary
                    .documents
                    .iter()
                    .cloned()
                    .map(|document| (document.document_id.0.clone(), document))
                    .collect::<FxHashMap<_, _>>()
            })
            .unwrap_or_default();
        for document in documents {
            by_document.insert(document.document_id.0.clone(), document);
        }
        let mut ref_by_document = existing
            .map(|summary| {
                summary
                    .document_refs
                    .into_iter()
                    .map(|document| (document.document_id.clone(), document))
                    .collect::<FxHashMap<_, _>>()
            })
            .unwrap_or_default();
        for document_ref in document_refs {
            ref_by_document.insert(document_ref.document_id.clone(), document_ref);
        }
        let mut merged_documents = by_document.into_values().collect::<Vec<_>>();
        merged_documents.sort_by(|left, right| left.document_id.0.cmp(&right.document_id.0));
        let mut merged_document_refs = ref_by_document.into_values().collect::<Vec<_>>();
        merged_document_refs.sort_by(|left, right| left.document_id.cmp(&right.document_id));
        SessionArchive {
            session_id,
            session_ord: None,
            documents: merged_documents,
            document_refs: merged_document_refs,
            discovery_candidate_count,
            span_count,
            graph_vertex_count,
            graph_edge_count,
            graph_generation: 0,
            lex_generation: 0,
            updated_at,
            archive_version: 2,
        }
    }

    pub fn compile_native_lane(
        &self,
        text: &str,
        scope: &ScopeKey,
        resolver_seed: &[phoenix_types::ResolverEntitySeed],
    ) -> (SurfaceCompileArtifacts, Vec<PreparedMentionRecord>, Vec<phoenix_types::Proposition>, SemanticBundle) {
        let artifacts = self.surface_compiler.compile(text, scope, resolver_seed);
        let mentions = self.prepare_mentions(&artifacts);
        let propositions = self.lower_propositions(&artifacts);
        let semantics = self.lower_semantics(&propositions);
        (artifacts, mentions, propositions, semantics)
    }

    pub fn compile_native_lane_with_causality(
        &self,
        text: &str,
        scope: &ScopeKey,
        resolver_seed: &[phoenix_types::ResolverEntitySeed],
        temporal_bindings: &[TemporalBinding],
    ) -> (
        SurfaceCompileArtifacts,
        Vec<PreparedMentionRecord>,
        Vec<phoenix_types::Proposition>,
        SemanticBundle,
        CausalBundle,
    ) {
        let (artifacts, mentions, propositions, semantics) =
            self.compile_native_lane(text, scope, resolver_seed);
        let causality = self.lower_causality(
            text,
            &artifacts,
            &propositions,
            &semantics,
            temporal_bindings,
        );
        (artifacts, mentions, propositions, semantics, causality)
    }
}

impl Default for PhoenixRuntimeNative {
    fn default() -> Self {
        Self {
            deterministic_kernel: DeterministicKernel::default(),
            surface_compiler: SurfaceCompiler::default(),
            ingest_engine: PhoenixIngestNative::default(),
        }
    }
}

fn normalize_surface(value: &str) -> String {
    value
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|character: char| !character.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn term_score(query: &str, candidate: &str) -> f64 {
    if query == candidate {
        1.0
    } else if candidate.contains(query) || query.contains(candidate) {
        0.8
    } else {
        0.6
    }
}
