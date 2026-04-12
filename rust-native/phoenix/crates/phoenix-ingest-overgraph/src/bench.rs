use super::*;
use serde::Serialize;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestBenchmarkCounts {
    pub text_bytes: usize,
    pub sentence_count: usize,
    pub mention_count: usize,
    pub chunk_count: usize,
    pub relation_seed_count: usize,
    pub entity_count: usize,
    pub relation_count: usize,
    pub alias_confirmation_count: usize,
    pub coref_cluster_count: usize,
    pub causal_proposition_count: usize,
    pub causal_link_count: usize,
    pub temporal_proposition_count: usize,
    pub temporal_anchor_count: usize,
    pub event_identity_seed_count: usize,
    pub lexical_span_count: usize,
    pub lexical_alias_entry_count: usize,
    pub segment_count: usize,
    pub segment_bytes: usize,
    pub graph_vertex_count: usize,
    pub graph_edge_count: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct IngestBenchmarkReport {
    pub document_id: String,
    pub document_total_us: u64,
    pub scan_bundle_us: u64,
    pub resolve_us: u64,
    pub post_resolve_total_us: u64,
    pub causal_substrate_us: u64,
    pub temporal_substrate_us: u64,
    pub event_identity_substrate_us: u64,
    pub lexical_postings_us: u64,
    pub segment_encode_us: u64,
    pub counts: IngestBenchmarkCounts,
}

impl PhoenixInvarantV3 {
    pub fn benchmark_document_pipeline(
        &self,
        document: &IngestDocument,
        session_id: Option<&SessionId>,
        created_at: i64,
    ) -> Result<IngestBenchmarkReport, StoreError> {
        let assignment = benchmark_assignment(document);
        let entity_memory = NativeEntityMemory::default();

        let document_started = Instant::now();

        let started = Instant::now();
        let scan_bundle = self.scan_document_bundle(document)?;
        let scan_bundle_us = elapsed_us(started);

        let started = Instant::now();
        let resolution_bundle =
            self.resolve_document_bundle(document, &scan_bundle, &entity_memory)?;
        let resolve_us = elapsed_us(started);

        let post_resolve_started = Instant::now();
        let mention_count = scan_bundle.scan.mentions.len();
        let _ = self.build_document_state(
            document,
            session_id,
            &assignment,
            created_at,
            &scan_bundle.boundaries,
            scan_bundle.chunks.len(),
            &resolution_bundle.entities,
            &resolution_bundle.kernel_batch,
            resolution_bundle.discovery_count,
            mention_count,
        );

        let started = Instant::now();
        let causal_substrate = build_document_causal_substrate(document, &scan_bundle, created_at);
        let causal_substrate_us = elapsed_us(started);

        let started = Instant::now();
        let temporal_substrate =
            build_document_temporal_substrate(document, &scan_bundle, created_at);
        let temporal_substrate_us = elapsed_us(started);

        let started = Instant::now();
        let event_identity_substrate = build_document_event_identity_substrate(
            document,
            assignment.revision,
            &causal_substrate,
            &temporal_substrate,
        );
        let event_identity_substrate_us = elapsed_us(started);

        let lexical_span_count = scan_bundle.indexed_spans.len();
        let started = Instant::now();
        let lexical = build_lexical_postings_segment(
            scan_bundle.indexed_spans,
            &resolution_bundle.entities,
            &document.document_id.0,
        );
        let lexical_postings_us = elapsed_us(started);

        let started = Instant::now();
        let mut segments = Vec::<PreparedDocumentSegment>::new();
        let mut segment_refs = Vec::<DocumentSegmentRef>::new();
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::AliasConfirmationTable,
            resolution_bundle.alias_confirmations.len(),
            &resolution_bundle.alias_confirmations,
        )?;
        if !resolution_bundle.coref_clusters.is_empty() {
            self.push_segment(
                &mut segments,
                &mut segment_refs,
                DocumentSegmentKind::CorefClusterTable,
                resolution_bundle.coref_clusters.len(),
                &resolution_bundle.coref_clusters,
            )?;
        }
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::ChunkTable,
            scan_bundle.chunks.len(),
            &scan_bundle.chunks,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::EntityTable,
            resolution_bundle.entities.len(),
            &resolution_bundle.entities,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::RelationTable,
            resolution_bundle.relations.len(),
            &resolution_bundle.relations,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::CausalSubstrateTable,
            causal_substrate.propositions.len()
                + causal_substrate.semantic_events.len()
                + causal_substrate.semantic_states.len()
                + causal_substrate.semantic_claims.len()
                + causal_substrate.semantic_relations.len()
                + causal_substrate.temporal_bindings.len()
                + causal_substrate.causal_candidates.len()
                + causal_substrate.causal_links.len()
                + causal_substrate.causal_diagnostics.len(),
            &causal_substrate,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::TemporalSubstrateTable,
            temporal_substrate.propositions.len()
                + temporal_substrate.semantic_events.len()
                + temporal_substrate.semantic_states.len()
                + temporal_substrate.semantic_claims.len()
                + temporal_substrate.surface_temporal_cues.len()
                + temporal_substrate.timex_records.len()
                + temporal_substrate.anchor_candidates.len()
                + temporal_substrate.axis_records.len()
                + temporal_substrate.reference_timex_edges.len()
                + temporal_substrate.reference_event_edges.len()
                + temporal_substrate.temporal_claims.len()
                + temporal_substrate.temporal_constraints.len()
                + temporal_substrate.temporal_diagnostics.len(),
            &temporal_substrate,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::EventIdentitySubstrateTable,
            event_identity_substrate.mention_seeds.len()
                + event_identity_substrate.diagnostics.len(),
            &event_identity_substrate,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::LexicalPostings,
            lexical.spans.len() + lexical.alias_entries.len(),
            &lexical,
        )?;
        let segment_encode_us = elapsed_us(started);
        let segment_bytes = segments.iter().map(|segment| segment.payload.len()).sum();

        Ok(IngestBenchmarkReport {
            document_id: document.document_id.0.clone(),
            document_total_us: elapsed_us(document_started),
            scan_bundle_us,
            resolve_us,
            post_resolve_total_us: elapsed_us(post_resolve_started),
            causal_substrate_us,
            temporal_substrate_us,
            event_identity_substrate_us,
            lexical_postings_us,
            segment_encode_us,
            counts: IngestBenchmarkCounts {
                text_bytes: document.text.len(),
                sentence_count: scan_bundle.scan.sentences.len(),
                mention_count,
                chunk_count: scan_bundle.chunks.len(),
                relation_seed_count: scan_bundle.structure.relation_seeds.len(),
                entity_count: resolution_bundle.entities.len(),
                relation_count: resolution_bundle.relations.len(),
                alias_confirmation_count: resolution_bundle.alias_confirmations.len(),
                coref_cluster_count: resolution_bundle.coref_clusters.len(),
                causal_proposition_count: causal_substrate.propositions.len(),
                causal_link_count: causal_substrate.causal_links.len(),
                temporal_proposition_count: temporal_substrate.propositions.len(),
                temporal_anchor_count: temporal_substrate.anchor_candidates.len(),
                event_identity_seed_count: event_identity_substrate.mention_seeds.len(),
                lexical_span_count,
                lexical_alias_entry_count: lexical.alias_entries.len(),
                segment_count: segments.len(),
                segment_bytes,
                graph_vertex_count: resolution_bundle.kernel_batch.vertices.len(),
                graph_edge_count: resolution_bundle.kernel_batch.edges.len(),
            },
        })
    }

    pub fn build_archive_for_benchmark(
        &self,
        document: &IngestDocument,
        session_id: Option<&SessionId>,
        created_at: i64,
    ) -> Result<DocumentArchive, StoreError> {
        let assignment = benchmark_assignment(document);
        let entity_memory = NativeEntityMemory::default();
        let outcome = self.build_document_outcome(
            document,
            session_id,
            &assignment,
            &entity_memory,
            created_at,
        )?;
        Ok(outcome.archive)
    }
}

fn benchmark_assignment(document: &IngestDocument) -> DocumentOrdinalAssignment {
    DocumentOrdinalAssignment {
        document_id: document.document_id.0.clone(),
        scope: document.scope.clone(),
        scope_key: scope_storage_key(&document.scope),
        scope_ord: ScopeOrd(0),
        document_ord: DocumentOrd(0),
        revision: 1,
    }
}

fn elapsed_us(started: Instant) -> u64 {
    started.elapsed().as_micros() as u64
}
