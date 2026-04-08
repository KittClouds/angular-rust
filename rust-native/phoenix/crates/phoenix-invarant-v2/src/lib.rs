use std::collections::BTreeMap;

use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use memchr::memchr3_iter;
use phoenix_chunker::{build_chunks, ChunkerConfig};
use phoenix_graph_kernel::{
    KernelEdge, KernelEdgeType, KernelEntityResolveRequest, KernelGraphLayer,
    KernelGraphSnapshot, KernelMutationBatch,
    KernelMutationScope, KernelProvenance, KernelResolutionFacet, KernelVertex, KernelVertexId,
    PhoenixGraphKernel,
};
use phoenix_semantic_v2::{
    scope_storage_key, AliasConfirmation, AliasEntry, AliasPosting, CandidateEntity,
    CandidateEvidence, ChunkId, ChunkRecord, DirtyScopeRecord, DocumentArchive, DocumentManifest,
    DocumentOrd, DocumentOrdinalAssignment, DocumentRevisionRef, DocumentSegmentHeader,
    DocumentSegmentKind, DocumentSegmentRef, DocumentVersionId, LexicalPostingsSegment,
    PreparedDocument, PreparedDocumentSegment, ResolutionDecision, ResolvedMention,
    ScopeLexSidecar, ScopeOrd, SemanticEntityRecord, SemanticRelationRecord, SessionArchive,
};
use phoenix_store_cozo::StoreError;
use phoenix_store_native::{
    BundleHeader, BundleKey, BundleKind, PhoenixArchiveStoreV2, PhoenixBundleStoreV2,
};
use phoenix_types::{
    BoundaryKind, ChunkKind, ChunkSpan, Diagnostic, DocumentId, EntityId, EntityKind,
    EvidenceSpan, FrameSlot, IndexedSpan, IndexedTextField, IngestDocument, IngestDocumentSummary,
    IngestResult, LexicalField, MentionEntityRef, MentionSource, MentionSpan,
    NarrativeTransitivity, NarrativeVerbHit, PosTag, RelationCandidate, ResolverEntitySeed,
    ResolverLink, ResolverLinkKind, ScopeKey, ScanArtifact, SentenceFrame, SentenceSpan,
    SessionDocumentState, SessionId, StructureArtifact, TextRange, TokenClass, TokenSpan,
    VerbFrame,
};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::json;
use smallvec::SmallVec;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InvarantV2Config {
    pub chunk_size: usize,
    pub overlap: usize,
}

fn encode_archive<T: Serialize>(value: &T) -> Result<Vec<u8>, StoreError> {
    let payload =
        rmp_serde::to_vec_named(value).map_err(|error| StoreError::Query(error.to_string()))?;
    Ok(compress_prepend_size(&payload))
}

fn decode_archive<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, StoreError> {
    let payload =
        decompress_size_prepended(bytes).map_err(|error| StoreError::Query(error.to_string()))?;
    rmp_serde::from_slice(&payload).map_err(|error| StoreError::Query(error.to_string()))
}

fn encode_segment_payload<T: Serialize>(value: &T) -> Result<(Vec<u8>, usize), StoreError> {
    let payload =
        rmp_serde::to_vec_named(value).map_err(|error| StoreError::Query(error.to_string()))?;
    let uncompressed_len = payload.len();
    Ok((compress_prepend_size(&payload), uncompressed_len))
}

fn document_archive_header(
    scope: &ScopeKey,
    document_id: &str,
    revision: u64,
    byte_len: usize,
    created_at: i64,
) -> BundleHeader {
    BundleHeader {
        key: BundleKey {
            kind: BundleKind::DocumentArchive,
            scope: scope_storage_key(scope),
            entity_key: document_id.to_owned(),
            revision,
        },
        byte_len,
        created_at,
    }
}

fn session_archive_header(
    session_id: &SessionId,
    revision: u64,
    byte_len: usize,
    created_at: i64,
) -> BundleHeader {
    BundleHeader {
        key: BundleKey {
            kind: BundleKind::SessionArchive,
            scope: session_id.0.clone(),
            entity_key: session_id.0.clone(),
            revision,
        },
        byte_len,
        created_at,
    }
}

fn scope_lex_sidecar_header(
    scope: &ScopeKey,
    revision: u64,
    byte_len: usize,
    created_at: i64,
) -> BundleHeader {
    let scope_key = scope_storage_key(scope);
    BundleHeader {
        key: BundleKey {
            kind: BundleKind::ScopeLexSidecar,
            scope: scope_key.clone(),
            entity_key: scope_key,
            revision,
        },
        byte_len,
        created_at,
    }
}

fn tokenize(text: &str) -> Vec<TokenSpan> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }
        if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
            let mut end = start + ch.len_utf8();
            while let Some((next_ix, next)) = chars.peek().copied() {
                if next.is_alphanumeric() || next == '\'' || next == '-' {
                    chars.next();
                    end = next_ix + next.len_utf8();
                } else {
                    break;
                }
            }
            let token = &text[start..end];
            let lower = token.to_ascii_lowercase();
            let pos = if is_pronoun(&lower) {
                Some(PosTag::Pronoun)
            } else if is_verb_token(&lower) {
                Some(PosTag::Verb)
            } else if token.chars().next().is_some_and(|value| value.is_uppercase()) {
                Some(PosTag::ProperNoun)
            } else {
                Some(PosTag::Noun)
            };
            tokens.push(TokenSpan {
                range: to_range(start, end),
                token_class: Some(if token.chars().all(|value| value.is_numeric()) {
                    TokenClass::Number
                } else {
                    TokenClass::Word
                }),
                pos,
                masked: false,
                capitalized: token.chars().next().is_some_and(|value| value.is_uppercase()),
            });
        } else {
            tokens.push(TokenSpan {
                range: to_range(start, start + ch.len_utf8()),
                token_class: Some(if ch.is_ascii_punctuation() {
                    TokenClass::Punctuation
                } else {
                    TokenClass::Symbol
                }),
                pos: Some(PosTag::Punctuation),
                masked: false,
                capitalized: false,
            });
        }
    }
    tokens
}

fn sentence_spans(text: &str) -> Vec<SentenceSpan> {
    if text.trim().is_empty() {
        return Vec::new();
    }
    let mut ranges = Vec::new();
    let mut current_start = 0usize;
    for boundary in memchr3_iter(b'.', b'!', b'?', text.as_bytes()) {
        let end = boundary + 1;
        let trimmed_start = trim_start_offset(text, current_start, end);
        let trimmed_end = trim_end_offset(text, trimmed_start, end);
        if trimmed_start < trimmed_end {
            ranges.push((trimmed_start, trimmed_end));
        }
        current_start = end;
    }
    if current_start < text.len() {
        let trimmed_start = trim_start_offset(text, current_start, text.len());
        let trimmed_end = trim_end_offset(text, trimmed_start, text.len());
        if trimmed_start < trimmed_end {
            ranges.push((trimmed_start, trimmed_end));
        }
    }
    ranges
        .into_iter()
        .enumerate()
        .map(|(index, (start, end))| SentenceSpan {
            index,
            range: to_range(start, end),
        })
        .collect()
}

fn discover_mentions(
    text: &str,
    tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
    resolver_seed: &[ResolverEntitySeed],
) -> Vec<MentionSpan> {
    let seed_map = resolver_seed
        .iter()
        .flat_map(|seed| {
            let mut forms = SmallVec::<[String; 4]>::new();
            forms.push(normalize_surface(&seed.canonical_name));
            for alias in &seed.aliases {
                forms.push(normalize_surface(alias));
            }
            forms.into_iter().map(move |form| (form, seed))
        })
        .collect::<FxHashMap<_, _>>();

    let mut mentions = Vec::new();
    let mut index = 0usize;
    let mut sentence_cursor = 0usize;
    while index < tokens.len() {
        let token = &tokens[index];
        let surface = slice_or_empty(text, token.range);
        let sentence_index = locate_sentence_cursor(sentences, &mut sentence_cursor, token.range);
        let normalized = (!seed_map.is_empty()).then(|| normalize_token_surface(surface));

        if let Some(seed) = normalized.as_ref().and_then(|value| seed_map.get(value)) {
            mentions.push(MentionSpan {
                range: token.range,
                surface: surface.to_owned(),
                kind: seed.kind.clone(),
                entity_ref: Some(MentionEntityRef::Known(seed.entity_id.clone())),
                source: Some(MentionSource::Known),
                confidence: 0.98,
                sentence_index,
            });
            index += 1;
            continue;
        }

        if matches!(token.pos, Some(PosTag::Pronoun)) {
            mentions.push(MentionSpan {
                range: token.range,
                surface: surface.to_owned(),
                kind: None,
                entity_ref: None,
                source: Some(MentionSource::Discovery),
                confidence: 0.65,
                sentence_index,
            });
            index += 1;
            continue;
        }

        if token.capitalized && matches!(token.token_class, Some(TokenClass::Word)) {
            let mut end = token.range.end;
            let mut last = index;
            while let Some(next) = tokens.get(last + 1) {
                if next.capitalized && matches!(next.token_class, Some(TokenClass::Word)) {
                    end = next.range.end;
                    last += 1;
                } else {
                    break;
                }
            }
            let surface = &text[token.range.start as usize..end as usize];
            mentions.push(MentionSpan {
                range: TextRange {
                    start: token.range.start,
                    end,
                },
                surface: surface.to_owned(),
                kind: Some(EntityKind::Character),
                entity_ref: Some(MentionEntityRef::Speculative(normalize_surface(surface))),
                source: Some(MentionSource::Discovery),
                confidence: 0.82,
                sentence_index,
            });
            index = last + 1;
            continue;
        }

        index += 1;
    }
    mentions
}

fn build_resolver_links(mentions: &[MentionSpan]) -> Vec<ResolverLink> {
    let mut links = Vec::new();
    let mut last_entity_by_surface = FxHashMap::<String, usize>::default();
    let mut antecedent = None::<usize>;
    for (index, mention) in mentions.iter().enumerate() {
        let normalized = normalize_surface(&mention.surface);
        if is_pronoun(&normalized) {
            if let Some(target_ix) = antecedent {
                let target = &mentions[target_ix];
                links.push(ResolverLink {
                    source_range: mention.range,
                    target_range: Some(target.range),
                    target_entity: target.entity_ref.clone(),
                    link_kind: Some(ResolverLinkKind::Pronoun),
                    confidence: 0.72,
                    sentence_index: mention.sentence_index,
                });
            }
            continue;
        }
        if let Some(previous_ix) = last_entity_by_surface.get(&normalized).copied() {
            let previous = &mentions[previous_ix];
            links.push(ResolverLink {
                source_range: mention.range,
                target_range: Some(previous.range),
                target_entity: previous.entity_ref.clone(),
                link_kind: Some(ResolverLinkKind::AliasCandidate),
                confidence: 0.61,
                sentence_index: mention.sentence_index,
            });
        }
        if mention.entity_ref.is_some() {
            antecedent = Some(index);
        }
        last_entity_by_surface.insert(normalized, index);
    }
    links
}

fn discover_narrative_hits(
    text: &str,
    tokens: &[TokenSpan],
    sentences: &[SentenceSpan],
) -> Vec<NarrativeVerbHit> {
    let mut hits = Vec::new();
    let mut sentence_cursor = 0usize;
    for token in tokens {
        if !matches!(token.pos, Some(PosTag::Verb)) {
            continue;
        }
        let normalized = normalize_token_surface(slice_or_empty(text, token.range));
        let (lemma, event_class, relation_type, transitivity) = classify_verb(&normalized);
        hits.push(NarrativeVerbHit {
            range: token.range,
            lemma,
            event_class,
            relation_type,
            transitivity,
            sentence_index: locate_sentence_cursor(sentences, &mut sentence_cursor, token.range),
            confidence: 0.7,
        });
    }
    hits
}

fn build_chunk_records(
    document: &IngestDocument,
    boundaries: &[BoundaryRecord],
    chunk_ranges: &[phoenix_chunker::Chunk],
) -> (Vec<ChunkRecord>, Vec<IndexedSpan>) {
    let mut chunks = Vec::with_capacity(chunk_ranges.len());
    let mut indexed_spans = Vec::with_capacity(chunk_ranges.len());
    let mut boundary_ix = 0usize;

    for (index, chunk) in chunk_ranges.iter().enumerate() {
        while boundary_ix + 1 < boundaries.len()
            && boundaries[boundary_ix + 1].range.start <= chunk.start as u32
        {
            boundary_ix += 1;
        }
        let boundary = boundaries.get(boundary_ix);
        let chapter_id = boundary.map(|value| value.chapter_id).unwrap_or(0);
        let boundary_label = boundary.map(|value| value.label.clone());
        let chunk_id = format!("{}:{index}", document.document_id.0);
        let text = document.text[chunk.start..chunk.end].trim().to_owned();
        let record = ChunkRecord {
            chunk_id: ChunkId(chunk_id.clone()),
            range: to_range(chunk.start, chunk.end),
            chapter_id,
            boundary_label: boundary_label.clone(),
            text: text.clone(),
        };
        let mut fields = Vec::new();
        if !document.title.trim().is_empty() {
            fields.push(IndexedTextField {
                field: LexicalField::Title,
                text: document.title.clone(),
            });
        }
        if !text.is_empty() {
            fields.push(IndexedTextField {
                field: LexicalField::Body,
                text,
            });
        }
        if let Some(summary) = boundary_label.as_deref().filter(|value| !value.trim().is_empty()) {
            fields.push(IndexedTextField {
                field: LexicalField::Summary,
                text: summary.to_owned(),
            });
        }
        indexed_spans.push(IndexedSpan {
            span_id: chunk_id,
            note_id: document.note_id.clone(),
            document_id: Some(document.document_id.clone()),
            scope: document.scope.clone(),
            fields,
        });
        chunks.push(record);
    }

    (chunks, indexed_spans)
}

fn build_native_entity_memory(snapshot: Option<&KernelGraphSnapshot>) -> NativeEntityMemory {
    let mut memory = NativeEntityMemory::default();
    let Some(snapshot) = snapshot else {
        return memory;
    };
    let _ = memory.kernel.rebuild_from_kernel_batches(vec![
        KernelMutationBatch {
            layer: KernelGraphLayer::Asserted,
            scope: KernelMutationScope::Full,
            recorded_at: None,
            vertices: snapshot.vertices.clone(),
            edges: snapshot.asserted_edges.clone(),
        },
        KernelMutationBatch {
            layer: KernelGraphLayer::Candidate,
            scope: KernelMutationScope::Full,
            recorded_at: None,
            vertices: Vec::new(),
            edges: snapshot.candidate_edges.clone(),
        },
    ]);
    for vertex in &snapshot.vertices {
        if let Some(facet) = vertex.entity_facet.as_ref() {
            if let Some(entity_id) = facet
                .canonical_entity_id
                .clone()
                .or_else(|| vertex.entity_id.clone())
            {
                if let Some(kind) = facet.entity_kind.clone() {
                    memory.entity_kinds.insert(entity_id, kind);
                }
            }
        }
    }
    for edge in &snapshot.asserted_edges {
        if edge.edge_type.0 != "alias_of" {
            continue;
        }
        let alias_surface = snapshot
            .vertices
            .iter()
            .find(|vertex| vertex.id.0 == edge.source_id.0)
            .and_then(|vertex| {
                vertex
                    .entity_facet
                    .as_ref()
                    .and_then(|facet| facet.surface.clone())
                    .or_else(|| {
                        vertex
                            .value
                            .get("name")
                            .and_then(|value| value.as_str())
                            .map(str::to_owned)
                    })
            });
        let entity_id = snapshot
            .vertices
            .iter()
            .find(|vertex| vertex.id.0 == edge.target_id.0)
            .and_then(|vertex| {
                vertex
                    .entity_facet
                    .as_ref()
                    .and_then(|facet| facet.canonical_entity_id.clone())
                    .or_else(|| vertex.entity_id.clone())
            });
        if let (Some(alias_surface), Some(entity_id)) = (alias_surface, entity_id) {
            memory
                .known_aliases
                .insert((entity_id, normalize_surface(&alias_surface)));
        }
    }
    memory
}

fn mention_id(document: &IngestDocument, mention_ix: usize) -> phoenix_semantic_v2::MentionId {
    phoenix_semantic_v2::MentionId(format!("mention::{}:{mention_ix}", document.document_id.0))
}

fn candidate_evidence(kind: &str, detail: impl Into<String>) -> CandidateEvidence {
    CandidateEvidence {
        kind: kind.to_owned(),
        detail: detail.into(),
    }
}

fn merge_candidate(
    candidates: &mut BTreeMap<String, CandidateEntity>,
    entity_id: String,
    source: &str,
    score_millis: i32,
    evidence: CandidateEvidence,
) {
    let entry = candidates
        .entry(entity_id.clone())
        .or_insert_with(|| CandidateEntity {
            entity_id,
            source: source.to_owned(),
            score_millis,
            evidence: Vec::new(),
        });
    if score_millis > entry.score_millis {
        entry.score_millis = score_millis;
        entry.source = source.to_owned();
    }
    if !entry.evidence.iter().any(|existing| existing == &evidence) {
        entry.evidence.push(evidence);
    }
}

fn build_prepared_mentions(
    document: &IngestDocument,
    scan: &ScanArtifact,
    chunks: &[ChunkRecord],
) -> Vec<PreparedMention> {
    let mention_by_range = scan
        .mentions
        .iter()
        .enumerate()
        .map(|(index, mention)| ((mention.range.start, mention.range.end), index))
        .collect::<FxHashMap<_, _>>();
    let mut links_by_source = FxHashMap::<usize, SmallVec<[usize; 4]>>::default();
    for link in &scan.resolver_links {
        let Some(source_ix) = mention_by_range
            .get(&(link.source_range.start, link.source_range.end))
            .copied()
        else {
            continue;
        };
        let Some(target_range) = link.target_range else {
            continue;
        };
        let Some(target_ix) = mention_by_range
            .get(&(target_range.start, target_range.end))
            .copied()
        else {
            continue;
        };
        links_by_source.entry(source_ix).or_default().push(target_ix);
    }

    scan.mentions
        .iter()
        .enumerate()
        .map(|(mention_ix, mention)| PreparedMention {
            mention_ix,
            mention_id: mention_id(document, mention_ix),
            normalized: normalize_surface(&mention.surface),
            chunk_id: chunks
                .iter()
                .find(|chunk| range_contains(chunk.range, mention.range))
                .map(|chunk| chunk.chunk_id.0.clone()),
            linked_mentions: links_by_source.remove(&mention_ix).unwrap_or_default(),
        })
        .collect()
}

fn resolve_mentions(
    document: &IngestDocument,
    scan: &ScanArtifact,
    _structure: &StructureArtifact,
    chunks: &[ChunkRecord],
    entity_memory: &NativeEntityMemory,
) -> (
    Vec<MentionResolution>,
    Vec<ResolvedMention>,
    Vec<AliasConfirmation>,
    Vec<Diagnostic>,
) {
    let prepared = build_prepared_mentions(document, scan, chunks);
    let mut diagnostics = Vec::<Diagnostic>::new();
    let mut base_candidates = Vec::<BTreeMap<String, CandidateEntity>>::with_capacity(prepared.len());
    let mut surface_support = FxHashMap::<String, FxHashMap<String, usize>>::default();

    for prepared_mention in &prepared {
        let mention = &scan.mentions[prepared_mention.mention_ix];
        let mut candidates = BTreeMap::<String, CandidateEntity>::new();

        if let Some(MentionEntityRef::Known(entity_id)) = mention.entity_ref.as_ref() {
            merge_candidate(
                &mut candidates,
                entity_id.0.clone(),
                "seed",
                1800,
                candidate_evidence("seed", mention.surface.clone()),
            );
        }

        let kernel_candidates = entity_memory.kernel.entity_candidates(KernelEntityResolveRequest {
            surface: Some(mention.surface.clone()),
            mention_vertex_id: Some(prepared_mention.mention_id.0.clone()),
            include_candidate_graph: true,
            limit: Some(8),
            ..KernelEntityResolveRequest::default()
        });
        for candidate in kernel_candidates {
            let relation = candidate.relation_type.as_deref().unwrap_or("kernel");
            let (source, bonus) = match relation {
                "alias_of" => ("kernel_alias", 100),
                "resolved_to" => ("kernel_resolved", 120),
                "candidate_same_as" => ("kernel_candidate", 0),
                _ => ("kernel_alias", 0),
            };
            merge_candidate(
                &mut candidates,
                candidate.entity_id.clone(),
                source,
                ((candidate.score * 1000.0).round() as i32) + 700 + bonus,
                candidate_evidence("kernel", relation.to_owned()),
            );
        }

        for link in scan
            .resolver_links
            .iter()
            .filter(|link| link.source_range == mention.range)
        {
            let Some(target_entity) = link.target_entity.as_ref() else {
                continue;
            };
            let Some(entity_id) = entity_id_from_ref(document, target_entity) else {
                continue;
            };
            let (source, score) = match link.link_kind {
                Some(ResolverLinkKind::Pronoun) => ("pronoun_link", 1150),
                Some(ResolverLinkKind::AliasCandidate) => ("alias_link", 950),
                None => ("alias_link", 850),
            };
            merge_candidate(
                &mut candidates,
                entity_id.0.clone(),
                source,
                score,
                candidate_evidence("resolver_link", format!("{:?}", link.link_kind)),
            );
        }

        if !prepared_mention.normalized.is_empty() {
            for other in &prepared {
                if other.mention_ix == prepared_mention.mention_ix
                    || other.normalized != prepared_mention.normalized
                {
                    continue;
                }
                let other_mention = &scan.mentions[other.mention_ix];
                if let Some(MentionEntityRef::Known(entity_id)) = other_mention.entity_ref.as_ref() {
                    merge_candidate(
                        &mut candidates,
                        entity_id.0.clone(),
                        "local_surface",
                        900,
                        candidate_evidence("local_surface", other_mention.surface.clone()),
                    );
                }
            }
        }

        if !is_pronoun(&prepared_mention.normalized) && !prepared_mention.normalized.is_empty() {
            merge_candidate(
                &mut candidates,
                speculative_entity_id(&document.document_id, &mention.surface),
                "new_speculative",
                420,
                candidate_evidence("new_speculative", prepared_mention.normalized.clone()),
            );
        }

        for candidate in candidates.values() {
            if matches!(
                candidate.source.as_str(),
                "seed" | "kernel_alias" | "kernel_resolved" | "pronoun_link" | "alias_link"
            ) {
                surface_support
                    .entry(prepared_mention.normalized.clone())
                    .or_default()
                    .entry(candidate.entity_id.clone())
                    .and_modify(|count| *count += 1)
                    .or_insert(1);
            }
        }
        base_candidates.push(candidates);
    }

    let mut resolutions = Vec::with_capacity(prepared.len());
    let mut resolved_mentions = Vec::with_capacity(prepared.len());
    let mut alias_confirmations = Vec::<AliasConfirmation>::new();

    for (index, prepared_mention) in prepared.iter().enumerate() {
        let mention = &scan.mentions[prepared_mention.mention_ix];
        let mut candidates = base_candidates[index].clone();
        if let Some(surface_entities) = surface_support.get(&prepared_mention.normalized) {
            for (entity_id, support) in surface_entities {
                merge_candidate(
                    &mut candidates,
                    entity_id.clone(),
                    "local_surface",
                    780 + (*support as i32 * 80),
                    candidate_evidence("surface_cluster", support.to_string()),
                );
            }
        }
        for linked_ix in &prepared_mention.linked_mentions {
            if let Some(candidate) = base_candidates
                .get(*linked_ix)
                .and_then(|map| map.values().max_by_key(|candidate| candidate.score_millis))
            {
                merge_candidate(
                    &mut candidates,
                    candidate.entity_id.clone(),
                    candidate.source.as_str(),
                    candidate.score_millis + if candidate.source == "pronoun_link" { 180 } else { 120 },
                    candidate_evidence("linked_mention", linked_ix.to_string()),
                );
            }
        }
        for candidate in candidates.values_mut() {
            if let Some(kind) = mention.kind.as_ref() {
                if let Some(existing_kind) = entity_memory.entity_kinds.get(&candidate.entity_id) {
                    if existing_kind != entity_kind_name(kind) {
                        candidate.score_millis -= 220;
                        candidate
                            .evidence
                            .push(candidate_evidence("kind_penalty", existing_kind.clone()));
                    }
                }
            }
            if candidate.source == "new_speculative" && mention.confidence >= 0.8 {
                candidate.score_millis += 180;
            }
            if prepared_mention.chunk_id.is_some() {
                candidate.score_millis += 20;
            }
        }

        let mut candidate_list = candidates.into_values().collect::<Vec<_>>();
        candidate_list.sort_by(|left, right| {
            right
                .score_millis
                .cmp(&left.score_millis)
                .then_with(|| left.entity_id.cmp(&right.entity_id))
        });
        let top = candidate_list.first().cloned();
        let runner_up = candidate_list.get(1).cloned();
        let top_score = top.as_ref().map(|candidate| candidate.score_millis).unwrap_or_default();
        let margin = top_score.saturating_sub(
            runner_up
                .as_ref()
                .map(|candidate| candidate.score_millis)
                .unwrap_or_default(),
        );

        let pronoun = is_pronoun(&prepared_mention.normalized);
        let resolved = top.as_ref().and_then(|candidate| {
            let threshold = if pronoun { 1100 } else { 900 };
            let margin_threshold = if pronoun { 220 } else { 180 };
            let speculative = candidate.source == "new_speculative";
            if candidate.score_millis >= threshold
                && margin >= margin_threshold
                && (!speculative || candidate.score_millis >= 700)
            {
                Some(EntityId(candidate.entity_id.clone()))
            } else {
                None
            }
        });

        let decision = if let Some(entity_id) = resolved.clone() {
            diagnostics.push(Diagnostic {
                code: match top.as_ref().map(|candidate| candidate.source.as_str()) {
                    Some("seed") => "er_known_seed_match",
                    Some("kernel_alias") | Some("kernel_resolved") => "er_kernel_alias_match",
                    Some("pronoun_link") => "er_pronoun_link_match",
                    Some("alias_link") | Some("local_surface") => "er_alias_link_match",
                    Some("new_speculative") => "er_new_speculative_entity",
                    _ => "er_collective_merge",
                }
                .to_owned(),
                message: format!(
                    "Resolved mention '{}' to '{}' with score {} and margin {}.",
                    mention.surface, entity_id.0, top_score, margin
                ),
            });
            ResolutionDecisionState {
                kind: ResolvedMentionKind::Resolved,
                entity_id: Some(entity_id),
                confidence_millis: top_score.max(0) as u32,
                margin_millis: margin.max(0) as u32,
            }
        } else if !candidate_list.is_empty() {
            diagnostics.push(Diagnostic {
                code: "er_ambiguous_resolution".to_owned(),
                message: format!(
                    "Mention '{}' stayed ambiguous; top candidate score {} margin {}.",
                    mention.surface, top_score, margin
                ),
            });
            ResolutionDecisionState {
                kind: ResolvedMentionKind::Ambiguous,
                entity_id: None,
                confidence_millis: top_score.max(0) as u32,
                margin_millis: margin.max(0) as u32,
            }
        } else {
            ResolutionDecisionState {
                kind: ResolvedMentionKind::Unresolved,
                entity_id: None,
                confidence_millis: 0,
                margin_millis: 0,
            }
        };

        if let (ResolvedMentionKind::Resolved, Some(entity_id), Some(top_candidate)) = (
            &decision.kind,
            decision.entity_id.as_ref(),
            top.as_ref(),
        ) {
            let normalized = normalize_surface(&mention.surface);
            if !normalized.is_empty()
                && !pronoun
                && normalized != normalize_surface(&entity_id.0)
                && top_candidate.source != "seed"
                && top_candidate.source != "kernel_resolved"
                && top_candidate.source != "pronoun_link"
                && candidate_has_alias_signal(top_candidate)
                && decision.confidence_millis >= 1000
                && decision.margin_millis >= 260
                && !entity_memory
                    .known_aliases
                    .contains(&(entity_id.0.clone(), normalized.clone()))
            {
                alias_confirmations.push(AliasConfirmation {
                    alias_surface: mention.surface.clone(),
                    normalized,
                    entity_id: entity_id.clone(),
                    confidence_millis: decision.confidence_millis,
                    mention_id: prepared_mention.mention_id.clone(),
                });
            } else if !normalized.is_empty()
                && normalized != normalize_surface(&entity_id.0)
                && !entity_memory
                    .known_aliases
                    .contains(&(entity_id.0.clone(), normalized.clone()))
            {
                diagnostics.push(Diagnostic {
                    code: "er_alias_rejected_low_margin".to_owned(),
                    message: format!(
                        "Alias confirmation for '{}' -> '{}' was rejected because the evidence was not alias-specific enough.",
                        mention.surface, entity_id.0
                    ),
                });
            }
        } else if top.is_some() && margin < 180 {
            diagnostics.push(Diagnostic {
                code: "er_alias_rejected_low_margin".to_owned(),
                message: format!(
                    "Alias-style resolution for '{}' was rejected because the candidate margin was too small.",
                    mention.surface
                ),
            });
        }

        resolutions.push(MentionResolution {
            mention_ix: prepared_mention.mention_ix,
            mention_id: prepared_mention.mention_id.clone(),
            entity_id: decision.entity_id.clone(),
            candidates: candidate_list.clone(),
            decision: decision.clone(),
        });
        resolved_mentions.push(ResolvedMention {
            mention_id: prepared_mention.mention_id.clone(),
            mention_index: prepared_mention.mention_ix,
            range: mention.range,
            surface: mention.surface.clone(),
            normalized: prepared_mention.normalized.clone(),
            kind: mention.kind.clone(),
            entity_id: decision.entity_id.clone(),
            decision: ResolutionDecision {
                status: match decision.kind {
                    ResolvedMentionKind::Resolved => "resolved".to_owned(),
                    ResolvedMentionKind::Ambiguous => "ambiguous".to_owned(),
                    ResolvedMentionKind::Unresolved => "unresolved".to_owned(),
                },
                confidence_millis: decision.confidence_millis,
                margin_millis: decision.margin_millis,
            },
            candidates: candidate_list,
        });
    }

    alias_confirmations.sort_by(|left, right| {
        left.entity_id
            .0
            .cmp(&right.entity_id.0)
            .then_with(|| left.normalized.cmp(&right.normalized))
            .then_with(|| left.mention_id.0.cmp(&right.mention_id.0))
    });
    alias_confirmations.dedup_by(|left, right| {
        left.entity_id == right.entity_id && left.normalized == right.normalized
    });

    (resolutions, resolved_mentions, alias_confirmations, diagnostics)
}

fn build_semantic_records(
    _document: &IngestDocument,
    scan: &ScanArtifact,
    structure: &StructureArtifact,
    chunks: &[ChunkRecord],
    resolutions: &[MentionResolution],
) -> (
    Vec<SemanticEntityRecord>,
    Vec<SemanticRelationRecord>,
    usize,
    Vec<Diagnostic>,
) {
    let mut entities = FxHashMap::<String, EntityAccumulator>::default();
    let mut diagnostics = Vec::<Diagnostic>::new();
    let resolution_by_range = resolutions
        .iter()
        .map(|resolution| {
            let mention = &scan.mentions[resolution.mention_ix];
            ((mention.range.start, mention.range.end), resolution)
        })
        .collect::<FxHashMap<_, _>>();
    for resolution in resolutions {
        let Some(entity_id) = resolution.entity_id.clone() else {
            continue;
        };
        let mention = &scan.mentions[resolution.mention_ix];
        let chunk_id = chunks
            .iter()
            .find(|chunk| range_contains(chunk.range, mention.range))
            .map(|chunk| chunk.chunk_id.0.clone());
        let entry = entities
            .entry(entity_id.0.clone())
            .or_insert_with(|| EntityAccumulator {
                entity_id: entity_id.clone(),
                canonical_name: mention.surface.clone(),
                aliases: SmallVec::new(),
                kind: mention.kind.clone(),
                mention_count: 0,
                chunk_ids: SmallVec::new(),
            });
        entry.mention_count += 1;
        if entry.canonical_name != mention.surface
            && !entry.aliases.iter().any(|alias| alias == &mention.surface)
        {
            entry.aliases.push(mention.surface.clone());
        }
        if let Some(chunk_id) = chunk_id {
            if !entry.chunk_ids.iter().any(|existing| existing == &chunk_id) {
                entry.chunk_ids.push(chunk_id);
            }
        }
    }

    let relation_records = structure
        .relations
        .iter()
        .filter_map(|relation| {
            let source_entity_id = relation
                .subject
                .as_ref()
                .and_then(|slot| resolution_by_range.get(&(slot.range.start, slot.range.end)))
                .and_then(|resolution| resolution.entity_id.clone());
            let target_entity_id = relation
                .object
                .as_ref()
                .and_then(|slot| resolution_by_range.get(&(slot.range.start, slot.range.end)))
                .and_then(|resolution| resolution.entity_id.clone());
            if source_entity_id.is_none() || target_entity_id.is_none() {
                diagnostics.push(Diagnostic {
                    code: "er_relation_skipped_unresolved_entity".to_owned(),
                    message: format!(
                        "Skipped asserted relation '{}' in sentence {} because one or more arguments stayed unresolved.",
                        relation.relation_type, relation.sentence_index
                    ),
                });
            }
            Some(SemanticRelationRecord {
                source_entity_id: source_entity_id?,
                target_entity_id: target_entity_id?,
                edge_type: relation.relation_type.clone(),
                sentence_index: relation.sentence_index,
                chunk_id: chunks
                    .iter()
                    .find(|chunk| {
                        chunk.range.start <= relation.verb_range.start
                            && chunk.range.end >= relation.verb_range.end
                    })
                    .map(|chunk| chunk.chunk_id.0.clone()),
            })
        })
        .collect::<Vec<_>>();

    let mut entity_records = entities
        .into_values()
        .map(|entity| SemanticEntityRecord {
            entity_id: entity.entity_id,
            canonical_name: entity.canonical_name,
            aliases: entity.aliases.into_vec(),
            kind: entity.kind,
            mention_count: entity.mention_count,
            chunk_ids: entity.chunk_ids.into_vec(),
        })
        .collect::<Vec<_>>();
    entity_records.sort_by(|left, right| left.entity_id.0.cmp(&right.entity_id.0));

    let discovery_count = scan
        .mentions
        .iter()
        .filter(|mention| {
            matches!(
                mention.source,
                Some(MentionSource::Discovery | MentionSource::Fuzzy)
            )
        })
        .count();

    (entity_records, relation_records, discovery_count, diagnostics)
}

fn build_kernel_batch(
    document: &IngestDocument,
    scan: &ScanArtifact,
    chunks: &[ChunkRecord],
    entities: &[SemanticEntityRecord],
    relations: &[SemanticRelationRecord],
    resolutions: &[MentionResolution],
    alias_confirmations: &[AliasConfirmation],
    evidence_spans: &[EvidenceSpan],
) -> KernelMutationBatch {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();
    let document_vertex_id = format!("doc::{}", document.document_id.0);
    let scope_key = scope_storage_key(&document.scope);
    vertices.push(KernelVertex {
        id: KernelVertexId(document_vertex_id.clone()),
        kind: "document".to_owned(),
        labels: vec!["document".to_owned()],
        weight: 1,
        value: json!({ "title": document.title }),
        attributes: json!({
            "documentId": document.document_id.0,
            "noteId": document.note_id.as_ref().map(|note_id| note_id.0.clone()),
            "scopeKey": scope_key,
        }),
        document_id: Some(document.document_id.0.clone()),
        chapter_id: Some(0),
        chapters: vec![0],
        ..KernelVertex::default()
    });

    for chunk in chunks {
        vertices.push(KernelVertex {
            id: KernelVertexId(chunk.chunk_id.0.clone()),
            kind: "chunk".to_owned(),
            labels: vec!["chunk".to_owned()],
            weight: 1,
            value: json!({ "text": chunk.text }),
            attributes: json!({
                "documentId": document.document_id.0,
                "chunkId": chunk.chunk_id.0,
                "scopeKey": scope_key,
            }),
            search_chunk_id: Some(chunk.chunk_id.0.clone()),
            document_id: Some(document.document_id.0.clone()),
            chapter_id: Some(chunk.chapter_id),
            chapters: vec![chunk.chapter_id],
            boundary_ordinal: Some(chunk.chapter_id),
            boundary_kind: Some(BoundaryKind::Section),
            boundary_ordinals: vec![chunk.chapter_id],
            ..KernelVertex::default()
        });
        edges.push(KernelEdge {
            source_id: KernelVertexId(document_vertex_id.clone()),
            target_id: KernelVertexId(chunk.chunk_id.0.clone()),
            edge_type: KernelEdgeType("contains".to_owned()),
            weight: 1,
            attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
            document_id: Some(document.document_id.0.clone()),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        });
    }

    for entity in entities {
        let entity_vertex_id = format!("entity::{}", entity.entity_id.0);
        vertices.push(KernelVertex {
            id: KernelVertexId(entity_vertex_id.clone()),
            kind: "entity".to_owned(),
            labels: vec!["entity".to_owned()],
            weight: entity.mention_count.max(1) as i64,
            value: json!({ "name": entity.canonical_name, "aliases": entity.aliases }),
            attributes: json!({
                "documentId": document.document_id.0,
                "entityId": entity.entity_id.0,
                "scopeKey": scope_key,
            }),
            entity_id: Some(entity.entity_id.0.clone()),
            document_id: Some(document.document_id.0.clone()),
            chapter_id: Some(0),
            chapters: vec![0],
            ..KernelVertex::default()
        });
        edges.push(KernelEdge {
            source_id: KernelVertexId(document_vertex_id.clone()),
            target_id: KernelVertexId(entity_vertex_id.clone()),
            edge_type: KernelEdgeType("entity".to_owned()),
            weight: entity.mention_count.max(1) as i64,
            attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
            document_id: Some(document.document_id.0.clone()),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        });
    }

    for resolution in resolutions {
        let mention = &scan.mentions[resolution.mention_ix];
        let mention_vertex_id = resolution.mention_id.0.clone();
        let chunk_id = chunks
            .iter()
            .find(|chunk| range_contains(chunk.range, mention.range))
            .map(|chunk| chunk.chunk_id.0.clone());
        vertices.push(KernelVertex {
            id: KernelVertexId(mention_vertex_id.clone()),
            kind: "mention".to_owned(),
            labels: vec!["mention".to_owned()],
            weight: 1,
            value: json!({ "surface": mention.surface }),
            attributes: json!({
                "documentId": document.document_id.0,
                "mentionIndex": resolution.mention_ix,
                "source": mention.source,
                "sentenceIndex": mention.sentence_index,
                "scopeKey": scope_key,
            }),
            document_id: Some(document.document_id.0.clone()),
            entity_facet: Some(phoenix_graph_kernel::KernelEntityFacet {
                canonical_entity_id: resolution.entity_id.as_ref().map(|value| value.0.clone()),
                surface: Some(mention.surface.clone()),
                entity_kind: mention
                    .kind
                    .as_ref()
                    .map(|kind| entity_kind_name(kind).to_owned()),
            }),
            provenance: KernelProvenance {
                resolver: Some("native_collective_er".to_owned()),
                source: Some("mention".to_owned()),
                confidence: Some(resolution.decision.confidence_millis as f64 / 1000.0),
                evidence_refs: resolution
                    .candidates
                    .iter()
                    .flat_map(|candidate| {
                        candidate
                            .evidence
                            .iter()
                            .map(|evidence| format!("{}:{}", evidence.kind, evidence.detail))
                    })
                    .collect(),
            },
            ..KernelVertex::default()
        });
        if let Some(chunk_id) = chunk_id {
            edges.push(KernelEdge {
                source_id: KernelVertexId(chunk_id),
                target_id: KernelVertexId(mention_vertex_id.clone()),
                edge_type: KernelEdgeType("mentions".to_owned()),
                weight: 1,
                attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
                document_id: Some(document.document_id.0.clone()),
                layer: KernelGraphLayer::Asserted,
                ..KernelEdge::default()
            });
        }
        if let Some(entity_id) = resolution.entity_id.as_ref() {
            edges.push(KernelEdge {
                source_id: KernelVertexId(mention_vertex_id.clone()),
                target_id: KernelVertexId(format!("entity::{}", entity_id.0)),
                edge_type: KernelEdgeType("resolved_to".to_owned()),
                weight: 1,
                attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
                document_id: Some(document.document_id.0.clone()),
                provenance: KernelProvenance {
                    resolver: Some("native_collective_er".to_owned()),
                    source: Some(
                        resolution
                            .candidates
                            .first()
                            .map(|candidate| candidate.source.clone())
                            .unwrap_or_else(|| "resolved".to_owned()),
                    ),
                    confidence: Some(resolution.decision.confidence_millis as f64 / 1000.0),
                    evidence_refs: resolution
                        .candidates
                        .iter()
                        .flat_map(|candidate| {
                            candidate
                                .evidence
                                .iter()
                                .map(|evidence| format!("{}:{}", evidence.kind, evidence.detail))
                        })
                        .collect(),
                },
                resolution_facet: Some(KernelResolutionFacet {
                    strategy: Some("collective".to_owned()),
                    candidate_rank: Some(0),
                    confidence: Some(resolution.decision.confidence_millis as f64 / 1000.0),
                    replaced_edge_key: None,
                }),
                layer: KernelGraphLayer::Asserted,
                ..KernelEdge::default()
            });
        } else {
            for (rank, candidate) in resolution.candidates.iter().take(4).enumerate() {
                edges.push(KernelEdge {
                    source_id: KernelVertexId(mention_vertex_id.clone()),
                    target_id: KernelVertexId(format!("entity::{}", candidate.entity_id)),
                    edge_type: KernelEdgeType("candidate_same_as".to_owned()),
                    weight: candidate.score_millis.max(1) as i64,
                    attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
                    document_id: Some(document.document_id.0.clone()),
                    provenance: KernelProvenance {
                        resolver: Some("native_collective_er".to_owned()),
                        source: Some(candidate.source.clone()),
                        confidence: Some(candidate.score_millis as f64 / 1000.0),
                        evidence_refs: candidate
                            .evidence
                            .iter()
                            .map(|evidence| format!("{}:{}", evidence.kind, evidence.detail))
                            .collect(),
                    },
                    resolution_facet: Some(KernelResolutionFacet {
                        strategy: Some("collective".to_owned()),
                        candidate_rank: Some(rank as u32),
                        confidence: Some(candidate.score_millis as f64 / 1000.0),
                        replaced_edge_key: None,
                    }),
                    layer: KernelGraphLayer::Candidate,
                    ..KernelEdge::default()
                });
            }
        }
    }

    for confirmation in alias_confirmations {
        let alias_vertex_id = format!(
            "alias::{}::{}",
            confirmation.entity_id.0, confirmation.normalized
        );
        vertices.push(KernelVertex {
            id: KernelVertexId(alias_vertex_id.clone()),
            kind: "alias".to_owned(),
            labels: vec!["alias".to_owned()],
            weight: 1,
            value: json!({ "name": confirmation.alias_surface }),
            attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
            document_id: Some(document.document_id.0.clone()),
            entity_facet: Some(phoenix_graph_kernel::KernelEntityFacet {
                canonical_entity_id: Some(confirmation.entity_id.0.clone()),
                surface: Some(confirmation.alias_surface.clone()),
                entity_kind: Some("alias".to_owned()),
            }),
            ..KernelVertex::default()
        });
        edges.push(KernelEdge {
            source_id: KernelVertexId(alias_vertex_id),
            target_id: KernelVertexId(format!("entity::{}", confirmation.entity_id.0)),
            edge_type: KernelEdgeType("alias_of".to_owned()),
            weight: confirmation.confidence_millis.max(1) as i64,
            attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
            document_id: Some(document.document_id.0.clone()),
            provenance: KernelProvenance {
                resolver: Some("native_collective_er".to_owned()),
                source: Some("confirmed_alias".to_owned()),
                confidence: Some(confirmation.confidence_millis as f64 / 1000.0),
                evidence_refs: vec![confirmation.mention_id.0.clone()],
            },
            resolution_facet: Some(KernelResolutionFacet {
                strategy: Some("collective".to_owned()),
                candidate_rank: Some(0),
                confidence: Some(confirmation.confidence_millis as f64 / 1000.0),
                replaced_edge_key: None,
            }),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        });
    }

    for relation in relations {
        edges.push(KernelEdge {
            source_id: KernelVertexId(format!("entity::{}", relation.source_entity_id.0)),
            target_id: KernelVertexId(format!("entity::{}", relation.target_entity_id.0)),
            edge_type: KernelEdgeType(relation.edge_type.clone()),
            weight: 1,
            attributes: json!({
                "documentId": document.document_id.0,
                "sentenceIndex": relation.sentence_index,
                "scopeKey": scope_key,
            }),
            document_id: Some(document.document_id.0.clone()),
            layer: KernelGraphLayer::Asserted,
            ..KernelEdge::default()
        });
    }

    vertices.extend(evidence_spans.iter().enumerate().map(|(index, evidence)| KernelVertex {
        id: KernelVertexId(format!("evidence::{}:{index}", document.document_id.0)),
        kind: "evidence".to_owned(),
        labels: vec!["evidence".to_owned()],
        weight: 1,
        value: json!({ "label": evidence.label }),
        attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
        document_id: Some(document.document_id.0.clone()),
        ..KernelVertex::default()
    }));

    for (index, evidence) in evidence_spans.iter().enumerate() {
        let evidence_vertex_id = format!("evidence::{}:{index}", document.document_id.0);
        for resolution in resolutions.iter().filter(|resolution| {
            resolution.entity_id.is_some()
                && range_contains(evidence.range, scan.mentions[resolution.mention_ix].range)
        }) {
            if let Some(entity_id) = resolution.entity_id.as_ref() {
                edges.push(KernelEdge {
                    source_id: KernelVertexId(evidence_vertex_id.clone()),
                    target_id: KernelVertexId(format!("entity::{}", entity_id.0)),
                    edge_type: KernelEdgeType("evidence_for".to_owned()),
                    weight: 1,
                    attributes: json!({ "documentId": document.document_id.0, "scopeKey": scope_key }),
                    document_id: Some(document.document_id.0.clone()),
                    provenance: KernelProvenance {
                        resolver: Some("native_collective_er".to_owned()),
                        source: Some("evidence_span".to_owned()),
                        confidence: Some(1.0),
                        evidence_refs: vec![evidence_vertex_id.clone()],
                    },
                    layer: KernelGraphLayer::Asserted,
                    ..KernelEdge::default()
                });
            }
        }
    }

    KernelMutationBatch {
        layer: KernelGraphLayer::Asserted,
        scope: KernelMutationScope::Document {
            document_id: document.document_id.0.clone(),
        },
        recorded_at: None,
        vertices,
        edges,
    }
}

fn frame_slot_from_mention(mention: &MentionSpan) -> FrameSlot {
    FrameSlot {
        range: mention.range,
        entity_ref: mention.entity_ref.clone(),
        confidence: mention.confidence,
    }
}

fn locate_sentence(sentences: &[SentenceSpan], range: TextRange) -> Option<usize> {
    let mut left = 0usize;
    let mut right = sentences.len();
    while left < right {
        let middle = left + (right - left) / 2;
        let sentence = &sentences[middle];
        if sentence.range.end < range.end {
            left = middle + 1;
        } else if sentence.range.start > range.start {
            right = middle;
        } else {
            return Some(sentence.index);
        }
    }
    None
}

fn locate_sentence_cursor(
    sentences: &[SentenceSpan],
    cursor: &mut usize,
    range: TextRange,
) -> usize {
    if sentences.is_empty() {
        return 0;
    }
    while *cursor + 1 < sentences.len() && sentences[*cursor].range.end < range.end {
        *cursor += 1;
    }
    sentences
        .get(*cursor)
        .filter(|sentence| sentence.range.start <= range.start && sentence.range.end >= range.end)
        .map(|sentence| sentence.index)
        .or_else(|| locate_sentence(sentences, range))
        .unwrap_or_default()
}

fn extract_boundaries(text: &str) -> Vec<BoundaryRecord> {
    let mut boundaries = Vec::new();
    let mut cursor = 0usize;
    let mut chapter_id = 0u32;
    for line in text.lines() {
        let start = cursor;
        let end = start + line.len();
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            let lower = trimmed.to_ascii_lowercase();
            let is_heading = trimmed.starts_with('#')
                || lower.starts_with("chapter ")
                || matches!(lower.as_str(), "prologue" | "epilogue" | "preface" | "introduction");
            if is_heading {
                let label = trimmed.trim_start_matches('#').trim().to_owned();
                boundaries.push(BoundaryRecord {
                    label,
                    range: to_range(start, end),
                    chapter_id,
                    is_chapter: is_heading,
                });
                chapter_id = chapter_id.saturating_add(1);
            }
        }
        cursor = end + 1;
    }
    boundaries
}

fn normalize_surface(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ch.is_whitespace())
        .flat_map(|ch| ch.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_token_surface(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_alphanumeric() {
            normalized.extend(ch.to_lowercase());
        }
    }
    normalized
}

fn candidate_has_alias_signal(candidate: &CandidateEntity) -> bool {
    matches!(candidate.source.as_str(), "kernel_alias" | "alias_link" | "local_surface")
        || candidate.evidence.iter().any(|evidence| {
            matches!(
                evidence.kind.as_str(),
                "kernel" | "resolver_link" | "local_surface" | "surface_cluster"
            )
        })
}

fn entity_kind_name(kind: &EntityKind) -> &'static str {
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

fn document_alias_entries(document: &DocumentArchive) -> Vec<AliasEntry> {
    let mut entries = BTreeMap::<String, BTreeMap<(String, String), usize>>::new();
    for entity in &document.entities {
        let forms = std::iter::once(entity.canonical_name.clone()).chain(entity.aliases.clone());
        for form in forms {
            let normalized = normalize_surface(&form);
            if normalized.is_empty() {
                continue;
            }
            entries
                .entry(normalized)
                .or_default()
                .entry((entity.entity_id.0.clone(), document.manifest.document_id.clone()))
                .and_modify(|count| *count += entity.mention_count)
                .or_insert(entity.mention_count);
        }
    }

    entries
        .into_iter()
        .map(|(normalized, postings)| AliasEntry {
            normalized,
            postings: postings
                .into_iter()
                .map(|((entity_id, document_id), mention_count)| AliasPosting {
                    entity_id,
                    document_id,
                    mention_count,
                })
                .collect(),
        })
        .collect()
}

fn speculative_entity_id(document_id: &DocumentId, surface: &str) -> String {
    format!("{}::{}", document_id.0, normalize_surface(surface).replace(' ', "_"))
}

fn entity_id_from_ref(document: &IngestDocument, entity_ref: &MentionEntityRef) -> Option<EntityId> {
    match entity_ref {
        MentionEntityRef::Known(entity_id) => Some(entity_id.clone()),
        MentionEntityRef::Speculative(speculative) => Some(EntityId(format!(
            "{}::{}",
            document.document_id.0,
            speculative.replace(' ', "_")
        ))),
    }
}

fn is_pronoun(value: &str) -> bool {
    matches!(
        value,
        "he" | "she" | "they" | "them" | "him" | "her" | "we" | "us" | "i" | "you" | "it"
    )
}

fn is_verb_token(value: &str) -> bool {
    matches!(
        value,
        "attack" | "attacked" | "attacks" | "met" | "meet" | "meets" | "rose" | "rise"
            | "rises" | "woke" | "wake" | "wakes" | "wrote" | "write" | "writes"
            | "mapped" | "map" | "maps" | "gave" | "give" | "gives" | "waited" | "wait"
            | "waits" | "saw" | "see" | "sees" | "found" | "find" | "finds" | "fought"
            | "fight" | "fights" | "moved" | "move" | "moves" | "crossed" | "cross"
            | "crosses"
    ) || value.ends_with("ed")
}

fn classify_verb(value: &str) -> (String, String, String, Option<NarrativeTransitivity>) {
    match value {
        "attack" | "attacked" | "attacks" | "fight" | "fought" | "fights" => (
            "attack".to_owned(),
            "conflict".to_owned(),
            "attacks".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
        "met" | "meet" | "meets" => (
            "meet".to_owned(),
            "interaction".to_owned(),
            "meets".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
        "gave" | "give" | "gives" => (
            "give".to_owned(),
            "transfer".to_owned(),
            "gives".to_owned(),
            Some(NarrativeTransitivity::Ditransitive),
        ),
        "wrote" | "write" | "writes" | "mapped" | "map" | "maps" => (
            value.trim_end_matches("ed").trim_end_matches('s').to_owned(),
            "creation".to_owned(),
            "writes".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
        other => (
            other.trim_end_matches("ed").trim_end_matches('s').to_owned(),
            "action".to_owned(),
            "relates_to".to_owned(),
            Some(NarrativeTransitivity::Transitive),
        ),
    }
}

fn range_contains(container: TextRange, inner: TextRange) -> bool {
    container.start <= inner.start && container.end >= inner.end
}

fn count_token_words(tokens: &[TokenSpan]) -> usize {
    tokens
        .iter()
        .filter(|token| matches!(token.token_class, Some(TokenClass::Word)))
        .count()
}

fn slice_or_empty(text: &str, range: TextRange) -> &str {
    text.get(range.start as usize..range.end as usize)
        .unwrap_or_default()
}

fn to_range(start: usize, end: usize) -> TextRange {
    TextRange {
        start: start.min(u32::MAX as usize) as u32,
        end: end.min(u32::MAX as usize) as u32,
    }
}

fn trim_start_offset(text: &str, start: usize, end: usize) -> usize {
    let slice = &text[start..end];
    start + slice.len().saturating_sub(slice.trim_start().len())
}

fn trim_end_offset(text: &str, start: usize, end: usize) -> usize {
    let slice = &text[start..end];
    start + slice.trim_end().len()
}

fn is_front_matter_label(label: &str) -> bool {
    matches!(
        label.trim().to_ascii_lowercase().as_str(),
        "prologue" | "preface" | "introduction"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_graph_kernel::KernelVertexClass;
    use phoenix_store_lmdb::PhoenixLmdbStore;
    use phoenix_types::{GenderHint, NoteId};
    use std::env;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> std::path::PathBuf {
        env::temp_dir().join(format!(
            "phoenix-invarant-v2-test-{name}-{}",
            SystemTime::now().duration_since(UNIX_EPOCH).expect("time").as_nanos()
        ))
    }

    #[test]
    fn scan_and_structure_capture_mentions_and_relations() {
        let engine = PhoenixInvarantV2::default();
        let scan = engine.scan_parts(
            "Luffy attacked Zoro.",
            &ScopeKey::default(),
            &[
                ResolverEntitySeed {
                    entity_id: EntityId("luffy".to_owned()),
                    canonical_name: "Luffy".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Character),
                    gender: Some(GenderHint::Male),
                    number: None,
                    scope: ScopeKey::default(),
                },
                ResolverEntitySeed {
                    entity_id: EntityId("zoro".to_owned()),
                    canonical_name: "Zoro".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(EntityKind::Character),
                    gender: Some(GenderHint::Male),
                    number: None,
                    scope: ScopeKey::default(),
                },
            ],
        );
        assert_eq!(scan.mentions.len(), 2);
        let structure = engine.build_structure_parts("Luffy attacked Zoro.", &scan);
        assert_eq!(structure.sentence_frames.len(), 1);
        assert_eq!(structure.relations.len(), 1);
        assert_eq!(structure.relations[0].relation_type, "attacks");
    }

    #[test]
    fn ingest_persists_archives_and_sidecars() {
        let store = PhoenixLmdbStore::open(temp_path("archives")).expect("store");
        store.init_bundle_schema().expect("schema");
        let engine = PhoenixInvarantV2::default();
        let (ingest, _artifacts) = engine
            .ingest_documents(
                &store,
                Some(&SessionId("session-1".to_owned())),
                &[IngestDocument {
                    document_id: DocumentId("doc-1".to_owned()),
                    note_id: Some(NoteId("note-1".to_owned())),
                    title: "Harbor".to_owned(),
                    text: "Ryan met Len at the harbor.".to_owned(),
                    scope: ScopeKey::default(),
                }],
                0,
                10,
            )
            .expect("ingest");
        assert_eq!(ingest.document_count, 1);
        let archives = engine
            .load_latest_document_archives(&store, Some(&ScopeKey::default()))
            .expect("archives");
        assert_eq!(archives.len(), 1);
        let spans = engine
            .load_latest_lex_spans(&store, Some(&ScopeKey::default()))
            .expect("spans");
        assert_eq!(spans.len(), 1);
        let sidecar = engine
            .load_scope_lex_sidecar(&store, &ScopeKey::default())
            .expect("sidecar")
            .expect("present");
        assert_eq!(sidecar.spans.len(), 1);
        assert_eq!(archives[0].resolved_mentions.len(), 2);
    }

    #[test]
    fn collective_er_resolves_pronouns_to_seeded_entities() {
        let engine = PhoenixInvarantV2::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-pronoun".to_owned()),
            note_id: None,
            title: "Pronoun".to_owned(),
            text: "Luffy waited. He smiled.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = engine.scan_parts(
            &document.text,
            &document.scope,
            &[ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            }],
        );
        let structure = engine.build_structure_parts(&document.text, &scan);
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );
        let (resolutions, resolved_mentions, aliases, diagnostics) = resolve_mentions(
            &document,
            &scan,
            &structure,
            &chunks,
            &NativeEntityMemory::default(),
        );
        assert_eq!(resolutions.len(), 2);
        let pronoun = resolved_mentions
            .iter()
            .find(|mention| mention.surface == "He")
            .expect("pronoun mention");
        assert_eq!(pronoun.decision.status, "resolved");
        assert_eq!(
            pronoun.entity_id.as_ref().map(|entity_id| entity_id.0.as_str()),
            Some("luffy")
        );
        assert!(!pronoun
            .candidates
            .iter()
            .any(|candidate| candidate.source == "new_speculative"));
        assert!(aliases.is_empty());
        assert!(diagnostics.iter().any(|diagnostic| {
            matches!(
                diagnostic.code.as_str(),
                "er_pronoun_link_match" | "er_collective_merge" | "er_known_seed_match"
            )
        }));
    }

    #[test]
    fn collective_er_does_not_confirm_aliases_from_prior_resolution_history_alone() {
        let engine = PhoenixInvarantV2::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-history-alias".to_owned()),
            note_id: None,
            title: "History".to_owned(),
            text: "Captain waited.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = engine.scan_parts(&document.text, &document.scope, &[]);
        let structure = engine.build_structure_parts(&document.text, &scan);
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );
        let mention_vertex_id = "mention::doc-history-alias:0".to_owned();
        let snapshot = KernelGraphSnapshot {
            vertices: vec![
                KernelVertex {
                    id: KernelVertexId(mention_vertex_id.clone()),
                    kind: "mention".to_owned(),
                    class: KernelVertexClass::Mention,
                    entity_facet: Some(phoenix_graph_kernel::KernelEntityFacet {
                        canonical_entity_id: Some("luffy".to_owned()),
                        surface: Some("Captain".to_owned()),
                        entity_kind: Some("Character".to_owned()),
                    }),
                    ..KernelVertex::default()
                },
                KernelVertex {
                    id: KernelVertexId("entity::luffy".to_owned()),
                    kind: "entity".to_owned(),
                    class: KernelVertexClass::Entity,
                    entity_id: Some("luffy".to_owned()),
                    ..KernelVertex::default()
                },
            ],
            asserted_edges: vec![KernelEdge {
                source_id: KernelVertexId(mention_vertex_id),
                target_id: KernelVertexId("entity::luffy".to_owned()),
                edge_type: KernelEdgeType("resolved_to".to_owned()),
                ..KernelEdge::default()
            }],
            candidate_edges: Vec::new(),
        };
        let memory = build_native_entity_memory(Some(&snapshot));
        let (_resolutions, resolved_mentions, aliases, diagnostics) = resolve_mentions(
            &document,
            &scan,
            &structure,
            &chunks,
            &memory,
        );
        assert_eq!(resolved_mentions.len(), 1);
        assert_eq!(resolved_mentions[0].decision.status, "resolved");
        assert_eq!(
            resolved_mentions[0]
                .entity_id
                .as_ref()
                .map(|entity_id| entity_id.0.as_str()),
            Some("luffy")
        );
        assert!(aliases.is_empty());
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.code == "er_alias_rejected_low_margin"
                && diagnostic.message.contains("not alias-specific enough")
        }));
    }

    #[test]
    fn collective_er_keeps_ambiguous_kernel_aliases_unresolved() {
        let engine = PhoenixInvarantV2::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-ambiguous".to_owned()),
            note_id: None,
            title: "Ambiguous".to_owned(),
            text: "Ace waited.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = engine.scan_parts(&document.text, &document.scope, &[]);
        let structure = engine.build_structure_parts(&document.text, &scan);
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );
        let snapshot = KernelGraphSnapshot {
            vertices: vec![
                KernelVertex {
                    id: KernelVertexId("alias::luffy::ace".to_owned()),
                    kind: "alias".to_owned(),
                    entity_facet: Some(phoenix_graph_kernel::KernelEntityFacet {
                        canonical_entity_id: Some("luffy".to_owned()),
                        surface: Some("Ace".to_owned()),
                        entity_kind: Some("Character".to_owned()),
                    }),
                    ..KernelVertex::default()
                },
                KernelVertex {
                    id: KernelVertexId("entity::luffy".to_owned()),
                    kind: "entity".to_owned(),
                    entity_id: Some("luffy".to_owned()),
                    ..KernelVertex::default()
                },
                KernelVertex {
                    id: KernelVertexId("alias::sabo::ace".to_owned()),
                    kind: "alias".to_owned(),
                    entity_facet: Some(phoenix_graph_kernel::KernelEntityFacet {
                        canonical_entity_id: Some("sabo".to_owned()),
                        surface: Some("Ace".to_owned()),
                        entity_kind: Some("Character".to_owned()),
                    }),
                    ..KernelVertex::default()
                },
                KernelVertex {
                    id: KernelVertexId("entity::sabo".to_owned()),
                    kind: "entity".to_owned(),
                    entity_id: Some("sabo".to_owned()),
                    ..KernelVertex::default()
                },
            ],
            asserted_edges: vec![
                KernelEdge {
                    source_id: KernelVertexId("alias::luffy::ace".to_owned()),
                    target_id: KernelVertexId("entity::luffy".to_owned()),
                    edge_type: KernelEdgeType("alias_of".to_owned()),
                    ..KernelEdge::default()
                },
                KernelEdge {
                    source_id: KernelVertexId("alias::sabo::ace".to_owned()),
                    target_id: KernelVertexId("entity::sabo".to_owned()),
                    edge_type: KernelEdgeType("alias_of".to_owned()),
                    ..KernelEdge::default()
                },
            ],
            candidate_edges: Vec::new(),
        };
        let memory = build_native_entity_memory(Some(&snapshot));
        let (_resolutions, resolved_mentions, aliases, diagnostics) = resolve_mentions(
            &document,
            &scan,
            &structure,
            &chunks,
            &memory,
        );
        assert_eq!(resolved_mentions.len(), 1);
        assert_eq!(resolved_mentions[0].decision.status, "ambiguous");
        assert!(resolved_mentions[0].entity_id.is_none());
        assert!(aliases.is_empty());
        assert!(diagnostics
            .iter()
            .any(|diagnostic| diagnostic.code == "er_ambiguous_resolution"));
    }

    #[test]
    fn collective_er_is_stable_across_identical_reruns() {
        let engine = PhoenixInvarantV2::default();
        let document = IngestDocument {
            document_id: DocumentId("doc-stable".to_owned()),
            note_id: None,
            title: "Stable".to_owned(),
            text: "Luffy waited. He smiled.".to_owned(),
            scope: ScopeKey::default(),
        };
        let scan = engine.scan_parts(
            &document.text,
            &document.scope,
            &[ResolverEntitySeed {
                entity_id: EntityId("luffy".to_owned()),
                canonical_name: "Luffy".to_owned(),
                aliases: Vec::new(),
                kind: Some(EntityKind::Character),
                gender: Some(GenderHint::Male),
                number: None,
                scope: ScopeKey::default(),
            }],
        );
        let structure = engine.build_structure_parts(&document.text, &scan);
        let (chunks, _) = build_chunk_records(
            &document,
            &extract_boundaries(&document.text),
            &build_chunks(
                &document.text,
                &ChunkerConfig {
                    chunk_size: 512,
                    overlap: 64,
                },
            ),
        );

        let run_once = || {
            let (resolutions, resolved_mentions, aliases, diagnostics) = resolve_mentions(
                &document,
                &scan,
                &structure,
                &chunks,
                &NativeEntityMemory::default(),
            );
            (
                resolutions
                    .iter()
                    .map(|resolution| {
                        (
                            resolution.mention_id.0.clone(),
                            resolution.entity_id.as_ref().map(|entity_id| entity_id.0.clone()),
                            resolution
                                .candidates
                                .iter()
                                .map(|candidate| {
                                    (
                                        candidate.entity_id.clone(),
                                        candidate.source.clone(),
                                        candidate.score_millis,
                                    )
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>(),
                resolved_mentions
                    .iter()
                    .map(|mention| {
                        (
                            mention.mention_id.0.clone(),
                            mention.decision.status.clone(),
                            mention.entity_id.as_ref().map(|entity_id| entity_id.0.clone()),
                        )
                    })
                    .collect::<Vec<_>>(),
                aliases
                    .iter()
                    .map(|alias| {
                        (
                            alias.entity_id.0.clone(),
                            alias.normalized.clone(),
                            alias.confidence_millis,
                        )
                    })
                    .collect::<Vec<_>>(),
                diagnostics
                    .iter()
                    .map(|diagnostic| (diagnostic.code.clone(), diagnostic.message.clone()))
                    .collect::<Vec<_>>(),
            )
        };

        let first = run_once();
        let second = run_once();
        let third = run_once();
        assert_eq!(first, second);
        assert_eq!(second, third);
    }
}

impl Default for InvarantV2Config {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 64,
        }
    }
}

#[derive(Default)]
pub struct PhoenixInvarantV2 {
    config: InvarantV2Config,
}

#[derive(Clone, Debug, Default)]
pub struct V2IngestArtifacts {
    pub kernel_batches: Vec<KernelMutationBatch>,
    pub session_documents: Vec<SessionDocumentState>,
    pub document_refs: Vec<DocumentRevisionRef>,
    pub document_manifests: Vec<DocumentManifest>,
    pub manifest_namespaces: Vec<String>,
    pub span_count: usize,
    pub discovery_candidate_count: usize,
    pub touched_scopes: Vec<ScopeKey>,
}

#[derive(Clone, Debug)]
struct BoundaryRecord {
    label: String,
    range: TextRange,
    chapter_id: u32,
    is_chapter: bool,
}

#[derive(Clone, Debug)]
struct PreparedMention {
    mention_ix: usize,
    mention_id: phoenix_semantic_v2::MentionId,
    normalized: String,
    chunk_id: Option<String>,
    linked_mentions: SmallVec<[usize; 4]>,
}

#[derive(Default)]
struct NativeEntityMemory {
    kernel: PhoenixGraphKernel,
    entity_kinds: FxHashMap<String, String>,
    known_aliases: FxHashSet<(String, String)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ResolvedMentionKind {
    Resolved,
    Ambiguous,
    Unresolved,
}

#[derive(Clone, Debug)]
struct ResolutionDecisionState {
    kind: ResolvedMentionKind,
    entity_id: Option<EntityId>,
    confidence_millis: u32,
    margin_millis: u32,
}

#[derive(Clone, Debug)]
struct MentionResolution {
    mention_ix: usize,
    mention_id: phoenix_semantic_v2::MentionId,
    entity_id: Option<EntityId>,
    candidates: Vec<CandidateEntity>,
    decision: ResolutionDecisionState,
}

#[derive(Clone, Debug)]
struct EntityAccumulator {
    entity_id: EntityId,
    canonical_name: String,
    aliases: SmallVec<[String; 4]>,
    kind: Option<EntityKind>,
    mention_count: usize,
    chunk_ids: SmallVec<[String; 4]>,
}

#[derive(Clone, Debug)]
struct DocumentOutcome {
    assignment: DocumentOrdinalAssignment,
    archive: DocumentArchive,
    kernel_batch: KernelMutationBatch,
    document_summary: IngestDocumentSummary,
    session_document: SessionDocumentState,
    scope: ScopeKey,
    span_count: usize,
    discovery_count: usize,
    diagnostics: Vec<Diagnostic>,
}

impl PhoenixInvarantV2 {
    pub fn new(config: InvarantV2Config) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &InvarantV2Config {
        &self.config
    }

    pub fn scan_parts(
        &self,
        text: &str,
        _scope: &ScopeKey,
        resolver_seed: &[ResolverEntitySeed],
    ) -> ScanArtifact {
        let progress = native_progress_enabled();
        let phase_started = Instant::now();
        let tokens = tokenize(text);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=tokenize wall_ms={} tokens={}",
                phase_started.elapsed().as_millis(),
                tokens.len(),
            );
        }
        let phase_started = Instant::now();
        let sentences = sentence_spans(text);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=sentence_spans wall_ms={} sentences={}",
                phase_started.elapsed().as_millis(),
                sentences.len(),
            );
        }
        let phase_started = Instant::now();
        let mentions = discover_mentions(text, &tokens, &sentences, resolver_seed);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=discover_mentions wall_ms={} mentions={}",
                phase_started.elapsed().as_millis(),
                mentions.len(),
            );
        }
        let phase_started = Instant::now();
        let resolver_links = build_resolver_links(&mentions);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=build_resolver_links wall_ms={} resolver_links={}",
                phase_started.elapsed().as_millis(),
                resolver_links.len(),
            );
        }
        let phase_started = Instant::now();
        let narrative_hits = discover_narrative_hits(text, &tokens, &sentences);
        if progress {
            eprintln!(
                "[runtime-ingest] scan_subphase=discover_narrative_hits wall_ms={} narrative_hits={}",
                phase_started.elapsed().as_millis(),
                narrative_hits.len(),
            );
        }
        let chunks = sentences
            .iter()
            .map(|sentence| ChunkSpan {
                kind: Some(ChunkKind::Clause),
                range: sentence.range,
                head: sentence.range,
                modifiers: Vec::new(),
                sentence_index: sentence.index,
            })
            .collect::<Vec<_>>();
        ScanArtifact {
            diagnostics: vec![Diagnostic {
                code: "PX_INVARANT_V2_SCAN".to_owned(),
                message: format!(
                    "Invarant V2 scanned {} tokens, {} sentences, and {} mentions.",
                    count_token_words(&tokens),
                    sentences.len(),
                    mentions.len()
                ),
            }],
            sentences,
            tokens,
            mentions,
            chunks,
            resolver_links,
            narrative_hits,
        }
    }

    pub fn build_structure_parts(&self, text: &str, scan: &ScanArtifact) -> StructureArtifact {
        let mut sentence_mentions = vec![Vec::<MentionSpan>::new(); scan.sentences.len()];
        for mention in &scan.mentions {
            if let Some(bucket) = sentence_mentions.get_mut(mention.sentence_index) {
                bucket.push(mention.clone());
            }
        }
        let mut sentence_chunks = vec![Vec::<ChunkSpan>::new(); scan.sentences.len()];
        for chunk in &scan.chunks {
            if let Some(bucket) = sentence_chunks.get_mut(chunk.sentence_index) {
                bucket.push(chunk.clone());
            }
        }
        let mut sentence_hits = vec![Vec::<NarrativeVerbHit>::new(); scan.sentences.len()];
        for hit in &scan.narrative_hits {
            if let Some(bucket) = sentence_hits.get_mut(hit.sentence_index) {
                bucket.push(hit.clone());
            }
        }

        let mut sentence_frames = Vec::with_capacity(scan.sentences.len());
        let mut relations = Vec::new();
        let mut evidence_spans = Vec::new();

        for sentence in &scan.sentences {
            let index = sentence.index;
            let mentions = sentence_mentions.get(index).cloned().unwrap_or_default();
            let chunks = sentence_chunks.get(index).cloned().unwrap_or_default();
            let hits = sentence_hits.get(index).cloned().unwrap_or_default();
            let mut diagnostics = Vec::new();
            let mut verb_frames = Vec::new();

            for hit in hits {
                let subject = mentions
                    .iter()
                    .filter(|mention| mention.range.end <= hit.range.start)
                    .max_by_key(|mention| mention.range.end)
                    .map(frame_slot_from_mention);
                let trailing = mentions
                    .iter()
                    .filter(|mention| mention.range.start >= hit.range.end)
                    .collect::<Vec<_>>();
                let object = trailing.first().map(|mention| frame_slot_from_mention(mention));
                let recipient = trailing.get(1).map(|mention| frame_slot_from_mention(mention));
                let evidence = vec![EvidenceSpan {
                    document_id: None,
                    note_id: None,
                    label: slice_or_empty(text, sentence.range).trim().to_owned(),
                    kind: Some("sentence".to_owned()),
                    range: sentence.range,
                }];
                evidence_spans.extend(evidence.iter().cloned());
                let attachments = chunks
                    .iter()
                    .filter(|chunk| chunk.range.start >= hit.range.end)
                    .take(1)
                    .map(|chunk| chunk.range)
                    .collect::<Vec<_>>();
                if subject.is_none() {
                    diagnostics.push(Diagnostic {
                        code: "PX_INVARANT_V2_STRUCTURE_SUBJECT_GAP".to_owned(),
                        message: format!(
                            "Invarant V2 inferred relation '{}' without a clear subject.",
                            hit.relation_type
                        ),
                    });
                }
                relations.push(RelationCandidate {
                    sentence_index: index,
                    verb_range: hit.range,
                    lemma: hit.lemma.clone(),
                    event_class: hit.event_class.clone(),
                    relation_type: hit.relation_type.clone(),
                    subject: subject.clone(),
                    object: object.clone(),
                    recipient: recipient.clone(),
                    attachments: attachments.clone(),
                    evidence: evidence.clone(),
                });
                verb_frames.push(VerbFrame {
                    verb_range: hit.range,
                    lemma: hit.lemma,
                    event_class: hit.event_class,
                    relation_type: hit.relation_type,
                    transitivity: hit.transitivity,
                    subject_candidates: subject.into_iter().collect(),
                    object_candidates: object.into_iter().collect(),
                    recipient_candidates: recipient.into_iter().collect(),
                    pp_attachments: attachments,
                    clause_range: sentence.range,
                    evidence,
                });
            }

            sentence_frames.push(SentenceFrame {
                sentence: sentence.clone(),
                mentions,
                chunks,
                verb_frames,
                clause_ranges: vec![sentence.range],
                diagnostics,
            });
        }

        StructureArtifact {
            sentence_frames,
            relations,
            evidence_spans,
            diagnostics: vec![Diagnostic {
                code: "PX_INVARANT_V2_STRUCTURE".to_owned(),
                message: "Invarant V2 built sentence frames and relation candidates.".to_owned(),
            }],
        }
    }

    pub fn ingest_documents(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        session_id: Option<&SessionId>,
        documents: &[IngestDocument],
        revision: u64,
        created_at: i64,
    ) -> Result<(IngestResult, V2IngestArtifacts), StoreError> {
        let entity_memory = NativeEntityMemory::default();
        let assignments = documents
            .iter()
            .map(|document| DocumentOrdinalAssignment {
                document_id: document.document_id.0.clone(),
                scope: document.scope.clone(),
                scope_key: scope_storage_key(&document.scope),
                scope_ord: ScopeOrd(0),
                document_ord: DocumentOrd(0),
                revision: revision + 1,
            })
            .collect::<Vec<_>>();
        let drafts = documents
            .par_iter()
            .enumerate()
            .map(|(index, document)| {
                self.build_document_outcome(
                    document,
                    session_id,
                    &assignments[index],
                    &entity_memory,
                    created_at,
                )
            })
            .collect::<Vec<_>>();
        let mut outcomes = Vec::with_capacity(drafts.len());
        for draft in drafts {
            outcomes.push(draft?);
        }
        outcomes.sort_by(|left, right| {
            left.document_summary
                .document_id
                .0
                .cmp(&right.document_summary.document_id.0)
        });

        let mut touched_scope_keys = FxHashSet::default();
        for outcome in &outcomes {
            self.persist_document_archive(store, &outcome.archive, revision + 1, created_at)?;
            touched_scope_keys.insert(scope_storage_key(&outcome.scope));
        }

        let mut touched_scopes = outcomes.iter().map(|outcome| outcome.scope.clone()).collect::<Vec<_>>();
        touched_scopes.sort_by_key(scope_storage_key);
        touched_scopes.dedup_by(|left, right| scope_storage_key(left) == scope_storage_key(right));
        for scope in &touched_scopes {
            let sidecar = self.build_scope_lex_sidecar(store, scope, revision + 1, created_at)?;
            self.persist_scope_lex_sidecar(store, &sidecar, revision + 1, created_at)?;
        }

        let total_entities = outcomes
            .iter()
            .map(|outcome| outcome.archive.entities.len())
            .sum();
        let total_mentions = outcomes
            .iter()
            .map(|outcome| outcome.archive.manifest.mention_count)
            .sum();
        let total_discovery = outcomes.iter().map(|outcome| outcome.discovery_count).sum();
        let total_span_count = outcomes.iter().map(|outcome| outcome.span_count).sum();
        let total_leaves = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.leaf_count)
            .sum();
        let total_chapters = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.chapter_count)
            .sum();
        let total_boundaries = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.boundary_count)
            .sum();
        let total_parents = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.parent_count)
            .sum();
        let graph_edge_count: usize = outcomes
            .iter()
            .map(|outcome| outcome.kernel_batch.edges.len())
            .sum();
        let graph_vertex_count: usize = outcomes
            .iter()
            .map(|outcome| outcome.kernel_batch.vertices.len())
            .sum();
        let mut ingest_diagnostics = vec![
            Diagnostic {
                code: "PX_INGEST_INVARANT_V2".to_owned(),
                message: format!(
                    "Invarant V2 ingested {} archives, {} chunks, {} entities, {} graph vertices, and {} graph edges.",
                    outcomes.len(),
                    total_leaves,
                    total_entities,
                    graph_vertex_count,
                    graph_edge_count
                ),
            },
            Diagnostic {
                code: "PX_INGEST_INVARANT_V2_ARCHIVE".to_owned(),
                message: format!(
                    "Native V2 persisted {} document archives, {} scope sidecars, and emitted {} kernel vertices.",
                    outcomes.len(),
                    touched_scope_keys.len(),
                    graph_vertex_count
                ),
            },
        ];
        ingest_diagnostics.extend(
            outcomes
                .iter()
                .flat_map(|outcome| outcome.diagnostics.iter().cloned()),
        );

        Ok((
            IngestResult {
                session_id: session_id.cloned(),
                document_count: outcomes.len(),
                warning_count: 2,
                documents: outcomes
                    .iter()
                    .map(|outcome| outcome.document_summary.clone())
                    .collect(),
                chunk_stats: Some(phoenix_types::ChunkStats {
                    documents: outcomes.len(),
                    total_chapters,
                    total_boundaries,
                    total_parents,
                    total_leaves,
                }),
                graph_summary: Some(phoenix_types::GraphSummary {
                    documents: outcomes.len(),
                    total_chapters,
                    total_boundaries,
                    total_leaves,
                    total_entities,
                    total_mentions,
                    total_edges: graph_edge_count,
                    cross_chapter_links: 0,
                }),
                entity_summary: Some(phoenix_types::EntitySummary {
                    total_entities,
                    total_aliases: outcomes
                        .iter()
                        .flat_map(|outcome| outcome.archive.entities.iter())
                        .map(|entity| entity.aliases.len())
                        .sum(),
                    total_mentions,
                    multi_chapter_entities: 0,
                }),
                discovery_summary: Some(phoenix_types::DiscoverySummary {
                    candidate_count: total_discovery,
                    mention_count: total_mentions,
                    persisted_count: total_entities,
                }),
                retrieval_summary: Some(phoenix_types::RetrievalSummary {
                    qgram_documents: outcomes.len(),
                    gldr_chunks: total_leaves,
                    gldr_entities: total_entities,
                    gldr_edges: graph_edge_count,
                    raptor_documents: outcomes.len(),
                    raptor_leaves: total_leaves,
                    raptor_enabled: true,
                }),
                relation_counts: Vec::new(),
                diagnostics: ingest_diagnostics,
            },
            V2IngestArtifacts {
                kernel_batches: outcomes
                    .iter()
                    .map(|outcome| outcome.kernel_batch.clone())
                    .collect(),
                session_documents: outcomes
                    .iter()
                    .map(|outcome| outcome.session_document.clone())
                    .collect(),
                document_refs: outcomes
                    .iter()
                    .map(|outcome| DocumentRevisionRef {
                        document_id: outcome.archive.manifest.document_id.clone(),
                        scope: outcome.archive.manifest.scope.clone(),
                        scope_ord: outcome.archive.manifest.scope_ord,
                        document_ord: outcome.archive.manifest.document_ord,
                        revision: outcome.archive.manifest.revision,
                    })
                    .collect(),
                document_manifests: outcomes
                    .iter()
                    .map(|outcome| outcome.archive.manifest.clone())
                    .collect(),
                manifest_namespaces: vec![
                    "invarant-v2.document".to_owned(),
                    "invarant-v2.session".to_owned(),
                    "invarant-v2.scope_lex".to_owned(),
                ],
                span_count: total_span_count,
                discovery_candidate_count: total_discovery,
                touched_scopes,
            },
        ))
    }

    pub fn ingest_documents_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        session_id: Option<&SessionId>,
        documents: &[IngestDocument],
        revision: u64,
        created_at: i64,
    ) -> Result<(IngestResult, V2IngestArtifacts), StoreError> {
        let progress = native_progress_enabled();
        let started = Instant::now();
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=prepare_ingest_context start documents={}",
                documents.len()
            );
        }
        let context = store.prepare_ingest_context(session_id, documents, revision)?;
        let entity_memory = build_native_entity_memory(context.kernel_snapshot.as_ref());
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=prepare_ingest_context finish assignments={} wall_ms={}",
                context.assignments.len(),
                started.elapsed().as_millis()
            );
        }
        let started = Instant::now();
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=build_document_outcome start documents={}",
                documents.len()
            );
        }
        let drafts = documents
            .par_iter()
            .enumerate()
            .map(|(index, document)| {
                self.build_document_outcome(
                    document,
                    session_id,
                    &context.assignments[index],
                    &entity_memory,
                    created_at,
                )
            })
            .collect::<Vec<_>>();
        let mut outcomes = Vec::with_capacity(drafts.len());
        for draft in drafts {
            outcomes.push(draft?);
        }
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=build_document_outcome finish outcomes={} wall_ms={}",
                outcomes.len(),
                started.elapsed().as_millis()
            );
        }
        outcomes.sort_by(|left, right| {
            left.document_summary
                .document_id
                .0
                .cmp(&right.document_summary.document_id.0)
        });

        let started = Instant::now();
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=prepare_document start documents={}",
                outcomes.len()
            );
        }
        let prepared = outcomes
            .iter()
            .map(|outcome| self.prepare_document(&outcome.archive, &outcome.assignment))
            .collect::<Result<Vec<_>, _>>()?;
        if progress {
            let segment_count = prepared
                .iter()
                .map(|document| document.segments.len())
                .sum::<usize>();
            eprintln!(
                "[runtime-ingest] subphase=prepare_document finish prepared={} segments={} wall_ms={}",
                prepared.len(),
                segment_count,
                started.elapsed().as_millis()
            );
        }
        let dirty_scopes = self.build_dirty_scope_records(&outcomes, created_at);
        let started = Instant::now();
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=persist_prepared_documents start documents={} dirty_scopes={}",
                prepared.len(),
                dirty_scopes.len()
            );
        }
        store.persist_prepared_documents(&prepared, None, &dirty_scopes, created_at)?;
        if progress {
            eprintln!(
                "[runtime-ingest] subphase=persist_prepared_documents finish documents={} dirty_scopes={} wall_ms={}",
                prepared.len(),
                dirty_scopes.len(),
                started.elapsed().as_millis()
            );
        }

        let total_entities = outcomes
            .iter()
            .map(|outcome| outcome.archive.entities.len())
            .sum();
        let total_mentions = outcomes
            .iter()
            .map(|outcome| outcome.archive.manifest.mention_count)
            .sum();
        let total_discovery = outcomes.iter().map(|outcome| outcome.discovery_count).sum();
        let total_span_count = outcomes.iter().map(|outcome| outcome.span_count).sum();
        let total_leaves = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.leaf_count)
            .sum();
        let total_chapters = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.chapter_count)
            .sum();
        let total_boundaries = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.boundary_count)
            .sum();
        let total_parents = outcomes
            .iter()
            .map(|outcome| outcome.document_summary.parent_count)
            .sum();
        let graph_edge_count: usize = outcomes
            .iter()
            .map(|outcome| outcome.kernel_batch.edges.len())
            .sum();
        let graph_vertex_count: usize = outcomes
            .iter()
            .map(|outcome| outcome.kernel_batch.vertices.len())
            .sum();
        let touched_scopes = dirty_scopes
            .iter()
            .map(|record| record.scope.clone())
            .collect::<Vec<_>>();
        let mut ingest_diagnostics = vec![
            Diagnostic {
                code: "PX_INGEST_INVARANT_V2".to_owned(),
                message: format!(
                    "Invarant V2 ingested {} archives, {} chunks, {} entities, {} graph vertices, and {} graph edges.",
                    outcomes.len(),
                    total_leaves,
                    total_entities,
                    graph_vertex_count,
                    graph_edge_count
                ),
            },
            Diagnostic {
                code: "PX_INGEST_INVARANT_V2_SEGMENTED".to_owned(),
                message: format!(
                    "Native V2 persisted {} manifests, {} prepared segment sets, and marked {} scopes dirty without synchronous sidecar rebuilds.",
                    outcomes.len(),
                    prepared.len(),
                    dirty_scopes.len()
                ),
            },
        ];
        ingest_diagnostics.extend(
            outcomes
                .iter()
                .flat_map(|outcome| outcome.diagnostics.iter().cloned()),
        );

        Ok((
            IngestResult {
                session_id: session_id.cloned(),
                document_count: outcomes.len(),
                warning_count: 2,
                documents: outcomes
                    .iter()
                    .map(|outcome| outcome.document_summary.clone())
                    .collect(),
                chunk_stats: Some(phoenix_types::ChunkStats {
                    documents: outcomes.len(),
                    total_chapters,
                    total_boundaries,
                    total_parents,
                    total_leaves,
                }),
                graph_summary: Some(phoenix_types::GraphSummary {
                    documents: outcomes.len(),
                    total_chapters,
                    total_boundaries,
                    total_leaves,
                    total_entities,
                    total_mentions,
                    total_edges: graph_edge_count,
                    cross_chapter_links: 0,
                }),
                entity_summary: Some(phoenix_types::EntitySummary {
                    total_entities,
                    total_aliases: outcomes
                        .iter()
                        .flat_map(|outcome| outcome.archive.entities.iter())
                        .map(|entity| entity.aliases.len())
                        .sum(),
                    total_mentions,
                    multi_chapter_entities: 0,
                }),
                discovery_summary: Some(phoenix_types::DiscoverySummary {
                    candidate_count: total_discovery,
                    mention_count: total_mentions,
                    persisted_count: total_entities,
                }),
                retrieval_summary: Some(phoenix_types::RetrievalSummary {
                    qgram_documents: outcomes.len(),
                    gldr_chunks: total_leaves,
                    gldr_entities: total_entities,
                    gldr_edges: graph_edge_count,
                    raptor_documents: outcomes.len(),
                    raptor_leaves: total_leaves,
                    raptor_enabled: true,
                }),
                relation_counts: Vec::new(),
                diagnostics: ingest_diagnostics,
            },
            V2IngestArtifacts {
                kernel_batches: outcomes
                    .iter()
                    .map(|outcome| outcome.kernel_batch.clone())
                    .collect(),
                session_documents: outcomes
                    .iter()
                    .map(|outcome| outcome.session_document.clone())
                    .collect(),
                document_refs: outcomes
                    .iter()
                    .map(|outcome| DocumentRevisionRef {
                        document_id: outcome.archive.manifest.document_id.clone(),
                        scope: outcome.archive.manifest.scope.clone(),
                        scope_ord: outcome.archive.manifest.scope_ord,
                        document_ord: outcome.archive.manifest.document_ord,
                        revision: outcome.archive.manifest.revision,
                    })
                    .collect(),
                document_manifests: outcomes
                    .iter()
                    .map(|outcome| outcome.archive.manifest.clone())
                    .collect(),
                manifest_namespaces: vec![
                    "invarant-v2.document".to_owned(),
                    "invarant-v2.session".to_owned(),
                    "invarant-v2.scope_lex".to_owned(),
                ],
                span_count: total_span_count,
                discovery_candidate_count: total_discovery,
                touched_scopes,
            },
        ))
    }

    fn persist_value<T: Serialize>(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        mut header: BundleHeader,
        value: &T,
    ) -> Result<(), StoreError> {
        let payload = encode_archive(value)?;
        header.byte_len = payload.len();
        store.put_bundle(&header, &payload)
    }

    fn load_latest_value<T: DeserializeOwned>(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        kind: BundleKind,
        scope: Option<&str>,
        entity_key: &str,
    ) -> Result<Option<T>, StoreError> {
        let header = store
            .list_bundle_headers(kind, scope)?
            .into_iter()
            .filter(|header| header.key.entity_key == entity_key)
            .max_by_key(|header| header.key.revision);
        let Some(header) = header else {
            return Ok(None);
        };
        let Some(payload) = store.get_bundle(&header.key)? else {
            return Ok(None);
        };
        Ok(Some(decode_archive(&payload)?))
    }

    fn build_scope_lex_sidecar(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        scope: &ScopeKey,
        revision: u64,
        created_at: i64,
    ) -> Result<ScopeLexSidecar, StoreError> {
        let archives = self.load_latest_document_archives(store, Some(scope))?;
        let mut spans = Vec::new();
        let mut document_ids = Vec::with_capacity(archives.len());
        let mut entries = FxHashMap::<String, AliasEntry>::default();
        let mut entity_ids = FxHashSet::<String>::default();

        for archive in archives {
            document_ids.push(archive.manifest.document_id.clone());
            spans.extend(archive.indexed_spans);
            for entity in archive.entities {
                entity_ids.insert(entity.entity_id.0.clone());
                let forms = std::iter::once(entity.canonical_name)
                    .chain(entity.aliases.into_iter())
                    .collect::<SmallVec<[String; 4]>>();
                for form in forms {
                    let normalized = normalize_surface(&form);
                    if normalized.is_empty() {
                        continue;
                    }
                    let entry = entries
                        .entry(normalized.clone())
                        .or_insert_with(|| AliasEntry {
                            normalized,
                            postings: Vec::new(),
                        });
                    if let Some(existing) = entry
                        .postings
                        .iter_mut()
                        .find(|posting| posting.entity_id == entity.entity_id.0)
                    {
                        existing.mention_count += entity.mention_count;
                    } else {
                        entry.postings.push(AliasPosting {
                            entity_id: entity.entity_id.0.clone(),
                            document_id: archive.manifest.document_id.clone(),
                            mention_count: entity.mention_count,
                        });
                    }
                }
            }
        }

        spans.sort_by(|left, right| left.span_id.cmp(&right.span_id));
        document_ids.sort();
        document_ids.dedup();

        let mut alias_entries = entries.into_values().collect::<Vec<_>>();
        alias_entries.sort_by(|left, right| left.normalized.cmp(&right.normalized));
        for entry in &mut alias_entries {
            entry.postings.sort_by(|left, right| {
                left.entity_id
                    .cmp(&right.entity_id)
                    .then_with(|| left.document_id.cmp(&right.document_id))
            });
        }

        Ok(ScopeLexSidecar {
            scope: scope.clone(),
            scope_key: scope_storage_key(scope),
            scope_ord: None,
            spans,
            alias_entries,
            document_ids,
            entity_count: entity_ids.len(),
            generated_at: created_at,
            generation: revision,
        })
    }

    pub fn persist_document_archive(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        archive: &DocumentArchive,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        self.persist_value(
            store,
            document_archive_header(
                &archive.manifest.scope,
                &archive.manifest.document_id,
                revision,
                0,
                created_at,
            ),
            archive,
        )
    }

    pub fn persist_session_summary(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        summary: &SessionArchive,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        self.persist_value(
            store,
            session_archive_header(&summary.session_id, revision, 0, created_at),
            summary,
        )
    }

    pub fn persist_scope_lex_sidecar(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        sidecar: &ScopeLexSidecar,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        self.persist_value(
            store,
            scope_lex_sidecar_header(&sidecar.scope, revision, 0, created_at),
            sidecar,
        )
    }

    pub fn persist_session_summary_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        summary: &SessionArchive,
        revision: u64,
        created_at: i64,
    ) -> Result<(), StoreError> {
        store.persist_session_archive(summary, revision, created_at)
    }

    pub fn load_latest_session_summary(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        session_id: &SessionId,
    ) -> Result<Option<SessionArchive>, StoreError> {
        self.load_latest_value(
            store,
            BundleKind::SessionArchive,
            Some(&session_id.0),
            &session_id.0,
        )
    }

    pub fn load_latest_session_summary_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        session_id: &SessionId,
    ) -> Result<Option<SessionArchive>, StoreError> {
        store.load_latest_session_archive(session_id)
    }

    pub fn load_latest_lex_spans(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<IndexedSpan>, StoreError> {
        if let Some(scope) = scope {
            let scope_key = scope_storage_key(scope);
            let sidecar = self.load_latest_value::<ScopeLexSidecar>(
                store,
                BundleKind::ScopeLexSidecar,
                Some(&scope_key),
                &scope_key,
            )?;
            return Ok(sidecar.map(|value| value.spans).unwrap_or_default());
        }

        let headers = store.list_bundle_headers(BundleKind::ScopeLexSidecar, None)?;
        let mut latest = FxHashMap::<String, BundleHeader>::default();
        for header in headers {
            match latest.get(&header.key.entity_key) {
                Some(existing) if existing.key.revision >= header.key.revision => {}
                _ => {
                    latest.insert(header.key.entity_key.clone(), header);
                }
            }
        }
        let mut spans = Vec::new();
        for header in latest.into_values() {
            let Some(payload) = store.get_bundle(&header.key)? else {
                continue;
            };
            let sidecar: ScopeLexSidecar = decode_archive(&payload)?;
            spans.extend(sidecar.spans);
        }
        spans.sort_by(|left, right| left.span_id.cmp(&right.span_id));
        Ok(spans)
    }

    pub fn load_latest_lex_spans_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<IndexedSpan>, StoreError> {
        store.load_lex_spans(scope)
    }

    pub fn load_scope_lex_sidecar(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        scope: &ScopeKey,
    ) -> Result<Option<ScopeLexSidecar>, StoreError> {
        let scope_key = scope_storage_key(scope);
        self.load_latest_value(
            store,
            BundleKind::ScopeLexSidecar,
            Some(&scope_key),
            &scope_key,
        )
    }

    pub fn load_latest_document_archives(
        &self,
        store: &dyn PhoenixBundleStoreV2,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<DocumentArchive>, StoreError> {
        let scope_key = scope.map(scope_storage_key);
        let headers = store.list_bundle_headers(BundleKind::DocumentArchive, scope_key.as_deref())?;
        let mut latest = FxHashMap::<String, BundleHeader>::default();
        for header in headers {
            match latest.get(&header.key.entity_key) {
                Some(existing) if existing.key.revision >= header.key.revision => {}
                _ => {
                    latest.insert(header.key.entity_key.clone(), header);
                }
            }
        }
        let mut headers = latest.into_values().collect::<Vec<_>>();
        headers.sort_by(|left, right| left.key.entity_key.cmp(&right.key.entity_key));
        let mut values = Vec::with_capacity(headers.len());
        for header in headers {
            let Some(payload) = store.get_bundle(&header.key)? else {
                continue;
            };
            values.push(decode_archive(&payload)?);
        }
        Ok(values)
    }

    pub fn load_latest_document_archives_native(
        &self,
        store: &dyn PhoenixArchiveStoreV2,
        scope: Option<&ScopeKey>,
    ) -> Result<Vec<DocumentArchive>, StoreError> {
        store.load_latest_document_archives(scope)
    }

    pub fn merge_session_summary(
        &self,
        existing: Option<SessionArchive>,
        session_id: SessionId,
        documents: Vec<SessionDocumentState>,
        document_refs: Vec<DocumentRevisionRef>,
        span_count: usize,
        discovery_candidate_count: usize,
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

    fn prepare_document(
        &self,
        archive: &DocumentArchive,
        assignment: &DocumentOrdinalAssignment,
    ) -> Result<PreparedDocument, StoreError> {
        let progress = native_progress_enabled();
        let mut manifest = archive.manifest.clone();
        let mut segments = Vec::<PreparedDocumentSegment>::new();
        let mut segment_refs = Vec::<DocumentSegmentRef>::new();

        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::StringArena,
            archive.tokens.len(),
            &archive.tokens,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::SentenceTable,
            archive.sentences.len(),
            &archive.sentences,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::MentionTable,
            archive.mentions.len(),
            &archive.mentions,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::ResolverLinkTable,
            archive.resolver_links.len(),
            &archive.resolver_links,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::ResolvedMentionTable,
            archive.resolved_mentions.len(),
            &archive.resolved_mentions,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::AliasConfirmationTable,
            archive.alias_confirmations.len(),
            &archive.alias_confirmations,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::ChunkTable,
            archive.chunks.len(),
            &archive.chunks,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::EntityTable,
            archive.entities.len(),
            &archive.entities,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::RelationTable,
            archive.relations.len(),
            &archive.relations,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::EvidenceTable,
            archive.evidence_spans.len(),
            &archive.evidence_spans,
        )?;
        let lexical_started = Instant::now();
        let lexical = LexicalPostingsSegment {
            spans: archive.indexed_spans.clone(),
            alias_entries: document_alias_entries(archive),
        };
        if progress {
            eprintln!(
                "[runtime-ingest] prepare_subphase=build_lexical_postings alias_entries={} spans={} wall_ms={}",
                lexical.alias_entries.len(),
                lexical.spans.len(),
                lexical_started.elapsed().as_millis(),
            );
        }
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::LexicalPostings,
            lexical.spans.len() + lexical.alias_entries.len(),
            &lexical,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::NarrativeHitTable,
            archive.relation_candidates.len(),
            &archive.relation_candidates,
        )?;
        self.push_segment(
            &mut segments,
            &mut segment_refs,
            DocumentSegmentKind::GraphMutation,
            archive.graph_batch.vertices.len() + archive.graph_batch.edges.len(),
            &archive.graph_batch,
        )?;
        if let Some(structure) = archive.structure.as_ref() {
            self.push_segment(
                &mut segments,
                &mut segment_refs,
                DocumentSegmentKind::StructureRelations,
                structure.relations.len(),
                structure,
            )?;
        }
        manifest.segment_refs = segment_refs;
        manifest.scope_ord = assignment.scope_ord;
        manifest.document_ord = assignment.document_ord;
        manifest.revision = assignment.revision;

        Ok(PreparedDocument {
            assignment: assignment.clone(),
            manifest,
            segments,
            kernel_batch: archive.graph_batch.clone(),
        })
    }

    fn push_segment<T: Serialize>(
        &self,
        segments: &mut Vec<PreparedDocumentSegment>,
        refs: &mut Vec<DocumentSegmentRef>,
        kind: DocumentSegmentKind,
        row_count: usize,
        value: &T,
    ) -> Result<(), StoreError> {
        let started = Instant::now();
        let (payload, uncompressed_len) = encode_segment_payload(value)?;
        let ordinal = segments.len() as u32;
        let header = DocumentSegmentHeader::new(
            kind,
            ordinal,
            row_count as u32,
            uncompressed_len,
            payload.len(),
        );
        refs.push(DocumentSegmentRef {
            kind,
            ordinal,
            row_count: row_count as u32,
            byte_len: payload.len() as u32,
            uncompressed_len: uncompressed_len as u32,
        });
        segments.push(PreparedDocumentSegment { header, payload });
        if native_progress_enabled() {
            eprintln!(
                "[runtime-ingest] prepare_segment kind={kind:?} ordinal={} rows={} wall_ms={} uncompressed_bytes={} compressed_bytes={}",
                ordinal,
                row_count,
                started.elapsed().as_millis(),
                uncompressed_len,
                refs.last().map(|segment_ref| segment_ref.byte_len).unwrap_or_default(),
            );
        }
        Ok(())
    }

    fn build_dirty_scope_records(
        &self,
        outcomes: &[DocumentOutcome],
        created_at: i64,
    ) -> Vec<DirtyScopeRecord> {
        let mut by_scope = FxHashMap::<String, DirtyScopeRecord>::default();
        for outcome in outcomes {
            let entry = by_scope
                .entry(outcome.assignment.scope_key.clone())
                .or_insert_with(|| DirtyScopeRecord {
                    scope: outcome.assignment.scope.clone(),
                    scope_key: outcome.assignment.scope_key.clone(),
                    scope_ord: outcome.assignment.scope_ord,
                    document_ords: Vec::new(),
                    updated_at: created_at,
                });
            entry.document_ords.push(outcome.assignment.document_ord);
            entry.updated_at = created_at;
        }
        let mut values = by_scope.into_values().collect::<Vec<_>>();
        for value in &mut values {
            value.document_ords.sort_by_key(|document_ord| document_ord.0);
            value.document_ords.dedup_by(|left, right| left.0 == right.0);
        }
        values.sort_by(|left, right| left.scope_key.cmp(&right.scope_key));
        values
    }

    fn build_document_outcome(
        &self,
        document: &IngestDocument,
        session_id: Option<&SessionId>,
        assignment: &DocumentOrdinalAssignment,
        entity_memory: &NativeEntityMemory,
        created_at: i64,
    ) -> Result<DocumentOutcome, StoreError> {
        let progress = native_progress_enabled();
        let document_started = Instant::now();
        let document_version_id = DocumentVersionId(format!(
            "{}::{}",
            document.document_id.0, assignment.revision
        ));

        let phase_started = Instant::now();
        let scan = self.scan_parts(&document.text, &document.scope, &[]);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=scan_parts document_id={} wall_ms={} tokens={} sentences={} mentions={} narrative_hits={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                scan.tokens.len(),
                scan.sentences.len(),
                scan.mentions.len(),
                scan.narrative_hits.len(),
            );
        }

        let phase_started = Instant::now();
        let structure = self.build_structure_parts(&document.text, &scan);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_structure_parts document_id={} wall_ms={} sentence_frames={} relations={} evidence_spans={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                structure.sentence_frames.len(),
                structure.relations.len(),
                structure.evidence_spans.len(),
            );
        }

        let phase_started = Instant::now();
        let boundaries = extract_boundaries(&document.text);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=extract_boundaries document_id={} wall_ms={} boundaries={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                boundaries.len(),
            );
        }

        let phase_started = Instant::now();
        let chunk_ranges = build_chunks(
            &document.text,
            &ChunkerConfig {
                chunk_size: self.config.chunk_size,
                overlap: self.config.overlap,
            },
        );
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_chunks document_id={} wall_ms={} chunk_ranges={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                chunk_ranges.len(),
            );
        }

        let phase_started = Instant::now();
        let (chunks, indexed_spans) = build_chunk_records(document, &boundaries, &chunk_ranges);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_chunk_records document_id={} wall_ms={} chunks={} indexed_spans={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                chunks.len(),
                indexed_spans.len(),
            );
        }

        let phase_started = Instant::now();
        let (resolutions, resolved_mentions, alias_confirmations, mut er_diagnostics) =
            resolve_mentions(document, &scan, &structure, &chunks, entity_memory);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=resolve_mentions document_id={} wall_ms={} resolutions={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                resolutions.len(),
            );
        }

        let phase_started = Instant::now();
        let (entities, relations, discovery_count, mut relation_diagnostics) =
            build_semantic_records(document, &scan, &structure, &chunks, &resolutions);
        er_diagnostics.append(&mut relation_diagnostics);
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_semantic_records document_id={} wall_ms={} entities={} relations={} discovery_count={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                entities.len(),
                relations.len(),
                discovery_count,
            );
        }

        let phase_started = Instant::now();
        let kernel_batch = build_kernel_batch(
            document,
            &scan,
            &chunks,
            &entities,
            &relations,
            &resolutions,
            &alias_confirmations,
            &structure.evidence_spans,
        );
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=build_kernel_batch document_id={} wall_ms={} vertices={} edges={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                kernel_batch.vertices.len(),
                kernel_batch.edges.len(),
            );
        }

        let phase_started = Instant::now();
        let document_summary = IngestDocumentSummary {
            document_id: document.document_id.clone(),
            note_id: document.note_id.clone(),
            chapter_count: boundaries.iter().filter(|boundary| boundary.is_chapter).count(),
            boundary_count: boundaries.len(),
            parent_count: boundaries.len(),
            leaf_count: chunks.len(),
            entity_count: entities.len(),
            edge_count: kernel_batch.edges.len(),
            has_front_matter_chapter: boundaries.iter().any(|boundary| {
                boundary.is_chapter && is_front_matter_label(&boundary.label)
            }),
            has_front_matter_boundary: boundaries
                .iter()
                .any(|boundary| is_front_matter_label(&boundary.label)),
        };
        let session_document = SessionDocumentState {
            document_id: document.document_id.clone(),
            note_id: document.note_id.clone(),
            chapter_count: document_summary.chapter_count,
            boundary_count: document_summary.boundary_count,
            chapter_titles: boundaries
                .iter()
                .filter(|boundary| boundary.is_chapter)
                .map(|boundary| boundary.label.clone())
                .collect(),
            boundary_labels: boundaries.iter().map(|boundary| boundary.label.clone()).collect(),
            parent_count: document_summary.parent_count,
            leaf_count: document_summary.leaf_count,
            entity_count: document_summary.entity_count,
            discovery_count,
            has_front_matter_chapter: document_summary.has_front_matter_chapter,
            has_front_matter_boundary: document_summary.has_front_matter_boundary,
            updated_at: created_at,
        };
        let mention_count = scan.mentions.len();
        let scope_key = assignment.scope_key.clone();
        let manifest = DocumentManifest {
            document_id: document.document_id.0.clone(),
            document_version_id,
            note_id: document.note_id.clone(),
            scope: document.scope.clone(),
            scope_key,
            scope_ord: assignment.scope_ord,
            document_ord: assignment.document_ord,
            revision: assignment.revision,
            title: document.title.clone(),
            text_len: document.text.len(),
            fingerprint: format!("{}:{}", document.document_id.0, document.text.len()),
            config_hash: "invarant-v2::document-archive".to_owned(),
            session_id: session_id.cloned(),
            document_summary: document_summary.clone(),
            session_document: session_document.clone(),
            discovery_count,
            mention_count,
            span_count: document_summary.leaf_count,
            entity_count: entities.len(),
            alias_count: entities.iter().map(|entity| entity.aliases.len()).sum(),
            graph_edge_count: kernel_batch.edges.len(),
            graph_vertex_count: kernel_batch.vertices.len(),
            segment_refs: Vec::new(),
            created_at,
            archive_version: 2,
        };
        let archive = DocumentArchive {
            manifest,
            tokens: scan.tokens.clone(),
            sentences: scan.sentences.clone(),
            mentions: scan.mentions.clone(),
            resolver_links: scan.resolver_links.clone(),
            resolved_mentions,
            alias_confirmations,
            coref_clusters: Vec::new(),
            er_summary: Default::default(),
            coref_summary: Default::default(),
            chunks,
            indexed_spans,
            entities,
            relations,
            evidence_spans: structure.evidence_spans.clone(),
            relation_candidates: structure.relations.clone(),
            graph_batch: kernel_batch.clone(),
            structure: Some(structure.clone()),
            causal_substrate: None,
        };
        if progress {
            eprintln!(
                "[runtime-ingest] doc_phase=assemble_archive document_id={} wall_ms={} mention_count={} alias_count={} total_wall_ms={}",
                document.document_id.0,
                phase_started.elapsed().as_millis(),
                mention_count,
                archive.manifest.alias_count,
                document_started.elapsed().as_millis(),
            );
        }
        Ok(DocumentOutcome {
            assignment: assignment.clone(),
            archive,
            kernel_batch,
            document_summary: document_summary.clone(),
            session_document,
            scope: document.scope.clone(),
            span_count: document_summary.leaf_count,
            discovery_count,
            diagnostics: er_diagnostics,
        })
    }
}

fn native_progress_enabled() -> bool {
    std::env::var_os("PHOENIX_PERF_PROGRESS").is_some()
        || std::env::var_os("PHOENIX_INGEST_PROGRESS").is_some()
}
