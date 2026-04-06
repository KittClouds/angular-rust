use std::collections::BTreeMap;
use std::env;
use std::time::Instant;

use phoenix_graph::{
    GraphEdgeRecord, GraphLayer, GraphMutationBatch, GraphMutationScope, GraphVertexRecord,
};
use phoenix_graptor::BorrowedIngestDocument;
use phoenix_store_cozo::StoreError;
use phoenix_types::{
    BoundaryKind, DocumentId, EntityId, EntityKind, IngestDocumentSummary, MentionSource, NoteId,
    RelationCount, ScopeKey, SessionId,
};
use rustc_hash::FxHashMap;
use serde_json::{json, Map, Value};

use crate::{
    completed_stage, semantic::MentionCandidate, AnalysisContext, CanonicalEntity,
    DocumentAnalysisStage, DocumentSemanticBundle, InvarantStore, NativeBoundary, NativeChapter,
    NativeDocumentProjection, NativeLeaf, NativeRelationRows, ProposedEntityLink,
    ResolvedMention, ResolutionStatus, INVARANT_DOCUMENTS_NAMESPACE, INVARANT_MANIFEST_NAMESPACE,
};
use phoenix_types::{ScanArtifact, StructureArtifact};

pub(crate) fn legacy_native_scanner_enabled() -> bool {
    matches!(
        env::var("PHOENIX_INVARANT_USE_LEGACY_SCANNER").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

pub(crate) fn write_relation_rows(
    store: &dyn InvarantStore,
    relation: &str,
    rows: &[Value],
    stages: &mut Vec<DocumentAnalysisStage>,
) -> Result<(), StoreError> {
    if rows.is_empty() {
        return Ok(());
    }
    let started = Instant::now();
    store.put_rows(relation, rows)?;
    stages.push(completed_stage(
        &format!("write_{relation}"),
        started,
        BTreeMap::from([("rowCount".to_owned(), rows.len())]),
    )?);
    Ok(())
}

pub(crate) fn relation_counts_from_rows(rows: &NativeRelationRows) -> Vec<RelationCount> {
    [
        ("notes", rows.notes.len()),
        ("docid_map", rows.docid_map.len()),
        ("chunkid_map", rows.chunkid_map.len()),
        ("chunks", rows.chunks.len()),
        ("document_boundaries", rows.document_boundaries.len()),
        ("entities", rows.entities.len()),
        ("spans", rows.spans.len()),
        ("span_mentions", rows.span_mentions.len()),
        ("discovery_candidates", rows.discovery_candidates.len()),
        ("edges", rows.edges.len()),
        ("scoped_documents", rows.scoped_documents.len()),
        ("scoped_definitions", rows.scoped_definitions.len()),
    ]
    .into_iter()
    .filter(|(_, rows)| *rows > 0)
    .map(|(relation, rows)| RelationCount {
        relation: relation.to_owned(),
        rows,
    })
    .collect()
}

pub(crate) fn project_native_document(
    document: &BorrowedIngestDocument<'_>,
    session_id: Option<&SessionId>,
    context: &AnalysisContext,
    config_hash: &str,
    scan: &ScanArtifact,
    structure: &StructureArtifact,
    bundle: &DocumentSemanticBundle,
    now: i64,
) -> Result<NativeDocumentProjection, StoreError> {
    let note_id = document
        .note_id
        .clone()
        .unwrap_or_else(|| NoteId(document.document_id.0.clone()));
    let (chapters, has_front_matter) = native_chapters(document, bundle);
    let boundaries = chapters
        .iter()
        .enumerate()
        .map(|(index, chapter)| NativeBoundary {
            boundary_id: chapter.boundary_id,
            kind: "chapter",
            depth: 1,
            label: Some(chapter.title.clone()),
            ordinal: (index + 1) as i64,
            parent_boundary_id: None,
            start: chapter.start as i64,
            end: chapter.end as i64,
        })
        .collect::<Vec<_>>();
    let leaves = native_leaves(document, bundle, &chapters);
    let summary = IngestDocumentSummary {
        document_id: document.document_id.clone(),
        note_id: Some(note_id.clone()),
        chapter_count: chapters.len(),
        boundary_count: boundaries.len(),
        parent_count: 0,
        leaf_count: leaves.len(),
        entity_count: bundle.resolution.canonical_entities.len(),
        edge_count: bundle.semantics.relations.len(),
        has_front_matter_chapter: has_front_matter,
        has_front_matter_boundary: has_front_matter,
    };
    let graph_batch = asserted_graph_batch(document, &note_id, &chapters, &leaves, bundle);
    let manifest_payload = native_document_manifest(
        document,
        session_id,
        &summary,
        &boundaries,
        &chapters,
        bundle
            .annotation
            .mention_candidates
            .iter()
            .filter(|mention| matches!(mention.source, Some(MentionSource::Discovery)))
            .count(),
        now,
    );

    let mut rows = NativeRelationRows::default();
    rows.notes.push(note_row(document, &note_id, now));
    rows.docid_map.push(docid_map_row(document, now));
    rows.document_boundaries.extend(
        boundaries
            .iter()
            .map(|boundary| document_boundary_row_native(document, &note_id, boundary, now)),
    );
    rows.chunks
        .extend(chapters.iter().map(|chapter| chapter_chunk_row_native(document, chapter, now)));
    rows.chunkid_map.extend(
        chapters
            .iter()
            .map(|chapter| chapter_chunkid_row_native(document, chapter, now)),
    );
    rows.chunks.extend(
        leaves
            .iter()
            .map(|leaf| leaf_chunk_row_native(document, leaf, &document.scope, now)),
    );
    rows.chunkid_map.extend(
        leaves
            .iter()
            .map(|leaf| leaf_chunkid_row_native(document, leaf, now)),
    );
    rows.entities.extend(
        bundle
            .resolution
            .canonical_entities
            .iter()
            .map(|entity| entity_row_native(entity, document, &note_id, now)),
    );
    let resolution_by_mention = bundle
        .resolution
        .resolved_mentions
        .iter()
        .map(|mention| (mention.mention_id.0.as_str(), mention))
        .collect::<FxHashMap<_, _>>();
    rows.spans.extend(bundle.annotation.mention_candidates.iter().map(|mention| {
        mention_span_row_native(
            mention,
            resolution_by_mention.get(mention.mention_id.0.as_str()).copied(),
            document,
            &note_id,
            now,
        )
    }));
    rows.span_mentions.extend(
        bundle
            .resolution
            .resolved_mentions
            .iter()
            .filter_map(|mention| span_mention_row_native(mention, now)),
    );
    rows.span_mentions.extend(
        bundle
            .resolution
            .proposed_links
            .iter()
            .map(|link| proposed_span_mention_row_native(link, now)),
    );
    rows.discovery_candidates
        .extend(discovery_candidate_rows_native(bundle, now));
    rows.edges
        .extend(relation_edge_rows_native(bundle, &note_id, now));

    rows.scoped_definitions.push(crate::scoped_definition_row(
        document,
        "source",
        &crate::source_payload(document, context, bundle),
        now,
    ));
    rows.scoped_definitions.push(crate::scoped_definition_row(
        document,
        "scan",
        &crate::scan_payload(document, context, scan, config_hash),
        now,
    ));
    rows.scoped_definitions.push(crate::scoped_definition_row(
        document,
        "segmentation",
        &crate::segmentation_payload(document, context, bundle),
        now,
    ));
    rows.scoped_definitions.push(crate::scoped_definition_row(
        document,
        "structure",
        &crate::structure_payload(document, context, structure, bundle),
        now,
    ));
    rows.scoped_definitions.push(crate::scoped_definition_row(
        document,
        "annotation",
        &crate::annotation_payload(document, context, bundle),
        now,
    ));
    rows.scoped_definitions.push(crate::scoped_definition_row(
        document,
        "resolution",
        &crate::resolution_payload(document, context, bundle),
        now,
    ));
    rows.scoped_definitions.push(crate::scoped_definition_row(
        document,
        "semantic",
        &crate::semantic_payload(document, context, bundle),
        now,
    ));
    rows.scoped_documents
        .push(crate::semantic_document_row(document, bundle, now));
    rows.scoped_definitions
        .extend(crate::semantic_object_rows(document, bundle, now));
    rows.scoped_documents
        .push(native_scoped_document_row(document, &manifest_payload, now));
    rows.scoped_definitions
        .push(native_manifest_definition_row(document, &manifest_payload, now));

    Ok(NativeDocumentProjection {
        summary,
        rows,
        graph_batch,
        alias_count: bundle
            .resolution
            .canonical_entities
            .iter()
            .map(|entity| entity.aliases.len())
            .sum(),
        discovery_count: bundle
            .annotation
            .mention_candidates
            .iter()
            .filter(|mention| matches!(mention.source, Some(MentionSource::Discovery)))
            .count(),
        mention_span_count: bundle.annotation.mention_candidates.len(),
    })
}

fn native_chapters(
    document: &BorrowedIngestDocument<'_>,
    bundle: &DocumentSemanticBundle,
) -> (Vec<NativeChapter>, bool) {
    let mut headings = bundle
        .scanned_document
        .spans
        .iter()
        .filter(|span| matches!(span.kind, crate::StructuralKind::Heading))
        .cloned()
        .collect::<Vec<_>>();
    headings.sort_by_key(|span| (span.range.start, span.range.end));
    if headings.is_empty() {
        return (
            vec![NativeChapter {
                chapter_id: 1,
                boundary_id: stable_int("invarant_boundary", &[document.document_id.0.as_str(), "1"]),
                boundary_ordinal: 1,
                title: document.title.to_owned(),
                start: 0,
                end: document.text.len(),
                chunk_id: stable_int(
                    "invarant_chapter_chunk",
                    &[document.document_id.0.as_str(), "1"],
                ),
            }],
            false,
        );
    }

    let mut chapters = Vec::new();
    let has_front_matter = headings
        .first()
        .map(|heading| heading.range.start > 0)
        .unwrap_or(false);
    let mut chapter_id = 1u32;
    if has_front_matter {
        chapters.push(NativeChapter {
            chapter_id,
            boundary_id: stable_int("invarant_boundary", &[document.document_id.0.as_str(), "front"]),
            boundary_ordinal: chapter_id,
            title: document.title.to_owned(),
            start: 0,
            end: headings[0].range.start as usize,
            chunk_id: stable_int(
                "invarant_chapter_chunk",
                &[document.document_id.0.as_str(), "front"],
            ),
        });
        chapter_id += 1;
    }
    for (index, heading) in headings.iter().enumerate() {
        let next_start = headings
            .get(index + 1)
            .map(|span| span.range.start as usize)
            .unwrap_or(document.text.len());
        chapters.push(NativeChapter {
            chapter_id,
            boundary_id: stable_int(
                "invarant_boundary",
                &[document.document_id.0.as_str(), &chapter_id.to_string()],
            ),
            boundary_ordinal: chapter_id,
            title: heading
                .label
                .clone()
                .unwrap_or_else(|| document.title.to_owned()),
            start: heading.range.start as usize,
            end: next_start.max(heading.range.end as usize),
            chunk_id: stable_int(
                "invarant_chapter_chunk",
                &[document.document_id.0.as_str(), &chapter_id.to_string()],
            ),
        });
        chapter_id += 1;
    }
    (chapters, has_front_matter)
}

fn native_leaves(
    document: &BorrowedIngestDocument<'_>,
    bundle: &DocumentSemanticBundle,
    chapters: &[NativeChapter],
) -> Vec<NativeLeaf> {
    bundle
        .leaf_chunks
        .iter()
        .enumerate()
        .map(|(index, chunk)| {
            let chapter = chapters
                .iter()
                .find(|chapter| {
                    chunk.range.start as usize >= chapter.start
                        && (chunk.range.start as usize) < chapter.end
                })
                .unwrap_or_else(|| chapters.last().expect("at least one chapter"));
            let search_id = format!(
                "{}:{}:leaf:{}",
                document.document_id.0,
                chapter.chapter_id,
                index
            );
            let start = chunk.range.start as usize;
            let end = chunk.range.end as usize;
            NativeLeaf {
                chunk_id: stable_int(
                    "invarant_leaf_chunk",
                    &[
                        document.document_id.0.as_str(),
                        &start.to_string(),
                        &end.to_string(),
                        &index.to_string(),
                    ],
                ),
                search_id,
                chapter_id: chapter.chapter_id,
                boundary_id: chapter.boundary_id,
                boundary_ordinal: chapter.boundary_ordinal,
                parent_id: chapter.chunk_id,
                start,
                end,
                text: safe_text_slice(document.text, start, end).to_owned(),
            }
        })
        .collect()
}

fn note_row(document: &BorrowedIngestDocument<'_>, note_id: &NoteId, now: i64) -> Value {
    json!({
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
        "change_reason": "phoenix_invarant_ingest",
    })
}

fn docid_map_row(document: &BorrowedIngestDocument<'_>, now: i64) -> Value {
    json!({
        "id": stable_int("docid", &[document.document_id.0.as_str()]),
        "docid": document.document_id.0,
        "created_at": now,
    })
}

fn document_boundary_row_native(
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    boundary: &NativeBoundary,
    now: i64,
) -> Value {
    json!({
        "doc_id": document.document_id.0,
        "boundary_id": boundary.boundary_id,
        "kind": boundary.kind,
        "depth": boundary.depth,
        "label": boundary.label,
        "ordinal": boundary.ordinal,
        "parent_boundary_id": boundary.parent_boundary_id,
        "note_id": note_id.0,
        "start_char": boundary.start,
        "end_char": boundary.end,
        "created_at": now,
    })
}

fn chapter_chunk_row_native(
    document: &BorrowedIngestDocument<'_>,
    chapter: &NativeChapter,
    now: i64,
) -> Value {
    json!({
        "chunk_id": chapter.chunk_id,
        "doc_id": document.document_id.0,
        "level": 2,
        "start": chapter.start as i64,
        "end": chapter.end as i64,
        "text": chapter.title,
        "parent_id": null,
        "scope_narrative": document.scope.narrative_id,
        "scope_folder": document.scope.folder_id,
        "created_at": now,
    })
}

fn chapter_chunkid_row_native(
    document: &BorrowedIngestDocument<'_>,
    chapter: &NativeChapter,
    now: i64,
) -> Value {
    json!({
        "id": chapter.chunk_id,
        "chunk_key": format!("{}:{}:chapter", document.document_id.0, chapter.chapter_id),
        "doc_id": document.document_id.0,
        "created_at": now,
    })
}

fn leaf_chunk_row_native(
    document: &BorrowedIngestDocument<'_>,
    leaf: &NativeLeaf,
    scope: &ScopeKey,
    now: i64,
) -> Value {
    json!({
        "chunk_id": leaf.chunk_id,
        "doc_id": document.document_id.0,
        "level": 0,
        "start": leaf.start as i64,
        "end": leaf.end as i64,
        "text": leaf.text,
        "parent_id": leaf.parent_id,
        "scope_narrative": scope.narrative_id,
        "scope_folder": scope.folder_id,
        "created_at": now,
    })
}

fn leaf_chunkid_row_native(
    document: &BorrowedIngestDocument<'_>,
    leaf: &NativeLeaf,
    now: i64,
) -> Value {
    json!({
        "id": leaf.chunk_id,
        "chunk_key": leaf.search_id,
        "doc_id": document.document_id.0,
        "created_at": now,
    })
}

fn entity_row_native(
    entity: &CanonicalEntity,
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    now: i64,
) -> Value {
    json!({
        "id": entity.entity_id.0,
        "label": entity.label,
        "kind": kind_to_string(entity.kind.as_ref()),
        "subtype": null,
        "aliases": entity.aliases,
        "first_note": note_id.0,
        "total_mentions": entity.mention_ids.len() as i64,
        "narrative_id": document.scope.narrative_id,
        "created_by": "invarant",
        "created_at": now,
        "updated_at": now,
    })
}

fn mention_span_row_native(
    mention: &MentionCandidate,
    resolved: Option<&ResolvedMention>,
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    now: i64,
) -> Value {
    json!({
        "id": stable_hex("mention_span", &[mention.mention_id.0.as_str()]),
        "world_id": document.scope.world_id,
        "note_id": note_id.0,
        "narrative_id": document.scope.narrative_id,
        "start": mention.range.start as i64,
        "end": mention.range.end as i64,
        "text": mention.surface,
        "content_hash": stable_hex("spanhash", &[mention.surface.as_str()]),
        "span_kind": "entity_mention",
        "status": resolved.map(|value| resolution_status_str(&value.status)).unwrap_or("observed"),
        "created_by": "invarant",
        "created_at": now,
        "updated_at": now,
    })
}

fn span_mention_row_native(mention: &ResolvedMention, now: i64) -> Option<Value> {
    Some(json!({
        "id": stable_hex("span_mention", &[mention.mention_id.0.as_str(), "resolved"]),
        "span_id": stable_hex("mention_span", &[mention.mention_id.0.as_str()]),
        "candidate_entity_id": mention.entity_id.as_ref()?.0,
        "match_type": resolution_status_str(&mention.status),
        "confidence": mention.confidence,
        "ev_frequency": null,
        "ev_capital_ratio": null,
        "ev_context_score": null,
        "ev_cooccurrence": null,
        "status": resolution_status_str(&mention.status),
        "created_at": now,
        "updated_at": now,
    }))
}

fn proposed_span_mention_row_native(link: &ProposedEntityLink, now: i64) -> Value {
    json!({
        "id": stable_hex("span_mention", &[link.mention_id.0.as_str(), "proposed"]),
        "span_id": stable_hex("mention_span", &[link.mention_id.0.as_str()]),
        "candidate_entity_id": link.entity_id.0,
        "match_type": link.reason,
        "confidence": link.confidence,
        "ev_frequency": null,
        "ev_capital_ratio": null,
        "ev_context_score": null,
        "ev_cooccurrence": null,
        "status": "proposed",
        "created_at": now,
        "updated_at": now,
    })
}

fn discovery_candidate_rows_native(bundle: &DocumentSemanticBundle, now: i64) -> Vec<Value> {
    let mut aggregated = BTreeMap::<String, (Option<EntityKind>, f64, i64)>::new();
    for mention in &bundle.annotation.mention_candidates {
        if !matches!(mention.source, Some(MentionSource::Discovery)) {
            continue;
        }
        let entry = aggregated
            .entry(mention.normalized_surface.clone())
            .or_insert((mention.kind.clone(), 0.0, 0));
        entry.1 += mention.confidence as f64;
        entry.2 += 1;
    }
    aggregated
        .into_iter()
        .map(|(token, (kind, score_sum, count))| {
            json!({
                "token": token,
                "kind": kind_code(kind.as_ref()),
                "score": if count > 0 { score_sum / count as f64 } else { 0.0 },
                "status": 0,
                "last_seen": now,
                "first_seen": now,
                "count": count,
            })
        })
        .collect()
}

fn relation_edge_rows_native(
    bundle: &DocumentSemanticBundle,
    note_id: &NoteId,
    now: i64,
) -> Vec<Value> {
    let mut rows = Vec::new();
    for relation in &bundle.semantics.relations {
        if let (Some(source), Some(target)) = (&relation.source_entity_id, &relation.target_entity_id) {
            rows.push(json!({
                "id": stable_hex("edge", &[source.0.as_str(), target.0.as_str(), relation.relation_type.as_str()]),
                "source_id": source.0,
                "target_id": target.0,
                "rel_type": relation.relation_type,
                "confidence": relation.confidence,
                "bidirectional": false,
                "source_note": note_id.0,
                "created_at": now,
            }));
        }
    }
    rows
}

fn native_document_manifest(
    document: &BorrowedIngestDocument<'_>,
    session_id: Option<&SessionId>,
    summary: &IngestDocumentSummary,
    boundaries: &[NativeBoundary],
    chapters: &[NativeChapter],
    discovery_count: usize,
    now: i64,
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
                "kind": boundary.kind,
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
                "parentCount": 0,
                "parentIds": Vec::<i64>::new(),
            })
        }).collect::<Vec<_>>(),
        "updatedAt": now,
    })
}

fn native_scoped_document_row(
    document: &BorrowedIngestDocument<'_>,
    payload: &Value,
    now: i64,
) -> Value {
    json!({
        "id": stable_hex("scoped_document", &[document.document_id.0.as_str(), "native"]),
        "scope_folder_id": document.scope.folder_id.clone().unwrap_or_else(|| "__root__".to_owned()),
        "narrative_id": document.scope.narrative_id.clone().unwrap_or_else(|| "__global__".to_owned()),
        "namespace": INVARANT_DOCUMENTS_NAMESPACE,
        "document_key": document.document_id.0,
        "payload": payload,
        "seeded_from_scope_folder_id": document.scope.folder_id,
        "created_at": now,
        "updated_at": now,
    })
}

fn native_manifest_definition_row(
    document: &BorrowedIngestDocument<'_>,
    payload: &Value,
    now: i64,
) -> Value {
    json!({
        "id": stable_hex("scoped_definition", &[document.document_id.0.as_str(), "manifest-native"]),
        "narrative_id": document.scope.narrative_id.clone().unwrap_or_else(|| "__global__".to_owned()),
        "namespace": INVARANT_MANIFEST_NAMESPACE,
        "definition_key": format!("document:{}", document.document_id.0),
        "payload": payload,
        "created_at": now,
        "updated_at": now,
    })
}

fn asserted_graph_batch(
    document: &BorrowedIngestDocument<'_>,
    note_id: &NoteId,
    chapters: &[NativeChapter],
    leaves: &[NativeLeaf],
    bundle: &DocumentSemanticBundle,
) -> GraphMutationBatch {
    let mut vertices = Vec::new();
    let mut edges = Vec::new();
    vertices.push(GraphVertexRecord {
        id: document_vertex_id(&document.document_id),
        kind: "document".to_owned(),
        weight: 1,
        value: json!({
            "kind": "document",
            "documentId": document.document_id.0,
            "title": document.title,
        }),
        attributes: graph_attributes(
            graph_metadata(1.0, vec![format!("document:{}", document.document_id.0)]),
            json!({
                "documentId": document.document_id.0,
                "noteId": note_id.0,
                "scope": document.scope,
            }),
        ),
        entity_id: None,
        search_chunk_id: None,
        document_id: Some(document.document_id.0.clone()),
        chapter_id: None,
        chapters: chapters.iter().map(|chapter| chapter.chapter_id).collect(),
        boundary_id: None,
        boundary_ordinal: None,
        boundary_kind: None,
        boundary_ordinals: chapters.iter().map(|chapter| chapter.boundary_ordinal).collect(),
    });
    for chapter in chapters {
        vertices.push(GraphVertexRecord {
            id: chapter_vertex_id(&document.document_id, chapter.chapter_id),
            kind: "chapter".to_owned(),
            weight: 1,
            value: json!({
                "kind": "chapter",
                "chapterId": chapter.chapter_id,
                "boundaryId": chapter.boundary_id,
                "boundaryOrdinal": chapter.boundary_ordinal,
                "title": chapter.title,
            }),
            attributes: graph_attributes(
                graph_metadata(
                    1.0,
                    vec![
                        format!("document:{}", document.document_id.0),
                        format!("boundary:{}:{}", document.document_id.0, chapter.boundary_id),
                    ],
                ),
                json!({
                    "documentId": document.document_id.0,
                    "boundaryId": chapter.boundary_id,
                    "boundaryOrdinal": chapter.boundary_ordinal,
                    "boundaryKind": "chapter",
                    "start": chapter.start,
                    "end": chapter.end,
                }),
            ),
            entity_id: None,
            search_chunk_id: None,
            document_id: Some(document.document_id.0.clone()),
            chapter_id: Some(chapter.chapter_id),
            chapters: vec![chapter.chapter_id],
            boundary_id: Some(chapter.boundary_id as u32),
            boundary_ordinal: Some(chapter.boundary_ordinal),
            boundary_kind: Some(BoundaryKind::Chapter),
            boundary_ordinals: vec![chapter.boundary_ordinal],
        });
        edges.push(GraphEdgeRecord {
            source_id: document_vertex_id(&document.document_id),
            target_id: chapter_vertex_id(&document.document_id, chapter.chapter_id),
            edge_type: "contains".to_owned(),
            weight: 1,
            attributes: graph_attributes(
                graph_metadata(
                    1.0,
                    vec![
                        format!("document:{}", document.document_id.0),
                        format!("boundary:{}:{}", document.document_id.0, chapter.boundary_id),
                    ],
                ),
                json!({
                    "documentId": document.document_id.0,
                    "boundaryId": chapter.boundary_id,
                    "boundaryOrdinal": chapter.boundary_ordinal,
                    "boundaryKind": "chapter",
                }),
            ),
            data: None,
            document_id: Some(document.document_id.0.clone()),
            narrative_id: document.scope.narrative_id.clone(),
            layer: GraphLayer::Asserted,
        });
    }
    for leaf in leaves {
        vertices.push(GraphVertexRecord {
            id: leaf_vertex_id(leaf.search_id.as_str()),
            kind: "leaf".to_owned(),
            weight: 1,
            value: json!({
                "kind": "leaf",
                "searchChunkId": leaf.search_id,
                "chunkId": leaf.chunk_id,
                "boundaryId": leaf.boundary_id,
                "boundaryOrdinal": leaf.boundary_ordinal,
            }),
            attributes: graph_attributes(
                graph_metadata(
                    0.94,
                    vec![
                        format!("document:{}", document.document_id.0),
                        format!("leaf:{}", leaf.search_id),
                    ],
                ),
                json!({
                    "documentId": document.document_id.0,
                    "noteId": note_id.0,
                    "chapterId": leaf.chapter_id,
                    "boundaryId": leaf.boundary_id,
                    "boundaryOrdinal": leaf.boundary_ordinal,
                    "boundaryKind": "chapter",
                    "searchChunkId": leaf.search_id,
                    "start": leaf.start,
                    "end": leaf.end,
                }),
            ),
            entity_id: None,
            search_chunk_id: Some(leaf.search_id.clone()),
            document_id: Some(document.document_id.0.clone()),
            chapter_id: Some(leaf.chapter_id),
            chapters: vec![leaf.chapter_id],
            boundary_id: Some(leaf.boundary_id as u32),
            boundary_ordinal: Some(leaf.boundary_ordinal),
            boundary_kind: Some(BoundaryKind::Chapter),
            boundary_ordinals: vec![leaf.boundary_ordinal],
        });
        edges.push(GraphEdgeRecord {
            source_id: chapter_vertex_id(&document.document_id, leaf.chapter_id),
            target_id: leaf_vertex_id(leaf.search_id.as_str()),
            edge_type: "contains_leaf".to_owned(),
            weight: 1,
            attributes: graph_attributes(
                graph_metadata(
                    0.94,
                    vec![
                        format!("document:{}", document.document_id.0),
                        format!("leaf:{}", leaf.search_id),
                    ],
                ),
                json!({
                    "documentId": document.document_id.0,
                    "chapterId": leaf.chapter_id,
                    "boundaryId": leaf.boundary_id,
                    "boundaryOrdinal": leaf.boundary_ordinal,
                    "boundaryKind": "chapter",
                    "searchChunkId": leaf.search_id,
                }),
            ),
            data: None,
            document_id: Some(document.document_id.0.clone()),
            narrative_id: document.scope.narrative_id.clone(),
            layer: GraphLayer::Asserted,
        });
    }
    for entity in &bundle.resolution.canonical_entities {
        vertices.push(GraphVertexRecord {
            id: entity_vertex_id(&entity.entity_id),
            kind: "entity".to_owned(),
            weight: entity.mention_ids.len().max(1) as i64,
            value: json!({
                "kind": "entity",
                "entityId": entity.entity_id.0,
                "label": entity.label,
                "entityKind": kind_to_string(entity.kind.as_ref()),
            }),
            attributes: graph_attributes(
                graph_metadata(
                    entity.confidence as f64,
                    std::iter::once(format!("entity:{}", entity.entity_id.0))
                        .chain(entity.evidence_ids.iter().map(|id| format!("evidence:{}", id.0)))
                        .collect(),
                ),
                json!({
                    "documentId": document.document_id.0,
                    "aliases": entity.aliases,
                    "scope": entity.scope,
                }),
            ),
            entity_id: Some(entity.entity_id.0.clone()),
            search_chunk_id: None,
            document_id: Some(document.document_id.0.clone()),
            chapter_id: None,
            chapters: Vec::new(),
            boundary_id: None,
            boundary_ordinal: None,
            boundary_kind: None,
            boundary_ordinals: Vec::new(),
        });
    }
    let leaf_search_by_chunk = leaves
        .iter()
        .map(|leaf| (leaf.chunk_id, leaf.search_id.clone()))
        .collect::<FxHashMap<_, _>>();
    let mention_by_id = bundle
        .annotation
        .mention_candidates
        .iter()
        .map(|mention| (mention.mention_id.0.as_str(), mention))
        .collect::<FxHashMap<_, _>>();
    let mut leaf_entity_edges = FxHashMap::<(String, String), f32>::default();
    for resolved in &bundle.resolution.resolved_mentions {
        let Some(entity_id) = resolved.entity_id.as_ref() else {
            continue;
        };
        let Some(mention) = mention_by_id.get(resolved.mention_id.0.as_str()).copied() else {
            continue;
        };
        let search_chunk = mention
            .chunk_id
            .as_ref()
            .and_then(|chunk_id| {
                leaves
                    .iter()
                    .find(|leaf| {
                        leaf.start as u32 <= mention.range.start && leaf.end as u32 >= mention.range.end
                    })
                    .map(|leaf| leaf.search_id.clone())
                    .or_else(|| {
                        leaf_search_by_chunk.get(&stable_int(
                            "invarant_leaf_chunk",
                            &[
                                document.document_id.0.as_str(),
                                &mention.range.start.to_string(),
                                &mention.range.end.to_string(),
                                chunk_id.0.as_str(),
                            ],
                        ))
                        .cloned()
                    })
            });
        let Some(search_chunk) = search_chunk else {
            continue;
        };
        let key = (search_chunk, entity_id.0.clone());
        let weight = leaf_entity_edges.entry(key).or_insert(0.0);
        *weight = (*weight).max(resolved.confidence);
    }
    for ((chunk_key, entity_id), confidence) in leaf_entity_edges {
        edges.push(GraphEdgeRecord {
            source_id: leaf_vertex_id(chunk_key.as_str()),
            target_id: entity_vertex_id(&EntityId(entity_id.clone())),
            edge_type: "mentions".to_owned(),
            weight: 1,
            attributes: graph_attributes(
                graph_metadata(
                    confidence as f64,
                    vec![
                        format!("leaf:{chunk_key}"),
                        format!("entity:{entity_id}"),
                        format!("document:{}", document.document_id.0),
                    ],
                ),
                json!({
                    "documentId": document.document_id.0,
                    "searchChunkId": chunk_key,
                    "entityId": entity_id,
                    "confidence": confidence,
                }),
            ),
            data: None,
            document_id: Some(document.document_id.0.clone()),
            narrative_id: document.scope.narrative_id.clone(),
            layer: GraphLayer::Asserted,
        });
    }
    for relation in &bundle.semantics.relations {
        if let (Some(source), Some(target)) = (&relation.source_entity_id, &relation.target_entity_id) {
            edges.push(GraphEdgeRecord {
                source_id: entity_vertex_id(source),
                target_id: entity_vertex_id(target),
                edge_type: relation.relation_type.clone(),
                weight: 1,
                attributes: graph_attributes(
                    graph_metadata(
                        relation.confidence as f64,
                        relation
                            .evidence_ids
                            .iter()
                            .map(|id| format!("evidence:{}", id.0))
                            .collect(),
                    ),
                    json!({
                        "documentId": document.document_id.0,
                        "confidence": relation.confidence,
                    }),
                ),
                data: None,
                document_id: Some(document.document_id.0.clone()),
                narrative_id: document.scope.narrative_id.clone(),
                layer: GraphLayer::Asserted,
            });
        }
    }
    GraphMutationBatch {
        layer: GraphLayer::Asserted,
        scope: GraphMutationScope::Document {
            document_id: document.document_id.0.clone(),
        },
        vertices,
        edges,
    }
}

fn graph_attributes(graph: Value, base: Value) -> Value {
    let mut object = base.as_object().cloned().unwrap_or_else(Map::new);
    object.insert("graph".to_owned(), graph);
    Value::Object(object)
}

fn graph_metadata(confidence: f64, evidence_refs: Vec<String>) -> Value {
    json!({
        "layer": "asserted",
        "status": "asserted",
        "resolver": "phoenix-invarant",
        "confidence": confidence,
        "evidence_refs": evidence_refs,
    })
}

fn document_vertex_id(document_id: &DocumentId) -> String {
    format!("doc::{}", document_id.0)
}

fn chapter_vertex_id(document_id: &DocumentId, chapter_id: u32) -> String {
    format!("chapter::{}::{}", document_id.0, chapter_id)
}

fn leaf_vertex_id(search_id: &str) -> String {
    format!("leaf::{search_id}")
}

fn entity_vertex_id(entity_id: &EntityId) -> String {
    format!("entity::{}", entity_id.0)
}

fn kind_to_string(kind: Option<&EntityKind>) -> &'static str {
    match kind {
        Some(EntityKind::Character) => "character",
        Some(EntityKind::Npc) => "npc",
        Some(EntityKind::Faction) => "faction",
        Some(EntityKind::Organization) => "organization",
        Some(EntityKind::Location) => "location",
        Some(EntityKind::Item) => "item",
        Some(EntityKind::Concept) => "concept",
        Some(EntityKind::Event) => "event",
        Some(EntityKind::Other) | None => "other",
    }
}

fn kind_code(kind: Option<&EntityKind>) -> Option<i64> {
    match kind {
        Some(EntityKind::Character) => Some(1),
        Some(EntityKind::Npc) => Some(2),
        Some(EntityKind::Faction) => Some(3),
        Some(EntityKind::Organization) => Some(4),
        Some(EntityKind::Location) => Some(5),
        Some(EntityKind::Item) => Some(6),
        Some(EntityKind::Concept) => Some(7),
        Some(EntityKind::Event) => Some(8),
        Some(EntityKind::Other) => Some(9),
        None => None,
    }
}

fn resolution_status_str(status: &ResolutionStatus) -> &'static str {
    match status {
        ResolutionStatus::Unresolved => "unresolved",
        ResolutionStatus::Proposed => "proposed",
        ResolutionStatus::Resolved => "resolved",
    }
}

fn safe_text_slice(text: &str, start: usize, end: usize) -> &str {
    let start = start.min(text.len());
    let end = end.min(text.len());
    if start >= end {
        ""
    } else {
        text.get(start..end).unwrap_or("")
    }
}

fn stable_int(prefix: &str, parts: &[&str]) -> i64 {
    (stable_hash(prefix, parts) & 0x7fff_ffff) as i64
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

fn stable_hex(prefix: &str, parts: &[&str]) -> String {
    format!("{:016x}", stable_hash(prefix, parts))
}
