use std::env;

use phoenix_types::{
    ChunkKind, ChunkSpan, EvidenceSpan, FrameSlot, RelationCandidate, ResolverLink,
    ResolverLinkKind, ScanArtifact, SentenceFrame, StructureArtifact, TextRange, VerbFrame,
};

#[derive(Default)]
pub struct NativeStructureBuilder;

impl NativeStructureBuilder {
    pub fn build_parts(&self, text: &str, scan: &ScanArtifact) -> StructureArtifact {
        let sentence_count = scan.sentences.len();
        let mut mentions_by_sentence = vec![Vec::new(); sentence_count];
        let mut chunks_by_sentence = vec![Vec::new(); sentence_count];
        let mut links_by_sentence = vec![Vec::new(); sentence_count];
        let mut hits_by_sentence = vec![Vec::new(); sentence_count];

        for mention in &scan.mentions {
            if let Some(bucket) = mentions_by_sentence.get_mut(mention.sentence_index) {
                bucket.push(mention.clone());
            }
        }
        for chunk in &scan.chunks {
            if let Some(bucket) = chunks_by_sentence.get_mut(chunk.sentence_index) {
                bucket.push(chunk.clone());
            }
        }
        for link in &scan.resolver_links {
            if let Some(bucket) = links_by_sentence.get_mut(link.sentence_index) {
                bucket.push(link);
            }
        }
        for hit in &scan.narrative_hits {
            if let Some(bucket) = hits_by_sentence.get_mut(hit.sentence_index) {
                bucket.push(hit);
            }
        }

        let mut evidence_spans = Vec::with_capacity(
            scan.narrative_hits.len().saturating_add(scan.resolver_links.len()),
        );
        let mut relations = Vec::with_capacity(scan.narrative_hits.len());
        let sentence_frames = scan
            .sentences
            .iter()
            .map(|sentence| {
                let mentions = mentions_by_sentence
                    .get(sentence.index)
                    .cloned()
                    .unwrap_or_default();
                let chunks = chunks_by_sentence
                    .get(sentence.index)
                    .cloned()
                    .unwrap_or_default();
                let sentence_links = links_by_sentence
                    .get(sentence.index)
                    .cloned()
                    .unwrap_or_default();
                let sentence_hits = hits_by_sentence
                    .get(sentence.index)
                    .cloned()
                    .unwrap_or_default();
                let clause_ranges = chunks
                    .iter()
                    .filter(|chunk| chunk.kind == Some(ChunkKind::Clause))
                    .map(|chunk| chunk.range)
                    .collect::<Vec<_>>();
                let verb_frames = sentence_hits
                    .iter()
                    .map(|hit| {
                        build_verb_frame(
                            text,
                            sentence.index,
                            hit.range,
                            &hit.lemma,
                            &hit.event_class,
                            &hit.relation_type,
                            hit.transitivity.clone(),
                            &chunks,
                            &mentions,
                            &sentence_links,
                        )
                    })
                    .collect::<Vec<_>>();
                relations.extend(
                    verb_frames
                        .iter()
                        .map(|frame| relation_from_frame(sentence.index, frame)),
                );

                evidence_spans.extend(sentence_hits.iter().map(|hit| EvidenceSpan {
                    document_id: None,
                    note_id: None,
                    label: hit.lemma.clone(),
                    kind: Some("narrativeVerb".to_owned()),
                    range: hit.range,
                }));
                evidence_spans.extend(sentence_links.iter().map(|link| EvidenceSpan {
                    document_id: None,
                    note_id: None,
                    label: match link.link_kind {
                        Some(ResolverLinkKind::Pronoun) => "pronounLink".to_owned(),
                        Some(ResolverLinkKind::AliasCandidate) => "aliasCandidate".to_owned(),
                        None => "resolverLink".to_owned(),
                    },
                    kind: Some("resolverLink".to_owned()),
                    range: link.source_range,
                }));

                SentenceFrame {
                    sentence: sentence.clone(),
                    mentions,
                    chunks,
                    verb_frames,
                    clause_ranges,
                    diagnostics: Vec::new(),
                }
            })
            .collect::<Vec<_>>();

        StructureArtifact {
            sentence_frames,
            relations,
            evidence_spans,
            diagnostics: Vec::new(),
        }
    }
}

pub(crate) fn legacy_native_structure_enabled() -> bool {
    matches!(
        env::var("PHOENIX_INVARANT_USE_LEGACY_STRUCTURE").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES")
    )
}

fn build_verb_frame(
    text: &str,
    sentence_index: usize,
    verb_range: TextRange,
    lemma: &str,
    event_class: &str,
    relation_type: &str,
    transitivity: Option<phoenix_types::NarrativeTransitivity>,
    chunks: &[ChunkSpan],
    mentions: &[phoenix_types::MentionSpan],
    resolver_links: &[&ResolverLink],
) -> VerbFrame {
    let clause_range = chunks
        .iter()
        .find(|chunk| chunk.kind == Some(ChunkKind::Clause) && contains(chunk.range, verb_range))
        .map(|chunk| chunk.range)
        .unwrap_or(verb_range);
    let subject_candidates = nearest_np_before(verb_range, chunks, mentions, resolver_links);
    let object_candidates = nearest_np_after(verb_range, chunks, mentions, resolver_links);
    let recipient_candidates =
        recipient_candidates(text, verb_range, chunks, mentions, resolver_links);
    let pp_attachments = pp_after(verb_range, chunks);

    let _ = sentence_index;
    VerbFrame {
        verb_range,
        lemma: lemma.to_owned(),
        event_class: event_class.to_owned(),
        relation_type: relation_type.to_owned(),
        transitivity,
        subject_candidates,
        object_candidates,
        recipient_candidates,
        pp_attachments,
        clause_range,
        evidence: vec![EvidenceSpan {
            document_id: None,
            note_id: None,
            label: lemma.to_owned(),
            kind: Some("verbFrame".to_owned()),
            range: verb_range,
        }],
    }
}

fn nearest_np_before(
    verb_range: TextRange,
    chunks: &[ChunkSpan],
    mentions: &[phoenix_types::MentionSpan],
    resolver_links: &[&ResolverLink],
) -> Vec<FrameSlot> {
    chunks
        .iter()
        .rev()
        .find(|chunk| chunk.kind == Some(ChunkKind::Np) && chunk.range.end <= verb_range.start)
        .map(|chunk| vec![slot_from_chunk(chunk, mentions, resolver_links)])
        .unwrap_or_default()
}

fn nearest_np_after(
    verb_range: TextRange,
    chunks: &[ChunkSpan],
    mentions: &[phoenix_types::MentionSpan],
    resolver_links: &[&ResolverLink],
) -> Vec<FrameSlot> {
    chunks
        .iter()
        .find(|chunk| chunk.kind == Some(ChunkKind::Np) && chunk.range.start >= verb_range.end)
        .map(|chunk| vec![slot_from_chunk(chunk, mentions, resolver_links)])
        .unwrap_or_default()
}

fn recipient_candidates(
    text: &str,
    verb_range: TextRange,
    chunks: &[ChunkSpan],
    mentions: &[phoenix_types::MentionSpan],
    resolver_links: &[&ResolverLink],
) -> Vec<FrameSlot> {
    chunks
        .iter()
        .filter(|chunk| chunk.kind == Some(ChunkKind::Pp) && chunk.range.start >= verb_range.end)
        .filter_map(|chunk| {
            let prep = text
                .get(chunk.head.start as usize..chunk.head.end as usize)
                .unwrap_or_default()
                .to_lowercase();
            if prep != "to" && prep != "for" {
                return None;
            }
            chunks
                .iter()
                .find(|candidate| {
                    candidate.kind == Some(ChunkKind::Np)
                        && candidate.range.start >= chunk.range.start
                        && candidate.range.end <= chunk.range.end
                })
                .map(|candidate| slot_from_chunk(candidate, mentions, resolver_links))
        })
        .collect()
}

fn pp_after(verb_range: TextRange, chunks: &[ChunkSpan]) -> Vec<TextRange> {
    chunks
        .iter()
        .filter(|chunk| chunk.kind == Some(ChunkKind::Pp) && chunk.range.start >= verb_range.end)
        .map(|chunk| chunk.range)
        .collect()
}

fn slot_from_chunk(
    chunk: &ChunkSpan,
    mentions: &[phoenix_types::MentionSpan],
    resolver_links: &[&ResolverLink],
) -> FrameSlot {
    let entity_ref = mentions
        .iter()
        .find(|mention| overlaps(chunk.range, mention.range))
        .and_then(|mention| mention.entity_ref.clone());
    let entity_ref = entity_ref.or_else(|| {
        resolver_links
            .iter()
            .find(|link| overlaps(chunk.range, link.source_range))
            .and_then(|link| link.target_entity.clone())
    });
    let confidence = if entity_ref.is_some() { 0.9 } else { 0.6 };
    FrameSlot {
        range: chunk.range,
        entity_ref,
        confidence,
    }
}

fn relation_from_frame(sentence_index: usize, frame: &VerbFrame) -> RelationCandidate {
    let mut object = frame.object_candidates.first().cloned();
    if should_skip_generic_object(&object) {
        if let Some(replacement) = frame.recipient_candidates.first().cloned() {
            object = Some(replacement);
        }
    }
    RelationCandidate {
        sentence_index,
        verb_range: frame.verb_range,
        lemma: frame.lemma.clone(),
        event_class: frame.event_class.clone(),
        relation_type: frame.relation_type.clone(),
        subject: frame.subject_candidates.first().cloned(),
        object,
        recipient: frame.recipient_candidates.first().cloned(),
        attachments: frame.pp_attachments.clone(),
        evidence: frame.evidence.clone(),
    }
}

fn should_skip_generic_object(slot: &Option<FrameSlot>) -> bool {
    let Some(slot) = slot else {
        return false;
    };
    let generic_len = slot.range.end.saturating_sub(slot.range.start);
    generic_len <= 12 && slot.entity_ref.is_none()
}

fn contains(outer: TextRange, inner: TextRange) -> bool {
    outer.start <= inner.start && outer.end >= inner.end
}

fn overlaps(left: TextRange, right: TextRange) -> bool {
    left.start < right.end && right.start < left.end
}
