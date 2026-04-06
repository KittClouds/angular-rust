use phoenix_types::{
    ChunkKind, ChunkSpan, EvidenceSpan, FrameSlot, RelationCandidate, ResolverLink,
    ResolverLinkKind, SentenceFrame, StructureArtifact, StructureRequest, TextRange, VerbFrame,
};

pub struct PhoenixStructure;

impl PhoenixStructure {
    pub fn new() -> Self {
        Self
    }

    pub fn build(&self, request: &StructureRequest) -> StructureArtifact {
        self.build_parts(&request.text, &request.scan)
    }

    pub fn build_parts(&self, text: &str, scan: &phoenix_types::ScanArtifact) -> StructureArtifact {
        let mut evidence_spans = Vec::new();
        let mut relations = Vec::new();
        let sentence_frames = scan
            .sentences
            .iter()
            .map(|sentence| {
                let mentions = scan
                    .mentions
                    .iter()
                    .filter(|mention| contains(sentence.range, mention.range))
                    .cloned()
                    .collect::<Vec<_>>();
                let chunks = scan
                    .chunks
                    .iter()
                    .filter(|chunk| contains(sentence.range, chunk.range))
                    .cloned()
                    .collect::<Vec<_>>();
                let clause_ranges = chunks
                    .iter()
                    .filter(|chunk| chunk.kind == Some(ChunkKind::Clause))
                    .map(|chunk| chunk.range)
                    .collect::<Vec<_>>();
                let sentence_links = scan
                    .resolver_links
                    .iter()
                    .filter(|link| link.sentence_index == sentence.index)
                    .collect::<Vec<_>>();
                let verb_frames = scan
                    .narrative_hits
                    .iter()
                    .filter(|hit| hit.sentence_index == sentence.index)
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

                evidence_spans.extend(
                    scan.narrative_hits
                        .iter()
                        .filter(|hit| hit.sentence_index == sentence.index)
                        .map(|hit| EvidenceSpan {
                            document_id: None,
                            note_id: None,
                            label: hit.lemma.clone(),
                            kind: Some("narrativeVerb".to_owned()),
                            range: hit.range,
                        }),
                );
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

impl Default for PhoenixStructure {
    fn default() -> Self {
        Self::new()
    }
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

#[cfg(test)]
mod tests {
    use phoenix_types::{
        ChunkKind, ChunkSpan, EntityId, MentionEntityRef, MentionSource, MentionSpan,
        NarrativeVerbHit, ResolverLink, ResolverLinkKind, ScanArtifact, SentenceSpan, TextRange,
    };

    use super::*;

    #[test]
    fn sentence_frames_capture_subject_object_and_pp() {
        let structure = PhoenixStructure::new();
        let scan = ScanArtifact {
            sentences: vec![SentenceSpan {
                index: 0,
                range: TextRange { start: 0, end: 35 },
            }],
            tokens: Vec::new(),
            mentions: vec![
                MentionSpan {
                    range: TextRange { start: 0, end: 5 },
                    surface: "Luffy".to_owned(),
                    kind: None,
                    entity_ref: Some(MentionEntityRef::Known(EntityId("luffy".to_owned()))),
                    source: Some(MentionSource::Known),
                    confidence: 1.0,
                    sentence_index: 0,
                },
                MentionSpan {
                    range: TextRange { start: 18, end: 22 },
                    surface: "Zoro".to_owned(),
                    kind: None,
                    entity_ref: Some(MentionEntityRef::Known(EntityId("zoro".to_owned()))),
                    source: Some(MentionSource::Known),
                    confidence: 1.0,
                    sentence_index: 0,
                },
            ],
            chunks: vec![
                ChunkSpan {
                    kind: Some(ChunkKind::Np),
                    range: TextRange { start: 0, end: 5 },
                    head: TextRange { start: 0, end: 5 },
                    modifiers: Vec::new(),
                    sentence_index: 0,
                },
                ChunkSpan {
                    kind: Some(ChunkKind::Vp),
                    range: TextRange { start: 6, end: 17 },
                    head: TextRange { start: 6, end: 14 },
                    modifiers: Vec::new(),
                    sentence_index: 0,
                },
                ChunkSpan {
                    kind: Some(ChunkKind::Np),
                    range: TextRange { start: 18, end: 22 },
                    head: TextRange { start: 18, end: 22 },
                    modifiers: Vec::new(),
                    sentence_index: 0,
                },
                ChunkSpan {
                    kind: Some(ChunkKind::Pp),
                    range: TextRange { start: 23, end: 35 },
                    head: TextRange { start: 23, end: 27 },
                    modifiers: Vec::new(),
                    sentence_index: 0,
                },
            ],
            resolver_links: vec![ResolverLink {
                source_range: TextRange { start: 30, end: 34 },
                target_range: Some(TextRange { start: 18, end: 22 }),
                target_entity: Some(MentionEntityRef::Known(EntityId("zoro".to_owned()))),
                link_kind: Some(ResolverLinkKind::Pronoun),
                confidence: 0.9,
                sentence_index: 0,
            }],
            narrative_hits: vec![NarrativeVerbHit {
                range: TextRange { start: 6, end: 14 },
                lemma: "attacked".to_owned(),
                event_class: "battle".to_owned(),
                relation_type: "attacks".to_owned(),
                transitivity: None,
                sentence_index: 0,
                confidence: 0.9,
            }],
            diagnostics: Vec::new(),
        };

        let artifact = structure.build(&StructureRequest {
            text: "Luffy attacked Zoro with fury.".to_owned(),
            scan,
        });

        assert_eq!(artifact.sentence_frames.len(), 1);
        let frame = &artifact.sentence_frames[0];
        assert_eq!(frame.verb_frames.len(), 1);
        assert_eq!(frame.verb_frames[0].subject_candidates.len(), 1);
        assert_eq!(frame.verb_frames[0].object_candidates.len(), 1);
        assert_eq!(frame.verb_frames[0].pp_attachments.len(), 1);
        assert_eq!(artifact.relations.len(), 1);
        assert!(artifact.relations[0].subject.is_some());
        assert!(artifact.relations[0].object.is_some());
        assert!(!artifact.evidence_spans.is_empty());
    }

    #[test]
    fn no_cst_baseline_works_with_clause_ranges_only_when_present() {
        let structure = PhoenixStructure::new();
        let artifact = structure.build(&StructureRequest {
            text: "Luffy smiled.".to_owned(),
            scan: ScanArtifact {
                sentences: vec![SentenceSpan {
                    index: 0,
                    range: TextRange { start: 0, end: 14 },
                }],
                tokens: Vec::new(),
                mentions: Vec::new(),
                chunks: vec![ChunkSpan {
                    kind: Some(ChunkKind::Vp),
                    range: TextRange { start: 6, end: 12 },
                    head: TextRange { start: 6, end: 12 },
                    modifiers: Vec::new(),
                    sentence_index: 0,
                }],
                resolver_links: Vec::new(),
                narrative_hits: vec![NarrativeVerbHit {
                    range: TextRange { start: 6, end: 12 },
                    lemma: "smiled".to_owned(),
                    event_class: "dialogue".to_owned(),
                    relation_type: "speaksTo".to_owned(),
                    transitivity: None,
                    sentence_index: 0,
                    confidence: 0.7,
                }],
                diagnostics: Vec::new(),
            },
        });

        assert!(artifact.sentence_frames[0].clause_ranges.is_empty());
        assert_eq!(artifact.relations.len(), 1);
    }
}
