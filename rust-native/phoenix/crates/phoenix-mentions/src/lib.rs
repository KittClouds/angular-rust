use compact_str::CompactString;
use phoenix_machine::SurfaceCompileArtifacts;
use phoenix_types::{
    MentionClass, MentionContext, MentionEntityRef, MentionFeatures, MentionSource,
    PreparedMentionRecord, SourceRange, SurfaceUnitKind,
};

pub struct MentionCompiler;

impl MentionCompiler {
    pub fn prepare(artifacts: &SurfaceCompileArtifacts) -> Vec<PreparedMentionRecord> {
        artifacts
            .scan
            .mentions
            .iter()
            .map(|mention| PreparedMentionRecord {
                mention_id: None,
                range: SourceRange::from(mention.range),
                surface: CompactString::from(mention.surface.as_str()),
                class: match mention.source.clone().unwrap_or(MentionSource::Discovery) {
                    MentionSource::Known | MentionSource::Alias => MentionClass::Named,
                    MentionSource::Fuzzy => MentionClass::Nominal,
                    MentionSource::Discovery => {
                        if matches!(mention.entity_ref, Some(MentionEntityRef::Speculative(_))) {
                            MentionClass::Discovery
                        } else {
                            MentionClass::Named
                        }
                    }
                },
                features: MentionFeatures {
                    normalized: CompactString::from(mention.surface.to_lowercase()),
                    sentence_index: mention.sentence_index,
                    chunk_index: artifacts
                        .surface
                        .units
                        .iter()
                        .find(|unit| {
                            unit.kind == SurfaceUnitKind::Sentence
                                && unit.sentence_index == mention.sentence_index
                        })
                        .map(|_| mention.sentence_index as u32),
                    kind: mention.kind.clone(),
                    entity_ref: mention.entity_ref.clone(),
                    confidence_millis: (mention.confidence.clamp(0.0, 1.0) * 1000.0) as u32,
                },
                context: MentionContext {
                    unit_key: artifacts
                        .surface
                        .units
                        .iter()
                        .find(|unit| unit.sentence_index == mention.sentence_index)
                        .and_then(|unit| unit.key),
                    sentence_range: artifacts
                        .surface
                        .sentences
                        .get(mention.sentence_index)
                        .map(|sentence| sentence.range),
                    clause_range: artifacts
                        .surface
                        .clauses
                        .iter()
                        .find(|clause| clause.sentence_index == mention.sentence_index)
                        .map(|clause| clause.range),
                },
            })
            .collect()
    }
}
