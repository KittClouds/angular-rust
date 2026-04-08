use std::collections::BTreeMap;
use std::path::Path;

use gliner::model::input::text::TextInput;
use gliner::model::params::Parameters;
use gliner::model::pipeline::span::SpanMode;
use gliner::model::GLiNER;
use gliner::text::span::Span;
use orp::params::RuntimeParameters;

#[derive(Clone, Debug, PartialEq)]
pub struct RelationSeededSpan {
    pub input_id: String,
    pub surface: String,
    pub label: String,
    pub probability: f32,
    pub span_start: usize,
    pub span_end: usize,
}

pub struct RelationMentionSeeder {
    model: GLiNER<SpanMode>,
    threshold: f32,
}

const RELATION_SEED_BATCH_SIZE: usize = 4;

impl RelationMentionSeeder {
    pub fn load(model_root: &Path, threshold: f32) -> Result<Self, String> {
        let tokenizer_path = model_root.join("tokenizer.json");
        let model_path = model_root.join("onnx").join("model.onnx");
        let model = GLiNER::<SpanMode>::new(
            Parameters::default().with_threshold(threshold),
            RuntimeParameters::default(),
            tokenizer_path
                .to_str()
                .ok_or_else(|| "gliner tokenizer path contains invalid utf-8".to_owned())?,
            model_path
                .to_str()
                .ok_or_else(|| "gliner model path contains invalid utf-8".to_owned())?,
        )
        .map_err(|error| {
            format!(
                "failed to load gliner x-small from {}: {error}",
                model_root.display()
            )
        })?;
        Ok(Self { model, threshold })
    }

    pub fn seed_chunk_mentions(
        &self,
        inputs: &[(String, String)],
    ) -> Result<Vec<RelationSeededSpan>, String> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let labels = relation_seed_labels();
        let mut seeded = Vec::new();
        for batch in inputs.chunks(RELATION_SEED_BATCH_SIZE) {
            let texts = batch
                .iter()
                .map(|(_, text)| text.as_str())
                .collect::<Vec<_>>();
            let output = self
                .model
                .inference(TextInput::from_str(&texts, &labels).map_err(|error| error.to_string())?)
                .map_err(|error| error.to_string())?;
            for ((input_id, _), sequence) in batch.iter().zip(output.spans.into_iter()) {
                for span in filter_seed_sequence(sequence, self.threshold) {
                    seeded.push(RelationSeededSpan {
                        input_id: input_id.clone(),
                        surface: span.text().trim().to_owned(),
                        label: span.class().to_owned(),
                        probability: span.probability(),
                        span_start: span.offsets().0,
                        span_end: span.offsets().1,
                    });
                }
            }
        }
        Ok(seeded)
    }
}

fn relation_seed_labels() -> Vec<&'static str> {
    vec!["person", "organization", "location"]
}

fn filter_seed_sequence(sequence: Vec<Span>, threshold: f32) -> Vec<Span> {
    let mut best_by_surface = BTreeMap::<String, Span>::new();
    for span in sequence {
        if !relation_seed_span_allowed(&span, threshold) {
            continue;
        }
        let key = normalize_surface(span.text());
        match best_by_surface.get(&key) {
            Some(current) if !prefer_seed_span(&span, current) => {}
            _ => {
                best_by_surface.insert(key, span);
            }
        }
    }
    best_by_surface.into_values().collect()
}

fn relation_seed_span_allowed(span: &Span, threshold: f32) -> bool {
    if span.probability() < threshold {
        return false;
    }
    let surface = span.text().trim();
    if surface.is_empty() || is_structural_surface(surface) || is_generic_relation_surface(surface)
    {
        return false;
    }
    if !surface.chars().any(|value| value.is_alphabetic()) {
        return false;
    }
    match span.class() {
        "person" | "organization" => true,
        "location" => {
            surface.split_whitespace().count() >= 2 || span.probability() >= threshold.max(0.8)
        }
        _ => false,
    }
}

fn prefer_seed_span(candidate: &Span, current: &Span) -> bool {
    let probability_gap = candidate.probability() - current.probability();
    if probability_gap.abs() >= 0.05 {
        return probability_gap > 0.0;
    }
    let candidate_priority = label_priority(candidate.class());
    let current_priority = label_priority(current.class());
    if candidate_priority != current_priority {
        return candidate_priority > current_priority;
    }
    if candidate.text().len() != current.text().len() {
        return candidate.text().len() > current.text().len();
    }
    candidate.probability() > current.probability()
}

fn label_priority(label: &str) -> i32 {
    match label {
        "person" => 3,
        "organization" => 2,
        "location" => 1,
        _ => 0,
    }
}

fn normalize_surface(surface: &str) -> String {
    let mut out = String::with_capacity(surface.len());
    let mut previous_space = false;
    for ch in surface.chars() {
        if ch.is_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            previous_space = false;
        } else if ch.is_whitespace() && !previous_space && !out.is_empty() {
            out.push(' ');
            previous_space = true;
        }
    }
    out.trim().to_owned()
}

fn is_structural_surface(surface: &str) -> bool {
    let trimmed = surface.trim();
    trimmed.starts_with('#')
        || trimmed.eq_ignore_ascii_case("table of contents")
        || trimmed.to_ascii_lowercase().starts_with("chapter ")
}

fn is_generic_relation_surface(surface: &str) -> bool {
    matches!(
        normalize_surface(surface).as_str(),
        "hero"
            | "heroes"
            | "villain"
            | "villains"
            | "monster"
            | "monsters"
            | "criminal"
            | "criminals"
            | "city"
            | "town"
            | "team"
            | "group"
            | "member"
            | "members"
            | "people"
            | "person"
            | "security"
            | "driving"
            | "chapter"
            | "genius"
            | "guard"
            | "guards"
            | "officer"
            | "officers"
            | "doctor"
            | "chief"
            | "teacher"
            | "father"
            | "mother"
    )
}
