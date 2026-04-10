use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::gliclass::GliclassLabelScore;

const LABEL_TOKEN: &str = "<<LABEL>>";
const SEP_TOKEN: &str = "<<SEP>>";
const EXAMPLE_TOKEN: &str = "<<EXAMPLE>>";

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GliclassInstructLabel {
    pub label: String,
    #[serde(default)]
    pub description: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GliclassInstructExample {
    pub text: String,
    #[serde(default, alias = "true_labels")]
    pub labels: Vec<String>,
}

pub fn build_input(
    prompt_first: bool,
    text: &str,
    labels: &[GliclassInstructLabel],
    examples: &[GliclassInstructExample],
    prompt: Option<&str>,
) -> String {
    let prompt = prompt
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or_default();
    let examples = format_examples_prompt(examples);
    let label_bytes = labels
        .iter()
        .map(|label| LABEL_TOKEN.len() + label.label.len() + description_len(label))
        .sum::<usize>();
    let mut prefix = String::with_capacity(label_bytes + SEP_TOKEN.len() + prompt.len() + 1);
    for label in labels {
        prefix.push_str(LABEL_TOKEN);
        prefix.push_str(&label.label);
        if let Some(description) = label.description.as_deref().map(str::trim) {
            if !description.is_empty() {
                prefix.push_str(": ");
                prefix.push_str(description);
            }
        }
    }
    prefix.push_str(SEP_TOKEN);
    if !prompt.is_empty() {
        prefix.push_str(prompt);
    }

    let mut input = String::with_capacity(prefix.len() + text.len() + examples.len());
    if prompt_first {
        input.push_str(&prefix);
        input.push_str(text);
    } else {
        input.push_str(text);
        input.push_str(&prefix);
    }
    input.push_str(&examples);
    input
}

pub fn flatten_hierarchical_labels(value: &Value, separator: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    flatten_labels_inner(value, "", separator, &mut out)?;
    Ok(out)
}

pub fn build_hierarchical_scores(
    original_labels: &Value,
    scores: &[GliclassLabelScore],
    separator: &str,
) -> Result<Value, String> {
    let score_lookup = scores
        .iter()
        .map(|row| (row.label.as_str(), row.score))
        .collect::<std::collections::BTreeMap<_, _>>();
    build_hierarchical_inner(original_labels, "", separator, &score_lookup)
}

fn description_len(label: &GliclassInstructLabel) -> usize {
    label
        .description
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.len() + 2)
        .unwrap_or(0)
}

fn format_examples_prompt(examples: &[GliclassInstructExample]) -> String {
    if examples.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for example in examples {
        out.push_str(EXAMPLE_TOKEN);
        out.push_str(example.text.trim());
        out.push_str(" \nLabels:\n ");
        out.push_str(&example.labels.join(", "));
    }
    out.push_str(SEP_TOKEN);
    out
}

fn flatten_labels_inner(
    value: &Value,
    prefix: &str,
    separator: &str,
    out: &mut Vec<String>,
) -> Result<(), String> {
    match value {
        Value::String(label) => {
            if prefix.is_empty() {
                out.push(label.clone());
            } else {
                out.push(format!("{prefix}{separator}{label}"));
            }
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                flatten_labels_inner(item, prefix, separator, out)?;
            }
            Ok(())
        }
        Value::Object(map) => {
            for (key, nested) in map {
                let next_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}{separator}{key}")
                };
                flatten_labels_inner(nested, &next_prefix, separator, out)?;
            }
            Ok(())
        }
        other => Err(format!("unsupported hierarchical label type: {}", other)),
    }
}

fn build_hierarchical_inner(
    value: &Value,
    prefix: &str,
    separator: &str,
    score_lookup: &std::collections::BTreeMap<&str, f32>,
) -> Result<Value, String> {
    match value {
        Value::String(label) => {
            let full_label = if prefix.is_empty() {
                label.clone()
            } else {
                format!("{prefix}{separator}{label}")
            };
            Ok(Value::from(
                score_lookup
                    .get(full_label.as_str())
                    .copied()
                    .unwrap_or(0.0),
            ))
        }
        Value::Array(items) => {
            let mut map = Map::new();
            for item in items {
                let Value::String(label) = item else {
                    return Err("hierarchical arrays must only contain strings".to_owned());
                };
                let full_label = if prefix.is_empty() {
                    label.clone()
                } else {
                    format!("{prefix}{separator}{label}")
                };
                map.insert(
                    label.clone(),
                    Value::from(
                        score_lookup
                            .get(full_label.as_str())
                            .copied()
                            .unwrap_or(0.0),
                    ),
                );
            }
            Ok(Value::Object(map))
        }
        Value::Object(items) => {
            let mut map = Map::new();
            for (key, nested) in items {
                let next_prefix = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}{separator}{key}")
                };
                map.insert(
                    key.clone(),
                    build_hierarchical_inner(nested, &next_prefix, separator, score_lookup)?,
                );
            }
            Ok(Value::Object(map))
        }
        other => Err(format!("unsupported hierarchical label type: {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_input_supports_descriptions_and_examples() {
        let labels = vec![GliclassInstructLabel {
            label: "space".to_owned(),
            description: Some("astronomy and planetary science".to_owned()),
        }];
        let examples = vec![GliclassInstructExample {
            text: "Mars mission update".to_owned(),
            labels: vec!["space".to_owned()],
        }];
        let input = build_input(
            true,
            "NASA launched a rover.",
            &labels,
            &examples,
            Some("Classify the topic:"),
        );
        assert_eq!(
            input,
            "<<LABEL>>space: astronomy and planetary science<<SEP>>Classify the topic:NASA launched a rover.<<EXAMPLE>>Mars mission update \nLabels:\n space<<SEP>>"
        );
    }

    #[test]
    fn flatten_hierarchical_labels_uses_dot_notation() {
        let value = serde_json::json!({
            "science": ["space", "biology"],
            "society": { "public": ["policy"] }
        });
        let flattened = flatten_hierarchical_labels(&value, ".").expect("flatten labels");
        assert_eq!(
            flattened,
            vec![
                "science.space".to_owned(),
                "science.biology".to_owned(),
                "society.public.policy".to_owned()
            ]
        );
    }

    #[test]
    fn build_hierarchical_scores_reconstructs_shape() {
        let value = serde_json::json!({
            "sentiment": ["positive", "negative"]
        });
        let scores = vec![
            GliclassLabelScore {
                label: "sentiment.positive".to_owned(),
                logit: 1.0,
                score: 0.9,
            },
            GliclassLabelScore {
                label: "sentiment.negative".to_owned(),
                logit: -1.0,
                score: 0.1,
            },
        ];
        let hierarchical =
            build_hierarchical_scores(&value, &scores, ".").expect("build hierarchical scores");
        let sentiment = hierarchical
            .get("sentiment")
            .and_then(Value::as_object)
            .expect("sentiment object");
        let positive = sentiment
            .get("positive")
            .and_then(Value::as_f64)
            .expect("positive score");
        let negative = sentiment
            .get("negative")
            .and_then(Value::as_f64)
            .expect("negative score");
        assert!((positive - 0.9).abs() < 1e-5);
        assert!((negative - 0.1).abs() < 1e-5);
    }
}
