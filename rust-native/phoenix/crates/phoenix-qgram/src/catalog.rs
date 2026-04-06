use phoenix_alex::{normalize_raw, scope_matches};
use phoenix_types::{IndexedSpan, IndexedTextField, LexicalField, NoteId, ScopeKey, TextRange};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpanOrdinal(pub u32);

#[derive(Clone, Debug)]
pub struct CatalogField {
    pub field: LexicalField,
    pub arena_range: TextRange,
    pub field_length: u32,
}

#[derive(Clone, Debug)]
pub struct CatalogSpan {
    pub span_id: String,
    pub note_id: Option<NoteId>,
    pub document_id: Option<phoenix_types::DocumentId>,
    pub scope: ScopeKey,
    pub fields: Vec<CatalogField>,
}

#[derive(Clone, Debug, Default)]
pub struct CorpusStats {
    pub total_spans: usize,
    pub average_field_lengths: FxHashMap<LexicalField, f64>,
}

#[derive(Clone, Debug, Default)]
pub struct SpanCatalog {
    arena: String,
    spans: Vec<CatalogSpan>,
    stats: CorpusStats,
}

impl SpanCatalog {
    pub fn build(spans: &[IndexedSpan]) -> Self {
        let mut arena = String::new();
        let mut catalog_spans = Vec::with_capacity(spans.len());
        let mut field_sums = FxHashMap::<LexicalField, usize>::default();
        let mut field_counts = FxHashMap::<LexicalField, usize>::default();

        for span in spans {
            let mut fields = Vec::new();
            for field in &span.fields {
                let normalized = normalize_field(field);
                if normalized.is_empty() {
                    continue;
                }
                let start = arena.len();
                arena.push_str(&normalized);
                let end = arena.len();
                arena.push('\n');
                fields.push(CatalogField {
                    field: field.field.clone(),
                    arena_range: TextRange {
                        start: start as u32,
                        end: end as u32,
                    },
                    field_length: normalized.len() as u32,
                });
                *field_sums.entry(field.field.clone()).or_insert(0) += normalized.len();
                *field_counts.entry(field.field.clone()).or_insert(0) += 1;
            }

            catalog_spans.push(CatalogSpan {
                span_id: span.span_id.clone(),
                note_id: span.note_id.clone(),
                document_id: span.document_id.clone(),
                scope: span.scope.clone(),
                fields,
            });
        }

        let average_field_lengths = field_sums
            .into_iter()
            .map(|(field, sum)| {
                let count = field_counts.get(&field).copied().unwrap_or(1);
                (field, sum as f64 / count as f64)
            })
            .collect();

        Self {
            arena,
            spans: catalog_spans,
            stats: CorpusStats {
                total_spans: spans.len(),
                average_field_lengths,
            },
        }
    }

    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn stats(&self) -> &CorpusStats {
        &self.stats
    }

    pub fn all_ordinals(&self) -> Vec<u32> {
        (0..self.spans.len() as u32).collect()
    }

    pub fn span(&self, ordinal: SpanOrdinal) -> Option<&CatalogSpan> {
        self.spans.get(ordinal.0 as usize)
    }

    pub fn field_text<'a>(&'a self, field: &CatalogField) -> &'a str {
        self.arena
            .get(field.arena_range.start as usize..field.arena_range.end as usize)
            .unwrap_or_default()
    }

    pub fn scope_matches(&self, ordinal: SpanOrdinal, scope: &ScopeKey) -> bool {
        self.span(ordinal)
            .map(|span| scope_matches(&span.scope, scope))
            .unwrap_or(false)
    }

    pub fn filtered_ordinals(&self, scope: &ScopeKey) -> Vec<u32> {
        self.spans
            .iter()
            .enumerate()
            .filter_map(|(index, span)| {
                if scope_matches(&span.scope, scope) {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .collect()
    }
}

fn normalize_field(field: &IndexedTextField) -> String {
    normalize_raw(&field.text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_types::{IndexedSpan, IndexedTextField, LexicalField};

    #[test]
    fn span_catalog_normalizes_fields_into_one_arena() {
        let catalog = SpanCatalog::build(&[IndexedSpan {
            span_id: "span-1".to_owned(),
            note_id: None,
            document_id: None,
            scope: ScopeKey::default(),
            fields: vec![
                IndexedTextField {
                    field: LexicalField::Title,
                    text: "Grand Line".to_owned(),
                },
                IndexedTextField {
                    field: LexicalField::Body,
                    text: "Monkey D. Luffy sailed fast.".to_owned(),
                },
            ],
        }]);

        assert_eq!(catalog.len(), 1);
        let span = catalog.span(SpanOrdinal(0)).expect("span");
        assert_eq!(span.fields.len(), 2);
        assert_eq!(catalog.field_text(&span.fields[0]), "grand line");
        assert_eq!(
            catalog.field_text(&span.fields[1]),
            "monkey d. luffy sailed fast."
        );
    }
}
