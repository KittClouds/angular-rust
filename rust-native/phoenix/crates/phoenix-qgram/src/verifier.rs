use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use phoenix_types::LexicalField;
use rustc_hash::FxHashMap;

use crate::catalog::{SpanCatalog, SpanOrdinal};
use crate::query::Clause;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MatchDetail {
    pub count: usize,
    pub positions: Vec<usize>,
    pub field_length: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PatternMatch {
    pub field_matches: FxHashMap<LexicalField, MatchDetail>,
    pub total_occ: usize,
    pub segment_mask: u32,
}

pub struct QueryVerifier {
    pub clauses: Vec<Clause>,
    matcher: Option<DoubleArrayAhoCorasick>,
    pattern_clause_indices: Vec<Vec<usize>>,
}

impl QueryVerifier {
    pub fn new(clauses: &[Clause]) -> Self {
        if clauses.is_empty() {
            return Self {
                clauses: Vec::new(),
                matcher: None,
                pattern_clause_indices: Vec::new(),
            };
        }

        let mut unique_patterns = Vec::<String>::new();
        let mut pattern_clause_indices = Vec::<Vec<usize>>::new();

        for (index, clause) in clauses.iter().enumerate() {
            if let Some(existing) = unique_patterns
                .iter()
                .position(|pattern| pattern == &clause.pattern)
            {
                pattern_clause_indices[existing].push(index);
            } else {
                unique_patterns.push(clause.pattern.clone());
                pattern_clause_indices.push(vec![index]);
            }
        }

        let matcher = if unique_patterns.is_empty() {
            None
        } else {
            DoubleArrayAhoCorasickBuilder::new()
                .match_kind(MatchKind::Standard)
                .build(&unique_patterns)
                .ok()
        };

        Self {
            clauses: clauses.to_vec(),
            matcher,
            pattern_clause_indices,
        }
    }

    pub fn verify_span(
        &self,
        catalog: &SpanCatalog,
        ordinal: SpanOrdinal,
    ) -> (Vec<Option<PatternMatch>>, usize) {
        let Some(span) = catalog.span(ordinal) else {
            return (Vec::new(), 0);
        };
        let Some(matcher) = &self.matcher else {
            return (Vec::new(), 0);
        };

        let mut matches = vec![None; self.clauses.len()];
        let mut matched_count = 0usize;

        for field in &span.fields {
            let normalized = catalog.field_text(field);
            if normalized.is_empty() {
                continue;
            }
            let mut iter = matcher.find_overlapping_iter(normalized.as_bytes());
            while let Some(found) = iter.next() {
                let pattern_index = found.value() as usize;
                let start = found.start();
                for clause_index in &self.pattern_clause_indices[pattern_index] {
                    let entry = matches[*clause_index].get_or_insert_with(|| {
                        matched_count += 1;
                        PatternMatch::default()
                    });
                    let detail = entry
                        .field_matches
                        .entry(field.field.clone())
                        .or_insert_with(|| MatchDetail {
                            count: 0,
                            positions: Vec::new(),
                            field_length: field.field_length as usize,
                        });
                    detail.count += 1;
                    detail.positions.push(start);
                    entry.total_occ += 1;
                    let mut segment = (start * 32) / (field.field_length as usize).max(1);
                    if segment >= 32 {
                        segment = 31;
                    }
                    entry.segment_mask |= 1 << segment;
                }
            }
        }

        (matches, matched_count)
    }
}

#[cfg(test)]
mod tests {
    use phoenix_types::{IndexedSpan, IndexedTextField, ScopeKey};

    use super::*;
    use crate::query::{parse_query, ClauseType};

    #[test]
    fn overlapping_verification_matches_go_style_positions() {
        let catalog = SpanCatalog::build(&[IndexedSpan {
            span_id: "doc-1".to_owned(),
            note_id: None,
            document_id: None,
            scope: ScopeKey::default(),
            fields: vec![IndexedTextField {
                field: LexicalField::Body,
                text: "banana".to_owned(),
            }],
        }]);

        let verifier = QueryVerifier::new(&parse_query("ana"));
        let (matches, matched_count) = verifier.verify_span(&catalog, SpanOrdinal(0));

        assert_eq!(matched_count, 1);
        let matched = matches[0].as_ref().expect("match");
        assert_eq!(
            matched.field_matches[&LexicalField::Body].positions,
            vec![1, 3]
        );
    }

    #[test]
    fn phrase_pattern_keeps_punctuation() {
        let clauses = parse_query(r#""C.E.O.""#);
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].clause_type, ClauseType::Phrase);
        assert_eq!(clauses[0].pattern, "c.e.o.");
    }
}
