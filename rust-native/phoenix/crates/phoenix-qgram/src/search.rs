use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap};

use phoenix_types::{
    Diagnostic, ImplicitMatchHit, IndexedSpan, LexicalField, LexicalSearchResult, ScopeKey, SpanHit,
};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

use crate::catalog::{SpanCatalog, SpanOrdinal};
use crate::grams::{extract_packed_gram_pair, extract_packed_grams, PackedGram};
use crate::implicit::match_implicit as match_implicit_impl;
use crate::postings::PostingSet;
use crate::query::{parse_query, Clause, ClauseType};
use crate::verifier::{PatternMatch, QueryVerifier};

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub k1: f64,
    pub b: f64,
    pub field_weights: BTreeMap<LexicalField, f64>,
    pub coverage_lambda: f64,
    pub coverage_epsilon: f64,
    pub phrase_hard: bool,
    pub proximity_alpha: f64,
    pub proximity_decay: f64,
    pub max_segments: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            field_weights: BTreeMap::new(),
            coverage_lambda: 3.0,
            coverage_epsilon: 0.1,
            phrase_hard: true,
            proximity_alpha: 0.5,
            proximity_decay: 0.1,
            max_segments: 32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct QgramConfig {
    pub trigram_width: usize,
    pub bigram_width: usize,
    pub bitmap_threshold: usize,
    pub max_clause_candidates: usize,
    pub search: SearchConfig,
}

impl Default for QgramConfig {
    fn default() -> Self {
        Self {
            trigram_width: 3,
            bigram_width: 2,
            bitmap_threshold: 256,
            max_clause_candidates: 256,
            search: SearchConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct QgramIndex {
    config: QgramConfig,
    catalog: SpanCatalog,
    trigram_postings: FxHashMap<PackedGram, PostingSet>,
    bigram_postings: FxHashMap<PackedGram, PostingSet>,
}

impl QgramIndex {
    pub fn build(spans: &[IndexedSpan], config: QgramConfig) -> Self {
        let catalog = SpanCatalog::build(spans);
        let mut index = Self {
            config,
            catalog,
            trigram_postings: FxHashMap::default(),
            bigram_postings: FxHashMap::default(),
        };
        index.reindex();
        index
    }

    pub fn rebuild_from_catalog(&mut self, spans: &[IndexedSpan]) {
        self.catalog = SpanCatalog::build(spans);
        self.reindex();
    }

    pub fn search(&self, query: &str, scope: &ScopeKey, limit: usize) -> LexicalSearchResult {
        let Some(prepared) = self.prepare_query(query) else {
            return LexicalSearchResult {
                span_hits: Vec::new(),
                diagnostics: vec![Diagnostic {
                    code: "PX_QGRAM_EMPTY_QUERY".to_owned(),
                    message: "No searchable clauses were produced from the query.".to_owned(),
                }],
            };
        };

        let mut workspace = QueryWorkspace::new(self.catalog.len());
        let mut clause_dfs = Vec::with_capacity(prepared.clauses.len());
        for clause in &prepared.clauses {
            let candidates = self.clause_candidates(clause, scope);
            for &ordinal in &candidates {
                workspace.bump(ordinal);
            }
            clause_dfs.push(candidates.len());
        }

        let mut candidate_ordinals = workspace.into_ranked_ordinals();
        candidate_ordinals
            .sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));

        let idfs = self.clause_idfs(&prepared, &clause_dfs);
        let mut span_hits = Vec::new();
        let mut top_hits = TopHits::new(limit);

        for (ordinal, _) in candidate_ordinals {
            let (matches, matched_count) = prepared
                .verifier
                .verify_span(&self.catalog, SpanOrdinal(ordinal));
            if matched_count == 0 {
                continue;
            }
            if self.config.search.phrase_hard
                && prepared.clauses.iter().enumerate().any(|(index, clause)| {
                    clause.clause.clause_type == ClauseType::Phrase
                        && matches.get(index).and_then(Option::as_ref).is_none()
                })
            {
                continue;
            }

            let score = self.score_span(&matches, matched_count, &idfs);
            let Some(span) = self.catalog.span(SpanOrdinal(ordinal)) else {
                continue;
            };
            let hit = SpanHit {
                span_id: span.span_id.clone(),
                note_id: span.note_id.clone(),
                document_id: span.document_id.clone(),
                score,
                coverage: matched_count as f32 / prepared.clauses.len() as f32,
            };

            if limit == 0 {
                span_hits.push(hit);
            } else {
                top_hits.push(hit);
            }
        }

        if limit > 0 {
            span_hits = top_hits.into_sorted_vec();
        } else {
            span_hits.sort_by(compare_span_hits);
        }

        LexicalSearchResult {
            span_hits,
            diagnostics: vec![Diagnostic {
                code: "PX_QGRAM_OK".to_owned(),
                message: "Qgram lexical search completed.".to_owned(),
            }],
        }
    }

    pub fn match_implicit(
        &self,
        text: &str,
        scope: &ScopeKey,
        lexicon: &phoenix_alex::Lexicon,
    ) -> Vec<ImplicitMatchHit> {
        match_implicit_impl(text, scope, lexicon)
    }

    fn reindex(&mut self) {
        self.trigram_postings.clear();
        self.bigram_postings.clear();

        let mut bigrams_buffer = Vec::new();
        let mut trigrams_buffer = Vec::new();
        for ordinal in 0..self.catalog.len() as u32 {
            let Some(span) = self.catalog.span(SpanOrdinal(ordinal)) else {
                continue;
            };
            for field in &span.fields {
                let normalized = self.catalog.field_text(field);

                extract_packed_gram_pair(
                    normalized,
                    self.config.bigram_width,
                    &mut bigrams_buffer,
                    self.config.trigram_width,
                    &mut trigrams_buffer,
                );
                for gram in &trigrams_buffer {
                    self.trigram_postings
                        .entry(*gram)
                        .or_default()
                        .add(ordinal, self.config.bitmap_threshold);
                }

                for gram in &bigrams_buffer {
                    self.bigram_postings
                        .entry(*gram)
                        .or_default()
                        .add(ordinal, self.config.bitmap_threshold);
                }
            }
        }
    }

    fn prepare_query(&self, query: &str) -> Option<PreparedQuery> {
        let clauses = parse_query(query);
        if clauses.is_empty() {
            return None;
        }

        let mut prepared_clauses = clauses
            .into_iter()
            .map(|clause| {
                let (gram_kind, grams, df_hint) = self.prepare_clause(&clause);
                PreparedClause {
                    clause,
                    gram_kind,
                    grams,
                    df_hint,
                }
            })
            .collect::<Vec<_>>();
        prepared_clauses.sort_by(|left, right| {
            left.df_hint
                .cmp(&right.df_hint)
                .then_with(|| left.clause.pattern.cmp(&right.clause.pattern))
        });

        let verifier = QueryVerifier::new(
            &prepared_clauses
                .iter()
                .map(|prepared| prepared.clause.clone())
                .collect::<Vec<_>>(),
        );

        Some(PreparedQuery {
            clauses: prepared_clauses,
            verifier,
        })
    }

    fn prepare_clause(
        &self,
        clause: &Clause,
    ) -> (ClauseGramKind, SmallVec<[PackedGram; 8]>, usize) {
        let fallback_df = self.catalog.stats().total_spans.max(1);
        match clause.pattern.len() {
            0 | 1 => (ClauseGramKind::AllOrdinals, SmallVec::new(), fallback_df),
            2 => {
                let mut grams = Vec::new();
                extract_packed_grams(&clause.pattern, self.config.bigram_width, &mut grams);
                let grams = SmallVec::from_vec(grams);
                let df_hint =
                    self.prepared_clause_df_hint(&grams, ClauseGramKind::Bigram, fallback_df);
                (ClauseGramKind::Bigram, grams, df_hint)
            }
            _ => {
                let mut grams = Vec::new();
                extract_packed_grams(&clause.pattern, self.config.trigram_width, &mut grams);
                let grams = SmallVec::from_vec(grams);
                let df_hint =
                    self.prepared_clause_df_hint(&grams, ClauseGramKind::Trigram, fallback_df);
                (ClauseGramKind::Trigram, grams, df_hint)
            }
        }
    }

    fn clause_candidates(&self, clause: &PreparedClause, scope: &ScopeKey) -> Vec<u32> {
        match clause.gram_kind {
            ClauseGramKind::AllOrdinals => self.catalog.filtered_ordinals(scope),
            ClauseGramKind::Bigram => {
                self.candidates_for_grams(&clause.grams, &self.bigram_postings, scope)
            }
            ClauseGramKind::Trigram => {
                self.candidates_for_grams(&clause.grams, &self.trigram_postings, scope)
            }
        }
    }

    fn candidates_for_grams(
        &self,
        grams: &[PackedGram],
        postings: &FxHashMap<PackedGram, PostingSet>,
        scope: &ScopeKey,
    ) -> Vec<u32> {
        if grams.is_empty() {
            return self.catalog.filtered_ordinals(scope);
        }

        let mut gram_postings = grams
            .iter()
            .filter_map(|gram| postings.get(gram).map(|posting| (*gram, posting)))
            .collect::<Vec<_>>();
        if gram_postings.len() != grams.len() {
            return Vec::new();
        }

        gram_postings.sort_by_key(|(_, posting)| posting.len());

        let mut current = gram_postings[0].1.clone();
        for (_, posting) in gram_postings.into_iter().skip(1) {
            current = current.intersect(posting, self.config.bitmap_threshold);
            if current.is_empty() {
                break;
            }
            if current.len() <= self.config.max_clause_candidates {
                break;
            }
        }

        current
            .to_vec()
            .into_iter()
            .filter(|ordinal| self.catalog.scope_matches(SpanOrdinal(*ordinal), scope))
            .collect()
    }

    fn clause_idfs(&self, prepared: &PreparedQuery, clause_dfs: &[usize]) -> Vec<f64> {
        prepared
            .clauses
            .iter()
            .zip(clause_dfs.iter().copied())
            .map(|(clause, fallback_df)| {
                let df = match clause.gram_kind {
                    ClauseGramKind::AllOrdinals => fallback_df.max(1),
                    ClauseGramKind::Bigram => {
                        self.gram_df(&clause.grams, &self.bigram_postings, fallback_df)
                    }
                    ClauseGramKind::Trigram => {
                        self.gram_df(&clause.grams, &self.trigram_postings, fallback_df)
                    }
                };
                idf(self.catalog.stats().total_spans.max(1), df.max(1))
            })
            .collect()
    }

    fn prepared_clause_df_hint(
        &self,
        grams: &[PackedGram],
        gram_kind: ClauseGramKind,
        fallback_df: usize,
    ) -> usize {
        match gram_kind {
            ClauseGramKind::AllOrdinals => fallback_df.max(1),
            ClauseGramKind::Bigram => self.gram_df(grams, &self.bigram_postings, fallback_df),
            ClauseGramKind::Trigram => self.gram_df(grams, &self.trigram_postings, fallback_df),
        }
    }

    fn gram_df(
        &self,
        grams: &[PackedGram],
        postings: &FxHashMap<PackedGram, PostingSet>,
        fallback_df: usize,
    ) -> usize {
        grams
            .iter()
            .filter_map(|gram| postings.get(gram).map(PostingSet::len))
            .min()
            .unwrap_or(fallback_df.max(1))
    }

    fn score_span(
        &self,
        matches: &[Option<PatternMatch>],
        matched_count: usize,
        idfs: &[f64],
    ) -> f64 {
        let mut base_sum = 0.0;
        let mut masks: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
        let stats = self.catalog.stats();

        for (index, matched) in matches.iter().enumerate() {
            let Some(matched) = matched else {
                continue;
            };
            let mut tf_star = 0.0;
            for (field, detail) in &matched.field_matches {
                let weight = self
                    .config
                    .search
                    .field_weights
                    .get(field)
                    .copied()
                    .unwrap_or(1.0);
                let avg_len = stats
                    .average_field_lengths
                    .get(field)
                    .copied()
                    .unwrap_or(100.0);
                let normalized_tf = normalized_term_frequency(
                    detail.count,
                    detail.field_length,
                    avg_len,
                    self.config.search.b,
                );
                tf_star += weight * normalized_tf;
            }
            base_sum += idfs[index] * saturate(tf_star, self.config.search.k1);
            masks.push(matched.segment_mask);
        }

        let coverage = matched_count as f64 / matches.len().max(1) as f64;
        let coverage_mult = (self.config.search.coverage_epsilon + coverage)
            .powf(self.config.search.coverage_lambda);

        let proximity = if masks.len() > 1 {
            pattern_proximity(
                &masks,
                self.config.search.proximity_alpha,
                self.config.search.max_segments,
                self.config.search.proximity_decay,
            )
        } else {
            1.0
        };

        base_sum * coverage_mult * proximity
    }
}

fn normalized_term_frequency(tf: usize, field_len: usize, avg_len: f64, b: f64) -> f64 {
    tf as f64 / (1.0 - b + b * field_len as f64 / avg_len.max(1.0))
}

fn saturate(tf_star: f64, k1: f64) -> f64 {
    (k1 + 1.0) * tf_star / (k1 + tf_star.max(1e-9))
}

fn pattern_proximity(masks: &[u32], alpha: f64, max_segments: u32, decay_lambda: f64) -> f64 {
    let common = masks
        .iter()
        .copied()
        .reduce(|left, right| left & right)
        .unwrap_or(0);
    let overlap = common.count_ones();
    let denom = max_segments.min(masks.len() as u32).max(1) as f64;
    let decay = (-decay_lambda).exp();
    1.0 + alpha * (overlap as f64 / denom) * decay
}

fn idf(total_spans: usize, df: usize) -> f64 {
    let total = total_spans as f64;
    let doc_freq = df as f64;
    (1.0 + (total - doc_freq + 0.5) / (doc_freq + 0.5)).ln()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClauseGramKind {
    AllOrdinals,
    Bigram,
    Trigram,
}

#[derive(Clone, Debug)]
struct PreparedClause {
    clause: Clause,
    gram_kind: ClauseGramKind,
    grams: SmallVec<[PackedGram; 8]>,
    df_hint: usize,
}

struct PreparedQuery {
    clauses: Vec<PreparedClause>,
    verifier: QueryVerifier,
}

#[derive(Debug)]
struct QueryWorkspace {
    clause_counts: Vec<u16>,
    touched: Vec<u32>,
}

impl QueryWorkspace {
    fn new(span_count: usize) -> Self {
        Self {
            clause_counts: vec![0; span_count],
            touched: Vec::new(),
        }
    }

    #[inline]
    fn bump(&mut self, ordinal: u32) {
        let slot = &mut self.clause_counts[ordinal as usize];
        if *slot == 0 {
            self.touched.push(ordinal);
        }
        *slot = slot.saturating_add(1);
    }

    fn into_ranked_ordinals(mut self) -> Vec<(u32, u16)> {
        let mut ranked = Vec::with_capacity(self.touched.len());
        for ordinal in self.touched.drain(..) {
            let index = ordinal as usize;
            ranked.push((ordinal, self.clause_counts[index]));
            self.clause_counts[index] = 0;
        }
        ranked
    }
}

#[derive(Clone, Debug)]
struct RankedSpanHit(SpanHit);

impl PartialEq for RankedSpanHit {
    fn eq(&self, other: &Self) -> bool {
        compare_span_hits(&self.0, &other.0) == Ordering::Equal
    }
}

impl Eq for RankedSpanHit {}

impl PartialOrd for RankedSpanHit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RankedSpanHit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| self.0.span_id.cmp(&other.0.span_id))
    }
}

#[derive(Debug)]
struct TopHits {
    limit: usize,
    hits: BinaryHeap<std::cmp::Reverse<RankedSpanHit>>,
}

impl TopHits {
    fn new(limit: usize) -> Self {
        Self {
            limit,
            hits: BinaryHeap::with_capacity(limit.max(1)),
        }
    }

    fn push(&mut self, hit: SpanHit) {
        if self.limit == 0 {
            return;
        }

        let ranked = RankedSpanHit(hit);
        if self.hits.len() < self.limit {
            self.hits.push(std::cmp::Reverse(ranked));
            return;
        }

        let should_replace = self
            .hits
            .peek()
            .map(|worst| ranked > worst.0)
            .unwrap_or(true);
        if should_replace {
            let _ = self.hits.pop();
            self.hits.push(std::cmp::Reverse(ranked));
        }
    }

    fn into_sorted_vec(self) -> Vec<SpanHit> {
        let mut hits = self
            .hits
            .into_iter()
            .map(|ranked| ranked.0 .0)
            .collect::<Vec<_>>();
        hits.sort_by(compare_span_hits);
        hits
    }
}

fn compare_span_hits(left: &SpanHit, right: &SpanHit) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.span_id.cmp(&right.span_id))
}

#[cfg(test)]
mod tests {
    use phoenix_types::{IndexedTextField, LexicalField, ScopeKey};

    use super::*;

    fn span(id: &str, title: &str, body: &str) -> IndexedSpan {
        IndexedSpan {
            span_id: id.to_owned(),
            note_id: Some(phoenix_types::NoteId(id.to_owned())),
            document_id: Some(phoenix_types::DocumentId(id.to_owned())),
            scope: ScopeKey::default(),
            fields: vec![
                IndexedTextField {
                    field: LexicalField::Title,
                    text: title.to_owned(),
                },
                IndexedTextField {
                    field: LexicalField::Body,
                    text: body.to_owned(),
                },
            ],
        }
    }

    #[test]
    fn bigram_sidecar_supports_two_character_queries() {
        let index = QgramIndex::build(
            &[span("doc-1", "CEO", "The C.E.O. arrived.")],
            QgramConfig::default(),
        );

        let results = index.search("ed", &ScopeKey::default(), 10);
        assert_eq!(results.span_hits.len(), 1);
        assert_eq!(results.span_hits[0].span_id, "doc-1");
    }

    #[test]
    fn phrase_and_punctuation_queries_verify_exactly() {
        let index = QgramIndex::build(
            &[
                span("doc-1", "Leader", "The C.E.O. arrived."),
                span("doc-2", "Leader", "The CEO arrived."),
            ],
            QgramConfig::default(),
        );

        let results = index.search(r#""C.E.O.""#, &ScopeKey::default(), 10);
        assert_eq!(results.span_hits.len(), 1);
        assert_eq!(results.span_hits[0].span_id, "doc-1");
    }

    #[test]
    fn coverage_and_proximity_favor_full_close_matches() {
        let index = QgramIndex::build(
            &[
                span("doc-1", "Alpha", "alpha bravo charlie"),
                span("doc-2", "Alpha", "alpha xxxxxxxxxxxxxxxxxxxxxxxxx bravo"),
                span("doc-3", "Alpha", "alpha only"),
            ],
            QgramConfig::default(),
        );

        let results = index.search("alpha bravo", &ScopeKey::default(), 10);
        assert_eq!(results.span_hits[0].span_id, "doc-1");
        assert_eq!(results.span_hits[1].span_id, "doc-2");
        assert_eq!(results.span_hits[2].span_id, "doc-3");
    }

    #[test]
    fn top_k_search_keeps_best_hits_without_sorting_everything() {
        let index = QgramIndex::build(
            &[
                span("doc-1", "Alpha", "alpha bravo charlie"),
                span("doc-2", "Alpha", "alpha bravo"),
                span("doc-3", "Alpha", "alpha"),
                span("doc-4", "Alpha", "alpha bravo delta"),
            ],
            QgramConfig::default(),
        );

        let results = index.search("alpha bravo", &ScopeKey::default(), 2);

        assert_eq!(results.span_hits.len(), 2);
        assert!(results.span_hits[0].score >= results.span_hits[1].score);
    }
}
