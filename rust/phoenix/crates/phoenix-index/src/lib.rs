use phoenix_alex::{AlexError, Lexicon};
use phoenix_qgram::{QgramConfig, QgramIndex};
use phoenix_types::{IndexedSpan, LexicalSearchResult, LexiconEntry, ScopeKey};

pub struct DeterministicIndex {
    lexical: QgramIndex,
    alias_lexicon: Lexicon,
}

impl DeterministicIndex {
    pub fn build(spans: &[IndexedSpan], entries: &[LexiconEntry], config: QgramConfig) -> Result<Self, AlexError> {
        Ok(Self {
            lexical: QgramIndex::build(spans, config),
            alias_lexicon: Lexicon::from_entries(entries)?,
        })
    }

    pub fn search(&self, query: &str, scope: &ScopeKey, limit: usize) -> LexicalSearchResult {
        self.lexical.search(query, scope, limit)
    }

    pub fn alias_lexicon(&self) -> &Lexicon {
        &self.alias_lexicon
    }

    pub fn lexical(&self) -> &QgramIndex {
        &self.lexical
    }
}
