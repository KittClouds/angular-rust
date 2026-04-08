//! Stable public entrypoints for the Alex resolved-entity lexicon layer.
//!
//! This stage owns lexicon compilation and text scanning for already-known
//! entities. It expects `LexiconEntry` rows plus text/scope inputs and returns
//! known-match results. It persists nothing directly; callers persist snapshots
//! if they want reuse. Prefer these functions over reaching into `Lexicon`
//! internals from higher-level pipeline code.

use phoenix_types::{KnownMatch, LexiconEntry, LexiconSnapshot, ScopeKey};

use crate::{AlexError, ExactSurfacePattern, Lexicon, LexiconBuilder};

pub fn build_snapshot(entries: &[LexiconEntry]) -> Result<LexiconSnapshot, AlexError> {
    LexiconBuilder::build(entries)
}

pub fn build_lexicon(entries: &[LexiconEntry]) -> Result<Lexicon, AlexError> {
    Lexicon::from_entries(entries)
}

pub fn load_lexicon(snapshot: LexiconSnapshot) -> Result<Lexicon, AlexError> {
    Lexicon::from_snapshot(snapshot)
}

pub fn lookup(lexicon: &Lexicon, surface: &str, scope: &ScopeKey) -> Vec<LexiconEntry> {
    lexicon.lookup(surface, scope)
}

pub fn scan_text(lexicon: &Lexicon, text: &str, scope: &ScopeKey) -> Vec<KnownMatch> {
    lexicon.scan(text, scope)
}

pub fn fuzzy_anchor(lexicon: &Lexicon, token: &str, scope: &ScopeKey) -> Option<KnownMatch> {
    lexicon.fuzzy_anchor(token, scope)
}

pub fn exact_surface_patterns(lexicon: &Lexicon) -> Vec<ExactSurfacePattern> {
    lexicon.exact_surface_patterns()
}
