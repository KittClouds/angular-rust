mod catalog;
mod grams;
mod implicit;
mod postings;
mod query;
mod search;
mod verifier;

pub use catalog::{CatalogSpan, CorpusStats, SpanCatalog, SpanOrdinal};
pub use grams::{extract_packed_grams, pack_ngram, unpack_ngram, PackedGram};
pub use implicit::match_implicit;
pub use postings::PostingSet;
pub use query::{parse_query, Clause, ClauseType};
pub use search::{QgramConfig, QgramIndex, SearchConfig};
pub use verifier::{MatchDetail, PatternMatch, QueryVerifier};
