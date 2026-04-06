mod fuzzy;
mod normalize;
mod sentence;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use daachorse::{DoubleArrayAhoCorasick, DoubleArrayAhoCorasickBuilder, MatchKind};
use fst::{Map, MapBuilder};
use phoenix_types::{
    KnownMatch, KnownMatchSource, LexiconEntry, LexiconSnapshot, LexiconStats,
    LexiconSurfaceSource, ScopeKey,
};
use thiserror::Error;

use fuzzy::find_matching_anchors;
pub use normalize::{
    canonicalize_with_offsets, generate_auto_aliases, is_sentence_guard, is_stop_word,
    is_stop_word_with_profile, normalize_raw, normalized_has_meaningful_token, phrase_key,
    scope_matches, strip_possessive, tokenize_norm, tokens_from_normalized, MAX_PHRASE_TOKENS,
    TOK_SEP,
};
pub use sentence::split_sentence_ranges;

#[derive(Debug, Error)]
pub enum AlexError {
    #[error("fst build failed")]
    FstBuild,
    #[error("fst load failed")]
    FstLoad,
    #[error("aho-corasick build failed")]
    ExactMatcherBuild,
    #[error("snapshot serialization failed")]
    SnapshotSerialize,
    #[error("snapshot deserialization failed")]
    SnapshotDeserialize,
}

#[derive(Clone, Debug)]
struct ExactPattern {
    surface: String,
    bucket_index: usize,
    source: LexiconSurfaceSource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExactSurfacePattern {
    pub surface: String,
    pub bucket_index: usize,
    pub source: LexiconSurfaceSource,
}

pub struct LexiconBuilder;

impl LexiconBuilder {
    pub fn build(entries: &[LexiconEntry]) -> Result<LexiconSnapshot, AlexError> {
        let compiled_at = now_ms();

        let mut phrase_key_to_entries: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
        let mut exact_surface_to_meta: BTreeMap<String, (String, LexiconSurfaceSource)> =
            BTreeMap::new();
        let mut entry_tokens: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut token_df: HashMap<String, usize> = HashMap::new();
        let mut token_owner: HashMap<String, usize> = HashMap::new();

        for (index, entry) in entries.iter().enumerate() {
            let mut surfaces = Vec::with_capacity(1 + entry.aliases.len() + 4);
            surfaces.push((entry.label.clone(), LexiconSurfaceSource::Canonical));
            surfaces.extend(
                entry
                    .aliases
                    .iter()
                    .cloned()
                    .map(|alias| (alias, LexiconSurfaceSource::Alias)),
            );
            surfaces.extend(
                generate_auto_aliases(&entry.label, entry.kind.as_ref())
                    .into_iter()
                    .map(|alias| (alias, LexiconSurfaceSource::AutoAlias)),
            );

            for (surface, source) in surfaces {
                let Some(key) = phrase_key(&surface) else {
                    continue;
                };
                let normalized_surface = normalize_raw(&surface);
                phrase_key_to_entries
                    .entry(key.clone())
                    .or_default()
                    .insert(index);
                exact_surface_to_meta
                    .entry(normalized_surface)
                    .and_modify(|current| {
                        if surface_source_rank(&source) < surface_source_rank(&current.1) {
                            current.1 = source.clone();
                        }
                    })
                    .or_insert((key, source));
            }

            let tokens = tokenize_norm(&entry.label);
            let unique_tokens = tokens.iter().cloned().collect::<BTreeSet<_>>();
            for token in unique_tokens {
                *token_df.entry(token.clone()).or_insert(0) += 1;
                token_owner.entry(token).or_insert(index);
            }
            entry_tokens.insert(entry.entity_id.0.clone(), tokens);
        }

        let mut builder = MapBuilder::memory();
        let mut buckets = Vec::with_capacity(phrase_key_to_entries.len());
        let mut phrase_key_to_bucket_index = BTreeMap::new();
        for (bucket_index, (key, entry_indices)) in phrase_key_to_entries.into_iter().enumerate() {
            builder
                .insert(key.as_str(), bucket_index as u64)
                .map_err(|_| AlexError::FstBuild)?;
            phrase_key_to_bucket_index.insert(key, bucket_index);
            buckets.push(entry_indices.into_iter().collect::<Vec<_>>());
        }
        let fst_bytes = builder.into_inner().map_err(|_| AlexError::FstBuild)?;

        let mut exact_patterns = Vec::with_capacity(exact_surface_to_meta.len());
        for (surface, (key, source)) in exact_surface_to_meta {
            if let Some(bucket_index) = phrase_key_to_bucket_index.get(&key) {
                exact_patterns.push(ExactPattern {
                    surface,
                    bucket_index: *bucket_index,
                    source,
                });
            }
        }
        exact_patterns.sort_by(|left, right| left.surface.cmp(&right.surface));

        let mut unique_token_to_entry = BTreeMap::new();
        for (token, df) in &token_df {
            if *df == 1 {
                if let Some(owner) = token_owner.get(token) {
                    unique_token_to_entry.insert(token.clone(), *owner);
                }
            }
        }

        let mut anchor_to_entries: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (index, entry) in entries.iter().enumerate() {
            let tokens = entry_tokens
                .get(&entry.entity_id.0)
                .cloned()
                .unwrap_or_default();
            let mut candidates = tokens
                .iter()
                .filter(|token| token.len() >= 3)
                .cloned()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            candidates.sort_by_key(|token| token_df.get(token).copied().unwrap_or(usize::MAX));
            for token in candidates.into_iter().take(3) {
                anchor_to_entries.entry(token).or_default().push(index);
            }
        }

        let anchor_count = anchor_to_entries.len();
        let unique_token_count = unique_token_to_entry.len();

        Ok(LexiconSnapshot {
            version: 1,
            compiled_at,
            fst_bytes,
            entries: entries.to_vec(),
            buckets,
            exact_surfaces: exact_patterns
                .iter()
                .map(|pattern| pattern.surface.clone())
                .collect(),
            exact_surface_bucket_indices: exact_patterns
                .iter()
                .map(|pattern| pattern.bucket_index)
                .collect(),
            exact_surface_sources: exact_patterns
                .iter()
                .map(|pattern| pattern.source.clone())
                .collect(),
            unique_token_to_entry,
            anchor_to_entries,
            entry_tokens,
            stats: LexiconStats {
                entity_count: entries.len(),
                exact_surface_count: exact_patterns.len(),
                anchor_count,
                unique_token_count,
            },
        })
    }
}

fn now_ms() -> i64 {
    #[cfg(target_arch = "wasm32")]
    {
        js_sys::Date::now() as i64
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_millis() as i64
    }
}

pub struct Lexicon {
    snapshot: LexiconSnapshot,
    fst: Map<Vec<u8>>,
    exact_matcher: Option<DoubleArrayAhoCorasick>,
}

impl Lexicon {
    pub fn from_entries(entries: &[LexiconEntry]) -> Result<Self, AlexError> {
        let snapshot = LexiconBuilder::build(entries)?;
        Self::from_snapshot(snapshot)
    }

    pub fn from_snapshot(snapshot: LexiconSnapshot) -> Result<Self, AlexError> {
        let fst = Map::new(snapshot.fst_bytes.clone()).map_err(|_| AlexError::FstLoad)?;
        let exact_matcher = if snapshot.exact_surfaces.is_empty() {
            None
        } else {
            Some(
                DoubleArrayAhoCorasickBuilder::new()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(&snapshot.exact_surfaces)
                    .map_err(|_| AlexError::ExactMatcherBuild)?,
            )
        };
        Ok(Self {
            snapshot,
            fst,
            exact_matcher,
        })
    }

    pub fn to_snapshot(&self) -> LexiconSnapshot {
        self.snapshot.clone()
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, AlexError> {
        bincode::serialize(&self.snapshot).map_err(|_| AlexError::SnapshotSerialize)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, AlexError> {
        let snapshot = bincode::deserialize(bytes).map_err(|_| AlexError::SnapshotDeserialize)?;
        Self::from_snapshot(snapshot)
    }

    pub fn stats(&self) -> LexiconStats {
        self.snapshot.stats.clone()
    }

    pub fn exact_surface_patterns(&self) -> Vec<ExactSurfacePattern> {
        self.snapshot
            .exact_surfaces
            .iter()
            .cloned()
            .zip(self.snapshot.exact_surface_bucket_indices.iter().copied())
            .zip(self.snapshot.exact_surface_sources.iter().cloned())
            .map(|((surface, bucket_index), source)| ExactSurfacePattern {
                surface,
                bucket_index,
                source,
            })
            .collect()
    }

    pub fn lookup(&self, surface: &str, scope: &ScopeKey) -> Vec<LexiconEntry> {
        let Some(key) = phrase_key(surface) else {
            return Vec::new();
        };
        let Some(bucket_index) = self.fst.get(key.as_str()) else {
            return Vec::new();
        };
        self.scoped_bucket(bucket_index as usize, scope)
    }

    pub fn scan(&self, text: &str, scope: &ScopeKey) -> Vec<KnownMatch> {
        let (canonicalized, offsets) = canonicalize_with_offsets(text);
        let Some(exact_matcher) = &self.exact_matcher else {
            return Vec::new();
        };
        exact_matcher
            .leftmost_find_iter(canonicalized.as_bytes())
            .filter_map(|matched| {
                let pattern_index = matched.value() as usize;
                let bucket_index = self.snapshot.exact_surface_bucket_indices[pattern_index];
                let source = &self.snapshot.exact_surface_sources[pattern_index];
                let entries = self.scoped_bucket(bucket_index, scope);
                if entries.is_empty() {
                    return None;
                }
                let start = offsets.get(matched.start()).copied().unwrap_or(0);
                let end = offsets.get(matched.end()).copied().unwrap_or(text.len());
                let surface = text.get(start..end).unwrap_or_default().to_owned();
                Some(KnownMatch {
                    range: phoenix_types::TextRange {
                        start: start as u32,
                        end: end as u32,
                    },
                    surface,
                    entries,
                    source: Some(match source {
                        LexiconSurfaceSource::Canonical => KnownMatchSource::ExactCanonical,
                        LexiconSurfaceSource::Alias => KnownMatchSource::ExactAlias,
                        LexiconSurfaceSource::AutoAlias => KnownMatchSource::ExactAutoAlias,
                    }),
                    confidence: 1.0,
                })
            })
            .collect()
    }

    pub fn fuzzy_anchor(&self, token: &str, scope: &ScopeKey) -> Option<KnownMatch> {
        let normalized = normalize_raw(token);
        let stripped = strip_possessive(&normalized);
        let exact_entry = self
            .snapshot
            .unique_token_to_entry
            .get(stripped)
            .copied()
            .and_then(|index| self.snapshot.entries.get(index).cloned())
            .filter(|entry| scope_matches(&entry.scope, scope));
        if let Some(entry) = exact_entry {
            return Some(single_entry_match(
                token,
                entry,
                KnownMatchSource::FuzzyAnchor,
                0.92,
            ));
        }

        if let Some(entry_indices) = self.snapshot.anchor_to_entries.get(stripped) {
            let entries = entry_indices
                .iter()
                .filter_map(|index| self.snapshot.entries.get(*index).cloned())
                .filter(|entry| scope_matches(&entry.scope, scope))
                .collect::<Vec<_>>();
            if !entries.is_empty() {
                return Some(KnownMatch {
                    range: phoenix_types::TextRange::default(),
                    surface: token.to_owned(),
                    entries,
                    source: Some(KnownMatchSource::FuzzyAnchor),
                    confidence: 0.9,
                });
            }
        }

        for anchor in find_matching_anchors(stripped, &self.snapshot.anchor_to_entries) {
            if let Some(entry_indices) = self.snapshot.anchor_to_entries.get(&anchor) {
                let entries = entry_indices
                    .iter()
                    .filter_map(|index| self.snapshot.entries.get(*index).cloned())
                    .filter(|entry| scope_matches(&entry.scope, scope))
                    .collect::<Vec<_>>();
                if !entries.is_empty() {
                    return Some(KnownMatch {
                        range: phoenix_types::TextRange::default(),
                        surface: token.to_owned(),
                        entries,
                        source: Some(KnownMatchSource::FuzzyAnchor),
                        confidence: 0.82,
                    });
                }
            }
        }

        None
    }

    fn scoped_bucket(&self, bucket_index: usize, scope: &ScopeKey) -> Vec<LexiconEntry> {
        self.snapshot
            .buckets
            .get(bucket_index)
            .into_iter()
            .flatten()
            .filter_map(|entry_index| self.snapshot.entries.get(*entry_index))
            .filter(|entry| scope_matches(&entry.scope, scope))
            .cloned()
            .collect()
    }
}

fn surface_source_rank(source: &LexiconSurfaceSource) -> usize {
    match source {
        LexiconSurfaceSource::Canonical => 0,
        LexiconSurfaceSource::Alias => 1,
        LexiconSurfaceSource::AutoAlias => 2,
    }
}

fn single_entry_match(
    token: &str,
    entry: LexiconEntry,
    source: KnownMatchSource,
    confidence: f32,
) -> KnownMatch {
    KnownMatch {
        range: phoenix_types::TextRange::default(),
        surface: token.to_owned(),
        entries: vec![entry],
        source: Some(source),
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use phoenix_types::{EntityId, EntityKind, GenderHint};

    use super::*;

    fn entry(
        id: &str,
        label: &str,
        kind: EntityKind,
        narrative_id: Option<&str>,
        aliases: &[&str],
    ) -> LexiconEntry {
        LexiconEntry {
            entity_id: EntityId(id.to_owned()),
            label: label.to_owned(),
            aliases: aliases.iter().map(|alias| (*alias).to_owned()).collect(),
            kind: Some(kind),
            gender: Some(GenderHint::Unknown),
            number: None,
            scope: ScopeKey {
                world_id: None,
                narrative_id: narrative_id.map(str::to_owned),
                folder_id: None,
                folder_path: None,
            },
        }
    }

    #[test]
    fn exact_multiword_lookup_handles_joiners_and_possessives() {
        let lexicon = Lexicon::from_entries(&[entry(
            "luffy",
            "Monkey D. Luffy",
            EntityKind::Character,
            None,
            &[],
        )])
        .expect("lexicon");

        let matches = lexicon.scan("Monkey D. Luffy's hat shimmered.", &ScopeKey::default());
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].surface, "Monkey D. Luffy");
    }

    #[test]
    fn lookup_uses_auto_aliases() {
        let lexicon = Lexicon::from_entries(&[
            entry("luffy", "Monkey D. Luffy", EntityKind::Character, None, &[]),
            entry("crew", "Straw Hat Pirates", EntityKind::Faction, None, &[]),
            entry("line", "Grand Line", EntityKind::Location, None, &[]),
        ])
        .expect("lexicon");

        assert_eq!(lexicon.lookup("Luffy", &ScopeKey::default()).len(), 1);
        assert_eq!(lexicon.lookup("SHP", &ScopeKey::default()).len(), 1);
        assert_eq!(lexicon.lookup("Grand", &ScopeKey::default()).len(), 1);
    }

    #[test]
    fn fuzzy_anchor_recovers_common_typos() {
        let lexicon = Lexicon::from_entries(&[
            entry("luffy", "Monkey D. Luffy", EntityKind::Character, None, &[]),
            entry("zoro", "Roronoa Zoro", EntityKind::Character, None, &[]),
        ])
        .expect("lexicon");

        let luffy = lexicon
            .fuzzy_anchor("Luffu", &ScopeKey::default())
            .expect("luffu match");
        assert_eq!(luffy.entries[0].entity_id.0, "luffy");

        let zoro = lexicon
            .fuzzy_anchor("Zoroo", &ScopeKey::default())
            .expect("zoroo match");
        assert_eq!(zoro.entries[0].entity_id.0, "zoro");
    }

    #[test]
    fn scope_filtering_and_snapshot_roundtrip_work() {
        let lexicon = Lexicon::from_entries(&[
            entry("global", "Global Hero", EntityKind::Character, None, &[]),
            entry(
                "n1",
                "Narrative One Hero",
                EntityKind::Character,
                Some("n1"),
                &[],
            ),
        ])
        .expect("lexicon");

        let scoped = ScopeKey {
            world_id: None,
            narrative_id: Some("n1".to_owned()),
            folder_id: None,
            folder_path: None,
        };
        assert_eq!(
            lexicon
                .scan("Global Hero met Narrative One Hero.", &scoped)
                .len(),
            2
        );
        assert_eq!(
            lexicon
                .scan("Narrative One Hero.", &ScopeKey::default())
                .len(),
            0
        );

        let bytes = lexicon.to_bytes().expect("snapshot bytes");
        let restored = Lexicon::from_bytes(&bytes).expect("restore lexicon");
        assert_eq!(
            restored.lookup("Global Hero", &ScopeKey::default()).len(),
            1
        );
    }
}
