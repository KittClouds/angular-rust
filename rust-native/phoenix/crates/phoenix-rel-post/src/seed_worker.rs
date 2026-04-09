use std::collections::{BTreeMap, BTreeSet};

use phoenix_chunker::{build_chunks, split_sentence_ranges, ChunkerConfig};
use phoenix_semantic_v2::{
    scope_storage_key, DirtyScopeRecord, DocumentArchive, ErScopePatchSidecar,
    RelationMentionSeedRecord, RelationMentionSeedScopeSidecar, ScopeLexSidecar, SessionArchive,
};
use phoenix_store_native_core::{
    PhoenixArchiveStoreV2, PhoenixErPatchStore, PhoenixRelationMentionSeedStore, StoreError,
};
use phoenix_types::{EntityKind, TextRange};
use rustc_hash::FxHashMap;
use serde::Serialize;

use crate::{
    derive_relation_entity_profiles, GlirelWorkerError, RelationEntityProfile,
    RelationMentionSeeder,
};

#[derive(Clone, Debug)]
pub struct RelationSeedConfig {
    pub threshold: f32,
    pub chunk_size: usize,
    pub overlap: usize,
    pub max_chunks_per_archive: usize,
    pub max_windows_per_chunk: usize,
    pub max_microchunks_per_archive: usize,
}

impl Default for RelationSeedConfig {
    fn default() -> Self {
        Self {
            threshold: 0.55,
            chunk_size: 320,
            overlap: 64,
            max_chunks_per_archive: 8,
            max_windows_per_chunk: 4,
            max_microchunks_per_archive: 24,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RelationSeedReport {
    pub scope_key: String,
    pub archive_count: usize,
    pub candidate_chunk_count: usize,
    pub microchunk_count: usize,
    pub seed_count: usize,
    pub generation: u64,
}

#[derive(Clone, Debug)]
struct SeedMicrochunk {
    input_id: String,
    document_id: String,
    revision: u64,
    chunk_id: String,
    text: String,
    doc_start: usize,
    sentence_index: Option<usize>,
    evidence: Vec<String>,
}

pub fn build_relation_mention_seed_sidecar(
    archives: &[DocumentArchive],
    session: Option<&SessionArchive>,
    dirty: Option<&DirtyScopeRecord>,
    sidecar: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    existing: Option<&RelationMentionSeedScopeSidecar>,
    seeder: &RelationMentionSeeder,
    config: &RelationSeedConfig,
    created_at: i64,
) -> Result<(RelationMentionSeedScopeSidecar, RelationSeedReport), GlirelWorkerError> {
    let scope = archives
        .first()
        .map(|archive| archive.manifest.scope.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope.clone()))
        .or_else(|| sidecar.as_ref().map(|value| value.scope.clone()))
        .unwrap_or_default();
    let scope_key = archives
        .first()
        .map(|archive| archive.manifest.scope_key.clone())
        .or_else(|| dirty.as_ref().map(|record| record.scope_key.clone()))
        .or_else(|| sidecar.as_ref().map(|value| value.scope_key.clone()))
        .unwrap_or_else(|| scope_storage_key(&scope));
    let scope_ord = archives
        .first()
        .map(|archive| archive.manifest.scope_ord)
        .or_else(|| dirty.as_ref().map(|record| record.scope_ord))
        .or_else(|| sidecar.as_ref().and_then(|value| value.scope_ord));
    let session_id = archives
        .iter()
        .find_map(|archive| archive.manifest.session_id.clone())
        .or_else(|| session.map(|value| value.session_id.clone()));

    let profiles = derive_relation_entity_profiles(archives, sidecar, er_sidecar, session);
    let profile_surface_index = build_profile_surface_index(&profiles);

    let mut report = RelationSeedReport {
        scope_key: scope_key.clone(),
        archive_count: archives.len(),
        generation: existing.map(|value| value.generation + 1).unwrap_or(1),
        ..Default::default()
    };
    let mut seeds = Vec::new();
    for archive in archives {
        let candidate_chunks =
            select_candidate_chunks(archive, &profiles, config.max_chunks_per_archive);
        report.candidate_chunk_count += candidate_chunks.len();
        let microchunks = build_microchunks(archive, &candidate_chunks, config);
        report.microchunk_count += microchunks.len();
        if microchunks.is_empty() {
            continue;
        }
        let inputs = microchunks
            .iter()
            .map(|chunk| (chunk.input_id.clone(), chunk.text.clone()))
            .collect::<Vec<_>>();
        let microchunk_by_id = microchunks
            .iter()
            .map(|chunk| (chunk.input_id.as_str(), chunk))
            .collect::<FxHashMap<_, _>>();
        for seeded in seeder
            .seed_chunk_mentions(&inputs)
            .map_err(GlirelWorkerError::Seeder)?
        {
            let Some(microchunk) = microchunk_by_id.get(seeded.input_id.as_str()) else {
                continue;
            };
            let normalized = normalize_surface(&seeded.surface);
            let Some(candidates) = profile_surface_index.get(normalized.as_str()) else {
                continue;
            };
            let Some(profile) = choose_seed_profile(candidates, &seeded.label) else {
                continue;
            };
            if !is_seed_surface_allowed(&seeded.surface, profile.kind.as_ref()) {
                continue;
            }
            seeds.push(RelationMentionSeedRecord {
                document_id: microchunk.document_id.clone(),
                revision: microchunk.revision,
                chunk_id: microchunk.chunk_id.clone(),
                entity_id: profile.entity_id.clone(),
                surface: seeded.surface.clone(),
                normalized,
                kind: profile.kind.clone(),
                range: TextRange {
                    start: (microchunk.doc_start + seeded.span_start) as u32,
                    end: (microchunk.doc_start + seeded.span_end) as u32,
                },
                sentence_index: microchunk.sentence_index,
                confidence_millis: (seeded.probability.clamp(0.0, 1.0) * 1000.0).round() as u32,
                seed_label: seeded.label,
                evidence: microchunk.evidence.clone(),
                created_at,
            });
        }
    }
    dedupe_seed_records(&mut seeds);
    report.seed_count = seeds.len();

    Ok((
        RelationMentionSeedScopeSidecar {
            scope,
            scope_key,
            scope_ord,
            session_id,
            updated_at: created_at,
            generation: report.generation,
            seeds,
        },
        report,
    ))
}

pub fn build_relation_mention_seed_sidecar_from_store<S>(
    store: &S,
    dirty: &DirtyScopeRecord,
    session: Option<&SessionArchive>,
    seeder: &RelationMentionSeeder,
    config: &RelationSeedConfig,
    created_at: i64,
) -> Result<(RelationMentionSeedScopeSidecar, RelationSeedReport), GlirelWorkerError>
where
    S: PhoenixArchiveStoreV2 + PhoenixErPatchStore + PhoenixRelationMentionSeedStore,
{
    let archives = store.load_latest_document_archives(Some(&dirty.scope))?;
    let sidecar = store.load_scope_sidecar(&dirty.scope)?;
    let er_sidecar = store.load_er_patch_sidecar(&dirty.scope)?;
    let existing = store.load_relation_mention_seed_sidecar(&dirty.scope)?;
    build_relation_mention_seed_sidecar(
        &archives,
        session,
        Some(dirty),
        sidecar.as_ref(),
        er_sidecar.as_ref(),
        existing.as_ref(),
        seeder,
        config,
        created_at,
    )
}

pub fn persist_relation_mention_seed_sidecar<S>(
    store: &S,
    sidecar: &RelationMentionSeedScopeSidecar,
) -> Result<(), StoreError>
where
    S: PhoenixRelationMentionSeedStore,
{
    store.persist_relation_mention_seed_sidecar(sidecar)
}

fn select_candidate_chunks<'a>(
    archive: &'a DocumentArchive,
    profiles: &[RelationEntityProfile],
    max_chunks: usize,
) -> Vec<&'a phoenix_semantic_v2::ChunkRecord> {
    let profile_by_entity = profiles
        .iter()
        .map(|profile| (profile.entity_id.0.as_str(), profile))
        .collect::<FxHashMap<_, _>>();
    let chunk_by_id = archive
        .chunks
        .iter()
        .map(|chunk| (chunk.chunk_id.0.as_str(), chunk))
        .collect::<FxHashMap<_, _>>();
    let mut selected = Vec::new();
    let mut seen = BTreeSet::new();

    for relation in &archive.relations {
        if let Some(chunk_id) = relation.chunk_id.as_deref() {
            if seen.insert(chunk_id.to_owned()) {
                if let Some(chunk) = chunk_by_id.get(chunk_id) {
                    selected.push(*chunk);
                    if selected.len() >= max_chunks {
                        return selected;
                    }
                }
            }
        }
        for entity_id in [&relation.source_entity_id.0, &relation.target_entity_id.0] {
            let Some(profile) = profile_by_entity.get(entity_id.as_str()) else {
                continue;
            };
            for chunk_id in &profile.chunk_ids {
                if seen.insert(chunk_id.clone()) {
                    if let Some(chunk) = chunk_by_id.get(chunk_id.as_str()) {
                        selected.push(*chunk);
                        if selected.len() >= max_chunks {
                            return selected;
                        }
                    }
                }
            }
        }
    }

    if selected.is_empty() && !archive.relations.is_empty() {
        for chunk in archive.chunks.iter().take(max_chunks) {
            selected.push(chunk);
        }
    }
    selected
}

fn build_microchunks(
    archive: &DocumentArchive,
    chunks: &[&phoenix_semantic_v2::ChunkRecord],
    config: &RelationSeedConfig,
) -> Vec<SeedMicrochunk> {
    let mut microchunks = Vec::new();
    for chunk in chunks {
        let mut chunk_windows = if split_sentence_ranges(&chunk.text).is_empty() {
            vec![phoenix_chunker::Chunk {
                start: 0,
                end: chunk.text.len(),
            }]
        } else {
            build_chunks(
                &chunk.text,
                &ChunkerConfig {
                    chunk_size: config.chunk_size,
                    overlap: config.overlap,
                },
            )
        };
        if chunk_windows.is_empty() {
            chunk_windows.push(phoenix_chunker::Chunk {
                start: 0,
                end: chunk.text.len(),
            });
        }
        for (window_index, window) in chunk_windows
            .into_iter()
            .take(config.max_windows_per_chunk)
            .enumerate()
        {
            let text = chunk.text[window.start..window.end].trim().to_owned();
            if text.is_empty() {
                continue;
            }
            microchunks.push(SeedMicrochunk {
                input_id: format!(
                    "{}::{}::{}::{}",
                    archive.manifest.document_id,
                    archive.manifest.revision,
                    chunk.chunk_id.0,
                    window_index
                ),
                document_id: archive.manifest.document_id.clone(),
                revision: archive.manifest.revision,
                chunk_id: chunk.chunk_id.0.clone(),
                text,
                doc_start: chunk.range.start as usize + window.start,
                sentence_index: None,
                evidence: vec![
                    "seed_source:chunker_microchunk".to_owned(),
                    format!("seed_chunk:{}", chunk.chunk_id.0),
                ],
            });
            if microchunks.len() >= config.max_microchunks_per_archive {
                return microchunks;
            }
        }
    }
    microchunks
}

fn build_profile_surface_index<'a>(
    profiles: &'a [RelationEntityProfile],
) -> FxHashMap<String, Vec<&'a RelationEntityProfile>> {
    let mut by_surface = FxHashMap::<String, Vec<&'a RelationEntityProfile>>::default();
    for profile in profiles {
        let canonical = normalize_surface(&profile.canonical_name);
        if !canonical.is_empty() {
            by_surface.entry(canonical).or_default().push(profile);
        }
        for alias in &profile.aliases {
            let normalized = normalize_surface(alias);
            if normalized.is_empty() {
                continue;
            }
            by_surface.entry(normalized).or_default().push(profile);
        }
    }
    by_surface
}

fn choose_seed_profile<'a>(
    profiles: &[&'a RelationEntityProfile],
    label: &str,
) -> Option<&'a RelationEntityProfile> {
    let mut candidates = profiles
        .iter()
        .copied()
        .filter(|profile| profile_kind_matches_seed_label(profile.kind.as_ref(), label))
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        candidates = profiles.to_vec();
    }
    candidates.into_iter().max_by(|left, right| {
        left.continuity_score_millis
            .cmp(&right.continuity_score_millis)
            .then_with(|| left.mention_count.cmp(&right.mention_count))
            .then_with(|| left.canonical_name.len().cmp(&right.canonical_name.len()))
    })
}

fn profile_kind_matches_seed_label(kind: Option<&EntityKind>, label: &str) -> bool {
    match label {
        "person" => matches!(kind, Some(EntityKind::Character | EntityKind::Npc)),
        "organization" => matches!(kind, Some(EntityKind::Organization | EntityKind::Faction)),
        "location" => matches!(kind, Some(EntityKind::Location)),
        _ => false,
    }
}

fn is_seed_surface_allowed(surface: &str, kind: Option<&EntityKind>) -> bool {
    let normalized = normalize_surface(surface);
    if normalized.is_empty() {
        return false;
    }
    if matches!(
        normalized.as_str(),
        "hero"
            | "heroes"
            | "villain"
            | "villains"
            | "monster"
            | "monsters"
            | "security"
            | "driving"
            | "chapter"
            | "guard"
            | "guards"
            | "doctor"
            | "chief"
            | "teacher"
            | "father"
            | "mother"
    ) {
        return false;
    }
    match kind {
        Some(EntityKind::Location) => surface.split_whitespace().count() >= 2,
        _ => surface.chars().any(|value| value.is_uppercase()) || surface.len() >= 4,
    }
}

fn normalize_surface(surface: &str) -> String {
    surface
        .trim()
        .to_ascii_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn dedupe_seed_records(seeds: &mut Vec<RelationMentionSeedRecord>) {
    let mut best = BTreeMap::<(String, String, u32, u32), RelationMentionSeedRecord>::new();
    for seed in seeds.drain(..) {
        let key = (
            seed.document_id.clone(),
            seed.entity_id.0.clone(),
            seed.range.start,
            seed.range.end,
        );
        match best.get(&key) {
            Some(current) if current.confidence_millis >= seed.confidence_millis => {}
            _ => {
                best.insert(key, seed);
            }
        }
    }
    *seeds = best.into_values().collect::<Vec<_>>();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_microchunks_prefers_small_sentence_windows() {
        let archive = DocumentArchive {
            manifest: Default::default(),
            chunks: vec![phoenix_semantic_v2::ChunkRecord {
                chunk_id: phoenix_semantic_v2::ChunkId("chunk-1".to_owned()),
                range: TextRange { start: 0, end: 84 },
                chapter_id: 0,
                boundary_label: None,
                text: "Alice works for Dynamis. Dynamis is in New Rome. Ryan later meets Len."
                    .to_owned(),
            }],
            ..Default::default()
        };
        let chunks = archive.chunks.iter().collect::<Vec<_>>();
        let microchunks = build_microchunks(&archive, &chunks, &RelationSeedConfig::default());
        assert!(!microchunks.is_empty());
        assert!(microchunks.iter().all(|chunk| chunk.text.len() <= 320));
    }

    #[test]
    fn choose_seed_profile_prefers_kind_match() {
        let left = RelationEntityProfile {
            entity_id: phoenix_types::EntityId("e1".to_owned()),
            canonical_name: "Dynamis".to_owned(),
            kind: Some(EntityKind::Organization),
            continuity_score_millis: 100,
            ..Default::default()
        };
        let right = RelationEntityProfile {
            entity_id: phoenix_types::EntityId("e2".to_owned()),
            canonical_name: "Dynamis".to_owned(),
            kind: Some(EntityKind::Location),
            continuity_score_millis: 500,
            ..Default::default()
        };
        let chosen = choose_seed_profile(&[&left, &right], "organization").unwrap();
        assert_eq!(chosen.entity_id.0, "e1");
    }
}
