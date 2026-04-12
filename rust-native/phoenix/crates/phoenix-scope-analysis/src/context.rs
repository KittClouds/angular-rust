use std::collections::BTreeSet;
use std::sync::Arc;

use hashbrown::HashSet;
use phoenix_semantic_v2::{
    scope_storage_key, DirtyScopeRecord, DocumentArchive, DocumentRevisionRef, ErScopePatchSidecar,
    ScopeLexSidecar, SemanticEntityRecord, SemanticRelationRecord, SessionArchive,
};
use phoenix_store_native_core::ScopeRuntimeImage;
use phoenix_types::{EntityId, EntityKind, ScopeKey, SessionId};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ScopeEntityOrd(pub u32);

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopeEntityProfile {
    pub ord: ScopeEntityOrd,
    pub entity_id: EntityId,
    pub canonical_name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub base_kind: Option<EntityKind>,
    pub effective_kind: Option<EntityKind>,
    pub mention_count: usize,
    pub linked_mention_count: usize,
    #[serde(default)]
    pub document_ids: Vec<String>,
    #[serde(default)]
    pub chunk_ids: Vec<String>,
    #[serde(default)]
    pub continuity_refs: Vec<String>,
    pub continuity_score_millis: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RawArchivedRelationKey {
    pub document_id: String,
    pub source_entity_id: String,
    pub target_entity_id: String,
    pub edge_type: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScopedArchivedRelation {
    pub document_id: String,
    pub created_at: i64,
    pub relation: SemanticRelationRecord,
}

#[derive(Clone, Debug)]
pub struct ScopeAnalysisContext {
    pub runtime: ScopeRuntimeImage,
    pub scope: ScopeKey,
    pub scope_key: String,
    pub session_id: Option<SessionId>,
    pub dirty: DirtyScopeRecord,
    pub document_refs: Arc<[DocumentRevisionRef]>,
    pub entity_profiles: Arc<[ScopeEntityProfile]>,
    pub label_by_entity: Arc<FxHashMap<String, String>>,
    pub entity_ord_by_id: Arc<FxHashMap<String, ScopeEntityOrd>>,
    pub persisted_relations: Arc<[SemanticRelationRecord]>,
    pub archived_relations: Arc<[ScopedArchivedRelation]>,
    pub raw_archived_relation_keys: Arc<HashSet<RawArchivedRelationKey>>,
    pub continuity_hints: Arc<FxHashMap<(String, String), BTreeSet<String>>>,
}

impl ScopeAnalysisContext {
    pub fn from_runtime_image(
        runtime: ScopeRuntimeImage,
        session: Option<&SessionArchive>,
    ) -> Self {
        let scope = runtime
            .archives
            .first()
            .map(|archive| archive.manifest.scope.clone())
            .unwrap_or_else(|| runtime.dirty.scope.clone());
        let scope_key = runtime
            .archives
            .first()
            .map(|archive| archive.manifest.scope_key.clone())
            .unwrap_or_else(|| runtime.dirty.scope_key.clone());
        let session_id = runtime
            .archives
            .iter()
            .find_map(|archive| archive.manifest.session_id.clone())
            .or_else(|| session.map(|value| value.session_id.clone()));
        let document_refs = collect_document_refs(session, &scope_key);
        let document_hits = document_ref_counts(&document_refs);
        let entity_profiles = build_entity_profiles(
            runtime.archives.as_ref(),
            runtime.sidecars.lexical.as_ref(),
            runtime.sidecars.er.as_ref(),
            &document_hits,
        );
        let label_by_entity = entity_profiles
            .iter()
            .map(|profile| (profile.entity_id.0.clone(), profile.canonical_name.clone()))
            .collect::<FxHashMap<_, _>>();
        let entity_ord_by_id = entity_profiles
            .iter()
            .map(|profile| (profile.entity_id.0.clone(), profile.ord))
            .collect::<FxHashMap<_, _>>();
        let persisted_relations = build_persisted_relations(runtime.archives.as_ref());
        let archived_relations = build_archived_relations(runtime.archives.as_ref());
        let raw_archived_relation_keys = build_raw_archived_relation_keys(&archived_relations);
        let continuity_hints = build_continuity_hints(&persisted_relations);

        Self {
            scope,
            scope_key,
            session_id,
            dirty: runtime.dirty.clone(),
            runtime,
            document_refs,
            entity_profiles: entity_profiles.into(),
            label_by_entity: Arc::new(label_by_entity),
            entity_ord_by_id: Arc::new(entity_ord_by_id),
            persisted_relations: persisted_relations.into(),
            archived_relations: archived_relations.into(),
            raw_archived_relation_keys: Arc::new(raw_archived_relation_keys),
            continuity_hints: Arc::new(continuity_hints),
        }
    }

    pub fn archives(&self) -> &[DocumentArchive] {
        self.runtime.archives.as_ref()
    }
}

fn collect_document_refs(
    session: Option<&SessionArchive>,
    scope_key: &str,
) -> Arc<[DocumentRevisionRef]> {
    session
        .map(|value| {
            value
                .document_refs
                .iter()
                .filter(|reference| scope_storage_key(&reference.scope) == scope_key)
                .cloned()
                .collect::<Vec<_>>()
                .into()
        })
        .unwrap_or_else(|| Arc::from([]))
}

fn document_ref_counts(document_refs: &[DocumentRevisionRef]) -> FxHashMap<String, usize> {
    let mut rows = FxHashMap::<String, usize>::default();
    for reference in document_refs {
        *rows.entry(reference.document_id.clone()).or_default() += 1;
    }
    rows
}

fn build_entity_profiles(
    archives: &[DocumentArchive],
    lexical: Option<&ScopeLexSidecar>,
    er_sidecar: Option<&ErScopePatchSidecar>,
    document_hits: &FxHashMap<String, usize>,
) -> Vec<ScopeEntityProfile> {
    let mut by_entity = FxHashMap::<String, ScopeEntityProfile>::default();
    for archive in archives {
        for entity in &archive.entities {
            let entry = by_entity
                .entry(entity.entity_id.0.clone())
                .or_insert_with(|| profile_from_record(archive, entity, document_hits));
            merge_profile(entry, archive, entity);
        }
    }
    apply_lexical_aliases(&mut by_entity, lexical);
    apply_er_sidecar(&mut by_entity, er_sidecar);

    let mut rows = by_entity.into_values().collect::<Vec<_>>();
    rows.sort_by(|left, right| left.entity_id.0.cmp(&right.entity_id.0));
    for (index, row) in rows.iter_mut().enumerate() {
        row.ord = ScopeEntityOrd(index as u32);
        row.aliases.sort();
        row.aliases.dedup();
        row.document_ids.sort();
        row.document_ids.dedup();
        row.chunk_ids.sort();
        row.chunk_ids.dedup();
        row.continuity_refs.sort();
        row.continuity_refs.dedup();
        if row.effective_kind.is_none() {
            row.effective_kind = row.base_kind.clone();
        }
    }
    rows
}

fn profile_from_record(
    archive: &DocumentArchive,
    entity: &SemanticEntityRecord,
    document_hits: &FxHashMap<String, usize>,
) -> ScopeEntityProfile {
    let seen = document_hits
        .get(&archive.manifest.document_id)
        .copied()
        .unwrap_or_default();
    ScopeEntityProfile {
        ord: ScopeEntityOrd::default(),
        entity_id: entity.entity_id.clone(),
        canonical_name: entity.canonical_name.clone(),
        aliases: entity.aliases.clone(),
        base_kind: entity.kind.clone(),
        effective_kind: entity.kind.clone(),
        mention_count: entity.mention_count,
        linked_mention_count: 0,
        document_ids: vec![archive.manifest.document_id.clone()],
        chunk_ids: entity.chunk_ids.clone(),
        continuity_refs: Vec::new(),
        continuity_score_millis: ((seen as i32) * 125).min(500)
            + ((entity.mention_count.min(8) as i32) * 40),
    }
}

fn merge_profile(
    profile: &mut ScopeEntityProfile,
    archive: &DocumentArchive,
    entity: &SemanticEntityRecord,
) {
    profile.mention_count = profile.mention_count.max(entity.mention_count);
    if !profile.document_ids.contains(&archive.manifest.document_id) {
        profile
            .document_ids
            .push(archive.manifest.document_id.clone());
        profile.continuity_score_millis += 90;
    }
    profile.aliases.extend(entity.aliases.iter().cloned());
    profile.chunk_ids.extend(entity.chunk_ids.iter().cloned());
    if profile.base_kind.is_none() {
        profile.base_kind = entity.kind.clone();
    }
    if profile.effective_kind.is_none() {
        profile.effective_kind = entity.kind.clone();
    }
}

fn apply_lexical_aliases(
    by_entity: &mut FxHashMap<String, ScopeEntityProfile>,
    lexical: Option<&ScopeLexSidecar>,
) {
    let Some(lexical) = lexical else {
        return;
    };
    for alias in &lexical.alias_entries {
        for posting in &alias.postings {
            if let Some(profile) = by_entity.get_mut(&posting.entity_id) {
                profile.aliases.push(alias.normalized.clone());
                profile.continuity_refs.push(format!(
                    "lexical:{}:{}",
                    alias.normalized, posting.document_id
                ));
            }
        }
    }
}

fn apply_er_sidecar(
    by_entity: &mut FxHashMap<String, ScopeEntityProfile>,
    er_sidecar: Option<&ErScopePatchSidecar>,
) {
    let Some(er_sidecar) = er_sidecar else {
        return;
    };
    for alias in &er_sidecar.alias_additions {
        if let Some(profile) = by_entity.get_mut(&alias.entity_id.0) {
            profile.aliases.push(alias.alias_surface.clone());
            profile
                .continuity_refs
                .push(format!("er_alias:{}", alias.case_id));
        }
    }
    for override_row in &er_sidecar.type_overrides {
        if let Some(profile) = by_entity.get_mut(&override_row.entity_id.0) {
            profile.effective_kind = Some(override_row.kind.clone());
            profile
                .continuity_refs
                .push(format!("er_type:{}", override_row.case_id));
        }
    }
    for link in &er_sidecar.entity_links {
        if let Some(profile) = by_entity.get_mut(&link.entity_id.0) {
            profile.linked_mention_count += 1;
            profile
                .continuity_refs
                .push(format!("er_link:{}", link.case_id));
        }
    }
}

fn build_persisted_relations(archives: &[DocumentArchive]) -> Vec<SemanticRelationRecord> {
    let mut rows = Vec::new();
    let mut seen = HashSet::<(String, String, String, String)>::new();
    for archive in archives {
        for relation in &archive.relations {
            let key = (
                relation.source_entity_id.0.clone(),
                relation.target_entity_id.0.clone(),
                relation.edge_type.clone(),
                relation.chunk_id.clone().unwrap_or_default(),
            );
            if seen.insert(key) {
                rows.push(relation.clone());
            }
        }
    }
    rows
}

fn build_archived_relations(archives: &[DocumentArchive]) -> Vec<ScopedArchivedRelation> {
    let mut rows = Vec::new();
    for archive in archives {
        for relation in &archive.relations {
            rows.push(ScopedArchivedRelation {
                document_id: archive.manifest.document_id.clone(),
                created_at: archive.manifest.created_at,
                relation: relation.clone(),
            });
        }
    }
    rows
}

fn build_raw_archived_relation_keys(
    relations: &[ScopedArchivedRelation],
) -> HashSet<RawArchivedRelationKey> {
    let mut rows = HashSet::new();
    for row in relations {
        rows.insert(RawArchivedRelationKey {
            document_id: row.document_id.clone(),
            source_entity_id: row.relation.source_entity_id.0.clone(),
            target_entity_id: row.relation.target_entity_id.0.clone(),
            edge_type: row.relation.edge_type.clone(),
        });
    }
    rows
}

fn build_continuity_hints(
    persisted_relations: &[SemanticRelationRecord],
) -> FxHashMap<(String, String), BTreeSet<String>> {
    let mut rows = FxHashMap::<(String, String), BTreeSet<String>>::default();
    for relation in persisted_relations {
        rows.entry((
            relation.source_entity_id.0.clone(),
            relation.target_entity_id.0.clone(),
        ))
        .or_default()
        .insert(relation.edge_type.clone());
    }
    rows
}
