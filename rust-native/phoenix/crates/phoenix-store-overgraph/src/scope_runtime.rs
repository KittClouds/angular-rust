use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use phoenix_semantic_v2::{
    DirtyScopeRecord, DocumentArchive, DocumentManifest, DocumentOrd, DocumentSegmentKind,
    LexicalPostingsSegment,
};
use phoenix_store_native_core::{
    ArchiveSegmentMask, PhoenixScopeRuntimeStore, ScopeImageSpec, ScopeRuntimeImage,
    ScopeRuntimeIndices, ScopeSidecarBundle,
};

use crate::{load_segment_payload, store_query_error};
use crate::{DatabaseEngine, PhoenixOvergraphStore, StoreError};

const SCOPE_RUNTIME_CACHE_LIMIT: usize = 64;

#[derive(Debug)]
pub(crate) struct CachedScopeDocumentProjection {
    scope_key: String,
    updated_at: i64,
    document_ord_fingerprint: u64,
    archive_segments: ArchiveSegmentMask,
    manifests: Arc<[DocumentManifest]>,
    archives: Arc<[DocumentArchive]>,
    indices: Arc<ScopeRuntimeIndices>,
}

#[derive(Debug)]
pub(crate) struct CachedScopeRuntimeImage {
    scope_key: String,
    updated_at: i64,
    document_ord_fingerprint: u64,
    spec: ScopeImageSpec,
    image: Arc<ScopeRuntimeImage>,
}

impl PhoenixOvergraphStore {
    fn document_ord_fingerprint(document_ords: &[DocumentOrd]) -> u64 {
        let mut fingerprint = 0xcbf29ce484222325u64 ^ (document_ords.len() as u64);
        for document_ord in document_ords {
            fingerprint ^= document_ord.0.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            fingerprint = fingerprint.rotate_left(13).wrapping_mul(0x100000001b3);
        }
        fingerprint
    }

    fn runtime_cache_key_parts(dirty: &DirtyScopeRecord) -> (String, i64, u64) {
        (
            dirty.scope_key.clone(),
            dirty.updated_at,
            Self::document_ord_fingerprint(&dirty.document_ords),
        )
    }

    fn retain_cache_limit<T>(entries: &mut Vec<Arc<T>>) {
        if entries.len() >= SCOPE_RUNTIME_CACHE_LIMIT {
            let overflow = entries.len() + 1 - SCOPE_RUNTIME_CACHE_LIMIT;
            entries.drain(0..overflow);
        }
    }

    fn load_cached_runtime_image(
        &self,
        dirty: &DirtyScopeRecord,
        spec: ScopeImageSpec,
    ) -> Result<Option<ScopeRuntimeImage>, StoreError> {
        let (scope_key, updated_at, document_ord_fingerprint) =
            Self::runtime_cache_key_parts(dirty);
        let cache = self.scope_runtime_image_cache.lock().map_err(|_| {
            StoreError::Query("scope runtime image cache mutex poisoned".to_owned())
        })?;
        Ok(cache
            .iter()
            .rev()
            .find(|cached| {
                cached.scope_key == scope_key
                    && cached.updated_at == updated_at
                    && cached.document_ord_fingerprint == document_ord_fingerprint
                    && cached.spec == spec
            })
            .map(|cached| cached.image.as_ref().clone()))
    }

    fn remember_runtime_image(
        &self,
        dirty: &DirtyScopeRecord,
        spec: ScopeImageSpec,
        image: ScopeRuntimeImage,
    ) -> Result<ScopeRuntimeImage, StoreError> {
        let (scope_key, updated_at, document_ord_fingerprint) =
            Self::runtime_cache_key_parts(dirty);
        let cached = Arc::new(CachedScopeRuntimeImage {
            scope_key,
            updated_at,
            document_ord_fingerprint,
            spec,
            image: Arc::new(image.clone()),
        });
        let mut cache = self.scope_runtime_image_cache.lock().map_err(|_| {
            StoreError::Query("scope runtime image cache mutex poisoned".to_owned())
        })?;
        cache.retain(|entry| {
            !(entry.scope_key == cached.scope_key
                && entry.updated_at == cached.updated_at
                && entry.document_ord_fingerprint == cached.document_ord_fingerprint
                && entry.spec == cached.spec)
        });
        Self::retain_cache_limit(&mut cache);
        cache.push(cached);
        Ok(image)
    }

    fn load_cached_document_projection(
        &self,
        dirty: &DirtyScopeRecord,
        requested_mask: ArchiveSegmentMask,
    ) -> Result<Option<Arc<CachedScopeDocumentProjection>>, StoreError> {
        let (scope_key, updated_at, document_ord_fingerprint) =
            Self::runtime_cache_key_parts(dirty);
        let cache = self.scope_runtime_document_cache.lock().map_err(|_| {
            StoreError::Query("scope runtime document cache mutex poisoned".to_owned())
        })?;

        let mut best = None::<Arc<CachedScopeDocumentProjection>>;
        for cached in cache.iter().rev() {
            if cached.scope_key != scope_key
                || cached.updated_at != updated_at
                || cached.document_ord_fingerprint != document_ord_fingerprint
                || !cached.archive_segments.contains_all(requested_mask)
            {
                continue;
            }
            match best.as_ref() {
                Some(current)
                    if current.archive_segments.bit_count()
                        <= cached.archive_segments.bit_count() => {}
                _ => best = Some(Arc::clone(cached)),
            }
        }
        Ok(best)
    }

    fn remember_document_projection(
        &self,
        projection: Arc<CachedScopeDocumentProjection>,
    ) -> Result<(), StoreError> {
        let mut cache = self.scope_runtime_document_cache.lock().map_err(|_| {
            StoreError::Query("scope runtime document cache mutex poisoned".to_owned())
        })?;
        cache.retain(|entry| {
            !(entry.scope_key == projection.scope_key
                && entry.updated_at == projection.updated_at
                && entry.document_ord_fingerprint == projection.document_ord_fingerprint
                && entry.archive_segments == projection.archive_segments)
        });
        Self::retain_cache_limit(&mut cache);
        cache.push(projection);
        Ok(())
    }

    pub(crate) fn invalidate_scope_runtime_image_cache(
        &self,
        scope_key: &str,
    ) -> Result<(), StoreError> {
        let mut cache = self.scope_runtime_image_cache.lock().map_err(|_| {
            StoreError::Query("scope runtime image cache mutex poisoned".to_owned())
        })?;
        cache.retain(|entry| entry.scope_key != scope_key);
        Ok(())
    }

    pub(crate) fn invalidate_scope_runtime_document_cache(
        &self,
        scope_key: &str,
    ) -> Result<(), StoreError> {
        let mut cache = self.scope_runtime_document_cache.lock().map_err(|_| {
            StoreError::Query("scope runtime document cache mutex poisoned".to_owned())
        })?;
        cache.retain(|entry| entry.scope_key != scope_key);
        Ok(())
    }

    pub(crate) fn invalidate_scope_runtime_caches(
        &self,
        scope_key: &str,
    ) -> Result<(), StoreError> {
        self.invalidate_scope_runtime_image_cache(scope_key)?;
        self.invalidate_scope_runtime_document_cache(scope_key)
    }

    fn load_runtime_manifests_with_engine(
        &self,
        engine: &mut DatabaseEngine,
        dirty: &DirtyScopeRecord,
    ) -> Result<Vec<DocumentManifest>, StoreError> {
        if dirty.document_ords.is_empty() {
            return self.load_latest_document_manifests_with_engine(engine, Some(&dirty.scope));
        }

        let manifests = self.load_latest_document_manifests_for_ords_with_engine(
            engine,
            dirty.scope_ord,
            &dirty.document_ords,
        )?;
        if manifests.len() == dirty.document_ords.len() {
            return Ok(manifests);
        }

        let wanted = dirty
            .document_ords
            .iter()
            .map(|document_ord| document_ord.0)
            .collect::<BTreeSet<_>>();
        let mut fallback =
            self.load_latest_document_manifests_with_engine(engine, Some(&dirty.scope))?;
        fallback.retain(|manifest| wanted.contains(&manifest.document_ord.0));
        Ok(fallback)
    }

    fn load_projected_document_archive_from_manifest_with_engine(
        &self,
        engine: &mut DatabaseEngine,
        manifest: &DocumentManifest,
        mask: ArchiveSegmentMask,
    ) -> Result<DocumentArchive, StoreError> {
        let mut archive = DocumentArchive {
            manifest: manifest.clone(),
            ..Default::default()
        };
        let mut lexical = None::<LexicalPostingsSegment>;

        for segment_ref in &manifest.segment_refs {
            if !mask.contains(segment_ref.kind) {
                continue;
            }

            let key = crate::segment_key(
                manifest.scope_ord,
                manifest.document_ord,
                manifest.revision,
                segment_ref.kind,
                segment_ref.ordinal,
            );
            let Some(node) = engine
                .get_node_by_key(crate::TYPE_DOCUMENT_SEGMENT, &key)
                .map_err(store_query_error)?
            else {
                return Err(StoreError::Query(format!(
                    "missing projected segment {} for {}@{}",
                    key, manifest.document_id, manifest.revision
                )));
            };
            let payload = load_segment_payload(&node)?;
            match segment_ref.kind {
                DocumentSegmentKind::StringArena => {
                    archive.tokens = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::SentenceTable => {
                    archive.sentences = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::MentionTable => {
                    archive.mentions = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::ResolverLinkTable => {
                    archive.resolver_links = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::ResolvedMentionTable => {
                    archive.resolved_mentions = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::AliasConfirmationTable => {
                    archive.alias_confirmations = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::CorefClusterTable => {
                    archive.coref_clusters = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::CausalSubstrateTable => {
                    archive.causal_substrate = Some(crate::decode_segment_payload(&payload)?);
                }
                DocumentSegmentKind::TemporalSubstrateTable => {
                    archive.temporal_substrate = Some(crate::decode_segment_payload(&payload)?);
                }
                DocumentSegmentKind::EventIdentitySubstrateTable => {
                    archive.event_identity_substrate =
                        Some(crate::decode_segment_payload(&payload)?);
                }
                DocumentSegmentKind::ChunkTable => {
                    archive.chunks = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::EntityTable => {
                    archive.entities = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::RelationTable => {
                    archive.relations = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::EvidenceTable => {
                    archive.evidence_spans = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::LexicalPostings => {
                    lexical = Some(crate::decode_segment_payload(&payload)?);
                }
                DocumentSegmentKind::NarrativeHitTable => {
                    archive.relation_candidates = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::GraphMutation => {
                    archive.graph_batch = crate::decode_segment_payload(&payload)?;
                }
                DocumentSegmentKind::StructureRelations => {
                    archive.structure = Some(crate::decode_segment_payload(&payload)?);
                }
                DocumentSegmentKind::BoundaryTable => {}
            }
        }

        if let Some(lexical) = lexical {
            archive.indexed_spans = lexical.spans;
        }

        Ok(archive)
    }

    fn load_projected_document_archives_with_engine(
        &self,
        engine: &mut DatabaseEngine,
        manifests: &[DocumentManifest],
        mask: ArchiveSegmentMask,
    ) -> Result<Vec<DocumentArchive>, StoreError> {
        manifests
            .iter()
            .map(|manifest| {
                self.load_projected_document_archive_from_manifest_with_engine(
                    engine, manifest, mask,
                )
            })
            .collect()
    }

    fn load_scope_sidecar_bundle_with_engine(
        &self,
        engine: &mut DatabaseEngine,
        dirty: &DirtyScopeRecord,
        spec: ScopeImageSpec,
    ) -> Result<ScopeSidecarBundle, StoreError> {
        let mut bundle = ScopeSidecarBundle::default();
        if spec.sidecars.includes_lexical() {
            bundle.lexical = self.load_native_scope_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_er() {
            bundle.er = self.load_native_er_patch_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_relation() {
            bundle.relation =
                self.load_native_relation_patch_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_memory() {
            bundle.memory =
                self.load_native_memory_patch_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_event_identity() {
            bundle.event_identity =
                self.load_native_event_identity_patch_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_state_schema() {
            bundle.state_schema =
                self.load_native_state_schema_patch_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_causal() {
            bundle.causal =
                self.load_native_causal_patch_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_temporal() {
            bundle.temporal =
                self.load_native_temporal_patch_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_graph() {
            bundle.graph =
                self.load_native_graph_patch_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_semantic_graph() {
            bundle.semantic_graph =
                self.load_native_semantic_graph_patch_sidecar_with_engine(engine, &dirty.scope)?;
        }
        if spec.sidecars.includes_relation_seed() {
            bundle.relation_seed =
                self.load_native_relation_mention_seed_sidecar_with_engine(engine, &dirty.scope)?;
        }
        Ok(bundle)
    }

    fn build_scope_runtime_indices(&self, manifests: &[DocumentManifest]) -> ScopeRuntimeIndices {
        let document_ids = manifests
            .iter()
            .map(|manifest| manifest.document_id.clone())
            .collect::<Vec<_>>()
            .into();
        let document_created_at = manifests
            .iter()
            .map(|manifest| (manifest.document_id.clone(), manifest.created_at))
            .collect::<BTreeMap<_, _>>();
        ScopeRuntimeIndices {
            document_ids,
            document_created_at,
        }
    }

    fn load_or_build_document_projection_with_engine(
        &self,
        engine: &mut DatabaseEngine,
        dirty: &DirtyScopeRecord,
        mask: ArchiveSegmentMask,
    ) -> Result<Arc<CachedScopeDocumentProjection>, StoreError> {
        if let Some(cached) = self.load_cached_document_projection(dirty, mask)? {
            return Ok(cached);
        }

        let manifests = self.load_runtime_manifests_with_engine(engine, dirty)?;
        let archives =
            self.load_projected_document_archives_with_engine(engine, &manifests, mask)?;
        let indices = self.build_scope_runtime_indices(&manifests);
        let (scope_key, updated_at, document_ord_fingerprint) =
            Self::runtime_cache_key_parts(dirty);
        let projection = Arc::new(CachedScopeDocumentProjection {
            scope_key,
            updated_at,
            document_ord_fingerprint,
            archive_segments: mask,
            manifests: Arc::from(manifests),
            archives: Arc::from(archives),
            indices: Arc::new(indices),
        });
        self.remember_document_projection(Arc::clone(&projection))?;
        Ok(projection)
    }
}

impl PhoenixScopeRuntimeStore for PhoenixOvergraphStore {
    fn load_scope_runtime_image(
        &self,
        dirty: &DirtyScopeRecord,
        spec: ScopeImageSpec,
    ) -> Result<ScopeRuntimeImage, StoreError> {
        if let Some(cached) = self.load_cached_runtime_image(dirty, spec)? {
            return Ok(cached);
        }

        let documents = self.with_engine(|engine| {
            self.load_or_build_document_projection_with_engine(engine, dirty, spec.archive_segments)
        })?;
        let sidecars = self.with_engine(|engine| {
            self.load_scope_sidecar_bundle_with_engine(engine, dirty, spec)
        })?;

        let image = ScopeRuntimeImage {
            dirty: dirty.clone(),
            manifests: Arc::clone(&documents.manifests),
            archives: Arc::clone(&documents.archives),
            sidecars: Arc::new(sidecars),
            indices: Arc::clone(&documents.indices),
            archive_segments: documents.archive_segments,
            sidecar_mask: spec.sidecars,
        };
        self.remember_runtime_image(dirty, spec, image)
    }
}
