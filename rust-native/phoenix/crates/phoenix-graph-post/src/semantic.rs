use std::env;
use std::io;
use std::path::PathBuf;
use std::str;

use hashbrown::{HashMap, HashSet};
use memchr::memrchr;
use phoenix_chunker::{build_chunks, ChunkerConfig};
use phoenix_embed::{
    default_embedding_model_root, OrtTextEmbedConfig, OrtTextEmbedError, OrtTextEmbedder,
    TextEmbeddingProfile,
};
use phoenix_store_native_core::{
    NativeSemanticNodeVectorRecord, PhoenixSemanticIndexStore, StoreError, SEMANTIC_VECTOR_DIM,
};
use phoenix_types::ScopeKey;
use serde::{Deserialize, Serialize};
use thiserror::Error;

const CHUNK_KIND: &str = "chunk";
pub type SnowflakeOrtEmbedder = OrtTextEmbedder;
pub use phoenix_embed::{default_ort_dylib_path, workspace_root};

#[derive(Debug, Error)]
pub enum SemanticNeighborError {
    #[error("document bytes were not valid utf-8: {0}")]
    Utf8(#[from] str::Utf8Error),
    #[error("failed to read embedding assets: {0}")]
    Io(#[from] io::Error),
    #[error("semantic store error: {0}")]
    Store(#[from] StoreError),
    #[error("embedding model error: {0}")]
    Model(String),
    #[error("embedding count mismatch: expected {expected}, got {actual}")]
    EmbeddingCountMismatch { expected: usize, actual: usize },
    #[error("embedding dimension mismatch for row {row}: expected {expected}, got {actual}")]
    EmbeddingDimensionMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    #[error(transparent)]
    Embed(#[from] OrtTextEmbedError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticEmbedConfig {
    pub model_root: PathBuf,
    pub batch_size: usize,
    pub max_length: usize,
    pub profile: TextEmbeddingProfile,
}

impl Default for SemanticEmbedConfig {
    fn default() -> Self {
        Self {
            model_root: default_embedding_model_root(),
            batch_size: 12,
            max_length: 512,
            profile: TextEmbeddingProfile::Native384,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SemanticChunkConfig {
    pub chunk_size: usize,
    pub overlap: usize,
    pub query_count: usize,
    pub neighbor_limit: usize,
    pub oversample: usize,
}

impl Default for SemanticChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 680,
            overlap: 120,
            query_count: 8,
            neighbor_limit: 3,
            oversample: 16,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticChunkNode {
    pub node_id: String,
    pub chunk_index: usize,
    pub start: usize,
    pub end: usize,
    pub byte_len: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticChunkNeighbor {
    pub source_node_id: String,
    pub source_chunk_index: usize,
    pub source_start: usize,
    pub source_end: usize,
    pub target_node_id: String,
    pub target_chunk_index: Option<usize>,
    pub target_start: Option<usize>,
    pub target_end: Option<usize>,
    pub distance: f64,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

pub fn first_half_markdown(bytes: &[u8]) -> Result<&str, SemanticNeighborError> {
    let midpoint = bytes.len() / 2;
    let boundary = split_boundary_before(bytes, midpoint);
    str::from_utf8(&bytes[..boundary]).map_err(Into::into)
}

pub fn split_boundary_before(bytes: &[u8], midpoint: usize) -> usize {
    let midpoint = midpoint.min(bytes.len());
    let prefix = &bytes[..midpoint];
    let mut cursor = prefix.len();
    while let Some(hit) = memrchr(b'\n', &prefix[..cursor]) {
        if hit > 0 && prefix[hit - 1] == b'\n' {
            return hit - 1;
        }
        cursor = hit;
    }
    memrchr(b'\n', prefix).unwrap_or(midpoint)
}

pub fn build_chunk_nodes(
    document_id: &str,
    text: &str,
    config: &SemanticChunkConfig,
) -> Vec<SemanticChunkNode> {
    let ranges = build_chunks(
        text,
        &ChunkerConfig {
            chunk_size: config.chunk_size.max(64),
            overlap: config.overlap.min(config.chunk_size.saturating_sub(1)),
        },
    );
    let mut nodes = Vec::with_capacity(ranges.len());
    for (chunk_index, range) in ranges.into_iter().enumerate() {
        if range.end <= range.start {
            continue;
        }
        if text[range.start..range.end].trim().is_empty() {
            continue;
        }
        nodes.push(SemanticChunkNode {
            node_id: format!("chunk::{document_id}::{chunk_index:04}"),
            chunk_index,
            start: range.start,
            end: range.end,
            byte_len: range.end - range.start,
        });
    }
    nodes
}

pub fn embed_chunks(
    model: &SnowflakeOrtEmbedder,
    text: &str,
    chunks: &[SemanticChunkNode],
) -> Result<Vec<Vec<f32>>, SemanticNeighborError> {
    let mut slices = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        slices.push(text[chunk.start..chunk.end].trim());
    }
    Ok(model.embed_texts(&slices)?)
}

pub fn build_chunk_node_records(
    scope: &ScopeKey,
    document_id: &str,
    narrative_id: Option<&str>,
    chunks: &[SemanticChunkNode],
    embeddings: &[Vec<f32>],
) -> Result<Vec<NativeSemanticNodeVectorRecord>, SemanticNeighborError> {
    if chunks.len() != embeddings.len() {
        return Err(SemanticNeighborError::EmbeddingCountMismatch {
            expected: chunks.len(),
            actual: embeddings.len(),
        });
    }
    let mut rows = Vec::with_capacity(chunks.len());
    for (row, (chunk, embedding)) in chunks.iter().zip(embeddings.iter()).enumerate() {
        if embedding.len() != SEMANTIC_VECTOR_DIM {
            return Err(SemanticNeighborError::EmbeddingDimensionMismatch {
                row,
                expected: SEMANTIC_VECTOR_DIM,
                actual: embedding.len(),
            });
        }
        rows.push(NativeSemanticNodeVectorRecord {
            scope: scope.clone(),
            node_id: chunk.node_id.clone(),
            node_kind: CHUNK_KIND.to_owned(),
            document_id: Some(document_id.to_owned()),
            narrative_id: narrative_id.map(str::to_owned),
            folder_id: None,
            values: embedding.clone(),
            evidence_refs: vec![format!(
                "document:{document_id}#bytes:{}-{}",
                chunk.start, chunk.end
            )],
            updated_at: 1,
        });
    }
    Ok(rows)
}

pub fn evenly_spaced_query_indices(total: usize, desired: usize) -> Vec<usize> {
    if total == 0 || desired == 0 {
        return Vec::new();
    }
    if desired >= total {
        return (0..total).collect();
    }
    let mut indices = Vec::with_capacity(desired + 1);
    let mut seen = HashSet::with_capacity(desired + 1);
    for slot in 0..desired {
        let index = slot * total / desired;
        if seen.insert(index) {
            indices.push(index);
        }
    }
    if let Some(last) = total.checked_sub(1) {
        if seen.insert(last) {
            indices.push(last);
        }
    }
    indices.sort_unstable();
    indices
}

pub fn collect_chunk_neighbors<S: PhoenixSemanticIndexStore>(
    store: &S,
    scope: &ScopeKey,
    chunks: &[SemanticChunkNode],
    embeddings: &[Vec<f32>],
    config: &SemanticChunkConfig,
) -> Result<Vec<SemanticChunkNeighbor>, SemanticNeighborError> {
    let mut node_index = HashMap::with_capacity(chunks.len());
    for chunk in chunks {
        node_index.insert(chunk.node_id.as_str(), chunk);
    }
    let mut seen = HashSet::with_capacity(chunks.len() * config.neighbor_limit.max(1));
    let mut hits = Vec::new();
    for query_index in evenly_spaced_query_indices(chunks.len(), config.query_count) {
        let source = &chunks[query_index];
        for neighbor in store.query_semantic_node_neighbors(
            &embeddings[query_index],
            scope,
            CHUNK_KIND,
            Some(&source.node_id),
            config.neighbor_limit.max(1),
            config.oversample.max(config.neighbor_limit.max(1)),
        )? {
            if !seen.insert((source.node_id.clone(), neighbor.node_id.clone())) {
                continue;
            }
            let target = node_index.get(neighbor.node_id.as_str()).copied();
            hits.push(SemanticChunkNeighbor {
                source_node_id: source.node_id.clone(),
                source_chunk_index: source.chunk_index,
                source_start: source.start,
                source_end: source.end,
                target_node_id: neighbor.node_id,
                target_chunk_index: target.map(|value| value.chunk_index),
                target_start: target.map(|value| value.start),
                target_end: target.map(|value| value.end),
                distance: neighbor.distance,
                evidence_refs: neighbor.evidence_refs,
            });
        }
    }
    Ok(hits)
}

pub fn semantic_embedder(
    config: &SemanticEmbedConfig,
) -> Result<SnowflakeOrtEmbedder, SemanticNeighborError> {
    let _ = ensure_ort_dylib_path();
    Ok(SnowflakeOrtEmbedder::load(&OrtTextEmbedConfig {
        model_root: config.model_root.clone(),
        batch_size: config.batch_size,
        max_length: config.max_length,
        profile: config.profile,
        prefix_passage: true,
    })?)
}

pub fn ensure_ort_dylib_path() -> Option<PathBuf> {
    if let Some(existing) = env::var_os("ORT_DYLIB_PATH") {
        return Some(PathBuf::from(existing));
    }
    let root = workspace_root();
    let path = default_ort_dylib_path(&root)?;
    env::set_var("ORT_DYLIB_PATH", &path);
    Some(path)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use phoenix_store_overgraph::PhoenixOvergraphStore;

    use super::*;

    #[test]
    fn split_boundary_prefers_blank_line_before_midpoint() {
        let text = "alpha\n\nbeta beta beta\n\ngamma gamma gamma\n";
        let midpoint = text.find("gamma").expect("gamma");
        let boundary = split_boundary_before(text.as_bytes(), midpoint);
        assert_eq!(&text[..boundary], "alpha\n\nbeta beta beta");
    }

    #[test]
    fn build_chunk_node_records_validate_dimensions() {
        let scope = ScopeKey::default();
        let chunks = vec![SemanticChunkNode {
            node_id: "chunk::doc::0000".to_owned(),
            chunk_index: 0,
            start: 0,
            end: 12,
            byte_len: 12,
        }];
        let error = build_chunk_node_records(&scope, "doc", None, &chunks, &[vec![0.0; 8]])
            .expect_err("dimension mismatch");
        assert!(matches!(
            error,
            SemanticNeighborError::EmbeddingDimensionMismatch { .. }
        ));
    }

    #[test]
    fn chunk_neighbor_roundtrip_queries_from_overgraph() {
        let path = std::env::temp_dir().join(format!(
            "phoenix-graph-semantic-{}-{}",
            std::process::id(),
            17_001_u64
        ));
        let _ = fs::remove_dir_all(&path);
        let store = PhoenixOvergraphStore::open(&path).expect("open store");
        let scope = ScopeKey::default();
        let chunks = vec![
            SemanticChunkNode {
                node_id: "chunk::doc::0000".to_owned(),
                chunk_index: 0,
                start: 0,
                end: 10,
                byte_len: 10,
            },
            SemanticChunkNode {
                node_id: "chunk::doc::0001".to_owned(),
                chunk_index: 1,
                start: 10,
                end: 20,
                byte_len: 10,
            },
            SemanticChunkNode {
                node_id: "chunk::doc::0002".to_owned(),
                chunk_index: 2,
                start: 20,
                end: 30,
                byte_len: 10,
            },
        ];
        let embeddings = vec![
            synthetic_embedding(0),
            synthetic_embedding(1),
            synthetic_embedding(2),
        ];
        let rows =
            build_chunk_node_records(&scope, "doc", None, &chunks, &embeddings).expect("rows");
        store
            .upsert_semantic_node_vectors_native(&rows)
            .expect("upsert vectors");

        let hits = collect_chunk_neighbors(
            &store,
            &scope,
            &chunks,
            &embeddings,
            &SemanticChunkConfig {
                query_count: 2,
                neighbor_limit: 1,
                oversample: 4,
                ..Default::default()
            },
        )
        .expect("collect neighbors");

        assert!(!hits.is_empty(), "expected at least one semantic hit");
        assert!(hits
            .iter()
            .all(|hit| hit.source_node_id != hit.target_node_id));
    }

    fn synthetic_embedding(seed: usize) -> Vec<f32> {
        let mut values = vec![0.0; SEMANTIC_VECTOR_DIM];
        values[seed % SEMANTIC_VECTOR_DIM] = 1.0;
        values[(seed + 11) % SEMANTIC_VECTOR_DIM] = 0.5;
        values
    }
}
