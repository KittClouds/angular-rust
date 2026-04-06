use phoenix_qgram::{QgramConfig, QgramIndex};
use phoenix_store_cozo::{CompactRowView, PhoenixCozoStore, StoreError};
use phoenix_types::{
    DocumentId, ImplicitMatchHit, IndexedSpan, IndexedTextField, LexicalField, LexicalSearchResult,
    NoteId, ScopeKey,
};
use rustc_hash::FxHashMap;

pub use phoenix_qgram::{parse_query, Clause, ClauseType, SearchConfig};
pub use phoenix_qgram::{
    CatalogSpan, CorpusStats, PackedGram, PostingSet, SpanCatalog, SpanOrdinal,
};

pub type LexConfig = QgramConfig;

#[derive(Clone, Debug)]
pub struct LexIndex {
    engine: QgramIndex,
    config: LexConfig,
}

impl Default for LexIndex {
    fn default() -> Self {
        let config = LexConfig::default();
        Self {
            engine: QgramIndex::build(&[], config.clone()),
            config,
        }
    }
}

impl LexIndex {
    pub fn build(spans: &[IndexedSpan], config: LexConfig) -> Self {
        Self {
            engine: QgramIndex::build(spans, config.clone()),
            config,
        }
    }

    pub fn from_store(store: &PhoenixCozoStore, config: LexConfig) -> Result<Self, StoreError> {
        let spans = indexed_spans_from_store(store)?;
        Ok(Self::build(&spans, config))
    }

    pub fn rebuild_from_spans(&mut self, spans: &[IndexedSpan]) {
        self.engine.rebuild_from_catalog(spans);
    }

    pub fn rebuild_from_store(&mut self, store: &PhoenixCozoStore) -> Result<usize, StoreError> {
        let spans = indexed_spans_from_store(store)?;
        let count = spans.len();
        self.rebuild_from_spans(&spans);
        Ok(count)
    }

    pub fn search(&self, query: &str, scope: &ScopeKey, limit: usize) -> LexicalSearchResult {
        self.engine.search(query, scope, limit)
    }

    pub fn match_implicit(
        &self,
        text: &str,
        scope: &ScopeKey,
        lexicon: &phoenix_alex::Lexicon,
    ) -> Vec<ImplicitMatchHit> {
        self.engine.match_implicit(text, scope, lexicon)
    }

    pub fn config(&self) -> &LexConfig {
        &self.config
    }
}

pub fn indexed_spans_from_store(store: &PhoenixCozoStore) -> Result<Vec<IndexedSpan>, StoreError> {
    const CHUNK_COLUMNS: &[&str] = &["chunk_id", "doc_id", "text", "parent_id", "level"];
    const CHUNK_ID_COLUMNS: &[&str] = &["id", "chunk_key"];
    const NOTE_COLUMNS: &[&str] = &[
        "id",
        "owner_id",
        "title",
        "world_id",
        "narrative_id",
        "folder_id",
    ];

    let chunks = store.fetch_compact_rows_with_columns("chunks", CHUNK_COLUMNS)?;
    let chunk_keys = store
        .fetch_compact_rows_with_columns("chunkid_map", CHUNK_ID_COLUMNS)?
        .into_iter()
        .filter_map(|row| {
            let row = CompactRowView::new(CHUNK_ID_COLUMNS, &row);
            Some((row.get_i64("id")?, row.get_str("chunk_key")?.to_owned()))
        })
        .collect::<FxHashMap<_, _>>();
    let notes = store.fetch_compact_rows_with_columns("notes", NOTE_COLUMNS)?;
    #[derive(Clone, Debug)]
    struct NoteMeta {
        note_id: NoteId,
        title: Option<String>,
        scope: ScopeKey,
    }
    let note_titles = notes
        .into_iter()
        .filter_map(|row| {
            let row = CompactRowView::new(NOTE_COLUMNS, &row);
            let note_id = row.get_str("id")?.to_owned();
            let owner_id = row
                .get_str("owner_id")
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .unwrap_or_else(|| note_id.clone());
            Some((
                owner_id,
                NoteMeta {
                    note_id: NoteId(note_id),
                    title: row
                        .get_str("title")
                        .filter(|value| !value.trim().is_empty())
                        .map(str::to_owned),
                    scope: ScopeKey {
                        world_id: row
                            .get_str("world_id")
                            .filter(|value| !value.is_empty())
                            .map(str::to_owned),
                        narrative_id: row
                            .get_str("narrative_id")
                            .filter(|value| !value.is_empty())
                            .map(str::to_owned),
                        folder_id: row
                            .get_str("folder_id")
                            .filter(|value| !value.is_empty())
                            .map(str::to_owned),
                        folder_path: None,
                    },
                },
            ))
        })
        .collect::<FxHashMap<_, _>>();
    let chunk_rows = chunks
        .into_iter()
        .filter(|row| CompactRowView::new(CHUNK_COLUMNS, row).get_i64("level") == Some(0))
        .collect::<Vec<_>>();
    if chunk_rows.is_empty() {
        return indexed_spans_from_notes(store);
    }

    let parent_text_by_chunk_id = chunk_rows
        .iter()
        .filter_map(|row| {
            let row = CompactRowView::new(CHUNK_COLUMNS, row);
            Some((
                row.get_i64("chunk_id")?,
                row.get_str("text").unwrap_or_default().to_owned(),
            ))
        })
        .collect::<FxHashMap<_, _>>();

    Ok(chunk_rows
        .into_iter()
        .filter_map(|row| {
            let row = CompactRowView::new(CHUNK_COLUMNS, &row);
            let chunk_id = row.get_i64("chunk_id")?;
            let span_id = chunk_keys
                .get(&chunk_id)
                .cloned()
                .unwrap_or_else(|| format!("chunk:{chunk_id}"));
            let doc_id = row.get_str("doc_id")?.to_owned();
            let note_meta = note_titles.get(&doc_id)?;
            let body = row.get_str("text")?.to_owned();
            let mut fields = Vec::new();
            if let Some(title) = note_meta.title.as_deref() {
                fields.push(IndexedTextField {
                    field: LexicalField::Title,
                    text: title.to_owned(),
                });
            }
            if !body.trim().is_empty() {
                fields.push(IndexedTextField {
                    field: LexicalField::Body,
                    text: body,
                });
            }
            if let Some(parent_id) = row.get_i64("parent_id") {
                if let Some(parent_text) = parent_text_by_chunk_id
                    .get(&parent_id)
                    .filter(|value: &&String| !value.trim().is_empty())
                {
                    fields.push(IndexedTextField {
                        field: LexicalField::Summary,
                        text: parent_text.to_owned(),
                    });
                }
            }
            if fields.is_empty() {
                return None;
            }

            Some(IndexedSpan {
                span_id,
                note_id: Some(note_meta.note_id.clone()),
                document_id: Some(DocumentId(doc_id.clone())),
                scope: note_meta.scope.clone(),
                fields,
            })
        })
        .collect())
}

fn indexed_spans_from_notes(store: &PhoenixCozoStore) -> Result<Vec<IndexedSpan>, StoreError> {
    const NOTE_COLUMNS: &[&str] = &[
        "id",
        "title",
        "content",
        "markdown_content",
        "is_current",
        "world_id",
        "narrative_id",
        "folder_id",
    ];
    let rows = store.fetch_compact_rows_with_columns("notes", NOTE_COLUMNS)?;
    Ok(rows
        .into_iter()
        .filter(|row| {
            CompactRowView::new(NOTE_COLUMNS, row)
                .get_bool("is_current")
                .unwrap_or(true)
        })
        .filter_map(|row| note_row_to_indexed_span(CompactRowView::new(NOTE_COLUMNS, &row)))
        .collect())
}

fn note_row_to_indexed_span(row: CompactRowView<'_>) -> Option<IndexedSpan> {
    let id = row.get_str("id")?.to_owned();
    let title = row.get_str("title").unwrap_or_default().to_owned();
    let body = row
        .get_str("content")
        .or_else(|| row.get_str("markdown_content"))
        .unwrap_or_default()
        .to_owned();

    let mut fields = Vec::new();
    if !title.trim().is_empty() {
        fields.push(IndexedTextField {
            field: LexicalField::Title,
            text: title,
        });
    }
    if !body.trim().is_empty() {
        fields.push(IndexedTextField {
            field: LexicalField::Body,
            text: body,
        });
    }
    if fields.is_empty() {
        return None;
    }

    Some(IndexedSpan {
        span_id: id.clone(),
        note_id: Some(NoteId(id.clone())),
        document_id: Some(DocumentId(id)),
        scope: ScopeKey {
            world_id: optional_row_string(row, "world_id"),
            narrative_id: optional_row_string(row, "narrative_id"),
            folder_id: optional_row_string(row, "folder_id"),
            folder_path: None,
        },
        fields,
    })
}

fn optional_row_string(row: CompactRowView<'_>, key: &str) -> Option<String> {
    row.get_str(key)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facade_builds_and_searches() {
        let spans = vec![IndexedSpan {
            span_id: "doc-1:0:0:0-24".to_owned(),
            note_id: Some(NoteId("note-1".to_owned())),
            document_id: Some(DocumentId("doc-1".to_owned())),
            scope: ScopeKey::default(),
            fields: vec![IndexedTextField {
                field: LexicalField::Body,
                text: "The phoenix woke again.".to_owned(),
            }],
        }];

        let lex = LexIndex::build(&spans, LexConfig::default());
        let result = lex.search("phoenix", &ScopeKey::default(), 5);

        assert_eq!(result.span_hits.len(), 1);
        assert_eq!(result.span_hits[0].span_id, "doc-1:0:0:0-24");
    }

    #[test]
    fn public_qgram_query_types_stay_exposed() {
        let clauses: Vec<Clause> = parse_query("\"sea king\"");
        assert_eq!(clauses.len(), 1);
        assert_eq!(clauses[0].clause_type, ClauseType::Phrase);
        let _config = SearchConfig::default();
    }
}
