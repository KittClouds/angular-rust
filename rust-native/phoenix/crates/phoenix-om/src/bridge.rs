use std::collections::{BTreeMap, BTreeSet};

use phoenix_scanner::PhoenixScanner;
use phoenix_store_cozo::{CompactRowView, PhoenixCozoStore, StoreError};
use phoenix_structure::PhoenixStructure;
use phoenix_types::{
    OmGraphIndexRecord, OmIndexResult, OmLostMemoryHit, OmMemorySearchHit, ThreadMessage,
};
use serde_json::{json, Value};

const KIND_MESSAGES: &str = "messages";
const KIND_OBSERVATION: &str = "observation";
const KIND_REFLECTION: &str = "reflection";
const MAX_RELATION_SUMMARIES: usize = 4;
const MAX_SNIPPET_CHARS: usize = 240;
const MAX_SOURCE_KEYS_PER_HIT: usize = 4;

const OM_GRAPH_INDEX_COLUMNS: &[&str] = &[
    "thread_id",
    "kind",
    "source_key",
    "document_id",
    "entity_count",
    "edge_count",
    "created_at",
];
const OM_GRAPH_ENTITY_COLUMNS: &[&str] = &[
    "thread_id",
    "document_id",
    "entity_id",
    "label",
    "aliases",
    "mention_count",
    "snippet",
    "created_at",
];
const OM_GRAPH_RELATION_COLUMNS: &[&str] = &[
    "thread_id",
    "document_id",
    "entity_id",
    "relation_ord",
    "summary",
    "created_at",
];

#[derive(Default)]
pub struct OmNativeBridge;

impl OmNativeBridge {
    pub fn index_message_window(
        &self,
        store: &PhoenixCozoStore,
        _scanner: &PhoenixScanner,
        _structure: &PhoenixStructure,
        thread_id: &str,
        source_key: &str,
        messages: &[ThreadMessage],
    ) -> Result<OmIndexResult, StoreError> {
        if messages.is_empty() {
            return Ok(OmIndexResult {
                kind: KIND_MESSAGES.to_owned(),
                source_key: source_key.to_owned(),
                ..OmIndexResult::default()
            });
        }

        let document_id = message_document_id(thread_id, source_key);
        let units = messages
            .iter()
            .map(|message| message.content.as_str())
            .filter(|text| !text.trim().is_empty())
            .collect::<Vec<_>>();
        let draft = build_document_draft(&units);
        self.persist_document_index(
            store,
            thread_id,
            KIND_MESSAGES,
            source_key,
            &document_id,
            draft,
        )
    }

    pub fn index_observation_delta(
        &self,
        store: &PhoenixCozoStore,
        _scanner: &PhoenixScanner,
        _structure: &PhoenixStructure,
        thread_id: &str,
        source_key: &str,
        observation_text: &str,
    ) -> Result<OmIndexResult, StoreError> {
        self.index_text_document(
            store,
            thread_id,
            KIND_OBSERVATION,
            source_key,
            observation_text,
        )
    }

    pub fn index_reflection_summary(
        &self,
        store: &PhoenixCozoStore,
        _scanner: &PhoenixScanner,
        _structure: &PhoenixStructure,
        thread_id: &str,
        source_key: &str,
        reflection_text: &str,
    ) -> Result<OmIndexResult, StoreError> {
        self.index_text_document(
            store,
            thread_id,
            KIND_REFLECTION,
            source_key,
            reflection_text,
        )
    }

    pub fn recover_lost_memory(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        limit: usize,
        focus: Option<&str>,
    ) -> Result<Vec<OmLostMemoryHit>, StoreError> {
        let indices = self.load_index_rows(store, thread_id)?;
        let message_docs = indices_for_kind(&indices, KIND_MESSAGES);
        if message_docs.is_empty() {
            return Ok(Vec::new());
        }
        let summary_docs = indices
            .iter()
            .filter(|row| row.kind == KIND_OBSERVATION || row.kind == KIND_REFLECTION)
            .map(|row| row.document_id.clone())
            .collect::<Vec<_>>();
        let message_rows = self.load_entity_rows(
            store,
            thread_id,
            &message_docs.keys().cloned().collect::<Vec<_>>(),
        )?;
        if message_rows.is_empty() {
            return Ok(Vec::new());
        }
        let summary_entity_ids = self
            .load_entity_rows(store, thread_id, &summary_docs)?
            .into_iter()
            .map(|row| row.entity_id)
            .collect::<BTreeSet<_>>();
        let normalized_focus = focus
            .map(normalize_surface)
            .filter(|value| !value.is_empty());

        let mut by_entity = BTreeMap::<String, AggregatedEntity>::new();
        for row in message_rows {
            let entry =
                by_entity
                    .entry(row.entity_id.clone())
                    .or_insert_with(|| AggregatedEntity {
                        entity_id: row.entity_id.clone(),
                        label: row.label.clone(),
                        aliases: row.aliases.clone(),
                        total_mentions: 0,
                        snippet: row.snippet.clone(),
                        documents: BTreeSet::new(),
                    });
            entry.total_mentions += row.mention_count;
            entry.documents.insert(row.document_id.clone());
            if entry.snippet.is_empty() && !row.snippet.is_empty() {
                entry.snippet = row.snippet.clone();
            }
        }

        let mut lost = Vec::new();
        for entity in by_entity.into_values() {
            if summary_entity_ids.contains(&entity.entity_id) {
                continue;
            }
            if let Some(ref needle) = normalized_focus {
                let label_match = normalize_surface(&entity.label).contains(needle)
                    || entity
                        .aliases
                        .iter()
                        .any(|alias| normalize_surface(alias).contains(needle));
                if !label_match {
                    continue;
                }
            }

            let mut source_keys = entity
                .documents
                .iter()
                .filter_map(|document_id| message_docs.get(document_id).cloned())
                .take(MAX_SOURCE_KEYS_PER_HIT)
                .collect::<Vec<_>>();
            source_keys.sort();
            lost.push(OmLostMemoryHit {
                entity_id: entity.entity_id.clone(),
                label: entity.label.clone(),
                total_mentions: entity.total_mentions,
                source_keys,
                relation_summaries: self.load_relation_summaries(
                    store,
                    thread_id,
                    &entity.documents.iter().cloned().collect::<Vec<_>>(),
                    &entity.entity_id,
                    MAX_RELATION_SUMMARIES,
                )?,
            });
        }
        lost.sort_by(|left, right| {
            right
                .total_mentions
                .cmp(&left.total_mentions)
                .then_with(|| left.label.cmp(&right.label))
        });
        lost.truncate(limit.max(1));
        Ok(lost)
    }

    pub fn memory_graph_search(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<OmMemorySearchHit>, StoreError> {
        let query = normalize_surface(query);
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let indices = self.load_index_rows(store, thread_id)?;
        if indices.is_empty() {
            return Ok(Vec::new());
        }
        let allowed_docs = indices
            .into_iter()
            .map(|row| (row.document_id, (row.kind, row.source_key)))
            .collect::<BTreeMap<_, _>>();
        let entity_rows = self.load_entity_rows(
            store,
            thread_id,
            &allowed_docs.keys().cloned().collect::<Vec<_>>(),
        )?;
        if entity_rows.is_empty() {
            return Ok(Vec::new());
        }

        let mut by_entity = BTreeMap::<String, AggregatedEntity>::new();
        for row in entity_rows {
            let entry =
                by_entity
                    .entry(row.entity_id.clone())
                    .or_insert_with(|| AggregatedEntity {
                        entity_id: row.entity_id.clone(),
                        label: row.label.clone(),
                        aliases: row.aliases.clone(),
                        total_mentions: 0,
                        snippet: row.snippet.clone(),
                        documents: BTreeSet::new(),
                    });
            entry.total_mentions += row.mention_count;
            entry.documents.insert(row.document_id.clone());
            if entry.snippet.is_empty() && !row.snippet.is_empty() {
                entry.snippet = row.snippet.clone();
            }
        }

        let mut hits = Vec::new();
        for entity in by_entity.into_values() {
            let relation_summaries = self.load_relation_summaries(
                store,
                thread_id,
                &entity.documents.iter().cloned().collect::<Vec<_>>(),
                &entity.entity_id,
                MAX_RELATION_SUMMARIES,
            )?;
            let alias_match = entity
                .aliases
                .iter()
                .any(|alias| normalize_surface(alias).contains(&query));
            let relation_match = relation_summaries
                .iter()
                .any(|summary| normalize_surface(summary).contains(&query));
            if !normalize_surface(&entity.label).contains(&query) && !alias_match && !relation_match
            {
                continue;
            }

            let Some(document_id) = entity.documents.iter().next().cloned() else {
                continue;
            };
            let Some((source_kind, source_key)) = allowed_docs.get(&document_id).cloned() else {
                continue;
            };
            hits.push(OmMemorySearchHit {
                label: entity.label,
                kind: "entity".to_owned(),
                document_id,
                source_kind,
                source_key,
                snippet: entity.snippet,
                relation_summaries,
            });
        }
        hits.sort_by(|left, right| left.label.cmp(&right.label));
        hits.truncate(limit.max(1));
        Ok(hits)
    }

    fn index_text_document(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        kind: &str,
        source_key: &str,
        text: &str,
    ) -> Result<OmIndexResult, StoreError> {
        let document_id = summary_document_id(thread_id, kind, source_key);
        let units = split_text_units(text);
        let draft = build_document_draft(&units);
        self.persist_document_index(store, thread_id, kind, source_key, &document_id, draft)
    }

    fn persist_document_index(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        kind: &str,
        source_key: &str,
        document_id: &str,
        draft: DocumentDraft,
    ) -> Result<OmIndexResult, StoreError> {
        self.clear_document_rows(store, thread_id, kind, source_key, document_id)?;

        let now = now_ms();
        let entity_rows = draft
            .entities
            .iter()
            .map(|entity| {
                json!({
                    "thread_id": thread_id,
                    "document_id": document_id,
                    "entity_id": entity.entity_id,
                    "label": entity.label,
                    "aliases": entity.aliases,
                    "mention_count": entity.mention_count,
                    "snippet": entity.snippet,
                    "created_at": now,
                })
            })
            .collect::<Vec<_>>();
        store.put_rows("om_graph_entities", &entity_rows)?;

        let relation_rows = draft
            .relations
            .iter()
            .enumerate()
            .map(|(index, relation)| {
                json!({
                    "thread_id": thread_id,
                    "document_id": document_id,
                    "entity_id": relation.entity_id,
                    "relation_ord": index as i64,
                    "summary": relation.summary,
                    "created_at": now,
                })
            })
            .collect::<Vec<_>>();
        store.put_rows("om_graph_relations", &relation_rows)?;

        let index = OmIndexResult {
            kind: kind.to_owned(),
            source_key: source_key.to_owned(),
            document_id: document_id.to_owned(),
            entity_count: draft.entities.len() as i64,
            edge_count: draft.relations.len() as i64,
        };
        store.put_row(
            "om_graph_index",
            json!({
                "thread_id": thread_id,
                "kind": index.kind,
                "source_key": index.source_key,
                "document_id": index.document_id,
                "entity_count": index.entity_count,
                "edge_count": index.edge_count,
                "created_at": now,
            }),
        )?;
        Ok(index)
    }

    fn clear_document_rows(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        kind: &str,
        source_key: &str,
        document_id: &str,
    ) -> Result<(), StoreError> {
        let index_keys = store
            .fetch_compact_rows_where_str(
                "om_graph_index",
                OM_GRAPH_INDEX_COLUMNS,
                "thread_id",
                thread_id,
            )?
            .into_iter()
            .filter(|row| {
                let row = CompactRowView::new(OM_GRAPH_INDEX_COLUMNS, row);
                row.get_str("kind") == Some(kind)
                    && row.get_str("source_key") == Some(source_key)
                    && row.get_str("document_id") == Some(document_id)
            })
            .collect::<Vec<_>>();
        store.delete_key_rows("om_graph_index", &index_keys)?;

        let entity_keys = store
            .fetch_compact_rows_where_str(
                "om_graph_entities",
                OM_GRAPH_ENTITY_COLUMNS,
                "thread_id",
                thread_id,
            )?
            .into_iter()
            .filter(|row| {
                let row = CompactRowView::new(OM_GRAPH_ENTITY_COLUMNS, row);
                row.get_str("document_id") == Some(document_id)
            })
            .collect::<Vec<_>>();
        store.delete_key_rows("om_graph_entities", &entity_keys)?;

        let relation_keys = store
            .fetch_compact_rows_where_str(
                "om_graph_relations",
                OM_GRAPH_RELATION_COLUMNS,
                "thread_id",
                thread_id,
            )?
            .into_iter()
            .filter(|row| {
                let row = CompactRowView::new(OM_GRAPH_RELATION_COLUMNS, row);
                row.get_str("document_id") == Some(document_id)
            })
            .collect::<Vec<_>>();
        store.delete_key_rows("om_graph_relations", &relation_keys)?;
        Ok(())
    }

    fn load_index_rows(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
    ) -> Result<Vec<OmGraphIndexRecord>, StoreError> {
        let mut records = store
            .fetch_compact_rows_where_str(
                "om_graph_index",
                OM_GRAPH_INDEX_COLUMNS,
                "thread_id",
                thread_id,
            )?
            .into_iter()
            .map(|row| graph_index_from_row(&CompactRowView::new(OM_GRAPH_INDEX_COLUMNS, &row)))
            .collect::<Result<Vec<_>, _>>()?;
        records.sort_by(|left, right| left.created_at.cmp(&right.created_at));
        Ok(records)
    }

    fn load_entity_rows(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        document_ids: &[String],
    ) -> Result<Vec<OmGraphEntityRow>, StoreError> {
        if document_ids.is_empty() {
            return Ok(Vec::new());
        }
        let allowed = document_ids.iter().cloned().collect::<BTreeSet<_>>();
        let rows = store.fetch_compact_rows_where_str(
            "om_graph_entities",
            OM_GRAPH_ENTITY_COLUMNS,
            "thread_id",
            thread_id,
        )?;
        Ok(rows
            .into_iter()
            .filter_map(|row| {
                let row = CompactRowView::new(OM_GRAPH_ENTITY_COLUMNS, &row);
                let document_id = row.get_str("document_id")?.to_owned();
                allowed.contains(&document_id).then(|| OmGraphEntityRow {
                    document_id,
                    entity_id: row.get_str("entity_id").unwrap_or_default().to_owned(),
                    label: row.get_str("label").unwrap_or_default().to_owned(),
                    aliases: row
                        .get_json("aliases")
                        .and_then(|value| {
                            value.as_array().map(|items| {
                                items
                                    .iter()
                                    .filter_map(Value::as_str)
                                    .map(str::to_owned)
                                    .collect::<Vec<_>>()
                            })
                        })
                        .unwrap_or_default(),
                    mention_count: row.get_i64("mention_count").unwrap_or_default(),
                    snippet: row.get_str("snippet").unwrap_or_default().to_owned(),
                })
            })
            .collect::<Vec<_>>())
    }

    fn load_relation_summaries(
        &self,
        store: &PhoenixCozoStore,
        thread_id: &str,
        document_ids: &[String],
        entity_id: &str,
        limit: usize,
    ) -> Result<Vec<String>, StoreError> {
        if document_ids.is_empty() {
            return Ok(Vec::new());
        }
        let allowed = document_ids.iter().cloned().collect::<BTreeSet<_>>();
        let rows = store.fetch_compact_rows_where_str(
            "om_graph_relations",
            OM_GRAPH_RELATION_COLUMNS,
            "thread_id",
            thread_id,
        )?;
        let mut summaries = rows
            .into_iter()
            .filter_map(|row| {
                let row = CompactRowView::new(OM_GRAPH_RELATION_COLUMNS, &row);
                (row.get_str("entity_id") == Some(entity_id)
                    && row
                        .get_str("document_id")
                        .map(|value| allowed.contains(value))
                        .unwrap_or(false))
                .then(|| row.get_str("summary").unwrap_or_default().to_owned())
            })
            .collect::<Vec<_>>();
        summaries.sort();
        summaries.dedup();
        summaries.truncate(limit);
        Ok(summaries)
    }
}

#[derive(Clone, Debug, Default)]
struct DocumentDraft {
    entities: Vec<OmEntityDraft>,
    relations: Vec<OmRelationDraft>,
}

#[derive(Clone, Debug, Default)]
struct OmEntityDraft {
    entity_id: String,
    label: String,
    aliases: Vec<String>,
    mention_count: i64,
    snippet: String,
}

#[derive(Clone, Debug, Default)]
struct OmRelationDraft {
    entity_id: String,
    summary: String,
}

#[derive(Clone, Debug, Default)]
struct OmGraphEntityRow {
    document_id: String,
    entity_id: String,
    label: String,
    aliases: Vec<String>,
    mention_count: i64,
    snippet: String,
}

#[derive(Clone, Debug, Default)]
struct AggregatedEntity {
    entity_id: String,
    label: String,
    aliases: Vec<String>,
    total_mentions: i64,
    snippet: String,
    documents: BTreeSet<String>,
}

#[derive(Clone, Debug, Default)]
struct EntityAggregateDraft {
    label: String,
    aliases: BTreeSet<String>,
    mention_count: i64,
    snippet: String,
}

#[derive(Clone, Debug)]
struct MentionRecord {
    entity_id: String,
    label: String,
}

fn build_document_draft(units: &[&str]) -> DocumentDraft {
    let mut entities = BTreeMap::<String, EntityAggregateDraft>::new();
    let mut relation_counts = BTreeMap::<(String, String), usize>::new();

    for unit in units {
        let mentions = extract_mentions(unit);
        let mut unit_entities = BTreeSet::<String>::new();
        for mention in mentions {
            let entry = entities
                .entry(mention.entity_id.clone())
                .or_insert_with(|| EntityAggregateDraft {
                    label: mention.label.clone(),
                    ..EntityAggregateDraft::default()
                });
            if better_label(&mention.label, &entry.label) {
                entry.label = mention.label.clone();
            }
            entry.mention_count += 1;
            if entry.snippet.is_empty() {
                entry.snippet = clip_text(unit, MAX_SNIPPET_CHARS);
            }
            for alias in aliases_for_label(&mention.label) {
                if alias != entry.label {
                    entry.aliases.insert(alias);
                }
            }
            unit_entities.insert(mention.entity_id);
        }

        let labels = unit_entities
            .iter()
            .filter_map(|entity_id| {
                entities
                    .get(entity_id)
                    .map(|entity| (entity_id, entity.label.clone()))
            })
            .collect::<Vec<_>>();
        for (entity_id, _) in &labels {
            for (other_id, other_label) in &labels {
                if entity_id == other_id {
                    continue;
                }
                *relation_counts
                    .entry((
                        (*entity_id).clone(),
                        format!("co-mentioned with {other_label}"),
                    ))
                    .or_default() += 1;
            }
        }
    }

    let entities = entities
        .into_iter()
        .map(|(entity_id, entity)| OmEntityDraft {
            entity_id,
            label: entity.label,
            aliases: entity.aliases.into_iter().collect(),
            mention_count: entity.mention_count,
            snippet: entity.snippet,
        })
        .collect::<Vec<_>>();
    let mut relations = relation_counts
        .into_iter()
        .map(|((entity_id, summary), count)| (entity_id, summary, count))
        .collect::<Vec<_>>();
    relations.sort_by(|left, right| {
        right
            .2
            .cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
    });
    let relations = relations
        .into_iter()
        .map(|(entity_id, summary, _)| OmRelationDraft { entity_id, summary })
        .collect::<Vec<_>>();

    DocumentDraft {
        entities,
        relations,
    }
}

fn extract_mentions(text: &str) -> Vec<MentionRecord> {
    let tokens = tokenize(text);
    let mut mentions = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        if !tokens[index].is_entity_start() {
            index += 1;
            continue;
        }
        let start = index;
        let mut end = index + 1;
        while end < tokens.len() {
            if tokens[end].is_entity_token() {
                end += 1;
                continue;
            }
            if tokens[end].is_connector()
                && end + 1 < tokens.len()
                && tokens[end + 1].is_entity_token()
            {
                end += 2;
                continue;
            }
            break;
        }
        let label = tokens[start..end]
            .iter()
            .map(|token| token.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let canonical_label = strip_title_prefix(&label).unwrap_or_else(|| label.clone());
        let normalized = normalize_surface(&canonical_label);
        if !normalized.is_empty() && !is_common_non_entity(&normalized) {
            mentions.push(MentionRecord {
                entity_id: format!("om-entity::{normalized}"),
                label,
            });
        }
        index = end.max(start + 1);
    }
    mentions
}

#[derive(Clone, Debug)]
struct SimpleToken {
    text: String,
}

impl SimpleToken {
    fn is_entity_start(&self) -> bool {
        self.is_entity_token() && !is_connector_word(&self.text)
    }

    fn is_entity_token(&self) -> bool {
        is_uppercase_token(&self.text)
    }

    fn is_connector(&self) -> bool {
        is_connector_word(&self.text)
    }
}

fn tokenize(text: &str) -> Vec<SimpleToken> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() || matches!(ch, '\'' | '-' | '&') {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(SimpleToken {
                text: std::mem::take(&mut current),
            });
        }
    }
    if !current.is_empty() {
        tokens.push(SimpleToken { text: current });
    }
    tokens
}

fn split_text_units(text: &str) -> Vec<&str> {
    text.split(['\n', '.', '!', '?'])
        .map(str::trim)
        .filter(|segment| !segment.is_empty())
        .collect()
}

fn aliases_for_label(label: &str) -> Vec<String> {
    let mut aliases = Vec::new();
    if let Some(stripped) = strip_title_prefix(label) {
        aliases.push(stripped.to_owned());
    }
    let acronym = acronym_for_label(label);
    if !acronym.is_empty() {
        aliases.push(acronym);
    }
    aliases.sort();
    aliases.dedup();
    aliases
}

fn strip_title_prefix(label: &str) -> Option<String> {
    let parts = label.split_whitespace().collect::<Vec<_>>();
    let first = *parts.first()?;
    if !matches!(
        normalize_surface(first).as_str(),
        "mr" | "mrs"
            | "ms"
            | "dr"
            | "prof"
            | "professor"
            | "capt"
            | "captain"
            | "sir"
            | "lady"
            | "lord"
    ) {
        return None;
    }
    let stripped = parts[1..].join(" ");
    (!stripped.is_empty()).then_some(stripped)
}

fn acronym_for_label(label: &str) -> String {
    let letters = label
        .split_whitespace()
        .filter(|part| !is_connector_word(part))
        .filter_map(|part| part.chars().next())
        .filter(|ch| ch.is_alphabetic())
        .collect::<String>();
    (letters.len() > 1)
        .then(|| letters.to_ascii_uppercase())
        .unwrap_or_default()
}

fn is_uppercase_token(token: &str) -> bool {
    let mut chars = token.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if token.len() == 1 {
        return false;
    }
    first.is_uppercase() || token.chars().all(|ch| !ch.is_lowercase())
}

fn is_connector_word(token: &str) -> bool {
    matches!(
        normalize_surface(token).as_str(),
        "of" | "the" | "and" | "de" | "del" | "la" | "van" | "von" | "&"
    )
}

fn is_common_non_entity(normalized: &str) -> bool {
    matches!(
        normalized,
        "the" | "a" | "an" | "he" | "she" | "they" | "we" | "i" | "you" | "it"
    )
}

fn normalize_surface(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len());
    let mut last_space = true;
    for ch in text.chars() {
        let mapped = if ch.is_alphanumeric() {
            Some(ch.to_ascii_lowercase())
        } else {
            None
        };
        match mapped {
            Some(value) => {
                normalized.push(value);
                last_space = false;
            }
            None if !last_space => {
                normalized.push(' ');
                last_space = true;
            }
            None => {}
        }
    }
    normalized.trim().to_owned()
}

fn indices_for_kind(rows: &[OmGraphIndexRecord], kind: &str) -> BTreeMap<String, String> {
    rows.iter()
        .filter(|row| row.kind == kind)
        .map(|row| (row.document_id.clone(), row.source_key.clone()))
        .collect()
}

fn graph_index_from_row(row: &CompactRowView<'_>) -> Result<OmGraphIndexRecord, StoreError> {
    Ok(OmGraphIndexRecord {
        thread_id: row.get_str("thread_id").unwrap_or_default().to_owned(),
        kind: row.get_str("kind").unwrap_or_default().to_owned(),
        source_key: row.get_str("source_key").unwrap_or_default().to_owned(),
        document_id: row.get_str("document_id").unwrap_or_default().to_owned(),
        entity_count: row.get_i64("entity_count").unwrap_or_default(),
        edge_count: row.get_i64("edge_count").unwrap_or_default(),
        created_at: row.get_i64("created_at").unwrap_or_default(),
    })
}

fn message_document_id(thread_id: &str, source_key: &str) -> String {
    format!("om::thread::{thread_id}::messages::{source_key}")
}

fn summary_document_id(thread_id: &str, kind: &str, source_key: &str) -> String {
    format!("om::thread::{thread_id}::{kind}::{source_key}")
}

fn clip_text(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    let clipped = text.chars().take(limit).collect::<String>();
    format!("{clipped}...")
}

fn better_label(candidate: &str, current: &str) -> bool {
    candidate.len() > current.len() || (candidate.len() == current.len() && candidate < current)
}

fn now_ms() -> i64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        return SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as i64)
            .unwrap_or_default();
    }
    #[cfg(target_arch = "wasm32")]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::OmNativeBridge;
    use phoenix_scanner::PhoenixScanner;
    use phoenix_store_cozo::{PhoenixCozoStore, StoreConfig};
    use phoenix_structure::PhoenixStructure;
    use phoenix_types::{StorageMode, ThreadMessage};

    fn store() -> PhoenixCozoStore {
        let store = PhoenixCozoStore::open(StoreConfig {
            mode: StorageMode::CozoMem,
            path: None,
        })
        .expect("open store");
        store.init_schema().expect("init schema");
        store
    }

    fn message(id: &str, content: &str, created_at: i64) -> ThreadMessage {
        ThreadMessage {
            id: id.to_owned(),
            thread_id: "thread-1".to_owned(),
            role: "user".to_owned(),
            content: content.to_owned(),
            narrative_id: String::new(),
            created_at,
            updated_at: created_at,
            is_streaming: false,
            token_count: None,
            is_observed: false,
        }
    }

    #[test]
    fn native_bridge_indexes_and_searches_thread_memory() {
        let store = store();
        let bridge = OmNativeBridge;
        let scanner = PhoenixScanner::default();
        let structure = PhoenixStructure::default();

        bridge
            .index_message_window(
                &store,
                &scanner,
                &structure,
                "thread-1",
                "msg:1:2",
                &[
                    message("msg-1", "Captain Luffy met Nami in Water Seven.", 1),
                    message("msg-2", "Nami warned Luffy about the Marines.", 2),
                ],
            )
            .expect("index messages");
        bridge
            .index_observation_delta(
                &store,
                &scanner,
                &structure,
                "thread-1",
                "obs:2",
                "Nami remains alert.",
            )
            .expect("index observation");

        let lost = bridge
            .recover_lost_memory(&store, "thread-1", 10, Some("luffy"))
            .expect("recover lost memory");
        assert_eq!(lost.len(), 1);
        assert!(lost[0].label.contains("Luffy"));

        let hits = bridge
            .memory_graph_search(&store, "thread-1", "marines", 10)
            .expect("memory graph search");
        assert!(!hits.is_empty());
        assert!(hits
            .iter()
            .any(|hit| hit.label.contains("Luffy") || hit.label.contains("Nami")));
    }
}
