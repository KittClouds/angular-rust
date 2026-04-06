#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhoenixColumnType {
    String,
    Int,
    Float,
    Bool,
    Json,
    VectorF32(usize),
}

impl PhoenixColumnType {
    pub fn as_cozo(self) -> String {
        match self {
            Self::String => "String".to_owned(),
            Self::Int => "Int".to_owned(),
            Self::Float => "Float".to_owned(),
            Self::Bool => "Bool".to_owned(),
            Self::Json => "Json".to_owned(),
            Self::VectorF32(dim) => format!("<F32; {dim}>"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhoenixColumnSpec {
    pub name: &'static str,
    pub ty: PhoenixColumnType,
    pub optional: bool,
    pub key: bool,
}

impl PhoenixColumnSpec {
    pub const fn new(name: &'static str, ty: PhoenixColumnType, optional: bool, key: bool) -> Self {
        Self {
            name,
            ty,
            optional,
            key,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhoenixRelationSpec {
    pub name: &'static str,
    pub columns: &'static [PhoenixColumnSpec],
}

impl PhoenixRelationSpec {
    pub const fn new(name: &'static str, columns: &'static [PhoenixColumnSpec]) -> Self {
        Self { name, columns }
    }

    pub fn key_columns(&self) -> impl Iterator<Item = &PhoenixColumnSpec> {
        self.columns.iter().filter(|column| column.key)
    }

    pub fn value_columns(&self) -> impl Iterator<Item = &PhoenixColumnSpec> {
        self.columns.iter().filter(|column| !column.key)
    }
}

const fn col(
    name: &'static str,
    ty: PhoenixColumnType,
    optional: bool,
    key: bool,
) -> PhoenixColumnSpec {
    PhoenixColumnSpec::new(name, ty, optional, key)
}

const PHOENIX_SCHEMA_STATE: &[PhoenixColumnSpec] = &[
    col("version", PhoenixColumnType::String, false, true),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const PHOENIX_SESSIONS: &[PhoenixColumnSpec] = &[
    col("session_id", PhoenixColumnType::String, false, true),
    col("label", PhoenixColumnType::String, false, false),
    col("world_id", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("folder_id", PhoenixColumnType::String, true, false),
    col("folder_path", PhoenixColumnType::String, true, false),
    col("status", PhoenixColumnType::String, false, false),
    col("revision", PhoenixColumnType::Int, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const PHOENIX_COMMITS: &[PhoenixColumnSpec] = &[
    col("commit_id", PhoenixColumnType::String, false, true),
    col("session_id", PhoenixColumnType::String, false, false),
    col("reason", PhoenixColumnType::String, true, false),
    col("revision", PhoenixColumnType::Int, false, false),
    col("committed_at", PhoenixColumnType::Int, false, false),
];

const PHOENIX_INGEST_LOG: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("session_id", PhoenixColumnType::String, true, false),
    col("document_count", PhoenixColumnType::Int, false, false),
    col("commit_requested", PhoenixColumnType::Bool, false, false),
    col("request_json", PhoenixColumnType::Json, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const PHOENIX_QUERY_LOG: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("session_id", PhoenixColumnType::String, true, false),
    col("query", PhoenixColumnType::String, false, false),
    col("limit", PhoenixColumnType::Int, true, false),
    col("request_json", PhoenixColumnType::Json, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const NOTES: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("version", PhoenixColumnType::Int, false, true),
    col("world_id", PhoenixColumnType::String, false, false),
    col("title", PhoenixColumnType::String, false, false),
    col("content", PhoenixColumnType::String, false, false),
    col("markdown_content", PhoenixColumnType::String, true, false),
    col("folder_id", PhoenixColumnType::String, true, false),
    col("entity_kind", PhoenixColumnType::String, true, false),
    col("entity_subtype", PhoenixColumnType::String, true, false),
    col("is_entity", PhoenixColumnType::Bool, false, false),
    col("is_pinned", PhoenixColumnType::Bool, false, false),
    col("favorite", PhoenixColumnType::Bool, false, false),
    col("owner_id", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("order", PhoenixColumnType::Float, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
    col("valid_from", PhoenixColumnType::Int, false, false),
    col("valid_to", PhoenixColumnType::Int, true, false),
    col("is_current", PhoenixColumnType::Bool, false, false),
    col("change_reason", PhoenixColumnType::String, true, false),
];

const ENTITIES: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("label", PhoenixColumnType::String, false, false),
    col("kind", PhoenixColumnType::String, false, false),
    col("subtype", PhoenixColumnType::String, true, false),
    col("aliases", PhoenixColumnType::Json, true, false),
    col("first_note", PhoenixColumnType::String, true, false),
    col("total_mentions", PhoenixColumnType::Int, false, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("created_by", PhoenixColumnType::String, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const EDGES: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("source_id", PhoenixColumnType::String, false, false),
    col("target_id", PhoenixColumnType::String, false, false),
    col("rel_type", PhoenixColumnType::String, false, false),
    col("confidence", PhoenixColumnType::Float, false, false),
    col("bidirectional", PhoenixColumnType::Bool, false, false),
    col("source_note", PhoenixColumnType::String, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const FOLDERS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("name", PhoenixColumnType::String, false, false),
    col("parent_id", PhoenixColumnType::String, true, false),
    col("world_id", PhoenixColumnType::String, false, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("folder_order", PhoenixColumnType::Float, false, false),
    col("entity_kind", PhoenixColumnType::String, true, false),
    col("entity_subtype", PhoenixColumnType::String, true, false),
    col("entity_label", PhoenixColumnType::String, true, false),
    col("color", PhoenixColumnType::String, true, false),
    col("is_typed_root", PhoenixColumnType::Bool, false, false),
    col("is_subtype_root", PhoenixColumnType::Bool, false, false),
    col("collapsed", PhoenixColumnType::Bool, false, false),
    col("owner_id", PhoenixColumnType::String, true, false),
    col("is_narrative_root", PhoenixColumnType::Bool, false, false),
    col("attributes", PhoenixColumnType::Json, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const THREADS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("world_id", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("title", PhoenixColumnType::String, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const THREAD_MESSAGES: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("thread_id", PhoenixColumnType::String, false, false),
    col("role", PhoenixColumnType::String, false, false),
    col("content", PhoenixColumnType::String, false, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, true, false),
    col("is_streaming", PhoenixColumnType::Bool, false, false),
    col("token_count", PhoenixColumnType::Int, true, false),
    col("is_observed", PhoenixColumnType::Bool, true, false),
];

const MEMORIES: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("content", PhoenixColumnType::String, false, false),
    col("memory_type", PhoenixColumnType::String, false, false),
    col("confidence", PhoenixColumnType::Float, false, false),
    col("source_role", PhoenixColumnType::String, true, false),
    col("entity_id", PhoenixColumnType::String, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const MEMORY_THREADS: &[PhoenixColumnSpec] = &[
    col("memory_id", PhoenixColumnType::String, false, true),
    col("thread_id", PhoenixColumnType::String, false, true),
    col("message_id", PhoenixColumnType::String, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const OM_RECORDS: &[PhoenixColumnSpec] = &[
    col("thread_id", PhoenixColumnType::String, false, true),
    col("observations", PhoenixColumnType::String, false, false),
    col("current_task", PhoenixColumnType::String, false, false),
    col(
        "suggested_continuation",
        PhoenixColumnType::String,
        true,
        false,
    ),
    col("last_observed_at", PhoenixColumnType::Int, false, false),
    col("obs_token_count", PhoenixColumnType::Int, false, false),
    col("generation_num", PhoenixColumnType::Int, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const OM_GENERATIONS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("thread_id", PhoenixColumnType::String, false, false),
    col("generation", PhoenixColumnType::Int, false, false),
    col("input_tokens", PhoenixColumnType::Int, false, false),
    col("output_tokens", PhoenixColumnType::Int, false, false),
    col("input_text", PhoenixColumnType::String, false, false),
    col("output_text", PhoenixColumnType::String, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const OM_GRAPH_INDEX: &[PhoenixColumnSpec] = &[
    col("thread_id", PhoenixColumnType::String, false, true),
    col("kind", PhoenixColumnType::String, false, true),
    col("source_key", PhoenixColumnType::String, false, true),
    col("document_id", PhoenixColumnType::String, false, false),
    col("entity_count", PhoenixColumnType::Int, false, false),
    col("edge_count", PhoenixColumnType::Int, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const SEMANTIC_VECTORS: &[PhoenixColumnSpec] = &[
    col("span_id", PhoenixColumnType::String, false, true),
    col("vec", PhoenixColumnType::VectorF32(384), false, false),
    col("model_id", PhoenixColumnType::String, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const SEMANTIC_DOCUMENTS: &[PhoenixColumnSpec] = &[
    col("document_id", PhoenixColumnType::String, false, true),
    col("vec", PhoenixColumnType::VectorF32(384), false, false),
    col("model_id", PhoenixColumnType::String, false, false),
    col("leaf_count", PhoenixColumnType::Int, false, false),
    col("evidence_refs", PhoenixColumnType::Json, true, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const SEMANTIC_NODE_PROTOTYPES: &[PhoenixColumnSpec] = &[
    col("node_id", PhoenixColumnType::String, false, true),
    col("node_kind", PhoenixColumnType::String, false, false),
    col("document_id", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("folder_id", PhoenixColumnType::String, true, false),
    col("vec", PhoenixColumnType::VectorF32(384), false, false),
    col("model_id", PhoenixColumnType::String, false, false),
    col("evidence_refs", PhoenixColumnType::Json, true, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const WORKSPACE_ARTIFACTS: &[PhoenixColumnSpec] = &[
    col("key", PhoenixColumnType::String, false, true),
    col("thread_id", PhoenixColumnType::String, false, true),
    col("narrative_id", PhoenixColumnType::String, false, true),
    col("folder_id", PhoenixColumnType::String, false, true),
    col("kind", PhoenixColumnType::String, false, false),
    col("payload", PhoenixColumnType::Json, false, false),
    col("pinned", PhoenixColumnType::Bool, false, false),
    col("produced_by", PhoenixColumnType::String, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const CHAT_RUNS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("thread_id", PhoenixColumnType::String, false, false),
    col("user_prompt", PhoenixColumnType::String, false, false),
    col("status", PhoenixColumnType::String, false, false),
    col("options_json", PhoenixColumnType::Json, false, false),
    col("capabilities_json", PhoenixColumnType::Json, false, false),
    col("prepared_context", PhoenixColumnType::String, false, false),
    col(
        "prepared_system_prompt",
        PhoenixColumnType::String,
        false,
        false,
    ),
    col(
        "planner_messages_json",
        PhoenixColumnType::Json,
        false,
        false,
    ),
    col("evidence_json", PhoenixColumnType::Json, false, false),
    col(
        "missing_capabilities_json",
        PhoenixColumnType::Json,
        false,
        false,
    ),
    col("error", PhoenixColumnType::String, true, false),
    col("final_response", PhoenixColumnType::String, true, false),
    col(
        "assistant_message_id",
        PhoenixColumnType::String,
        true,
        false,
    ),
    col("deadline_at", PhoenixColumnType::Int, false, false),
    col("completed_at", PhoenixColumnType::Int, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const CHAT_RUN_EVENTS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("run_id", PhoenixColumnType::String, false, false),
    col("phase", PhoenixColumnType::String, false, false),
    col("kind", PhoenixColumnType::String, false, false),
    col("label", PhoenixColumnType::String, false, false),
    col("detail", PhoenixColumnType::String, true, false),
    col("status", PhoenixColumnType::String, true, false),
    col("payload", PhoenixColumnType::String, true, false),
    col("latency_ms", PhoenixColumnType::Int, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const CHAT_TOOL_CALLS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("run_id", PhoenixColumnType::String, false, false),
    col("tool_call_id", PhoenixColumnType::String, false, false),
    col("tool_name", PhoenixColumnType::String, false, false),
    col("host", PhoenixColumnType::String, false, false),
    col("class", PhoenixColumnType::String, false, false),
    col("status", PhoenixColumnType::String, false, false),
    col("arguments_json", PhoenixColumnType::Json, false, false),
    col("result_json", PhoenixColumnType::String, false, false),
    col("error", PhoenixColumnType::String, false, false),
    col("idempotency_key", PhoenixColumnType::String, false, false),
    col("approval_id", PhoenixColumnType::String, false, false),
    col("started_at", PhoenixColumnType::Int, false, false),
    col("completed_at", PhoenixColumnType::Int, false, false),
    col("latency_ms", PhoenixColumnType::Int, false, false),
];

const CHAT_APPROVAL_REQUESTS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("run_id", PhoenixColumnType::String, false, false),
    col("tool_call_id", PhoenixColumnType::String, false, false),
    col("tool_name", PhoenixColumnType::String, false, false),
    col("status", PhoenixColumnType::String, false, false),
    col("affected_note_id", PhoenixColumnType::String, false, false),
    col("summary", PhoenixColumnType::String, false, false),
    col("diff_preview", PhoenixColumnType::String, false, false),
    col("expected_revision", PhoenixColumnType::Int, false, false),
    col("rollback_token", PhoenixColumnType::String, false, false),
    col("proposal_json", PhoenixColumnType::Json, false, false),
    col("decision_json", PhoenixColumnType::Json, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const HNSW_INDEX: &[PhoenixColumnSpec] = &[
    col("dim", PhoenixColumnType::Int, false, true),
    col("version", PhoenixColumnType::Int, false, false),
    col("bytes", PhoenixColumnType::Json, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const DOCID_MAP: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::Int, false, true),
    col("docid", PhoenixColumnType::String, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const CHUNKID_MAP: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::Int, false, true),
    col("chunk_key", PhoenixColumnType::String, false, false),
    col("doc_id", PhoenixColumnType::String, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const CHUNKS: &[PhoenixColumnSpec] = &[
    col("chunk_id", PhoenixColumnType::Int, false, true),
    col("doc_id", PhoenixColumnType::String, false, false),
    col("level", PhoenixColumnType::Int, false, false),
    col("start", PhoenixColumnType::Int, false, false),
    col("end", PhoenixColumnType::Int, false, false),
    col("text", PhoenixColumnType::String, false, false),
    col("parent_id", PhoenixColumnType::Int, true, false),
    col("scope_narrative", PhoenixColumnType::String, true, false),
    col("scope_folder", PhoenixColumnType::String, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const RAPTOR_NODES: &[PhoenixColumnSpec] = &[
    col("node_id", PhoenixColumnType::Int, false, true),
    col("doc_id", PhoenixColumnType::String, false, false),
    col("node_type", PhoenixColumnType::Int, false, false),
    col("level", PhoenixColumnType::Int, false, false),
    col("start", PhoenixColumnType::Int, true, false),
    col("end", PhoenixColumnType::Int, true, false),
    col("text", PhoenixColumnType::String, false, false),
    col("vector", PhoenixColumnType::Json, true, false),
    col("parent_id", PhoenixColumnType::Int, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const RAPTOR_EDGES: &[PhoenixColumnSpec] = &[
    col("parent_id", PhoenixColumnType::Int, false, true),
    col("child_id", PhoenixColumnType::Int, false, true),
    col("doc_id", PhoenixColumnType::String, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const EPISODES: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("scope_id", PhoenixColumnType::String, false, false),
    col("note_id", PhoenixColumnType::String, true, false),
    col("ts", PhoenixColumnType::Int, false, false),
    col("boundary_doc_id", PhoenixColumnType::String, true, false),
    col("boundary_id", PhoenixColumnType::Int, true, false),
    col("action_type", PhoenixColumnType::String, false, false),
    col("target_id", PhoenixColumnType::String, false, false),
    col("target_kind", PhoenixColumnType::String, false, false),
    col("payload", PhoenixColumnType::Json, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
];

const SPANS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("world_id", PhoenixColumnType::String, true, false),
    col("note_id", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("start", PhoenixColumnType::Int, true, false),
    col("end", PhoenixColumnType::Int, true, false),
    col("text", PhoenixColumnType::String, true, false),
    col("content_hash", PhoenixColumnType::String, true, false),
    col("span_kind", PhoenixColumnType::String, true, false),
    col("status", PhoenixColumnType::String, true, false),
    col("created_by", PhoenixColumnType::String, true, false),
    col("created_at", PhoenixColumnType::Int, true, false),
    col("updated_at", PhoenixColumnType::Int, true, false),
];

const WORMHOLES: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("src_span_id", PhoenixColumnType::String, true, false),
    col("dst_span_id", PhoenixColumnType::String, true, false),
    col("mode", PhoenixColumnType::String, true, false),
    col("confidence", PhoenixColumnType::Float, true, false),
    col("rationale", PhoenixColumnType::String, true, false),
    col("wormhole_type", PhoenixColumnType::String, true, false),
    col("bidirectional", PhoenixColumnType::Bool, true, false),
    col("created_at", PhoenixColumnType::Int, true, false),
    col("updated_at", PhoenixColumnType::Int, true, false),
];

const SPAN_MENTIONS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("span_id", PhoenixColumnType::String, true, false),
    col(
        "candidate_entity_id",
        PhoenixColumnType::String,
        true,
        false,
    ),
    col("match_type", PhoenixColumnType::String, true, false),
    col("confidence", PhoenixColumnType::Float, true, false),
    col("ev_frequency", PhoenixColumnType::Float, true, false),
    col("ev_capital_ratio", PhoenixColumnType::Float, true, false),
    col("ev_context_score", PhoenixColumnType::Float, true, false),
    col("ev_cooccurrence", PhoenixColumnType::Float, true, false),
    col("status", PhoenixColumnType::String, true, false),
    col("created_at", PhoenixColumnType::Int, true, false),
    col("updated_at", PhoenixColumnType::Int, true, false),
];

const NETWORK_INSTANCE: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("name", PhoenixColumnType::String, true, false),
    col("schema_id", PhoenixColumnType::String, true, false),
    col("network_kind", PhoenixColumnType::String, true, false),
    col("network_subtype", PhoenixColumnType::String, true, false),
    col("root_folder_id", PhoenixColumnType::String, true, false),
    col("root_entity_id", PhoenixColumnType::String, true, false),
    col("namespace", PhoenixColumnType::String, true, false),
    col("description", PhoenixColumnType::String, true, false),
    col("tags", PhoenixColumnType::Json, true, false),
    col("member_count", PhoenixColumnType::Int, true, false),
    col("relationship_count", PhoenixColumnType::Int, true, false),
    col("max_depth", PhoenixColumnType::Int, true, false),
    col("created_at", PhoenixColumnType::Int, true, false),
    col("updated_at", PhoenixColumnType::Int, true, false),
    col("group_id", PhoenixColumnType::String, true, false),
    col("scope_type", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
];

const NETWORK_MEMBERSHIP: &[PhoenixColumnSpec] = &[
    col("network_id", PhoenixColumnType::String, false, true),
    col("entity_id", PhoenixColumnType::String, false, true),
    col("x", PhoenixColumnType::Float, true, false),
    col("y", PhoenixColumnType::Float, true, false),
    col("fixed", PhoenixColumnType::Bool, true, false),
];

const NETWORK_RELATIONSHIP: &[PhoenixColumnSpec] = &[
    col("network_id", PhoenixColumnType::String, false, true),
    col("relationship_id", PhoenixColumnType::String, false, true),
    col("source_entity_id", PhoenixColumnType::String, false, false),
    col("target_entity_id", PhoenixColumnType::String, false, false),
];

const DISCOVERY_CANDIDATES: &[PhoenixColumnSpec] = &[
    col("token", PhoenixColumnType::String, false, true),
    col("kind", PhoenixColumnType::Int, true, false),
    col("score", PhoenixColumnType::Float, true, false),
    col("status", PhoenixColumnType::Int, true, false),
    col("last_seen", PhoenixColumnType::Int, true, false),
    col("first_seen", PhoenixColumnType::Int, true, false),
    col("count", PhoenixColumnType::Int, true, false),
];

const ENTITY_CARDS: &[PhoenixColumnSpec] = &[
    col("entity_id", PhoenixColumnType::String, false, true),
    col("card_id", PhoenixColumnType::String, false, true),
    col("name", PhoenixColumnType::String, true, false),
    col("color", PhoenixColumnType::String, true, false),
    col("icon", PhoenixColumnType::String, true, false),
    col("display_order", PhoenixColumnType::Int, true, false),
    col("is_collapsed", PhoenixColumnType::Bool, true, false),
    col("created_at", PhoenixColumnType::Int, true, false),
    col("updated_at", PhoenixColumnType::Int, true, false),
];

const FOLDER_SCHEMAS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("entity_kind", PhoenixColumnType::String, true, false),
    col("subtype", PhoenixColumnType::String, true, false),
    col("name", PhoenixColumnType::String, true, false),
    col("description", PhoenixColumnType::String, true, false),
    col("allowed_subfolders", PhoenixColumnType::Json, true, false),
    col("allowed_note_types", PhoenixColumnType::Json, true, false),
    col("is_vault_root", PhoenixColumnType::Bool, true, false),
    col("container_only", PhoenixColumnType::Bool, true, false),
    col(
        "propagate_kind_to_children",
        PhoenixColumnType::Bool,
        true,
        false,
    ),
    col("icon", PhoenixColumnType::String, true, false),
    col("is_system", PhoenixColumnType::Bool, true, false),
    col("created_at", PhoenixColumnType::Int, true, false),
    col("updated_at", PhoenixColumnType::Int, true, false),
];

const SCOPED_DOCUMENTS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("scope_folder_id", PhoenixColumnType::String, false, false),
    col("narrative_id", PhoenixColumnType::String, false, false),
    col("namespace", PhoenixColumnType::String, false, false),
    col("document_key", PhoenixColumnType::String, false, false),
    col("payload", PhoenixColumnType::Json, false, false),
    col(
        "seeded_from_scope_folder_id",
        PhoenixColumnType::String,
        true,
        false,
    ),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const SCOPED_ENTITY_FIELDS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("entity_id", PhoenixColumnType::String, false, false),
    col("scope_folder_id", PhoenixColumnType::String, false, false),
    col("narrative_id", PhoenixColumnType::String, false, false),
    col("field_key", PhoenixColumnType::String, false, false),
    col("value_json", PhoenixColumnType::Json, false, false),
    col(
        "seeded_from_scope_folder_id",
        PhoenixColumnType::String,
        true,
        false,
    ),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const SCOPED_DEFINITIONS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("narrative_id", PhoenixColumnType::String, false, false),
    col("namespace", PhoenixColumnType::String, false, false),
    col("definition_key", PhoenixColumnType::String, false, false),
    col("payload", PhoenixColumnType::Json, false, false),
    col("created_at", PhoenixColumnType::Int, false, false),
    col("updated_at", PhoenixColumnType::Int, false, false),
];

const BLOCKS: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("note_id", PhoenixColumnType::String, true, false),
    col("ord", PhoenixColumnType::Int, true, false),
    col("text", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("created_at", PhoenixColumnType::Int, true, false),
];

const DOCUMENT_BOUNDARIES: &[PhoenixColumnSpec] = &[
    col("doc_id", PhoenixColumnType::String, false, true),
    col("boundary_id", PhoenixColumnType::Int, false, true),
    col("kind", PhoenixColumnType::String, false, false),
    col("depth", PhoenixColumnType::Int, false, false),
    col("label", PhoenixColumnType::String, true, false),
    col("ordinal", PhoenixColumnType::Int, false, false),
    col("parent_boundary_id", PhoenixColumnType::Int, true, false),
    col("note_id", PhoenixColumnType::String, true, false),
    col("start_char", PhoenixColumnType::Int, false, false),
    col("end_char", PhoenixColumnType::Int, true, false),
    col("created_at", PhoenixColumnType::Int, false, false),
];

const GRAPH_VERTICES: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("document_id", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("value", PhoenixColumnType::Json, false, false),
    col("weight", PhoenixColumnType::Int, false, false),
    col("attributes", PhoenixColumnType::Json, false, false),
];

const GRAPH_EDGES: &[PhoenixColumnSpec] = &[
    col("source_id", PhoenixColumnType::String, false, true),
    col("target_id", PhoenixColumnType::String, false, true),
    col("document_id", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("valid_from_doc", PhoenixColumnType::String, true, false),
    col("valid_from_boundary", PhoenixColumnType::Int, true, false),
    col("valid_to_doc", PhoenixColumnType::String, true, false),
    col("valid_to_boundary", PhoenixColumnType::Int, true, false),
    col("assertion_kind", PhoenixColumnType::String, true, false),
    col("weight", PhoenixColumnType::Int, false, false),
    col("attributes", PhoenixColumnType::Json, false, false),
    col("data", PhoenixColumnType::Json, true, false),
    col("edge_type", PhoenixColumnType::String, false, false),
];

const GRAPH_CANDIDATE_EDGES: &[PhoenixColumnSpec] = &[
    col("source_id", PhoenixColumnType::String, false, true),
    col("target_id", PhoenixColumnType::String, false, true),
    col("edge_type", PhoenixColumnType::String, false, true),
    col("document_id", PhoenixColumnType::String, true, false),
    col("narrative_id", PhoenixColumnType::String, true, false),
    col("valid_from_doc", PhoenixColumnType::String, true, false),
    col("valid_from_boundary", PhoenixColumnType::Int, true, false),
    col("valid_to_doc", PhoenixColumnType::String, true, false),
    col("valid_to_boundary", PhoenixColumnType::Int, true, false),
    col("assertion_kind", PhoenixColumnType::String, true, false),
    col("weight", PhoenixColumnType::Int, false, false),
    col("attributes", PhoenixColumnType::Json, false, false),
    col("data", PhoenixColumnType::Json, true, false),
];

const GRAPH_NODE_INDEX: &[PhoenixColumnSpec] = &[
    col("id", PhoenixColumnType::String, false, true),
    col("idx", PhoenixColumnType::Int, false, false),
];

const GRAPH_PROPERTIES: &[PhoenixColumnSpec] = &[
    col("owner_id", PhoenixColumnType::String, false, true),
    col("owner_type", PhoenixColumnType::String, false, true),
    col("key", PhoenixColumnType::String, false, true),
    col("valid_from", PhoenixColumnType::Int, false, true),
    col("value_type", PhoenixColumnType::String, false, false),
    col("value_blob", PhoenixColumnType::Json, false, false),
    col("valid_until", PhoenixColumnType::Int, true, false),
    col("txn_id", PhoenixColumnType::Int, false, false),
];

const GRAPH_VERTEX_LABELS: &[PhoenixColumnSpec] = &[
    col("vertex_id", PhoenixColumnType::String, false, true),
    col("label", PhoenixColumnType::String, false, true),
];

const GRAPH_NAMED_RULES: &[PhoenixColumnSpec] = &[
    col("name", PhoenixColumnType::String, false, true),
    col("query_json", PhoenixColumnType::Json, false, false),
    col("materialized", PhoenixColumnType::Bool, false, false),
    col("last_run", PhoenixColumnType::Int, true, false),
    col("invalidated", PhoenixColumnType::Bool, false, false),
];

const GRAPH_RULE_RESULTS: &[PhoenixColumnSpec] = &[
    col("rule_name", PhoenixColumnType::String, false, true),
    col("created_at", PhoenixColumnType::Int, false, true),
    col("row_json", PhoenixColumnType::Json, false, true),
];

pub const ALL_RELATIONS: &[PhoenixRelationSpec] = &[
    PhoenixRelationSpec::new("phoenix_schema_state", PHOENIX_SCHEMA_STATE),
    PhoenixRelationSpec::new("phoenix_sessions", PHOENIX_SESSIONS),
    PhoenixRelationSpec::new("phoenix_commits", PHOENIX_COMMITS),
    PhoenixRelationSpec::new("phoenix_ingest_log", PHOENIX_INGEST_LOG),
    PhoenixRelationSpec::new("phoenix_query_log", PHOENIX_QUERY_LOG),
    PhoenixRelationSpec::new("notes", NOTES),
    PhoenixRelationSpec::new("entities", ENTITIES),
    PhoenixRelationSpec::new("edges", EDGES),
    PhoenixRelationSpec::new("folders", FOLDERS),
    PhoenixRelationSpec::new("threads", THREADS),
    PhoenixRelationSpec::new("thread_messages", THREAD_MESSAGES),
    PhoenixRelationSpec::new("memories", MEMORIES),
    PhoenixRelationSpec::new("memory_threads", MEMORY_THREADS),
    PhoenixRelationSpec::new("om_records", OM_RECORDS),
    PhoenixRelationSpec::new("om_generations", OM_GENERATIONS),
    PhoenixRelationSpec::new("om_graph_index", OM_GRAPH_INDEX),
    PhoenixRelationSpec::new("semantic_vectors", SEMANTIC_VECTORS),
    PhoenixRelationSpec::new("semantic_documents", SEMANTIC_DOCUMENTS),
    PhoenixRelationSpec::new("semantic_node_prototypes", SEMANTIC_NODE_PROTOTYPES),
    PhoenixRelationSpec::new("workspace_artifacts", WORKSPACE_ARTIFACTS),
    PhoenixRelationSpec::new("chat_runs", CHAT_RUNS),
    PhoenixRelationSpec::new("chat_run_events", CHAT_RUN_EVENTS),
    PhoenixRelationSpec::new("chat_tool_calls", CHAT_TOOL_CALLS),
    PhoenixRelationSpec::new("chat_approval_requests", CHAT_APPROVAL_REQUESTS),
    PhoenixRelationSpec::new("hnsw_index", HNSW_INDEX),
    PhoenixRelationSpec::new("docid_map", DOCID_MAP),
    PhoenixRelationSpec::new("chunkid_map", CHUNKID_MAP),
    PhoenixRelationSpec::new("chunks", CHUNKS),
    PhoenixRelationSpec::new("raptor_nodes", RAPTOR_NODES),
    PhoenixRelationSpec::new("raptor_edges", RAPTOR_EDGES),
    PhoenixRelationSpec::new("episodes", EPISODES),
    PhoenixRelationSpec::new("spans", SPANS),
    PhoenixRelationSpec::new("wormholes", WORMHOLES),
    PhoenixRelationSpec::new("span_mentions", SPAN_MENTIONS),
    PhoenixRelationSpec::new("network_instance", NETWORK_INSTANCE),
    PhoenixRelationSpec::new("network_membership", NETWORK_MEMBERSHIP),
    PhoenixRelationSpec::new("network_relationship", NETWORK_RELATIONSHIP),
    PhoenixRelationSpec::new("discovery_candidates", DISCOVERY_CANDIDATES),
    PhoenixRelationSpec::new("entity_cards", ENTITY_CARDS),
    PhoenixRelationSpec::new("folder_schemas", FOLDER_SCHEMAS),
    PhoenixRelationSpec::new("scoped_documents", SCOPED_DOCUMENTS),
    PhoenixRelationSpec::new("scoped_entity_fields", SCOPED_ENTITY_FIELDS),
    PhoenixRelationSpec::new("scoped_definitions", SCOPED_DEFINITIONS),
    PhoenixRelationSpec::new("blocks", BLOCKS),
    PhoenixRelationSpec::new("document_boundaries", DOCUMENT_BOUNDARIES),
    PhoenixRelationSpec::new("graph_vertices", GRAPH_VERTICES),
    PhoenixRelationSpec::new("graph_edges", GRAPH_EDGES),
    PhoenixRelationSpec::new("graph_candidate_edges", GRAPH_CANDIDATE_EDGES),
    PhoenixRelationSpec::new("graph_node_index", GRAPH_NODE_INDEX),
    PhoenixRelationSpec::new("graph_properties", GRAPH_PROPERTIES),
    PhoenixRelationSpec::new("graph_vertex_labels", GRAPH_VERTEX_LABELS),
    PhoenixRelationSpec::new("graph_named_rules", GRAPH_NAMED_RULES),
    PhoenixRelationSpec::new("graph_rule_results", GRAPH_RULE_RESULTS),
];

pub const DERIVED_SNAPSHOT_RELATIONS: &[&str] = &[
    "semantic_vectors",
    "semantic_documents",
    "semantic_node_prototypes",
    "hnsw_index",
    "docid_map",
    "chunkid_map",
    "chunks",
    "raptor_nodes",
    "raptor_edges",
    "episodes",
    "blocks",
    "document_boundaries",
    "graph_vertices",
    "graph_edges",
    "graph_candidate_edges",
    "graph_node_index",
    "graph_properties",
    "graph_vertex_labels",
    "graph_named_rules",
    "graph_rule_results",
];

pub const CONTENT_SNAPSHOT_RELATIONS: &[&str] = &[
    "phoenix_schema_state",
    "notes",
    "entities",
    "edges",
    "folders",
    "threads",
    "thread_messages",
    "memories",
    "memory_threads",
    "om_records",
    "om_generations",
    "om_graph_index",
    "workspace_artifacts",
    "chat_runs",
    "chat_run_events",
    "chat_tool_calls",
    "chat_approval_requests",
    "spans",
    "wormholes",
    "span_mentions",
    "network_instance",
    "network_membership",
    "network_relationship",
    "discovery_candidates",
    "entity_cards",
    "folder_schemas",
    "scoped_documents",
    "scoped_entity_fields",
    "scoped_definitions",
];

pub const CORE_RELATIONS: &[&str] = &[
    "notes",
    "entities",
    "edges",
    "folders",
    "threads",
    "thread_messages",
    "om_graph_index",
    "semantic_vectors",
    "semantic_documents",
    "semantic_node_prototypes",
    "spans",
    "chunks",
    "raptor_nodes",
    "workspace_artifacts",
    "scoped_documents",
    "scoped_entity_fields",
    "scoped_definitions",
    "document_boundaries",
    "graph_vertices",
    "graph_edges",
    "graph_candidate_edges",
    "graph_properties",
];
