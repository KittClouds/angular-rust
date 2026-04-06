use std::borrow::Cow;
use std::cell::RefCell;

use phoenix_runtime_wasm::{
    AnalyzeTextRequestView, IngestDocumentView, IngestRequestView, PhoenixRuntime,
    QueryRequestView, ScanRequestView, ScopeKeyView, SnapshotPartition, StructureRequestView,
};
#[cfg(target_arch = "wasm32")]
use phoenix_types::SnapshotPolicy;
use phoenix_types::{
    AnalyzeTextBinaryRequestHeader, CommitRequest, CreateSessionRequest, Diagnostic,
    EmbedUpsertBinaryRequestHeader, GraphDeltaRequest, IngestBinaryRequestHeader,
    IngestDocumentBinaryRecord, PacketHeader, PacketKind, QueryBinaryRequestHeader, QueryTarget,
    RebuildRequest, RuntimeInitRequest, ScanBinaryRequestHeader, SessionId, SessionStateRequest,
    SessionStatsRequest, StoreCommandRequest, StructureBinaryRequestHeader, TemporalMarker,
    BINARY_REQUEST_LAYOUT_VERSION,
};
use serde::Deserialize;
#[cfg(target_arch = "wasm32")]
use serde::Serialize;

#[cfg(target_arch = "wasm32")]
mod opfs;

pub const PHOENIX_PROTOCOL_VERSION: u32 = 6;
pub const DEFAULT_PACKET_REGION_SIZE: usize = 64 * 1024;

thread_local! {
    static RUNTIME: RefCell<Option<PhoenixRuntime>> = const { RefCell::new(None) };
    #[cfg(target_arch = "wasm32")]
    static OPFS_STATUS: RefCell<OpfsStatus> = RefCell::new(OpfsStatus::idle());
}

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct OpfsStatus {
    phase: &'static str,
    operation: &'static str,
    snapshot_bytes: usize,
    recovered_from_backup: bool,
    message: String,
}

#[cfg(target_arch = "wasm32")]
impl OpfsStatus {
    const fn idle() -> Self {
        Self {
            phase: "idle",
            operation: "none",
            snapshot_bytes: 0,
            recovered_from_backup: false,
            message: String::new(),
        }
    }

    fn pending(operation: &'static str, message: &str) -> Self {
        Self {
            phase: "pending",
            operation,
            snapshot_bytes: 0,
            recovered_from_backup: false,
            message: message.to_owned(),
        }
    }

    fn success(
        operation: &'static str,
        snapshot_bytes: usize,
        recovered_from_backup: bool,
        message: &str,
    ) -> Self {
        Self {
            phase: "succeeded",
            operation,
            snapshot_bytes,
            recovered_from_backup,
            message: message.to_owned(),
        }
    }

    fn failed(operation: &'static str, message: &str) -> Self {
        Self {
            phase: "failed",
            operation,
            snapshot_bytes: 0,
            recovered_from_backup: false,
            message: message.to_owned(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BorrowedScopeKey<'a> {
    #[serde(borrow)]
    world_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    narrative_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    folder_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    folder_path: Option<Cow<'a, str>>,
}

impl<'a> BorrowedScopeKey<'a> {
    fn as_view(&self) -> ScopeKeyView<'_> {
        ScopeKeyView {
            world_id: self.world_id.as_deref(),
            narrative_id: self.narrative_id.as_deref(),
            folder_id: self.folder_id.as_deref(),
            folder_path: self.folder_path.as_deref(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BorrowedQueryRequest<'a> {
    #[serde(borrow)]
    session_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    query: Cow<'a, str>,
    scope: BorrowedScopeKey<'a>,
    targets: Vec<QueryTarget>,
    limit: Option<usize>,
    temporal: Option<TemporalMarker>,
    #[serde(default)]
    include_candidate_graph: bool,
}

#[derive(Debug)]
struct BorrowedEmbedUpsertRecord<'a> {
    span_id: &'a str,
    values: &'a [f32],
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BorrowedAnalyzeTextRequest<'a> {
    #[serde(borrow)]
    text: Cow<'a, str>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BorrowedIngestDocument<'a> {
    #[serde(borrow)]
    document_id: Cow<'a, str>,
    #[serde(borrow)]
    note_id: Option<Cow<'a, str>>,
    #[serde(borrow)]
    title: Cow<'a, str>,
    #[serde(borrow)]
    text: Cow<'a, str>,
    scope: BorrowedScopeKey<'a>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BorrowedIngestRequest<'a> {
    #[serde(borrow)]
    session_id: Option<Cow<'a, str>>,
    documents: Vec<BorrowedIngestDocument<'a>>,
    commit: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BorrowedScanRequest<'a> {
    #[serde(borrow)]
    text: Cow<'a, str>,
    scope: BorrowedScopeKey<'a>,
    #[serde(borrow)]
    session_id: Option<Cow<'a, str>>,
    resolver_seed: Vec<phoenix_types::ResolverEntitySeed>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BorrowedStructureRequest<'a> {
    #[serde(borrow)]
    text: Cow<'a, str>,
    scan: phoenix_types::ScanArtifact,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SnapshotExportRequest {
    partition: Option<String>,
}

pub fn packet_header_size() -> usize {
    PacketHeader::BYTE_LEN
}

#[cfg(target_arch = "wasm32")]
fn opfs_status_json() -> String {
    OPFS_STATUS.with(|cell| serde_json::to_string(&*cell.borrow()).expect("serialize opfs status"))
}

#[cfg(target_arch = "wasm32")]
fn set_opfs_status(status: OpfsStatus) {
    OPFS_STATUS.with(|cell| {
        *cell.borrow_mut() = status;
    });
}

#[cfg(target_arch = "wasm32")]
fn should_auto_save_on_commit() -> bool {
    RUNTIME.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|runtime| runtime.config.snapshot_policy == SnapshotPolicy::OnCommit)
            .unwrap_or(false)
    })
}

fn decode_json_borrowed<'a, T>(bytes: &'a [u8]) -> Result<T, String>
where
    T: Deserialize<'a>,
{
    serde_json::from_slice(bytes).map_err(|error| error.to_string())
}

fn parse_wire_header<T: Copy>(bytes: &[u8], expected_len: usize) -> Result<T, String> {
    if bytes.len() < expected_len {
        return Err("binary request header is truncated".to_owned());
    }
    Ok(unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const T) })
}

fn read_le_u32(bytes: [u8; 4]) -> u32 {
    u32::from_le_bytes(bytes)
}

fn read_f32_slice(bytes: &[u8]) -> Result<&[f32], String> {
    let (prefix, values, suffix) = unsafe { bytes.align_to::<f32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
        return Err("vector block is not aligned to f32".to_owned());
    }
    Ok(values)
}

fn read_string_from_arena<'a>(
    arena: &'a [u8],
    offset: u32,
    len: u32,
) -> Result<Option<&'a str>, String> {
    if len == 0 {
        return Ok(None);
    }
    let start = offset as usize;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| "string ref overflow".to_owned())?;
    let slice = arena
        .get(start..end)
        .ok_or_else(|| "string ref exceeds arena".to_owned())?;
    let text = std::str::from_utf8(slice).map_err(|error| error.to_string())?;
    Ok(Some(text))
}

fn read_json_from_arena<T>(arena: &[u8], offset: u32, len: u32) -> Result<Option<T>, String>
where
    T: serde::de::DeserializeOwned,
{
    let Some(text) = read_string_from_arena(arena, offset, len)? else {
        return Ok(None);
    };
    serde_json::from_str(text)
        .map(Some)
        .map_err(|error| error.to_string())
}

fn query_targets_from_flags(flags: u32) -> Vec<QueryTarget> {
    let mut targets = Vec::new();
    if flags & phoenix_types::REQUEST_FLAG_TARGET_CHUNKS != 0 {
        targets.push(QueryTarget::Chunks);
    }
    if flags & phoenix_types::REQUEST_FLAG_TARGET_NODES != 0 {
        targets.push(QueryTarget::Nodes);
    }
    if flags & phoenix_types::REQUEST_FLAG_TARGET_GRAPH != 0 {
        targets.push(QueryTarget::Graph);
    }
    if flags & phoenix_types::REQUEST_FLAG_TARGET_SEMANTIC != 0 {
        targets.push(QueryTarget::Semantic);
    }
    targets
}

fn with_query_json_request<T>(
    bytes: &[u8],
    op: impl FnOnce(QueryRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let request: BorrowedQueryRequest<'_> = decode_json_borrowed(bytes)?;
    let session_id = request
        .session_id
        .as_deref()
        .map(|value| SessionId(value.to_owned()));
    let view = QueryRequestView {
        session_id,
        query: request.query.as_ref(),
        scope: request.scope.as_view(),
        targets: &request.targets,
        limit: request.limit,
        temporal: request.temporal.as_ref(),
        semantic_query_vector: None,
        include_candidate_graph: request.include_candidate_graph,
    };
    op(view)
}

fn with_query_binary_request<T>(
    bytes: &[u8],
    op: impl FnOnce(QueryRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let header: QueryBinaryRequestHeader =
        parse_wire_header(bytes, QueryBinaryRequestHeader::BYTE_LEN)?;
    if read_le_u32(header.version) != BINARY_REQUEST_LAYOUT_VERSION {
        return Err("unsupported query binary request version".to_owned());
    }
    let arena_offset = read_le_u32(header.arena_offset) as usize;
    let arena_len = read_le_u32(header.arena_len) as usize;
    let arena = bytes
        .get(arena_offset..arena_offset + arena_len)
        .ok_or_else(|| "query request arena exceeds payload".to_owned())?;
    let session_id = read_string_from_arena(
        arena,
        read_le_u32(header.session_offset),
        read_le_u32(header.session_len),
    )?
    .map(|value| SessionId(value.to_owned()));
    let query = read_string_from_arena(
        arena,
        read_le_u32(header.query_offset),
        read_le_u32(header.query_len),
    )?
    .ok_or_else(|| "query text is required".to_owned())?;
    let scope = ScopeKeyView {
        world_id: read_string_from_arena(
            arena,
            read_le_u32(header.world_offset),
            read_le_u32(header.world_len),
        )?,
        narrative_id: read_string_from_arena(
            arena,
            read_le_u32(header.narrative_offset),
            read_le_u32(header.narrative_len),
        )?,
        folder_id: read_string_from_arena(
            arena,
            read_le_u32(header.folder_id_offset),
            read_le_u32(header.folder_id_len),
        )?,
        folder_path: read_string_from_arena(
            arena,
            read_le_u32(header.folder_path_offset),
            read_le_u32(header.folder_path_len),
        )?,
    };
    let temporal = read_json_from_arena::<TemporalMarker>(
        arena,
        read_le_u32(header.temporal_offset),
        read_le_u32(header.temporal_len),
    )?;
    let vector_dim = read_le_u32(header.query_vector_dim) as usize;
    let vector_len = read_le_u32(header.query_vector_len) as usize;
    let semantic_query_vector = if vector_len == 0 {
        None
    } else {
        let vector_offset = read_le_u32(header.query_vector_offset) as usize;
        let vector_bytes_len = vector_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| "query vector length overflow".to_owned())?;
        let vector_bytes = bytes
            .get(vector_offset..vector_offset + vector_bytes_len)
            .ok_or_else(|| "query vector exceeds payload".to_owned())?;
        let vector = read_f32_slice(vector_bytes)?;
        if vector_dim != 0 && vector_dim != vector.len() {
            return Err("query vector dimension does not match vector length".to_owned());
        }
        Some(vector)
    };
    let flags = read_le_u32(header.flags);
    let targets = query_targets_from_flags(flags);
    let raw_limit = read_le_u32(header.limit);
    let limit = if raw_limit == u32::MAX {
        None
    } else {
        Some(raw_limit as usize)
    };
    let view = QueryRequestView {
        session_id,
        query,
        scope,
        targets: &targets,
        limit,
        temporal: temporal.as_ref(),
        semantic_query_vector,
        include_candidate_graph: flags & phoenix_types::REQUEST_FLAG_INCLUDE_CANDIDATE_GRAPH != 0,
    };
    op(view)
}

fn parse_embed_upsert_binary_request(
    bytes: &[u8],
) -> Result<Vec<BorrowedEmbedUpsertRecord<'_>>, String> {
    let header: EmbedUpsertBinaryRequestHeader =
        parse_wire_header(bytes, EmbedUpsertBinaryRequestHeader::BYTE_LEN)?;
    if read_le_u32(header.version) != BINARY_REQUEST_LAYOUT_VERSION {
        return Err("unsupported embed upsert binary request version".to_owned());
    }
    let count = read_le_u32(header.count) as usize;
    let dim = read_le_u32(header.dim) as usize;
    let table_offset = EmbedUpsertBinaryRequestHeader::BYTE_LEN;
    let table_len = count
        .checked_mul(phoenix_types::StringRefRecord::BYTE_LEN)
        .ok_or_else(|| "embed upsert record table overflow".to_owned())?;
    let table = bytes
        .get(table_offset..table_offset + table_len)
        .ok_or_else(|| "embed upsert record table exceeds payload".to_owned())?;
    let vector_offset = table_offset + table_len;
    let vector_bytes_len = count
        .checked_mul(dim)
        .and_then(|size| size.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| "embed upsert vector block overflow".to_owned())?;
    let vector_bytes = bytes
        .get(vector_offset..vector_offset + vector_bytes_len)
        .ok_or_else(|| "embed upsert vector block exceeds payload".to_owned())?;
    let arena_offset = read_le_u32(header.arena_offset) as usize;
    let arena = bytes
        .get(arena_offset..)
        .ok_or_else(|| "embed upsert arena exceeds payload".to_owned())?;
    let vector_values = read_f32_slice(vector_bytes)?;
    let mut records = Vec::with_capacity(count);
    for index in 0..count {
        let start = index * phoenix_types::StringRefRecord::BYTE_LEN;
        let end = start + phoenix_types::StringRefRecord::BYTE_LEN;
        let record: phoenix_types::StringRefRecord =
            parse_wire_header(&table[start..end], phoenix_types::StringRefRecord::BYTE_LEN)?;
        let span_id = read_string_from_arena(arena, record.offset, record.len)?
            .ok_or_else(|| "embed upsert span id is required".to_owned())?;
        let vector_start = index
            .checked_mul(dim)
            .ok_or_else(|| "embed upsert vector offset overflow".to_owned())?;
        let values = vector_values
            .get(vector_start..vector_start + dim)
            .ok_or_else(|| "embed upsert vector slice exceeds payload".to_owned())?;
        records.push(BorrowedEmbedUpsertRecord { span_id, values });
    }
    Ok(records)
}

fn with_analyze_json_request<T>(
    bytes: &[u8],
    op: impl FnOnce(AnalyzeTextRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let request: BorrowedAnalyzeTextRequest<'_> = decode_json_borrowed(bytes)?;
    op(AnalyzeTextRequestView {
        text: request.text.as_ref(),
    })
}

fn with_analyze_binary_request<T>(
    bytes: &[u8],
    op: impl FnOnce(AnalyzeTextRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let header: AnalyzeTextBinaryRequestHeader =
        parse_wire_header(bytes, AnalyzeTextBinaryRequestHeader::BYTE_LEN)?;
    if read_le_u32(header.version) != BINARY_REQUEST_LAYOUT_VERSION {
        return Err("unsupported analyzeText binary request version".to_owned());
    }
    let arena_offset = read_le_u32(header.arena_offset) as usize;
    let arena_len = read_le_u32(header.arena_len) as usize;
    let arena = bytes
        .get(arena_offset..arena_offset + arena_len)
        .ok_or_else(|| "analyzeText arena exceeds payload".to_owned())?;
    let text = read_string_from_arena(
        arena,
        read_le_u32(header.text_offset),
        read_le_u32(header.text_len),
    )?
    .ok_or_else(|| "analyzeText text is required".to_owned())?;
    op(AnalyzeTextRequestView { text })
}

fn with_ingest_json_request<T>(
    bytes: &[u8],
    op: impl FnOnce(IngestRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let request: BorrowedIngestRequest<'_> = decode_json_borrowed(bytes)?;
    let session_id = request
        .session_id
        .as_deref()
        .map(|value| SessionId(value.to_owned()));
    let documents = request
        .documents
        .iter()
        .map(|document| IngestDocumentView {
            document_id: phoenix_types::DocumentId(document.document_id.as_ref().to_owned()),
            note_id: document
                .note_id
                .as_deref()
                .map(|value| phoenix_types::NoteId(value.to_owned())),
            title: document.title.as_ref(),
            text: document.text.as_ref(),
            scope: document.scope.as_view(),
        })
        .collect::<Vec<_>>();
    op(IngestRequestView {
        session_id,
        documents,
        commit: request.commit,
    })
}

fn with_ingest_binary_request<T>(
    bytes: &[u8],
    op: impl FnOnce(IngestRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let header: IngestBinaryRequestHeader =
        parse_wire_header(bytes, IngestBinaryRequestHeader::BYTE_LEN)?;
    if read_le_u32(header.version) != BINARY_REQUEST_LAYOUT_VERSION {
        return Err("unsupported ingest binary request version".to_owned());
    }
    let table_offset = read_le_u32(header.table1_offset) as usize;
    let table_count = read_le_u32(header.table1_count) as usize;
    let table_len = table_count
        .checked_mul(IngestDocumentBinaryRecord::BYTE_LEN)
        .ok_or_else(|| "ingest document table overflow".to_owned())?;
    let table = bytes
        .get(table_offset..table_offset + table_len)
        .ok_or_else(|| "ingest document table exceeds payload".to_owned())?;
    let arena_offset = read_le_u32(header.arena_offset) as usize;
    let arena_len = read_le_u32(header.arena_len) as usize;
    let arena = bytes
        .get(arena_offset..arena_offset + arena_len)
        .ok_or_else(|| "ingest arena exceeds payload".to_owned())?;
    let session_id = read_string_from_arena(
        arena,
        read_le_u32(header.session_offset),
        read_le_u32(header.session_len),
    )?
    .map(|value| SessionId(value.to_owned()));
    let mut documents = Vec::with_capacity(table_count);
    for index in 0..table_count {
        let start = index * IngestDocumentBinaryRecord::BYTE_LEN;
        let end = start + IngestDocumentBinaryRecord::BYTE_LEN;
        let record: IngestDocumentBinaryRecord =
            parse_wire_header(&table[start..end], IngestDocumentBinaryRecord::BYTE_LEN)?;
        let document_id = read_string_from_arena(
            arena,
            read_le_u32(record.document_id_offset),
            read_le_u32(record.document_id_len),
        )?
        .ok_or_else(|| "ingest document id is required".to_owned())?;
        let title = read_string_from_arena(
            arena,
            read_le_u32(record.title_offset),
            read_le_u32(record.title_len),
        )?
        .ok_or_else(|| "ingest title is required".to_owned())?;
        let text = read_string_from_arena(
            arena,
            read_le_u32(record.text_offset),
            read_le_u32(record.text_len),
        )?
        .ok_or_else(|| "ingest text is required".to_owned())?;
        documents.push(IngestDocumentView {
            document_id: phoenix_types::DocumentId(document_id.to_owned()),
            note_id: read_string_from_arena(
                arena,
                read_le_u32(record.note_id_offset),
                read_le_u32(record.note_id_len),
            )?
            .map(|value| phoenix_types::NoteId(value.to_owned())),
            title,
            text,
            scope: ScopeKeyView {
                world_id: read_string_from_arena(
                    arena,
                    read_le_u32(record.world_offset),
                    read_le_u32(record.world_len),
                )?,
                narrative_id: read_string_from_arena(
                    arena,
                    read_le_u32(record.narrative_offset),
                    read_le_u32(record.narrative_len),
                )?,
                folder_id: read_string_from_arena(
                    arena,
                    read_le_u32(record.folder_id_offset),
                    read_le_u32(record.folder_id_len),
                )?,
                folder_path: read_string_from_arena(
                    arena,
                    read_le_u32(record.folder_path_offset),
                    read_le_u32(record.folder_path_len),
                )?,
            },
        });
    }
    op(IngestRequestView {
        session_id,
        documents,
        commit: read_le_u32(header.flags) & phoenix_types::REQUEST_FLAG_COMMIT != 0,
    })
}

fn with_scan_json_request<T>(
    bytes: &[u8],
    op: impl FnOnce(ScanRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let request: BorrowedScanRequest<'_> = decode_json_borrowed(bytes)?;
    let session_id = request
        .session_id
        .as_deref()
        .map(|value| SessionId(value.to_owned()));
    let view = ScanRequestView {
        text: request.text.as_ref(),
        scope: request.scope.as_view(),
        session_id,
        resolver_seed: &request.resolver_seed,
    };
    op(view)
}

fn with_scan_binary_request<T>(
    bytes: &[u8],
    op: impl FnOnce(ScanRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let header: ScanBinaryRequestHeader =
        parse_wire_header(bytes, ScanBinaryRequestHeader::BYTE_LEN)?;
    if read_le_u32(header.version) != BINARY_REQUEST_LAYOUT_VERSION {
        return Err("unsupported scan binary request version".to_owned());
    }
    let arena_offset = read_le_u32(header.arena_offset) as usize;
    let arena_len = read_le_u32(header.arena_len) as usize;
    let arena = bytes
        .get(arena_offset..arena_offset + arena_len)
        .ok_or_else(|| "scan arena exceeds payload".to_owned())?;
    let session_id = read_string_from_arena(
        arena,
        read_le_u32(header.session_offset),
        read_le_u32(header.session_len),
    )?
    .map(|value| SessionId(value.to_owned()));
    let text = read_string_from_arena(
        arena,
        read_le_u32(header.text_offset),
        read_le_u32(header.text_len),
    )?
    .ok_or_else(|| "scan text is required".to_owned())?;
    let resolver_seed = read_json_from_arena::<Vec<phoenix_types::ResolverEntitySeed>>(
        arena,
        read_le_u32(header.resolver_seed_offset),
        read_le_u32(header.resolver_seed_len),
    )?
    .unwrap_or_default();
    let view = ScanRequestView {
        text,
        scope: ScopeKeyView {
            world_id: read_string_from_arena(
                arena,
                read_le_u32(header.world_offset),
                read_le_u32(header.world_len),
            )?,
            narrative_id: read_string_from_arena(
                arena,
                read_le_u32(header.narrative_offset),
                read_le_u32(header.narrative_len),
            )?,
            folder_id: read_string_from_arena(
                arena,
                read_le_u32(header.folder_id_offset),
                read_le_u32(header.folder_id_len),
            )?,
            folder_path: read_string_from_arena(
                arena,
                read_le_u32(header.folder_path_offset),
                read_le_u32(header.folder_path_len),
            )?,
        },
        session_id,
        resolver_seed: &resolver_seed,
    };
    op(view)
}

fn with_structure_json_request<T>(
    bytes: &[u8],
    op: impl FnOnce(StructureRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let request: BorrowedStructureRequest<'_> = decode_json_borrowed(bytes)?;
    op(StructureRequestView {
        text: request.text.as_ref(),
        scan: &request.scan,
    })
}

fn with_structure_binary_request<T>(
    bytes: &[u8],
    op: impl FnOnce(StructureRequestView<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let header: StructureBinaryRequestHeader =
        parse_wire_header(bytes, StructureBinaryRequestHeader::BYTE_LEN)?;
    if read_le_u32(header.version) != BINARY_REQUEST_LAYOUT_VERSION {
        return Err("unsupported structure binary request version".to_owned());
    }
    let arena_offset = read_le_u32(header.arena_offset) as usize;
    let arena_len = read_le_u32(header.arena_len) as usize;
    let arena = bytes
        .get(arena_offset..arena_offset + arena_len)
        .ok_or_else(|| "structure arena exceeds payload".to_owned())?;
    let text = read_string_from_arena(
        arena,
        read_le_u32(header.text_offset),
        read_le_u32(header.text_len),
    )?
    .ok_or_else(|| "structure text is required".to_owned())?;
    let scan = read_json_from_arena::<phoenix_types::ScanArtifact>(
        arena,
        read_le_u32(header.scan_offset),
        read_le_u32(header.scan_len),
    )?
    .ok_or_else(|| "structure scan artifact is required".to_owned())?;
    op(StructureRequestView { text, scan: &scan })
}

pub fn process_packet_buffer(buffer: &mut [u8]) -> Result<(), String> {
    if buffer.len() < PacketHeader::BYTE_LEN {
        return Err("buffer too small for packet header".to_owned());
    }

    let header_bytes: [u8; PacketHeader::BYTE_LEN] = buffer[..PacketHeader::BYTE_LEN]
        .try_into()
        .map_err(|_| "invalid packet header".to_owned())?;
    let header = PacketHeader::from_le_bytes(header_bytes);
    let payload_end = PacketHeader::BYTE_LEN + header.payload_len as usize;
    if payload_end > buffer.len() {
        return Err("payload length exceeds packet region".to_owned());
    }

    match header.packet_kind() {
        PacketKind::InitRuntimeRequest => {
            let request: RuntimeInitRequest =
                decode_json(&buffer[PacketHeader::BYTE_LEN..payload_end])?;
            let runtime =
                PhoenixRuntime::open(request.config.clone(), request.storage_path.map(Into::into))
                    .map_err(|error| error.to_string())?;
            let result = runtime.init().map_err(|error| error.to_string())?;
            RUNTIME.with(|cell| {
                *cell.borrow_mut() = Some(runtime);
            });
            write_json_response(
                buffer,
                PacketKind::InitRuntimeResult,
                header.request_id,
                &result,
            )
        }
        PacketKind::CreateSessionRequest => with_runtime(|runtime| {
            let request: CreateSessionRequest =
                decode_json(&buffer[PacketHeader::BYTE_LEN..payload_end])?;
            let result = runtime
                .create_session(request)
                .map_err(|error| error.to_string())?;
            write_json_response(
                buffer,
                PacketKind::CreateSessionResult,
                header.request_id,
                &result,
            )
        }),
        PacketKind::CommitRequest => with_runtime(|runtime| {
            let request: CommitRequest = decode_json(&buffer[PacketHeader::BYTE_LEN..payload_end])?;
            let result = runtime.commit(request).map_err(|error| error.to_string())?;
            write_json_response(buffer, PacketKind::CommitResult, header.request_id, &result)?;
            #[cfg(target_arch = "wasm32")]
            if should_auto_save_on_commit() {
                let _ = phoenix_opfs_save_snapshot();
            }
            Ok(())
        }),
        PacketKind::RebuildRequest => with_runtime(|runtime| {
            let request: RebuildRequest =
                decode_json(&buffer[PacketHeader::BYTE_LEN..payload_end])?;
            let result = runtime
                .rebuild(request)
                .map_err(|error| error.to_string())?;
            write_json_response(
                buffer,
                PacketKind::RebuildResult,
                header.request_id,
                &result,
            )
        }),
        PacketKind::IngestRequest => with_runtime(|runtime| {
            let (should_auto_save_flag, result) = with_ingest_json_request(
                &buffer[PacketHeader::BYTE_LEN..payload_end],
                |request| {
                    let should_auto_save = request.commit;
                    let result = runtime
                        .ingest_view(request)
                        .map_err(|error| error.to_string())?;
                    Ok((should_auto_save, result))
                },
            )?;
            write_json_response(buffer, PacketKind::IngestResult, header.request_id, &result)?;
            #[cfg(not(target_arch = "wasm32"))]
            let _ = should_auto_save_flag;
            #[cfg(target_arch = "wasm32")]
            if should_auto_save_flag && should_auto_save_on_commit() {
                let _ = phoenix_opfs_save_snapshot();
            }
            Ok(())
        }),
        PacketKind::IngestBinaryRequest => with_runtime(|runtime| {
            let (should_auto_save_flag, result) = with_ingest_binary_request(
                &buffer[PacketHeader::BYTE_LEN..payload_end],
                |request| {
                    let should_auto_save = request.commit;
                    let result = runtime
                        .ingest_view(request)
                        .map_err(|error| error.to_string())?;
                    Ok((should_auto_save, result))
                },
            )?;
            write_json_response(buffer, PacketKind::IngestResult, header.request_id, &result)?;
            #[cfg(not(target_arch = "wasm32"))]
            let _ = should_auto_save_flag;
            #[cfg(target_arch = "wasm32")]
            if should_auto_save_flag && should_auto_save_on_commit() {
                let _ = phoenix_opfs_save_snapshot();
            }
            Ok(())
        }),
        PacketKind::QueryRequest => with_runtime(|runtime| {
            let result =
                with_query_json_request(&buffer[PacketHeader::BYTE_LEN..payload_end], |request| {
                    runtime
                        .query_view(request)
                        .map_err(|error| error.to_string())
                })?;
            write_binary_response_with(
                buffer,
                PacketKind::QueryResult,
                header.request_id,
                |payload| {
                    runtime
                        .encode_query_result_into(&result, payload)
                        .map_err(|error| error.to_string())
                },
            )
        }),
        PacketKind::QueryBinaryRequest => with_runtime(|runtime| {
            let result = with_query_binary_request(
                &buffer[PacketHeader::BYTE_LEN..payload_end],
                |request| {
                    runtime
                        .query_view(request)
                        .map_err(|error| error.to_string())
                },
            )?;
            write_binary_response_with(
                buffer,
                PacketKind::QueryResult,
                header.request_id,
                |payload| {
                    runtime
                        .encode_query_result_into(&result, payload)
                        .map_err(|error| error.to_string())
                },
            )
        }),
        PacketKind::EmbedUpsertBinaryRequest => with_runtime(|runtime| {
            let records =
                parse_embed_upsert_binary_request(&buffer[PacketHeader::BYTE_LEN..payload_end])?;
            let now = {
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
            };
            let rows = records
                .iter()
                .map(|record| phoenix_store_cozo::SemanticVectorRow {
                    span_id: record.span_id,
                    values: record.values,
                    model_id: phoenix_store_cozo::SEMANTIC_MODEL_ID,
                    updated_at: now,
                })
                .collect::<Vec<_>>();
            runtime
                .store
                .upsert_semantic_vectors(&rows)
                .map_err(|error| error.to_string())?;
            write_json_response(
                buffer,
                PacketKind::EmbedUpsertResult,
                header.request_id,
                &serde_json::json!({
                    "inserted": rows.len(),
                    "modelId": phoenix_store_cozo::SEMANTIC_MODEL_ID,
                    "dimension": phoenix_store_cozo::SEMANTIC_VECTOR_DIM,
                }),
            )
        }),
        PacketKind::ScanRequest => with_runtime(|runtime| {
            let result =
                with_scan_json_request(&buffer[PacketHeader::BYTE_LEN..payload_end], |request| {
                    Ok(runtime.scan_text_view(request))
                })?;
            write_json_response(buffer, PacketKind::ScanResult, header.request_id, &result)
        }),
        PacketKind::ScanBinaryRequest => with_runtime(|runtime| {
            let result = with_scan_binary_request(
                &buffer[PacketHeader::BYTE_LEN..payload_end],
                |request| Ok(runtime.scan_text_view(request)),
            )?;
            write_json_response(buffer, PacketKind::ScanResult, header.request_id, &result)
        }),
        PacketKind::StructureRequest => with_runtime(|runtime| {
            let result = with_structure_json_request(
                &buffer[PacketHeader::BYTE_LEN..payload_end],
                |request| Ok(runtime.build_structure_view(request)),
            )?;
            write_json_response(
                buffer,
                PacketKind::StructureResult,
                header.request_id,
                &result,
            )
        }),
        PacketKind::StructureBinaryRequest => with_runtime(|runtime| {
            let result = with_structure_binary_request(
                &buffer[PacketHeader::BYTE_LEN..payload_end],
                |request| Ok(runtime.build_structure_view(request)),
            )?;
            write_json_response(
                buffer,
                PacketKind::StructureResult,
                header.request_id,
                &result,
            )
        }),
        PacketKind::AnalyzeTextRequest => with_runtime(|runtime| {
            let result = with_analyze_json_request(
                &buffer[PacketHeader::BYTE_LEN..payload_end],
                |request| Ok(runtime.analyze_text_view(request)),
            )?;
            write_json_response(
                buffer,
                PacketKind::AnalyzeTextResult,
                header.request_id,
                &result,
            )
        }),
        PacketKind::AnalyzeTextBinaryRequest => with_runtime(|runtime| {
            let result = with_analyze_binary_request(
                &buffer[PacketHeader::BYTE_LEN..payload_end],
                |request| Ok(runtime.analyze_text_view(request)),
            )?;
            write_json_response(
                buffer,
                PacketKind::AnalyzeTextResult,
                header.request_id,
                &result,
            )
        }),
        PacketKind::GraphDeltaRequest => with_runtime(|runtime| {
            let request: GraphDeltaRequest =
                decode_json(&buffer[PacketHeader::BYTE_LEN..payload_end])?;
            write_binary_response_with(
                buffer,
                PacketKind::GraphDeltaResult,
                header.request_id,
                |payload| {
                    runtime
                        .graph_delta_binary_into(request, payload)
                        .map_err(|error| error.to_string())
                },
            )
        }),
        PacketKind::SessionStateRequest => with_runtime(|runtime| {
            let request: SessionStateRequest =
                decode_json(&buffer[PacketHeader::BYTE_LEN..payload_end])?;
            write_binary_response_with(
                buffer,
                PacketKind::SessionStateResult,
                header.request_id,
                |payload| {
                    runtime
                        .session_state_binary_into(&request.session_id, payload)
                        .map_err(|error| error.to_string())
                },
            )
        }),
        PacketKind::SessionStatsRequest => with_runtime(|runtime| {
            let request: SessionStatsRequest =
                decode_json(&buffer[PacketHeader::BYTE_LEN..payload_end])?;
            write_binary_response_with(
                buffer,
                PacketKind::SessionStatsResult,
                header.request_id,
                |payload| {
                    runtime
                        .session_stats_binary_into(&request.session_id, payload)
                        .map_err(|error| error.to_string())
                },
            )
        }),
        PacketKind::SnapshotExportRequest => with_runtime(|runtime| {
            let request = if header.payload_len == 0 {
                SnapshotExportRequest { partition: None }
            } else {
                decode_json::<SnapshotExportRequest>(&buffer[PacketHeader::BYTE_LEN..payload_end])?
            };
            let partition = match request.partition.as_deref() {
                Some(value) => SnapshotPartition::from_str(value)
                    .ok_or_else(|| format!("unsupported snapshot partition: {value}"))?,
                None => SnapshotPartition::All,
            };
            let bytes = runtime
                .export_snapshot_partition(partition)
                .map_err(|error| error.to_string())?;
            write_binary_response(
                buffer,
                PacketKind::SnapshotResult,
                header.request_id,
                &bytes,
            )
        }),
        PacketKind::SnapshotImportRequest => with_runtime(|runtime| {
            let snapshot_len = header.payload_len as usize;
            let envelope = runtime
                .import_snapshot(&buffer[PacketHeader::BYTE_LEN..payload_end])
                .map_err(|error| error.to_string())?;
            let descriptor = runtime.snapshot_descriptor(envelope.created_at, snapshot_len);
            write_json_response(
                buffer,
                PacketKind::SnapshotResult,
                header.request_id,
                &descriptor,
            )
        }),
        PacketKind::StoreCommandRequest => with_runtime(|runtime| {
            let request: StoreCommandRequest =
                decode_json(&buffer[PacketHeader::BYTE_LEN..payload_end])?;
            let result = runtime
                .store_command(request)
                .map_err(|error| error.to_string())?;
            write_json_response(
                buffer,
                PacketKind::StoreCommandResult,
                header.request_id,
                &result,
            )
        }),
        kind => write_error_response(
            buffer,
            header.request_id,
            &format!("unsupported packet kind: {kind:?}"),
        ),
    }
}

fn with_runtime<T>(
    operation: impl FnOnce(&PhoenixRuntime) -> Result<T, String>,
) -> Result<T, String> {
    RUNTIME.with(|cell| {
        let borrow = cell.borrow();
        let runtime = borrow
            .as_ref()
            .ok_or_else(|| "runtime not initialized".to_owned())?;
        operation(runtime)
    })
}

fn decode_json<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    serde_json::from_slice(bytes).map_err(|error| error.to_string())
}

fn write_json_response<T: serde::Serialize>(
    buffer: &mut [u8],
    kind: PacketKind,
    request_id: u32,
    payload: &T,
) -> Result<(), String> {
    let bytes = serde_json::to_vec(payload).map_err(|error| error.to_string())?;
    write_binary_response(buffer, kind, request_id, &bytes)
}

fn write_error_response(buffer: &mut [u8], request_id: u32, message: &str) -> Result<(), String> {
    let diagnostic = Diagnostic {
        code: "PX_PACKET_ERROR".to_owned(),
        message: message.to_owned(),
    };
    write_json_response(buffer, PacketKind::Status, request_id, &diagnostic)
}

fn write_binary_response(
    buffer: &mut [u8],
    kind: PacketKind,
    request_id: u32,
    payload: &[u8],
) -> Result<(), String> {
    let total_len = PacketHeader::BYTE_LEN + payload.len();
    if total_len > buffer.len() {
        return Err("response payload exceeds packet region".to_owned());
    }

    let header = PacketHeader::new(1, kind, request_id, payload.len() as u32);
    buffer[..PacketHeader::BYTE_LEN].copy_from_slice(&header.to_le_bytes());
    buffer[PacketHeader::BYTE_LEN..total_len].copy_from_slice(payload);
    Ok(())
}

fn write_binary_response_with(
    buffer: &mut [u8],
    kind: PacketKind,
    request_id: u32,
    writer: impl FnOnce(&mut [u8]) -> Result<usize, String>,
) -> Result<(), String> {
    let payload = &mut buffer[PacketHeader::BYTE_LEN..];
    let payload_len = writer(payload)?;
    if PacketHeader::BYTE_LEN + payload_len > buffer.len() {
        return Err("response payload exceeds packet region".to_owned());
    }
    let header = PacketHeader::new(1, kind, request_id, payload_len as u32);
    buffer[..PacketHeader::BYTE_LEN].copy_from_slice(&header.to_le_bytes());
    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_wasm_protocol_version() -> u32 {
    PHOENIX_PROTOCOL_VERSION
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_packet_header_size() -> usize {
    packet_header_size()
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_alloc(size: usize) -> *mut u8 {
    let mut bytes = Vec::<u8>::with_capacity(size.max(1));
    let ptr = bytes.as_mut_ptr();
    std::mem::forget(bytes);
    ptr
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_dealloc(ptr: *mut u8, capacity: usize) {
    if !ptr.is_null() {
        unsafe {
            let _ = Vec::from_raw_parts(ptr, 0, capacity.max(1));
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_process_packet_at(offset: usize, capacity: usize) -> i32 {
    let result = unsafe {
        let slice = std::slice::from_raw_parts_mut(offset as *mut u8, capacity);
        process_packet_buffer(slice)
    };

    match result {
        Ok(()) => 0,
        Err(error) => {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(offset as *mut u8, capacity);
                let _ = write_error_response(slice, 0, &error);
            }
            -1
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_opfs_save_snapshot() -> i32 {
    let snapshot = match with_runtime(|runtime| {
        runtime.export_snapshot().map_err(|error| error.to_string())
    }) {
        Ok(bytes) => bytes,
        Err(error) => {
            set_opfs_status(OpfsStatus::failed("save", &error));
            return -1;
        }
    };

    set_opfs_status(OpfsStatus::pending(
        "save",
        "Saving Phoenix snapshot to OPFS",
    ));
    wasm_bindgen_futures::spawn_local(async move {
        match opfs::save_snapshot(&snapshot).await {
            Ok(snapshot_bytes) => {
                set_opfs_status(OpfsStatus::success(
                    "save",
                    snapshot_bytes,
                    false,
                    "Phoenix snapshot saved to OPFS",
                ));
            }
            Err(error) => {
                set_opfs_status(OpfsStatus::failed("save", &error));
            }
        }
    });

    0
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_opfs_load_snapshot() -> i32 {
    if with_runtime(|_| Ok(())).is_err() {
        let message = "runtime not initialized";
        set_opfs_status(OpfsStatus::failed("load", message));
        return -1;
    }

    set_opfs_status(OpfsStatus::pending(
        "load",
        "Loading Phoenix snapshot from OPFS",
    ));
    wasm_bindgen_futures::spawn_local(async move {
        match opfs::load_snapshot().await {
            Ok(load) => match load.bytes {
                Some(bytes) => {
                    let recovered_from_backup = load.recovered_from_backup;
                    let snapshot_bytes = bytes.len();
                    let import_result = with_runtime(|runtime| {
                        runtime
                            .import_snapshot(&bytes)
                            .map_err(|error| error.to_string())
                    });

                    match import_result {
                        Ok(_) => {
                            let message = if recovered_from_backup {
                                "Phoenix snapshot restored from OPFS backup"
                            } else {
                                "Phoenix snapshot restored from OPFS"
                            };
                            set_opfs_status(OpfsStatus::success(
                                "load",
                                snapshot_bytes,
                                recovered_from_backup,
                                message,
                            ));
                        }
                        Err(error) => {
                            set_opfs_status(OpfsStatus::failed("load", &error));
                        }
                    }
                }
                None => {
                    set_opfs_status(OpfsStatus::success(
                        "load",
                        0,
                        false,
                        "No Phoenix snapshot found in OPFS",
                    ));
                }
            },
            Err(error) => {
                set_opfs_status(OpfsStatus::failed("load", &error));
            }
        }
    });

    0
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_opfs_clear_snapshot() -> i32 {
    set_opfs_status(OpfsStatus::pending(
        "clear",
        "Clearing Phoenix snapshot from OPFS",
    ));
    wasm_bindgen_futures::spawn_local(async move {
        match opfs::clear_snapshot().await {
            Ok(()) => {
                set_opfs_status(OpfsStatus::success(
                    "clear",
                    0,
                    false,
                    "Phoenix snapshot cleared from OPFS",
                ));
            }
            Err(error) => {
                set_opfs_status(OpfsStatus::failed("clear", &error));
            }
        }
    });

    0
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_opfs_status_len() -> usize {
    opfs_status_json().len()
}

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn phoenix_opfs_write_status_at(offset: usize, capacity: usize) -> usize {
    let bytes = opfs_status_json().into_bytes();
    let write_len = bytes.len().min(capacity);
    unsafe {
        let dest = std::slice::from_raw_parts_mut(offset as *mut u8, capacity);
        dest[..write_len].copy_from_slice(&bytes[..write_len]);
    }
    bytes.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use phoenix_types::{
        AnalyzeTextRequest, CreateSessionRequest, DocumentId, GraphDeltaRequest,
        GraphDeltaResultHeader, IngestRequest, QueryRequest, QueryResultHeader, QueryTarget,
        RuntimeConfig, RuntimeInitResult, ScanArtifact, ScanRequest, ScopeKey, SessionRecord,
        SessionStateRequest, SessionStateResultHeader, SessionStatsRequest,
        SessionStatsResultHeader, SnapshotDto, StructureArtifact, StructureRequest,
    };

    fn packet(kind: PacketKind, request_id: u32, payload: &[u8]) -> Vec<u8> {
        let mut buffer = vec![0_u8; DEFAULT_PACKET_REGION_SIZE];
        let header = PacketHeader::new(1, kind, request_id, payload.len() as u32);
        buffer[..PacketHeader::BYTE_LEN].copy_from_slice(&header.to_le_bytes());
        buffer[PacketHeader::BYTE_LEN..PacketHeader::BYTE_LEN + payload.len()]
            .copy_from_slice(payload);
        buffer
    }

    fn decode_header(buffer: &[u8]) -> PacketHeader {
        PacketHeader::from_le_bytes(buffer[..PacketHeader::BYTE_LEN].try_into().expect("header"))
    }

    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().expect("u32"))
    }

    fn read_u64(bytes: &[u8], offset: usize) -> u64 {
        u64::from_le_bytes(bytes[offset..offset + 8].try_into().expect("u64"))
    }

    fn read_utf8(bytes: &[u8], arena_offset: usize, string_offset: u32, string_len: u32) -> String {
        let start = arena_offset + string_offset as usize;
        let end = start + string_len as usize;
        String::from_utf8(bytes[start..end].to_vec()).expect("utf8")
    }

    #[test]
    fn packet_header_size_matches_shared_contract() {
        assert_eq!(packet_header_size(), 16);
    }

    #[test]
    fn shared_memory_runtime_init_and_session_round_trip() {
        let init_payload = serde_json::to_vec(&RuntimeInitRequest {
            config: RuntimeConfig::default(),
            storage_path: None,
            force_reset: false,
        })
        .expect("init payload");
        let mut init_packet = packet(PacketKind::InitRuntimeRequest, 7, &init_payload);
        process_packet_buffer(&mut init_packet).expect("init packet");

        let init_header = decode_header(&init_packet);
        assert_eq!(init_header.packet_kind(), PacketKind::InitRuntimeResult);
        let init_result: RuntimeInitResult = serde_json::from_slice(
            &init_packet
                [PacketHeader::BYTE_LEN..PacketHeader::BYTE_LEN + init_header.payload_len as usize],
        )
        .expect("init result");
        assert!(init_result.ready);

        let session_payload = serde_json::to_vec(&CreateSessionRequest {
            session_id: None,
            label: "Shared".to_owned(),
            scope: ScopeKey::default(),
        })
        .expect("session payload");
        let mut session_packet = packet(PacketKind::CreateSessionRequest, 8, &session_payload);
        process_packet_buffer(&mut session_packet).expect("session packet");

        let session_header = decode_header(&session_packet);
        assert_eq!(
            session_header.packet_kind(),
            PacketKind::CreateSessionResult
        );
        let session: SessionRecord = serde_json::from_slice(
            &session_packet[PacketHeader::BYTE_LEN
                ..PacketHeader::BYTE_LEN + session_header.payload_len as usize],
        )
        .expect("session result");
        assert_eq!(session.label, "Shared");
    }

    #[test]
    fn shared_memory_ingest_query_and_snapshot_round_trip() {
        let init_payload = serde_json::to_vec(&RuntimeInitRequest {
            config: RuntimeConfig::default(),
            storage_path: None,
            force_reset: false,
        })
        .expect("init payload");
        let mut init_packet = packet(PacketKind::InitRuntimeRequest, 1, &init_payload);
        process_packet_buffer(&mut init_packet).expect("init packet");

        let session_payload = serde_json::to_vec(&CreateSessionRequest {
            session_id: None,
            label: "RoundTrip".to_owned(),
            scope: ScopeKey::default(),
        })
        .expect("session payload");
        let mut session_packet = packet(PacketKind::CreateSessionRequest, 2, &session_payload);
        process_packet_buffer(&mut session_packet).expect("session packet");
        let session_header = decode_header(&session_packet);
        let session: SessionRecord = serde_json::from_slice(
            &session_packet[PacketHeader::BYTE_LEN
                ..PacketHeader::BYTE_LEN + session_header.payload_len as usize],
        )
        .expect("session result");

        let ingest_payload = serde_json::to_vec(&IngestRequest {
            session_id: Some(session.session_id.clone()),
            documents: vec![phoenix_types::IngestDocument {
                document_id: DocumentId("packet-doc".to_owned()),
                note_id: None,
                title: "Packet Note".to_owned(),
                text: "Phoenix packets are alive.".to_owned(),
                scope: ScopeKey::default(),
            }],
            commit: false,
        })
        .expect("ingest payload");
        let mut ingest_packet = packet(PacketKind::IngestRequest, 3, &ingest_payload);
        process_packet_buffer(&mut ingest_packet).expect("ingest packet");

        let query_payload = serde_json::to_vec(&QueryRequest {
            session_id: Some(session.session_id),
            query: "phoenix".to_owned(),
            scope: ScopeKey::default(),
            targets: vec![QueryTarget::Chunks],
            limit: Some(3),
            temporal: None,
            semantic_query_vector: None,
            include_candidate_graph: false,
        })
        .expect("query payload");
        let mut query_packet = packet(PacketKind::QueryRequest, 4, &query_payload);
        process_packet_buffer(&mut query_packet).expect("query packet");
        let query_header = decode_header(&query_packet);
        assert_eq!(query_header.packet_kind(), PacketKind::QueryResult);
        let query_payload = &query_packet
            [PacketHeader::BYTE_LEN..PacketHeader::BYTE_LEN + query_header.payload_len as usize];
        assert!(query_payload.len() >= QueryResultHeader::BYTE_LEN);
        let chunk_count = read_u32(query_payload, 20);
        let arena_offset = read_u32(query_payload, 48) as usize;
        let session_offset = read_u32(query_payload, 8);
        let session_len = read_u32(query_payload, 12);
        let first_chunk_id = read_utf8(
            query_payload,
            arena_offset,
            read_u32(query_payload, QueryResultHeader::BYTE_LEN),
            read_u32(query_payload, QueryResultHeader::BYTE_LEN + 4),
        );
        assert_eq!(chunk_count, 1);
        assert!(
            read_utf8(query_payload, arena_offset, session_offset, session_len)
                .starts_with("session-")
        );
        assert!(first_chunk_id.starts_with("packet-doc:"));

        let mut export_packet = packet(PacketKind::SnapshotExportRequest, 5, &[]);
        process_packet_buffer(&mut export_packet).expect("snapshot export");
        let export_header = decode_header(&export_packet);
        assert_eq!(export_header.packet_kind(), PacketKind::SnapshotResult);

        let snapshot_bytes = export_packet
            [PacketHeader::BYTE_LEN..PacketHeader::BYTE_LEN + export_header.payload_len as usize]
            .to_vec();
        let mut import_packet = packet(PacketKind::SnapshotImportRequest, 6, &snapshot_bytes);
        process_packet_buffer(&mut import_packet).expect("snapshot import");
        let import_header = decode_header(&import_packet);
        let snapshot_result: SnapshotDto = serde_json::from_slice(
            &import_packet[PacketHeader::BYTE_LEN
                ..PacketHeader::BYTE_LEN + import_header.payload_len as usize],
        )
        .expect("snapshot descriptor");
        assert_eq!(snapshot_result.schema_version, "phoenix.cozo.v3");
    }

    #[test]
    fn shared_memory_scan_and_structure_round_trip() {
        let init_payload = serde_json::to_vec(&RuntimeInitRequest {
            config: RuntimeConfig::default(),
            storage_path: None,
            force_reset: false,
        })
        .expect("init payload");
        let mut init_packet = packet(PacketKind::InitRuntimeRequest, 1, &init_payload);
        process_packet_buffer(&mut init_packet).expect("init packet");

        let scan_payload = serde_json::to_vec(&ScanRequest {
            text: "Luffy attacked Zoro.".to_owned(),
            scope: ScopeKey::default(),
            session_id: Some(phoenix_types::SessionId("packet-scan".to_owned())),
            resolver_seed: vec![
                phoenix_types::ResolverEntitySeed {
                    entity_id: phoenix_types::EntityId("luffy".to_owned()),
                    canonical_name: "Luffy".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(phoenix_types::EntityKind::Character),
                    gender: Some(phoenix_types::GenderHint::Male),
                    number: None,
                    scope: ScopeKey::default(),
                },
                phoenix_types::ResolverEntitySeed {
                    entity_id: phoenix_types::EntityId("zoro".to_owned()),
                    canonical_name: "Zoro".to_owned(),
                    aliases: Vec::new(),
                    kind: Some(phoenix_types::EntityKind::Character),
                    gender: Some(phoenix_types::GenderHint::Male),
                    number: None,
                    scope: ScopeKey::default(),
                },
            ],
        })
        .expect("scan payload");
        let mut scan_packet = packet(PacketKind::ScanRequest, 2, &scan_payload);
        process_packet_buffer(&mut scan_packet).expect("scan packet");
        let scan_header = decode_header(&scan_packet);
        let scan: ScanArtifact = serde_json::from_slice(
            &scan_packet
                [PacketHeader::BYTE_LEN..PacketHeader::BYTE_LEN + scan_header.payload_len as usize],
        )
        .expect("scan result");
        assert_eq!(scan.mentions.len(), 2);

        let structure_payload = serde_json::to_vec(&StructureRequest {
            text: "Luffy attacked Zoro.".to_owned(),
            scan,
        })
        .expect("structure payload");
        let mut structure_packet = packet(PacketKind::StructureRequest, 3, &structure_payload);
        process_packet_buffer(&mut structure_packet).expect("structure packet");
        let structure_header = decode_header(&structure_packet);
        let structure: StructureArtifact = serde_json::from_slice(
            &structure_packet[PacketHeader::BYTE_LEN
                ..PacketHeader::BYTE_LEN + structure_header.payload_len as usize],
        )
        .expect("structure result");
        assert_eq!(structure.sentence_frames.len(), 1);
    }

    #[test]
    fn shared_memory_analyze_text_round_trip() {
        let init_payload = serde_json::to_vec(&RuntimeInitRequest {
            config: RuntimeConfig::default(),
            storage_path: None,
            force_reset: false,
        })
        .expect("init payload");
        let mut init_packet = packet(PacketKind::InitRuntimeRequest, 21, &init_payload);
        process_packet_buffer(&mut init_packet).expect("init packet");

        let analytics_payload = serde_json::to_vec(&AnalyzeTextRequest {
            text: "The iron gate slammed shut. The iron gate rattled again.".to_owned(),
        })
        .expect("analytics payload");
        let mut analytics_packet = packet(PacketKind::AnalyzeTextRequest, 22, &analytics_payload);
        process_packet_buffer(&mut analytics_packet).expect("analytics packet");
        let analytics_header = decode_header(&analytics_packet);
        assert_eq!(
            analytics_header.packet_kind(),
            PacketKind::AnalyzeTextResult
        );
        let analytics: serde_json::Value = serde_json::from_slice(
            &analytics_packet[PacketHeader::BYTE_LEN
                ..PacketHeader::BYTE_LEN + analytics_header.payload_len as usize],
        )
        .expect("analytics result");
        assert!(analytics["wordCount"].as_i64().unwrap_or_default() > 0);
        assert_eq!(analytics["sentenceCount"].as_i64().unwrap_or_default(), 2);
        assert!(analytics["repetition"]["items"]
            .as_array()
            .map(|items| !items.is_empty())
            .unwrap_or(false));
    }

    #[test]
    fn shared_memory_graph_delta_and_session_packets_are_binary() {
        let init_payload = serde_json::to_vec(&RuntimeInitRequest {
            config: RuntimeConfig::default(),
            storage_path: None,
            force_reset: false,
        })
        .expect("init payload");
        let mut init_packet = packet(PacketKind::InitRuntimeRequest, 11, &init_payload);
        process_packet_buffer(&mut init_packet).expect("init packet");

        let session_payload = serde_json::to_vec(&CreateSessionRequest {
            session_id: None,
            label: "GraphState".to_owned(),
            scope: ScopeKey::default(),
        })
        .expect("session payload");
        let mut session_packet = packet(PacketKind::CreateSessionRequest, 12, &session_payload);
        process_packet_buffer(&mut session_packet).expect("session packet");
        let session_header = decode_header(&session_packet);
        let session: SessionRecord = serde_json::from_slice(
            &session_packet[PacketHeader::BYTE_LEN
                ..PacketHeader::BYTE_LEN + session_header.payload_len as usize],
        )
        .expect("session result");

        let ingest_payload = serde_json::to_vec(&IngestRequest {
            session_id: Some(session.session_id.clone()),
            documents: vec![phoenix_types::IngestDocument {
                document_id: DocumentId("graph-doc".to_owned()),
                note_id: None,
                title: "Graph Packet".to_owned(),
                text: "Ryan attacked Len. Ryan gave Len a blade.".to_owned(),
                scope: ScopeKey::default(),
            }],
            commit: false,
        })
        .expect("ingest payload");
        let mut ingest_packet = packet(PacketKind::IngestRequest, 13, &ingest_payload);
        process_packet_buffer(&mut ingest_packet).expect("ingest packet");

        let graph_delta_payload = serde_json::to_vec(&GraphDeltaRequest {
            session_id: session.session_id.clone(),
            scope: ScopeKey::default(),
            changed_documents: vec![DocumentId("graph-doc".to_owned())],
            limit: Some(8),
            since_commit: None,
            include_candidate_graph: false,
        })
        .expect("graph delta payload");
        let mut graph_packet = packet(PacketKind::GraphDeltaRequest, 14, &graph_delta_payload);
        process_packet_buffer(&mut graph_packet).expect("graph packet");
        let graph_header = decode_header(&graph_packet);
        assert_eq!(graph_header.packet_kind(), PacketKind::GraphDeltaResult);
        let graph_payload = &graph_packet
            [PacketHeader::BYTE_LEN..PacketHeader::BYTE_LEN + graph_header.payload_len as usize];
        assert!(graph_payload.len() >= GraphDeltaResultHeader::BYTE_LEN);
        assert!(read_u32(graph_payload, 20) >= 1);
        assert!(read_u32(graph_payload, 36) >= 1);

        let state_payload = serde_json::to_vec(&SessionStateRequest {
            session_id: session.session_id.clone(),
        })
        .expect("state payload");
        let mut state_packet = packet(PacketKind::SessionStateRequest, 15, &state_payload);
        process_packet_buffer(&mut state_packet).expect("state packet");
        let state_header = decode_header(&state_packet);
        assert_eq!(state_header.packet_kind(), PacketKind::SessionStateResult);
        let state_payload = &state_packet
            [PacketHeader::BYTE_LEN..PacketHeader::BYTE_LEN + state_header.payload_len as usize];
        assert!(state_payload.len() >= SessionStateResultHeader::BYTE_LEN);
        assert_eq!(read_u32(state_payload, 20), 1);

        let stats_payload = serde_json::to_vec(&SessionStatsRequest {
            session_id: session.session_id,
        })
        .expect("stats payload");
        let mut stats_packet = packet(PacketKind::SessionStatsRequest, 16, &stats_payload);
        process_packet_buffer(&mut stats_packet).expect("stats packet");
        let stats_header = decode_header(&stats_packet);
        assert_eq!(stats_header.packet_kind(), PacketKind::SessionStatsResult);
        let stats_payload = &stats_packet
            [PacketHeader::BYTE_LEN..PacketHeader::BYTE_LEN + stats_header.payload_len as usize];
        assert!(stats_payload.len() >= SessionStatsResultHeader::BYTE_LEN);
        assert_eq!(read_u32(stats_payload, 20), 1);
        assert!(read_u64(stats_payload, SessionStatsResultHeader::BYTE_LEN + 36) > 0);
    }
}
