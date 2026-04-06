use phoenix_store_cozo::StoreError;
use phoenix_types::{
    ChunkHitRecord, Diagnostic, DiagnosticRecord, GraphDeltaChunk, GraphDeltaChunkRecord,
    GraphDeltaEdge, GraphDeltaEdgeRecord, GraphDeltaNode, GraphDeltaNodeRecord, GraphDeltaResult,
    GraphDeltaResultHeader, NodeHitRecord, QueryResult, QueryResultHeader, SessionDocumentRecord,
    SessionState, SessionStateResultHeader, SessionStats, SessionStatsRecord,
    SessionStatsResultHeader, StringRefRecord, BINARY_LAYOUT_VERSION, FLAG_HAS_DOCUMENT_ID,
    FLAG_HAS_ENTITY_ID, FLAG_HAS_FRONT_MATTER, FLAG_HAS_NOTE_ID, FLAG_HAS_SESSION,
};

pub fn encode_query_result(result: &QueryResult) -> Result<Vec<u8>, StoreError> {
    let plan = QueryPayloadPlan::build(result);
    let mut bytes = vec![0; plan.total_len()];
    let written = plan.write_into(&mut bytes)?;
    bytes.truncate(written);
    Ok(bytes)
}

pub fn encode_query_result_into(
    buffer: &mut [u8],
    result: &QueryResult,
) -> Result<usize, StoreError> {
    QueryPayloadPlan::build(result).write_into(buffer)
}

pub fn encode_graph_delta(result: &GraphDeltaResult) -> Result<Vec<u8>, StoreError> {
    let plan = GraphDeltaPayloadPlan::build(result);
    let mut bytes = vec![0; plan.total_len()];
    let written = plan.write_into(&mut bytes)?;
    bytes.truncate(written);
    Ok(bytes)
}

pub fn encode_graph_delta_into(
    buffer: &mut [u8],
    result: &GraphDeltaResult,
) -> Result<usize, StoreError> {
    GraphDeltaPayloadPlan::build(result).write_into(buffer)
}

pub fn encode_session_state(state: &SessionState) -> Result<Vec<u8>, StoreError> {
    let plan = SessionStatePayloadPlan::build(state);
    let mut bytes = vec![0; plan.total_len()];
    let written = plan.write_into(&mut bytes)?;
    bytes.truncate(written);
    Ok(bytes)
}

pub fn encode_session_state_into(
    buffer: &mut [u8],
    state: &SessionState,
) -> Result<usize, StoreError> {
    SessionStatePayloadPlan::build(state).write_into(buffer)
}

pub fn encode_session_stats(stats: &SessionStats) -> Result<Vec<u8>, StoreError> {
    let plan = SessionStatsPayloadPlan::build(stats);
    let mut bytes = vec![0; plan.total_len()];
    let written = plan.write_into(&mut bytes)?;
    bytes.truncate(written);
    Ok(bytes)
}

pub fn encode_session_stats_into(
    buffer: &mut [u8],
    stats: &SessionStats,
) -> Result<usize, StoreError> {
    SessionStatsPayloadPlan::build(stats).write_into(buffer)
}

#[derive(Clone, Copy, Debug, Default)]
struct PackedStringRef {
    offset: u32,
    len: u32,
}

#[derive(Default)]
struct BinaryBuilder {
    arena: Vec<u8>,
}

impl BinaryBuilder {
    fn push_string(&mut self, value: &str) -> PackedStringRef {
        let offset = self.arena.len() as u32;
        self.arena.extend_from_slice(value.as_bytes());
        PackedStringRef {
            offset,
            len: value.len() as u32,
        }
    }
}

struct QueryPayloadPlan {
    header: QueryResultHeader,
    chunk_records: Vec<ChunkHitRecord>,
    node_records: Vec<NodeHitRecord>,
    diagnostic_records: Vec<DiagnosticRecord>,
    arena: Vec<u8>,
}

impl QueryPayloadPlan {
    fn build(result: &QueryResult) -> Self {
        let mut builder = BinaryBuilder::default();
        let session = result
            .session_id
            .as_ref()
            .map(|session_id| builder.push_string(&session_id.0))
            .unwrap_or_default();
        let chunk_offset = QueryResultHeader::BYTE_LEN as u32;
        let mut chunk_records = Vec::with_capacity(result.chunk_hits.len());
        for hit in &result.chunk_hits {
            let chunk_id = builder.push_string(&hit.chunk_id);
            chunk_records.push(ChunkHitRecord {
                chunk_id_offset: chunk_id.offset,
                chunk_id_len: chunk_id.len,
                score_bits: hit.score.to_bits(),
            });
        }
        let node_offset = chunk_offset + (chunk_records.len() * ChunkHitRecord::BYTE_LEN) as u32;
        let mut node_records = Vec::with_capacity(result.node_hits.len());
        for hit in &result.node_hits {
            let entity_id = hit
                .entity_id
                .as_ref()
                .map(|entity_id| builder.push_string(&entity_id.0))
                .unwrap_or_default();
            node_records.push(NodeHitRecord {
                entity_id_offset: entity_id.offset,
                entity_id_len: entity_id.len,
                score_bits: hit.score.to_bits(),
            });
        }
        let diagnostic_offset = node_offset + (node_records.len() * NodeHitRecord::BYTE_LEN) as u32;
        let diagnostic_records = diagnostics_to_records(&mut builder, &result.diagnostics);
        let arena_offset =
            diagnostic_offset + (diagnostic_records.len() * DiagnosticRecord::BYTE_LEN) as u32;
        let header = QueryResultHeader {
            version: BINARY_LAYOUT_VERSION,
            flags: if result.session_id.is_some() {
                FLAG_HAS_SESSION
            } else {
                0
            },
            session_offset: session.offset,
            session_len: session.len,
            table1_offset: chunk_offset,
            table1_count: chunk_records.len() as u32,
            table2_offset: node_offset,
            table2_count: node_records.len() as u32,
            table3_offset: diagnostic_offset,
            table3_count: diagnostic_records.len() as u32,
            table4_offset: 0,
            table4_count: 0,
            arena_offset,
            arena_len: builder.arena.len() as u32,
        };
        Self {
            header,
            chunk_records,
            node_records,
            diagnostic_records,
            arena: builder.arena,
        }
    }

    fn total_len(&self) -> usize {
        self.header.arena_offset as usize + self.arena.len()
    }

    fn write_into(&self, buffer: &mut [u8]) -> Result<usize, StoreError> {
        ensure_capacity(buffer, self.total_len())?;
        let mut offset = 0;
        write_query_header(buffer, &mut offset, &self.header);
        for record in &self.chunk_records {
            write_chunk_hit_record(buffer, &mut offset, record);
        }
        for record in &self.node_records {
            write_node_hit_record(buffer, &mut offset, record);
        }
        for record in &self.diagnostic_records {
            write_diagnostic_record(buffer, &mut offset, record);
        }
        write_bytes(buffer, &mut offset, &self.arena);
        Ok(offset)
    }
}

struct GraphDeltaPayloadPlan {
    header: GraphDeltaResultHeader,
    chunk_records: Vec<GraphDeltaChunkRecord>,
    node_records: Vec<GraphDeltaNodeRecord>,
    edge_records: Vec<GraphDeltaEdgeRecord>,
    diagnostic_records: Vec<DiagnosticRecord>,
    arena: Vec<u8>,
}

impl GraphDeltaPayloadPlan {
    fn build(result: &GraphDeltaResult) -> Self {
        let mut builder = BinaryBuilder::default();
        let session = builder.push_string(&result.session_id.0);
        let header_size = GraphDeltaResultHeader::BYTE_LEN as u32;
        let mut chunk_records = Vec::with_capacity(result.chunks.len());
        for chunk in &result.chunks {
            chunk_records.push(graph_delta_chunk_record(&mut builder, chunk));
        }
        let table2_offset =
            header_size + (chunk_records.len() * GraphDeltaChunkRecord::BYTE_LEN) as u32;
        let mut node_records = Vec::with_capacity(result.nodes.len());
        for node in &result.nodes {
            node_records.push(graph_delta_node_record(&mut builder, node));
        }
        let table3_offset =
            table2_offset + (node_records.len() * GraphDeltaNodeRecord::BYTE_LEN) as u32;
        let mut edge_records = Vec::with_capacity(result.edges.len());
        for edge in &result.edges {
            edge_records.push(graph_delta_edge_record(&mut builder, edge));
        }
        let table4_offset =
            table3_offset + (edge_records.len() * GraphDeltaEdgeRecord::BYTE_LEN) as u32;
        let diagnostic_records = diagnostics_to_records(&mut builder, &result.diagnostics);
        let arena_offset =
            table4_offset + (diagnostic_records.len() * DiagnosticRecord::BYTE_LEN) as u32;
        let header = GraphDeltaResultHeader {
            version: BINARY_LAYOUT_VERSION,
            flags: FLAG_HAS_SESSION,
            session_offset: session.offset,
            session_len: session.len,
            table1_offset: header_size,
            table1_count: chunk_records.len() as u32,
            table2_offset,
            table2_count: node_records.len() as u32,
            table3_offset,
            table3_count: edge_records.len() as u32,
            table4_offset,
            table4_count: diagnostic_records.len() as u32,
            arena_offset,
            arena_len: builder.arena.len() as u32,
        };
        Self {
            header,
            chunk_records,
            node_records,
            edge_records,
            diagnostic_records,
            arena: builder.arena,
        }
    }

    fn total_len(&self) -> usize {
        self.header.arena_offset as usize + self.arena.len()
    }

    fn write_into(&self, buffer: &mut [u8]) -> Result<usize, StoreError> {
        ensure_capacity(buffer, self.total_len())?;
        let mut offset = 0;
        write_graph_delta_header(buffer, &mut offset, &self.header);
        for record in &self.chunk_records {
            write_graph_delta_chunk_record(buffer, &mut offset, record);
        }
        for record in &self.node_records {
            write_graph_delta_node_record(buffer, &mut offset, record);
        }
        for record in &self.edge_records {
            write_graph_delta_edge_record(buffer, &mut offset, record);
        }
        for record in &self.diagnostic_records {
            write_diagnostic_record(buffer, &mut offset, record);
        }
        write_bytes(buffer, &mut offset, &self.arena);
        Ok(offset)
    }
}

struct SessionStatePayloadPlan {
    header: SessionStateResultHeader,
    document_records: Vec<SessionDocumentRecord>,
    title_records: Vec<StringRefRecord>,
    namespace_records: Vec<StringRefRecord>,
    arena: Vec<u8>,
}

impl SessionStatePayloadPlan {
    fn build(state: &SessionState) -> Self {
        let mut builder = BinaryBuilder::default();
        let session = builder.push_string(&state.session_id.0);
        let header_size = SessionStateResultHeader::BYTE_LEN as u32;
        let mut title_records = Vec::new();
        let mut document_records = Vec::with_capacity(state.documents.len());
        for document in &state.documents {
            let title_start = title_records.len() as u32;
            for title in &document.chapter_titles {
                let title_ref = builder.push_string(title);
                title_records.push(StringRefRecord {
                    offset: title_ref.offset,
                    len: title_ref.len,
                });
            }
            let document_id = builder.push_string(&document.document_id.0);
            let note_id = document
                .note_id
                .as_ref()
                .map(|note_id| builder.push_string(&note_id.0))
                .unwrap_or_default();
            let mut flags = 0;
            if document.note_id.is_some() {
                flags |= FLAG_HAS_NOTE_ID;
            }
            if document.has_front_matter_chapter {
                flags |= FLAG_HAS_FRONT_MATTER;
            }
            document_records.push(SessionDocumentRecord {
                document_id_offset: document_id.offset,
                document_id_len: document_id.len,
                note_id_offset: note_id.offset,
                note_id_len: note_id.len,
                chapter_titles_start: title_start,
                chapter_titles_count: document.chapter_titles.len() as u32,
                chapter_count: document.chapter_count as u32,
                parent_count: document.parent_count as u32,
                leaf_count: document.leaf_count as u32,
                entity_count: document.entity_count as u32,
                discovery_count: document.discovery_count as u32,
                flags,
                updated_at_bits: document.updated_at as u64,
            });
        }
        let table2_offset =
            header_size + (document_records.len() * SessionDocumentRecord::BYTE_LEN) as u32;
        let namespace_records = state
            .manifest_namespaces
            .iter()
            .map(|namespace| {
                let string_ref = builder.push_string(namespace);
                StringRefRecord {
                    offset: string_ref.offset,
                    len: string_ref.len,
                }
            })
            .collect::<Vec<_>>();
        let arena_offset = table2_offset
            + (title_records.len() * StringRefRecord::BYTE_LEN) as u32
            + (namespace_records.len() * StringRefRecord::BYTE_LEN) as u32;
        let header = SessionStateResultHeader {
            version: BINARY_LAYOUT_VERSION,
            flags: FLAG_HAS_SESSION,
            session_offset: session.offset,
            session_len: session.len,
            table1_offset: header_size,
            table1_count: document_records.len() as u32,
            table2_offset,
            table2_count: title_records.len() as u32,
            table3_offset: table2_offset + (title_records.len() * StringRefRecord::BYTE_LEN) as u32,
            table3_count: namespace_records.len() as u32,
            table4_offset: 0,
            table4_count: 0,
            arena_offset,
            arena_len: builder.arena.len() as u32,
        };
        Self {
            header,
            document_records,
            title_records,
            namespace_records,
            arena: builder.arena,
        }
    }

    fn total_len(&self) -> usize {
        self.header.arena_offset as usize + self.arena.len()
    }

    fn write_into(&self, buffer: &mut [u8]) -> Result<usize, StoreError> {
        ensure_capacity(buffer, self.total_len())?;
        let mut offset = 0;
        write_session_state_header(buffer, &mut offset, &self.header);
        for record in &self.document_records {
            write_session_document_record(buffer, &mut offset, record);
        }
        for record in &self.title_records {
            write_string_ref_record(buffer, &mut offset, record);
        }
        for record in &self.namespace_records {
            write_string_ref_record(buffer, &mut offset, record);
        }
        write_bytes(buffer, &mut offset, &self.arena);
        Ok(offset)
    }
}

struct SessionStatsPayloadPlan {
    header: SessionStatsResultHeader,
    record: SessionStatsRecord,
    arena: Vec<u8>,
}

impl SessionStatsPayloadPlan {
    fn build(stats: &SessionStats) -> Self {
        let mut builder = BinaryBuilder::default();
        let session = builder.push_string(&stats.session_id.0);
        let header = SessionStatsResultHeader {
            version: BINARY_LAYOUT_VERSION,
            flags: FLAG_HAS_SESSION,
            session_offset: session.offset,
            session_len: session.len,
            table1_offset: SessionStatsResultHeader::BYTE_LEN as u32,
            table1_count: 1,
            table2_offset: 0,
            table2_count: 0,
            table3_offset: 0,
            table3_count: 0,
            table4_offset: 0,
            table4_count: 0,
            arena_offset: (SessionStatsResultHeader::BYTE_LEN + SessionStatsRecord::BYTE_LEN)
                as u32,
            arena_len: builder.arena.len() as u32,
        };
        let record = SessionStatsRecord {
            document_count: stats.document_count as u32,
            chapter_count: stats.chapter_count as u32,
            parent_count: stats.parent_count as u32,
            leaf_count: stats.leaf_count as u32,
            entity_count: stats.entity_count as u32,
            discovery_candidate_count: stats.discovery_candidate_count as u32,
            graph_vertex_count: stats.graph_vertex_count as u32,
            graph_edge_count: stats.graph_edge_count as u32,
            span_count: stats.span_count as u32,
            updated_at_bits: stats.updated_at as u64,
        };
        Self {
            header,
            record,
            arena: builder.arena,
        }
    }

    fn total_len(&self) -> usize {
        self.header.arena_offset as usize + self.arena.len()
    }

    fn write_into(&self, buffer: &mut [u8]) -> Result<usize, StoreError> {
        ensure_capacity(buffer, self.total_len())?;
        let mut offset = 0;
        write_session_stats_header(buffer, &mut offset, &self.header);
        write_session_stats_record(buffer, &mut offset, &self.record);
        write_bytes(buffer, &mut offset, &self.arena);
        Ok(offset)
    }
}

fn diagnostics_to_records(
    builder: &mut BinaryBuilder,
    diagnostics: &[Diagnostic],
) -> Vec<DiagnosticRecord> {
    diagnostics
        .iter()
        .map(|diagnostic| {
            let code = builder.push_string(&diagnostic.code);
            let message = builder.push_string(&diagnostic.message);
            DiagnosticRecord {
                code_offset: code.offset,
                code_len: code.len,
                message_offset: message.offset,
                message_len: message.len,
            }
        })
        .collect()
}

fn graph_delta_chunk_record(
    builder: &mut BinaryBuilder,
    chunk: &GraphDeltaChunk,
) -> GraphDeltaChunkRecord {
    let vertex_id = builder.push_string(&chunk.vertex_id);
    let chunk_id = builder.push_string(&chunk.chunk_id);
    let document_id = builder.push_string(&chunk.document_id.0);
    let note_id = chunk
        .note_id
        .as_ref()
        .map(|note_id| builder.push_string(&note_id.0))
        .unwrap_or_default();
    GraphDeltaChunkRecord {
        vertex_id_offset: vertex_id.offset,
        vertex_id_len: vertex_id.len,
        chunk_id_offset: chunk_id.offset,
        chunk_id_len: chunk_id.len,
        document_id_offset: document_id.offset,
        document_id_len: document_id.len,
        note_id_offset: note_id.offset,
        note_id_len: note_id.len,
        chapter_id: chunk.chapter_id,
        start: chunk.range.start,
        end: chunk.range.end,
        flags: if chunk.note_id.is_some() {
            FLAG_HAS_NOTE_ID
        } else {
            0
        },
    }
}

fn graph_delta_node_record(
    builder: &mut BinaryBuilder,
    node: &GraphDeltaNode,
) -> GraphDeltaNodeRecord {
    let node_id = builder.push_string(&node.node_id);
    let kind = builder.push_string(&node.kind);
    let label = builder.push_string(&node.label);
    let entity_id = node
        .entity_id
        .as_ref()
        .map(|entity_id| builder.push_string(&entity_id.0))
        .unwrap_or_default();
    let document_id = node
        .document_id
        .as_ref()
        .map(|document_id| builder.push_string(&document_id.0))
        .unwrap_or_default();
    let mut flags = 0;
    if node.entity_id.is_some() {
        flags |= FLAG_HAS_ENTITY_ID;
    }
    if node.document_id.is_some() {
        flags |= FLAG_HAS_DOCUMENT_ID;
    }
    GraphDeltaNodeRecord {
        node_id_offset: node_id.offset,
        node_id_len: node_id.len,
        kind_offset: kind.offset,
        kind_len: kind.len,
        label_offset: label.offset,
        label_len: label.len,
        entity_id_offset: entity_id.offset,
        entity_id_len: entity_id.len,
        document_id_offset: document_id.offset,
        document_id_len: document_id.len,
        chapter_id: node.chapter_id.unwrap_or_default(),
        weight: node.weight,
        flags,
    }
}

fn graph_delta_edge_record(
    builder: &mut BinaryBuilder,
    edge: &GraphDeltaEdge,
) -> GraphDeltaEdgeRecord {
    let source_id = builder.push_string(&edge.source_id);
    let target_id = builder.push_string(&edge.target_id);
    let edge_type = builder.push_string(&edge.edge_type);
    GraphDeltaEdgeRecord {
        source_id_offset: source_id.offset,
        source_id_len: source_id.len,
        target_id_offset: target_id.offset,
        target_id_len: target_id.len,
        edge_type_offset: edge_type.offset,
        edge_type_len: edge_type.len,
        weight: edge.weight,
        flags: 0,
    }
}

fn ensure_capacity(buffer: &[u8], required: usize) -> Result<(), StoreError> {
    if buffer.len() < required {
        return Err(StoreError::Query(format!(
            "binary buffer too small: required {required} bytes, had {}",
            buffer.len()
        )));
    }
    Ok(())
}

fn write_bytes(buffer: &mut [u8], offset: &mut usize, bytes: &[u8]) {
    let end = *offset + bytes.len();
    buffer[*offset..end].copy_from_slice(bytes);
    *offset = end;
}

fn write_u32(buffer: &mut [u8], offset: &mut usize, value: u32) {
    write_bytes(buffer, offset, &value.to_le_bytes());
}

fn write_u64(buffer: &mut [u8], offset: &mut usize, value: u64) {
    write_bytes(buffer, offset, &value.to_le_bytes());
}

fn write_i32(buffer: &mut [u8], offset: &mut usize, value: i32) {
    write_bytes(buffer, offset, &value.to_le_bytes());
}

fn write_query_header(buffer: &mut [u8], offset: &mut usize, header: &QueryResultHeader) {
    write_common_header(
        buffer,
        offset,
        header.version,
        header.flags,
        header.session_offset,
        header.session_len,
        header.table1_offset,
        header.table1_count,
        header.table2_offset,
        header.table2_count,
        header.table3_offset,
        header.table3_count,
        header.table4_offset,
        header.table4_count,
        header.arena_offset,
        header.arena_len,
    );
}

fn write_graph_delta_header(
    buffer: &mut [u8],
    offset: &mut usize,
    header: &GraphDeltaResultHeader,
) {
    write_common_header(
        buffer,
        offset,
        header.version,
        header.flags,
        header.session_offset,
        header.session_len,
        header.table1_offset,
        header.table1_count,
        header.table2_offset,
        header.table2_count,
        header.table3_offset,
        header.table3_count,
        header.table4_offset,
        header.table4_count,
        header.arena_offset,
        header.arena_len,
    );
}

fn write_session_state_header(
    buffer: &mut [u8],
    offset: &mut usize,
    header: &SessionStateResultHeader,
) {
    write_common_header(
        buffer,
        offset,
        header.version,
        header.flags,
        header.session_offset,
        header.session_len,
        header.table1_offset,
        header.table1_count,
        header.table2_offset,
        header.table2_count,
        header.table3_offset,
        header.table3_count,
        header.table4_offset,
        header.table4_count,
        header.arena_offset,
        header.arena_len,
    );
}

fn write_session_stats_header(
    buffer: &mut [u8],
    offset: &mut usize,
    header: &SessionStatsResultHeader,
) {
    write_common_header(
        buffer,
        offset,
        header.version,
        header.flags,
        header.session_offset,
        header.session_len,
        header.table1_offset,
        header.table1_count,
        header.table2_offset,
        header.table2_count,
        header.table3_offset,
        header.table3_count,
        header.table4_offset,
        header.table4_count,
        header.arena_offset,
        header.arena_len,
    );
}

fn write_common_header(
    buffer: &mut [u8],
    offset: &mut usize,
    version: u32,
    flags: u32,
    session_offset: u32,
    session_len: u32,
    table1_offset: u32,
    table1_count: u32,
    table2_offset: u32,
    table2_count: u32,
    table3_offset: u32,
    table3_count: u32,
    table4_offset: u32,
    table4_count: u32,
    arena_offset: u32,
    arena_len: u32,
) {
    write_u32(buffer, offset, version);
    write_u32(buffer, offset, flags);
    write_u32(buffer, offset, session_offset);
    write_u32(buffer, offset, session_len);
    write_u32(buffer, offset, table1_offset);
    write_u32(buffer, offset, table1_count);
    write_u32(buffer, offset, table2_offset);
    write_u32(buffer, offset, table2_count);
    write_u32(buffer, offset, table3_offset);
    write_u32(buffer, offset, table3_count);
    write_u32(buffer, offset, table4_offset);
    write_u32(buffer, offset, table4_count);
    write_u32(buffer, offset, arena_offset);
    write_u32(buffer, offset, arena_len);
}

fn write_chunk_hit_record(buffer: &mut [u8], offset: &mut usize, record: &ChunkHitRecord) {
    write_u32(buffer, offset, record.chunk_id_offset);
    write_u32(buffer, offset, record.chunk_id_len);
    write_u64(buffer, offset, record.score_bits);
}

fn write_node_hit_record(buffer: &mut [u8], offset: &mut usize, record: &NodeHitRecord) {
    write_u32(buffer, offset, record.entity_id_offset);
    write_u32(buffer, offset, record.entity_id_len);
    write_u64(buffer, offset, record.score_bits);
}

fn write_diagnostic_record(buffer: &mut [u8], offset: &mut usize, record: &DiagnosticRecord) {
    write_u32(buffer, offset, record.code_offset);
    write_u32(buffer, offset, record.code_len);
    write_u32(buffer, offset, record.message_offset);
    write_u32(buffer, offset, record.message_len);
}

fn write_graph_delta_chunk_record(
    buffer: &mut [u8],
    offset: &mut usize,
    record: &GraphDeltaChunkRecord,
) {
    write_u32(buffer, offset, record.vertex_id_offset);
    write_u32(buffer, offset, record.vertex_id_len);
    write_u32(buffer, offset, record.chunk_id_offset);
    write_u32(buffer, offset, record.chunk_id_len);
    write_u32(buffer, offset, record.document_id_offset);
    write_u32(buffer, offset, record.document_id_len);
    write_u32(buffer, offset, record.note_id_offset);
    write_u32(buffer, offset, record.note_id_len);
    write_u32(buffer, offset, record.chapter_id);
    write_u32(buffer, offset, record.start);
    write_u32(buffer, offset, record.end);
    write_u32(buffer, offset, record.flags);
}

fn write_graph_delta_node_record(
    buffer: &mut [u8],
    offset: &mut usize,
    record: &GraphDeltaNodeRecord,
) {
    write_u32(buffer, offset, record.node_id_offset);
    write_u32(buffer, offset, record.node_id_len);
    write_u32(buffer, offset, record.kind_offset);
    write_u32(buffer, offset, record.kind_len);
    write_u32(buffer, offset, record.label_offset);
    write_u32(buffer, offset, record.label_len);
    write_u32(buffer, offset, record.entity_id_offset);
    write_u32(buffer, offset, record.entity_id_len);
    write_u32(buffer, offset, record.document_id_offset);
    write_u32(buffer, offset, record.document_id_len);
    write_u32(buffer, offset, record.chapter_id);
    write_i32(buffer, offset, record.weight);
    write_u32(buffer, offset, record.flags);
}

fn write_graph_delta_edge_record(
    buffer: &mut [u8],
    offset: &mut usize,
    record: &GraphDeltaEdgeRecord,
) {
    write_u32(buffer, offset, record.source_id_offset);
    write_u32(buffer, offset, record.source_id_len);
    write_u32(buffer, offset, record.target_id_offset);
    write_u32(buffer, offset, record.target_id_len);
    write_u32(buffer, offset, record.edge_type_offset);
    write_u32(buffer, offset, record.edge_type_len);
    write_i32(buffer, offset, record.weight);
    write_u32(buffer, offset, record.flags);
}

fn write_string_ref_record(buffer: &mut [u8], offset: &mut usize, record: &StringRefRecord) {
    write_u32(buffer, offset, record.offset);
    write_u32(buffer, offset, record.len);
}

fn write_session_document_record(
    buffer: &mut [u8],
    offset: &mut usize,
    record: &SessionDocumentRecord,
) {
    write_u32(buffer, offset, record.document_id_offset);
    write_u32(buffer, offset, record.document_id_len);
    write_u32(buffer, offset, record.note_id_offset);
    write_u32(buffer, offset, record.note_id_len);
    write_u32(buffer, offset, record.chapter_titles_start);
    write_u32(buffer, offset, record.chapter_titles_count);
    write_u32(buffer, offset, record.chapter_count);
    write_u32(buffer, offset, record.parent_count);
    write_u32(buffer, offset, record.leaf_count);
    write_u32(buffer, offset, record.entity_count);
    write_u32(buffer, offset, record.discovery_count);
    write_u32(buffer, offset, record.flags);
    write_u64(buffer, offset, record.updated_at_bits);
}

fn write_session_stats_record(buffer: &mut [u8], offset: &mut usize, record: &SessionStatsRecord) {
    write_u32(buffer, offset, record.document_count);
    write_u32(buffer, offset, record.chapter_count);
    write_u32(buffer, offset, record.parent_count);
    write_u32(buffer, offset, record.leaf_count);
    write_u32(buffer, offset, record.entity_count);
    write_u32(buffer, offset, record.discovery_candidate_count);
    write_u32(buffer, offset, record.graph_vertex_count);
    write_u32(buffer, offset, record.graph_edge_count);
    write_u32(buffer, offset, record.span_count);
    write_u64(buffer, offset, record.updated_at_bits);
}
