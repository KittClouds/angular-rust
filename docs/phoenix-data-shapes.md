# Phoenix Data Shapes

This document maps the current Phoenix-only data flow from Angular into the worker, across the packet buffer boundary, through Rust `phoenix-wasm`, into scanner and graptor pipelines, into Cozo relations, and out to OPFS snapshot storage.

Source-of-truth files:

- `src/app/services/phoenix-wasm.service.ts`
- `src/app/workers/phoenix.worker.ts`
- `src/app/lib/phoenix/wasm-protocol.ts`
- `rust/phoenix/crates/phoenix-types/src/lib.rs`
- `rust/phoenix/crates/phoenix-types/src/binary.rs`
- `rust/phoenix/crates/phoenix-runtime/src/binary.rs`
- `rust/phoenix/crates/phoenix-wasm/src/lib.rs`
- `rust/phoenix/crates/phoenix-wasm/src/opfs.rs`
- `rust/phoenix/crates/phoenix-scanner/src/lib.rs`
- `rust/phoenix/crates/phoenix-graptor/src/lib.rs`
- `rust/phoenix/crates/phoenix-store-cozo/src/schema.rs`
- `rust/phoenix/crates/phoenix-store-cozo/src/lib.rs`

Not covered here:

- GoKitt
- legacy Go discovery registry semantics
- pre-Phoenix scanner paths

## Live Path

The live Phoenix path is:

1. `src/app/services/phoenix-wasm.service.ts`
2. `src/app/workers/phoenix.worker.ts`
3. `rust/phoenix/crates/phoenix-wasm`
4. `phoenix-scanner`
5. `phoenix-graptor`
6. `phoenix-store-cozo`
7. OPFS snapshot persistence

At a high level:

- Angular builds a packet buffer and writes a `PacketHeader` plus payload bytes.
- The worker receives the buffer, loads `phoenix_wasm.wasm`, copies the packet into WASM linear memory, and calls `phoenix_process_packet_at`.
- Rust decodes the request, runs the requested runtime/scanner/graptor/store action, and writes the response back into the same packet region.
- The worker copies the response bytes back into the original shared or transferable buffer.
- Angular decodes either JSON payloads or one of the compact binary result layouts.

## Wire Format

### Packet Region

The outer packet region is defined by `PacketHeader` in both TS and Rust.

`PacketHeader` shape:

| Field | Type | Meaning |
| --- | --- | --- |
| `ready` | `u32` | Packet slot is populated. TS writes `1`. |
| `kind` | `u32` | `PacketKind` discriminant. |
| `request_id` / `requestId` | `u32` | Correlates request and response. |
| `payload_len` / `payloadLen` | `u32` | Number of bytes after the 16-byte header. |

Details:

- Header size is always `16` bytes.
- TS writes it with `writePacketHeader(...)` in `src/app/lib/phoenix/wasm-protocol.ts`.
- Rust defines the same layout in `phoenix-types/src/lib.rs` as `PacketHeader`.

### Packet Kinds Used by the Angular Phoenix Client

The Angular client currently names these request kinds in `src/app/lib/phoenix/wasm-protocol.ts`:

| Kind | Number | Purpose |
| --- | --- | --- |
| `status` | `1` | Status probe |
| `initRuntimeRequest` | `2` | Runtime bootstrap |
| `createSessionRequest` | `4` | Session creation |
| `commitRequest` | `6` | Commit |
| `rebuildRequest` | `8` | Rebuild |
| `ingestRequest` | `10` | JSON ingest |
| `queryRequest` | `12` | JSON query |
| `snapshotExportRequest` | `14` | Snapshot export |
| `snapshotImportRequest` | `16` | Snapshot import |
| `scanRequest` | `17` | JSON scan |
| `structureRequest` | `19` | JSON structure |
| `graphDeltaRequest` | `21` | Graph delta |
| `sessionStateRequest` | `23` | Session state |
| `sessionStatsRequest` | `25` | Session stats |
| `analyzeTextRequest` | `27` | Text analysis |
| `queryBinaryRequest` | `29` | Compact binary query |
| `storeCommandRequest` | `34` | Store command |
| `embedUpsertBinaryRequest` | `36` | Compact binary embedding upsert |

Rust `PacketKind` contains the full request/result enum, including result kinds such as `IngestResult`, `ScanResult`, `GraphDeltaResult`, `SessionStateResult`, `SessionStatsResult`, and `Ack`.

### Shared Buffer Boundary

The browser-side shared-memory story is:

- `PhoenixWasmService.createPacketBuffer(...)` prefers `SharedArrayBuffer`.
- If `SharedArrayBuffer` is unavailable or the page is not isolated, it falls back to `ArrayBuffer`.
- The worker accepts either in `PROCESS_PACKET`.
- If the buffer is a plain `ArrayBuffer`, Angular transfers ownership to the worker with `postMessage(..., [buffer])`.

Important nuance:

- The main thread and worker can share the packet region with `SharedArrayBuffer`.
- Phoenix still copies packet bytes into WASM linear memory before Rust processes them.
- So the transport is shared-memory between JS threads, but not zero-copy into the WASM runtime itself.

Worker flow in `src/app/workers/phoenix.worker.ts`:

1. Receive `PROCESS_PACKET { capacity, buffer }`.
2. Create a `Uint8Array` view over the provided buffer.
3. Allocate a region inside WASM memory with `phoenix_alloc(capacity)`.
4. Copy request bytes from JS buffer into WASM memory.
5. Call `phoenix_process_packet_at(ptr, capacity)`.
6. Read the response header from WASM memory.
7. Copy the used response bytes back into the original JS buffer.
8. Return `PROCESS_PACKET_RESULT { status }`.

### Protocol Versions and Flags

Current protocol constants in `src/app/lib/phoenix/wasm-protocol.ts`:

- `PROTOCOL_VERSION = 6`
- `BINARY_REQUEST_LAYOUT_VERSION = 2`
- `DEFAULT_PACKET_REGION_SIZE = 64 * 1024`

Current request flags used by the TS client:

- `REQUEST_FLAG_HAS_SESSION`
- `REQUEST_FLAG_HAS_TEMPORAL`
- `REQUEST_FLAG_TARGET_CHUNKS`
- `REQUEST_FLAG_TARGET_NODES`
- `REQUEST_FLAG_TARGET_GRAPH`
- `REQUEST_FLAG_TARGET_SEMANTIC`
- `REQUEST_FLAG_INCLUDE_CANDIDATE_GRAPH`

Current binary result layout version in Rust:

- `BINARY_LAYOUT_VERSION = 1`

### Binary Request Layouts

The compact binary request headers are defined in `rust/phoenix/crates/phoenix-types/src/binary.rs`.

| Header | Purpose | Shape |
| --- | --- | --- |
| `QueryBinaryRequestHeader` | Query request | Fixed header with flags, session ref, query ref, scope refs, limit, optional temporal ref, optional query vector ref, arena ref |
| `AnalyzeTextBinaryRequestHeader` | Analyze-text request | Fixed header with flags, text ref, arena ref |
| `IngestBinaryRequestHeader` | Ingest request | Fixed header with flags, optional session ref, document table offset/count, arena ref |
| `IngestDocumentBinaryRecord` | One ingested document | Document id ref, note id ref, title ref, text ref, scope refs, flags |
| `ScanBinaryRequestHeader` | Scan request | Fixed header with session ref, text ref, scope refs, resolver-seed JSON ref, arena ref |
| `StructureBinaryRequestHeader` | Structure request | Fixed header with text ref, scan-artifact JSON ref, arena ref |
| `EmbedUpsertBinaryRequestHeader` | Embedding upsert | Version, count, dimension, arena offset |

All string and JSON-heavy values live in an arena section. The header stores offsets and lengths into that arena.

### Binary Result Layouts

Phoenix currently uses a common 56-byte binary result header shape for:

- `QueryResultHeader`
- `GraphDeltaResultHeader`
- `SessionStateResultHeader`
- `SessionStatsResultHeader`

Common result header fields:

| Field | Type |
| --- | --- |
| `version` | `u32` |
| `flags` | `u32` |
| `session_offset` | `u32` |
| `session_len` | `u32` |
| `table1_offset` | `u32` |
| `table1_count` | `u32` |
| `table2_offset` | `u32` |
| `table2_count` | `u32` |
| `table3_offset` | `u32` |
| `table3_count` | `u32` |
| `table4_offset` | `u32` |
| `table4_count` | `u32` |
| `arena_offset` | `u32` |
| `arena_len` | `u32` |

The TS service decodes these payloads in `src/app/services/phoenix-wasm.service.ts`.

#### Query Result

Rust source type: `QueryResult`

TS decode interface: `PhoenixQueryBinaryResult`

Table assignment:

| Table | Record type | Shape |
| --- | --- | --- |
| `table1` | `ChunkHitRecord` | `chunk_id_offset`, `chunk_id_len`, `score_bits` |
| `table2` | `NodeHitRecord` | `entity_id_offset`, `entity_id_len`, `score_bits` |
| `table3` | `DiagnosticRecord` | code/message string refs |
| `table4` | unused | `0` |

Decoded TS shape:

```ts
{
  sessionId: string;
  chunkHits: { chunkId: string; score: number }[];
  nodeHits: { entityId: string; score: number }[];
  diagnostics: { code: string; message: string }[];
}
```

#### Graph Delta Result

Rust source type: `GraphDeltaResult`

TS decode interface: `PhoenixGraphDeltaBinaryResult`

Table assignment:

| Table | Record type | Byte len | Meaning |
| --- | --- | --- | --- |
| `table1` | `GraphDeltaChunkRecord` | `48` | chunk vertices |
| `table2` | `GraphDeltaNodeRecord` | `52` | entity/event/other nodes |
| `table3` | `GraphDeltaEdgeRecord` | `32` | graph edges |
| `table4` | `DiagnosticRecord` | `16` | diagnostics |

Decoded TS shape:

```ts
{
  sessionId: string;
  chunks: {
    vertexId: string;
    chunkId: string;
    documentId: string;
    noteId?: string;
    chapterId: number;
    start: number;
    end: number;
  }[];
  nodes: {
    nodeId: string;
    kind: string;
    label: string;
    entityId?: string;
    documentId?: string;
    chapterId?: number;
    weight: number;
  }[];
  edges: {
    sourceId: string;
    targetId: string;
    edgeType: string;
    weight: number;
  }[];
  diagnostics: PhoenixDiagnostic[];
}
```

#### Session State Result

Rust source type: `SessionState`

TS decode interface: `PhoenixSessionStateBinaryResult`

Table assignment:

| Table | Record type | Byte len | Meaning |
| --- | --- | --- | --- |
| `table1` | `SessionDocumentRecord` | `56` | one row per document in the session |
| `table2` | `StringRefRecord` | `8` | chapter title refs |
| `table3` | `StringRefRecord` | `8` | manifest namespace refs |
| `table4` | unused | `0` | unused |

Decoded TS shape:

```ts
{
  sessionId: string;
  documents: {
    documentId: string;
    noteId?: string;
    chapterTitles: string[];
    chapterCount: number;
    parentCount: number;
    leafCount: number;
    entityCount: number;
    discoveryCount: number;
    hasFrontMatterChapter: boolean;
    updatedAt: number;
  }[];
  manifestNamespaces: string[];
}
```

#### Session Stats Result

Rust source type: `SessionStats`

TS decode interface: `PhoenixSessionStatsBinaryResult`

Table assignment:

| Table | Record type | Byte len |
| --- | --- | --- |
| `table1` | `SessionStatsRecord` | `44` |
| `table2` | unused | `0` |
| `table3` | unused | `0` |
| `table4` | unused | `0` |

Decoded TS shape:

```ts
{
  sessionId: string;
  documentCount: number;
  chapterCount: number;
  parentCount: number;
  leafCount: number;
  entityCount: number;
  discoveryCandidateCount: number;
  graphVertexCount: number;
  graphEdgeCount: number;
  spanCount: number;
  updatedAt: number;
}
```

## Scanner Shapes

The scanner boundary is defined by `phoenix-types/src/lib.rs` and implemented by `rust/phoenix/crates/phoenix-scanner/src/lib.rs`.

### Request Shape

`ScanRequest`:

```ts
{
  text: string;
  scope: ScopeKey;
  sessionId?: SessionId;
  resolverSeed: ResolverEntitySeed[];
}
```

Supporting shapes:

- `ScopeKey`
  - `worldId?: string`
  - `narrativeId?: string`
  - `folderId?: string`
  - `folderPath?: string`
- `ResolverEntitySeed`
  - `entityId`
  - `canonicalName`
  - `aliases: string[]`
  - `kind?: EntityKind`
  - `gender?: GenderHint`
  - `number?: NumberHint`
  - `scope: ScopeKey`

### Output Shape

`ScanArtifact`:

```ts
{
  sentences: SentenceSpan[];
  tokens: TokenSpan[];
  mentions: MentionSpan[];
  chunks: ChunkSpan[];
  resolverLinks: ResolverLink[];
  narrativeHits: NarrativeVerbHit[];
  diagnostics: Diagnostic[];
}
```

Core scanner payloads:

| Type | Important fields |
| --- | --- |
| `SentenceSpan` | `index`, `range` |
| `TokenSpan` | `range`, `tokenClass`, `pos`, `masked`, `capitalized` |
| `MentionSpan` | `range`, `surface`, `kind`, `entityRef`, `source`, `confidence`, `sentenceIndex` |
| `ChunkSpan` | `kind`, `range`, `head`, `modifiers`, `sentenceIndex` |
| `ResolverLink` | `sourceRange`, `targetRange`, `targetEntity`, `linkKind`, `confidence`, `sentenceIndex` |
| `NarrativeVerbHit` | `range`, `lemma`, `eventClass`, `relationType`, `transitivity`, `sentenceIndex`, `confidence` |

### Discovery in Scanner Output

Phoenix discovery is part of the scanner, not a separate legacy registry pipeline.

The scanner flow in `phoenix-scanner/src/lib.rs` is:

1. Seed resolver state from `resolver_seed`.
2. Build exact known-entity candidates from the lexicon.
3. Add fuzzy candidates if configured.
4. Build first-pass artifacts.
5. Run `build_discovery_mentions(...)`.
6. Add discovery mentions to the candidate set.
7. Re-run final pass and resolver links.

Discovery surfaces are emitted as normal `MentionSpan` records with:

- `source = MentionSource::Discovery`
- `entity_ref = MentionEntityRef::Speculative(...)` or no resolved known entity
- `confidence` from the discovery heuristics

This matters because the scanner output shape does not contain a separate `DiscoveryRecord` list. Discovery first appears on the scanner boundary as mention records.

## Graptor Shapes

The ingest boundary is defined by `IngestDocument`, `IngestRequest`, `IngestResult`, and related summary structs in `phoenix-types/src/lib.rs`, then implemented in `rust/phoenix/crates/phoenix-graptor/src/lib.rs`.

### Ingest Request

`IngestDocument`:

```ts
{
  documentId: DocumentId;
  noteId?: NoteId;
  title: string;
  text: string;
  scope: ScopeKey;
}
```

`IngestRequest`:

```ts
{
  sessionId?: SessionId;
  documents: IngestDocument[];
  commit: boolean;
}
```

### Ingest Result

`IngestResult`:

```ts
{
  sessionId?: SessionId;
  documentCount: number;
  warningCount: number;
  documents: IngestDocumentSummary[];
  chunkStats?: ChunkStats;
  graphSummary?: GraphSummary;
  entitySummary?: EntitySummary;
  discoverySummary?: DiscoverySummary;
  retrievalSummary?: RetrievalSummary;
  relationCounts: RelationCount[];
  diagnostics: Diagnostic[];
}
```

Important summary structs:

- `IngestDocumentSummary`
  - `documentId`
  - `noteId`
  - `chapterCount`
  - `boundaryCount`
  - `parentCount`
  - `leafCount`
  - `entityCount`
  - `edgeCount`
  - `hasFrontMatterChapter`
  - `hasFrontMatterBoundary`
- `DiscoverySummary`
  - `candidateCount`
  - `mentionCount`
  - `persistedCount`
- `SessionState`
  - `sessionId`
  - `documents`
  - `manifestNamespaces`
  - `updatedAt`
- `SessionStats`
  - `sessionId`
  - `documentCount`
  - `chapterCount`
  - `boundaryCount`
  - `parentCount`
  - `leafCount`
  - `entityCount`
  - `discoveryCandidateCount`
  - `graphVertexCount`
  - `graphEdgeCount`
  - `spanCount`
  - `updatedAt`

### Internal Graptor Materialization Shapes

Inside graptor, the important transient shapes are:

- scanner `ScanArtifact`
- structure output for chunk and boundary segmentation
- `RelationCandidate`
- mention records resolved against the entity registry
- evidence spans

`RelationCandidate` shape:

```ts
{
  sentenceIndex: number;
  verbRange: TextRange;
  lemma: string;
  eventClass: string;
  relationType: string;
  subject?: FrameSlot;
  object?: FrameSlot;
  recipient?: FrameSlot;
  attachments: TextRange[];
  evidence: EvidenceSpan[];
}
```

Materialization steps inside `phoenix-graptor/src/lib.rs`:

1. Segment document into chapter, parent, and leaf chunks.
2. Scan each leaf with `PhoenixScanner`.
3. Convert mentions into entity-linked `MentionRecord`s.
4. Persist discovery candidates separately into `discovery_candidates`.
5. Persist mention spans and span/entity joins into `spans` and `span_mentions`.
6. Convert `RelationCandidate` values into graph event vertices, graph edges, legacy `edges`, and evidence span rows.
7. Flush buffered rows into Cozo relations through `CompactRelationBuffer`.

## Cozo Relation Shapes

The Phoenix ingest outputs relevant to this map are defined in `rust/phoenix/crates/phoenix-store-cozo/src/schema.rs`.

### `chunkid_map`

Purpose: numeric chunk id to stable search key mapping

| Column | Type |
| --- | --- |
| `id` | `Int` key |
| `chunk_key` | `String` |
| `doc_id` | `String` |
| `created_at` | `Int` |

Common row producers:

- chapter chunk id rows
- parent chunk id rows
- leaf chunk id rows

### `chunks`

Purpose: persisted chunk tree

| Column | Type |
| --- | --- |
| `chunk_id` | `Int` key |
| `doc_id` | `String` |
| `level` | `Int` |
| `start` | `Int` |
| `end` | `Int` |
| `text` | `String` |
| `parent_id` | `Int?` |
| `scope_narrative` | `String?` |
| `scope_folder` | `String?` |
| `created_at` | `Int` |

Current level convention in graptor row builders:

- `2` = chapter
- `1` = parent chunk
- `0` = leaf chunk

### `document_boundaries`

Purpose: chapter/heading/section/act boundary records

| Column | Type |
| --- | --- |
| `doc_id` | `String` key |
| `boundary_id` | `Int` key |
| `kind` | `String` |
| `depth` | `Int` |
| `label` | `String?` |
| `ordinal` | `Int` |
| `parent_boundary_id` | `Int?` |
| `note_id` | `String?` |
| `start_char` | `Int` |
| `end_char` | `Int?` |
| `created_at` | `Int` |

### `spans`

Purpose: mention spans and evidence spans

| Column | Type |
| --- | --- |
| `id` | `String` key |
| `world_id` | `String?` |
| `note_id` | `String?` |
| `narrative_id` | `String?` |
| `start` | `Int?` |
| `end` | `Int?` |
| `text` | `String?` |
| `content_hash` | `String?` |
| `span_kind` | `String?` |
| `status` | `String?` |
| `created_by` | `String?` |
| `created_at` | `Int?` |
| `updated_at` | `Int?` |

Current graptor row builders use:

- mention spans: `span_kind = "entity_mention"`, `status = "resolved"`, `created_by = "graptor"`
- evidence spans: `span_kind = evidence.kind || "evidence"`, `status = "derived"`, `created_by = "graptor"`

### `span_mentions`

Purpose: join a span to a candidate/resolved entity

| Column | Type |
| --- | --- |
| `id` | `String` key |
| `span_id` | `String?` |
| `candidate_entity_id` | `String?` |
| `match_type` | `String?` |
| `confidence` | `Float?` |
| `ev_frequency` | `Float?` |
| `ev_capital_ratio` | `Float?` |
| `ev_context_score` | `Float?` |
| `ev_cooccurrence` | `Float?` |
| `status` | `String?` |
| `created_at` | `Int?` |
| `updated_at` | `Int?` |

Current graptor mention rows use:

- `match_type = "exact"`
- `status = "resolved"`

### `discovery_candidates`

Purpose: persisted speculative discovery outputs

| Column | Type |
| --- | --- |
| `token` | `String` key |
| `kind` | `Int?` |
| `score` | `Float?` |
| `status` | `Int?` |
| `last_seen` | `Int?` |
| `first_seen` | `Int?` |
| `count` | `Int?` |

Current graptor discovery row builder writes:

```json
{
  "token": "<stable discovery row id>",
  "kind": 0,
  "score": <confidence>,
  "status": 1,
  "last_seen": <now_ms>,
  "first_seen": <now_ms>,
  "count": 1
}
```

Important nuance:

- the relation key is called `token`
- graptor currently stores a stable synthetic discovery id there, not the raw surface string
- UI-side discovery candidate stores still normalize and coalesce on `token`

### `edges`

Purpose: compact legacy relationship rows, separate from graph-edge persistence

| Column | Type |
| --- | --- |
| `id` | `String` key |
| `source_id` | `String` |
| `target_id` | `String` |
| `rel_type` | `String` |
| `confidence` | `Float` |
| `bidirectional` | `Bool` |
| `source_note` | `String?` |
| `created_at` | `Int` |

Current relation rows use relation-role suffixes such as:

- `attacks:object`
- `speaksTo:recipient`

### `graph_vertices`

Purpose: graph nodes for chunks, entities, events, and other materialized graph objects

| Column | Type |
| --- | --- |
| `id` | `String` key |
| `document_id` | `String?` |
| `narrative_id` | `String?` |
| `value` | `Json` |
| `weight` | `Int` |
| `attributes` | `Json` |

The `value` object is type-specific. Examples:

- parent chunk vertex
  - `{"kind":"parent","chunkId":...,"chapterId":...,"boundaryId":...}`
- event vertex
  - `{"kind":"event","lemma":...,"eventClass":...,"relationType":...}`

### `graph_edges`

Purpose: richer typed graph-edge persistence

| Column | Type |
| --- | --- |
| `source_id` | `String` key |
| `target_id` | `String` key |
| `document_id` | `String?` |
| `narrative_id` | `String?` |
| `valid_from_doc` | `String?` |
| `valid_from_boundary` | `Int?` |
| `valid_to_doc` | `String?` |
| `valid_to_boundary` | `Int?` |
| `assertion_kind` | `String?` |
| `weight` | `Int` |
| `attributes` | `Json` |
| `data` | `Json?` |
| `edge_type` | `String` |

The row builder pulls several fields from the `attributes` JSON:

- `documentId`
- `boundaryId`
- `validToBoundary`
- `assertionKind`

### `graph_properties`

Purpose: time-aware property rows attached to graph owners

| Column | Type |
| --- | --- |
| `owner_id` | `String` key |
| `owner_type` | `String` key |
| `key` | `String` key |
| `valid_from` | `Int` key |
| `value_type` | `String` |
| `value_blob` | `Json` |
| `valid_until` | `Int?` |
| `txn_id` | `Int` |

## OPFS Snapshot Shape

The OPFS layer is in `rust/phoenix/crates/phoenix-wasm/src/opfs.rs`.

Paths:

- primary snapshot: `/phoenix/runtime.snapshot.json`
- backup snapshot: `/phoenix/runtime.snapshot.json.bak`
- temp prefix: `/phoenix/runtime.snapshot.tmp-<timestamp>`

Important nuance:

- Despite the `.json` filename, the saved payload is the raw snapshot byte stream returned by `runtime.export_snapshot()`.
- Current export comes from `phoenix-store-cozo` binary snapshot encoding.
- Import still supports older legacy JSON snapshots for compatibility, but current export is binary.

Current save/load behavior:

1. Export snapshot bytes from the runtime.
2. Reject snapshots larger than `64 MiB`.
3. Ensure `/phoenix` exists.
4. Write temp file.
5. Copy current primary to backup if present.
6. Copy temp file to primary.
7. Delete temp file.
8. On load, prefer primary.
9. If primary is missing, try backup.
10. If neither exists, report no snapshot.

`LoadSnapshot` shape:

```ts
{
  bytes?: Uint8Array;
  recoveredFromBackup: boolean;
}
```

Snapshot export/import boundary in Rust:

- `phoenix-wasm` packet handling supports `SnapshotExportRequest` and `SnapshotImportRequest`
- `phoenix-runtime` delegates to `phoenix-store-cozo`
- `phoenix-store-cozo` exports a snapshot envelope made of relation blocks

Current snapshot payload format:

- binary snapshot envelope
- relation block headers with relation ids
- per-relation compact row payloads
- optional compression at the block level

## End-to-End Example

This example shows the shape transitions for a single ingested document.

### 1. Angular Builds an Ingest Packet

Representative JSON ingest request:

```json
{
  "sessionId": "session-1",
  "documents": [
    {
      "documentId": "doc-1",
      "noteId": "note-1",
      "title": "Chapter One",
      "text": "Aria attacked Bram in the courtyard.",
      "scope": {
        "worldId": "world-1",
        "narrativeId": "narrative-1",
        "folderId": "folder-1"
      }
    }
  ],
  "commit": true
}
```

Angular writes:

- 16-byte `PacketHeader`
- payload bytes after it
- buffer type is `SharedArrayBuffer` when available, otherwise `ArrayBuffer`

### 2. Worker Copies the Packet into WASM

The worker sees:

```ts
{
  type: 'PROCESS_PACKET',
  id: 42,
  capacity: 65536,
  buffer: SharedArrayBuffer | ArrayBuffer
}
```

It copies:

- request buffer -> WASM linear memory
- response bytes -> same JS-visible packet region

### 3. Rust Ingests the Document

`IngestRequest` becomes `BorrowedIngestRequest`, then graptor processes one document.

Chunk persistence examples:

`document_boundaries` row:

```json
{
  "doc_id": "doc-1",
  "boundary_id": 1,
  "kind": "chapter",
  "depth": 0,
  "label": "Chapter One",
  "ordinal": 1,
  "parent_boundary_id": null,
  "note_id": "note-1",
  "start_char": 0,
  "end_char": 39,
  "created_at": 1710000000000
}
```

`chunks` leaf row:

```json
{
  "chunk_id": 1001,
  "doc_id": "doc-1",
  "level": 0,
  "start": 0,
  "end": 39,
  "text": "Aria attacked Bram in the courtyard.",
  "parent_id": 1000,
  "scope_narrative": "narrative-1",
  "scope_folder": "folder-1",
  "created_at": 1710000000000
}
```

`chunkid_map` row:

```json
{
  "id": 1001,
  "chunk_key": "<leaf-search-id>",
  "doc_id": "doc-1",
  "created_at": 1710000000000
}
```

### 4. Scanner Produces a `ScanArtifact`

Representative scanner output shape:

```json
{
  "sentences": [
    { "index": 0, "range": { "start": 0, "end": 39 } }
  ],
  "mentions": [
    {
      "range": { "start": 0, "end": 4 },
      "surface": "Aria",
      "source": "known",
      "confidence": 1.0,
      "sentenceIndex": 0
    },
    {
      "range": { "start": 15, "end": 19 },
      "surface": "Bram",
      "source": "discovery",
      "confidence": 0.81,
      "sentenceIndex": 0
    }
  ],
  "resolverLinks": [],
  "narrativeHits": [
    {
      "range": { "start": 5, "end": 13 },
      "lemma": "attack",
      "eventClass": "conflict",
      "relationType": "attacks",
      "sentenceIndex": 0,
      "confidence": 0.95
    }
  ]
}
```

### 5. Graptor Persists Discovery, Mention, Relation, and Graph Rows

Persisted discovery candidate row:

```json
{
  "token": "<stable-discovery-row-id>",
  "kind": 0,
  "score": 0.81,
  "status": 1,
  "last_seen": 1710000000000,
  "first_seen": 1710000000000,
  "count": 1
}
```

Persisted mention span row:

```json
{
  "id": "<stable-span-id>",
  "world_id": "world-1",
  "note_id": "note-1",
  "narrative_id": "narrative-1",
  "start": 0,
  "end": 4,
  "text": "Aria",
  "content_hash": "<stable-span-hash>",
  "span_kind": "entity_mention",
  "status": "resolved",
  "created_by": "graptor",
  "created_at": 1710000000000,
  "updated_at": 1710000000000
}
```

Persisted span/entity join row:

```json
{
  "id": "<stable-span-mention-id>",
  "span_id": "<stable-span-id>",
  "candidate_entity_id": "entity-aria",
  "match_type": "exact",
  "confidence": 1.0,
  "status": "resolved",
  "created_at": 1710000000000,
  "updated_at": 1710000000000
}
```

Persisted event vertex row:

```json
{
  "id": "<stable-event-id>",
  "document_id": "doc-1",
  "narrative_id": "narrative-1",
  "value": {
    "kind": "event",
    "lemma": "attack",
    "eventClass": "conflict",
    "relationType": "attacks"
  },
  "weight": 1,
  "attributes": {
    "documentId": "doc-1",
    "noteId": "note-1",
    "chapterId": 1,
    "boundaryId": 1,
    "searchChunkId": "<leaf-search-id>",
    "verbRange": { "start": 5, "end": 13 }
  }
}
```

Persisted graph edge row:

```json
{
  "source_id": "entity::entity-aria",
  "target_id": "<stable-event-id>",
  "document_id": "doc-1",
  "narrative_id": "narrative-1",
  "valid_from_doc": "doc-1",
  "valid_from_boundary": 1,
  "assertion_kind": "current",
  "weight": 100,
  "attributes": {
    "role": "subject",
    "documentId": "doc-1",
    "boundaryId": 1,
    "assertionKind": "current"
  },
  "data": null,
  "edge_type": "event_subject"
}
```

Persisted compact edge row:

```json
{
  "id": "<stable-edge-id>",
  "source_id": "entity-aria",
  "target_id": "entity-bram",
  "rel_type": "attacks:object",
  "confidence": 0.95,
  "bidirectional": false,
  "source_note": "note-1",
  "created_at": 1710000000000
}
```

### 6. Rust Returns an `IngestResult`

Representative result shape:

```json
{
  "sessionId": "session-1",
  "documentCount": 1,
  "warningCount": 0,
  "documents": [
    {
      "documentId": "doc-1",
      "noteId": "note-1",
      "chapterCount": 1,
      "boundaryCount": 1,
      "parentCount": 1,
      "leafCount": 1,
      "entityCount": 2,
      "edgeCount": 1,
      "hasFrontMatterChapter": false,
      "hasFrontMatterBoundary": false
    }
  ],
  "discoverySummary": {
    "candidateCount": 1,
    "mentionCount": 1,
    "persistedCount": 1
  }
}
```

### 7. Snapshot Export and OPFS Persistence

When Phoenix saves to OPFS:

1. Runtime calls `export_snapshot()`.
2. Cozo emits a binary snapshot envelope containing compact rows partitioned by relation.
3. `phoenix-wasm` saves those raw bytes to `/phoenix/runtime.snapshot.json`.
4. A backup is maintained at `/phoenix/runtime.snapshot.json.bak`.

That means the OPFS persistence boundary shape is:

- input: opaque snapshot byte vector
- storage unit: one full exported snapshot payload
- restore: import the full payload back into Cozo relations

## Quick Answers

If you are asking “what shape is it here?”:

- Angular to worker: `PacketHeader` + payload bytes inside `SharedArrayBuffer` or `ArrayBuffer`
- worker to WASM: copied packet bytes in WASM linear memory
- scanner input: `ScanRequest`
- scanner output: `ScanArtifact`
- graptor input: `IngestRequest` with `IngestDocument[]`
- graptor output: `IngestResult`
- persisted discovery: rows in `discovery_candidates`
- persisted mentions: rows in `spans` and `span_mentions`
- persisted graph: rows in `graph_vertices`, `graph_edges`, `graph_properties`
- persisted compact relationships: rows in `edges`
- OPFS: full exported snapshot byte stream, not live row-by-row JSON
