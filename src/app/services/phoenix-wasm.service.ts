import { Injectable, NgZone, inject } from '@angular/core';
import {
    BINARY_REQUEST_LAYOUT_VERSION,
    decodePacketHeader,
    DEFAULT_PACKET_REGION_SIZE,
    isRetriablePacketMessage,
    PACKET_HEADER_SIZE,
    PACKET_KIND,
    PhoenixChatStreamEvent,
    PhoenixChatStreamRequest,
    PROTOCOL_VERSION,
    REQUEST_FLAG_HAS_SESSION,
    REQUEST_FLAG_HAS_TEMPORAL,
    REQUEST_FLAG_INCLUDE_CANDIDATE_GRAPH,
    REQUEST_FLAG_TARGET_CHUNKS,
    REQUEST_FLAG_TARGET_GRAPH,
    REQUEST_FLAG_TARGET_NODES,
    REQUEST_FLAG_TARGET_SEMANTIC,
    tryDecodeJson,
    writePacketHeader,
} from '../lib/phoenix/wasm-protocol';
import { EmbeddingWorkerService } from '../lib/services/embedding-worker.service';
import {
    NliWorkerService,
    type NliClassificationResult,
    type NliPairClassificationInput,
} from '../lib/services/nli-worker.service';
import { createNoopNgZone, createWorkerOutsideAngular } from '../lib/core/worker-zone';

export interface PhoenixScope {
    worldId?: string;
    narrativeId?: string;
    folderId?: string;
    folderPath?: string;
}

export interface PhoenixChunkHit {
    chunkId: string;
    score: number;
}

export interface PhoenixNodeHit {
    entityId: string;
    score: number;
}

export interface PhoenixDiagnostic {
    code: string;
    message: string;
}

export interface PhoenixQueryBinaryResult {
    sessionId: string;
    chunkHits: PhoenixChunkHit[];
    nodeHits: PhoenixNodeHit[];
    diagnostics: PhoenixDiagnostic[];
}

export interface PhoenixGraphDeltaBinaryResult {
    sessionId: string;
    chunks: Array<{
        vertexId: string;
        chunkId: string;
        documentId: string;
        noteId?: string;
        chapterId: number;
        start: number;
        end: number;
    }>;
    nodes: Array<{
        nodeId: string;
        kind: string;
        label: string;
        entityId?: string;
        documentId?: string;
        chapterId?: number;
        weight: number;
    }>;
    edges: Array<{ sourceId: string; targetId: string; edgeType: string; weight: number }>;
    diagnostics: PhoenixDiagnostic[];
}

export interface PhoenixSessionStateBinaryResult {
    sessionId: string;
    documents: Array<{
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
    }>;
    manifestNamespaces: string[];
}

export interface PhoenixSessionStatsBinaryResult {
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

export type PhoenixSnapshotPartition = 'all' | 'content' | 'derived';

type PacketResponse = {
    kind: number;
    requestId: number;
    bytes: Uint8Array;
    json: any;
};

type PhoenixWorkerMessage =
    | { type: 'INIT'; id: number }
    | { type: 'PROCESS_PACKET'; id: number; capacity: number; buffer: SharedArrayBuffer | ArrayBuffer }
    | { type: 'CHAT_STREAM_START'; id: number; request: PhoenixChatStreamRequest }
    | { type: 'CHAT_STREAM_CANCEL'; id: number };

type PhoenixWorkerMessageInput =
    | { type: 'INIT' }
    | { type: 'PROCESS_PACKET'; capacity: number; buffer: SharedArrayBuffer | ArrayBuffer };

type PhoenixWorkerResponse =
    | { type: 'INIT_RESULT'; id: number; protocolVersion: number }
    | { type: 'PROCESS_PACKET_RESULT'; id: number; status: number; buffer?: ArrayBuffer }
    | { type: 'CHAT_STREAM_EVENT'; id: number; event: PhoenixChatStreamEvent }
    | { type: 'ERROR'; id: number; error: string };

type PhoenixChatStreamCallbacks = {
    onChunk: (chunk: string) => void;
    onComplete: (response: string) => void;
    onError: (error: Error) => void;
    onReasoningChunk?: (chunk: string) => void;
    onEvent?: (event: { stage: 'reasoning' | 'stream'; status: 'running' | 'done' | 'error'; detail?: string }) => void;
};

export interface PhoenixOmPendingAction {
    kind: 'observe' | 'reflect';
    threadId: string;
    model: string;
    systemPrompt: string;
    userPrompt: string;
    messageIds: string[];
    reflectorToolingEnabled: boolean;
    reflectorMaxToolRounds: number;
}

export interface PhoenixOmReflectorToolSpec {
    name: string;
    description: string;
    parametersJson: unknown;
}

export interface PhoenixOmReflectorToolCall {
    id: string;
    name: string;
    argumentsJson: string;
}

export interface PhoenixOmReflectorToolResult {
    toolCallId: string;
    name: string;
    resultJson: string;
}

export interface PhoenixOmReflectorMessage {
    role: string;
    content: string;
    name?: string | null;
    toolCallId?: string | null;
    toolCalls: PhoenixOmReflectorToolCall[];
}

export interface PhoenixOmReflectorModelRequest {
    sessionId: string;
    threadId: string;
    model: string;
    allowTools: boolean;
    tools: PhoenixOmReflectorToolSpec[];
    messages: PhoenixOmReflectorMessage[];
}

export interface PhoenixOmReflectorModelResponse {
    content: string;
    toolCalls: PhoenixOmReflectorToolCall[];
}

export type PhoenixOmReflectorStep =
    | {
          kind: 'modelRequest';
          request: PhoenixOmReflectorModelRequest;
      }
    | {
          kind: 'toolCalls';
          sessionId: string;
          threadId: string;
          toolCalls: PhoenixOmReflectorToolCall[];
      }
    | {
          kind: 'complete';
          sessionId: string;
          threadId: string;
          response: string;
      };

export interface PhoenixOmTransportConfig {
    apiKey: string;
    defaultModel: string;
    omModel?: string;
    temperature?: number;
    maxTokens?: number;
}

export interface PhoenixChatWorkspaceArtifact {
    key: string;
    runId: string;
    narrativeId: string;
    folderId: string;
    kind: string;
    payload: unknown;
    pinned: boolean;
    producedBy: string;
    createdAt: number;
    updatedAt: number;
}

export interface PhoenixChatPlannerToolSpec {
    name: string;
    description: string;
    parametersJson: unknown;
}

export interface PhoenixChatPlannerToolCall {
    id: string;
    name: string;
    argumentsJson: string;
}

export interface PhoenixChatPlannerMessage {
    role: string;
    content: string;
    name?: string | null;
    toolCallId?: string | null;
    toolCalls: PhoenixChatPlannerToolCall[];
}

export interface PhoenixChatPlannerModelRequest {
    runId: string;
    threadId: string;
    model: string;
    allowTools: boolean;
    tools: PhoenixChatPlannerToolSpec[];
    messages: PhoenixChatPlannerMessage[];
}

export interface PhoenixChatPlannerModelResponse {
    content: string;
    toolCalls: PhoenixChatPlannerToolCall[];
}

export type PhoenixChatPlannerStep =
    | {
          kind: 'modelRequest';
          request: PhoenixChatPlannerModelRequest;
      }
    | {
          kind: 'toolCalls';
          runId: string;
          toolCalls: PhoenixChatPlannerToolCall[];
      }
    | {
          kind: 'complete';
          runId: string;
          response: string;
      };

export interface PhoenixChatPlannerTransportConfig {
    apiKey: string;
    defaultModel: string;
    temperature?: number;
    maxTokens?: number;
}

type PhoenixSemanticLeafChunk = {
    spanId: string;
    documentId: string;
    text: string;
    narrativeId?: string;
    folderId?: string;
};

type PhoenixSemanticDocumentVector = {
    documentId: string;
    values: Float32Array;
    leafCount: number;
    evidenceRefs: string[];
};

type PhoenixSemanticCandidatePrototypeInput = {
    nodeId: string;
    nodeKind: string;
    documentId?: string;
    narrativeId?: string;
    folderId?: string;
    text: string;
    evidenceRefs: string[];
};

type PhoenixSemanticPrototypeVectorRecord = {
    nodeId: string;
    nodeKind: string;
    documentId: string | undefined;
    narrativeId: string | undefined;
    folderId: string | undefined;
    evidenceRefs: string[];
    values: Float32Array;
};

type PhoenixSemanticNliJudgmentInput = NliPairClassificationInput;

const SEMANTIC_EMBEDDING_MODEL_ID = 'mongodb-leaf';
const SEMANTIC_NLI_MODEL_ID = 'onnx-community/ModernBERT-base-nli-ONNX';
const SEMANTIC_VECTOR_DIM = 384;
const SEMANTIC_EMBED_BATCH_SIZE = 16;
const SEMANTIC_NLI_BATCH_SIZE = 4;
const QUERY_BINARY_HEADER_LEN = 22 * 4;
const EMBED_UPSERT_HEADER_LEN = 4 * 4;

@Injectable({ providedIn: 'root' })
export class PhoenixWasmService {
    private readonly embeddingWorker = inject(EmbeddingWorkerService);
    private readonly nliWorker = inject(NliWorkerService);
    private worker: Worker | null = null;
    private workerReady = false;
    private loadPromise: Promise<void> | null = null;
    private readyCallbacks: Array<() => void> = [];
    private nextPacketRequestId = 1;
    private nextWorkerMessageId = 1;
    private readonly pendingWorkerMessages = new Map<
        number,
        {
            resolve: (message: PhoenixWorkerResponse) => void;
            reject: (error: Error) => void;
        }
    >();
    private readonly activeChatStreams = new Map<
        number,
        {
            callbacks: PhoenixChatStreamCallbacks;
            resolve: () => void;
        }
    >();
    private readonly pendingOmByThread = new Map<string, Promise<boolean>>();
    private readonly pendingPlannerByRun = new Map<string, Promise<boolean>>();
    private readonly pendingSemanticDocsBySession = new Map<string, Set<string>>();
    private loggedTransferFallback = false;
    private semanticEmbeddingQueue: Promise<void> = Promise.resolve();

    get isReady(): boolean {
        return this.workerReady;
    }

    constructor(private readonly ngZone: NgZone = createNoopNgZone()) {}

    onReady(callback: () => void): void {
        if (this.isReady) {
            callback();
            return;
        }
        this.readyCallbacks.push(callback);
    }

    async loadWasm(): Promise<void> {
        if (this.loadPromise) {
            return this.loadPromise;
        }
        this.loadPromise = this.loadWasmInternal().catch((error) => {
            this.loadPromise = null;
            throw error;
        });
        return this.loadPromise;
    }

    async initRuntime(forceReset = false): Promise<any> {
        await this.loadWasm();
        return (await this.sendJson(PACKET_KIND.initRuntimeRequest, {
            config: {
                target: 'wasm',
                storage: 'cozoMem',
                snapshotPolicy: 'manual',
                featureFlags: {
                    scanner: true,
                    structure: true,
                    graptor: true,
                    gldr: true,
                    semantic: true,
                    candidateGraph: true,
                },
            },
            storagePath: null,
            forceReset,
        })).json;
    }

    async createSession(label: string, scope: PhoenixScope = {}): Promise<any> {
        await this.loadWasm();
        return (await this.sendJson(PACKET_KIND.createSessionRequest, {
            sessionId: null,
            label,
            scope,
        })).json;
    }

    async ingest(request: Record<string, unknown>): Promise<any> {
        await this.loadWasm();
        const sessionId = typeof request['sessionId'] === 'string' ? String(request['sessionId']) : null;
        const documentIds = extractDocumentIds(request);
        if (sessionId && documentIds.length) {
            this.trackPendingSemanticDocuments(sessionId, documentIds);
        }
        const result = (await this.sendJson(PACKET_KIND.ingestRequest, request, 512 * 1024)).json;
        if (sessionId && request['commit'] === true) {
            this.scheduleSemanticIndexForSession(sessionId, documentIds);
        }
        return result;
    }

    async query(request: Record<string, unknown>): Promise<PhoenixQueryBinaryResult> {
        await this.loadWasm();
        if (!queryTargetsIncludeSemantic(request['targets'])) {
            const payload = (await this.sendJson(PACKET_KIND.queryRequest, request, 512 * 1024)).bytes;
            return decodeQueryResult(payload);
        }

        const semanticVector =
            this.normalizeSemanticVector(request['semanticQueryVector']) ??
            (await this.embedQueryText(String(request['query'] ?? '')));
        const payloadBytes = buildSemanticQueryBinaryPayload(request, semanticVector);
        const payload = (
            await this.sendBytes(
                PACKET_KIND.queryBinaryRequest,
                payloadBytes,
                Math.max(512 * 1024, payloadBytes.byteLength + 4096),
            )
        ).bytes;
        return decodeQueryResult(payload);
    }

    async commit(sessionId: string, request: Record<string, unknown> = {}): Promise<any> {
        await this.loadWasm();
        const result = (await this.sendJson(PACKET_KIND.commitRequest, {
            sessionId,
            reason: request['reason'] ?? null,
        })).json;
        this.scheduleSemanticIndexForSession(sessionId);
        return result;
    }

    async rebuild(request: Record<string, unknown> = {}): Promise<any> {
        await this.loadWasm();
        return (await this.sendJson(PACKET_KIND.rebuildRequest, request)).json;
    }

    async scan(request: Record<string, unknown>): Promise<any> {
        await this.loadWasm();
        return (await this.sendJson(PACKET_KIND.scanRequest, request, 512 * 1024)).json;
    }

    async buildStructure(request: Record<string, unknown>): Promise<any> {
        await this.loadWasm();
        return (await this.sendJson(PACKET_KIND.structureRequest, request, 512 * 1024)).json;
    }

    async analyzeText(text: string): Promise<any> {
        await this.loadWasm();
        return (await this.sendJson(PACKET_KIND.analyzeTextRequest, { text }, 512 * 1024)).json;
    }

    async graphDelta(request: Record<string, unknown>): Promise<PhoenixGraphDeltaBinaryResult> {
        await this.loadWasm();
        const payload = (await this.sendJson(PACKET_KIND.graphDeltaRequest, request, 512 * 1024)).bytes;
        return decodeGraphDeltaResult(payload);
    }

    async sessionState(sessionId: string): Promise<PhoenixSessionStateBinaryResult> {
        await this.loadWasm();
        const payload = (await this.sendJson(
            PACKET_KIND.sessionStateRequest,
            { sessionId },
            512 * 1024,
        )).bytes;
        return decodeSessionStateResult(payload);
    }

    async sessionStats(sessionId: string): Promise<PhoenixSessionStatsBinaryResult> {
        await this.loadWasm();
        const payload = (await this.sendJson(
            PACKET_KIND.sessionStatsRequest,
            { sessionId },
            128 * 1024,
        )).bytes;
        return decodeSessionStatsResult(payload);
    }

    async exportSnapshot(
        partition: PhoenixSnapshotPartition = 'all',
        capacityHint = 8 * 1024 * 1024,
    ): Promise<Uint8Array> {
        await this.loadWasm();
        if (partition === 'all') {
            return (await this.sendBytes(PACKET_KIND.snapshotExportRequest, new Uint8Array(0), capacityHint)).bytes;
        }
        return (await this.sendJson(
            PACKET_KIND.snapshotExportRequest,
            { partition },
            capacityHint,
        )).bytes;
    }

    async importSnapshot(bytes: Uint8Array): Promise<any> {
        await this.loadWasm();
        return (await this.sendBytes(
            PACKET_KIND.snapshotImportRequest,
            bytes,
            Math.max(bytes.byteLength + 4096, 256 * 1024),
        )).json;
    }

    async storeCommand(command: string, payload: Record<string, unknown> = {}): Promise<any> {
        await this.loadWasm();
        const result = (await this.sendJson(
            PACKET_KIND.storeCommandRequest,
            { command, payload },
            512 * 1024,
        )).json;
        if (!result?.success) {
            throw new Error(result?.error || `Phoenix store command failed: ${command}`);
        }
        return result.payload ?? null;
    }

    async chatInit(config: Record<string, unknown>): Promise<any> {
        return this.storeCommand('chat:init', { config });
    }

    async chatCreateThread(worldId: string, narrativeId: string, title?: string): Promise<any> {
        return this.storeCommand('chat:createThread', { worldId, narrativeId, ...(title ? { title } : {}) });
    }

    async chatGetThread(id: string): Promise<any> {
        return this.storeCommand('chat:getThread', { id });
    }

    async chatListThreads(worldId?: string): Promise<any> {
        return this.storeCommand('chat:listThreads', worldId ? { worldId } : {});
    }

    async chatDeleteThread(id: string): Promise<void> {
        await this.storeCommand('chat:deleteThread', { id });
    }

    async chatAddMessage(threadId: string, role: string, content: string, narrativeId?: string): Promise<any> {
        return this.storeCommand('chat:addMessage', {
            threadId,
            role,
            content,
            ...(narrativeId ? { narrativeId } : {}),
        });
    }

    async chatListMessages(threadId: string): Promise<any> {
        return this.storeCommand('chat:listMessages', { threadId });
    }

    async chatUpdateMessage(messageId: string, content: string): Promise<any> {
        return this.storeCommand('chat:updateMessage', { messageId, content });
    }

    async chatAppendMessage(messageId: string, chunk: string): Promise<any> {
        return this.storeCommand('chat:appendMessage', { messageId, chunk });
    }

    async chatStartStreamingMessage(threadId: string, narrativeId?: string): Promise<any> {
        return this.storeCommand('chat:startStreamingMessage', {
            threadId,
            ...(narrativeId ? { narrativeId } : {}),
        });
    }

    async chatClearThread(threadId: string): Promise<void> {
        await this.storeCommand('chat:clearThread', { threadId });
    }

    async chatExportThread(threadId: string): Promise<string> {
        return (await this.storeCommand('chat:exportThread', { threadId })) || '{}';
    }

    async chatStartRun(threadId: string, prompt: string, options: Record<string, unknown>): Promise<any> {
        return this.storeCommand('chat:startRun', { threadId, prompt, options });
    }

    async chatPollRun(runId: string): Promise<any> {
        return this.storeCommand('chat:pollRun', { runId });
    }

    async chatResumeRun(runId: string): Promise<any> {
        return this.storeCommand('chat:resumeRun', { runId });
    }

    async chatCancelRun(runId: string): Promise<any> {
        return this.storeCommand('chat:cancelRun', { runId });
    }

    async chatListRunEvents(threadId: string, limit = 100): Promise<any> {
        return this.storeCommand('chat:listRunEvents', { threadId, limit });
    }

    async chatMarkRunStreaming(runId: string, assistantMessageId: string): Promise<any> {
        return this.storeCommand('chat:markRunStreaming', { runId, assistantMessageId });
    }

    async chatCompleteRun(
        runId: string,
        assistantMessageId: string,
        finalResponse: string,
        finalError?: string,
    ): Promise<any> {
        return this.storeCommand('chat:completeRun', {
            runId,
            assistantMessageId,
            finalResponse,
            ...(finalError ? { finalError } : {}),
        });
    }

    async chatGetPlannerStep(runId: string): Promise<PhoenixChatPlannerStep | null> {
        return this.storeCommand('chat:getPlannerStep', { runId });
    }

    async chatSubmitPlannerModelResponse(
        runId: string,
        response: PhoenixChatPlannerModelResponse,
    ): Promise<PhoenixChatPlannerStep | null> {
        return this.storeCommand('chat:submitPlannerModelResponse', { runId, response });
    }

    async chatAdvancePlannerRun(runId: string): Promise<PhoenixChatPlannerStep | null> {
        return this.storeCommand('chat:advancePlannerRun', { runId });
    }

    async chatDegradePlannerRun(runId: string, reason: string): Promise<any> {
        return this.storeCommand('chat:degradePlannerRun', { runId, reason });
    }

    async chatListPlannerArtifacts(runId: string): Promise<PhoenixChatWorkspaceArtifact[]> {
        const payload = await this.storeCommand('chat:listPlannerArtifacts', { runId });
        return Array.isArray(payload) ? payload : [];
    }

    async chatPinPlannerArtifact(
        runId: string,
        key: string,
        pinned = true,
    ): Promise<PhoenixChatWorkspaceArtifact | null> {
        return this.storeCommand('chat:pinPlannerArtifact', { runId, key, pinned });
    }

    async chatPrepareOm(threadId: string): Promise<PhoenixOmPendingAction | null> {
        return this.storeCommand('chat:prepareOm', { threadId });
    }

    async chatApplyOmAction(action: PhoenixOmPendingAction, response: string): Promise<boolean> {
        return !!(await this.storeCommand('chat:applyOmAction', { action, response }));
    }

    async omStartReflector(action: PhoenixOmPendingAction): Promise<PhoenixOmReflectorStep> {
        return this.storeCommand('om:startReflector', { action });
    }

    async omSubmitReflectorModelResponse(
        sessionId: string,
        response: PhoenixOmReflectorModelResponse,
    ): Promise<PhoenixOmReflectorStep> {
        return this.storeCommand('om:submitReflectorModelResponse', { sessionId, response });
    }

    async omSubmitReflectorToolResults(
        sessionId: string,
        results: PhoenixOmReflectorToolResult[],
    ): Promise<PhoenixOmReflectorStep> {
        return this.storeCommand('om:submitReflectorToolResults', { sessionId, results });
    }

    async omDropReflectorSession(sessionId: string): Promise<boolean> {
        return !!(await this.storeCommand('om:dropReflectorSession', { sessionId }));
    }

    async omRecoverLostMemory(threadId: string, limit = 10, focus?: string): Promise<any[]> {
        return this.storeCommand('om:recoverLostMemory', {
            threadId,
            limit,
            ...(focus ? { focus } : {}),
        });
    }

    async omMemoryGraphSearch(threadId: string, query: string, limit = 10): Promise<any[]> {
        return this.storeCommand('om:memoryGraphSearch', {
            threadId,
            query,
            limit,
        });
    }

    async chatProcessOm(threadId: string, config: PhoenixOmTransportConfig): Promise<boolean> {
        const existing = this.pendingOmByThread.get(threadId);
        if (existing) {
            return existing;
        }

        const task = this.chatProcessOmInternal(threadId, config).finally(() => {
            this.pendingOmByThread.delete(threadId);
        });
        this.pendingOmByThread.set(threadId, task);
        return task;
    }

    async chatProcessPlannerRun(
        runId: string,
        config: PhoenixChatPlannerTransportConfig,
    ): Promise<boolean> {
        const existing = this.pendingPlannerByRun.get(runId);
        if (existing) {
            return existing;
        }

        const task = this.chatProcessPlannerRunInternal(runId, config).finally(() => {
            this.pendingPlannerByRun.delete(runId);
        });
        this.pendingPlannerByRun.set(runId, task);
        return task;
    }

    async chatSubmitToolResults(runId: string, results: unknown[]): Promise<any> {
        return this.storeCommand('chat:submitToolResults', { runId, results });
    }

    async chatSubmitApproval(
        runId: string,
        approvalId: string,
        approved: boolean,
        decisionJson?: string,
    ): Promise<any> {
        return this.storeCommand('chat:submitApproval', {
            runId,
            approvalId,
            approved,
            ...(decisionJson ? { decisionJson } : {}),
        });
    }

    private trackPendingSemanticDocuments(sessionId: string, documentIds: string[]): void {
        if (!documentIds.length) {
            return;
        }
        let pending = this.pendingSemanticDocsBySession.get(sessionId);
        if (!pending) {
            pending = new Set<string>();
            this.pendingSemanticDocsBySession.set(sessionId, pending);
        }
        for (const documentId of documentIds) {
            pending.add(documentId);
        }
    }

    private drainPendingSemanticDocuments(sessionId: string, seedDocumentIds: string[] = []): string[] {
        const pending = this.pendingSemanticDocsBySession.get(sessionId);
        const drained = new Set<string>(seedDocumentIds);
        if (pending) {
            for (const documentId of pending) {
                drained.add(documentId);
            }
            this.pendingSemanticDocsBySession.delete(sessionId);
        }
        return Array.from(drained);
    }

    private scheduleSemanticIndexForSession(sessionId: string, seedDocumentIds: string[] = []): void {
        const documentIds = this.drainPendingSemanticDocuments(sessionId, seedDocumentIds);
        if (!documentIds.length) {
            return;
        }
        void this.enqueueEmbeddingTask(async () => {
            await this.indexCommittedSemanticDocuments(documentIds);
        }).catch((error) => {
            console.warn('[PhoenixWasmService] Semantic indexing failed:', error);
        });
    }

    private enqueueEmbeddingTask<T>(task: () => Promise<T>): Promise<T> {
        const run = this.semanticEmbeddingQueue.then(task, task);
        this.semanticEmbeddingQueue = run.then(
            () => undefined,
            () => undefined,
        );
        return run;
    }

    private async ensureSemanticEmbeddingWorker(): Promise<void> {
        await this.embeddingWorker.initialize(SEMANTIC_EMBEDDING_MODEL_ID);
    }

    private async ensureSemanticNliWorker(): Promise<void> {
        await this.nliWorker.initialize(SEMANTIC_NLI_MODEL_ID);
    }

    private async embedQueryText(queryText: string): Promise<Float32Array> {
        return this.enqueueEmbeddingTask(async () => {
            await this.ensureSemanticEmbeddingWorker();
            const embeddings = await this.embeddingWorker.embed([queryText], 1);
            return this.toSemanticVector(embeddings[0] || []);
        });
    }

    private async indexCommittedSemanticDocuments(documentIds: string[]): Promise<void> {
        if (!documentIds.length) {
            return;
        }
        const rows = await this.storeCommand('semantic:listLeafChunks', { documentIds });
        const chunks = normalizeSemanticLeafChunks(rows);
        if (!chunks.length) {
            return;
        }

        await this.ensureSemanticEmbeddingWorker();
        let writeChain = Promise.resolve();
        const documentAccumulator = new Map<
            string,
            { sum: Float32Array; leafCount: number; evidenceRefs: Set<string> }
        >();
        await this.embeddingWorker.embedStream(
            chunks.map((chunk) => chunk.text),
            (batch) => {
                const batchStart = batch.batchIndex * SEMANTIC_EMBED_BATCH_SIZE;
                const records = batch.embeddings
                    .map((embedding, index) => {
                        const chunk = chunks[batchStart + index];
                        if (!chunk) {
                            return null;
                        }
                        const values = this.toSemanticVector(embedding);
                        accumulateSemanticDocumentVector(
                            documentAccumulator,
                            chunk.documentId,
                            chunk.spanId,
                            values,
                        );
                        return {
                            spanId: chunk.spanId,
                            values,
                        };
                    })
                    .filter((record): record is { spanId: string; values: Float32Array } => record !== null);
                if (!records.length) {
                    return;
                }
                writeChain = writeChain.then(() => this.upsertSemanticBatch(records));
            },
            SEMANTIC_EMBED_BATCH_SIZE,
        );
        await writeChain;
        const documentVectors = finalizeSemanticDocumentVectors(documentAccumulator);
        if (documentVectors.length) {
            await this.storeCommand('semantic:upsertDocumentVectors', {
                rows: documentVectors.map((record) => ({
                    documentId: record.documentId,
                    leafCount: record.leafCount,
                    values: Array.from(record.values),
                    evidenceRefs: record.evidenceRefs,
                })),
            });
        }

        const prototypeInputs = normalizeSemanticCandidatePrototypeInputs(
            await this.storeCommand('semantic:listCandidatePrototypeInputs', { documentIds }),
        );
        if (prototypeInputs.length) {
            let prototypeWriteChain = Promise.resolve();
            await this.embeddingWorker.embedStream(
                prototypeInputs.map((row) => row.text),
                (batch) => {
                    const batchStart = batch.batchIndex * SEMANTIC_EMBED_BATCH_SIZE;
                    const records = batch.embeddings
                        .map((embedding, index) => {
                            const row = prototypeInputs[batchStart + index];
                            if (!row) {
                                return null;
                            }
                            return {
                                nodeId: row.nodeId,
                                nodeKind: row.nodeKind,
                                documentId: row.documentId,
                                narrativeId: row.narrativeId,
                                folderId: row.folderId,
                                evidenceRefs: row.evidenceRefs,
                                values: this.toSemanticVector(embedding),
                            };
                        })
                        .filter(
                            (
                                record,
                            ): record is PhoenixSemanticPrototypeVectorRecord => record !== null,
                        );
                    if (!records.length) {
                        return;
                    }
                    prototypeWriteChain = prototypeWriteChain.then(() =>
                        this.storeCommand('semantic:upsertPrototypeVectors', {
                            rows: records.map((record) => ({
                                nodeId: record.nodeId,
                                nodeKind: record.nodeKind,
                                documentId: record.documentId ?? null,
                                narrativeId: record.narrativeId ?? null,
                                folderId: record.folderId ?? null,
                                evidenceRefs: record.evidenceRefs,
                                values: Array.from(record.values),
                            })),
                        }),
                    );
                },
                SEMANTIC_EMBED_BATCH_SIZE,
            );
            await prototypeWriteChain;
        }

        await this.storeCommand('semantic:refreshCandidateGraphEdges', {
            documentIds,
            nodeIds: prototypeInputs.map((row) => row.nodeId),
        });

        const nliInputs = normalizeSemanticNliJudgmentInputs(
            await this.storeCommand('semantic:listNliJudgmentInputs', {
                documentIds,
                nodeIds: prototypeInputs.map((row) => row.nodeId),
            }),
        );
        if (!nliInputs.length) {
            return;
        }

        await this.embeddingWorker.dispose().catch(() => undefined);
        await this.ensureSemanticNliWorker();
        const nliStatus = await this.nliWorker.getStatus();
        const nliResults: NliClassificationResult[] = [];
        try {
            await this.nliWorker.classifyStream(
                nliInputs,
                (batch) => {
                    nliResults.push(...batch.results);
                },
                SEMANTIC_NLI_BATCH_SIZE,
            );
            const summary = await this.storeCommand('semantic:applyNliJudgments', {
                modelId: SEMANTIC_NLI_MODEL_ID,
                device: nliStatus.device,
                results: nliResults,
            });
            console.info('[PhoenixWasmService] Applied local NLI candidate judgments.', summary);
        } finally {
            await this.nliWorker.dispose().catch(() => undefined);
        }
    }

    private async upsertSemanticBatch(
        records: Array<{ spanId: string; values: Float32Array }>,
    ): Promise<void> {
        if (!records.length) {
            return;
        }
        const payload = buildEmbedUpsertBinaryPayload(records);
        await this.sendBytes(
            PACKET_KIND.embedUpsertBinaryRequest,
            payload,
            Math.max(256 * 1024, payload.byteLength + 4096),
        );
    }

    private normalizeSemanticVector(source: unknown): Float32Array | null {
        if (source instanceof Float32Array) {
            return this.toSemanticVector(source);
        }
        if (Array.isArray(source)) {
            return this.toSemanticVector(source);
        }
        return null;
    }

    private toSemanticVector(source: ArrayLike<number>): Float32Array {
        const vector = new Float32Array(source.length);
        for (let index = 0; index < source.length; index += 1) {
            vector[index] = Number(source[index] ?? 0);
        }
        if (vector.length !== SEMANTIC_VECTOR_DIM) {
            throw new Error(
                `Semantic vector dimension mismatch: expected ${SEMANTIC_VECTOR_DIM}, got ${vector.length}`,
            );
        }
        return vector;
    }

    async streamChat(request: PhoenixChatStreamRequest, callbacks: PhoenixChatStreamCallbacks): Promise<void> {
        await this.loadWasm();
        if (!this.worker) {
            callbacks.onError(new Error('[PhoenixWasmService] Worker not initialized'));
            return;
        }

        const id = this.nextWorkerMessageId++;
        return new Promise((resolve) => {
            this.activeChatStreams.set(id, { callbacks, resolve });
            this.worker!.postMessage({
                type: 'CHAT_STREAM_START',
                id,
                request,
            } as PhoenixWorkerMessage);
        });
    }

    private async chatProcessOmInternal(threadId: string, config: PhoenixOmTransportConfig): Promise<boolean> {
        if (!config.apiKey?.trim()) {
            return false;
        }

        let mutated = false;
        for (let iteration = 0; iteration < 4; iteration += 1) {
            const action = await this.chatPrepareOm(threadId);
            if (!action) {
                break;
            }

            const response = await this.runOmAction(action, config);
            mutated = (await this.chatApplyOmAction(action, response)) || mutated;
        }

        return mutated;
    }

    private async chatProcessPlannerRunInternal(
        runId: string,
        config: PhoenixChatPlannerTransportConfig,
    ): Promise<boolean> {
        if (!config.apiKey?.trim()) {
            await this.chatDegradePlannerRun(runId, 'Planner requires an OpenRouter API key.').catch(
                () => undefined,
            );
            return false;
        }

        try {
            let step = await this.chatGetPlannerStep(runId);
            while (step) {
                if (step.kind === 'complete') {
                    return true;
                }
                if (step.kind === 'toolCalls') {
                    step = await this.chatAdvancePlannerRun(step.runId);
                    continue;
                }

                const payload = await this.fetchOmChatCompletion(
                    step.request.model || config.defaultModel,
                    config,
                    this.buildToolCallingRequestBody(step.request),
                );
                step = await this.chatSubmitPlannerModelResponse(
                    step.request.runId,
                    this.buildPlannerModelResponse(payload),
                );
            }
            return false;
        } catch (error) {
            const reason = error instanceof Error ? error.message : String(error);
            await this.chatDegradePlannerRun(runId, reason).catch(() => undefined);
            return false;
        }
    }

    private async runOmAction(action: PhoenixOmPendingAction, config: PhoenixOmTransportConfig): Promise<string> {
        if (action.kind === 'reflect' && action.reflectorToolingEnabled) {
            return this.runReflectorWithRuntime(action, config);
        }
        return this.runSimpleOmAction(action, config);
    }

    private async runSimpleOmAction(action: PhoenixOmPendingAction, config: PhoenixOmTransportConfig): Promise<string> {
        const model = action.model || config.omModel || config.defaultModel;
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${config.apiKey.trim()}`,
                'Content-Type': 'application/json',
                Accept: 'application/json',
                'HTTP-Referer': globalThis.location?.origin || 'http://localhost',
                'X-Title': 'KittClouds Phoenix',
            },
            body: JSON.stringify({
                model,
                messages: [
                    { role: 'system', content: action.systemPrompt },
                    { role: 'user', content: action.userPrompt },
                ],
                stream: false,
                temperature: typeof config.temperature === 'number' ? config.temperature : 0.3,
                max_tokens: typeof config.maxTokens === 'number' && config.maxTokens > 0
                    ? config.maxTokens
                    : 2_048,
            }),
        });

        if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || `OM request failed with status ${response.status}`);
        }

        const payload = await response.json();
        const content =
            extractOmResponseText(payload?.choices?.[0]?.message?.content) ||
            extractOmResponseText(payload?.choices?.[0]?.content);
        if (!content) {
            throw new Error('OM response did not include content.');
        }
        return content;
    }

    private async runReflectorWithRuntime(
        action: PhoenixOmPendingAction,
        config: PhoenixOmTransportConfig,
    ): Promise<string> {
        let step = await this.omStartReflector(action);
        let sessionId: string | null = null;
        try {
            while (true) {
                if (step.kind === 'complete') {
                    return step.response;
                }
                if (step.kind === 'toolCalls') {
                    sessionId = step.sessionId;
                    const results: PhoenixOmReflectorToolResult[] = [];
                    for (const toolCall of step.toolCalls) {
                        results.push(await this.executeOmToolCall(step.threadId, toolCall));
                    }
                    step = await this.omSubmitReflectorToolResults(step.sessionId, results);
                    continue;
                }

                sessionId = step.request.sessionId;
                const payload = await this.fetchOmChatCompletion(
                    step.request.model || action.model || config.omModel || config.defaultModel,
                    config,
                    this.buildToolCallingRequestBody(step.request),
                );
                step = await this.omSubmitReflectorModelResponse(
                    step.request.sessionId,
                    this.buildReflectorModelResponse(payload),
                );
            }
        } catch (error) {
            if (sessionId) {
                await this.omDropReflectorSession(sessionId).catch(() => undefined);
            }
            throw error;
        }
    }

    private async fetchOmChatCompletion(
        model: string,
        config: PhoenixOmTransportConfig,
        body: Record<string, unknown>,
    ): Promise<any> {
        const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${config.apiKey.trim()}`,
                'Content-Type': 'application/json',
                Accept: 'application/json',
                'HTTP-Referer': globalThis.location?.origin || 'http://localhost',
                'X-Title': 'KittClouds Phoenix',
            },
            body: JSON.stringify({
                model,
                stream: false,
                temperature: typeof config.temperature === 'number' ? config.temperature : 0.3,
                max_tokens:
                    typeof config.maxTokens === 'number' && config.maxTokens > 0
                        ? config.maxTokens
                        : 2_048,
                ...body,
            }),
        });

        if (!response.ok) {
            const detail = await response.text();
            throw new Error(detail || `OM request failed with status ${response.status}`);
        }

        return response.json();
    }

    private buildToolCallingRequestBody(request: {
        allowTools: boolean;
        tools: Array<{
            name: string;
            description: string;
            parametersJson: unknown;
        }>;
        messages: Array<{
            role: string;
            content: string;
            name?: string | null;
            toolCallId?: string | null;
            toolCalls: Array<{
                id: string;
                name: string;
                argumentsJson: string;
            }>;
        }>;
    }): Record<string, unknown> {
        return {
            messages: request.messages.map((message) => ({
                role: message.role,
                content: message.content,
                ...(message.name ? { name: message.name } : {}),
                ...(message.toolCallId ? { tool_call_id: message.toolCallId } : {}),
                ...(message.toolCalls.length
                    ? {
                          tool_calls: message.toolCalls.map((toolCall) => ({
                              id: toolCall.id,
                              type: 'function',
                              function: {
                                  name: toolCall.name,
                                  arguments: toolCall.argumentsJson,
                              },
                          })),
                      }
                    : {}),
            })),
            ...(request.allowTools
                ? {
                      tools: request.tools.map((tool) => ({
                          type: 'function',
                          function: {
                              name: tool.name,
                              description: tool.description,
                              parameters: tool.parametersJson,
                          },
                      })),
                      tool_choice: 'auto',
                  }
                : {}),
        };
    }

    private buildReflectorModelResponse(payload: any): PhoenixOmReflectorModelResponse {
        const response = this.buildToolCallingModelResponse(payload);
        return {
            content: response.content,
            toolCalls: response.toolCalls,
        };
    }

    private buildPlannerModelResponse(payload: any): PhoenixChatPlannerModelResponse {
        const response = this.buildToolCallingModelResponse(payload);
        return {
            content: response.content,
            toolCalls: response.toolCalls,
        };
    }

    private buildToolCallingModelResponse(payload: any): {
        content: string;
        toolCalls: Array<{ id: string; name: string; argumentsJson: string }>;
    } {
        const choice = payload?.choices?.[0];
        const toolCalls = Array.isArray(choice?.message?.tool_calls) ? choice.message.tool_calls : [];
        return {
            content:
                extractOmResponseText(choice?.message?.content) ||
                extractOmResponseText(choice?.content) ||
                '',
            toolCalls: toolCalls.map((toolCall: any) => ({
                id: String(toolCall?.id || ''),
                name: String(toolCall?.function?.name || ''),
                argumentsJson: String(toolCall?.function?.arguments || '{}'),
            })),
        };
    }

    private async executeOmToolCall(
        threadId: string,
        toolCall: PhoenixOmReflectorToolCall,
    ): Promise<PhoenixOmReflectorToolResult> {
        const toolName = String(toolCall?.name || '');
        const rawArgs = String(toolCall?.argumentsJson || '{}');
        let args: Record<string, unknown> = {};
        try {
            args = JSON.parse(rawArgs) as Record<string, unknown>;
        } catch {
            args = {};
        }

        let result: unknown;
        switch (toolName) {
            case 'recover_lost_memory':
                result = await this.omRecoverLostMemory(
                    threadId,
                    typeof args['limit'] === 'number' ? args['limit'] : 10,
                    typeof args['focus'] === 'string' ? args['focus'] : undefined,
                );
                break;
            case 'memory_graph_search':
                result = await this.omMemoryGraphSearch(
                    threadId,
                    typeof args['query'] === 'string' ? args['query'] : '',
                    typeof args['limit'] === 'number' ? args['limit'] : 10,
                );
                break;
            default:
                result = { error: `Unsupported OM tool: ${toolName}` };
                break;
        }
        return {
            toolCallId: toolCall.id,
            name: toolName,
            resultJson: JSON.stringify(result),
        };
    }

    cancelStreamChat(streamId: number): void {
        if (!this.worker || !this.activeChatStreams.has(streamId)) {
            return;
        }
        this.worker.postMessage({ type: 'CHAT_STREAM_CANCEL', id: streamId } as PhoenixWorkerMessage);
    }

    private async loadWasmInternal(): Promise<void> {
        this.worker?.terminate();
        this.worker = createWorkerOutsideAngular(
            this.ngZone,
            () =>
                new Worker(new URL('../workers/phoenix.worker', import.meta.url), {
                    type: 'module',
                    name: 'phoenix-wasm',
                }),
            (worker) => {
                worker.onmessage = (event: MessageEvent<PhoenixWorkerResponse>) => {
                    this.handleWorkerMessage(event.data);
                };

                worker.onerror = (event) => {
                    const error = new Error(
                        `[PhoenixWasmService] Worker error: ${event.message || 'Unknown worker error'}`,
                    );
                    this.ngZone.run(() => {
                        console.error('[PhoenixWasmService] Worker error:', event);
                        this.worker?.terminate();
                        this.worker = null;
                        this.workerReady = false;
                        this.loadPromise = null;
                        this.rejectPendingWorkerMessages(error);
                        this.rejectActiveChatStreams(error);
                    });
                };
            },
        );
        this.workerReady = false;

        const response = await this.sendWorkerMessage({ type: 'INIT' });
        if (response.type !== 'INIT_RESULT') {
            throw new Error('[PhoenixWasmService] Worker failed to initialize');
        }
        if (response.protocolVersion !== PROTOCOL_VERSION) {
            throw new Error(
                `[PhoenixWasmService] Protocol mismatch: expected ${PROTOCOL_VERSION}, got ${response.protocolVersion}`,
            );
        }
        this.workerReady = true;
        this.notifyReady();
    }

    private notifyReady(): void {
        this.ngZone.run(() => {
            for (const callback of this.readyCallbacks) {
                try {
                    callback();
                } catch (error) {
                    console.error('[PhoenixWasmService] Ready callback failed:', error);
                }
            }
            this.readyCallbacks = [];
        });
    }

    private async sendJson(kind: number, payload: unknown, capacityHint?: number): Promise<PacketResponse> {
        const encoder = new TextEncoder();
        return this.sendPayload(kind, encoder.encode(JSON.stringify(payload)), capacityHint);
    }

    private async sendBytes(kind: number, payload: Uint8Array, capacityHint?: number): Promise<PacketResponse> {
        return this.sendPayload(kind, payload, capacityHint);
    }

    private async sendPayload(kind: number, payload: Uint8Array, capacityHint?: number): Promise<PacketResponse> {
        let capacity = Math.max(
            capacityHint ?? 0,
            DEFAULT_PACKET_REGION_SIZE,
            payload.byteLength * 4 + 4096,
        );

        for (let attempt = 0; attempt < 6; attempt++) {
            const buffer = this.createPacketBuffer(capacity);
            const bytes = new Uint8Array(buffer, 0, capacity);
            const requestId = this.nextPacketRequestId++;
            writePacketHeader(bytes, kind, requestId, payload.byteLength);
            bytes.set(payload, PACKET_HEADER_SIZE);

            const response = await this.sendWorkerMessage({
                type: 'PROCESS_PACKET',
                capacity,
                buffer,
            });
            if (response.type !== 'PROCESS_PACKET_RESULT') {
                throw new Error('[PhoenixWasmService] Unexpected worker response');
            }

            const responseBuffer = response.buffer ?? buffer;
            const responseRegion = new Uint8Array(responseBuffer, 0, capacity);
            const header = decodePacketHeader(responseRegion);
            const responseBytes = responseRegion.slice(
                PACKET_HEADER_SIZE,
                PACKET_HEADER_SIZE + header.payloadLen,
            );
            const json = tryDecodeJson(responseBytes);

            if (response.status === 0) {
                return {
                    kind: header.kind,
                    requestId: header.requestId,
                    bytes: responseBytes,
                    json,
                };
            }

            const message = json?.message || json?.error || 'Phoenix packet failed';
            if (attempt < 5 && typeof message === 'string' && isRetriablePacketMessage(message)) {
                capacity *= 2;
                continue;
            }
            throw new Error(message);
        }

        throw new Error('[PhoenixWasmService] Packet retry budget exhausted');
    }

    private createPacketBuffer(capacity: number): SharedArrayBuffer | ArrayBuffer {
        if (
            typeof SharedArrayBuffer !== 'undefined' &&
            (typeof crossOriginIsolated === 'undefined' || crossOriginIsolated)
        ) {
            return new SharedArrayBuffer(capacity);
        }
        if (!this.loggedTransferFallback) {
            this.loggedTransferFallback = true;
            console.warn('[PhoenixWasmService] SharedArrayBuffer unavailable, falling back to transferable buffers.');
        }
        return new ArrayBuffer(capacity);
    }

    private async sendWorkerMessage(message: PhoenixWorkerMessageInput): Promise<PhoenixWorkerResponse> {
        if (!this.worker) {
            throw new Error('[PhoenixWasmService] Worker not initialized');
        }

        const id = this.nextWorkerMessageId++;
        return new Promise((resolve, reject) => {
            this.pendingWorkerMessages.set(id, { resolve, reject });
            const fullMessage = { ...message, id } as PhoenixWorkerMessage;
            if (message.type === 'PROCESS_PACKET' && !this.isSharedBuffer(message.buffer)) {
                this.worker!.postMessage(fullMessage, [message.buffer]);
                return;
            }
            this.worker!.postMessage(fullMessage);
        });
    }

    private handleWorkerMessage(message: PhoenixWorkerResponse): void {
        if (message.type === 'CHAT_STREAM_EVENT') {
            this.handleChatStreamEvent(message.id, message.event);
            return;
        }

        const pending = this.pendingWorkerMessages.get(message.id);
        if (!pending) {
            if (message.type === 'ERROR') {
                this.failChatStream(message.id, new Error(message.error));
            }
            return;
        }
        this.pendingWorkerMessages.delete(message.id);
        this.ngZone.run(() => {
            if (message.type === 'ERROR') {
                pending.reject(new Error(message.error));
                return;
            }
            pending.resolve(message);
        });
    }

    private rejectPendingWorkerMessages(error: Error): void {
        for (const pending of this.pendingWorkerMessages.values()) {
            this.ngZone.run(() => {
                pending.reject(error);
            });
        }
        this.pendingWorkerMessages.clear();
    }

    private handleChatStreamEvent(streamId: number, event: PhoenixChatStreamEvent): void {
        const active = this.activeChatStreams.get(streamId);
        if (!active) {
            return;
        }

        switch (event.type) {
            case 'delta':
                active.callbacks.onEvent?.({ stage: 'stream', status: 'running' });
                active.callbacks.onChunk(event.chunk);
                return;
            case 'reasoning':
                active.callbacks.onEvent?.({ stage: 'reasoning', status: 'running' });
                active.callbacks.onReasoningChunk?.(event.chunk);
                return;
            case 'done':
                this.activeChatStreams.delete(streamId);
                this.ngZone.run(() => {
                    active.callbacks.onEvent?.({ stage: 'stream', status: 'done' });
                    active.callbacks.onComplete(event.response);
                    active.resolve();
                });
                return;
            case 'cancelled':
                this.activeChatStreams.delete(streamId);
                this.ngZone.run(() => {
                    active.callbacks.onEvent?.({ stage: 'stream', status: 'error', detail: 'Cancelled' });
                    active.callbacks.onError(new Error('Chat stream cancelled'));
                    active.resolve();
                });
                return;
            case 'error':
                this.activeChatStreams.delete(streamId);
                this.ngZone.run(() => {
                    active.callbacks.onEvent?.({ stage: 'stream', status: 'error', detail: event.error });
                    active.callbacks.onError(new Error(event.error));
                    active.resolve();
                });
                return;
        }
    }

    private failChatStream(streamId: number, error: Error): void {
        const active = this.activeChatStreams.get(streamId);
        if (!active) {
            return;
        }
        this.activeChatStreams.delete(streamId);
        this.ngZone.run(() => {
            active.callbacks.onEvent?.({ stage: 'stream', status: 'error', detail: error.message });
            active.callbacks.onError(error);
            active.resolve();
        });
    }

    private rejectActiveChatStreams(error: Error): void {
        for (const [streamId] of this.activeChatStreams) {
            this.failChatStream(streamId, error);
        }
    }

    private isSharedBuffer(buffer: SharedArrayBuffer | ArrayBuffer): buffer is SharedArrayBuffer {
        return typeof SharedArrayBuffer !== 'undefined' && buffer instanceof SharedArrayBuffer;
    }
}

type BinaryHeader = {
    sessionOffset: number;
    sessionLen: number;
    table1Offset: number;
    table1Count: number;
    table2Offset: number;
    table2Count: number;
    table3Offset: number;
    table3Count: number;
    table4Offset: number;
    table4Count: number;
    arenaOffset: number;
};

function decodeBinaryHeader(view: DataView): BinaryHeader {
    return {
        sessionOffset: view.getUint32(8, true),
        sessionLen: view.getUint32(12, true),
        table1Offset: view.getUint32(16, true),
        table1Count: view.getUint32(20, true),
        table2Offset: view.getUint32(24, true),
        table2Count: view.getUint32(28, true),
        table3Offset: view.getUint32(32, true),
        table3Count: view.getUint32(36, true),
        table4Offset: view.getUint32(40, true),
        table4Count: view.getUint32(44, true),
        arenaOffset: view.getUint32(48, true),
    };
}

function readArenaString(bytes: Uint8Array, arenaOffset: number, stringOffset: number, stringLen: number): string {
    return new TextDecoder().decode(
        bytes.slice(arenaOffset + stringOffset, arenaOffset + stringOffset + stringLen),
    );
}

function decodeDiagnostics(
    bytes: Uint8Array,
    view: DataView,
    tableOffset: number,
    tableCount: number,
    arenaOffset: number,
): PhoenixDiagnostic[] {
    return Array.from({ length: tableCount }, (_, index) => {
        const base = tableOffset + index * 16;
        return {
            code: readArenaString(bytes, arenaOffset, view.getUint32(base, true), view.getUint32(base + 4, true)),
            message: readArenaString(bytes, arenaOffset, view.getUint32(base + 8, true), view.getUint32(base + 12, true)),
        };
    });
}

function decodeQueryResult(bytes: Uint8Array): PhoenixQueryBinaryResult {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const header = decodeBinaryHeader(view);
    const sessionId = readArenaString(bytes, header.arenaOffset, header.sessionOffset, header.sessionLen);
    const chunkHits = Array.from({ length: header.table1Count }, (_, index) => {
        const base = header.table1Offset + index * 16;
        return {
            chunkId: readArenaString(bytes, header.arenaOffset, view.getUint32(base, true), view.getUint32(base + 4, true)),
            score: view.getFloat64(base + 8, true),
        };
    });
    const nodeHits = Array.from({ length: header.table2Count }, (_, index) => {
        const base = header.table2Offset + index * 16;
        return {
            entityId: readArenaString(bytes, header.arenaOffset, view.getUint32(base, true), view.getUint32(base + 4, true)),
            score: view.getFloat64(base + 8, true),
        };
    });
    const diagnostics = decodeDiagnostics(bytes, view, header.table3Offset, header.table3Count, header.arenaOffset);
    return { sessionId, chunkHits, nodeHits, diagnostics };
}

function decodeGraphDeltaResult(bytes: Uint8Array): PhoenixGraphDeltaBinaryResult {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const header = decodeBinaryHeader(view);
    const sessionId = readArenaString(bytes, header.arenaOffset, header.sessionOffset, header.sessionLen);
    const chunks = Array.from({ length: header.table1Count }, (_, index) => {
        const base = header.table1Offset + index * 48;
        return {
            vertexId: readArenaString(bytes, header.arenaOffset, view.getUint32(base, true), view.getUint32(base + 4, true)),
            chunkId: readArenaString(bytes, header.arenaOffset, view.getUint32(base + 8, true), view.getUint32(base + 12, true)),
            documentId: readArenaString(bytes, header.arenaOffset, view.getUint32(base + 16, true), view.getUint32(base + 20, true)),
            noteId:
                view.getUint32(base + 28, true) > 0
                    ? readArenaString(bytes, header.arenaOffset, view.getUint32(base + 24, true), view.getUint32(base + 28, true))
                    : undefined,
            chapterId: view.getUint32(base + 32, true),
            start: view.getUint32(base + 36, true),
            end: view.getUint32(base + 40, true),
        };
    });
    const nodes = Array.from({ length: header.table2Count }, (_, index) => {
        const base = header.table2Offset + index * 52;
        return {
            nodeId: readArenaString(bytes, header.arenaOffset, view.getUint32(base, true), view.getUint32(base + 4, true)),
            kind: readArenaString(bytes, header.arenaOffset, view.getUint32(base + 8, true), view.getUint32(base + 12, true)),
            label: readArenaString(bytes, header.arenaOffset, view.getUint32(base + 16, true), view.getUint32(base + 20, true)),
            entityId:
                view.getUint32(base + 28, true) > 0
                    ? readArenaString(bytes, header.arenaOffset, view.getUint32(base + 24, true), view.getUint32(base + 28, true))
                    : undefined,
            documentId:
                view.getUint32(base + 36, true) > 0
                    ? readArenaString(bytes, header.arenaOffset, view.getUint32(base + 32, true), view.getUint32(base + 36, true))
                    : undefined,
            chapterId: view.getUint32(base + 40, true) || undefined,
            weight: view.getInt32(base + 44, true),
        };
    });
    const edges = Array.from({ length: header.table3Count }, (_, index) => {
        const base = header.table3Offset + index * 32;
        return {
            sourceId: readArenaString(bytes, header.arenaOffset, view.getUint32(base, true), view.getUint32(base + 4, true)),
            targetId: readArenaString(bytes, header.arenaOffset, view.getUint32(base + 8, true), view.getUint32(base + 12, true)),
            edgeType: readArenaString(bytes, header.arenaOffset, view.getUint32(base + 16, true), view.getUint32(base + 20, true)),
            weight: view.getInt32(base + 24, true),
        };
    });
    const diagnostics = decodeDiagnostics(bytes, view, header.table4Offset, header.table4Count, header.arenaOffset);
    return { sessionId, chunks, nodes, edges, diagnostics };
}

function decodeSessionStateResult(bytes: Uint8Array): PhoenixSessionStateBinaryResult {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const header = decodeBinaryHeader(view);
    const sessionId = readArenaString(bytes, header.arenaOffset, header.sessionOffset, header.sessionLen);
    const titleRefs = Array.from({ length: header.table2Count }, (_, index) => {
        const base = header.table2Offset + index * 8;
        return {
            offset: view.getUint32(base, true),
            len: view.getUint32(base + 4, true),
        };
    });
    const documents = Array.from({ length: header.table1Count }, (_, index) => {
        const base = header.table1Offset + index * 56;
        const titleStart = view.getUint32(base + 16, true);
        const titleCount = view.getUint32(base + 20, true);
        return {
            documentId: readArenaString(bytes, header.arenaOffset, view.getUint32(base, true), view.getUint32(base + 4, true)),
            noteId:
                view.getUint32(base + 12, true) > 0
                    ? readArenaString(bytes, header.arenaOffset, view.getUint32(base + 8, true), view.getUint32(base + 12, true))
                    : undefined,
            chapterTitles: titleRefs
                .slice(titleStart, titleStart + titleCount)
                .map((title) => readArenaString(bytes, header.arenaOffset, title.offset, title.len)),
            chapterCount: view.getUint32(base + 24, true),
            parentCount: view.getUint32(base + 28, true),
            leafCount: view.getUint32(base + 32, true),
            entityCount: view.getUint32(base + 36, true),
            discoveryCount: view.getUint32(base + 40, true),
            hasFrontMatterChapter: (view.getUint32(base + 44, true) & (1 << 4)) !== 0,
            updatedAt: Number(view.getBigUint64(base + 48, true)),
        };
    });
    const manifestNamespaces = Array.from({ length: header.table3Count }, (_, index) => {
        const base = header.table3Offset + index * 8;
        return readArenaString(bytes, header.arenaOffset, view.getUint32(base, true), view.getUint32(base + 4, true));
    });
    return { sessionId, documents, manifestNamespaces };
}

function decodeSessionStatsResult(bytes: Uint8Array): PhoenixSessionStatsBinaryResult {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const header = decodeBinaryHeader(view);
    const sessionId = readArenaString(bytes, header.arenaOffset, header.sessionOffset, header.sessionLen);
    const base = header.table1Offset;
    return {
        sessionId,
        documentCount: view.getUint32(base, true),
        chapterCount: view.getUint32(base + 4, true),
        parentCount: view.getUint32(base + 8, true),
        leafCount: view.getUint32(base + 12, true),
        entityCount: view.getUint32(base + 16, true),
        discoveryCandidateCount: view.getUint32(base + 20, true),
        graphVertexCount: view.getUint32(base + 24, true),
        graphEdgeCount: view.getUint32(base + 28, true),
        spanCount: view.getUint32(base + 32, true),
        updatedAt: Number(view.getBigUint64(base + 36, true)),
    };
}

function extractDocumentIds(request: Record<string, unknown>): string[] {
    const documents = Array.isArray(request['documents']) ? request['documents'] : [];
    const unique = new Set<string>();
    for (const document of documents) {
        if (!document || typeof document !== 'object') {
            continue;
        }
        const documentId = (document as Record<string, unknown>)['documentId'];
        if (typeof documentId === 'string' && documentId) {
            unique.add(documentId);
        }
    }
    return Array.from(unique);
}

function queryTargetsIncludeSemantic(targets: unknown): boolean {
    if (!Array.isArray(targets)) {
        return false;
    }
    return targets.some((target) => String(target).toLowerCase() === 'semantic');
}

function normalizeSemanticLeafChunks(value: unknown): PhoenixSemanticLeafChunk[] {
    if (!Array.isArray(value)) {
        return [];
    }
    const rows = value
        .map((row): PhoenixSemanticLeafChunk | null => {
            if (!row || typeof row !== 'object') {
                return null;
            }
            const record = row as Record<string, unknown>;
            if (typeof record['spanId'] !== 'string' || typeof record['text'] !== 'string') {
                return null;
            }
            return {
                spanId: record['spanId'],
                documentId: typeof record['documentId'] === 'string' ? record['documentId'] : '',
                text: record['text'],
                narrativeId: typeof record['narrativeId'] === 'string' ? record['narrativeId'] : undefined,
                folderId: typeof record['folderId'] === 'string' ? record['folderId'] : undefined,
            };
        });
    return rows.filter((row): row is PhoenixSemanticLeafChunk => row !== null && !!row.text);
}

function accumulateSemanticDocumentVector(
    accumulator: Map<string, { sum: Float32Array; leafCount: number; evidenceRefs: Set<string> }>,
    documentId: string,
    spanId: string,
    values: Float32Array,
): void {
    if (!documentId) {
        return;
    }
    const normalized = l2NormalizeSemanticVector(values);
    if (!normalized) {
        return;
    }
    let entry = accumulator.get(documentId);
    if (!entry) {
        entry = { sum: new Float32Array(normalized.length), leafCount: 0, evidenceRefs: new Set<string>() };
        accumulator.set(documentId, entry);
    }
    for (let index = 0; index < normalized.length; index += 1) {
        entry.sum[index] += normalized[index];
    }
    entry.leafCount += 1;
    if (spanId) {
        entry.evidenceRefs.add(`chunk:${spanId}`);
    }
}

function finalizeSemanticDocumentVectors(
    accumulator: Map<string, { sum: Float32Array; leafCount: number; evidenceRefs: Set<string> }>,
): PhoenixSemanticDocumentVector[] {
    const rows: PhoenixSemanticDocumentVector[] = [];
    for (const [documentId, entry] of accumulator) {
        if (!documentId || entry.leafCount <= 0) {
            continue;
        }
        const mean = new Float32Array(entry.sum.length);
        for (let index = 0; index < entry.sum.length; index += 1) {
            mean[index] = entry.sum[index] / entry.leafCount;
        }
        const values = l2NormalizeSemanticVector(mean);
        if (!values) {
            continue;
        }
        rows.push({
            documentId,
            values,
            leafCount: entry.leafCount,
            evidenceRefs: Array.from(entry.evidenceRefs),
        });
    }
    return rows;
}

function normalizeSemanticCandidatePrototypeInputs(value: unknown): PhoenixSemanticCandidatePrototypeInput[] {
    if (!Array.isArray(value)) {
        return [];
    }
    return value
        .map((row): PhoenixSemanticCandidatePrototypeInput | null => {
            if (!row || typeof row !== 'object') {
                return null;
            }
            const record = row as Record<string, unknown>;
            if (typeof record['nodeId'] !== 'string' || typeof record['nodeKind'] !== 'string') {
                return null;
            }
            const text = typeof record['text'] === 'string' ? record['text'].trim() : '';
            if (!text) {
                return null;
            }
            return {
                nodeId: record['nodeId'],
                nodeKind: record['nodeKind'],
                documentId: typeof record['documentId'] === 'string' ? record['documentId'] : undefined,
                narrativeId: typeof record['narrativeId'] === 'string' ? record['narrativeId'] : undefined,
                folderId: typeof record['folderId'] === 'string' ? record['folderId'] : undefined,
                text,
                evidenceRefs: Array.isArray(record['evidenceRefs'])
                    ? record['evidenceRefs'].filter((value): value is string => typeof value === 'string')
                    : [],
            };
        })
        .filter((row): row is PhoenixSemanticCandidatePrototypeInput => row !== null);
}

function normalizeSemanticNliJudgmentInputs(value: unknown): PhoenixSemanticNliJudgmentInput[] {
    if (!Array.isArray(value)) {
        return [];
    }
    return value
        .map((row): PhoenixSemanticNliJudgmentInput | null => {
            if (!row || typeof row !== 'object') {
                return null;
            }
            const record = row as Record<string, unknown>;
            const requiredFields = [
                'judgmentId',
                'groupId',
                'sourceId',
                'targetId',
                'edgeType',
                'direction',
                'premise',
                'hypothesis',
            ] as const;
            if (requiredFields.some((field) => typeof record[field] !== 'string')) {
                return null;
            }
            return {
                judgmentId: record['judgmentId'] as string,
                groupId: record['groupId'] as string,
                sourceId: record['sourceId'] as string,
                targetId: record['targetId'] as string,
                edgeType: record['edgeType'] as string,
                direction: record['direction'] as string,
                premise: record['premise'] as string,
                hypothesis: record['hypothesis'] as string,
            };
        })
        .filter((row): row is PhoenixSemanticNliJudgmentInput => row !== null);
}

function l2NormalizeSemanticVector(values: Float32Array): Float32Array | null {
    let magnitudeSquared = 0;
    for (let index = 0; index < values.length; index += 1) {
        const value = values[index];
        magnitudeSquared += value * value;
    }
    if (!Number.isFinite(magnitudeSquared) || magnitudeSquared <= 0) {
        return null;
    }
    const magnitude = Math.sqrt(magnitudeSquared);
    if (!Number.isFinite(magnitude) || magnitude <= 0) {
        return null;
    }
    const normalized = new Float32Array(values.length);
    for (let index = 0; index < values.length; index += 1) {
        normalized[index] = values[index] / magnitude;
    }
    return normalized;
}

function buildSemanticQueryBinaryPayload(
    request: Record<string, unknown>,
    semanticVector: Float32Array,
): Uint8Array {
    const arena = new ArenaBuilder();
    const sessionId = typeof request['sessionId'] === 'string' ? request['sessionId'] : undefined;
    const query = typeof request['query'] === 'string' ? request['query'] : '';
    const scope =
        request['scope'] && typeof request['scope'] === 'object'
            ? (request['scope'] as Record<string, unknown>)
            : {};
    const temporalJson =
        request['temporal'] === null || request['temporal'] === undefined
            ? undefined
            : JSON.stringify(request['temporal']);
    const flags =
        (sessionId ? REQUEST_FLAG_HAS_SESSION : 0) |
        (temporalJson ? REQUEST_FLAG_HAS_TEMPORAL : 0) |
        (request['includeCandidateGraph'] === true ? REQUEST_FLAG_INCLUDE_CANDIDATE_GRAPH : 0) |
        buildQueryTargetFlags(request['targets']);

    const sessionRef = arena.push(sessionId);
    const queryRef = arena.push(query);
    const worldRef = arena.push(typeof scope['worldId'] === 'string' ? scope['worldId'] : undefined);
    const narrativeRef = arena.push(
        typeof scope['narrativeId'] === 'string' ? scope['narrativeId'] : undefined,
    );
    const folderIdRef = arena.push(typeof scope['folderId'] === 'string' ? scope['folderId'] : undefined);
    const folderPathRef = arena.push(
        typeof scope['folderPath'] === 'string' ? scope['folderPath'] : undefined,
    );
    const temporalRef = arena.push(temporalJson);

    const vectorOffset = alignTo(QUERY_BINARY_HEADER_LEN + arena.length, 4);
    const totalLength = vectorOffset + semanticVector.byteLength;
    const bytes = new Uint8Array(totalLength);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    view.setUint32(0, BINARY_REQUEST_LAYOUT_VERSION, true);
    view.setUint32(4, flags, true);
    view.setUint32(8, sessionRef.offset, true);
    view.setUint32(12, sessionRef.length, true);
    view.setUint32(16, queryRef.offset, true);
    view.setUint32(20, queryRef.length, true);
    view.setUint32(24, worldRef.offset, true);
    view.setUint32(28, worldRef.length, true);
    view.setUint32(32, narrativeRef.offset, true);
    view.setUint32(36, narrativeRef.length, true);
    view.setUint32(40, folderIdRef.offset, true);
    view.setUint32(44, folderIdRef.length, true);
    view.setUint32(48, folderPathRef.offset, true);
    view.setUint32(52, folderPathRef.length, true);
    view.setUint32(56, normalizeQueryLimit(request['limit']), true);
    view.setUint32(60, temporalRef.offset, true);
    view.setUint32(64, temporalRef.length, true);
    view.setUint32(68, vectorOffset, true);
    view.setUint32(72, semanticVector.length, true);
    view.setUint32(76, semanticVector.length, true);
    view.setUint32(80, QUERY_BINARY_HEADER_LEN, true);
    view.setUint32(84, arena.length, true);
    bytes.set(arena.bytes(), QUERY_BINARY_HEADER_LEN);
    bytes.set(new Uint8Array(semanticVector.buffer, semanticVector.byteOffset, semanticVector.byteLength), vectorOffset);
    return bytes;
}

function buildEmbedUpsertBinaryPayload(
    records: Array<{ spanId: string; values: Float32Array }>,
): Uint8Array {
    const arena = new ArenaBuilder();
    const refs = records.map((record) => arena.push(record.spanId));
    const tableOffset = EMBED_UPSERT_HEADER_LEN;
    const tableLength = records.length * 8;
    const vectorOffset = tableOffset + tableLength;
    const vectorLength = records.reduce((sum, record) => sum + record.values.byteLength, 0);
    const arenaOffset = vectorOffset + vectorLength;
    const totalLength = arenaOffset + arena.length;
    const bytes = new Uint8Array(totalLength);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    view.setUint32(0, BINARY_REQUEST_LAYOUT_VERSION, true);
    view.setUint32(4, records.length, true);
    view.setUint32(8, SEMANTIC_VECTOR_DIM, true);
    view.setUint32(12, arenaOffset, true);

    let cursor = tableOffset;
    for (const ref of refs) {
        view.setUint32(cursor, ref.offset, true);
        view.setUint32(cursor + 4, ref.length, true);
        cursor += 8;
    }

    let vectorCursor = vectorOffset;
    for (const record of records) {
        bytes.set(
            new Uint8Array(record.values.buffer, record.values.byteOffset, record.values.byteLength),
            vectorCursor,
        );
        vectorCursor += record.values.byteLength;
    }

    bytes.set(arena.bytes(), arenaOffset);
    return bytes;
}

function buildQueryTargetFlags(targets: unknown): number {
    if (!Array.isArray(targets) || !targets.length) {
        return REQUEST_FLAG_TARGET_CHUNKS;
    }
    let flags = 0;
    for (const target of targets) {
        switch (String(target).toLowerCase()) {
            case 'chunks':
                flags |= REQUEST_FLAG_TARGET_CHUNKS;
                break;
            case 'nodes':
                flags |= REQUEST_FLAG_TARGET_NODES;
                break;
            case 'graph':
                flags |= REQUEST_FLAG_TARGET_GRAPH;
                break;
            case 'semantic':
                flags |= REQUEST_FLAG_TARGET_SEMANTIC;
                break;
        }
    }
    return flags || REQUEST_FLAG_TARGET_CHUNKS;
}

function normalizeQueryLimit(limit: unknown): number {
    return typeof limit === 'number' && Number.isFinite(limit) && limit >= 0
        ? Math.floor(limit)
        : 0xffffffff;
}

function alignTo(value: number, alignment: number): number {
    return Math.ceil(value / alignment) * alignment;
}

class ArenaBuilder {
    private readonly encoder = new TextEncoder();
    private readonly chunks: Uint8Array[] = [];
    private size = 0;

    push(text?: string): { offset: number; length: number } {
        if (!text) {
            return { offset: 0, length: 0 };
        }
        const bytes = this.encoder.encode(text);
        const offset = this.size;
        this.chunks.push(bytes);
        this.size += bytes.byteLength;
        return { offset, length: bytes.byteLength };
    }

    get length(): number {
        return this.size;
    }

    bytes(): Uint8Array {
        const merged = new Uint8Array(this.size);
        let offset = 0;
        for (const chunk of this.chunks) {
            merged.set(chunk, offset);
            offset += chunk.byteLength;
        }
        return merged;
    }
}

function extractOmResponseText(value: unknown): string {
    if (typeof value === 'string') {
        return value;
    }
    if (Array.isArray(value)) {
        return value
            .map((item) => {
                if (typeof item === 'string') {
                    return item;
                }
                if (item && typeof item === 'object' && typeof (item as Record<string, unknown>)['text'] === 'string') {
                    return String((item as Record<string, unknown>)['text']);
                }
                return '';
            })
            .join('');
    }
    if (value && typeof value === 'object' && typeof (value as Record<string, unknown>)['text'] === 'string') {
        return String((value as Record<string, unknown>)['text']);
    }
    return '';
}
