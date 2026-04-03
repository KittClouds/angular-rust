import { Injectable, inject } from '@angular/core';

import {
    type PhoenixChatPlannerModelResponse,
    type PhoenixChatPlannerTransportConfig,
    type PhoenixChatPlannerStep,
    type PhoenixGraphDeltaBinaryResult,
    type PhoenixOmPendingAction,
    type PhoenixOmTransportConfig,
    type PhoenixQueryBinaryResult,
    type PhoenixSessionStateBinaryResult,
    type PhoenixSessionStatsBinaryResult,
    type PhoenixSnapshotPartition,
    type PhoenixChatWorkspaceArtifact,
    PhoenixWasmService,
} from './phoenix-wasm.service';

type PhoenixTransportMethodName =
    | 'onReady'
    | 'loadWasm'
    | 'initRuntime'
    | 'createSession'
    | 'ingest'
    | 'query'
    | 'commit'
    | 'rebuild'
    | 'scan'
    | 'buildStructure'
    | 'analyzeText'
    | 'graphDelta'
    | 'sessionState'
    | 'sessionStats'
    | 'exportSnapshot'
    | 'importSnapshot'
    | 'storeCommand'
    | 'chatInit'
    | 'chatCreateThread'
    | 'chatGetThread'
    | 'chatListThreads'
    | 'chatDeleteThread'
    | 'chatAddMessage'
    | 'chatListMessages'
    | 'chatUpdateMessage'
    | 'chatAppendMessage'
    | 'chatStartStreamingMessage'
    | 'chatClearThread'
    | 'chatExportThread'
    | 'chatStartRun'
    | 'chatPollRun'
    | 'chatResumeRun'
    | 'chatCancelRun'
    | 'chatListRunEvents'
    | 'chatMarkRunStreaming'
    | 'chatCompleteRun'
    | 'chatGetPlannerStep'
    | 'chatSubmitPlannerModelResponse'
    | 'chatAdvancePlannerRun'
    | 'chatDegradePlannerRun'
    | 'chatListPlannerArtifacts'
    | 'chatPinPlannerArtifact'
    | 'chatPrepareOm'
    | 'chatApplyOmAction'
    | 'chatProcessOm'
    | 'chatProcessPlannerRun'
    | 'chatSubmitToolResults'
    | 'chatSubmitApproval'
    | 'streamChat';

type PhoenixNativeMethodName = Exclude<PhoenixTransportMethodName, 'loadWasm'>;

export type PhoenixTransportSurface = Pick<PhoenixWasmService, 'isReady' | PhoenixTransportMethodName>;

export type PhoenixNativeBridge = Pick<PhoenixWasmService, 'isReady' | PhoenixNativeMethodName> & {
    loadRuntime(): Promise<void>;
};

export type PhoenixRuntimeTarget = 'web' | 'native';

declare global {
    interface Window {
        __PHOENIX_RUNTIME_TARGET__?: PhoenixRuntimeTarget;
        __PHOENIX_NATIVE_BACKEND__?: PhoenixNativeBridge;
        __TAURI_INTERNALS__?: unknown;
    }
}

export function registerPhoenixNativeBackend(bridge: PhoenixNativeBridge): void {
    if (typeof window === 'undefined') {
        return;
    }
    window.__PHOENIX_NATIVE_BACKEND__ = bridge;
    window.__PHOENIX_RUNTIME_TARGET__ = 'native';
}

export function setPhoenixRuntimeTarget(target: PhoenixRuntimeTarget): void {
    if (typeof window === 'undefined') {
        return;
    }
    window.__PHOENIX_RUNTIME_TARGET__ = target;
}

export function detectPhoenixRuntimeTarget(): PhoenixRuntimeTarget {
    if (typeof window === 'undefined') {
        return 'web';
    }
    const explicit = window.__PHOENIX_RUNTIME_TARGET__;
    if (explicit === 'native' || explicit === 'web') {
        return explicit;
    }
    return window.__TAURI_INTERNALS__ ? 'native' : 'web';
}

@Injectable({ providedIn: 'root' })
export class PhoenixBackendService {
    private readonly wasm = inject(PhoenixWasmService);

    get target(): PhoenixRuntimeTarget {
        return detectPhoenixRuntimeTarget();
    }

    get isReady(): boolean {
        return this.target === 'native'
            ? Boolean(this.nativeBridgeOrNull()?.isReady)
            : this.wasm.isReady;
    }

    onReady(callback: () => void): void {
        if (this.target === 'native') {
            this.requireNativeBridge().onReady(callback);
            return;
        }
        this.wasm.onReady(callback);
    }

    async loadRuntime(): Promise<void> {
        if (this.target === 'native') {
            await this.requireNativeBridge().loadRuntime();
            return;
        }
        await this.wasm.loadWasm();
    }

    async loadWasm(): Promise<void> {
        await this.loadRuntime();
    }

    async initRuntime(forceReset = false): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().initRuntime(forceReset)
            : this.wasm.initRuntime(forceReset);
    }

    async createSession(
        label: string,
        scope: Parameters<PhoenixWasmService['createSession']>[1] = {},
    ): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().createSession(label, scope)
            : this.wasm.createSession(label, scope);
    }

    async ingest(request: Record<string, unknown>): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().ingest(request)
            : this.wasm.ingest(request);
    }

    async query(request: Record<string, unknown>): Promise<PhoenixQueryBinaryResult> {
        return this.target === 'native'
            ? this.requireNativeBridge().query(request)
            : this.wasm.query(request);
    }

    async commit(sessionId: string, request: Record<string, unknown> = {}): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().commit(sessionId, request)
            : this.wasm.commit(sessionId, request);
    }

    async rebuild(request: Record<string, unknown> = {}): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().rebuild(request)
            : this.wasm.rebuild(request);
    }

    async scan(request: Record<string, unknown>): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().scan(request)
            : this.wasm.scan(request);
    }

    async buildStructure(request: Record<string, unknown>): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().buildStructure(request)
            : this.wasm.buildStructure(request);
    }

    async analyzeText(text: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().analyzeText(text)
            : this.wasm.analyzeText(text);
    }

    async graphDelta(request: Record<string, unknown>): Promise<PhoenixGraphDeltaBinaryResult> {
        return this.target === 'native'
            ? this.requireNativeBridge().graphDelta(request)
            : this.wasm.graphDelta(request);
    }

    async sessionState(sessionId: string): Promise<PhoenixSessionStateBinaryResult> {
        return this.target === 'native'
            ? this.requireNativeBridge().sessionState(sessionId)
            : this.wasm.sessionState(sessionId);
    }

    async sessionStats(sessionId: string): Promise<PhoenixSessionStatsBinaryResult> {
        return this.target === 'native'
            ? this.requireNativeBridge().sessionStats(sessionId)
            : this.wasm.sessionStats(sessionId);
    }

    async exportSnapshot(
        partition: PhoenixSnapshotPartition = 'all',
        capacityHint = 8 * 1024 * 1024,
    ): Promise<Uint8Array> {
        return this.target === 'native'
            ? this.requireNativeBridge().exportSnapshot(partition, capacityHint)
            : this.wasm.exportSnapshot(partition, capacityHint);
    }

    async importSnapshot(bytes: Uint8Array): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().importSnapshot(bytes)
            : this.wasm.importSnapshot(bytes);
    }

    async storeCommand(command: string, payload: Record<string, unknown> = {}): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().storeCommand(command, payload)
            : this.wasm.storeCommand(command, payload);
    }

    async chatInit(config: Record<string, unknown>): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatInit(config)
            : this.wasm.chatInit(config);
    }

    async chatCreateThread(worldId: string, narrativeId: string, title?: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatCreateThread(worldId, narrativeId, title)
            : this.wasm.chatCreateThread(worldId, narrativeId, title);
    }

    async chatGetThread(id: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatGetThread(id)
            : this.wasm.chatGetThread(id);
    }

    async chatListThreads(worldId?: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatListThreads(worldId)
            : this.wasm.chatListThreads(worldId);
    }

    async chatDeleteThread(id: string): Promise<void> {
        if (this.target === 'native') {
            await this.requireNativeBridge().chatDeleteThread(id);
            return;
        }
        await this.wasm.chatDeleteThread(id);
    }

    async chatAddMessage(threadId: string, role: string, content: string, narrativeId?: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatAddMessage(threadId, role, content, narrativeId)
            : this.wasm.chatAddMessage(threadId, role, content, narrativeId);
    }

    async chatListMessages(threadId: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatListMessages(threadId)
            : this.wasm.chatListMessages(threadId);
    }

    async chatUpdateMessage(messageId: string, content: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatUpdateMessage(messageId, content)
            : this.wasm.chatUpdateMessage(messageId, content);
    }

    async chatAppendMessage(messageId: string, chunk: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatAppendMessage(messageId, chunk)
            : this.wasm.chatAppendMessage(messageId, chunk);
    }

    async chatStartStreamingMessage(threadId: string, narrativeId?: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatStartStreamingMessage(threadId, narrativeId)
            : this.wasm.chatStartStreamingMessage(threadId, narrativeId);
    }

    async chatClearThread(threadId: string): Promise<void> {
        if (this.target === 'native') {
            await this.requireNativeBridge().chatClearThread(threadId);
            return;
        }
        await this.wasm.chatClearThread(threadId);
    }

    async chatExportThread(threadId: string): Promise<string> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatExportThread(threadId)
            : this.wasm.chatExportThread(threadId);
    }

    async chatStartRun(threadId: string, prompt: string, options: Record<string, unknown>): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatStartRun(threadId, prompt, options)
            : this.wasm.chatStartRun(threadId, prompt, options);
    }

    async chatPollRun(runId: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatPollRun(runId)
            : this.wasm.chatPollRun(runId);
    }

    async chatResumeRun(runId: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatResumeRun(runId)
            : this.wasm.chatResumeRun(runId);
    }

    async chatCancelRun(runId: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatCancelRun(runId)
            : this.wasm.chatCancelRun(runId);
    }

    async chatListRunEvents(threadId: string, limit = 100): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatListRunEvents(threadId, limit)
            : this.wasm.chatListRunEvents(threadId, limit);
    }

    async chatMarkRunStreaming(runId: string, assistantMessageId: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatMarkRunStreaming(runId, assistantMessageId)
            : this.wasm.chatMarkRunStreaming(runId, assistantMessageId);
    }

    async chatCompleteRun(
        runId: string,
        assistantMessageId: string,
        finalResponse: string,
        finalError?: string,
    ): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatCompleteRun(runId, assistantMessageId, finalResponse, finalError)
            : this.wasm.chatCompleteRun(runId, assistantMessageId, finalResponse, finalError);
    }

    async chatGetPlannerStep(runId: string): Promise<PhoenixChatPlannerStep | null> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatGetPlannerStep(runId)
            : this.wasm.chatGetPlannerStep(runId);
    }

    async chatSubmitPlannerModelResponse(
        runId: string,
        response: PhoenixChatPlannerModelResponse,
    ): Promise<PhoenixChatPlannerStep | null> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatSubmitPlannerModelResponse(runId, response)
            : this.wasm.chatSubmitPlannerModelResponse(runId, response);
    }

    async chatAdvancePlannerRun(runId: string): Promise<PhoenixChatPlannerStep | null> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatAdvancePlannerRun(runId)
            : this.wasm.chatAdvancePlannerRun(runId);
    }

    async chatDegradePlannerRun(runId: string, reason: string): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatDegradePlannerRun(runId, reason)
            : this.wasm.chatDegradePlannerRun(runId, reason);
    }

    async chatListPlannerArtifacts(runId: string): Promise<PhoenixChatWorkspaceArtifact[]> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatListPlannerArtifacts(runId)
            : this.wasm.chatListPlannerArtifacts(runId);
    }

    async chatPinPlannerArtifact(runId: string, key: string, pinned = true): Promise<PhoenixChatWorkspaceArtifact | null> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatPinPlannerArtifact(runId, key, pinned)
            : this.wasm.chatPinPlannerArtifact(runId, key, pinned);
    }

    async chatPrepareOm(threadId: string): Promise<PhoenixOmPendingAction | null> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatPrepareOm(threadId)
            : this.wasm.chatPrepareOm(threadId);
    }

    async chatApplyOmAction(action: PhoenixOmPendingAction, response: string): Promise<boolean> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatApplyOmAction(action, response)
            : this.wasm.chatApplyOmAction(action, response);
    }

    async chatProcessOm(threadId: string, config: PhoenixOmTransportConfig): Promise<boolean> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatProcessOm(threadId, config)
            : this.wasm.chatProcessOm(threadId, config);
    }

    async chatProcessPlannerRun(
        runId: string,
        config: PhoenixChatPlannerTransportConfig,
    ): Promise<boolean> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatProcessPlannerRun(runId, config)
            : this.wasm.chatProcessPlannerRun(runId, config);
    }

    async chatSubmitToolResults(runId: string, results: unknown[]): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatSubmitToolResults(runId, results)
            : this.wasm.chatSubmitToolResults(runId, results);
    }

    async chatSubmitApproval(
        runId: string,
        approvalId: string,
        approved: boolean,
        decisionJSON?: string,
    ): Promise<any> {
        return this.target === 'native'
            ? this.requireNativeBridge().chatSubmitApproval(runId, approvalId, approved, decisionJSON)
            : this.wasm.chatSubmitApproval(runId, approvalId, approved, decisionJSON);
    }

    async streamChat(
        request: Parameters<PhoenixWasmService['streamChat']>[0],
        callbacks: Parameters<PhoenixWasmService['streamChat']>[1],
    ): Promise<void> {
        if (this.target === 'native') {
            await this.requireNativeBridge().streamChat(request, callbacks);
            return;
        }
        await this.wasm.streamChat(request, callbacks);
    }

    private nativeBridgeOrNull(): PhoenixNativeBridge | null {
        if (typeof window === 'undefined') {
            return null;
        }
        return window.__PHOENIX_NATIVE_BACKEND__ ?? null;
    }

    private requireNativeBridge(): PhoenixNativeBridge {
        const bridge = this.nativeBridgeOrNull();
        if (bridge) {
            return bridge;
        }
        throw new Error(
            'Phoenix native backend was selected, but no native bridge was registered. ' +
            'Register a taurpc-backed bridge with registerPhoenixNativeBackend() before Angular boot.',
        );
    }
}
