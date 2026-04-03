import { Injectable, computed, inject, signal } from '@angular/core';

import { getSetting, setSetting } from '../dexie/settings.service';
import { ScopeService } from './scope.service';
import { PhoenixBackendService } from '../../services/phoenix-backend.service';
import { PhoenixStoreService } from '../../services/phoenix-store.service';

export interface Thread {
    id: string;
    world_id: string;
    narrative_id: string;
    title: string;
    created_at: number;
    updated_at: number;
}

export interface ThreadMessage {
    id: string;
    thread_id: string;
    role: 'user' | 'assistant' | 'system';
    content: string;
    narrative_id: string;
    created_at: number;
    updated_at: number;
    is_streaming: boolean;
}

export interface Memory {
    id: string;
    content: string;
    memory_type: 'fact' | 'preference' | 'entity_mention' | 'relation';
    confidence: number;
    source_role: string;
    entity_id: string;
    created_at: number;
    updated_at: number;
}

export interface OpenRouterStructuredOutputConfig {
    enabled?: boolean;
    type?: 'json_schema' | 'json_object';
    schema?: unknown;
    strict?: boolean;
    name?: string;
    description?: string;
}

export interface OpenRouterPlugin {
    id: string;
}

export interface OpenRouterRequestOptions {
    structuredOutput?: OpenRouterStructuredOutputConfig;
    plugins?: OpenRouterPlugin[];
}

export interface ChatConfig {
    apiKey: string;
    model: string;
    structuredOutput?: OpenRouterStructuredOutputConfig;
    plugins?: OpenRouterPlugin[];
    omEnabled: boolean;
    omModel?: string;
    observeThreshold?: number;
    reflectThreshold?: number;
    temperature?: number;
    maxTokens?: number;
    reasoningEnabled?: boolean;
    reasoningEffort?: 'low' | 'medium' | 'high';
    reasoningMaxTokens?: number;
    includeReasoning?: boolean;
}

export interface CreateThreadOptions {
    worldId?: string;
    narrativeId?: string;
}

export interface OpenRouterMessage {
    role: 'system' | 'user' | 'assistant';
    content: string | null;
}

export interface ChatProgressEvent {
    stage: 'reasoning' | 'stream';
    status: 'running' | 'done' | 'error';
    detail?: string;
}

export type ChatRunStatus =
    | 'queued'
    | 'gathering'
    | 'planning'
    | 'executing_tools'
    | 'awaiting_tool_host'
    | 'awaiting_approval'
    | 'ready_to_answer'
    | 'streaming'
    | 'completed'
    | 'degraded'
    | 'failed'
    | 'cancelled';

export interface RunOptions {
    finalProvider: string;
    finalModel: string;
    plannerModel?: string;
    omModel?: string;
    plannerEnabled: boolean;
    omEnabled: boolean;
    workspaceEnabled: boolean;
    mutationsEnabled: boolean;
    deadlineMs: number;
    mutationPolicy: 'confirm' | 'trusted_auto_edit' | 'full_autonomy';
    narrativeId?: string;
    folderId?: string;
    scopeId?: string;
    baseSystemPrompt?: string;
    initialExternalContext?: string;
}

export interface CapabilityProfile {
    omEnabled: boolean;
    workspaceEnabled: boolean;
    plannerEnabled: boolean;
    goToolHost: boolean;
    tsToolHost: boolean;
    blockSearch: boolean;
}

export interface EvidenceItem {
    id: string;
    source: string;
    title?: string;
    content: string;
    score?: number;
    metadata?: Record<string, unknown>;
}

export interface ChatWorkspaceArtifact {
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

export interface ChatPlannerToolSpec {
    name: string;
    description: string;
    parametersJson: unknown;
}

export interface ChatPlannerToolCall {
    id: string;
    name: string;
    argumentsJson: string;
}

export interface ChatPlannerMessage {
    role: string;
    content: string;
    name?: string;
    toolCallId?: string;
    toolCalls: ChatPlannerToolCall[];
}

export interface ChatPlannerModelRequest {
    runId: string;
    threadId: string;
    model: string;
    allowTools: boolean;
    tools: ChatPlannerToolSpec[];
    messages: ChatPlannerMessage[];
}

export interface ChatPlannerModelResponse {
    content: string;
    toolCalls: ChatPlannerToolCall[];
}

export type ChatPlannerStep =
    | {
          kind: 'modelRequest';
          request: ChatPlannerModelRequest;
      }
    | {
          kind: 'toolCalls';
          runId: string;
          toolCalls: ChatPlannerToolCall[];
      }
    | {
          kind: 'complete';
          runId: string;
          response: string;
      };

export interface ChatRun {
    id: string;
    threadId: string;
    userPrompt: string;
    status: ChatRunStatus;
    options: RunOptions;
    capabilities: CapabilityProfile;
    preparedContext: string;
    preparedSystemPrompt: string;
    plannerMessagesJson: string;
    evidenceJson: string;
    missingCapabilitiesJson: string;
    error?: string;
    finalResponse?: string;
    assistantMessageId?: string;
    deadlineAt: number;
    completedAt?: number;
    createdAt: number;
    updatedAt: number;
}

export interface ChatRunEvent {
    id: string;
    runId: string;
    phase: string;
    kind: string;
    label: string;
    detail?: string;
    status?: string;
    payload?: string;
    latencyMs?: number;
    createdAt: number;
}

export interface ChatToolCall {
    id: string;
    runId: string;
    toolCallId: string;
    toolName: string;
    host: 'go' | 'typescript';
    class: 'read' | 'proposal' | 'write';
    status: string;
    argumentsJson: string;
    resultJson?: string;
    error?: string;
    idempotencyKey?: string;
    approvalId?: string;
    startedAt?: number;
    completedAt?: number;
    latencyMs?: number;
}

export interface ToolProposal {
    proposalId: string;
    toolName: string;
    affectedNoteId?: string;
    summary: string;
    diffPreview?: string;
    expectedRevision?: number;
    rollbackToken?: string;
    payloadJson?: string;
}

export interface ChatApprovalRequest {
    id: string;
    runId: string;
    toolCallId: string;
    toolName: string;
    status: string;
    affectedNoteId?: string;
    summary: string;
    diffPreview?: string;
    expectedRevision?: number;
    rollbackToken?: string;
    proposalJson?: string;
    decisionJson?: string;
    createdAt: number;
    updatedAt: number;
}

export interface ChatRunSnapshot {
    run: ChatRun;
    events: ChatRunEvent[];
    toolCalls: ChatToolCall[];
    approvals: ChatApprovalRequest[];
    evidence: EvidenceItem[];
    missingCapabilities: string[];
    plannerStep?: ChatPlannerStep | null;
    artifacts: ChatWorkspaceArtifact[];
}

export interface ToolResultSubmission {
    callId?: string;
    toolCallId?: string;
    resultJson?: string;
    error?: string;
    proposal?: ToolProposal;
}

type PhoenixThreadRecord = {
    id: string;
    worldId?: string;
    world_id?: string;
    narrativeId?: string;
    narrative_id?: string;
    title?: string;
    createdAt?: number;
    created_at?: number;
    updatedAt?: number;
    updated_at?: number;
};

type PhoenixThreadMessageRecord = {
    id: string;
    threadId?: string;
    thread_id?: string;
    role?: string;
    content?: string;
    narrativeId?: string;
    narrative_id?: string;
    createdAt?: number;
    created_at?: number;
    updatedAt?: number;
    updated_at?: number;
    isStreaming?: boolean;
    is_streaming?: boolean;
};

@Injectable({ providedIn: 'root' })
export class PhoenixChatService {
    private readonly phoenix = inject(PhoenixBackendService);
    private readonly scopeService = inject(ScopeService);
    private readonly storeService = inject(PhoenixStoreService);

    readonly ready = signal(false);
    readonly initialized = signal(false);
    readonly currentThread = signal<Thread | null>(null);
    readonly messages = signal<ThreadMessage[]>([]);
    readonly threads = signal<Thread[]>([]);
    readonly loading = signal(false);

    readonly messageCount = computed(() => this.messages().length);
    readonly hasThread = computed(() => this.currentThread() !== null);

    private snapshotTimeout: ReturnType<typeof setTimeout> | null = null;

    constructor() {
        console.log('[PhoenixChatService] Service created');
    }

    async init(config?: ChatConfig): Promise<void> {
        if (this.initialized()) {
            return;
        }

        await this.storeService.initialize();

        const savedConfig = config ?? getSetting<ChatConfig | null>('openrouter:config', null) ?? undefined;
        await this.applyRuntimeConfig(savedConfig);

        this.ready.set(true);
        this.initialized.set(true);
        console.log('[PhoenixChatService] Chat service initialized');

        await this.loadThreads();
        await this.restoreLastThread();
    }

    async updateConfig(config: ChatConfig): Promise<void> {
        await this.storeService.initialize();
        await this.applyRuntimeConfig(config);
    }

    async createThread(options?: CreateThreadOptions): Promise<Thread | null> {
        await this.ensureInitialized();
        const scope = this.scopeService.activeScope();
        const worldId = options?.worldId || scope.id || 'default';
        const narrativeId = options?.narrativeId || scope.narrativeId || 'default';

        try {
            const raw = await this.phoenix.chatCreateThread(worldId, narrativeId);
            const thread = toThread(raw);
            this.currentThread.set(thread);
            this.messages.set([]);
            this.threads.update((threads) => [thread, ...threads.filter((item) => item.id !== thread.id)]);
            setSetting('chat:activeThreadId', thread.id);
            this.scheduleSnapshot();
            return thread;
        } catch (error) {
            console.error('[PhoenixChatService] Create thread error:', error);
            return null;
        }
    }

    async loadThread(threadId: string): Promise<void> {
        await this.ensureInitialized();
        this.loading.set(true);
        try {
            const raw = await this.phoenix.chatGetThread(threadId);
            if (!raw) {
                console.warn('[PhoenixChatService] Thread not found (stale ID):', threadId);
                return;
            }

            const thread = toThread(raw);
            this.currentThread.set(thread);
            await this.loadMessages(threadId);
            setSetting('chat:activeThreadId', threadId);
        } catch (error) {
            console.error('[PhoenixChatService] Load thread error:', error);
        } finally {
            this.loading.set(false);
        }
    }

    async loadThreads(): Promise<void> {
        await this.ensureInitialized();
        const scope = this.scopeService.activeScope();
        const worldId = scope.id || '';

        try {
            const payload = await this.phoenix.chatListThreads(worldId);
            this.threads.set(Array.isArray(payload) ? payload.map(toThread) : []);
        } catch (error) {
            console.error('[PhoenixChatService] Load threads error:', error);
            this.threads.set([]);
        }
    }

    async deleteThread(threadId: string): Promise<boolean> {
        await this.ensureInitialized();
        try {
            await this.phoenix.chatDeleteThread(threadId);
            this.threads.update((threads) => threads.filter((thread) => thread.id !== threadId));
            if (this.currentThread()?.id === threadId) {
                this.currentThread.set(null);
                this.messages.set([]);
                setSetting('chat:activeThreadId', null);
            }
            this.scheduleSnapshot();
            return true;
        } catch (error) {
            console.error('[PhoenixChatService] Delete thread error:', error);
            return false;
        }
    }

    async getOrCreateThread(): Promise<Thread | null> {
        if (this.currentThread()) {
            return this.currentThread();
        }

        const lastThreadId = getSetting<string | null>('chat:activeThreadId', null);
        if (lastThreadId) {
            await this.loadThread(lastThreadId);
            if (this.currentThread()) {
                return this.currentThread();
            }
        }

        return this.createThread();
    }

    async addMessage(
        role: 'user' | 'assistant' | 'system',
        content: string,
    ): Promise<ThreadMessage | null> {
        const thread = await this.getOrCreateThread();
        if (!thread) {
            return null;
        }

        try {
            const raw = await this.phoenix.chatAddMessage(
                thread.id,
                role,
                content,
                thread.narrative_id,
            );
            const message = toThreadMessage(raw);
            this.messages.update((messages) => [...messages, message]);
            this.scheduleSnapshot();
            void this.triggerOm(thread.id);
            return message;
        } catch (error) {
            console.error('[PhoenixChatService] Add message error:', error);
            return null;
        }
    }

    async addUserMessage(content: string): Promise<ThreadMessage | null> {
        return this.addMessage('user', content);
    }

    async addAssistantMessage(content: string): Promise<ThreadMessage | null> {
        return this.addMessage('assistant', content);
    }

    async updateMessage(messageId: string, content: string): Promise<boolean> {
        await this.ensureInitialized();
        try {
            const raw = await this.phoenix.chatUpdateMessage(messageId, content);
            const message = raw ? toThreadMessage(raw) : null;
            if (!message) {
                return false;
            }
            this.messages.update((messages) =>
                messages.map((current) => (current.id === messageId ? message : current)),
            );
            this.scheduleSnapshot();
            void this.triggerOm(message.thread_id);
            return true;
        } catch (error) {
            console.error('[PhoenixChatService] Update message error:', error);
            return false;
        }
    }

    async appendMessage(messageId: string, chunk: string): Promise<boolean> {
        await this.ensureInitialized();
        try {
            const raw = await this.phoenix.chatAppendMessage(messageId, chunk);
            const message = raw ? toThreadMessage(raw) : null;
            if (!message) {
                return false;
            }
            this.messages.update((messages) =>
                messages.map((current) => (current.id === messageId ? message : current)),
            );
            return true;
        } catch (error) {
            console.error('[PhoenixChatService] Append message error:', error);
            return false;
        }
    }

    async startStreamingMessage(): Promise<ThreadMessage | null> {
        await this.ensureInitialized();
        const thread = this.currentThread();
        if (!thread) {
            return null;
        }

        try {
            const raw = await this.phoenix.chatStartStreamingMessage(thread.id, thread.narrative_id);
            const message = toThreadMessage(raw);
            this.messages.update((messages) => [...messages, message]);
            return message;
        } catch (error) {
            console.error('[PhoenixChatService] Start streaming message error:', error);
            return null;
        }
    }

    async getMemories(): Promise<Memory[]> {
        return [];
    }

    async getContext(): Promise<string> {
        return '';
    }

    async clearThread(): Promise<boolean> {
        await this.ensureInitialized();
        const thread = this.currentThread();
        if (!thread) {
            return false;
        }

        try {
            await this.phoenix.chatClearThread(thread.id);
            this.messages.set([]);
            this.scheduleSnapshot();
            return true;
        } catch (error) {
            console.error('[PhoenixChatService] Clear thread error:', error);
            return false;
        }
    }

    async exportThread(): Promise<string> {
        await this.ensureInitialized();
        const thread = this.currentThread();
        if (!thread) {
            return '{}';
        }
        try {
            return await this.phoenix.chatExportThread(thread.id);
        } catch (error) {
            console.error('[PhoenixChatService] Export thread error:', error);
            return '{}';
        }
    }

    async startRun(prompt: string, options: RunOptions): Promise<ChatRun | null> {
        const thread = await this.getOrCreateThread();
        if (!thread) {
            return null;
        }

        const normalized: RunOptions = {
            ...options,
            narrativeId: options.narrativeId || thread.narrative_id || '',
            scopeId: options.scopeId || options.narrativeId || thread.narrative_id || '',
        };

        try {
            const raw = await this.phoenix.chatStartRun(
                thread.id,
                prompt,
                normalized as unknown as Record<string, unknown>,
            );
            this.scheduleSnapshot();
            return raw ? toChatRun(raw) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Start run error:', error);
            return null;
        }
    }

    async pollRun(runId: string): Promise<ChatRunSnapshot | null> {
        await this.ensureInitialized();
        try {
            const raw = await this.phoenix.chatPollRun(runId);
            return raw ? toChatRunSnapshot(raw) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Poll run error:', error);
            return null;
        }
    }

    async getPlannerStep(runId: string): Promise<ChatPlannerStep | null> {
        await this.ensureInitialized();
        try {
            const raw = await this.phoenix.chatGetPlannerStep(runId);
            return raw ? toChatPlannerStep(raw) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Get planner step error:', error);
            return null;
        }
    }

    async submitPlannerModelResponse(
        runId: string,
        response: ChatPlannerModelResponse,
    ): Promise<ChatPlannerStep | null> {
        await this.ensureInitialized();
        try {
            const raw = await this.phoenix.chatSubmitPlannerModelResponse(runId, response);
            return raw ? toChatPlannerStep(raw) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Submit planner model response error:', error);
            return null;
        }
    }

    async advancePlannerRun(runId: string): Promise<ChatPlannerStep | null> {
        await this.ensureInitialized();
        try {
            const raw = await this.phoenix.chatAdvancePlannerRun(runId);
            return raw ? toChatPlannerStep(raw) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Advance planner run error:', error);
            return null;
        }
    }

    async degradePlannerRun(runId: string, reason: string): Promise<ChatRunSnapshot | null> {
        await this.ensureInitialized();
        try {
            const raw = await this.phoenix.chatDegradePlannerRun(runId, reason);
            this.scheduleSnapshot();
            return raw ? toChatRunSnapshot(raw) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Degrade planner run error:', error);
            return null;
        }
    }

    async listPlannerArtifacts(runId: string): Promise<ChatWorkspaceArtifact[]> {
        await this.ensureInitialized();
        try {
            const payload = await this.phoenix.chatListPlannerArtifacts(runId);
            return Array.isArray(payload) ? payload.map(toChatWorkspaceArtifact) : [];
        } catch (error) {
            console.error('[PhoenixChatService] List planner artifacts error:', error);
            return [];
        }
    }

    async pinPlannerArtifact(
        runId: string,
        key: string,
        pinned = true,
    ): Promise<ChatWorkspaceArtifact | null> {
        await this.ensureInitialized();
        try {
            const payload = await this.phoenix.chatPinPlannerArtifact(runId, key, pinned);
            return payload ? toChatWorkspaceArtifact(payload) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Pin planner artifact error:', error);
            return null;
        }
    }

    async processPlannerRun(runId: string): Promise<boolean> {
        await this.ensureInitialized();
        const config = getSetting<ChatConfig | null>('openrouter:config', null);
        if (!config?.apiKey?.trim()) {
            await this.degradePlannerRun(runId, 'Planner requires an OpenRouter API key.');
            return false;
        }

        try {
            const processed = await this.phoenix.chatProcessPlannerRun(runId, {
                apiKey: config.apiKey,
                defaultModel: config.model || DEFAULT_CHAT_MODEL,
                temperature: config.temperature,
                maxTokens: config.maxTokens,
            });
            this.scheduleSnapshot();
            return processed;
        } catch (error) {
            console.error('[PhoenixChatService] Planner processing error:', error);
            const reason = error instanceof Error ? error.message : String(error);
            await this.degradePlannerRun(runId, reason);
            return false;
        }
    }

    async submitToolResults(runId: string, results: ToolResultSubmission[]): Promise<ChatRunSnapshot | null> {
        await this.ensureInitialized();
        try {
            const payload = await this.phoenix.chatSubmitToolResults(runId, results);
            this.scheduleSnapshot();
            return payload ? toChatRunSnapshot(payload) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Submit tool results error:', error);
            return null;
        }
    }

    async submitApproval(
        runId: string,
        approvalId: string,
        approved: boolean,
        decisionJSON?: string,
    ): Promise<ChatRunSnapshot | null> {
        await this.ensureInitialized();
        try {
            const payload = await this.phoenix.chatSubmitApproval(runId, approvalId, approved, decisionJSON);
            this.scheduleSnapshot();
            return payload ? toChatRunSnapshot(payload) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Submit approval error:', error);
            return null;
        }
    }

    async resumeRun(runId: string): Promise<ChatRun | null> {
        await this.ensureInitialized();
        try {
            const raw = await this.phoenix.chatResumeRun(runId);
            this.scheduleSnapshot();
            return raw ? toChatRun(raw) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Resume run error:', error);
            return null;
        }
    }

    async cancelRun(runId: string): Promise<boolean> {
        await this.ensureInitialized();
        try {
            await this.phoenix.chatCancelRun(runId);
            this.scheduleSnapshot();
            return true;
        } catch (error) {
            console.error('[PhoenixChatService] Cancel run error:', error);
            return false;
        }
    }

    async listRunEvents(threadId: string, limit = 100): Promise<ChatRunEvent[]> {
        await this.ensureInitialized();
        try {
            const payload = await this.phoenix.chatListRunEvents(threadId, limit);
            return Array.isArray(payload) ? payload.map(toChatRunEvent) : [];
        } catch (error) {
            console.error('[PhoenixChatService] List run events error:', error);
            return [];
        }
    }

    async markRunStreaming(runId: string, assistantMessageId: string): Promise<ChatRunSnapshot | null> {
        await this.ensureInitialized();
        try {
            const payload = await this.phoenix.chatMarkRunStreaming(runId, assistantMessageId);
            this.scheduleSnapshot();
            return payload ? toChatRunSnapshot(payload) : null;
        } catch (error) {
            console.error('[PhoenixChatService] Mark run streaming error:', error);
            return null;
        }
    }

    async completeRun(
        runId: string,
        assistantMessageId: string,
        finalResponse: string,
        finalError?: string,
    ): Promise<ChatRunSnapshot | null> {
        await this.ensureInitialized();
        try {
            const payload = await this.phoenix.chatCompleteRun(
                runId,
                assistantMessageId,
                finalResponse,
                finalError,
            );
            const snapshot = payload ? toChatRunSnapshot(payload) : null;
            this.scheduleSnapshot();
            if (snapshot?.run.threadId) {
                void this.triggerOm(snapshot.run.threadId);
            }
            return snapshot;
        } catch (error) {
            console.error('[PhoenixChatService] Complete run error:', error);
            return null;
        }
    }

    async streamChat(
        messages: OpenRouterMessage[],
        callbacks: {
            onChunk: (chunk: string) => void;
            onComplete: (full: string) => void;
            onError: (err: Error) => void;
            onReasoningChunk?: (chunk: string) => void;
            onEvent?: (event: ChatProgressEvent) => void;
        },
        systemPrompt?: string,
        requestOptions?: OpenRouterRequestOptions,
    ): Promise<void> {
        await this.ensureInitialized();
        const config = getSetting<ChatConfig | null>('openrouter:config', null);
        if (!config?.apiKey) {
            callbacks.onEvent?.({
                stage: 'stream',
                status: 'error',
                detail: 'OpenRouter API key is not configured.',
            });
            callbacks.onError(new Error('OpenRouter API key is not configured.'));
            return;
        }

        try {
            await this.phoenix.streamChat(
                {
                    config: {
                        apiKey: config.apiKey,
                        model: config.model || DEFAULT_CHAT_MODEL,
                        temperature: config.temperature,
                        maxTokens: config.maxTokens,
                        reasoningEnabled: config.reasoningEnabled ?? false,
                        reasoningEffort: config.reasoningEffort ?? 'medium',
                        reasoningMaxTokens: config.reasoningMaxTokens,
                        includeReasoning: config.includeReasoning ?? false,
                    },
                    messages,
                    ...(systemPrompt ? { systemPrompt } : {}),
                    ...(requestOptions ? { requestOptions } : {}),
                },
                callbacks,
            );
        } catch (error) {
            callbacks.onEvent?.({
                stage: 'stream',
                status: 'error',
                detail: error instanceof Error ? error.message : String(error),
            });
            callbacks.onError(error instanceof Error ? error : new Error(String(error)));
        }
    }

    async newSession(): Promise<Thread | null> {
        this.messages.set([]);
        return this.createThread();
    }

    private async loadMessages(threadId: string): Promise<void> {
        try {
            const payload = await this.phoenix.chatListMessages(threadId);
            this.messages.set(Array.isArray(payload) ? payload.map(toThreadMessage) : []);
        } catch (error) {
            console.error('[PhoenixChatService] Load messages error:', error);
            this.messages.set([]);
        }
    }

    private async applyRuntimeConfig(config?: ChatConfig): Promise<void> {
        const runtimeConfig = {
            model: config?.model || DEFAULT_CHAT_MODEL,
            temperatureMilli:
                typeof config?.temperature === 'number'
                    ? Math.max(0, Math.min(65_535, Math.round(config.temperature * 1000)))
                    : null,
            maxTokens: config?.maxTokens ?? null,
            reasoningEnabled: config?.reasoningEnabled ?? false,
            reasoningEffort: config?.reasoningEffort ?? 'medium',
            reasoningMaxTokens: config?.reasoningMaxTokens ?? null,
            includeReasoning: config?.includeReasoning ?? false,
            omEnabled: config?.omEnabled ?? false,
            omModel: config?.omModel ?? null,
            observeThreshold: config?.observeThreshold ?? null,
            reflectThreshold: config?.reflectThreshold ?? null,
        };
        await this.phoenix.chatInit(runtimeConfig);
    }

    private async restoreLastThread(): Promise<void> {
        const lastThreadId = getSetting<string | null>('chat:activeThreadId', null);
        if (!lastThreadId) {
            return;
        }
        await this.loadThread(lastThreadId);
        if (!this.currentThread()) {
            setSetting('chat:activeThreadId', null);
        }
    }

    private async ensureInitialized(): Promise<void> {
        if (!this.initialized()) {
            await this.init();
        }
    }

    private async triggerOm(threadId: string): Promise<void> {
        const config = getSetting<ChatConfig | null>('openrouter:config', null);
        if (!threadId || !config?.omEnabled || !config.apiKey?.trim()) {
            return;
        }

        try {
            const mutated = await this.phoenix.chatProcessOm(threadId, {
                apiKey: config.apiKey,
                defaultModel: config.model || DEFAULT_CHAT_MODEL,
                omModel: config.omModel,
                temperature: config.temperature,
                maxTokens: config.maxTokens,
            });
            if (mutated) {
                this.scheduleSnapshot();
            }
        } catch (error) {
            console.error('[PhoenixChatService] OM processing error:', error);
        }
    }

    private scheduleSnapshot(): void {
        if (this.snapshotTimeout) {
            clearTimeout(this.snapshotTimeout);
        }
        this.snapshotTimeout = setTimeout(() => {
            void this.storeService.triggerSnapshot().catch((error) => {
                console.error('[PhoenixChatService] Failed to persist Phoenix snapshot:', error);
            });
        }, 1200);
    }
}

const DEFAULT_CHAT_MODEL = 'meta-llama/llama-3.3-70b-instruct:free';

function numeric(value: unknown, fallback = Date.now()): number {
    return typeof value === 'number' ? value : fallback;
}

function stringValue(value: unknown, fallback = ''): string {
    return typeof value === 'string' ? value : fallback;
}

function toThread(raw: PhoenixThreadRecord): Thread {
    return {
        id: raw.id,
        world_id: stringValue(raw.world_id ?? raw.worldId),
        narrative_id: stringValue(raw.narrative_id ?? raw.narrativeId),
        title: stringValue(raw.title),
        created_at: numeric(raw.created_at ?? raw.createdAt),
        updated_at: numeric(raw.updated_at ?? raw.updatedAt),
    };
}

function toThreadMessage(raw: PhoenixThreadMessageRecord): ThreadMessage {
    return {
        id: raw.id,
        thread_id: stringValue(raw.thread_id ?? raw.threadId),
        role: (stringValue(raw.role, 'assistant') as ThreadMessage['role']),
        content: stringValue(raw.content),
        narrative_id: stringValue(raw.narrative_id ?? raw.narrativeId),
        created_at: numeric(raw.created_at ?? raw.createdAt),
        updated_at: numeric(raw.updated_at ?? raw.updatedAt),
        is_streaming: Boolean(raw.is_streaming ?? raw.isStreaming),
    };
}

function toCapabilityProfile(raw: any): CapabilityProfile {
    return {
        omEnabled: !!raw?.omEnabled,
        workspaceEnabled: !!raw?.workspaceEnabled,
        plannerEnabled: !!raw?.plannerEnabled,
        goToolHost: !!raw?.goToolHost,
        tsToolHost: !!raw?.tsToolHost,
        blockSearch: !!raw?.blockSearch,
    };
}

function toRunOptions(raw: any): RunOptions {
    return {
        finalProvider: stringValue(raw?.finalProvider),
        finalModel: stringValue(raw?.finalModel),
        plannerModel: raw?.plannerModel ? String(raw.plannerModel) : undefined,
        omModel: raw?.omModel ? String(raw.omModel) : undefined,
        plannerEnabled: !!raw?.plannerEnabled,
        omEnabled: !!raw?.omEnabled,
        workspaceEnabled: !!raw?.workspaceEnabled,
        mutationsEnabled: !!raw?.mutationsEnabled,
        deadlineMs: numeric(raw?.deadlineMs, 0),
        mutationPolicy: (stringValue(raw?.mutationPolicy, 'confirm') as RunOptions['mutationPolicy']),
        narrativeId: raw?.narrativeId ? String(raw.narrativeId) : undefined,
        folderId: raw?.folderId ? String(raw.folderId) : undefined,
        scopeId: raw?.scopeId ? String(raw.scopeId) : undefined,
        baseSystemPrompt: raw?.baseSystemPrompt ? String(raw.baseSystemPrompt) : undefined,
        initialExternalContext: raw?.initialExternalContext
            ? String(raw.initialExternalContext)
            : undefined,
    };
}

function toChatRun(raw: any): ChatRun {
    return {
        id: stringValue(raw?.id),
        threadId: stringValue(raw?.threadId),
        userPrompt: stringValue(raw?.userPrompt),
        status: stringValue(raw?.status, 'queued') as ChatRunStatus,
        options: toRunOptions(raw?.options),
        capabilities: toCapabilityProfile(raw?.capabilities),
        preparedContext: stringValue(raw?.preparedContext),
        preparedSystemPrompt: stringValue(raw?.preparedSystemPrompt),
        plannerMessagesJson: stringValue(raw?.plannerMessagesJson, '[]'),
        evidenceJson: stringValue(raw?.evidenceJson, '[]'),
        missingCapabilitiesJson: stringValue(raw?.missingCapabilitiesJson, '[]'),
        error: raw?.error ? String(raw.error) : undefined,
        finalResponse: raw?.finalResponse ? String(raw.finalResponse) : undefined,
        assistantMessageId: raw?.assistantMessageId ? String(raw.assistantMessageId) : undefined,
        deadlineAt: numeric(raw?.deadlineAt, 0),
        completedAt: typeof raw?.completedAt === 'number' ? raw.completedAt : undefined,
        createdAt: numeric(raw?.createdAt),
        updatedAt: numeric(raw?.updatedAt),
    };
}

function toChatRunEvent(raw: any): ChatRunEvent {
    return {
        id: stringValue(raw?.id),
        runId: stringValue(raw?.runId),
        phase: stringValue(raw?.phase),
        kind: stringValue(raw?.kind),
        label: stringValue(raw?.label),
        detail: raw?.detail ? String(raw.detail) : undefined,
        status: raw?.status ? String(raw.status) : undefined,
        payload: raw?.payload ? String(raw.payload) : undefined,
        latencyMs: typeof raw?.latencyMs === 'number' ? raw.latencyMs : undefined,
        createdAt: numeric(raw?.createdAt),
    };
}

function toChatWorkspaceArtifact(raw: any): ChatWorkspaceArtifact {
    return {
        key: stringValue(raw?.key),
        runId: stringValue(raw?.runId),
        narrativeId: stringValue(raw?.narrativeId),
        folderId: stringValue(raw?.folderId),
        kind: stringValue(raw?.kind),
        payload: raw?.payload ?? null,
        pinned: !!raw?.pinned,
        producedBy: stringValue(raw?.producedBy),
        createdAt: numeric(raw?.createdAt),
        updatedAt: numeric(raw?.updatedAt),
    };
}

function toChatPlannerToolCall(raw: any): ChatPlannerToolCall {
    return {
        id: stringValue(raw?.id),
        name: stringValue(raw?.name),
        argumentsJson: stringValue(raw?.argumentsJson, '{}'),
    };
}

function toChatPlannerMessage(raw: any): ChatPlannerMessage {
    return {
        role: stringValue(raw?.role),
        content: stringValue(raw?.content),
        name: raw?.name ? String(raw.name) : undefined,
        toolCallId: raw?.toolCallId ? String(raw.toolCallId) : undefined,
        toolCalls: Array.isArray(raw?.toolCalls) ? raw.toolCalls.map(toChatPlannerToolCall) : [],
    };
}

function toChatPlannerModelRequest(raw: any): ChatPlannerModelRequest {
    return {
        runId: stringValue(raw?.runId),
        threadId: stringValue(raw?.threadId),
        model: stringValue(raw?.model),
        allowTools: !!raw?.allowTools,
        tools: Array.isArray(raw?.tools)
            ? raw.tools.map((tool: any) => ({
                  name: stringValue(tool?.name),
                  description: stringValue(tool?.description),
                  parametersJson: tool?.parametersJson ?? null,
              }))
            : [],
        messages: Array.isArray(raw?.messages) ? raw.messages.map(toChatPlannerMessage) : [],
    };
}

function toChatPlannerStep(raw: any): ChatPlannerStep | null {
    const kind = stringValue(raw?.kind);
    switch (kind) {
        case 'modelRequest':
            return {
                kind,
                request: toChatPlannerModelRequest(raw?.request),
            };
        case 'toolCalls':
            return {
                kind,
                runId: stringValue(raw?.runId),
                toolCalls: Array.isArray(raw?.toolCalls)
                    ? raw.toolCalls.map(toChatPlannerToolCall)
                    : [],
            };
        case 'complete':
            return {
                kind,
                runId: stringValue(raw?.runId),
                response: stringValue(raw?.response),
            };
        default:
            return null;
    }
}

function toChatRunSnapshot(raw: any): ChatRunSnapshot {
    return {
        run: toChatRun(raw?.run),
        events: Array.isArray(raw?.events) ? raw.events.map(toChatRunEvent) : [],
        toolCalls: Array.isArray(raw?.toolCalls) ? raw.toolCalls : [],
        approvals: Array.isArray(raw?.approvals) ? raw.approvals : [],
        evidence: Array.isArray(raw?.evidence) ? raw.evidence : [],
        missingCapabilities: Array.isArray(raw?.missingCapabilities) ? raw.missingCapabilities : [],
        plannerStep: raw?.plannerStep ? toChatPlannerStep(raw.plannerStep) : null,
        artifacts: Array.isArray(raw?.artifacts) ? raw.artifacts.map(toChatWorkspaceArtifact) : [],
    };
}
