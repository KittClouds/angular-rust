export type WasmExports = {
    memory: WebAssembly.Memory;
    phoenix_alloc: (size: number) => number;
    phoenix_dealloc: (ptr: number, capacity: number) => void;
    phoenix_process_packet_at: (offset: number, capacity: number) => number;
    phoenix_packet_header_size: () => number;
    phoenix_wasm_protocol_version: () => number;
};

export type PhoenixChatStreamMessage = {
    role: 'system' | 'user' | 'assistant';
    content: string | null;
};

export type PhoenixChatStreamRequestOptions = {
    structuredOutput?: {
        enabled?: boolean;
        type?: 'json_schema' | 'json_object';
        schema?: unknown;
        strict?: boolean;
        name?: string;
        description?: string;
    };
    plugins?: Array<{ id: string }>;
};

export type PhoenixChatStreamConfig = {
    apiKey: string;
    model: string;
    temperature?: number;
    maxTokens?: number;
    reasoningEnabled?: boolean;
    reasoningEffort?: 'low' | 'medium' | 'high';
    reasoningMaxTokens?: number;
    includeReasoning?: boolean;
};

export type PhoenixChatStreamRequest = {
    config: PhoenixChatStreamConfig;
    messages: PhoenixChatStreamMessage[];
    systemPrompt?: string;
    requestOptions?: PhoenixChatStreamRequestOptions;
};

export type PhoenixChatStreamEvent =
    | { type: 'delta'; chunk: string }
    | { type: 'reasoning'; chunk: string }
    | { type: 'done'; response: string; reasoning?: string }
    | { type: 'error'; error: string }
    | { type: 'cancelled' };

export const PACKET_KIND = {
    status: 1,
    initRuntimeRequest: 2,
    createSessionRequest: 4,
    commitRequest: 6,
    rebuildRequest: 8,
    ingestRequest: 10,
    queryRequest: 12,
    snapshotExportRequest: 14,
    snapshotImportRequest: 16,
    scanRequest: 17,
    structureRequest: 19,
    graphDeltaRequest: 21,
    sessionStateRequest: 23,
    sessionStatsRequest: 25,
    analyzeTextRequest: 27,
    queryBinaryRequest: 29,
    storeCommandRequest: 34,
    embedUpsertBinaryRequest: 36,
} as const;

export const DEFAULT_PACKET_REGION_SIZE = 64 * 1024;
export const PACKET_HEADER_SIZE = 16;
export const PROTOCOL_VERSION = 6;
export const BINARY_REQUEST_LAYOUT_VERSION = 2;
export const REQUEST_FLAG_HAS_SESSION = 1 << 1;
export const REQUEST_FLAG_HAS_TEMPORAL = 1 << 2;
export const REQUEST_FLAG_TARGET_CHUNKS = 1 << 8;
export const REQUEST_FLAG_TARGET_NODES = 1 << 9;
export const REQUEST_FLAG_TARGET_GRAPH = 1 << 10;
export const REQUEST_FLAG_TARGET_SEMANTIC = 1 << 11;
export const REQUEST_FLAG_INCLUDE_CANDIDATE_GRAPH = 1 << 12;
export const PHOENIX_WASM_CANDIDATE_URLS = ['/assets/phoenix_wasm.wasm', '/assets/wasm/phoenix_wasm.wasm'];

export type PacketHeader = {
    ready: number;
    kind: number;
    requestId: number;
    payloadLen: number;
};

export function writePacketHeader(bytes: Uint8Array, kind: number, requestId: number, payloadLen: number): void {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    view.setUint32(0, 1, true);
    view.setUint32(4, kind, true);
    view.setUint32(8, requestId, true);
    view.setUint32(12, payloadLen, true);
}

export function decodePacketHeader(bytes: Uint8Array): PacketHeader {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    return {
        ready: view.getUint32(0, true),
        kind: view.getUint32(4, true),
        requestId: view.getUint32(8, true),
        payloadLen: view.getUint32(12, true),
    };
}

export function tryDecodeJson(bytes: Uint8Array): any {
    if (!bytes.byteLength) {
        return null;
    }
    try {
        return JSON.parse(new TextDecoder().decode(bytes));
    } catch {
        return null;
    }
}

export function isRetriablePacketMessage(message: string): boolean {
    return (
        message.includes('packet region') ||
        message.includes('buffer too small') ||
        message.includes('exceeds')
    );
}

export function createImportObject(): Record<string, Record<string, (...args: any[]) => any>> {
    return new Proxy(
        {},
        {
            get(_target, moduleName: string) {
                return new Proxy(
                    {},
                    {
                        get(_innerTarget, importName: string) {
                            if (moduleName === '__wbindgen_externref_xform__') {
                                if (importName === '__wbindgen_externref_table_grow') {
                                    return () => 0;
                                }
                                if (importName === '__wbindgen_externref_table_set_null') {
                                    return () => {};
                                }
                            }
                            return (..._args: any[]) => {
                                if (importName.includes('now')) {
                                    return Date.now();
                                }
                                return 0;
                            };
                        },
                    },
                );
            },
        },
    ) as Record<string, Record<string, (...args: any[]) => any>>;
}
