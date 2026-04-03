import { Injectable, inject } from '@angular/core';

import type { DecorationSpan, EntityKind } from '../lib/Scanner/types';
import {
    type PhoenixDiscoveryCandidate,
    groupDiscoveryMentions,
} from '../lib/phoenix/phoenix-discovery';
import { smartGraphRegistry } from '../lib/registry';
import { PhoenixBackendService } from './phoenix-backend.service';
import { PhoenixStoreService } from './phoenix-store.service';

export interface ProvenanceContext {
    vaultId?: string;
    worldId: string;
    parentPath?: string;
    folderType?: string;
}

export interface SearchScope {
    narrativeId?: string;
    folderPath?: string;
}

export interface KnowledgeGraphNode {
    id: string;
    kind: string;
    label: string;
    props?: Record<string, unknown>;
}

export interface KnowledgeGraphEdge {
    source: string;
    target: string;
    relation: string;
    weight: number;
    props?: Record<string, unknown>;
}

export interface KnowledgeGraphData {
    nodes: Record<string, KnowledgeGraphNode>;
    edges: KnowledgeGraphEdge[];
}

type NoteIngestInput = {
    id: string;
    title: string;
    text: string;
    narrativeId?: string;
    folderPath?: string;
    version?: number;
};

type DictionaryEntry = {
    id: string;
    label: string;
    kind: string;
    aliases: string[];
};

@Injectable({ providedIn: 'root' })
export class PhoenixUiApiService {
    private readonly phoenix = inject(PhoenixBackendService);
    private readonly store = inject(PhoenixStoreService);

    private ready = false;
    private readyPromise: Promise<void> | null = null;
    private readonly readyCallbacks = new Set<() => void>();

    private mainSessionId: string | null = null;
    private gldrSessionId: string | null = null;
    private dictionary: DictionaryEntry[] = [];
    private knowledgeGraph: KnowledgeGraphData = { nodes: {}, edges: [] };

    get isReady(): boolean {
        return this.ready;
    }

    onReady(callback: () => void): void {
        if (this.ready) {
            callback();
            return;
        }
        this.readyCallbacks.add(callback);
    }

    async loadRuntime(): Promise<void> {
        if (this.readyPromise) {
            return this.readyPromise;
        }

        this.readyPromise = this.initialize().catch((error) => {
            this.readyPromise = null;
            throw error;
        });
        return this.readyPromise;
    }

    async loadWasm(): Promise<void> {
        await this.loadRuntime();
    }

    async hydrateWithEntities(): Promise<void> {
        await this.loadRuntime();
        await this.hydrateWithEntitiesInternal();
    }

    private async hydrateWithEntitiesInternal(): Promise<void> {
        const entities = smartGraphRegistry.getAll().map((entity) => ({
            id: entity.id,
            label: entity.label,
            kind: entity.kind,
            aliases: entity.aliases || [],
        }));
        await this.rebuildDictionary(entities);
    }

    async rebuildDictionary(
        entities: Array<{ id: string; label: string; kind: string; aliases?: string[] }>,
    ): Promise<void> {
        this.dictionary = entities
            .map((entity) => ({
                id: entity.id,
                label: entity.label,
                kind: entity.kind,
                aliases: entity.aliases || [],
            }))
            .sort((left, right) => {
                const leftLen = Math.max(left.label.length, ...left.aliases.map((alias) => alias.length), 0);
                const rightLen = Math.max(right.label.length, ...right.aliases.map((alias) => alias.length), 0);
                return rightLen - leftLen || left.label.localeCompare(right.label);
            });
    }

    async hydrateNotes(
        notes: Array<{
            id: string;
            text: string;
            title?: string;
            version?: number;
            narrativeId?: string;
            folderPath?: string;
        }>,
    ): Promise<{ success: boolean; error?: string }> {
        await this.loadRuntime();
        const sessionId = await this.ensureMainSession();
        const documents = notes.map((note) => this.noteInputToDocument({
            id: note.id,
            title: note.title || note.id,
            text: note.text || '',
            narrativeId: note.narrativeId || undefined,
            folderPath: note.folderPath || undefined,
            version: note.version,
        }));

        if (!documents.length) {
            return { success: true };
        }

        await this.ingestDocumentsIntoSession(sessionId, documents);
        return { success: true };
    }

    async upsertNote(
        id: string,
        text: string,
        version?: number,
        meta?: { title?: string; narrativeId?: string; folderPath?: string },
    ): Promise<{ success: boolean; error?: string }> {
        await this.loadRuntime();
        const note = await this.resolveNoteIngestInput(id, text, version, meta);
        await this.ingestNoteInputs([note], await this.ensureMainSession());
        return { success: true };
    }

    async indexNote(id: string, text: string, scope?: SearchScope): Promise<void> {
        await this.loadRuntime();
        const note = await this.resolveNoteIngestInput(id, text, undefined, {
            narrativeId: scope?.narrativeId,
            folderPath: scope?.folderPath,
        });
        await this.ingestNoteInputs([note], await this.ensureMainSession());
    }

    async search(query: string, limit = 20): Promise<any[]> {
        return this.searchScoped(query, limit);
    }

    async searchScoped(query: string, limit = 20, scope?: SearchScope): Promise<any[]> {
        await this.loadRuntime();
        if (!query.trim()) {
            return [];
        }

        const sessionId = await this.ensureMainSession();
        const result = await this.phoenix.query({
            sessionId,
            query,
            scope: this.toPhoenixScope(scope),
            targets: ['chunks'],
            limit,
            temporal: null,
        });

        const byNote = new Map<string, { DocID: string; Score: number; ChunkID: string }>();
        for (const hit of result.chunkHits || []) {
            const noteId = chunkIdToDocumentId(hit.chunkId);
            const current = byNote.get(noteId);
            if (!current || hit.score > current.Score) {
                byNote.set(noteId, {
                    DocID: noteId,
                    Score: hit.score,
                    ChunkID: hit.chunkId,
                });
            }
        }

        return Array.from(byNote.values()).sort((left, right) => right.Score - left.Score).slice(0, limit);
    }

    async scan(text: string, provenance?: ProvenanceContext): Promise<any> {
        await this.loadRuntime();
        const scan = await this.phoenix.scan({
            text,
            scope: this.toPhoenixScope({ folderPath: provenance?.parentPath }),
            sessionId: 'phoenix-ui-scan',
            resolverSeed: [],
        });
        const structure = await this.phoenix.buildStructure({ text, scan });
        const graph = buildGraphFromScanResult(text, scan, structure);
        return {
            ...scan,
            structure,
            graph: {
                Nodes: graph.nodes,
                Edges: graph.edges,
            },
            timing_us: 0,
        };
    }

    async scanDiscovery(text: string): Promise<PhoenixDiscoveryCandidate[]> {
        await this.loadRuntime();
        const scan = await this.phoenix.scan({
            text,
            scope: {},
            sessionId: 'phoenix-ui-discovery',
            resolverSeed: [],
        });

        const mentions = Array.isArray(scan?.mentions) ? scan.mentions : [];
        return groupDiscoveryMentions(text, mentions);
    }

    async scanImplicitAsync(text: string): Promise<DecorationSpan[]> {
        await this.loadRuntime();
        if (!this.dictionary.length) {
            await this.hydrateWithEntities();
        }
        return matchDictionarySpans(text, this.dictionary);
    }

    async analyzeText(text: string): Promise<any> {
        await this.loadRuntime();
        return this.phoenix.analyzeText(text);
    }

    async knowledgeInit(): Promise<{ success: boolean; message?: string; error?: string }> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('knowledge-init');
        await this.refreshKnowledgeGraph();
        return { success: true, message: 'Phoenix knowledge graph ready' };
    }

    async knowledgeLoad(): Promise<{ success: boolean; message?: string; error?: string }> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('knowledge-load');
        await this.refreshKnowledgeGraph();
        return { success: true, message: 'Phoenix knowledge graph loaded' };
    }

    async knowledgeSync(): Promise<{ success: boolean; message?: string; error?: string }> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('knowledge-sync');
        await this.refreshKnowledgeGraph();
        return { success: true, message: 'Phoenix knowledge graph synced' };
    }

    async knowledgeSave(): Promise<{ success: boolean; message?: string; error?: string }> {
        await this.loadRuntime();
        return { success: true, message: 'Phoenix graph is already persisted' };
    }

    async knowledgeAddNode(node: {
        id: string;
        kind: string;
        label?: string;
        props?: Record<string, unknown>;
    }): Promise<{ success: boolean; message?: string; error?: string }> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('knowledge-add-node');
        await this.phoenix.storeCommand('relation:upsert', {
            relation: 'graph_vertices',
            row: {
                id: node.id,
                value: {
                    id: node.id,
                    kind: node.kind,
                    label: node.label || node.id,
                },
                weight: 1,
                attributes: node.props || {},
            },
        });
        await this.phoenix.storeCommand('relation:upsert', {
            relation: 'graph_vertex_labels',
            row: {
                vertex_id: node.id,
                label: node.label || node.id,
            },
        });
        this.store.markDerivedDirty();
        await this.refreshKnowledgeGraph();
        return { success: true };
    }

    async knowledgeAddEdge(edge: {
        source: string;
        target: string;
        relation: string;
        weight?: number;
        props?: Record<string, unknown>;
    }): Promise<{ success: boolean; message?: string; error?: string }> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('knowledge-add-edge');
        await this.phoenix.storeCommand('relation:upsert', {
            relation: 'graph_edges',
            row: {
                source_id: edge.source,
                target_id: edge.target,
                weight: edge.weight ?? 1,
                attributes: edge.props || {},
                data: null,
                edge_type: edge.relation,
            },
        });
        this.store.markDerivedDirty();
        await this.refreshKnowledgeGraph();
        return { success: true };
    }

    async knowledgeGetNode(id: string): Promise<any> {
        await this.ensureKnowledgeGraph();
        return this.knowledgeGraph.nodes[id] || null;
    }

    async knowledgeGetChildren(id: string, relation?: string): Promise<any[]> {
        await this.ensureKnowledgeGraph();
        return this.knowledgeGraph.edges
            .filter((edge) => edge.source === id && (!relation || edge.relation === relation))
            .map((edge) => this.knowledgeGraph.nodes[edge.target])
            .filter(Boolean);
    }

    async knowledgeGetParents(id: string, relation?: string): Promise<any[]> {
        await this.ensureKnowledgeGraph();
        return this.knowledgeGraph.edges
            .filter((edge) => edge.target === id && (!relation || edge.relation === relation))
            .map((edge) => this.knowledgeGraph.nodes[edge.source])
            .filter(Boolean);
    }

    async knowledgeGetAncestors(id: string, relation?: string, maxDepth = -1): Promise<any[]> {
        return this.walkKnowledgeGraph(id, 'parents', relation, maxDepth);
    }

    async knowledgeGetDescendants(id: string, relation?: string, maxDepth = -1): Promise<any[]> {
        return this.walkKnowledgeGraph(id, 'children', relation, maxDepth);
    }

    async knowledgeGetNeighborhood(id: string): Promise<any[]> {
        await this.ensureKnowledgeGraph();
        const neighbors = new Map<string, KnowledgeGraphNode>();
        for (const edge of this.knowledgeGraph.edges) {
            if (edge.source === id && this.knowledgeGraph.nodes[edge.target]) {
                neighbors.set(edge.target, this.knowledgeGraph.nodes[edge.target]);
            }
            if (edge.target === id && this.knowledgeGraph.nodes[edge.source]) {
                neighbors.set(edge.source, this.knowledgeGraph.nodes[edge.source]);
            }
        }
        return Array.from(neighbors.values());
    }

    async knowledgeGetGraph(): Promise<KnowledgeGraphData> {
        await this.ensureKnowledgeGraph();
        return this.knowledgeGraph;
    }

    async gldrInit(): Promise<{ success: boolean; error?: string }> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('gldr-init');
        const session = await this.phoenix.createSession('phoenix-ui-gldr', {});
        this.gldrSessionId = String(session?.sessionId || '');
        return { success: true };
    }

    async gldrRegisterEntity(_name: string, _entityId: string): Promise<{ success: boolean; error?: string }> {
        await this.loadRuntime();
        return { success: true };
    }

    async gldrIndexChunk(
        chunkId: string,
        fields: Record<string, string>,
        _mentions: Array<{ entityId: string; count: number }>,
    ): Promise<{ success: boolean; error?: string }> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('gldr-index-chunk');
        const sessionId = await this.ensureGldrSession();
        await this.ingestDocumentsIntoSession(sessionId, [{
            documentId: chunkId,
            noteId: chunkId,
            title: fields['title'] || chunkId,
            text: fields['content'] || '',
            scope: {},
        }]);
        return { success: true };
    }

    async gldrIndexChunksWithEmbeddings(
        items: Array<{
            chunkId: string;
            fields: Record<string, string>;
            mentions: Array<{ entityId: string; count: number }>;
            embedding: Float32Array;
        }>,
    ): Promise<{ success: boolean; error?: string; count?: number; dim?: number }> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('gldr-index-embeddings');
        const sessionId = await this.ensureGldrSession();
        await this.ingestDocumentsIntoSession(
            sessionId,
            items.map((item) => ({
                documentId: item.chunkId,
                noteId: item.chunkId,
                title: item.fields['title'] || item.chunkId,
                text: item.fields['content'] || '',
                scope: {},
            })),
        );
        return {
            success: true,
            count: items.length,
            dim: items[0]?.embedding?.length || 0,
        };
    }

    async gldrAddGraphEdge(
        _sourceId: string,
        _edge: { targetId: string; relType: string; confidence?: number; source?: string },
    ): Promise<{ success: boolean; error?: string }> {
        await this.loadRuntime();
        return { success: true };
    }

    async gldrLoadCooccurrences(_minCount = 2): Promise<{ success: boolean; edgesLoaded?: number; error?: string }> {
        await this.loadRuntime();
        return { success: true, edgesLoaded: 0 };
    }

    async gldrSearch(query: string, config: Record<string, unknown> = {}): Promise<string> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('gldr-search');
        const sessionId = await this.ensureGldrSession();
        const limit = typeof config['topChunks'] === 'number' ? Number(config['topChunks']) : 12;
        const result = await this.phoenix.query({
            sessionId,
            query,
            scope: {},
            targets: ['chunks'],
            limit,
            temporal: null,
        });
        return JSON.stringify(
            (result.chunkHits || []).map((hit) => ({
                chunkId: hit.chunkId,
                chunkScore: hit.score,
                lexScore: hit.score,
                graphScore: 0,
                matchedEntities: [],
            })),
        );
    }

    async gldrSearchWithEmbedding(
        query: string,
        embedding: Float32Array,
        config: Record<string, unknown> = {},
    ): Promise<string> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('gldr-search-embedding');
        const sessionId = await this.ensureGldrSession();
        const limit = typeof config['topChunks'] === 'number' ? Number(config['topChunks']) : 12;
        const result = await this.phoenix.query({
            sessionId,
            query,
            scope: {},
            targets: ['chunks', 'semantic'],
            limit,
            temporal: null,
            semanticQueryVector: embedding,
        });
        return JSON.stringify(
            (result.chunkHits || []).map((hit) => ({
                chunkId: hit.chunkId,
                chunkScore: hit.score,
                lexScore: hit.score,
                graphScore: 0,
                matchedEntities: [],
            })),
        );
    }

    async gldrSearchNodes(query: string, config: Record<string, unknown> = {}): Promise<string> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('gldr-search-nodes');
        const sessionId = await this.ensureGldrSession();
        const limit = typeof config['topChunks'] === 'number' ? Number(config['topChunks']) : 12;
        const result = await this.phoenix.query({
            sessionId,
            query,
            scope: {},
            targets: ['nodes'],
            limit,
            temporal: null,
        });
        return JSON.stringify(
            (result.nodeHits || []).map((hit) => ({
                entityId: hit.entityId,
                nodeScore: hit.score,
                topChunks: [],
                proximityFromQuery: hit.score,
            })),
        );
    }

    async gldrSearchNodesWithEmbedding(
        query: string,
        embedding: Float32Array,
        config: Record<string, unknown> = {},
    ): Promise<string> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('gldr-search-nodes-embedding');
        const sessionId = await this.ensureGldrSession();
        const limit = typeof config['topChunks'] === 'number' ? Number(config['topChunks']) : 12;
        const result = await this.phoenix.query({
            sessionId,
            query,
            scope: {},
            targets: ['nodes', 'semantic'],
            limit,
            temporal: null,
            semanticQueryVector: embedding,
        });
        return JSON.stringify(
            (result.nodeHits || []).map((hit) => ({
                entityId: hit.entityId,
                nodeScore: hit.score,
                topChunks: [],
                proximityFromQuery: hit.score,
            })),
        );
    }

    async gldrStats(): Promise<string> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('gldr-stats');
        if (!this.gldrSessionId) {
            return JSON.stringify({ entities: 0, chunks: 0, edges: 0 });
        }
        const stats = await this.phoenix.sessionStats(this.gldrSessionId);
        return JSON.stringify({
            entities: stats.entityCount,
            chunks: stats.leafCount,
            edges: stats.graphEdgeCount,
        });
    }

    async systemCreateSession(config: Record<string, unknown> = {}): Promise<{ sessionId: string }> {
        await this.loadRuntime();
        const label = typeof config['label'] === 'string' ? String(config['label']) : 'phoenix-ui-session';
        const scope = this.toPhoenixScope(config['scope'] as SearchScope | undefined);
        const response = await this.phoenix.createSession(label, scope);
        return { sessionId: String(response?.sessionId || '') };
    }

    async systemIngest<T = any>(sessionId: string, request: Record<string, unknown>): Promise<T> {
        await this.loadRuntime();
        await this.store.ensureDerivedLoaded('system-ingest');
        const result = await (this.phoenix.ingest({ sessionId, ...request }) as Promise<T>);
        this.store.markDerivedDirty();
        return result;
    }

    async systemSearch<T = any>(sessionId: string, request: Record<string, unknown>): Promise<T> {
        await this.loadRuntime();
        return this.phoenix.query({ sessionId, ...request }) as Promise<T>;
    }

    async systemCommit<T = any>(sessionId: string, request: Record<string, unknown> = {}): Promise<T> {
        await this.loadRuntime();
        return this.phoenix.commit(sessionId, request) as Promise<T>;
    }

    async systemGetState<T = any>(sessionId: string): Promise<T> {
        await this.loadRuntime();
        return this.phoenix.sessionState(sessionId) as Promise<T>;
    }

    async systemGetStats<T = any>(sessionId: string): Promise<T> {
        await this.loadRuntime();
        return this.phoenix.sessionStats(sessionId) as Promise<T>;
    }

    async systemClose(sessionId: string): Promise<{ success: boolean; error?: string }> {
        await this.loadRuntime();
        await this.phoenix.storeCommand('session:close', { sessionId });
        return { success: true };
    }

    async systemRun<T = any>(request: Record<string, unknown>): Promise<T> {
        await this.loadRuntime();
        const created = typeof request['sessionId'] === 'string' ? null : await this.systemCreateSession({});
        const sessionId = typeof request['sessionId'] === 'string' ? String(request['sessionId']) : created!.sessionId;
        const result: Record<string, unknown> = { sessionId };

        try {
            if (request['ingest']) {
                result['ingest'] = await this.systemIngest(sessionId, request['ingest'] as Record<string, unknown>);
            }
            if (request['search']) {
                result['search'] = await this.systemSearch(sessionId, request['search'] as Record<string, unknown>);
            }
            if (request['commit']) {
                result['commit'] = await this.systemCommit(sessionId, request['commit'] as Record<string, unknown>);
            }
            if (request['state']) {
                result['state'] = await this.systemGetState(sessionId);
            }
            if (request['stats']) {
                result['stats'] = await this.systemGetStats(sessionId);
            }
        } finally {
            if (created) {
                try {
                    await this.systemClose(created.sessionId);
                } catch (error) {
                    console.warn('[PhoenixUiApi] Failed to close disposable Phoenix session:', error);
                }
            }
        }

        return result as T;
    }

    private async initialize(): Promise<void> {
        const startedAt = Date.now();
        console.log('[PhoenixUiApi] initialize:start');
        console.log('[PhoenixUiApi] initialize:store.initialize:start');
        await this.store.initialize();
        console.log(`[PhoenixUiApi] initialize:store.initialize:complete (${Date.now() - startedAt}ms)`);
        console.log('[PhoenixUiApi] initialize:ensureMainSession:start');
        await this.ensureMainSession();
        console.log(`[PhoenixUiApi] initialize:ensureMainSession:complete (${Date.now() - startedAt}ms)`);
        if (!this.dictionary.length) {
            console.log('[PhoenixUiApi] initialize:hydrateWithEntities:start');
            await this.hydrateWithEntitiesInternal();
            console.log(`[PhoenixUiApi] initialize:hydrateWithEntities:complete (${Date.now() - startedAt}ms)`);
        }
        this.ready = true;
        const readyCallbacks = Array.from(this.readyCallbacks);
        this.readyCallbacks.clear();
        queueMicrotask(() => {
            if (typeof window !== 'undefined') {
                window.dispatchEvent(new CustomEvent('phoenix-ready'));
                window.dispatchEvent(new CustomEvent('gokitt-ready'));
            }
            for (const callback of readyCallbacks) {
                try {
                    callback();
                } catch (error) {
                    console.error('[PhoenixUiApi] Ready callback failed:', error);
                }
            }
        });
        console.log(`[PhoenixUiApi] initialize:complete (${Date.now() - startedAt}ms)`);
    }

    private async ensureMainSession(): Promise<string> {
        if (this.mainSessionId) {
            return this.mainSessionId;
        }
        const response = await this.phoenix.createSession('phoenix-ui-main', {});
        this.mainSessionId = String(response?.sessionId || '');
        return this.mainSessionId;
    }

    private async ensureGldrSession(): Promise<string> {
        if (this.gldrSessionId) {
            return this.gldrSessionId;
        }
        const response = await this.phoenix.createSession('phoenix-ui-gldr', {});
        this.gldrSessionId = String(response?.sessionId || '');
        return this.gldrSessionId;
    }

    private async ingestNoteInputs(notes: NoteIngestInput[], sessionId: string): Promise<void> {
        await this.ingestDocumentsIntoSession(
            sessionId,
            notes.map((note) => this.noteInputToDocument(note)),
        );
    }

    private async ingestDocumentsIntoSession(
        sessionId: string,
        documents: Array<{
            documentId: string;
            noteId: string;
            title: string;
            text: string;
            scope: Record<string, unknown>;
        }>,
    ): Promise<void> {
        if (!documents.length) {
            return;
        }
        await this.store.ensureDerivedLoaded('session-ingest');
        await this.phoenix.ingest({
            sessionId,
            documents,
            commit: false,
        });
        this.store.markDerivedDirty();
    }

    private noteInputToDocument(note: NoteIngestInput): {
        documentId: string;
        noteId: string;
        title: string;
        text: string;
        scope: Record<string, unknown>;
    } {
        return {
            documentId: note.id,
            noteId: note.id,
            title: note.title || note.id,
            text: note.text || '',
            scope: this.toPhoenixScope({
                narrativeId: note.narrativeId,
                folderPath: note.folderPath,
            }),
        };
    }

    private async resolveNoteIngestInput(
        id: string,
        text: string,
        version?: number,
        meta?: { title?: string; narrativeId?: string; folderPath?: string },
    ): Promise<NoteIngestInput> {
        if (meta?.title && (meta?.narrativeId !== undefined || meta?.folderPath !== undefined)) {
            return {
                id,
                title: meta.title || id,
                text,
                narrativeId: meta.narrativeId || undefined,
                folderPath: meta.folderPath || undefined,
                version,
            };
        }

        const header = await this.store.getNoteHeader(id);
        return {
            id,
            title: meta?.title || header?.title || id,
            text,
            narrativeId: meta?.narrativeId ?? header?.narrativeId ?? undefined,
            folderPath: meta?.folderPath ?? header?.folderId ?? undefined,
            version,
        };
    }

    private toPhoenixScope(scope?: SearchScope | Record<string, unknown>): Record<string, unknown> {
        if (!scope) {
            return {};
        }
        const scopeRecord = scope as Record<string, unknown>;
        return {
            narrativeId: typeof scopeRecord['narrativeId'] === 'string' ? scopeRecord['narrativeId'] : undefined,
            folderPath: typeof scopeRecord['folderPath'] === 'string' ? scopeRecord['folderPath'] : undefined,
            worldId: typeof scopeRecord['worldId'] === 'string' ? scopeRecord['worldId'] : undefined,
            folderId: typeof scopeRecord['folderId'] === 'string' ? scopeRecord['folderId'] : undefined,
        };
    }

    private async ensureKnowledgeGraph(): Promise<void> {
        if (Object.keys(this.knowledgeGraph.nodes).length || this.knowledgeGraph.edges.length) {
            return;
        }
        await this.store.ensureDerivedLoaded('knowledge-graph-cache');
        await this.refreshKnowledgeGraph();
    }

    private async refreshKnowledgeGraph(): Promise<void> {
        const vertices = await this.phoenix.storeCommand('relation:list', { relation: 'graph_vertices' });
        const graphNodes: Record<string, KnowledgeGraphNode> = {};
        for (const row of Array.isArray(vertices) ? vertices : []) {
            const value = asRecord((row as Record<string, unknown>)?.['value']);
            const attributes = asRecord((row as Record<string, unknown>)?.['attributes']);
            const id = String((row as Record<string, unknown>)?.['id'] || value['id'] || '');
            if (!id) {
                continue;
            }
            graphNodes[id] = {
                id,
                kind: String(value['kind'] || 'unknown'),
                label: String(value['label'] || value['entityId'] || id),
                props: attributes,
            };
        }

        const edgeRows = await this.phoenix.storeCommand('relation:list', { relation: 'graph_edges' });
        const graphEdges: KnowledgeGraphEdge[] = (Array.isArray(edgeRows) ? edgeRows : [])
            .map((row: any) => ({
                source: String(row?.source_id || ''),
                target: String(row?.target_id || ''),
                relation: String(row?.edge_type || 'edge'),
                weight: Number(row?.weight || 0),
                props: asRecord(row?.attributes),
            }))
            .filter((edge) => edge.source && edge.target);

        if (!Object.keys(graphNodes).length) {
            const entityRows = await this.store.listEntities();
            for (const entity of entityRows) {
                graphNodes[entity.id] = {
                    id: entity.id,
                    kind: entity.kind,
                    label: entity.label,
                    props: { narrativeId: entity.narrativeId || '' },
                };
            }
        }

        if (!graphEdges.length) {
            const edgeRowsFallback = await this.store.listAllEdges();
            for (const edge of edgeRowsFallback) {
                graphEdges.push({
                    source: edge.sourceId,
                    target: edge.targetId,
                    relation: edge.relType,
                    weight: edge.confidence,
                    props: {},
                });
            }
        }

        this.knowledgeGraph = {
            nodes: graphNodes,
            edges: graphEdges,
        };
    }

    private async walkKnowledgeGraph(
        startId: string,
        direction: 'children' | 'parents',
        relation?: string,
        maxDepth = -1,
    ): Promise<any[]> {
        await this.ensureKnowledgeGraph();
        const results = new Map<string, KnowledgeGraphNode>();
        const queue: Array<{ id: string; depth: number }> = [{ id: startId, depth: 0 }];
        const seen = new Set<string>([startId]);

        while (queue.length) {
            const current = queue.shift();
            if (!current) {
                continue;
            }

            const matches = this.knowledgeGraph.edges.filter((edge) => {
                if (relation && edge.relation !== relation) {
                    return false;
                }
                return direction === 'children' ? edge.source === current.id : edge.target === current.id;
            });

            for (const edge of matches) {
                const nextId = direction === 'children' ? edge.target : edge.source;
                if (seen.has(nextId)) {
                    continue;
                }
                seen.add(nextId);
                const node = this.knowledgeGraph.nodes[nextId];
                if (node) {
                    results.set(nextId, node);
                }
                if (maxDepth < 0 || current.depth + 1 < maxDepth) {
                    queue.push({ id: nextId, depth: current.depth + 1 });
                }
            }
        }

        return Array.from(results.values());
    }
}

function buildGraphFromScanResult(text: string, scan: any, structure: any): {
    nodes: Record<string, { id: string; label: string; kind: string }>;
    edges: Array<{ Source: string; Target: string; Relation: string }>;
} {
    const nodes: Record<string, { id: string; label: string; kind: string }> = {};
    const mentionByKey = new Map<string, { id: string; label: string; kind: string }>();
    const mentions = Array.isArray(scan?.mentions) ? scan.mentions : [];

    for (const mention of mentions) {
        const key = mentionKey(mention, text);
        if (!key) {
            continue;
        }
        const label = String(mention?.surface || sliceRange(text, mention?.range) || key);
        const kind = String(mention?.kind || 'UNKNOWN');
        const node = { id: key, label, kind };
        nodes[key] = node;
        mentionByKey.set(key, node);
    }

    const edges: Array<{ Source: string; Target: string; Relation: string }> = [];
    const relations = Array.isArray(structure?.relations) ? structure.relations : [];
    for (const relation of relations) {
        const source = frameSlotToNodeId(relation?.subject, mentionByKey, text);
        const target = frameSlotToNodeId(relation?.object || relation?.recipient, mentionByKey, text);
        if (!source || !target) {
            continue;
        }
        edges.push({
            Source: source,
            Target: target,
            Relation: String(relation?.relationType || relation?.lemma || 'RELATED_TO'),
        });
    }

    return { nodes, edges };
}

function frameSlotToNodeId(
    slot: any,
    mentionByKey: Map<string, { id: string; label: string; kind: string }>,
    text: string,
): string | null {
    if (!slot) {
        return null;
    }
    const entityRef = slot?.entityRef;
    if (typeof entityRef === 'string' && mentionByKey.has(entityRef)) {
        return entityRef;
    }
    if (entityRef && typeof entityRef === 'object') {
        if (typeof entityRef.Known === 'string') return entityRef.Known;
        if (typeof entityRef.known === 'string') return entityRef.known;
        if (typeof entityRef.Speculative === 'string') return entityRef.Speculative;
        if (typeof entityRef.speculative === 'string') return entityRef.speculative;
    }
    const label = sliceRange(text, slot?.range);
    return label || null;
}

function mentionKey(mention: any, text: string): string | null {
    const entityRef = mention?.entityRef;
    if (entityRef && typeof entityRef === 'object') {
        if (typeof entityRef.Known === 'string') return entityRef.Known;
        if (typeof entityRef.known === 'string') return entityRef.known;
        if (typeof entityRef.Speculative === 'string') return entityRef.Speculative;
        if (typeof entityRef.speculative === 'string') return entityRef.speculative;
    }
    return String(mention?.surface || sliceRange(text, mention?.range) || '').trim() || null;
}

function matchDictionarySpans(text: string, dictionary: DictionaryEntry[]): DecorationSpan[] {
    const spans: DecorationSpan[] = [];
    const claimed: Array<{ from: number; to: number }> = [];
    const lower = text.toLowerCase();

    for (const entry of dictionary) {
        const surfaces = [entry.label, ...entry.aliases].filter(Boolean);
        for (const surface of surfaces) {
            const needle = surface.toLowerCase();
            if (!needle) {
                continue;
            }

            let from = 0;
            while (from < lower.length) {
                const index = lower.indexOf(needle, from);
                if (index < 0) {
                    break;
                }
                const end = index + needle.length;
                from = index + 1;

                if (!isWordBoundary(text, index, end)) {
                    continue;
                }
                if (claimed.some((span) => rangesOverlap(span.from, span.to, index, end))) {
                    continue;
                }

                claimed.push({ from: index, to: end });
                spans.push({
                    type: 'entity_implicit',
                    from: index,
                    to: end,
                    label: entry.label,
                    kind: entry.kind as EntityKind,
                    target: entry.label,
                    matchedText: text.slice(index, end),
                    entityId: entry.id,
                    resolved: true,
                });
            }
        }
    }

    return spans.sort((left, right) => left.from - right.from || left.to - right.to);
}

function isWordBoundary(text: string, from: number, to: number): boolean {
    const before = from > 0 ? text[from - 1] : '';
    const after = to < text.length ? text[to] : '';
    return !isWordChar(before) && !isWordChar(after);
}

function isWordChar(char: string): boolean {
    return !!char && /[\p{L}\p{N}_]/u.test(char);
}

function rangesOverlap(leftFrom: number, leftTo: number, rightFrom: number, rightTo: number): boolean {
    return leftFrom < rightTo && rightFrom < leftTo;
}

function chunkIdToDocumentId(chunkId: string): string {
    const separator = chunkId.indexOf(':');
    return separator >= 0 ? chunkId.slice(0, separator) : chunkId;
}

function sliceRange(text: string, range: any): string {
    const start = Number(range?.start ?? range?.from ?? 0);
    const end = Number(range?.end ?? range?.to ?? 0);
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
        return '';
    }
    return text.slice(start, end);
}

function asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === 'object' && !Array.isArray(value)
        ? value as Record<string, unknown>
        : {};
}
