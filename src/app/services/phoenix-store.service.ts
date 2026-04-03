import { Injectable, inject } from '@angular/core';

import {
    SqlitePersistenceService,
} from '../lib/sqlite/persistence/SqlitePersistenceService';
import {
    type LoadedPhoenixManifestState,
    type PersistenceManifest,
    type PhoenixWalBatch,
    type RecoveryState,
    PHOENIX_DERIVED_CHECKPOINT_MS,
    PHOENIX_WAL_IDLE_CHECKPOINT_MS,
    collectManifestFiles,
    createEmptyPhoenixManifest,
    decodeWalBytes,
    finalizeContentCheckpointManifest,
    finalizeDerivedCheckpointManifest,
    nextManifestWithWalAppend,
    shouldCheckpointContent,
} from '../lib/sqlite/persistence/phoenix-wal';
import {
    assertPhoenixRuntimeCapabilities,
    isPhoenixWasmMismatchError,
    normalizePhoenixRuntimeCompatibilityError,
} from '../lib/phoenix/phoenix-runtime-compat';
import { PhoenixBackendService } from './phoenix-backend.service';
import { PhoenixSnapshotPartition } from './phoenix-wasm.service';

export interface StoreNote {
    id: string;
    worldId: string;
    title: string;
    content: string;
    markdownContent: string;
    folderId: string;
    entityKind: string;
    entitySubtype: string;
    isEntity: boolean;
    isPinned: boolean;
    favorite: boolean;
    ownerId: string;
    narrativeId: string;
    order: number;
    createdAt: number;
    updatedAt: number;
    version?: number;
}

export interface StoreNoteHeader {
    id: string;
    worldId: string;
    title: string;
    folderId: string;
    entityKind: string;
    entitySubtype: string;
    isEntity: boolean;
    isPinned: boolean;
    favorite: boolean;
    ownerId: string;
    narrativeId: string;
    order: number;
    createdAt: number;
    updatedAt: number;
    version?: number;
}

export interface StoreEntity {
    id: string;
    label: string;
    kind: string;
    subtype?: string;
    aliases: string[];
    firstNote: string;
    totalMentions: number;
    narrativeId?: string;
    createdBy: 'user' | 'extraction' | 'auto';
    createdAt: number;
    updatedAt: number;
}

export interface StoreEdge {
    id: string;
    sourceId: string;
    targetId: string;
    relType: string;
    confidence: number;
    bidirectional: boolean;
    sourceNote?: string;
    createdAt: number;
}

export interface StoreFolder {
    id: string;
    name: string;
    parentId?: string;
    worldId: string;
    narrativeId?: string;
    folderOrder: number;
    entityKind: string;
    entitySubtype: string;
    entityLabel: string;
    color: string;
    isTypedRoot: boolean;
    isSubtypeRoot: boolean;
    collapsed: boolean;
    ownerId: string;
    isNarrativeRoot: boolean;
    attributes?: string;
    createdAt: number;
    updatedAt: number;
}

export interface StoreScopedDocument {
    id: string;
    scopeFolderId: string;
    narrativeId: string;
    namespace: string;
    documentKey: string;
    payload: string;
    seededFromScopeFolderId?: string;
    createdAt: number;
    updatedAt: number;
}

export interface StoreScopedEntityField {
    id: string;
    entityId: string;
    scopeFolderId: string;
    narrativeId: string;
    fieldKey: string;
    valueJson: string;
    seededFromScopeFolderId?: string;
    createdAt: number;
    updatedAt: number;
}

export interface StoreScopedDefinition {
    id: string;
    narrativeId: string;
    namespace: string;
    definitionKey: string;
    payload: string;
    createdAt: number;
    updatedAt: number;
}

export interface StoreDiscoveryCandidate {
    token: string;
    kind: number;
    score: number;
    status: number;
    lastSeen: number;
    firstSeen: number;
    count: number;
}

export interface StoreEntityCard {
    entityId: string;
    cardId: string;
    name: string;
    color: string;
    icon: string;
    displayOrder: number;
    isCollapsed: boolean;
    createdAt: number;
    updatedAt: number;
}

export interface StoreFolderSchema {
    id: string;
    entityKind: string;
    subtype?: string;
    name: string;
    description?: string;
    allowedSubfolders: unknown[];
    allowedNoteTypes: unknown[];
    isVaultRoot: boolean;
    containerOnly: boolean;
    propagateKindToChildren: boolean;
    icon?: string;
    isSystem: boolean;
    createdAt: number;
    updatedAt: number;
}

export interface StoreNetworkInstance {
    id: string;
    schemaId: string;
    name: string;
    rootFolderId: string;
    rootEntityId?: string;
    entityIds: string[];
    narrativeId: string;
    description?: string;
    createdAt: number;
    updatedAt: number;
}

export interface StoreNetworkMembership {
    networkId: string;
    entityId: string;
    x: number;
    y: number;
    fixed: boolean;
}

export interface StoreNetworkRelationship {
    id: string;
    networkId: string;
    sourceEntityId: string;
    targetEntityId: string;
    relationshipCode: string;
    strength?: number;
    startDate?: number;
    endDate?: number;
    notes?: string;
    createdAt: number;
    updatedAt: number;
}

export function hasActivePhoenixPersistence(state: LoadedPhoenixManifestState): boolean {
    return Boolean(
        state.manifest ||
        state.manifestBytes > 0 ||
        state.backupManifestBytes > 0 ||
        state.contentCheckpoint?.bytes ||
        state.derivedCheckpoint?.bytes ||
        state.closedSegments.some((segment) => segment.bytes > 0) ||
        state.activeSegment?.bytes,
    );
}

export function formatPhoenixPersistenceSummary(state: LoadedPhoenixManifestState): string {
    const closedWalBytes = state.closedSegments.reduce((sum, segment) => sum + segment.bytes, 0);
    return [
        `manifest=${state.manifest ? 'yes' : 'no'}`,
        `manifestBytes=${state.manifestBytes}`,
        `backupManifestBytes=${state.backupManifestBytes}`,
        `contentCheckpointBytes=${state.contentCheckpoint?.bytes || 0}`,
        `derivedCheckpointBytes=${state.derivedCheckpoint?.bytes || 0}`,
        `closedWalBytes=${closedWalBytes}`,
        `activeWalBytes=${state.activeSegment?.bytes || 0}`,
        `legacyArtifacts=${state.staleLegacyFiles.length}`,
    ].join(', ');
}

interface PhoenixCheckpointBundle {
    contentSnapshot: Uint8Array;
    derivedSnapshot: Uint8Array;
}

interface ContentWalMutation {
    command: string;
    payload: Record<string, unknown>;
}

type DerivedLoadState = 'cold' | 'loading' | 'ready';

@Injectable({ providedIn: 'root' })
export class PhoenixStoreService {
    private readonly phoenix = inject(PhoenixBackendService);
    private readonly persistence = inject(SqlitePersistenceService);

    private initialized = false;
    private initPromise: Promise<void> | null = null;
    private mutationChain: Promise<void> = Promise.resolve();
    private contentCheckpointTimeout: ReturnType<typeof setTimeout> | null = null;
    private derivedCheckpointTimeout: ReturnType<typeof setTimeout> | null = null;
    private snapshotsPaused = false;
    private manifest: PersistenceManifest | null = null;
    private manifestMeta: LoadedPhoenixManifestState | null = null;
    private derivedDirty = false;
    private derivedLoadState: DerivedLoadState = 'cold';
    private derivedLoadPromise: Promise<void> | null = null;
    private recoveryState: RecoveryState = {
        contentRecovered: false,
        derivedRecovered: false,
        replayedRecords: 0,
        lastRecoveredSeq: 0,
        manifestGeneration: 0,
    };

    async initialize(): Promise<void> {
        if (this.initialized) return;
        if (this.initPromise) return this.initPromise;

        this.initPromise = this.initializeInternal().catch((error) => {
            this.initPromise = null;
            throw error;
        });
        return this.initPromise;
    }

    async tryInitialize(): Promise<boolean> {
        if (this.initialized) return true;
        if (!this.canInitialize) return false;

        try {
            await this.initialize();
            return this.initialized;
        } catch {
            return false;
        }
    }

    get isReady(): boolean {
        return this.initialized;
    }

    get canInitialize(): boolean {
        return this.phoenix.isReady;
    }

    get isDerivedReady(): boolean {
        return this.derivedLoadState === 'ready';
    }

    pauseSnapshots(): void {
        this.snapshotsPaused = true;
        this.clearCheckpointTimers();
    }

    resumeSnapshots(): void {
        this.snapshotsPaused = false;
        if (this.initialized) {
            this.scheduleContentCheckpoint();
            if (this.derivedDirty && this.isDerivedReady) {
                this.scheduleDerivedCheckpoint();
            }
        }
    }

    async triggerSnapshot(): Promise<void> {
        if (!this.initialized || this.snapshotsPaused) {
            return;
        }
        await this.runSerialized(async () => {
            await this.flushContentCheckpoint(true);
            if (this.derivedDirty && this.isDerivedReady) {
                await this.flushDerivedCheckpoint(true);
            }
        });
    }

    markDerivedDirty(): void {
        if (!this.isDerivedReady) {
            console.warn('[PhoenixStoreService] Ignoring derived dirty mark while the derived partition is still cold.');
            return;
        }
        this.derivedDirty = true;
        if (this.snapshotsPaused || !this.initialized) {
            return;
        }
        this.scheduleDerivedCheckpoint();
    }

    async ensureDerivedLoaded(reason = 'unknown'): Promise<void> {
        await this.ensureInitialized();
        if (this.derivedLoadState === 'ready') {
            return;
        }
        if (this.derivedLoadPromise) {
            return this.derivedLoadPromise;
        }

        this.derivedLoadPromise = this.runSerialized(() => this.loadDerivedCheckpointIntoRuntime(reason)).finally(() => {
            this.derivedLoadPromise = null;
        });

        return this.derivedLoadPromise;
    }

    private async loadDerivedCheckpointIntoRuntime(reason: string): Promise<void> {
        if (this.derivedLoadState === 'ready') {
            return;
        }

        const manifest = this.requireManifest();
        const checkpointMeta = manifest.derived.checkpointFile
            ? this.manifestMeta?.derivedCheckpoint || {
                file: manifest.derived.checkpointFile,
                bytes: 0,
            }
            : null;

        if (!checkpointMeta?.file) {
            this.derivedLoadState = 'ready';
            this.recoveryState = {
                ...this.recoveryState,
                derivedRecovered: true,
            };
            return;
        }

        this.derivedLoadState = 'loading';
        const startedAt = Date.now();
        console.log(
            `[PhoenixStoreService] Derived load requested (${reason}) -> ${checkpointMeta.file} (${checkpointMeta.bytes} bytes)`,
        );

        try {
            const bytes = await this.persistence.readCheckpointFile(checkpointMeta.file);
            if (!bytes?.byteLength) {
                throw new Error(`Missing derived checkpoint: ${checkpointMeta.file}`);
            }
            await this.phoenix.importSnapshot(bytes);
            await this.phoenix.storeCommand('persistence:clearDerivedEphemera');
            this.derivedLoadState = 'ready';
            this.recoveryState = {
                ...this.recoveryState,
                derivedRecovered: true,
            };
            console.log(
                `[PhoenixStoreService] Derived load complete (${Date.now() - startedAt}ms, ${bytes.byteLength} bytes)`,
            );
        } catch (error) {
            console.warn('[PhoenixStoreService] Derived checkpoint load failed; clearing derived partition.', error);
            await this.phoenix.storeCommand('persistence:clearDerived');
            this.derivedLoadState = 'ready';
            this.derivedDirty = false;
            this.recoveryState = {
                ...this.recoveryState,
                derivedRecovered: false,
            };
        }
    }

    async upsertNote(note: StoreNote): Promise<void> {
        await this.runContentMutation([{ command: 'note:upsert', payload: { row: noteToRow(note) } }]);
    }

    async getNote(id: string): Promise<StoreNote | null> {
        const row = await this.getNoteRow(id, true);
        return row ? rowToNote(row) : null;
    }

    async getNoteHeader(id: string): Promise<StoreNoteHeader | null> {
        const row = await this.getNoteRow(id, false);
        return row ? rowToNoteHeader(row) : null;
    }

    async deleteNote(id: string): Promise<void> {
        await this.runContentMutation([{ command: 'note:delete', payload: { id } }]);
    }

    async listNotes(folderId?: string): Promise<StoreNote[]> {
        const rows = await this.listNoteRows(folderId, true);
        return rows.map(rowToNote);
    }

    async listNoteHeaders(folderId?: string): Promise<StoreNoteHeader[]> {
        const rows = await this.listNoteRows(folderId, false);
        return rows.map(rowToNoteHeader);
    }

    async getNotesByIds(ids: string[]): Promise<StoreNote[]> {
        if (!ids.length) {
            return [];
        }
        await this.ensureInitialized();
        const payload = await this.phoenix.storeCommand('note:listByIds', {
            ids,
            includeBody: true,
        });
        return Array.isArray(payload) ? payload.map(rowToNote) : [];
    }

    async upsertEntity(entity: StoreEntity): Promise<void> {
        await this.runContentRelationUpsert('entities', entityToRow(entity));
    }

    async getEntity(id: string): Promise<StoreEntity | null> {
        await this.ensureInitialized();
        const row = await this.relationGetFirst<any>('entities', { id });
        return row ? rowToEntity(row) : null;
    }

    async getEntityByLabel(label: string): Promise<StoreEntity | null> {
        await this.ensureInitialized();
        const rows = await this.relationList<any>('entities');
        const normalized = label.trim().toLowerCase();
        const row = rows.find((candidate) => String(candidate.label || '').trim().toLowerCase() === normalized);
        return row ? rowToEntity(row) : null;
    }

    async deleteEntity(id: string): Promise<void> {
        await this.runContentRelationDelete('entities', { id });
    }

    async listEntities(kind?: string): Promise<StoreEntity[]> {
        await this.ensureInitialized();
        const rows = await this.relationList<any>('entities', kind ? { kind } : undefined);
        return rows
            .map(rowToEntity)
            .sort((left, right) => left.label.localeCompare(right.label));
    }

    async upsertEdge(edge: StoreEdge): Promise<void> {
        await this.runContentRelationUpsert('edges', edgeToRow(edge));
    }

    async getEdge(id: string): Promise<StoreEdge | null> {
        await this.ensureInitialized();
        const row = await this.relationGetFirst<any>('edges', { id });
        return row ? rowToEdge(row) : null;
    }

    async deleteEdge(id: string): Promise<void> {
        await this.runContentRelationDelete('edges', { id });
    }

    async listEdgesForEntity(entityId: string): Promise<StoreEdge[]> {
        await this.ensureInitialized();
        const rows = await this.relationList<any>('edges');
        return rows
            .filter((row) => row.source_id === entityId || row.target_id === entityId)
            .map(rowToEdge);
    }

    async listAllEdges(): Promise<StoreEdge[]> {
        await this.ensureInitialized();
        const rows = await this.relationList<any>('edges');
        return rows.map(rowToEdge);
    }

    async upsertFolder(folder: StoreFolder): Promise<void> {
        await this.runContentRelationUpsert('folders', folderToRow(folder));
    }

    async getFolder(id: string): Promise<StoreFolder | null> {
        await this.ensureInitialized();
        const row = await this.relationGetFirst<any>('folders', { id });
        return row ? rowToFolder(row) : null;
    }

    async deleteFolder(id: string): Promise<void> {
        await this.runContentRelationDelete('folders', { id });
    }

    async listFolders(parentId?: string): Promise<StoreFolder[]> {
        await this.ensureInitialized();
        const rows = await this.relationList<any>(
            'folders',
            parentId !== undefined ? { parent_id: parentId } : undefined,
        );
        return rows
            .map(rowToFolder)
            .sort((left, right) => left.folderOrder - right.folderOrder || left.name.localeCompare(right.name));
    }

    async upsertScopedDocument(document: StoreScopedDocument): Promise<void> {
        await this.runContentRelationUpsert('scoped_documents', scopedDocumentToRow(document));
    }

    async getScopedDocument(
        scopeFolderId: string,
        namespace: string,
        documentKey: string,
    ): Promise<StoreScopedDocument | null> {
        await this.ensureInitialized();
        const row = await this.relationGetFirst<any>('scoped_documents', {
            scope_folder_id: scopeFolderId,
            namespace,
            document_key: documentKey,
        });
        return row ? rowToScopedDocument(row) : null;
    }

    async listScopedDocuments(scopeFolderId: string, namespace?: string): Promise<StoreScopedDocument[]> {
        await this.ensureInitialized();
        const rows = await this.relationList<any>('scoped_documents', {
            scope_folder_id: scopeFolderId,
            ...(namespace ? { namespace } : {}),
        });
        return rows
            .map(rowToScopedDocument)
            .sort((left, right) => left.documentKey.localeCompare(right.documentKey));
    }

    async deleteScopedDocument(
        scopeFolderId: string,
        namespace: string,
        documentKey: string,
    ): Promise<void> {
        await this.runContentRelationDelete('scoped_documents', {
            scope_folder_id: scopeFolderId,
            namespace,
            document_key: documentKey,
        });
    }

    async upsertScopedEntityField(field: StoreScopedEntityField): Promise<void> {
        await this.runContentRelationUpsert('scoped_entity_fields', scopedEntityFieldToRow(field));
    }

    async getScopedEntityField(
        entityId: string,
        scopeFolderId: string,
        fieldKey: string,
    ): Promise<StoreScopedEntityField | null> {
        await this.ensureInitialized();
        const row = await this.relationGetFirst<any>('scoped_entity_fields', {
            entity_id: entityId,
            scope_folder_id: scopeFolderId,
            field_key: fieldKey,
        });
        return row ? rowToScopedEntityField(row) : null;
    }

    async listScopedEntityFields(
        scopeFolderId: string,
        entityId?: string,
    ): Promise<StoreScopedEntityField[]> {
        await this.ensureInitialized();
        const rows = await this.relationList<any>('scoped_entity_fields', {
            scope_folder_id: scopeFolderId,
            ...(entityId ? { entity_id: entityId } : {}),
        });
        return rows
            .map(rowToScopedEntityField)
            .sort((left, right) => left.fieldKey.localeCompare(right.fieldKey));
    }

    async deleteScopedEntityField(
        entityId: string,
        scopeFolderId: string,
        fieldKey: string,
    ): Promise<void> {
        await this.runContentRelationDelete('scoped_entity_fields', {
            entity_id: entityId,
            scope_folder_id: scopeFolderId,
            field_key: fieldKey,
        });
    }

    async upsertScopedDefinition(definition: StoreScopedDefinition): Promise<void> {
        await this.runContentRelationUpsert('scoped_definitions', scopedDefinitionToRow(definition));
    }

    async getScopedDefinition(
        narrativeId: string,
        namespace: string,
        definitionKey: string,
    ): Promise<StoreScopedDefinition | null> {
        await this.ensureInitialized();
        const row = await this.relationGetFirst<any>('scoped_definitions', {
            narrative_id: narrativeId,
            namespace,
            definition_key: definitionKey,
        });
        return row ? rowToScopedDefinition(row) : null;
    }

    async listScopedDefinitions(narrativeId: string, namespace?: string): Promise<StoreScopedDefinition[]> {
        await this.ensureInitialized();
        const rows = await this.relationList<any>('scoped_definitions', {
            narrative_id: narrativeId,
            ...(namespace ? { namespace } : {}),
        });
        return rows
            .map(rowToScopedDefinition)
            .sort((left, right) => left.definitionKey.localeCompare(right.definitionKey));
    }

    async deleteScopedDefinition(
        narrativeId: string,
        namespace: string,
        definitionKey: string,
    ): Promise<void> {
        await this.runContentRelationDelete('scoped_definitions', {
            narrative_id: narrativeId,
            namespace,
            definition_key: definitionKey,
        });
    }

    async storeUpsertDiscoveryCandidate(candidate: StoreDiscoveryCandidate): Promise<{ success: boolean; error?: string }> {
        await this.runContentRelationUpsert('discovery_candidates', {
            token: candidate.token,
            kind: candidate.kind,
            score: candidate.score,
            status: candidate.status,
            last_seen: candidate.lastSeen,
            first_seen: candidate.firstSeen,
            count: candidate.count,
        });
        return { success: true };
    }

    async storeListDiscoveryCandidates(): Promise<StoreDiscoveryCandidate[]> {
        await this.ensureInitialized();
        const rows = await this.relationList<any>('discovery_candidates');
        return rows
            .map((row) => ({
                token: String(row.token || ''),
                kind: Number(row.kind || 0),
                score: Number(row.score || 0),
                status: Number(row.status || 0),
                lastSeen: Number(row.last_seen || 0),
                firstSeen: Number(row.first_seen || 0),
                count: Number(row.count || 0),
            }))
            .sort((left, right) => right.score - left.score || left.token.localeCompare(right.token));
    }

    async upsertEntityCards(cards: StoreEntityCard[]): Promise<void> {
        await this.runContentMutation([
            {
                command: 'entityCards:upsertBatch',
                payload: {
                    cards: cards.map(entityCardToPhoenix),
                },
            },
        ]);
    }

    async getEntityCards(entityId: string): Promise<StoreEntityCard[]> {
        await this.ensureInitialized();
        const payload = await this.phoenix.storeCommand('entityCards:get', { entityId });
        return Array.isArray(payload) ? payload.map(phoenixEntityCardToStore) : [];
    }

    async storeUpsertEntityCards(cards: StoreEntityCard[]): Promise<{ success: boolean; error?: string }> {
        await this.upsertEntityCards(cards);
        return { success: true };
    }

    async storeGetEntityCards(entityId: string): Promise<StoreEntityCard[]> {
        return this.getEntityCards(entityId);
    }

    async upsertFolderSchema(schema: StoreFolderSchema): Promise<void> {
        await this.runContentMutation([
            {
                command: 'folderSchema:upsert',
                payload: {
                    schema: folderSchemaToPhoenix(schema),
                },
            },
        ]);
    }

    async getFolderSchema(id: string): Promise<StoreFolderSchema | null> {
        await this.ensureInitialized();
        const payload = await this.phoenix.storeCommand('folderSchema:get', { id });
        return payload ? phoenixFolderSchemaToStore(payload) : null;
    }

    async storeUpsertFolderSchema(schema: StoreFolderSchema): Promise<{ success: boolean; error?: string }> {
        await this.upsertFolderSchema(schema);
        return { success: true };
    }

    async storeGetFolderSchema(id: string): Promise<StoreFolderSchema | null> {
        return this.getFolderSchema(id);
    }

    async saveNetworkView(view: {
        instance: StoreNetworkInstance;
        members: StoreNetworkMembership[];
        relationships: StoreNetworkRelationship[];
    }): Promise<void> {
        await this.runContentMutation([
            {
                command: 'networkView:save',
                payload: {
                    view: networkViewToPhoenix(view),
                },
            },
        ]);
    }

    async getNetworkView(id: string): Promise<{
        instance: StoreNetworkInstance;
        members: StoreNetworkMembership[];
        relationships: StoreNetworkRelationship[];
    } | null> {
        await this.ensureInitialized();
        const payload = await this.phoenix.storeCommand('networkView:get', { id });
        return payload ? phoenixNetworkViewToStore(payload) : null;
    }

    async listNetworkViews(): Promise<StoreNetworkInstance[]> {
        await this.ensureInitialized();
        const payload = await this.phoenix.storeCommand('networkView:list');
        return Array.isArray(payload) ? payload.map(phoenixNetworkInstanceToStore) : [];
    }

    async deleteNetworkView(id: string): Promise<void> {
        await this.runContentMutation([{ command: 'networkView:delete', payload: { id } }]);
    }

    async storeUpsertNetworkInstance(instance: StoreNetworkInstance): Promise<{ success: boolean; error?: string }> {
        const current = (await this.getNetworkView(instance.id)) || {
            instance,
            members: instance.entityIds.map((entityId) => ({
                networkId: instance.id,
                entityId,
                x: 0,
                y: 0,
                fixed: false,
            })),
            relationships: [],
        };
        current.instance = instance;
        current.members = reconcileMembers(instance.id, current.members, instance.entityIds);
        await this.saveNetworkView(current);
        return { success: true };
    }

    async storeGetNetworkInstance(id: string): Promise<StoreNetworkInstance | null> {
        const view = await this.getNetworkView(id);
        return view?.instance || null;
    }

    async storeListNetworkInstances(): Promise<StoreNetworkInstance[]> {
        return this.listNetworkViews();
    }

    async storeDeleteNetworkInstance(id: string): Promise<{ success: boolean; error?: string }> {
        await this.deleteNetworkView(id);
        return { success: true };
    }

    async storeUpsertNetworkMembership(member: StoreNetworkMembership): Promise<{ success: boolean; error?: string }> {
        const view = (await this.getNetworkView(member.networkId)) || {
            instance: emptyNetworkInstance(member.networkId),
            members: [],
            relationships: [],
        };
        const others = view.members.filter((current) => current.entityId !== member.entityId);
        view.members = [...others, member];
        view.instance.entityIds = Array.from(new Set(view.members.map((current) => current.entityId)));
        await this.saveNetworkView(view);
        return { success: true };
    }

    async storeGetNetworkMembers(networkId: string): Promise<StoreNetworkMembership[]> {
        const view = await this.getNetworkView(networkId);
        return view?.members || [];
    }

    async storeDeleteNetworkMembership(networkId: string, entityId: string): Promise<{ success: boolean; error?: string }> {
        const view = await this.getNetworkView(networkId);
        if (!view) {
            return { success: true };
        }
        view.members = view.members.filter((member) => member.entityId !== entityId);
        view.relationships = view.relationships.filter(
            (relationship) =>
                relationship.sourceEntityId !== entityId && relationship.targetEntityId !== entityId,
        );
        view.instance.entityIds = Array.from(new Set(view.members.map((member) => member.entityId)));
        await this.saveNetworkView(view);
        return { success: true };
    }

    async storeUpsertNetworkRelationship(
        relationship: StoreNetworkRelationship,
    ): Promise<{ success: boolean; error?: string }> {
        const view = (await this.getNetworkView(relationship.networkId)) || {
            instance: emptyNetworkInstance(relationship.networkId),
            members: [],
            relationships: [],
        };
        const others = view.relationships.filter((current) => current.id !== relationship.id);
        view.relationships = [...others, relationship];
        await this.saveNetworkView(view);
        return { success: true };
    }

    async storeGetNetworkRelationships(networkId: string): Promise<StoreNetworkRelationship[]> {
        const view = await this.getNetworkView(networkId);
        return view?.relationships || [];
    }

    async storeDeleteNetworkRelationship(
        networkId: string,
        relationshipId: string,
    ): Promise<{ success: boolean; error?: string }> {
        const view = await this.getNetworkView(networkId);
        if (!view) {
            return { success: true };
        }
        view.relationships = view.relationships.filter((relationship) => relationship.id !== relationshipId);
        await this.saveNetworkView(view);
        return { success: true };
    }

    async exportDatabase(): Promise<PhoenixCheckpointBundle> {
        await this.ensureInitialized();
        const [contentSnapshot, derivedSnapshot] = await Promise.all([
            this.exportSnapshotPartition('content'),
            this.exportSnapshotPartition('derived'),
        ]);
        return {
            contentSnapshot,
            derivedSnapshot,
        };
    }

    async importDatabase(data: Uint8Array): Promise<void> {
        await this.ensureInitialized();
        await this.runSerialized(async () => {
            await this.phoenix.importSnapshot(data);
            this.manifest = createEmptyPhoenixManifest();
            this.derivedDirty = true;
            await this.flushContentCheckpoint(true);
            await this.flushDerivedCheckpoint(true);
        });
    }

    async countNotes(): Promise<number> {
        const notes = await this.listNoteHeaders();
        return notes.length;
    }

    static fromDexieNote(dexieNote: any): StoreNote {
        return {
            id: dexieNote.id,
            worldId: dexieNote.worldId || '',
            title: dexieNote.title || '',
            content: typeof dexieNote.content === 'string' ? dexieNote.content : JSON.stringify(dexieNote.content || ''),
            markdownContent: dexieNote.markdownContent || '',
            folderId: dexieNote.folderId || '',
            entityKind: dexieNote.entityKind || '',
            entitySubtype: dexieNote.entitySubtype || '',
            isEntity: dexieNote.isEntity || false,
            isPinned: dexieNote.isPinned || false,
            favorite: dexieNote.favorite || false,
            ownerId: dexieNote.ownerId || '',
            narrativeId: dexieNote.narrativeId || '',
            order: dexieNote.order || 0,
            createdAt: dexieNote.createdAt || Date.now(),
            updatedAt: dexieNote.updatedAt || Date.now(),
            version: dexieNote.version,
        };
    }

    static fromDexieEntity(dexieEntity: any): StoreEntity {
        return {
            id: dexieEntity.id,
            label: dexieEntity.label || '',
            kind: dexieEntity.kind || 'UNKNOWN',
            subtype: dexieEntity.subtype,
            aliases: dexieEntity.aliases || [],
            firstNote: dexieEntity.firstNote || '',
            totalMentions: dexieEntity.totalMentions || 0,
            narrativeId: dexieEntity.narrativeId,
            createdBy: dexieEntity.createdBy || 'user',
            createdAt: dexieEntity.createdAt || Date.now(),
            updatedAt: dexieEntity.updatedAt || Date.now(),
        };
    }

    static fromDexieEdge(dexieEdge: any): StoreEdge {
        return {
            id: dexieEdge.id,
            sourceId: dexieEdge.sourceId,
            targetId: dexieEdge.targetId,
            relType: dexieEdge.relType || 'RELATED_TO',
            confidence: dexieEdge.confidence ?? 1.0,
            bidirectional: dexieEdge.bidirectional || false,
            sourceNote: dexieEdge.sourceNote,
            createdAt: dexieEdge.createdAt || Date.now(),
        };
    }

    static fromDexieFolder(dexieFolder: any): StoreFolder {
        return {
            id: dexieFolder.id,
            name: dexieFolder.name || '',
            parentId: dexieFolder.parentId,
            worldId: dexieFolder.worldId || '',
            narrativeId: dexieFolder.narrativeId,
            folderOrder: dexieFolder.folderOrder ?? dexieFolder.order ?? 0,
            entityKind: dexieFolder.entityKind || '',
            entitySubtype: dexieFolder.entitySubtype || '',
            entityLabel: dexieFolder.entityLabel || '',
            color: dexieFolder.color || '',
            isTypedRoot: dexieFolder.isTypedRoot || false,
            isSubtypeRoot: dexieFolder.isSubtypeRoot || false,
            collapsed: dexieFolder.collapsed || false,
            ownerId: dexieFolder.ownerId || '',
            isNarrativeRoot: dexieFolder.isNarrativeRoot || false,
            attributes: dexieFolder.attributes
                ? typeof dexieFolder.attributes === 'string'
                    ? dexieFolder.attributes
                    : JSON.stringify(dexieFolder.attributes)
                : undefined,
            createdAt: dexieFolder.createdAt || Date.now(),
            updatedAt: dexieFolder.updatedAt || Date.now(),
        };
    }

    private async initializeInternal(): Promise<void> {
        const startedAt = Date.now();
        console.log('[PhoenixStoreService] initialize:start');
        console.log('[PhoenixStoreService] initialize:wasm.load:start');
        await this.phoenix.loadWasm();
        console.log(`[PhoenixStoreService] initialize:wasm.load:complete (${Date.now() - startedAt}ms)`);
        console.log('[PhoenixStoreService] initialize:initRuntime:start');
        await this.phoenix.initRuntime(false);
        console.log(`[PhoenixStoreService] initialize:initRuntime:complete (${Date.now() - startedAt}ms)`);
        console.log('[PhoenixStoreService] initialize:runtime.compat:start');
        await this.ensureRuntimeCompatibility();
        console.log(`[PhoenixStoreService] initialize:runtime.compat:complete (${Date.now() - startedAt}ms)`);

        console.log('[PhoenixStoreService] initialize:persistence.load:start');
        const persisted = await this.persistence.loadManifestMeta();
        this.manifestMeta = persisted;
        const closedWalBytes = persisted.closedSegments.reduce((sum, segment) => sum + segment.bytes, 0);
        console.log(
            `[PhoenixStoreService] initialize:persistence.load:complete (${Date.now() - startedAt}ms, manifest=${persisted.manifest ? 'yes' : 'no'}, manifestBytes=${persisted.manifestBytes}, content=${persisted.contentCheckpoint?.bytes || 0} bytes, derived=${persisted.derivedCheckpoint?.bytes || 0} bytes, closedWal=${closedWalBytes} bytes, activeWal=${persisted.activeSegment?.bytes || 0} bytes)`,
        );
        console.log(`[PhoenixStoreService] Persistence state -> ${formatPhoenixPersistenceSummary(persisted)}`);
        if (persisted.recoveredFromBackup) {
            console.warn('[PhoenixStoreService] Recovered Phoenix manifest from backup copy.');
        }
        if (persisted.staleLegacyFiles.length) {
            console.warn(
                `[PhoenixStoreService] Ignoring legacy Phoenix snapshot artifacts (not used for restore): ${persisted.staleLegacyFiles.join(', ')}`,
            );
        }
        if (!hasActivePhoenixPersistence(persisted)) {
            console.log(
                '[PhoenixStoreService] No active Phoenix manifest/WAL state found. Boot will start from an empty Phoenix manifest; Dexie hydration should stay empty unless data is reseeded elsewhere.',
            );
        }

        if (persisted.manifest) {
            this.manifest = persisted.manifest;
            try {
                this.recoveryState = await this.recoverRuntimeFromManifest(persisted);
                this.derivedLoadState = persisted.derivedCheckpoint?.file ? 'cold' : 'ready';
                if (this.derivedLoadState === 'cold') {
                    console.log('[PhoenixStoreService] Derived checkpoint recovery intentionally deferred until first use.');
                }
            } catch (error) {
                console.error('[PhoenixStoreService] Phoenix WAL recovery failed. Resetting Phoenix runtime.', error);
                await this.resetPhoenixPersistence(persisted.manifest);
                await this.phoenix.initRuntime(true);
                this.manifest = createEmptyPhoenixManifest();
                this.manifestMeta = null;
                this.derivedLoadState = 'ready';
            }
        } else {
            this.manifest = createEmptyPhoenixManifest();
            this.derivedLoadState = 'ready';
        }

        this.initialized = true;
        console.log(`[PhoenixStoreService] initialize:complete (${Date.now() - startedAt}ms)`);
    }

    private async ensureRuntimeCompatibility(): Promise<void> {
        try {
            const payload = await this.phoenix.storeCommand('runtime:capabilities');
            assertPhoenixRuntimeCapabilities(payload);
        } catch (error) {
            const normalized = normalizePhoenixRuntimeCompatibilityError(error);
            if (normalized !== error || isPhoenixWasmMismatchError(normalized)) {
                console.error('[PhoenixStoreService] Phoenix WASM compatibility check failed.', normalized);
            }
            throw normalized;
        }
    }

    private async ensureInitialized(): Promise<void> {
        if (!this.initialized) {
            await this.initialize();
        }
    }

    private clearCheckpointTimers(): void {
        if (this.contentCheckpointTimeout) {
            clearTimeout(this.contentCheckpointTimeout);
            this.contentCheckpointTimeout = null;
        }
        if (this.derivedCheckpointTimeout) {
            clearTimeout(this.derivedCheckpointTimeout);
            this.derivedCheckpointTimeout = null;
        }
    }

    private scheduleContentCheckpoint(delayMs = PHOENIX_WAL_IDLE_CHECKPOINT_MS): void {
        const manifest = this.manifest;
        if (!manifest || this.snapshotsPaused) {
            return;
        }
        const lastSeq = manifest.content.nextSeq - 1;
        if (lastSeq <= manifest.content.lastCheckpointSeq && !shouldCheckpointContent(manifest)) {
            return;
        }
        if (shouldCheckpointContent(manifest)) {
            delayMs = 0;
        }
        if (this.contentCheckpointTimeout) {
            clearTimeout(this.contentCheckpointTimeout);
        }
        this.contentCheckpointTimeout = setTimeout(() => {
            void this.runSerialized(async () => {
                try {
                    await this.flushContentCheckpoint(false);
                } catch (error) {
                    console.error('[PhoenixStoreService] Failed to checkpoint Phoenix content partition:', error);
                }
            });
        }, delayMs);
    }

    private scheduleDerivedCheckpoint(delayMs = PHOENIX_DERIVED_CHECKPOINT_MS): void {
        if (!this.initialized || this.snapshotsPaused || !this.derivedDirty || !this.isDerivedReady) {
            return;
        }
        if (this.derivedCheckpointTimeout) {
            clearTimeout(this.derivedCheckpointTimeout);
        }
        this.derivedCheckpointTimeout = setTimeout(() => {
            void this.runSerialized(async () => {
                try {
                    await this.flushDerivedCheckpoint(false);
                } catch (error) {
                    console.error('[PhoenixStoreService] Failed to checkpoint Phoenix derived partition:', error);
                }
            });
        }, delayMs);
    }

    private async exportSnapshotPartition(partition: PhoenixSnapshotPartition): Promise<Uint8Array> {
        if (partition === 'derived') {
            await this.ensureDerivedLoaded('snapshot-export');
        }
        return this.phoenix.exportSnapshot(partition);
    }

    private async getNoteRow(id: string, includeBody: boolean): Promise<any | null> {
        await this.ensureInitialized();
        const payload = await this.phoenix.storeCommand('note:get', { id, includeBody });
        return payload || null;
    }

    private async listNoteRows(folderId: string | undefined, includeBody: boolean): Promise<any[]> {
        await this.ensureInitialized();
        const payload = await this.phoenix.storeCommand('note:list', {
            includeBody,
            ...(folderId !== undefined ? { folderId } : {}),
        });
        return Array.isArray(payload) ? payload : [];
    }

    private async runContentRelationUpsert(relation: string, row: Record<string, unknown>): Promise<void> {
        await this.runContentMutation([{ command: 'relation:upsert', payload: { relation, row } }]);
    }

    private async relationGetFirst<T>(relation: string, filter: Record<string, unknown>): Promise<T | null> {
        const payload = await this.phoenix.storeCommand('relation:getFirst', { relation, filter });
        return (payload as T | null) || null;
    }

    private async relationList<T>(relation: string, filter?: Record<string, unknown>): Promise<T[]> {
        const payload = await this.phoenix.storeCommand('relation:list', {
            relation,
            ...(filter ? { filter } : {}),
        });
        return Array.isArray(payload) ? (payload as T[]) : [];
    }

    private async runContentRelationDelete(relation: string, filter: Record<string, unknown>): Promise<void> {
        await this.runContentMutation([{ command: 'relation:delete', payload: { relation, filter } }]);
    }

    private async runContentMutation(mutations: ContentWalMutation[]): Promise<void> {
        if (!mutations.length) {
            return;
        }
        await this.ensureInitialized();
        await this.runSerialized(async () => {
            const manifest = this.requireManifest();
            const batch = this.buildWalBatch(mutations, manifest.content.nextSeq);
            const appendResult = await this.persistence.appendWalBatch(batch);
            const nextManifest = nextManifestWithWalAppend(manifest, batch, appendResult);

            await this.persistence.commitManifest(nextManifest);

            try {
                await this.phoenix.storeCommand('persistence:applyWalBatch', { records: batch.records });
            } catch (error) {
                console.error('[PhoenixStoreService] Runtime apply failed after WAL commit. Rebuilding runtime.', error);
                await this.reloadRuntimeFromPersistence();
                this.scheduleContentCheckpoint();
                return;
            }

            this.manifest = nextManifest;
            this.recoveryState = {
                ...this.recoveryState,
                contentRecovered: true,
                replayedRecords: this.recoveryState.replayedRecords + batch.records.length,
                lastRecoveredSeq: batch.records[batch.records.length - 1]?.seq || this.recoveryState.lastRecoveredSeq,
                manifestGeneration: nextManifest.generation,
            };
            this.scheduleContentCheckpoint();
        });
    }

    private buildWalBatch(mutations: ContentWalMutation[], nextSeq: number): PhoenixWalBatch {
        const writtenAt = Date.now();
        return {
            records: mutations.map((mutation, index) => ({
                seq: nextSeq + index,
                command: mutation.command,
                payload: mutation.payload,
                partition: 'content',
                writtenAt,
            })),
        };
    }

    private async runSerialized<T>(task: () => Promise<T>): Promise<T> {
        const run = this.mutationChain.then(task, task);
        this.mutationChain = run.then(
            () => undefined,
            () => undefined,
        );
        return run;
    }

    private async flushContentCheckpoint(force: boolean): Promise<void> {
        if (this.contentCheckpointTimeout) {
            clearTimeout(this.contentCheckpointTimeout);
            this.contentCheckpointTimeout = null;
        }
        const manifest = this.requireManifest();
        const lastSeq = manifest.content.nextSeq - 1;
        if (!force && lastSeq <= manifest.content.lastCheckpointSeq && !shouldCheckpointContent(manifest)) {
            return;
        }

        const snapshot = await this.exportSnapshotPartition('content');
        const checkpoint = await this.persistence.writeCheckpoint('content', manifest.generation + 1, snapshot);
        const finalized = finalizeContentCheckpointManifest(manifest, checkpoint.file, lastSeq);
        await this.persistence.commitManifest(finalized.manifest);
        if (finalized.pruneFiles.length) {
            await this.persistence.pruneFiles(finalized.pruneFiles);
        }
        this.manifest = finalized.manifest;
        this.recoveryState = {
            ...this.recoveryState,
            contentRecovered: true,
            lastRecoveredSeq: Math.max(this.recoveryState.lastRecoveredSeq, lastSeq),
            manifestGeneration: finalized.manifest.generation,
        };
    }

    private async flushDerivedCheckpoint(force: boolean): Promise<void> {
        if (this.derivedCheckpointTimeout) {
            clearTimeout(this.derivedCheckpointTimeout);
            this.derivedCheckpointTimeout = null;
        }
        if ((!this.derivedDirty && !force) || !this.isDerivedReady) {
            return;
        }

        const manifest = this.requireManifest();
        const snapshot = await this.exportSnapshotPartition('derived');
        const checkpoint = await this.persistence.writeCheckpoint('derived', manifest.generation + 1, snapshot);
        const finalized = finalizeDerivedCheckpointManifest(manifest, checkpoint.file, checkpoint.writtenAt);
        await this.persistence.commitManifest(finalized.manifest);
        if (finalized.pruneFiles.length) {
            await this.persistence.pruneFiles(finalized.pruneFiles);
        }
        this.manifest = finalized.manifest;
        this.derivedDirty = false;
        this.recoveryState = {
            ...this.recoveryState,
            derivedRecovered: true,
            manifestGeneration: finalized.manifest.generation,
        };
    }

    private requireManifest(): PersistenceManifest {
        if (!this.manifest) {
            this.manifest = createEmptyPhoenixManifest();
        }
        return this.manifest;
    }

    private async reloadRuntimeFromPersistence(): Promise<void> {
        const restoreDerivedAfterReload = this.isDerivedReady;
        await this.phoenix.initRuntime(true);
        const persisted = await this.persistence.loadManifestMeta();
        this.manifestMeta = persisted;
        if (!persisted.manifest) {
            this.manifest = createEmptyPhoenixManifest();
            this.derivedLoadState = 'ready';
            this.recoveryState = {
                contentRecovered: false,
                derivedRecovered: false,
                replayedRecords: 0,
                lastRecoveredSeq: 0,
                manifestGeneration: 0,
            };
            return;
        }
        this.manifest = persisted.manifest;
        this.recoveryState = await this.recoverRuntimeFromManifest(persisted);
        this.derivedLoadState = persisted.derivedCheckpoint?.file ? 'cold' : 'ready';
        if (restoreDerivedAfterReload && persisted.derivedCheckpoint?.file) {
            await this.loadDerivedCheckpointIntoRuntime('runtime-reload');
        }
    }

    private async recoverRuntimeFromManifest(persisted: LoadedPhoenixManifestState): Promise<RecoveryState> {
        const manifest = persisted.manifest;
        if (!manifest) {
            return {
                contentRecovered: false,
                derivedRecovered: false,
                replayedRecords: 0,
                lastRecoveredSeq: 0,
                manifestGeneration: 0,
            };
        }

        if (manifest.content.checkpointFile) {
            if (!persisted.contentCheckpoint?.bytes) {
                throw new Error(`Missing content checkpoint: ${manifest.content.checkpointFile}`);
            }
            console.log(
                `[PhoenixStoreService] Content recovery importing ${persisted.contentCheckpoint.file} (${persisted.contentCheckpoint.bytes} bytes)`,
            );
            const bytes = await this.persistence.readCheckpointFile(persisted.contentCheckpoint.file);
            if (!bytes?.byteLength) {
                throw new Error(`Missing content checkpoint: ${manifest.content.checkpointFile}`);
            }
            await this.phoenix.importSnapshot(bytes);
        }

        console.log('[PhoenixStoreService] Content recovery replaying Phoenix WAL incrementally.');
        const replayResult = await this.replayWalSegments(persisted, manifest);
        console.log(
            `[PhoenixStoreService] Content recovery complete (${replayResult.replayedRecords} records replayed, lastSeq=${replayResult.lastRecoveredSeq}).`,
        );

        return {
            contentRecovered: true,
            derivedRecovered: false,
            replayedRecords: replayResult.replayedRecords,
            lastRecoveredSeq: replayResult.lastRecoveredSeq,
            manifestGeneration: manifest.generation,
        };
    }

    private async replayWalSegments(
        persisted: LoadedPhoenixManifestState,
        manifest: PersistenceManifest,
    ): Promise<{ replayedRecords: number; lastRecoveredSeq: number }> {
        const seenSeq = new Set<number>();
        const maxSeqExclusive = manifest.content.nextSeq;
        let replayedRecords = 0;
        let lastRecoveredSeq = manifest.content.lastCheckpointSeq;

        const closedOrder = new Map(manifest.content.closedSegments.map((segment, index) => [segment.file, index]));
        const sortedClosed = [...persisted.closedSegments].sort(
            (left, right) => (closedOrder.get(left.file) || 0) - (closedOrder.get(right.file) || 0),
        );
        for (const segment of sortedClosed) {
            if (!segment.bytes) {
                throw new Error(`Missing closed WAL segment: ${segment.file}`);
            }
            const result = await this.applyReplaySegment(
                segment.file,
                false,
                manifest.content.lastCheckpointSeq,
                maxSeqExclusive,
                seenSeq,
            );
            replayedRecords += result.replayedRecords;
            lastRecoveredSeq = Math.max(lastRecoveredSeq, result.lastRecoveredSeq);
        }
        if (persisted.activeSegment?.bytes) {
            const result = await this.applyReplaySegment(
                persisted.activeSegment.file,
                true,
                manifest.content.lastCheckpointSeq,
                maxSeqExclusive,
                seenSeq,
            );
            replayedRecords += result.replayedRecords;
            lastRecoveredSeq = Math.max(lastRecoveredSeq, result.lastRecoveredSeq);
        }

        return { replayedRecords, lastRecoveredSeq };
    }

    private async applyReplaySegment(
        file: string,
        allowTailCorruption: boolean,
        lastCheckpointSeq: number,
        maxSeqExclusive: number,
        seenSeq: Set<number>,
    ): Promise<{ replayedRecords: number; lastRecoveredSeq: number }> {
        const bytes = await this.persistence.readWalSegment(file);
        if (!bytes?.byteLength) {
            if (allowTailCorruption) {
                return { replayedRecords: 0, lastRecoveredSeq: lastCheckpointSeq };
            }
            throw new Error(`Missing closed WAL segment: ${file}`);
        }

        const decoded = decodeWalBytes(bytes, { maxSeqExclusive });
        if (decoded.tailCorrupted) {
            if (!allowTailCorruption) {
                throw new Error(`Corrupt WAL segment: ${file}`);
            }
            console.warn(`[PhoenixStoreService] Ignoring corrupt WAL tail in ${file}.`);
        }

        const records = decoded.records.filter((record) => {
            if (record.seq <= lastCheckpointSeq || seenSeq.has(record.seq)) {
                return false;
            }
            seenSeq.add(record.seq);
            return true;
        });

        if (!records.length) {
            return {
                replayedRecords: 0,
                lastRecoveredSeq: lastCheckpointSeq,
            };
        }

        await this.phoenix.storeCommand('persistence:applyWalBatch', { records });
        return {
            replayedRecords: records.length,
            lastRecoveredSeq: records[records.length - 1]?.seq || lastCheckpointSeq,
        };
    }

    private async resetPhoenixPersistence(manifest: PersistenceManifest): Promise<void> {
        await this.persistence.pruneFiles(collectManifestFiles(manifest));
    }
}

function noteToRow(note: StoreNote): Record<string, unknown> {
    const version = note.version ?? note.updatedAt ?? Date.now();
    const validFrom = note.createdAt || note.updatedAt || Date.now();
    return {
        id: note.id,
        version,
        world_id: note.worldId || '',
        title: note.title || '',
        content: note.content || '',
        markdown_content: note.markdownContent || '',
        folder_id: note.folderId || null,
        entity_kind: note.entityKind || null,
        entity_subtype: note.entitySubtype || null,
        is_entity: note.isEntity || false,
        is_pinned: note.isPinned || false,
        favorite: note.favorite || false,
        owner_id: note.ownerId || null,
        narrative_id: note.narrativeId || null,
        order: Number(note.order || 0),
        created_at: note.createdAt || Date.now(),
        updated_at: note.updatedAt || Date.now(),
        valid_from: validFrom,
        valid_to: null,
        is_current: true,
        change_reason: null,
    };
}

function rowToNote(row: any): StoreNote {
    return {
        ...rowToNoteHeader(row),
        content: String(row.content || ''),
        markdownContent: String(row.markdown_content || ''),
    };
}

function rowToNoteHeader(row: any): StoreNoteHeader {
    return {
        id: String(row.id || ''),
        worldId: String(row.world_id || ''),
        title: String(row.title || ''),
        folderId: String(row.folder_id || ''),
        entityKind: String(row.entity_kind || ''),
        entitySubtype: String(row.entity_subtype || ''),
        isEntity: !!row.is_entity,
        isPinned: !!row.is_pinned,
        favorite: !!row.favorite,
        ownerId: String(row.owner_id || ''),
        narrativeId: String(row.narrative_id || ''),
        order: Number(row.order || 0),
        createdAt: Number(row.created_at || 0),
        updatedAt: Number(row.updated_at || 0),
        version: Number(row.version || 0),
    };
}

function entityToRow(entity: StoreEntity): Record<string, unknown> {
    return {
        id: entity.id,
        label: entity.label,
        kind: entity.kind,
        subtype: entity.subtype || null,
        aliases: entity.aliases || [],
        first_note: entity.firstNote || null,
        total_mentions: entity.totalMentions || 0,
        narrative_id: entity.narrativeId || null,
        created_by: entity.createdBy || 'user',
        created_at: entity.createdAt || Date.now(),
        updated_at: entity.updatedAt || Date.now(),
    };
}

function rowToEntity(row: any): StoreEntity {
    return {
        id: String(row.id || ''),
        label: String(row.label || ''),
        kind: String(row.kind || 'UNKNOWN'),
        subtype: row.subtype ? String(row.subtype) : undefined,
        aliases: Array.isArray(row.aliases) ? row.aliases.map(String) : [],
        firstNote: String(row.first_note || ''),
        totalMentions: Number(row.total_mentions || 0),
        narrativeId: row.narrative_id ? String(row.narrative_id) : undefined,
        createdBy: (row.created_by || 'user') as 'user' | 'extraction' | 'auto',
        createdAt: Number(row.created_at || 0),
        updatedAt: Number(row.updated_at || 0),
    };
}

function edgeToRow(edge: StoreEdge): Record<string, unknown> {
    return {
        id: edge.id,
        source_id: edge.sourceId,
        target_id: edge.targetId,
        rel_type: edge.relType,
        confidence: edge.confidence ?? 1,
        bidirectional: edge.bidirectional || false,
        source_note: edge.sourceNote || null,
        created_at: edge.createdAt || Date.now(),
    };
}

function rowToEdge(row: any): StoreEdge {
    return {
        id: String(row.id || ''),
        sourceId: String(row.source_id || ''),
        targetId: String(row.target_id || ''),
        relType: String(row.rel_type || ''),
        confidence: Number(row.confidence || 0),
        bidirectional: !!row.bidirectional,
        sourceNote: row.source_note ? String(row.source_note) : undefined,
        createdAt: Number(row.created_at || 0),
    };
}

function folderToRow(folder: StoreFolder): Record<string, unknown> {
    return {
        id: folder.id,
        name: folder.name,
        parent_id: folder.parentId || null,
        world_id: folder.worldId || '',
        narrative_id: folder.narrativeId || null,
        folder_order: Number(folder.folderOrder || 0),
        entity_kind: folder.entityKind || null,
        entity_subtype: folder.entitySubtype || null,
        entity_label: folder.entityLabel || null,
        color: folder.color || null,
        is_typed_root: folder.isTypedRoot || false,
        is_subtype_root: folder.isSubtypeRoot || false,
        collapsed: folder.collapsed || false,
        owner_id: folder.ownerId || null,
        is_narrative_root: folder.isNarrativeRoot || false,
        attributes: parseJsonString(folder.attributes),
        created_at: folder.createdAt || Date.now(),
        updated_at: folder.updatedAt || Date.now(),
    };
}

function rowToFolder(row: any): StoreFolder {
    return {
        id: String(row.id || ''),
        name: String(row.name || ''),
        parentId: row.parent_id ? String(row.parent_id) : undefined,
        worldId: String(row.world_id || ''),
        narrativeId: row.narrative_id ? String(row.narrative_id) : undefined,
        folderOrder: Number(row.folder_order || 0),
        entityKind: String(row.entity_kind || ''),
        entitySubtype: String(row.entity_subtype || ''),
        entityLabel: String(row.entity_label || ''),
        color: String(row.color || ''),
        isTypedRoot: !!row.is_typed_root,
        isSubtypeRoot: !!row.is_subtype_root,
        collapsed: !!row.collapsed,
        ownerId: String(row.owner_id || ''),
        isNarrativeRoot: !!row.is_narrative_root,
        attributes: serializeJsonField(row.attributes),
        createdAt: Number(row.created_at || 0),
        updatedAt: Number(row.updated_at || 0),
    };
}

function scopedDocumentToRow(document: StoreScopedDocument): Record<string, unknown> {
    return {
        id: document.id,
        scope_folder_id: document.scopeFolderId,
        narrative_id: document.narrativeId,
        namespace: document.namespace,
        document_key: document.documentKey,
        payload: parseJsonString(document.payload),
        seeded_from_scope_folder_id: document.seededFromScopeFolderId || null,
        created_at: document.createdAt,
        updated_at: document.updatedAt,
    };
}

function rowToScopedDocument(row: any): StoreScopedDocument {
    return {
        id: String(row.id || ''),
        scopeFolderId: String(row.scope_folder_id || ''),
        narrativeId: String(row.narrative_id || ''),
        namespace: String(row.namespace || ''),
        documentKey: String(row.document_key || ''),
        payload: serializeJsonField(row.payload),
        seededFromScopeFolderId: row.seeded_from_scope_folder_id
            ? String(row.seeded_from_scope_folder_id)
            : undefined,
        createdAt: Number(row.created_at || 0),
        updatedAt: Number(row.updated_at || 0),
    };
}

function scopedEntityFieldToRow(field: StoreScopedEntityField): Record<string, unknown> {
    return {
        id: field.id,
        entity_id: field.entityId,
        scope_folder_id: field.scopeFolderId,
        narrative_id: field.narrativeId,
        field_key: field.fieldKey,
        value_json: parseJsonString(field.valueJson),
        seeded_from_scope_folder_id: field.seededFromScopeFolderId || null,
        created_at: field.createdAt,
        updated_at: field.updatedAt,
    };
}

function rowToScopedEntityField(row: any): StoreScopedEntityField {
    return {
        id: String(row.id || ''),
        entityId: String(row.entity_id || ''),
        scopeFolderId: String(row.scope_folder_id || ''),
        narrativeId: String(row.narrative_id || ''),
        fieldKey: String(row.field_key || ''),
        valueJson: serializeJsonField(row.value_json),
        seededFromScopeFolderId: row.seeded_from_scope_folder_id
            ? String(row.seeded_from_scope_folder_id)
            : undefined,
        createdAt: Number(row.created_at || 0),
        updatedAt: Number(row.updated_at || 0),
    };
}

function scopedDefinitionToRow(definition: StoreScopedDefinition): Record<string, unknown> {
    return {
        id: definition.id,
        narrative_id: definition.narrativeId,
        namespace: definition.namespace,
        definition_key: definition.definitionKey,
        payload: parseJsonString(definition.payload),
        created_at: definition.createdAt,
        updated_at: definition.updatedAt,
    };
}

function rowToScopedDefinition(row: any): StoreScopedDefinition {
    return {
        id: String(row.id || ''),
        narrativeId: String(row.narrative_id || ''),
        namespace: String(row.namespace || ''),
        definitionKey: String(row.definition_key || ''),
        payload: serializeJsonField(row.payload),
        createdAt: Number(row.created_at || 0),
        updatedAt: Number(row.updated_at || 0),
    };
}

function parseJsonString(value?: string): unknown {
    if (!value) {
        return null;
    }

    try {
        return JSON.parse(value);
    } catch {
        return value;
    }
}

function serializeJsonField(value: unknown): string {
    if (typeof value === 'string') {
        return value;
    }

    if (value === null || value === undefined) {
        return '';
    }

    try {
        return JSON.stringify(value);
    } catch {
        return String(value);
    }
}

function entityCardToPhoenix(card: StoreEntityCard): Record<string, unknown> {
    return {
        entityId: card.entityId,
        cardId: card.cardId,
        name: card.name,
        color: card.color,
        icon: card.icon,
        displayOrder: card.displayOrder,
        isCollapsed: card.isCollapsed,
        createdAt: card.createdAt,
        updatedAt: card.updatedAt,
    };
}

function phoenixEntityCardToStore(card: any): StoreEntityCard {
    return {
        entityId: String(card?.entityId || ''),
        cardId: String(card?.cardId || ''),
        name: String(card?.name || ''),
        color: String(card?.color || ''),
        icon: String(card?.icon || ''),
        displayOrder: Number(card?.displayOrder || 0),
        isCollapsed: !!card?.isCollapsed,
        createdAt: Number(card?.createdAt || 0),
        updatedAt: Number(card?.updatedAt || 0),
    };
}

function folderSchemaToPhoenix(schema: StoreFolderSchema): Record<string, unknown> {
    return {
        id: schema.id,
        entityKind: schema.entityKind,
        subtype: schema.subtype || '',
        name: schema.name,
        description: schema.description || '',
        allowedSubfolders: JSON.stringify(schema.allowedSubfolders || []),
        allowedNoteTypes: JSON.stringify(schema.allowedNoteTypes || []),
        isVaultRoot: schema.isVaultRoot,
        containerOnly: schema.containerOnly,
        propagateKindToChildren: schema.propagateKindToChildren,
        icon: schema.icon || '',
        isSystem: schema.isSystem,
        createdAt: schema.createdAt,
        updatedAt: schema.updatedAt,
    };
}

function phoenixFolderSchemaToStore(schema: any): StoreFolderSchema {
    return {
        id: String(schema?.id || ''),
        entityKind: String(schema?.entityKind || ''),
        subtype: schema?.subtype ? String(schema.subtype) : undefined,
        name: String(schema?.name || ''),
        description: schema?.description ? String(schema.description) : undefined,
        allowedSubfolders: parseJsonString(String(schema?.allowedSubfolders || '[]')) as unknown[],
        allowedNoteTypes: parseJsonString(String(schema?.allowedNoteTypes || '[]')) as unknown[],
        isVaultRoot: !!schema?.isVaultRoot,
        containerOnly: !!schema?.containerOnly,
        propagateKindToChildren: !!schema?.propagateKindToChildren,
        icon: schema?.icon ? String(schema.icon) : undefined,
        isSystem: !!schema?.isSystem,
        createdAt: Number(schema?.createdAt || 0),
        updatedAt: Number(schema?.updatedAt || 0),
    };
}

function networkViewToPhoenix(view: {
    instance: StoreNetworkInstance;
    members: StoreNetworkMembership[];
    relationships: StoreNetworkRelationship[];
}): Record<string, unknown> {
    const memberCount = view.members.length || view.instance.entityIds.length;
    return {
        instance: {
            id: view.instance.id,
            name: view.instance.name,
            schemaId: view.instance.schemaId,
            networkKind: 'saved_view',
            networkSubtype: '',
            rootFolderId: view.instance.rootFolderId,
            rootEntityId: view.instance.rootEntityId || '',
            namespace: 'ui.network',
            description: view.instance.description || '',
            tags: [],
            memberCount,
            relationshipCount: view.relationships.length,
            maxDepth: 0,
            createdAt: view.instance.createdAt,
            updatedAt: view.instance.updatedAt,
            groupId: '',
            scopeType: 'folder',
            narrativeId: view.instance.narrativeId,
        },
        members: view.members.map((member) => ({
            networkId: member.networkId,
            entityId: member.entityId,
            x: member.x,
            y: member.y,
            fixed: member.fixed,
        })),
        relationships: view.relationships.map((relationship) => ({
            networkId: relationship.networkId,
            sourceEntityId: relationship.sourceEntityId,
            targetEntityId: relationship.targetEntityId,
            relationshipId: relationship.id,
        })),
    };
}

function phoenixNetworkViewToStore(view: any): {
    instance: StoreNetworkInstance;
    members: StoreNetworkMembership[];
    relationships: StoreNetworkRelationship[];
} {
    return {
        instance: phoenixNetworkInstanceToStore(view?.instance),
        members: Array.isArray(view?.members) ? view.members.map(phoenixNetworkMembershipToStore) : [],
        relationships: Array.isArray(view?.relationships)
            ? view.relationships.map(phoenixNetworkRelationshipToStore)
            : [],
    };
}

function phoenixNetworkInstanceToStore(instance: any): StoreNetworkInstance {
    return {
        id: String(instance?.id || ''),
        schemaId: String(instance?.schemaId || ''),
        name: String(instance?.name || ''),
        rootFolderId: String(instance?.rootFolderId || ''),
        rootEntityId: instance?.rootEntityId ? String(instance.rootEntityId) : undefined,
        entityIds: [],
        narrativeId: String(instance?.narrativeId || ''),
        description: instance?.description ? String(instance.description) : undefined,
        createdAt: Number(instance?.createdAt || 0),
        updatedAt: Number(instance?.updatedAt || 0),
    };
}

function phoenixNetworkMembershipToStore(member: any): StoreNetworkMembership {
    return {
        networkId: String(member?.networkId || ''),
        entityId: String(member?.entityId || ''),
        x: Number(member?.x || 0),
        y: Number(member?.y || 0),
        fixed: !!member?.fixed,
    };
}

function phoenixNetworkRelationshipToStore(relationship: any): StoreNetworkRelationship {
    return {
        id: String(relationship?.relationshipId || relationship?.id || ''),
        networkId: String(relationship?.networkId || ''),
        sourceEntityId: String(relationship?.sourceEntityId || ''),
        targetEntityId: String(relationship?.targetEntityId || ''),
        relationshipCode: String(relationship?.relationshipCode || relationship?.relationshipId || ''),
        strength: undefined,
        startDate: undefined,
        endDate: undefined,
        notes: undefined,
        createdAt: 0,
        updatedAt: 0,
    };
}

function emptyNetworkInstance(networkId: string): StoreNetworkInstance {
    const now = Date.now();
    return {
        id: networkId,
        schemaId: '',
        name: networkId,
        rootFolderId: '',
        rootEntityId: undefined,
        entityIds: [],
        narrativeId: '',
        description: '',
        createdAt: now,
        updatedAt: now,
    };
}

function reconcileMembers(
    networkId: string,
    currentMembers: StoreNetworkMembership[],
    entityIds: string[],
): StoreNetworkMembership[] {
    const existing = new Map(currentMembers.map((member) => [member.entityId, member]));
    return entityIds.map((entityId) => {
        const member = existing.get(entityId);
        if (member) {
            return member;
        }
        return {
            networkId,
            entityId,
            x: 0,
            y: 0,
            fixed: false,
        };
    });
}
