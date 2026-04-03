import { Injectable, NgZone, computed, signal } from '@angular/core';

import { createNoopNgZone, createWorkerOutsideAngular } from '../core/worker-zone';
import type { PhoenixNliCanonicalLabel } from '../nli/nli-utils';

export interface NliProgress {
    type: 'init' | 'batch' | 'complete';
    current: number;
    total: number;
    message: string;
}

export interface NliPairClassificationInput {
    judgmentId: string;
    groupId: string;
    sourceId: string;
    targetId: string;
    edgeType: string;
    direction: string;
    premise: string;
    hypothesis: string;
}

export interface NliClassificationResult extends NliPairClassificationInput {
    entailment: number;
    neutral: number;
    contradiction: number;
    predictedLabel: PhoenixNliCanonicalLabel;
    confidence: number;
}

export interface NliBatch {
    results: NliClassificationResult[];
    batchIndex: number;
    totalBatches: number;
}

export type NliProgressCallback = (progress: NliProgress) => void;
export type NliBatchCallback = (batch: NliBatch) => void;

@Injectable({ providedIn: 'root' })
export class NliWorkerService {
    private worker: Worker | null = null;
    private pendingCallbacks = new Map<
        number,
        {
            resolve: Function;
            reject: Function;
            onProgress?: NliProgressCallback;
            onBatch?: NliBatchCallback;
        }
    >();
    private callbackId = 0;

    readonly isInitialized = signal(false);
    readonly modelId = signal<string | null>(null);
    readonly device = signal<string>('wasm');
    readonly isProcessing = signal(false);
    readonly progress = signal<NliProgress | null>(null);

    readonly isReady = computed(() => this.isInitialized() && !this.isProcessing());

    constructor(private readonly ngZone: NgZone = createNoopNgZone()) {}

    async initialize(modelId: string, onProgress?: NliProgressCallback): Promise<void> {
        if (this.isInitialized() && this.modelId() === modelId) {
            return;
        }

        if (this.worker) {
            await this.dispose();
        }

        this.progress.set({ type: 'init', current: 0, total: 100, message: 'Starting...' });
        onProgress?.({ type: 'init', current: 0, total: 100, message: 'Starting...' });

        this.worker = createWorkerOutsideAngular(
            this.ngZone,
            () =>
                new Worker(new URL('../../workers/nli.worker', import.meta.url), {
                    type: 'module',
                }),
            (worker) => {
                worker.onmessage = (event) => this.handleWorkerMessage(event);
                worker.onerror = (event) => {
                    this.ngZone.run(() => {
                        console.error('[NliWorkerService] Worker error:', event);
                        this.progress.set(null);
                    });
                };
            },
        );

        const id = this.nextId();
        const promise = new Promise<void>((resolve, reject) => {
            this.pendingCallbacks.set(id, {
                resolve: (payload?: { device?: string }) => {
                    this.isInitialized.set(true);
                    this.modelId.set(modelId);
                    if (payload?.device) {
                        this.device.set(payload.device);
                    }
                    resolve();
                },
                reject,
                onProgress,
            });
        });

        this.worker.postMessage({
            type: 'INIT',
            payload: { modelId },
            _id: id,
        });

        return promise;
    }

    async classifyStream(
        pairs: NliPairClassificationInput[],
        onBatch: NliBatchCallback,
        batchSize = 4,
        onProgress?: NliProgressCallback,
    ): Promise<void> {
        if (!this.isReady()) {
            throw new Error('Worker not ready. Call initialize() first.');
        }

        this.isProcessing.set(true);
        this.progress.set({
            type: 'batch',
            current: 0,
            total: Math.ceil(pairs.length / batchSize),
            message: 'Starting...',
        });

        const id = this.nextId();
        const promise = new Promise<void>((resolve, reject) => {
            this.pendingCallbacks.set(id, { resolve, reject, onBatch, onProgress });
        });

        this.worker!.postMessage({
            type: 'CLASSIFY_STREAM',
            payload: { pairs, batchSize },
            _id: id,
        });

        try {
            await promise;
            this.isProcessing.set(false);
            this.progress.set({ type: 'complete', current: 1, total: 1, message: 'Complete' });
        } catch (error) {
            this.isProcessing.set(false);
            this.progress.set(null);
            throw error;
        }
    }

    async getStatus(): Promise<{ initialized: boolean; modelId: string | null; device: string }> {
        if (!this.worker) {
            return { initialized: false, modelId: null, device: 'wasm' };
        }

        const id = this.nextId();
        const promise = new Promise<{ initialized: boolean; modelId: string | null; device: string }>(
            (resolve, reject) => {
                this.pendingCallbacks.set(id, { resolve, reject });
            },
        );

        this.worker.postMessage({ type: 'GET_STATUS', _id: id });
        return promise;
    }

    async dispose(): Promise<void> {
        if (!this.worker) {
            return;
        }

        const id = this.nextId();
        const promise = new Promise<void>((resolve, reject) => {
            this.pendingCallbacks.set(id, { resolve, reject });
        });

        this.worker.postMessage({ type: 'DISPOSE', _id: id });
        await promise;

        this.worker.terminate();
        this.worker = null;
        this.isInitialized.set(false);
        this.modelId.set(null);
        this.progress.set(null);
        this.device.set('wasm');
    }

    private nextId(): number {
        return ++this.callbackId;
    }

    private handleWorkerMessage(event: MessageEvent): void {
        const { type, payload, _id } = event.data;

        if (type === 'init_progress' || type === 'classify_progress') {
            const progress: NliProgress = {
                type: type === 'init_progress' ? 'init' : 'batch',
                current: event.data.current,
                total: event.data.total,
                message: event.data.message,
            };
            this.progress.set(progress);
            if (_id !== undefined && this.pendingCallbacks.has(_id)) {
                const { onProgress } = this.pendingCallbacks.get(_id)!;
                onProgress?.(progress);
            }
            return;
        }

        if (type === 'CLASSIFY_BATCH') {
            if (_id !== undefined && this.pendingCallbacks.has(_id)) {
                const { onBatch, onProgress } = this.pendingCallbacks.get(_id)!;
                onBatch?.(payload);
                onProgress?.({
                    type: 'batch',
                    current: payload.batchIndex,
                    total: payload.totalBatches,
                    message: `Batch ${payload.batchIndex}/${payload.totalBatches}`,
                });
            }
            return;
        }

        if (type === 'CLASSIFY_COMPLETE') {
            if (_id !== undefined && this.pendingCallbacks.has(_id)) {
                const { resolve } = this.pendingCallbacks.get(_id)!;
                this.pendingCallbacks.delete(_id);
                resolve();
            }
            return;
        }

        if (type === 'STATUS') {
            if (_id !== undefined && this.pendingCallbacks.has(_id)) {
                const { resolve } = this.pendingCallbacks.get(_id)!;
                this.pendingCallbacks.delete(_id);
                this.ngZone.run(() => resolve(payload));
            }
            return;
        }

        if (_id !== undefined && this.pendingCallbacks.has(_id)) {
            const { resolve, reject } = this.pendingCallbacks.get(_id)!;
            this.pendingCallbacks.delete(_id);

            if (type === 'ERROR') {
                this.ngZone.run(() => reject(new Error(payload?.message || 'Worker error')));
                return;
            }

            if (type === 'INIT_COMPLETE') {
                this.ngZone.run(() => resolve(payload));
                return;
            }

            if (type === 'DISPOSED') {
                this.ngZone.run(() => resolve());
                return;
            }

            this.ngZone.run(() => resolve(payload));
        }
    }
}
