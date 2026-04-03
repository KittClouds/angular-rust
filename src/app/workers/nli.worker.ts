/// <reference lib="webworker" />

import { pipeline, env } from '@huggingface/transformers';

import {
    buildNliPipelineBatch,
    normalizeNliScores,
    topNliLabel,
    type PhoenixNliCanonicalLabel,
} from '../lib/nli/nli-utils';

env.allowLocalModels = false;
env.useBrowserCache = true;

const hasWebGPU = typeof self !== 'undefined' && 'gpu' in (self as any).navigator;

type NliPairClassificationInput = {
    judgmentId: string;
    groupId: string;
    sourceId: string;
    targetId: string;
    edgeType: string;
    direction: string;
    premise: string;
    hypothesis: string;
};

type NliClassificationResult = NliPairClassificationInput & {
    entailment: number;
    neutral: number;
    contradiction: number;
    predictedLabel: PhoenixNliCanonicalLabel;
    confidence: number;
};

interface InitPayload {
    modelId: string;
    device?: 'webgpu' | 'wasm';
}

interface ClassifyStreamPayload {
    pairs: NliPairClassificationInput[];
    batchSize?: number;
}

type WorkerMessage =
    | { type: 'INIT'; payload: InitPayload; _id: number }
    | { type: 'CLASSIFY_STREAM'; payload: ClassifyStreamPayload; _id: number }
    | { type: 'DISPOSE'; payload?: never; _id: number }
    | { type: 'GET_STATUS'; payload?: never; _id: number };

interface ProgressUpdate {
    type: 'init_progress' | 'classify_progress';
    current: number;
    total: number;
    message: string;
    _id: number;
}

interface ResponseMessage {
    type:
        | 'INIT_COMPLETE'
        | 'CLASSIFY_BATCH'
        | 'CLASSIFY_COMPLETE'
        | 'DISPOSED'
        | 'STATUS'
        | 'ERROR';
    payload?: any;
    _id: number;
}

class NliWorker {
    private classifier: any = null;
    private modelId: string | null = null;
    private initialized = false;
    private device: 'webgpu' | 'wasm' = 'wasm';
    private id2label: Record<string, string> | null = null;

    async initialize(payload: InitPayload, _id: number): Promise<void> {
        if (this.initialized) {
            this.sendResponse({
                type: 'INIT_COMPLETE',
                payload: { device: this.device },
                _id,
            });
            return;
        }

        this.modelId = payload.modelId;
        const attemptedDevices: Array<'webgpu' | 'wasm'> = payload.device
            ? [payload.device]
            : hasWebGPU
              ? ['webgpu', 'wasm']
              : ['wasm'];

        let lastError: unknown = null;

        for (const device of attemptedDevices) {
            this.sendProgress({
                type: 'init_progress',
                current: 0,
                total: 100,
                message: `Loading NLI model on ${device}...`,
                _id,
            });
            try {
                this.classifier = await pipeline('text-classification', this.modelId, {
                    device,
                    progress_callback: (progress: { status?: string; loaded?: number; total?: number }) => {
                        if (
                            progress?.status === 'progress' &&
                            typeof progress.loaded === 'number' &&
                            typeof progress.total === 'number' &&
                            progress.total > 0
                        ) {
                            this.sendProgress({
                                type: 'init_progress',
                                current: progress.loaded,
                                total: progress.total,
                                message: `Loading NLI model on ${device}...`,
                                _id,
                            });
                        }
                    },
                });
                this.device = device;
                this.initialized = true;
                this.id2label = this.readId2Label();
                this.sendResponse({
                    type: 'INIT_COMPLETE',
                    payload: { device: this.device },
                    _id,
                });
                return;
            } catch (error) {
                lastError = error;
                this.classifier = null;
            }
        }

        throw lastError instanceof Error ? lastError : new Error('Failed to initialize NLI worker');
    }

    async *classifyStream(
        payload: ClassifyStreamPayload,
        _id: number,
    ): AsyncGenerator<{ results: NliClassificationResult[]; batchIndex: number; totalBatches: number }> {
        if (!this.initialized || !this.classifier) {
            throw new Error('Worker not initialized');
        }

        const { pairs, batchSize = 4 } = payload;
        const totalBatches = Math.ceil(pairs.length / batchSize);

        for (let index = 0; index < pairs.length; index += batchSize) {
            const batchPairs = pairs.slice(index, index + batchSize);
            const batchIndex = Math.floor(index / batchSize) + 1;

            this.sendProgress({
                type: 'classify_progress',
                current: batchIndex,
                total: totalBatches,
                message: `Evaluating NLI batch ${batchIndex}/${totalBatches}`,
                _id,
            });

            const pipelineInputs = buildNliPipelineBatch(
                batchPairs.map((pair) => ({
                    premise: pair.premise,
                    hypothesis: pair.hypothesis,
                })),
            );
            const output = await this.classifier(pipelineInputs, {
                top_k: 3,
            });

            const rows = Array.isArray(output) ? output : [output];
            const results = batchPairs.map((pair, pairIndex) => {
                const labels = Array.isArray(rows[pairIndex]) ? rows[pairIndex] : [rows[pairIndex]];
                const scores = normalizeNliScores(
                    labels
                        .filter((value: any) => value && typeof value.label === 'string')
                        .map((value: any) => ({
                            label: value.label as string,
                            score: typeof value.score === 'number' ? value.score : 0,
                        })),
                    this.id2label,
                );
                const predictedLabel = topNliLabel(scores);
                return {
                    ...pair,
                    entailment: scores.entailment,
                    neutral: scores.neutral,
                    contradiction: scores.contradiction,
                    predictedLabel,
                    confidence: scores[predictedLabel],
                };
            });

            yield {
                results,
                batchIndex,
                totalBatches,
            };

            await new Promise((resolve) => setTimeout(resolve, 5));
        }
    }

    async dispose(_id: number): Promise<void> {
        if (this.classifier && typeof this.classifier.dispose === 'function') {
            await this.classifier.dispose();
        }
        this.classifier = null;
        this.modelId = null;
        this.initialized = false;
        this.device = 'wasm';
        this.id2label = null;
        this.sendResponse({ type: 'DISPOSED', _id });
    }

    getStatus(): { initialized: boolean; modelId: string | null; device: string } {
        return {
            initialized: this.initialized,
            modelId: this.modelId,
            device: this.device,
        };
    }

    private readId2Label(): Record<string, string> | null {
        const config = this.classifier?.model?.config;
        const raw = config?.id2label;
        if (!raw || typeof raw !== 'object') {
            return null;
        }
        const map: Record<string, string> = {};
        for (const [key, value] of Object.entries(raw)) {
            if (typeof value === 'string') {
                map[String(key)] = value;
            }
        }
        return Object.keys(map).length ? map : null;
    }

    private sendResponse(message: ResponseMessage): void {
        self.postMessage(message);
    }

    private sendProgress(update: ProgressUpdate): void {
        self.postMessage(update);
    }
}

const worker = new NliWorker();

self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
    const { type, payload, _id } = event.data;

    try {
        switch (type) {
            case 'INIT':
                await worker.initialize(payload, _id);
                break;
            case 'CLASSIFY_STREAM': {
                const generator = worker.classifyStream(payload, _id);
                for await (const batch of generator) {
                    self.postMessage({
                        type: 'CLASSIFY_BATCH',
                        payload: batch,
                        _id,
                    } satisfies ResponseMessage);
                }
                self.postMessage({
                    type: 'CLASSIFY_COMPLETE',
                    _id,
                } satisfies ResponseMessage);
                break;
            }
            case 'GET_STATUS':
                self.postMessage({
                    type: 'STATUS',
                    payload: worker.getStatus(),
                    _id,
                } satisfies ResponseMessage);
                break;
            case 'DISPOSE':
                await worker.dispose(_id);
                break;
        }
    } catch (error) {
        self.postMessage({
            type: 'ERROR',
            payload: {
                message: error instanceof Error ? error.message : 'Unknown NLI worker error',
            },
            _id,
        } satisfies ResponseMessage);
    }
};
