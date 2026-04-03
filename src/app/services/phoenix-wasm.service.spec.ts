import '@angular/compiler';
import {
    Injector,
    createEnvironmentInjector,
    runInInjectionContext,
    type EnvironmentInjector,
} from '@angular/core';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { EmbeddingWorkerService } from '../lib/services/embedding-worker.service';
import { NliWorkerService } from '../lib/services/nli-worker.service';
import { PhoenixWasmService } from './phoenix-wasm.service';

type MockEmbeddingWorker = Pick<EmbeddingWorkerService, 'initialize' | 'embedStream' | 'dispose'>;
type MockNliWorker = Pick<
    NliWorkerService,
    'initialize' | 'classifyStream' | 'getStatus' | 'dispose'
>;

function makeEmbedding(first: number, second = 0): number[] {
    const values = new Array<number>(384).fill(0);
    values[0] = first;
    values[1] = second;
    return values;
}

describe('PhoenixWasmService semantic sidecar', () => {
    let injector: EnvironmentInjector;
    let service: PhoenixWasmService;
    let embeddingWorkerMock: {
        initialize: ReturnType<typeof vi.fn>;
        embedStream: ReturnType<typeof vi.fn>;
        dispose: ReturnType<typeof vi.fn>;
    };
    let nliWorkerMock: {
        initialize: ReturnType<typeof vi.fn>;
        classifyStream: ReturnType<typeof vi.fn>;
        getStatus: ReturnType<typeof vi.fn>;
        dispose: ReturnType<typeof vi.fn>;
    };

    beforeEach(() => {
        embeddingWorkerMock = {
            initialize: vi.fn().mockResolvedValue(undefined),
            embedStream: vi.fn().mockImplementation(async (texts, onBatch) => {
                if (
                    JSON.stringify(texts) ===
                    JSON.stringify([
                        'Crimson harbor bells at dawn.',
                        'Fog rolled over the cranes.',
                        'Moonlit observatory above the desert.',
                    ])
                ) {
                    onBatch({
                        embeddings: [
                            makeEmbedding(3, 0),
                            makeEmbedding(0, 4),
                            makeEmbedding(5, 0),
                        ],
                        batchIndex: 0,
                        totalBatches: 1,
                    });
                    return;
                }
                expect(texts).toEqual([
                    'entity: Harbor bells\nsupport: Crimson harbor bells at dawn.',
                    'entity: Harbor bell tower\nsupport: Fog rolled over the cranes.',
                ]);
                onBatch({
                    embeddings: [makeEmbedding(2, 2), makeEmbedding(2, 1)],
                    batchIndex: 0,
                    totalBatches: 1,
                });
            }),
            dispose: vi.fn().mockResolvedValue(undefined),
        };
        nliWorkerMock = {
            initialize: vi.fn().mockResolvedValue(undefined),
            classifyStream: vi.fn().mockImplementation(async (pairs, onBatch) => {
                expect(pairs).toEqual([
                    {
                        judgmentId: 'candidate_corefers_with::entity::harbor-bells::entity::harbor-bells-alt::forward',
                        groupId: 'candidate_corefers_with::entity::harbor-bells::entity::harbor-bells-alt',
                        sourceId: 'entity::harbor-bells',
                        targetId: 'entity::harbor-bells-alt',
                        edgeType: 'candidate_corefers_with',
                        direction: 'forward',
                        premise: 'entity: Harbor bells\nsupport: Crimson harbor bells at dawn.',
                        hypothesis:
                            'This refers to the same entity as:\nentity: Harbor bell tower\nsupport: Fog rolled over the cranes.',
                    },
                    {
                        judgmentId: 'candidate_corefers_with::entity::harbor-bells::entity::harbor-bells-alt::reverse',
                        groupId: 'candidate_corefers_with::entity::harbor-bells::entity::harbor-bells-alt',
                        sourceId: 'entity::harbor-bells',
                        targetId: 'entity::harbor-bells-alt',
                        edgeType: 'candidate_corefers_with',
                        direction: 'reverse',
                        premise: 'entity: Harbor bell tower\nsupport: Fog rolled over the cranes.',
                        hypothesis:
                            'This refers to the same entity as:\nentity: Harbor bells\nsupport: Crimson harbor bells at dawn.',
                    },
                ]);
                onBatch({
                    results: [
                        {
                            ...pairs[0],
                            entailment: 0.82,
                            neutral: 0.12,
                            contradiction: 0.06,
                            predictedLabel: 'entailment',
                            confidence: 0.82,
                        },
                        {
                            ...pairs[1],
                            entailment: 0.79,
                            neutral: 0.14,
                            contradiction: 0.07,
                            predictedLabel: 'entailment',
                            confidence: 0.79,
                        },
                    ],
                    batchIndex: 1,
                    totalBatches: 1,
                });
            }),
            getStatus: vi.fn().mockResolvedValue({
                initialized: true,
                modelId: 'onnx-community/ModernBERT-base-nli-ONNX',
                device: 'wasm',
            }),
            dispose: vi.fn().mockResolvedValue(undefined),
        };

        injector = createEnvironmentInjector([
            { provide: EmbeddingWorkerService, useValue: embeddingWorkerMock satisfies MockEmbeddingWorker },
            { provide: NliWorkerService, useValue: nliWorkerMock satisfies MockNliWorker },
        ], Injector.create({ providers: [] }));

        service = runInInjectionContext(injector, () => new PhoenixWasmService());
    });

    afterEach(() => {
        injector.destroy();
    });

    it('derives document centroids from the streamed leaf embeddings without a second embedding pass', async () => {
        const storeCommand = vi
            .spyOn(service, 'storeCommand')
            .mockImplementation(async (command: string) => {
                if (command === 'semantic:listLeafChunks') {
                    return [
                        {
                            spanId: 'doc-1:1:0:0-29',
                            documentId: 'doc-1',
                            text: 'Crimson harbor bells at dawn.',
                        },
                        {
                            spanId: 'doc-1:2:0:30-58',
                            documentId: 'doc-1',
                            text: 'Fog rolled over the cranes.',
                        },
                        {
                            spanId: 'doc-2:1:0:0-36',
                            documentId: 'doc-2',
                            text: 'Moonlit observatory above the desert.',
                        },
                    ];
                }
                if (command === 'semantic:listCandidatePrototypeInputs') {
                    return [
                        {
                            nodeId: 'entity::harbor-bells',
                            nodeKind: 'entity',
                            documentId: 'doc-1',
                            narrativeId: 'nar-1',
                            folderId: 'folder-1',
                            text: 'entity: Harbor bells\nsupport: Crimson harbor bells at dawn.',
                            evidenceRefs: ['graph_vertex:entity::harbor-bells', 'chunk:doc-1:1:0:0-29'],
                        },
                        {
                            nodeId: 'entity::harbor-bells-alt',
                            nodeKind: 'entity',
                            documentId: 'doc-1',
                            narrativeId: 'nar-1',
                            folderId: 'folder-1',
                            text: 'entity: Harbor bell tower\nsupport: Fog rolled over the cranes.',
                            evidenceRefs: ['graph_vertex:entity::harbor-bells-alt', 'chunk:doc-1:2:0:30-58'],
                        },
                    ];
                }
                if (command === 'semantic:listNliJudgmentInputs') {
                    return [
                        {
                            judgmentId:
                                'candidate_corefers_with::entity::harbor-bells::entity::harbor-bells-alt::forward',
                            groupId: 'candidate_corefers_with::entity::harbor-bells::entity::harbor-bells-alt',
                            sourceId: 'entity::harbor-bells',
                            targetId: 'entity::harbor-bells-alt',
                            edgeType: 'candidate_corefers_with',
                            direction: 'forward',
                            premise: 'entity: Harbor bells\nsupport: Crimson harbor bells at dawn.',
                            hypothesis:
                                'This refers to the same entity as:\nentity: Harbor bell tower\nsupport: Fog rolled over the cranes.',
                        },
                        {
                            judgmentId:
                                'candidate_corefers_with::entity::harbor-bells::entity::harbor-bells-alt::reverse',
                            groupId: 'candidate_corefers_with::entity::harbor-bells::entity::harbor-bells-alt',
                            sourceId: 'entity::harbor-bells',
                            targetId: 'entity::harbor-bells-alt',
                            edgeType: 'candidate_corefers_with',
                            direction: 'reverse',
                            premise: 'entity: Harbor bell tower\nsupport: Fog rolled over the cranes.',
                            hypothesis:
                                'This refers to the same entity as:\nentity: Harbor bells\nsupport: Crimson harbor bells at dawn.',
                        },
                    ];
                }
                return { inserted: 2 };
            });
        const sendBytes = vi.spyOn(service as any, 'sendBytes').mockResolvedValue({
            kind: 0,
            requestId: 1,
            bytes: new Uint8Array(),
            json: { inserted: 3 },
        });

        await (service as any).indexCommittedSemanticDocuments(['doc-1', 'doc-2']);

        expect(embeddingWorkerMock.initialize).toHaveBeenCalledTimes(1);
        expect(embeddingWorkerMock.initialize).toHaveBeenCalledWith('mongodb-leaf');
        expect(embeddingWorkerMock.dispose).toHaveBeenCalledTimes(1);
        expect(embeddingWorkerMock.embedStream).toHaveBeenCalledTimes(2);
        expect(nliWorkerMock.initialize).toHaveBeenCalledWith('onnx-community/ModernBERT-base-nli-ONNX');
        expect(nliWorkerMock.classifyStream).toHaveBeenCalledTimes(1);
        expect(nliWorkerMock.dispose).toHaveBeenCalledTimes(1);
        expect(sendBytes).toHaveBeenCalledTimes(1);
        expect(storeCommand).toHaveBeenNthCalledWith(1, 'semantic:listLeafChunks', {
            documentIds: ['doc-1', 'doc-2'],
        });
        expect(storeCommand).toHaveBeenNthCalledWith(
            2,
            'semantic:upsertDocumentVectors',
            expect.any(Object),
        );
        expect(storeCommand).toHaveBeenNthCalledWith(3, 'semantic:listCandidatePrototypeInputs', {
            documentIds: ['doc-1', 'doc-2'],
        });
        expect(storeCommand).toHaveBeenNthCalledWith(
            4,
            'semantic:upsertPrototypeVectors',
            expect.any(Object),
        );
        expect(storeCommand).toHaveBeenNthCalledWith(5, 'semantic:refreshCandidateGraphEdges', {
            documentIds: ['doc-1', 'doc-2'],
            nodeIds: ['entity::harbor-bells', 'entity::harbor-bells-alt'],
        });
        expect(storeCommand).toHaveBeenNthCalledWith(6, 'semantic:listNliJudgmentInputs', {
            documentIds: ['doc-1', 'doc-2'],
            nodeIds: ['entity::harbor-bells', 'entity::harbor-bells-alt'],
        });
        expect(storeCommand).toHaveBeenNthCalledWith(
            7,
            'semantic:applyNliJudgments',
            expect.objectContaining({
                modelId: 'onnx-community/ModernBERT-base-nli-ONNX',
                device: 'wasm',
                results: expect.any(Array),
            }),
        );

        const rows = (storeCommand.mock.calls[1]?.[1] as {
            rows: Array<{ documentId: string; leafCount: number; values: number[]; evidenceRefs: string[] }>;
        }).rows;
        expect(rows).toHaveLength(2);
        expect(rows[0].documentId).toBe('doc-1');
        expect(rows[0].leafCount).toBe(2);
        expect(rows[0].values[0]).toBeCloseTo(Math.SQRT1_2, 5);
        expect(rows[0].values[1]).toBeCloseTo(Math.SQRT1_2, 5);
        expect(rows[0].evidenceRefs).toEqual(['chunk:doc-1:1:0:0-29', 'chunk:doc-1:2:0:30-58']);
        expect(rows[1].documentId).toBe('doc-2');
        expect(rows[1].leafCount).toBe(1);
        expect(rows[1].values[0]).toBeCloseTo(1, 5);
        expect(rows[1].values[1]).toBeCloseTo(0, 5);
        expect(rows[1].evidenceRefs).toEqual(['chunk:doc-2:1:0:0-36']);

        const prototypeRows = (storeCommand.mock.calls[3]?.[1] as {
            rows: Array<{ nodeId: string; nodeKind: string; values: number[]; evidenceRefs: string[] }>;
        }).rows;
        expect(prototypeRows).toEqual([
            {
                nodeId: 'entity::harbor-bells',
                nodeKind: 'entity',
                documentId: 'doc-1',
                narrativeId: 'nar-1',
                folderId: 'folder-1',
                evidenceRefs: ['graph_vertex:entity::harbor-bells', 'chunk:doc-1:1:0:0-29'],
                values: expect.any(Array),
            },
            {
                nodeId: 'entity::harbor-bells-alt',
                nodeKind: 'entity',
                documentId: 'doc-1',
                narrativeId: 'nar-1',
                folderId: 'folder-1',
                evidenceRefs: ['graph_vertex:entity::harbor-bells-alt', 'chunk:doc-1:2:0:30-58'],
                values: expect.any(Array),
            },
        ]);
    });
});
