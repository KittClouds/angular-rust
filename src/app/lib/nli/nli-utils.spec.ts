import { describe, expect, it } from 'vitest';

import {
    buildNliPipelineBatch,
    canonicalizeNliLabel,
    normalizeNliScores,
    topNliLabel,
} from './nli-utils';

describe('nli-utils', () => {
    it('builds text-pair classifier batches with text and text_pair fields', () => {
        expect(
            buildNliPipelineBatch([
                {
                    premise: 'turn [user]: Ryan waited by the harbor.',
                    hypothesis: 'This text is about Ryan.',
                },
            ]),
        ).toEqual([
            {
                text: 'turn [user]: Ryan waited by the harbor.',
                text_pair: 'This text is about Ryan.',
            },
        ]);
    });

    it('normalizes label strings into entailment neutral contradiction scores', () => {
        const scores = normalizeNliScores([
            { label: 'ENTAILMENT', score: 0.82 },
            { label: 'neutral', score: 0.11 },
            { label: 'contradiction', score: 0.07 },
        ]);

        expect(scores).toEqual({
            entailment: 0.82,
            neutral: 0.11,
            contradiction: 0.07,
        });
        expect(topNliLabel(scores)).toBe('entailment');
    });

    it('falls back through id2label when outputs use LABEL_n ids', () => {
        expect(
            canonicalizeNliLabel('LABEL_2', {
                '0': 'contradiction',
                '1': 'neutral',
                '2': 'entailment',
            }),
        ).toBe('entailment');
    });
});
