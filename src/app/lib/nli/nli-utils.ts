export type PhoenixNliCanonicalLabel = 'entailment' | 'neutral' | 'contradiction';

export type PhoenixNliPairInput = {
    premise: string;
    hypothesis: string;
};

export type PhoenixNliPipelineInput = {
    text: string;
    text_pair: string;
};

export type PhoenixNliScoreMap = Record<PhoenixNliCanonicalLabel, number>;

export type PhoenixNliPipelineLabel = {
    label: string;
    score: number;
};

export function buildNliPipelineBatch(inputs: PhoenixNliPairInput[]): PhoenixNliPipelineInput[] {
    return inputs.map((input) => ({
        text: input.premise,
        text_pair: input.hypothesis,
    }));
}

export function normalizeNliScores(
    labels: PhoenixNliPipelineLabel[],
    id2label?: Record<string, string> | null,
): PhoenixNliScoreMap {
    const scores: PhoenixNliScoreMap = {
        entailment: 0,
        neutral: 0,
        contradiction: 0,
    };

    for (const item of labels) {
        const canonical = canonicalizeNliLabel(item.label, id2label);
        if (!canonical) {
            continue;
        }
        scores[canonical] = item.score;
    }

    return scores;
}

export function canonicalizeNliLabel(
    label: string,
    id2label?: Record<string, string> | null,
): PhoenixNliCanonicalLabel | null {
    const normalized = normalizeLabelText(label, id2label);
    if (!normalized) {
        return null;
    }
    if (normalized.includes('entail')) {
        return 'entailment';
    }
    if (normalized.includes('contrad')) {
        return 'contradiction';
    }
    if (normalized.includes('neutral')) {
        return 'neutral';
    }
    return null;
}

export function topNliLabel(scores: PhoenixNliScoreMap): PhoenixNliCanonicalLabel {
    return (Object.entries(scores).sort((left, right) => right[1] - left[1])[0]?.[0] ??
        'neutral') as PhoenixNliCanonicalLabel;
}

function normalizeLabelText(label: string, id2label?: Record<string, string> | null): string {
    const direct = label.trim().toLowerCase();
    if (direct && !/^label[_\s-]?\d+$/.test(direct)) {
        return direct;
    }

    const numericSuffix = direct.match(/(\d+)$/)?.[1];
    if (!numericSuffix || !id2label) {
        return direct;
    }

    const mapped =
        id2label[numericSuffix] ??
        id2label[`LABEL_${numericSuffix}`] ??
        id2label[`label_${numericSuffix}`];

    return typeof mapped === 'string' ? mapped.trim().toLowerCase() : direct;
}
