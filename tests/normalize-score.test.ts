import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import type { Client } from '../src/client.js';

function setup(
    rows: Array<{ raw: number; complexity: number }>,
    options?: ConstructorParameters<typeof GeneLSTM>[1],
): { glstm: GeneLSTM; byRow: Client[] } {
    const glstm = new GeneLSTM(rows.length, options);
    const byRow = [...glstm.clients];
    rows.forEach((row, i) => {
        byRow[i].scoreRaw = row.raw;
        byRow[i].complexity = row.complexity;
    });

    return { glstm, byRow };
}

function order(glstm: GeneLSTM, byRow: Client[]): number[] {
    return glstm.clients.map(c => byRow.indexOf(c));
}

describe('_normalizeScore()', () => {
    it('gives score 1 to all clients when raw scores and complexities are equal', () => {
        const { glstm, byRow } = setup([
            { raw: 0.3, complexity: 1.25 },
            { raw: 0.3, complexity: 1.25 },
            { raw: 0.3, complexity: 1.25 },
        ]);
        glstm['_normalizeScore']();

        for (const c of byRow) {
            expect(c.score).toBe(1);
            expect(c.adjustedScore).toBeCloseTo(0.3 - 0.01 * 0.05, 15);
        }
        expect(order(glstm, byRow)).toEqual([0, 1, 2]);
    });

    it('normalizes negative raw scores to [0, 1] and sorts by score', () => {
        const { glstm, byRow } = setup([
            { raw: -3, complexity: 1.25 },
            { raw: -1, complexity: 1.25 },
            { raw: -2, complexity: 1.25 },
        ]);
        glstm['_normalizeScore']();

        expect(byRow.map(c => c.adjustedScore)).toEqual(
            [-3 - 0.02, -1 - 0.02, -2 - 0.02].map(v => expect.closeTo(v, 12)),
        );
        expect(byRow.map(c => c.score)).toEqual([0, 1, 0.5].map(v => expect.closeTo(v, 12)));
        expect(order(glstm, byRow)).toEqual([1, 2, 0]);
    });

    it('uses a span floor of 0.05 for the complexity penalty when raw scores are equal', () => {
        const { glstm, byRow } = setup([
            { raw: 1, complexity: 3 },
            { raw: 1, complexity: 0 },
        ]);
        glstm['_normalizeScore']();

        expect(byRow[0].adjustedScore).toBeCloseTo(1 - 0.01 * 1 * 0.05, 15);
        expect(byRow[1].adjustedScore).toBe(1);
        expect(byRow[0].score).toBe(0);
        expect(byRow[1].score).toBe(1);
        expect(order(glstm, byRow)).toEqual([1, 0]);
    });

    it('sorts by lower complexity when scores differ by EPS or less', () => {
        const { glstm, byRow } = setup(
            [
                { raw: 1 + 2e-7, complexity: 5 },
                { raw: 1, complexity: 1 },
                { raw: 1 + 1e-7, complexity: 3 },
            ],
            { LAMBDA_LOW: 0 },
        );
        glstm['_normalizeScore']();

        expect(byRow.map(c => c.score)).toEqual([1, 1, 1]);
        expect(order(glstm, byRow)).toEqual([1, 2, 0]);
    });

    it('sorts by score before complexity when scores differ by more than EPS', () => {
        const { glstm, byRow } = setup(
            [
                { raw: 0, complexity: 1 },
                { raw: 1, complexity: 9 },
            ],
            { LAMBDA_LOW: 0 },
        );
        glstm['_normalizeScore']();

        expect(order(glstm, byRow)).toEqual([1, 0]);
    });

    for (const [mode, optimization, lambda] of [
        ['LAMBDA_LOW', false, 0.01],
        ['LAMBDA_HIGH', true, 0.1],
    ] as const) {
        it(`uses ${mode} (${lambda}) when _optimization is ${optimization}`, () => {
            const { glstm, byRow } = setup([
                { raw: 1, complexity: 3 },
                { raw: 0.5, complexity: 1 },
                { raw: 0, complexity: 0 },
            ]);
            glstm['_optimization'] = optimization;
            glstm['_normalizeScore']();

            expect(byRow[0].adjustedScore).toBeCloseTo(1 - lambda, 15);
            expect(byRow[1].adjustedScore).toBeCloseTo(0.5 - lambda * 0.5, 15);
            expect(byRow[2].adjustedScore).toBe(0);
            expect(byRow[0].score).toBe(1);
            expect(byRow[1].score).toBeCloseTo(0.5, 15);
            expect(byRow[2].score).toBe(0);
        });
    }
});
