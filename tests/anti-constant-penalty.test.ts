import { describe, it, expect, vi } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import type { SeqInput } from '../src/types/index.js';
import { withSeededRandom } from './helpers/seededRandom.js';

const LAMBDA = 0.05;

function firstStep(input: SeqInput): number {
    const v = input[0];
    if (typeof v !== 'number') throw new Error('scalar input expected');

    return v;
}

function fitWithKnownPredictions(predict: (x: number) => number[], options: { shuffleEachEpoch?: boolean } = {}) {
    const glstm = new GeneLSTM(4);
    for (const client of glstm.clients) client.calculate = input => predict(firstStep(input));
    vi.spyOn(glstm, 'evolve').mockImplementation(() => undefined);

    const history = glstm.fit([[0], [1], [0], [1]], [0, 1, 0, 1], {
        epochs: 1,
        verbose: 0,
        loss: 'mae',
        antiConstantPenalty: true,
        antiConstantLambda: LAMBDA,
        ...options,
    });

    return { glstm, history };
}

describe('fit() antiConstantPenalty', () => {
    it('uses the real predictions of the epoch: [0, 1, 0, 1] → mean 0.5, variance 0.25', () => {
        const { glstm, history } = fitWithKnownPredictions(x => [x]);
        const expected = Math.min(10, 0 + LAMBDA * Math.abs(0.5 - 0.5) + (LAMBDA * 0.5) / (0.25 + 1e-6));

        expect(history.error[0]).toBe(expected);
        for (const client of glstm.clients) expect(client.error).toBe(expected);
    });

    it('adds the mean penalty to the loss', () => {
        const { history } = fitWithKnownPredictions(x => [0.5 + 0.5 * x], { shuffleEachEpoch: false });
        const expected = 0.25 + LAMBDA * 0.25 + (LAMBDA * 0.5) / (0.0625 + 1e-6);

        expect(history.error[0]).toBeCloseTo(expected, 12);
    });

    it('uses p[0] when the prediction is longer than the target', () => {
        const { history } = fitWithKnownPredictions(x => [x, 99]);
        const expected = (LAMBDA * 0.5) / (0.25 + 1e-6);

        expect(history.error[0]).toBe(expected);
    });

    it('still clamps a constant prediction to 10', () => {
        const { history } = fitWithKnownPredictions(() => [0.5]);

        expect(history.error[0]).toBe(10);
    });

    it('gives different client errors below 10 on the README lastBit data', () => {
        const inputs: number[][] = [];
        const outputs: number[] = [];
        for (let n = 0; n < 16; n++) {
            inputs.push([(n >> 3) & 1, (n >> 2) & 1, (n >> 1) & 1, n & 1]);
            outputs.push(n & 1);
        }

        const { result } = withSeededRandom(11, () => {
            const glstm = new GeneLSTM(100);
            const history = glstm.fit(inputs, outputs, { epochs: 100, verbose: 0, antiConstantPenalty: true });

            return { history, errors: glstm.clients.map(c => c.error) };
        });

        expect(result.history.error[result.history.error.length - 1]).toBeLessThan(10);
        expect(result.errors.filter(e => e < 10).length).toBeGreaterThan(result.errors.length / 2);
        expect(new Set(result.errors).size).toBeGreaterThan(1);
    });
});
