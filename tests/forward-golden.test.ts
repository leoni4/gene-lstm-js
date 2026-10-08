import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import type { LstmOptions } from '../src/types/index.js';
import { withSeededRandom } from './helpers/seededRandom.js';

const INPUT = [
    [0.5, -1],
    [1, 0.25],
];
const C_T = -0.5351784449088207;
const H_T = -0.30188100281197916;
const S = -0.6528215042179688;
const GOLDEN = {
    sigmoid: 0.34235399976758674,
    tanh: -0.5735663204500124,
    identity: S,
} as const;

function layer(alpha: number): LstmOptions {
    return {
        hiddenSize: 1,
        forgetGate: [{ weight1: 0.3, weight2: 0.9, bias: 0.1, weightIn: [0.2, -0.4] }],
        potentialLongToRem: [{ weight1: -0.5, weight2: 0.9, bias: 0.05, weightIn: [0.6, 0.1] }],
        potentialLongMemory: [{ weight1: 0.7, weight2: 0.9, bias: -0.2, weightIn: [-0.3, 0.8] }],
        shortMemoryToRemember: [{ weight1: 0.4, weight2: 0.9, bias: 0.0, weightIn: [0.5, 0.25] }],
        readoutW: [[1.5]],
        readoutB: [-0.2],
        alpha,
    };
}

function run(activation: keyof typeof GOLDEN, alpha: number) {
    const glstm = new GeneLSTM(1, {
        INPUT_FEATURES: 2,
        OUTPUT_ACTIVATION: activation,
        loadData: [layer(alpha)],
        loadPercent: 1,
    });
    const client = glstm.clients[0];
    const { result, log } = withSeededRandom(1, () => client.calculate(INPUT));

    return { result, log, lstm: client.genome.lstmArray[0] };
}

describe('forward pass golden values (H = 1, T = 2, F = 2)', () => {
    for (const activation of ['sigmoid', 'tanh', 'identity'] as const) {
        it(`computes the hand-computed output with ${activation} output activation`, () => {
            const { result, log, lstm } = run(activation, 1);

            expect(result).toHaveLength(1);
            expect(result[0]).toBeCloseTo(GOLDEN[activation], 12);
            expect(lstm.longMemory[0]).toBeCloseTo(C_T, 12);
            expect(lstm.shortMemory[0]).toBeCloseTo(H_T, 12);
            expect(log).toHaveLength(0);
        });
    }

    it('mixes (1 − alpha) · x[last][0] into output 0 after the output activation (alpha 0.5)', () => {
        const xLast0 = INPUT[INPUT.length - 1][0];

        const identity = run('identity', 0.5).result;
        expect(identity[0]).toBeCloseTo(0.5 * xLast0 + 0.5 * S, 12);
        expect(identity[0]).toBeCloseTo(0.1735892478910156, 12);

        const sigmoid = run('sigmoid', 0.5).result;
        expect(sigmoid[0]).toBeCloseTo(0.5 * xLast0 + 0.5 * GOLDEN.sigmoid, 12);
    });
});
