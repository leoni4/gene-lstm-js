import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Genome } from '../src/genome.js';
import { LSTM } from '../src/lstm.js';
import type { GeneOptions, SeqInput } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput, makeLayerOptions, type WeightInMode } from './helpers/randomGenome.js';
import { referenceGenomeCalculate, referenceLstmCalculate, type ReferenceInput } from './helpers/referenceForward.js';

type InputKind = 'matrix' | 'scalar';

const F = 5;
const HIDDEN = [3, 2, 5, 1];

function makeTestInput(kind: InputKind, rng: () => number, T: number): ReferenceInput {
    if (kind === 'matrix') return makeInput(rng, T, F);

    return Array.from({ length: T }, () => rng() * 2 - 1);
}

function firstMismatch(a: readonly number[], b: readonly number[]): number {
    const n = Math.max(a.length, b.length);
    for (let i = 0; i < n; i++) {
        if (i >= a.length || i >= b.length || !Object.is(a[i], b[i])) return i;
    }

    return -1;
}

function expectDense(values: readonly number[]) {
    for (let i = 0; i < values.length; i++) {
        expect(i in values, `index ${i} is a hole`).toBe(true);
    }
}

function expectSameLayerState(a: LSTM, b: LSTM, label: string) {
    const gatesA = [a['_forgetGate'], a['_potentialLongToRem'], a['_potentialLongMemory'], a['_shortMemoryToRemember']];
    const gatesB = [b['_forgetGate'], b['_potentialLongToRem'], b['_potentialLongMemory'], b['_shortMemoryToRemember']];
    for (let g = 0; g < gatesA.length; g++) {
        for (let k = 0; k < gatesA[g].length; k++) {
            const x = gatesA[g][k].weightIn;
            const y = gatesB[g][k].weightIn;
            if (x === undefined || y === undefined) {
                expect(x, `${label} gate ${g} unit ${k} weightIn`).toBe(y);
            } else {
                expect(firstMismatch(x, y), `${label} gate ${g} unit ${k} weightIn`).toBe(-1);
            }
        }
    }
    expect(firstMismatch(a.longMemory, b.longMemory), `${label} longMemory`).toBe(-1);
    expect(firstMismatch(a.shortMemory, b.shortMemory), `${label} shortMemory`).toBe(-1);
}

function expectSameRun(lib: { result: number[]; log: number[] }, ref: { result: number[]; log: number[] }) {
    expect(lib.result.length).toBe(ref.result.length);
    expect(firstMismatch(lib.result, ref.result), 'output').toBe(-1);
    expect(lib.log.length).toBe(ref.log.length);
    expect(firstMismatch(lib.log, ref.log), 'Math.random log').toBe(-1);
}

function twoGenomes(glstm: GeneLSTM, options: GeneOptions) {
    const json = JSON.stringify(options);
    const built = withSeededRandom(1, () => ({
        a: new Genome(glstm, JSON.parse(json)),
        b: new Genome(glstm, JSON.parse(json)),
    }));
    expect(built.log).toHaveLength(0);

    return built.result;
}

function twoLayers(glstm: GeneLSTM, H: number, outDim: number, mode: WeightInMode, seed: number) {
    const json = JSON.stringify(makeLayerOptions(mulberry32(seed), H, F, outDim, mode));

    return { a: new LSTM(glstm, JSON.parse(json)), b: new LSTM(glstm, JSON.parse(json)) };
}

const genomeCases = [2, 3, 4].flatMap(layers =>
    [1, 3].flatMap(outDim =>
        (['sigmoid', 'tanh', 'identity'] as const).flatMap(activation =>
            [0, 1, 64].flatMap(T =>
                (['matrix', 'scalar'] as const).map(input => ({
                    name: `${layers} layers out=${outDim} ${activation} T${T} ${input}`,
                    layers,
                    outDim,
                    activation,
                    T,
                    input,
                })),
            ),
        ),
    ),
);

describe('fused scalar layers and flat fullSeq readout match the test-side reference', () => {
    it.each(genomeCases)('$name', ({ layers, outDim, activation, T, input: kind }) => {
        const seed = 500 + layers * 100 + outDim * 10 + T;
        const rng = mulberry32(seed);
        const options = makeGeneOptions(rng, layers, HIDDEN.slice(0, layers), F, outDim, 'mixed');
        const input = makeTestInput(kind, rng, T);
        const glstm = new GeneLSTM(1, { INPUT_FEATURES: F, OUTPUT_DIM: outDim, OUTPUT_ACTIVATION: activation });
        const { a, b } = twoGenomes(glstm, options);

        const lib = withSeededRandom(seed, () => a.calculate(input as SeqInput));
        const ref = withSeededRandom(seed, () => referenceGenomeCalculate(b, input));

        expectSameRun(lib, ref);
        for (let l = 0; l < layers; l++) {
            expectSameLayerState(a.lstmArray[l], b.lstmArray[l], `L${l}`);
        }
    });

    const layerCases = [1, 3].flatMap(outDim =>
        [0, 1, 64].flatMap(T => (['matrix', 'scalar'] as const).map(input => ({ outDim, T, input }))),
    );

    it.each(layerCases)('LSTM.calculate(input, true): out=$outDim T$T $input', ({ outDim, T, input: kind }) => {
        const glstm = new GeneLSTM(1, { INPUT_FEATURES: F, OUTPUT_DIM: outDim, OUTPUT_ACTIVATION: 'tanh' });
        const { a, b } = twoLayers(glstm, 4, outDim, 'none', 70 + T);
        const input = makeTestInput(kind, mulberry32(80 + T), T);

        const lib = withSeededRandom(7, () => a.calculate(input as SeqInput, true));
        const ref = withSeededRandom(7, () => referenceLstmCalculate(b, input, true));

        expect(lib.result).toHaveLength(T * outDim);
        expectDense(lib.result);
        expectSameRun(lib, ref);
        expectSameLayerState(a, b, 'layer');
    });

    it('1-D input with array items sends each array item to the vector path', () => {
        const glstm = new GeneLSTM(1, { INPUT_FEATURES: F, OUTPUT_DIM: 2, OUTPUT_ACTIVATION: 'sigmoid' });
        const { a, b } = twoLayers(glstm, 3, 2, 'none', 91);
        const rng = mulberry32(92);
        const row = () => Array.from({ length: F }, () => rng() * 2 - 1);
        const input: ReferenceInput = [rng() - 0.5, row(), rng() - 0.5, row(), rng() - 0.5];

        const lib = withSeededRandom(8, () => a.calculate(input as SeqInput, true));
        const ref = withSeededRandom(8, () => referenceLstmCalculate(b, input, true));

        expect(lib.log).toHaveLength(4 * 3 * F);
        expectSameRun(lib, ref);
        expectSameLayerState(a, b, 'layer');
    });
});
