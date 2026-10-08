import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Genome } from '../src/genome.js';
import { LSTM } from '../src/lstm.js';
import type { GeneOptions, LstmOptions, SeqInput } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput, makeLayerOptions, type WeightInMode } from './helpers/randomGenome.js';
import { referenceGenomeCalculate, referenceLstmCalculate, type ReferenceInput } from './helpers/referenceForward.js';

type Activation = 'sigmoid' | 'tanh' | 'identity';
type InputKind = 'matrix' | 'ragged' | 'numberRow';

const ACTIVATIONS: Activation[] = ['sigmoid', 'tanh', 'identity'];
const MODES: WeightInMode[] = ['all', 'none', 'mixed'];
const INPUT_KINDS: InputKind[] = ['matrix', 'ragged', 'numberRow'];
const HIDDEN_SIZES = [1, 2, 3, 5, 8, 17, 60, 121];

interface FuzzCase {
    name: string;
    seed: number;
    H: number[];
    F: number;
    T: number;
    mode: WeightInMode;
    outDim: number;
    activation: Activation;
    input: InputKind;
}

function pick<T>(rng: () => number, values: readonly T[]): T {
    return values[Math.floor(rng() * values.length)];
}

function buildFuzzCases(count: number): FuzzCase[] {
    const rng = mulberry32(2024);
    const cases: FuzzCase[] = [];
    for (let n = 0; n < count; n++) {
        const layers = 1 + Math.floor(rng() * 4);
        const H = Array.from({ length: layers }, () => pick(rng, HIDDEN_SIZES));
        const c = {
            seed: 9000 + n,
            H,
            F: 1 + Math.floor(rng() * 40),
            T: Math.max(...H) > 17 ? 1 + Math.floor(rng() * 8) : 1 + Math.floor(rng() * 24),
            mode: pick(rng, MODES),
            outDim: 1 + Math.floor(rng() * 3),
            activation: pick(rng, ACTIVATIONS),
            input: pick(rng, INPUT_KINDS),
        };
        const name =
            `#${n} ${c.input} H=[${c.H.join(',')}] F${c.F} T${c.T} weightIn=${c.mode} ` +
            `out=${c.outDim} ${c.activation}`;
        cases.push({ name, ...c });
    }

    return cases;
}

function makeFuzzInput(c: FuzzCase, rng: () => number): ReferenceInput {
    if (c.input === 'ragged') {
        return Array.from({ length: c.T }, () =>
            Array.from({ length: c.F + Math.floor(rng() * 3) - 1 || 1 }, () => rng() * 2 - 1),
        );
    }
    const rows: (number | number[])[] = makeInput(rng, c.T, c.F);
    if (c.input === 'numberRow' && c.T > 1) rows[1 + Math.floor(rng() * (c.T - 1))] = rng() * 2 - 1;

    return rows;
}

function firstMismatch(a: readonly number[], b: readonly number[]): number {
    const n = Math.max(a.length, b.length);
    for (let i = 0; i < n; i++) {
        if (i >= a.length || i >= b.length || !Object.is(a[i], b[i])) return i;
    }

    return -1;
}

function gates(lstm: LSTM) {
    return [
        lstm['_forgetGate'],
        lstm['_potentialLongToRem'],
        lstm['_potentialLongMemory'],
        lstm['_shortMemoryToRemember'],
    ];
}

function expectSameLayerState(a: LSTM, b: LSTM, label: string) {
    const ga = gates(a);
    const gb = gates(b);
    for (let g = 0; g < ga.length; g++) {
        expect(ga[g].length).toBe(gb[g].length);
        for (let k = 0; k < ga[g].length; k++) {
            const x = ga[g][k].weightIn;
            const y = gb[g][k].weightIn;
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

function expectSameGenomeState(a: Genome, b: Genome) {
    expect(a.lstmArray.length).toBe(b.lstmArray.length);
    for (let l = 0; l < a.lstmArray.length; l++) {
        expectSameLayerState(a.lstmArray[l], b.lstmArray[l], `L${l}`);
    }
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

describe('fused vector forward pass matches the test-side reference', () => {
    it.each(buildFuzzCases(60))('$name', c => {
        const rng = mulberry32(c.seed);
        const options = makeGeneOptions(rng, c.H.length, c.H, c.F, c.outDim, c.mode);
        const input = makeFuzzInput(c, rng);
        const glstm = new GeneLSTM(1, { INPUT_FEATURES: c.F, OUTPUT_DIM: c.outDim, OUTPUT_ACTIVATION: c.activation });
        const { a, b } = twoGenomes(glstm, options);

        const lib = withSeededRandom(c.seed, () => a.calculate(input as SeqInput));
        const ref = withSeededRandom(c.seed, () => referenceGenomeCalculate(b, input));

        expectSameRun(lib, ref);
        expectSameGenomeState(a, b);
    });

    it('repeated calls with a change of input width re-randomize weightIn in the same order', () => {
        const rng = mulberry32(31);
        const glstm = new GeneLSTM(1, { INPUT_FEATURES: 36, OUTPUT_DIM: 2, OUTPUT_ACTIVATION: 'tanh' });
        const { a, b } = twoGenomes(glstm, makeGeneOptions(rng, 2, [5, 3], 36, 2, 'mixed'));
        const inputs = [makeInput(rng, 6, 36), makeInput(rng, 6, 12), makeInput(rng, 6, 12), makeInput(rng, 4, 36)];
        const expectedLogLengths: number[] = [];

        inputs.forEach((input, call) => {
            const lib = withSeededRandom(100 + call, () => a.calculate(input));
            const ref = withSeededRandom(100 + call, () => referenceGenomeCalculate(b, input));
            expectSameRun(lib, ref);
            expectSameGenomeState(a, b);
            expectedLogLengths.push(lib.log.length);
        });

        expect(expectedLogLengths[1]).toBe(4 * 5 * 12);
        expect(expectedLogLengths[2]).toBe(0);
        expect(expectedLogLengths[3]).toBe(4 * 5 * 36);
    });

    it('LSTM.calculate(input, true) returns the same per-step readout as the reference', () => {
        const rng = mulberry32(57);
        const glstm = new GeneLSTM(1, { INPUT_FEATURES: 7, OUTPUT_DIM: 3, OUTPUT_ACTIVATION: 'sigmoid' });
        const options: LstmOptions = makeLayerOptions(rng, 4, 7, 3, 'none');
        const json = JSON.stringify(options);
        const a = new LSTM(glstm, JSON.parse(json));
        const b = new LSTM(glstm, JSON.parse(json));
        const input = makeInput(rng, 10, 7);

        const lib = withSeededRandom(3, () => a.calculate(input, true));
        const ref = withSeededRandom(3, () => referenceLstmCalculate(b, input, true));

        expect(lib.result).toHaveLength(10 * 3);
        expectSameRun(lib, ref);
        expectSameLayerState(a, b, 'layer');
    });
});
