import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Genome } from '../src/genome.js';
import type { GeneOptions, SeqInput } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput, type WeightInMode } from './helpers/randomGenome.js';
import { referenceGenomeCalculate, type ReferenceInput } from './helpers/referenceForward.js';

type Activation = 'sigmoid' | 'tanh' | 'identity';
type AlphaMode = 'one' | 'lt1' | 'generated';
type InputKind = 'matrix' | 'vector' | 'ragged' | 'numberRowInMiddle' | 'numberRowAtEnd';

interface ForwardCase {
    name: string;
    layers: number;
    H: number;
    F: number;
    T: number;
    mode: WeightInMode;
    outDim: number;
    activation: Activation;
    alpha: AlphaMode;
    input: InputKind;
}

const ACTIVATIONS: Activation[] = ['sigmoid', 'tanh', 'identity'];
const MODES: WeightInMode[] = ['all', 'none', 'mixed'];

function makeCase(c: Omit<ForwardCase, 'name'>): ForwardCase {
    const name =
        `${c.input} L${c.layers} H${c.H} F${c.F} T${c.T} weightIn=${c.mode} ` +
        `out=${c.outDim} ${c.activation} alpha=${c.alpha}`;

    return { name, ...c };
}

function buildCases(): ForwardCase[] {
    const cases: ForwardCase[] = [];
    let n = 0;

    for (const layers of [1, 2, 3, 4]) {
        for (const H of [1, 3, 17]) {
            for (const mode of MODES) {
                for (const F of [1, 36]) {
                    cases.push(
                        makeCase({
                            layers,
                            H,
                            F,
                            T: 8,
                            mode,
                            outDim: n % 2 ? 3 : 1,
                            activation: ACTIVATIONS[n % 3],
                            alpha: n % 4 === 0 ? 'lt1' : n % 4 === 1 ? 'one' : 'generated',
                            input: 'matrix',
                        }),
                    );
                    n++;
                }
            }
        }
    }

    for (const T of [0, 1, 8, 64]) {
        for (const layers of [1, 2, 4]) {
            for (const input of ['matrix', 'vector'] as const) {
                cases.push(
                    makeCase({
                        layers,
                        H: 3,
                        F: 36,
                        T,
                        mode: 'mixed',
                        outDim: 1,
                        activation: 'sigmoid',
                        alpha: 'lt1',
                        input,
                    }),
                );
            }
        }
    }

    for (const layers of [1, 2, 3, 4]) {
        for (const H of [1, 3, 17]) {
            cases.push(
                makeCase({
                    layers,
                    H,
                    F: 36,
                    T: 8,
                    mode: H === 3 ? 'none' : 'all',
                    outDim: H === 17 ? 3 : 1,
                    activation: ACTIVATIONS[H % 3],
                    alpha: 'lt1',
                    input: 'vector',
                }),
            );
        }
    }

    for (const layers of [1, 2, 3]) {
        for (const outDim of [1, 3]) {
            for (const activation of ACTIVATIONS) {
                for (const alpha of ['one', 'lt1'] as const) {
                    cases.push(
                        makeCase({
                            layers,
                            H: 3,
                            F: 36,
                            T: 8,
                            mode: 'all',
                            outDim,
                            activation,
                            alpha,
                            input: 'matrix',
                        }),
                    );
                }
            }
        }
    }

    for (const H of [60, 121]) {
        for (const layers of [1, 4]) {
            for (const mode of ['all', 'none'] as const) {
                cases.push(
                    makeCase({
                        layers,
                        H,
                        F: 36,
                        T: H === 121 ? 16 : 64,
                        mode,
                        outDim: layers === 4 ? 3 : 1,
                        activation: 'identity',
                        alpha: 'generated',
                        input: 'matrix',
                    }),
                );
            }
        }
    }

    for (const input of ['ragged', 'numberRowInMiddle', 'numberRowAtEnd'] as const) {
        for (const layers of [1, 3]) {
            for (const mode of MODES) {
                cases.push(
                    makeCase({ layers, H: 3, F: 36, T: 9, mode, outDim: 1, activation: 'tanh', alpha: 'lt1', input }),
                );
            }
        }
    }

    return cases;
}

function makeCaseInput(c: ForwardCase, rng: () => number): ReferenceInput {
    switch (c.input) {
        case 'vector':
            return Array.from({ length: c.T }, () => rng() * 2 - 1);
        case 'ragged':
            return Array.from({ length: c.T }, (_, t) =>
                Array.from({ length: t % 3 === 2 ? c.F + 3 : c.F }, () => rng() * 2 - 1),
            );
        case 'numberRowInMiddle': {
            const rows: (number | number[])[] = makeInput(rng, c.T, c.F);
            rows[Math.floor(c.T / 2)] = rng() * 2 - 1;

            return rows;
        }
        case 'numberRowAtEnd': {
            const rows: (number | number[])[] = makeInput(rng, c.T, c.F);
            rows[c.T - 1] = rng() * 2 - 1;

            return rows;
        }
        default:
            return makeInput(rng, c.T, c.F);
    }
}

function makeCaseOptions(c: ForwardCase, rng: () => number): GeneOptions {
    const options = makeGeneOptions(rng, c.layers, c.H, c.F, c.outDim, c.mode);
    const last = options[options.length - 1];
    if (c.alpha === 'one') last.alpha = 1;
    if (c.alpha === 'lt1') last.alpha = 0.37;

    return options;
}

function firstMismatch(a: readonly number[], b: readonly number[]): number {
    const n = Math.max(a.length, b.length);
    for (let i = 0; i < n; i++) {
        if (i >= a.length || i >= b.length || !Object.is(a[i], b[i])) return i;
    }

    return -1;
}

function weightInState(genome: Genome): (number[] | undefined)[] {
    const out: (number[] | undefined)[] = [];
    for (const lstm of genome.lstmArray) {
        for (const gate of [
            lstm['_forgetGate'],
            lstm['_potentialLongToRem'],
            lstm['_potentialLongMemory'],
            lstm['_shortMemoryToRemember'],
        ]) {
            for (const block of gate) out.push(block.weightIn);
        }
    }

    return out;
}

function expectSameState(a: Genome, b: Genome) {
    const wa = weightInState(a);
    const wb = weightInState(b);
    expect(wa.length).toBe(wb.length);
    for (let i = 0; i < wa.length; i++) {
        const x = wa[i];
        const y = wb[i];
        if (x === undefined || y === undefined) {
            expect(x, `weightIn ${i}`).toBe(y);
        } else {
            expect(firstMismatch(x, y), `weightIn ${i}`).toBe(-1);
        }
    }

    for (let l = 0; l < a.lstmArray.length; l++) {
        expect(firstMismatch(a.lstmArray[l].longMemory, b.lstmArray[l].longMemory), `longMemory L${l}`).toBe(-1);
        expect(firstMismatch(a.lstmArray[l].shortMemory, b.lstmArray[l].shortMemory), `shortMemory L${l}`).toBe(-1);
    }
}

const geneLstmCache = new Map<string, GeneLSTM>();

function geneLstmFor(c: ForwardCase): GeneLSTM {
    const key = `${c.F}|${c.outDim}|${c.activation}`;
    let glstm = geneLstmCache.get(key);
    if (!glstm) {
        glstm = new GeneLSTM(1, { INPUT_FEATURES: c.F, OUTPUT_DIM: c.outDim, OUTPUT_ACTIVATION: c.activation });
        geneLstmCache.set(key, glstm);
    }

    return glstm;
}

describe('forward pass matches the test-side reference', () => {
    const cases = buildCases();

    it.each(cases.map((c, index) => ({ ...c, index })))('$name', c => {
        const rng = mulberry32(1000 + c.index);
        const options = makeCaseOptions(c, rng);
        const input = makeCaseInput(c, rng);
        const json = JSON.stringify(options);
        const glstm = geneLstmFor(c);

        const built = withSeededRandom(1, () => ({
            a: new Genome(glstm, JSON.parse(json)),
            b: new Genome(glstm, JSON.parse(json)),
        }));
        expect(built.log).toHaveLength(0);
        const { a, b } = built.result;

        const seed = 5000 + c.index;
        const lib = withSeededRandom(seed, () => a.calculate(input as SeqInput));
        const ref = withSeededRandom(seed, () => referenceGenomeCalculate(b, input));

        expect(lib.result.length).toBe(ref.result.length);
        expect(firstMismatch(lib.result, ref.result), 'output').toBe(-1);
        expect(lib.result.every(Number.isFinite)).toBe(true);

        expect(lib.log.length).toBe(ref.log.length);
        expect(firstMismatch(lib.log, ref.log), 'Math.random log').toBe(-1);

        expectSameState(a, b);
    });

    it('lazy weightIn set-up uses 4·H·F Math.random calls on the first call and 0 on the second', () => {
        const rng = mulberry32(42);
        const glstm = new GeneLSTM(1, { INPUT_FEATURES: 36 });
        const genome = new Genome(glstm, makeGeneOptions(rng, 1, 1, 36, 1, 'none'));
        const input = makeInput(rng, 8, 36);

        const first = withSeededRandom(7, () => genome.calculate(input));
        const second = withSeededRandom(7, () => genome.calculate(input));

        expect(first.log).toHaveLength(4 * 1 * 36);
        expect(second.log).toHaveLength(0);
    });
});

describe('seeded evolution is deterministic', () => {
    const input = makeInput(mulberry32(7), 8, 4);

    const seededRun = (seed: number) =>
        withSeededRandom(seed, () => {
            const glstm = new GeneLSTM(30, { INPUT_FEATURES: 4 });
            for (let gen = 0; gen < 10; gen++) {
                for (const client of glstm.clients) {
                    const y = client.calculate(input);
                    client.score = 1 / (1 + Math.abs(y[0] - 0.75));
                }
                glstm.evolve();
            }

            return JSON.stringify(glstm.model());
        });

    it('two runs with the same seed give the same model() JSON and the same Math.random log', () => {
        const first = seededRun(123);
        const second = seededRun(123);

        expect(second.result).toBe(first.result);
        expect(second.log.length).toBe(first.log.length);
        expect(firstMismatch(second.log, first.log)).toBe(-1);

        expect(seededRun(124).result).not.toBe(first.result);
    });

    it('withSeededRandom restores Math.random also when the callback throws', () => {
        const original = Math.random;
        expect(() =>
            withSeededRandom(1, () => {
                throw new Error('boom');
            }),
        ).toThrow('boom');
        expect(Math.random).toBe(original);
    });
});
