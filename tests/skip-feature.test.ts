import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import type { GeneLSTMOptions, GeneOptions } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput, type WeightInMode } from './helpers/randomGenome.js';
import { referenceGenomeCalculate } from './helpers/referenceForward.js';

const F = 5;
const ALPHA = 0.5;

function makeLoadData(layers: number, outDim: number, weightInMode: WeightInMode = 'all'): GeneOptions {
    const data = makeGeneOptions(mulberry32(17), layers, [3, 2, 4].slice(0, layers), F, outDim, weightInMode);
    for (const layer of data) layer.alpha = ALPHA;

    return data;
}

function build(loadData: GeneOptions, skipFeature?: GeneLSTMOptions['skipFeature']): GeneLSTM {
    const outDim = loadData[0].readoutW?.length ?? 1;

    return withSeededRandom(1, () => {
        return new GeneLSTM(2, {
            INPUT_FEATURES: F,
            OUTPUT_DIM: outDim,
            OUTPUT_ACTIVATION: 'identity',
            loadData: JSON.parse(JSON.stringify(loadData)),
            loadPercent: 1,
            skipFeature,
        });
    }).result;
}

function run(glstm: GeneLSTM, input: number[] | number[][]) {
    return withSeededRandom(11, () => glstm.clients[0].calculate(input));
}

function lastReadout(glstm: GeneLSTM): number[] {
    const layers = glstm.clients[0].genome.lstmArray;

    return layers[layers.length - 1]['_readout']();
}

function expectBitEqual(actual: number[], expected: number[]) {
    expect(actual.length).toBe(expected.length);
    actual.forEach((v, i) => expect(Object.is(v, expected[i])).toBe(true));
}

describe('skipFeature option', () => {
    it('defaults to 0: outputs and Math.random log equal to the reference', () => {
        for (const layers of [1, 2, 3]) {
            for (const weightInMode of ['all', 'none'] as const) {
                const loadData = makeLoadData(layers, 2, weightInMode);
                const input = makeInput(mulberry32(3 + layers), 6, F);
                const reference = build(loadData);
                const ref = withSeededRandom(11, () => referenceGenomeCalculate(reference.clients[0].genome, input));

                for (const glstm of [build(loadData), build(loadData, 0)]) {
                    expect(glstm.skipFeature).toBe(0);
                    const lib = run(glstm, input);
                    expectBitEqual(lib.result, ref.result);
                    expect(lib.log).toEqual(ref.log);
                }
            }
        }
    });

    it('a number selects the feature of the last step (2-D input, 1 layer)', () => {
        const loadData = makeLoadData(1, 3, 'none');
        const input = makeInput(mulberry32(5), 4, F);
        const none = run(build(loadData, 'none'), input);

        for (let k = 0; k < F; k++) {
            const lib = run(build(loadData, k), input);
            const expected = (1 - ALPHA) * input[input.length - 1][k] + ALPHA * none.result[0];
            expect(Object.is(lib.result[0], expected)).toBe(true);
            expectBitEqual(lib.result.slice(1), none.result.slice(1));
            expect(lib.log).toEqual(none.log);
            expect(lib.log.length).toBeGreaterThan(0);
        }
    });

    it("'none': output 0 is the readout y[0], other outputs and Math.random log do not change", () => {
        for (const layers of [1, 2]) {
            const loadData = makeLoadData(layers, 2, 'mixed');
            const input = makeInput(mulberry32(9), 5, F);
            const legacy = run(build(loadData), input);
            const glstm = build(loadData, 'none');
            const lib = run(glstm, input);

            expect(glstm.skipFeature).toBe('none');
            expectBitEqual(lib.result, lastReadout(glstm));
            expectBitEqual(lib.result.slice(1), legacy.result.slice(1));
            expect(lib.result[0]).not.toBe(legacy.result[0]);
            expect(lib.log).toEqual(legacy.log);
        }

        const scalar = build(makeLoadData(1, 1), 'none');
        const scalarOut = run(scalar, [0.3, -0.2, 0.9]).result;
        expectBitEqual(scalarOut, lastReadout(scalar));
    });

    it('a number has no effect for 1-D input and for genomes with 2+ layers', () => {
        const oneLayer = makeLoadData(1, 1);
        const scalarInput = [0.4, -0.7, 0.25, 0.8];
        expectBitEqual(run(build(oneLayer, 3), scalarInput).result, run(build(oneLayer), scalarInput).result);

        const twoLayers = makeLoadData(2, 1);
        const input = makeInput(mulberry32(21), 5, F);
        expectBitEqual(run(build(twoLayers, 3), input).result, run(build(twoLayers), input).result);
    });

    it('a number index out of the row gives a skip value of 0; values are clamped to integers >= 0', () => {
        const loadData = makeLoadData(1, 1);
        const input = makeInput(mulberry32(5), 4, F);
        const none = run(build(loadData, 'none'), input);
        const outside = run(build(loadData, F + 2), input);
        expect(Object.is(outside.result[0], (1 - ALPHA) * 0 + ALPHA * none.result[0])).toBe(true);

        expect(build(loadData, -3).skipFeature).toBe(0);
        expect(build(loadData, 2.9).skipFeature).toBe(2);
        expectBitEqual(run(build(loadData, 2.9), input).result, run(build(loadData, 2), input).result);
    });
});
