import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import type { GeneOptions, LstmOptions } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput } from './helpers/randomGenome.js';

const F = 4;

function outputs(glstm: GeneLSTM, inputs: number[][][]): number[][] {
    return inputs.map(x => glstm.clients[0].calculate(x));
}

function expectBitEqual(a: number[][], b: number[][]) {
    expect(a.length).toBe(b.length);
    for (let i = 0; i < a.length; i++) {
        expect(a[i].length).toBe(b[i].length);
        for (let j = 0; j < a[i].length; j++) expect(Object.is(a[i][j], b[i][j])).toBe(true);
    }
}

describe('model() saves the output settings', () => {
    const inputRng = mulberry32(11);
    const inputs = Array.from({ length: 5 }, () => makeInput(inputRng, 6, F));

    it('writes outputDim and outputActivation of each layer', () => {
        const glstm = new GeneLSTM(1, {
            INPUT_FEATURES: F,
            OUTPUT_DIM: 2,
            OUTPUT_ACTIVATION: 'tanh',
            loadData: makeGeneOptions(mulberry32(1), 2, [3, 2], F, 2, 'all'),
            loadPercent: 1,
        });

        for (const layer of glstm.clients[0].model()) {
            expect(layer.outputDim).toBe(2);
            expect(layer.outputActivation).toBe('tanh');
        }
    });

    it('a saved identity model loads with default options and gives bit-equal outputs', () => {
        withSeededRandom(3, () => {
            const source = new GeneLSTM(1, {
                INPUT_FEATURES: F,
                OUTPUT_DIM: 2,
                OUTPUT_ACTIVATION: 'identity',
                loadData: makeGeneOptions(mulberry32(2), 2, [3, 2], F, 2, 'all'),
                loadPercent: 1,
            });
            const saved = JSON.parse(JSON.stringify(source.clients[0].model())) as GeneOptions;

            const loaded = new GeneLSTM(1, { INPUT_FEATURES: F, loadData: saved, loadPercent: 1 });
            expectBitEqual(outputs(loaded, inputs), outputs(source, inputs));

            const stripped = saved.map(layer => {
                const copy: LstmOptions = { ...layer };
                delete copy.outputDim;
                delete copy.outputActivation;

                return copy;
            });
            const legacy = new GeneLSTM(1, { INPUT_FEATURES: F, loadData: stripped, loadPercent: 1 });
            const legacyOut = outputs(legacy, inputs);
            expect(legacyOut[0].length).toBe(1);
            expect(legacyOut[0][0]).not.toBe(outputs(source, inputs)[0][0]);
        });
    });

    it('JSON without the fields (1.0.10 format) uses the GeneLSTM options as before', () => {
        const data = makeGeneOptions(mulberry32(4), 1, 3, F, 2, 'all');
        expect(data[0].outputDim).toBeUndefined();

        const glstm = new GeneLSTM(1, {
            INPUT_FEATURES: F,
            OUTPUT_DIM: 3,
            OUTPUT_ACTIVATION: 'identity',
            loadData: data,
            loadPercent: 1,
        });
        const lstm = glstm.clients[0].genome.lstmArray[0];

        expect(lstm['_outputDim']).toBe(3);
        expect(lstm['_outputActivation']).toBe('identity');
        expect(glstm.clients[0].calculate(inputs[0]).length).toBe(3);
        expect(lstm.model()).toMatchObject({ outputDim: 3, outputActivation: 'identity' });
    });

    it('the old single-output format saves outputDim 1', () => {
        const [layer] = makeGeneOptions(mulberry32(5), 1, 3, F, 1, 'all');
        const old: LstmOptions = { ...layer, readoutW: (layer.readoutW as number[][])[0], readoutB: 0.1 };

        const glstm = new GeneLSTM(1, { INPUT_FEATURES: F, OUTPUT_DIM: 2, loadData: [old], loadPercent: 1 });

        expect(glstm.clients[0].model()[0].outputDim).toBe(1);
    });
});
