import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import type { Client } from '../src/client.js';
import type { GeneOptions, LstmOptions } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput } from './helpers/randomGenome.js';

const F = 4;

function collectRefs(clients: Client[]): object[] {
    const refs: object[] = [];
    for (const c of clients) {
        const genome = c.genome;
        refs.push(genome, genome.lstmArray);
        for (const lstm of genome.lstmArray) {
            refs.push(lstm, lstm.readoutW, lstm.readoutB, lstm.longMemory, lstm.shortMemory, ...lstm.readoutW);
            const gates = [
                lstm['_forgetGate'],
                lstm['_potentialLongToRem'],
                lstm['_potentialLongMemory'],
                lstm['_shortMemoryToRemember'],
            ];
            for (const gate of gates) {
                refs.push(gate);
                for (const block of gate) {
                    refs.push(block);
                    if (block.weightIn) refs.push(block.weightIn);
                }
            }
        }
    }

    return refs;
}

function collectDataRefs(data: GeneOptions): object[] {
    const refs: object[] = [data];
    for (const layer of data) {
        refs.push(layer);
        if (Array.isArray(layer.readoutW)) refs.push(layer.readoutW, ...layer.readoutW.filter(Array.isArray));
        if (Array.isArray(layer.readoutB)) refs.push(layer.readoutB);
        const gates = [
            layer.forgetGate,
            layer.potentialLongToRem,
            layer.potentialLongMemory,
            layer.shortMemoryToRemember,
        ];
        for (const gate of gates) {
            refs.push(gate);
            for (const unit of gate) {
                refs.push(unit);
                if (unit.weightIn) refs.push(unit.weightIn);
            }
        }
    }

    return refs;
}

function expectNoDuplicates(refs: object[]) {
    const seen = new Set<object>();
    let duplicates = 0;
    for (const r of refs) {
        if (seen.has(r)) duplicates++;
        seen.add(r);
    }
    expect(duplicates).toBe(0);
}

function makeData(seed: number): GeneOptions {
    return makeGeneOptions(mulberry32(seed), 2, [3, 2], F, 1, 'all');
}

function layerState(c: Client): LstmOptions[] {
    return c.genome.lstmArray.map(l => l.copyOptions());
}

describe('loadData: own genome for each loaded client', () => {
    for (const loadPercent of [0.5, 1]) {
        it(`shares no Genome, LSTM, block or array (loadPercent ${loadPercent})`, () => {
            const loadData = makeData(1);
            const glstm = new GeneLSTM(10, { INPUT_FEATURES: F, loadData, loadPercent });

            const loaded = glstm.clients.filter((_, i) => i / 10 < loadPercent);
            expect(loaded.length).toBe(loadPercent * 10);
            for (const c of loaded) {
                expect(JSON.stringify(layerState(c))).toBe(JSON.stringify(layerState(loaded[0])));
            }

            expectNoDuplicates([...collectDataRefs(loadData), ...collectRefs(glstm.clients)]);
        });
    }

    it('does not write into the loadData object during training', () => {
        const loadData = makeData(2);
        const before = JSON.stringify(loadData);
        const inputs = [0, 1, 2].map(k => makeInput(mulberry32(100 + k), 8, F));

        withSeededRandom(7, () => {
            const glstm = new GeneLSTM(10, {
                INPUT_FEATURES: F,
                loadData,
                loadPercent: 1,
                PROBABILITY_MUTATE_WEIGHT_SHIFT: 1,
            });
            for (let gen = 0; gen < 30; gen++) {
                for (const c of glstm.clients) {
                    c.score = -inputs.reduce((s, x) => s + Math.abs(c.calculate(x)[0] - 0.3), 0);
                }
                glstm.evolve();
            }
        });

        expect(JSON.stringify(loadData)).toBe(before);
    });

    it('two instances built from one loadData object share no array', () => {
        const loadData = makeData(3);
        const a = new GeneLSTM(4, { INPUT_FEATURES: F, loadData, loadPercent: 1 });
        const b = new GeneLSTM(4, { INPUT_FEATURES: F, loadData, loadPercent: 1 });

        expectNoDuplicates([...collectRefs(a.clients), ...collectRefs(b.clients)]);
    });

    it('builds the loaded genome once: same Math.random calls as one loaded client', () => {
        const u = { weight1: 0.5, weight2: 0.4, bias: 0.1 };
        const incomplete: GeneOptions = [
            {
                hiddenSize: 3,
                forgetGate: [{ ...u }],
                potentialLongToRem: [{ ...u }],
                potentialLongMemory: [{ ...u }],
                shortMemoryToRemember: [{ ...u }],
                readoutW: [[0.1, 0.2, 0.3]],
                readoutB: [0],
                alpha: 0.8,
            },
        ];

        const one = withSeededRandom(11, () => new GeneLSTM(1, { loadData: incomplete, loadPercent: 1 }));
        const many = withSeededRandom(11, () => new GeneLSTM(10, { loadData: incomplete, loadPercent: 1 }));

        expect(one.log.length).toBeGreaterThan(0);
        expect(many.log).toEqual(one.log);

        const first = JSON.stringify(layerState(one.result.clients[0]));
        for (const c of many.result.clients) {
            expect(JSON.stringify(layerState(c))).toBe(first);
        }
    });

    it('keeps output settings of the loaded layers in each copy', () => {
        const unit = { weight1: 0.1, weight2: 0.2, bias: 0.3 };
        const gates = (H: number) => ({
            hiddenSize: H,
            forgetGate: Array.from({ length: H }, () => ({ ...unit })),
            potentialLongToRem: Array.from({ length: H }, () => ({ ...unit })),
            potentialLongMemory: Array.from({ length: H }, () => ({ ...unit })),
            shortMemoryToRemember: Array.from({ length: H }, () => ({ ...unit })),
            alpha: 0.5,
        });
        const cases: { OUTPUT_DIM?: number; data: GeneOptions; outLen: number }[] = [
            { data: [{ ...gates(2), readoutW: [[0.4, 0.5]], readoutB: [0.1], outputActivation: 'tanh' }], outLen: 1 },
            {
                data: [
                    {
                        ...gates(2),
                        readoutW: [
                            [0.4, 0.5],
                            [0.6, 0.7],
                        ],
                        readoutB: [0.1, 0.2],
                        outputDim: 2,
                    },
                ],
                outLen: 2,
            },
            { OUTPUT_DIM: 3, data: [{ ...gates(2), readoutW: [0.4, 0.5], readoutB: 0.1 }], outLen: 1 },
        ];
        const input = [0.3, 0.7, 0.1];

        for (const { OUTPUT_DIM, data, outLen } of cases) {
            const glstm = new GeneLSTM(4, { OUTPUT_DIM, loadData: data, loadPercent: 1 });
            const expected = glstm.clients[0].calculate(input);
            expect(expected).toHaveLength(outLen);
            for (const c of glstm.clients) {
                expect(c.calculate(input)).toEqual(expected);
            }
        }
    });

    it('data without weightIn: each copy draws its own weightIn at its first calculate()', () => {
        const loadData = makeGeneOptions(mulberry32(4), 1, 2, F, 1, 'none');
        const glstm = new GeneLSTM(2, { INPUT_FEATURES: F, loadData, loadPercent: 1 });
        const input = makeInput(mulberry32(5), 3, F);

        withSeededRandom(6, () => glstm.clients.forEach(c => c.calculate(input)));

        const w0 = glstm.clients[0].genome.lstmArray[0]['_forgetGate'][0].weightIn;
        const w1 = glstm.clients[1].genome.lstmArray[0]['_forgetGate'][0].weightIn;
        expect(w0).toHaveLength(F);
        expect(w1).toHaveLength(F);
        expect(w1).not.toBe(w0);
        expect(w1).not.toEqual(w0);
    });
});
