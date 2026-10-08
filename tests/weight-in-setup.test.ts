import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import type { LSTM } from '../src/lstm.js';
import type { GeneLSTMOptions, GeneOptions } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput } from './helpers/randomGenome.js';

const F = 5;
const N = 4;

function build(weightInSetup?: GeneLSTMOptions['weightInSetup'], extra?: GeneLSTMOptions) {
    return withSeededRandom(3, () => new GeneLSTM(N, { INPUT_FEATURES: F, weightInSetup, ...extra }));
}

function weightInLengths(data: GeneOptions): (number | undefined)[] {
    return data.flatMap(layer =>
        [
            ...layer.forgetGate,
            ...layer.potentialLongToRem,
            ...layer.potentialLongMemory,
            ...layer.shortMemoryToRemember,
        ].map(u => u.weightIn?.length),
    );
}

function unitCount(data: GeneOptions): number {
    return data.reduce((s, layer) => s + layer.hiddenSize, 0);
}

describe('weightInSetup option', () => {
    it("defaults to 'lazy': no weightIn before the first calculate(), same draws as explicit 'lazy'", () => {
        const byDefault = build();
        const lazy = build('lazy');

        expect(byDefault.result.weightInSetup).toBe('lazy');
        expect(byDefault.log).toEqual(lazy.log);
        for (const client of byDefault.result.clients) {
            expect(weightInLengths(client.model()).every(n => n === undefined)).toBe(true);
        }
    });

    it("'construct': a fresh population has weightIn of length INPUT_FEATURES in every unit", () => {
        const lazy = build('lazy');
        const construct = build('construct');

        for (const client of construct.result.clients) {
            const lengths = weightInLengths(client.model());
            expect(lengths.length).toBeGreaterThan(0);
            expect(lengths.every(n => n === F)).toBe(true);
        }
        const units = construct.result.clients.reduce((s, c) => s + unitCount(c.model()), 0);
        expect(construct.log.length - lazy.log.length).toBe(units * 4 * F);
    });

    it("'construct': the first calculate() with the matching width calls no Math.random", () => {
        const glstm = build('construct').result;
        const input = makeInput(mulberry32(9), 6, F);

        for (const client of glstm.clients) {
            const before = JSON.stringify(client.model());
            const { log } = withSeededRandom(4, () => client.calculate(input));
            expect(log).toEqual([]);
            expect(JSON.stringify(client.model())).toBe(before);
        }
    });

    it("'construct': units added by mutation and by size repair get weightIn", () => {
        const glstm = build('construct').result;
        const lstm: LSTM = glstm.clients[0].genome.lstmArray[0];

        withSeededRandom(5, () => {
            lstm['_mutateAddUnit']();
            lstm['_mutateAddUnit']();
        });
        expect(lstm.shortMemory.length).toBe(3);
        expect(weightInLengths([lstm.model()]).every(n => n === F)).toBe(true);

        lstm.readoutW[0].push(0.1);
        const model = withSeededRandom(7, () => lstm.model()).result;
        expect(model.hiddenSize).toBe(4);
        expect(weightInLengths([model]).every(n => n === F)).toBe(true);
    });

    it("'construct': loaded data without weightIn is set up at load time, once for all copies", () => {
        const data = makeGeneOptions(mulberry32(8), 2, [3, 2], F, 1, 'none');
        const glstm = withSeededRandom(
            2,
            () => new GeneLSTM(3, { INPUT_FEATURES: F, weightInSetup: 'construct', loadData: data, loadPercent: 1 }),
        ).result;

        const models = glstm.clients.map(c => JSON.stringify(c.model()));
        expect(new Set(models).size).toBe(1);
        expect(weightInLengths(glstm.clients[0].model()).every(n => n === F)).toBe(true);
        expect(weightInLengths(data).every(n => n === undefined)).toBe(true);

        const input = makeInput(mulberry32(9), 6, F);
        const outputs = glstm.clients.map(c => withSeededRandom(4, () => c.calculate(input)));
        for (const out of outputs) {
            expect(out.log).toEqual([]);
            out.result.forEach((v, i) => expect(Object.is(v, outputs[0].result[i])).toBe(true));
        }
    });

    it("'construct': two loads of model() right after construction give the same outputs", () => {
        const saved = build('construct').result.clients[0].model();
        const input = makeInput(mulberry32(10), 6, F);

        const run = (seed: number) =>
            withSeededRandom(seed, () => {
                const glstm = new GeneLSTM(1, { INPUT_FEATURES: F, loadData: saved, loadPercent: 1 });

                return glstm.clients[0].calculate(input);
            }).result;

        const a = run(11);
        const b = run(12);
        expect(a.length).toBe(b.length);
        a.forEach((v, i) => expect(Object.is(v, b[i])).toBe(true));
    });

    it("'construct': a different input width still replaces weightIn with random values", () => {
        const glstm = build('construct').result;
        const before = glstm.clients[0].model();
        expect(before.length).toBe(1);
        const { log } = withSeededRandom(4, () => glstm.clients[0].calculate(makeInput(mulberry32(9), 3, F + 1)));

        expect(log.length).toBe(unitCount(before) * 4 * (F + 1));
        expect(weightInLengths(glstm.clients[0].model()).every(n => n === F + 1)).toBe(true);
    });
});
