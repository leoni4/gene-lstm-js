import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { EMutationPressure } from '../src/types/index.js';

const { NORMAL, BOOST, ESCAPE, PANIC } = EMutationPressure;

const DEFAULT_FLAT_SEQUENCE: Array<[number, EMutationPressure]> = [
    [16, BOOST],
    [46, ESCAPE],
    [106, PANIC],
    [136, ESCAPE],
    [196, PANIC],
    [226, ESCAPE],
    [286, PANIC],
];

function makeFlatPressureGlstm(options?: ConstructorParameters<typeof GeneLSTM>[1]): GeneLSTM {
    const glstm = new GeneLSTM(2, options);
    glstm['_bestHistory'] = new Array(50).fill(0.5);
    glstm['_complexityHistory'] = new Array(50).fill(1.25);

    return glstm;
}

function runFlat(glstm: GeneLSTM, fromGen: number, toGen: number, fitness = 0.5): Array<[number, EMutationPressure]> {
    const changes: Array<[number, EMutationPressure]> = [];
    for (let gen = fromGen; gen <= toGen; gen++) {
        const before = glstm.mutationPressure;
        glstm.updateMutationPressure(fitness, gen);
        if (glstm.mutationPressure !== before) {
            changes.push([gen, glstm.mutationPressure]);
        }
    }

    return changes;
}

function improvedGenerations(glstm: GeneLSTM, fitnessAt: (gen: number) => number, toGen: number): number[] {
    const improved: number[] = [];
    for (let gen = 1; gen <= toGen; gen++) {
        glstm.updateMutationPressure(fitnessAt(gen), gen);
        if (glstm['_generationsSinceImprovement'] === 0) {
            improved.push(gen);
        }
    }

    return improved;
}

const range = (from: number, to: number) => Array.from({ length: to - from + 1 }, (_, i) => from + i);

const spikeAt10 = (gen: number) => (gen === 10 ? 2.0 : 0.5 + 0.01 * gen);

describe('stagnation reference: defaults', () => {
    it('resolves the legacy defaults', () => {
        const glstm = new GeneLSTM(2);
        expect(glstm.stagnationReference).toBe('bestEver');
        expect(glstm.stagnationWindow).toBe(50);
    });

    it('keeps the default level sequence when the new options are set to their default values', () => {
        const glstm = makeFlatPressureGlstm({ stagnationReference: 'bestEver', stagnationWindow: 50 });
        expect(runFlat(glstm, 1, 300)).toEqual(DEFAULT_FLAT_SEQUENCE);
    });

    it('does not fill the rolling window in the default mode', () => {
        const glstm = makeFlatPressureGlstm();
        runFlat(glstm, 1, 20);
        expect(glstm['_rollingBestHistory']).toEqual([]);
    });

    it('rounds stagnationWindow down with a minimum of 1', () => {
        expect(new GeneLSTM(2, { stagnationWindow: 7.9 }).stagnationWindow).toBe(7);
        expect(new GeneLSTM(2, { stagnationWindow: 0 }).stagnationWindow).toBe(1);
        expect(new GeneLSTM(2, { stagnationWindow: -3 }).stagnationWindow).toBe(1);
    });
});

describe("stagnationReference: 'rolling'", () => {
    it('gives the default level sequence with flat fitness and records each generation in the window', () => {
        const glstm = makeFlatPressureGlstm({ stagnationReference: 'rolling' });
        expect(runFlat(glstm, 1, 300)).toEqual(DEFAULT_FLAT_SEQUENCE);
        expect(glstm['_rollingBestHistory']).toEqual(new Array(50).fill(0.5));
    });

    it('stops using a lucky spike as the reference after stagnationWindow generations', () => {
        const bestEver = makeFlatPressureGlstm();
        expect(improvedGenerations(bestEver, spikeAt10, 60)).toEqual(range(1, 10));

        const rolling = makeFlatPressureGlstm({ stagnationReference: 'rolling', stagnationWindow: 5 });
        expect(improvedGenerations(rolling, spikeAt10, 60)).toEqual([...range(1, 10), ...range(16, 60)]);
        expect(rolling['_bestFitnessEver']).toBe(2.0);
    });

    it('uses the threshold relative to the rolling reference', () => {
        const glstm = makeFlatPressureGlstm({ stagnationReference: 'rolling', stagnationWindow: 1 });
        glstm.updateMutationPressure(10.0, 1);
        glstm.updateMutationPressure(1.0, 2);
        expect(glstm['_generationsSinceImprovement']).toBe(1);
        glstm.updateMutationPressure(1.0005, 3);
        expect(glstm['_generationsSinceImprovement']).toBe(2);
        glstm.updateMutationPressure(1.0016, 4);
        expect(glstm['_generationsSinceImprovement']).toBe(0);
    });

    it('lowers the pressure again when fitness rises above the recent window', () => {
        const fitnessAt = (gen: number) => (gen === 1 ? 1.0 : gen <= 16 ? 0.5 : 0.5 + 0.01 * (gen - 16));

        const bestEver = makeFlatPressureGlstm();
        const rolling = makeFlatPressureGlstm({ stagnationReference: 'rolling', stagnationWindow: 5 });

        const changes = (glstm: GeneLSTM) => {
            const out: Array<[number, EMutationPressure]> = [];
            for (let gen = 1; gen <= 60; gen++) {
                const before = glstm.mutationPressure;
                glstm.updateMutationPressure(fitnessAt(gen), gen);
                if (glstm.mutationPressure !== before) {
                    out.push([gen, glstm.mutationPressure]);
                }
            }

            return out;
        };

        expect(changes(bestEver)).toEqual([
            [16, BOOST],
            [46, ESCAPE],
        ]);
        expect(changes(rolling)).toEqual([
            [16, BOOST],
            [17, NORMAL],
        ]);
    });

    it('keeps a separate window when stagnationWindow is larger than the 50-generation history', () => {
        const fitnessAt = (luckyGen: number) => (gen: number) => (gen === 1 ? 1.0 : gen === luckyGen ? 0.6 : 0.5);

        const inWindow = makeFlatPressureGlstm({ stagnationReference: 'rolling', stagnationWindow: 80 });
        expect(improvedGenerations(inWindow, fitnessAt(81), 81)).toEqual([1]);
        expect(inWindow['_rollingBestHistory']).toHaveLength(80);

        const expired = makeFlatPressureGlstm({ stagnationReference: 'rolling', stagnationWindow: 80 });
        expect(improvedGenerations(expired, fitnessAt(82), 82)).toEqual([1, 82]);
    });
});
