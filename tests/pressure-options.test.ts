import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { EMutationPressure, MUTATION_PRESSURE_CONST } from '../src/types/index.js';

const { COMPACT, NORMAL, BOOST, ESCAPE, PANIC } = EMutationPressure;
const ALL_LEVELS = [COMPACT, NORMAL, BOOST, ESCAPE, PANIC];

const DEFAULT_FLAT_SEQUENCE: Array<[number, EMutationPressure]> = [
    [16, BOOST],
    [46, ESCAPE],
    [106, PANIC],
    [136, ESCAPE],
    [196, PANIC],
    [226, ESCAPE],
    [286, PANIC],
];

function makeFlatPressureGlstm(
    options?: ConstructorParameters<typeof GeneLSTM>[1],
    complexityHistory: number[] = new Array(50).fill(1.25),
): GeneLSTM {
    const glstm = new GeneLSTM(2, options);
    glstm['_bestHistory'] = new Array(50).fill(0.5);
    glstm['_complexityHistory'] = complexityHistory;

    return glstm;
}

function growingComplexity(): number[] {
    return Array.from({ length: 50 }, (_, i) => (i < 49 ? 1.25 : 10));
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

function multipliersAt(glstm: GeneLSTM, level: EMutationPressure) {
    glstm.mutationPressure = level;

    return glstm.getMutationPressure();
}

describe('pressure options: defaults', () => {
    it('resolves the legacy defaults', () => {
        const glstm = new GeneLSTM(2);
        expect(glstm.pressureTopologyBoost).toBe(true);
        expect(glstm.panicMaxGenerations).toBe(30);
        expect(glstm.panicCooldownGenerations).toBe(60);
        expect(glstm.compactTrigger).toBe('levelCounter');
    });

    it('returns the exported MUTATION_PRESSURE_CONST entries (same objects) by default', () => {
        const glstm = new GeneLSTM(2);
        for (const level of ALL_LEVELS) {
            expect(multipliersAt(glstm, level)).toBe(MUTATION_PRESSURE_CONST[level]);
        }
    });

    it('keeps the default level sequence when the new options are set to their default values', () => {
        const glstm = makeFlatPressureGlstm({
            pressureTopologyBoost: true,
            panicMaxGenerations: 30,
            panicCooldownGenerations: 60,
            compactTrigger: 'levelCounter',
        });
        expect(runFlat(glstm, 1, 300)).toEqual(DEFAULT_FLAT_SEQUENCE);
    });
});

describe('pressureTopologyBoost: false', () => {
    it('uses topology 1 for BOOST / ESCAPE / PANIC and keeps the weight multipliers and COMPACT', () => {
        const glstm = new GeneLSTM(2, { pressureTopologyBoost: false });
        expect(glstm.pressureTopologyBoost).toBe(false);

        expect(multipliersAt(glstm, COMPACT)).toEqual({ topology: 0.1, weights: 0.8 });
        expect(multipliersAt(glstm, NORMAL)).toEqual({ topology: 1, weights: 1 });
        expect(multipliersAt(glstm, BOOST)).toEqual({ topology: 1, weights: 1.5 });
        expect(multipliersAt(glstm, ESCAPE)).toEqual({ topology: 1, weights: 2 });
        expect(multipliersAt(glstm, PANIC)).toEqual({ topology: 1, weights: 4 });
    });

    it('does not change the exported table or other instances', () => {
        const before = JSON.stringify(MUTATION_PRESSURE_CONST);
        new GeneLSTM(2, { pressureTopologyBoost: false });
        const other = new GeneLSTM(2);

        expect(JSON.stringify(MUTATION_PRESSURE_CONST)).toBe(before);
        expect(multipliersAt(other, PANIC)).toEqual({ topology: 2, weights: 4 });
    });

    it('returns the same precomputed object on each call', () => {
        const glstm = new GeneLSTM(2, { pressureTopologyBoost: false });
        glstm.mutationPressure = PANIC;
        expect(glstm.getMutationPressure()).toBe(glstm.getMutationPressure());
    });

    it('does not change the level sequence', () => {
        const glstm = makeFlatPressureGlstm({ pressureTopologyBoost: false });
        expect(runFlat(glstm, 1, 300)).toEqual(DEFAULT_FLAT_SEQUENCE);
    });

    it('keeps COMPACT reachable', () => {
        const glstm = makeFlatPressureGlstm({ pressureTopologyBoost: false }, growingComplexity());
        expect(runFlat(glstm, 1, 100)).toEqual([
            [16, BOOST],
            [46, ESCAPE],
            [97, COMPACT],
        ]);
        expect(glstm.getMutationPressure()).toEqual({ topology: 0.1, weights: 0.8 });
    });
});

describe('panicMaxGenerations and panicCooldownGenerations', () => {
    it('ends PANIC after panicMaxGenerations generations', () => {
        const glstm = makeFlatPressureGlstm({ panicMaxGenerations: 10 });
        expect(runFlat(glstm, 1, 200)).toEqual([
            [16, BOOST],
            [46, ESCAPE],
            [106, PANIC],
            [116, ESCAPE],
            [176, PANIC],
            [186, ESCAPE],
        ]);
    });

    it('blocks ESCAPE → PANIC once with a cooldown of 61 and the default threshold', () => {
        const glstm = makeFlatPressureGlstm({ panicCooldownGenerations: 61 });
        expect(runFlat(glstm, 1, 136)).toEqual(DEFAULT_FLAT_SEQUENCE.slice(0, 4));
        expect(glstm['_panicCooldownCounter']).toBe(61);

        runFlat(glstm, 137, 196);
        expect(glstm.mutationPressure).toBe(ESCAPE);
        expect(glstm['_stagnationCounter']).toBe(0);

        expect(runFlat(glstm, 197, 300)).toEqual([
            [256, PANIC],
            [286, ESCAPE],
        ]);
    });

    it('gives the default sequence with a cooldown of 0', () => {
        const glstm = makeFlatPressureGlstm({ panicCooldownGenerations: 0 });
        expect(runFlat(glstm, 1, 300)).toEqual(DEFAULT_FLAT_SEQUENCE);
    });

    it('rounds down and applies the minimum values', () => {
        const low = new GeneLSTM(2, { panicMaxGenerations: 0, panicCooldownGenerations: -5 });
        expect(low.panicMaxGenerations).toBe(1);
        expect(low.panicCooldownGenerations).toBe(0);

        const fractional = new GeneLSTM(2, { panicMaxGenerations: 12.9, panicCooldownGenerations: 70.5 });
        expect(fractional.panicMaxGenerations).toBe(12);
        expect(fractional.panicCooldownGenerations).toBe(70);
    });
});

describe("compactTrigger: 'sinceImprovement'", () => {
    it("starts COMPACT only in the ESCAPE window with 'levelCounter' (default)", () => {
        const glstm = makeFlatPressureGlstm(undefined, growingComplexity());
        expect(runFlat(glstm, 1, 100)).toEqual([
            [16, BOOST],
            [46, ESCAPE],
            [97, COMPACT],
        ]);
    });

    it('starts COMPACT after 51 generations without improvement at any level', () => {
        const glstm = makeFlatPressureGlstm({ compactTrigger: 'sinceImprovement' }, growingComplexity());
        expect(glstm.compactTrigger).toBe('sinceImprovement');

        expect(runFlat(glstm, 1, 51)).toEqual([
            [16, BOOST],
            [46, ESCAPE],
        ]);
        runFlat(glstm, 52, 52);
        expect(glstm.mutationPressure).toBe(COMPACT);
        expect(glstm['_stagnationCounter']).toBe(6);
        expect(glstm['_optimization']).toBe(true);
    });

    it('resets the counter on improvement only', () => {
        const glstm = makeFlatPressureGlstm({ compactTrigger: 'sinceImprovement' }, growingComplexity());
        runFlat(glstm, 1, 52);
        expect(glstm.mutationPressure).toBe(COMPACT);

        glstm.updateMutationPressure(0.9, 53);
        expect(glstm.mutationPressure).toBe(NORMAL);
        expect(glstm['_generationsSinceImprovement']).toBe(0);

        expect(runFlat(glstm, 54, 120, 0.9)).toEqual([
            [68, BOOST],
            [98, ESCAPE],
            [104, COMPACT],
        ]);
    });

    it('counts the PANIC timeout generation as stagnant and keeps the default sequence without growth', () => {
        const glstm = makeFlatPressureGlstm({ compactTrigger: 'sinceImprovement' });
        expect(runFlat(glstm, 1, 136)).toEqual(DEFAULT_FLAT_SEQUENCE.slice(0, 4));
        expect(glstm['_generationsSinceImprovement']).toBe(135);

        expect(runFlat(glstm, 137, 300)).toEqual(DEFAULT_FLAT_SEQUENCE.slice(4));
    });
});
