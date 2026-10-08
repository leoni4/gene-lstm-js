import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { EMutationPressure } from '../src/types/index.js';

const { NORMAL, BOOST, ESCAPE, PANIC } = EMutationPressure;

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

describe('updateMutationPressure() state machine (default options)', () => {
    it('escalates NORMAL → BOOST → ESCAPE → PANIC and cycles ESCAPE ↔ PANIC with flat fitness', () => {
        const glstm = makeFlatPressureGlstm();
        expect(glstm.mutationPressure).toBe(NORMAL);

        expect(runFlat(glstm, 1, 300)).toEqual([
            [16, BOOST],
            [46, ESCAPE],
            [106, PANIC],
            [136, ESCAPE],
            [196, PANIC],
            [226, ESCAPE],
            [286, PANIC],
        ]);
    });

    it('ends PANIC after 30 generations and sets a cooldown of 60 that never blocks the next PANIC', () => {
        const glstm = makeFlatPressureGlstm();

        runFlat(glstm, 1, 105);
        expect(glstm.mutationPressure).toBe(ESCAPE);
        expect(glstm['_panicCooldownCounter']).toBe(0);

        runFlat(glstm, 106, 106);
        expect(glstm.mutationPressure).toBe(PANIC);

        runFlat(glstm, 107, 135);
        expect(glstm.mutationPressure).toBe(PANIC);
        expect(glstm['_panicCounter']).toBe(29);

        runFlat(glstm, 136, 136);
        expect(glstm.mutationPressure).toBe(ESCAPE);
        expect(glstm['_panicCounter']).toBe(0);
        expect(glstm['_panicCooldownCounter']).toBe(60);

        runFlat(glstm, 137, 195);
        expect(glstm.mutationPressure).toBe(ESCAPE);
        expect(glstm['_panicCooldownCounter']).toBe(1);
        expect(glstm['_stagnationCounter']).toBe(59);

        runFlat(glstm, 196, 196);
        expect(glstm.mutationPressure).toBe(PANIC);
        expect(glstm['_panicCooldownCounter']).toBe(0);
    });

    it('steps down one level for each improvement until NORMAL', () => {
        const glstm = makeFlatPressureGlstm();
        runFlat(glstm, 1, 106);
        expect(glstm.mutationPressure).toBe(PANIC);

        const levels: EMutationPressure[] = [];
        for (const fitness of [0.6, 0.7, 0.8, 0.9]) {
            glstm.updateMutationPressure(fitness);
            levels.push(glstm.mutationPressure);
            expect(glstm['_stagnationCounter']).toBe(0);
            expect(glstm['_panicCounter']).toBe(0);
        }

        expect(levels).toEqual([ESCAPE, BOOST, NORMAL, NORMAL]);
    });

    it('needs an improvement larger than max(1e-6, |best| · 1e-3)', () => {
        const glstm = makeFlatPressureGlstm();
        runFlat(glstm, 1, 16);
        expect(glstm.mutationPressure).toBe(BOOST);

        glstm.updateMutationPressure(0.5004);
        expect(glstm.mutationPressure).toBe(BOOST);
        expect(glstm['_stagnationCounter']).toBe(1);

        glstm.updateMutationPressure(0.5006);
        expect(glstm.mutationPressure).toBe(NORMAL);
        expect(glstm['_stagnationCounter']).toBe(0);
    });

    it('halves the PANIC cooldown on improvement', () => {
        const glstm = makeFlatPressureGlstm();
        runFlat(glstm, 1, 136);
        expect(glstm['_panicCooldownCounter']).toBe(60);

        glstm.updateMutationPressure(0.6);
        expect(glstm.mutationPressure).toBe(BOOST);
        expect(glstm['_panicCooldownCounter']).toBe(29);
    });

    it('keeps the level when enablePressureEscalation is false', () => {
        const glstm = makeFlatPressureGlstm({ enablePressureEscalation: false, mutationPressure: BOOST });

        expect(runFlat(glstm, 1, 300)).toEqual([]);
        glstm.updateMutationPressure(10);
        glstm.updateMutationPressure(20);

        expect(glstm.mutationPressure).toBe(BOOST);
        expect(glstm['_stagnationCounter']).toBe(0);
        expect(glstm['_panicCooldownCounter']).toBe(0);
    });
});
