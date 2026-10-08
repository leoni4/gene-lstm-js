import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Genome } from '../src/genome.js';
import type { GeneLSTMOptions } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions } from './helpers/randomGenome.js';

const structOptions: GeneLSTMOptions = {
    PROBABILITY_MUTATE_LSTM_BLOCK: 1,
    PROBABILITY_MUTATE_ADD_UNIT: 0,
    PROBABILITY_MUTATE_REMOVE_UNIT: 0,
};

function mutateOnce(options: GeneLSTMOptions, layers: number, seed: number) {
    const glstm = new GeneLSTM(2, { ...structOptions, ...options });
    const genome = new Genome(glstm, makeGeneOptions(mulberry32(17), layers, 2, 1, 1));
    const before = [...genome.lstmArray];
    const { log } = withSeededRandom(seed, () => genome.mutate());

    return { before, after: genome.lstmArray, log };
}

describe('frontStructureChanges', () => {
    it('defaults to true', () => {
        expect(new GeneLSTM(2).frontStructureChanges).toBe(true);
        expect(new GeneLSTM(2, { frontStructureChanges: false }).frontStructureChanges).toBe(false);
    });

    it('with false, a prepend becomes an append with the same draws', () => {
        const opts: GeneLSTMOptions = { PROBABILITY_REMOVE_BLOCK: 0, PROBABILITY_ADD_BLOCK_APPEND: 0 };
        const on = mutateOnce(opts, 2, 5);
        const off = mutateOnce({ ...opts, frontStructureChanges: false }, 2, 5);

        expect(on.after).toHaveLength(3);
        expect(on.after.slice(1)).toEqual(on.before);
        expect(off.after).toHaveLength(3);
        expect(off.after.slice(0, 2)).toEqual(off.before);
        expect(off.log).toEqual(on.log);
    });

    it('with false, a front removal becomes an end removal with the same draws', () => {
        const opts: GeneLSTMOptions = { PROBABILITY_REMOVE_BLOCK: 1 };
        let frontRemovals = 0;

        for (let seed = 1; seed <= 40; seed++) {
            const on = mutateOnce(opts, 3, seed);
            const off = mutateOnce({ ...opts, frontStructureChanges: false }, 3, seed);
            expect(off.log).toEqual(on.log);

            if (on.after.length === 2 && on.after[0] !== on.before[0]) {
                frontRemovals++;
                expect(off.after).toEqual(off.before.slice(0, 2));
            }
            expect(off.after[0]).toBe(off.before[0]);
        }
        expect(frontRemovals).toBeGreaterThan(0);
    });

    it.each([
        [true, 'changes'],
        [false, 'never changes'],
    ])('with %s, 1000 structural mutations %s layer 0', frontStructureChanges => {
        const glstm = new GeneLSTM(2, {
            ...structOptions,
            PROBABILITY_REMOVE_BLOCK: 0.5,
            PROBABILITY_ADD_BLOCK_APPEND: 0.5,
            MAX_LAYERS: 4,
            frontStructureChanges,
        });
        const genome = new Genome(glstm, makeGeneOptions(mulberry32(3), 2, 2, 1, 1));
        const first = genome.lstmArray[0];
        let changed = 0;

        withSeededRandom(23, () => {
            for (let i = 0; i < 1000; i++) {
                genome.mutate();
                if (genome.lstmArray[0] !== first) changed++;
            }
        });

        if (frontStructureChanges) expect(changed).toBeGreaterThan(0);
        else expect(changed).toBe(0);
    });
});
