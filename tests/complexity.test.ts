import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Genome } from '../src/genome.js';
import { Client } from '../src/client.js';
import { EMutationPressure } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions } from './helpers/randomGenome.js';

const { BOOST, ESCAPE, COMPACT } = EMutationPressure;

const F = 3;

function clientWithShape(glstm: GeneLSTM, H: number[], outDim = 1): Client {
    return new Client(new Genome(glstm, makeGeneOptions(mulberry32(H.length * 1000 + H[0]), H.length, H, F, outDim)));
}

describe('_calcClientComplexity()', () => {
    it('counts the hidden units of each layer', () => {
        const glstm = new GeneLSTM(1);

        expect(glstm['_calcClientComplexity'](clientWithShape(glstm, [60, 60]))).toEqual({
            blocks: 2,
            units: 120,
            complexity: 32,
        });
        expect(glstm['_calcClientComplexity'](clientWithShape(glstm, [3, 5]))).toEqual({
            blocks: 2,
            units: 8,
            complexity: 4,
        });
        expect(glstm['_calcClientComplexity'](clientWithShape(glstm, [1]))).toEqual({
            blocks: 1,
            units: 1,
            complexity: 1.25,
        });
    });

    it('does not count OUTPUT_DIM as units', () => {
        const glstm = new GeneLSTM(1, { OUTPUT_DIM: 3 });

        expect(glstm['_calcClientComplexity'](clientWithShape(glstm, [60, 60], 3))).toEqual({
            blocks: 2,
            units: 120,
            complexity: 32,
        });
    });
});

describe('complexity penalty with hidden units', () => {
    it('gives a larger genome a larger penalty when raw scores are equal', () => {
        const glstm = new GeneLSTM(2);
        const small = clientWithShape(glstm, [1, 1]);
        const large = clientWithShape(glstm, [60, 60]);
        glstm.clients.splice(0, glstm.clients.length, large, small);
        small.score = 0.5;
        large.score = 0.5;

        withSeededRandom(3, () => glstm['_prepareRawScores']());
        expect(small.complexity).toBe(2.5);
        expect(large.complexity).toBe(32);

        glstm['_normalizeScore']();

        expect(large.adjustedScore).toBeLessThan(small.adjustedScore);
        expect(glstm.clients).toEqual([small, large]);
    });
});

describe('COMPACT pressure with hidden-unit growth', () => {
    it('starts COMPACT in the ESCAPE window (stagnation counter 51) when the best genome grows by 8 units with flat fitness', () => {
        const glstm = new GeneLSTM(2);
        const before = clientWithShape(glstm, [1]);
        const after = clientWithShape(glstm, [9]);

        const changes: Array<[number, EMutationPressure]> = [];
        for (let gen = 1; gen <= 105; gen++) {
            const best = gen <= 60 ? before : after;
            best.scoreRaw = 0.5;
            best.complexity = glstm['_calcClientComplexity'](best).complexity;
            glstm['_updateChampion'](best);

            const pressure = glstm.mutationPressure;
            glstm.updateMutationPressure(0.5, gen);
            if (glstm.mutationPressure !== pressure) {
                changes.push([gen, glstm.mutationPressure]);
            }
        }

        expect(after.complexity - before.complexity).toBe(2);
        expect(changes).toEqual([
            [16, BOOST],
            [46, ESCAPE],
            [97, COMPACT],
        ]);
        expect(glstm.mutationPressure).toBe(COMPACT);
        expect(glstm['_stagnationCounter']).toBe(59);
        expect(glstm['_optimization']).toBe(true);
    });
});
