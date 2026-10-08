import { describe, it, expect, vi } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Genome } from '../src/genome.js';
import { LSTM } from '../src/lstm.js';
import { Species } from '../src/species.js';
import type { Client } from '../src/client.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput, type WeightInMode } from './helpers/randomGenome.js';

const F = 6;

interface PairCase {
    name: string;
    layers1: number;
    layers2: number;
    H1: number | number[];
    H2: number | number[];
    mode1: WeightInMode;
    mode2: WeightInMode;
    outDim: number;
}

const PAIR_CASES: PairCase[] = [
    { name: 'same shape, all weightIn', layers1: 1, layers2: 1, H1: 3, H2: 3, mode1: 'all', mode2: 'all', outDim: 1 },
    { name: 'different H', layers1: 1, layers2: 1, H1: 2, H2: 5, mode1: 'all', mode2: 'all', outDim: 1 },
    { name: 'different layers', layers1: 1, layers2: 3, H1: 3, H2: [3, 4, 2], mode1: 'all', mode2: 'all', outDim: 1 },
    { name: 'weightIn missing', layers1: 2, layers2: 2, H1: 4, H2: 4, mode1: 'none', mode2: 'none', outDim: 1 },
    { name: 'weightIn set vs missing', layers1: 2, layers2: 2, H1: 4, H2: 4, mode1: 'all', mode2: 'none', outDim: 1 },
    {
        name: 'mixed weightIn',
        layers1: 4,
        layers2: 2,
        H1: [3, 1, 5, 2],
        H2: 6,
        mode1: 'mixed',
        mode2: 'mixed',
        outDim: 2,
    },
    { name: 'OUTPUT_DIM 3', layers1: 2, layers2: 3, H1: 7, H2: [1, 7, 3], mode1: 'mixed', mode2: 'all', outDim: 3 },
];

describe('Genome.distance() with a flatten cache', () => {
    for (const [index, c] of PAIR_CASES.entries()) {
        it(`gives the same value as without the cache: ${c.name}`, () => {
            const rng = mulberry32(1000 + index);
            const glstm = new GeneLSTM(1, { INPUT_FEATURES: F, OUTPUT_DIM: c.outDim });
            const g1 = new Genome(glstm, makeGeneOptions(rng, c.layers1, c.H1, F, c.outDim, c.mode1));
            const g2 = new Genome(glstm, makeGeneOptions(rng, c.layers2, c.H2, F, c.outDim, c.mode2));

            const { result, log } = withSeededRandom(99, () => {
                const plain = [g1.distance(g2), g2.distance(g1), g1.distance(g1)];

                const cache = new Map<LSTM, number[]>();
                const firstFill = [g1.distance(g2, cache), g2.distance(g1, cache), g1.distance(g1, cache)];
                const filledSize = cache.size;
                const fromFilled = [g1.distance(g2, cache), g2.distance(g1, cache), g1.distance(g1, cache)];

                return { plain, firstFill, fromFilled, filledSize, cacheSize: cache.size };
            });

            for (let i = 0; i < result.plain.length; i++) {
                expect(Object.is(result.firstFill[i], result.plain[i])).toBe(true);
                expect(Object.is(result.fromFilled[i], result.plain[i])).toBe(true);
            }
            expect(result.plain[0]).toBeGreaterThan(0);
            expect(result.filledSize).toBe(Math.min(c.layers1, c.layers2) + c.layers1);
            expect(result.cacheSize).toBe(result.filledSize);
            expect(log).toHaveLength(0);
        });
    }

    it('flattens each LSTM at most once when the cache is passed', () => {
        const rng = mulberry32(2024);
        const glstm = new GeneLSTM(1, { INPUT_FEATURES: F });
        const genomes = Array.from(
            { length: 5 },
            (_, i) => new Genome(glstm, makeGeneOptions(rng, 1 + (i % 3), 3, F, 1)),
        );
        const spy = vi.spyOn(LSTM.prototype, 'flattenWeights');
        try {
            const cache = new Map<LSTM, number[]>();
            for (const a of genomes) {
                for (const b of genomes) a.distance(b, cache);
            }
            const layerCount = genomes.reduce((n, g) => n + g.lstmArray.length, 0);
            expect(spy).toHaveBeenCalledTimes(layerCount);
        } finally {
            spy.mockRestore();
        }
    });
});

describe('_genSpecies() flatten cache in evolve()', () => {
    const input = makeInput(mulberry32(31), 6, 4);
    const originalPut = Species.prototype.put;

    const seededRun = (passCache: boolean) => {
        const putSpy = vi.spyOn(Species.prototype, 'put').mockImplementation(function (
            this: Species,
            client: Client,
            force = false,
            cache?: Map<LSTM, number[]>,
        ) {
            return originalPut.call(this, client, force, passCache ? cache : undefined);
        });
        const flattenCalls = new Map<LSTM, number>();
        const originalFlatten = LSTM.prototype.flattenWeights;
        const flattenSpy = vi.spyOn(LSTM.prototype, 'flattenWeights').mockImplementation(function (this: LSTM) {
            flattenCalls.set(this, (flattenCalls.get(this) ?? 0) + 1);

            return originalFlatten.call(this);
        });

        try {
            return withSeededRandom(4321, () => {
                const glstm = new GeneLSTM(30, {
                    INPUT_FEATURES: 4,
                    PROBABILITY_MUTATE_LSTM_BLOCK: 0.2,
                    PROBABILITY_MUTATE_ADD_UNIT: 0.2,
                });
                const speciesSizes: number[][] = [];
                let maxFlattenPerLstm = 0;
                let distancePuts = 0;
                for (let gen = 0; gen < 12; gen++) {
                    for (const client of glstm.clients) {
                        const y = client.calculate(input);
                        client.score = 1 / (1 + Math.abs(y[0] - 0.4));
                    }
                    flattenCalls.clear();
                    const putsBefore = putSpy.mock.calls.filter(call => call[1] !== true).length;
                    glstm.evolve();
                    distancePuts += putSpy.mock.calls.filter(call => call[1] !== true).length - putsBefore;
                    maxFlattenPerLstm = Math.max(maxFlattenPerLstm, ...flattenCalls.values());
                    speciesSizes.push(glstm['_species'].map(s => s.size()));
                }

                return {
                    json: JSON.stringify(glstm.clients.map(c => c.model())),
                    speciesSizes,
                    maxFlattenPerLstm,
                    distancePuts,
                };
            });
        } finally {
            flattenSpy.mockRestore();
            putSpy.mockRestore();
        }
    };

    it('gives the same population, species, and Math.random log as a run without the cache', () => {
        const withCache = seededRun(true);
        const withoutCache = seededRun(false);

        expect(withCache.result.json).toBe(withoutCache.result.json);
        expect(withCache.result.speciesSizes).toEqual(withoutCache.result.speciesSizes);
        expect(withCache.log.length).toBe(withoutCache.log.length);
        expect(withCache.log.every((v, i) => Object.is(v, withoutCache.log[i]))).toBe(true);

        expect(withCache.result.distancePuts).toBeGreaterThan(0);
        expect(withCache.result.speciesSizes.some(sizes => sizes.length > 1)).toBe(true);

        expect(withCache.result.maxFlattenPerLstm).toBe(1);
        expect(withoutCache.result.maxFlattenPerLstm).toBeGreaterThan(1);
    });
});
