import { describe, it, expect, vi, afterEach } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Genome } from '../src/genome.js';
import { Client } from '../src/client.js';
import { Species } from '../src/species.js';
import { RandomSelector } from '../src/randomSelector.js';
import type { SpeciesSelection } from '../src/types/index.js';
import { withSeededRandom } from './helpers/seededRandom.js';

const PICKS = 100_000;

function equalSpecies(geneLstm: GeneLSTM, count: number): Species[] {
    const list: Species[] = [];
    for (let i = 0; i < count; i++) {
        const client = new Client(new Genome(geneLstm));
        client.score = 1;
        const species = new Species(client);
        species.evaluateScore();
        list.push(species);
    }

    return list;
}

function histogram(selector: RandomSelector, species: Species[], seed: number) {
    const { result, log } = withSeededRandom(seed, () => {
        const counts = species.map(() => 0);
        for (let n = 0; n < PICKS; n++) counts[species.indexOf(selector.random())]++;

        return counts;
    });

    return { counts: result, randomCalls: log.length };
}

function makeSelector(species: Species[], mode?: SpeciesSelection): RandomSelector {
    const selector = mode === undefined ? new RandomSelector(0.6) : new RandomSelector(0.6, mode);
    for (const s of species) selector.add(s);

    return selector;
}

function seededRun(options: { speciesSelection?: SpeciesSelection }) {
    return withSeededRandom(21, () => {
        const g = new GeneLSTM(30, { INPUT_FEATURES: 2, ...options });
        for (let gen = 0; gen < 15; gen++) {
            for (const c of g.clients) c.score = Math.random();
            g.evolve();
        }

        return JSON.stringify(g.clients.map(c => c.model()));
    });
}

afterEach(() => {
    vi.restoreAllMocks();
});

describe('speciesSelection option', () => {
    it("defaults to 'legacy' and stores the given mode", () => {
        expect(new GeneLSTM(4).speciesSelection).toBe('legacy');
        expect(new GeneLSTM(4, { speciesSelection: 'legacy' }).speciesSelection).toBe('legacy');
        expect(new GeneLSTM(4, { speciesSelection: 'proportional' }).speciesSelection).toBe('proportional');
    });

    it("'legacy' never picks the species outside the first SURVIVORS share (5 equal species)", () => {
        const species = equalSpecies(new GeneLSTM(4), 5);
        const { counts, randomCalls } = histogram(makeSelector(species, 'legacy'), species, 5);

        expect(randomCalls).toBe(PICKS);
        expect(counts[3]).toBe(0);
        expect(counts[4]).toBe(0);
        for (let i = 0; i < 3; i++) expect(counts[i] / PICKS).toBeCloseTo(1 / 3, 1);
    });

    it('the default selector mode gives the same picks as legacy', () => {
        const species = equalSpecies(new GeneLSTM(4), 5);
        const byDefault = histogram(makeSelector(species), species, 5);
        const legacy = histogram(makeSelector(species, 'legacy'), species, 5);

        expect(byDefault).toEqual(legacy);
    });

    it("'proportional' picks each of 5 equal species about 20% of the time, with one Math.random call per pick", () => {
        const species = equalSpecies(new GeneLSTM(4), 5);
        const { counts, randomCalls } = histogram(makeSelector(species, 'proportional'), species, 5);

        expect(randomCalls).toBe(PICKS);
        for (const c of counts) expect(Math.abs(c / PICKS - 0.2)).toBeLessThan(0.01);
    });

    it('evolve() passes the option to the species selector', () => {
        const random = vi.spyOn(RandomSelector.prototype, 'random');
        for (const mode of ['legacy', 'proportional'] as const) {
            random.mockClear();
            withSeededRandom(4, () => {
                const g = new GeneLSTM(20, { INPUT_FEATURES: 2, speciesSelection: mode });
                for (const c of g.clients) c.score = Math.random();
                g.evolve();
            });
            expect(random.mock.contexts.length).toBeGreaterThan(0);
            for (const selector of random.mock.contexts) {
                expect(selector).toBeInstanceOf(RandomSelector);
                if (selector instanceof RandomSelector) expect(selector['_mode']).toBe(mode);
            }
        }
    });

    it("an explicit 'legacy' gives the same seeded run as the default; 'proportional' gives another run", () => {
        const byDefault = seededRun({});
        const legacy = seededRun({ speciesSelection: 'legacy' });
        const proportional = seededRun({ speciesSelection: 'proportional' });

        expect(legacy.result).toBe(byDefault.result);
        expect(legacy.log).toEqual(byDefault.log);
        expect(proportional.result).not.toBe(byDefault.result);
    });
});
