import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Species } from '../src/species.js';
import { withSeededRandom } from './helpers/seededRandom.js';

function expectSpeciesInPopulation(g: GeneLSTM) {
    const population = new Set(g.clients);
    for (const s of g['_species']) {
        expect(s.size()).toBeGreaterThan(0);
        expect(population.has(s['_representative'])).toBe(true);
        for (const c of s.clients) {
            expect(population.has(c)).toBe(true);
            expect(c.species).toBe(s);
        }
    }
}

function countReinsertions(g: GeneLSTM): () => number {
    let count = 0;
    const original = g['_reinsertElitesIfStagnant'].bind(g);
    g['_reinsertElitesIfStagnant'] = () => {
        const reinserted = original();
        if (reinserted) count++;

        return reinserted;
    };

    return () => count;
}

describe('elite re-insertion and species membership', () => {
    it('keeps only population clients in species during a flat-fitness run', () => {
        withSeededRandom(9, () => {
            const g = new GeneLSTM(30, { INPUT_FEATURES: 2 });
            const reinsertions = countReinsertions(g);
            for (let gen = 0; gen < 120; gen++) {
                for (const c of g.clients) c.score = 0.5;
                g.evolve();
                expectSpeciesInPopulation(g);
            }
            expect(reinsertions()).toBeGreaterThan(0);
        });
    });

    it('removes replaced clients from their species and drops a species that becomes empty', () => {
        withSeededRandom(3, () => {
            const g = new GeneLSTM(8, { INPUT_FEATURES: 2 });
            for (const c of g.clients) c.score = Math.random();
            g.evolve();
            for (const c of g.clients) c.score = Math.random();
            g['_prepareRawScores']();
            g['_normalizeScore']();

            const last = g.clients[g.clients.length - 1];
            const secondLast = g.clients[g.clients.length - 2];
            const lastOldSpecies = last.species;
            expect(lastOldSpecies).not.toBeNull();
            if (lastOldSpecies === null) return;
            lastOldSpecies.remove(last);
            if (lastOldSpecies.size() === 0) {
                g['_species'].splice(g['_species'].indexOf(lastOldSpecies), 1);
            }
            const lonely = new Species(last);
            g['_species'].push(lonely);
            const secondLastSpecies = secondLast.species;

            expect(g['_champion']).not.toBeNull();
            expect(g['_runnerUp']).not.toBeNull();
            g['_championStagnationCount'] = g['_championStagnationThreshold'];
            expect(g['_reinsertElitesIfStagnant']()).toBe(true);

            expect(g['_species']).not.toContain(lonely);
            expect(last.species).toBeNull();
            expect(secondLast.species).toBeNull();
            expect(secondLastSpecies?.clients ?? []).not.toContain(secondLast);
            expect(g.clients).not.toContain(last);
            expect(g.clients).not.toContain(secondLast);

            for (const c of g.clients) c.score = Math.random();
            expect(() => g.evolve()).not.toThrow();
            expectSpeciesInPopulation(g);
        });
    });
});
