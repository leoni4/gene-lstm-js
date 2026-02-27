import { describe, it, expect, beforeEach } from 'vitest';
import { RandomSelector } from '../src/randomSelector.js';
import { Species } from '../src/species.js';
import { Client } from '../src/client.js';
import { Genome } from '../src/genome.js';
import { GeneLSTM } from '../src/gLstm.js';

describe('RandomSelector', () => {
    let geneLstm: GeneLSTM;
    let randomSelector: RandomSelector;

    beforeEach(() => {
        geneLstm = new GeneLSTM(10);
        randomSelector = new RandomSelector(0.5);
    });

    describe('constructor', () => {
        it('should initialize with survivors parameter', () => {
            expect(randomSelector.objects).toEqual([]);
            expect(randomSelector.totalScore).toBe(0);
        });
    });

    describe('add', () => {
        it('should add species to the selector', () => {
            const genome = new Genome(geneLstm);
            const client = new Client(genome);
            client.score = 0.8;
            const species = new Species(client);

            randomSelector.add(species);

            expect(randomSelector.objects).toHaveLength(1);
            expect(randomSelector.objects[0]).toBe(species);
        });

        it('should update total score when adding species', () => {
            const genome = new Genome(geneLstm);
            const client = new Client(genome);
            client.score = 0.8;
            const species = new Species(client);
            species.evaluateScore();

            randomSelector.add(species);

            expect(randomSelector.totalScore).toBeGreaterThan(0);
        });

        it('should sort species by score in descending order', () => {
            const genome1 = new Genome(geneLstm);
            const client1 = new Client(genome1);
            client1.score = 0.5;
            const species1 = new Species(client1);
            species1.evaluateScore();

            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            client2.score = 0.9;
            const species2 = new Species(client2);
            species2.evaluateScore();

            const genome3 = new Genome(geneLstm);
            const client3 = new Client(genome3);
            client3.score = 0.7;
            const species3 = new Species(client3);
            species3.evaluateScore();

            randomSelector.add(species1);
            randomSelector.add(species2);
            randomSelector.add(species3);

            expect(randomSelector.objects).toHaveLength(3);
            expect(randomSelector.objects[0].score).toBeGreaterThanOrEqual(randomSelector.objects[1].score);
            expect(randomSelector.objects[1].score).toBeGreaterThanOrEqual(randomSelector.objects[2].score);
        });

        it('should accumulate total score from multiple species', () => {
            const genome1 = new Genome(geneLstm);
            const client1 = new Client(genome1);
            client1.score = 0.5;
            const species1 = new Species(client1);
            species1.evaluateScore();

            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            client2.score = 0.3;
            const species2 = new Species(client2);
            species2.evaluateScore();

            randomSelector.add(species1);
            const scoreAfterFirst = randomSelector.totalScore;
            randomSelector.add(species2);

            expect(randomSelector.totalScore).toBeGreaterThan(scoreAfterFirst);
        });
    });

    describe('random', () => {
        it('should return a species from the selector', () => {
            const genome = new Genome(geneLstm);
            const client = new Client(genome);
            client.score = 0.8;
            const species = new Species(client);
            species.evaluateScore();

            randomSelector.add(species);

            const selected = randomSelector.random();

            expect(selected).toBe(species);
        });

        it('should return first species when only one exists', () => {
            const genome = new Genome(geneLstm);
            const client = new Client(genome);
            client.score = 0.8;
            const species = new Species(client);
            species.evaluateScore();

            randomSelector.add(species);

            for (let i = 0; i < 10; i++) {
                const selected = randomSelector.random();
                expect(selected).toBe(species);
            }
        });

        it('should select from multiple species based on score', () => {
            const speciesArray: Species[] = [];

            for (let i = 0; i < 5; i++) {
                const genome = new Genome(geneLstm);
                const client = new Client(genome);
                client.score = 0.5 + i * 0.1;
                const species = new Species(client);
                species.evaluateScore();
                speciesArray.push(species);
                randomSelector.add(species);
            }

            const selections = new Set<Species>();

            // Run multiple selections to see distribution
            for (let i = 0; i < 50; i++) {
                const selected = randomSelector.random();
                selections.add(selected);
                expect(randomSelector.objects).toContain(selected);
            }

            // Should potentially select different species (though not guaranteed)
            expect(selections.size).toBeGreaterThan(0);
        });

        it('should handle edge case with zero total score', () => {
            const genome = new Genome(geneLstm);
            const client = new Client(genome);
            client.score = 0;
            const species = new Species(client);
            species.evaluateScore();

            randomSelector.add(species);

            const selected = randomSelector.random();

            expect(selected).toBe(species);
        });
    });

    describe('reset', () => {
        it('should clear all objects and reset total score', () => {
            const genome = new Genome(geneLstm);
            const client = new Client(genome);
            client.score = 0.8;
            const species = new Species(client);
            species.evaluateScore();

            randomSelector.add(species);

            expect(randomSelector.objects).toHaveLength(1);
            expect(randomSelector.totalScore).toBeGreaterThan(0);

            randomSelector.reset();

            expect(randomSelector.objects).toEqual([]);
            expect(randomSelector.totalScore).toBe(0);
        });

        it('should allow adding new species after reset', () => {
            const genome1 = new Genome(geneLstm);
            const client1 = new Client(genome1);
            client1.score = 0.8;
            const species1 = new Species(client1);
            species1.evaluateScore();

            randomSelector.add(species1);
            randomSelector.reset();

            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            client2.score = 0.6;
            const species2 = new Species(client2);
            species2.evaluateScore();

            randomSelector.add(species2);

            expect(randomSelector.objects).toHaveLength(1);
            expect(randomSelector.objects[0]).toBe(species2);
        });
    });

    describe('getters', () => {
        it('should return correct objects array', () => {
            const objects = randomSelector.objects;
            expect(Array.isArray(objects)).toBe(true);
        });

        it('should return correct total score', () => {
            const score = randomSelector.totalScore;
            expect(typeof score).toBe('number');
            expect(score).toBe(0);
        });
    });
});
