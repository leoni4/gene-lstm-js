import { describe, it, expect, beforeEach } from 'vitest';
import { Client } from '../src/client.js';
import { Genome } from '../src/genome.js';
import { GeneLSTM } from '../src/gLstm.js';

describe('Client', () => {
    let geneLstm: GeneLSTM;
    let genome: Genome;
    let client: Client;

    beforeEach(() => {
        geneLstm = new GeneLSTM(1);
        genome = new Genome(geneLstm);
        client = new Client(genome);
    });

    describe('constructor', () => {
        it('should initialize with a genome', () => {
            expect(client.genome).toBe(genome);
            expect(client.species).toBeNull();
            expect(client.bestScore).toBe(false);
            expect(client.error).toBe(0);
            expect(client.score).toBe(0);
        });
    });

    describe('mutate', () => {
        it('should mutate genome when bestScore is false', () => {
            const originalModel = client.genome.lstmArray[0].model();
            client.mutate();
            const mutatedModel = client.genome.lstmArray[0].model();

            // Mutation may or may not change the model depending on random probabilities
            expect(client.genome).toBeDefined();
            expect(JSON.stringify(originalModel) === JSON.stringify(mutatedModel)).toBeFalsy();
        });

        it('should not mutate when bestScore is true and force is false', () => {
            client.bestScore = true;
            const originalDepth = client.genome.lstmArray.length;

            // Run multiple times to increase chance of mutation attempt
            for (let i = 0; i < 10; i++) {
                client.mutate(false);
            }

            // Depth should remain the same
            expect(client.genome.lstmArray.length).toBe(originalDepth);
        });

        it('should mutate when bestScore is true and force is true', () => {
            client.bestScore = true;

            // Force mutation with high probability settings
            const glstmWithHighMutation = new GeneLSTM(1, {
                PROBABILITY_MUTATE_LSTM_BLOCK: 0.5,
                MUTATION_RATE: 1.0,
            });
            const genomeWithHighMutation = new Genome(glstmWithHighMutation);
            const clientWithHighMutation = new Client(genomeWithHighMutation);
            clientWithHighMutation.bestScore = true;

            clientWithHighMutation.mutate(true);

            // Genome should still be valid after forced mutation
            expect(clientWithHighMutation.genome).toBeDefined();
        });
    });

    describe('distance', () => {
        it('should calculate distance between two clients', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);

            const distance = client.distance(client2);

            expect(distance).toBeGreaterThanOrEqual(0);
            expect(typeof distance).toBe('number');
        });

        it('should return 0 distance for identical clients', () => {
            const distance = client.distance(client);

            expect(distance).toBe(0);
        });

        it('should calculate symmetric distance', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);

            const distance1 = client.distance(client2);
            const distance2 = client2.distance(client);

            expect(distance1).toBe(distance2);
        });
    });

    describe('calculate', () => {
        it('should calculate output for number[] input', () => {
            const input = [0.5, 0.3, 0.2];
            const output = client.calculate(input);

            expect(Array.isArray(output)).toBe(true);
            expect(output.length).toBeGreaterThan(0);
            output.forEach(val => {
                expect(typeof val).toBe('number');
                expect(isFinite(val)).toBe(true);
            });
        });

        it('should calculate output for number[][] input', () => {
            const input = [
                [0.5, 0.3],
                [0.2, 0.4],
                [0.1, 0.6],
            ];
            const output = client.calculate(input);

            expect(Array.isArray(output)).toBe(true);
            expect(output.length).toBeGreaterThan(0);
            output.forEach(val => {
                expect(typeof val).toBe('number');
                expect(isFinite(val)).toBe(true);
            });
        });

        it('should produce consistent output for same input', () => {
            const input = [0.5, 0.3, 0.2];
            const output1 = client.calculate(input);
            const output2 = client.calculate(input);

            expect(output1).toEqual(output2);
        });
    });

    describe('score and error properties', () => {
        it('should allow setting score', () => {
            client.score = 0.75;
            expect(client.score).toBe(0.75);
        });

        it('should allow setting error', () => {
            client.error = 0.25;
            expect(client.error).toBe(0.25);
        });

        it('should allow setting bestScore', () => {
            client.bestScore = true;
            expect(client.bestScore).toBe(true);
        });
    });
});
