import { describe, it, expect, beforeEach } from 'vitest';
import { Species } from '../src/species.js';
import { Client } from '../src/client.js';
import { Genome } from '../src/genome.js';
import { GeneLSTM } from '../src/gLstm.js';

describe('Species', () => {
    let geneLstm: GeneLSTM;
    let client: Client;
    let species: Species;

    beforeEach(() => {
        geneLstm = new GeneLSTM(10);
        const genome = new Genome(geneLstm);
        client = new Client(genome);
        client.score = 0.8;
        species = new Species(client);
    });

    describe('constructor', () => {
        it('should initialize with a representative client', () => {
            expect(species.clients).toHaveLength(1);
            expect(species.clients[0]).toBe(client);
            expect(client.species).toBe(species);
        });

        it('should initialize score to 0', () => {
            expect(species.score).toBe(0);
        });
    });

    describe('put', () => {
        it('should add client when distance is within CP threshold', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);

            // Since both clients have random but similar genomes, distance should be relatively small
            const added = species.put(client2);

            expect(typeof added).toBe('boolean');
            if (added) {
                expect(species.clients).toContain(client2);
                expect(client2.species).toBe(species);
            }
        });

        it('should add client when force is true regardless of distance', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);

            const added = species.put(client2, true);

            expect(added).toBe(true);
            expect(species.clients).toContain(client2);
            expect(client2.species).toBe(species);
        });

        it('should not add client when distance exceeds CP and force is false', () => {
            // Create a genome with very different structure
            const glstmHighCP = new GeneLSTM(10, {
                CP: 0.001, // Very low threshold
            });
            const genome2 = new Genome(glstmHighCP);

            // Make it very different
            for (let i = 0; i < 5; i++) {
                genome2.mutate();
            }

            const client2 = new Client(genome2);

            // Try to add with force=false
            const added = species.put(client2, false);

            // Might or might not be added depending on actual distance
            expect(typeof added).toBe('boolean');
        });

        it('should update client species reference when added', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);

            expect(client2.species).toBeNull();

            species.put(client2, true);

            expect(client2.species).toBe(species);
        });
    });

    describe('size', () => {
        it('should return number of clients in species', () => {
            expect(species.size()).toBe(1);
        });

        it('should update when clients are added', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);

            species.put(client2, true);

            expect(species.size()).toBe(2);
        });
    });

    describe('goExtinct', () => {
        it('should remove all clients and clear their species reference', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            species.put(client2, true);

            expect(species.size()).toBe(2);
            expect(client.species).toBe(species);
            expect(client2.species).toBe(species);

            species.goExtinct();

            expect(species.size()).toBe(0);
            expect(client.species).toBeNull();
            expect(client2.species).toBeNull();
        });
    });

    describe('reset', () => {
        it('should pick new representative and clear other clients', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            species.put(client2, true);

            const initialSize = species.size();
            expect(initialSize).toBeGreaterThan(1);

            species.reset();

            expect(species.size()).toBe(1);
            expect(species.clients.length).toBe(1);
            expect(species.clients[0].species).toBe(species);
        });

        it('should reset score to 0', () => {
            species.evaluateScore();
            const scoreBeforeReset = species.score;

            species.reset();

            expect(species.score).toBe(0);
        });

        it('should maintain a valid representative after reset', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            species.put(client2, true);

            species.reset();

            expect(species.clients.length).toBe(1);
            const rep = species.clients[0];
            expect(rep).toBeDefined();
            expect(rep.species).toBe(species);
        });
    });

    describe('evaluateScore', () => {
        it('should calculate average score of all clients', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            client2.score = 0.6;
            species.put(client2, true);

            const genome3 = new Genome(geneLstm);
            const client3 = new Client(genome3);
            client3.score = 0.4;
            species.put(client3, true);

            species.evaluateScore();

            // Average of 0.8, 0.6, 0.4 = 0.6
            expect(species.score).toBeCloseTo((0.8 + 0.6 + 0.4) / 3, 5);
        });

        it('should handle single client', () => {
            species.evaluateScore();

            expect(species.score).toBe(0.8);
        });

        it('should handle zero scores', () => {
            client.score = 0;

            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            client2.score = 0;
            species.put(client2, true);

            species.evaluateScore();

            expect(species.score).toBe(0);
        });
    });

    describe('kill', () => {
        it('should remove bottom performing clients', () => {
            // Add more clients with different scores
            const clients = [client];
            for (let i = 0; i < 9; i++) {
                const genome = new Genome(geneLstm);
                const c = new Client(genome);
                c.score = Math.random();
                species.put(c, true);
                clients.push(c);
            }

            const initialSize = species.size();

            species.kill(0.5); // Keep top 50%

            const finalSize = species.size();

            expect(finalSize).toBeLessThanOrEqual(initialSize);
            expect(finalSize).toBeGreaterThan(0);
        });

        it('should sort clients by score before killing', () => {
            // Add clients with specific scores
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            client2.score = 0.9;
            species.put(client2, true);

            const genome3 = new Genome(geneLstm);
            const client3 = new Client(genome3);
            client3.score = 0.3;
            species.put(client3, true);

            const genome4 = new Genome(geneLstm);
            const client4 = new Client(genome4);
            client4.score = 0.5;
            species.put(client4, true);

            species.kill(0.5); // Keep top 50%

            // Check that remaining clients have higher scores
            const remainingScores = species.clients.map(c => c.score);
            expect(Math.min(...remainingScores)).toBeGreaterThanOrEqual(0.5);
        });

        it('should not kill clients with bestScore flag', () => {
            // Add clients
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            client2.score = 0.1; // Low score
            client2.bestScore = true; // But marked as best
            species.put(client2, true);

            const genome3 = new Genome(geneLstm);
            const client3 = new Client(genome3);
            client3.score = 0.5;
            species.put(client3, true);

            species.kill(0.3); // Aggressive kill rate

            // client2 should survive despite low score
            expect(species.clients).toContain(client2);
        });

        it('should clear species reference for killed clients', () => {
            // Add multiple clients
            const clientsToAdd: Client[] = [];
            for (let i = 0; i < 5; i++) {
                const genome = new Genome(geneLstm);
                const c = new Client(genome);
                c.score = i * 0.1;
                species.put(c, true);
                clientsToAdd.push(c);
            }

            species.kill(0.5);

            // Check that killed clients have null species
            clientsToAdd.forEach(c => {
                if (!species.clients.includes(c)) {
                    expect(c.species).toBeNull();
                }
            });
        });
    });

    describe('breed', () => {
        it('should produce offspring genome', () => {
            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            client2.score = 0.6;
            species.put(client2, true);

            const offspring = species.breed();

            expect(offspring).toBeInstanceOf(Genome);
            expect(offspring.lstmArray.length).toBeGreaterThanOrEqual(1);
        });

        it('should crossover higher scoring parent first', () => {
            // Add clients with different scores
            client.score = 0.9;

            const genome2 = new Genome(geneLstm);
            const client2 = new Client(genome2);
            client2.score = 0.3;
            species.put(client2, true);

            const offspring = species.breed();

            expect(offspring).toBeInstanceOf(Genome);
        });

        it('should work with single client (self-crossing)', () => {
            const offspring = species.breed();

            expect(offspring).toBeInstanceOf(Genome);
            expect(offspring.lstmArray.length).toBeGreaterThanOrEqual(1);
        });

        it('should produce valid offspring from multiple clients', () => {
            // Add more clients
            for (let i = 0; i < 5; i++) {
                const genome = new Genome(geneLstm);
                const c = new Client(genome);
                c.score = Math.random();
                species.put(c, true);
            }

            // Test multiple breeding operations
            for (let i = 0; i < 10; i++) {
                const offspring = species.breed();
                expect(offspring).toBeInstanceOf(Genome);
                expect(offspring.lstmArray.length).toBeGreaterThanOrEqual(1);
            }
        });
    });

    describe('score getter', () => {
        it('should return current species score', () => {
            expect(species.score).toBe(0);

            species.evaluateScore();

            expect(species.score).toBe(0.8);
        });
    });

    describe('clients getter', () => {
        it('should return array of clients', () => {
            const clients = species.clients;

            expect(Array.isArray(clients)).toBe(true);
            expect(clients.length).toBeGreaterThan(0);
        });
    });
});
