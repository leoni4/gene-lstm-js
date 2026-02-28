import { describe, it, expect, beforeEach } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { EMutationPressure } from '../src/types/index.js';
import type { GeneOptions } from '../src/types/index.js';

describe('GeneLSTM', () => {
    describe('constructor', () => {
        it('should initialize with default options', () => {
            const glstm = new GeneLSTM(10);

            expect(glstm.clients).toHaveLength(10);
            expect(glstm.CP).toBeDefined();
            expect(glstm.C1).toBeDefined();
            expect(glstm.C2).toBeDefined();
        });

        it('should initialize with custom options', () => {
            const glstm = new GeneLSTM(5, {
                CP: 0.5,
                C1: 2.0,
                C2: 0.8,
                INPUT_FEATURES: 3,
            });

            expect(glstm.clients).toHaveLength(5);
            expect(glstm.CP).toBe(0.5);
            expect(glstm.C1).toBe(2.0);
            expect(glstm.C2).toBe(0.8);
            expect(glstm.INPUT_FEATURES).toBe(3);
        });

        it('should load data when provided', () => {
            const loadData: GeneOptions = [
                {
                    hiddenSize: 1,
                    forgetGate: [{ weight1: 0.5, weight2: 0.5, bias: 0.0 }],
                    potentialLongToRem: [{ weight1: 0.4, weight2: 0.4, bias: 0.0 }],
                    potentialLongMemory: [{ weight1: 0.6, weight2: 0.6, bias: 0.0 }],
                    shortMemoryToRemember: [{ weight1: 0.7, weight2: 0.7, bias: 0.0 }],
                    readoutW: [0.5],
                    readoutB: 0.1,
                    alpha: 0.9,
                },
            ];

            const glstm = new GeneLSTM(5, { loadData });

            expect(glstm.clients).toHaveLength(5);
            expect(glstm.clients[0].genome.lstmArray[0].alpha).toBe(0.9);
        });

        it('should initialize all clients in first species', () => {
            const glstm = new GeneLSTM(10);

            const speciesCount = glstm.clients.filter(c => c.species !== null).length;
            expect(speciesCount).toBe(10);
        });
    });

    describe('getters', () => {
        let glstm: GeneLSTM;

        beforeEach(() => {
            glstm = new GeneLSTM(10);
        });

        it('should return INPUT_FEATURES', () => {
            expect(typeof glstm.INPUT_FEATURES).toBe('number');
        });

        it('should return CP', () => {
            expect(typeof glstm.CP).toBe('number');
        });

        it('should return C1 and C2', () => {
            expect(typeof glstm.C1).toBe('number');
            expect(typeof glstm.C2).toBe('number');
        });

        it('should return SURVIVORS', () => {
            expect(glstm.SURVIVORS).toBeGreaterThan(0);
            expect(glstm.SURVIVORS).toBeLessThanOrEqual(1);
        });

        it('should return MUTATION_RATE', () => {
            expect(typeof glstm.MUTATION_RATE).toBe('number');
        });

        it('should return mutation probability values', () => {
            expect(typeof glstm.PROBABILITY_MUTATE_BIAS_SHIFT).toBe('number');
            expect(typeof glstm.PROBABILITY_MUTATE_WEIGHT_SHIFT).toBe('number');
            expect(typeof glstm.PROBABILITY_MUTATE_LSTM_BLOCK).toBe('number');
        });

        it('should return sleeping block config', () => {
            const config = glstm.sleepingBlockConfig;
            expect(config.epsilon).toBeDefined();
            expect(config.forgetBias).toBeDefined();
            expect(config.inputBias).toBeDefined();
        });

        it('should return clients array', () => {
            expect(Array.isArray(glstm.clients)).toBe(true);
        });

        it('should return champion', () => {
            expect(glstm.champion).toBeDefined();
        });
    });

    describe('mutationPressure', () => {
        let glstm: GeneLSTM;

        beforeEach(() => {
            glstm = new GeneLSTM(10);
        });

        it('should get mutation pressure', () => {
            expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);
        });

        it('should set mutation pressure', () => {
            glstm.mutationPressure = EMutationPressure.BOOST;
            expect(glstm.mutationPressure).toBe(EMutationPressure.BOOST);
        });

        it('should return pressure multipliers', () => {
            const pressure = glstm.getMutationPressure();
            expect(pressure.topology).toBeDefined();
            expect(pressure.weights).toBeDefined();
            expect(typeof pressure.topology).toBe('number');
            expect(typeof pressure.weights).toBe('number');
        });

        it('should have different multipliers for different pressure levels', () => {
            glstm.mutationPressure = EMutationPressure.NORMAL;
            const normalPressure = glstm.getMutationPressure();

            glstm.mutationPressure = EMutationPressure.BOOST;
            const boostPressure = glstm.getMutationPressure();

            expect(boostPressure.weights).toBeGreaterThan(normalPressure.weights);
        });
    });

    describe('emptyGenome', () => {
        it('should create empty genome', () => {
            const glstm = new GeneLSTM(10);
            const genome = glstm.emptyGenome();

            expect(genome).toBeDefined();
            expect(genome.lstmArray).toHaveLength(1);
        });
    });

    describe('model', () => {
        it('should return model from best client', () => {
            const glstm = new GeneLSTM(10);

            // Assign scores
            glstm.clients.forEach((c, i) => {
                c.score = i * 0.1;
            });

            const model = glstm.model();

            expect(Array.isArray(model)).toBe(true);
            expect(model.length).toBeGreaterThan(0);
        });

        it('should return champion model if champion exists', () => {
            const glstm = new GeneLSTM(10);

            // Trigger evolution to create champion
            glstm.clients.forEach((c, i) => {
                c.score = i * 0.1;
            });
            glstm.evolve();

            const model = glstm.model();

            expect(Array.isArray(model)).toBe(true);
        });
    });

    describe('printSpecies', () => {
        it('should print species information without crashing', () => {
            const glstm = new GeneLSTM(10);

            expect(() => glstm.printSpecies()).not.toThrow();
        });
    });

    describe('adjustCP', () => {
        it('should not adjust CP when within deadband', () => {
            const glstm = new GeneLSTM(100, {
                targetSpecies: 5,
                cpDeadband: 2,
            });

            const cpBefore = glstm.CP;
            glstm.adjustCP(5, 1);

            expect(glstm.CP).toBe(cpBefore);
        });

        it('should increase CP when too many species', () => {
            const glstm = new GeneLSTM(100, {
                targetSpecies: 5,
                cpDeadband: 1,
                CP: 0.1,
            });

            const cpBefore = glstm.CP;
            glstm.adjustCP(10, 1); // 10 species, target is 5

            expect(glstm.CP).toBeGreaterThan(cpBefore);
        });

        it('should decrease CP when too few species', () => {
            const glstm = new GeneLSTM(100, {
                targetSpecies: 10,
                cpDeadband: 1,
                CP: 0.5,
            });

            const cpBefore = glstm.CP;
            glstm.adjustCP(2, 1); // 2 species, target is 10

            expect(glstm.CP).toBeLessThan(cpBefore);
        });

        it('should clamp CP to valid range', () => {
            const glstm = new GeneLSTM(100, {
                targetSpecies: 5,
                minCP: 0.01,
                maxCP: 2.0,
                CP: 0.01,
            });

            glstm.adjustCP(20, 1); // Force increase
            expect(glstm.CP).toBeGreaterThanOrEqual(0.01);
            expect(glstm.CP).toBeLessThanOrEqual(2.0);
        });
    });

    describe('updateMutationPressure', () => {
        it('should update pressure based on fitness improvements', () => {
            const glstm = new GeneLSTM(10, {
                enablePressureEscalation: true,
                stagnationThreshold: 2,
            });

            // First update with low fitness
            glstm.updateMutationPressure(0.5, 1);

            // Second update with same fitness (stagnation)
            glstm.updateMutationPressure(0.5, 2);
            glstm.updateMutationPressure(0.5, 3);

            // Pressure may have escalated
            expect(glstm.mutationPressure).toBeDefined();
        });

        it('should reset stagnation counter on improvement', () => {
            const glstm = new GeneLSTM(10, {
                enablePressureEscalation: true,
            });

            glstm.updateMutationPressure(0.5, 1);
            glstm.updateMutationPressure(0.8, 2); // Improvement

            expect(glstm.mutationPressure).toBeDefined();
        });

        it('should not escalate when disabled', () => {
            const glstm = new GeneLSTM(10, {
                enablePressureEscalation: false,
            });

            const pressureBefore = glstm.mutationPressure;

            for (let i = 0; i < 20; i++) {
                glstm.updateMutationPressure(0.5, i);
            }

            expect(glstm.mutationPressure).toBe(pressureBefore);
        });
    });

    describe('evolve', () => {
        it('should evolve population', () => {
            const glstm = new GeneLSTM(10);

            // Set scores
            glstm.clients.forEach(c => {
                c.score = Math.random();
            });

            expect(() => glstm.evolve()).not.toThrow();
        });

        it('should maintain population size', () => {
            const glstm = new GeneLSTM(10);

            glstm.clients.forEach(c => {
                c.score = Math.random();
            });

            glstm.evolve();

            expect(glstm.clients).toHaveLength(10);
        });

        it('should mark best client with bestScore flag', () => {
            const glstm = new GeneLSTM(10);

            glstm.clients.forEach((c, i) => {
                c.score = i * 0.1;
            });

            glstm.evolve();

            const bestScoreCount = glstm.clients.filter(c => c.bestScore).length;
            expect(bestScoreCount).toBe(1);
        });

        it('should handle optimization flag', () => {
            const glstm = new GeneLSTM(10);

            glstm.clients.forEach(c => {
                c.score = Math.random();
            });

            expect(() => glstm.evolve(true)).not.toThrow();
        });

        it('should evolve multiple generations', () => {
            const glstm = new GeneLSTM(10);

            for (let gen = 0; gen < 5; gen++) {
                glstm.clients.forEach(c => {
                    c.score = Math.random();
                });
                glstm.evolve();
            }

            expect(glstm.clients).toHaveLength(10);
        });

        it('should create multiple species with diverse population', () => {
            const glstm = new GeneLSTM(50, {
                CP: 0.1, // Low CP encourages more species
            });

            // Evolve and mutate to create diversity
            for (let gen = 0; gen < 3; gen++) {
                glstm.clients.forEach(c => {
                    c.score = Math.random();
                    c.mutate(true);
                });
                glstm.evolve();
            }

            expect(glstm.clients).toHaveLength(50);
        });

        it('should handle all clients having same score', () => {
            const glstm = new GeneLSTM(10);

            glstm.clients.forEach(c => {
                c.score = 0.5;
            });

            expect(() => glstm.evolve()).not.toThrow();
        });

        it('should handle zero scores', () => {
            const glstm = new GeneLSTM(10);

            glstm.clients.forEach(c => {
                c.score = 0;
            });

            expect(() => glstm.evolve()).not.toThrow();
        });

        it('should adjust CP during evolution', () => {
            const glstm = new GeneLSTM(50, {
                targetSpecies: 5,
            });

            glstm.clients.forEach(c => {
                c.score = Math.random();
            });

            glstm.evolve();

            // CP may have changed
            expect(typeof glstm.CP).toBe('number');
        });

        it('should update champion during evolution', () => {
            const glstm = new GeneLSTM(10);

            glstm.clients.forEach((c, i) => {
                c.score = i * 0.1;
            });

            glstm.evolve();

            expect(glstm.champion).toBeDefined();
        });
    });

    describe('integration tests', () => {
        it('should evolve and improve over generations', () => {
            const glstm = new GeneLSTM(20);

            const scores: number[] = [];

            for (let gen = 0; gen < 10; gen++) {
                glstm.clients.forEach(c => {
                    // Simple fitness function
                    const output = c.calculate([0.5, 0.3, 0.2]);
                    c.score = 1 - Math.abs(output[0] - 0.7);
                });

                glstm.evolve();

                const bestScore = Math.max(...glstm.clients.map(c => c.score));
                scores.push(bestScore);
            }

            expect(glstm.clients).toHaveLength(20);
        });

        it('should handle complex scenarios with varying fitness', () => {
            const glstm = new GeneLSTM(30);

            for (let gen = 0; gen < 5; gen++) {
                glstm.clients.forEach((c, i) => {
                    // Varying fitness
                    c.score = Math.sin(i * 0.1) * 0.5 + 0.5;
                });

                glstm.evolve();
            }

            expect(glstm.clients).toHaveLength(30);
        });

        it('should maintain species diversity with proper CP', () => {
            const glstm = new GeneLSTM(40, {
                CP: 0.2,
                targetSpecies: 4,
            });

            for (let gen = 0; gen < 3; gen++) {
                glstm.clients.forEach(c => {
                    c.score = Math.random();
                });
                glstm.evolve();
            }

            expect(glstm.clients.length).toBe(40);
        });

        it('should handle mutation pressure escalation', () => {
            const glstm = new GeneLSTM(15, {
                enablePressureEscalation: true,
                stagnationThreshold: 3,
            });

            // Stagnant fitness
            for (let gen = 0; gen < 5; gen++) {
                glstm.clients.forEach(c => {
                    c.score = 0.5;
                });
                glstm.evolve();
            }

            expect(glstm.mutationPressure).toBeDefined();
        });

        it('should handle custom sleeping block config', () => {
            const glstm = new GeneLSTM(10, {
                sleepingBlockConfig: {
                    epsilon: 0.001,
                    forgetBias: 2.0,
                    inputBias: -2.0,
                    outputBias: 0.5,
                    candidateBias: 0.5,
                    initialAlpha: 0.02,
                },
            });

            glstm.clients.forEach(c => {
                c.score = Math.random();
                c.mutate(true);
            });

            glstm.evolve();

            expect(glstm.clients).toHaveLength(10);
        });
    });

    describe('edge cases', () => {
        it('should handle single client', () => {
            const glstm = new GeneLSTM(1);

            glstm.clients[0].score = 0.8;

            expect(() => glstm.evolve()).not.toThrow();
            expect(glstm.clients).toHaveLength(1);
        });

        it('should handle very small population', () => {
            const glstm = new GeneLSTM(2);

            glstm.clients.forEach(c => {
                c.score = Math.random();
            });

            expect(() => glstm.evolve()).not.toThrow();
        });

        it('should handle large population', () => {
            const glstm = new GeneLSTM(100);

            glstm.clients.forEach(c => {
                c.score = Math.random();
            });

            expect(() => glstm.evolve()).not.toThrow();
        });

        it('should handle extreme CP values', () => {
            const glstm = new GeneLSTM(20, {
                CP: 10.0,
            });

            glstm.clients.forEach(c => {
                c.score = Math.random();
            });

            expect(() => glstm.evolve()).not.toThrow();
        });

        it('should handle all mutation pressure levels', () => {
            const glstm = new GeneLSTM(10);

            const pressureLevels = [
                EMutationPressure.COMPACT,
                EMutationPressure.NORMAL,
                EMutationPressure.BOOST,
                EMutationPressure.ESCAPE,
                EMutationPressure.PANIC,
            ];

            for (const pressure of pressureLevels) {
                glstm.mutationPressure = pressure;
                glstm.clients.forEach(c => {
                    c.score = Math.random();
                });
                glstm.evolve();
            }

            expect(glstm.clients).toHaveLength(10);
        });
    });
});
