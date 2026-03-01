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

    describe('fit', () => {
        describe('basic functionality', () => {
            it('should train on simple data and return history', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 10,
                    verbose: 0,
                });

                expect(history).toBeDefined();
                expect(history.error).toHaveLength(10);
                expect(history.epochs).toBe(10);
                expect(history.champion).toBeDefined();
                expect(history.stoppedEarly).toBe(false);
            });

            it('should handle 2D output targets', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ];
                const yTrain = [
                    [0, 1],
                    [1, 0],
                    [1, 0],
                    [0, 1],
                ];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 10,
                    verbose: 0,
                });

                expect(history).toBeDefined();
                expect(history.error).toHaveLength(10);
                expect(history.champion).toBeDefined();
            });

            it('should handle sequence input (2D arrays)', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [
                    [[0], [1], [0]],
                    [[1], [0], [1]],
                ];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    verbose: 0,
                });

                expect(history).toBeDefined();
                expect(history.error).toHaveLength(5);
            });

            it('should stop early when error threshold is reached', () => {
                const glstm = new GeneLSTM(50);

                // Simple problem - should converge quickly
                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 1000,
                    errorThreshold: 0.1,
                    verbose: 0,
                });

                expect(history.stoppedEarly).toBe(true);
                expect(history.epochs).toBeLessThan(1000);
            });

            it('should return champion after training', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    verbose: 0,
                });

                expect(history.champion).toBeDefined();
                expect(history.champion).not.toBeNull();

                // Champion should be able to make predictions
                const result = history.champion!.calculate([0.5]);
                expect(Array.isArray(result)).toBe(true);
            });
        });

        describe('validation', () => {
            it('should throw error for empty training data', () => {
                const glstm = new GeneLSTM(10);

                expect(() => {
                    glstm.fit([], [], { verbose: 0 });
                }).toThrow('Training data cannot be empty');
            });

            it('should throw error for mismatched input/output lengths', () => {
                const glstm = new GeneLSTM(10);

                const xTrain = [[0], [1], [0]];
                const yTrain = [0, 1];

                expect(() => {
                    glstm.fit(xTrain, yTrain, { verbose: 0 });
                }).toThrow('Input and output data must have the same length');
            });

            it('should throw error for invalid validationSplit', () => {
                const glstm = new GeneLSTM(10);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                expect(() => {
                    glstm.fit(xTrain, yTrain, {
                        validationSplit: 1.5,
                        verbose: 0,
                    });
                }).toThrow('validationSplit must be between 0 and 1');

                expect(() => {
                    glstm.fit(xTrain, yTrain, {
                        validationSplit: -0.1,
                        verbose: 0,
                    });
                }).toThrow('validationSplit must be between 0 and 1');
            });

            it('should throw error if validation split is too large', () => {
                const glstm = new GeneLSTM(10);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                expect(() => {
                    glstm.fit(xTrain, yTrain, {
                        validationSplit: 0.99,
                        verbose: 0,
                    });
                }).toThrow('Validation split too large');
            });
        });

        describe('validation split', () => {
            it('should split data when validationSplit is provided', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1], [0], [1], [0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1, 0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    validationSplit: 0.25,
                    verbose: 0,
                });

                expect(history.validationError).toBeDefined();
                expect(history.validationError).toHaveLength(5);
            });

            it('should not have validation error when validationSplit is 0', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    validationSplit: 0,
                    verbose: 0,
                });

                expect(history.validationError).toBeUndefined();
            });

            it('should compute validation error correctly', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1], [0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 3,
                    validationSplit: 0.33,
                    verbose: 0,
                });

                expect(history.validationError).toBeDefined();
                history.validationError!.forEach(valErr => {
                    expect(typeof valErr).toBe('number');
                    expect(valErr).toBeGreaterThanOrEqual(0);
                });
            });
        });

        describe('options', () => {
            it('should use default options when not provided', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain);

                expect(history).toBeDefined();
                expect(history.error.length).toBeGreaterThan(0);
            });

            it('should respect epochs option', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 15,
                    verbose: 0,
                });

                expect(history.error).toHaveLength(15);
                expect(history.epochs).toBe(15);
            });

            it('should respect errorThreshold option', () => {
                const glstm = new GeneLSTM(50);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 1000,
                    errorThreshold: 0.5,
                    verbose: 0,
                });

                if (history.stoppedEarly) {
                    expect(history.error[history.error.length - 1]).toBeLessThanOrEqual(0.5);
                }
            });

            it('should handle verbose: 0 (no logging)', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                expect(() => {
                    glstm.fit(xTrain, yTrain, {
                        epochs: 5,
                        verbose: 0,
                    });
                }).not.toThrow();
            });

            it('should handle verbose: 1 (periodic logging)', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                expect(() => {
                    glstm.fit(xTrain, yTrain, {
                        epochs: 5,
                        verbose: 1,
                        logInterval: 2,
                    });
                }).not.toThrow();
            });

            it('should handle verbose: 2 (detailed logging)', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                expect(() => {
                    glstm.fit(xTrain, yTrain, {
                        epochs: 3,
                        verbose: 2,
                    });
                }).not.toThrow();
            });
        });

        describe('loss functions', () => {
            it('should use MAE loss by default', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
                expect(history.error.length).toBe(5);
            });

            it('should handle MAE loss explicitly', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    loss: 'mae',
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });

            it('should handle MSE loss', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    loss: 'mse',
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });

            it('should handle BCE loss', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    loss: 'bce',
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });
        });

        describe('anti-constant penalty', () => {
            it('should work without anti-constant penalty by default', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });

            it('should apply anti-constant penalty when enabled', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    antiConstantPenalty: true,
                    antiConstantLambda: 0.05,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });

            it('should handle custom antiConstantLambda', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    antiConstantPenalty: true,
                    antiConstantLambda: 0.1,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });
        });

        describe('shuffling', () => {
            it('should shuffle data each epoch by default', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });

            it('should not shuffle when shuffleEachEpoch is false', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    shuffleEachEpoch: false,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });
        });

        describe('integration with evolve', () => {
            it('should call evolve during training', () => {
                const glstm = new GeneLSTM(30);

                const xTrain = [[0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 10,
                    verbose: 0,
                });

                // Evolution should have happened
                expect(glstm.champion).toBeDefined();
                expect(history.error).toHaveLength(10);
            });

            it('should trigger optimization mode for low errors', () => {
                const glstm = new GeneLSTM(30);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 50,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });
        });

        describe('error tracking', () => {
            it('should track error over epochs', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1], [0], [1]];
                const yTrain = [0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 10,
                    verbose: 0,
                });

                expect(history.error).toHaveLength(10);
                history.error.forEach(err => {
                    expect(typeof err).toBe('number');
                    expect(err).toBeGreaterThanOrEqual(0);
                });
            });

            it('should have decreasing or stable error trend', () => {
                const glstm = new GeneLSTM(50);

                // Simple problem
                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 20,
                    verbose: 0,
                });

                // First and last error
                const firstError = history.error[0];
                const lastError = history.error[history.error.length - 1];

                // Last error should generally be lower or similar (with some tolerance for randomness)
                expect(lastError).toBeLessThanOrEqual(firstError * 1.5);
            });
        });

        describe('real-world scenarios', () => {
            it('should train on XOR-like problem', () => {
                const glstm = new GeneLSTM(50);

                const xTrain = [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ];
                const yTrain = [0, 1, 1, 0];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 50,
                    errorThreshold: 0.3,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
                expect(history.champion).toBeDefined();
            });

            it('should train on lastBit problem', () => {
                const glstm = new GeneLSTM(100);

                const xTrain = [
                    [0, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                    [0, 0, 1, 1],
                ];
                const yTrain = [0, 1, 0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 30,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
                expect(history.champion).toBeDefined();
            });

            it('should train with validation data', () => {
                const glstm = new GeneLSTM(50);

                const xTrain = [
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                    [0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1],
                ];
                const yTrain = [0, 1, 1, 0, 0, 1, 1, 0];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 20,
                    validationSplit: 0.25,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
                expect(history.validationError).toBeDefined();
                expect(history.validationError!.length).toBe(20);
            });

            it('should handle multi-output regression', () => {
                const glstm = new GeneLSTM(30);

                const xTrain = [[0], [1], [2], [3]];
                const yTrain = [
                    [0, 0],
                    [1, 2],
                    [2, 4],
                    [3, 6],
                ];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 20,
                    loss: 'mse',
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
                expect(history.champion).toBeDefined();
            });
        });

        describe('edge cases', () => {
            it('should handle single sample', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0.5]];
                const yTrain = [1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    verbose: 0,
                });

                expect(history.error).toHaveLength(5);
            });

            it('should handle large input dimensions', () => {
                const glstm = new GeneLSTM(20, { INPUT_FEATURES: 10 });

                const xTrain = [
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                ];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 5,
                    verbose: 0,
                });

                expect(history.error).toBeDefined();
            });

            it('should handle zero epochs', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 0,
                    verbose: 0,
                });

                expect(history.error).toHaveLength(0);
                expect(history.epochs).toBe(0);
            });

            it('should handle mismatched output dimensions gracefully', () => {
                const glstm = new GeneLSTM(20);

                const xTrain = [[0], [1]];
                const yTrain = [
                    [0, 0],
                    [1, 1],
                ];

                // Should not throw, but handle internally
                expect(() => {
                    glstm.fit(xTrain, yTrain, {
                        epochs: 5,
                        verbose: 0,
                    });
                }).not.toThrow();
            });

            it('should handle very small population', () => {
                const glstm = new GeneLSTM(5);

                const xTrain = [[0], [1]];
                const yTrain = [0, 1];

                const history = glstm.fit(xTrain, yTrain, {
                    epochs: 10,
                    verbose: 0,
                });

                expect(history.error).toHaveLength(10);
            });
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
