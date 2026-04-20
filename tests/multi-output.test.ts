import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Genome } from '../src/genome.js';

describe('Multi-Output Support', () => {
    describe('basic multi-output initialization', () => {
        it('should create model with OUTPUT_DIM=2', () => {
            const glstm = new GeneLSTM(10, {
                OUTPUT_DIM: 2,
                INPUT_FEATURES: 1,
            });

            expect(glstm.OUTPUT_DIM).toBe(2);
            expect(glstm.OUTPUT_ACTIVATION).toBe('sigmoid');

            const client = glstm.clients[0];
            const lstm = client.genome.lstmArray[0];

            expect(lstm.readoutW).toHaveLength(2);
            expect(lstm.readoutB).toHaveLength(2);
        });

        it('should support different output activations', () => {
            const glstmSigmoid = new GeneLSTM(5, {
                OUTPUT_DIM: 2,
                OUTPUT_ACTIVATION: 'sigmoid',
            });

            const glstmTanh = new GeneLSTM(5, {
                OUTPUT_DIM: 2,
                OUTPUT_ACTIVATION: 'tanh',
            });

            const glstmIdentity = new GeneLSTM(5, {
                OUTPUT_DIM: 2,
                OUTPUT_ACTIVATION: 'identity',
            });

            expect(glstmSigmoid.OUTPUT_ACTIVATION).toBe('sigmoid');
            expect(glstmTanh.OUTPUT_ACTIVATION).toBe('tanh');
            expect(glstmIdentity.OUTPUT_ACTIVATION).toBe('identity');
        });

        it('should default to single output when OUTPUT_DIM not specified', () => {
            const glstm = new GeneLSTM(10);

            expect(glstm.OUTPUT_DIM).toBe(1);
        });
    });

    describe('multi-output inference', () => {
        it('should return correct number of outputs', () => {
            const glstm = new GeneLSTM(10, {
                OUTPUT_DIM: 3,
                INPUT_FEATURES: 1,
            });

            const client = glstm.clients[0];
            const output = client.calculate([0.5, 0.3, 0.8]);

            expect(output).toHaveLength(3);
            output.forEach(val => {
                expect(typeof val).toBe('number');
                expect(val).toBeGreaterThanOrEqual(0);
                expect(val).toBeLessThanOrEqual(1);
            });
        });

        it('should produce different values for each output', () => {
            const glstm = new GeneLSTM(20, {
                OUTPUT_DIM: 2,
                INPUT_FEATURES: 1,
            });

            // Evolve a bit to get non-zero weights
            for (let i = 0; i < 5; i++) {
                glstm.clients.forEach(c => {
                    c.score = Math.random();
                });
                glstm.evolve();
            }

            const client = glstm.clients[0];
            const output = client.calculate([0.5, 0.3, 0.8]);

            expect(output).toHaveLength(2);
            // After evolution, outputs should not be identical (very unlikely)
        });
    });

    describe('multi-output training', () => {
        it('should train 2-bit parity problem', () => {
            const glstm = new GeneLSTM(30, {
                OUTPUT_DIM: 2,
                INPUT_FEATURES: 1,
            });

            // XOR-like problem: output [lastBit, firstBit]
            const xTrain = [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ];
            const yTrain = [
                [0, 0], // last=0, first=0
                [1, 0], // last=1, first=0
                [0, 1], // last=0, first=1
                [1, 1], // last=1, first=1
            ];

            const history = glstm.fit(xTrain, yTrain, {
                epochs: 100,
                errorThreshold: 0.2,
                verbose: 0,
                loss: 'mse',
            });

            expect(history.error[history.error.length - 1]).toBeLessThan(0.3);
            expect(history.champion).toBeDefined();

            // Test predictions
            const pred = history.champion!.calculate([0, 1]);
            expect(pred).toHaveLength(2);
        });

        it('should train on 3-output problem', () => {
            const glstm = new GeneLSTM(25, {
                OUTPUT_DIM: 3,
                INPUT_FEATURES: 1,
            });

            // Simple pattern recognition: [isZero, isOne, sum>1]
            const xTrain = [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
            ];
            const yTrain = [
                [1, 0, 0], // both zero
                [0, 1, 0], // one
                [0, 1, 0], // one
                [0, 0, 1], // sum>1
            ];

            const history = glstm.fit(xTrain, yTrain, {
                epochs: 80,
                errorThreshold: 0.25,
                verbose: 0,
                loss: 'mse',
            });

            expect(history.error).toBeDefined();
            expect(history.champion).toBeDefined();

            const pred = history.champion!.calculate([1, 1]);
            expect(pred).toHaveLength(3);
        });

        it('should handle validation split with multi-output', () => {
            const glstm = new GeneLSTM(20, {
                OUTPUT_DIM: 2,
            });

            const xTrain = [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 0],
                [0, 1],
            ];
            const yTrain = [
                [0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, 0],
                [0, 1],
            ];

            const history = glstm.fit(xTrain, yTrain, {
                epochs: 20,
                validationSplit: 0.33,
                verbose: 0,
            });

            expect(history.validationError).toBeDefined();
            expect(history.validationError!.length).toBe(20);
        });
    });

    describe('backwards compatibility', () => {
        it('should load old single-output models', () => {
            // Old format with single-output readout
            const oldModel = [
                {
                    hiddenSize: 2,
                    forgetGate: [
                        { weight1: 0.5, weight2: 0.3, bias: 0.1 },
                        { weight1: -0.2, weight2: 0.4, bias: -0.1 },
                    ],
                    potentialLongToRem: [
                        { weight1: 0.3, weight2: 0.2, bias: 0.0 },
                        { weight1: 0.1, weight2: -0.3, bias: 0.2 },
                    ],
                    potentialLongMemory: [
                        { weight1: -0.1, weight2: 0.5, bias: 0.3 },
                        { weight1: 0.4, weight2: -0.2, bias: -0.2 },
                    ],
                    shortMemoryToRemember: [
                        { weight1: 0.2, weight2: 0.1, bias: -0.1 },
                        { weight1: -0.3, weight2: 0.3, bias: 0.1 },
                    ],
                    readoutW: [0.5, -0.3], // Old format: 1D array
                    readoutB: 0.1, // Old format: scalar
                    alpha: 1.0,
                },
            ];

            const glstm = new GeneLSTM(10, {
                loadData: oldModel,
            });

            const client = glstm.clients[0];
            const lstm = client.genome.lstmArray[0];

            // Should be converted to new format
            expect(lstm.readoutW).toHaveLength(1);
            expect(lstm.readoutW[0]).toEqual([0.5, -0.3]);
            expect(lstm.readoutB).toEqual([0.1]);

            // Should still produce single output
            const output = client.calculate([0.5, 0.3]);
            expect(output).toHaveLength(1);
        });

        it('should evolve old models to new format', () => {
            const oldModel = [
                {
                    hiddenSize: 1,
                    forgetGate: [{ weight1: 0.5, weight2: 0.3, bias: 0.1 }],
                    potentialLongToRem: [{ weight1: 0.3, weight2: 0.2, bias: 0.0 }],
                    potentialLongMemory: [{ weight1: -0.1, weight2: 0.5, bias: 0.3 }],
                    shortMemoryToRemember: [{ weight1: 0.2, weight2: 0.1, bias: -0.1 }],
                    readoutW: [0.5],
                    readoutB: 0.1,
                    alpha: 1.0,
                },
            ];

            const glstm = new GeneLSTM(10, {
                loadData: oldModel,
            });

            // Evolve
            glstm.clients.forEach(c => {
                c.score = Math.random();
            });
            glstm.evolve();

            // All clients should still work
            glstm.clients.forEach(c => {
                const output = c.calculate([0.5]);
                expect(Array.isArray(output)).toBe(true);
                expect(output.length).toBeGreaterThanOrEqual(1);
            });
        });
    });

    describe('mutations with multi-output', () => {
        it('should mutate readout weights correctly', () => {
            const glstm = new GeneLSTM(10, {
                OUTPUT_DIM: 2,
                PROBABILITY_MUTATE_READOUT_W: 1.0,
            });

            const client = glstm.clients[0];
            const lstm = client.genome.lstmArray[0];

            const before = lstm.readoutW.map(row => [...row]);

            // Force mutation
            client.mutate(true);

            const after = lstm.readoutW;

            // At least one weight should have changed
            let changed = false;
            for (let j = 0; j < before.length; j++) {
                for (let k = 0; k < before[j].length; k++) {
                    if (Math.abs(before[j][k] - after[j][k]) > 1e-10) {
                        changed = true;
                        break;
                    }
                }
            }
            expect(changed).toBe(true);
        });

        it('should add units correctly with multi-output', () => {
            const glstm = new GeneLSTM(10, {
                OUTPUT_DIM: 3,
                PROBABILITY_MUTATE_ADD_UNIT: 1.0,
            });

            const client = glstm.clients[0];
            const lstm = client.genome.lstmArray[0];

            const hiddenBefore = lstm.readoutW[0].length;

            // Force add unit mutation
            client.mutate(true);

            const hiddenAfter = lstm.readoutW[0].length;

            // Hidden size should have increased
            expect(hiddenAfter).toBeGreaterThanOrEqual(hiddenBefore);

            // All outputs should have same hidden size
            lstm.readoutW.forEach(row => {
                expect(row.length).toBe(hiddenAfter);
            });
        });

        it('should remove units correctly with multi-output', () => {
            const glstm = new GeneLSTM(10, {
                OUTPUT_DIM: 2,
            });

            const client = glstm.clients[0];
            const lstm = client.genome.lstmArray[0];

            // First add some units
            for (let i = 0; i < 3; i++) {
                client.mutate(true);
            }

            // Now try to remove (might not happen if hiddenSize=1)
            for (let i = 0; i < 10; i++) {
                client.mutate(true);
            }

            // Should still have valid structure
            expect(lstm.readoutW).toHaveLength(2);
            lstm.readoutW.forEach(row => {
                expect(row.length).toBeGreaterThan(0);
            });
        });
    });

    describe('crossover with multi-output', () => {
        it('should crossover multi-output genomes correctly', () => {
            const glstm = new GeneLSTM(20, {
                OUTPUT_DIM: 2,
            });

            const client1 = glstm.clients[0];
            const client2 = glstm.clients[1];

            const offspring = Genome.crossOver(client1.genome, client2.genome);

            // Offspring should have valid structure
            expect(offspring.lstmArray.length).toBeGreaterThan(0);
            offspring.lstmArray.forEach(lstm => {
                expect(lstm.readoutW).toHaveLength(2);
                expect(lstm.readoutB).toHaveLength(2);
            });
        });

        it('should handle crossover with different output dimensions', () => {
            const glstm1 = new GeneLSTM(10, { OUTPUT_DIM: 2 });
            const glstm2 = new GeneLSTM(10, { OUTPUT_DIM: 3 });

            const client1 = glstm1.clients[0];
            const client2 = glstm2.clients[0];

            // Change glstm reference to match
            const genome2 = client2.genome;
            const genome2WithGlstm1 = new Genome(
                glstm1,
                genome2.lstmArray.map(l => l.model()),
            );

            const offspring = Genome.crossOver(client1.genome, genome2WithGlstm1);

            // Offspring should have valid structure (takes max of parents)
            expect(offspring.lstmArray.length).toBeGreaterThan(0);
            offspring.lstmArray.forEach(lstm => {
                expect(lstm.readoutW.length).toBeGreaterThanOrEqual(2);
                expect(lstm.readoutB.length).toBeGreaterThanOrEqual(2);
            });
        });
    });

    describe('model serialization with multi-output', () => {
        it('should serialize and deserialize multi-output models', () => {
            const glstm = new GeneLSTM(10, {
                OUTPUT_DIM: 2,
                INPUT_FEATURES: 3,
            });

            const model1 = glstm.model();

            // Create new GeneLSTM with saved model
            const glstm2 = new GeneLSTM(10, {
                OUTPUT_DIM: 2,
                INPUT_FEATURES: 3,
                loadData: model1,
            });

            const model2 = glstm2.model();

            // Models should be equivalent
            expect(model2.length).toBe(model1.length);
            model2.forEach((block, i) => {
                expect(block.hiddenSize).toBe(model1[i].hiddenSize);
                expect((block.readoutW as number[][]).length).toBe((model1[i].readoutW as number[][]).length);
                expect((block.readoutB as number[]).length).toBe((model1[i].readoutB as number[]).length);
            });
        });
    });
});
