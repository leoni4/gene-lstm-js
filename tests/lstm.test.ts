import { describe, it, expect, beforeEach } from 'vitest';
import { LSTM, ShortMemoryBlock, OutputBlock } from '../src/lstm.js';
import { GeneLSTM } from '../src/gLstm.js';
import type { LstmOptions } from '../src/types/index.js';

describe('ShortMemoryBlock', () => {
    describe('constructor', () => {
        it('should initialize with sigmoid activation', () => {
            const block = new ShortMemoryBlock('sigmoid');
            expect(block.weight1).toBeGreaterThanOrEqual(-1);
            expect(block.weight1).toBeLessThanOrEqual(1);
        });

        it('should initialize with tanh activation', () => {
            const block = new ShortMemoryBlock('tanh');
            expect(block.weight2).toBeGreaterThanOrEqual(-1);
            expect(block.weight2).toBeLessThanOrEqual(1);
        });

        it('should accept custom weights', () => {
            const block = new ShortMemoryBlock('sigmoid', 0.5, 0.3, 0.1);
            expect(block.weight1).toBe(0.5);
            expect(block.weight2).toBe(0.3);
            expect(block.bias).toBe(0.1);
        });

        it('should accept weightIn parameter', () => {
            const weightIn = [0.1, 0.2, 0.3];
            const block = new ShortMemoryBlock('sigmoid', 0.5, 0.3, 0.1, weightIn);
            expect(block.weightIn).toEqual(weightIn);
        });
    });

    describe('calculate', () => {
        it('should calculate with scalar input', () => {
            const block = new ShortMemoryBlock('sigmoid', 0.5, 0.5, 0.0);
            const result = block.calculate(0.5, 0.5);

            expect(typeof result).toBe('number');
            expect(isFinite(result)).toBe(true);
            expect(result).toBeGreaterThanOrEqual(0);
            expect(result).toBeLessThanOrEqual(1);
        });

        it('should calculate with vector input', () => {
            const block = new ShortMemoryBlock('sigmoid', 0.5, 0.5, 0.0);
            const input = [0.5, 0.3, 0.2];
            const result = block.calculate(input, 0.5);

            expect(typeof result).toBe('number');
            expect(isFinite(result)).toBe(true);
        });

        it('should initialize weightIn automatically for vector input', () => {
            const block = new ShortMemoryBlock('sigmoid', 0.5, 0.5, 0.0);
            expect(block.weightIn).toBeUndefined();

            const input = [0.5, 0.3];
            block.calculate(input, 0.5);

            expect(block.weightIn).toBeDefined();
            expect(block.weightIn).toHaveLength(2);
        });

        it('should use tanh activation when specified', () => {
            const block = new ShortMemoryBlock('tanh', 0.5, 0.5, 0.0);
            const result = block.calculate(0.5, 0.5);

            expect(result).toBeGreaterThanOrEqual(-1);
            expect(result).toBeLessThanOrEqual(1);
        });
    });
});

describe('OutputBlock', () => {
    it('should calculate output from long and short memory', () => {
        const block = new OutputBlock();
        const result = block.calculate(0.5, 0.8);

        expect(typeof result).toBe('number');
        expect(isFinite(result)).toBe(true);
    });

    it('should handle zero values', () => {
        const block = new OutputBlock();
        const result = block.calculate(0, 0);

        expect(result).toBe(0);
    });

    it('should handle negative values', () => {
        const block = new OutputBlock();
        const result = block.calculate(-0.5, 0.8);

        expect(typeof result).toBe('number');
        expect(isFinite(result)).toBe(true);
    });
});

describe('LSTM', () => {
    let geneLstm: GeneLSTM;

    beforeEach(() => {
        geneLstm = new GeneLSTM(10);
    });

    describe('constructor', () => {
        it('should initialize with default options', () => {
            const lstm = new LSTM(geneLstm);

            expect(lstm.alpha).toBe(1.0);
            expect(lstm.longMemory).toBeDefined();
            expect(lstm.shortMemory).toBeDefined();
            expect(lstm.readoutW).toBeDefined();
        });

        it('should initialize with custom options', () => {
            const options: LstmOptions = {
                hiddenSize: 2,
                forgetGate: [
                    { weight1: 0.5, weight2: 0.5, bias: 0.0 },
                    { weight1: 0.6, weight2: 0.6, bias: 0.1 },
                ],
                potentialLongToRem: [
                    { weight1: 0.4, weight2: 0.4, bias: 0.0 },
                    { weight1: 0.5, weight2: 0.5, bias: 0.1 },
                ],
                potentialLongMemory: [
                    { weight1: 0.6, weight2: 0.6, bias: 0.0 },
                    { weight1: 0.7, weight2: 0.7, bias: 0.1 },
                ],
                shortMemoryToRemember: [
                    { weight1: 0.7, weight2: 0.7, bias: 0.0 },
                    { weight1: 0.8, weight2: 0.8, bias: 0.1 },
                ],
                readoutW: [0.5, 0.3],
                readoutB: 0.1,
                alpha: 0.9,
            };

            const lstm = new LSTM(geneLstm, options);

            expect(lstm.alpha).toBe(0.9);
            expect(lstm.readoutW).toEqual([0.5, 0.3]);
            expect(lstm.readoutB).toBe(0.1);
        });

        it('should handle weightIn in gate options', () => {
            const options: LstmOptions = {
                hiddenSize: 1,
                forgetGate: [{ weight1: 0.5, weight2: 0.5, bias: 0.0, weightIn: [0.1, 0.2] }],
                potentialLongToRem: [{ weight1: 0.4, weight2: 0.4, bias: 0.0, weightIn: [0.2, 0.3] }],
                potentialLongMemory: [{ weight1: 0.6, weight2: 0.6, bias: 0.0, weightIn: [0.3, 0.4] }],
                shortMemoryToRemember: [{ weight1: 0.7, weight2: 0.7, bias: 0.0, weightIn: [0.4, 0.5] }],
                readoutW: [0.5],
                readoutB: 0.1,
                alpha: 0.9,
            };

            const lstm = new LSTM(geneLstm, options);

            expect(lstm).toBeDefined();
        });
    });

    describe('alpha property', () => {
        it('should get and set alpha', () => {
            const lstm = new LSTM(geneLstm);

            lstm.alpha = 0.5;
            expect(lstm.alpha).toBe(0.5);
        });

        it('should clamp alpha to [0, 1]', () => {
            const lstm = new LSTM(geneLstm);

            lstm.alpha = 1.5;
            expect(lstm.alpha).toBe(1.0);

            lstm.alpha = -0.5;
            expect(lstm.alpha).toBe(0.0);
        });
    });

    describe('calculate', () => {
        it('should process scalar sequence', () => {
            const lstm = new LSTM(geneLstm);
            const input = [0.5, 0.3, 0.2];

            const output = lstm.calculate(input);

            expect(Array.isArray(output)).toBe(true);
            expect(output.length).toBe(1);
            expect(typeof output[0]).toBe('number');
        });

        it('should process vector sequence', () => {
            const lstm = new LSTM(geneLstm);
            const input = [
                [0.5, 0.3],
                [0.2, 0.4],
                [0.1, 0.6],
            ];

            const output = lstm.calculate(input);

            expect(Array.isArray(output)).toBe(true);
            expect(output.length).toBe(1);
        });

        it('should return full sequence when fullSeq is true', () => {
            const lstm = new LSTM(geneLstm);
            const input = [0.5, 0.3, 0.2, 0.4];

            const output = lstm.calculate(input, true);

            expect(Array.isArray(output)).toBe(true);
            expect(output.length).toBe(input.length);
        });

        it('should apply alpha mixing', () => {
            const lstm = new LSTM(geneLstm);
            lstm.alpha = 0.5;

            const input = [0.5, 0.3, 0.2];
            const output = lstm.calculate(input);

            expect(output).toBeDefined();
            expect(output.length).toBe(1);
        });

        it('should handle empty input', () => {
            const lstm = new LSTM(geneLstm);
            const input: number[] = [];

            const output = lstm.calculate(input);

            expect(Array.isArray(output)).toBe(true);
        });

        it('should process 2D sequences with fullSeq', () => {
            const lstm = new LSTM(geneLstm);
            const input = [
                [0.5, 0.3],
                [0.2, 0.4],
            ];

            const output = lstm.calculate(input, true);

            expect(Array.isArray(output)).toBe(true);
            expect(output.length).toBe(input.length);
        });
    });

    describe('flattenWeights', () => {
        it('should flatten all weights to array', () => {
            const lstm = new LSTM(geneLstm);

            const weights = lstm.flattenWeights();

            expect(Array.isArray(weights)).toBe(true);
            expect(weights.length).toBeGreaterThan(0);
            weights.forEach(w => {
                expect(typeof w).toBe('number');
                expect(isFinite(w)).toBe(true);
            });
        });

        it('should include readout weights and alpha', () => {
            const lstm = new LSTM(geneLstm);
            lstm.alpha = 0.75;

            const weights = lstm.flattenWeights();

            // Alpha should be at the end
            expect(weights[weights.length - 1]).toBe(0.75);
        });
    });

    describe('model', () => {
        it('should serialize LSTM to options', () => {
            const lstm = new LSTM(geneLstm);

            const model = lstm.model();

            expect(model.hiddenSize).toBeDefined();
            expect(model.forgetGate).toBeDefined();
            expect(model.potentialLongToRem).toBeDefined();
            expect(model.potentialLongMemory).toBeDefined();
            expect(model.shortMemoryToRemember).toBeDefined();
            expect(model.readoutW).toBeDefined();
            expect(model.readoutB).toBeDefined();
            expect(model.alpha).toBeDefined();
        });

        it('should preserve alpha in model', () => {
            const lstm = new LSTM(geneLstm);
            lstm.alpha = 0.65;

            const model = lstm.model();

            expect(model.alpha).toBe(0.65);
        });

        it('should create model that can be used to recreate LSTM', () => {
            const lstm1 = new LSTM(geneLstm);
            const model = lstm1.model();

            const lstm2 = new LSTM(geneLstm, model);

            expect(lstm2.alpha).toBe(lstm1.alpha);
            expect(lstm2.readoutB).toBe(lstm1.readoutB);
        });
    });

    describe('mutate', () => {
        it('should mutate without crashing', () => {
            const lstm = new LSTM(geneLstm);

            expect(() => lstm.mutate()).not.toThrow();
        });

        it('should potentially change weights', () => {
            const glstmHighMutation = new GeneLSTM(10, {
                PROBABILITY_MUTATE_WEIGHT_SHIFT: 1.0,
                MUTATION_RATE: 1.0,
            });

            const lstm = new LSTM(glstmHighMutation);

            lstm.mutate();

            const weightsAfter = lstm.flattenWeights();

            // Weights should still be valid after mutation
            expect(weightsAfter.length).toBeGreaterThan(0);
            weightsAfter.forEach(w => {
                expect(typeof w).toBe('number');
                expect(isFinite(w)).toBe(true);
            });
        });

        it('should add units with high add probability', () => {
            const glstmHighAdd = new GeneLSTM(10, {
                PROBABILITY_MUTATE_ADD_UNIT: 1.0,
            });

            const lstm = new LSTM(glstmHighAdd);
            const sizeBefore = lstm.readoutW.length;

            lstm.mutate();

            expect(lstm.readoutW.length).toBeGreaterThan(sizeBefore);
        });

        it('should remove units with high remove probability', () => {
            const glstmHighRemove = new GeneLSTM(10, {
                PROBABILITY_MUTATE_REMOVE_UNIT: 1.0,
            });

            const options: LstmOptions = {
                hiddenSize: 3,
                forgetGate: [
                    { weight1: 0.5, weight2: 0.5, bias: 0.0 },
                    { weight1: 0.5, weight2: 0.5, bias: 0.0 },
                    { weight1: 0.5, weight2: 0.5, bias: 0.0 },
                ],
                potentialLongToRem: [
                    { weight1: 0.4, weight2: 0.4, bias: 0.0 },
                    { weight1: 0.4, weight2: 0.4, bias: 0.0 },
                    { weight1: 0.4, weight2: 0.4, bias: 0.0 },
                ],
                potentialLongMemory: [
                    { weight1: 0.6, weight2: 0.6, bias: 0.0 },
                    { weight1: 0.6, weight2: 0.6, bias: 0.0 },
                    { weight1: 0.6, weight2: 0.6, bias: 0.0 },
                ],
                shortMemoryToRemember: [
                    { weight1: 0.7, weight2: 0.7, bias: 0.0 },
                    { weight1: 0.7, weight2: 0.7, bias: 0.0 },
                    { weight1: 0.7, weight2: 0.7, bias: 0.0 },
                ],
                readoutW: [0.5, 0.3, 0.2],
                readoutB: 0.1,
                alpha: 0.9,
            };

            const lstm = new LSTM(glstmHighRemove, options);
            const sizeBefore = lstm.readoutW.length;

            lstm.mutate();

            // Should remove at least one unit (but keep at least 1)
            expect(lstm.readoutW.length).toBeLessThanOrEqual(sizeBefore);
            expect(lstm.readoutW.length).toBeGreaterThanOrEqual(1);
        });

        it('should not remove last unit', () => {
            const glstmHighRemove = new GeneLSTM(10, {
                PROBABILITY_MUTATE_REMOVE_UNIT: 1.0,
            });

            const lstm = new LSTM(glstmHighRemove);

            // Try multiple mutations
            for (let i = 0; i < 10; i++) {
                lstm.mutate();
            }

            // Should always keep at least 1 unit
            expect(lstm.readoutW.length).toBeGreaterThanOrEqual(1);
        });

        it('should mutate readout weights', () => {
            const glstmReadoutMutation = new GeneLSTM(10, {
                PROBABILITY_MUTATE_READOUT_W: 1.0,
                MUTATION_RATE: 1.0,
            });

            const lstm = new LSTM(glstmReadoutMutation);
            const readoutBefore = [...lstm.readoutW];

            lstm.mutate();

            expect(lstm.readoutW).toBeDefined();
            expect(lstm.readoutW.length).toBe(readoutBefore.length);
        });

        it('should mutate readout bias', () => {
            const glstmBiasMutation = new GeneLSTM(10, {
                PROBABILITY_MUTATE_READOUT_B: 1.0,
                MUTATION_RATE: 1.0,
            });

            const lstm = new LSTM(glstmBiasMutation);

            lstm.mutate();

            expect(typeof lstm.readoutB).toBe('number');
        });

        it('should mutate alpha', () => {
            const glstmAlphaMutation = new GeneLSTM(10, {
                PROBABILITY_MUTATE_ALPHA_SHIFT: 1.0,
                MUTATION_RATE: 1.0,
            });

            const lstm = new LSTM(glstmAlphaMutation);
            lstm.alpha = 0.5;

            lstm.mutate();

            expect(lstm.alpha).toBeGreaterThanOrEqual(0);
            expect(lstm.alpha).toBeLessThanOrEqual(1);
        });

        it('should respect mutation pressure', () => {
            const glstm = new GeneLSTM(10, {
                PROBABILITY_MUTATE_WEIGHT_SHIFT: 1.0,
                MUTATION_RATE: 1.0,
            });

            const lstm = new LSTM(glstm);

            // Mutate multiple times to test pressure scaling
            for (let i = 0; i < 5; i++) {
                lstm.mutate();
            }

            expect(lstm).toBeDefined();
        });
    });

    describe('edge cases', () => {
        it('should handle very small alpha values', () => {
            const lstm = new LSTM(geneLstm);
            lstm.alpha = 0.01;

            const output = lstm.calculate([0.5, 0.3]);

            expect(output).toBeDefined();
            expect(output.length).toBeGreaterThan(0);
        });

        it('should handle large hidden sizes', () => {
            const options: LstmOptions = {
                hiddenSize: 10,
                forgetGate: Array(10)
                    .fill(null)
                    .map(() => ({ weight1: 0.5, weight2: 0.5, bias: 0.0 })),
                potentialLongToRem: Array(10)
                    .fill(null)
                    .map(() => ({ weight1: 0.4, weight2: 0.4, bias: 0.0 })),
                potentialLongMemory: Array(10)
                    .fill(null)
                    .map(() => ({ weight1: 0.6, weight2: 0.6, bias: 0.0 })),
                shortMemoryToRemember: Array(10)
                    .fill(null)
                    .map(() => ({ weight1: 0.7, weight2: 0.7, bias: 0.0 })),
                readoutW: Array(10).fill(0.5),
                readoutB: 0.1,
                alpha: 0.9,
            };

            const lstm = new LSTM(geneLstm, options);
            const output = lstm.calculate([0.5, 0.3]);

            expect(output).toBeDefined();
        });

        it('should handle long sequences', () => {
            const lstm = new LSTM(geneLstm);
            const input = Array(100)
                .fill(0)
                .map(() => Math.random());

            const output = lstm.calculate(input);

            expect(output).toBeDefined();
            expect(output.length).toBe(1);
        });
    });
});
