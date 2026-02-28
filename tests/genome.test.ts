import { describe, it, expect, beforeEach } from 'vitest';
import { Genome } from '../src/genome.js';
import { GeneLSTM } from '../src/gLstm.js';
import type { GeneOptions } from '../src/types/index.js';

describe('Genome', () => {
    let geneLstm: GeneLSTM;

    beforeEach(() => {
        geneLstm = new GeneLSTM(10);
    });

    describe('constructor', () => {
        it('should create genome with default LSTM when no data provided', () => {
            const genome = new Genome(geneLstm);

            expect(genome.lstmArray).toHaveLength(1);
            expect(genome.glstm).toBe(geneLstm);
        });

        it('should create genome with provided data', () => {
            const data: GeneOptions = [
                {
                    hiddenSize: 2,
                    forgetGate: [
                        { weight1: 0.5, weight2: 0.5, bias: 0.0 },
                        { weight1: 0.3, weight2: 0.3, bias: 0.1 },
                    ],
                    potentialLongToRem: [
                        { weight1: 0.4, weight2: 0.4, bias: 0.0 },
                        { weight1: 0.2, weight2: 0.2, bias: 0.1 },
                    ],
                    potentialLongMemory: [
                        { weight1: 0.6, weight2: 0.6, bias: 0.0 },
                        { weight1: 0.1, weight2: 0.1, bias: 0.1 },
                    ],
                    shortMemoryToRemember: [
                        { weight1: 0.7, weight2: 0.7, bias: 0.0 },
                        { weight1: 0.8, weight2: 0.8, bias: 0.1 },
                    ],
                    readoutW: [0.5, 0.3],
                    readoutB: 0.1,
                    alpha: 0.9,
                },
            ];

            const genome = new Genome(geneLstm, data);

            expect(genome.lstmArray).toHaveLength(1);
            expect(genome.lstmArray[0].alpha).toBe(0.9);
        });

        it('should create genome with multiple LSTM blocks', () => {
            const data: GeneOptions = [
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
                {
                    hiddenSize: 1,
                    forgetGate: [{ weight1: 0.3, weight2: 0.3, bias: 0.1 }],
                    potentialLongToRem: [{ weight1: 0.2, weight2: 0.2, bias: 0.1 }],
                    potentialLongMemory: [{ weight1: 0.4, weight2: 0.4, bias: 0.1 }],
                    shortMemoryToRemember: [{ weight1: 0.5, weight2: 0.5, bias: 0.1 }],
                    readoutW: [0.3],
                    readoutB: 0.2,
                    alpha: 0.8,
                },
            ];

            const genome = new Genome(geneLstm, data);

            expect(genome.lstmArray).toHaveLength(2);
        });
    });

    describe('distance', () => {
        it('should return 0 for identical genomes', () => {
            const genome1 = new Genome(geneLstm);
            const distance = genome1.distance(genome1);

            expect(distance).toBe(0);
        });

        it('should calculate distance between different genomes', () => {
            const genome1 = new Genome(geneLstm);
            const genome2 = new Genome(geneLstm);

            const distance = genome1.distance(genome2);

            expect(typeof distance).toBe('number');
            expect(distance).toBeGreaterThanOrEqual(0);
        });

        it('should handle genomes with different depths', () => {
            const data1: GeneOptions = [
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

            const data2: GeneOptions = [
                ...data1,
                {
                    hiddenSize: 1,
                    forgetGate: [{ weight1: 0.3, weight2: 0.3, bias: 0.1 }],
                    potentialLongToRem: [{ weight1: 0.2, weight2: 0.2, bias: 0.1 }],
                    potentialLongMemory: [{ weight1: 0.4, weight2: 0.4, bias: 0.1 }],
                    shortMemoryToRemember: [{ weight1: 0.5, weight2: 0.5, bias: 0.1 }],
                    readoutW: [0.3],
                    readoutB: 0.2,
                    alpha: 0.8,
                },
            ];

            const genome1 = new Genome(geneLstm, data1);
            const genome2 = new Genome(geneLstm, data2);

            const distance = genome1.distance(genome2);

            expect(distance).toBeGreaterThan(0);
        });

        it('should be symmetric', () => {
            const genome1 = new Genome(geneLstm);
            const genome2 = new Genome(geneLstm);

            const distance1 = genome1.distance(genome2);
            const distance2 = genome2.distance(genome1);

            expect(distance1).toBe(distance2);
        });
    });

    describe('mutate', () => {
        it('should mutate genome without crashing', () => {
            const genome = new Genome(geneLstm);

            genome.mutate();

            expect(genome.lstmArray.length).toBeGreaterThanOrEqual(1);
        });

        it('should potentially add LSTM blocks', () => {
            const glstmHighAdd = new GeneLSTM(10, {
                PROBABILITY_MUTATE_LSTM_BLOCK: 1.0,
                MUTATION_RATE: 1.0,
                PROBABILITY_REMOVE_BLOCK: 0.0,
            });

            const genome = new Genome(glstmHighAdd);
            const initialDepth = genome.lstmArray.length;

            // Mutate multiple times
            for (let i = 0; i < 10; i++) {
                genome.mutate();
            }

            expect(genome.lstmArray.length).toBeGreaterThan(initialDepth);
        });

        it('should potentially remove LSTM blocks', () => {
            const glstmHighRemove = new GeneLSTM(10, {
                PROBABILITY_MUTATE_LSTM_BLOCK: 1.0,
                MUTATION_RATE: 1.0,
                PROBABILITY_REMOVE_BLOCK: 1.0,
            });

            // Create genome with multiple blocks
            const data: GeneOptions = Array(5)
                .fill(null)
                .map(() => ({
                    hiddenSize: 1,
                    forgetGate: [{ weight1: 0.5, weight2: 0.5, bias: 0.0 }],
                    potentialLongToRem: [{ weight1: 0.4, weight2: 0.4, bias: 0.0 }],
                    potentialLongMemory: [{ weight1: 0.6, weight2: 0.6, bias: 0.0 }],
                    shortMemoryToRemember: [{ weight1: 0.7, weight2: 0.7, bias: 0.0 }],
                    readoutW: [0.5],
                    readoutB: 0.1,
                    alpha: 0.9,
                }));

            const genome = new Genome(glstmHighRemove, data);
            const initialDepth = genome.lstmArray.length;

            // Mutate multiple times
            for (let i = 0; i < 10; i++) {
                genome.mutate();
            }

            // Should have removed some blocks (but always keep at least 1)
            expect(genome.lstmArray.length).toBeLessThanOrEqual(initialDepth);
            expect(genome.lstmArray.length).toBeGreaterThanOrEqual(1);
        });
    });

    describe('calculate', () => {
        it('should calculate output for number[] input', () => {
            const genome = new Genome(geneLstm);
            const input = [0.5, 0.3, 0.2];

            const output = genome.calculate(input);

            expect(Array.isArray(output)).toBe(true);
            expect(output.length).toBeGreaterThan(0);
        });

        it('should calculate output for number[][] input', () => {
            const genome = new Genome(geneLstm);
            const input = [
                [0.5, 0.3],
                [0.2, 0.4],
            ];

            const output = genome.calculate(input);

            expect(Array.isArray(output)).toBe(true);
            expect(output.length).toBeGreaterThan(0);
        });

        it('should process through all LSTM blocks', () => {
            const data: GeneOptions = Array(3)
                .fill(null)
                .map(() => ({
                    hiddenSize: 1,
                    forgetGate: [{ weight1: 0.5, weight2: 0.5, bias: 0.0 }],
                    potentialLongToRem: [{ weight1: 0.4, weight2: 0.4, bias: 0.0 }],
                    potentialLongMemory: [{ weight1: 0.6, weight2: 0.6, bias: 0.0 }],
                    shortMemoryToRemember: [{ weight1: 0.7, weight2: 0.7, bias: 0.0 }],
                    readoutW: [0.5],
                    readoutB: 0.1,
                    alpha: 0.9,
                }));

            const genome = new Genome(geneLstm, data);
            const input = [0.5, 0.3, 0.2];

            const output = genome.calculate(input);

            expect(output).toBeDefined();
            expect(Array.isArray(output)).toBe(true);
        });
    });

    describe('crossGateUnit', () => {
        it('should crossover two gate units', () => {
            const unitA = {
                weight1: 0.5,
                weight2: 0.6,
                bias: 0.7,
                weightIn: [0.1, 0.2, 0.3],
            };

            const unitB = {
                weight1: 0.8,
                weight2: 0.9,
                bias: 1.0,
                weightIn: [0.4, 0.5, 0.6],
            };

            const result = Genome.crossGateUnit(unitA, unitB);

            expect(result.weight1).toBeDefined();
            expect(result.weight2).toBeDefined();
            expect(result.bias).toBeDefined();
            expect(result.weightIn).toBeDefined();
            expect(result.weightIn).toHaveLength(3);
        });

        it('should handle units without weightIn', () => {
            const unitA = { weight1: 0.5, weight2: 0.6, bias: 0.7 };
            const unitB = { weight1: 0.8, weight2: 0.9, bias: 1.0 };

            const result = Genome.crossGateUnit(unitA, unitB);

            expect(result.weight1).toBeDefined();
            expect(result.weight2).toBeDefined();
            expect(result.bias).toBeDefined();
        });

        it('should handle one unit with weightIn and one without', () => {
            const unitA = {
                weight1: 0.5,
                weight2: 0.6,
                bias: 0.7,
                weightIn: [0.1, 0.2],
            };
            const unitB = { weight1: 0.8, weight2: 0.9, bias: 1.0 };

            const result = Genome.crossGateUnit(unitA, unitB);

            expect(result.weight1).toBeDefined();
            expect(result.weightIn).toBeDefined();
        });
    });

    describe('crossOver', () => {
        it('should create offspring from two genomes', () => {
            const genome1 = new Genome(geneLstm);
            const genome2 = new Genome(geneLstm);

            const offspring = Genome.crossOver(genome1, genome2);

            expect(offspring).toBeInstanceOf(Genome);
            expect(offspring.lstmArray.length).toBeGreaterThanOrEqual(1);
        });

        it('should handle genomes with different depths', () => {
            const data1: GeneOptions = Array(2)
                .fill(null)
                .map(() => ({
                    hiddenSize: 1,
                    forgetGate: [{ weight1: 0.5, weight2: 0.5, bias: 0.0 }],
                    potentialLongToRem: [{ weight1: 0.4, weight2: 0.4, bias: 0.0 }],
                    potentialLongMemory: [{ weight1: 0.6, weight2: 0.6, bias: 0.0 }],
                    shortMemoryToRemember: [{ weight1: 0.7, weight2: 0.7, bias: 0.0 }],
                    readoutW: [0.5],
                    readoutB: 0.1,
                    alpha: 0.9,
                }));

            const data2: GeneOptions = Array(4)
                .fill(null)
                .map(() => ({
                    hiddenSize: 1,
                    forgetGate: [{ weight1: 0.3, weight2: 0.3, bias: 0.1 }],
                    potentialLongToRem: [{ weight1: 0.2, weight2: 0.2, bias: 0.1 }],
                    potentialLongMemory: [{ weight1: 0.4, weight2: 0.4, bias: 0.1 }],
                    shortMemoryToRemember: [{ weight1: 0.5, weight2: 0.5, bias: 0.1 }],
                    readoutW: [0.3],
                    readoutB: 0.2,
                    alpha: 0.8,
                }));

            const genome1 = new Genome(geneLstm, data1);
            const genome2 = new Genome(geneLstm, data2);

            const offspring = Genome.crossOver(genome1, genome2);

            expect(offspring.lstmArray.length).toBeGreaterThanOrEqual(2);
            expect(offspring.lstmArray.length).toBeLessThanOrEqual(4);
        });

        it('should handle genomes with different hidden sizes', () => {
            const data1: GeneOptions = [
                {
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
                    readoutW: [0.5, 0.6],
                    readoutB: 0.1,
                    alpha: 0.9,
                },
            ];

            const data2: GeneOptions = [
                {
                    hiddenSize: 3,
                    forgetGate: [
                        { weight1: 0.3, weight2: 0.3, bias: 0.1 },
                        { weight1: 0.4, weight2: 0.4, bias: 0.2 },
                        { weight1: 0.5, weight2: 0.5, bias: 0.3 },
                    ],
                    potentialLongToRem: [
                        { weight1: 0.2, weight2: 0.2, bias: 0.1 },
                        { weight1: 0.3, weight2: 0.3, bias: 0.2 },
                        { weight1: 0.4, weight2: 0.4, bias: 0.3 },
                    ],
                    potentialLongMemory: [
                        { weight1: 0.4, weight2: 0.4, bias: 0.1 },
                        { weight1: 0.5, weight2: 0.5, bias: 0.2 },
                        { weight1: 0.6, weight2: 0.6, bias: 0.3 },
                    ],
                    shortMemoryToRemember: [
                        { weight1: 0.5, weight2: 0.5, bias: 0.1 },
                        { weight1: 0.6, weight2: 0.6, bias: 0.2 },
                        { weight1: 0.7, weight2: 0.7, bias: 0.3 },
                    ],
                    readoutW: [0.3, 0.4, 0.5],
                    readoutB: 0.2,
                    alpha: 0.8,
                },
            ];

            const genome1 = new Genome(geneLstm, data1);
            const genome2 = new Genome(geneLstm, data2);

            const offspring = Genome.crossOver(genome1, genome2);

            expect(offspring).toBeInstanceOf(Genome);
            expect(offspring.lstmArray[0]).toBeDefined();
        });
    });
});
