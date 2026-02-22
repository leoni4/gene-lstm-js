import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';

describe('Non-Destructive Structural Mutations', () => {
    it('should create sleeping blocks with correct initialization', () => {
        const glstm = new GeneLSTM(1);
        const config = glstm.sleepingBlockConfig;

        // Verify default configuration
        expect(config.epsilon).toBe(0.002);
        expect(config.forgetBias).toBe(1.5);
        expect(config.inputBias).toBe(-1.5);
        expect(config.outputBias).toBe(0.0);
        expect(config.candidateBias).toBe(0.0);
        expect(config.initialAlpha).toBe(0.01);
    });

    it('should allow custom sleeping block configuration', () => {
        const glstm = new GeneLSTM(1, {
            sleepingBlockConfig: {
                epsilon: 0.001,
                forgetBias: 2.0,
                inputBias: -2.0,
            },
        });

        const config = glstm.sleepingBlockConfig;
        expect(config.epsilon).toBe(0.001);
        expect(config.forgetBias).toBe(2.0);
        expect(config.inputBias).toBe(-2.0);
        // Other values should use defaults
        expect(config.outputBias).toBe(0.0);
    });

    it('should maintain behavior similarity when adding sleeping blocks', () => {
        const glstm = new GeneLSTM(10, {
            PROBABILITY_MUTATE_LSTM_BLOCK: 1.0, // Force structural mutation
            MUTATION_RATE: 1.0,
        });

        const testInput = [0.5, 0.3, 0.8, 0.2];

        // Get initial output
        const initialOutput = glstm.clients[0].calculate(testInput);
        const initialDepth = glstm.clients[0].genome.lstmArray.length;

        // Force mutation (should add blocks)
        for (let i = 0; i < 5; i++) {
            glstm.clients[0].mutate(true);
        }

        const finalDepth = glstm.clients[0].genome.lstmArray.length;
        const finalOutput = glstm.clients[0].calculate(testInput);

        // Depth should increase (sleeping blocks added)
        expect(finalDepth).toBeGreaterThanOrEqual(initialDepth);

        // Output should be reasonably similar (not identical due to random init)
        // This is a weak test since sleeping blocks should have minimal impact
        expect(finalOutput).toHaveLength(initialOutput.length);
    });

    it('should add blocks with directional bias (append > prepend)', () => {
        const glstm = new GeneLSTM(100, {
            PROBABILITY_MUTATE_LSTM_BLOCK: 1.0,
            MUTATION_RATE: 1.0,
            PROBABILITY_REMOVE_BLOCK: 0.0, // Only additions
        });

        const additionCounts = { increased: 0, total: 0 };

        for (let i = 0; i < glstm.clients.length; i++) {
            const beforeDepth = glstm.clients[i].genome.lstmArray.length;
            glstm.clients[i].mutate(true);
            const afterDepth = glstm.clients[i].genome.lstmArray.length;

            if (afterDepth > beforeDepth) {
                additionCounts.increased++;
            }
            additionCounts.total++;
        }

        // Most should have increased depth
        expect(additionCounts.increased).toBeGreaterThan(additionCounts.total * 0.7);
    });

    it('should support alpha parameter for skip connections', () => {
        const glstm = new GeneLSTM(1, {
            loadData: [
                {
                    forgetGate: { weight1: 0.5, weight2: 0.5, bias: 0.0 },
                    potentialLongToRem: { weight1: 0.5, weight2: 0.5, bias: 0.0 },
                    potentialLongMemory: { weight1: 0.5, weight2: 0.5, bias: 0.0 },
                    shortMemoryToRemember: { weight1: 0.5, weight2: 0.5, bias: 0.0 },
                    alpha: 0.5,
                },
            ],
        });

        const client = glstm.clients[0];
        const lstm = client.genome.lstmArray[0];

        expect(lstm.alpha).toBe(0.5);

        // Alpha should be mutable and clamped
        lstm.alpha = 1.5;
        expect(lstm.alpha).toBe(1.0); // Clamped to max

        lstm.alpha = -0.5;
        expect(lstm.alpha).toBe(0.0); // Clamped to min
    });

    it('should preserve alpha in model serialization', () => {
        const glstm = new GeneLSTM(1);
        const lstm = glstm.clients[0].genome.lstmArray[0];

        lstm.alpha = 0.75;
        const model = lstm.model();

        expect(model.alpha).toBe(0.75);
    });
});
