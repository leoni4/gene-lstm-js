import { describe, it, expect, vi } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput } from './helpers/randomGenome.js';

const F = 3;

function loadedPopulation(size: number): GeneLSTM {
    const loadData = makeGeneOptions(mulberry32(7), 2, [3, 5], F, 1);

    return new GeneLSTM(size, { loadData, loadPercent: 1 });
}

function capturedLines(spy: { mock: { calls: unknown[][] } }): string[] {
    return spy.mock.calls.map(args => args.map(String).join(' '));
}

describe('unit count in logs', () => {
    it('fit() log prints the sum of hidden units over the layers', () => {
        const rng = mulberry32(11);
        const xTrain = [makeInput(rng, 4, F), makeInput(rng, 4, F)];
        const yTrain = [0.2, 0.8];

        const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
        try {
            withSeededRandom(5, () => loadedPopulation(4).fit(xTrain, yTrain, { epochs: 1, verbose: 1 }));
            const epochLine = capturedLines(spy).find(line => line.startsWith('Epoch 0'));

            expect(epochLine).toContain('blocks: 2 - units: 8');
        } finally {
            spy.mockRestore();
        }
    });

    it('printSpecies() prints the sum of hidden units over all clients', () => {
        const glstm = loadedPopulation(4);

        const spy = vi.spyOn(console, 'log').mockImplementation(() => {});
        try {
            glstm.printSpecies();
            const header = capturedLines(spy).find(line => line.startsWith('### Species:'));

            expect(header).toContain('units=32 (avg/block 4.00)');
        } finally {
            spy.mockRestore();
        }
    });
});
