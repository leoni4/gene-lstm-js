import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { Genome } from '../src/genome.js';
import type { GeneLSTMOptions } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions } from './helpers/randomGenome.js';

const shape = (genome: Genome) => genome.lstmArray.map(lstm => lstm.shortMemory.length);

function makeParents(options: GeneLSTMOptions | undefined, h1: number[], h2: number[]) {
    const glstm = new GeneLSTM(2, options);
    const rng = mulberry32(4242);
    const g1 = new Genome(glstm, makeGeneOptions(rng, h1.length, h1, 1, 1));
    const g2 = new Genome(glstm, makeGeneOptions(rng, h2.length, h2, 1, 1));

    return { glstm, g1, g2 };
}

function crossSeeded(options: GeneLSTMOptions | undefined, h1: number[], h2: number[], seed = 7) {
    const { g1, g2 } = makeParents(options, h1, h2);
    const { result, log } = withSeededRandom(seed, () => Genome.crossOver(g1, g2));

    return { g1, g2, child: result, log };
}

describe('growth limits: defaults', () => {
    it('resolves the legacy defaults', () => {
        const glstm = new GeneLSTM(2);
        expect(glstm.crossoverStructure).toBe('max');
        expect(glstm.MAX_LAYERS).toBe(Infinity);
        expect(glstm.MAX_UNITS_PER_LAYER).toBe(Infinity);
    });

    it('floors the limits and keeps them at 1 or more', () => {
        const glstm = new GeneLSTM(2, { MAX_LAYERS: 2.7, MAX_UNITS_PER_LAYER: 0 });
        expect(glstm.MAX_LAYERS).toBe(2);
        expect(glstm.MAX_UNITS_PER_LAYER).toBe(1);
    });

    it("'max' crossover still gives the larger structure", () => {
        expect(shape(crossSeeded(undefined, [2], [5, 5, 5]).child)).toEqual([5, 5, 5]);
        expect(shape(crossSeeded(undefined, [5, 5, 5], [2]).child)).toEqual([5, 5, 5]);
    });

    it('limits above both parents give the same child and the same draws as the default', () => {
        const base = crossSeeded(undefined, [3, 6], [5, 2, 4]);
        const limited = crossSeeded(
            { crossoverStructure: 'max', MAX_LAYERS: 3, MAX_UNITS_PER_LAYER: 6 },
            [3, 6],
            [5, 2, 4],
        );
        expect(JSON.stringify(limited.child.lstmArray.map(l => l.model()))).toBe(
            JSON.stringify(base.child.lstmArray.map(l => l.model())),
        );
        expect(limited.log).toEqual(base.log);
    });
});

describe("growth limits: 'fitter' crossover", () => {
    it('gives the child the structure of g1', () => {
        const opts: GeneLSTMOptions = { crossoverStructure: 'fitter' };
        expect(shape(crossSeeded(opts, [2], [5, 5, 5]).child)).toEqual([2]);
        expect(shape(crossSeeded(opts, [5, 5, 5], [2]).child)).toEqual([5, 5, 5]);
        expect(shape(crossSeeded(opts, [3, 1], [2, 4, 6]).child)).toEqual([3, 1]);
    });

    it('copies the excess units and layers of g1 and crosses the shared units', () => {
        const { g1, g2, child } = crossSeeded({ crossoverStructure: 'fitter' }, [4, 2], [2]);
        const c0 = child.lstmArray[0].model();
        const a0 = g1.lstmArray[0].model();
        const b0 = g2.lstmArray[0].model();

        for (let k = 0; k < 2; k++) {
            const unit = c0.forgetGate[k];
            expect([a0.forgetGate[k].weight1, b0.forgetGate[k].weight1]).toContain(unit.weight1);
        }
        expect(c0.forgetGate.slice(2)).toEqual(a0.forgetGate.slice(2));
        expect((c0.readoutW as number[][])[0].slice(2)).toEqual((a0.readoutW as number[][])[0].slice(2));
        expect(child.lstmArray[1].model()).toEqual(g1.lstmArray[1].model());
    });

    it('gives a child that runs', () => {
        const { child } = crossSeeded({ crossoverStructure: 'fitter' }, [3, 2], [5, 5, 5]);
        const out = child.calculate([0.1, -0.2, 0.3]);
        expect(out).toHaveLength(1);
        expect(Number.isFinite(out[0])).toBe(true);
    });
});

describe("growth limits: 'max' crossover at the limits", () => {
    it('caps layers and units that only grow past the limit', () => {
        const opts: GeneLSTMOptions = { MAX_LAYERS: 2, MAX_UNITS_PER_LAYER: 3 };
        expect(shape(crossSeeded(opts, [2], [5, 5, 5]).child)).toEqual([3, 3]);
        expect(shape(crossSeeded(opts, [5, 5, 5], [2]).child)).toEqual([3, 3]);
    });

    it('keeps sizes that both parents already have (no pruning)', () => {
        const opts: GeneLSTMOptions = { MAX_LAYERS: 1, MAX_UNITS_PER_LAYER: 2 };
        expect(shape(crossSeeded(opts, [10, 4], [8, 6, 6]).child)).toEqual([8, 4]);
    });

    it('a capped copied layer keeps its first units and readout columns', () => {
        const { g2, child } = crossSeeded({ MAX_UNITS_PER_LAYER: 2 }, [1], [1, 4]);
        const copied = child.lstmArray[1].model();
        const source = g2.lstmArray[1].model();
        expect(copied.hiddenSize).toBe(2);
        expect(copied.shortMemoryToRemember).toEqual(source.shortMemoryToRemember.slice(0, 2));
        expect(copied.readoutW).toEqual((source.readoutW as number[][]).map(row => row.slice(0, 2)));
        expect(Number.isFinite(child.calculate([0.5, 0.25])[0])).toBe(true);
    });
});

describe('growth limits: mutation', () => {
    const growOptions: GeneLSTMOptions = {
        PROBABILITY_MUTATE_LSTM_BLOCK: 1,
        PROBABILITY_REMOVE_BLOCK: 0,
        PROBABILITY_MUTATE_ADD_UNIT: 1,
        PROBABILITY_MUTATE_REMOVE_UNIT: 0,
    };

    it('without limits, forced add mutations grow the genome', () => {
        const glstm = new GeneLSTM(2, growOptions);
        const genome = new Genome(glstm);
        withSeededRandom(11, () => {
            for (let i = 0; i < 20; i++) genome.mutate();
        });
        expect(genome.lstmArray.length).toBeGreaterThan(3);
        expect(Math.max(...shape(genome))).toBeGreaterThan(4);
    });

    it('stops adding blocks and units at the limits', () => {
        const glstm = new GeneLSTM(2, { ...growOptions, MAX_LAYERS: 3, MAX_UNITS_PER_LAYER: 4 });
        const genome = new Genome(glstm);
        withSeededRandom(11, () => {
            for (let i = 0; i < 40; i++) genome.mutate();
        });
        expect(genome.lstmArray.length).toBe(3);
        expect(shape(genome)).toEqual([4, 4, 4]);
    });

    it('at a limit, only the decision draws stay', () => {
        const glstm = new GeneLSTM(2, {
            ...growOptions,
            PROBABILITY_MUTATE_READOUT_W: 0,
            PROBABILITY_MUTATE_READOUT_B: 0,
            PROBABILITY_MUTATE_WEIGHT_RANDOM: 0,
            PROBABILITY_MUTATE_BIAS_RANDOM: 0,
            PROBABILITY_MUTATE_WEIGHT_SHIFT: 0,
            PROBABILITY_MUTATE_BIAS_SHIFT: 0,
            PROBABILITY_MUTATE_ALPHA_SHIFT: 0,
            MAX_LAYERS: 1,
            MAX_UNITS_PER_LAYER: 1,
        });
        const genome = new Genome(glstm, makeGeneOptions(mulberry32(1), 1, 1, 1, 1));

        const { log } = withSeededRandom(3, () => genome.mutate());
        expect(log).toHaveLength(2 + 7 + 3);
        expect(shape(genome)).toEqual([1]);
    });

    it('does not prune a loaded genome above the limit', () => {
        const loadData = makeGeneOptions(mulberry32(5), 3, [6, 6, 6], 1, 1);
        const glstm = new GeneLSTM(3, { ...growOptions, loadData, MAX_LAYERS: 2, MAX_UNITS_PER_LAYER: 2 });
        const genome = glstm.clients[0].genome;
        withSeededRandom(9, () => {
            for (let i = 0; i < 10; i++) genome.mutate();
        });
        expect(shape(genome)).toEqual([6, 6, 6]);
    });
});
