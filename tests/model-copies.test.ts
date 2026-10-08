import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import type { LSTM } from '../src/lstm.js';
import type { GeneOptions } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions } from './helpers/randomGenome.js';

const F = 4;

function makeData(seed: number, outDim = 2): GeneOptions {
    return makeGeneOptions(mulberry32(seed), 2, [3, 2], F, outDim, 'all');
}

function liveArrays(lstm: LSTM): object[] {
    const refs: object[] = [lstm.readoutW, lstm.readoutB, ...lstm.readoutW];
    const gates = [
        lstm['_forgetGate'],
        lstm['_potentialLongToRem'],
        lstm['_potentialLongMemory'],
        lstm['_shortMemoryToRemember'],
    ];
    for (const gate of gates) {
        refs.push(gate);
        for (const block of gate) {
            refs.push(block);
            if (block.weightIn) refs.push(block.weightIn);
        }
    }

    return refs;
}

function snapshotArrays(data: GeneOptions): object[] {
    const refs: object[] = [];
    for (const layer of data) {
        if (Array.isArray(layer.readoutW)) refs.push(layer.readoutW, ...layer.readoutW.filter(Array.isArray));
        if (Array.isArray(layer.readoutB)) refs.push(layer.readoutB);
        for (const gate of [
            layer.forgetGate,
            layer.potentialLongToRem,
            layer.potentialLongMemory,
            layer.shortMemoryToRemember,
        ]) {
            refs.push(gate);
            for (const unit of gate) {
                refs.push(unit);
                if (unit.weightIn) refs.push(unit.weightIn);
            }
        }
    }

    return refs;
}

function expectNoSharedArray(lstms: LSTM[], data: GeneOptions) {
    const live = new Set<object>(lstms.flatMap(liveArrays));
    const shared = snapshotArrays(data).filter(r => live.has(r));
    expect(shared.length).toBe(0);
}

describe('model() returns copies', () => {
    it('LSTM.model() and copyOptions() share no array with the LSTM', () => {
        const glstm = new GeneLSTM(1, { INPUT_FEATURES: F, loadData: makeData(1), loadPercent: 1 });
        const lstms = glstm.clients[0].genome.lstmArray;

        expectNoSharedArray(
            lstms,
            lstms.map(l => l.model()),
        );
        expectNoSharedArray(
            lstms,
            lstms.map(l => l.copyOptions()),
        );
    });

    it('Client.model() and GeneLSTM.model() share no array with the population', () => {
        const glstm = new GeneLSTM(3, { INPUT_FEATURES: F, loadData: makeData(2), loadPercent: 1 });
        const lstms = glstm.clients.flatMap(c => c.genome.lstmArray);

        for (const c of glstm.clients) expectNoSharedArray(lstms, c.model());
        expectNoSharedArray(lstms, glstm.model());
    });

    it('a Client.model() snapshot does not change when the client mutates later', () => {
        withSeededRandom(5, () => {
            const glstm = new GeneLSTM(2, {
                INPUT_FEATURES: F,
                loadData: makeData(3),
                loadPercent: 1,
                PROBABILITY_MUTATE_READOUT_W: 1,
                PROBABILITY_MUTATE_READOUT_B: 1,
            });
            const client = glstm.clients[0];
            const snapshot = client.model();
            const before = JSON.stringify(snapshot);

            for (let k = 0; k < 20; k++) client.mutate(true);

            expect(JSON.stringify(snapshot)).toBe(before);

            const liveReadout = JSON.stringify(client.genome.lstmArray.map(l => [l.readoutW, l.readoutB]));
            const snapReadout = JSON.stringify(snapshot.map(l => [l.readoutW, l.readoutB]));
            expect(liveReadout).not.toBe(snapReadout);
        });
    });
});
