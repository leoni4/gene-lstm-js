import { describe, it, expect, vi, afterEach } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import type { GeneLSTMOptions } from '../src/types/index.js';
import { mulberry32, withSeededRandom } from './helpers/seededRandom.js';
import { makeGeneOptions, makeInput } from './helpers/randomGenome.js';
import { referenceGenomeCalculate } from './helpers/referenceForward.js';

const F = 4;

function build(inputCheck?: GeneLSTMOptions['inputCheck']): GeneLSTM {
    const loadData = makeGeneOptions(mulberry32(7), 2, [3, 2], F, 1);

    return withSeededRandom(1, () => new GeneLSTM(2, { INPUT_FEATURES: F, loadData, loadPercent: 1, inputCheck }))
        .result;
}

function expectSameAsReference(glstm: GeneLSTM, input: number[][], seed: number) {
    const reference = build();
    const lib = withSeededRandom(seed, () => glstm.clients[0].calculate(input));
    const ref = withSeededRandom(seed, () => referenceGenomeCalculate(reference.clients[0].genome, input));

    expect(lib.result.length).toBe(ref.result.length);
    lib.result.forEach((v, i) => expect(Object.is(v, ref.result[i])).toBe(true));
    expect(lib.log).toEqual(ref.log);

    return lib.log;
}

afterEach(() => {
    vi.restoreAllMocks();
});

describe('inputCheck option', () => {
    it('defaults to off: no warning, same outputs and Math.random log as the reference', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const input = makeInput(mulberry32(3), 5, F + 2);

        for (const glstm of [build(), build('off')]) {
            expect(glstm.inputCheck).toBe('off');
            const log = expectSameAsReference(glstm, input, 11);
            expect(log.length).toBeGreaterThan(0);
        }
        expect(warn).not.toHaveBeenCalled();
    });

    it("'warn' writes one warning per GeneLSTM instance and does not change results", () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const input = makeInput(mulberry32(3), 5, F + 2);

        const glstm = build('warn');
        expectSameAsReference(glstm, input, 11);
        for (let n = 0; n < 10; n++) {
            for (const client of glstm.clients) client.calculate(input);
        }
        expect(warn).toHaveBeenCalledTimes(1);
        expect(String(warn.mock.calls[0][0])).toContain(`input row has ${F + 2} features, but INPUT_FEATURES is ${F}`);

        build('warn').clients[0].calculate(input);
        expect(warn).toHaveBeenCalledTimes(2);
    });

    it("'throw' throws before the genome changes and without Math.random calls", () => {
        const glstm = build('throw');
        const before = JSON.stringify(glstm.clients[0].model());
        const rows = makeInput(mulberry32(3), 3, F);
        const ragged = [...rows, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]];

        for (const input of [makeInput(mulberry32(3), 5, F + 2), ragged]) {
            const { log } = withSeededRandom(5, () => {
                expect(() => glstm.clients[0].calculate(input)).toThrow(
                    `input row has ${F + 2} features, but INPUT_FEATURES is ${F}`,
                );
            });
            expect(log).toEqual([]);
            expect(JSON.stringify(glstm.clients[0].model())).toBe(before);
        }
    });

    it('matching width: no warning, no error, same outputs as the reference', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        const input = makeInput(mulberry32(3), 5, F);

        for (const mode of ['warn', 'throw'] as const) {
            const log = expectSameAsReference(build(mode), input, 11);
            expect(log).toEqual([]);
        }
        expect(warn).not.toHaveBeenCalled();
    });
});
