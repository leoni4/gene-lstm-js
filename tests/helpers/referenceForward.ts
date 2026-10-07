import type { Genome } from '../../src/genome.js';
import type { LSTM, ShortMemoryBlock } from '../../src/lstm.js';

export type ReferenceInput = readonly (number | number[])[];

type Activation = (x: number) => number;

function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

function referenceGate(block: ShortMemoryBlock, input: number | number[], shortMemory: number, act: Activation) {
    const rec = block.weight1 * shortMemory;

    let inTerm = 0;

    if (Array.isArray(input)) {
        if (!block.weightIn || block.weightIn.length !== input.length) {
            block.weightIn = new Array(input.length).fill(0).map(() => Math.random() * 2 - 1);
        }
        for (let i = 0; i < input.length; i++) {
            inTerm += block.weightIn[i] * input[i];
        }
    } else {
        inTerm = block.weight2 * input;
    }

    const sum = rec + inTerm + block.bias;

    return act(sum);
}

function referenceReadout(lstm: LSTM): number[] {
    const outputDim = lstm['_outputDim'];
    const activation = lstm['_outputActivation'];
    const short = lstm.shortMemory;
    const output: number[] = new Array(outputDim);

    for (let j = 0; j < outputDim; j++) {
        let s = lstm.readoutB[j];
        for (let k = 0; k < short.length; k++) {
            s += lstm.readoutW[j][k] * short[k];
        }
        if (activation === 'sigmoid') {
            output[j] = sigmoid(s);
        } else if (activation === 'tanh') {
            output[j] = Math.tanh(s);
        } else {
            output[j] = s;
        }
    }

    return output;
}

export function referenceLstmCalculate(lstm: LSTM, input: ReferenceInput, fullSeq: boolean): number[] {
    lstm['_ensureConsistentSizes']();

    const long = lstm.longMemory;
    const short = lstm.shortMemory;
    long.fill(0);
    short.fill(0);

    const F = lstm['_forgetGate'];
    const I = lstm['_potentialLongToRem'];
    const G = lstm['_potentialLongMemory'];
    const O = lstm['_shortMemoryToRemember'];
    const H = short.length;

    const isMatrix = Array.isArray(input[0]);
    const mem: number[][] = [];

    for (const x of input) {
        for (let k = 0; k < H; k++) {
            const hPrev = short[k];
            const f = referenceGate(F[k], x, hPrev, sigmoid);
            long[k] *= f;
            const i = referenceGate(I[k], x, hPrev, sigmoid);
            const g = referenceGate(G[k], x, hPrev, Math.tanh);
            long[k] += i * g;
            const o = referenceGate(O[k], x, hPrev, sigmoid);
            short[k] = Math.tanh(long[k]) * o;
        }
        if (fullSeq) mem.push(referenceReadout(lstm));
    }

    if (fullSeq) return mem.flat();

    const y = referenceReadout(lstm);
    let yPrev: number;
    if (isMatrix) {
        const last = input.length ? input[input.length - 1] : [0];
        yPrev = Array.isArray(last) && typeof last[0] === 'number' ? last[0] : 0;
    } else {
        const lastIn = input.length ? input[input.length - 1] : 0;
        if (typeof lastIn !== 'number') {
            throw new Error('referenceLstmCalculate: 1-D input with an array row is not supported');
        }
        yPrev = lastIn;
    }

    const a = lstm['_alpha'];
    const result = [...y];
    result[0] = (1 - a) * yPrev + a * y[0];

    return result;
}

export function referenceGenomeCalculate(genome: Genome, input: ReferenceInput): number[] {
    const layers = genome.lstmArray;
    let passed: ReferenceInput = input;
    let out: number[] = [];

    for (let i = 0; i < layers.length; i++) {
        const fullSeq = layers.length > 1 && layers.length > i + 1;
        out = referenceLstmCalculate(layers[i], passed, fullSeq);
        passed = out;
    }

    return out;
}
