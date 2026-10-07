import type { GateUnitOptions, GeneOptions, LstmOptions } from '../../src/types/index.js';

export type WeightInMode = 'all' | 'none' | 'mixed';

export function makeLayerOptions(
    rng: () => number,
    H: number,
    nIn: number,
    outDim: number,
    weightInMode: WeightInMode = 'all',
): LstmOptions {
    const u = () => rng() * 2 - 1;
    const unit = (k: number): GateUnitOptions => {
        const g: GateUnitOptions = { weight1: u(), weight2: u(), bias: u() };
        const set = weightInMode === 'all' || (weightInMode === 'mixed' && (k + Math.floor(rng() * 2)) % 2 === 0);
        if (set) g.weightIn = Array.from({ length: nIn }, u);

        return g;
    };
    const gate = () => Array.from({ length: H }, (_, k) => unit(k));

    return {
        hiddenSize: H,
        forgetGate: gate(),
        potentialLongToRem: gate(),
        potentialLongMemory: gate(),
        shortMemoryToRemember: gate(),
        readoutW: Array.from({ length: outDim }, () => Array.from({ length: H }, () => u() * 0.5)),
        readoutB: Array.from({ length: outDim }, () => u() * 0.2),
        alpha: rng() < 0.3 ? rng() : 1,
    };
}

export function makeGeneOptions(
    rng: () => number,
    layers: number,
    H: number | number[],
    nIn: number,
    outDim: number,
    weightInMode: WeightInMode = 'all',
): GeneOptions {
    const out: GeneOptions = [];
    for (let l = 0; l < layers; l++) {
        const h = Array.isArray(H) ? H[l] : H;
        out.push(makeLayerOptions(rng, h, nIn, outDim, weightInMode));
    }

    return out;
}

export function makeInput(rng: () => number, T: number, F: number): number[][] {
    return Array.from({ length: T }, () => Array.from({ length: F }, () => rng() * 2 - 1));
}
