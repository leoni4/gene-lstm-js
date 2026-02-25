import { GeneLSTM } from './gLstm.js';
import type { LstmOptions } from './types/index.js';

type ActivationName = 'sigmoid' | 'tanh';
type ActivationFunction = (x: number) => number;
type WeightTarget = { kind: 'scalar'; key: 'weight1' | 'weight2' } | { kind: 'vector'; index: number };

function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

const flattenBlock = (b: ShortMemoryBlock): number[] => {
    const base = [b.weight1, b.weight2, b.bias];

    if (b.weightIn && b.weightIn.length) {
        base.push(...b.weightIn);
    }

    return base;
};

const blockToOptions = (b: ShortMemoryBlock) => ({
    weight1: b.weight1,
    weight2: b.weight2,
    bias: b.bias,
    weightIn: b.weightIn ? [...b.weightIn] : undefined,
});

export class ShortMemoryBlock {
    private _activationName: ActivationName;
    private _activationFunction: ActivationFunction;
    weight1: number = 0;
    weight2: number = 0;
    weightIn?: number[];
    bias: number = 0;

    constructor(activation: ActivationName, weight1?: number, weight2?: number, bias?: number, weightIn?: number[]) {
        this._activationName = activation;
        if (this._activationName === 'sigmoid') {
            this._activationFunction = sigmoid;
        } else {
            this._activationFunction = Math.tanh;
        }
        this.weight1 = weight1 ?? Math.random() * 2 - 1;
        this.weight2 = weight2 ?? Math.random() * 2 - 1;
        this.bias = bias ?? Math.random() * 2 - 1;
        this.weightIn = weightIn;
    }

    calculate(input: number | number[], shortMemory: number): number {
        const rec = this.weight1 * shortMemory;

        let inTerm = 0;

        if (Array.isArray(input)) {
            // vector input
            if (!this.weightIn || this.weightIn.length !== input.length) {
                // init once, stable size
                this.weightIn = new Array(input.length).fill(0).map(() => Math.random() * 2 - 1);
            }
            for (let i = 0; i < input.length; i++) {
                inTerm += this.weightIn[i] * input[i];
            }
        } else {
            // scalar input (old behavior)
            inTerm = this.weight2 * input;
        }

        const sum = rec + inTerm + this.bias;

        return this._activationFunction(sum);
    }
}

export class OutputBlock {
    calculate(longMemory: number, shortMemory: number) {
        const out = Math.tanh(longMemory) * shortMemory;

        return out;
    }
}

export class LSTM {
    private _geneLstm: GeneLSTM;

    longMemory: number[];
    shortMemory: number[];

    readoutW: number[];
    readoutB: number;

    private _forgetGate: ShortMemoryBlock[];
    private _potentialLongToRem: ShortMemoryBlock[];
    private _potentialLongMemory: ShortMemoryBlock[];

    private _shortMemoryToRemember: ShortMemoryBlock[];

    // Skip connection strength for non-destructive mutations
    private _alpha: number;

    constructor(geneLstm: GeneLSTM, options?: LstmOptions) {
        this._geneLstm = geneLstm;
        this._alpha = options?.alpha ?? 1.0;

        const H = options?.hiddenSize ?? 1;

        const makeBlock = (act: ActivationName, u?: any) =>
            new ShortMemoryBlock(act, u?.weight1, u?.weight2, u?.bias, u?.weightIn);

        if (options) {
            // gates restored from arrays
            this._forgetGate = new Array(H).fill(0).map((_, i) => makeBlock('sigmoid', options.forgetGate[i]));
            this._potentialLongToRem = new Array(H)
                .fill(0)
                .map((_, i) => makeBlock('sigmoid', options.potentialLongToRem[i]));
            this._potentialLongMemory = new Array(H)
                .fill(0)
                .map((_, i) => makeBlock('tanh', options.potentialLongMemory[i]));
            this._shortMemoryToRemember = new Array(H)
                .fill(0)
                .map((_, i) => makeBlock('sigmoid', options.shortMemoryToRemember[i]));

            this.readoutW = options.readoutW?.length === H ? [...options.readoutW] : new Array(H).fill(0);
            this.readoutB = options.readoutB ?? 0;
        } else {
            // fresh random
            this._forgetGate = new Array(H).fill(0).map(() => new ShortMemoryBlock('sigmoid'));
            this._potentialLongToRem = new Array(H).fill(0).map(() => new ShortMemoryBlock('sigmoid'));
            this._potentialLongMemory = new Array(H).fill(0).map(() => new ShortMemoryBlock('tanh'));
            this._shortMemoryToRemember = new Array(H).fill(0).map(() => new ShortMemoryBlock('sigmoid'));

            const eps = 0.1;
            this.readoutW = new Array(H).fill(0).map(() => (Math.random() * 2 - 1) * eps);
            this.readoutB = 0;
        }

        this.longMemory = new Array(H).fill(0);
        this.shortMemory = new Array(H).fill(0);

        // normalize sizes if something was off
        this._ensureConsistentSizes();
    }

    get alpha(): number {
        return this._alpha;
    }

    set alpha(value: number) {
        // Clamp alpha between 0 and 1
        this._alpha = Math.max(0, Math.min(1, value));
    }

    private _hiddenSize(): number {
        return Math.max(1, this.readoutW.length || 1);
    }

    private _ensureConsistentSizes() {
        const H = this._hiddenSize();

        // init memories if needed
        if (!this.longMemory || this.longMemory.length !== H) this.longMemory = new Array(H).fill(0);
        if (!this.shortMemory || this.shortMemory.length !== H) this.shortMemory = new Array(H).fill(0);

        const ensureGate = (gate: ShortMemoryBlock[], activation: ActivationName) => {
            while (gate.length < H) gate.push(new ShortMemoryBlock(activation));
            while (gate.length > H) gate.pop();
        };

        ensureGate(this._forgetGate, 'sigmoid');
        ensureGate(this._potentialLongToRem, 'sigmoid');
        ensureGate(this._potentialLongMemory, 'tanh');
        ensureGate(this._shortMemoryToRemember, 'sigmoid');

        // readout weights must match H
        while (this.readoutW.length < H) this.readoutW.push(0);
        while (this.readoutW.length > H) this.readoutW.pop();
    }

    flattenWeights(): number[] {
        this._ensureConsistentSizes();

        const out: number[] = [];

        for (const b of this._forgetGate) out.push(...flattenBlock(b));
        for (const b of this._potentialLongToRem) out.push(...flattenBlock(b));
        for (const b of this._potentialLongMemory) out.push(...flattenBlock(b));
        for (const b of this._shortMemoryToRemember) out.push(...flattenBlock(b));

        out.push(...this.readoutW, this.readoutB, this._alpha);
        return out;
    }

    private _pickWeightTarget(block: ShortMemoryBlock): WeightTarget {
        const canVector = !!block.weightIn && block.weightIn.length > 0;

        if (canVector && Math.random() < 0.3) {
            const idx = Math.floor(Math.random() * block.weightIn!.length);

            return { kind: 'vector', index: idx };
        }

        const key = `weight${Math.floor(Math.random() * 2 + 1)}` as 'weight1' | 'weight2';

        return { kind: 'scalar', key };
    }

    private _ensureWeightIn(block: ShortMemoryBlock) {
        if (!block.weightIn || block.weightIn.length === 0) {
            const n = this._geneLstm.INPUT_FEATURES ?? 1;
            block.weightIn = new Array(n).fill(0).map(() => Math.random() * 2 - 1);
        }
    }

    calculate(input: number[] | number[][], fullSeq = false): number[] {
        this._ensureConsistentSizes();

        // reset state each forward (как у тебя было)
        this.longMemory.fill(0);
        this.shortMemory.fill(0);

        const fullSeqMemory: number[] = [];

        if (Array.isArray(input[0])) {
            const seq = input as number[][];
            for (const x_t of seq) {
                this._predictUnit(x_t);
                if (fullSeq) fullSeqMemory.push(this._readout()); // можно fullSeq по readout
            }
            return fullSeq ? fullSeqMemory : [this._readout()];
        }

        const seq = input as number[];
        for (const num of seq) {
            this._predictUnit(num);
            if (fullSeq) fullSeqMemory.push(this._readout());
        }
        return fullSeq ? fullSeqMemory : [this._readout()];
    }

    private _readout(): number {
        // y = sigmoid(dot(W, h) + b)
        let s = this.readoutB;
        for (let k = 0; k < this.shortMemory.length; k++) s += this.readoutW[k] * this.shortMemory[k];
        return sigmoid(s);
    }

    private _predictUnit(input: number | number[]) {
        const H = this.shortMemory.length;

        for (let k = 0; k < H; k++) {
            const hPrev = this.shortMemory[k];

            const f = this._forgetGate[k].calculate(input, hPrev);
            this.longMemory[k] *= f;

            const i = this._potentialLongToRem[k].calculate(input, hPrev);
            const g = this._potentialLongMemory[k].calculate(input, hPrev);

            this.longMemory[k] += i * g;

            const o = this._shortMemoryToRemember[k].calculate(input, hPrev);

            // output per-unit
            this.shortMemory[k] = Math.tanh(this.longMemory[k]) * o;
        }
    }

    model(): LstmOptions {
        this._ensureConsistentSizes();
        const H = this.shortMemory.length;

        return {
            hiddenSize: H,

            forgetGate: this._forgetGate.map(blockToOptions),
            potentialLongToRem: this._potentialLongToRem.map(blockToOptions),
            potentialLongMemory: this._potentialLongMemory.map(blockToOptions),
            shortMemoryToRemember: this._shortMemoryToRemember.map(blockToOptions),

            readoutW: [...this.readoutW],
            readoutB: this.readoutB,

            alpha: this._alpha,
        };
    }

    private _getBlockToMutate(): ShortMemoryBlock {
        this._ensureConsistentSizes();
        const H = this.shortMemory.length;
        const unit = Math.floor(Math.random() * H);

        const gateNum = Math.floor(Math.random() * 4);
        switch (gateNum) {
            case 0:
                return this._forgetGate[unit];
            case 1:
                return this._potentialLongToRem[unit];
            case 2:
                return this._potentialLongMemory[unit];
            default:
                return this._shortMemoryToRemember[unit];
        }
    }

    private _mutateWeightRandom(pressureScale: number) {
        const block = this._getBlockToMutate();
        this._ensureWeightIn(block);

        const target = this._pickWeightTarget(block);

        if (target.kind === 'scalar') {
            const key = target.key;
            const base = Math.abs(block[key] ?? 1); // avoid 0 base
            let newWeight = (Math.random() * 2 - 1) * base * this._geneLstm.WEIGHT_RANDOM_STRENGTH * pressureScale;

            block[key] = Math.max(-10, Math.min(10, newWeight));

            return;
        }

        const i = target.index;

        const range = this._geneLstm.WEIGHT_RANDOM_STRENGTH * pressureScale;
        const newWeight = (Math.random() * 2 - 1) * range;
        block.weightIn![i] = newWeight;
    }

    private _mutateBiasRandom(pressureScale: number) {
        const block = this._getBlockToMutate();

        const range = this._geneLstm.BIAS_RANDOM_STRENGTH * pressureScale;
        const newBias = (Math.random() * 2 - 1) * range;

        block.bias = Math.max(-10, Math.min(10, newBias));
    }

    private _mutateWeightShift(pressureScale: number) {
        const block = this._getBlockToMutate();

        this._ensureWeightIn(block);

        const target = this._pickWeightTarget(block);

        if (target.kind === 'scalar') {
            const key = target.key;
            const current = block[key] ?? this._geneLstm.WEIGHT_SHIFT_STRENGTH;

            let newWeight = current + (Math.random() * 2 - 1) * this._geneLstm.WEIGHT_SHIFT_STRENGTH * pressureScale;
            block[key] = Math.max(-10, Math.min(10, newWeight));

            return;
        }

        // vector
        const i = target.index;
        const current = block.weightIn![i] ?? 0;

        let newWeight = current + (Math.random() * 2 - 1) * this._geneLstm.WEIGHT_SHIFT_STRENGTH * pressureScale;
        block.weightIn![i] = Math.max(-10, Math.min(10, newWeight));
    }

    private _mutateBiasShift(pressureScale: number) {
        const block = this._getBlockToMutate();

        let newBias = block.bias + (Math.random() * 2 - 1) * this._geneLstm.BIAS_SHIFT_STRENGTH * pressureScale;

        // Clamp to reasonable range
        block.bias = Math.max(-10, Math.min(10, newBias));
    }

    private _mutateAlpha(pressureScale: number) {
        const delta = (Math.random() * 2 - 1) * this._geneLstm.ALPHA_SHIFT_STRENGTH * pressureScale;
        this.alpha = this._alpha + delta;
    }

    private _mutateAddUnit() {
        this._ensureConsistentSizes();

        // add new unit to each gate
        this._forgetGate.push(new ShortMemoryBlock('sigmoid'));
        this._potentialLongToRem.push(new ShortMemoryBlock('sigmoid'));
        this._potentialLongMemory.push(new ShortMemoryBlock('tanh'));
        this._shortMemoryToRemember.push(new ShortMemoryBlock('sigmoid'));

        // extend memories
        this.longMemory.push(0);
        this.shortMemory.push(0);

        // NEW unit initially does nothing -> readout weight 0
        this.readoutW.push(0);

        // optional: keep readoutB unchanged

        // (опционально) можно “усыпить” веса новому юниту маленькими значениями
    }

    private _mutateRemoveUnit() {
        this._ensureConsistentSizes();
        const H = this.shortMemory.length;
        if (H <= 1) return;

        // pick unit to remove:
        // лучше удалять с минимальным |readoutW| чтобы меньше ломать фенотип
        let idx = 0;
        let best = Math.abs(this.readoutW[0]);
        for (let k = 1; k < H; k++) {
            const v = Math.abs(this.readoutW[k]);
            if (v < best) {
                best = v;
                idx = k;
            }
        }

        const removeAt = <T>(arr: T[]) => arr.splice(idx, 1);

        removeAt(this._forgetGate);
        removeAt(this._potentialLongToRem);
        removeAt(this._potentialLongMemory);
        removeAt(this._shortMemoryToRemember);

        removeAt(this.longMemory);
        removeAt(this.shortMemory);
        removeAt(this.readoutW);
    }

    private _mutateReadoutWeightShift(pressureScale: number) {
        this._ensureConsistentSizes();
        const i = Math.floor(Math.random() * this.readoutW.length);
        const delta = (Math.random() * 2 - 1) * this._geneLstm.WEIGHT_SHIFT_STRENGTH * pressureScale;
        this.readoutW[i] = Math.max(-10, Math.min(10, this.readoutW[i] + delta));
    }

    private _mutateReadoutBiasShift(pressureScale: number) {
        const delta = (Math.random() * 2 - 1) * this._geneLstm.BIAS_SHIFT_STRENGTH * pressureScale;
        this.readoutB = Math.max(-10, Math.min(10, this.readoutB + delta));
    }

    mutate() {
        // Apply weights mutation pressure to all weight/bias mutations
        const pressure = this._geneLstm.getMutationPressure();
        const weightsPressure = pressure.weights;
        const topologyPressure = pressure.topology;

        if (Math.random() < this._geneLstm.PROBABILITY_MUTATE_ADD_UNIT * topologyPressure) {
            this._mutateAddUnit();
        }
        if (Math.random() < this._geneLstm.PROBABILITY_MUTATE_REMOVE_UNIT * topologyPressure) {
            this._mutateRemoveUnit();
        }

        if (
            Math.random() <
            this._geneLstm.PROBABILITY_MUTATE_READOUT_W * this._geneLstm.MUTATION_RATE * weightsPressure
        ) {
            this._mutateReadoutWeightShift(weightsPressure);
        }

        if (
            Math.random() <
            this._geneLstm.PROBABILITY_MUTATE_READOUT_B * this._geneLstm.MUTATION_RATE * weightsPressure
        ) {
            this._mutateReadoutBiasShift(weightsPressure);
        }

        if (
            Math.random() <
            this._geneLstm.PROBABILITY_MUTATE_WEIGHT_RANDOM * this._geneLstm.MUTATION_RATE * weightsPressure
        ) {
            // Refactored to use simpler Bernoulli sampling with pressure scaling
            this._mutateWeightRandom(weightsPressure);
        }

        if (
            Math.random() <
            this._geneLstm.PROBABILITY_MUTATE_BIAS_RANDOM * this._geneLstm.MUTATION_RATE * weightsPressure
        ) {
            this._mutateBiasRandom(weightsPressure);
        }

        if (
            Math.random() <
            this._geneLstm.PROBABILITY_MUTATE_WEIGHT_SHIFT * this._geneLstm.MUTATION_RATE * weightsPressure
        ) {
            this._mutateWeightShift(weightsPressure);
        }

        if (
            Math.random() <
            this._geneLstm.PROBABILITY_MUTATE_BIAS_SHIFT * this._geneLstm.MUTATION_RATE * weightsPressure
        ) {
            this._mutateBiasShift(weightsPressure);
        }

        // Mutate alpha (skip connection strength) occasionally
        if (
            Math.random() <
            this._geneLstm.PROBABILITY_MUTATE_ALPHA_SHIFT * this._geneLstm.MUTATION_RATE * weightsPressure
        ) {
            this._mutateAlpha(weightsPressure);
        }
    }
}
