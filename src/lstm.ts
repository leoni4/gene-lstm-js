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
            if (!this.weightIn || this.weightIn.length !== input.length) {
                this.weightIn = new Array(input.length).fill(0).map(() => Math.random() * 2 - 1);
            }
            for (let i = 0; i < input.length; i++) {
                inTerm += this.weightIn[i] * input[i];
            }
        } else {
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

    readoutW: number[][];
    readoutB: number[];

    private _outputDim: number;
    private _outputActivation: 'sigmoid' | 'tanh' | 'identity';

    private _forgetGate: ShortMemoryBlock[];
    private _potentialLongToRem: ShortMemoryBlock[];
    private _potentialLongMemory: ShortMemoryBlock[];

    private _shortMemoryToRemember: ShortMemoryBlock[];

    private _alpha: number;

    constructor(geneLstm: GeneLSTM, options?: LstmOptions) {
        this._geneLstm = geneLstm;
        this._alpha = options?.alpha ?? 1.0;

        const H = options?.hiddenSize ?? 1;

        this._outputDim = options?.outputDim ?? this._geneLstm.OUTPUT_DIM ?? 1;
        this._outputActivation = options?.outputActivation ?? this._geneLstm.OUTPUT_ACTIVATION ?? 'sigmoid';

        const makeBlock = (
            act: ActivationName,
            u?: { weight1?: number; weight2?: number; bias?: number; weightIn?: number[] },
        ) => new ShortMemoryBlock(act, u?.weight1, u?.weight2, u?.bias, u?.weightIn ? [...u.weightIn] : undefined);

        if (options) {
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

            if (options.readoutW && !Array.isArray(options.readoutW[0])) {
                const oldW = options.readoutW as number[];
                const oldB = options.readoutB as number;
                this.readoutW = [oldW.length === H ? [...oldW] : new Array(H).fill(0)];
                this.readoutB = [oldB ?? 0];
                this._outputDim = 1;
            } else {
                const optW = options.readoutW as number[][];
                const optB = options.readoutB as number[];

                if (optW && Array.isArray(optW[0])) {
                    this.readoutW = optW.map(row => (row.length === H ? [...row] : new Array(H).fill(0)));
                } else {
                    this.readoutW = new Array(this._outputDim).fill(0).map(() => new Array(H).fill(0));
                }

                if (optB && Array.isArray(optB)) {
                    this.readoutB = [...optB];
                } else {
                    this.readoutB = new Array(this._outputDim).fill(0);
                }
            }
        } else {
            this._forgetGate = new Array(H).fill(0).map(() => new ShortMemoryBlock('sigmoid'));
            this._potentialLongToRem = new Array(H).fill(0).map(() => new ShortMemoryBlock('sigmoid'));
            this._potentialLongMemory = new Array(H).fill(0).map(() => new ShortMemoryBlock('tanh'));
            this._shortMemoryToRemember = new Array(H).fill(0).map(() => new ShortMemoryBlock('sigmoid'));

            const eps = 0.2;
            this.readoutW = new Array(this._outputDim)
                .fill(0)
                .map(() => new Array(H).fill(0).map(() => (Math.random() * 2 - 1) * eps));
            this.readoutB = new Array(this._outputDim).fill(0).map(() => (Math.random() * 2 - 1) * eps);
        }

        this.longMemory = new Array(H).fill(0);
        this.shortMemory = new Array(H).fill(0);

        this._ensureConsistentSizes();
    }

    get alpha(): number {
        return this._alpha;
    }

    set alpha(value: number) {
        this._alpha = Math.max(0, Math.min(1, value));
    }

    private _hiddenSize(): number {
        return Math.max(1, this.readoutW[0]?.length || 1);
    }

    private _ensureConsistentSizes() {
        const H = this._hiddenSize();

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

        while (this.readoutW.length < this._outputDim) {
            this.readoutW.push(new Array(H).fill(0));
        }
        while (this.readoutW.length > this._outputDim) {
            this.readoutW.pop();
        }

        for (let j = 0; j < this.readoutW.length; j++) {
            if (!this.readoutW[j] || this.readoutW[j].length !== H) {
                this.readoutW[j] = new Array(H).fill(0);
            }
        }

        while (this.readoutB.length < this._outputDim) {
            this.readoutB.push(0);
        }
        while (this.readoutB.length > this._outputDim) {
            this.readoutB.pop();
        }
    }

    flattenWeights(): number[] {
        this._ensureConsistentSizes();

        const out: number[] = [];

        for (const b of this._forgetGate) out.push(...flattenBlock(b));
        for (const b of this._potentialLongToRem) out.push(...flattenBlock(b));
        for (const b of this._potentialLongMemory) out.push(...flattenBlock(b));
        for (const b of this._shortMemoryToRemember) out.push(...flattenBlock(b));

        for (const row of this.readoutW) {
            out.push(...row);
        }
        out.push(...this.readoutB, this._alpha);

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

        this.longMemory.fill(0);
        this.shortMemory.fill(0);

        const D = this._outputDim;
        const fullSeqMemory: number[] | null = fullSeq ? new Array(input.length * D) : null;
        let offset = 0;

        if (Array.isArray(input[0])) {
            const seq = input as number[][];
            for (const x_t of seq) {
                if (Array.isArray(x_t)) this._predictUnitVector(x_t);
                else this._predictUnit(x_t);
                if (fullSeqMemory) {
                    this._readoutInto(fullSeqMemory, offset);
                    offset += D;
                }
            }

            if (fullSeqMemory) return fullSeqMemory;

            const y = this._readout();

            const last = seq.length ? seq[seq.length - 1] : [0];
            const yPrev = typeof last[0] === 'number' ? last[0] : 0;

            const a = this._alpha;

            const result = [...y];
            result[0] = (1 - a) * yPrev + a * y[0];

            return result;
        }

        const seq = input as number[];
        for (const num of seq) {
            if (Array.isArray(num)) this._predictUnitVector(num);
            else this._predictUnit(num);
            if (fullSeqMemory) {
                this._readoutInto(fullSeqMemory, offset);
                offset += D;
            }
        }

        if (fullSeqMemory) return fullSeqMemory;

        const y = this._readout();

        const lastIn = seq.length ? seq[seq.length - 1] : 0;
        const a = this._alpha;

        const result = [...y];
        result[0] = (1 - a) * lastIn + a * y[0];

        return result;
    }

    private _readout(): number[] {
        const output = new Array(this._outputDim);
        this._readoutInto(output, 0);

        return output;
    }

    private _readoutInto(output: number[], offset: number) {
        for (let j = 0; j < this._outputDim; j++) {
            let s = this.readoutB[j];
            for (let k = 0; k < this.shortMemory.length; k++) {
                s += this.readoutW[j][k] * this.shortMemory[k];
            }

            if (this._outputActivation === 'sigmoid') {
                output[offset + j] = sigmoid(s);
            } else if (this._outputActivation === 'tanh') {
                output[offset + j] = Math.tanh(s);
            } else {
                output[offset + j] = s;
            }
        }
    }

    private _predictUnit(x: number) {
        const H = this.shortMemory.length;
        const long = this.longMemory;
        const short = this.shortMemory;

        for (let k = 0; k < H; k++) {
            const fb = this._forgetGate[k];
            const ib = this._potentialLongToRem[k];
            const gb = this._potentialLongMemory[k];
            const ob = this._shortMemoryToRemember[k];
            const hPrev = short[k];

            const f = sigmoid(fb.weight1 * hPrev + fb.weight2 * x + fb.bias);
            long[k] *= f;

            const i = sigmoid(ib.weight1 * hPrev + ib.weight2 * x + ib.bias);
            const g = Math.tanh(gb.weight1 * hPrev + gb.weight2 * x + gb.bias);
            long[k] += i * g;

            const o = sigmoid(ob.weight1 * hPrev + ob.weight2 * x + ob.bias);
            short[k] = Math.tanh(long[k]) * o;
        }
    }

    private _predictUnitVector(x: number[]) {
        const H = this.shortMemory.length;
        const n = x.length;
        const long = this.longMemory;
        const short = this.shortMemory;

        for (let k = 0; k < H; k++) {
            const fb = this._forgetGate[k];
            const ib = this._potentialLongToRem[k];
            const gb = this._potentialLongMemory[k];
            const ob = this._shortMemoryToRemember[k];

            if (!fb.weightIn || fb.weightIn.length !== n) {
                fb.weightIn = new Array(n).fill(0).map(() => Math.random() * 2 - 1);
            }
            if (!ib.weightIn || ib.weightIn.length !== n) {
                ib.weightIn = new Array(n).fill(0).map(() => Math.random() * 2 - 1);
            }
            if (!gb.weightIn || gb.weightIn.length !== n) {
                gb.weightIn = new Array(n).fill(0).map(() => Math.random() * 2 - 1);
            }
            if (!ob.weightIn || ob.weightIn.length !== n) {
                ob.weightIn = new Array(n).fill(0).map(() => Math.random() * 2 - 1);
            }

            const fw = fb.weightIn;
            const iw = ib.weightIn;
            const gw = gb.weightIn;
            const ow = ob.weightIn;
            const hPrev = short[k];

            let fd = 0;
            let id = 0;
            let gd = 0;
            let od = 0;
            for (let j = 0; j < n; j++) {
                const xj = x[j];
                fd += fw[j] * xj;
                id += iw[j] * xj;
                gd += gw[j] * xj;
                od += ow[j] * xj;
            }

            const f = sigmoid(fb.weight1 * hPrev + fd + fb.bias);
            long[k] *= f;

            const i = sigmoid(ib.weight1 * hPrev + id + ib.bias);
            const g = Math.tanh(gb.weight1 * hPrev + gd + gb.bias);
            long[k] += i * g;

            const o = sigmoid(ob.weight1 * hPrev + od + ob.bias);
            short[k] = Math.tanh(long[k]) * o;
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

    copyOptions(): LstmOptions {
        return { ...this.model(), outputDim: this._outputDim, outputActivation: this._outputActivation };
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

        const i = target.index;
        const current = block.weightIn![i] ?? 0;

        let newWeight = current + (Math.random() * 2 - 1) * this._geneLstm.WEIGHT_SHIFT_STRENGTH * pressureScale;
        block.weightIn![i] = Math.max(-10, Math.min(10, newWeight));
    }

    private _mutateBiasShift(pressureScale: number) {
        const block = this._getBlockToMutate();

        let newBias = block.bias + (Math.random() * 2 - 1) * this._geneLstm.BIAS_SHIFT_STRENGTH * pressureScale;

        block.bias = Math.max(-10, Math.min(10, newBias));
    }

    private _mutateAlpha(pressureScale: number) {
        const delta = (Math.random() * 2 - 1) * this._geneLstm.ALPHA_SHIFT_STRENGTH * pressureScale;
        this.alpha = this._alpha + delta;
    }

    private _mutateAddUnit() {
        this._ensureConsistentSizes();

        this._forgetGate.push(new ShortMemoryBlock('sigmoid'));
        this._potentialLongToRem.push(new ShortMemoryBlock('sigmoid'));
        this._potentialLongMemory.push(new ShortMemoryBlock('tanh'));
        this._shortMemoryToRemember.push(new ShortMemoryBlock('sigmoid'));

        this.longMemory.push(0);
        this.shortMemory.push(0);

        for (let j = 0; j < this._outputDim; j++) {
            this.readoutW[j].push(0);
        }
    }

    private _mutateRemoveUnit() {
        this._ensureConsistentSizes();
        const H = this.shortMemory.length;
        if (H <= 1) return;

        let idx = 0;
        let best = Infinity;
        for (let k = 0; k < H; k++) {
            let totalWeight = 0;
            for (let j = 0; j < this._outputDim; j++) {
                totalWeight += Math.abs(this.readoutW[j][k]);
            }
            if (totalWeight < best) {
                best = totalWeight;
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

        for (let j = 0; j < this._outputDim; j++) {
            this.readoutW[j].splice(idx, 1);
        }
    }

    private _mutateReadoutWeightShift(pressureScale: number) {
        this._ensureConsistentSizes();
        const j = Math.floor(Math.random() * this._outputDim);
        const k = Math.floor(Math.random() * this.readoutW[j].length);
        const delta = (Math.random() * 2 - 1) * this._geneLstm.WEIGHT_SHIFT_STRENGTH * pressureScale;
        this.readoutW[j][k] = Math.max(-10, Math.min(10, this.readoutW[j][k] + delta));
    }

    private _mutateReadoutBiasShift(pressureScale: number) {
        this._ensureConsistentSizes();
        const j = Math.floor(Math.random() * this._outputDim);
        const delta = (Math.random() * 2 - 1) * this._geneLstm.BIAS_SHIFT_STRENGTH * pressureScale;
        this.readoutB[j] = Math.max(-10, Math.min(10, this.readoutB[j] + delta));
    }

    mutate() {
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

        if (
            Math.random() <
            this._geneLstm.PROBABILITY_MUTATE_ALPHA_SHIFT * this._geneLstm.MUTATION_RATE * weightsPressure
        ) {
            this._mutateAlpha(weightsPressure);
        }
    }
}
