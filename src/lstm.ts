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

    longMemory: number = 0;
    shortMemory: number = 0;

    private _forgetGate: ShortMemoryBlock;
    private _potentialLongToRem: ShortMemoryBlock;
    private _potentialLongMemory: ShortMemoryBlock;

    private _shortMemoryToRemember: ShortMemoryBlock;

    private _outputGate: OutputBlock;

    // Skip connection strength for non-destructive mutations
    private _alpha: number;

    constructor(GeneLSTM: GeneLSTM, options?: LstmOptions) {
        this._geneLstm = GeneLSTM;
        this._alpha = options?.alpha ?? 1.0; // Default: no skip connection

        if (options) {
            this._forgetGate = new ShortMemoryBlock(
                'sigmoid',
                options.forgetGate.weight1,
                options.forgetGate.weight2,
                options.forgetGate.bias,
            );
            this._potentialLongToRem = new ShortMemoryBlock(
                'sigmoid',
                options.potentialLongToRem.weight1,
                options.potentialLongToRem.weight2,
                options.potentialLongToRem.bias,
            );
            this._potentialLongMemory = new ShortMemoryBlock(
                'tanh',
                options.potentialLongMemory.weight1,
                options.potentialLongMemory.weight2,
                options.potentialLongMemory.bias,
            );
            this._shortMemoryToRemember = new ShortMemoryBlock(
                'sigmoid',
                options.shortMemoryToRemember.weight1,
                options.shortMemoryToRemember.weight2,
                options.shortMemoryToRemember.bias,
            );
        } else {
            this._forgetGate = new ShortMemoryBlock('sigmoid');
            this._potentialLongToRem = new ShortMemoryBlock('sigmoid');
            this._potentialLongMemory = new ShortMemoryBlock('tanh');
            this._shortMemoryToRemember = new ShortMemoryBlock('sigmoid');
        }
        this._outputGate = new OutputBlock();
    }

    get alpha(): number {
        return this._alpha;
    }

    set alpha(value: number) {
        // Clamp alpha between 0 and 1
        this._alpha = Math.max(0, Math.min(1, value));
    }

    flattenWeights(): number[] {
        return [
            ...flattenBlock(this._forgetGate),
            ...flattenBlock(this._potentialLongToRem),
            ...flattenBlock(this._potentialLongMemory),
            ...flattenBlock(this._shortMemoryToRemember),
        ];
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
            const n = this._geneLstm.INPUT_FEATURES ?? 3;
            block.weightIn = new Array(n).fill(0).map(() => Math.random() * 2 - 1);
        }
    }

    calculate(input: number[] | number[][], fullSeq = false): number[] {
        this.longMemory = 0;
        this.shortMemory = 0;

        const fullSeqMemory: number[] = [];

        // input is number[][]
        if (Array.isArray(input[0])) {
            const seq = input as number[][];
            for (const x_t of seq) {
                this._predictUnit(x_t);
                if (fullSeq) fullSeqMemory.push(this.shortMemory);
            }

            return fullSeq ? fullSeqMemory : [this.shortMemory];
        }

        // input is number[]
        const seq = input as number[];
        for (const num of seq) {
            this._predictUnit(num);
            if (fullSeq) fullSeqMemory.push(this.shortMemory);
        }

        return fullSeq ? fullSeqMemory : [this.shortMemory];
    }

    private _predictUnit(input: number | number[]) {
        const forgetOut = this._forgetGate.calculate(input, this.shortMemory);
        this.longMemory *= forgetOut;

        const potentialLongToRem = this._potentialLongToRem.calculate(input, this.shortMemory);
        const potentialLong = this._potentialLongMemory.calculate(input, this.shortMemory);

        this.longMemory += potentialLongToRem * potentialLong;

        const potentialShortToRem = this._shortMemoryToRemember.calculate(input, this.shortMemory);

        const output = this._outputGate.calculate(this.longMemory, potentialShortToRem);
        this.shortMemory = output;

        return output;
    }

    model(): LstmOptions {
        return {
            forgetGate: {
                weight1: this._forgetGate.weight1,
                weight2: this._forgetGate.weight2,
                bias: this._forgetGate.bias,
            },
            potentialLongToRem: {
                weight1: this._potentialLongToRem.weight1,
                weight2: this._potentialLongToRem.weight2,
                bias: this._potentialLongToRem.bias,
            },
            potentialLongMemory: {
                weight1: this._potentialLongMemory.weight1,
                weight2: this._potentialLongMemory.weight2,
                bias: this._potentialLongMemory.bias,
            },
            shortMemoryToRemember: {
                weight1: this._shortMemoryToRemember.weight1,
                weight2: this._shortMemoryToRemember.weight2,
                bias: this._shortMemoryToRemember.bias,
            },
            alpha: this._alpha,
        };
    }

    private _getBlockToMutate(): ShortMemoryBlock {
        const blockNum = Math.floor(Math.random() * 4 + 1);

        let block: ShortMemoryBlock = this._forgetGate;
        switch (blockNum) {
            case 1:
                block = this._forgetGate;
                break;
            case 2:
                block = this._potentialLongToRem;
                break;
            case 3:
                block = this._potentialLongMemory;
                break;
            case 4:
                block = this._shortMemoryToRemember;
                break;
        }

        return block;
    }

    private _mutateWeightRandom(pressureScale: number) {
        const block = this._getBlockToMutate();
        this._ensureWeightIn(block);

        const target = this._pickWeightTarget(block);

        if (target.kind === 'scalar') {
            const key = target.key;
            const base = Math.abs(block[key] ?? 1); // avoid 0 base
            let newWeight = block[key] ?? 0;

            while (newWeight === block[key]) {
                newWeight = (Math.random() * 2 - 1) * base * this._geneLstm.WEIGHT_RANDOM_STRENGTH * pressureScale;
            }

            block[key] = newWeight;

            return;
        }

        const i = target.index;
        let newWeight = block.weightIn![i] ?? 0;

        while (newWeight === block.weightIn![i]) {
            const range = this._geneLstm.WEIGHT_RANDOM_STRENGTH * pressureScale;
            newWeight = (Math.random() * 2 - 1) * range;
        }

        block.weightIn![i] = newWeight;
    }

    private _mutateBiasRandom(pressureScale: number) {
        const block = this._getBlockToMutate();

        let newWeight = block.bias ?? this._geneLstm.BIAS_RANDOM_STRENGTH;
        while (newWeight === block.bias) {
            newWeight =
                (Math.random() * newWeight * 2 - newWeight) * this._geneLstm.BIAS_RANDOM_STRENGTH * pressureScale;
        }
        block.bias = newWeight;
    }

    private _mutateWeightShift(pressureScale: number) {
        const block = this._getBlockToMutate();

        this._ensureWeightIn(block);

        const target = this._pickWeightTarget(block);

        if (target.kind === 'scalar') {
            const key = target.key;
            const current = block[key] ?? this._geneLstm.WEIGHT_SHIFT_STRENGTH;

            let newWeight = current;
            while (newWeight === current) {
                newWeight = current + (Math.random() * 2 - 1) * this._geneLstm.WEIGHT_SHIFT_STRENGTH * pressureScale;
            }
            block[key] = Math.max(-10, Math.min(10, newWeight));

            return;
        }

        // vector
        const i = target.index;
        const current = block.weightIn![i] ?? 0;

        let newWeight = current;
        while (newWeight === current) {
            newWeight = current + (Math.random() * 2 - 1) * this._geneLstm.WEIGHT_SHIFT_STRENGTH * pressureScale;
        }
        block.weightIn![i] = Math.max(-10, Math.min(10, newWeight));
    }

    private _mutateBiasShift(pressureScale: number) {
        const block = this._getBlockToMutate();

        let newBias = block.bias ?? this._geneLstm.BIAS_SHIFT_STRENGTH;
        while (newBias === block.bias) {
            newBias = block.bias + (Math.random() * 2 - 1) * this._geneLstm.BIAS_SHIFT_STRENGTH * pressureScale;
        }
        // Clamp to reasonable range
        block.bias = Math.max(-10, Math.min(10, newBias));
    }

    private _mutateAlpha(pressureScale: number) {
        const delta = (Math.random() * 2 - 1) * this._geneLstm.ALPHA_SHIFT_STRENGTH * pressureScale;
        this.alpha = this._alpha + delta;
    }

    mutate() {
        // Apply weights mutation pressure to all weight/bias mutations
        const pressure = this._geneLstm.getMutationPressure();
        const weightsPressure = pressure.weights;

        // Refactored to use simpler Bernoulli sampling with pressure scaling
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

        // Mutate alpha (skip connection strength) occasionally
        if (
            Math.random() <
            this._geneLstm.PROBABILITY_MUTATE_ALPHA_SHIFT * this._geneLstm.MUTATION_RATE * weightsPressure
        ) {
            this._mutateAlpha(weightsPressure);
        }
    }
}
