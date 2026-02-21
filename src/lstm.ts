import { GeneLSTM } from './gLstm.js';
import type { LstmOptions } from './types/index.js';

type ActivationName = 'sigmoid' | 'tanh';
type ActivationFunction = (x: number) => number;

function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

export class ShortMemoryBlock {
    private _activationName: ActivationName;
    private _activationFunction: ActivationFunction;
    weight1: number = 0;
    weight2: number = 0;
    bias: number = 0;

    constructor(activation: ActivationName, weight1?: number, weight2?: number, bias?: number) {
        this._activationName = activation;
        if (this._activationName === 'sigmoid') {
            this._activationFunction = sigmoid;
        } else {
            this._activationFunction = Math.tanh;
        }
        this.weight1 = weight1 || Math.random() * 2 - 1;
        this.weight2 = weight2 || Math.random() * 2 - 1;
        this.bias = bias || Math.random() * 2 - 1;
    }

    calculate(input: number, shortMemory: number): number {
        const shortMemoryCalculated = this.weight1 * shortMemory;
        const inputCalculated = this.weight2 * input;
        const summ = shortMemoryCalculated + inputCalculated + this.bias;
        const out = this._activationFunction(summ);

        return out;
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
            this._forgetGate.weight1,
            this._forgetGate.weight2,
            this._forgetGate.bias,
            this._potentialLongToRem.weight1,
            this._potentialLongToRem.weight2,
            this._potentialLongToRem.bias,
            this._potentialLongMemory.weight1,
            this._potentialLongMemory.weight2,
            this._potentialLongMemory.bias,
            this._shortMemoryToRemember.weight1,
            this._shortMemoryToRemember.weight2,
            this._shortMemoryToRemember.bias,
        ];
    }

    private _predictUnit(input: number) {
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

    calculate(input: number[], fullSeq = false): number[] {
        this.longMemory = 0;
        this.shortMemory = 0;
        const fullSeqMemory: number[] = [];
        input.forEach(num => {
            this._predictUnit(num);
            if (fullSeq) {
                fullSeqMemory.push(this.shortMemory);
            }
        });

        return fullSeq ? fullSeqMemory : [this.shortMemory];
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

    private _mutateWeightRandom() {
        const block = this._getBlockToMutate();
        const weightNum = `weight${Math.floor(Math.random() * 2 + 1)}` as 'weight1' | 'weight2';

        let newWeight = block[weightNum] || this._geneLstm.WEIGHT_RANDOM_STRENGTH;
        while (newWeight === block[weightNum]) {
            newWeight = (Math.random() * newWeight * 2 - newWeight) * this._geneLstm.WEIGHT_RANDOM_STRENGTH;
        }
        block[weightNum] = newWeight;
    }

    private _mutateBiasRandom() {
        const block = this._getBlockToMutate();

        let newWeight = block.bias || this._geneLstm.BIAS_RANDOM_STRENGTH;
        while (newWeight === block.bias) {
            newWeight = (Math.random() * newWeight * 2 - newWeight) * this._geneLstm.BIAS_RANDOM_STRENGTH;
        }
        block.bias = newWeight;
    }

    private _mutateWeightShift() {
        const block = this._getBlockToMutate();
        const weightNum = `weight${Math.floor(Math.random() * 2 + 1)}` as 'weight1' | 'weight2';

        let newWeight = block[weightNum] || this._geneLstm.WEIGHT_SHIFT_STRENGTH;
        while (newWeight === block[weightNum]) {
            newWeight = block[weightNum] + (Math.random() * 2 - 1) * this._geneLstm.WEIGHT_SHIFT_STRENGTH;
        }
        // Clamp to reasonable range
        block[weightNum] = Math.max(-10, Math.min(10, newWeight));
    }

    private _mutateBiasShift() {
        const block = this._getBlockToMutate();

        let newBias = block.bias || this._geneLstm.BIAS_SHIFT_STRENGTH;
        while (newBias === block.bias) {
            newBias = block.bias + (Math.random() * 2 - 1) * this._geneLstm.BIAS_SHIFT_STRENGTH;
        }
        // Clamp to reasonable range
        block.bias = Math.max(-10, Math.min(10, newBias));
    }

    private _mutateAlpha() {
        const delta = (Math.random() * 2 - 1) * this._geneLstm.ALPHA_SHIFT_STRENGTH; // ±10% change
        this.alpha = this._alpha + delta;
    }

    mutate() {
        // Refactored to use simpler Bernoulli sampling
        if (Math.random() < this._geneLstm.PROBABILITY_MUTATE_WEIGHT_RANDOM * this._geneLstm.MUTATION_RATE) {
            this._mutateWeightRandom();
        }

        if (Math.random() < this._geneLstm.PROBABILITY_MUTATE_BIAS_RANDOM * this._geneLstm.MUTATION_RATE) {
            this._mutateBiasRandom();
        }

        if (Math.random() < this._geneLstm.PROBABILITY_MUTATE_WEIGHT_SHIFT * this._geneLstm.MUTATION_RATE) {
            this._mutateWeightShift();
        }

        if (Math.random() < this._geneLstm.PROBABILITY_MUTATE_BIAS_SHIFT * this._geneLstm.MUTATION_RATE) {
            this._mutateBiasShift();
        }

        // Mutate alpha (skip connection strength) occasionally
        if (Math.random() < this._geneLstm.PROBABILITY_MUTATE_ALPHA_SHIFT * this._geneLstm.MUTATION_RATE) {
            this._mutateAlpha();
        }
    }
}
