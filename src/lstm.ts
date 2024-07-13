import { GeneLSTM } from './gLstm';

interface ShortMemory {
    weight1: number;
    weight2: number;
    bias: number;
}

interface LstmOptions {
    forgetGate: ShortMemory;
    potentialLongToRem: ShortMemory;
    potentialLongMemory: ShortMemory;
    shortMemoryToRemember: ShortMemory;
}

type ActivationName = 'sigmoid' | 'tanh';
type ActivationFunction = (x: number) => number;

function sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
}

class ShortMemoryBlock {
    #activationName: ActivationName;
    #activationFunction: ActivationFunction;
    #weight1: number = 0;
    #weight2: number = 0;
    #bias: number = 0;

    constructor(activation: ActivationName, weight1?: number, weight2?: number, bias?: number) {
        this.#activationName = activation;
        if (this.#activationName === 'sigmoid') {
            this.#activationFunction = sigmoid;
        } else {
            this.#activationFunction = Math.tanh;
        }
        this.#weight1 = weight1 || Math.random() * 2 - 1;
        this.#weight2 = weight2 || Math.random() * 2 - 1;
        this.#bias = bias || Math.random() * 2 - 1;
    }

    get weight1() {
        return this.#weight1;
    }
    get weight2() {
        return this.#weight1;
    }
    get bias() {
        return this.#weight1;
    }
    set weight1(num: number) {
        this.#weight1 = num;
    }
    set weight2(num: number) {
        this.#weight2 = num;
    }
    set bias(num: number) {
        this.#bias = num;
    }

    calculate(input: number, shortMemory: number): number {
        const shortMemoryCalculated = this.#weight1 * shortMemory;
        const inputCalculated = this.#weight2 * input;
        const summ = shortMemoryCalculated + inputCalculated + this.#bias;
        const out = this.#activationFunction(summ);
        return out;
    }
}

class OutputBlock {
    calculate(longMemory: number, shortMemory: number) {
        const out = Math.tanh(longMemory) * shortMemory;
        return out;
    }
}

export class LSTM {
    #geneLstm: GeneLSTM;

    #longMemory: number = 0;
    #shortMemory: number = 0;

    #forgetGate: ShortMemoryBlock;
    #potentialLongToRem: ShortMemoryBlock;
    #potentialLongMemory: ShortMemoryBlock;

    #shortMemoryToRemember: ShortMemoryBlock;

    #outputGate: OutputBlock;

    constructor(GeneLSTM: GeneLSTM, options?: LstmOptions) {
        this.#geneLstm = GeneLSTM;
        if (options) {
            this.#forgetGate = new ShortMemoryBlock(
                'sigmoid',
                options.forgetGate.weight1,
                options.forgetGate.weight2,
                options.forgetGate.bias,
            );
            this.#potentialLongToRem = new ShortMemoryBlock(
                'sigmoid',
                options.potentialLongToRem.weight1,
                options.potentialLongToRem.weight2,
                options.potentialLongToRem.bias,
            );
            this.#potentialLongMemory = new ShortMemoryBlock(
                'tanh',
                options.potentialLongMemory.weight1,
                options.potentialLongMemory.weight2,
                options.potentialLongMemory.bias,
            );
            this.#shortMemoryToRemember = new ShortMemoryBlock(
                'sigmoid',
                options.shortMemoryToRemember.weight1,
                options.shortMemoryToRemember.weight2,
                options.shortMemoryToRemember.bias,
            );
        } else {
            this.#forgetGate = new ShortMemoryBlock('sigmoid');
            this.#potentialLongToRem = new ShortMemoryBlock('sigmoid');
            this.#potentialLongMemory = new ShortMemoryBlock('tanh');
            this.#shortMemoryToRemember = new ShortMemoryBlock('sigmoid');
        }
        this.#outputGate = new OutputBlock();
    }

    set longMemory(num: number) {
        this.#longMemory = num;
    }

    set shortMemory(num: number) {
        this.#shortMemory = num;
    }
    get longMemory() {
        return this.#longMemory;
    }

    get shortMemory() {
        return this.#shortMemory;
    }

    #predictUnit(input: number) {
        const forgetOut = this.#forgetGate.calculate(input, this.#shortMemory);
        this.#longMemory *= forgetOut;

        const potentialLongToRem = this.#potentialLongToRem.calculate(input, this.#shortMemory);
        const potentialLong = this.#potentialLongMemory.calculate(input, this.#shortMemory);

        this.#longMemory += potentialLongToRem * potentialLong;

        const potentialShortToRem = this.#shortMemoryToRemember.calculate(input, this.#shortMemory);

        const output = this.#outputGate.calculate(this.#longMemory, potentialShortToRem);
        this.#shortMemory = output;
        return output;
    }

    calculate(input: number[]) {
        this.#longMemory = 0;
        this.#shortMemory = 0;
        input.forEach(num => {
            this.#predictUnit(num);
        });
        return this.#shortMemory;
    }

    #getBlockToMutate(): ShortMemoryBlock {
        const blockNum = Math.floor(Math.random() * 4 + 1);

        let block: ShortMemoryBlock = this.#forgetGate;
        switch (blockNum) {
            case 1:
                block = this.#forgetGate;
                break;
            case 2:
                block = this.#potentialLongToRem;
                break;
            case 3:
                block = this.#potentialLongMemory;
                break;
            case 4:
                block = this.#shortMemoryToRemember;
                break;
        }

        return block;
    }

    #mutateWeightRandom() {
        const block = this.#getBlockToMutate();
        const weightNum = `weight${Math.floor(Math.random() * 2 + 1)}` as 'weight1' | 'weight2';

        let newWeight = block[weightNum] || this.#geneLstm.WEIGHT_RANDOM_STRENGTH;
        while (newWeight === block[weightNum]) {
            newWeight = (Math.random() * newWeight * 2 - newWeight) * this.#geneLstm.WEIGHT_RANDOM_STRENGTH;
        }
        block[weightNum] = newWeight;
    }

    #mutateBiasRandom() {
        const block = this.#getBlockToMutate();

        let newWeight = block.bias || this.#geneLstm.BIAS_RANDOM_STRENGTH;
        while (newWeight === block.bias) {
            newWeight = (Math.random() * newWeight * 2 - newWeight) * this.#geneLstm.BIAS_RANDOM_STRENGTH;
        }
        block.bias = newWeight;
    }

    #mutateWeightShift() {
        const block = this.#getBlockToMutate();

        let newWeight = block.bias || this.#geneLstm.WEIGHT_SHIFT_STRENGTH;
        while (newWeight === block.bias) {
            newWeight = block.bias + (Math.random() * 2 - 1) * this.#geneLstm.WEIGHT_SHIFT_STRENGTH;
        }
        block.bias = newWeight;
    }

    #mutateBiasShift() {
        const block = this.#getBlockToMutate();
        const weightNum = `weight${Math.floor(Math.random() * 2 + 1)}` as 'weight1' | 'weight2';

        let newWeight = block[weightNum] || this.#geneLstm.WEIGHT_SHIFT_STRENGTH;
        while (newWeight === block[weightNum]) {
            newWeight = block[weightNum] + (Math.random() * 2 - 1) * this.#geneLstm.WEIGHT_SHIFT_STRENGTH;
        }
        block[weightNum] = newWeight;
    }

    mutate() {
        let prob: number;

        prob = this.#geneLstm.PROBABILITY_MUTATE_WEIGHT_RANDOM * this.#geneLstm.MUTATION_RATE;
        while (prob > Math.random()) {
            prob--;
            this.#mutateWeightRandom();
        }

        prob = this.#geneLstm.PROBABILITY_MUTATE_BIAS_RANDOM * this.#geneLstm.MUTATION_RATE;
        while (prob > Math.random()) {
            prob--;
            this.#mutateBiasRandom();
        }

        prob = this.#geneLstm.PROBABILITY_MUTATE_WEIGHT_SHIFT * this.#geneLstm.MUTATION_RATE;
        while (prob > Math.random()) {
            prob--;
            this.#mutateWeightShift();
        }

        prob = this.#geneLstm.PROBABILITY_MUTATE_BIAS_SHIFT * this.#geneLstm.MUTATION_RATE;
        while (prob > Math.random()) {
            prob--;
            this.#mutateBiasShift();
        }
    }
}
