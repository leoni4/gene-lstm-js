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

export class LSTM {
    #longMemory: number = 0;
    #shortMemory: number = 0;

    #forgetGate: ShortMemoryBlock;
    #potentialLongToRem: ShortMemoryBlock;
    #potentialLongMemory: ShortMemoryBlock;

    #shortMemoryToRemember: ShortMemoryBlock;

    #outputGate: OutputBlock;

    constructor(options?: LstmOptions) {
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

    evolve() {}
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
        if (weight1) {
            this.#weight1 = weight1;
        }
        if (weight2) {
            this.#weight2 = weight2;
        }
        if (bias) {
            this.#bias = bias;
        }
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
