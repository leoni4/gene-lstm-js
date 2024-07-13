import { LSTM } from './lstm';

export class Client {
    #lstm: LSTM;
    #bestScore: boolean = false;

    constructor(LSTM: LSTM) {
        this.#lstm = LSTM;
    }

    mutate(force = false) {
        if (this.#bestScore && !force) {
            return;
        }
        this.#lstm.mutate();
    }

    calculate(input: number[]): number {
        return this.#lstm.calculate(input);
    }
}
