import { Genome } from './genome';

export class Client {
    #genome: Genome;
    #bestScore: boolean = false;

    constructor(LSTM: Genome) {
        this.#genome = LSTM;
    }

    mutate(force = false) {
        if (this.#bestScore && !force) {
            return;
        }
        this.#genome.mutate();
    }

    calculate(input: number[]): number[] {
        return this.#genome.calculate(input);
    }
}
