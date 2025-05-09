import { Genome } from './genome';
import { Species } from './species';

export class Client {
    #species: Species | null;
    #genome: Genome;
    #bestScore: boolean = false;
    #error: number = 0;
    #score: number = 0;

    constructor(LSTM: Genome) {
        this.#genome = LSTM;
        this.#species = null;
    }

    get error() {
        return this.#error;
    }
    set error(num: number) {
        this.#error = num;
    }
    get score() {
        return this.#score;
    }
    set score(num: number) {
        this.#score = num;
    }
    get bestScore() {
        return this.#bestScore;
    }
    set bestScore(bol: boolean) {
        this.#bestScore = bol;
    }
    get genome() {
        return this.#genome;
    }

    set species(species: Species | null) {
        this.#species = species;
    }

    get species() {
        return this.#species;
    }

    mutate(force = false) {
        if (this.#bestScore && !force) {
            return;
        }
        this.#genome.mutate();
    }

    distance(client: Client): number {
        return this.#genome.distance(client.genome);
    }

    calculate(input: number[]): number[] {
        return this.#genome.calculate(input);
    }
}
