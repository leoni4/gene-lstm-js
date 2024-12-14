import { Genome } from './genome';
import { Species } from './species';

export class Client {
    #species: Species | null;
    #genome: Genome;
    #bestScore: boolean = false;

    constructor(LSTM: Genome) {
        this.#genome = LSTM;
        this.#species = null;
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
