import { Genome } from './genome.js';
import { Species } from './species.js';

export class Client {
    species: Species | null;
    genome: Genome;
    bestScore: boolean = false;
    error: number = 0;
    score: number = 0;

    constructor(LSTM: Genome) {
        this.genome = LSTM;
        this.species = null;
    }

    mutate(force = false) {
        if (this.bestScore && !force) {
            return;
        }
        this.genome.mutate();
    }

    distance(client: Client): number {
        return this.genome.distance(client.genome);
    }

    calculate(input: number[]): number[] {
        return this.genome.calculate(input);
    }
}
