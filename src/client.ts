import { Genome } from './genome.js';
import { Species } from './species.js';
import type { LSTM } from './lstm.js';
import type { SeqInput } from './types/index.js';

export class Client {
    species: Species | null;
    genome: Genome;
    bestScore: boolean = false;
    error: number = 0;
    score: number = 0;
    scoreRaw = 0;
    adjustedScore = 0;
    complexity = 0;

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

    distance(client: Client, cache?: Map<LSTM, number[]>): number {
        return this.genome.distance(client.genome, cache);
    }

    calculate(input: SeqInput): number[] {
        return this.genome.calculate(input);
    }

    model() {
        return this.genome.lstmArray.map(l => l.model());
    }
}
