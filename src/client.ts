import { Genome } from './genome.js';
import { Species } from './species.js';
import type { SeqInput } from './types/index.js';

export class Client {
    species: Species | null;
    genome: Genome;
    bestScore: boolean = false;
    error: number = 0;
    score: number = 0;
    /**
     * Fitness assigned by fit() or manually by the library user.
     * Higher is better.
     */
    scoreRaw = 0;

    /**
     * Raw fitness after applying the complexity penalty,
     * before normalization into the selection score.
     */
    adjustedScore = 0;

    /**
     * Cached structural complexity for the current generation.
     */
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

    distance(client: Client): number {
        return this.genome.distance(client.genome);
    }

    calculate(input: SeqInput): number[] {
        return this.genome.calculate(input);
    }
}
