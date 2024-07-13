import { Client } from './client';
import { Species } from './species';
import { LSTM } from './lstm';

interface GeneLSTMOptions {
    SURVIVORS?: number;
    MUTATION_RATE?: number;
    BIAS_SHIFT_STRENGTH?: number;
    BIAS_RANDOM_STRENGTH?: number;
    PROBABILITY_MUTATE_BIAS_SHIFT?: number;
    PROBABILITY_MUTATE_BIAS_RANDOM?: number;
    WEIGHT_SHIFT_STRENGTH?: number;
    WEIGHT_RANDOM_STRENGTH?: number;
    PROBABILITY_MUTATE_WEIGHT_SHIFT?: number;
    PROBABILITY_MUTATE_WEIGHT_RANDOM?: number;
    loadData?: object;
}

export class GeneLSTM {
    #clients: Client[] = [];
    #species: Species[] = [];
    #maxClients: number;

    #SURVIVORS: number;
    #MUTATION_RATE: number;

    #BIAS_SHIFT_STRENGTH: number;
    #BIAS_RANDOM_STRENGTH: number;
    #PROBABILITY_MUTATE_BIAS_SHIFT: number;
    #PROBABILITY_MUTATE_BIAS_RANDOM: number;

    #WEIGHT_SHIFT_STRENGTH: number;
    #WEIGHT_RANDOM_STRENGTH: number;
    #PROBABILITY_MUTATE_WEIGHT_SHIFT: number;
    #PROBABILITY_MUTATE_WEIGHT_RANDOM: number;

    constructor(clients: number, options?: GeneLSTMOptions) {
        this.#maxClients = clients;

        this.#SURVIVORS = options?.SURVIVORS || 1;
        this.#MUTATION_RATE = options?.MUTATION_RATE || 1;
        this.#BIAS_SHIFT_STRENGTH = options?.BIAS_SHIFT_STRENGTH || 1;
        this.#BIAS_RANDOM_STRENGTH = options?.BIAS_RANDOM_STRENGTH || 1;
        this.#PROBABILITY_MUTATE_BIAS_SHIFT = options?.PROBABILITY_MUTATE_BIAS_SHIFT || 1;
        this.#PROBABILITY_MUTATE_BIAS_RANDOM = options?.PROBABILITY_MUTATE_BIAS_RANDOM || 1;
        this.#WEIGHT_SHIFT_STRENGTH = options?.WEIGHT_SHIFT_STRENGTH || 1;
        this.#WEIGHT_RANDOM_STRENGTH = options?.WEIGHT_RANDOM_STRENGTH || 1;
        this.#PROBABILITY_MUTATE_WEIGHT_SHIFT = options?.PROBABILITY_MUTATE_WEIGHT_SHIFT || 1;
        this.#PROBABILITY_MUTATE_WEIGHT_RANDOM = options?.PROBABILITY_MUTATE_WEIGHT_RANDOM || 1;

        if (options?.loadData) {
            this.#load(options?.loadData);
        } else {
            this.#reset();
        }
    }

    get SURVIVORS() {
        return this.#SURVIVORS;
    }
    get MUTATION_RATE() {
        return this.#MUTATION_RATE;
    }
    get BIAS_SHIFT_STRENGTH() {
        return this.#BIAS_SHIFT_STRENGTH;
    }
    get BIAS_RANDOM_STRENGTH() {
        return this.#BIAS_RANDOM_STRENGTH;
    }
    get PROBABILITY_MUTATE_BIAS_SHIFT() {
        return this.#PROBABILITY_MUTATE_BIAS_SHIFT;
    }
    get PROBABILITY_MUTATE_BIAS_RANDOM() {
        return this.#PROBABILITY_MUTATE_BIAS_RANDOM;
    }
    get WEIGHT_SHIFT_STRENGTH() {
        return this.#WEIGHT_SHIFT_STRENGTH;
    }
    get WEIGHT_RANDOM_STRENGTH() {
        return this.#WEIGHT_RANDOM_STRENGTH;
    }
    get PROBABILITY_MUTATE_WEIGHT_SHIFT() {
        return this.#PROBABILITY_MUTATE_WEIGHT_SHIFT;
    }
    get PROBABILITY_MUTATE_WEIGHT_RANDOM() {
        return this.#PROBABILITY_MUTATE_WEIGHT_RANDOM;
    }

    #load(data: object) {}

    #reset() {
        this.#clients = [];
        for (let i = 0; i < this.#maxClients; i += 1) {
            const c: Client = new Client(new LSTM(this));
            this.#clients.push(c);
        }
    }

    printSpecies() {
        for (let i = 0; i < this.#species.length; i += 1) {
            console.log(this.#species[i].score, this.#species[i].size());
        }
    }

    evolve() {}
}
