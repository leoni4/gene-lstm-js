import { Client } from './client';
import { Species } from './species';
import { Genome } from './genome';

import type { GeneOptions } from './types/index';

interface GeneLSTMOptions {
    CP?: number;
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
    PROBABILITY_MUTATE_NEW_LSTM?: number;
    loadData?: GeneOptions;
}

export class GeneLSTM {
    #clients: Client[] = [];
    #species: Species[] = [];
    #maxClients: number;

    #CP: number;

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

    #PROBABILITY_MUTATE_NEW_LSTM: number;

    constructor(clients: number, options?: GeneLSTMOptions) {
        this.#maxClients = clients;

        this.#CP = options?.CP || 0.5;
        this.#SURVIVORS = options?.SURVIVORS || 0.8;
        this.#MUTATION_RATE = options?.MUTATION_RATE || 0.5;
        this.#BIAS_SHIFT_STRENGTH = options?.BIAS_SHIFT_STRENGTH || 0.5;
        this.#BIAS_RANDOM_STRENGTH = options?.BIAS_RANDOM_STRENGTH || 0.5;
        this.#WEIGHT_SHIFT_STRENGTH = options?.WEIGHT_SHIFT_STRENGTH || 0.5;
        this.#WEIGHT_RANDOM_STRENGTH = options?.WEIGHT_RANDOM_STRENGTH || 0.5;
        this.#PROBABILITY_MUTATE_BIAS_SHIFT = options?.PROBABILITY_MUTATE_BIAS_SHIFT || 0.5;
        this.#PROBABILITY_MUTATE_BIAS_RANDOM = options?.PROBABILITY_MUTATE_BIAS_RANDOM || 0.5;
        this.#PROBABILITY_MUTATE_WEIGHT_SHIFT = options?.PROBABILITY_MUTATE_WEIGHT_SHIFT || 0.5;
        this.#PROBABILITY_MUTATE_WEIGHT_RANDOM = options?.PROBABILITY_MUTATE_WEIGHT_RANDOM || 0.5;
        this.#PROBABILITY_MUTATE_NEW_LSTM = options?.PROBABILITY_MUTATE_NEW_LSTM || 0.5;

        this.#init(options?.loadData);
    }

    get CP() {
        return this.#CP;
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
    get PROBABILITY_MUTATE_NEW_LSTM() {
        return this.#PROBABILITY_MUTATE_NEW_LSTM;
    }

    get clients() {
        return this.#clients;
    }

    #emptyGenome() {
        return new Genome(this);
    }

    #initGenome(data: GeneOptions) {
        return new Genome(this, data);
    }

    #init(data?: GeneOptions) {
        let genome: Genome;
        if (data) {
            genome = this.#initGenome(data);
        } else {
            genome = this.#emptyGenome();
        }
        this.#clients = [];
        for (let i = 0; i < this.#maxClients; i += 1) {
            const c: Client = new Client(genome);
            if (i === 0) {
                this.#species.push(new Species(c));
            } else {
                this.#species[0].put(c, true);
            }
            this.#clients.push(c);
        }
    }

    printSpecies() {
        console.log('### Species:', this.#species.length);
        for (let i = 0; i < this.#species.length; i += 1) {
            console.log(this.#species[i].score, this.#species[i].size());
        }
        console.log('###');
    }

    #mutate() {
        this.#clients.forEach(client => {
            client.mutate();
        });
    }

    evolve(optimization = false, error?: number) {
        // if (this.#lastError === error) {
        //     this.#sameErrorEpoch += 1;
        // } else {
        //     this.#sameErrorEpoch = 0;
        // }
        // this.#lastError = error;
        // this.#evolveCounts++;
        // this.#optimization = optimization || this.#evolveCounts % 10 === 0;
        // this.#normalizeScore();
        // this.#genSpecies();
        // this.#kill();
        // this.#removeExtinct();
        // this.#reproduce();
        this.#mutate();
        // for (let i = 0; i < this.#clients.length; i += 1) {
        //     this.#clients[i].generateCalculator();
        // }
    }
}
