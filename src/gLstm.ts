import { Client } from './client.js';
import { Species } from './species.js';
import { Genome } from './genome.js';
import { RandomSelector } from './randomSelector.js';

import type { GeneOptions } from './types/index.js';

interface GeneLSTMOptions {
    CP?: number;
    C1?: number;
    C2?: number;
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
    PROBABILITY_MUTATE_LSTM_BLOCK?: number;
    loadData?: GeneOptions;
}

export class GeneLSTM {
    #clients: Client[] = [];
    #species: Species[] = [];
    #maxClients: number;
    #sameErrorEpoch: number = 0;
    #lastError?: number;

    #CP: number;
    #C1: number;
    #C2: number;

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

    #PROBABILITY_MUTATE_LSTM_BLOCK: number;

    #evolveCounts = 0;
    #optimization = false;

    constructor(clients: number, options?: GeneLSTMOptions) {
        this.#maxClients = clients;

        this.#CP = options?.CP ?? 1.0;
        this.#C1 = options?.C1 ?? 1.0;
        this.#C2 = options?.C2 ?? 0.4;
        this.#SURVIVORS = options?.SURVIVORS ?? 0.8;
        this.#MUTATION_RATE = options?.MUTATION_RATE ?? 0.05;
        this.#BIAS_SHIFT_STRENGTH = options?.BIAS_SHIFT_STRENGTH ?? 0.2;
        this.#BIAS_RANDOM_STRENGTH = options?.BIAS_RANDOM_STRENGTH ?? 1.0;
        this.#WEIGHT_SHIFT_STRENGTH = options?.WEIGHT_SHIFT_STRENGTH ?? 0.2;
        this.#WEIGHT_RANDOM_STRENGTH = options?.WEIGHT_RANDOM_STRENGTH ?? 1.0;

        this.#PROBABILITY_MUTATE_BIAS_SHIFT = options?.PROBABILITY_MUTATE_BIAS_SHIFT ?? 0.8;
        this.#PROBABILITY_MUTATE_BIAS_RANDOM = options?.PROBABILITY_MUTATE_BIAS_RANDOM ?? 0.1;
        this.#PROBABILITY_MUTATE_WEIGHT_SHIFT = options?.PROBABILITY_MUTATE_WEIGHT_SHIFT ?? 0.8;
        this.#PROBABILITY_MUTATE_WEIGHT_RANDOM = options?.PROBABILITY_MUTATE_WEIGHT_RANDOM ?? 0.1;
        this.#PROBABILITY_MUTATE_LSTM_BLOCK = options?.PROBABILITY_MUTATE_LSTM_BLOCK ?? 0.05;

        this.#init(options?.loadData);
    }

    get CP() {
        return this.#CP;
    }

    get C1() {
        return this.#C1;
    }

    get C2() {
        return this.#C2;
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
    get PROBABILITY_MUTATE_LSTM_BLOCK() {
        return this.#PROBABILITY_MUTATE_LSTM_BLOCK;
    }

    get clients() {
        return this.#clients;
    }

    emptyGenome() {
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
            genome = this.emptyGenome();
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

    model() {
        return this.#clients[0].genome.lstmArray.map(l => l.model());
    }

    printSpecies() {
        console.log(
            '### Species:',
            this.#species.length,
            '# Complecity:',
            this.#clients.reduce((acc, c) => c.genome.lstmArray.length + acc, 0),
        );
        for (let i = 0; i < this.#species.length; i += 1) {
            console.log('#', this.#species[i].score, this.#species[i].size());
        }
        console.log('###');
    }

    evolve(optimization = false, error?: number) {
        if (this.#lastError === error) {
            this.#sameErrorEpoch += 1;
        } else {
            this.#sameErrorEpoch = 0;
        }
        this.#lastError = error;
        this.#evolveCounts++;
        this.#optimization = optimization || this.#evolveCounts % 10 === 0;
        this.#normalizeScore();
        this.#genSpecies();
        this.#kill();
        this.#removeExtinct();
        this.#reproduce();
        this.#mutate();
    }

    #normalizeScore() {
        let maxScore = 0;
        const bestScoreSet = [];
        let minScore = Infinity;

        for (let i = 0; i < this.#clients.length; i += 1) {
            const item = this.#clients[i];
            item.bestScore = false;
            maxScore = item.score > maxScore ? item.score : maxScore;
            minScore = item.score < minScore ? item.score : minScore;
        }

        for (let i = 0; i < this.#clients.length; i += 1) {
            const item = this.#clients[i];
            if (item.score === maxScore) {
                bestScoreSet.push(i);
                item.bestScore = true;
                item.score = 1;
            } else if (item.score === minScore) {
                item.score = 0;
            } else {
                item.score = (item.score - minScore) / (maxScore - minScore);
            }
        }

        if (bestScoreSet.length > 1) {
            bestScoreSet.forEach((i, index) => {
                if (index === 0) return;
                this.#clients[i].bestScore = false;
            });
        }

        this.#clients.sort((a, b) => {
            return a.score > b.score ? -1 : 1;
        });

        const cof = this.#optimization ? 0.1 : 0.01;

        this.#clients.forEach(item => {
            const allLayers = item.genome.lstmArray.length;
            item.score -= (Math.sqrt(Math.sqrt(allLayers)) - 1) * cof;
        });
    }

    #genSpecies() {
        for (let i = 0; i < this.#species.length; i += 1) {
            this.#species[i].reset();
        }
        for (let i = 0; i < this.#clients.length; i += 1) {
            const c = this.#clients[i];
            if (c.species !== null) {
                continue;
            }

            let found = false;
            for (let k = 0; k < this.#species.length; k += 1) {
                const s = this.#species[k];
                if (s.put(c)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                this.#species.push(new Species(c));
            }
        }
        for (let i = 0; i < this.#species.length; i += 1) {
            this.#species[i].evaluateScore();
        }
    }

    #kill() {
        for (let i = 0; i < this.#species.length; i += 1) {
            this.#species[i].kill(this.#SURVIVORS);
        }
    }

    #removeExtinct() {
        for (let i = this.#species.length - 1; i >= 0; i--) {
            if (this.#species[i].size() <= 1 && !this.#species[i].clients[0]?.bestScore && this.#species.length > 1) {
                this.#species[i].goExtinct();
                this.#species.splice(i, 1);
            }
        }
    }

    #reproduce() {
        const selector = new RandomSelector(this.#SURVIVORS);
        for (let i = 0; i < this.#species.length; i += 1) {
            selector.add(this.#species[i]);
        }
        for (let i = 0; i < this.#clients.length; i += 1) {
            const c = this.#clients[i];
            if (c.species === null) {
                const s = selector.random();
                c.genome = s.breed();
                s.put(c, true);
            }
        }
        selector.reset();
    }

    #mutate() {
        this.#clients.forEach(client => {
            client.mutate();
        });
    }
}
