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
    private _clients: Client[] = [];
    private _species: Species[] = [];
    private _maxClients: number;

    private _CP: number;
    private _C1: number;
    private _C2: number;

    private _SURVIVORS: number;
    private _MUTATION_RATE: number;

    private _BIAS_SHIFT_STRENGTH: number;
    private _BIAS_RANDOM_STRENGTH: number;
    private _PROBABILITY_MUTATE_BIAS_SHIFT: number;
    private _PROBABILITY_MUTATE_BIAS_RANDOM: number;

    private _WEIGHT_SHIFT_STRENGTH: number;
    private _WEIGHT_RANDOM_STRENGTH: number;
    private _PROBABILITY_MUTATE_WEIGHT_SHIFT: number;
    private _PROBABILITY_MUTATE_WEIGHT_RANDOM: number;

    private _PROBABILITY_MUTATE_LSTM_BLOCK: number;

    private _evolveCounts = 0;
    private _optimization = false;

    constructor(clients: number, options?: GeneLSTMOptions) {
        this._maxClients = clients;

        this._CP = options?.CP ?? 1.0;
        this._C1 = options?.C1 ?? 1.0;
        this._C2 = options?.C2 ?? 0.4;
        this._SURVIVORS = options?.SURVIVORS ?? 0.8;
        this._MUTATION_RATE = options?.MUTATION_RATE ?? 0.05;
        this._BIAS_SHIFT_STRENGTH = options?.BIAS_SHIFT_STRENGTH ?? 0.2;
        this._BIAS_RANDOM_STRENGTH = options?.BIAS_RANDOM_STRENGTH ?? 1.0;
        this._WEIGHT_SHIFT_STRENGTH = options?.WEIGHT_SHIFT_STRENGTH ?? 0.2;
        this._WEIGHT_RANDOM_STRENGTH = options?.WEIGHT_RANDOM_STRENGTH ?? 1.0;

        this._PROBABILITY_MUTATE_BIAS_SHIFT = options?.PROBABILITY_MUTATE_BIAS_SHIFT ?? 0.8;
        this._PROBABILITY_MUTATE_BIAS_RANDOM = options?.PROBABILITY_MUTATE_BIAS_RANDOM ?? 0.1;
        this._PROBABILITY_MUTATE_WEIGHT_SHIFT = options?.PROBABILITY_MUTATE_WEIGHT_SHIFT ?? 0.8;
        this._PROBABILITY_MUTATE_WEIGHT_RANDOM = options?.PROBABILITY_MUTATE_WEIGHT_RANDOM ?? 0.1;
        this._PROBABILITY_MUTATE_LSTM_BLOCK = options?.PROBABILITY_MUTATE_LSTM_BLOCK ?? 0.05;

        this._init(options?.loadData);
    }

    get CP() {
        return this._CP;
    }

    get C1() {
        return this._C1;
    }

    get C2() {
        return this._C2;
    }

    get SURVIVORS() {
        return this._SURVIVORS;
    }
    get MUTATION_RATE() {
        return this._MUTATION_RATE;
    }
    get BIAS_SHIFT_STRENGTH() {
        return this._BIAS_SHIFT_STRENGTH;
    }
    get BIAS_RANDOM_STRENGTH() {
        return this._BIAS_RANDOM_STRENGTH;
    }
    get PROBABILITY_MUTATE_BIAS_SHIFT() {
        return this._PROBABILITY_MUTATE_BIAS_SHIFT;
    }
    get PROBABILITY_MUTATE_BIAS_RANDOM() {
        return this._PROBABILITY_MUTATE_BIAS_RANDOM;
    }
    get WEIGHT_SHIFT_STRENGTH() {
        return this._WEIGHT_SHIFT_STRENGTH;
    }
    get WEIGHT_RANDOM_STRENGTH() {
        return this._WEIGHT_RANDOM_STRENGTH;
    }
    get PROBABILITY_MUTATE_WEIGHT_SHIFT() {
        return this._PROBABILITY_MUTATE_WEIGHT_SHIFT;
    }
    get PROBABILITY_MUTATE_WEIGHT_RANDOM() {
        return this._PROBABILITY_MUTATE_WEIGHT_RANDOM;
    }
    get PROBABILITY_MUTATE_LSTM_BLOCK() {
        return this._PROBABILITY_MUTATE_LSTM_BLOCK;
    }

    get clients() {
        return this._clients;
    }

    emptyGenome() {
        return new Genome(this);
    }

    private _initGenome(data: GeneOptions) {
        return new Genome(this, data);
    }

    private _init(data?: GeneOptions) {
        let genome: Genome;
        if (data) {
            genome = this._initGenome(data);
        } else {
            genome = this.emptyGenome();
        }
        this._clients = [];
        for (let i = 0; i < this._maxClients; i += 1) {
            const c: Client = new Client(genome);
            if (i === 0) {
                this._species.push(new Species(c));
            } else {
                this._species[0].put(c, true);
            }
            this._clients.push(c);
        }
    }

    model() {
        return this._clients[0].genome.lstmArray.map(l => l.model());
    }

    printSpecies() {
        console.log(
            '### Species:',
            this._species.length,
            '# Complecity:',
            this._clients.reduce((acc, c) => c.genome.lstmArray.length + acc, 0),
        );
        for (let i = 0; i < this._species.length; i += 1) {
            console.log('#', this._species[i].score, this._species[i].size());
        }
        console.log('###');
    }

    evolve(optimization = false) {
        this._evolveCounts++;
        this._optimization = optimization || this._evolveCounts % 10 === 0;
        this._normalizeScore();
        this._genSpecies();
        this._kill();
        this._removeExtinct();
        this._reproduce();
        this._mutate();
    }

    private _normalizeScore() {
        let maxScore = 0;
        const bestScoreSet = [];
        let minScore = Infinity;

        for (let i = 0; i < this._clients.length; i += 1) {
            const item = this._clients[i];
            item.bestScore = false;
            maxScore = item.score > maxScore ? item.score : maxScore;
            minScore = item.score < minScore ? item.score : minScore;
        }

        for (let i = 0; i < this._clients.length; i += 1) {
            const item = this._clients[i];
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
                this._clients[i].bestScore = false;
            });
        }

        this._clients.sort((a, b) => {
            return a.score > b.score ? -1 : 1;
        });

        const cof = this._optimization ? 0.1 : 0.01;

        this._clients.forEach(item => {
            const allLayers = item.genome.lstmArray.length;
            item.score -= (Math.sqrt(Math.sqrt(allLayers)) - 1) * cof;
        });
    }

    private _genSpecies() {
        for (let i = 0; i < this._species.length; i += 1) {
            this._species[i].reset();
        }
        for (let i = 0; i < this._clients.length; i += 1) {
            const c = this._clients[i];
            if (c.species !== null) {
                continue;
            }

            let found = false;
            for (let k = 0; k < this._species.length; k += 1) {
                const s = this._species[k];
                if (s.put(c)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                this._species.push(new Species(c));
            }
        }
        for (let i = 0; i < this._species.length; i += 1) {
            this._species[i].evaluateScore();
        }
    }

    private _kill() {
        for (let i = 0; i < this._species.length; i += 1) {
            this._species[i].kill(this._SURVIVORS);
        }
    }

    private _removeExtinct() {
        for (let i = this._species.length - 1; i >= 0; i--) {
            if (this._species[i].size() <= 1 && !this._species[i].clients[0]?.bestScore && this._species.length > 1) {
                this._species[i].goExtinct();
                this._species.splice(i, 1);
            }
        }
    }

    private _reproduce() {
        const selector = new RandomSelector(this._SURVIVORS);
        for (let i = 0; i < this._species.length; i += 1) {
            selector.add(this._species[i]);
        }
        for (let i = 0; i < this._clients.length; i += 1) {
            const c = this._clients[i];
            if (c.species === null) {
                const s = selector.random();
                c.genome = s.breed();
                s.put(c, true);
            }
        }
        selector.reset();
    }

    private _mutate() {
        this._clients.forEach(client => {
            client.mutate();
        });
    }
}
