import { Client } from './client.js';
import { Genome } from './genome.js';

export class Species {
    private _representative: Client;
    private _clients: Array<Client> = [];
    private _score: number = 0;

    constructor(client: Client) {
        this._representative = client;
        this._representative.species = this;
        this._clients.push(client);
    }

    get score(): number {
        return this._score;
    }

    get clients(): Array<Client> {
        return this._clients;
    }

    put(client: Client, force = false): boolean {
        if (force || client.distance(this._representative) < this._representative.genome.glstm.CP) {
            client.species = this;
            this._clients.push(client);

            return true;
        }

        return false;
    }

    size() {
        return this._clients.length;
    }

    private _getRandomClient(): Client {
        return this._clients[Math.floor(Math.random() * this._clients.length)];
    }

    goExtinct() {
        for (let i = 0; i < this._clients.length; i += 1) {
            const c = this._clients[i];
            c.species = null;
        }
        this._clients = [];
    }

    reset() {
        this._representative = this._getRandomClient();
        this.goExtinct();
        this._clients.push(this._representative);
        this._representative.species = this;
        this._score = 0;
    }

    evaluateScore() {
        let value = 0;
        for (let i = 0; i < this._clients.length; i += 1) {
            const c = this._clients[i];
            value += c.score;
        }
        this._score = value / this._clients.length;
    }

    kill(survivors = 0.5) {
        this._clients.sort((a, b) => {
            return a.score > b.score ? -1 : 1;
        });

        const elems = survivors * this._clients.length;
        for (let i = this._clients.length - 1; i > elems; i -= 1) {
            if (this._clients[i].bestScore) {
                continue;
            }
            this._clients[i].species = null;
            this._clients.splice(i, 1);
        }
    }

    breed(): Genome {
        const c1 = this._getRandomClient();
        const c2 = this._getRandomClient();
        if (c1.score >= c2.score) {
            return Genome.crossOver(c1.genome, c2.genome);
        } else {
            return Genome.crossOver(c2.genome, c1.genome);
        }
    }
}
