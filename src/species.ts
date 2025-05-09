import { Client } from './client';

export class Species {
    #representative: Client;
    #clients: Array<Client> = [];
    #score: number = 0;

    constructor(client: Client) {
        this.#representative = client;
        this.#representative.species = this;
        this.#clients.push(client);
    }

    get score(): number {
        return this.#score;
    }

    get clients(): Array<Client> {
        return this.#clients;
    }

    put(client: Client, force = false): boolean {
        if (force || client.distance(this.#representative) < this.#representative.genome.glstm.CP) {
            client.species = this;
            this.#clients.push(client);
            return true;
        }
        return false;
    }

    size() {
        return this.#clients.length;
    }
}
