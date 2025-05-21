import { Species } from './species';

export class RandomSelector {
    #objects: Array<Species> = [];
    #survivors: number;
    #totalScore = 0;

    constructor(survivors: number) {
        this.#survivors = survivors;
    }

    get objects(): Array<Species> {
        return this.#objects;
    }

    get totalScore(): number {
        return this.#totalScore;
    }

    add(object: Species) {
        this.#objects.push(object);
        this.#totalScore += object.score;
        this.#objects.sort((a, b) => {
            return a.score > b.score ? -1 : 1;
        });
    }

    random(): Species {
        const randomScore = Math.abs(Math.random() * this.#totalScore * this.#survivors);
        let scoreIndex = 0;
        for (let i = 0; i < this.#objects.length; i++) {
            scoreIndex += Math.abs(this.#objects[i].score);
            if (scoreIndex >= randomScore) {
                return this.#objects[i];
            }
        }
        return this.#objects[0];
    }

    reset() {
        this.#objects = [];
        this.#totalScore = 0;
    }
}
