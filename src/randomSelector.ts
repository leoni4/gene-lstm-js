import { Species } from './species.js';

export class RandomSelector {
    private _objects: Array<Species> = [];
    private _survivors: number;
    private _totalScore = 0;

    constructor(survivors: number) {
        this._survivors = survivors;
    }

    get objects(): Array<Species> {
        return this._objects;
    }

    get totalScore(): number {
        return this._totalScore;
    }

    add(object: Species) {
        this._objects.push(object);
        this._totalScore += object.score;
        this._objects.sort((a, b) => {
            return a.score > b.score ? -1 : 1;
        });
    }

    random(): Species {
        const randomScore = Math.abs(Math.random() * this._totalScore * this._survivors);
        let scoreIndex = 0;
        for (let i = 0; i < this._objects.length; i++) {
            scoreIndex += Math.abs(this._objects[i].score);
            if (scoreIndex >= randomScore) {
                return this._objects[i];
            }
        }

        return this._objects[0];
    }

    reset() {
        this._objects = [];
        this._totalScore = 0;
    }
}
