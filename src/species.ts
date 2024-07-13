export class Species {
    #score: number = 0;

    set score(score: number) {
        this.#score = score;
    }
    get score() {
        return this.#score;
    }

    size() {
        return 0;
    }
}
