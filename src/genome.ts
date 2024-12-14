import { LSTM } from './lstm';
import { GeneLSTM } from './gLstm';

export class Genome {
    #glstm: GeneLSTM;
    #lstmArray: LSTM[];

    constructor(glstm: GeneLSTM) {
        this.#glstm = glstm;
        this.#lstmArray = [new LSTM(this.#glstm)];
    }
    get glstm() {
        return this.#glstm;
    }

    distance(genome: Genome): number {
        throw new Error('Method not implemented.');
    }

    mutate() {
        this.#lstmArray.forEach(lstm => {
            lstm.mutate();
        });
        if (this.#glstm.PROBABILITY_MUTATE_NEW_LSTM > Math.random()) {
            if (Math.random() > 0.5 && this.#lstmArray.length > 1) {
                this.#lstmArray.pop();
            } else {
                this.#lstmArray.push(new LSTM(this.#glstm));
            }
        }
    }

    calculate(input: number[]) {
        let inputPassed = input;
        this.#lstmArray.forEach((lstm, i) => {
            const localClc = lstm.calculate(inputPassed, this.#lstmArray.length > 1 && this.#lstmArray.length > i + 1);
            inputPassed = localClc;
        });
        return inputPassed;
    }
}
