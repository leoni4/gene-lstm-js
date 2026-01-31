import { LSTM } from './lstm.js';
import { GeneLSTM } from './gLstm.js';
import type { GeneOptions, ShortMemory, LstmOptions } from './types/index.js';

export class Genome {
    #glstm: GeneLSTM;
    #lstmArray: LSTM[];

    constructor(glstm: GeneLSTM, data?: GeneOptions) {
        this.#glstm = glstm;
        this.#lstmArray = [];
        if (data) {
            data.forEach(option => {
                this.#lstmArray.push(new LSTM(this.#glstm, option));
            });
        } else {
            this.#lstmArray.push(new LSTM(this.#glstm));
        }
    }

    get glstm() {
        return this.#glstm;
    }

    get lstmArray() {
        return this.#lstmArray;
    }

    distance(g2passed: Genome): number {
        let g1: Genome = this;
        let g2 = g2passed;

        if (g1.lstmArray.length < g2.lstmArray.length) {
            [g1, g2] = [g2, g1];
        }

        let excess = 0;
        let weightDiff = 0;
        let similar = 0;

        const maxLen = Math.max(g1.lstmArray.length, g2.lstmArray.length);

        for (let i = 0; i < maxLen; i++) {
            const block1 = g1.lstmArray[i];
            const block2 = g2.lstmArray[i];

            if (block1 && block2) {
                const w1 = block1.flattenWeights();
                const w2 = block2.flattenWeights();

                const len = Math.min(w1.length, w2.length);
                for (let j = 0; j < len; j++) {
                    weightDiff += Math.abs(w1[j] - w2[j]);
                }

                weightDiff /= len || 1;
                similar++;
            } else {
                excess++;
            }
        }

        weightDiff /= similar || 1;

        return this.#glstm.C1 * excess + this.#glstm.C2 * weightDiff;
    }

    mutate() {
        this.#lstmArray.forEach(lstm => {
            lstm.mutate();
        });
        if (this.#glstm.PROBABILITY_MUTATE_LSTM_BLOCK * this.#glstm.MUTATION_RATE > Math.random()) {
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

    static crossOver(g1: Genome, g2: Genome): Genome {
        const lstms1 = g1.lstmArray;
        const lstms2 = g2.lstmArray;
        const geneOptions: LstmOptions[] = [];

        const maxLength = Math.max(lstms1.length, lstms2.length);

        for (let i = 0; i < maxLength; i++) {
            const block1 = lstms1[i];
            const block2 = lstms2[i];

            if (block1 && block2) {
                const b1Model = block1.model();
                const b2Model = block2.model();
                geneOptions.push({
                    forgetGate: Genome.crossShortMemory(b1Model.forgetGate, b2Model.forgetGate),
                    potentialLongToRem: Genome.crossShortMemory(b1Model.potentialLongToRem, b2Model.potentialLongToRem),
                    potentialLongMemory: Genome.crossShortMemory(
                        b1Model.potentialLongMemory,
                        b2Model.potentialLongMemory,
                    ),
                    shortMemoryToRemember: Genome.crossShortMemory(
                        b1Model.shortMemoryToRemember,
                        b2Model.shortMemoryToRemember,
                    ),
                });
            } else if (block1 && Math.random() < 0.5) {
                geneOptions.push(block1.model());
            } else if (block2) {
                geneOptions.push(block2.model());
            }
        }

        return new Genome(g1.glstm, geneOptions);
    }

    static crossShortMemory(a: ShortMemory, b: ShortMemory): ShortMemory {
        return {
            weight1: Math.random() < 0.5 ? a.weight1 : b.weight1,
            weight2: Math.random() < 0.5 ? a.weight2 : b.weight2,
            bias: Math.random() < 0.5 ? a.bias : b.bias,
        };
    }
}
