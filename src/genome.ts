import { LSTM } from './lstm.js';
import { GeneLSTM } from './gLstm.js';
import type { GeneOptions, ShortMemory, LstmOptions } from './types/index.js';

export class Genome {
    private _glstm: GeneLSTM;
    private _lstmArray: LSTM[];

    constructor(glstm: GeneLSTM, data?: GeneOptions) {
        this._glstm = glstm;
        this._lstmArray = [];
        if (data) {
            data.forEach(option => {
                this._lstmArray.push(new LSTM(this._glstm, option));
            });
        } else {
            this._lstmArray.push(new LSTM(this._glstm));
        }
    }

    get glstm() {
        return this._glstm;
    }

    get lstmArray() {
        return this._lstmArray;
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

        return this._glstm.C1 * excess + this._glstm.C2 * weightDiff;
    }

    mutate() {
        this._lstmArray.forEach(lstm => {
            lstm.mutate();
        });
        if (this._glstm.PROBABILITY_MUTATE_LSTM_BLOCK * this._glstm.MUTATION_RATE > Math.random()) {
            if (Math.random() > 0.5 && this._lstmArray.length > 1) {
                this._lstmArray.pop();
            } else {
                this._lstmArray.push(new LSTM(this._glstm));
            }
        }
    }

    calculate(input: number[]) {
        let inputPassed = input;
        this._lstmArray.forEach((lstm, i) => {
            const localClc = lstm.calculate(inputPassed, this._lstmArray.length > 1 && this._lstmArray.length > i + 1);
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
