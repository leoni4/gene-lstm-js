import { LSTM } from './lstm.js';
import { GeneLSTM } from './gLstm.js';
import type { GeneOptions, GateUnitOptions, LstmOptions, SeqInput } from './types/index.js';

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

    distance(g2passed: Genome, cache?: Map<LSTM, number[]>): number {
        const flatten = (lstm: LSTM): number[] => {
            if (!cache) return lstm.flattenWeights();
            let w = cache.get(lstm);
            if (!w) {
                w = lstm.flattenWeights();
                cache.set(lstm, w);
            }

            return w;
        };

        let g1: Genome = this;
        let g2 = g2passed;

        if (g1.lstmArray.length < g2.lstmArray.length) {
            [g1, g2] = [g2, g1];
        }

        let excess = 0;
        let weightDiffSum = 0;
        let similar = 0;

        const maxLen = Math.max(g1.lstmArray.length, g2.lstmArray.length);

        for (let i = 0; i < maxLen; i++) {
            const block1 = g1.lstmArray[i];
            const block2 = g2.lstmArray[i];

            if (block1 && block2) {
                const w1 = flatten(block1);
                const w2 = flatten(block2);

                const len = Math.min(w1.length, w2.length);
                let blockDiffSum = 0;
                for (let j = 0; j < len; j++) {
                    blockDiffSum += Math.abs(w1[j] - w2[j]);
                }

                const blockDiff = blockDiffSum / (len || 1);
                weightDiffSum += blockDiff;
                similar++;
            } else {
                excess++;
            }
        }

        const weightDiff = weightDiffSum / (similar || 1);

        return this._glstm.C1 * excess + this._glstm.C2 * weightDiff;
    }

    private _createSleepingBlock(): LSTM {
        const cfg = this._glstm.sleepingBlockConfig;
        const eps = cfg.epsilon;

        const randSmall = () => Math.random() * 2 * eps - eps;

        const inputN = this._glstm.INPUT_FEATURES || 1;
        const outputDim = this._glstm.OUTPUT_DIM || 1;

        const makeUnit = (bias: number): GateUnitOptions => ({
            weight1: randSmall(),
            weight2: randSmall(),
            bias,
            weightIn: new Array(inputN).fill(0).map(randSmall),
        });

        const H = 1; 

        const options: LstmOptions = {
            hiddenSize: H,

            forgetGate: new Array(H).fill(0).map(() => makeUnit(cfg.forgetBias)),
            potentialLongToRem: new Array(H).fill(0).map(() => makeUnit(cfg.inputBias)),
            potentialLongMemory: new Array(H).fill(0).map(() => makeUnit(cfg.candidateBias)),
            shortMemoryToRemember: new Array(H).fill(0).map(() => makeUnit(cfg.outputBias)),

            readoutW: new Array(outputDim).fill(0).map(() => new Array(H).fill(0)),
            readoutB: new Array(outputDim).fill(0),

            alpha: cfg.initialAlpha,
            outputDim,
            outputActivation: this._glstm.OUTPUT_ACTIVATION,
        };

        return new LSTM(this._glstm, options);
    }

    mutate() {
        this._lstmArray.forEach(lstm => {
            lstm.mutate();
        });

        const pressure = this._glstm.getMutationPressure();
        const structProb = this._glstm.PROBABILITY_MUTATE_LSTM_BLOCK * this._glstm.MUTATION_RATE * pressure.topology;

        if (structProb > Math.random()) {
            const scaledRemoveProb = this._glstm.PROBABILITY_REMOVE_BLOCK * pressure.topology;
            const shouldRemove = Math.random() < Math.min(scaledRemoveProb, 0.9);

            if (shouldRemove && this._lstmArray.length > 1) {
                const removeFromEnd = Math.random() < 0.5;
                if (removeFromEnd) {
                    this._lstmArray.pop();
                } else {
                    this._lstmArray.shift();
                }
            } else {
                const shouldAppend = Math.random() < this._glstm.PROBABILITY_ADD_BLOCK_APPEND;

                if (this._lstmArray.length < this._glstm.MAX_LAYERS) {
                    if (shouldAppend) {
                        this._lstmArray.push(this._createSleepingBlock());
                    } else {
                        this._lstmArray.unshift(this._createSleepingBlock());
                    }
                }
            }
        }
    }

    calculate(input: SeqInput): number[] {
        let inputPassed = input;

        this._lstmArray.forEach((lstm, i) => {
            const fullSeq = this._lstmArray.length > 1 && this._lstmArray.length > i + 1;
            inputPassed = lstm.calculate(inputPassed, fullSeq);
        });

        return inputPassed as number[];
    }

    static crossGateUnit(a: GateUnitOptions, b: GateUnitOptions): GateUnitOptions {
        const out: GateUnitOptions = {
            weight1: Math.random() < 0.5 ? a.weight1 : b.weight1,
            weight2: Math.random() < 0.5 ? a.weight2 : b.weight2,
            bias: Math.random() < 0.5 ? a.bias : b.bias,
        };

        const wa = a.weightIn;
        const wb = b.weightIn;

        if (wa && wb) {
            const n = Math.min(wa.length, wb.length);
            const w: number[] = new Array(n);
            for (let i = 0; i < n; i++) w[i] = Math.random() < 0.5 ? wa[i] : wb[i];
            out.weightIn = w;
        } else if (wa) {
            out.weightIn = [...wa];
        } else if (wb) {
            out.weightIn = [...wb];
        }

        return out;
    }

    static crossOver(g1: Genome, g2: Genome): Genome {
        const lstms1 = g1.lstmArray;
        const lstms2 = g2.lstmArray;
        const geneOptions: LstmOptions[] = [];

        const glstm = g1.glstm;
        const childSize = (n1: number, n2: number, limit: number): number =>
            glstm.crossoverStructure === 'fitter' ? n1 : Math.min(Math.max(n1, n2), Math.max(limit, Math.min(n1, n2)));

        const childLength = childSize(lstms1.length, lstms2.length, glstm.MAX_LAYERS);

        for (let i = 0; i < childLength; i++) {
            const block1 = lstms1[i];
            const block2 = lstms2[i];

            if (block1 && block2) {
                const a = block1.model();
                const b = block2.model();

                const H1 = a.hiddenSize ?? 1;
                const H2 = b.hiddenSize ?? 1;
                const H = childSize(H1, H2, glstm.MAX_UNITS_PER_LAYER);
                const minH = Math.min(H1, H2);

                const crossGateArray = (ga: GateUnitOptions[], gb: GateUnitOptions[]) => {
                    const out: GateUnitOptions[] = [];
                    for (let k = 0; k < H; k++) {
                        if (k < minH) {
                            out.push(Genome.crossGateUnit(ga[k], gb[k]));
                        } else {
                            const hasA = k < ga.length;
                            const hasB = k < gb.length;
                            if (hasA && hasB) out.push(Math.random() < 0.5 ? ga[k] : gb[k]);
                            else if (hasA) out.push(ga[k]);
                            else out.push(gb[k]);
                        }
                    }

                    return out;
                };

                const aReadoutW = a.readoutW as number[][];
                const bReadoutW = b.readoutW as number[][];
                const aReadoutB = a.readoutB as number[];
                const bReadoutB = b.readoutB as number[];

                const outputDim = Math.max(aReadoutW?.length ?? 1, bReadoutW?.length ?? 1);
                const readoutW: number[][] = [];

                for (let j = 0; j < outputDim; j++) {
                    const rowA = aReadoutW?.[j];
                    const rowB = bReadoutW?.[j];
                    const row: number[] = new Array(H);

                    for (let k = 0; k < H; k++) {
                        const wa = rowA?.[k];
                        const wb = rowB?.[k];
                        row[k] =
                            wa !== undefined && wb !== undefined ? (Math.random() < 0.5 ? wa : wb) : (wa ?? wb ?? 0);
                    }
                    readoutW.push(row);
                }

                const readoutB: number[] = [];
                for (let j = 0; j < outputDim; j++) {
                    const ba = aReadoutB?.[j];
                    const bb = bReadoutB?.[j];
                    readoutB.push(
                        ba !== undefined && bb !== undefined ? (Math.random() < 0.5 ? ba : bb) : (ba ?? bb ?? 0),
                    );
                }

                geneOptions.push({
                    hiddenSize: H,

                    forgetGate: crossGateArray(a.forgetGate, b.forgetGate),
                    potentialLongToRem: crossGateArray(a.potentialLongToRem, b.potentialLongToRem),
                    potentialLongMemory: crossGateArray(a.potentialLongMemory, b.potentialLongMemory),
                    shortMemoryToRemember: crossGateArray(a.shortMemoryToRemember, b.shortMemoryToRemember),

                    readoutW,
                    readoutB,

                    alpha: Math.random() < 0.5 ? a.alpha : b.alpha,
                });
            } else {
                const use1 = block1 && (!block2 || Math.random() < 0.75);
                const copied = use1 ? block1!.model() : block2!.model();

                const H = childSize(copied.hiddenSize, 0, glstm.MAX_UNITS_PER_LAYER);
                if (H >= copied.hiddenSize) {
                    geneOptions.push(copied);
                } else {
                    geneOptions.push({
                        ...copied,
                        hiddenSize: H,
                        forgetGate: copied.forgetGate.slice(0, H),
                        potentialLongToRem: copied.potentialLongToRem.slice(0, H),
                        potentialLongMemory: copied.potentialLongMemory.slice(0, H),
                        shortMemoryToRemember: copied.shortMemoryToRemember.slice(0, H),
                        readoutW: (copied.readoutW as number[][]).map(row => row.slice(0, H)),
                    });
                }
            }
        }

        return new Genome(g1.glstm, geneOptions);
    }
}
