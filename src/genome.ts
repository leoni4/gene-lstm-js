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

    distance(g2passed: Genome): number {
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
                const w1 = block1.flattenWeights();
                const w2 = block2.flattenWeights();

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

    /**
     * Creates a "sleeping block" - a new LSTM block initialized to be nearly transparent
     * to preserve the parent's learned behavior during structural mutations
     */
    private _createSleepingBlock(): LSTM {
        const cfg = this._glstm.sleepingBlockConfig;
        const eps = cfg.epsilon;

        const randSmall = () => Math.random() * 2 * eps - eps;

        const inputN = this._glstm.INPUT_FEATURES || 1;

        const makeUnit = (bias: number): GateUnitOptions => ({
            weight1: randSmall(),
            weight2: randSmall(),
            bias,
            weightIn: new Array(inputN).fill(0).map(randSmall),
        });

        const H = 1; // sleeping block starts minimal & non-destructive

        const options: LstmOptions = {
            hiddenSize: H,

            forgetGate: new Array(H).fill(0).map(() => makeUnit(cfg.forgetBias)),
            potentialLongToRem: new Array(H).fill(0).map(() => makeUnit(cfg.inputBias)),
            potentialLongMemory: new Array(H).fill(0).map(() => makeUnit(cfg.candidateBias)),
            shortMemoryToRemember: new Array(H).fill(0).map(() => makeUnit(cfg.outputBias)),

            // new unit should initially have ~0 influence on final output
            readoutW: new Array(H).fill(0),
            readoutB: 0,

            alpha: cfg.initialAlpha,
        };

        return new LSTM(this._glstm, options);
    }

    /**
     * Structural mutation with directional bias:
     * - 92% chance to APPEND (add to end) - preserves learned representations
     * - 8% chance to PREPEND (add to beginning) - explores new input transformations
     * - 10% chance to REMOVE (if depth > 1)
     *
     * Mutation pressure scaling:
     * - Topology pressure scales the probability of structural mutations (add/remove blocks)
     */
    mutate() {
        // Mutate parameters of existing blocks
        this._lstmArray.forEach(lstm => {
            lstm.mutate();
        });

        // Apply topology mutation pressure to structural mutations
        const pressure = this._glstm.getMutationPressure();
        const structProb = this._glstm.PROBABILITY_MUTATE_LSTM_BLOCK * this._glstm.MUTATION_RATE * pressure.topology;

        if (structProb > Math.random()) {
            // Decide whether to add or remove (remove probability also scaled by topology pressure)
            const scaledRemoveProb = this._glstm.PROBABILITY_REMOVE_BLOCK * pressure.topology;
            const shouldRemove = Math.random() < Math.min(scaledRemoveProb, 0.9); // Cap at 90% to avoid too aggressive pruning

            if (shouldRemove && this._lstmArray.length > 1) {
                // Remove block from either end
                const removeFromEnd = Math.random() < 0.5;
                if (removeFromEnd) {
                    this._lstmArray.pop();
                } else {
                    this._lstmArray.shift();
                }
            } else {
                // Add sleeping block
                const shouldAppend = Math.random() < this._glstm.PROBABILITY_ADD_BLOCK_APPEND;

                if (shouldAppend) {
                    // APPEND: add to end (90-95% of additions)
                    this._lstmArray.push(this._createSleepingBlock());
                } else {
                    // PREPEND: add to beginning (5-10% of additions)
                    this._lstmArray.unshift(this._createSleepingBlock());
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

        const maxLength = Math.max(lstms1.length, lstms2.length);

        for (let i = 0; i < maxLength; i++) {
            const block1 = lstms1[i];
            const block2 = lstms2[i];

            if (block1 && block2) {
                const a = block1.model();
                const b = block2.model();

                const H1 = a.hiddenSize ?? 1;
                const H2 = b.hiddenSize ?? 1;
                const H = Math.max(H1, H2);
                const minH = Math.min(H1, H2);

                const crossGateArray = (ga: GateUnitOptions[], gb: GateUnitOptions[]) => {
                    const out: GateUnitOptions[] = [];
                    for (let k = 0; k < H; k++) {
                        if (k < minH) {
                            out.push(Genome.crossGateUnit(ga[k], gb[k]));
                        } else {
                            // excess unit: inherit from whichever parent has it
                            const hasA = k < ga.length;
                            const hasB = k < gb.length;
                            if (hasA && hasB) out.push(Math.random() < 0.5 ? ga[k] : gb[k]);
                            else if (hasA) out.push(ga[k]);
                            else out.push(gb[k]);
                        }
                    }

                    return out;
                };

                const readoutW: number[] = new Array(H);
                for (let k = 0; k < H; k++) {
                    const wa = a.readoutW?.[k];
                    const wb = b.readoutW?.[k];
                    readoutW[k] =
                        wa !== undefined && wb !== undefined ? (Math.random() < 0.5 ? wa : wb) : (wa ?? wb ?? 0);
                }

                geneOptions.push({
                    hiddenSize: H,

                    forgetGate: crossGateArray(a.forgetGate, b.forgetGate),
                    potentialLongToRem: crossGateArray(a.potentialLongToRem, b.potentialLongToRem),
                    potentialLongMemory: crossGateArray(a.potentialLongMemory, b.potentialLongMemory),
                    shortMemoryToRemember: crossGateArray(a.shortMemoryToRemember, b.shortMemoryToRemember),

                    readoutW,
                    readoutB: Math.random() < 0.5 ? (a.readoutB ?? 0) : (b.readoutB ?? 0),

                    alpha: Math.random() < 0.5 ? a.alpha : b.alpha,
                });
            } else {
                // Only one parent has this block
                const use1 = block1 && (!block2 || Math.random() < 0.75);
                geneOptions.push(use1 ? block1!.model() : block2!.model());
            }
        }

        return new Genome(g1.glstm, geneOptions);
    }
}
