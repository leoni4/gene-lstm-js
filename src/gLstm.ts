import { Client } from './client.js';
import { Species } from './species.js';
import { Genome } from './genome.js';
import { RandomSelector } from './randomSelector.js';

import type { GeneOptions, SleepingBlockConfig, SeqInput, GeneLSTMOptions } from './types/index.js';
import { EMutationPressure, MUTATION_PRESSURE_CONST, IGlstmFitOptions, IGlstmFitHistory } from './types/index.js';
import { computeLoss, isY2D, mean, variance } from './helpers/index.js';

const HISTORY_WINDOW = 50;
const SMALL_GAIN_THRESHOLD = 0.01;
const COMPLEXITY_GROWTH_ABS = 2.0;
const COMPLEXITY_GROWTH_RATIO = 0.25;
const COMPLEXITY_RATIO_DENOM_FLOOR = 8;

type ResolvedGeneLSTMOptions = Required<Omit<GeneLSTMOptions, 'sleepingBlockConfig' | 'loadData'>> & {
    sleepingBlockConfig: SleepingBlockConfig;
    loadData?: GeneOptions;
};

function getDefaultTargetSpecies(clients: number): number {
    if (clients <= 100) return 5;
    if (clients <= 500) return 8;

    return 10;
}

function resolveGeneLstmOptions(clients: number, options?: GeneLSTMOptions): ResolvedGeneLSTMOptions {
    return {
        CP: options?.CP ?? 0.1,
        C1: options?.C1 ?? 1.0,
        C2: options?.C2 ?? 0.4,

        INPUT_FEATURES: options?.INPUT_FEATURES ?? 1,

        SURVIVORS: options?.SURVIVORS ?? 0.6,
        MUTATION_RATE: options?.MUTATION_RATE ?? 1,

        BIAS_SHIFT_STRENGTH: options?.BIAS_SHIFT_STRENGTH ?? 0.2,
        BIAS_RANDOM_STRENGTH: options?.BIAS_RANDOM_STRENGTH ?? 1.0,
        WEIGHT_SHIFT_STRENGTH: options?.WEIGHT_SHIFT_STRENGTH ?? 0.2,
        WEIGHT_RANDOM_STRENGTH: options?.WEIGHT_RANDOM_STRENGTH ?? 1.0,
        ALPHA_SHIFT_STRENGTH: options?.ALPHA_SHIFT_STRENGTH ?? 0.01,

        PROBABILITY_MUTATE_ALPHA_SHIFT: options?.PROBABILITY_MUTATE_ALPHA_SHIFT ?? 0.05,
        PROBABILITY_MUTATE_BIAS_SHIFT: options?.PROBABILITY_MUTATE_BIAS_SHIFT ?? 0.8,
        PROBABILITY_MUTATE_BIAS_RANDOM: options?.PROBABILITY_MUTATE_BIAS_RANDOM ?? 0.1,
        PROBABILITY_MUTATE_WEIGHT_SHIFT: options?.PROBABILITY_MUTATE_WEIGHT_SHIFT ?? 0.95,
        PROBABILITY_MUTATE_WEIGHT_RANDOM: options?.PROBABILITY_MUTATE_WEIGHT_RANDOM ?? 0.05,

        PROBABILITY_MUTATE_LSTM_BLOCK: options?.PROBABILITY_MUTATE_LSTM_BLOCK ?? 0.01,
        PROBABILITY_ADD_BLOCK_APPEND: options?.PROBABILITY_ADD_BLOCK_APPEND ?? 0.92,
        PROBABILITY_REMOVE_BLOCK: options?.PROBABILITY_REMOVE_BLOCK ?? 0.1,

        PROBABILITY_MUTATE_ADD_UNIT: options?.PROBABILITY_MUTATE_ADD_UNIT ?? 0.02,
        PROBABILITY_MUTATE_REMOVE_UNIT: options?.PROBABILITY_MUTATE_REMOVE_UNIT ?? 0.02,

        PROBABILITY_MUTATE_READOUT_W: options?.PROBABILITY_MUTATE_READOUT_W ?? 1,
        PROBABILITY_MUTATE_READOUT_B: options?.PROBABILITY_MUTATE_READOUT_B ?? 0.6,

        sleepingBlockConfig: {
            epsilon: 0.002,
            forgetBias: 1.5,
            inputBias: -1.5,
            outputBias: 0.0,
            candidateBias: 0.0,
            initialAlpha: 0.01,
            ...options?.sleepingBlockConfig,
        },

        targetSpecies: options?.targetSpecies ?? getDefaultTargetSpecies(clients),
        cpAdjustRate: options?.cpAdjustRate ?? 0.2,
        cpDeadband: options?.cpDeadband ?? 1,
        minCP: options?.minCP ?? 0.01,
        maxCP: options?.maxCP ?? 10.0,

        mutationPressure: options?.mutationPressure ?? EMutationPressure.NORMAL,
        enablePressureEscalation: options?.enablePressureEscalation ?? true,
        stagnationThreshold: options?.stagnationThreshold ?? 15,

        loadData: options?.loadData,

        verbose: options?.verbose ?? 0,
    };
}

export class GeneLSTM {
    private _clients: Client[] = [];
    private _species: Species[] = [];
    private _maxClients: number;
    private _INPUT_FEATURES: number;

    private _CP: number;
    private _C1: number;
    private _C2: number;

    private _SURVIVORS: number;
    private _MUTATION_RATE: number;

    private _BIAS_SHIFT_STRENGTH: number;
    private _BIAS_RANDOM_STRENGTH: number;
    private _ALPHA_SHIFT_STRENGTH: number;
    private _PROBABILITY_MUTATE_BIAS_SHIFT: number;
    private _PROBABILITY_MUTATE_BIAS_RANDOM: number;
    private _PROBABILITY_MUTATE_ALPHA_SHIFT: number;

    private _WEIGHT_SHIFT_STRENGTH: number;
    private _WEIGHT_RANDOM_STRENGTH: number;
    private _PROBABILITY_MUTATE_WEIGHT_SHIFT: number;
    private _PROBABILITY_MUTATE_WEIGHT_RANDOM: number;

    private _PROBABILITY_MUTATE_LSTM_BLOCK: number;
    private _PROBABILITY_ADD_BLOCK_APPEND: number;
    private _PROBABILITY_REMOVE_BLOCK: number;
    private _PROBABILITY_MUTATE_ADD_UNIT: number;
    private _PROBABILITY_MUTATE_REMOVE_UNIT: number;
    private _PROBABILITY_MUTATE_READOUT_W: number;
    private _PROBABILITY_MUTATE_READOUT_B: number;

    private _sleepingBlockConfig: SleepingBlockConfig;

    // Dynamic CP adjustment parameters
    private _targetSpecies: number;
    private _cpAdjustRate: number;
    private _cpDeadband: number;
    private _minCP: number;
    private _maxCP: number;

    // Mutation pressure parameters
    private _mutationPressure: EMutationPressure;
    private _enablePressureEscalation: boolean;
    private _stagnationThreshold: number;
    private _stagnationCounter = 0;
    private _bestFitnessEver = 0;
    // --- Pressure tuning knobs ---
    private _pressureImprovementAbs = 1e-6; // minimal absolute improvement
    private _pressureImprovementRel = 1e-3; // relative improvement factor (0.1%)

    // Panic control
    private _panicCounter = 0;
    private _panicMaxGenerations = 30; // how long we allow PANIC to run
    private _panicCooldownCounter = 0;
    private _panicCooldownGenerations = 60; // after PANIC ends, forbid re-entering for a while

    // Optional: per-level stagnation thresholds (instead of one global)
    private _stagnationThresholdByPressure: Record<EMutationPressure, number> = {
        [EMutationPressure.COMPACT]: 40,
        [EMutationPressure.NORMAL]: 20,
        [EMutationPressure.BOOST]: 40,
        [EMutationPressure.ESCAPE]: 80,
        [EMutationPressure.PANIC]: 999999, // not used for step-up (PANIC is max)
    };

    // Champion tracking parameters
    private _bestHistory: number[] = [];
    private _complexityHistory: number[] = [];
    private _champion: Client | null = null;
    private _championStagnationCount = 0;
    private _championStagnationThreshold = 10;

    private _evolveCounts = 0;
    private _optimization = false;

    private _verbose: number = 0;

    constructor(clients: number, options?: GeneLSTMOptions) {
        this._maxClients = clients;

        const o = resolveGeneLstmOptions(clients, options);

        this._CP = o.CP;
        this._C1 = o.C1;
        this._C2 = o.C2;

        this._INPUT_FEATURES = o.INPUT_FEATURES;

        this._SURVIVORS = o.SURVIVORS;
        this._MUTATION_RATE = o.MUTATION_RATE;

        this._BIAS_SHIFT_STRENGTH = o.BIAS_SHIFT_STRENGTH;
        this._BIAS_RANDOM_STRENGTH = o.BIAS_RANDOM_STRENGTH;
        this._WEIGHT_SHIFT_STRENGTH = o.WEIGHT_SHIFT_STRENGTH;
        this._WEIGHT_RANDOM_STRENGTH = o.WEIGHT_RANDOM_STRENGTH;
        this._ALPHA_SHIFT_STRENGTH = o.ALPHA_SHIFT_STRENGTH;

        this._PROBABILITY_MUTATE_ALPHA_SHIFT = o.PROBABILITY_MUTATE_ALPHA_SHIFT;
        this._PROBABILITY_MUTATE_BIAS_SHIFT = o.PROBABILITY_MUTATE_BIAS_SHIFT;
        this._PROBABILITY_MUTATE_BIAS_RANDOM = o.PROBABILITY_MUTATE_BIAS_RANDOM;
        this._PROBABILITY_MUTATE_WEIGHT_SHIFT = o.PROBABILITY_MUTATE_WEIGHT_SHIFT;
        this._PROBABILITY_MUTATE_WEIGHT_RANDOM = o.PROBABILITY_MUTATE_WEIGHT_RANDOM;

        this._PROBABILITY_MUTATE_LSTM_BLOCK = o.PROBABILITY_MUTATE_LSTM_BLOCK;
        this._PROBABILITY_ADD_BLOCK_APPEND = o.PROBABILITY_ADD_BLOCK_APPEND;
        this._PROBABILITY_REMOVE_BLOCK = o.PROBABILITY_REMOVE_BLOCK;

        this._PROBABILITY_MUTATE_ADD_UNIT = o.PROBABILITY_MUTATE_ADD_UNIT;
        this._PROBABILITY_MUTATE_REMOVE_UNIT = o.PROBABILITY_MUTATE_REMOVE_UNIT;

        this._PROBABILITY_MUTATE_READOUT_W = o.PROBABILITY_MUTATE_READOUT_W;
        this._PROBABILITY_MUTATE_READOUT_B = o.PROBABILITY_MUTATE_READOUT_B;

        this._sleepingBlockConfig = o.sleepingBlockConfig;

        this._targetSpecies = o.targetSpecies;
        this._cpAdjustRate = o.cpAdjustRate;
        this._cpDeadband = o.cpDeadband;
        this._minCP = o.minCP;
        this._maxCP = o.maxCP;

        this._mutationPressure = o.mutationPressure;
        this._enablePressureEscalation = o.enablePressureEscalation;
        this._stagnationThreshold = o.stagnationThreshold;

        this._verbose = o.verbose;

        this._init(o.loadData);
    }

    get INPUT_FEATURES() {
        return this._INPUT_FEATURES;
    }

    get CP() {
        return this._CP;
    }

    get C1() {
        return this._C1;
    }

    get C2() {
        return this._C2;
    }

    get SURVIVORS() {
        return this._SURVIVORS;
    }
    get MUTATION_RATE() {
        return this._MUTATION_RATE;
    }
    get BIAS_SHIFT_STRENGTH() {
        return this._BIAS_SHIFT_STRENGTH;
    }
    get BIAS_RANDOM_STRENGTH() {
        return this._BIAS_RANDOM_STRENGTH;
    }
    get PROBABILITY_MUTATE_BIAS_SHIFT() {
        return this._PROBABILITY_MUTATE_BIAS_SHIFT;
    }
    get PROBABILITY_MUTATE_BIAS_RANDOM() {
        return this._PROBABILITY_MUTATE_BIAS_RANDOM;
    }
    get PROBABILITY_MUTATE_ALPHA_SHIFT() {
        return this._PROBABILITY_MUTATE_ALPHA_SHIFT;
    }
    get ALPHA_SHIFT_STRENGTH() {
        return this._ALPHA_SHIFT_STRENGTH;
    }
    get WEIGHT_SHIFT_STRENGTH() {
        return this._WEIGHT_SHIFT_STRENGTH;
    }
    get WEIGHT_RANDOM_STRENGTH() {
        return this._WEIGHT_RANDOM_STRENGTH;
    }
    get PROBABILITY_MUTATE_WEIGHT_SHIFT() {
        return this._PROBABILITY_MUTATE_WEIGHT_SHIFT;
    }
    get PROBABILITY_MUTATE_WEIGHT_RANDOM() {
        return this._PROBABILITY_MUTATE_WEIGHT_RANDOM;
    }
    get PROBABILITY_MUTATE_LSTM_BLOCK() {
        return this._PROBABILITY_MUTATE_LSTM_BLOCK;
    }
    get PROBABILITY_ADD_BLOCK_APPEND() {
        return this._PROBABILITY_ADD_BLOCK_APPEND;
    }
    get PROBABILITY_REMOVE_BLOCK() {
        return this._PROBABILITY_REMOVE_BLOCK;
    }
    get PROBABILITY_MUTATE_ADD_UNIT() {
        return this._PROBABILITY_MUTATE_ADD_UNIT;
    }
    get PROBABILITY_MUTATE_REMOVE_UNIT() {
        return this._PROBABILITY_MUTATE_REMOVE_UNIT;
    }
    get PROBABILITY_MUTATE_READOUT_W() {
        return this._PROBABILITY_MUTATE_READOUT_W;
    }
    get PROBABILITY_MUTATE_READOUT_B() {
        return this._PROBABILITY_MUTATE_READOUT_B;
    }
    get sleepingBlockConfig() {
        return this._sleepingBlockConfig;
    }

    get clients() {
        return this._clients;
    }

    get mutationPressure(): EMutationPressure {
        return this._mutationPressure;
    }

    set mutationPressure(value: EMutationPressure) {
        this._mutationPressure = value;
    }

    get champion() {
        return this._champion;
    }

    /**
     * Returns the current mutation pressure multipliers for topology and weights.
     * These are used to scale mutation probabilities and magnitudes throughout the system.
     */
    getMutationPressure(): { topology: number; weights: number } {
        return MUTATION_PRESSURE_CONST[this._mutationPressure];
    }

    emptyGenome() {
        return new Genome(this);
    }

    private _initGenome(data: GeneOptions) {
        return new Genome(this, data);
    }

    private _init(data?: GeneOptions) {
        let genome: Genome;
        if (data) {
            genome = this._initGenome(data);
        } else {
            genome = this.emptyGenome();
        }
        this._clients = [];
        for (let i = 0; i < this._maxClients; i += 1) {
            const c: Client = new Client(genome);
            if (i === 0) {
                this._species.push(new Species(c));
            } else {
                this._species[0].put(c, true);
            }
            this._clients.push(c);
        }
    }

    model() {
        this._clients.sort((a, b) => {
            return a.score > b.score ? -1 : 1;
        });

        return (this._champion || this._clients[0]).genome.lstmArray.map(l => l.model());
    }

    printSpecies() {
        const totalClients = this._clients.length;

        const totalDepth = this._clients.reduce((acc, c) => acc + c.genome.lstmArray.length, 0);

        const totalUnits = this._clients.reduce((acc, c) => {
            return (
                acc +
                c.genome.lstmArray.reduce((uAcc, lstm) => {
                    // safest: readoutW length reflects hiddenSize after _ensureConsistentSizes
                    return uAcc + (lstm.readoutW?.length ?? lstm.model().hiddenSize ?? 1);
                }, 0)
            );
        }, 0);

        const totalBlocks = totalDepth;
        const avgDepth = totalClients ? totalDepth / totalClients : 0;
        const avgUnitsPerBlock = totalBlocks ? totalUnits / totalBlocks : 0;

        console.log(
            '### Species:',
            this._species.length,
            '| Complexity:',
            `depth=${totalDepth} (avg ${avgDepth.toFixed(2)})`,
            `units=${totalUnits} (avg/block ${avgUnitsPerBlock.toFixed(2)})`,
        );

        for (let i = 0; i < this._species.length; i += 1) {
            console.log('#', this._species[i].score, this._species[i].size());
        }
        console.log('###');
    }

    fit(xTrain: SeqInput[], yTrain: SeqInput, options: IGlstmFitOptions = {}): IGlstmFitHistory {
        // ---- validate ----
        if (!xTrain || !yTrain || xTrain.length === 0) {
            throw new Error('Training data cannot be empty');
        }
        if (xTrain.length !== yTrain.length) {
            throw new Error(
                `Input and output data must have the same length (got ${xTrain.length} vs ${yTrain.length})`,
            );
        }

        const maxEpochs = options.epochs ?? Infinity;
        const errorThreshold = options.errorThreshold ?? 0.01;
        const validationSplit = options.validationSplit ?? 0;
        const verbose = options.verbose ?? 1;
        const logInterval = options.logInterval ?? 100;

        const loss = options.loss ?? 'mae';
        const antiConst = options.antiConstantPenalty ?? false;
        const antiLambda = options.antiConstantLambda ?? 0.05;
        const shuffleEachEpoch = options.shuffleEachEpoch ?? true;

        if (validationSplit < 0 || validationSplit >= 1) {
            throw new Error('validationSplit must be between 0 and 1 (exclusive)');
        }

        // normalize y into 2D array targets
        const y2d: number[][] = isY2D(yTrain) ? yTrain : (yTrain as number[]).map(v => [v]);

        const outputDim = y2d[0]?.length ?? 1;

        // split train/val
        let trainX = xTrain;
        let trainY = y2d;
        let valX: SeqInput[] | null = null;
        let valY: number[][] | null = null;

        if (validationSplit > 0) {
            const splitIndex = Math.floor(xTrain.length * (1 - validationSplit));
            trainX = xTrain.slice(0, splitIndex);
            trainY = y2d.slice(0, splitIndex);
            valX = xTrain.slice(splitIndex);
            valY = y2d.slice(splitIndex);

            if (trainX.length === 0) throw new Error('Validation split too large, no training data remaining');
        }

        const history: IGlstmFitHistory = {
            error: [],
            validationError: valX ? [] : undefined,
            epochs: 0,
            champion: null,
            stoppedEarly: false,
        };

        // index array for shuffling
        const idx = new Array(trainX.length).fill(0).map((_, i) => i);

        let epoch = 0;

        while (epoch < maxEpochs) {
            if (shuffleEachEpoch) {
                for (let i = idx.length - 1; i > 0; i--) {
                    const j = Math.floor(Math.random() * (i + 1));
                    [idx[i], idx[j]] = [idx[j], idx[i]];
                }
            }

            let bestScore = -Infinity;
            let bestClient: Client = this._clients[0];
            let bestError = Infinity;

            // ---- evaluate population ----
            for (const client of this._clients) {
                let totalLoss = 0;

                // for anti-constant: collect predictions (for 1D output only)
                const predsForPenalty: number[] = antiConst && outputDim === 1 ? [] : [];

                for (let ii = 0; ii < idx.length; ii++) {
                    const i = idx[ii];
                    const pred = client.calculate(trainX[i]); // returns number[]
                    const target = trainY[i];

                    // safety: ensure correct dimension
                    if (pred.length !== outputDim) {
                        // tolerate by slicing/padding
                        const p = pred.slice(0, outputDim);
                        while (p.length < outputDim) p.push(0);
                        totalLoss += computeLoss(p, target, loss);
                        if (predsForPenalty.length) predsForPenalty.push(p[0]);
                    } else {
                        totalLoss += computeLoss(pred, target, loss);
                        if (predsForPenalty.length) predsForPenalty.push(pred[0]);
                    }
                }

                let err = totalLoss / trainX.length;

                // anti-constant penalty (helps avoid 0.5 plateaus for binary tasks)
                if (antiConst && outputDim === 1) {
                    const m = mean(predsForPenalty);
                    const v = variance(predsForPenalty);

                    // mean penalty discourages always ~0 or ~1
                    const meanPenalty = antiLambda * Math.abs(m - 0.5);

                    // variance penalty discourages always constant (esp. 0.5)
                    const varPenalty = (antiLambda * 0.5) / (v + 1e-6);

                    err = Math.min(10, err + meanPenalty + varPenalty);
                }

                client.error = err;

                // score: map lower error to higher score, keep in (0,1]
                // simple: score = 1 / (1 + err)  (more stable than 1-err for losses like BCE)
                client.score = 1 / (1 + err);

                // tiny tie-breaker noise (important on flat landscapes)
                client.score += Math.random() * 1e-9;

                if (client.score > bestScore) {
                    bestScore = client.score;
                    bestClient = client;
                    bestError = err;
                }
            }

            history.error.push(bestError);

            // ---- validation ----
            let valErr: number | undefined;
            if (valX && valY) {
                let totalVal = 0;
                for (let i = 0; i < valX.length; i++) {
                    const pred = bestClient.calculate(valX[i]);
                    const target = valY[i];

                    const p =
                        pred.length === outputDim
                            ? pred
                            : [...pred.slice(0, outputDim), ...new Array(Math.max(0, outputDim - pred.length)).fill(0)];
                    totalVal += computeLoss(p, target, loss);
                }
                valErr = totalVal / valX.length;
                history.validationError!.push(valErr);
            }

            // ---- logging ----
            if (verbose === 2 || (verbose === 1 && (epoch % logInterval === 0 || epoch === 0))) {
                const blocks = bestClient.genome.lstmArray.length;
                const units = bestClient.genome.lstmArray.reduce((acc, lstm) => acc + (lstm.readoutW?.length ?? 1), 0);
                let msg = `Epoch ${epoch} - error: ${bestError.toFixed(6)} - blocks: ${blocks} - units: ${units}`;

                if (valErr !== undefined) msg += ` - val_error: ${valErr.toFixed(6)}`;

                if (verbose === 2) {
                    msg += ` - species: ${this._species.length} - pressure: ${this._mutationPressure} - CP: ${this._CP.toFixed(4)}`;
                }
                console.log(msg);
            }

            // ---- early stop ----
            if (bestError <= errorThreshold) {
                history.stoppedEarly = true;
                history.epochs = epoch;
                history.champion = this._champion ?? bestClient;

                if (verbose > 0) console.log(`✓ Training completed: error threshold reached at epoch ${epoch}`);
                break;
            }

            // evolve
            const shouldOptimize = bestError <= 0.02; // можно связать с твоим OPT_ERR_THRESHOLD
            this.evolve(shouldOptimize);

            epoch++;
        }

        if (!history.stoppedEarly) {
            history.epochs = epoch;
            history.champion = this._champion ?? this._clients[0];
            if (verbose > 0) console.log(`Training completed: max epochs (${maxEpochs}) reached`);
        }

        return history;
    }

    /**
     * Dynamically adjusts the compatibility parameter (CP) to maintain species count near target.
     * Higher CP → species merge more easily → fewer species
     * Lower CP → species split more easily → more species
     *
     * @param speciesCount Current number of species
     * @param generation Optional generation number for logging
     */
    adjustCP(speciesCount: number, generation?: number): void {
        const cpBefore = this._CP;
        const error = speciesCount - this._targetSpecies;

        // Deadband: don't adjust if within acceptable range
        if (Math.abs(error) <= this._cpDeadband) {
            if (generation !== undefined && this._verbose === 2) {
                console.log(
                    `[Gen ${generation}] CP: ${this._CP.toFixed(4)} | Species: ${speciesCount}/${this._targetSpecies} (within deadband ±${this._cpDeadband}) | No adjustment`,
                );
            }

            return;
        }

        // Calculate adjustment factor
        // Positive error (too many species) → increase CP
        // Negative error (too few species) → decrease CP
        const errorRatio = error / this._targetSpecies;
        const adjustmentFactor = 1 + this._cpAdjustRate * errorRatio;

        // Apply multiplicative adjustment
        this._CP *= adjustmentFactor;

        // Clamp to valid range
        this._CP = Math.max(this._minCP, Math.min(this._maxCP, this._CP));

        // Debug logging
        if (generation !== undefined && this._verbose === 2) {
            const direction = error > 0 ? '↑ INCREASE' : '↓ DECREASE';
            console.log(
                `[Gen ${generation}] CP: ${cpBefore.toFixed(4)} → ${this._CP.toFixed(4)} (${direction}) | Species: ${speciesCount}/${this._targetSpecies} | Error: ${error > 0 ? '+' : ''}${error}`,
            );
        }
    }

    private _calcClientComplexity(client: Client): { blocks: number; units: number; complexity: number } {
        const blocks = client.genome.lstmArray.length;
        const units = client.genome.lstmArray.reduce((acc, lstm) => acc + (lstm.readoutW?.length ?? 1), 0);

        // можно другой коэффициент, но это нормальный старт
        const complexity = blocks + 0.25 * units;

        return { blocks, units, complexity };
    }

    /**
     * Automatically adjusts mutation pressure based on fitness stagnation.
     * State machine:
     * - If fitness improves: reset stagnation counter, gradually reduce pressure (if above NORMAL)
     * - If fitness stagnates for N generations: escalate pressure (NORMAL → BOOST → ESCAPE → PANIC)
     *
     * @param currentBestFitness The best fitness score in the current generation
     * @param generation Optional generation number for logging
     */
    updateMutationPressure(currentBestFitness: number, generation?: number): void {
        if (!this._enablePressureEscalation) {
            return;
        }
        const canCompact =
            this._bestHistory.length >= 2 &&
            this._complexityHistory.length >= 2 &&
            this._championStagnationCount > HISTORY_WINDOW;

        if (canCompact) {
            const scoreHist = this._bestHistory;
            const compHist = this._complexityHistory;

            const s0 = scoreHist[0];
            const sBest = Math.max(...scoreHist);
            const gain = sBest - s0;

            const c0 = compHist[0];
            const c1 = compHist[compHist.length - 1];
            const growthAbs = c1 - c0;
            const denom = Math.max(c0, COMPLEXITY_RATIO_DENOM_FLOOR);
            const growthRatio = growthAbs / denom;

            const growingMeaningfully = growthAbs >= COMPLEXITY_GROWTH_ABS || growthRatio >= COMPLEXITY_GROWTH_RATIO;

            const tinyProgress = gain <= SMALL_GAIN_THRESHOLD;

            if (growingMeaningfully && tinyProgress) {
                if (this._mutationPressure !== EMutationPressure.COMPACT) {
                    const old = this._mutationPressure;
                    this._mutationPressure = EMutationPressure.COMPACT;

                    if (this._verbose === 2) {
                        console.log(
                            `[Gen ${this._evolveCounts}] Mutation Pressure: ${old} → COMPACT (gain=${gain.toFixed(4)}, complexity ${c0.toFixed(2)}→${c1.toFixed(2)})`,
                        );
                    }
                }
            }
        }

        if (this._panicCooldownCounter > 0) {
            this._panicCooldownCounter--;
        }
        if (this._mutationPressure === EMutationPressure.PANIC) {
            this._panicCounter++;

            if (this._panicCounter >= this._panicMaxGenerations) {
                // Force exit PANIC even without improvement
                const oldPressure = this._mutationPressure;
                this._mutationPressure = EMutationPressure.ESCAPE;

                this._panicCounter = 0;
                this._stagnationCounter = 0; // reset stagnation when forcing down
                this._panicCooldownCounter = this._panicCooldownGenerations;

                if (generation !== undefined && this._verbose === 2) {
                    console.log(
                        `[Gen ${generation}] Mutation Pressure: ${oldPressure} → ${this._mutationPressure} (PANIC timeout ${this._panicMaxGenerations} gens, cooldown ${this._panicCooldownGenerations} gens)`,
                    );
                }

                return;
            }
        }

        // Abs + relative threshold: makes improvement detection stable for different fitness scales
        const improvementThreshold = Math.max(
            this._pressureImprovementAbs,
            Math.abs(this._bestFitnessEver) * this._pressureImprovementRel,
        );

        const hasImproved = currentBestFitness > this._bestFitnessEver + improvementThreshold;
        if (hasImproved) {
            // Fitness improved - reset stagnation and gradually reduce pressure
            this._bestFitnessEver = currentBestFitness;
            this._stagnationCounter = 0;

            // If we improved, clear panic tracking (we don't want to be stuck in panic mode)
            this._panicCounter = 0;
            // Optionally shorten cooldown because progress resumed:
            if (this._panicCooldownCounter > 0)
                this._panicCooldownCounter = Math.floor(this._panicCooldownCounter * 0.5);

            // Gradually reduce pressure if above NORMAL
            const pressureLevels = [
                EMutationPressure.PANIC,
                EMutationPressure.ESCAPE,
                EMutationPressure.BOOST,
                EMutationPressure.NORMAL,
            ];
            let currentIndex = pressureLevels.indexOf(this._mutationPressure);
            if (currentIndex === -1) {
                this._mutationPressure = EMutationPressure.NORMAL;
                currentIndex = pressureLevels.indexOf(this._mutationPressure);
            }

            if (currentIndex < pressureLevels.length - 1) {
                // Move one level down towards NORMAL
                this._mutationPressure = pressureLevels[currentIndex + 1];
                if (generation !== undefined && this._verbose === 2) {
                    console.log(
                        `[Gen ${generation}] Mutation Pressure: ${pressureLevels[currentIndex]} → ${this._mutationPressure} (fitness improved to ${currentBestFitness.toFixed(4)})`,
                    );
                }
            }
        } else {
            this._stagnationCounter++;

            // Use per-level stagnation threshold (more patient as pressure increases)
            const levelThreshold =
                this._stagnationThresholdByPressure?.[this._mutationPressure] ?? this._stagnationThreshold;

            if (this._stagnationCounter >= levelThreshold) {
                const pressureLevels = [
                    EMutationPressure.NORMAL,
                    EMutationPressure.BOOST,
                    EMutationPressure.ESCAPE,
                    EMutationPressure.PANIC,
                ];

                let currentIndex = pressureLevels.indexOf(this._mutationPressure);
                if (currentIndex === -1) {
                    this._mutationPressure = EMutationPressure.NORMAL;
                    currentIndex = pressureLevels.indexOf(this._mutationPressure);
                }

                // If PANIC is on cooldown, do not escalate into PANIC
                const nextPressure = pressureLevels[currentIndex + 1];
                const panicBlocked = nextPressure === EMutationPressure.PANIC && this._panicCooldownCounter > 0;

                if (currentIndex < pressureLevels.length - 1 && !panicBlocked) {
                    const oldPressure = this._mutationPressure;
                    this._mutationPressure = nextPressure;
                    this._stagnationCounter = 0; // Reset counter after escalation

                    // entering panic: reset panic counter
                    if (this._mutationPressure === EMutationPressure.PANIC) {
                        this._panicCounter = 0;
                    }

                    if (generation !== undefined && this._verbose === 2) {
                        console.log(
                            `[Gen ${generation}] Mutation Pressure: ${oldPressure} → ${this._mutationPressure} (stagnated for ${levelThreshold} generations, best: ${this._bestFitnessEver.toFixed(4)})`,
                        );
                    }
                } else {
                    // If we could not escalate (max or panic blocked), you may still want to reset counter to avoid spam
                    this._stagnationCounter = 0;

                    if (generation !== undefined && panicBlocked && this._verbose === 2) {
                        console.log(
                            `[Gen ${generation}] Mutation Pressure: ${this._mutationPressure} (PANIC blocked by cooldown ${this._panicCooldownCounter} gens)`,
                        );
                    }
                }
            }
        }
    }

    evolve(optimization = false) {
        this._evolveCounts++;
        this._optimization = optimization || this._evolveCounts % 10 === 0;
        this._updateChampion();

        if (this._enablePressureEscalation && this._clients.length > 0) {
            const bestClient = this._clients[0]; // Already sorted by score in _normalizeScore
            const currentBestScore = bestClient.score;
            this.updateMutationPressure(currentBestScore, this._evolveCounts);
        }

        this._normalizeScore();

        this._genSpecies();

        // Dynamically adjust CP based on current species count
        this.adjustCP(this._species.length, this._evolveCounts);

        this._kill();
        this._removeExtinct();
        this._reproduce();
        this._mutate();
    }

    private _normalizeScore() {
        let maxScore = 0;
        const bestScoreSet: number[] = [];
        let minScore = Infinity;

        for (let i = 0; i < this._clients.length; i += 1) {
            const item = this._clients[i];
            item.bestScore = false;
            maxScore = item.score > maxScore ? item.score : maxScore;
            minScore = item.score < minScore ? item.score : minScore;
        }

        for (let i = 0; i < this._clients.length; i += 1) {
            const item = this._clients[i];
            if (item.score === maxScore) {
                bestScoreSet.push(i);
                item.bestScore = true;
                item.score = 1;
            } else if (item.score === minScore) {
                item.score = 0;
            } else {
                item.score = (item.score - minScore) / (maxScore - minScore);
            }
        }

        if (bestScoreSet.length > 1) {
            bestScoreSet.forEach((i, index) => {
                if (index === 0) return;
                this._clients[i].bestScore = false;
            });
        }

        this._clients.sort((a, b) => (a.score > b.score ? -1 : 1));

        const cof = this._optimization ? 0.1 : 0.01;

        // separate knobs (tune as needed)
        const depthCof = cof; // penalty for extra LSTM blocks
        const unitsCof = cof * 0.35; // penalty for hidden units (softer)

        this._clients.forEach(item => {
            const blocks = item.genome.lstmArray.length;

            // sum hidden size across all blocks
            const units = item.genome.lstmArray.reduce((acc, lstm) => acc + (lstm.readoutW?.length ?? 1), 0);

            // soft growth penalties
            const depthPenalty = (Math.sqrt(Math.sqrt(blocks)) - 1) * depthCof;
            const unitsPenalty = (Math.sqrt(units) - 1) * unitsCof;

            item.score -= depthPenalty + unitsPenalty;
        });
    }

    private _genSpecies() {
        for (let i = 0; i < this._species.length; i += 1) {
            this._species[i].reset();
        }
        for (let i = 0; i < this._clients.length; i += 1) {
            const c = this._clients[i];
            if (c.species !== null) {
                continue;
            }

            let found = false;
            for (let k = 0; k < this._species.length; k += 1) {
                const s = this._species[k];
                if (s.put(c)) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                this._species.push(new Species(c));
            }
        }
        for (let i = 0; i < this._species.length; i += 1) {
            this._species[i].evaluateScore();
        }
    }

    private _kill() {
        for (let i = 0; i < this._species.length; i += 1) {
            this._species[i].kill(this._SURVIVORS);
        }
    }

    private _removeExtinct() {
        for (let i = this._species.length - 1; i >= 0; i--) {
            if (this._species[i].size() <= 1 && !this._species[i].clients[0]?.bestScore && this._species.length > 1) {
                this._species[i].goExtinct();
                this._species.splice(i, 1);
            }
        }
    }

    private _reproduce() {
        const selector = new RandomSelector(this._SURVIVORS);
        for (let i = 0; i < this._species.length; i += 1) {
            selector.add(this._species[i]);
        }
        for (let i = 0; i < this._clients.length; i += 1) {
            const c = this._clients[i];
            if (c.species === null) {
                const s = selector.random();
                c.genome = s.breed();
                s.put(c, true);
            }
        }
        selector.reset();
    }

    private _mutate() {
        this._clients.forEach(client => {
            client.mutate();
        });
    }

    /**
     * Updates the champion (best client ever seen) and handles re-insertion.
     * The champion is a saved copy of the best performing client.
     *
     * Logic:
     * 1. Compare current best client with stored champion
     * 2. If current best is better (or no champion exists), save a copy and reset stagnation
     * 3. If champion hasn't been improved for N epochs (default 10), re-insert it into population
     *    by replacing the worst performing client
     * 4. After re-insertion, reset stagnation counter
     */
    private _updateChampion(): void {
        if (this._clients.length === 0) {
            return;
        }

        this._clients.sort((a, b) => {
            return a.score > b.score ? -1 : 1;
        });

        // Current best client (already sorted by score in _normalizeScore)
        const currentBest = this._clients[0];

        const { complexity: currentBestComplexity } = this._calcClientComplexity(currentBest);

        // Initialize champion if not exists
        if (this._champion === null) {
            // Create a deep copy of the best client
            this._champion = this._copyClient(currentBest);
            this._championStagnationCount = 0;

            this._bestHistory = [currentBest.score];
            this._complexityHistory = [currentBestComplexity];

            if (this._verbose === 2)
                console.log(
                    `[Gen ${this._evolveCounts}] Champion initialized with score ${currentBest.score.toFixed(4)}`,
                );

            return;
        }

        // Compare current best with champion (use raw score comparison)
        // Since scores are normalized, we compare the actual fitness before normalization
        // We'll use a simple comparison: if current best has higher normalized score, it's better
        const championImproved = currentBest.score > this._champion.score;

        if (championImproved) {
            // Update champion with new best
            this._champion = this._copyClient(currentBest);
            this._championStagnationCount = 0;
            if (this._verbose === 2)
                console.log(`[Gen ${this._evolveCounts}] Champion updated with score ${currentBest.score.toFixed(4)}`);
        } else {
            // No improvement - increment stagnation counter
            this._championStagnationCount++;

            // Check if we should re-insert champion
            if (this._championStagnationCount >= this._championStagnationThreshold) {
                // Re-insert champion by replacing the worst client
                const worstClientIndex = this._clients.length - 1;
                const worstClient = this._clients[worstClientIndex];
                if (this._verbose === 2)
                    console.log(
                        `[Gen ${this._evolveCounts}] Champion re-inserted after ${this._championStagnationCount} stagnant epochs (replacing worst client with score ${worstClient.score.toFixed(4)})`,
                    );

                // Detach worst client from its species
                // The species will clean up in the next _genSpecies call
                if (worstClient.species !== null) {
                    worstClient.species = null;
                }

                // Create a fresh copy of champion and insert it
                const championCopy = this._copyClient(this._champion);
                this._clients[worstClientIndex] = championCopy;

                // Champion will be assigned to a species in the next _genSpecies call
                // Reset stagnation counter
                this._championStagnationCount = 0;
            }
        }

        this._bestHistory.push(currentBest.score);
        this._complexityHistory.push(currentBestComplexity);

        if (this._bestHistory.length > HISTORY_WINDOW) this._bestHistory.shift();
        if (this._complexityHistory.length > HISTORY_WINDOW) this._complexityHistory.shift();
    }

    /**
     * Creates a deep copy of a client, including its genome.
     * This ensures the champion is preserved independently of population changes.
     */
    private _copyClient(client: Client): Client {
        // Create a new genome by copying the structure from the original
        const geneOptions = client.genome.lstmArray.map(lstm => lstm.model());
        const newGenome = new Genome(this, geneOptions);

        // Create a new client with the copied genome
        const newClient = new Client(newGenome);
        newClient.score = client.score;
        newClient.bestScore = client.bestScore;
        newClient.error = client.error;

        return newClient;
    }
}
