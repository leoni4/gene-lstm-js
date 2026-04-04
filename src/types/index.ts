import { Client } from '../client.js';

export enum EMutationPressure {
    COMPACT = 'COMPACT',
    NORMAL = 'NORMAL',
    BOOST = 'BOOST',
    ESCAPE = 'ESCAPE',
    PANIC = 'PANIC',
}

export type MutationPressureType = 'topology' | 'weights';

export const MUTATION_PRESSURE_CONST: Record<EMutationPressure, Record<MutationPressureType, number>> = {
    COMPACT: {
        topology: 0.1,
        weights: 0.8,
    },
    NORMAL: {
        topology: 1,
        weights: 1,
    },
    BOOST: {
        topology: 1.2,
        weights: 1.5,
    },
    ESCAPE: {
        topology: 1.5,
        weights: 2,
    },
    PANIC: {
        topology: 2,
        weights: 4,
    },
};

export type LstmOptions = {
    hiddenSize: number;

    forgetGate: GateUnitOptions[];
    potentialLongToRem: GateUnitOptions[];
    potentialLongMemory: GateUnitOptions[];
    shortMemoryToRemember: GateUnitOptions[];

    readoutW: number[];
    readoutB: number;

    alpha: number;
};

export type GeneOptions = LstmOptions[];

export interface SleepingBlockConfig {
    epsilon: number; // Small weight range
    forgetBias: number; // Positive: remember everything
    inputBias: number; // Negative: write little
    outputBias: number; // Neutral
    candidateBias: number; // Neutral
    initialAlpha: number; // Skip connection initial value
}

export type SeqInput = number[] | number[][];

export type GateUnitOptions = {
    weight1: number; // recurrent
    weight2: number; // scalar input fallback (можно хранить)
    bias: number;
    weightIn?: number[]; // vector input weights (len = INPUT_FEATURES)
};

export interface IGlstmFitOptions {
    epochs?: number;
    errorThreshold?: number;
    validationSplit?: number;
    verbose?: 0 | 1 | 2;
    logInterval?: number;

    // new
    loss?: 'mae' | 'mse' | 'bce';
    outputMode?: 'auto' | 'binary' | 'regression'; // mostly for logging/thresholding
    antiConstantPenalty?: boolean;
    antiConstantLambda?: number; // default 0.05
    shuffleEachEpoch?: boolean;
}

export interface IGlstmFitHistory {
    error: number[];
    validationError?: number[];
    epochs: number;
    champion: Client | null;
    stoppedEarly: boolean;
}

export interface GeneLSTMOptions {
    CP?: number;
    C1?: number;
    C2?: number;
    INPUT_FEATURES?: number;
    SURVIVORS?: number;
    MUTATION_RATE?: number;
    BIAS_SHIFT_STRENGTH?: number;
    BIAS_RANDOM_STRENGTH?: number;
    ALPHA_SHIFT_STRENGTH?: number;
    PROBABILITY_MUTATE_BIAS_SHIFT?: number;
    PROBABILITY_MUTATE_BIAS_RANDOM?: number;
    WEIGHT_SHIFT_STRENGTH?: number;
    WEIGHT_RANDOM_STRENGTH?: number;
    PROBABILITY_MUTATE_ALPHA_SHIFT?: number;
    PROBABILITY_MUTATE_WEIGHT_SHIFT?: number;
    PROBABILITY_MUTATE_WEIGHT_RANDOM?: number;
    PROBABILITY_MUTATE_LSTM_BLOCK?: number;
    PROBABILITY_ADD_BLOCK_APPEND?: number;
    PROBABILITY_REMOVE_BLOCK?: number;
    PROBABILITY_MUTATE_ADD_UNIT?: number;
    PROBABILITY_MUTATE_REMOVE_UNIT?: number;
    PROBABILITY_MUTATE_READOUT_W?: number;
    PROBABILITY_MUTATE_READOUT_B?: number;
    sleepingBlockConfig?: Partial<SleepingBlockConfig>;
    loadData?: GeneOptions;
    // Dynamic CP adjustment parameters
    targetSpecies?: number;
    cpAdjustRate?: number;
    cpDeadband?: number;
    minCP?: number;
    maxCP?: number;
    // Mutation pressure parameters
    mutationPressure?: EMutationPressure;
    enablePressureEscalation?: boolean;
    stagnationThreshold?: number;

    verbose?: number;
}
