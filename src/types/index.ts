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
        weights: 2,
    },
};

export interface ShortMemory {
    weight1: number;
    weight2: number;
    bias: number;
}

export interface LstmOptions {
    forgetGate: ShortMemory;
    potentialLongToRem: ShortMemory;
    potentialLongMemory: ShortMemory;
    shortMemoryToRemember: ShortMemory;
    alpha?: number; // Skip connection strength
}

export type GeneOptions = LstmOptions[];

export interface SleepingBlockConfig {
    epsilon: number; // Small weight range
    forgetBias: number; // Positive: remember everything
    inputBias: number; // Negative: write little
    outputBias: number; // Neutral
    candidateBias: number; // Neutral
    initialAlpha: number; // Skip connection initial value
}
