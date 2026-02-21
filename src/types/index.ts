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
