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
}

export type GeneOptions = LstmOptions[];
