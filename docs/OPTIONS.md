# GeneLSTM Configuration Options

Complete reference guide for all configuration options available in the `GeneLSTMOptions` interface.

## Table of Contents

- [Overview](#overview)
- [Basic Configuration](#basic-configuration)
- [Speciation Parameters](#speciation-parameters)
- [Evolution Parameters](#evolution-parameters)
- [Weight Mutation Parameters](#weight-mutation-parameters)
- [Bias Mutation Parameters](#bias-mutation-parameters)
- [Skip Connection (Alpha) Mutation](#skip-connection-alpha-mutation)
- [Topology Mutation Parameters](#topology-mutation-parameters)
- [Readout Layer Mutation](#readout-layer-mutation)
- [Sleeping Block Configuration](#sleeping-block-configuration)
- [Dynamic Speciation](#dynamic-speciation)
- [Mutation Pressure System](#mutation-pressure-system)
- [Pre-trained Models](#pre-trained-models)
- [Logging](#logging)
- [Complete Example](#complete-example)

## Overview

The `GeneLSTMOptions` interface provides extensive control over the evolutionary process, network topology, and mutation strategies. All parameters are optional and have sensible defaults.

```typescript
interface GeneLSTMOptions {
    // Speciation
    CP?: number;
    C1?: number;
    C2?: number;

    // Basic configuration
    INPUT_FEATURES?: number;
    SURVIVORS?: number;
    MUTATION_RATE?: number;

    // Weight mutations
    WEIGHT_SHIFT_STRENGTH?: number;
    WEIGHT_RANDOM_STRENGTH?: number;
    PROBABILITY_MUTATE_WEIGHT_SHIFT?: number;
    PROBABILITY_MUTATE_WEIGHT_RANDOM?: number;

    // Bias mutations
    BIAS_SHIFT_STRENGTH?: number;
    BIAS_RANDOM_STRENGTH?: number;
    PROBABILITY_MUTATE_BIAS_SHIFT?: number;
    PROBABILITY_MUTATE_BIAS_RANDOM?: number;

    // Alpha (skip connection) mutations
    ALPHA_SHIFT_STRENGTH?: number;
    PROBABILITY_MUTATE_ALPHA_SHIFT?: number;

    // Topology mutations
    PROBABILITY_MUTATE_LSTM_BLOCK?: number;
    PROBABILITY_ADD_BLOCK_APPEND?: number;
    PROBABILITY_REMOVE_BLOCK?: number;
    PROBABILITY_MUTATE_ADD_UNIT?: number;
    PROBABILITY_MUTATE_REMOVE_UNIT?: number;

    // Readout layer mutations
    PROBABILITY_MUTATE_READOUT_W?: number;
    PROBABILITY_MUTATE_READOUT_B?: number;

    // Advanced configuration
    sleepingBlockConfig?: Partial<SleepingBlockConfig>;
    loadData?: GeneOptions;

    // Dynamic speciation
    targetSpecies?: number;
    cpAdjustRate?: number;
    cpDeadband?: number;
    minCP?: number;
    maxCP?: number;

    // Mutation pressure
    mutationPressure?: EMutationPressure;
    enablePressureEscalation?: boolean;
    stagnationThreshold?: number;

    // Logging
    verbose?: number;
}
```

## Basic Configuration

### `INPUT_FEATURES`

**Type:** `number`  
**Default:** `1`

Number of input features per time step. This determines the dimensionality of the input vectors.

**Example:**

```typescript
// For single scalar input
const glstm = new GeneLSTM(100, {
    INPUT_FEATURES: 1,
});

// For multi-dimensional input (e.g., 10 features)
const glstm = new GeneLSTM(100, {
    INPUT_FEATURES: 10,
});

// Usage with 3 features
const input = [0.5, 0.3, 0.8]; // Must match INPUT_FEATURES
const output = glstm.clients[0].calculate(input);
```

**Use Cases:**

- Time series with multiple sensors: `INPUT_FEATURES: 5`
- Financial data (OHLCV): `INPUT_FEATURES: 5`
- Single value prediction: `INPUT_FEATURES: 1`

---

## Speciation Parameters

Speciation groups similar networks together, promoting diversity and preventing premature convergence.

### `CP` (Compatibility Parameter)

**Type:** `number`  
**Default:** `0.1`  
**Range:** `0.01` - `10.0`

Distance threshold for species membership. Lower values create more species (stricter), higher values create fewer species (more permissive).

**Example:**

```typescript
// Many small species (high diversity)
const glstm = new GeneLSTM(300, {
    CP: 0.05,
});

// Few large species (faster convergence)
const glstm = new GeneLSTM(300, {
    CP: 0.3,
});
```

**Guidelines:**

- Start with default `0.1`
- Increase if too many species form
- Decrease if all networks cluster in one species
- Use with [Dynamic Speciation](#dynamic-speciation) for automatic adjustment

### `C1` (Topology Difference Weight)

**Type:** `number`  
**Default:** `1.0`

Weight coefficient for structural differences in distance calculation. Higher values make topology differences more important for species separation.

**Example:**

```typescript
// Emphasize topology differences
const glstm = new GeneLSTM(300, {
    C1: 2.0, // Topology differences count twice as much
    C2: 0.5, // Weight differences count half as much
});
```

### `C2` (Weight Difference Weight)

**Type:** `number`  
**Default:** `0.4`

Weight coefficient for parameter differences in distance calculation. Higher values make weight differences more important for species separation.

**Example:**

```typescript
// Emphasize weight differences
const glstm = new GeneLSTM(300, {
    C1: 0.5, // Topology differences count half as much
    C2: 1.5, // Weight differences count 1.5x as much
});
```

---

## Evolution Parameters

### `SURVIVORS`

**Type:** `number`  
**Default:** `0.6`  
**Range:** `0.0` - `1.0`

Fraction of the population that survives each generation. The rest are replaced by offspring.

**Example:**

```typescript
// Conservative: 80% survive (slow evolution)
const glstm = new GeneLSTM(300, {
    SURVIVORS: 0.8,
});

// Aggressive: 40% survive (fast evolution, higher variance)
const glstm = new GeneLSTM(300, {
    SURVIVORS: 0.4,
});
```

**Guidelines:**

- **0.7-0.8**: Stable, slow evolution
- **0.5-0.6**: Balanced (recommended)
- **0.3-0.4**: Fast, exploratory

### `MUTATION_RATE`

**Type:** `number`  
**Default:** `1.0`

Global mutation probability multiplier. Scales all mutation probabilities.

**Example:**

```typescript
// Half mutation rate
const glstm = new GeneLSTM(300, {
    MUTATION_RATE: 0.5,
});

// Double mutation rate
const glstm = new GeneLSTM(300, {
    MUTATION_RATE: 2.0,
});
```

---

## Weight Mutation Parameters

Weight mutations adjust the connection strengths in the LSTM gates.

### `WEIGHT_SHIFT_STRENGTH`

**Type:** `number`  
**Default:** `0.2`

Maximum magnitude for incremental weight adjustments.

**Example:**

```typescript
// Fine-tuning phase
const glstm = new GeneLSTM(300, {
    WEIGHT_SHIFT_STRENGTH: 0.05, // Small adjustments
});

// Exploration phase
const glstm = new GeneLSTM(300, {
    WEIGHT_SHIFT_STRENGTH: 0.5, // Large adjustments
});
```

### `WEIGHT_RANDOM_STRENGTH`

**Type:** `number`  
**Default:** `1.0`

Range for complete weight randomization `[-WEIGHT_RANDOM_STRENGTH, +WEIGHT_RANDOM_STRENGTH]`.

**Example:**

```typescript
const glstm = new GeneLSTM(300, {
    WEIGHT_RANDOM_STRENGTH: 2.0, // Weights reset to [-2, +2]
});
```

### `PROBABILITY_MUTATE_WEIGHT_SHIFT`

**Type:** `number`  
**Default:** `0.95`  
**Range:** `0.0` - `1.0`

Probability of applying incremental weight shift per mutation attempt.

**Example:**

```typescript
// Frequent small weight adjustments
const glstm = new GeneLSTM(300, {
    PROBABILITY_MUTATE_WEIGHT_SHIFT: 0.99,
    WEIGHT_SHIFT_STRENGTH: 0.1,
});
```

### `PROBABILITY_MUTATE_WEIGHT_RANDOM`

**Type:** `number`  
**Default:** `0.05`  
**Range:** `0.0` - `1.0`

Probability of complete weight randomization per mutation attempt.

**Example:**

```typescript
// More random exploration
const glstm = new GeneLSTM(300, {
    PROBABILITY_MUTATE_WEIGHT_RANDOM: 0.15,
});
```

**Weight Mutation Strategy Example:**

```typescript
// Fine-tuning configuration
const fineTuning = new GeneLSTM(300, {
    PROBABILITY_MUTATE_WEIGHT_SHIFT: 0.99, // Almost always shift
    PROBABILITY_MUTATE_WEIGHT_RANDOM: 0.01, // Rarely randomize
    WEIGHT_SHIFT_STRENGTH: 0.05, // Small shifts
    WEIGHT_RANDOM_STRENGTH: 0.5, // Small range if randomized
});

// Exploration configuration
const exploration = new GeneLSTM(300, {
    PROBABILITY_MUTATE_WEIGHT_SHIFT: 0.8,
    PROBABILITY_MUTATE_WEIGHT_RANDOM: 0.2, // More randomization
    WEIGHT_SHIFT_STRENGTH: 0.3,
    WEIGHT_RANDOM_STRENGTH: 2.0, // Large range
});
```

---

## Bias Mutation Parameters

Bias mutations adjust the activation thresholds in LSTM gates.

### `BIAS_SHIFT_STRENGTH`

**Type:** `number`  
**Default:** `0.2`

Maximum magnitude for incremental bias adjustments.

### `BIAS_RANDOM_STRENGTH`

**Type:** `number`  
**Default:** `1.0`

Range for complete bias randomization.

### `PROBABILITY_MUTATE_BIAS_SHIFT`

**Type:** `number`  
**Default:** `0.8`  
**Range:** `0.0` - `1.0`

Probability of incremental bias adjustment.

### `PROBABILITY_MUTATE_BIAS_RANDOM`

**Type:** `number`  
**Default:** `0.1`  
**Range:** `0.0` - `1.0`

Probability of complete bias randomization.

**Example:**

```typescript
// Bias-focused evolution
const glstm = new GeneLSTM(300, {
    PROBABILITY_MUTATE_BIAS_SHIFT: 0.95,
    BIAS_SHIFT_STRENGTH: 0.3,
    PROBABILITY_MUTATE_BIAS_RANDOM: 0.05,
});
```

---

## Skip Connection (Alpha) Mutation

Alpha controls the skip connection strength between input and output of LSTM blocks.

### `ALPHA_SHIFT_STRENGTH`

**Type:** `number`  
**Default:** `0.01`

Maximum magnitude for alpha adjustments. Typically smaller than weight/bias shifts.

### `PROBABILITY_MUTATE_ALPHA_SHIFT`

**Type:** `number`  
**Default:** `0.05`  
**Range:** `0.0` - `1.0`

Probability of adjusting the skip connection strength.

**Example:**

```typescript
// Enable skip connection exploration
const glstm = new GeneLSTM(300, {
    PROBABILITY_MUTATE_ALPHA_SHIFT: 0.15,
    ALPHA_SHIFT_STRENGTH: 0.02,
});
```

---

## Topology Mutation Parameters

Topology mutations modify the network architecture itself.

### `PROBABILITY_MUTATE_LSTM_BLOCK`

**Type:** `number`  
**Default:** `0.01`  
**Range:** `0.0` - `1.0`

Probability of attempting a block-level mutation (add or remove).

**Example:**

```typescript
// Allow more architecture changes
const glstm = new GeneLSTM(300, {
    PROBABILITY_MUTATE_LSTM_BLOCK: 0.05,
});
```

### `PROBABILITY_ADD_BLOCK_APPEND`

**Type:** `number`  
**Default:** `0.92`  
**Range:** `0.0` - `1.0`

When adding a block, probability of appending (vs. other strategies).

### `PROBABILITY_REMOVE_BLOCK`

**Type:** `number`  
**Default:** `0.1`  
**Range:** `0.0` - `1.0`

Probability of removing a block when block mutation occurs.

**Example:**

```typescript
// Prefer growing networks
const glstm = new GeneLSTM(300, {
    PROBABILITY_ADD_BLOCK_APPEND: 0.95,
    PROBABILITY_REMOVE_BLOCK: 0.05,
});

// Allow more pruning
const glstm = new GeneLSTM(300, {
    PROBABILITY_ADD_BLOCK_APPEND: 0.7,
    PROBABILITY_REMOVE_BLOCK: 0.3,
});
```

### `PROBABILITY_MUTATE_ADD_UNIT`

**Type:** `number`  
**Default:** `0.02`  
**Range:** `0.0` - `1.0`

Probability of adding a hidden unit to a block.

### `PROBABILITY_MUTATE_REMOVE_UNIT`

**Type:** `number`  
**Default:** `0.02`  
**Range:** `0.0` - `1.0`

Probability of removing a hidden unit from a block.

**Example:**

```typescript
// Encourage wider blocks
const glstm = new GeneLSTM(300, {
    PROBABILITY_MUTATE_ADD_UNIT: 0.08,
    PROBABILITY_MUTATE_REMOVE_UNIT: 0.01,
});
```

---

## Readout Layer Mutation

The readout layer produces final outputs from LSTM hidden states.

### `PROBABILITY_MUTATE_READOUT_W`

**Type:** `number`  
**Default:** `1.0`  
**Range:** `0.0` - `1.0`

Probability of mutating readout weights.

### `PROBABILITY_MUTATE_READOUT_B`

**Type:** `number`  
**Default:** `0.6`  
**Range:** `0.0` - `1.0`

Probability of mutating readout bias.

**Example:**

```typescript
// Focus on output layer
const glstm = new GeneLSTM(300, {
    PROBABILITY_MUTATE_READOUT_W: 1.0,
    PROBABILITY_MUTATE_READOUT_B: 0.9,
});
```

---

## Sleeping Block Configuration

Sleeping blocks are initialized with nearly-dormant weights for stable evolution.

### `sleepingBlockConfig`

**Type:** `Partial<SleepingBlockConfig>`  
**Default:** See below

```typescript
interface SleepingBlockConfig {
    epsilon: number; // Small weight range
    forgetBias: number; // Positive: remember everything
    inputBias: number; // Negative: write little
    outputBias: number; // Neutral
    candidateBias: number; // Neutral
    initialAlpha: number; // Skip connection initial value
}
```

**Defaults:**

```typescript
{
    epsilon: 0.002,
    forgetBias: 1.5,
    inputBias: -1.5,
    outputBias: 0.0,
    candidateBias: 0.0,
    initialAlpha: 0.01,
}
```

**Example:**

```typescript
// More dormant initialization
const glstm = new GeneLSTM(300, {
    sleepingBlockConfig: {
        epsilon: 0.001, // Smaller weights
        forgetBias: 2.0, // Remember more
        inputBias: -2.0, // Write even less
        initialAlpha: 0.005, // Weaker skip connection
    },
});

// More active initialization
const glstm = new GeneLSTM(300, {
    sleepingBlockConfig: {
        epsilon: 0.01, // Larger weights
        forgetBias: 1.0, // Forget more
        inputBias: -1.0, // Write more
        initialAlpha: 0.05, // Stronger skip connection
    },
});
```

**Parameter Guide:**

- **epsilon**: Initial weight magnitude (`±epsilon`)
- **forgetBias**: High positive → gates stay closed → remember more
- **inputBias**: Negative → write less to cell state
- **outputBias**: Controls output gate threshold
- **candidateBias**: Controls candidate activation threshold
- **initialAlpha**: Skip connection strength (0 = no skip, 1 = full bypass)

---

## Dynamic Speciation

Automatically adjusts the compatibility parameter to maintain a target number of species.

### `targetSpecies`

**Type:** `number`  
**Default:** Auto-calculated based on population:

- ≤100 clients: `5` species
- ≤500 clients: `8` species
- \>500 clients: `10` species

Target number of species to maintain.

### `cpAdjustRate`

**Type:** `number`  
**Default:** `0.2`

Rate of CP adjustment per generation (0-1). Higher values = faster adjustment.

### `cpDeadband`

**Type:** `number`  
**Default:** `1`

Tolerance range around target where no adjustment occurs (prevents oscillation).

### `minCP` / `maxCP`

**Type:** `number`  
**Default:** `0.01` / `10.0`

Bounds for CP adjustment.

**Example:**

```typescript
// Maintain exactly 6 species
const glstm = new GeneLSTM(300, {
    targetSpecies: 6,
    cpAdjustRate: 0.3, // Adjust faster
    cpDeadband: 0, // Strict targeting (5.5-6.5 acceptable)
    minCP: 0.05,
    maxCP: 5.0,
    verbose: 2, // See adjustment logs
});
```

**How it Works:**

1. Count species after each generation
2. If count > target: increase CP (merge species)
3. If count < target: decrease CP (split species)
4. Adjustment: `CP *= (1 + cpAdjustRate * error/target)`

---

## Mutation Pressure System

Dynamically adjusts mutation intensity based on fitness progress.

### `mutationPressure`

**Type:** `EMutationPressure`  
**Default:** `EMutationPressure.NORMAL`

Initial mutation pressure level:

```typescript
enum EMutationPressure {
    COMPACT = 'COMPACT', // Minimal mutations, favor simplicity
    NORMAL = 'NORMAL', // Balanced
    BOOST = 'BOOST', // Increased mutations
    ESCAPE = 'ESCAPE', // High exploration
    PANIC = 'PANIC', // Maximum mutations
}
```

**Pressure Effects:**

| Pressure | Topology Multiplier | Weights Multiplier |
| -------- | ------------------- | ------------------ |
| COMPACT  | 0.1x                | 0.8x               |
| NORMAL   | 1.0x                | 1.0x               |
| BOOST    | 1.2x                | 1.5x               |
| ESCAPE   | 1.5x                | 2.0x               |
| PANIC    | 2.0x                | 4.0x               |

### `enablePressureEscalation`

**Type:** `boolean`  
**Default:** `true`

Enable automatic pressure escalation when fitness stagnates.

### `stagnationThreshold`

**Type:** `number`  
**Default:** `15`

Generations without improvement before escalating pressure.

**Example:**

```typescript
// Start conservative, escalate if stuck
const glstm = new GeneLSTM(300, {
    mutationPressure: EMutationPressure.COMPACT,
    enablePressureEscalation: true,
    stagnationThreshold: 20,
});

// Fixed high pressure (no adaptation)
const glstm = new GeneLSTM(300, {
    mutationPressure: EMutationPressure.BOOST,
    enablePressureEscalation: false,
});
```

**Escalation Logic:**

1. Start at initial pressure level
2. If fitness improves: gradually reduce pressure → NORMAL
3. If stagnant for N generations: escalate (NORMAL → BOOST → ESCAPE → PANIC)
4. PANIC has timeout (30 gens) and cooldown (60 gens)
5. Growing complexity + tiny fitness gain → switch to COMPACT

**Complete Example:**

```typescript
import { GeneLSTM, EMutationPressure } from '@leoni4/gene-lstm-js';

const glstm = new GeneLSTM(500, {
    mutationPressure: EMutationPressure.NORMAL,
    enablePressureEscalation: true,
    stagnationThreshold: 15,
    verbose: 2, // See pressure changes
});

// Pressure automatically adjusts during evolution
for (let i = 0; i < 1000; i++) {
    // ... evaluate fitness ...
    glstm.evolve();
    // Console shows pressure transitions:
    // [Gen 50] Mutation Pressure: NORMAL → BOOST (stagnated 15 gens)
    // [Gen 75] Mutation Pressure: BOOST → NORMAL (fitness improved)
}
```

---

## Pre-trained Models

### `loadData`

**Type:** `GeneOptions`  
**Default:** `undefined`

Load a pre-trained model instead of random initialization.

**Example:**

```typescript
// Export trained model
const trainedModel = glstm.model();
// Save to file: JSON.stringify(trainedModel)

// Later, load it
import { PRE_TRAINED_DATA } from './saved-model.js';

const glstm = new GeneLSTM(1, {
    loadData: PRE_TRAINED_DATA,
});

// Use immediately
const result = glstm.clients[0].calculate(input);
```

**Model Format:**

```typescript
type GeneOptions = LstmOptions[];

interface LstmOptions {
    hiddenSize: number;
    forgetGate: GateUnitOptions[];
    potentialLongToRem: GateUnitOptions[];
    potentialLongMemory: GateUnitOptions[];
    shortMemoryToRemember: GateUnitOptions[];
    readoutW: number[];
    readoutB: number;
    alpha: number;
}
```

---

## Logging

### `verbose`

**Type:** `number`  
**Default:** `0`

Logging verbosity level:

- **0**: Silent
- **1**: Basic progress
- **2**: Detailed (species, CP, pressure changes)

**Example:**

```typescript
// Detailed logging
const glstm = new GeneLSTM(300, {
    verbose: 2,
});

glstm.evolve();
// Output:
// [Gen 1] CP: 0.1000 → 0.1200 (↑ INCREASE) | Species: 8/5 | Error: +3
// [Gen 1] Mutation Pressure: NORMAL (stagnation: 1)
// ### Species: 8 | Complexity: depth=240 (avg 0.80) units=480 (avg/block 2.00)
// # 0.95 42
// # 0.87 38
// ...
```

---

## Complete Example

Here's a comprehensive configuration demonstrating all major options:

```typescript
import { GeneLSTM, EMutationPressure } from '@leoni4/gene-lstm-js';

const glstm = new GeneLSTM(500, {
    // ===== Basic Configuration =====
    INPUT_FEATURES: 8,

    // ===== Speciation =====
    CP: 0.12,
    C1: 1.2,
    C2: 0.5,

    // ===== Evolution =====
    SURVIVORS: 0.65,
    MUTATION_RATE: 1.0,

    // ===== Weight Mutations =====
    WEIGHT_SHIFT_STRENGTH: 0.25,
    WEIGHT_RANDOM_STRENGTH: 1.5,
    PROBABILITY_MUTATE_WEIGHT_SHIFT: 0.92,
    PROBABILITY_MUTATE_WEIGHT_RANDOM: 0.08,

    // ===== Bias Mutations =====
    BIAS_SHIFT_STRENGTH: 0.25,
    BIAS_RANDOM_STRENGTH: 1.2,
    PROBABILITY_MUTATE_BIAS_SHIFT: 0.85,
    PROBABILITY_MUTATE_BIAS_RANDOM: 0.12,

    // ===== Alpha Mutations =====
    ALPHA_SHIFT_STRENGTH: 0.015,
    PROBABILITY_MUTATE_ALPHA_SHIFT: 0.08,

    // ===== Topology Mutations =====
    PROBABILITY_MUTATE_LSTM_BLOCK: 0.03,
    PROBABILITY_ADD_BLOCK_APPEND: 0.88,
    PROBABILITY_REMOVE_BLOCK: 0.12,
    PROBABILITY_MUTATE_ADD_UNIT: 0.05,
    PROBABILITY_MUTATE_REMOVE_UNIT: 0.03,

    // ===== Readout Mutations =====
    PROBABILITY_MUTATE_READOUT_W: 1.0,
    PROBABILITY_MUTATE_READOUT_B: 0.7,

    // ===== Sleeping Block Config =====
    sleepingBlockConfig: {
        epsilon: 0.003,
        forgetBias: 1.8,
        inputBias: -1.8,
        outputBias: 0.0,
        candidateBias: 0.0,
        initialAlpha: 0.015,
    },

    // ===== Dynamic Speciation =====
    targetSpecies: 8,
    cpAdjustRate: 0.25,
    cpDeadband: 1,
    minCP: 0.05,
    maxCP: 8.0,

    // ===== Mutation Pressure =====
    mutationPressure: EMutationPressure.NORMAL,
    enablePressureEscalation: true,
    stagnationThreshold: 18,

    // ===== Logging =====
    verbose: 2,
});

// Training loop
for (let epoch = 0; epoch < 1000; epoch++) {
    // Evaluate fitness
    for (const client of glstm.clients) {
        // ... your evaluation logic ...
        client.score = evaluateFitness(client);
    }

    // Evolve
    glstm.evolve();

    // Monitor progress
    if (epoch % 50 === 0) {
        glstm.printSpecies();
        console.log('Champion score:', glstm.champion?.score);
        console.log('Mutation pressure:', glstm.mutationPressure);
    }
}
```

---

## Quick Start Presets

### Preset 1: Fast Exploration

```typescript
const glstm = new GeneLSTM(200, {
    SURVIVORS: 0.4,
    PROBABILITY_MUTATE_WEIGHT_RANDOM: 0.15,
    PROBABILITY_MUTATE_LSTM_BLOCK: 0.05,
    mutationPressure: EMutationPressure.BOOST,
});
```

### Preset 2: Stable Refinement

```typescript
const glstm = new GeneLSTM(500, {
    SURVIVORS: 0.8,
    WEIGHT_SHIFT_STRENGTH: 0.1,
    PROBABILITY_MUTATE_WEIGHT_SHIFT: 0.99,
    PROBABILITY_MUTATE_WEIGHT_RANDOM: 0.01,
    PROBABILITY_MUTATE_LSTM_BLOCK: 0.005,
    mutationPressure: EMutationPressure.COMPACT,
});
```

### Preset 3: Balanced Evolution

```typescript
const glstm = new GeneLSTM(300, {
    INPUT_FEATURES: 5,
    SURVIVORS: 0.6,
    enablePressureEscalation: true,
    targetSpecies: 6,
    verbose: 1,
});
```

---

## Tips and Best Practices

1. **Start Simple**: Use defaults first, adjust only when needed
2. **Monitor Species**: Too many/few? Adjust `CP` or enable dynamic speciation
3. **Pressure Escalation**: Enable for automatic adaptation to difficult problems
4. **Complexity Control**: Use COMPACT pressure to prevent bloat
5. **Verbose Logging**: Use `verbose: 2` during development, `verbose: 0` in production
6. **Population Size**:
    - Small problems: 100-300
    - Medium problems: 300-500
    - Large problems: 500-1000+

7. **Survival Rate**:
    - Noisy fitness: 0.7-0.8 (more stability)
    - Smooth fitness: 0.4-0.6 (faster evolution)

8. **Topology Mutations**:
    - Start low (0.01-0.02)
    - Increase if stuck in local optimum
    - Decrease if networks grow too complex

9. **Weight vs. Bias**:
    - Weights affect connections
    - Biases affect thresholds
    - Usually mutate weights more frequently

10. **Champion Re-insertion**: Automatically happens after 10 generations of stagnation

---

For more examples, see the [demo folder](../demo/) in the repository.
