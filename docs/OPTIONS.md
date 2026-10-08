# GeneLSTM Configuration Options

Complete reference guide for all configuration options available in the `GeneLSTMOptions` interface.

## Table of Contents

- [Overview](#overview)
- [Basic Configuration](#basic-configuration)
- [Speciation Parameters](#speciation-parameters)
- [Evolution Parameters](#evolution-parameters)
- [Selection Score and Complexity Penalty](#selection-score-and-complexity-penalty)
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
    OUTPUT_DIM?: number;
    OUTPUT_ACTIVATION?: 'sigmoid' | 'tanh' | 'identity';
    SURVIVORS?: number;
    MUTATION_RATE?: number;

    // Selection score and complexity penalty
    OPT_ERR_THRESHOLD?: number;
    OPTIMIZATION_PERIOD?: number;
    LAMBDA_HIGH?: number;
    LAMBDA_LOW?: number;
    EPS?: number;

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
    loadPercent?: number;

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

Number of input features per time step of 2-D input (`number[][]`).

The input shape decides how the first block reads the input:

- `number[]` is a sequence of T time steps with one scalar value in each step. `INPUT_FEATURES` has no effect on this input.
- `number[][]` is a sequence of T time steps. Each step is a row of features, and each row must have `INPUT_FEATURES` values.

The library does not check the row width. During `calculate()`, a unit whose input-weight count is not equal to the row width gets new random input weights, without an error. New sleeping blocks and weight mutations on a unit without input weights create `INPUT_FEATURES` input weights, so with a wrong `INPUT_FEATURES` these weights become random weights. For example, the small weights (`±epsilon`) of a new sleeping block become weights in `[-1, 1]`.

**Example:**

```typescript
// 1-D input: a sequence of 3 scalar time steps (INPUT_FEATURES is not used)
const glstm1 = new GeneLSTM(100);
const output1 = glstm1.clients[0].calculate([0.5, 0.3, 0.8]);

// 2-D input: a sequence of 2 time steps with 3 features in each step
const glstm2 = new GeneLSTM(100, {
    INPUT_FEATURES: 3,
});
const output2 = glstm2.clients[0].calculate([
    [0.5, 0.3, 0.8], // step 1
    [0.1, 0.2, 0.4], // step 2
]);
```

**Use Cases:**

- Time series with multiple sensors: `INPUT_FEATURES: 5`
- Financial data (OHLCV): `INPUT_FEATURES: 5`
- Single value prediction: `INPUT_FEATURES: 1`

### `OUTPUT_DIM`

**Type:** `number`  
**Default:** `1`

Number of outputs. `calculate()` returns an array with `OUTPUT_DIM` values. Each block has `OUTPUT_DIM` readout rows.

With 2 or more blocks, each block except the last one gives `OUTPUT_DIM` values for each time step to the next block, as one flat scalar sequence with T × `OUTPUT_DIM` steps. The [skip connection](#skip-connection-alpha-mutation) changes only output 0.

`model()` does not save this value. Use the same value when you load a model: a smaller value removes readout rows without an error, and a larger value adds rows with zero weights.

### `OUTPUT_ACTIVATION`

**Type:** `'sigmoid' | 'tanh' | 'identity'`  
**Default:** `'sigmoid'`

Activation function of the readout of each block (also the blocks before the last one).

`model()` does not save this value. Use the same value when you load a model. For example, a model trained with `'identity'` and loaded with the default `'sigmoid'` gives different outputs.

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

Fraction of each species that survives each generation. The rest are replaced by offspring.

Exact rule: each species sorts its clients by `score` and keeps the first ⌊`SURVIVORS` × n⌋ + 1 clients (n = species size, at most n clients). It never removes the client with the highest raw score of the generation (`bestScore`). So with `SURVIVORS: 0.6`, a species of 10 clients keeps 7 clients.

The same value also changes how parent species are selected: the random value of the score-proportional selection is multiplied by `SURVIVORS`, so with a value below 1 the species with the highest scores are selected more often than their score share.

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

Global mutation probability multiplier. It scales the probabilities of the weight, bias, alpha, and readout mutations, and of the block mutation (`PROBABILITY_MUTATE_LSTM_BLOCK`). It does not scale the add-unit and remove-unit probabilities (`PROBABILITY_MUTATE_ADD_UNIT`, `PROBABILITY_MUTATE_REMOVE_UNIT`).

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

## Selection Score and Complexity Penalty

Before each `evolve()` call, each client must have a fitness in `client.score` (a higher value is better). `fit()` sets `score = 1 / (1 + error)`. With a manual loop, you set it.

`evolve()` uses the scores in this sequence:

1. A `NaN` or `±Infinity` score makes `evolve()` throw an error.
2. It adds tie-breaker noise smaller than `1e-9` to `score` and copies the value to `scoreRaw`. The champion and the mutation pressure use `scoreRaw`.
3. It calculates `complexity = blocks + 0.25 × (sum of the hidden units of all blocks)`. For example, 2 blocks with 60 hidden units each give `2 + 0.25 × 120 = 32`. `OUTPUT_DIM` does not change this value. (Versions 1.0.5–1.0.10 counted `OUTPUT_DIM` for each block instead of the hidden units.)
4. It calculates `adjustedScore = scoreRaw − λ × log1p(complexity) / log1p(maxComplexity) × max(rawSpan, 0.05)`. `maxComplexity` is the highest complexity in the population, and `rawSpan` is the difference between the highest and the lowest `scoreRaw`.
5. It replaces `score` with the normalized value `(adjustedScore − min) / (max − min)` in `[0, 1]`. If `max − min ≤ EPS`, all clients get `score = 1`. Then it sorts `clients` by `score`; when two scores have a difference of `EPS` or less, the client with the lower complexity comes first.

So after `evolve()`, `client.score` is a selection score, not your fitness. Read your fitness in `client.scoreRaw`.

λ is `LAMBDA_HIGH` in an optimization generation, and `LAMBDA_LOW` in other generations. A generation is an optimization generation when:

- you call `evolve(true)` (`fit()` does this when the best error of the epoch is `OPT_ERR_THRESHOLD` or less), or
- the number of `evolve()` calls is a multiple of `OPTIMIZATION_PERIOD`, or
- the mutation pressure changes to `COMPACT` in this generation.

### `OPT_ERR_THRESHOLD`

**Type:** `number`  
**Default:** `0.005`

Only `fit()` uses it. If the best error of an epoch is this value or less, `fit()` calls `evolve(true)`. Negative values become `0`.

### `OPTIMIZATION_PERIOD`

**Type:** `number`  
**Default:** `10`

Each `OPTIMIZATION_PERIOD`-th `evolve()` call is an optimization generation. The value is rounded down; values below `1` become `1`.

### `LAMBDA_HIGH`

**Type:** `number`  
**Default:** `0.1`

Complexity penalty factor in optimization generations. Negative values become `0`.

### `LAMBDA_LOW`

**Type:** `number`  
**Default:** `0.01`

Complexity penalty factor in other generations. Negative values become `0`.

### `EPS`

**Type:** `number`  
**Default:** `1e-6`

Tolerance for the score normalization and the sort (see step 5). Values below `Number.EPSILON` become `Number.EPSILON`.

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

Strength of the weight randomization. A random-weight mutation changes one weight of one gate unit. `p` is the weights multiplier of the current [mutation pressure](#mutation-pressure-system) and `r` is a random value in `[-1, 1]`:

- Input weight (`weightIn`, selected with 30% probability when the unit has input weights): the new value is `r × WEIGHT_RANDOM_STRENGTH × p`, so the range is `[-S·p, +S·p]`.
- Recurrent weight (`weight1`) or scalar input weight (`weight2`): the new value is `r × |old value| × WEIGHT_RANDOM_STRENGTH × p`, limited to `[-10, 10]`. The range depends on the old value, and a weight with the value `0` stays `0`.

**Example:**

```typescript
const glstm = new GeneLSTM(300, {
    WEIGHT_RANDOM_STRENGTH: 2.0, // Input weights reset to [-2, +2] at NORMAL pressure
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

Each block has an `alpha` value in `[0, 1]`. Only the last block of a genome uses it. The last block returns output 0 as:

```
output[0] = (1 − alpha) × x_last + alpha × y[0]
```

- `y[0]` is readout output 0 of the last time step.
- `x_last` is the last input value of the last block:
    - genome with 1 block and 2-D input: feature 0 of the last time step;
    - genome with 1 block and 1-D input: the last scalar step;
    - genome with 2 or more blocks: the last value that the block before gives (its output `OUTPUT_DIM − 1` of the last time step).
- Outputs 1 to `OUTPUT_DIM − 1` have no skip term.

So `alpha = 1` means no skip, and `alpha = 0` means that output 0 is equal to `x_last`. When `alpha < 1` in a genome with 1 block and 2-D input, feature 0 of the last step goes directly into output 0, so the order of the input features is part of the model.

Initial values: a new random block starts with `alpha = 1`. A new sleeping block starts with `sleepingBlockConfig.initialAlpha` (default `0.01`), so an appended sleeping block first gives almost the value of the block before. The alpha of the other blocks has no effect on the output, but it mutates and `model()` saves it.

### `ALPHA_SHIFT_STRENGTH`

**Type:** `number`  
**Default:** `0.01`

Maximum magnitude for alpha adjustments: `alpha += r × ALPHA_SHIFT_STRENGTH × p` (`r` in `[-1, 1]`, `p` = weights multiplier of the mutation pressure), limited to `[0, 1]`.

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

Probability of attempting a block-level mutation (add or remove). For each genome, the probability is `PROBABILITY_MUTATE_LSTM_BLOCK × MUTATION_RATE × t` (`t` = topology multiplier of the mutation pressure). When the mutation occurs:

1. It selects "remove" with the probability `min(PROBABILITY_REMOVE_BLOCK × t, 0.9)`.
2. If it selects "remove" and the genome has 2 or more blocks, it removes the last block or the first block (50% each).
3. In all other cases (also "remove" with only 1 block), it adds a sleeping block: at the end with the probability `PROBABILITY_ADD_BLOCK_APPEND`, else at the start.

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

When adding a block, probability of appending it at the end. Otherwise the block is added at the start (prepend).

**Note on prepend:** a new sleeping block has zero readout weights and a zero readout bias. When it is added at the start, it gives a constant sequence to the old first block, so the output of the model does not depend on the input until readout mutations change these weights. Set `PROBABILITY_ADD_BLOCK_APPEND: 1` to disable prepend.

### `PROBABILITY_REMOVE_BLOCK`

**Type:** `number`  
**Default:** `0.1`  
**Range:** `0.0` - `1.0`

Probability of removing a block when block mutation occurs. The probability is multiplied by the topology multiplier of the mutation pressure and is limited to `0.9`. A genome with 1 block adds a block instead. The removed block is the last block or the first block (50% each); when the first block is removed, the old second block gets the user input directly.

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

Probability of adding a hidden unit to a block. Each block tests it once per mutation. The probability is multiplied by the topology multiplier of the mutation pressure; `MUTATION_RATE` does not change it. The new unit has random gate weights and zero readout weights.

### `PROBABILITY_MUTATE_REMOVE_UNIT`

**Type:** `number`  
**Default:** `0.02`  
**Range:** `0.0` - `1.0`

Probability of removing a hidden unit from a block. Each block tests it once per mutation. The probability is multiplied by the topology multiplier of the mutation pressure; `MUTATION_RATE` does not change it. It removes the unit with the smallest sum of absolute readout weights. A block with 1 unit does not change.

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
        initialAlpha: 0.005, // Output 0 is closer to the value of the block before
    },
});

// More active initialization
const glstm = new GeneLSTM(300, {
    sleepingBlockConfig: {
        epsilon: 0.01, // Larger weights
        forgetBias: 1.0, // Forget more
        inputBias: -1.0, // Write more
        initialAlpha: 0.05, // The new block has more effect on output 0
    },
});
```

**Parameter Guide:**

- **epsilon**: Initial weight magnitude (`±epsilon`)
- **forgetBias**: High positive → gates stay closed → remember more
- **inputBias**: Negative → write less to cell state
- **outputBias**: Controls output gate threshold
- **candidateBias**: Controls candidate activation threshold
- **initialAlpha**: Initial `alpha` of the block (0 = output 0 is only the skip value, 1 = no skip). See [Skip Connection](#skip-connection-alpha-mutation)

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

Generations without improvement before escalating pressure. The real threshold depends on the current level:

| Step           | Generations without improvement |
| -------------- | ------------------------------- |
| NORMAL → BOOST | `stagnationThreshold`           |
| BOOST → ESCAPE | `2 × stagnationThreshold`       |
| ESCAPE → PANIC | `4 × stagnationThreshold`       |

The counter starts again from 0 after each level change. An improvement is a raw best score (`scoreRaw`) higher than the best score before plus `max(1e-6, |best| × 1e-3)`. The value is rounded down; values below `1` become `1`.

**Example:**

```typescript
// Start conservative, escalate if stuck
const glstm = new GeneLSTM(300, {
    mutationPressure: EMutationPressure.COMPACT, // NORMAL after the first evolve() while escalation is on
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

1. Start at initial pressure level. The first `evolve()` call always counts as an improvement, so a start at `COMPACT` changes to `NORMAL` at the first call.
2. If fitness improves: reduce pressure by one level (PANIC → ESCAPE → BOOST → NORMAL; COMPACT → NORMAL)
3. If stagnant: escalate (NORMAL → BOOST → ESCAPE → PANIC) after the thresholds in the table of [`stagnationThreshold`](#stagnationthreshold)
4. PANIC has a timeout of 30 generations. Then the pressure changes to ESCAPE, and a cooldown of 60 generations starts. During the cooldown, the step ESCAPE → PANIC is blocked; each improvement halves the remaining cooldown. The cooldown counts down in each generation, and ESCAPE → PANIC needs `4 × stagnationThreshold` generations without improvement, so the cooldown blocks PANIC only when `stagnationThreshold` is less than 15. With the default value (15), it never blocks PANIC.
5. After more than 50 generations without improvement: if the [complexity](#selection-score-and-complexity-penalty) of the best client grew in the last 50 generations by 2 or more (for example, 8 new hidden units), or by `0.25 × max(c0, 8)` or more (`c0` is the complexity 50 generations ago), and the raw best score grew by `0.01` or less, the pressure changes to COMPACT, and this generation is an [optimization generation](#selection-score-and-complexity-penalty). At the next generation without improvement where this condition is false, the pressure changes back to NORMAL.

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
    // Console shows pressure transitions (verbose: 2), for example:
    // [Gen 50] Mutation Pressure: NORMAL → BOOST (stagnated for 15 generations, best: 0.912345)
    // [Gen 75] Mutation Pressure: BOOST → NORMAL (fitness improved to 0.923456)
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
    // model() does not save these values. Use the values of the training run.
    INPUT_FEATURES: 4,
    OUTPUT_DIM: 1,
    OUTPUT_ACTIVATION: 'identity',
});

// Use immediately
const result = glstm.clients[0].calculate(input);
```

`model()` exports the layers of the champion (if there is no champion, the client with the highest `score`). It also sorts `glstm.clients` by `score`.

**Options that `model()` does not save:** `OUTPUT_DIM`, `OUTPUT_ACTIVATION`, and `INPUT_FEATURES`. When you load a model, a block uses the `outputDim` and `outputActivation` fields of its `LstmOptions` if they exist, else the `GeneLSTM` options. Because `model()` does not write these fields, the `GeneLSTM` options apply. Use the values of the training run:

- `OUTPUT_ACTIVATION`: a different value gives different outputs.
- `OUTPUT_DIM`: a smaller value removes readout rows without an error; a larger value adds rows with zero weights.
- `INPUT_FEATURES`: it does not change the outputs of the loaded weights, but new sleeping blocks and some weight mutations use it (see [`INPUT_FEATURES`](#input_features)).

**Model Format:**

```typescript
type GeneOptions = LstmOptions[]; // one entry per block, first block first

type LstmOptions = {
    hiddenSize: number;
    forgetGate: GateUnitOptions[]; // hiddenSize entries, one per unit
    potentialLongToRem: GateUnitOptions[];
    potentialLongMemory: GateUnitOptions[];
    shortMemoryToRemember: GateUnitOptions[];
    readoutW: number[] | number[][]; // model() writes number[][]: OUTPUT_DIM rows × hiddenSize
    readoutB: number | number[]; // model() writes number[]: OUTPUT_DIM values
    alpha: number;
    outputDim?: number; // model() does not write it
    outputActivation?: 'sigmoid' | 'tanh' | 'identity'; // model() does not write it
};

type GateUnitOptions = {
    weight1: number; // recurrent weight
    weight2: number; // weight of a scalar input step
    bias: number;
    weightIn?: number[]; // weights of a 2-D input row
};
```

The old format with `readoutW: number[]` and `readoutB: number` still loads. Such a block has 1 output, independent of `OUTPUT_DIM`. A readout row whose length is not `hiddenSize` loads as a row of zeros.

### `loadPercent`

**Type:** `number`  
**Default:** `0.5`

Share of the population that starts with the loaded model. The first ⌈`loadPercent` × clients⌉ clients get their own copy of `loadData`; the other clients get new random genomes. The value `0` becomes `0.5`. A value of `1` or more loads the model into all clients; a negative value loads it into no client. Without `loadData`, the option has no effect.

---

## Logging

### `verbose`

**Type:** `number`  
**Default:** `0`

Logging verbosity level of `evolve()`:

- **0**: Silent
- **1**: Silent (the same as 0 for this option; `fit()` has its own `verbose` option)
- **2**: Detailed (champion updates, CP changes, pressure changes, elite re-insertion)

`evolve()` does not call `printSpecies()`. Call it yourself to see the species.

**Example:**

```typescript
// Detailed logging
const glstm = new GeneLSTM(300, {
    verbose: 2,
});

glstm.evolve();
// Output, for example:
// [Gen 1] Champion initialized: rawScore=0.948910445840, complexity=1.25
// [Gen 1] CP: 0.1000 → 0.2000 (↑ INCREASE) | Species: 30/5 | Error: +25

glstm.printSpecies();
// Output, for example:
// ### Species: 4 | Complexity: depth=30 (avg 1.00) units=32 (avg/block 1.07)
// # 0.463594421337993 15
// # 0.4475283662191467 11
// ...
// ###
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
    mutationPressure: EMutationPressure.COMPACT, // NORMAL after the first evolve() while escalation is on
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
