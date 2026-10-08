# 🧬 Gene LSTM

A TypeScript implementation of evolutionary LSTM neural networks using genetic algorithms. This library combines Long Short-Term Memory (LSTM) networks with neuroevolution techniques to create adaptive, self-optimizing neural networks for sequence learning tasks.

## 🎮 Live Demo

Try the interactive demo: **[https://leoni4.github.io/gene-lstm-js/](https://leoni4.github.io/gene-lstm-js/)**

The demo showcases Gene LSTM solving various problems including LastBit, parity functions, and classification tasks with real-time visualization of the neural network evolution.

## Features

- **Neuroevolution**: Evolves LSTM architectures and weights through genetic algorithms
- **Speciation**: Maintains diversity through automatic species clustering
- **Dynamic Mutation Pressure**: Automatically adjusts mutation intensity based on fitness stagnation
- **Adaptive Complexity**: Balances network complexity with performance
- **Sleeping Block Initialization**: Smart initialization strategy for stable training
- **TypeScript**: Full type safety and modern ES modules

## Installation

```bash
npm install @leoni4/gene-lstm-js
```

## Usage

### Quick Start with `fit()` Method

The `fit()` method provides a simple, Keras-style API for training Gene LSTM networks:

```typescript
import { GeneLSTM } from '@leoni4/gene-lstm-js';

// Create a GeneLSTM instance with 300 clients
const glstm = new GeneLSTM(300);

// Training data (lastBit example - predict the last bit in a sequence)
// Each input is a 1-D array: a sequence of 4 scalar time steps.
const lastBit = {
    inputs: [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ],
    outputs: [
        0, // last = 0
        1, // last = 1
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
    ],
};

// Train the network
const history = glstm.fit(lastBit.inputs, lastBit.outputs, {
    epochs: 1000,
    verbose: 2,
});

console.log('Training completed in:', history.epochs, 'epochs');
console.log('Final error:', history.error[history.error.length - 1]);

// Use the trained champion to make predictions
const prediction = history.champion!.calculate([1, 0, 1, 1]);
console.log('Prediction:', prediction);
```

### Manual Library Usage

```typescript
import { GeneLSTM } from '@leoni4/gene-lstm-js';

// Training data just random example
// Each input is a 2-D array: a sequence of time steps, each step has 4 features.
const trainingData = {
    inputs: [
        [
            [0, 0.5, 0.25, 1],
            [1, 0.5, 0.25, 0],
        ],
        [
            [1, 0.5, 0.25, 1],
            [0, 0.5, 0.25, 1],
        ],
    ],
    outputs: [0, 1],
};

// Create a population of 300 clients
const glstm = new GeneLSTM(300, {
    INPUT_FEATURES: 4, // Must be equal to the number of features in each step
    verbose: 0, // Logging level (GeneLSTM writes logs only at 2)
});

// Training loop
for (let epoch = 0; epoch < 1000; epoch++) {
    // Evaluate each client
    for (const client of glstm.clients) {
        let errorSum = 0;

        for (let i = 0; i < trainingData.inputs.length; i++) {
            const output = client.calculate(trainingData.inputs[i]);
            const error = Math.abs(output[0] - trainingData.outputs[i]);
            errorSum += error;
        }

        const avgError = errorSum / trainingData.inputs.length;
        client.score = 1 - avgError; // Higher score is better
    }

    // Evolve population
    glstm.evolve();

    if (epoch % 100 === 0) {
        console.log(`Epoch ${epoch}`);
        glstm.printSpecies();
    }
}

// Use the best performing network (champion)
const champion = glstm.champion || glstm.clients[0];
const prediction = champion.calculate([
    [0.5, 0.5, 0.25, 1],
    [1, 0.5, 0.25, 0],
]);
console.log('Prediction:', prediction);
```

Before each `evolve()` call, give each client a new `score` (a higher score is better). `evolve()` replaces `client.score` with a normalized selection score and keeps your value in `client.scoreRaw`. A `NaN` or `±Infinity` score makes `evolve()` throw an error.

### Loading Pre-trained Models

```typescript
import { GeneLSTM } from '@leoni4/gene-lstm-js';
import { PRE_TRAINED_DATA } from './my-trained-model.js';

const glstm = new GeneLSTM(1, {
    loadData: PRE_TRAINED_DATA,
    // Use the values of the training run. Models saved by version 1.0.10 or earlier need them.
    INPUT_FEATURES: 4,
    OUTPUT_DIM: 1,
    OUTPUT_ACTIVATION: 'sigmoid',
});

const result = glstm.clients[0].calculate([
    [0, 0.5, 0.25, 1],
    [1, 0.5, 0.25, 0],
]);
console.log('Result:', result);
```

`model()` saves `outputDim` and `outputActivation` in each block. A loaded block uses these saved values, also when the `GeneLSTM` options are different. Models saved by version 1.0.10 or earlier do not contain these fields: for them, the `GeneLSTM` options apply, and a model trained with `OUTPUT_ACTIVATION: 'identity'` and loaded with the default `'sigmoid'` gives different outputs. `model()` does not save `INPUT_FEATURES`. New random clients and new blocks always use the `GeneLSTM` options, so use the values of the training run when you continue to train. See [Pre-trained Models](./docs/OPTIONS.md#pre-trained-models).

### Advanced Configuration

```typescript
import { GeneLSTM, EMutationPressure } from '@leoni4/gene-lstm-js';

const glstm = new GeneLSTM(500, {
    // Input configuration (features per step of 2-D input)
    INPUT_FEATURES: 10,

    // Species parameters
    CP: 0.15, // Compatibility threshold
    targetSpecies: 8, // Target number of species

    // Evolution parameters
    SURVIVORS: 0.7, // Survival rate (70%)
    MUTATION_RATE: 1.0,

    // Mutation pressure (adaptive)
    mutationPressure: EMutationPressure.NORMAL,
    enablePressureEscalation: true,
    stagnationThreshold: 20,

    // Topology mutations
    PROBABILITY_MUTATE_LSTM_BLOCK: 0.02,
    PROBABILITY_ADD_BLOCK_APPEND: 0.9,
    PROBABILITY_REMOVE_BLOCK: 0.15,

    // Weight mutations
    PROBABILITY_MUTATE_WEIGHT_SHIFT: 0.95,
    WEIGHT_SHIFT_STRENGTH: 0.3,

    // Logging
    verbose: 2,
});
```

### Available Scripts

For development and testing:

```bash
# Run demo
npm run demo

# Start interactive demo with Vite
npm start

# Build the project
npm run build

# Run tests
npm test

# Run tests in watch mode
npm run test:watch

# Lint code
npm run lint

# Type checking
npm run typecheck

# Build demo for production
npm run build:demo
```

## API

### Core Classes

#### `GeneLSTM`

Main class for managing the evolutionary process.

**Constructor:**

```typescript
new GeneLSTM(clients: number, options?: GeneLSTMOptions)
```

**Key Methods:**

- **`fit(xTrain: SeqInput[], yTrain: SeqInput, options?: IGlstmFitOptions): IGlstmFitHistory`**

    High-level training method with automatic evolution and error tracking. Provides a Keras-like API for training.

    **Parameters:**
    - `xTrain`: Array of input sequences (can be 1D or 2D arrays)
    - `yTrain`: Array of target outputs (can be 1D numbers or 2D arrays)
    - `options`: Training configuration object

    **Options (`IGlstmFitOptions`):**

    ```typescript
    {
      epochs?: number;              // Maximum number of training epochs (default: Infinity)
      errorThreshold?: number;      // Stop when error below this value (default: 0.01)
      validationSplit?: number;     // Fraction of data for validation (default: 0)
      verbose?: 0 | 1 | 2;         // Logging level (default: 1)
                                    // 0 = silent, 1 = periodic, 2 = detailed
      logInterval?: number;         // Log every N epochs when verbose=1 (default: 100)

      loss?: 'mae' | 'mse' | 'bce'; // Loss function (default: 'mae')
                                     // mae = Mean Absolute Error
                                     // mse = Mean Squared Error
                                     // bce = Binary Cross Entropy

      antiConstantPenalty?: boolean;     // Penalize constant predictions (default: false)
      antiConstantLambda?: number;       // Penalty strength (default: 0.05)
      shuffleEachEpoch?: boolean;        // Shuffle training data (default: true)
    }
    ```

    The type also has `outputMode`. The library does not use it.

    **Returns (`IGlstmFitHistory`):**

    ```typescript
    {
      error: number[];              // Training error per epoch
      validationError?: number[];   // Validation error per epoch (if validationSplit > 0)
      epochs: number;               // Total epochs completed
      champion: Client | null;      // Best trained network
      stoppedEarly: boolean;        // True if stopped due to errorThreshold
    }
    ```

    **Example:**

    ```typescript
    const history = glstm.fit(xTrain, yTrain, {
        epochs: 1000,
        errorThreshold: 0.01,
        validationSplit: 0.2,
        verbose: 2,
        loss: 'mae',
        antiConstantPenalty: true,
    });

    console.log('Final error:', history.error[history.error.length - 1]);
    const prediction = history.champion!.calculate([1, 0, 1]);
    ```

- `evolve(optimization?: boolean)` - Evolve the population for one generation
- `printSpecies()` - Print current species statistics
- `adjustCP(speciesCount: number, generation?: number)` - Dynamically adjust compatibility parameter
- `updateMutationPressure(currentBestFitness: number, generation?: number)` - Update mutation pressure based on progress
- `getMutationPressure(): { topology: number; weights: number }` - Multipliers of the current pressure level
- `model(): GeneOptions` - Export the layers of the champion. If there is no champion, it exports the client with the highest `score`. It sorts `clients` by `score`.

**Properties:**

- `clients: Client[]` - All clients in the population
- `champion: Client | null` - Copy of the client with the highest raw score (`scoreRaw`) seen in all generations
- `runnerUp: Client | null` - Copy of the best client of the last generation after the complexity penalty
- `mutationPressure: EMutationPressure` - Current mutation pressure level

#### `Client`

Represents an individual neural network in the population.

**Key Methods:**

- `calculate(input: SeqInput): number[]` - Forward pass through the network
- `mutate(force?: boolean)` - Mutate the client's genome
- `distance(client: Client): number` - Calculate genetic distance to another client
- `model(): GeneOptions` - Export the layers of this client (same format as `GeneLSTM.model()`)

**Properties:**

- `genome: Genome` - The LSTM architecture and weights
- `score: number` - Before `evolve()`: the fitness that you set (or that `fit()` sets). After `evolve()`: the normalized selection score (0-1), after the complexity penalty
- `scoreRaw: number` - The fitness that `evolve()` received (your `score` plus tie-breaker noise smaller than 1e-9)
- `adjustedScore: number` - `scoreRaw` minus the complexity penalty
- `complexity: number` - Complexity value that the penalty uses
- `error: number` - Error that `fit()` calculated
- `bestScore: boolean` - `true` for the client with the highest `scoreRaw` in the generation; this client does not mutate and `kill` does not remove it
- `species: Species | null` - Species membership

#### `EMutationPressure`

Enum for mutation pressure levels:

- `COMPACT` - Minimal mutations, favor simplicity
- `NORMAL` - Balanced mutation rate
- `BOOST` - Increased mutations to escape local optima
- `ESCAPE` - High mutation rate for exploration
- `PANIC` - Maximum mutations when severely stuck

### Data Structures

#### `SeqInput`

Input format for LSTM calculations:

```typescript
type SeqInput = number[] | number[][];
```

- **Scalar mode**: `number[]` - A sequence of T time steps with one scalar value in each step. `INPUT_FEATURES` has no effect on this input.
- **Vector mode**: `number[][]` - A sequence of T time steps; each step is a row of features. Each row must have `INPUT_FEATURES` values. During `calculate()`, a unit whose input-weight count is not equal to the row width gets new random input weights, without an error. New sleeping blocks and weight mutations on a unit without input weights create `INPUT_FEATURES` input weights, so with a wrong `INPUT_FEATURES` these weights become random weights.

#### `GeneLSTMOptions`

Configuration object for GeneLSTM initialization. See [detailed options documentation](./docs/OPTIONS.md) for all available parameters.

#### `GeneOptions`

Pre-trained model data format:

```typescript
type GeneOptions = LstmOptions[];
```

Export model data using:

```typescript
const modelData = glstm.model();
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** your changes following [Conventional Commits](https://www.conventionalcommits.org/):
    - `feat: add new feature`
    - `fix: resolve bug`
    - `docs: update documentation`
    - `test: add tests`
4. **Test** your changes: `npm test`
5. **Lint** your code: `npm run lint`
6. **Push** to your fork: `git push origin feature/my-feature`
7. **Submit** a pull request to the `main` branch

### Development Setup

```bash
# Clone the repository
git clone https://github.com/leoni4/gene-lstm-js.git
cd gene-lstm-js

# Install dependencies
npm install

# Run tests
npm test

# Start development demo
npm start
```

## License

MIT © [Leonid Lilo](https://github.com/leoni4)

See [LICENSE](./LICENSE) for details.

## Repository

**GitHub**: [leoni4/gene-lstm-js](https://github.com/leoni4/gene-lstm-js)

**Issues**: [Report bugs or request features](https://github.com/leoni4/gene-lstm-js/issues)

**NPM**: [@leoni4/gene-lstm-js](https://www.npmjs.com/package/@leoni4/gene-lstm-js)

## Further Documentation

- [Detailed Options Reference](./docs/OPTIONS.md) - Complete guide to all configuration options
- [Examples](./demo/) - More usage examples and problem implementations
