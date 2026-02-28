# 🧬 Gene LSTM

A TypeScript implementation of evolutionary LSTM neural networks using genetic algorithms. This library combines Long Short-Term Memory (LSTM) networks with neuroevolution techniques to create adaptive, self-optimizing neural networks for sequence learning tasks.

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

### Basic Library Usage

```typescript
import { GeneLSTM } from '@leoni4/gene-lstm-js';

// Create a GeneLSTM instance
const glstm = new GeneLSTM(300);

// Training data (lastBit example)
export const lastBit = {
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

const solved = glstm.fit(lastBit.inputs, lastBit.outputs, {
    verbose: 2,
});

console.log('solved in:', solved.epochs);
```

### Manual Library Usage

```typescript
import { GeneLSTM } from '@leoni4/gene-lstm-js';

// Training data
const trainingData = {
    inputs: [
        [0, 0.5, 0.25, 1],
        [1, 0.5, 0.25, 1],
    ],
    outputs: [0, 1],
};

// Create a population of 300 clients
const glstm = new GeneLSTM(300, {
    INPUT_FEATURES: 4, // Number of input features
    verbose: 1, // Logging level
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
const prediction = champion.calculate([0.5, 0.5, 0.25, 1]);
console.log('Prediction:', prediction);
```

### Loading Pre-trained Models

```typescript
import { GeneLSTM } from '@leoni4/gene-lstm-js';
import { PRE_TRAINED_DATA } from './my-trained-model.js';

const glstm = new GeneLSTM(1, {
    loadData: PRE_TRAINED_DATA,
});

const result = glstm.clients[0].calculate([0, 0.5, 0.25, 1]);
console.log('Result:', result);
```

### Advanced Configuration

```typescript
import { GeneLSTM, EMutationPressure } from '@leoni4/gene-lstm-js';

const glstm = new GeneLSTM(500, {
    // Input configuration
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

- `evolve(optimization?: boolean)` - Evolve the population for one generation
- `printSpecies()` - Print current species statistics
- `adjustCP(speciesCount: number, generation?: number)` - Dynamically adjust compatibility parameter
- `updateMutationPressure(currentBestFitness: number, generation?: number)` - Update mutation pressure based on progress
- `model()` - Export the best model's architecture

**Properties:**

- `clients: Client[]` - All clients in the population
- `champion: Client | null` - Best performing client ever seen
- `mutationPressure: EMutationPressure` - Current mutation pressure level

#### `Client`

Represents an individual neural network in the population.

**Key Methods:**

- `calculate(input: SeqInput): number[]` - Forward pass through the network
- `mutate(force?: boolean)` - Mutate the client's genome
- `distance(client: Client): number` - Calculate genetic distance to another client

**Properties:**

- `genome: Genome` - The LSTM architecture and weights
- `score: number` - Fitness score (0-1)
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

- **Scalar mode**: `number[]` - Single input vector
- **Vector mode**: `number[][]` - Sequence of input vectors

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
