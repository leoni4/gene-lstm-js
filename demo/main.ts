import { GeneLSTM } from '../src/gLstm.js';
import {
    lastBit,
    testLstmSineNext01,
    testLstmAdding01,
    testLstmParity01,
    testLstmTrend01,
    testLstmWaveMix01,
    testHierarchicalSegmentMajorityAdd,
    testHierarchicalSegmentXorAdd,
} from './problems.js';
import type { LstmOptions } from '../src/types/index.js';

// ========== Types ==========
interface Complexity {
    blocks: number;
    totalUnits: number;
    avgUnitsPerBlock: number;
}

// ========== Problem Definitions ==========
const problems = {
    lastBit: {
        build: () => lastBit.build(),
        inputFeatures: 1,
        description:
            '<strong>Last Bit:</strong> Learn to output the last bit of a 4-bit binary sequence. Simple memory task with 16 samples. Perfect for quick testing.',
    },
    sineNext: {
        build: () => testLstmSineNext01.build(),
        inputFeatures: 1,
        description:
            '<strong>Sine Wave Prediction:</strong> Predict the next value in a sine wave sequence. Tests the ability to learn periodic patterns. 512 samples, sequence length 20.',
    },
    adding: {
        build: () => testLstmAdding01.build(),
        inputFeatures: 2,
        description:
            '<strong>Adding Task:</strong> Add two marked values in a sequence of random numbers. Classic LSTM benchmark testing long-term dependencies. 2048 samples, sequence length 40.',
    },
    parity: {
        build: () => testLstmParity01.build(),
        inputFeatures: 1,
        description:
            '<strong>Parity Check:</strong> Calculate XOR (parity) of a binary sequence. Tests sequential logic and memory. 2048 samples, sequence length 32.',
    },
    trend: {
        build: () => testLstmTrend01.build(),
        inputFeatures: 1,
        description:
            '<strong>Trend Detection:</strong> Determine if a noisy time series is trending up or down. Tests pattern recognition in noisy data. 3000 samples, sequence length 30.',
    },
    waveMix: {
        build: () => testLstmWaveMix01.build(),
        inputFeatures: 1,
        description:
            '<strong>Wave Mix Prediction:</strong> Predict next value in a complex wave formed by mixing multiple sine waves. Tests learning of complex periodic patterns. 2048 samples, sequence length 25.',
    },
    hierarchicalMajority: {
        build: () => testHierarchicalSegmentMajorityAdd.build(),
        inputFeatures: 3,
        description:
            '<strong>Hierarchical Majority:</strong> Segment-wise majority voting with hierarchical structure. Each segment has marked values to add; output is 1 if majority of segments exceed threshold. Tests hierarchical reasoning. 512 samples, 4 segments of 5 timesteps each.',
    },
    hierarchicalXor: {
        build: () => testHierarchicalSegmentXorAdd.build(),
        inputFeatures: 3,
        description:
            '<strong>Hierarchical XOR:</strong> Segment-wise XOR parity with hierarchical structure. Each segment has marked values; compute local parity then global XOR. Tests hierarchical logic. 512 samples, 4 segments of 5 timesteps each.',
    },
};

// ========== Global State ==========
let glstm: GeneLSTM | null = null;
let isTraining = false;
let shouldStop = false;
let currentProblem = 'lastBit';
let dataset = problems[currentProblem].build();

// ========== DOM Elements ==========
const trainBtn = document.getElementById('trainBtn') as HTMLButtonElement;
const stopBtn = document.getElementById('stopBtn') as HTMLButtonElement;
const maxEpochsInput = document.getElementById('maxEpochs') as HTMLInputElement;
const errorThresholdInput = document.getElementById('errorThreshold') as HTMLInputElement;
const logContainer = document.getElementById('logContainer') as HTMLDivElement;
const modelViz = document.getElementById('modelViz') as HTMLDivElement;
const copyModelBtn = document.getElementById('copyModelBtn') as HTMLButtonElement;
const blockDetails = document.getElementById('blockDetails') as HTMLDivElement;
const closeDetailsBtn = document.getElementById('closeDetailsBtn') as HTMLButtonElement;
const problemSelect = document.getElementById('problemSelect') as HTMLSelectElement;
const problemDescription = document.getElementById('problemDescription') as HTMLDivElement;

// ========== Utility Functions ==========
function log(message: string, type: 'info' | 'success' | 'warning' | 'error' = 'info') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logContainer.appendChild(entry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function updateMetric(id: string, value: string | number) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = typeof value === 'number' ? value.toFixed(4) : value;
    }
}

function computeComplexity(model: LstmOptions[]): Complexity {
    const blocks = model.length;
    const totalUnits = model.reduce((acc, lstm) => acc + (lstm.hiddenSize || 1), 0);
    const avgUnitsPerBlock = blocks > 0 ? totalUnits / blocks : 0;

    return { blocks, totalUnits, avgUnitsPerBlock };
}

// ========== Visualization Functions ==========
function renderModelVisualization(model: LstmOptions[]) {
    modelViz.innerHTML = '';

    const blocksStrip = document.createElement('div');
    blocksStrip.className = 'blocks-strip';

    model.forEach((lstm, index) => {
        const blockCard = createBlockCard(lstm, index);
        blocksStrip.appendChild(blockCard);
    });

    modelViz.appendChild(blocksStrip);
}

function createBlockCard(lstm: LstmOptions, index: number): HTMLDivElement {
    const card = document.createElement('div');
    card.className = 'block-card';
    card.dataset.index = String(index);

    const readoutPreview = lstm.readoutW
        .slice(0, 6)
        .map(w => w.toFixed(2))
        .join(', ');
    const readoutSuffix = lstm.readoutW.length > 6 ? '...' : '';

    card.innerHTML = `
        <div class="block-header">
            <span class="block-index">Block ${index}</span>
        </div>
        <div class="block-info">
            <span class="block-info-label">Hidden Size:</span>
            <span class="block-info-value">${lstm.hiddenSize}</span>
        </div>
        <div class="block-info">
            <span class="block-info-label">Alpha:</span>
            <span class="block-info-value">${lstm.alpha.toFixed(3)}</span>
        </div>
        <div class="readout-preview">
            <div class="block-info-label">ReadoutW:</div>
            <div class="readout-weights">${readoutPreview}${readoutSuffix}</div>
        </div>
    `;

    card.addEventListener('click', () => showBlockDetails(lstm, index));

    return card;
}

function showBlockDetails(lstm: LstmOptions, index: number) {
    const detailsTitle = document.getElementById('blockDetailsTitle');
    const detailsContent = document.getElementById('blockDetailsContent');

    if (!detailsTitle || !detailsContent) return;

    detailsTitle.textContent = `Block ${index} - Detailed View`;
    detailsContent.innerHTML = '';

    // ReadoutW Bar Chart
    const barSection = document.createElement('div');
    barSection.className = 'bar-chart-section';
    barSection.innerHTML = '<h4>Readout Weights</h4>';
    const barChart = renderBarChart(lstm.readoutW);
    barSection.appendChild(barChart);
    detailsContent.appendChild(barSection);

    // Gate Heatmaps
    const gates = [
        { name: 'Forget Gate', data: lstm.forgetGate },
        { name: 'Potential Long To Remember', data: lstm.potentialLongToRem },
        { name: 'Potential Long Memory', data: lstm.potentialLongMemory },
        { name: 'Short Memory To Remember', data: lstm.shortMemoryToRemember },
    ];

    gates.forEach(gate => {
        const heatmapSection = document.createElement('div');
        heatmapSection.className = 'heatmap-section';
        heatmapSection.innerHTML = `<h4>${gate.name}</h4>`;
        const canvas = renderHeatmap(gate.data);
        heatmapSection.appendChild(canvas);
        detailsContent.appendChild(heatmapSection);
    });

    blockDetails.classList.remove('hidden');
}

function renderBarChart(weights: number[]): HTMLDivElement {
    const container = document.createElement('div');
    container.className = 'bar-chart';

    const maxAbsValue = Math.max(...weights.map(w => Math.abs(w)), 0.01);

    weights.forEach(weight => {
        const bar = document.createElement('div');
        bar.className = 'bar';
        if (weight < 0) bar.classList.add('negative');

        const height = (Math.abs(weight) / maxAbsValue) * 100;
        bar.style.height = `${height}%`;

        const valueLabel = document.createElement('span');
        valueLabel.className = 'bar-value';
        valueLabel.textContent = weight.toFixed(3);
        bar.appendChild(valueLabel);

        container.appendChild(bar);
    });

    return container;
}

function renderHeatmap(
    gateData: Array<{ weight1: number; weight2: number; bias: number; weightIn?: number[] }>,
): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.className = 'heatmap-canvas';

    const units = gateData.length;
    if (units === 0) return canvas;

    // Extract all weights from gate units
    const weights: number[][] = [];
    let maxFeatures = 0;

    gateData.forEach(unit => {
        const row: number[] = [unit.weight1, unit.weight2, unit.bias];
        if (unit.weightIn && Array.isArray(unit.weightIn)) {
            row.push(...unit.weightIn);
        }
        weights.push(row);
        maxFeatures = Math.max(maxFeatures, row.length);
    });

    // Canvas sizing
    const cellSize = 25;
    const padding = 40;
    const width = maxFeatures * cellSize + padding * 2;
    const height = units * cellSize + padding * 2;

    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return canvas;

    // Background
    ctx.fillStyle = '#2d323e';
    ctx.fillRect(0, 0, width, height);

    // Find global min/max for color scaling
    let minVal = Infinity;
    let maxVal = -Infinity;
    weights.forEach(row => {
        row.forEach(val => {
            minVal = Math.min(minVal, val);
            maxVal = Math.max(maxVal, val);
        });
    });

    // Color mapping function
    const getColor = (value: number): string => {
        const absMax = Math.max(Math.abs(minVal), Math.abs(maxVal), 0.01);
        const normalized = value / absMax; // -1 to 1

        if (normalized > 0) {
            // Positive: gray to red
            const intensity = Math.floor(normalized * 200);

            return `rgb(${100 + intensity}, ${50}, ${50})`;
        } else if (normalized < 0) {
            // Negative: gray to blue
            const intensity = Math.floor(-normalized * 200);

            return `rgb(${50}, ${50}, ${100 + intensity})`;
        } else {
            // Near zero: gray
            return 'rgb(80, 80, 80)';
        }
    };

    // Draw cells
    weights.forEach((row, rowIdx) => {
        row.forEach((value, colIdx) => {
            const x = padding + colIdx * cellSize;
            const y = padding + rowIdx * cellSize;

            ctx.fillStyle = getColor(value);
            ctx.fillRect(x, y, cellSize - 1, cellSize - 1);
        });
    });

    // Labels
    ctx.fillStyle = '#9aa0a6';
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';

    // Column labels
    const labels = ['w1', 'w2', 'b', ...Array.from({ length: maxFeatures - 3 }, (_, i) => `in${i}`)];
    labels.forEach((label, i) => {
        const x = padding + i * cellSize + cellSize / 2;
        ctx.fillText(label, x, padding - 10);
    });

    // Row labels
    ctx.textAlign = 'right';
    weights.forEach((_, i) => {
        const y = padding + i * cellSize + cellSize / 2 + 3;
        ctx.fillText(`u${i}`, padding - 10, y);
    });

    return canvas;
}

async function sleep(num: number = 0) {
    return new Promise(resolve => setTimeout(resolve, num));
}

// ========== Training Loop ==========
async function trainModel() {
    if (isTraining) return;

    isTraining = true;
    shouldStop = false;
    trainBtn.disabled = true;
    stopBtn.disabled = false;

    const maxEpochs = parseInt(maxEpochsInput.value);
    const errorThreshold = parseFloat(errorThresholdInput.value);

    log(`Starting training (max ${maxEpochs} epochs, threshold: ${errorThreshold})`, 'info');

    // Initialize GeneLSTM with correct INPUT_FEATURES for current problem
    const inputFeatures = problems[currentProblem].inputFeatures;
    glstm = new GeneLSTM(250, {
        INPUT_FEATURES: inputFeatures,
        verbose: 0,
    });

    let epoch = 0;
    let bestError = Infinity;
    let bestScore = 0;
    let handledCalcs = 0;
    const ALLOWED_CALCS = 5000;

    while (epoch < maxEpochs && !shouldStop) {
        // Evaluate all clients
        for (const client of glstm.clients) {
            let totalError = 0;

            for (let i = 0; i < dataset.inputs.length; i++) {
                handledCalcs++;
                if (handledCalcs >= ALLOWED_CALCS) {
                    await sleep(0);
                    handledCalcs = 0;
                }
                const input = dataset.inputs[i];
                const target = dataset.outputs[i];
                const prediction = client.calculate(input)[0];

                // MAE loss
                const error = Math.abs(prediction - target);
                totalError += error;
            }

            const avgError = totalError / dataset.inputs.length;
            client.error = avgError;

            // Score: higher is better
            client.score = 1 / (1 + avgError);
        }

        // Find best client
        glstm.clients.sort((a, b) => b.score - a.score);
        const bestClient = glstm.clients[0];
        bestError = bestClient.error;
        bestScore = bestClient.score;

        // Update metrics
        updateMetric('metricEpoch', epoch);
        updateMetric('metricError', bestError);
        updateMetric('metricScore', bestScore);
        updateMetric('metricSpecies', glstm['_species'].length);
        updateMetric('metricPressure', glstm.mutationPressure);
        updateMetric('metricCP', glstm.CP);

        const complexity = computeComplexity(bestClient.genome.lstmArray.map(l => l.model()));
        updateMetric('metricBlocks', complexity.blocks);
        updateMetric('metricUnits', complexity.totalUnits);
        updateMetric('metricAvgUnits', complexity.avgUnitsPerBlock);

        // Update visualization every 10 epochs or if champion changed
        if (epoch % 10 === 0 || epoch < 5) {
            const championModel = glstm.champion
                ? glstm.champion.genome.lstmArray.map(l => l.model())
                : bestClient.genome.lstmArray.map(l => l.model());
            renderModelVisualization(championModel);
        }

        // Log progress
        if (epoch % 50 === 0 || epoch < 5) {
            log(
                `Epoch ${epoch}: error=${bestError.toFixed(4)}, blocks=${complexity.blocks}, units=${complexity.totalUnits}`,
                'info',
            );
        }

        // Check early stopping
        if (bestError <= errorThreshold) {
            log(`Training completed! Reached error threshold at epoch ${epoch}`, 'success');
            break;
        }

        // Evolve
        const shouldOptimize = bestError <= 0.05;
        glstm.evolve(shouldOptimize);

        epoch++;

        // Non-blocking: yield every 5 iterations
        if (epoch % 5 === 0) {
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }

    if (shouldStop) {
        log('Training stopped by user', 'warning');
    } else if (epoch >= maxEpochs) {
        log(`Training completed: max epochs reached`, 'info');
    }

    // Final visualization
    if (glstm) {
        const finalModel = glstm.champion
            ? glstm.champion.genome.lstmArray.map(l => l.model())
            : glstm.clients[0].genome.lstmArray.map(l => l.model());
        renderModelVisualization(finalModel);
        log(`Final: error=${bestError.toFixed(4)}, score=${bestScore.toFixed(4)}`, 'success');
    }

    isTraining = false;
    trainBtn.disabled = false;
    stopBtn.disabled = true;
}

function stopTraining() {
    shouldStop = true;
    stopBtn.disabled = true;
}

function copyModelToClipboard() {
    if (!glstm) {
        log('No model to copy', 'error');

        return;
    }

    const model = glstm.champion
        ? glstm.champion.genome.lstmArray.map(l => l.model())
        : glstm.clients[0].genome.lstmArray.map(l => l.model());
    const json = JSON.stringify(model, null, 2);

    navigator.clipboard
        .writeText(json)
        .then(() => {
            log('Model JSON copied to clipboard', 'success');
        })
        .catch(err => {
            log(`Failed to copy: ${err.message}`, 'error');
        });
}

// ========== Problem Selection ==========
function handleProblemChange() {
    if (isTraining) {
        log('Cannot change problem while training', 'warning');
        problemSelect.value = currentProblem;

        return;
    }

    currentProblem = problemSelect.value as keyof typeof problems;
    const problem = problems[currentProblem];

    // Update dataset
    dataset = problem.build();

    // Update description
    problemDescription.innerHTML = problem.description;

    // Clear visualization
    modelViz.innerHTML = '<p class="placeholder">Start training to see the champion model...</p>';

    // Clear log and add new entries
    logContainer.innerHTML = '';
    log('Problem changed', 'info');
    log(
        `Dataset: ${currentProblem} (${dataset.inputs.length} samples, ${problem.inputFeatures} input features)`,
        'info',
    );
}

// ========== Event Listeners ==========
trainBtn.addEventListener('click', trainModel);
stopBtn.addEventListener('click', stopTraining);
copyModelBtn.addEventListener('click', copyModelToClipboard);
closeDetailsBtn.addEventListener('click', () => {
    blockDetails.classList.add('hidden');
});
problemSelect.addEventListener('change', handleProblemChange);

// Close details on background click
blockDetails.addEventListener('click', e => {
    if (e.target === blockDetails) {
        blockDetails.classList.add('hidden');
    }
});

// ========== Initialization ==========
log('Demo initialized. Click "Start Training" to begin.', 'info');
log(
    `Dataset: ${currentProblem} (${dataset.inputs.length} samples, ${problems[currentProblem].inputFeatures} input features)`,
    'info',
);
