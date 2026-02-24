import { GeneLSTM, Client } from '../src/index.js';
import type { LstmOptions } from '../src/types/index.js';
import {
    testLstmSineNext01,
    testLstmAdding01,
    testLstmParity01,
    testLstmTrend01,
    testLstmWaveMix01,
    testHierarchicalSegmentXorAdd,
} from './problems.js';

console.log(
    !testLstmSineNext01,
    !testLstmAdding01,
    !testLstmParity01,
    !testLstmTrend01,
    !testLstmWaveMix01,
    !testHierarchicalSegmentXorAdd,
);

const trainingData = {
    inputs: [
        [0, 0.5, 0.25, 1],
        [1, 0.5, 0.25, 1],
    ],
    outputs: [0, 1],
};

process.on('unhandledRejection', (reason: any) => {
    console.error('UNHANDLED REJECTION:', reason);
    console.error('STACK:', reason?.stack);
});

process.on('uncaughtException', (err: any) => {
    console.error('UNCAUGHT EXCEPTION:', err);
    console.error('STACK:', err?.stack);
    process.exitCode = 1;
});

// const testData = {
//     inputs: [
//         [0.1, 0.5, 0.25, 1], // ожидание: ближе к 0
//         [0.9, 0.5, 0.25, 1], // ожидание: ближе к 1
//         [0.5, 0.5, 0.25, 1], // пограничное — интересно как поведёт себя сеть
//         [0, 0.4, 0.3, 1], // похож на 0, но шум
//         [1, 0.6, 0.2, 1], // похож на 1, но шум
//     ],
//     outputs: [0, 1, 0.5, 0, 1], // предположения, не обязательны
// };

const usePreTrained = () => {
    const options: LstmOptions = {
        forgetGate: {
            weight1: 2.7,
            weight2: 1.63,
            bias: 1.62,
        },
        potentialLongToRem: {
            weight1: 2,
            weight2: 1.65,
            bias: 0.62,
        },
        potentialLongMemory: {
            weight1: 1.41,
            weight2: 0.94,
            bias: -0.32,
        },
        shortMemoryToRemember: {
            weight1: 4.38,
            weight2: -0.19,
            bias: 0.59,
        },
    };

    const glstm = new GeneLSTM(1, {
        loadData: [options],
    });

    const c = glstm.clients[0];
    const out = c.calculate(trainingData.inputs[0]);
    const out2 = c.calculate(trainingData.inputs[1]);

    console.log('---- PRE TRAINED -----');
    console.log('out1', out, `// should be ${trainingData.outputs[0]}`);
    console.log('out2', out2, `// should be ${trainingData.outputs[1]}`);
    console.log('---- ----------- -----');
};
usePreTrained();

const sleep = (num = 0) => new Promise(resolve => setTimeout(resolve, num));

const train = (glstm: GeneLSTM, data = trainingData, epochsPasseg = 1000) => {
    let epoch = 0;
    let iter = 0;
    let bestClient: Client | undefined;
    const EPOCHS = epochsPasseg;
    const LAMBDA_MEAN = 0.05;

    return new Promise<number>(resolve => {
        const session = async () => {
            epoch++;

            let bestError = Infinity;

            for (let c = 0; c < glstm.clients.length; c++) {
                const client = glstm.clients[c];
                client.bestScore = false;

                let localErrorSum = 0;
                let predSum = 0;
                let classSolved = 0;

                for (let t = 0; t < data.inputs.length; t++) {
                    const input = data.inputs[t];
                    const target = data.outputs[t];

                    iter++;
                    const out = client.calculate(input)[0]; // убери "as any" если типы уже обновил

                    predSum += out;
                    localErrorSum += Math.abs(out - target);

                    if ((target === 1 && out > 0.5) || (target === 0 && out < 0.5)) classSolved++;

                    if (iter % 1000 === 0) await sleep(0);
                }

                const localError = localErrorSum / data.inputs.length;
                const meanPred = predSum / data.inputs.length;
                const penalty = LAMBDA_MEAN * Math.abs(meanPred - 0.5);
                const penaltyClass = LAMBDA_MEAN * Math.abs(classSolved / data.inputs.length);
                const finalError = Math.min(1, localError + penalty + penaltyClass);

                client.error = finalError;
                client.score = 1 - finalError;

                // ВАЖНО: выбираем best по finalError, иначе penalty не работает
                if (finalError < bestError) {
                    bestError = finalError;
                    bestClient = client;
                }
            }

            console.log('Epoch:', epoch, ' Error:', bestError);

            if (epoch % 10 === 0) glstm.printSpecies();

            if (epoch >= EPOCHS || bestError < 0.01) {
                resolve(bestError);

                return;
            }

            if (bestClient) bestClient.bestScore = true;

            glstm.evolve();
            session();
        };

        session();
    });
};

const usetraining = async () => {
    const glstm = new GeneLSTM(100, {
        INPUT_FEATURES: 3,
    });
    glstm.printSpecies();
    const data = testHierarchicalSegmentXorAdd.build();
    console.log('---- START TRAIN -----');

    await train(glstm, data, 1000);
    console.log('---- END TRAIN -----');

    const c = glstm.champion || glstm.clients[0];

    console.log('---- TRAINED -----');
    data.inputs.forEach((input, i) => {
        const out = c.calculate(input);
        console.log('input N', i, 'out', out, `// should be ${data.outputs[i]}`);
    });
    console.log('---- ----------- -----');
};

usetraining();
