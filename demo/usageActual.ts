import { GeneLSTM } from '../src/index.js';
import { generateSlidingWindows } from './convertData.js';
import { testData, topModel } from './DATA.js';

const sleep = (num = 0) => new Promise(resolve => setTimeout(resolve, num));

const train = (glstm: GeneLSTM, data: any) => {
    let epoch = 0;
    let iter = 0;
    let bestClient: any;
    let time = Date.now();
    const EPOCHS = 1000;
    return new Promise(resolve => {
        const session = async () => {
            epoch++;
            let error = Infinity;
            for (let c = 0; c < glstm.clients.length; c++) {
                const client = glstm.clients[c];
                client.bestScore = false;
                let localError = 0;
                for (let t = 0; t < data.length; t++) {
                    const input = data[t].input;
                    const output = data[t].target;

                    iter++;
                    const out = client.calculate(input)[0];
                    localError += Math.abs(out - output);
                    if (Math.sign(out) !== Math.sign(output)) {
                        if (Math.sign(out) > 0 && Math.sign(output) < 0) {
                            localError += 1;
                        } else {
                            localError += 1;
                        }
                    }
                    if (iter % 1000 === 0) {
                        await sleep(0);
                    }
                }
                client.error = localError / data.length;
                client.score = 1 - client.error;
                if (error > client.error) {
                    error = client.error;
                    bestClient = client;
                }
            }
            bestClient.bestScore = true;
            console.log('Epoch:', epoch, ' Error:', error);
            if (epoch % 10 === 0) {
                glstm.printSpecies();
            }
            if (epoch % 100 === 0 || Date.now() - time > 1000 * 60 * 15) {
                time = Date.now();
                logResults(glstm, data);
                printModel(glstm);
            }

            if (epoch >= EPOCHS || error < 0.01) {
                resolve(error);
                return;
            }
            glstm.evolve();
            session();
        };
        session();
    });
};

const logResults = async (glstm: GeneLSTM, data: typeof testData) => {
    const client = glstm.clients[0];
    let localError = 0; // MAE (scaled)
    let mapeTotal = 0; // MAPE (%)
    let correctDirection = 0;

    let TP = 0; // true positive: predicted ↑, actual ↑
    let TN = 0; // true negative: predicted ↓, actual ↓
    let FP = 0; // false positive: predicted ↑, actual ↓
    let FN = 0; // false negative: predicted ↓, actual ↑

    for (let t = 0; t < data.length; t++) {
        const input = data[t].input;
        const target = data[t].target;
        const decode = data[t].decode;

        const out = client.calculate(input)[0];
        localError += Math.abs(out - target); // MAE

        const predictedPrice = decode(out);
        const actualPrice = decode(target);
        mapeTotal += Math.abs((actualPrice - predictedPrice) / actualPrice);

        const predSign = Math.sign(out);
        const realSign = Math.sign(target);

        if (predSign === realSign) {
            correctDirection++;
        }

        if (realSign > 0 && predSign > 0) TP++;
        else if (realSign < 0 && predSign < 0) TN++;
        else if (realSign < 0 && predSign > 0) FP++;
        else if (realSign > 0 && predSign < 0) FN++;

        if (t % 1000 === 0) {
            await sleep(0);
        }
    }

    const total = data.length;
    const mae = localError / total;
    const mape = (mapeTotal / total) * 100;
    const directionalAccuracy = (correctDirection / total) * 100;

    const precision = TP + FP === 0 ? 0 : TP / (TP + FP);
    const recall = TP + FN === 0 ? 0 : TP / (TP + FN);
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);

    console.log('MAE (scaled):', mae.toFixed(6));
    console.log('MAPE (%):', mape.toFixed(2));
    console.log('Directional Accuracy (%):', directionalAccuracy.toFixed(2));
    console.log('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN);
    console.log('Precision (for UP):', (precision * 100).toFixed(2) + '%');
    console.log('Recall (for UP):', (recall * 100).toFixed(2) + '%');
    console.log('F1-score:', (f1 * 100).toFixed(2) + '%');
};

const printModel = (glstm: GeneLSTM) => {
    console.log('-- MODEL --');
    console.log(glstm.model());
};

const usetraining = async () => {
    const glstm = new GeneLSTM(300, {
        //  MUTATION_RATE: 1,
        loadData: topModel,
    });
    let c = glstm.clients[0];
    glstm.printSpecies();

    const finalData = generateSlidingWindows(testData)[0];
    const trainData = generateSlidingWindows(testData, 90);

    console.log('---- PRE TRAINED -----');
    let out = c.calculate(finalData.input);
    console.log('out', finalData.decode(out[0]), `// should be ${finalData.actual}`);
    console.log('---- ----------- -----');

    console.log('---- START TRAIN -----');
    await logResults(glstm, trainData);
    await train(glstm, trainData);
    console.log('---- END TRAIN -----');

    c = glstm.clients[0];

    console.log('---- TRAINED -----');
    out = c.calculate(finalData.input);
    console.log('out', finalData.decode(out[0]), `// should be ${finalData.actual}`);
    console.log('---- ----------- -----');
    printModel(glstm);
};

const useTest = async () => {
    console.log('---- LOG -----');

    const glstm = new GeneLSTM(1, {
        loadData: topModel,
    });
    const data = generateSlidingWindows(testData, 90);
    await logResults(glstm, data);
    console.log('----     -----');
};

const test = false;

if (test) {
    useTest();
} else {
    usetraining();
}
