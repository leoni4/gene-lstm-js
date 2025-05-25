import { GeneLSTM } from '../index';
import { generateSlidingWindows } from './convertData';
import { testData, topModel2 } from './DATA';

const sleep = (num = 0) => new Promise(resolve => setTimeout(resolve, num));

const train = (glstm: GeneLSTM, data: any) => {
    let epoch = 0;
    let iter = 0;
    let bestClient: any;
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
                        localError += 1;
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
            if (epoch % 100 === 0) {
                logResults(glstm, data);
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
    let correctDirection = 0; // directional accuracy
    let iter = 0;

    for (let t = 0; t < data.length; t++) {
        const input = data[t].input;
        const target = data[t].target;
        const decode = data[t].decode;

        iter++;

        const out = client.calculate(input)[0];
        localError += Math.abs(out - target); // MAE

        // MAPE: сравниваем реальные цены
        const predictedPrice = decode(out);
        const actualPrice = decode(target);
        mapeTotal += Math.abs((actualPrice - predictedPrice) / actualPrice);

        // Directional accuracy
        if (Math.sign(out) === Math.sign(target)) {
            correctDirection++;
        }

        if (iter % 1000 === 0) {
            await sleep(0); // не блокирует основной поток
        }
    }

    const mae = localError / data.length;
    const mape = (mapeTotal / data.length) * 100;
    const directionalAccuracy = (correctDirection / data.length) * 100;

    console.log('MAE (scaled):', mae.toFixed(6));
    console.log('MAPE (%):', mape.toFixed(2));
    console.log('Directional Accuracy (%):', directionalAccuracy.toFixed(2));
};

const usetraining = async () => {
    const glstm = new GeneLSTM(500, {
        // MUTATION_RATE: 10,
        loadData: topModel2,
    });
    glstm.printSpecies();

    console.log('---- START TRAIN -----');
    await train(glstm, generateSlidingWindows(testData, 90));
    console.log('---- END TRAIN -----');

    const c = glstm.clients[0];

    console.log('---- TRAINED -----');
    const finalData = generateSlidingWindows(testData)[0];
    const out = c.calculate(finalData.input);
    console.log('out', finalData.decode(out[0]), `// should be ${finalData.actual}`);
    console.log('---- ----------- -----');
    console.log('-- MODEL --');
    console.log(glstm.model());
};

const useTest = () => {
    const glstm = new GeneLSTM(1, {
        loadData: topModel2,
    });
    const data = generateSlidingWindows(testData, 90);
    logResults(glstm, data);
};

const test = false;

if (test) {
    useTest();
} else {
    usetraining();
}
