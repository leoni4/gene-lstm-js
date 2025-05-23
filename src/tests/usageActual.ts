import { GeneLSTM } from '../index';
import { generateSlidingWindows } from './convertData';
import { testData, topModel } from './DATA';

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
            console.log('Epoch:', epoch, ' Error:', error);
            if (epoch % 10 === 0) {
                glstm.printSpecies();
            }
            if (epoch >= EPOCHS || error < 0.01) {
                resolve(error);
                return;
            }
            bestClient.bestScore = true;
            glstm.evolve();
            session();
        };
        session();
    });
};

const usetraining = async () => {
    const glstm = new GeneLSTM(1000, {
        // MUTATION_RATE: 10,
        loadData: topModel,
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
usetraining();
