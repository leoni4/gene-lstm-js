import { GeneLSTM } from '../index';
import { LSTM } from '../lstm';

import type { LstmOptions } from '../types/index';

const trainingData = {
    inputs: [
        [0, 0.5, 0.25, 1],
        [1, 0.5, 0.25, 1],
    ],
    outputs: [0, 1],
};

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

const glstm = new GeneLSTM(10);
const lstm = new LSTM(glstm, options);

const out = lstm.calculate(trainingData.inputs[0], false);

const out2 = lstm.calculate(trainingData.inputs[1], false);
console.log('---- PRE TRAINED -----');
console.log('out1', out, `// should be ${trainingData.outputs[0]}`);
console.log('out2', out2, `// should be ${trainingData.outputs[1]}`);
console.log('---- ----------- -----');

console.log('---- START TRAIN -----');
const realLSTM = new LSTM(glstm);
let epoch = 0;
const train = async () => {
    epoch++;
    realLSTM.mutate();
    let error = 0;
    trainingData.inputs.forEach((input, i) => {
        const out = realLSTM.calculate(input, false)[0];
        error += Math.abs(out - trainingData.outputs[i]);
    });
    error /= trainingData.inputs.length;
    console.log('epoch:', epoch, '- error:', error);

    if (epoch > 1000 || error < 0.01) {
        console.log('---- TRAINED -----');
        return;
    }
    setTimeout(train, 1);
};
