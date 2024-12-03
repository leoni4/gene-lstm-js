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

    const glstm = new GeneLSTM(1);
    const lstm = new LSTM(glstm, options);

    const out = lstm.calculate(trainingData.inputs[0], false);

    const out2 = lstm.calculate(trainingData.inputs[1], false);
    console.log('---- PRE TRAINED -----');
    console.log('out1', out, `// should be ${trainingData.outputs[0]}`);
    console.log('out2', out2, `// should be ${trainingData.outputs[1]}`);
    console.log('---- ----------- -----');
};
usePreTrained();

const usetraining = () => {
    const glstm = new GeneLSTM(100);
    glstm.printSpecies();
};
usetraining();

// console.log('---- START TRAIN -----');
// let realLSTM1 = new LSTM(glstm);
// let realLSTM2 = new LSTM(glstm);
// let epoch = 0;
// const train = async () => {
//     epoch++;
//     let error1 = 0;
//     let error2 = 0;
//     let bestLSTM;
//     trainingData.inputs.forEach((input, i) => {
//         const out = realLSTM1.calculate(input, false)[0];
//         error1 += Math.abs(out - trainingData.outputs[i]);
//         const out2 = realLSTM2.calculate(input, false)[0];
//         error2 += Math.abs(out2 - trainingData.outputs[i]);
//     });
//     error1 /= trainingData.inputs.length;
//     error2 /= trainingData.inputs.length;
//     const error = Math.min(error1, error2);
//     console.log('epoch:', epoch, '- error:', error, '#', error1 < error2 ? 1 : 2);
//     if (error1 < error2) {
//         bestLSTM = realLSTM1;
//         realLSTM2 = new LSTM(glstm, realLSTM1.model());
//         realLSTM2.mutate();
//     } else {
//         bestLSTM = realLSTM2;
//         realLSTM1 = new LSTM(glstm, realLSTM2.model());
//         realLSTM1.mutate();
//     }

//     if (epoch > 100000 || error < 0.01) {
//         console.log('---- TRAINED -----');
//         const out = bestLSTM.calculate(trainingData.inputs[0], false);
//         const out2 = bestLSTM.calculate(trainingData.inputs[1], false);
//         console.log('out1', out, `// should be ${trainingData.outputs[0]}`);
//         console.log('out2', out2, `// should be ${trainingData.outputs[1]}`);
//         console.log('---- ----------- -----');
//         return;
//     }
//     setTimeout(train, 1);
// };
// train();
