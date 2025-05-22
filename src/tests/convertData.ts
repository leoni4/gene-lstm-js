type PreparedData = {
    input: number[]; // Вход в сеть (scaled log-returns)
    target: number; // Целевое значение (scaled)
    actual: number; // Фактическая цена (P[t+1])
    decode: (scaledY: number) => number; // Расшифровка в цену
};

function logTransform(data: number[]): number[] {
    const logReturns: number[] = [];
    for (let i = 1; i < data.length; i++) {
        logReturns.push(Math.log(data[i] / data[i - 1]));
    }
    return logReturns;
}

function getScaleFactor(logReturns: number[]): number {
    const maxAbs = Math.max(...logReturns.map(x => Math.abs(x)));
    return 1 / maxAbs;
}

function scaleData(data: number[], scaleFactor: number): number[] {
    return data.map(x => x * scaleFactor);
}

export function generateSlidingWindows(prices: number[], windowSize?: number): PreparedData[] {
    const logReturns = logTransform(prices); // длина на 1 меньше
    const result: PreparedData[] = [];

    const maxWindowSize = logReturns.length - 1;
    const effectiveWindowSize = windowSize && windowSize <= maxWindowSize ? windowSize : maxWindowSize;

    for (let i = 0; i < logReturns.length - effectiveWindowSize; i++) {
        const rawInput = logReturns.slice(i, i + effectiveWindowSize);
        const targetRaw = logReturns[i + effectiveWindowSize]; // log(P[t+1]/P[t])

        const scaleFactor = getScaleFactor(rawInput.concat([targetRaw])); // учитываем и цель
        const input = scaleData(rawInput, scaleFactor);
        const target = targetRaw * scaleFactor;

        const lastKnownPrice = prices[i + effectiveWindowSize]; // P[t]
        const actualNextPrice = prices[i + effectiveWindowSize + 1]; // P[t+1]

        result.push({
            input,
            target,
            actual: actualNextPrice,
            decode: (scaledY: number) => lastKnownPrice * Math.exp(scaledY / scaleFactor),
        });

        // if (i === 0) {
        //     console.log('--- Window 0 ---');
        //     console.log('Input:', input);
        //     console.log('Target (raw):', targetRaw);
        //     console.log('Scale factor:', scaleFactor);
        //     console.log('P[t] (base):', lastKnownPrice);
        //     console.log('P[t+1] (actual):', actualNextPrice);
        //     console.log('Decoded from target:', lastKnownPrice * Math.exp(target / scaleFactor));
        // }
    }

    return result;
}

// const prices = [
//     10000, 10020, 10010, 10030, 10040, 10060, 10055, 10070, 10090, 10100, 10120, 10130, 10110, 10140, 10150, 10160,
//     10180, 10200, 10210, 10230, 10240, 10220, 10250, 10260, 10270, 10280, 10300, 10320, 10340, 10360, 10380,
// ];

// const trainingSet = generateSlidingWindows(prices, 5);

// console.log('prices:', prices);
// //console.log('trainingSet:', trainingSet);

// console.log('Input:', trainingSet[0].input);
// console.log('Raw output:', trainingSet[0].target);
// console.log('Predicted next price:', trainingSet[0].decode(trainingSet[0].target));
