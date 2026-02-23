export const testLstmSineNext01 = {
    name: 'lstm_sine_next_01',
    clients: 200,
    inputs: [] as number[][],
    outputs: [] as number[],
    build({ samples = 512, seqLen = 20, step = 0.2, noise = 0.0 } = {}) {
        const inputs: number[][] = [];
        const outputs: number[] = [];

        for (let s = 0; s < samples; s++) {
            const start = Math.random() * 20 - 10;
            const seq: number[] = [];
            for (let t = 0; t < seqLen; t++) {
                const x = start + t * step;
                let y = Math.sin(x);
                if (noise) y += (Math.random() * 2 - 1) * noise;
                seq.push((y + 1) / 2);
            }
            const nextX = start + seqLen * step;
            let nextY = Math.sin(nextX);
            if (noise) nextY += (Math.random() * 2 - 1) * noise;
            outputs.push((nextY + 1) / 2);
            inputs.push(seq);
        }

        return { ...this, inputs, outputs };
    },
} as const;

export const testLstmAdding01 = {
    name: 'lstm_adding_01',
    clients: 300,
    inputs: [] as number[][],
    outputs: [] as number[],
    build({ samples = 2048, seqLen = 40 } = {}) {
        const inputs: number[][] = [];
        const outputs: number[] = [];

        for (let s = 0; s < samples; s++) {
            const values = Array.from({ length: seqLen }, () => Math.random()); // [0..1]
            const markers = new Array(seqLen).fill(0);

            const i1 = Math.floor(Math.random() * seqLen);
            let i2 = Math.floor(Math.random() * seqLen);
            while (i2 === i1) i2 = Math.floor(Math.random() * seqLen);
            markers[i1] = 1;
            markers[i2] = 1;

            const seq: number[] = [];
            for (let t = 0; t < seqLen; t++) {
                seq.push(values[t], markers[t]); // интерливинг
            }

            inputs.push(seq);
            outputs.push(values[i1] + values[i2]); // [0..2]
        }

        return { ...this, inputs, outputs };
    },
} as const;

export const testLstmParity01 = {
    name: 'lstm_parity_01',
    clients: 300,
    inputs: [] as number[][],
    outputs: [] as number[],
    build({ samples = 2048, seqLen = 32 } = {}) {
        const inputs: number[][] = [];
        const outputs: number[] = [];

        for (let s = 0; s < samples; s++) {
            let parity = 0;
            const seq: number[] = [];
            for (let t = 0; t < seqLen; t++) {
                const bit = Math.random() < 0.5 ? 0 : 1;
                parity ^= bit;
                seq.push(bit);
            }
            inputs.push(seq);
            outputs.push(parity); // 0/1
        }

        return { ...this, inputs, outputs };
    },
} as const;

export const testLstmTrend01 = {
    name: 'lstm_trend_01',
    clients: 250,
    inputs: [] as number[][],
    outputs: [] as number[],
    build({ samples = 3000, seqLen = 30, noise = 0.05 } = {}) {
        const inputs: number[][] = [];
        const outputs: number[] = [];

        for (let s = 0; s < samples; s++) {
            let x = Math.random() * 0.2; // старт
            const drift = (Math.random() * 2 - 1) * 0.02; // наклон
            const seq: number[] = [];
            for (let t = 0; t < seqLen; t++) {
                x = x + drift + (Math.random() * 2 - 1) * noise;
                seq.push(x);
            }
            // нормализация приблизительно в [0..1]
            // (можно убрать, если твоя модель норм умеет)
            const min = Math.min(...seq);
            const max = Math.max(...seq);
            const norm = seq.map(v => (v - min) / (max - min || 1));

            inputs.push(norm);
            outputs.push(norm[seqLen - 1] > norm[0] ? 1 : 0);
        }

        return { ...this, inputs, outputs };
    },
} as const;

export const testLstmWaveMix01 = {
    name: 'lstm_wavemix_next_01',
    clients: 300,
    inputs: [] as number[][],
    outputs: [] as number[],
    build({ samples = 2048, seqLen = 25, step = 0.15, noise = 0.0 } = {}) {
        const f = (x: number) =>
            0.55 * Math.sin(1.0 * x) + 0.3 * Math.sin(2.3 * x + 0.7) + 0.15 * Math.sin(4.7 * x + 1.9);

        const inputs: number[][] = [];
        const outputs: number[] = [];

        for (let s = 0; s < samples; s++) {
            const start = Math.random() * 20 - 10;
            const seq: number[] = [];
            for (let t = 0; t < seqLen; t++) {
                const x = start + t * step;
                let y = f(x);
                if (noise) y += (Math.random() * 2 - 1) * noise;
                seq.push((y + 1) / 2); // [0..1]
            }

            const nextX = start + seqLen * step;
            let nextY = f(nextX);
            if (noise) nextY += (Math.random() * 2 - 1) * noise;
            outputs.push((nextY + 1) / 2);

            inputs.push(seq);
        }

        return { ...this, inputs, outputs };
    },
} as const;

export const testHierarchicalSegmentXorAdd = {
    name: 'hierarchical_segment_xor_add',
    inputs: [] as number[][],
    outputs: [] as number[],
    build({
        samples = 512,
        seqLen = 20,
        segLen = 5,
        threshold = 1,
        noise = 0,
        valueMin = 0.0,
        valueMax = 1.0,
        seed = 0,
    } = {}) {
        const inputs: number[][][] = [];
        const outputs: number[] = [];

        if (seqLen % segLen !== 0) {
            throw new Error(`seqLen (${seqLen}) must be divisible by segLen (${segLen})`);
        }

        const segments = seqLen / segLen;

        let s = seed >>> 0;
        const rand = () => {
            s = (1664525 * s + 1013904223) >>> 0;

            return s / 0xffffffff;
        };

        const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

        const xor = (a: number, b: number) => (a ^ b) & 1;

        for (let sample = 0; sample < samples; sample++) {
            const sampleVec: number[][] = [];

            let finalParity = 0;

            for (let seg = 0; seg < segments; seg++) {
                let p1 = Math.floor(rand() * segLen);
                let p2 = Math.floor(rand() * segLen);
                while (p2 === p1) p2 = Math.floor(rand() * segLen);

                const values: number[] = [];
                for (let i = 0; i < segLen; i++) {
                    let v = valueMin + (valueMax - valueMin) * rand();

                    if (noise > 0) {
                        v += (rand() * 2 - 1) * noise;
                    }

                    v = clamp01(v);
                    values.push(v);
                }

                const localSum = values[p1] + values[p2];
                const localBit = localSum > threshold ? 1 : 0;

                finalParity = xor(finalParity, localBit);

                for (let i = 0; i < segLen; i++) {
                    const value = values[i];
                    const marker = i === p1 || i === p2 ? 1 : 0;
                    const segStart = i === 0 ? 1 : 0;

                    const value11 = value * 2 - 1; // [0..1] -> [-1..1]
                    const marker11 = marker ? 1 : -1; // {0,1} -> {-1,1}
                    const segStart11 = segStart ? 1 : -1;
                    sampleVec.push([value11, marker11, segStart11]);
                }
            }

            inputs.push(sampleVec);
            outputs.push(finalParity);
        }
        const up = outputs.filter(a => a === 1).length;
        const down = outputs.length - up;
        console.log(`1=${up}, 0=${down}`);

        return { ...this, inputs, outputs };
    },
} as const;
