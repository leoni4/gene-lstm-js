export const testLstmSineNext01 = {
    name: 'lstm_sine_next_01',
    clients: 200,
    inputs: [] as number[][],
    outputs: [] as number[],
    build({ samples = 1024, seqLen = 20, step = 0.2, noise = 0.0 } = {}) {
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
