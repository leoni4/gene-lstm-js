import type { SeqInput } from '../types/index.js';

export function isY2D(y: SeqInput): y is number[][] {
    return Array.isArray(y[0]);
}

export function clamp01(v: number) {
    return Math.max(0, Math.min(1, v));
}

export function safeLog(v: number) {
    const x = Math.max(1e-12, Math.min(1 - 1e-12, v));

    return Math.log(x);
}

export function computeLoss(pred: number[], target: number[], loss: 'mae' | 'mse' | 'bce'): number {
    let s = 0;

    if (loss === 'mae') {
        for (let k = 0; k < target.length; k++) s += Math.abs(pred[k] - target[k]);

        return s / target.length;
    }

    if (loss === 'mse') {
        for (let k = 0; k < target.length; k++) {
            const d = pred[k] - target[k];
            s += d * d;
        }

        return s / target.length;
    }

    // bce
    for (let k = 0; k < target.length; k++) {
        const y = clamp01(target[k]);
        const p = clamp01(pred[k]);
        // -(y*log(p) + (1-y)*log(1-p))
        s += -(y * safeLog(p) + (1 - y) * safeLog(1 - p));
    }

    return s / target.length;
}

export function mean(arr: number[]) {
    return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

export function variance(arr: number[]) {
    if (arr.length <= 1) return 0;
    const m = mean(arr);
    let s = 0;
    for (const x of arr) {
        const d = x - m;
        s += d * d;
    }

    return s / arr.length;
}
