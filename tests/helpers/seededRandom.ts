export function mulberry32(seed: number): () => number {
    let a = seed >>> 0;

    return () => {
        a = (a + 0x6d2b79f5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);

        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function withSeededRandom<T>(seed: number, fn: () => T): { result: T; log: number[] } {
    const original = Math.random;
    const gen = mulberry32(seed);
    const log: number[] = [];
    Math.random = () => {
        const v = gen();
        log.push(v);

        return v;
    };
    try {
        const result = fn();

        return { result, log };
    } finally {
        Math.random = original;
    }
}
