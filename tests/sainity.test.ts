import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';

describe('sanity check', () => {
    it('works', () => {
        expect(new GeneLSTM(1)).toBeTruthy();
    });
});
