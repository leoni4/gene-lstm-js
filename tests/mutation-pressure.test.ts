import { describe, it, expect } from 'vitest';
import { GeneLSTM } from '../src/gLstm.js';
import { EMutationPressure, MUTATION_PRESSURE_CONST } from '../src/types/index.js';

describe('Mutation Pressure System', () => {
    it('should initialize with default NORMAL mutation pressure', () => {
        const glstm = new GeneLSTM(10);

        expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);

        const pressure = glstm.getMutationPressure();
        expect(pressure.topology).toBe(1);
        expect(pressure.weights).toBe(1);
    });

    it('should allow setting custom mutation pressure', () => {
        const glstm = new GeneLSTM(10, {
            mutationPressure: EMutationPressure.BOOST,
        });

        expect(glstm.mutationPressure).toBe(EMutationPressure.BOOST);

        const pressure = glstm.getMutationPressure();
        expect(pressure.topology).toBe(1.2);
        expect(pressure.weights).toBe(1.5);
    });

    it('should allow changing mutation pressure dynamically', () => {
        const glstm = new GeneLSTM(10);

        glstm.mutationPressure = EMutationPressure.ESCAPE;

        const pressure = glstm.getMutationPressure();
        expect(pressure.topology).toBe(1.5);
        expect(pressure.weights).toBe(2);
    });

    it('should return correct pressure values for all levels', () => {
        const glstm = new GeneLSTM(10);

        // Test COMPACT
        glstm.mutationPressure = EMutationPressure.COMPACT;
        let pressure = glstm.getMutationPressure();
        expect(pressure.topology).toBe(0.1);
        expect(pressure.weights).toBe(0.8);

        // Test NORMAL
        glstm.mutationPressure = EMutationPressure.NORMAL;
        pressure = glstm.getMutationPressure();
        expect(pressure.topology).toBe(1);
        expect(pressure.weights).toBe(1);

        // Test BOOST
        glstm.mutationPressure = EMutationPressure.BOOST;
        pressure = glstm.getMutationPressure();
        expect(pressure.topology).toBe(1.2);
        expect(pressure.weights).toBe(1.5);

        // Test ESCAPE
        glstm.mutationPressure = EMutationPressure.ESCAPE;
        pressure = glstm.getMutationPressure();
        expect(pressure.topology).toBe(1.5);
        expect(pressure.weights).toBe(2);

        // Test PANIC
        glstm.mutationPressure = EMutationPressure.PANIC;
        pressure = glstm.getMutationPressure();
        expect(pressure.topology).toBe(2);
        expect(pressure.weights).toBe(2);
    });

    it('should not escalate pressure when disabled', () => {
        const glstm = new GeneLSTM(10, {
            enablePressureEscalation: false,
        });

        expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);

        // Simulate stagnation
        glstm.updateMutationPressure(0.5);
        glstm.updateMutationPressure(0.5);
        glstm.updateMutationPressure(0.5);

        // Should remain NORMAL
        expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);
    });

    it('should escalate pressure on stagnation when enabled', () => {
        const glstm = new GeneLSTM(10, {
            enablePressureEscalation: true,
            stagnationThreshold: 3,
        });

        expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);

        // First call establishes baseline
        glstm.updateMutationPressure(0.5);
        expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);

        // Simulate stagnation for 3 generations
        glstm.updateMutationPressure(0.5); // stagnation count = 1
        expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);

        glstm.updateMutationPressure(0.5); // stagnation count = 2
        expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);

        glstm.updateMutationPressure(0.5); // stagnation count = 3, escalate!
        // Should escalate to BOOST after 3 stagnant generations
        expect(glstm.mutationPressure).toBe(EMutationPressure.BOOST);
    });

    it('should reduce pressure on improvement', () => {
        const glstm = new GeneLSTM(10, {
            enablePressureEscalation: true,
            stagnationThreshold: 2,
        });

        // Establish baseline and escalate to BOOST
        glstm.updateMutationPressure(0.5); // baseline
        glstm.updateMutationPressure(0.5); // stagnation count = 1
        glstm.updateMutationPressure(0.5); // stagnation count = 2, escalate to BOOST
        expect(glstm.mutationPressure).toBe(EMutationPressure.BOOST);

        // Improve fitness - should reduce pressure
        glstm.updateMutationPressure(0.7);
        expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);
    });

    it('should escalate through all pressure levels', () => {
        const glstm = new GeneLSTM(10, {
            enablePressureEscalation: true,
            stagnationThreshold: 2,
        });

        expect(glstm.mutationPressure).toBe(EMutationPressure.NORMAL);

        // Escalate to BOOST
        glstm.updateMutationPressure(0.5); // baseline
        glstm.updateMutationPressure(0.5); // stagnation count = 1
        glstm.updateMutationPressure(0.5); // stagnation count = 2, escalate to BOOST
        expect(glstm.mutationPressure).toBe(EMutationPressure.BOOST);

        // Escalate to ESCAPE
        glstm.updateMutationPressure(0.5); // stagnation count = 1
        glstm.updateMutationPressure(0.5); // stagnation count = 2, escalate to ESCAPE
        expect(glstm.mutationPressure).toBe(EMutationPressure.ESCAPE);

        // Escalate to PANIC
        glstm.updateMutationPressure(0.5); // stagnation count = 1
        glstm.updateMutationPressure(0.5); // stagnation count = 2, escalate to PANIC
        expect(glstm.mutationPressure).toBe(EMutationPressure.PANIC);

        // Stay at PANIC
        glstm.updateMutationPressure(0.5); // stagnation count = 1
        glstm.updateMutationPressure(0.5); // stagnation count = 2, but already at max
        expect(glstm.mutationPressure).toBe(EMutationPressure.PANIC);
    });

    it('should verify MUTATION_PRESSURE_CONST is correctly defined', () => {
        expect(MUTATION_PRESSURE_CONST[EMutationPressure.COMPACT]).toEqual({
            topology: 0.1,
            weights: 0.8,
        });

        expect(MUTATION_PRESSURE_CONST[EMutationPressure.NORMAL]).toEqual({
            topology: 1,
            weights: 1,
        });

        expect(MUTATION_PRESSURE_CONST[EMutationPressure.BOOST]).toEqual({
            topology: 1.2,
            weights: 1.5,
        });

        expect(MUTATION_PRESSURE_CONST[EMutationPressure.ESCAPE]).toEqual({
            topology: 1.5,
            weights: 2,
        });

        expect(MUTATION_PRESSURE_CONST[EMutationPressure.PANIC]).toEqual({
            topology: 2,
            weights: 2,
        });
    });
});
