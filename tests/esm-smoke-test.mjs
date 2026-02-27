#!/usr/bin/env node

/**
 * ESM Smoke Test
 *
 * This test verifies that the package exports work correctly when imported as ESM.
 * It checks that all exported classes, functions, and constants are properly available.
 */

import { GeneLSTM, Client, EMutationPressure, MUTATION_PRESSURE_CONST } from '../dist/index.js';

// Color codes for output
const GREEN = '\x1b[32m';
const RED = '\x1b[31m';
const YELLOW = '\x1b[33m';
const RESET = '\x1b[0m';

let testsPassed = 0;
let testsFailed = 0;

/**
 * Helper function to run a test
 */
function test(name, fn) {
    try {
        fn();
        console.log(`${GREEN}✓${RESET} ${name}`);
        testsPassed++;
    } catch (error) {
        console.error(`${RED}✗${RESET} ${name}`);
        console.error(`  ${RED}Error: ${error.message}${RESET}`);
        testsFailed++;
    }
}

/**
 * Helper function to assert a condition
 */
function assert(condition, message) {
    if (!condition) {
        throw new Error(message || 'Assertion failed');
    }
}

console.log(`${YELLOW}Running ESM smoke tests...${RESET}\n`);

// Test 1: GeneLSTM export
test('GeneLSTM is exported and is a class', () => {
    assert(typeof GeneLSTM === 'function', 'GeneLSTM should be a function/class');
    assert(GeneLSTM.name === 'GeneLSTM', 'GeneLSTM should have the correct name');
});

// Test 2: Client export
test('Client is exported and is a class', () => {
    assert(typeof Client === 'function', 'Client should be a function/class');
    assert(Client.name === 'Client', 'Client should have the correct name');
});

// Test 3: EMutationPressure export
test('EMutationPressure enum is exported', () => {
    assert(typeof EMutationPressure === 'object', 'EMutationPressure should be an object');
    assert(EMutationPressure !== null, 'EMutationPressure should not be null');
});

// Test 4: MUTATION_PRESSURE_CONST export
test('MUTATION_PRESSURE_CONST is exported', () => {
    assert(typeof MUTATION_PRESSURE_CONST === 'object', 'MUTATION_PRESSURE_CONST should be an object');
    assert(MUTATION_PRESSURE_CONST !== null, 'MUTATION_PRESSURE_CONST should not be null');
});

// Test 5: GeneLSTM can be instantiated
test('GeneLSTM can be instantiated with valid parameters', () => {
    const config = {
        populationSize: 100,
        eliteSize: 10,
        mutationRate: 0.1,
        crossoverRate: 0.7,
        weightMutationStep: 0.1,
        biasMutationStep: 0.1,
    };

    const geneLSTM = new GeneLSTM(config);
    assert(geneLSTM instanceof GeneLSTM, 'Should create a GeneLSTM instance');
});

// Test 6: Client can be instantiated
test('Client can be instantiated', () => {
    const client = new Client();
    assert(client instanceof Client, 'Should create a Client instance');
});

// Test 7: EMutationPressure has expected values
test('EMutationPressure has expected enum values', () => {
    assert('COMPACT' in EMutationPressure, 'EMutationPressure should have COMPACT');
    assert('NORMAL' in EMutationPressure, 'EMutationPressure should have NORMAL');
    assert('BOOST' in EMutationPressure, 'EMutationPressure should have BOOST');
    assert('ESCAPE' in EMutationPressure, 'EMutationPressure should have ESCAPE');
    assert('PANIC' in EMutationPressure, 'EMutationPressure should have PANIC');
});

// Test 8: MUTATION_PRESSURE_CONST has expected structure
test('MUTATION_PRESSURE_CONST has expected structure', () => {
    assert(typeof MUTATION_PRESSURE_CONST === 'object', 'Should be an object');
    assert(Object.keys(MUTATION_PRESSURE_CONST).length > 0, 'Should have keys');
});

// Test 9: Module import works with named imports
test('Named imports work correctly', () => {
    assert(GeneLSTM !== undefined, 'GeneLSTM should be defined');
    assert(Client !== undefined, 'Client should be defined');
    assert(EMutationPressure !== undefined, 'EMutationPressure should be defined');
    assert(MUTATION_PRESSURE_CONST !== undefined, 'MUTATION_PRESSURE_CONST should be defined');
});

// Test 10: Default import should work (import * as)
import * as geneLstmJs from '../dist/index.js';

test('Wildcard import works correctly', () => {
    assert(geneLstmJs.GeneLSTM !== undefined, 'GeneLSTM should be available via wildcard import');
    assert(geneLstmJs.Client !== undefined, 'Client should be available via wildcard import');
    assert(geneLstmJs.EMutationPressure !== undefined, 'EMutationPressure should be available via wildcard import');
    assert(
        geneLstmJs.MUTATION_PRESSURE_CONST !== undefined,
        'MUTATION_PRESSURE_CONST should be available via wildcard import',
    );
});

// Summary
console.log(`\n${YELLOW}Test Summary:${RESET}`);
console.log(`${GREEN}Passed: ${testsPassed}${RESET}`);
if (testsFailed) console.log(`${RED}Failed: ${testsFailed}${RESET}`);

if (testsFailed > 0) {
    console.error(`\n${RED}ESM smoke tests failed!${RESET}`);
    process.exit(1);
} else {
    console.log(`\n${GREEN}All ESM smoke tests passed!${RESET}`);
    process.exit(0);
}
