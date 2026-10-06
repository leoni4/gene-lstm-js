Always use ASD-STE100 Simplified Technical English (STE) when describing anything to the user.

# AGENTS.md

This file is the entry point to the `@leoni4/gene-lstm-js` repository instructions.

It is written for AI agents and human contributors.

`@leoni4/gene-lstm-js` is a reusable TypeScript library for evolutionary LSTM models.

The library combines recurrent neural-network behaviour with genetic evolution.

Correct sequence semantics, correct genome-to-network mapping, stable public APIs, reproducibility, and performance are more important than adding features quickly.

## Core priorities

When working in this repository, use this priority order:

1. Preserve LSTM and evolutionary algorithm correctness.
2. Preserve public API compatibility unless a breaking change is explicitly requested.
3. Preserve sequence and recurrent-state semantics.
4. Preserve deterministic and reproducible behaviour where the API supports it.
5. Preserve serialization and persisted-model compatibility when practical.
6. Keep training and evaluation efficient enough for evolutionary populations.
7. Keep the implementation understandable and maintainable.

Do not improve one of these areas by silently breaking another.

# Repository source of truth

Use the repository as the source of truth.

Before making assumptions, inspect:

- `package.json`;
- the lockfile;
- `README.md`;
- source code;
- tests;
- build configuration;
- exported types;
- examples;
- the nearest `AGENTS.md`, if one exists.

Do not rely on remembered package behaviour when the repository can answer the question.

Read dependency versions from repository files.

# General guidelines

- Use the package manager defined by the repository.
- Prefer existing `package.json` scripts.
- Make the smallest change that fully solves the task.
- Do not refactor unrelated code.
- Do not introduce a dependency when the existing stack can solve the problem reasonably well.
- Do not change genetic or LSTM parameters as incidental cleanup.
- Do not change model behaviour during a typing, documentation, formatting, or performance-only task.
- Prefer existing abstractions and established patterns.
- Do not manually edit generated build output when source files are available.

# Public library contract

This is a public reusable library.

Treat publicly exposed behaviour as a compatibility contract.

This can include:

- exported classes;
- exported functions;
- exported types;
- constructor parameters;
- configuration objects;
- model input shapes;
- model output shapes;
- sequence semantics;
- recurrent-state behaviour;
- default values;
- mutation configuration;
- serialization;
- package entry points.

Before changing a public API, inspect:

- declarations;
- usages;
- exports;
- tests;
- README examples;
- serialization;
- compatibility code.

Do not make an intentional breaking change accidentally.

If a breaking change is required, make it explicit.

# Semantic versioning

Treat semantic versioning as part of the package contract.

A change can require a major version when it intentionally breaks supported public behaviour.

A backward-compatible feature can normally require a minor version.

A compatible bug fix can normally require a patch version.

Do not change package versions unless the user asks.

Do not run release operations only because implementation work is complete.

# Model semantics

Changes to model construction or evaluation can silently invalidate trained genomes.

Treat the following as algorithm-sensitive:

- genome structure;
- gene ordering;
- gene-to-parameter mapping;
- LSTM block structure;
- gate calculation;
- recurrent connections;
- hidden state;
- cell state;
- state initialization;
- state reset;
- sequence order;
- input dimensions;
- output dimensions;
- output activation;
- nested or hierarchical blocks, if supported;
- mutation;
- crossover;
- fitness handling;
- population evolution;
- cloning;
- serialization.

Do not change these semantics as incidental cleanup.

# Genome-to-model mapping

The mapping from genes to executable model parameters is a core invariant.

A genome must not start controlling a different parameter only because code was reorganized.

When changing model construction, verify:

- gene ordering;
- parameter offsets;
- parameter counts;
- layer or block ordering;
- input dimensions;
- output dimensions;
- recurrent parameters;
- nested structures, when applicable;
- serialization;
- cloning.

If model structure changes intentionally, determine whether old serialized genomes remain valid.

Do not claim compatibility without checking the actual mapping.

# Sequence semantics

Sequence order is part of the model.

Do not accidentally:

- reverse a sequence;
- skip an item;
- duplicate an item;
- leak a later step into an earlier step;
- carry recurrent state between evaluations when the API expects a reset;
- reset recurrent state when the API expects continuation.

When changing evaluation logic, identify explicitly:

- when state is created;
- when state is reset;
- when state is retained;
- whether repeated calls are independent;
- how batch or sequence boundaries work.

Tests should cover these semantics when affected.

# Hidden and cell state

Hidden state and cell state are model state, not temporary implementation details.

When changing recurrent evaluation:

- preserve state dimensions;
- preserve update order;
- preserve reset semantics;
- avoid sharing mutable state between unrelated model instances;
- avoid state leakage between independent fitness evaluations;
- preserve cloning semantics.

A performance optimization must not reuse mutable recurrent state incorrectly.

# Inputs and outputs

Input and output dimensions are part of the model contract.

When the library supports configurable output dimensions, verify the complete path from configuration to returned output.

Check:

- configuration validation;
- gene count;
- model construction;
- output layer construction;
- returned output shape;
- serialization;
- cloning;
- tests;
- README examples.

Do not assume a single output when the API can support several outputs.

Do not silently flatten or reshape outputs.

Changing input or output dimensional semantics can invalidate persisted genomes.

# Genetic evolution

Evolution behaviour must remain separate from model evaluation behaviour.

Treat these areas as algorithm-sensitive:

- initialization;
- mutation;
- mutation pressure or mode;
- crossover;
- parent selection;
- elitism;
- population replacement;
- fitness ordering;
- stagnation handling;
- structural mutation, when supported;
- parameter mutation;
- cloning.

Do not change default evolution pressure or mutation behaviour during an unrelated refactor.

# Mutation

Mutation must preserve a valid genome.

When modifying mutation logic, verify:

- every gene remains mapped to a valid model parameter;
- parameter counts remain correct;
- mutation ranges remain intentional;
- mutation probabilities remain intentional;
- structural changes update dependent metadata;
- cloning remains independent;
- serialization still round-trips.

Do not add hidden normalization or clamping only to make training appear more stable.

# Crossover

Crossover must produce a valid genome for the target architecture.

Verify:

- parent compatibility;
- gene lengths;
- structural metadata;
- input and output dimensions;
- nested model structures, when applicable;
- cloning;
- mutation after crossover;
- serialization.

Do not silently combine incompatible parent architectures unless that behaviour is explicitly supported.

# Fitness

Fitness is caller-defined unless the API defines additional semantics.

Do not assume:

- fitness is positive;
- fitness is normalized;
- fitness has a fixed range;
- fitness represents accuracy;
- fitness represents profit;
- larger or smaller is better unless the current API defines it.

Preserve existing comparison semantics.

Do not transform caller fitness values silently.

# Randomness and reproducibility

Evolution depends on random-number generation.

If seeded or injected randomness is supported:

- use the established RNG path;
- do not bypass it with direct `Math.random()` calls;
- preserve deterministic behaviour when practical;
- include the configured RNG in new evolutionary operations.

Changing the number or order of random draws can change training results.

Treat this as a behaviour change when deterministic sequences are part of the tested API.

Do not claim determinism unless verified.

# Numerical behaviour

LSTM evaluation can be sensitive to numerical changes.

Be careful when changing:

- sigmoid implementation;
- tanh implementation;
- gate equations;
- multiplication or accumulation order;
- normalization;
- clipping;
- weight ranges;
- mutation magnitude;
- initialization;
- floating-point comparisons.

Do not add arbitrary rounding, clipping, normalization, or epsilon values only to hide instability.

Handle `NaN` and infinite values according to existing library policy.

If no policy exists and a task requires one, make the decision explicit.

# Serialization

Serialized genomes can represent expensive training work.

Treat persisted representation carefully.

When changing serialization:

- inspect current fields;
- inspect model reconstruction;
- preserve fields when practical;
- preserve dimensional metadata;
- preserve nested model structure;
- test serialization round trips;
- compare reconstructed outputs when practical.

Do not claim that old serialized models are compatible without verifying the schema and model mapping.

If compatibility cannot be maintained, make that explicit.

# Cloning

Evolution requires independent individuals.

A clone must not accidentally share mutable state with its source.

Pay special attention to:

- gene arrays;
- nested arrays;
- model configuration;
- recurrent state;
- mutation state;
- cached values.

When cloning behaviour changes, test that mutating the clone does not mutate the source.

# Performance

Evolutionary LSTM workloads can evaluate many models across long sequences.

Before adding work inside a hot path, determine how often it executes.

Important hot paths can include:

- every sequence step;
- every LSTM block;
- every model;
- every genome;
- every mutation;
- every generation.

Avoid unnecessary allocations in sequence loops.

Avoid rebuilding unchanged model structures repeatedly when safe reuse is possible.

Avoid recalculating identical values when correctness permits caching.

Do not sacrifice model correctness for speed.

A cache must include everything that affects the result.

Never cache mutable recurrent results across evaluations unless the state semantics make this explicitly safe.

# TypeScript

TypeScript correctness is mandatory.

Respect repository compiler settings.

Do not weaken types only to make code compile.

Avoid:

- unnecessary `any`;
- broad assertions;
- ignored errors;
- duplicated domain types;
- unsafe non-null assertions.

Use explicit narrowing for nullable values.

Use named interfaces or types for important reusable public structures.

Follow the established public API even when a new API style would look cleaner.

# Exports

A public symbol must be exported through the complete package export path.

When adding or changing public APIs, check:

- local exports;
- barrel files;
- package entry points;
- generated declarations;
- package `exports`, if present.

Do not accidentally export internal implementation details.

Do not remove existing exports as cleanup without checking compatibility.

# Tests

Tests for algorithmic code should focus on behaviour.

Prefer deterministic tests where practical.

Important areas include:

- model construction;
- sequence evaluation;
- recurrent state;
- state reset;
- multiple outputs;
- mutation;
- crossover;
- cloning;
- serialization;
- deterministic RNG behaviour;
- genome-to-model mapping;
- regression cases.

For bug fixes, add a focused regression test when possible.

Do not weaken existing assertions only to make a change pass.

# Documentation

README usage and examples are part of the package user experience.

When public behaviour changes, check:

- installation examples;
- model construction;
- input examples;
- output examples;
- evolution examples;
- configuration;
- serialization examples;
- migration notes, when appropriate.

Do not document behaviour that was not verified.

# Comments

Prefer clear code and naming.

Use comments to preserve non-obvious information such as:

- genome-to-parameter mapping;
- recurrent-state constraints;
- sequence-boundary behaviour;
- numerical-stability decisions;
- serialization compatibility;
- algorithm invariants.

Do not add comments that restate code.

Remove or update stale comments.

# Dependencies

Do not add dependencies without a concrete need.

Before updating a dependency:

1. Confirm that the task requires it.
2. Check compatibility impact.
3. Update the lockfile through the package manager.
4. Run relevant validation.

Do not update unrelated dependencies.

# Generated files

Do not manually edit generated artifacts when source files exist.

This can include:

- `dist/`;
- compiled JavaScript;
- generated declaration files;
- coverage output;
- generated bundles.

Change the source of truth.

Generate derived artifacts only when required by the repository workflow.

# Git

Unless the user explicitly asks for the specific operation, do not:

- stage files;
- commit;
- push;
- merge;
- rebase;
- force-push;
- create tags;
- create releases;
- publish packages;
- modify remote branches.

Read-only Git operations are allowed.

Treat existing uncommitted changes as user-owned work.

Never discard unrelated changes.

# Package publishing

Publishing changes external state.

Do not run:

- `npm publish`;
- `npm version`;
- tag creation;
- release creation;

unless the user explicitly requests that action.

Before a requested release:

1. Inspect the current package version.
2. Determine compatibility impact.
3. Run the repository's prepublish or release validation.
4. Verify the package build.
5. Verify intended package contents.
6. Report failures before publishing.

Never report a successful publish unless it completed successfully.

# Change discipline

Make the smallest change that fully solves the requested task.

Do not:

- refactor unrelated code;
- rename unrelated symbols;
- change model defaults as cleanup;
- change public APIs without need;
- combine an algorithm change with unrelated architecture work;
- perform broad formatting;
- introduce speculative abstractions;
- change training behaviour while working only on documentation or types.

Report unrelated problems instead of silently fixing them.

# Reuse before creating

Before adding a new:

- model abstraction;
- genome utility;
- mutation helper;
- serialization layer;
- sequence helper;
- type;
- configuration abstraction;

search for the same concept.

Prefer one source of truth.

Do not add layers only for possible future requirements.

# Verification

Before declaring a task complete:

1. Review the diff.
2. Check for unrelated changes.
3. Run the smallest relevant tests.
4. Run TypeScript validation when relevant.
5. Run lint when relevant.
6. Run the build when package output can be affected.
7. Run broader tests when the change affects shared algorithm semantics.
8. State what was verified.
9. State what remains unverified.

For model-semantic changes, consider whether validation must cover:

- sequence behaviour;
- recurrent state;
- genome mapping;
- output dimensions;
- mutation;
- crossover;
- cloning;
- serialization.

Do not claim success for a command that did not complete successfully.

# Error handling

When a change causes an error:

1. Stop expanding the scope.
2. Investigate the root cause.
3. Determine whether the current change caused it.
4. Make the smallest correction.
5. Re-run relevant validation.

Do not respond to a local error with broad refactoring.

Do not suppress type errors, lint errors, or tests only to make validation pass.

# Working on a task

Before making a change:

1. Read applicable repository instructions.
2. Inspect the relevant implementation.
3. Inspect related tests.
4. Search for existing patterns and usages.
5. Determine whether the task changes public API.
6. Determine whether it changes model semantics.
7. Determine whether persisted genomes can be affected.
8. Make the smallest complete change.
9. Run relevant validation.
10. Report unverified assumptions or results.

More specific nested `AGENTS.md` files can add or refine rules for their directory.

They must not silently weaken repository-wide compatibility or algorithm-correctness requirements.