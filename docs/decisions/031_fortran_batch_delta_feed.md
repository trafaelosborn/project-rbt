# 031 - Fortran Batch Delta Feed

**Date:** 2026-04-17  
**Status:** Implemented

## Problem

The first v5 Fortran integration only accelerated the cosine math after Python
had already done most of the expensive candidate materialization work.

For each proposal candidate, `IncrementalScoringState.evaluate_batch()` still:

1. computed a sequence delta
2. copied the committed char-counter state into new `Counter` objects
3. applied the changed-sequence updates to those copied counters
4. passed the fully materialized candidate counters into `FortranCosineScorer`

That meant the compiled path was real, but too narrow. Python was still paying
for `N` full char-counter worlds before Fortran saw the batch.

## Decision

Keep structural scoring unchanged for now, but change the batch form-scoring
feed from:

- full per-candidate char-counter copies

to:

- one committed baseline char-counter state
- plus sparse per-candidate char-ngram deltas

This preserves the current methodology and acceptance logic while moving more of
the hot candidate-evaluation setup into a baseline-plus-delta shape that is much
better suited to compiled batch scoring.

## Implementation

### `src/accelerate/fortran_cosine.py`

Added:

- `ProfileVocabIndex.counter_to_dense_counts()`
- `ProfileVocabIndex.add_counter_to_dense()`
- `FortranCosineScorer.score_form_batch_from_deltas()`

The new batch path:

1. builds one dense committed baseline vector per component
2. clones that baseline across the batch matrix
3. applies only sparse per-candidate deltas
4. normalizes rows in-place
5. calls the existing Fortran batch kernel

Also changed the existing batch matrices to Fortran-order buffers to reduce
layout-conversion overhead on the compiled call path.

### `src/accelerate/incremental_scoring_state.py`

Added:

- `_char_counter_deltas_from_sequences()`

Changed:

- `evaluate_batch()` now uses sparse char deltas when a `FortranCosineScorer`
  is present
- `_virtual_state_from_sequences()` now supports `include_char_counts=False`
  so the batch path can skip copying committed char counters entirely

Structural and coherence scoring remain unchanged in this pass. The goal here is
to reduce Python-side candidate materialization without changing the search.

## Why this order

This is the smallest change that puts Fortran in charge of more of the hot loop
without rewriting the whole candidate evaluator at once.

It does **not** yet make Fortran own the full proposal loop. It does:

- remove one major Python-side duplication cost in the batch path
- preserve correctness and current run behavior
- create a cleaner baseline for a later structural-batch pass

## Validation

Targeted correctness coverage:

- `tests/test_cosine_acceleration.py`
  - delta-fed batch scorer matches full batch scorer
  - batch `IncrementalScoringState` with Fortran scorer stays close to Python path
- `tests/test_incremental_scoring_state.py`
  - existing evaluate / evaluate_batch correctness suite still passes

## Benchmark signal

Two new benchmark artifacts were generated:

- `data/benchmarks/v5_candidate_scaling_hotloop_pass1/scaling_report.json`
- `data/benchmarks/v5_vs_v4_headtohead_hotloop_pass1/headtohead_report.json`

Key outcome:

- the improved Fortran batch path now wins clearly at `8` and `16` candidates
- `32` candidates is still unstable / not yet a clean win

That supports the current strategy:

- benchmark and stabilize `8`/`16` first
- only widen the search again after the batch architecture proves itself
