# 032 - Batch Structural Selection Path

**Date:** 2026-04-17  
**Status:** Implemented

## Problem

After the delta-fed Fortran form path in decision 031, the batch scorer still
did unnecessary Python work on the structural side during candidate selection.

In the Fortran-backed `evaluate_batch()` path, Python still:

- built full word bigram/trigram profile dicts for each candidate
- computed top-k coverage from those dicts
- ran structural/coherence scoring one candidate at a time

That work did not change methodology, but it kept too much selection-time logic
in a per-candidate Python shape.

## Decision

Keep the saved-stage and single-candidate reference path unchanged, but add a
specialized selection-time batch path for the Fortran-backed evaluator.

The new batch path:

1. computes structural features directly from raw counters
2. skips full word profile dict construction for candidate ranking
3. computes structural score and coherence metrics over a batch of vectors

This does **not** change the scoring definition. It only changes how the same
values are assembled for proposal selection.

## Implementation

### `src/accelerate/incremental_scoring_state.py`

Added:

- `_top_k_coverage_from_counter()`
- `_score_virtual_state_batch()`
- `_batch_cosine_similarity()`

Changed:

- `evaluate_batch()` now uses a specialized vectorized structural/coherence
  path when a `FortranCosineScorer` is present
- the Fortran-backed batch path no longer builds full word profile dicts for
  candidate ranking
- reference vectors and coherence-scale arrays are cached on the
  `IncrementalScoringState` instance for reuse

## Why this is safe

- candidate acceptance still uses the same composite score
- structural score is still the same Latin-distance formula
- coherence still uses the same real-language-centroid vs Markov-noise margin
- saved accepted stages still get full fingerprint artifacts from the standard
  `_save_stage()` path

Only the transient ranking path changed.

## Result

With the structural-side batch cleanup in place, the Fortran-backed path now
shows a clean speed win across `8`, `16`, and `32` candidates in the local
candidate-scaling benchmark, and remains ahead in the matched 16-candidate
head-to-head benchmark versus v4/v5 plain.

This is the first point where the v5 Fortran line looks like a stable
production candidate in the narrow-candidate regime.
