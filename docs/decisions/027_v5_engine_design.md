# 027 - V5 Engine Design

**Date:** 2026-04-12  
**Status:** Proposed

## Context

The long-running French -> Latin v4 continuation has reached near-zero
structural distance while remaining active on form and family alignment. As of
the design freeze, the live run in
`data/retrodiction/french/v4_until_plateau_from_30k/manifest.json` is still
running, but its structural score is effectively converged for ordinary
interpretation.

That makes v4 a strong baseline and a natural handoff point for v5.

## Decision

V5 will be treated as a new engine condition rather than a continuation of v4.
We will freeze the current v4 run as a completed baseline artifact, then build
v5 as a controlled methodological update with explicit ablations.

## Why Freeze V4 Now

Stopping v4 after the next clean block boundary preserves one coherent run under
one architecture. This matters more than squeezing a few extra ten-thousandths
out of the structural axis.

The closeout language should be careful:

- Use "near-zero structural convergence achieved" or "structural convergence
  band reached"
- Do not overclaim an exact natural halt if we manually stop the run
- Record the halt reason as an operator decision to begin v5 refinement

## V5 Design Pillars

### 1. Semantic Transparency Constraint

Add a semantic transparency term to the acceptance objective.

Intent:

- reward tokens whose meaning is recoverable from recognizable semantic roots
- penalize semantically empty glue that exists only as statistical filler

Important methodological constraint:

- this is not merely a speed or implementation change
- it introduces a human-designed lexical prior
- therefore it must be treated as an experimental condition with its own
  ablation, calibration, and documentation

Operationally:

- semantic transparency is additive, not replacing structural, form, coherence,
  or family alignment
- calibration must happen in a short probe run before any production v5 run
- results should be reported both with and without the transparency term when
  possible

### 2. Batched Candidate Evaluation

The next acceleration target is batching candidate evaluation, not merely
reusing the existing Fortran guidance layer as-is.

Important implementation note:

- the current Fortran batch code computes tensor adjustment batches for guidance
- it is not yet a drop-in full proposal scorer for 100 candidates
- therefore "use existing Fortran batch" still implies real engineering work

The correct v5 goal is:

- generate candidate mutations in Python
- evaluate candidate tensor math in batch
- keep acceptance, coherence, and decision logic in Python

### 3. Candidate Count Must Be Benchmarked

Do not assume 100 candidates is automatically optimal.

Candidate count affects:

- search breadth
- mutation generation cost
- Python-side coordination overhead
- quality of best-of-k selection

V5 should benchmark at least:

- 8
- 16
- 32
- 64
- 100

The production count should be chosen from evidence, not from intuition alone.

### 4. Larger Blocks Are Fine if Visibility Survives

Moving from 1000 to 5000 proposals per block is reasonable if:

- wall-clock per block remains acceptable
- lightweight status visibility still exists inside the block
- block-level validator and plateau logic remain unchanged

Without intermediate heartbeat reporting, 5000-proposal blocks are too opaque.

## Build Order

1. Freeze the current v4 run after the next block boundary.
2. Write a v4 final-state report and final validator-bank snapshot.
3. Build the semantic transparency scorer.
4. Calibrate transparency weight in a short probe run.
5. Implement batched candidate evaluation.
6. Benchmark candidate counts and choose the production count.
7. Benchmark v5 against frozen v4 on the same starting corpus.
8. Build CLI.
9. Build TUI.
10. Launch the first v5 production run.

## Required Documentation

- `docs/reports/2026-04-12_v4_final_state.md`
- `docs/decisions/028_semantic_transparency_constraint.md`
- `docs/reports/<date>_v5_batch_benchmark.md`
- updates to `README`
- updates to `METHODOLOGY.md`

## Production-Run Gate

Do not launch the first v5 production run until all of the following are true:

- frozen v4 baseline report exists
- final validator-bank snapshot exists for frozen v4
- transparency term has a documented calibration
- batched evaluation has a measured benchmark against v4
- candidate count has been selected empirically

## Consequence

This design preserves the strongest scientific property of the current work:

- v4 remains a clean baseline
- v5 becomes a clearly described new condition
- any improvement in speed, transparency, or trajectory can be attributed to
  documented architectural changes rather than to a muddled mid-run splice
