# 035 - V5.1 Guided and Span Sparse Paths

**Date:** 2026-04-17  
**Status:** Implemented

## Problem

After introducing sparse candidate payloads for the obvious rewrite-heavy
operators, two meaningful sources of full-corpus materialization still remained:

- `sequence_span_rewrite`
- guided operator variants used when batch guidance is enabled

Both were still constructing full candidate corpora even when the mutation
logically touched only a narrow span or a sparse set of rewritten rows.

## Decision

Push sparse payload handling one layer deeper:

- extract reusable local-span mutation logic in the v2 engine
- add sparse span splicing helpers
- make v4's span rewrite return sparse changed-row payloads
- make guided rewrite-heavy operators emit sparse payloads instead of eagerly
  materialized corpora

## Implementation

### Shared v2 helpers

Added to `src/retrodiction/engine_reinforced_v2.py`:

- `_splice_sequence_span_sparse(...)`
- `_rewrite_sequence_span_local(...)`

This keeps the actual span mutation logic shared between the original
materialized path and the new sparse v4 path.

### New sparse v4 paths

Added to `src/retrodiction/engine_reinforced_v4.py`:

- `_mutate_sequence_span_rewrite_sparse(...)`
- `_mutate_sequence_span_rewrite_guided_sparse(...)`
- `_mutate_token_char_edit_guided_sparse(...)`
- `_mutate_suffix_family_guided_sparse(...)`
- `_mutate_split_token_guided_sparse(...)`
- `_mutate_function_word_burst_guided_sparse(...)`
- `_mutate_paradigm_family_rewrite_guided_sparse(...)`
- `_apply_named_operator_guided_payload(...)`

`_mutate_candidate(...)` now routes:

- unguided `sequence_span_rewrite` through the sparse path
- guided rewrite-heavy operators through sparse payload generation

`macro_bundle_rewrite` remains materialized for now.

## Validation

Regression coverage added in `tests/test_engine_reinforced_v4.py`:

- sparse span rewrite returns a sparse payload
- guided token edit payload remains sparse

Combined targeted suite result:

- `24 passed`

## Benchmark Readout

Artifacts:

- `data/benchmarks/v5_candidate_scaling_sparse_payload_pass2/scaling_report.json`
- `data/benchmarks/v5_candidate_scaling_sparse_payload_pass2_p100/scaling_report.json`
- `data/benchmarks/v5_vs_v4_headtohead_sparse_payload_pass2/headtohead_report.json`

Observed behavior:

- The 50-proposal 16-candidate slice favored plain Python
  (`5153.5 p/h` vs `4430.0 p/h`)
- The 100-proposal 16-candidate slice favored Fortran batch
  (`4226.4 p/h` vs `3446.6 p/h`)
- In the matched 100-proposal v4/v5 head-to-head:
  - v4 baseline: `3803 p/h`
  - v5 plain: `3348 p/h`
  - v5 + Fortran batch: `4163 p/h`

So the strongest current conclusion is:

- these sparse-path changes help both v4 and v5
- plain v5 is still behind v4 at this matched shape
- v5 + Fortran batch remains the fastest overall matched condition at
  16 candidates / 100 proposals

## Next Step

The biggest remaining materialized island is `macro_bundle_rewrite`.

If we keep tuning v5.1, the next serious pass should target bundle composition
so batch candidates are assembled from sparse deltas instead of repeatedly
cloning a whole working corpus inside the bundle loop.
