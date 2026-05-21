# 034 - V5.1 Sparse Candidate Payloads

**Date:** 2026-04-17  
**Status:** Implemented

## Problem

V5's hot loop was still paying unnecessary Python overhead by materializing full
candidate corpora for rewrite-heavy operators before scoring them. That meant we
were doing large transient list allocations even when only a handful of
sequences changed, and even when the candidate would be rejected immediately.

This was especially wasteful for the operators that dominate proposal traffic:

- `token_char_edit`
- `suffix_family_rewrite`
- `split_token`
- `function_word_burst`
- `paradigm_family_rewrite`

The incremental scoring layer already knew how to score localized changes. The
missing piece was carrying those changes through the proposal loop without
eagerly rebuilding the whole corpus.

## Decision

Introduce a sparse mutation payload in the v4/v5 hot loop.

`MutationPayload` can now carry either:

- `sequences`: a fully materialized candidate corpus
- `changed_sequences`: only the touched row indices and replacement sequences

Rewrite-heavy operators now emit sparse payloads. The proposal scorer consumes
those payloads directly through new incremental scoring entry points:

- `IncrementalScoringState.evaluate_changed_sequences(...)`
- `IncrementalScoringState.evaluate_batch_changed_sequences(...)`

The accepted candidate is materialized only once, immediately before alignment,
tensor-state update, commit, and stage save.

## Implementation Notes

- Added sparse rewrite helpers in
  `src/retrodiction/engine_reinforced_v2.py`
- Added sparse virtual-state construction and sparse batch scoring in
  `src/accelerate/incremental_scoring_state.py`
- Wired `MutationPayload` through the proposal pool, scoring path, and accepted
  candidate commit path in `src/retrodiction/engine_reinforced_v4.py`
- Added regression coverage for:
  - sparse changed-row scoring vs full materialized scoring
  - sparse batch scoring vs full batch scoring
  - accepted sparse winner materialization before save

## Why This Shape

We explicitly did **not** move acceptance logic, coherence gating, or alignment
into Fortran here. This pass is about removing Python-side transient corpus
construction so the existing incremental and Fortran-assisted scorers receive
smaller, more local inputs.

This keeps methodology unchanged while reducing overhead in the hottest part of
the search loop.

## Validation

Regression tests:

- `tests/test_incremental_scoring_state.py`
- `tests/test_engine_reinforced_v4.py`

Result after this pass:

- `22 passed`

Benchmark artifacts written:

- `data/benchmarks/v5_candidate_scaling_sparse_payload_pass1/scaling_report.json`
- `data/benchmarks/v5_vs_v4_headtohead_sparse_payload_pass1/headtohead_report.json`

## Current Readout

On the matched 16-candidate probe:

- v4 baseline: `3307 p/h`
- v5 plain Python: `3164 p/h`
- v5 + Fortran batch: `4072 p/h`

So the sparse payload pass did not yet make plain v5 overtake v4, but it keeps
the Fortran path clearly ahead and reduces wasted materialization in the exact
part of the loop we wanted to attack.

## Next Step

The next likely win is to push more of candidate construction/selection into a
batch-oriented path so the engine spends less time in Python assembling proposal
objects around otherwise fast scoring kernels.
