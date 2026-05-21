# 036 - V5.1 Sparse Macro Bundle Composition

**Date:** 2026-04-17  
**Status:** Implemented

## Problem

After the earlier sparse-payload passes, `macro_bundle_rewrite` was still the
largest remaining materialized island in the hot loop.

Even though most rewrite-heavy operators had already been converted to sparse
payloads, macro bundles were still:

- cloning a working corpus
- applying several sub-operators one after another
- rebuilding a full candidate corpus at each sub-step

That meant the engine was still paying repeated full-corpus allocation overhead
inside one of the most expensive operator families.

## Decision

Keep macro-bundle semantics the same, but change its internal representation.

Macro bundles now:

- keep one shallow working sequence view
- apply sparse sub-operator payloads in place
- track only the rows that differ from the original base corpus
- emit a sparse `MutationPayload` at the end

Compatibility wrappers still exist for call sites that expect a fully
materialized corpus, but the main v4/v5 candidate loop now stays sparse through
macro-bundle generation.

## Implementation

Added to `src/retrodiction/engine_reinforced_v4.py`:

- `_apply_named_operator_payload(...)`
- `_apply_mutation_payload_in_place(...)`
- `_mutate_macro_bundle_rewrite_payload(...)`
- `_mutate_macro_bundle_rewrite_sparse(...)`
- `_mutate_macro_bundle_rewrite_guided_sparse(...)`

Also refactored:

- `_mutate_macro_bundle_rewrite(...)`
- `_mutate_macro_bundle_rewrite_guided(...)`
- `_mutate_candidate(...)`

The old materialized macro-bundle entry points now delegate to the sparse
payload core and materialize only once at the boundary when needed.

## Validation

Regression coverage added in `tests/test_engine_reinforced_v4.py`:

- sparse macro bundle returns a sparse payload
- materialized macro bundle wrapper still produces a full corpus correctly

Targeted suite result:

- `26 passed`

## Benchmark Readout

Artifacts:

- `data/benchmarks/v5_candidate_scaling_macro_bundle_pass1_p100/scaling_report.json`
- `data/benchmarks/v5_vs_v4_headtohead_macro_bundle_pass1/headtohead_report.json`

Matched 100-proposal / 16-candidate readout:

- v4 baseline: `3720 p/h`
- v5 plain Python: `3754 p/h`
- v5 + Fortran batch: `4604 p/h`

Interpretation:

- plain v5 is now effectively at parity with v4 on the matched probe
- the Fortran-backed lane is now clearly ahead of both
- this is the first pass where the v5 architecture is no longer obviously
  paying a structural hot-loop penalty relative to v4 in the matched benchmark

## Next Step

The remaining serious tuning opportunities are now smaller and more specific:

- reduce repeated token/bigram recounting inside bundle sub-steps
- decide whether widening beyond 16 candidates still pays off
- benchmark whether the current Fortran lane should become the default
  production condition for long French-to-Latin runs
