# 2026-04-09 Fortran Acceleration Phase 4 Integration

## Objective

Take the Phase 4 incremental tensor scaffold and wire it into the live
accelerated `v4` engine path.

The specific target was:

- stop rebuilding the current bridge tensor from sequences each proposal when an
  incremental live tensor state is available

## Files Added Or Updated

- Incremental guidance benchmark:
  [benchmark_v4_incremental_guidance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/benchmark_v4_incremental_guidance.py)
- Guidance builder:
  [v4_batch_guidance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/v4_batch_guidance.py)
- Engine integration:
  [engine_reinforced_v4.py](/C:/Code/Project%20RBT/project_rbt/src/retrodiction/engine_reinforced_v4.py)
- Guidance tests:
  [test_v4_batch_guidance.py](/C:/Code/Project%20RBT/project_rbt/tests/test_v4_batch_guidance.py)
- Engine tests:
  [test_engine_reinforced_v4.py](/C:/Code/Project%20RBT/project_rbt/tests/test_engine_reinforced_v4.py)
- Decision note:
  [024_phase4_v4_incremental_guidance_integration.md](/C:/Code/Project%20RBT/project_rbt/docs/decisions/024_phase4_v4_incremental_guidance_integration.md)
- Benchmark artifact:
  [fortran_v4_phase4_benchmark.json](/C:/Code/Project%20RBT/project_rbt/data/validation/fortran_v4_phase4_benchmark.json)

## What Changed

Before this step, `TensorBatchGuidanceBuilder` only knew how to build current
co-occurrence and positional slices from raw sequences.

After this step, it also knows how to:

1. create an initial live tensor state
2. consume that state directly for batch guidance
3. stay backward-compatible with the older sequence-only route

The `v4` engine now carries that tensor state alongside the accepted bridge and
updates it only when a mutation is accepted.

The integrated path also now supports bounded in-place anchor extension, so new
surface forms can be absorbed without immediately forcing a full rebuild when
the current anchor still has vocab headroom.

## Verification

Passed:

- `python -m py_compile src\accelerate\v4_batch_guidance.py src\retrodiction\engine_reinforced_v4.py src\accelerate\benchmark_v4_incremental_guidance.py tests\test_v4_batch_guidance.py tests\test_engine_reinforced_v4.py`
- `python -m pytest tests\test_v4_batch_guidance.py tests\test_engine_reinforced_v4.py tests\test_incremental_tensor_state.py -q -p no:tmpdir -p no:cacheprovider`

Result:

- `13 passed`

Notable new checks:

- builder `build_from_state(...)` matches legacy `build(sequences)` in NumPy mode
- the live `v4` loop now uses `build_from_state(...)` when the builder supports
  it
- the seed record now carries tensor-state diagnostics

## Benchmark

Artifact:

- [fortran_v4_phase4_benchmark.json](/C:/Code/Project%20RBT/project_rbt/data/validation/fortran_v4_phase4_benchmark.json)

Paired run setup:

- language: `french`
- `180` sampled sequences
- `6` proposals
- `4` candidates per proposal
- same seed for both modes

Results:

- Python-only: `4.378783s`
- Auto-batch: `5.179153s`
- speedup: `0.845463x`

So the integrated path is still slower in this small full-loop benchmark.

## Why It Is Still Slower

The important result is not just the timing. It is the update-mode breakdown from
the accelerated run:

- `seed_build = 1`
- `anchor_extend = 2`

That means the engine really is using the new Phase 4 path, and the accepted
mutations in this benchmark changed token forms enough to require new anchor
entries, but *not* enough to force full rebuild.

This is already an improvement over the first integration pass. The anchor-churn
problem is no longer expressed here as repeated full rebuild.

This shows that the performance ceiling is now constrained less by the tensor
math itself and more by the remaining Python-side whole-loop cost.

## Representative Live Diagnostic

The integrated accelerated run now records the tensor-state update mode directly
in stage diagnostics. For example, [FR_v4_001.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/_v4_phase4_bench_auto/records/FR_v4_001.json)
shows:

- `batch_guidance_backend = fortran`
- `batch_guidance_tensor_state_update_mode = anchor_extend`
- many `batch_guidance_tensor_oov_tokens`

So the new instrumentation is already paying off: we can explain the runtime
behavior instead of just observing it.

## Interpretation

Phase 4 integration succeeded architecturally:

- the live engine consumes a persistent tensor state
- the guidance builder can read from that state directly
- the diagnostics expose when the state remains incremental versus when it
  reanchors
- bounded anchor extension is now absorbing OOV-heavy accepted moves without
  dropping into full rebuild in the paired benchmark

Phase 4 integration did **not** yet produce a whole-loop speed win.

That is still useful. It tells us the next bottleneck is now:

- remaining Python-side whole-loop cost after accepted moves

not:

- the incremental tensor scaffold itself

## Next Sharp Move

If we want a genuine runtime win from the integrated path, the next iteration has
to reduce the non-kernel work that still dominates this small full-loop run.

The best options now look like:

1. a token-identity layer distinct from surface-form novelty
2. a benchmark mode with more accepted steps so state refresh wins can amortize
   setup and save costs
3. moving more post-accept bookkeeping off the hot path

Until that happens, the engine will keep paying enough Python overhead after
accepted moves that the acceleration layer cannot yet dominate the total run
time.
