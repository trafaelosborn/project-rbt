# 2026-04-09 Fortran Acceleration Phase 3

## Objective

Pick up from the Phase 2 batch kernel and wire it into the live `v4`
retrodiction engine as an optional acceleration mode without changing the
methodology:

- same operator families
- same Latin structural and form scoring
- same coherence gating
- same family-alignment diagnostic
- Python-only path retained as the reference implementation

## Files Added Or Updated

- Guidance layer: [v4_batch_guidance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/v4_batch_guidance.py)
- Engine integration: [engine_reinforced_v4.py](/C:/Code/Project%20RBT/project_rbt/src/retrodiction/engine_reinforced_v4.py)
- Guidance tests: [test_v4_batch_guidance.py](/C:/Code/Project%20RBT/project_rbt/tests/test_v4_batch_guidance.py)
- Updated engine smoke tests: [test_engine_reinforced_v4.py](/C:/Code/Project%20RBT/project_rbt/tests/test_engine_reinforced_v4.py)
- Decision note: [022_fortran_v4_guidance_integration.md](/C:/Code/Project%20RBT/project_rbt/docs/decisions/022_fortran_v4_guidance_integration.md)
- Mini benchmark artifact: [fortran_v4_phase3_benchmark.json](/C:/Code/Project%20RBT/project_rbt/data/validation/fortran_v4_phase3_benchmark.json)

## What Phase 3 Actually Does

Phase 3 does not replace `v4` acceptance with a tensor-only optimizer.

Instead:

1. Build the current bridge co-occurrence and positional slices in memory.
2. Align Latin co-occurrence and positional slices into that live anchor space.
3. Ask the batch kernel for a top-K tensor adjustment landscape.
4. Run a Hungarian frontier over that batch to keep a diverse non-overlapping
   subset of rows/targets.
5. Convert that subset into hotspot tokens, hotspot pairs, and positional hints.
6. Use those hints to steer existing `v4` operators toward more relevant parts
   of the live bridge.

The actual scoring and accept/reject step remain in Python.

## Verification

Targeted checks passed:

- `python -m py_compile project_rbt\src\accelerate\v4_batch_guidance.py project_rbt\src\retrodiction\engine_reinforced_v4.py project_rbt\tests\test_v4_batch_guidance.py project_rbt\tests\test_engine_reinforced_v4.py`
- `python -m pytest project_rbt\tests\test_v4_batch_guidance.py project_rbt\tests\test_engine_reinforced_v4.py -q -p no:tmpdir -p no:cacheprovider`

Result:

- `7 passed`

## Real Accelerated Smoke Run

I ran a small real French `v4` smoke in accelerated mode:

- output: [run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/_v4_phase3_smoke/run_summary.json)
- best stage: `FR_v4_002`
- final coherence: `coherent`
- guidance backend used: `fortran`
- selected guidance adjustments: `12`

That confirms the live engine is actually consuming the Fortran batch path end
to end, not just the synthetic test doubles.

## Mini End-To-End Benchmark

I also ran a small paired benchmark at:

- `180` sampled sequences
- `6` proposals
- `4` candidates per proposal
- same seed for both modes

Artifact:

- [fortran_v4_phase3_benchmark.json](/C:/Code/Project%20RBT/project_rbt/data/validation/fortran_v4_phase3_benchmark.json)

Results:

- Python-only: `3.705901s`
- Auto-batch: `4.233937s`

The auto-batch path used the `fortran` backend successfully, but it was still
slower at this small end-to-end scale.

That does **not** invalidate the direction. It means the current live loop still
spends enough time rebuilding tensors and translating guidance in Python that
the raw kernel win has not yet become a full-loop win.

## Interpretation

Phase 3 is successful on architecture, not yet on total runtime:

- successful:
  - optional accelerated mode exists
  - live `v4` can use it
  - Python-only path remains intact
  - real Fortran guidance is flowing into live operator selection

- not yet successful:
  - full-loop speedup
  - full replacement of sequential proposal generation with batch-native
    acceptance logic

## Next Sharp Move

The next real payoff likely requires deeper fusion, not more micro-optimizing:

1. move more of the current live tensor rebuild path into the acceleration layer
2. batch more than just hotspot discovery
3. keep Python on acceptance, coherence, and corpus semantics

That is where Phase 2's `61x` kernel win has a chance to become a meaningful
run-level win.
