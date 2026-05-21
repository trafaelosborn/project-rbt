# 2026-04-09 Fortran Acceleration Phase 4

## Objective

Start the deeper-fusion part of the Fortran roadmap by removing one of the main
Python-side costs in the live loop:

- rebuilding the full bridge fingerprint tensor after every accepted mutation

The goal for this session was not to rewrite `v4` yet. It was to build the
incremental tensor state that later integration can consume.

## Files Added Or Updated

- Incremental state: [incremental_tensor_state.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/incremental_tensor_state.py)
- Benchmark CLI: [benchmark_incremental_tensor_state.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/benchmark_incremental_tensor_state.py)
- Tests: [test_incremental_tensor_state.py](/C:/Code/Project%20RBT/project_rbt/tests/test_incremental_tensor_state.py)
- Decision note: [023_phase4_incremental_tensor_state.md](/C:/Code/Project%20RBT/project_rbt/docs/decisions/023_phase4_incremental_tensor_state.md)
- Benchmark artifact: [incremental_tensor_phase4_benchmark.json](/C:/Code/Project%20RBT/project_rbt/data/validation/incremental_tensor_phase4_benchmark.json)

## What Phase 4 Adds

The new state object keeps the live bridge fingerprint resident in memory:

1. sequences
2. anchor vocabulary
3. raw co-occurrence counts
4. positional accumulators
5. bigram and trigram counters
6. normalized tensor views
7. packed contiguous tensor buffer

It can then apply an accepted mutation in two modes:

- `incremental`
  subtract old changed-sequence contributions, add new ones, and refresh the
  views in place
- `full_rebuild`
  reanchor from scratch when new vocabulary or sequence-count drift makes the
  current anchor invalid

## Real-Data Bug Found And Fixed

The first benchmark mismatch was not in co-occurrence or positional state.

Those already matched the rebuild reference.

The real divergence was in the bigram vector, caused by top-N tie ordering. Two
semantically identical counters could choose different zero-boundary members in
the final sparse profile when counts were tied.

The fix was to make the Phase 4 state use deterministic profile ranking:

- descending count
- lexical tie break on the n-gram tuple

That change is now locked in by the new tied-profile regression test in
[test_incremental_tensor_state.py](/C:/Code/Project%20RBT/project_rbt/tests/test_incremental_tensor_state.py).

## Verification

Passed:

- `python -m py_compile src\accelerate\incremental_tensor_state.py src\accelerate\benchmark_incremental_tensor_state.py tests\test_incremental_tensor_state.py`
- `python -m pytest tests\test_incremental_tensor_state.py -q -p no:tmpdir -p no:cacheprovider`

Result:

- `5 passed`

## Benchmark

Artifact:

- [incremental_tensor_phase4_benchmark.json](/C:/Code/Project%20RBT/project_rbt/data/validation/incremental_tensor_phase4_benchmark.json)

Parameters:

- language: `french`
- sampled sequences: `800`
- changed sequences: `12`
- anchor vocab size: `4097`

Results:

- update mode: `incremental`
- incremental refresh: `0.259233s`
- same-anchor rebuild: `0.436647s`
- speedup vs rebuild: `1.684382x`

## Interpretation

This is the first Phase 4 sign that deeper fusion can matter at the live-loop
level, not just at the isolated kernel level.

The gain is still modest compared with the raw Phase 2 kernel speedup, but it is
honest:

- same live corpus size
- same tensor shape
- same anchor space
- same resulting tensor

That means the project now has a plausible next integration point:

- keep a live tensor state
- refresh it incrementally after accepted mutations
- hand the refreshed tensor directly to the Fortran batch layer

## Next Sharp Move

Wire the incremental tensor state into the accelerated `v4` path so that:

1. accepted mutations update the live tensor state instead of rebuilding from
   sequences
2. the live tensor state feeds the Phase 2/3 batch kernel directly
3. Python retains scoring, coherence, and Hungarian acceptance logic

That is the point where Phase 4 stops being a scaffold and starts becoming a
real runtime win for the retrodiction loop.
