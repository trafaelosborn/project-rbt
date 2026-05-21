# Decision 023: Phase 4 Incremental Tensor State

Date: 2026-04-09
Status: Accepted

## Context

Phase 3 proved that the Fortran batch landscape can guide the live `v4` engine,
but it did not yet make the end-to-end loop faster.

The bottleneck is no longer the isolated batch kernel. It is the repeated Python
work required to rebuild the live bridge tensor after each accepted mutation.

If every accepted move forces a full corpus-to-tensor rebuild, the project gives
back a large share of the speedup that the Fortran kernel created.

## Decision

Phase 4 adds an in-memory incremental fingerprint tensor state:

1. keep the live bridge corpus and its tensor slices resident in memory
2. update accepted mutations against that state instead of rebuilding by default
3. fall back to a full rebuild only when the anchor vocabulary must change

The new state tracks:

- anchor vocabulary
- raw co-occurrence counts
- positional accumulators
- bigram and trigram counters
- normalized matrix views
- packed contiguous tensor buffer

## Scope

Phase 4 is a scaffold, not a live-engine replacement.

It does **not** yet:

- replace the existing `v4` proposal loop
- move acceptance or scoring into Fortran
- change the operator families or methodology

It **does** provide the state object needed for the next integration step.

## Implementation Notes

Primary module:

- [incremental_tensor_state.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/incremental_tensor_state.py)

Benchmark:

- [benchmark_incremental_tensor_state.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/benchmark_incremental_tensor_state.py)

Tests:

- [test_incremental_tensor_state.py](/C:/Code/Project%20RBT/project_rbt/tests/test_incremental_tensor_state.py)

The incremental state:

- subtracts old changed-sequence contributions
- adds new changed-sequence contributions
- refreshes normalized matrix/vector views
- repacks the contiguous tensor buffer

If a mutation introduces OOV anchor drift or changes sequence count, the state
reanchors through a full rebuild.

## Determinism Rule

Real-data benchmarking exposed a tie-order instability in the top-N n-gram
profiles. The incremental and rebuild paths were semantically equivalent, but
their sparse n-gram vectors diverged because tied counts depended on Counter
iteration order.

Phase 4 therefore uses deterministic n-gram profile ranking inside the
incremental tensor state:

- sort by descending count
- break ties lexicographically on the n-gram tuple

This keeps incremental refreshes and same-anchor rebuilds exactly aligned.

## Consequences

### Good

- Accepted-mutation refresh can now be benchmarked honestly against full rebuild.
- The live tensor state is explicit and inspectable.
- Future Fortran integration has a cleaner handoff point.

### Bad

- This does not yet speed up the live `v4` engine by itself.
- Anchor drift still triggers full rebuild.
- Deterministic top-N ranking is now a Phase 4 internal convention that must be
  preserved in later acceleration work.

## Benchmark Read

On the first real French benchmark:

- sampled sequences: `800`
- changed sequences: `12`
- anchor vocab size: `4097`
- update mode: `incremental`
- incremental refresh: `0.259233s`
- same-anchor full rebuild: `0.436647s`
- speedup: `1.68x`

That is not yet the final acceleration story, but it is the first evidence that
Phase 4 is moving the end-to-end loop in the right direction.
