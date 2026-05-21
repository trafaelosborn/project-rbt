# 2026-04-09 Fortran Acceleration Phase 2

## Goal

Move from a compileable toy kernel to a real batch landscape kernel:

- align current and reference tensors onto the same source-vocabulary space
- scan co-occurrence plus positional slices in Fortran
- return a scored top-K adjustment batch to Python
- benchmark that batch kernel against the Python reference

## What Was Built

- Alignment layer: [aligned_tensor.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/aligned_tensor.py)
- Batch wrapper: [fortran_batch.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/fortran_batch.py)
- CLI benchmark: [benchmark_batch_candidates.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/benchmark_batch_candidates.py)
- Fortran batch kernel: [bridge_distance.f90](/C:/Code/Project%20RBT/project_rbt/src/accelerate/bridge_distance.f90)
- Tests: [test_aligned_tensor.py](/C:/Code/Project%20RBT/project_rbt/tests/test_aligned_tensor.py), [test_fortran_batch.py](/C:/Code/Project%20RBT/project_rbt/tests/test_fortran_batch.py)

## Alignment Choice

Phase 2 explicitly anchors the tensor space to the source/current vocabulary.

For the default French -> Latin benchmark pair:

- anchor vocab comes from [french_cooccurrence_meta.json](/C:/Code/Project%20RBT/project_rbt/data/matrices/french_cooccurrence_meta.json)
- Latin co-occurrence and positional slices are projected into that 5000-token anchor space
- positional matrices are reduced to `5000 x 6` before batching

This keeps the batch layer consistent with the project's earlier source-vocab alignment rule.

## Verification

Python-side tests now cover:

- source-vocab square-matrix alignment
- source-vocab feature-matrix alignment
- Python reference top-K candidate ordering

Compile-backed tests cover:

- Fortran top-K candidate extraction matches the Python reference exactly on a small deterministic example
- Full compile-backed acceleration subset passed: `6 passed`

## Benchmark

Benchmark artifact:

- [fortran_batch_benchmark_phase2.json](/C:/Code/Project%20RBT/project_rbt/data/validation/fortran_batch_benchmark_phase2.json)

The benchmark uses:

- anchor: `french`
- reference: `latin`
- slices: co-occurrence plus positional
- `top_k = 512`

Measured result:

- Python reference: `1.817311` seconds
- Fortran batch kernel: `0.029722` seconds
- Speedup vs Python: `61.143010x`

## Interpretation

Phase 2 is a meaningful architectural step because the Fortran side is now doing real batch landscape extraction instead of only a trivial elementwise operation.

Unlike the Phase 1 elementwise kernel, the Phase 2 batch kernel is now decisively faster than the Python reference on the aligned French -> Latin landscape.

The next step is to connect this candidate batch to Python-side selection logic without changing:

- scoring
- coherence gating
- operator families
- the reference sequential engine
