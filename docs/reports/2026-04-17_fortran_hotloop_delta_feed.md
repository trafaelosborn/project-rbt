# 2026-04-17 - Fortran Hot Loop Delta Feed

## Purpose

Reduce Python-side candidate setup in the v5 Fortran path without changing the
search methodology.

This pass specifically targeted the batch form-scoring boundary:

- before: Python copied full char-counter state for every candidate
- after: Python sends one committed baseline plus sparse per-candidate deltas

## Code changes

Changed files:

- `src/accelerate/fortran_cosine.py`
- `src/accelerate/incremental_scoring_state.py`
- `src/accelerate/benchmark_v5_candidate_scaling.py`
- `tests/test_cosine_acceleration.py`

Correctness verification:

- `python -m py_compile` on touched modules: passed
- `pytest tests/test_cosine_acceleration.py tests/test_incremental_scoring_state.py -q`
  - **33 passed**

## Benchmark 1 - candidate scaling in the decision range

Artifact:

- `data/benchmarks/v5_candidate_scaling_hotloop_pass1/scaling_report.json`

Config:

- source corpus: processed French
- proposals: 20
- seed: 77
- culture bombs: off
- semantic transparency: off
- incremental scoring: on

Results:

| Candidates | Plain p/h | Fortran p/h | Speedup |
|---|---:|---:|---:|
| 8 | 6869.6 | **8035.6** | **1.170x** |
| 16 | 3583.1 | **4193.4** | **1.170x** |
| 32 | **2380.5** | 2261.1 | 0.950x |

Interpretation:

- The new batch feed clearly helps in the `8` and `16` candidate range.
- `32` is still not a reliable win, which reinforces the decision to stabilize
  narrower candidate pools before widening again.

## Benchmark 2 - clean head-to-head

Artifact:

- `data/benchmarks/v5_vs_v4_headtohead_hotloop_pass1/headtohead_report.json`

Config:

- proposals: 100
- seed: 77
- candidates: 16
- start corpus: processed French

Results:

| Condition | p/h | Accepted | Final struct | Final form |
|---|---:|---:|---:|---:|
| v4 baseline | 3710 | 5 | -1.3641 | 0.6904 |
| v5 plain Python | 3726 | 5 | -1.3641 | 0.6904 |
| v5 Fortran batch | **3983** | 4 | **-1.3508** | 0.6639 |

Interpretation:

- In the matched 16-candidate configuration, the improved Fortran path is now
  the fastest of the three.
- The throughput gain over v4 baseline is about **7.4%**.
- The throughput gain over plain v5 is about **6.9%**.

This is not a moonshot yet, but it is finally a clean end-to-end win in the
configuration range we actually care about.

## Takeaway

The architectural diagnosis was correct:

- the problem was not “Fortran can’t help”
- the problem was “Fortran did not own enough of the hot path”

This pass moved the boundary in the right direction.

## Next move

1. Treat `16` as the primary matched benchmark configuration.
2. Keep `32` as a secondary stress test, not the default production setting.
3. If we want a bigger win, the next compiled target should be deeper candidate
   evaluation work beyond form alone, rather than simply increasing candidate
   count again.
