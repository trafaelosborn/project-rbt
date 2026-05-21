# 2026-04-17 - Fortran Structural Batch Pass

## Purpose

Follow the delta-fed form batch work with a structural-side batch cleanup for
candidate selection.

This pass did **not** add a new structural Fortran kernel. It removed Python
selection-time overhead by:

- computing top-k coverage directly from Counters
- skipping transient word-profile dict construction in the Fortran-backed batch path
- vectorizing structural and coherence calculations across the candidate batch

## Verification

Touched modules compiled successfully.

Targeted test run:

- `pytest tests/test_cosine_acceleration.py tests/test_incremental_scoring_state.py -q`
  - **33 passed**

## Benchmark 1 - candidate scaling

Artifact:

- `data/benchmarks/v5_candidate_scaling_hotloop_pass2/scaling_report.json`

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
| 8 | 5810.9 | **7222.2** | **1.243x** |
| 16 | 3009.8 | **3693.1** | **1.227x** |
| 32 | 1462.2 | **1972.9** | **1.349x** |

Interpretation:

- Unlike the previous pass, the Fortran-backed path now wins at `32` as well.
- The win is still strongest as a relative speedup at wider candidate counts,
  but `16` remains the clean matched-comparison setting.

## Benchmark 2 - matched head-to-head

Artifact:

- `data/benchmarks/v5_vs_v4_headtohead_hotloop_pass2/headtohead_report.json`

Config:

- proposals: 100
- seed: 77
- candidates: 16
- start corpus: processed French

Results:

| Condition | p/h | Accepted | Final struct | Final form |
|---|---:|---:|---:|---:|
| v4 baseline | 3184 | 5 | -1.3641 | 0.6904 |
| v5 plain Python | 3100 | 5 | -1.3641 | 0.6904 |
| v5 Fortran batch | **3839** | 4 | **-1.3508** | 0.6639 |

## Takeaway

This pass strengthens the earlier conclusion:

- the main issue really was architecture
- widening the compiled boundary improves the real run
- the v5 Fortran line is now measurably ahead of both matched baselines in the
  16-candidate configuration

The next decision is no longer “does Fortran help at all?”

It is:

- whether the current `16`-candidate Fortran build is good enough to start a
  fresh production trajectory now, or
- whether we want one more pass that pushes compiled work deeper still before
  committing to a long run.
