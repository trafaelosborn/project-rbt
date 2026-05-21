# 2026-04-16 — V5 Fortran Candidate Scaling

## Purpose

Measure whether the newly integrated `use_fortran_batch` path in v5 pays off
more as we widen the candidate pool, and use that to choose the next
production-facing configuration.

All probes in this report used:

- source corpus: processed French
- target: Latin
- engine: plain v5 (`enable_culture_bombs=False`)
- incremental scoring: on
- semantic transparency: off
- proposals per probe: 50
- seed: 77

The only difference between the paired runs is candidate scoring backend:

- `plain_python`
- `fortran_batch`

Benchmark artifact:

- `data/benchmarks/v5_candidate_scaling/scaling_report.json`

Runner:

- `src/accelerate/benchmark_v5_candidate_scaling.py`

## Results

| Candidates | Plain p/h | Fortran p/h | Speedup | Plain accepted | Fortran accepted | Plain final struct | Fortran final struct | Plain final form | Fortran final form |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 2280.2 | 2561.5 | **1.123x** | 4 | 3 | -1.3651 | **-1.3503** | **0.6672** | 0.6409 |
| 64 | 968.2 | 798.5 | **0.825x** | 2 | 3 | -1.3744 | **-1.3503** | 0.6283 | **0.6409** |
| 100 | 450.4 | 487.3 | **1.082x** | 2 | 5 | -1.3712 | **-1.3513** | 0.6372 | **0.7078** |

## Interpretation

### 1. Fortran is now real

The old question was whether Fortran had been integrated only on paper. That is
no longer true. The current v5 path is genuinely using batched scoring and, in
two of the three tested production-like settings, it wins end to end.

### 2. The win is not monotonic

The speedup does **not** increase cleanly with candidate count.

- `32` candidates is the strongest current throughput point.
- `100` candidates is still faster with Fortran than without it, but only by
  about 8%.
- `64` candidates regressed in this seed/run.

So the current architecture does **not** support the claim that “bigger batch
means automatically better throughput.”

### 3. There are two different optima

If the objective is **throughput**, `32 + fortran_batch` is the best current
choice in this benchmark.

If the objective is **absolute progress per 50 proposals**, `100 + fortran_batch`
looks strongest on the actual bridge metrics:

- best final structural score in the set
- best final form score in the set
- most accepted stages in the set

But it gets there much more slowly.

### 4. Progress per wall-clock still favors 32

Comparing only the Fortran runs:

- `32` candidates: struct gain from seed `+0.030402`, form gain `+0.073296`,
  `2561.5 proposals/hour`
- `100` candidates: struct gain from seed `+0.029415`, form gain `+0.140212`,
  `487.3 proposals/hour`

So `100` buys substantially more **form** improvement per proposal budget, but
`32` remains much better for practical wall-clock throughput.

## Recommendation

### Short answer

Use **`32` candidates with `use_fortran_batch=True`** for the next
production-oriented v5 tuning run.

### Why

- It is the fastest tested end-to-end setting.
- It still improves structural score relative to the plain Python counterpart.
- It avoids the very large wall-clock penalty of `100` candidates.
- It gives us a cleaner benchmark target while v5 is still being stabilized.

### What not to do yet

Do **not** jump straight to `100` candidates for a long production run based on
the current intuition alone. The benchmark says `100` can find better form
states, but the time cost is severe enough that it is not yet the sensible
default.

## Next move

1. Use `32` candidates as the working production baseline for the next v5
   tuning pass.
2. Treat `100` candidates as an explicit high-search-depth experimental mode.
3. If we want to rescue `64` or strengthen the case for `100`, run
   multi-seed scaling probes before committing to a long run.
