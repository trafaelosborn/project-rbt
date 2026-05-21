# 028 - V5 Controller Architecture: Shared Config, CLI, and TUI

**Date:** 2026-04-12  
**Status:** Implemented

## Context

V5 introduces a shared control layer (`src/control/`) that powers both CLI and
TUI from a single config model and controller abstraction. This replaces the
pattern of calling long_run drivers directly from one-off scripts.

## Architecture

```
src/control/
    run_config.py        # RunConfig, BenchmarkConfig, ConfigLockedError
    run_controller.py    # RunController (launch, chain, status, stop, validate, benchmark)
    cli.py               # Thin argparse wrapper over RunController
    tui.py               # Textual TUI scaffold over RunController
    _driver_adapter.py   # Internal: RunConfig → LongRunConfig translation
    _benchmark_runner.py # Internal: candidate-count throughput sweep
```

### One config model

`RunConfig` is the single source of truth for any run. It validates itself
at construction time and enforces bounded presets:

- `candidate_count` must be in `(8, 16, 32, 64, 100)`
- `block_proposals` must be in `(1000, 5000)`

### Config lock

`RunConfig.lock()` is called exactly once, immediately before the driver is
invoked. After lock, any field mutation raises `ConfigLockedError`. This
enforces the methodological constraint: run parameters cannot be changed
after a run has started.

The lock is not a UX restriction — it is a scientific integrity constraint.
Changing `use_semantic_transparency`, `use_fortran_batch`, or
`candidate_count` mid-run would invalidate the run as a controlled
experiment.

### Controller

`RunController` is the boundary between interfaces and engine internals.
It provides:

- `launch_run(config, output_dir)` — validates, locks, and starts
- `chain_run(from_manifest, config, output_dir)` — resolves endpoint, then launches
- `status(output_dir)` — reads manifest, returns `RunStatus` (non-destructive)
- `stop_run(output_dir)` — writes stop sentinel for clean halt at next block
- `validate_run(output_dir)` — scores best endpoint against validator bank
- `benchmark(config)` — candidate-count throughput sweep

Engine-specific details (`LongRunConfig`, block drivers, stage records) do
not cross the controller boundary.

### CLI

`src/control/cli.py` is an intentionally minimal argparse wrapper.
Commands: `retrodact`, `chain`, `benchmark`, `status`, `validate`.
No heavy CLI framework. Adding flags here is a last resort — the TUI is the
primary interactive interface.

### TUI

`src/control/tui.py` requires `textual` (pure Python, works on Windows).
Panels: `ConfigPanel`, `RunStatusPanel`, `ThroughputPanel`, `PreviewPanel`, `Log`.

The TUI uses `RunController` with a progress callback. The driver runs in a
background thread; the TUI receives `RunStatus` at each block boundary via
`call_from_thread`. Config fields are disabled (read-only) the moment the
[Launch] button is pressed.

## Fortran Batch Gap Analysis

The current Fortran batch layer (`src/accelerate/v4_batch_guidance.py`)
provides **guidance adjustments** — it computes which tensor dimensions to
prioritise for operator selection. It is not a full candidate scorer.

### What exists

- `BatchGuidance`: hotspot token weights, cooccurrence adjustment vectors
- Used as soft guidance in operator selection, not for scoring N candidates

### What v5 needs for batched candidate evaluation

To evaluate N candidates in batch (the actual acceleration target):

1. **Mutation generation** stays in Python: produce N mutated sequence lists
2. **Token array packing**: encode each mutated corpus as a fixed-shape tensor
   `(N, vocab_size)` or `(N, seq_count, max_seq_len)`
3. **Fortran batch scorer**: new routine that takes the packed tensors and
   returns `(N,)` float scores (structural + form components)
4. **Python acceptance logic**: receives `(N,)` scores, picks best, handles
   `min_improvement` check, commits state

### The gap

None of steps 2–4 exist yet. The existing Fortran code does not expose a
routine of the form `score_candidates(candidate_array) -> scores_array`.

### Scaffolded interface contract

```python
class BatchCandidateScorer(Protocol):
    """Interface that a Fortran or numpy batch scorer must satisfy."""

    def score_batch(
        self,
        candidate_sequences: list[list[list[str]]],  # shape: (N, seqs, tokens)
    ) -> list[float]:
        """
        Return one total score per candidate.
        Must be equivalent to calling _evaluate_sequences() for each.
        """
        ...
```

Until this interface is implemented and benchmarked, `use_fortran_batch=True`
with `candidate_count > 8` will emit a warning and fall back to Python
scoring. `use_fortran_batch=True` with `candidate_count=8` is the current
Fortran guidance path, which is functional but measured at 0.845x (slower
than Python at that count due to overhead).

### Why not implement it now

Implementing the Fortran batch scorer correctly requires:

- Deciding on a fixed vocabulary encoding (most efficient representation)
- Benchmarking the crossover point (at what N does batch beat Python loop)
- Validating that Fortran scores match Python scores exactly

These are non-trivial and belong in a dedicated engineering sprint. The
interface contract above ensures the slot exists in the architecture so the
implementation can drop in cleanly.

## Head-to-Head Results (measured 2026-04-12)

200-proposal probes from the frozen v4 endpoint (struct ≈ −3.7e−5, form = 0.756),
seed=77. All conditions start from the same corpus.

| Condition | struct | form | accepted | p/hour | halt |
|-----------|--------|------|----------|--------|------|
| v4 baseline | -3.7e-5 | 0.7565 | 3 | 565 | max_proposals |
| v5 culture bombs | -3.7e-5 | 0.7558 | 0 | 2,168 | culture_bomb_plateau |
| v5 transparency w=0.05 | -3.7e-5 | 0.7565 | 3 | 4,988 | max_proposals |

**Key findings:**

1. Culture bombs triggered plateau detection immediately (0 accepted stages,
   halt=culture_bomb_plateau). At the v4 endpoint, the search space for
   structural improvement is exhausted and culture bombs cannot escape the
   plateau. This is expected: the v5 culture bomb engine was designed for
   mid-run plateau escape, not for continuing from a near-zero struct endpoint.

2. The transparency condition (w=0.05) matches v4 baseline on accepted stages
   and form score. The transparency term does not hurt form convergence at this
   endpoint.

3. Throughput numbers from 200-proposal probes are noisy (one-time corpus
   loading dominates short trials). The candidate count benchmark provides
   more reliable throughput measurements.

## Candidate Count Benchmark (measured 2026-04-12)

From the frozen v4 endpoint, 150 proposals per trial, seed=42.

| c | p/hour | accepted stages | 
|---|--------|-----------------|
| 8 | 509 | 1 |
| **16** | **874** | **3** |
| 32 | 505 | 3 |
| 64 | 361 | 3 |
| 100 | 287 | 3 |

**Production candidate count: 16.**

c=16 matches c=32/64/100 on accepted stages but runs at 874 p/h vs 505/361/287.
Beyond c=16, coordination overhead per proposal outweighs additional search breadth.
c=8 accepts fewer stages, suggesting it misses good candidates.

Results written to `data/benchmarks/v5_candidate_count/`.

## What remains before full v5 production

See `docs/decisions/027_v5_engine_design.md` production-run gate. Specifically:

1. ~~Semantic transparency scorer~~ — done, see doc 029
2. ~~Transparency calibration probe~~ — done, endpoint score=0.265, weight=0.05 recommended
3. Fortran batch scorer implementation (above gap analysis) — not started
4. ~~Candidate count benchmark~~ — done, production count = 16 (874 p/h)
5. ~~V5 vs V4 head-to-head~~ — done, see results above
