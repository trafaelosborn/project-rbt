# 033 - V5.1 Hot Loop / Observability Split

**Date:** 2026-04-17  
**Status:** Implemented

## Problem

We want two things at once:

1. a faster v5 hot loop
2. stronger scientific outputs against attested validator corpora

Those goals conflict if every proposal candidate is treated like a full
scientific checkpoint. The search path needs cheap transient evaluation, while
the science path needs saved corpora and validator comparisons at meaningful
checkpoints.

## Decision

Split the system into:

- **search-time telemetry**
  - configurable Logan's Run live-event verbosity
  - buffered event writes instead of one file append per candidate
- **science-time validation**
  - optional validator-bank snapshots at block boundaries
  - snapshots run against saved block endpoints, not transient rejected
    candidates

This preserves methodology while reducing avoidable hot-loop overhead.

## Implemented changes

### 1. Live-event controls

Added to runtime config:

- `live_event_mode`
  - `all`
  - `selected`
  - `accepted_only`
  - `off`
- `live_event_buffer_size`

These flow through:

- `src/control/run_config.py`
- `src/control/cli.py`
- `src/control/_driver_adapter.py`
- `src/retrodiction/long_run_v4.py`
- `src/retrodiction/long_run_v5.py`
- `src/retrodiction/engine_reinforced_v4.py`

`engine_reinforced_v4.py` now buffers JSONL live events in memory and flushes
them in batches. This keeps the TUI-compatible event stream while reducing file
open/write frequency.

### 2. Block-level validator snapshots

Added to long-run config:

- `validator_set`
- `validator_snapshot_every_blocks`

`long_run_v4.py` now optionally calls
`compare_run_manifest_to_validator_bank(...)` after selected blocks complete and
writes the resulting artifacts into:

- `<run_dir>/validator_snapshots/`

The block entry in `manifest.json` now stores the snapshot artifact paths under
`validator_snapshot`.

### 3. Validator-bank compare flexibility

`src/validation/validator_bank_compare.py` now accepts:

- `validator_ids`
- `output_dir`
- `block_ids`

This allows one completed run manifest to produce:

- full-run validator comparisons
- single-block validator snapshots
- filtered validator subsets

without duplicating comparison logic.

## Why this matters

This change makes the v5.1 direction explicit:

- **rejected candidates** are for fast internal search
- **accepted stages and block endpoints** are for science

That is the right separation for a publishable system. We do not need to pay
historical-validator cost inside every proposal to preserve scientific value.

## Validation

Targeted checks passed:

- `tests/test_run_config.py`
- `tests/test_long_run_v4.py`
- `tests/test_engine_reinforced_v4.py`

All targeted tests passed after the change.

## Next step

The next performance pass should move deeper into the proposal hot loop:

- reduce transient candidate materialization
- move more selection-time work into delta-native or compiled batch paths
- keep validator snapshots at block boundaries as the science-facing output
