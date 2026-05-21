# Decision 024: Phase 4 V4 Incremental Guidance Integration

Date: 2026-04-09
Status: Accepted

## Context

Phase 4 introduced an incremental fingerprint tensor state, but that state still
lived beside the engine rather than inside it.

The next necessary step was to wire the state into the live `v4` acceleration
path so that batch guidance could read the current bridge tensor directly instead
of rebuilding co-occurrence and positional slices from sequences every proposal.

## Decision

Integrate the incremental tensor state into the accelerated `v4` loop while
keeping the methodology unchanged.

Python still owns:

- corpus mutation semantics
- Latin structural and form scoring
- coherence gating
- family-alignment diagnostics
- accept / reject decisions

The acceleration layer now additionally owns:

- initial live tensor state construction for the current bridge
- state-based batch guidance reads
- accepted-mutation tensor refresh when possible
- bounded in-place anchor extension when new forms still fit inside the current
  vocab ceiling

## Implementation

Updated modules:

- [v4_batch_guidance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/v4_batch_guidance.py)
- [engine_reinforced_v4.py](/C:/Code/Project%20RBT/project_rbt/src/retrodiction/engine_reinforced_v4.py)

New benchmark:

- [benchmark_v4_incremental_guidance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/benchmark_v4_incremental_guidance.py)

Updated tests:

- [test_v4_batch_guidance.py](/C:/Code/Project%20RBT/project_rbt/tests/test_v4_batch_guidance.py)
- [test_engine_reinforced_v4.py](/C:/Code/Project%20RBT/project_rbt/tests/test_engine_reinforced_v4.py)

The builder now supports:

1. `build_initial_state(sequences)`
2. `build_from_state(state)`
3. legacy `build(sequences)` as a compatibility fallback

The `v4` engine now:

1. builds one live tensor state for the seed bridge
2. asks the guidance builder to read from that state
3. updates the state only after accepted mutations
4. records whether the state refresh was:
   - `seed_build`
   - `incremental`
   - `anchor_extend`
   - `full_rebuild`

## Why This Is Still Phase 4

This step does not change what the engine is optimizing.

It changes where the current tensor comes from.

That makes it Phase 4 in the strict sense:

- reduce Python-side rebuild work
- keep a live tensor resident
- hand the same tensor semantics to the Fortran-guided path

## Consequences

### Good

- The accelerated `v4` loop now consumes a persistent live tensor state.
- The integration is backward-compatible with older guidance builders.
- Diagnostics now expose whether accepted mutations stayed incremental or forced
  reanchoring.

### Bad

- End-to-end speedup is still not guaranteed.
- The integrated path still pays for Python-side scoring, fingerprint writes,
  and artifact generation.
- Anchor extension reduces churn, but it does not yet erase the remaining
  whole-loop overhead.

## Benchmark Read

Artifact:

- [fortran_v4_phase4_benchmark.json](/C:/Code/Project%20RBT/project_rbt/data/validation/fortran_v4_phase4_benchmark.json)

Small paired benchmark result:

- `python_only`: `4.378783s`
- `auto_batch`: `5.179153s`
- speedup: `0.845463x`

The integrated path is therefore still slower at this scale.

The key explanatory signal is in the tensor-state update breakdown for the
accelerated run:

- `seed_build = 1`
- `anchor_extend = 2`

That means the engine is now using the new path correctly *and* absorbing the
accepted OOV-heavy mutations without dropping into full rebuild in this paired
benchmark.

The integrated path still does not beat Python-only end to end, but it is much
closer to parity than the first integration pass.

## Next Sharp Move

The next real performance step is no longer "reduce full rebuilds at all costs."
That part is already improved.

The next likely payoff is reducing the remaining Python-side whole-loop cost:

1. separate token-form novelty from tensor-anchor identity more cleanly, or
2. benchmark with more accepted proposals so the cheaper state refresh has more
   chances to amortize setup costs, or
3. move more post-accept bookkeeping off the hot path

Without that, the engine will keep paying enough Python overhead after accepted
mutations that Phase 4 cannot yet cash in its full theoretical runtime win.
