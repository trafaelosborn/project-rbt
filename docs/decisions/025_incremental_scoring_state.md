# 025 — Incremental Scoring State

**Date:** 2026-04-09  
**Status:** Implemented  
**Supersedes:** nothing (additive to 022, 024)

## Problem

The v4 engine running a 15,000-proposal French continuation was producing ~684
proposals/hour on this machine. At that rate, reaching 1M proposals would take
61 days. The research goal requires hundreds of thousands to millions of proposals
to adequately sample linguistic space.

Profiling the hot loop revealed three stacked O(corpus_size) operations per
candidate:

| Operation | Cost per call | Calls per proposal |
|---|---|---|
| `score_token` inside operators | ~0.26ms cold | 500–4000 |
| `_token_counts` + `_bigram_counts` in `_mutate_candidate` | ~20ms | 8 |
| `_evaluate_sequences` (char + word ngrams) | ~85ms | 8 |

Total measured: ~5,264ms per 8-candidate proposal. Observed rate: 684/hour.

## Solution

Three acceleration layers added via `IncrementalScoringState`
(`src/accelerate/incremental_scoring_state.py`):

### Layer 1 — `score_token` cache

`LatinFormReference.score_token` is a pure function: same token, same score,
always. Added `_score_token_cache: dict[str, float]` to `LatinFormReference`
(two lines in `engine_reinforced_v2.py`). After the first call, any repeated
token lookup is a dict read (~0.0001ms). Cache is valid for the engine lifetime
since the Latin reference never changes.

Measured speedup: **1868x** on repeated token lookups.

### Layer 2 — precomputed token/bigram counts

`_mutate_candidate` recomputed `_token_counts` and `_bigram_counts` from the
full corpus for every one of the 8 candidate calls per proposal. Since the
corpus only changes on acceptance (not between candidates), these are computed
once per proposal from `IncrementalScoringState.token_counts` and
`word_bigram_counts` (zero-cost properties backed by committed state) and passed
as optional args to the v4 override of `_mutate_candidate`.

### Layer 3 — incremental `evaluate()`

`IncrementalScoringState` maintains running Counters for:
- Character bigrams (weighted by token occurrence)
- Character trigrams
- Suffixes (length 3)
- Word bigrams
- Word trigrams
- Token counts, total tokens, total sequence length

Per candidate, `evaluate()` calls `compute_sequence_delta()` to find which
sequences changed (typically 1–5% of 800), then applies the delta to Counter
copies for those sequences only. The rest of the corpus is never touched.

The scoring computation from those counters is the same math as
`_evaluate_sequences` — tested to floating-point equality.

`commit(new_sequences)` updates all running state in-place after each accepted
stage.

## Results

Benchmark on the live French v4 corpus (800 sequences, 12,426 tokens, 8
candidates per proposal):

| Metric | Baseline | Incremental | Speedup |
|---|---|---|---|
| `score_token` (200 calls, warm) | 51.6ms | 0.03ms | **1868x** |
| `evaluate()` × 8 | 826ms | 360ms | **2.3x** |
| Full proposal (mutation + scoring) | 5,264ms | 246ms | **21.4x** |
| Proposals/hour (estimated) | 684 | 14,655 | **21x** |

## Constraints preserved

- `use_incremental_scoring=False` restores the exact original v4 loop
- All operator logic, reward shaping, coherence gating, and Hungarian
  alignment are untouched
- `evaluate()` is verified equal to `_evaluate_sequences()` by 11 tests
  (see `tests/test_incremental_scoring_state.py`)
- v2, v3, v5 engines are unchanged

## What's next

At 14,655 proposals/hour, 1M proposals ≈ 68 hours. That's progress but still
serial. The next layer is parallelism: run N engine instances across N CPU cores,
each with its own `IncrementalScoringState`. Python's `multiprocessing` or a
worker pool can distribute proposals without any shared-state coordination since
each candidate is independent.
