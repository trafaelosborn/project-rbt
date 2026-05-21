# 030 - Cosine Acceleration: Python Bug Fix + Fortran Layer

**Date:** 2026-04-13  
**Status:** Implemented

## Context

The v5 production run requires a fresh trajectory from the original French corpus.
Before launch, we need substantially higher throughput than the 874 p/h measured in
the v5 candidate count benchmark.

Doc 028 identified the Fortran batch scorer as the pending acceleration item. This
doc covers the investigation, the actual bottleneck discovered, and the fixes.

## What we actually profiled

The assumption was: cosine scoring in `IncrementalScoringState._score_from_counters`
was the bottleneck. Profiling revealed a different picture:

| Component | Time per proposal (c=16) |
|-----------|--------------------------|
| `_score_from_counters` (Python cosine) | 53ms |
| `compute_sequence_delta` | 3ms |
| `_mutate_candidate` × 16 | **~11,000ms** |

Mutation generation was the bottleneck, not scoring.

## Root cause in `_mutate_candidate`

Profiling `_apply_named_operator` per operator:

| Operator | Old time | Weight |
|----------|----------|--------|
| `suffix_family_rewrite` | 995ms median | 13% |
| `paradigm_family_rewrite` | 89ms median | 13% |
| `sequence_span_rewrite` | 32ms median | 14% |
| `token_char_edit` | 7ms median | 15% |

`suffix_family_rewrite` was 70x slower than it should be.

## Root cause: `_sparse_profile_cosine` union bug

```python
# OLD — iterates over set(a) | set(b) = O(|a| + |b|)
dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in set(a) | set(b))
```

For keys in `b` but not in `a`, `a.get(k) = 0.0` and the term contributes zero.
The union was mathematically correct but O(|a| + |b|) instead of O(|a|).

For `score_token` on a single token (|a| = 7 char bigrams), against the Latin
reference (|b| = 1018 bigrams), this meant **1018 iterations instead of 7** — a
145x overhead.

`suffix_family_rewrite` calls `score_token` ~750 times per invocation (8 candidate
suffixes × ~93 tokens in the family). Each `score_token` call did 3 cosine
operations with the full Latin reference. Total: 750 × 3 × 1018 iterations ≈ 2.3M
Python iterations per operator invocation.

## Fix 1: `_sparse_profile_cosine` — iterate over `a` only

```python
# NEW — mathematically identical, O(|a|) instead of O(|a| + |b|)
dot = sum(v * b.get(k, 0.0) for k, v in a.items())
```

Cross-set terms (k ∈ b, k ∉ a) always contribute zero to the dot product.

## Fix 2: `LatinFormReference.score_token` — fast path + precomputed norms

The fixed `_sparse_profile_cosine` still recomputes the Latin reference norm
`nb = sum(v*v for v in b.values()) ** 0.5` on every call. For the Latin reference
(1018-2500 entries), this is O(|b|) per call.

Solution:
1. Precompute `_bg_ref_norm`, `_tg_ref_norm`, `_sfx_ref_norm` once at init.
2. Rewrite `score_token` with an inline fast path that:
   - Extracts char n-grams for the single token directly (no `_extract_char_ngrams_from_sequences`)
   - Uses precomputed reference norms in the cosine computation
   - Avoids calling the full `self.score([[token]])` path

`score_token` speedup (uncached): 205µs → 10µs = **20x**.

## Fix 3: Fortran sparse cosine extension

A Fortran extension (`sparse_cosine.f90`) implementing `batch_form_scores_f32` was
compiled and wired into `IncrementalScoringState` via `FortranCosineScorer`. This
provides a secondary acceleration for corpus-level scoring (the `_score_from_counters`
path), replacing Python dict-intersection cosines with BLAS-backed dense float32
dot products.

The Fortran layer is correct to float32 precision (max error < 1e-7 vs numpy).

Architecture:
- `src/accelerate/sparse_cosine.f90` — Fortran module with `batch_form_scores_f32`
- `src/accelerate/fortran_cosine.py` — `FortranCosineScorer` with `score_single_form`
  and `score_form_batch`; numpy fallback if Fortran unavailable
- `IncrementalScoringState.from_sequences(fortran_cosine_scorer=...)` — optional
  scorer passed in from engine; replaces 3 Python cosines in `_score_from_counters`
- `ReinforcedV4Config.use_fortran_cosine: bool` — enables Fortran cosine at engine level
- `RunConfig.use_fortran_cosine: bool` — propagates through `LongRunConfig` and
  `_driver_adapter` to the engine

## Measured throughput

150-proposal benchmark from the v4 endpoint (c=16, seed=42):

| Condition | p/h |
|-----------|-----|
| v5 candidate count benchmark (doc 028) | 874 |
| After cosine fixes (Python + Fortran) | **18,867** |

**21.6x improvement.** The Fortran layer provides a secondary contribution on top
of the Python fix; both are active in the measured number.

## Operator timings after fix

| Operator | Before | After | Speedup |
|----------|--------|-------|---------|
| `suffix_family_rewrite` | 995ms | 14ms | 70x |
| `paradigm_family_rewrite` | 89ms | 4ms | 22x |
| `sequence_span_rewrite` | 32ms | 1.3ms | 24x |
| `token_char_edit` | 7ms | 0.8ms | 9x |

## Next step

The v5 production run can now be launched from the original French corpus. At
~18,000 p/h, a 48-hour window yields ~864,000 proposals — sufficient for
meaningful structural convergence from scratch.

Transparency (use_semantic_transparency) must NOT be enabled on the first
production run. It is a separate experimental condition requiring its own
trajectory and ablation.
