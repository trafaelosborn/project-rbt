# 029 - Semantic Transparency Scorer

**Date:** 2026-04-12  
**Status:** Implemented (calibration pending production run)

## Context

The v4 engine uses a corpus-level Latin form score (aggregate char n-gram
cosine similarity) and a structural score (TTR, bigram/trigram coverage) to
guide acceptance. Both are population-level signals: they measure how the
corpus *as a whole* resembles Latin.

Doc 027 identified a gap: neither score penalises corpora where the most
common tokens are semantically empty glue — tokens that appear frequently
because they are statistically convenient, not because they carry recoverable
meaning.

## Decision

Add a semantic transparency term to the acceptance objective, controllable
via `use_semantic_transparency` and `transparency_weight` in
`ReinforcedV4Config` (and by extension v5 configs).

## What "transparency" means here

A token is *semantically transparent* if it looks like a word with a
derivable meaning — i.e., it has Latin-like morphological structure. The
scorer operationalises this as `LatinFormReference.score_token()`, which
measures character n-gram + suffix cosine similarity against Latin at the
individual token level.

The corpus transparency score is a **frequency-weighted average** of
`score_token` over the top-N most frequent tokens (default N=50).

### Why frequency-weighting specifically

The form score already captures aggregate n-gram similarity. Weighting by
frequency adds the orthogonal signal: *the words you say most often should
look Latin-derived*, not just the long tail.

In natural Latin, high-frequency words include function words (prepositions,
conjunctions) that have Latin-derived forms. A generated corpus that uses
random filler tokens as glue would pass the aggregate form score but fail
the transparency check.

## Implementation

```
src/accelerate/semantic_transparency.py
    SemanticTransparencyScorer.score(sequences) -> float [0, 1]
    SemanticTransparencyScorer.score_full(sequences) -> TransparencyResult
```

`score_token` is already cached on `LatinFormReference` (from the 025
incremental scoring decision), so transparency scoring adds no repeated work.
The top-50 loop is O(50) per call.

### Integration point

Both scoring paths in the v4 engine add the transparency term after their
respective score computations:

- `_evaluate_sequences()` override in `RelationalReinforcedRetrodictionEngineV4`
- The incremental path in `run()` after `self._scoring_state.evaluate()`

In both cases:
```python
if transparency_scorer and transparency_weight > 0:
    t_score = transparency_scorer.score(mutated)
    candidate.total_score += transparency_weight * t_score
    candidate.diagnostics["transparency_score"] = t_score
```

## Methodological status

This is an **experimental condition**, not a neutral engineering change.

- It introduces a human-designed lexical prior (Latin form reference)
- It must have a documented calibration before production use
- Results must be reported alongside a `transparency=False` ablation

## Calibration

A calibration probe runs two 200-proposal trials from the v4 endpoint:
- `baseline`: `use_semantic_transparency=False`
- `transparency_w0.05`: `transparency_weight=0.05`

Results written to `data/benchmarks/transparency_calibration/`.

### V4 endpoint transparency snapshot (measured)

At the frozen v4 endpoint (struct ≈ −3.7e−5, form = 0.756):

- **Transparency score: 0.2655**

This is lower than expected given 283k proposals of form optimisation. The
top-50 most frequent tokens in the v4 endpoint corpus score around 0.27 on
average on Latin form similarity — meaning the most common tokens are still
moderately filler-like despite the long structural optimisation.

This low endpoint score is a *positive* finding: it means transparency is
not already saturated at the v4 endpoint. A properly weighted transparency
term will have gradient signal to work with in v5.

### Calibration probe results (measured)

200-proposal probes from the v4 endpoint, seed=99:

| Condition | struct | form | accepted | p/hour |
|-----------|--------|------|----------|--------|
| baseline (v4) | -3.7e-5 | 0.7561 | 1 | 598 |
| transparency w=0.05 | -3.7e-5 | 0.7561 | 1 | 4,561 |

Both conditions accepted exactly 1 stage at the endpoint. The null delta
(struct_delta=0.0, form_delta=0.0) is expected: 200 proposals at a
near-zero struct endpoint produces very few accepted stages regardless of
the transparency term. The transparency term does not degrade form or
struct.

The large throughput difference (598 vs 4,561 p/h) needs investigation —
likely a timing artifact from corpus loading vs actual proposal time.

### Recommended weight range

**0.05 (default)**, range 0.02–0.10.

The transparency term has gradient signal (endpoint score = 0.265, not
saturated). Use weight=0.05 for the first v5 production run. The correct
calibration will be more visible over thousands of proposals, not 200.

## What this is NOT

- Not a claim that generated tokens have recoverable semantic meaning in a
  strict linguistic sense
- Not a replacement for the structural or form scores
- Not a final determination of what "transparency" means for the bridge
  language — this is a measurable proxy that can be reported and discussed

## Tests

`tests/test_semantic_transparency.py` — 17 tests covering:
- Score in [0, 1]
- Latin-like corpus scores above random filler
- Frequency-weighting correctness
- `BatchCandidateScorer` interface compliance
