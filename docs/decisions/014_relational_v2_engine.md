# Decision 014: Relational Reinforced V2 Engine

Date: 2026-04-08

## Context

The reward-fixed reinforced engines in `src/retrodiction/engine_reinforced.py` can
improve French toward Latin in the trainable structural subspace, but they operate on
a frozen source vocabulary. They can rearrange or reweight French tokens, but they
cannot mutate token forms or model knock-on effects across related words.

That makes the original reinforced engines useful structural baselines, but too narrow
for the project's next question: what happens when the bridge itself is allowed to
change forms rather than only transition weights?

## Decision

Add `src/retrodiction/engine_reinforced_v2.py` as a second reinforced search engine
family.

The v2 engine works directly on sampled corpora rather than on a fixed bigram matrix.
Each accepted stage becomes the new baseline corpus, and each proposal mutates the
baseline at one of several scales:

1. token-level character edit
2. suffix-family rewrite across related token types
3. local bigram order swap
4. token split
5. bigram merge

Candidate scoring combines:

1. the existing Latin structural reward from `LatinReference`
2. a new Latin form reward from character n-grams and suffix profiles
3. the existing coherence margin against the Markov floor
4. a small mutation-cost regularizer

Form mutations are no longer purely random. Token edits and suffix rewrites first
sample several candidate changes and keep the locally most Latin-like proposal before
global scoring.

## Rationale

This keeps Latin inside the optimization loop, which is consistent with the active
single-blind methodology, while widening the search space enough to produce readable
bridge drift rather than only fixed-vocabulary reshuffling.

The multi-scale operator mix is intentionally evolutionary rather than purely
parametric:

- some proposals affect a single form
- some propagate across suffix families
- some perturb local order relations

That better matches the project's working intuition that linguistic change can have
both local and system-wide effects.

## Consequences

Positive:

- Reinforced search can now mutate actual token forms.
- Accepted stages produce a visibly changing bridge corpus.
- Operator counts become part of the experimental signal.
- The pipeline now has a dedicated `reinforced_v2` step.

Limitations:

- The engine is still heuristic and target-conditioned.
- It does not yet model latent mutations or delayed expression.
- Structural movement is still modest relative to form movement in the first French
  run.
