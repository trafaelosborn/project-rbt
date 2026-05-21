# Decision 017: Relational Reinforced V3 Engine

Date: 2026-04-08

## Context

The reinforced `v2` engine widened the search space enough to produce a real synthetic
French -> Latin bridge, but the long convergence run plateaued at `FR_v2_061`.

A follow-up meso-scale probe showed that simply increasing the perturbation window was
not enough. The best span-heavy continuation candidate improved the raw objective by
only about `0.000004` before cost, and the mutation penalty immediately erased that
gain.

So the next bottleneck was no longer just locality. The search needed:

1. stranger mutation families
2. a stronger Latin-side reward when a candidate is genuinely good on multiple axes

## Decision

Add `src/retrodiction/engine_reinforced_v3.py` as an experimental successor to `v2`.

The `v3` engine keeps the relational corpus-mutation design from `v2`, but adds:

1. `function_word_burst`
   coordinated rewrites of several short, high-frequency tokens in one move
2. `paradigm_family_rewrite`
   prefix-linked family rewrites across many related word types at once
3. `macro_bundle_rewrite`
   bundled multi-operator proposals that can mix local edits, family rewrites, and
   span rewrites into one accepted mutation

Candidate scoring is also amplified. In addition to the base `v2` score, `v3` adds:

1. explicit reward on structural gain relative to the current baseline
2. explicit reward on form gain relative to the current baseline
3. extra reward on suffix and trigram gains
4. a small joint bonus when structure and form improve together
5. penalty relief when the move improves both Latin axes without materially hurting
   coherence

## Rationale

This is a deliberate attempt to break the French plateau without abandoning the
project's interpretability.

The engine is still heuristic and target-conditioned, but the new operators let the
search make coordinated, system-level moves instead of only local edits. The amplified
reward makes Latin "speak louder" when a move is jointly good rather than only
marginally better in one narrow sub-score.

## Consequences

Positive:

- The search now has genuinely weird mutation operators rather than just larger windows.
- Accepted moves can propagate across multiple related tokens in one step.
- Reward shaping is now explicit and inspectable in per-stage diagnostics.

Observed immediately:

- A default French `v3` run from the original source corpus accepted two
  `macro_bundle_rewrite` moves very quickly, but did not outperform the best raw `v2`
  endpoint on the underlying Latin structural and form axes.
- A continuation run starting from `FR_v2_061` did improve both raw Latin signals:
  `latin_structural_score = -1.306957 -> -1.302373`
  `latin_form_score = 0.762744 -> 0.798087`

Interpretation:

- Weird mutations plus louder reward do create new motion.
- The cleanest result is not a full restart from French, but a continuation from the
  current `v2` attractor.
- `v3` is best understood, for now, as a breakout engine layered on top of `v2`, not
  a replacement for all earlier runs.
