# Decision 019: Add Alignment-Driven Operator Scheduling in Reinforced V4

Date: 2026-04-08

## Context

The project completed two precursor steps:

1. `v3` showed that stranger mutations plus louder Latin reward can push French
   beyond the old `v2` plateau.
2. the Hungarian family-alignment diagnostic showed that `v3` also improves under an
   independent global family-level measure.

That made it reasonable to test Phase 2 of the proposed `v4` direction:

- use global family alignment to schedule mutation weirdness
- but do not yet let that signal directly change acceptance criteria

## Decision

Add `src/retrodiction/engine_reinforced_v4.py`.

This engine keeps the `v3` mutation family and reward stack, but changes operator
selection:

1. current bridge state is scored against a Latin family inventory under Hungarian
   assignment
2. family alignment score is converted into an inverse-log weirdness level
3. weirdness changes the top-level operator weights dynamically

In this Phase 2 version:

- proposal selection is alignment-aware
- acceptance remains the existing `v3` total-score rule
- bundle size and penalty relief are not yet scheduled by alignment

## Rationale

This is the smallest real test of the `v4` idea.

It asks whether a global alignment controller can improve the search without yet
handing the controller full authority over acceptance or mutation depth.

## Consequences

Positive:

- The search no longer uses only fixed operator weights.
- Alignment is now present inside the generation loop, not only in validation.
- Every accepted stage records:
  - `family_alignment_score`
  - `family_alignment_cost`
  - `weirdness_level`
  - scheduled operator weights

Observed immediately on the French continuation probe from `FR_v3_008`:

- raw Latin structural score improved:
  `-1.302373 -> -1.295360`
- raw Latin form score improved:
  `0.798138 -> 0.806686`
- total score improved:
  `-0.548276 -> -0.476586`
- coherence remained `coherent`

But:

- family alignment did not improve monotonically
- it peaked early at `FR_v4_001 = 0.540554`
- the stable endpoint `FR_v4_006` ended slightly lower at `0.536294`

Interpretation:

- alignment-driven scheduling is strong enough to reopen movement
- it is not yet sufficient to guarantee monotonic improvement on the alignment axis
- Phase 2 therefore worked as a controller experiment, but it does not yet justify
  making family alignment a full optimization target
