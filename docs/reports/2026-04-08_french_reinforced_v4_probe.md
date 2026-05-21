# French Reinforced V4 Probe

Date: 2026-04-08

## Purpose

Implement Phase 2 of the proposed `v4` direction:

1. keep the `v3` mutation and reward stack
2. use Hungarian family alignment to schedule operator weights dynamically
3. test whether that controller can push the French bridge beyond the current `v3`
   endpoint

## Implementation

Code:

- `src/retrodiction/engine_reinforced_v4.py`

Tests:

- `tests/test_engine_reinforced_v4.py`
- `tests/test_validation_hungarian_alignment.py`

The new engine computes a current family alignment score against Latin, converts it to
an inverse-log weirdness level, and uses that to bias top-level operator weights.

In this Phase 2 version:

- selection changes
- acceptance does not

## Starting point

Input corpus:

- `data/retrodiction/french/v3_from_v2_endpoint/corpora/FR_v3_008_tokens.json`

Output:

- `data/retrodiction/french/v4_from_v3_endpoint/`

## Result

Run summary:

- total stages: `7`
- accepted mutation stages: `6`
- proposals attempted: `51`
- halt reason: `stable`
- best / final stage: `FR_v4_006`

Accepted operator counts:

- `function_word_burst = 1`
- `macro_bundle_rewrite = 2`
- `token_char_edit = 2`
- `paradigm_family_rewrite = 1`

Scores from the `v3` seed to the `v4` endpoint:

- total score: `-0.548276 -> -0.476586`
- Latin structural score: `-1.302373 -> -1.295360`
- Latin form score: `0.798138 -> 0.806686`
- coherence: `coherent` throughout

## The important nuance

The family-alignment signal did **not** improve monotonically.

Selected family alignment scores:

- `FR_v4_000 = 0.539860`
- `FR_v4_001 = 0.540554`
- `FR_v4_002 = 0.540121`
- `FR_v4_003 = 0.539426`
- `FR_v4_004 = 0.538500`
- `FR_v4_005 = 0.538535`
- `FR_v4_006 = 0.536294`

So the best family-aligned stage is:

- `FR_v4_001`

while the best raw-score endpoint is:

- `FR_v4_006`

## Interpretation

This is a meaningful but mixed result.

What worked:

- alignment-driven operator scheduling reopened movement
- raw Latin structural and form scores both improved again
- coherence remained intact

What did not fully work yet:

- the alignment controller did not keep the run climbing on the alignment axis
- the stable endpoint traded a small amount of family alignment for better raw
  structural/form scores

That means `v4` Phase 2 is useful, but not yet self-justifying as a final controller.
It helped the search move, but it did not yet unify all diagnostics into a single
upward path.

## Practical takeaway

The current best read is:

- `v3` proved the bridge could move past `v2`
- `v4` proved alignment-driven scheduling can move the bridge further
- but family alignment is not yet strong enough as a control signal by itself

This suggests two plausible next moves:

1. compare the `v4` ladder against attested intermediate validators
2. if staying on the `v4` track, let alignment influence acceptance or penalty relief
   more directly rather than only operator choice
