# French Reinforced V3 Probe

Date: 2026-04-08

## Purpose

Test whether two changes can break the French `v2` plateau:

1. stranger perturbations
2. stronger Latin-side reward when a move is jointly correct

The new `v3` engine adds bundled and family-level mutation operators plus reward
amplification for candidates that improve the active Latin signals together.

## Implementation

Code:

- `src/retrodiction/engine_reinforced_v3.py`
- `tests/test_engine_reinforced_v3.py`

New operator families:

1. `function_word_burst`
2. `paradigm_family_rewrite`
3. `macro_bundle_rewrite`

New reward diagnostics logged per accepted stage:

1. `reward_struct_gain`
2. `reward_form_gain`
3. `reward_suffix_gain`
4. `reward_trigram_gain`
5. `reward_bonus`
6. `reward_penalty_relief`
7. `reward_effective_total_score`

## Run A: fresh French -> Latin v3

Output:

- `data/retrodiction/french/v3/`

Configuration:

- default `ReinforcedV3Config`

Result:

- total stages: `3`
- accepted mutation stages: `2`
- proposals attempted: `25`
- halt reason: `stable`
- best / final stage: `FR_v3_002`

Scores:

- effective total score: `-0.814688 -> -0.359718`
- Latin structural score: `-1.378939 -> -1.371337`
- Latin form score: `0.566157 -> 0.614681`
- coherence: `coherent` throughout

Accepted operator mix:

- `macro_bundle_rewrite = 2`

Interpretation:

The default `v3` run clearly moves harder than early `v2`, but the big total-score
jump is partly a reward-amplification effect. On the underlying raw Latin axes, this
fresh restart still does not outperform the mature `v2` endpoint.

That means `v3` is not yet a clean replacement for `v2` as a from-scratch engine.

## Run B: v3 continuation from `FR_v2_061`

Starting state:

- `data/retrodiction/french/v2_convergence/corpora/FR_v2_061_tokens.json`

Output:

- `data/retrodiction/french/v3_from_v2_endpoint/`

Result:

- total stages: `9`
- accepted mutation stages: `8`
- proposals attempted: `57`
- halt reason: `stable`
- best / final stage: `FR_v3_008`

Scores from the `v2` endpoint seed to the `v3` continuation endpoint:

- effective total score: `-0.579637 -> -0.474742`
- Latin structural score: `-1.306957 -> -1.302373`
- Latin form score: `0.762744 -> 0.798087`
- language-likeness margin: `3.105235 -> 3.103579`
- coherence: `coherent` throughout

Accepted operator mix:

- `function_word_burst = 4`
- `suffix_family_rewrite = 2`
- `paradigm_family_rewrite = 1`
- `macro_bundle_rewrite = 1`

Interpretation:

This is the real success case.

Unlike the meso-scale continuation probe, the `v3` continuation did push past the
old plateau. It improved both raw Latin signals, not just the amplified total score.
The gains are modest on the structural axis and larger on the form axis, but they are
real and they survived to a new stable endpoint.

## Preview read

Best preview:

- `data/retrodiction/french/v3_from_v2_endpoint/previews/FR_v3_008_preview.txt`

Qualitative read:

- still not Latin
- still visibly synthetic
- more aggressively form-shifted than `FR_v2_061`
- still above the coherence floor rather than collapsing into noise

The bridge reads like a harsher, more bundled continuation of the v2 attractor rather
than a completely new path.

## Practical takeaway

The `v2` plateau was not absolute.

It appears to have been relative to the `v2` operator family and scoring geometry.
Once the project added stranger mutations and louder reward for jointly good Latin
moves, the French endpoint moved again.

That does not mean the project has reached Latin. It does mean the search still had a
live improvement basin above `FR_v2_061`, and `v3` found part of it.
