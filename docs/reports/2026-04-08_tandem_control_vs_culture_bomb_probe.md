# Tandem Control vs Culture-Bomb Probe

Date: 2026-04-08

## Purpose

Run two cheap continuation branches in tandem from the same seed corpus:

1. a plain `v4` continuation control
2. a shock-enabled `v5` branch with plateau-triggered culture bombs

The goal is to test whether exogenous-shock rescue produces a meaningfully
different path from the current smooth continuation.

## Assumption used

The user prompt suggested keeping the search alive through a plateau window "as
long as the amount of total iterations that preceded it."

For this probe, that was operationalized as:

- plateau window = `10`

Rationale:

- the current French `v4` lineage had `10` accepted mutation stages before this
  branch point (`6` in `v4_from_v3_endpoint` plus `4` in the first post-plateau
  probe)

## Shared seed

Both branches start from:

- `data/retrodiction/french/v4_post_plateau_50pct_probe/corpora/FR_v4_004_tokens.json`

## Branch A: v4 control

Output:

- `data/retrodiction/french/v4_control_long_from_post_plateau/`

Summary:

- total stages: `4`
- accepted mutation stages: `3`
- proposals attempted: `15`
- halt reason: `stable`
- best / final stage: `FR_v4_003`

Raw-axis improvement from the shared seed:

- Latin structural score: `-1.288622 -> -1.284454`
- Latin form score: `0.808937 -> 0.809516`
- family alignment score: `0.520696 -> 0.517184`

## Branch B: v5 culture-bomb continuation

Output:

- `data/retrodiction/french/v5_culture_bomb_from_post_plateau/`

Summary:

- total stages: `4`
- accepted mutation stages: `3`
- proposals attempted: `15`
- halt reason: `culture_bomb_plateau`
- culture bombs used: `1`
- best / final stage: `FR_v5_003`

Observed behavior:

1. `v5` tracked the same accepted path as the plain `v4` continuation
2. the plateau trigger did fire once
3. the culture bomb failed to produce an improving rescue candidate
4. the run ended on the same endpoint as the control branch, just under a more
   explicit halt label

## Main result

Under this configuration, the smooth continuation still had real headroom.

The culture-bomb branch did not discover a new basin beyond the control. It first
followed the same three accepted moves as the plain `v4` branch, then failed to
escape the next plateau when the shock operator was invoked.

So the answer from this tandem probe is:

1. continuing was worth it
2. the smooth line still produced meaningful movement
3. the first culture-bomb implementation did not outperform the smooth branch

## Historical check

The control endpoint was compared against the local `old_french` validator packet:

- comparison output:
  `data/validation/french_v4_control_long_from_post_plateau_vs_old_french.json`

Best structural match:

- `FR_v4_003`
- structural distance: `1.979246`

Best form match:

- `FR_v4_001`
- validator form score: `0.588735`

This improves the Old-French structural match again relative to the prior
post-plateau probe:

- `1.985277 -> 1.979246`

So the extra smooth continuation did not merely chase Latin. It also improved the
attested-validator structural signal again.

## Interpretation

The current picture is:

1. the French bridge still has smooth continuation room under the present scorer
2. the first exogenous-shock implementation is not yet buying us anything extra
3. structural historical legibility continues to improve even as family alignment
   and surface-form metrics remain in tension

## Practical takeaway

For now, the control branch wins.

Not because culture shocks are a bad idea in principle, but because the first shock
operator did not beat ordinary continuation under the current score geometry.

## Next move

Two good next moves now exist:

1. keep extending the smooth continuation in measured blocks while logging the
   validator axes
2. if revisiting shocks, make them more structured or lower-penalty rather than
   simply louder
