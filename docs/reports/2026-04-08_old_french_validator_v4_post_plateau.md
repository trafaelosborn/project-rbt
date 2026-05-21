# Old French Validator: V4 Post-Plateau Follow-up

Date: 2026-04-08

## Purpose

Compare the `v4_post_plateau_50pct_probe` ladder against the local `old_french`
validator packet and determine whether the extra post-plateau Latin-directed
movement remains historically legible.

This is a follow-up to both:

1. the earlier `v2_convergence` Old French pilot
2. the first `v4_from_v3_endpoint` Old French follow-up

## Validator corpus

The validator packet is unchanged:

1. `Sequence de sainte Eulalie`
2. `Serments de Strasbourg`
3. `La Vie de saint Alexis` (full locally extracted render)

Processed validator outputs:

- `data/processed/historical/old_french_tokens.json`
- `data/processed/historical/old_french_manifest.json`

Validator corpus stats:

- files: `3`
- sequences: `661`
- total tokens: `4620`
- unique types: `1378`
- type/token ratio: `0.298268`
- mean sequence length: `6.989410`

## Compared checkpoints

Full `v4_post_plateau_50pct_probe` ladder:

1. `FR_v4_000`
2. `FR_v4_001`
3. `FR_v4_002`
4. `FR_v4_003`
5. `FR_v4_004`

Comparison output:

- `data/validation/french_v4_post_plateau_50pct_probe_vs_old_french.json`

## Results

### Structural best match

Best by structural distance and structural cosine:

- stage: `FR_v4_004`
- structural distance: `1.985277`
- structural cosine: `0.999986`

Structural distance improved across the post-plateau branch:

- `FR_v4_000`: `1.995647`
- `FR_v4_001`: `1.994163`
- `FR_v4_002`: `1.990006`
- `FR_v4_003`: `1.990770`
- `FR_v4_004`: `1.985277`

### Form best match

Best by orthographic / form similarity:

- stage: `FR_v4_001`
- validator form score: `0.590938`

Form behavior in the post-plateau branch was nearly flat, with only a tiny local
uptick before drifting back down:

- `FR_v4_000`: `0.590564`
- `FR_v4_001`: `0.590938`
- `FR_v4_002`: `0.588614`
- `FR_v4_003`: `0.588520`
- `FR_v4_004`: `0.588722`

## Comparison to the earlier v4 validator run

Relative to the prior `v4_from_v3_endpoint` Old French comparison:

- best structural distance improved from `1.995647` to `1.985277`
- the structural best stage moved from `FR_v4_006` to the new endpoint `FR_v4_004`
- form similarity did not recover the earlier global best; it stayed around
  `0.59`, well below the earlier `FR_v4_000 = 0.611085`

So the extra post-plateau movement stayed historically legible on the structural
axis and even strengthened it further, while surface-form similarity remained
mostly flat-to-down.

## Interpretation

This is the clearest answer yet on what the extra continuation is doing.

1. The post-plateau branch is not merely target-chasing noise.
2. It continues to improve the Old-French structural match.
3. It does not restore the surface-form side of the split.

The cleanest read is that the current continuation is still following a historically
legible structural path even as it keeps drifting away from the validator packet in
surface form relative to the earliest stages.

## Limitations

This remains a provisional validator result:

1. The `old_french` packet is still small.
2. It is hand-assembled rather than a large scholarly corpus release.
3. Structural cosine remains compressed and should not be over-read.

## Next move

The most useful next moves are now:

1. add a hold-out Latin split so the endpoint has an out-of-loop target check
2. rerun the same validator comparison against a richer Old French corpus such as
   OTA `0176`
3. decide whether endpoint selection should be explicitly multi-objective across
   Latin reward, family alignment, and attested-validator structure
