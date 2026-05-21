# Old French Validator Pilot

Date: 2026-04-08

## Purpose

Run the first attested historical comparison against the French
`v2_convergence` checkpoint ladder.

This is a pilot validator pass, not a definitive historical validation study.

## Validator corpus

The local `old_french` packet was assembled from three attested witnesses /
transcriptions:

1. `Sequence de sainte Eulalie`
2. `Serments de Strasbourg`
3. `La Vie de saint Alexis` (full locally extracted render)

Raw sources live in:

- `data/raw/historical/old_french/`

Processed validator outputs:

- `data/processed/historical/old_french_tokens.json`
- `data/processed/historical/old_french_manifest.json`
- `data/matrices/old_french_*`

Validator corpus stats:

- files: `3`
- sequences: `661`
- total tokens: `4620`
- unique types: `1378`
- type/token ratio: `0.298268`
- mean sequence length: `6.989410`

## Compared checkpoints

The default French `v2_convergence` ladder:

1. `FR_v2_000`
2. `FR_v2_015`
3. `FR_v2_030`
4. `FR_v2_045`
5. `FR_v2_058`
6. `FR_v2_061`

Comparison output:

- `data/validation/french_v2_convergence_vs_old_french.json`

## Results

### Structural best match

Best by structural distance and structural cosine:

- stage: `FR_v2_061`
- structural distance: `2.026658`
- structural cosine: `0.999984`

Structural distance improved steadily across the ladder:

- `FR_v2_000`: `2.088336`
- `FR_v2_015`: `2.064684`
- `FR_v2_030`: `2.055166`
- `FR_v2_045`: `2.038354`
- `FR_v2_058`: `2.026795`
- `FR_v2_061`: `2.026658`

### Form best match

Best by orthographic / form similarity:

- stage: `FR_v2_000`
- validator form score: `0.755981`

Form similarity declined as the bridge moved toward its stable Latin-conditioned
endpoint:

- `FR_v2_000`: `0.755981`
- `FR_v2_015`: `0.736095`
- `FR_v2_030`: `0.691216`
- `FR_v2_045`: `0.677117`
- `FR_v2_058`: `0.673150`
- `FR_v2_061`: `0.673241`

## Interpretation

The pilot result splits cleanly:

1. Structurally, the later bridge checkpoints look more Old-French-like than the
   starting French sample.
2. In surface form, the starting French sample still looks more similar to the
   small Old French validator packet than the later Latin-conditioned bridge does.

That means the current `v2` path is not a simple "more historical in every way"
gradient. It appears to move toward the validator in structural space while moving
away from it in orthographic / suffix-form space.

This is a meaningful result. It suggests that:

- the bridge may be learning a structural historical drift,
- while the present form reward is still mainly Latin-seeking rather than
  intermediate-seeking.

## Limitations

This validator pass should still be interpreted cautiously:

1. The validator packet is small.
2. It mixes very early witnesses with a longer `Alexis` witness.
3. It is still a hand-assembled starter corpus, not yet a full scholarly historical corpus.
4. Structural cosine remains numerically compressed and should not be over-read.

## Next move

The immediate next move is to strengthen the validator side:

1. expand `old_french` with more attested text
2. rerun the same checkpoint comparison
3. compare the pilot result to a larger scholarly corpus such as BFM or OTA `0176`

Even in its current small form, the pilot already gives the project something it did
not previously have: a real attested-stage comparison with a directional result.
