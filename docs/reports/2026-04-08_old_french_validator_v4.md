# Old French Validator: V4 Follow-up

Date: 2026-04-08

## Purpose

Compare the full French `v4_from_v3_endpoint` ladder against the local
`old_french` validator packet and determine whether the extra `v4` movement is
historically legible.

This is a follow-up to the earlier `v2_convergence` Old French pilot, not a
replacement for a larger scholarly validator corpus.

## Validator corpus

The validator packet is the same local `old_french` set used in the earlier
pilot:

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

Full `v4_from_v3_endpoint` ladder:

1. `FR_v4_000`
2. `FR_v4_001`
3. `FR_v4_002`
4. `FR_v4_003`
5. `FR_v4_004`
6. `FR_v4_005`
7. `FR_v4_006`

Comparison output:

- `data/validation/french_v4_from_v3_endpoint_vs_old_french.json`

## Results

### Structural best match

Best by structural distance and structural cosine:

- stage: `FR_v4_006`
- structural distance: `1.995647`
- structural cosine: `0.999985`

Structural distance improved across the `v4` run:

- `FR_v4_000`: `2.016365`
- `FR_v4_001`: `2.016365`
- `FR_v4_002`: `2.012179`
- `FR_v4_003`: `2.012179`
- `FR_v4_004`: `2.012179`
- `FR_v4_005`: `2.002236`
- `FR_v4_006`: `1.995647`

### Form best match

Best by orthographic / form similarity:

- stage: `FR_v4_000`
- validator form score: `0.611085`

Form similarity did not track the structural gain:

- `FR_v4_000`: `0.611085`
- `FR_v4_001`: `0.597488`
- `FR_v4_002`: `0.595263`
- `FR_v4_003`: `0.598759`
- `FR_v4_004`: `0.604054`
- `FR_v4_005`: `0.601706`
- `FR_v4_006`: `0.590575`

## Comparison to the earlier v2 pilot

Relative to the earlier `v2_convergence` Old French comparison:

- best structural distance improved from `2.026658` to `1.995647`
- the structural best stage moved from `FR_v2_061` to `FR_v4_006`
- the best form stage is still the seed stage rather than the late endpoint

So the extra `v4` movement looks historically legible in structural space, even
though the surface-form split remains.

## Interpretation

The main pattern holds and strengthens:

1. Later Latin-conditioned checkpoints look more Old-French-like structurally than
   the earlier stages do.
2. Surface-form similarity still prefers the earlier stage over the late
   Latin-conditioned endpoint.
3. `v4` improves the structural side of that result rather than erasing it.

The cleanest read is that the current controller is still pulling surface form
toward Latin harder than toward the attested intermediate packet, while the
structural bridge continues to move in a historically legible direction.

## Limitations

This remains a provisional validator result:

1. The `old_french` packet is still small.
2. It is hand-assembled rather than a full scholarly corpus release.
3. Structural cosine remains numerically compressed and should not be over-read.

## Next move

The sharp next moves are:

1. rerun the same comparison against a richer Old French corpus such as OTA `0176`
2. add a hold-out Latin split so endpoint scoring is not only in-loop
3. decide whether the next controller revision should reward Old-French-legible
   structure directly or keep the validator fully post hoc
