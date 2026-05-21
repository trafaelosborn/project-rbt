# Hungarian Family Alignment Diagnostic

Date: 2026-04-08

## Purpose

Implement Phase 1 of the proposed `v4` direction:

1. add Hungarian family alignment as a diagnostic only
2. do not change mutation behavior
3. test whether the new global signal agrees with the current French bridge story

## Implementation

Code:

- `src/validation/hungarian_alignment.py`

Tests:

- `tests/test_validation_hungarian_alignment.py`

The diagnostic extracts three family types from each corpus:

1. suffix families
2. prefix families
3. short high-frequency token families

It then builds a cost matrix against a Latin family inventory and runs Hungarian
assignment to produce:

1. normalized family alignment score
2. normalized family alignment cost
3. matched-pair residuals

## Outputs

French `v2` ladder:

- `data/validation/french_v2_convergence_vs_latin_family_alignment.json`

French `v3` continuation ladder:

- `data/validation/french_v3_from_v2_endpoint_vs_latin_family_alignment.json`

## Result: `v2_convergence`

Selected ladder:

1. `FR_v2_000`
2. `FR_v2_015`
3. `FR_v2_030`
4. `FR_v2_045`
5. `FR_v2_058`
6. `FR_v2_061`

Family alignment scores:

- `FR_v2_000 = 0.397699`
- `FR_v2_015 = 0.458687`
- `FR_v2_030 = 0.480213`
- `FR_v2_045 = 0.499806`
- `FR_v2_058 = 0.519016`
- `FR_v2_061 = 0.518986`

Best `v2` checkpoint under this diagnostic:

- `FR_v2_058`

Interpretation:

The family-alignment signal improves steadily across the `v2` path and then flattens
at the end, which matches the earlier picture of a late stable basin near
`FR_v2_061`.

## Result: `v3_from_v2_endpoint`

Selected ladder:

1. `FR_v3_000`
2. `FR_v3_002`
3. `FR_v3_004`
4. `FR_v3_005`
5. `FR_v3_006`
6. `FR_v3_008`

Family alignment scores:

- `FR_v3_000 = 0.518986`
- `FR_v3_002 = 0.517469`
- `FR_v3_004 = 0.518238`
- `FR_v3_005 = 0.521783`
- `FR_v3_006 = 0.521782`
- `FR_v3_008 = 0.539860`

Best `v3` checkpoint under this diagnostic:

- `FR_v3_008`

Interpretation:

The `v3` continuation endpoint improves over the shared `v2` seed by about
`0.020874` family-alignment points:

- `0.518986 -> 0.539860`

So `v3` did not only improve under its own amplified reward. It also improved under
this new independent family-level alignment diagnostic.

## Read across existing metrics

The important thing is directional agreement:

- raw Latin structural score improved
- raw Latin form score improved
- Hungarian family alignment score improved

That makes the `v3` continuation result stronger than "the new scorer liked its own
output." Multiple diagnostics are now pointing the same way.

## What this does and does not mean

What it means:

- the bridge is becoming more Latin-aligned under a new global family signal
- the `v2` plateau was not just an artifact of the old scalar total score
- the new diagnostic is informative enough to be worth keeping

What it does not mean:

- that the bridge has reached Latin
- that Hungarian one-to-one family matching is the final correct ontology
- that the alignment signal is ready to control mutation behavior without further
  testing

## Practical takeaway

Phase 1 worked.

The project now has a real family-level global alignment diagnostic, and it supports
the same broad conclusion as the existing French runs:

- late `v2` is more Latin-aligned than early `v2`
- `v3` pushes the bridge beyond the old `v2` basin

That makes it reasonable to consider Phase 2 next:

- use the alignment signal to schedule operator weights
- still without fully handing control to it
