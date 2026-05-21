# Decision 018: Add Hungarian Family Alignment as a Diagnostic Layer

Date: 2026-04-08

## Context

The proposed `v4` control loop calls for a more global notion of mismatch than the
current scalar score alone provides. The project needed a first implementation step
that would:

1. remain diagnostic-only
2. not change mutation behavior
3. reveal whether a family-level global alignment signal tracks the existing French
   bridge trajectory in a useful way

## Decision

Add `src/validation/hungarian_alignment.py` as a validation module.

This module:

1. extracts mutable family inventories from a corpus
2. builds a family-to-family cost matrix against a Latin reference inventory
3. runs Hungarian assignment over those families
4. reports normalized family alignment score and residual structure

The first implementation uses three family types:

1. suffix families
2. prefix families
3. short high-frequency token families

## Rationale

This is the safest Phase 1 implementation of the `v4` idea.

It adds a global, interpretable alignment diagnostic without changing the bridge
generation engines. That lets the project ask:

- does the current bridge path look more Latin-aligned at the family level?
- does `v3` improve on `v2` under a new global criterion?

before using that signal to steer mutation behavior.

## Consequences

Positive:

- The project now has an explicit family-level Latin alignment score.
- The signal is computed out-of-loop in validation, not inside mutation.
- The diagnostic can be compared directly against raw Latin structural / form scores.

Observed immediately:

- On the French `v2_convergence` ladder, family alignment rises from
  `0.397699` at `FR_v2_000` to `0.518986` at `FR_v2_061`, with the best late
  checkpoint at `FR_v2_058 = 0.519016`.
- On the French `v3_from_v2_endpoint` continuation ladder, the shared starting point
  `FR_v3_000` begins at `0.518986` and the endpoint `FR_v3_008` reaches
  `0.539860`.

Interpretation:

- The new diagnostic agrees directionally with the project's current reading of the
  bridge path.
- `v3` did not merely inflate an internal reward. It also improved family-level
  Latin alignment under an independent diagnostic layer.

Limitations:

- The current matching units are heuristic families, not linguistic gold labels.
- Hungarian still imposes one-to-one matching, which may be too rigid for some
  historical relationships.
- This is a validation signal, not yet a mutation controller.
