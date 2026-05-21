# French Meso-Scale Probe

Date: 2026-04-08

## Purpose

Test whether a larger perturbation window can push the French `v2` endpoint beyond
its apparent plateau.

This probe starts from the current convergence endpoint `FR_v2_061` and heavily
weights the new contiguous span operator `sequence_span_rewrite`.

## Implementation

The `v2` engine now includes a meso-scale operator that rewrites contiguous spans of
`2-5` adjacent sentence sequences using bundled sub-edits and optional local sequence
reordering.

Code location:

- `src/retrodiction/engine_reinforced_v2.py`

## Probe setup

Starting state:

- `data/retrodiction/french/v2_convergence/corpora/FR_v2_061_tokens.json`

Probe output:

- `data/retrodiction/french/v2_meso_probe_short/`

Key probe settings:

- `max_proposals = 24`
- `max_accepted_stages = 8`
- `n_candidates = 4`
- operator weights heavily biased toward `sequence_span_rewrite`

## Result

The short continuation probe accepted no moves.

- total stages: `1`
- final stage: `FR_v2_000`
- final total score: `-0.579618`
- final Latin structural score: `-1.306957`
- final Latin form score: `0.762769`

This means the endpoint state remained unchanged under the short span-heavy search.

## Diagnostic follow-up

A direct diagnostic sample of span rewrites from the endpoint showed:

- current total score: `-0.5796183853`
- best sampled total score after cost: `-0.5850141009`
- best sampled raw score before cost: `-0.5796141009`

So the best sampled span mutation only improved the raw objective by about
`0.000004`, and the mutation-cost penalty more than erased that gain.

## Interpretation

This is a useful negative result.

It suggests that:

1. the `FR_v2_061` plateau is real under the current objective
2. larger contiguous-window mutations alone do not obviously unlock a new path
3. the next bottleneck is likely the score geometry or mutation family, not just the
   perturbation window size

## Practical takeaway

The project can now say:

- "We tested a larger perturbation window."
- "It did not materially break the French endpoint plateau under the current score."

That is better than guessing.
