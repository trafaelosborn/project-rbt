# Decision: Add meso-scale contiguous span mutation to reinforced v2

Date: 2026-04-08

## Context

The original reinforced `v2` search operated at local scales:

1. token character edit
2. suffix-family rewrite
3. local bigram swap
4. token split
5. bigram merge

That made the engine good at local form drift, but likely weak at coordinated
multi-sentence movement. The plateau at `FR_v2_061` raised the possibility that the
search was stuck because its perturbation window was too small.

## Decision

Add a new operator:

- `sequence_span_rewrite`

This operator chooses a contiguous span of `2-5` adjacent sentence sequences and
applies a small bundle of sub-edits inside that span, such as:

- token character edits
- suffix-family rewrites
- local bigram swaps
- token splits
- bigram merges
- local sequence-order swaps / rotations

The intent is to let the search make meso-scale coordinated moves without jumping
straight to full-document or paragraph-wide replacement.

## Consequences

Positive:

- The search space now includes bundled multi-sentence mutations.
- Plateau testing is no longer limited to purely local rewrites.
- The operator is still interpretable and traceable in stage records.

Observed immediately:

- A short continuation probe from `FR_v2_061` with span-heavy weighting did not
  accept any moves under the current score.
- Diagnostic sampling showed that the best unpenalized span candidate only barely
  improved the raw score, and the mutation-cost penalty pushed it negative.

Interpretation:

- The current French endpoint plateau is not *only* a sentence-window problem.
- Under the present scoring and cost regime, meso-scale moves do not yet open a new
  improvement basin in a meaningful way.
