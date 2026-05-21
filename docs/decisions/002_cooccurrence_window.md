# Decision: Co-occurrence window size = 2

Date: 2026-04-07
Phase: P2

## What was decided

The co-occurrence matrix is built with a symmetric sliding window of size 2 (two tokens
to the left, two to the right of each center token).

## Why

Window=1 captures only immediately adjacent pairs. For natural language prose, this
misses common short-range dependencies: determiners are often separated from their
nouns by one token (adjective), prepositions from their objects, etc. Window=1
produces a sparser matrix that undersells actual lexical association patterns.

Window=3 or larger dilutes the signal. In sentences of 8–15 tokens (typical Wikipedia
prose), a window of 3 includes a large fraction of the sentence as "context" for each
position. The co-occurrence matrix becomes a near-uniform fingerprint of overall
vocabulary frequency rather than a signal of local dependency structure.

Window=2 is the most common choice in distributional semantics work (Levy and Goldberg
2014, GloVe, etc.) for exactly this reason. It captures the strongest local
dependencies while remaining selective.

For the retrodiction purpose specifically: the shift from analytic (word-order-based)
to synthetic (case-based) grammar should show up as changes in the co-occurrence
structure of function words relative to content words. Window=2 is appropriate for
detecting these grammatical relationship signals.

## Impact

All co-occurrence matrices in the pipeline use window=2. The `--window` flag allows
override for sensitivity analysis.

## Revision history

- 2026-04-07: Initial decision.
