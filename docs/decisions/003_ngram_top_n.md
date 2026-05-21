# Decision: N-gram profile retains top 5,000 entries

Date: 2026-04-07
Phase: P2

## What was decided

The bigram and trigram frequency profiles retain the top 5,000 most frequent n-grams
by count, normalized to relative frequencies summing to 1.0.

## Why

The total unique bigram count for a 500-article Wikipedia corpus is approximately
200,000–400,000 types (depending on language morphological richness). Keeping all of
them would produce very sparse, high-dimensional profiles that are numerically noisy
for comparison.

Top-5,000 bigrams typically capture approximately 60–75% of all bigram occurrences in
the corpus (Zipf's law — a small number of bigrams account for most of the mass). This
is a stable, information-dense representation: the most frequent bigrams are grammatical
constructions (articles + nouns, prepositions + articles, auxiliary + verb) and these
are exactly the patterns that shift during the analytic→synthetic grammar transition.

For thin corpora (Occitan, Genoese), the unique bigram count may be well below 5,000.
The profile is capped at the actual count; no padding. The actual count is recorded in
the metadata.

The `--top-n` flag allows override for sensitivity analysis.

## Impact

Profile comparisons in the retrodiction scoring should use intersection-based similarity
(Jaccard or cosine on the shared key set) rather than assuming identical key sets.

## Revision history

- 2026-04-07: Initial decision.
