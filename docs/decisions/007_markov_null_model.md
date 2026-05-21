# Decision: Markov noise generator — n=2, uniform transitions, vocab=500, seed=42

Date: 2026-04-07
Phase: P2

## What was decided

The Markov noise floor null model uses:
- n=2 (bigram order; context window = 1 preceding token)
- Uniform transitions (each next token sampled uniformly from vocabulary)
- Vocabulary size = 500 abstract tokens ("0" through "499")
- Sequence length = 10 tokens per sequence
- 10,000 sequences total
- Random seed = 42 for reproducibility

## Why each parameter

**n=2 (bigram), uniform:**
The null model represents "no linguistic structure." True zero structure would be
n=1 (independent token sampling). However, n=1 uniform is so far from any real
language that it may be trivially distinguishable regardless of reconstruction quality.
n=2 uniform adds the minimum amount of sequential dependency (each token is conditioned
on one preceding token) while still producing sequences with zero genuine grammatical
or semantic structure.

In uniform mode, n has no effect on output statistics — transitions are uniform regardless
of context, so n=1 and n=2 uniform are identical in output. The n=2 parameter is
retained for documentation purposes and for when trained mode is used.

**Vocab size = 500:**
Romance language vocabularies at 500 articles are ~20,000–50,000 types. However, most
fingerprint statistics (positional distribution, co-occurrence) are dominated by the
most common ~500–1,000 tokens (function words, common nouns). A Markov vocabulary of
500 abstract types represents the "dense core" of a natural language vocabulary without
implicitly encoding any real vocabulary structure.

**Sequence length = 10, num_sequences = 10,000:**
10 tokens approximates average clause length in Wikipedia prose (shorter than a full
sentence). 10,000 sequences gives 100,000 total tokens — comparable to a thin-corpus
Romance language. This ensures the Markov fingerprint is built from a corpus of similar
size to the comparison targets.

**Seed = 42:**
Fixed seed ensures the Markov null fingerprint is fully reproducible. If the seed
changes between runs, null model scores will drift. The seed is recorded in the
corpus metadata.

## Impact

The Markov null fingerprint is built by the same pipeline as all other fingerprints.
Its `vs_markov_noise` scores in bridge stage records represent the distance from the
reconstruction at each stage to this baseline.

A reconstruction that scores below or near the Markov floor would indicate the
retrodiction algorithm has introduced noise rather than structure — a critical
failure mode that this null model is designed to detect.

## Revision history

- 2026-04-07: Initial decision.
