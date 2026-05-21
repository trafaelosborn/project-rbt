# Decision 011: Structural Similarity Metric — top-k Coverage vs Shannon Entropy

**Status:** Accepted  
**Date:** 2026-04-07  
**Context:** Phase 3 retrodiction scoring

---

## Problem

The retrodiction engine needs a vocabulary-independent metric that measures how
similar a generated intermediate corpus is to Latin (the target) vs Markov noise
(the null). The metric must discriminate meaningfully across all bridge stages.

---

## Rejected: Shannon entropy in cosine feature space

The initial implementation used a 4-dimensional structural feature vector:

```
[TTR, bigram_entropy, trigram_entropy, log_mean_seq_len]
```

**Why it fails:** Shannon entropy saturates for large vocabularies.

For a vocabulary of V=5000 tokens, maximum entropy is log(5000) ≈ 8.517 nats.
All profiles — French (structured), Markov noise (random), and generated
intermediates — concentrate in the range [7.8, 8.5]. Cosine similarities in
this space were consistently ≥ 0.999. The metric could not discriminate between
a structured Romance corpus and pure noise.

This is not a bug in the implementation; it is a fundamental property of
Shannon entropy for high-cardinality distributions. Entropy resolves well
near 0 (highly deterministic) but saturates in the high-entropy regime where
all natural language profiles live.

---

## Accepted: top-k coverage

Replace entropy with **top-k coverage**: the fraction of total n-gram mass
accounted for by the top-k most frequent n-grams.

```python
def top_k_coverage(profile: dict[str, float], k: int = 100) -> float:
    top_k_vals = sorted(profile.values(), reverse=True)[:k]
    return float(sum(top_k_vals))
```

**Why it works:** Coverage measures *concentration*, not dispersion.

For a uniform profile of N entries: coverage = k / N.  
For a fully concentrated profile: coverage → 1.0.

Analytic grammar (French, Spanish) has rigid collocations: article+noun,
preposition+article dominate the bigram distribution. A handful of bigrams
account for a large fraction of all bigram mass. Coverage is HIGH.

Synthetic grammar (Classical Latin) has free word order: any content word
can follow any other with roughly equal probability. Coverage is LOW.

Markov noise (pure uniform sampling) has the lowest possible coverage for
its vocabulary size.

| Corpus | bg_cov (top-100) | tg_cov (top-100) |
|--------|-----------------|-----------------|
| French | ~0.25           | ~0.18           |
| Romanian | ~0.22         | ~0.16           |
| Latin (reference) | ~0.16 | ~0.12         |
| Markov noise | ~0.02    | ~0.005          |

The signal is clear across the full range. Cosine similarities in
`[TTR, top100_bigram_cov, top100_trigram_cov, log_mean_seq_len]` space
discriminate well between all corpus types.

---

## Feature vector (final)

```
[0] type_token_ratio          — morphological complexity proxy
[1] top100_bigram_coverage    — fraction of bigram mass in top-100 bigrams
[2] top100_trigram_coverage   — fraction of trigram mass in top-100 trigrams
[3] log(1 + mean_seq_length)  — sequence length (structural, not content)
```

The retrodiction direction is: as the bigram model mixes toward Latin,
coverage falls from Romance levels (~0.22-0.25) toward Latin levels (~0.16).
This is the primary discriminating signal between analytic and synthetic grammar.

A valid intermediate stage should show monotonically decreasing coverage as
retrodiction progresses — coverage rising would indicate the algorithm is moving
away from Latin, toward greater rigidity (wrong direction).

---

## Alternative considered: top-k with k > 100

Top-k coverage becomes less sensitive as k grows (all profiles converge toward
1.0 for large k). k=100 was chosen as a balance: small enough to concentrate on
the most discriminating bigrams, large enough not to be dominated by a single
high-frequency collocation. The value is a hyperparameter; k=50 and k=200 were
also tested and produced qualitatively similar orderings.
