# Decision: Retrodiction algorithm — bigram mixing with generative validation

Date: 2026-04-07
Phase: P3

## What was decided

The retrodiction algorithm operates as follows at each iteration:

1. **Generate**: Sample a synthetic corpus from the current bigram language model.
2. **Fingerprint**: Compute the full statistical fingerprint of the generated corpus
   (co-occurrence matrix, positional distribution, n-gram profiles, TTR).
3. **Score**: Compare the fingerprint against Markov noise and Sumerian references
   using cosine similarity in structural feature space.
4. **Record**: Store the complete bridge stage record (fingerprint paths + scores).
5. **Transform**: Mix the bigram transition matrix toward uniform by factor alpha.
6. **Halt**: If the structural vector has not changed meaningfully (L2 delta below
   STABILITY_THRESHOLD), stop. Otherwise repeat from 1.

Default parameters: alpha=0.05, num_sequences=2000, max_iterations=200,
stability_threshold=0.005, seed=42.

## Why generative

Pure tensor interpolation between two fingerprints is mathematically trivial and
scientifically empty. Given fingerprints F_A and F_B, a weighted average
(1-t)*F_A + t*F_B produces a "valid" intermediate for any two corpora — French and
Zulu, Esperanto and Quechua, any pair. This says nothing about linguistic reality.

The generative step is what makes the intermediate stages scientifically meaningful.
At each iteration, the generated corpus is a REAL synthetic text — token sequences
produced by sampling from the evolved distributions. This corpus:

- Can be fingerprinted independently and compared to real attested historical stages
  (Vulgar Latin texts, Old French, Carolingian Latin) using the same metrics
- Demonstrates internal coherence: if the distributions are incoherent, the generated
  text will look like noise, and that failure is itself a finding
- Produces a path that is EMERGENT from the statistics rather than predetermined by
  the endpoints

Three possible outcomes:

**Case 1 — Noise**: The generated intermediate is statistically incoherent. No
language could have these distributions simultaneously. This means either the
algorithm is broken (engineering problem) or the ontological assumption is wrong
(the statistical fingerprint does not preserve genealogical information). Either
outcome is meaningful.

**Case 2 — Match**: The generated intermediate fingerprints to something statistically
indistinguishable from an attested historical stage. The gradient found the real path.

**Case 3 — Ghost language**: The generated intermediate is fully coherent as a language
but does not match any attested stage. A statistically valid path through linguistic
space that history did not take. Evidence that language evolution has more degrees of
freedom than the historical record shows.

Cases 2 and 3 are both genuine results. The degree to which independent
single-language reconstructions agree with each other before the ground truth is
opened quantifies the convergence geometry of the Romance family.

## Why bigram mixing toward uniform

The backward transformation mixes the bigram transition matrix toward uniform:
    T_new[i,j] = (1 - alpha) * T[i,j] + alpha * (1/V)

Rationale: Modern analytic Romance languages have relatively rigid word order, which
appears in the bigram distribution as high concentration (certain bigrams dominate).
Classical Latin has freer word order (case-based agreement rather than position-based),
which corresponds to a more uniform bigram distribution (any content word can follow
any other with more nearly equal probability).

Mixing toward uniform therefore moves the statistical structure in the direction of
greater syntactic freedom — from analytic toward synthetic grammar. This is the
statistically motivated backward direction.

## Why vocabulary is held fixed

The generated intermediate corpora use the same vocabulary as the source corpus
(e.g., French tokens). The vocabulary does not evolve. This is deliberate:

- The retrodiction is about STRUCTURE, not lexicon. The claim is that the
  distributional statistics of a language encode its genealogy, independent of
  what the tokens mean.
- Vocabulary evolution (introduction of morphological variants, archaic forms)
  would require linguistic knowledge we explicitly do not have.
- For the Minos application, the tokens are unknown. The methodology must work
  with fixed abstract vocabulary.

The generated intermediate will therefore be "French tokens in increasingly Latin-like
patterns" — not reconstructed Latin. Comparison to attested historical stages is
done at the fingerprint (structural) level, not by word comparison.

## Similarity metric

The similarity between a generated stage and a reference corpus (Markov, Sumerian,
Portuguese, Latin) is cosine similarity in a 4-dimensional structural feature space:

    [type_token_ratio, bigram_entropy, trigram_entropy, log(1 + mean_seq_len)]

All four features are vocabulary-independent. This allows direct comparison between
corpora with different token sets (French intermediate vs. Sumerian, vs. Markov noise
with abstract "0"..."499" tokens).

## Halting criterion

The algorithm halts when the L2 distance between consecutive structural vectors falls
below STABILITY_THRESHOLD (default 0.005). This is the empirical stable point — the
fingerprint is no longer changing meaningfully. The stable point is NOT tuned toward
Latin; the algorithm does not know where Latin is.

A secondary halt is MAX_ITERATIONS (default 200) to prevent runaway runs.

## Scoring convention

All four scores (vs_markov, vs_sumerian, vs_portuguese, vs_latin) are cosine
similarities in structural feature space, range [0, 1]. Higher = more similar to
that reference.

During retrodiction (Phase 3): vs_portuguese_control and vs_latin_ground_truth are
null — filled in by separate post-processing passes that explicitly unlock
sequestration.

## Phase 3 scope

Phase 3 runs four independent single-language reconstructions:
French (FR), Italian (IT), Spanish (ES), Romanian (RO).

Occitan and Genoese are excluded from Phase 3 due to thin corpora (potential
instability in bigram model).

The combined multi-language reconstruction is Phase 4.

## Revision history

- 2026-04-07: Initial decision.
