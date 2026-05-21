# Decision 013: Reinforced-Stage Coherence Diagnostic

Date: 2026-04-08

## Context

Reward-fixed reinforced runs can improve toward Latin in the trainable reward
subspace without yet telling us whether the bridge remains language-like or is merely
optimizer junk.

Attested intermediate corpora are not yet ingested, so the project needs an internal
diagnostic that can separate "coherent alternate bridge" from "noise-like collapse"
before historical validation is available.

## Decision

Add a post-generation coherence diagnostic for reinforced stages based on scaled
structural-vector distance.

For each generated stage:

1. Compute the existing structural vector:
   - `type_token_ratio`
   - `bigram_coverage`
   - `trigram_coverage`
   - `log_mean_seq_len`
2. Build a real-language reference manifold from in-repo attested non-noise corpora:
   - French
   - Italian
   - Spanish
   - Romanian
   - Occitan
   - Genoese
   - Sumerian
3. Compute the centroid and per-feature standard deviation of that set.
4. Measure scaled Euclidean distance from the stage to:
   - the real-language centroid
   - the Markov noise reference
5. Define:
   - `language_likeness_margin = distance_to_markov_noise - distance_to_real_language_centroid`

Interpretation:

- `margin >= 1.0` -> `coherent`
- `0.0 <= margin < 1.0` -> `borderline`
- `margin < 0.0` -> `noise_like`

## Rationale

This gives the project a first-pass answer to the question "are we just setting tokens
on fire?" without using Latin as the sole judge of bridge quality.

The diagnostic is intentionally conservative:

- it does not claim historical correctness
- it does not replace attested-stage comparison
- it only asks whether the bridge still occupies language-like structural space more
  than it resembles the Markov floor

Including Sumerian in the real-language manifold helps keep the diagnostic from being
merely "stay close to Romance."

## Consequences

Positive:

- Reinforced summaries now report whether a bridge remains coherent under null-model
  comparison.
- The method can distinguish Latin improvement from outright structural collapse.
- The same diagnostic can be reused for future languages and alternate targets.

Limitations:

- The diagnostic still operates on a low-dimensional structural representation.
- High coherence does not imply historical plausibility.
- A future attested-stage layer may supersede or refine the current thresholding.
