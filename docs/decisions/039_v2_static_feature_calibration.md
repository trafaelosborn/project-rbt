# Decision 039: v2 Static Feature Calibration

Date: 2026-06-12

Status: frozen for H1 as part of the v2 preregistration freeze.

## Context

After the v2 corpus bank reached `h1_ready`, the remaining preregistration
blockers were feature-count ambiguity and in-loop feature collinearity. The H1
draft registered 21 candidate features, but the proposed sentence-length tail
shape feature contained two scalar moments: skewness and excess kurtosis.

The H1 feature tier must be calibrated using only fixed corpus fingerprints,
not retrodiction trajectories or endpoints.

## Decision

Resolve the F21 ambiguity as:

```text
F21: sentence-length skewness
AUX01: sentence-length excess kurtosis
```

`AUX01` is emitted for reporting and diagnostics but is not counted as a
registered H1 feature. The registered candidate pool therefore remains 21
features.

## Static Calibration Artifacts

Generated artifacts:

```text
data/v2/static_feature_matrix.csv
data/v2/static_feature_spearman.csv
data/v2/static_feature_tiers.json
```

Generation command:

```text
python -m src.validation.static_feature_matrix --budget 20000
```

Calibration settings:

```text
corpus_budget: 20000
sample_seed: 1729
corpus_count: 22
collinearity_threshold_abs_spearman: 0.85
```

The static matrix includes H1-required ready corpora plus the ready optional
Klingon control. Japanese and Mandarin remain visible in the corpus manifest as
post-H1 tokenization-policy rows, but are not included in the H1 static matrix.
Modern Occitan is now visible as a non-required auxiliary positive diagnostic,
but it is excluded from in-loop tier selection for the same reason: it is a
reporting sanity check, not a calibration knob.

## Retained In-Loop Tier

The retained in-loop tier is:

```text
F02 top-100 bigram coverage
F09 mean word length
F15 suffix entropy
F17 conditional character entropy
F18 dis-legomena to hapax ratio
F19 word-edge character asymmetry
```

This gives six in-loop features, within the preregistered target range of five
to seven.

## Demotion

`F16 Heaps' law exponent` was demoted to diagnostic by the static collinearity
rule.

Strongest conflict:

```text
F16 Heaps' law exponent vs F09 mean word length
Spearman rho = 0.8690005646527386
```

Rationale:

Both axes are meaningful, but the in-loop tier keeps the more direct and
trainable word-form scale axis (`F09`) when the indirect vocabulary-growth axis
(`F16`) exceeds the collinearity threshold. `F16` remains available as a
measured diagnostic.

## Feature Formula Notes

The calibration module is:

```text
src/validation/static_feature_matrix.py
```

Important registered formula choices:

- `F19` is occurrence-weighted `H(final_char_distribution) -
  H(initial_char_distribution)`.
- `F20` uses the broad Unicode vowel set registered in the calibration module
  and remains diagnostic.
- `F21` is standardized third central moment of sequence lengths.
- `AUX01` is sequence-length excess kurtosis.

## Consequences

- The corpus and feature-count blockers are resolved at the draft-artifact
  level.
- H1 received author sign-off on 2026-06-12 before freeze, commit, and tag.
- Any change to the retained tier after first v2 run launch must be reported as
  a new registration rather than a quiet edit.
