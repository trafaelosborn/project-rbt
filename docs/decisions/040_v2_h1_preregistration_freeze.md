# Decision 040: v2 H1 Preregistration Freeze

Date: 2026-06-12

Status: frozen for H1.

## Author Approval

Taylor Osborn approved the v2 H1 preregistration freeze on 2026-06-12.

Approval statement:

```text
My name is Taylor Osborn and I approve this (wild and likely ill-advised) experiment
```

## Frozen Scope

This freeze covers:

```text
docs/features_v2.md
docs/decisions/037_v2_corpus_bank.md
docs/decisions/038_v2_historical_validator_depth.md
docs/decisions/039_v2_static_feature_calibration.md
data/v2/corpora_manifest.json
data/v2/static_feature_matrix.csv
data/v2/static_feature_spearman.csv
data/v2/static_feature_tiers.json
```

## Preregistration Hash

Frozen feature preregistration document:

```text
path: docs/features_v2.md
sha256: 551a94d99b5426bf15968b9edf1769b76e609e6fe2927a1c8e2fdf30b9990a87
```

The freeze commit is identified by the git tag:

```text
v2-prereg-freeze
```

Registered H1 runs must write both the freeze commit and the feature document
SHA-256 into their run config / run manifest:

```text
features_freeze_commit
features_freeze_sha256
corpus_budget
seed
```

## Frozen H1 Configuration Summary

```text
corpus manifest status: h1_ready
registered feature count: 21
auxiliary diagnostics: AUX01 sentence-length excess kurtosis
static corpus budget: 20000
static sample seed: 1729
static matrix corpus count: 22
retained in-loop feature count: 6
```

Retained in-loop features:

```text
F02 top-100 bigram coverage
F09 mean word length
F15 suffix entropy
F17 conditional character entropy
F18 dis-legomena to hapax ratio
F19 word-edge character asymmetry
```

Deferred or auxiliary rows:

```text
Japanese: post-H1, needs tokenization policy
Mandarin: post-H1 alternate, needs tokenization policy
Modern Occitan: auxiliary positive diagnostic, not required for H1
Klingon: optional constructed-language stress control
```

## Consequence

Any change to the corpus set, feature formulas, retained in-loop tier,
registered analysis statistics, null/control policy, or thresholds after this
freeze must be treated as a new registration or a disclosed post-freeze
amendment.
