# Project RBT v2 Feature and Analysis Pre-Registration

Status: FROZEN FOR H1. USE THIS VERSION FOR REGISTERED V2 H1 RUNS.

Prepared: 2026-06-11
Approved: 2026-06-12 by Taylor Osborn
Project: v2 Romance calibration build
Primary question: Do family-anchored statistical retrodictions converge toward
their shared ancestor, and does the resulting typological map reproduce the
attested structure of the Romance family?

This document is the frozen H1 pre-registration. It enumerates the v2 candidate
feature pool, retained feature tiers, required static collinearity calibration,
analysis defaults, and freeze rules. It is immutable after author approval,
commit, and tag `v2-prereg-freeze`.

No registered v2 H1 run may deviate from this document without being reported
as a new registration or a post-freeze amendment.

## Freeze Resolution

1. The v2 corpus bank manifest exists at `data/v2/corpora_manifest.json` and
   currently reports `h1_ready`. Japanese is retained as a visible post-H1
   optional row with `needs_tokenization_policy`, but it is not required for H1.
   Modern Occitan is retained as a non-required auxiliary positive-control
   diagnostic and does not alter the required corpus set.
2. F21 is resolved as sentence-length skewness. Sentence-length excess kurtosis
   is emitted as auxiliary diagnostic `AUX01`, not as a registered H1 feature.
3. Static calibration artifacts now exist at `data/v2/static_feature_matrix.csv`,
   `data/v2/static_feature_spearman.csv`, and
   `data/v2/static_feature_tiers.json`. Taylor Osborn approved these artifacts
   and the retained in-loop tier on 2026-06-12.

## Feature Tiering Rule

Tiering is instrument calibration, not outcome analysis. It uses only static
fingerprints of fixed corpora and never uses run trajectories or endpoints.

Default rule:

1. Subsample every corpus to the common token budget before feature computation.
2. Compute the full candidate feature matrix for all v2 corpora.
3. A feature qualifies for the in-loop reward tier only if it maps to a named
   typological axis and its pairwise Spearman `abs(rho)` is below 0.85 with every
   other retained in-loop feature on the static matrix.
4. The in-loop tier may contain at most 7 retained features or feature families.
5. Where two features are collinear, retain the more size-robust feature in-loop
   and demote the other to measured diagnostic.

The working expectation is 5 to 7 in-loop features, consistent with the v1
intrinsic dimensionality estimate of about 5 principal components.

## Candidate Feature Pool

The first 15 features are the v1 post-hoc measured set. Features 16 to 21 are
v2 additions from the project brief.

| ID | Feature | Formula / computation | Typological rationale | Expected behavior | Proposed tier |
|---|---|---|---|---|---|
| F01 | Type-token ratio | `unique_token_types / total_tokens` | Vocabulary breadth and inflectional/productive variation. | Higher in morphologically rich or lexically broad corpora, but strongly size-sensitive. | Excluded if F16 is stable; otherwise diagnostic |
| F02 | Top-100 bigram coverage | Sum of the 100 largest relative word-bigram probabilities. | Local syntactic concentration vs dispersion. | Higher when a few word transitions dominate; lower for broader combinatorics. | In-loop candidate |
| F03 | Top-100 trigram coverage | Sum of the 100 largest relative word-trigram probabilities. | Higher-order phrase-pattern concentration. | Higher for formulaic or repetitive syntax; lower for diffuse phrase structure. | Measured diagnostic |
| F04 | Log mean sequence length | `log(1 + mean_tokens_per_sequence)` | Sentence or clause length scale. | Detects register/segmentation differences; weakly trainable under current operators. | Excluded from reward unless length-changing operators are registered |
| F05 | Top-500 bigram coverage | Sum of the 500 largest relative word-bigram probabilities. | Broader version of F02. | Tracks the same concentration axis at lower resolution. | Measured diagnostic |
| F06 | Top-500 trigram coverage | Sum of the 500 largest relative word-trigram probabilities. | Broader version of F03. | Tracks phrase-pattern concentration beyond the top 100. | Measured diagnostic |
| F07 | Top-1000 unigram coverage | Fraction of all token mass in the 1000 most frequent token types. | Frequency concentration and vocabulary breadth. | Higher for restricted or repetitive vocabulary; lower for broad lexical distribution. | Measured diagnostic |
| F08 | Sequence-length variance | Population variance of token counts per sequence. | Register and syntactic packaging variation. | Higher when sentence/clause lengths vary strongly. | Measured diagnostic |
| F09 | Mean word length | Mean character length over tokens. | Morphological and orthographic compactness proxy. | Higher in languages/corpora with longer average word forms. | In-loop candidate |
| F10 | Word-length variance | Population variance of token character lengths. | Diversity of morphological/orthographic word sizes. | Higher when short function forms and long content/inflected forms coexist. | Measured diagnostic |
| F11 | Hapax ratio | Count of types with frequency 1 divided by total type count. | Low-frequency tail shape and lexical productivity. | Higher for broad/productive or under-sampled corpora. | Measured diagnostic; likely demoted if F18 retained |
| F12 | Zipf slope, ranks 10-1000 | Least-squares slope of `log(freq)` on `log(rank)`. | Frequency-tail shape. | More negative means steeper concentration; flatter slopes indicate broader tails. | Measured diagnostic |
| F13 | Bigram entropy | Shannon entropy of normalized word-bigram profile. | Local combinatorial dispersion. | Higher for diffuse transition structure; likely anticorrelated with F02/F05. | Measured diagnostic or in-loop alternative to F02 |
| F14 | Trigram entropy | Shannon entropy of normalized word-trigram profile. | Higher-order combinatorial dispersion. | Higher for diffuse phrase structure; likely related to F03/F06. | Measured diagnostic |
| F15 | Suffix entropy | Shannon entropy of top suffix profile using the existing suffix extractor. | Suffixal morphological richness. | Higher for richer or more diverse suffix inventories. | In-loop candidate |
| F16 | Heaps' law exponent | Fit `V(n) = K n^beta` over prefix samples; use `beta`. | Size-robust vocabulary growth and morphological productivity. | Higher beta means vocabulary continues to grow rapidly with corpus size. | Demoted to diagnostic by static collinearity calibration |
| F17 | Conditional character entropy | `H(char_i | char_{i-1})` over token-internal character bigrams. | Phonotactic/orthographic predictability; agglutinative vs fusional concatenation proxy. | Higher for less predictable local character continuation. | In-loop candidate |
| F18 | Dis-legomena to hapax ratio | `count(freq == 2) / max(count(freq == 1), 1)`. | Frequency-tail shape beyond hapax count. | Distinguishes one-off noise from a sustained low-frequency tail. | In-loop candidate |
| F19 | Word-edge character asymmetry | `H(final_char_distribution) - H(initial_char_distribution)` over token occurrences. | Prefixing vs suffixing tendency through word-edge behavior. | Positive values indicate more diverse endings than beginnings. | In-loop candidate |
| F20 | Consonant-vowel ratio | Count consonant characters divided by vowel characters after Unicode letter filtering and the registered broad vowel set in `src.validation.static_feature_matrix`. | Broad phonotactic proxy. | Higher values indicate consonant-heavy orthographic/phonotactic profile. | Measured diagnostic |
| F21 | Sentence-length skewness | Standardized third central moment of token counts per sequence. | Distributional shape beyond mean and variance. | Positive skewness tracks right-tailed sentence/clause length distributions. | Measured diagnostic |

## Static-Calibrated In-Loop Reward Subset

The 20,000-token static calibration matrix retains six in-loop features. This
tier satisfies the registered Spearman `abs(rho) < 0.85` rule and remains a
draft only until author sign-off.

| Feature | Axis |
|---|---|
| F02 top-100 bigram coverage | word-transition concentration |
| F09 mean word length | word-form scale |
| F15 suffix entropy | suffixal morphological richness |
| F17 conditional character entropy | character-level predictability |
| F18 dis-legomena to hapax ratio | low-frequency tail shape |
| F19 word-edge character asymmetry | prefixing/suffixing edge tendency |

Demotion rule: if any pair in this subset has Spearman `abs(rho) >= 0.85`, keep
the more size-robust or more directly typological feature and demote the other
to measured diagnostic. Expected likely demotions are:

| Conflict | Default action |
|---|---|
| F02 vs F13 | Retain F02 unless entropy separates controls more stably across budgets. |
| F01 vs F16 | Retain F16; demote TTR. |
| F11 vs F18 | Retain F18 if stable; demote hapax ratio. |
| F02/F03/F05/F06/F13/F14 cluster | Retain at most one bigram axis and at most one trigram axis in-loop. |
| F08/F21 cluster | Keep as diagnostics unless length-changing operators are registered. |

Observed static-calibration demotion:

```text
F16 Heaps' law exponent -> diagnostic
strongest conflict: F09 mean word length, Spearman rho = 0.8690005646527386
```

## Static Collinearity Matrix

Generated static-calibration artifacts:

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
registered_feature_count: 21
auxiliary_diagnostics: AUX01 sentence-length excess kurtosis
```

Required corpus rows:

```text
latin
french
spanish
italian
portuguese
romanian
catalan_or_sardinian
medieval_romance_validators_available_for_v2
turkish
finnish
mandarin_or_vietnamese
sumerian
klingon_tatoeba_tlh_optional
modern_standard_arabic
bulgarian_or_serbian
german
markov_noise
```

Auxiliary non-required positive diagnostic:

```text
modern_occitan
```

Required matrix rule:

```text
For every proposed in-loop pair (Fi, Fj):
  retain both only if Spearman abs(rho(Fi, Fj)) < 0.85
  otherwise demote one before author sign-off.
```

Matrix status:

```text
COMPUTED. Awaiting author sign-off before freeze.
```

## Corpus-Size Normalization

Before any feature computation, each corpus is subsampled to a common token
budget. The token budget is a run/config parameter and must be recorded in every
manifest as `corpus_budget`.

Registered default budgets:

| Use | Token budgets |
|---|---|
| Main static matrix and full runs | 20000, 30000, 50000 where corpora permit |
| Small-corpus stress test | about 7000 tokens, single-register |
| Bootstrap contact test | same budget grid as the main static matrix |

If a corpus cannot satisfy a budget, that budget is excluded for all corpora in
that comparison rather than allowing unmatched budgets.

## In-Loop Structural Score

The v1/v5 production score used negative scaled Euclidean distance on only the
first three structural features:

```text
S_struct = -score_scale * ||candidate_reward_vec - latin_reward_vec||_2
```

v2 retains the same basic form but replaces the reward vector with the frozen
in-loop feature subset selected at H1:

```text
S_struct_v2 = -score_scale * ||z(candidate_inloop) - z(latin_inloop)||_2
```

where `z(.)` uses the frozen static-matrix mean and scale from the H1
calibration artifact. The same frozen feature basis and scaling must be used in
candidate acceptance, trajectory reporting, endpoint geometry, and validator
comparison.

If length-dependent features are retained in-loop, the mutation operator set
must include registered length-changing operators. Otherwise length-dependent
features remain measured diagnostics only.

## Registered Experiments

### E1. Convergence Gate

Question: do Romance-source trajectories funnel into a common region earlier
than unrelated controls when retrodicted toward Latin?

Default statistic:

1. For each trajectory, compute progress
   `p = (S - S_start) / (S_final - S_start)` using the frozen in-loop composite
   score.
2. Sample each trajectory at deciles `p = 0.0, 0.1, ..., 1.0`.
3. At each decile, compute pairwise distances in frozen in-loop feature space.
4. Normalize each pair's distance curve by its distance at `p = 0`.
5. Compute AUC over `p in [0, 1]` for each pair.
6. Compare Romance-Romance AUCs against control-pair AUCs.

Gate decision:

```text
YES if Romance-pair AUC is lower than control-pair AUC by one-sided
Mann-Whitney U at alpha = 0.05 and the direction replicates in every seed.
NO otherwise.
```

### E2. Backbone Reproduction

Question: does the static size-normalized map reproduce the consensus Romance
backbone?

Registered expected structure:

```text
Italian and Spanish should be relatively close.
Portuguese should lie near Spanish.
Romanian should be the clearest Romance outlier.
```

This experiment validates coarse consensus structure only. It must not chase
fine-grained or contested Romance subgrouping claims.

### E3. Contact-Detection Test

Question: does the map surface structural contact overlays, or only genealogical
backbone structure?

Required donor comparators:

```text
Modern Standard Arabic for Spanish
Bulgarian or Serbian for Romanian
German for French, flagged as an imperfect Frankish proxy
```

Default statistic:

```text
delta(X, D) = scalar projection of (X - C) onto unit(C -> D)
```

where `C` is the Romance centroid excluding `X`, computed in frozen in-loop plus
diagnostic feature space. Estimate uncertainty by bootstrap over corpus
subsamples at the common budget with `B = 1000`.

Registered predictions:

1. Romanian displacement toward the Slavic donor proxy exceeds every other
   Romance source's Slavic displacement, with non-overlapping bootstrap 95
   percent confidence intervals, replicated across token budgets.
2. Spanish displacement toward Arabic is statistically indistinguishable from
   the other Romance sources' Arabic displacements by overlapping bootstrap 95
   percent confidence intervals.
3. French/Germanic displacement is exploratory with no registered direction.

Power note: with five Romance sources, an across-source permutation test bottoms
out at `p = 0.2`; this experiment uses bootstrap CI separation plus
cross-budget replication rather than pretending to have more source-level power.

### E4. Small-Corpus Stress Test

Question: does the typological geometry survive degradation to small,
single-register corpora?

Default design:

```text
Budget: about 7000 tokens.
Register: administrative or legal preferred; fallback must remain formulaic
and restricted-vocabulary.
Experiments repeated: E1 convergence gate and E2 backbone reproduction.
```

If the required register family is unavailable for the Romance sources, document
the fallback before running. Do not substitute an unrelated single register such
as lyric poetry.

## Robustness Defaults

| Requirement | Default |
|---|---|
| Seeds | Multiple full seeds per source; exact seed list to be written before freeze |
| Nulls | Random-init trajectories, Markov-noise corpora, typological controls |
| Typological controls | Turkish, Finnish, Vietnamese, Sumerian, Klingon via Tatoeba as an optional constructed-language stress test; Japanese deferred post-H1 pending tokenization policy |
| Auxiliary positive diagnostic | Modern Occitan, non-required and excluded from static in-loop tier selection |
| Withheld Romance relative | Catalan or Sardinian |
| Donor comparators | MSA, Bulgarian or Serbian, German |
| Dimensionality check | PCA on frozen in-loop plus diagnostic feature matrix |
| Corpus provenance | `data/v2/corpora_manifest.json` with source, URL, access date, license, and pre-cleaning token count |

## Freeze Enforcement

After author H1 sign-off:

1. Commit this file.
2. Tag the commit `v2-prereg-freeze`.
3. Compute SHA-256 for this file.
4. Store both the freeze commit and SHA-256 in the v2 run config.
5. At run launch, recompute SHA-256 and refuse to start if it differs.
6. Write `features_freeze_commit`, `features_freeze_sha256`, `corpus_budget`,
   and `seed` into every run manifest.

Any change to feature formulas, rationales, tiers, analysis statistics, nulls,
thresholds, or registered predictions after the first run result invalidates the
freeze and must be reported as a new registration rather than a quiet edit.
