# Latin Self-Similarity Calibration

**Date:** 2026-05-19
**Run under test:** `v5_fortran_c16_seed45_paper_run` final block
**Analysis script:** `tmp_self_similarity/calibration.py`
**Numeric outputs:** `data/validation/2026-05-19_self_similarity/`

## 1. Summary

The v5 paper reports three endpoint metrics — Latin structural score
-0.00010808926278147055, Latin form score 0.7611594796180725, family
alignment score 0.513866759690557 — as if 0, 1, 1 are the achievable
maxima. This report checks those maxima by scoring Latin against itself
through the production scoring functions (`LatinReference`,
`LatinFormReference`, `hungarian_alignment_diagnostics`) with no path
through mutation or candidate generation. Two findings: (a) the structural
formula stated in the paper (`cosine_similarity − 1`) is not what the code
computes — the engine actually computes `-REWARD_SCORE_SCALE · ||cand[:3] −
latin_reward[:3]||`, where `REWARD_SCORE_SCALE = 5.0` and only the first
three features (TTR, top-100 bg cov, top-100 tg cov) participate; (b) the
family-alignment ceiling is structurally 0.9583, not 1.0, because 2-character
short-token families with empty suffix profiles incur a fixed self-cost.

## 2. Sanity-check reproduction

Reproduced one validator-bank row — `(block_0614, old_french)` — using the
same loading and scoring path the validator-bank analysis uses
(`_load_corpus_and_profiles`, `structural_vector`, `cosine_similarity`,
`scaled_euclidean_distance`, `CorpusFormReference`).

| Field | Persisted CSV | Recomputed | Δ |
|---|---:|---:|---:|
| validator_structural_cosine | 0.999214086216678 | 0.999214086216678 | 0.0 |
| validator_structural_distance | 2.024154993404628 | 2.024154993404628 | 0.0 |
| validator_form_score | 0.383759842107158 | 0.383759842107158 | 0.0 |
| validator_char_bigram_cosine | 0.578808184556165 | 0.578808184556165 | 0.0 |
| validator_char_trigram_cosine | 0.302167292576561 | 0.302167292576561 | 0.0 |
| validator_suffix_cosine | 0.156848256270341 | 0.156848256270341 | 0.0 |

Max absolute delta: **0.0**. Loading path is identical.

## 3. Per-corpus self-similarity table

Each corpus is scored against itself by building a per-corpus reference
the same way the production code builds the Latin reference (structural
vector from corpus n-gram profiles, `CorpusFormReference.from_sequences`
for form, `extract_family_inventory` + `hungarian_alignment_diagnostics`
for family) and scoring the same sequences against that reference.

| Corpus | n_seq | n_tok | struct (cos−1) | struct (-5·‖Δ‖) | form | family | family pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| latin | 897 076 | 7 089 280 | +2.2e−16 | −0.0 | 1.000 | **0.9653** | 36 |
| old_french | 12 756 | 80 049 | +2.2e−16 | −0.0 | 1.000 | 0.9375 | 36 |
| middle_french | 6 217 | 31 818 | −1.1e−16 | −0.0 | 1.000 | 0.9375 | 36 |
| anglo_norman | 5 440 | 27 612 | 0.0 | −0.0 | 1.000 | 0.9444 | 36 |
| langue_d_oil | 6 008 | 33 873 | −1.1e−16 | −0.0 | 1.000 | 0.9375 | 36 |
| old_spanish | 4 174 | 29 651 | −2.2e−16 | −0.0 | 1.000 | 0.9583 | 36 |
| old_occitan | 9 899 | 70 826 | +2.2e−16 | −0.0 | 1.000 | 0.9444 | 36 |
| markov | 10 000 | 100 000 | 0.0 | −0.0 | 1.000 | **0.9375** | 12 |
| sumerian | 1 510 | 79 113 | 0.0 | −0.0 | 1.000 | 0.9722 | 36 |
| portuguese_withheld | 15 282 | 202 211 | 0.0 | −0.0 | 1.000 | 0.9444 | 36 |

When the per-corpus reference is built from the same sequences passed as
the candidate, structural cos−1 ≈ 0 and form score = 1 to floating-point
precision. The family-alignment column shows a consistent shortfall from
1.0 — explained in §4.

(Markov noise has only 12 family entries because its tokens are
synthetic and don't meet the `min_family_types` threshold for suffix or
prefix groupings; the family-alignment math still applies but operates
on a thinner inventory.)

### 3a. Production Latin scoring path

The table above uses the validator-bank `CorpusFormReference` form scorer
and a "build profile from the same input as the candidate" structural
reference. The production engine path is slightly different — both
`LatinReference` and `LatinFormReference` truncate to `sequences[:50_000]`
and load Latin's n-gram profile from a pre-built matrix file. Scoring
Latin against itself through the production classes gives:

| Quantity | Latin candidate = Latin[:50 000] | Latin candidate = full Latin |
|---|---:|---:|
| `LatinReference.score(vec)` (production formula) | **−0.17789** | −0.37107 |
| `cosine_similarity(cand, ref) − 1` (paper formula) | **−1.26e−4** | −5.36e−4 |
| `LatinFormReference.score` form score | **1.0000** | 0.9826 |
| Hungarian family alignment | **0.9583** | 0.8203 |

The matched-slice column is the "Latin candidate the production path can
naturally produce." The full-Latin column shows the additional asymmetry
that exists when the candidate corpus is larger than the 50 000-sequence
reference slice.

## 4. Deviations from (0, 1, 1) — diagnosis

### 4.1 The structural-score formula in the paper is not the engine's formula

Paper text (§2.2):

> The structural score is the cosine similarity between the candidate's
> four-dimensional vector and the target (Latin) four-dimensional vector,
> minus one (so that zero indicates an exact match and negative values
> indicate divergence).

Code (`src/retrodiction/engine_reinforced.py::LatinReference.score`,
also `IncrementalScoringState._score_virtual_state_batch`):

```python
candidate = vec[:reward_vec.shape[0]]            # first 3 features only
return -float(score_scale * np.linalg.norm(candidate - reward_vec))
# = -5.0 * Euclidean distance over [TTR, top100_bg_cov, top100_tg_cov]
```

The two formulas differ in three ways: (a) Euclidean vs cosine, (b) 3
features vs 4 (log mean sequence length is dropped from the reward
because the generator preserves source sentence-length distribution), (c)
scaled by 5.0. The manifest's
`final_latin_structural_score = -0.00010808926278147055` is the
Euclidean form — equivalent to `‖cand[:3] − latin_reward[:3]‖ = 2.16e-5`.

Both formulas indicate a near-perfect structural match in this run; the
paper claim is correct in spirit but the equation is mis-stated.

### 4.2 Plugging Latin in as the candidate doesn't reach the production ceiling

With Latin[:50 000] as the candidate, the production structural score is
**−0.178** (cos−1 form: −1.26e−4). It is not 0 because of a profile-build
asymmetry inside the Latin structural reference:

- Reference `vec` is built by `structural_vector(sequences[:50 000],
  latin_ngram_meta["bigrams"], latin_ngram_meta["trigrams"])`. The two
  n-gram profiles in `latin_ngram_meta.json` were built from the **full**
  897 076-sequence Latin corpus at pipeline time.
- Candidate `vec` is built by `structural_vector(seqs, bg_prof, tg_prof)`
  where `bg_prof = build_profile(extract_ngrams(seqs, 2), 5000)` — top-5000
  bigram profile built from whatever the candidate sequences are.

Even when the candidate sequences equal Latin[:50 000], its top-5000
profile differs from the pre-built full-Latin profile (different sample,
different head distribution). The resulting top-100 coverages differ:

```
Reference vec  : TTR=0.1290  top100_bg=0.1620  top100_tg=0.1244  log_seq=2.226
Candidate vec  : TTR=0.1290  top100_bg=0.1324  top100_tg=0.1047  log_seq=2.226
                                  ^^^^^^^^^^^         ^^^^^^^^^^^
```

`‖cand[:3] − ref[:3]‖ = √(0² + 0.0296² + 0.0198²) = 0.0356`, times 5.0 =
0.178. Exact match for the observed score.

The engine, however, has direct control over the candidate's top-5000
profile (it builds it freshly each iteration and mutates the underlying
corpus). It can drive the candidate's top-100 coverages toward the
reference's pre-built coverages independent of whether the candidate
corpus "looks like" Latin. **This is why the v5 search reached
−0.000108, smaller than the −0.178 you'd get by plugging Latin in
directly.** The production ceiling is effectively 0; the v5 run is at
~99.997 % of that ceiling.

### 4.3 Form ceiling: 1.0 is reachable only on the matched slice

`LatinFormReference` truncates Latin to `sequences[:50_000]` when building
the bg/tg/sfx reference profiles. Scoring `LatinFormReference.score(seqs)`:

- with `seqs = Latin[:50_000]` (same sequences the reference used): form
  score = **1.0000** exactly.
- with `seqs = full Latin` (897 076 sequences): form score = **0.9826**
  due to sampling drift between the full corpus and its [:50 000] head.

In production, the engine's candidate is 800 sequences sampled per
iteration. Plugging "Latin-as-candidate" through the production scorer
therefore does not reach 1.0 in practice; the achievable maximum on
real candidates depends on what the engine's 800-sample composition
looks like, but the metric itself can reach 1.0 if the candidate is the
matched [:50 000] slice. The v5 endpoint at **0.7611** has substantial
headroom against this ceiling.

### 4.4 Family alignment ceiling is 0.9583, not 1.0

For Latin scored against itself with matched inputs, the family
alignment score is **0.9583**, not 1.0. Diagnosed in
`tmp_self_similarity/debug_family.py`:

- Latin's family inventory has 36 entries: 12 suffix families, 12 prefix
  families, 12 short-token families (the most frequent 2–4-character
  tokens).
- 6 of the short-token families have 2-character tokens: `in`, `et`,
  `ut`, `ad`, `me`, `si`. `SUFFIX_LEN = 3`, so
  `_extract_suffixes_from_sequences` skips tokens with length < 3 and
  emits an empty suffix counter for these families.
- `_sparse_profile_cosine(empty, empty)` returns 0.0 by its
  short-circuit guard (`if not a or not b: return 0.0`), so even when
  comparing one of these families to itself, `suffix_cost = 1 −
  cos = 1`.
- With `suffix_weight = 0.25` in the cost function, each of these 6
  families contributes a fixed self-cost of 0.25, regardless of the
  candidate.
- Total fixed cost: 6 × 0.25 = 1.5. Normalised by 36 family slots:
  1.5 / 36 = 0.0417. Score: 1 − 0.0417 = **0.9583**. Exact.

The per-corpus family ceilings in §3 vary (0.9375 to 0.9722) by how many
of each corpus's top-12 short-token families are 2-character entries:

- old_french / middle_french / langue_d_oil / markov: 0.9375 (8 short
  families with empty suffix profile → 8 × 0.25 / 36 = 0.056).
- anglo_norman / old_occitan / portuguese_withheld: 0.9444 (7 short → 1
  − 7×0.25/36).
- old_spanish: 0.9583 (6 short).
- latin (full corpus): 0.9653 (5 short — slightly different short-token
  set than Latin[:50 000]).
- sumerian: 0.9722 (4 short — many Sumerian tokens are 3+ chars after
  romanisation, e.g. `lugal`, `kalam`, `gibil`).

The ceiling is an artifact of how the cost function handles empty
suffix profiles, not a property of the candidate corpus. Two ways to
read it: (a) the metric maxes out at ~0.94–0.97 depending on the
target language's character of its short-function-word inventory; (b)
the v5 endpoint at 0.5139 represents **53.6 %** of the actual
achievable Latin ceiling (0.5139 / 0.9583), not 51.4 % of 1.0.

## 5. Bottom line — how to read the v5 endpoint

Three corrections to apply to the manuscript's endpoint table:

| Metric | Paper's stated ceiling | Actual achievable ceiling (this code path) | v5 endpoint | % of actual ceiling |
|---|---:|---:|---:|---:|
| Latin structural score | 0 (cos−1) | 0 (production: `−5·‖Δ‖`; *not* cos−1) | −0.000108 | **99.997 %** of `-5·‖Δ‖=0` |
| Latin form score | 1 | 1 (when candidate matches `Latin[:50 000]`) | 0.7612 | **76.1 %** |
| Family alignment score | 1 | **0.9583** (Latin matched-slice ceiling) | 0.5139 | **53.6 %** of 0.9583 |

Recommendations for the manuscript update:

1. **Fix the structural-score formula in §2.2.** Either:
   - State the actual production formula:
     `S_struct = −5.0 · ‖[TTR_cand, bg_cov_cand, tg_cov_cand] −
     [TTR_Latin, bg_cov_Latin, tg_cov_Latin]‖₂`,
     and note that the 4th feature (log mean sequence length) is
     excluded from the structural reward because the generator
     preserves source sentence-length distribution;
   - or change the engine to actually compute the stated cos−1 formula
     (post-paper change, separate PR — not recommended for this paper
     since the existing artifacts use the Euclidean form).

2. **Report the family-alignment ceiling alongside the endpoint.** The
   ceiling of 0.9583 for Latin should appear in §3 or §4 so the reader
   does not mistakenly compare 0.5139 against a 1.0 maximum that the
   metric cannot reach. The v5 endpoint represents ~54 % of achievable,
   not ~51 % of nominal. This is also a Limitation candidate (the
   metric has a known structural ceiling < 1 for any language with
   common 2-character function words, which is most Indo-European
   languages).

3. **Form score has real headroom.** v5's 0.7611 is only 76 % of the
   1.0 ceiling. The plateau stopping rule fired despite this, which is
   a methodological observation worth noting: the plateau is a stop
   condition on relative improvement, not on closeness to ceiling. The
   form axis is genuinely far from a Latin-form match at the endpoint,
   and that gap is real, not an artifact of ceiling-vs-formula
   confusion.

4. **The "structural ceiling" framing for the paper's structural
   score.** Because the engine can match the reference's top-100
   coverage values directly (without the candidate corpus being
   anywhere near Latin in any deeper sense), the structural ceiling
   is reachable for *any* candidate corpus that happens to share those
   three summary statistics with Latin. v5's near-zero structural
   score therefore certifies summary-statistic agreement, not
   linguistic resemblance. The dimensional-robustness report
   (2026-05-19_dimensional_robustness_check.md) already reframes
   the metric as "typological proximity detector"; this calibration
   adds that even within that frame, the ceiling is easily reached
   by direct summary-statistic matching, independent of typological
   substance.
