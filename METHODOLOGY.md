# Project RBT: Methodology

**Romance Baby Talk: A Statistical Retrodiction Engine**  
Independent Research / Space Ranger Press  
Draft as of: 2026-04-08  
Phase: P3R - Reinforcement-guided bridge generation

---

> This document is the paper-in-progress. It is updated after every completed phase.
> It must accurately reflect the active methodology and the current state of the
> pipeline, including null results, baseline runs, and methodological revisions.

---

## 1. Experimental Design

### 1.1 From blind retrodiction to reinforced bridge generation

The original project design treated Latin as a fully sequestered endpoint and asked
whether a backward statistical gradient from modern Romance languages would approach
Latin without ever seeing it.

That blind design has now been demoted to a baseline.

The active Phase 3 experiment is single-blind, target-conditioned bridge generation.
Modern Romance corpora are the starting state. Latin is the reinforcer. The model is
allowed to move toward Latin under explicit reward or gradient pressure, and the
scientific question is not whether arrival is possible. Arrival is possible by
construction. The scientific question is what path the system takes through linguistic
space while it is being pushed toward Latin.

Three outcomes are methodologically meaningful:

1. The generated bridge aligns with attested intermediate languages.
2. The generated bridge is internally coherent but does not align with attested stages.
3. The generated bridge collapses into statistical junk, indicating either a broken
   model or an incorrect ontology.

### 1.2 Single-blind architecture

The corpus partition now has three roles:

**Island A - Modern Romance:** Italian, French, Spanish, Romanian, Occitan, Genoese.
These are the source corpora whose token vocabularies are preserved during generation.

**Latin reward corpus:** Classical Latin from Perseus Digital Library. Latin remains
sequestered on disk and is only unlocked by the reinforcement engines with a logged
reason string. In Phase 3R it serves as the reward signal and directed-gradient target.
Latin is not copied token-for-token into generated bridge corpora; it conditions the
path rather than supplying emitted text.

**Portuguese positive control:** Portuguese remains withheld from the reinforcement loop.
It is reserved for post hoc comparison against generated bridges as an external Romance
validator not used during training.

### 1.3 Null models and validators

| Model / corpus | Description | Role | Source |
|---|---|---|---|
| Markov noise | Random token sequences, uniform transitions, n=2 | Noise floor | Generated (`src/nullmodel/markov.py`) |
| Sumerian | Attested language isolate, zero Indo-European relationship | Structured out-family diagnostic | ORACC DCCLT |
| Portuguese | Withheld Romance language | Positive control outside training | Wikipedia API |
| Latin | Reinforcement target and endpoint validator | In-loop target, out-of-loop interpretation | Perseus Digital Library |

Nulls remain diagnostic even though Latin is in the optimization loop. They help
distinguish coherent bridge languages from reward-hacked noise.

For reinforced stages, internal coherence is now operationalized as scaled distance to
an attested real-language centroid versus the Markov noise floor. A bridge remains
`coherent` when it stays materially closer to real-language space than to the Markov
reference.

---

## 2. Input Representation

All languages are represented as statistical fingerprints. No phonetic values. No
graphemic assumptions. Tokens are abstract units whose behavioral statistics are the
sole basis for inference.

Each language corpus is fingerprinted with four components:

1. **Co-occurrence matrix** - token adjacency frequencies, L2-normalized. Window = 2.
   Implementation: `src/fingerprint/cooccurrence.py`.
2. **Positional frequency distribution** - initial / medial / final rates, mean
   normalized position, standard deviation of normalized position, and log-normalized
   frequency. Implementation: `src/fingerprint/positional.py`.
3. **N-gram profiles** - top-5000 bigrams and trigrams by relative frequency.
   Implementation: `src/fingerprint/ngram.py`.
4. **Type/token ratio** - unique token types divided by total tokens. Computed in
   `src/fingerprint/ngram.py`.

The bridge-stage record format stores matrix paths plus structural summaries for every
generated stage.

---

## 3. Corpus Sources

| Language | Source | Register | Notes |
|---|---|---|---|
| Italian | Wikipedia API (it.wikipedia.org) | Formal encyclopedic | Primary Romance input |
| French | Wikipedia API (fr.wikipedia.org) | Formal encyclopedic | Primary Romance input |
| Spanish | Wikipedia API (es.wikipedia.org) | Formal encyclopedic | Primary Romance input |
| Romanian | Wikipedia API (ro.wikipedia.org) | Formal encyclopedic | Eastern branch |
| Occitan | Wikipedia API (oc.wikipedia.org) | Formal encyclopedic | Thin corpus - flagged |
| Genoese | Wikipedia API (lij.wikipedia.org) | Formal encyclopedic | Very thin corpus - flagged |
| Portuguese | Wikipedia API (pt.wikipedia.org) | Formal encyclopedic | Sequestered positive control |
| Latin | Perseus Digital Library | Classical formal | Reinforcement target |
| Sumerian | ORACC DCCLT | Lexical | Structured out-family diagnostic |
| Markov noise | Generated | N/A | Noise floor |

Register consistency remains a methodological asset. Classical Latin as preserved is
overwhelmingly formal. Wikipedia prose in all Romance languages is likewise formal and
encyclopedic. This reduces register mismatch as a confound, though it does not remove
the distinction between Classical Latin and unattested Vulgar Latin.

### 3.1 Wikipedia API strategy

Articles are fetched via `action=query&generator=random&grnnamespace=0&prop=extracts&explaintext=true`,
retrieving up to 20 random articles with plain-text extracts per request. Processing is
streaming: raw article text is tokenized on receipt and discarded. No raw Wikipedia
dumps are stored locally. Articles shorter than 200 characters are skipped as stubs.

### 3.2 Tokenization

All languages are tokenized identically using a language-agnostic Unicode tokenizer in
`src/ingest/tokenize.py`. Tokens are maximal runs of alphabetic Unicode characters.
Digits and punctuation are removed. Sentences are split on terminal punctuation followed
by uppercase. Tokens shorter than 2 characters are dropped.

---

## 4. Results

Phase P2 corpus construction is complete. Blind retrodiction runs were executed as an
exploratory baseline, but the primary experiment is now reinforcement-guided bridge
generation with Latin in the loop.

### 4.1 Corpus statistics (Phase P2)

All corpora ingested 2026-04-07.

**Island A - Modern Romance**

| Language | Articles | Sequences | Total tokens | Unique types | TTR |
|---|---|---|---|---|---|
| Italian | 500 | 30,576 | 450,393 | 49,811 | 0.1106 |
| French | 500 | 17,276 | 224,016 | 28,910 | 0.1291 |
| Spanish | 500 | 14,952 | 234,349 | 31,314 | 0.1336 |
| Romanian | 500 | 11,966 | 136,608 | 27,829 | 0.2037 |
| Occitan | 500 | 10,857 | 105,768 | 20,530 | 0.1941 |
| Genoese | 500 | 8,439 | 67,347 | 18,257 | 0.2711 |

**Held-out / auxiliary corpora**

| Corpus | Files / articles | Sequences | Total tokens | Unique types | TTR |
|---|---|---|---|---|---|
| Portuguese (positive control) | 500 | 15,282 | 202,211 | 30,347 | 0.1501 |
| Latin (reinforcement target) | 428 | 897,076 | 7,089,280 | 388,265 | 0.0548 |

**Null / diagnostic corpora**

| Corpus | Sequences | Total tokens | Unique types | TTR | Notes |
|---|---|---|---|---|---|
| Markov noise | 10,000 | 100,000 | 500 | 0.0050 | Uniform transitions, seed = 42 |
| Sumerian (ORACC DCCLT) | 1,510 | 79,113 | 17,607 | 0.2226 | CDLI primary unreachable; ORACC fallback |

### 4.2 Blind baseline runs (archived)

Unguided retrodiction runs were completed for French, Italian, Spanish, and Romanian
using the earlier `mix_toward_uniform` engine. These runs are retained as a baseline
for comparison but are no longer the primary methodological claim.

### 4.3 Reinforced bridge generation (Phase P3R)

The active experiment uses two Latin-conditioned algorithms implemented in
`src/retrodiction/engine_reinforced.py`:

1. `stochastic` - random perturbation plus Latin reward ("baby babble" selection)
2. `gradient` - direct mixing toward the Latin transition matrix

The primary initial case study is French -> Latin. Generated bridge corpora are first
evaluated for:

1. Internal coherence
2. Cross-algorithm convergence or divergence
3. Comparison against nulls and held-out controls

Comparison to attested intermediate corpora is the next validation layer and requires
separate historical-corpus ingestion.

That validator layer is now implemented locally via:

1. `src/ingest/historical.py` for on-disk attested corpus ingestion
2. `src/validation/checkpoint_compare.py` for checkpoint-ladder comparison

No attested validator corpus is bundled in the repo yet, but the workflow now exists
for immediate use once a text collection is dropped into `data/raw/historical/`.

A first pilot `old_french` validator packet has now been ingested locally from:

1. `Sequence de sainte Eulalie`
2. `Serments de Strasbourg`
3. `La Vie de saint Alexis` (full locally extracted render)

This pilot corpus is intentionally small and should be read as a first validator pass,
not the final historical benchmark.

The first comparison against the French `v2_convergence` ladder produced a split
result:

- best structural match: `FR_v2_061`
- best form match: `FR_v2_000`

So the current bridge appears to move toward the Old French pilot in structural space
while moving away from it in orthographic / suffix-form space. That is the first real
historical signal in the repo, but it is still provisional because the validator
packet is hand-assembled.

After expanding the `old_french` packet to include a full locally extracted
`Saint Alexis` render, the split result held:

- validator corpus stats: `661` sequences, `4620` tokens, `1378` types
- best structural match remained `FR_v2_061`
- best form match remained `FR_v2_000`

So the directional pattern survived the larger validator packet rather than
disappearing as a small-sample artifact.

The project also tested whether the French `v2` plateau might be a locality artifact.
A new meso-scale operator now rewrites contiguous spans of `2-5` sentence sequences
inside the reinforced `v2` engine. A short continuation probe from `FR_v2_061`,
with span-heavy weighting, accepted no moves. Diagnostic sampling found only
vanishingly small raw-score improvements from the best span mutations, which were
more than erased by the mutation-cost penalty.

So, under the current objective, the French endpoint plateau does not appear to be
caused only by too-small perturbation windows.

That negative result motivated a new experimental branch aimed at two specific
questions:

1. can stranger mutation families escape the current basin?
2. can Latin reward speak louder when a move is jointly correct on multiple axes?

French reinforced `v3` was added on 2026-04-08 in
`src/retrodiction/engine_reinforced_v3.py`. It extends the relational search with:

1. `function_word_burst`
2. `paradigm_family_rewrite`
3. `macro_bundle_rewrite`

It also adds explicit reward amplification on:

1. structural gain relative to the current baseline
2. form gain relative to the current baseline
3. suffix gain
4. trigram gain
5. joint-improvement bonuses and mutation-penalty relief

The first fresh French `v3` run from the original source corpus accepted two
`macro_bundle_rewrite` moves before stabilizing:

- `FR_v3_000 -> FR_v3_002`
- `latin_structural_score: -1.378939 -> -1.371337`
- `latin_form_score: 0.566157 -> 0.614681`

This confirmed that `v3` can move quickly, but it did not by itself beat the mature
`v2` endpoint on the raw Latin axes.

The more important result came from a continuation run that used the `v2`
convergence endpoint `FR_v2_061` as the source corpus for `v3`. That run produced a
new stable endpoint `FR_v3_008` and improved both raw Latin signals:

- `latin_structural_score: -1.306957 -> -1.302373`
- `latin_form_score: 0.762744 -> 0.798087`
- `language-likeness margin: 3.105235 -> 3.103579`

So the French plateau was not absolute. It was specific to the `v2` operator family
and reward geometry. Under stranger relational mutations plus louder reward for
jointly good Latin moves, the search found a new coherent improvement basin above
`FR_v2_061`.

The project now also has a diagnostic-only Phase 1 implementation of the proposed
`v4` control loop in `src/validation/hungarian_alignment.py`. This module does not
change mutation behavior. It extracts mutable family inventories from bridge corpora
and compares them to a Latin family reference under Hungarian assignment.

On the French `v2_convergence` ladder, the resulting family alignment score rises
from `0.397699` at `FR_v2_000` to `0.518986` at `FR_v2_061`, with the best late
checkpoint at `FR_v2_058 = 0.519016`.

On the French `v3_from_v2_endpoint` continuation ladder, the shared start
`FR_v3_000` begins at `0.518986` and the endpoint `FR_v3_008` reaches
`0.539860`.

So the new family-level diagnostic agrees with the existing raw Latin signals:
late `v2` is more Latin-aligned than early `v2`, and `v3` improves further beyond
the old `v2` basin under an independent global-alignment measure.

Phase 2 of the proposed `v4` direction is now also implemented in
`src/retrodiction/engine_reinforced_v4.py`. This engine keeps the `v3` mutation and
reward stack but uses the current Hungarian family-alignment score to schedule
operator weights through an inverse-log weirdness curve.

A continuation run from `FR_v3_008` produced a new stable endpoint `FR_v4_006` with
improved raw Latin scores:

- `latin_structural_score: -1.302373 -> -1.295360`
- `latin_form_score: 0.798138 -> 0.806686`

But the family-alignment signal did not improve monotonically. It peaked early at
`FR_v4_001 = 0.540554` and the stable endpoint `FR_v4_006` ended slightly lower at
`0.536294`.

So the alignment-driven scheduler appears strong enough to reopen movement, but not
yet strong enough to make the alignment axis itself the stable maximand. In other
words, Phase 2 worked as a controller experiment, but it does not yet justify giving
family alignment full control over the search.

The first Old French validator follow-up on the full `v4_from_v3_endpoint` ladder
is now complete. It preserved the same split seen in the earlier `v2` pilot, but
with a stronger structural result:

- best structural match: `FR_v4_006`
- validator structural distance: `1.995647`
- best form match: `FR_v4_000`
- validator form score: `0.611085`

Relative to the earlier `v2` pilot, the best Old-French structural match improved
from `2.026658` at `FR_v2_061` to `1.995647` at `FR_v4_006`, while the form side
still preferred the earlier stage over the late Latin-conditioned endpoint.

So the latest controller branch appears to strengthen historical legibility in
structural space without yet resolving the surface-form split. That makes the
current Old French validator signal more informative, not less: `v4` is not merely
chasing Latin blindly, but it is still pulling surface form harder toward Latin than
toward the attested intermediate packet.

The project also now has a direct test of the earlier `v4` plateau claim. A
continuation probe started from `FR_v4_006` and granted an extra `26` proposals,
approximately 50% of the original `51`-proposal run length. That probe did not
remain stuck. It accepted four additional mutations and halted only because the
added budget ran out, not because it re-entered a `stable` condition.

On the raw Latin axes, the post-plateau probe improved beyond the earlier endpoint:

- `latin_structural_score: -1.295360 -> -1.288622`
- `latin_form_score: 0.806686 -> 0.808937`

But the independent family-alignment diagnostic worsened over the same continuation:

- `family_alignment_score: 0.536294 -> 0.520696`

So the strongest form of the earlier plateau interpretation is now falsified. The
current `v4` branch can still move under the same rules, but the extra movement is
not diagnostically free. It is improving the direct Latin reward axes faster than
the independent family-alignment signal.

The post-plateau branch has now also been compared against the Old French validator
packet. That follow-up strengthens the earlier historical reading rather than
weakening it.

Across the `v4_post_plateau_50pct_probe` ladder:

- best structural match: `FR_v4_004`
- validator structural distance: `1.985277`
- best form match: `FR_v4_001`
- validator form score: `0.590938`

Relative to the prior `v4_from_v3_endpoint` validator comparison, the best
Old-French structural distance improved again from `1.995647` to `1.985277`.
Surface-form similarity, however, stayed in roughly the same lower band around
`0.59` and did not recover the earlier global best from the pre-probe ladder.

So the current interpretation sharpens:

1. extra continuation is not merely blind target-chasing, because it still improves
   the attested-validator structural match
2. the structural / surface-form split remains unresolved
3. endpoint choice is now better understood as a multi-objective selection problem,
   not as a single obvious winner under one scalar score

The project has also now tested a first explicit "culture bomb" idea rather than
only discussing it conceptually. A tandem probe branched from the current
post-plateau seed and ran:

1. a plain `v4` continuation control
2. a shock-enabled `v5` branch with plateau-triggered culture bombs

For this experiment, the plateau window was set to `10`, operationalizing the
user prompt as the number of accepted `v4` mutation stages that preceded the branch
point in the current French lineage.

Under that setup, both branches first followed the same three accepted moves. The
plain continuation halted as `stable` at `FR_v4_003`, while the shock branch fired
one culture bomb and then halted as `culture_bomb_plateau` without finding a better
rescue candidate.

So the first exogenous-shock implementation did not outperform the smooth branch.
At this point in the search, ordinary continuation still had more headroom than the
shock rescue did.

The smooth control endpoint was then compared against the Old French validator
packet and improved the best structural match again:

- `validator_structural_distance: 1.985277 -> 1.979246`

That means the current control branch is still moving in a historically legible
direction structurally, even though family alignment and surface-form scores remain
in tension. The present lesson is therefore not "culture shocks are useless," but
"the first culture-bomb operator is weaker than the current smooth continuation
under this score geometry."

French reinforced rerun completed 2026-04-08 with default configuration
(`num_sequences=2000`, `n_candidates=20`, `alpha=0.05`, `seed=42`) after revising
the Latin reward to exclude `log_mean_seq_len`, which the generator cannot optimize
because it preserves the source sentence-length distribution.

- `stochastic` ran for 10 stages and improved from `latin_score=-0.239143`
  to `-0.144710`, with a best stage of `FR_stoch_007` at `-0.125597`.
- `gradient` ran for 6 stages and improved from `latin_score=-0.292680`
  to `-0.242540`, with a best stage of `FR_grad_004` at `-0.242410`.

The earlier apparent gradient failure was therefore a reward-design artifact, not a
clean falsification of the reinforced method. Once the reward was restricted to the
trainable subspace `[type_token_ratio, bigram_coverage, trigram_coverage]`, both
algorithms moved French closer to the Latin reference under the active score.

The rerun also exposed an important constraint: French/Latin token overlap in the
capped source vocabulary is only `3.02%`, so the directed gradient path should be
interpreted as heuristic guidance rather than a literal token-aligned route. That low
overlap helps explain why the stochastic search still finds a stronger best stage than
the deterministic mixer under the current representation. A detailed rerun note is
stored in `docs/reports/2026-04-08_french_reinforced_rerun.md`, while the original
2026-04-07 report is retained as an archived pre-fix run.

The coherence diagnostic now resolves the immediate "junk vs bridge" question for this
case study. Across all 10 stochastic stages and all 6 gradient stages, the generated
French -> Latin bridges remained `coherent`, with positive language-likeness margins
throughout (`stochastic`: `2.117820 -> 2.606176`; `gradient`: `1.966161 -> 2.283905`).
In other words, the current reinforced runs are not merely drifting toward the Markov
floor while improving their Latin reward. They remain substantially closer to the
attested real-language manifold than to noise under the active diagnostic.

The older cosine similarities to Markov and Sumerian remain logged for continuity, but
they are still numerically compressed near `1.0` and should not be treated as the
primary discriminator of bridge quality. The margin-based coherence diagnostic is the
current operative filter for "coherent bridge" versus "noise-like collapse."

French relational reinforced v2 also completed on 2026-04-08 using
`src/retrodiction/engine_reinforced_v2.py`. This engine mutates actual sampled corpora
instead of only transition weights, using multi-scale operators:

1. token-level character edits
2. suffix-family rewrites across related word types
3. local bigram swaps
4. token splits
5. bigram merges

The first full French v2 run produced `18` total stages (`17` accepted mutations)
before halting at `max_accepted_stages`. The best stage `FR_v2_017` improved:

- `total_score`: `-0.814688 -> -0.699032`
- `latin_structural_score`: `-1.378939 -> -1.371562`
- `latin_form_score`: `0.566157 -> 0.715272`

All accepted v2 stages remained `coherent`. The accepted operator mix was:

- `suffix_family_rewrite = 9`
- `token_char_edit = 7`
- `swap_bigram_order = 1`

This is the first run in the project that produces an explicitly mutating,
source-derived synthetic bridge corpus rather than only a fixed-vocabulary structural
bridge. The detailed run note is stored in
`docs/reports/2026-04-08_french_reinforced_v2_run.md`.

---

## 5. Decisions Log

All parameter and design decisions are documented in `docs/decisions/`. Key decisions:

- `001_copy_vs_import.md` - fingerprint infrastructure copied from Minos rather than imported
- `002_cooccurrence_window.md` - co-occurrence window
- `003_ngram_top_n.md` - n-gram profile cap
- `004_sentence_splitting.md` - cross-language sentence splitting rule
- `005_wikipedia_api_strategy.md` - encyclopedic register via streaming Wikipedia ingest
- `006_corpus_size_target.md` - corpus-size target
- `007_markov_null_model.md` - Markov floor null
- `008_latin_corpus_source.md` - Latin source corpus
- `009_sumerian_source.md` - Sumerian source corpus
- `010_retrodiction_algorithm.md` - blind baseline retrodiction algorithm
- `011_similarity_metric.md` - structural similarity metric
- `012_reinforcement_protocol.md` - Latin-conditioned bridge-generation protocol
- `013_coherence_diagnostic.md` - language-likeness margin against Markov noise
- `014_relational_v2_engine.md` - multi-scale relational reinforced search over corpora
- `015_historical_validator_layer.md` - local attested-corpus ingestion and checkpoint comparison
- `016_meso_scale_span_operator.md` - contiguous multi-sentence span mutation for plateau testing
- `017_reinforced_v3_engine.md` - weird bundled mutations plus amplified Latin reward
- `018_hungarian_alignment_diagnostic.md` - family-level Latin alignment via Hungarian assignment
- `019_reinforced_v4_operator_schedule.md` - alignment-driven operator scheduling

---

## 6. Sequestration Protocol

The sequestration firewall is still enforced at the code level, but the project now
distinguishes between forbidden access and authorized reinforcement access.

- Ingest, fingerprint, and null-model modules remain forbidden from calling
  `load_sequestered()` or `unlock_sequestration()`.
- This restriction is enforced by `tests/test_sequestration.py`, which scans those
  modules for forbidden calls.
- Reinforced retrodiction engines may unlock the Latin corpus with a required, logged
  reason string. This is now part of the methodology rather than a violation.
- Portuguese remains outside the optimization loop and is reserved for post hoc scoring.
- Generated bridge corpora remain source-vocabulary synthetic corpora; Latin is used
  as a conditioning target, not copied text.

---

*This document will continue to expand as reinforced runs, controls, and attested-stage
comparisons are added.*
