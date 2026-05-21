# Project RBT Snapshot

Date: 2026-04-08

## Purpose

This document consolidates the current state of Project RBT into one place:

1. methodology
2. completed runs
3. current results
4. what has and has not been validated
5. key output locations
6. recommended next steps

It is intended as the single status snapshot for the repo as of 2026-04-08.

## Current status

- Active phase: `P3R - Reinforcement-guided bridge generation`
- Pipeline status: `25/25 complete`
- Repo state:
  - ingest complete
  - fingerprinting complete
  - blind structural retrodiction complete for four Romance languages
- reinforced French runs complete
- relational reinforced v2 French runs complete
- meso-scale continuation probe complete
- relational reinforced v3 pilot complete
- alignment-scheduled reinforced v4 continuation complete
- v4 vs Old French validator comparison complete
- v4 post-plateau 50% continuation probe complete
- v4 post-plateau vs Old French validator comparison complete
- tandem v4 control vs v5 culture-bomb probe complete

## Core question

Project RBT is no longer asking the narrow question:

- "Can French blindly discover Latin?"

The active project question is:

- "If French is driven toward Latin under explicit reinforcement, what path does it take through linguistic space?"
- "Is that path coherent as a language-like bridge?"
- "Does that path align with attested intermediates, or does it reveal plausible unattested alternatives?"

## Methodology in its current form

### 1. Blind baseline

The original retrodiction engine treated Latin as fully sequestered and moved descendant
languages backward by a structural reverse operator. That baseline remains useful, but
it is now understood as a structural experiment rather than a full lexical
reconstruction engine.

### 2. Reinforced bridge generation

The active methodology is single-blind, target-conditioned bridge generation:

- modern Romance language = starting state
- Latin = reinforcement target
- generated corpus path = primary object of study

This means:

- the endpoint can be measured against Latin
- but endpoint agreement with Latin is not independent validation, because Latin is
  in the optimization loop
- the strongest future validator is attested historical intermediate material

### 3. Coherence filter

Bridge quality is currently screened with the language-likeness margin against the
Markov floor:

- `coherent`
- `borderline`
- `noise_like`

This is the project's current operational answer to:

- "Are we generating a bridge or just setting tokens on fire?"

### 4. Reinforced v2

The relational reinforced v2 engine widens the search space by mutating actual sampled
corpora rather than only transition weights.

Operator family:

1. `token_char_edit`
2. `suffix_family_rewrite`
3. `swap_bigram_order`
4. `split_token`
5. `merge_bigram`

Score family:

1. Latin structural reward
2. Latin form reward
3. coherence margin
4. mutation-cost penalty

### 5. Reinforced v3

The relational reinforced v3 engine keeps the v2 corpus-mutation model but adds
stranger operators and louder reward when a move is jointly good on the Latin axes.

Operator additions:

1. `function_word_burst`
2. `paradigm_family_rewrite`
3. `macro_bundle_rewrite`

Reward additions:

1. structural gain bonus
2. form gain bonus
3. suffix / trigram gain bonus
4. joint-improvement bonus
5. mutation-penalty relief when the move is good and coherence-safe

## Corpus inventory

### Modern / reference corpora present

- French
- Italian
- Spanish
- Romanian
- Occitan
- Genoese

### Controls / nulls present

- Latin
- Portuguese
- Sumerian
- Markov noise

### Validator layer

A first pilot attested historical corpus has now been ingested:

- `old_french`

The current `old_french` packet contains:

- `Sequence de sainte Eulalie`
- `Serments de Strasbourg`
- `La Vie de saint Alexis` (full locally extracted render)

The validator tooling also now exists in-repo:

- `src/ingest/historical.py`
- `src/validation/checkpoint_compare.py`

That means:

- endpoint-vs-Latin scoring is available now
- pilot intermediate-vs-attested-stage comparison is now available
- larger-scale validator comparison is still pending richer corpora

## Completed runs

### A. Blind structural retrodiction

Completed stable runs:

| Language | Total stages | Final stage | Stable |
|---|---:|---|---|
| French | 14 | `FR_retro_013` | yes |
| Italian | 21 | `IT_retro_020` | yes |
| Romanian | 5 | `RO_retro_004` | yes |
| Spanish | 40 | `ES_retro_039` | yes |

Important limitation:

- all blind baseline runs still have `vs_portuguese_control = null`
- all blind baseline runs still have `vs_latin_ground_truth = null`

So the structural baseline is complete as a generation step, but not yet complete as a
validator-scored experiment.

### B. French reinforced v1: reward-fixed rerun

After removing `log_mean_seq_len` from the Latin reward, the reinforced engine became
trainable in the dimensions it actually controls.

#### `stochastic`

- total stages: `10`
- stable: yes
- final stage: `FR_stoch_009`
- final Latin score: `-0.144710`
- best stage: `FR_stoch_007`
- best Latin score: `-0.125597`
- final coherence: `coherent`
- final language-likeness margin: `2.390170`

#### `gradient`

- total stages: `6`
- stable: yes
- final stage: `FR_grad_005`
- final Latin score: `-0.242540`
- best stage: `FR_grad_004`
- best Latin score: `-0.242410`
- final coherence: `coherent`
- final language-likeness margin: `2.283905`
- Latin vocab overlap ratio: `0.0302`

Interpretation:

- both algorithms improved toward Latin under the corrected reward
- both remained coherent
- the stochastic path outperformed the directed gradient path under the current
  representation

### C. French reinforced v2: first full run

This was the first run where the bridge itself mutated as a corpus.

- total stages: `18`
- accepted mutation stages: `17`
- halt reason: `max_accepted_stages`
- best / final stage: `FR_v2_017`
- final total score: `-0.699032`
- final Latin structural score: `-1.371562`
- final Latin form score: `0.715272`
- final coherence: `coherent`

Accepted operator counts:

- `suffix_family_rewrite = 9`
- `token_char_edit = 7`
- `swap_bigram_order = 1`

Interpretation:

- v2 successfully produced a path instead of stalling
- most movement was in form drift, not large structural change

### D. French reinforced v2: convergence run

This is the current deepest French run and the best statement of where the present v2
engine actually converges.

- total stages: `62`
- accepted mutation stages: `61`
- proposals attempted: `175`
- halt reason: `stable`
- best / final stage: `FR_v2_061`
- final total score: `-0.580618`
- final Latin structural score: `-1.306957`
- final Latin form score: `0.762769`
- final coherence: `coherent`

Accepted operator counts:

- `token_char_edit = 25`
- `suffix_family_rewrite = 18`
- `split_token = 12`
- `swap_bigram_order = 6`

Notable improvement from v2 seed to stable endpoint:

- total score: `-0.814688 -> -0.580618`
- Latin structural score: `-1.378939 -> -1.306957`
- Latin form score: `0.566157 -> 0.762769`
- language-likeness margin: `2.792672 -> 3.105235`

Interpretation:

- the current v2 engine does not reach literal Latin
- it does reach a stable Latin-conditioned attractor
- that attractor remains language-like under the current coherence screen
- the bridge is now visibly synthetic rather than merely reweighted French

### E. French reinforced v3: fresh run

This was the first test of weird bundled mutations plus amplified Latin reward.

- total stages: `3`
- accepted mutation stages: `2`
- halt reason: `stable`
- best / final stage: `FR_v3_002`
- final total score: `-0.359718`
- final Latin structural score: `-1.371337`
- final Latin form score: `0.614681`
- final coherence: `coherent`

Accepted operator counts:

- `macro_bundle_rewrite = 2`

Interpretation:

- `v3` moved quickly from a cold start
- most of the apparent win was reward amplification plus form drift
- the fresh `v3` restart did not beat the mature raw `v2` endpoint

### F. French reinforced v3: continuation from `FR_v2_061`

This is the more important `v3` result.

Starting point:

- `data/retrodiction/french/v2_convergence/corpora/FR_v2_061_tokens.json`

Output:

- `data/retrodiction/french/v3_from_v2_endpoint/`

Run summary:

- total stages: `9`
- accepted mutation stages: `8`
- proposals attempted: `57`
- halt reason: `stable`
- best / final stage: `FR_v3_008`
- final total score: `-0.474742`
- final Latin structural score: `-1.302373`
- final Latin form score: `0.798087`
- final coherence: `coherent`

Accepted operator counts:

- `function_word_burst = 4`
- `suffix_family_rewrite = 2`
- `paradigm_family_rewrite = 1`
- `macro_bundle_rewrite = 1`

Improvement relative to the `v2` endpoint seed:

- total score: `-0.579637 -> -0.474742`
- Latin structural score: `-1.306957 -> -1.302373`
- Latin form score: `0.762744 -> 0.798087`
- language-likeness margin: `3.105235 -> 3.103579`

Interpretation:

- the `v2` plateau was not absolute
- weird perturbations plus louder Latin reward found a new coherent basin
- the new gains are real on the raw Latin axes, not only on the amplified total score
- the movement is stronger on form than on structure

### G. French reinforced v4: continuation from `FR_v3_008`

This is Phase 2 of the proposed `v4` direction.

Starting point:

- `data/retrodiction/french/v3_from_v2_endpoint/corpora/FR_v3_008_tokens.json`

Output:

- `data/retrodiction/french/v4_from_v3_endpoint/`

Run summary:

- total stages: `7`
- accepted mutation stages: `6`
- proposals attempted: `51`
- halt reason: `stable`
- best / final stage: `FR_v4_006`
- final total score: `-0.476586`
- final Latin structural score: `-1.295360`
- final Latin form score: `0.806686`
- final family alignment score: `0.536294`
- final coherence: `coherent`

Accepted operator counts:

- `function_word_burst = 1`
- `macro_bundle_rewrite = 2`
- `token_char_edit = 2`
- `paradigm_family_rewrite = 1`

Improvement relative to the `v3` endpoint seed:

- total score: `-0.548276 -> -0.476586`
- Latin structural score: `-1.302373 -> -1.295360`
- Latin form score: `0.798138 -> 0.806686`

Important nuance:

- family alignment peaked early at `FR_v4_001 = 0.540554`
- the stable endpoint `FR_v4_006` ended lower at `0.536294`

Interpretation:

- alignment-driven operator scheduling is enough to reopen movement again
- it improves the raw Latin axes
- it does not yet produce monotonic improvement on the family-alignment axis
- `v4` Phase 2 therefore looks useful, but not yet complete as a controller

### H. Old French validator follow-up on `v4`

This is the first attested-validator comparison for the new `v4` ladder.

Compared checkpoints:

1. `FR_v4_000`
2. `FR_v4_001`
3. `FR_v4_002`
4. `FR_v4_003`
5. `FR_v4_004`
6. `FR_v4_005`
7. `FR_v4_006`

Comparison output:

- `data/validation/french_v4_from_v3_endpoint_vs_old_french.json`

Best structural match:

- `FR_v4_006`
- structural distance: `1.995647`

Best form match:

- `FR_v4_000`
- validator form score: `0.611085`

Interpretation:

- the old structural / form split persists under `v4`
- but the structural side strengthens relative to the earlier `v2` pilot
- best structural distance improved from `2.026658` at `FR_v2_061` to `1.995647`
  at `FR_v4_006`
- so the extra `v4` movement appears historically legible in structural space,
  even though surface form is still being pulled away from the Old French packet

### I. French reinforced v4: post-plateau 50% probe

This probe tests whether the earlier `v4` endpoint `FR_v4_006` was a true
stopping point or just the end of the original budget.

Starting point:

- `data/retrodiction/french/v4_from_v3_endpoint/corpora/FR_v4_006_tokens.json`

Output:

- `data/retrodiction/french/v4_post_plateau_50pct_probe/`

Probe budget:

- extra proposals: `26`
- rationale: approximately 50% of the original `51`-proposal `v4` run

Run summary:

- total stages: `5`
- accepted mutation stages: `4`
- proposals attempted: `26`
- halt reason: `max_proposals`
- best / final stage: `FR_v4_004`
- final coherence: `coherent`

Raw-axis comparison to the prior endpoint:

- Latin structural score: `-1.295360 -> -1.288622`
- Latin form score: `0.806686 -> 0.808937`
- family alignment score: `0.536294 -> 0.520696`

Interpretation:

- the earlier `v4` plateau was not absolute
- a modest post-plateau continuation found four more accepted moves
- the extra movement improved the raw Latin axes but worsened the independent
  family-alignment diagnostic
- the probe halted on budget, not on `stable`

### J. Old French validator follow-up on the post-plateau branch

This is the first attested-validator comparison for the new post-plateau `v4`
continuation.

Compared checkpoints:

1. `FR_v4_000`
2. `FR_v4_001`
3. `FR_v4_002`
4. `FR_v4_003`
5. `FR_v4_004`

Comparison output:

- `data/validation/french_v4_post_plateau_50pct_probe_vs_old_french.json`

Best structural match:

- `FR_v4_004`
- structural distance: `1.985277`

Best form match:

- `FR_v4_001`
- validator form score: `0.590938`

Interpretation:

- the post-plateau continuation remained historically legible in structural space
- best structural distance improved again from `1.995647` to `1.985277`
- form similarity stayed roughly flat around `0.59` and did not recover the earlier
  global best from the pre-probe `v4` ladder
- so the extra continuation still looks like real structural bridge movement, not
  just blind target-chasing, but the structural / surface-form split remains

### K. Tandem control vs culture-bomb probe

This probe branched from the current post-plateau seed and ran:

1. a plain `v4` continuation control
2. a shock-enabled `v5` continuation with plateau-triggered culture bombs

Shared seed:

- `data/retrodiction/french/v4_post_plateau_50pct_probe/corpora/FR_v4_004_tokens.json`

Plateau assumption used for both branches:

- `10` proposal plateau window

Rationale:

- `10` accepted `v4` mutation stages preceded this branch point in the current
  French lineage

Control result:

- output: `data/retrodiction/french/v4_control_long_from_post_plateau/`
- final stage: `FR_v4_003`
- halt reason: `stable`
- Latin structural score: `-1.288622 -> -1.284454`
- Latin form score: `0.808937 -> 0.809516`

Culture-bomb result:

- output: `data/retrodiction/french/v5_culture_bomb_from_post_plateau/`
- final stage: `FR_v5_003`
- halt reason: `culture_bomb_plateau`
- culture bombs used: `1`

Interpretation:

- both branches found the same three accepted moves before the next plateau
- the first culture-bomb trigger fired, but it did not rescue the branch into a new
  basin
- under this configuration, the smooth continuation still outperformed the shock
  idea in practice simply by already having more headroom

Historical follow-up:

- the control endpoint improved the Old-French structural match again to `1.979246`
  in `data/validation/french_v4_control_long_from_post_plateau_vs_old_french.json`
- so the extra smooth continuation remained historically legible

## What the project currently shows

### Strongest positive results

1. The pipeline runs end to end and is stable enough to generate repeated bridge paths.
2. The reward-fixed reinforced engines improve toward Latin without collapsing into
   Markov-like junk.
3. The relational v2 engine produces a genuinely mutating synthetic bridge path.
4. The deep v2 convergence run finds a stable endpoint rather than wandering forever.
5. The `v4` scheduling experiment shows the bridge can still move under a more
   global controller without losing coherence.
6. The `v4` ladder improves the best Old-French structural match beyond the earlier
   `v2` validator result.
7. A short post-plateau continuation shows the current `v4` branch can still move
   past `FR_v4_006` under the same rules.
8. That post-plateau continuation also improves the best Old-French structural
   match again.
9. A tandem control vs culture-bomb probe shows the current smooth continuation
   still beats the first shock-rescue implementation.

### Strongest current limitations

1. Attested historical intermediate material is still limited; the current
   `old_french` packet is a pilot rather than a decisive benchmark.
2. Endpoint-vs-Latin is not independent validation because Latin is the reinforcer.
3. The blind baseline completed generation but still lacks Portuguese / Latin
   validation fields.
4. The original reinforced v1 engine remains structurally useful but is still narrow
   because its vocabulary constraints are stronger than the v2 engine.
5. Even `v4` only uses alignment to schedule operator choice; it does not yet use
   alignment directly in acceptance or penalty relief.
6. The Old-French split still persists, so historical legibility is currently
   stronger in structure than in surface form.
7. Extra post-plateau movement currently improves the raw Latin axes while degrading
   family alignment, so the diagnostics are not yet unified.
8. The best surface-form match still lives earlier on the path than the strongest
   structural endpoint does.
9. The first culture-bomb implementation did not yet open a better basin than the
   plain continuation branch.

## What can be claimed right now

Safe current claims:

- The project can generate coherent French -> Latin-conditioned bridge paths.
- Different engine families produce different bridge geometries.
- The current v2 engine converges to a stable synthetic endpoint.
- The current bridges are not noise-like under the active coherence diagnostic.

Claims not yet justified:

- That any checkpoint matches an attested historical intermediate.
- That the v2 endpoint is historically correct rather than merely target-conditioned.
- That the observed path generalizes across descendant languages.

## Validation status

### What is validated now

- pipeline completion
- targeted unit tests
- coherence under the current diagnostic
- endpoint movement toward Latin under in-loop scoring
- pilot attested intermediate comparison for both `v2` and `v4`
- that the earlier `v4` plateau was budget-sensitive rather than absolute
- that the post-plateau `v4` continuation improves the Old-French structural match
- that the current smooth continuation branch also beats the first `v5`
  shock-rescue probe from the same seed

### What is still missing

1. Richer attested intermediate comparison beyond the pilot `old_french` packet
2. Hold-out Latin validation
3. Portuguese control scoring for completed baseline retrodictions
4. Cross-language relational v2 runs beyond French
5. A decisive answer on where the new post-plateau `v4` continuation truly stabilizes

## Recommended next steps

### Immediate next move

Add a hold-out Latin split and rerun endpoint evaluation with an out-of-loop target
check. The smooth continuation branch remains the best current cheap experiment, and
its latest endpoint has stayed historically legible on the structural axis.

### After that

1. Add a hold-out Latin split and rerun the French ladder with an out-of-loop
   endpoint check.
2. Replace the hand-built `old_french` packet with a richer scholarly corpus such
   as OTA `0176`.
3. Add Portuguese and Latin post hoc scoring for the blind baseline run summaries.
4. Repeat v2 or later engines on another descendant language for convergence
   comparison.
5. If revisiting culture shocks, tune the shock operator itself rather than assuming
   more shock quantity will help.

### Parallel research track

Design and, if warranted, implement a `v4` engine where:

1. family-level Hungarian assignment measures global bridge-vs-Latin mismatch
2. an inverse-log schedule turns that mismatch into mutation weirdness
3. proposal weirdness is uncapped, but acceptance remains coherence-gated

Summary document:

- `docs/reports/2026-04-08_v4_hungarian_weirdness_summary.md`

## Key file locations

### Core methodology / decisions

- `METHODOLOGY.md`
- `docs/decisions/012_reinforcement_protocol.md`
- `docs/decisions/013_coherence_diagnostic.md`
- `docs/decisions/014_relational_v2_engine.md`
- `docs/decisions/015_historical_validator_layer.md`
- `docs/decisions/017_reinforced_v3_engine.md`
- `docs/decisions/018_hungarian_alignment_diagnostic.md`
- `docs/decisions/019_reinforced_v4_operator_schedule.md`

### Run reports

- `docs/reports/2026-04-07_french_reinforced_run.md`
- `docs/reports/2026-04-08_french_reinforced_rerun.md`
- `docs/reports/2026-04-08_french_reinforced_v2_run.md`
- `docs/reports/2026-04-08_french_reinforced_v3_probe.md`
- `docs/reports/2026-04-08_french_reinforced_v4_probe.md`
- `docs/reports/2026-04-08_french_v4_post_plateau_50pct_probe.md`
- `docs/reports/2026-04-08_old_french_validator_pilot.md`
- `docs/reports/2026-04-08_old_french_validator_v4.md`
- `docs/reports/2026-04-08_old_french_validator_v4_post_plateau.md`
- `docs/reports/2026-04-08_tandem_control_vs_culture_bomb_probe.md`
- `docs/reports/2026-04-08_french_meso_probe.md`
- `docs/reports/2026-04-08_v4_hungarian_weirdness_summary.md`
- `docs/reports/2026-04-08_hungarian_alignment_diagnostic.md`

### Baseline retrodiction summaries

- `data/retrodiction/french/run_summary.json`
- `data/retrodiction/italian/run_summary.json`
- `data/retrodiction/romanian/run_summary.json`
- `data/retrodiction/spanish/run_summary.json`

### Reinforced French summaries

- `data/retrodiction/french/stochastic/run_summary.json`
- `data/retrodiction/french/gradient/run_summary.json`
- `data/retrodiction/french/v2/run_summary.json`
- `data/retrodiction/french/v2_convergence/run_summary.json`
- `data/retrodiction/french/v3/run_summary.json`
- `data/retrodiction/french/v3_from_v2_endpoint/run_summary.json`
- `data/retrodiction/french/v4_from_v3_endpoint/run_summary.json`
- `data/retrodiction/french/v4_post_plateau_50pct_probe/run_summary.json`
- `data/retrodiction/french/v4_control_long_from_post_plateau/run_summary.json`
- `data/retrodiction/french/v5_culture_bomb_from_post_plateau/run_summary.json`

### Best current French bridge artifacts

- `v2` stable endpoint summary: `data/retrodiction/french/v2_convergence/run_summary.json`
- `v2` stable endpoint preview: `data/retrodiction/french/v2_convergence/previews/FR_v2_061_preview.txt`
- `v2` stable endpoint corpus: `data/retrodiction/french/v2_convergence/corpora/FR_v2_061_tokens.json`
- `v3` continuation endpoint summary: `data/retrodiction/french/v3_from_v2_endpoint/run_summary.json`
- `v3` continuation endpoint preview: `data/retrodiction/french/v3_from_v2_endpoint/previews/FR_v3_008_preview.txt`
- `v3` continuation endpoint corpus: `data/retrodiction/french/v3_from_v2_endpoint/corpora/FR_v3_008_tokens.json`
- `v4` continuation endpoint summary: `data/retrodiction/french/v4_from_v3_endpoint/run_summary.json`
- `v4` continuation endpoint preview: `data/retrodiction/french/v4_from_v3_endpoint/previews/FR_v4_006_preview.txt`
- `v4` continuation endpoint corpus: `data/retrodiction/french/v4_from_v3_endpoint/corpora/FR_v4_006_tokens.json`
- `v4` post-plateau probe summary: `data/retrodiction/french/v4_post_plateau_50pct_probe/run_summary.json`
- `v4` post-plateau probe preview: `data/retrodiction/french/v4_post_plateau_50pct_probe/previews/FR_v4_004_preview.txt`
- `v4` post-plateau probe corpus: `data/retrodiction/french/v4_post_plateau_50pct_probe/corpora/FR_v4_004_tokens.json`
- `v4` tandem-control summary: `data/retrodiction/french/v4_control_long_from_post_plateau/run_summary.json`
- `v4` tandem-control preview: `data/retrodiction/french/v4_control_long_from_post_plateau/previews/FR_v4_003_preview.txt`
- `v5` culture-bomb summary: `data/retrodiction/french/v5_culture_bomb_from_post_plateau/run_summary.json`
- `v5` culture-bomb preview: `data/retrodiction/french/v5_culture_bomb_from_post_plateau/previews/FR_v5_003_preview.txt`

### New validation outputs

- `data/validation/french_v2_convergence_vs_latin_family_alignment.json`
- `data/validation/french_v3_from_v2_endpoint_vs_latin_family_alignment.json`
- `data/validation/french_v4_from_v3_endpoint_vs_latin_family_alignment.json`
- `data/validation/french_v2_convergence_vs_old_french.json`
- `data/validation/french_v4_from_v3_endpoint_vs_old_french.json`
- `data/validation/french_v4_post_plateau_50pct_probe_vs_old_french.json`
- `data/validation/french_v4_control_long_from_post_plateau_vs_old_french.json`

## Verification

Recent direct verification completed on 2026-04-08:

- `python pipeline.py --status` -> `25/25 complete`
- targeted `py_compile` passed for v2 / v3 engines, pipeline, and tests
- targeted pytest passed for:
  - `tests/test_engine_reinforced_v2.py`
  - `tests/test_engine_reinforced_v3.py`
  - `tests/test_engine_reinforced_v4.py`
  - `tests/test_retrodiction_similarity.py`
  - `tests/test_validation_hungarian_alignment.py`

Known environment issue:

- full tmpdir-backed pytest remains flaky on this Windows machine because of temp
  directory permission cleanup behavior

## Bottom line

Project RBT is past the stage of "does anything run?" and into the stage of "what
exactly have we built, and how do we validate it historically?"

The current best answer is:

- we have a coherent, stable, synthetic French -> Latin-conditioned bridge path
- we have now pushed that bridge beyond the original `v2` plateau with `v3`
- the new Hungarian family-alignment diagnostic agrees with that `v3` improvement
- `v4` pushes the raw Latin scores a bit further, but alignment peaks early rather
  than at the stable endpoint
- a post-plateau continuation shows `v4` can move further still under the same rules
- that extra movement improves raw Latin structure/form but worsens family alignment
- the `v4` endpoint is the best Old-French structural match seen so far, while the
  surface-form split still persists
- the post-plateau endpoint improves the Old-French structural match yet again, but
  surface-form similarity stays roughly flat around the same lower band
- the latest smooth continuation improves the Old-French structural match again to
  `1.979246`
- the first `v5` culture-bomb branch did not beat the plain continuation from the
  same seed
- we still do not have decisive attested historical validation because the current
  Old French packet is only a pilot
- the next decisive move is validator comparison on the new post-plateau endpoint,
  now followed by hold-out Latin plus a richer historical validator corpus
