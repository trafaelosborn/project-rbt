# 2026-04-25 - Paper LLM Handoff

## Purpose

This document is a single paper-facing handoff note for the current French ->
Latin Project RBT manuscript. It is meant to give another LLM or human editor a
complete, conservative source of truth about:

- what the system is
- what was actually run
- what engineering changes were made
- what the completed paper run produced
- what validation and control analyses were performed afterward
- what claims are safe to make
- what claims should be avoided

Use this as a manuscript input, not as a marketing brief.

## One-Sentence Framing

Project RBT currently supports a **target-conditioned, non-ML retrodictive
search** from a modern Romance-language corpus toward a Latin target, and the
completed French -> Latin `v5` paper run shows that this search can reach a
near-zero Latin structural endpoint under the active metric while preserving a
non-random, in-family trajectory under held-out validator and control analyses.

## Claim Boundaries

The paper should stay inside these boundaries.

### Safe

- The method is **target-conditioned**.
- The method is **non-ML**.
- The search is driven by explicit scoring axes and operator families.
- The validator bank is **held out** from the optimization loop.
- The control bank is **post hoc** only.
- The completed `v5` French -> Latin run reached a near-zero Latin structural
  endpoint under the active structural fingerprint metric.
- The run shows a form/structure tradeoff.
- The completed run stayed closer to relevant attested/withheld controls than
  to Markov noise.

### Avoid

- "unsupervised reconstruction"
- "zero supervision"
- "blind discovery"
- "reconstructed Latin"
- "structurally indistinguishable from Latin" unless explicitly qualified as
  "under the active structural fingerprint metric"
- "chronologically replayed French history"

The clean description is: **target-conditioned retrodictive search with held-out
historical validation and post hoc controls**.

## Project State Relevant to the Paper

### Main manuscript source

- [docs/manuscript/french_to_latin_retrodiction.tex](/C:/Code/Project%20RBT/project_rbt/docs/manuscript/french_to_latin_retrodiction.tex)

### Main paper-run report

- [docs/reports/2026-04-18_v5_paper_run.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-18_v5_paper_run.md)

### Supporting reports now relevant to the manuscript

- [docs/reports/2026-04-24_v5_paper_run_seed_audit.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-24_v5_paper_run_seed_audit.md)
- [docs/reports/2026-04-25_v5_control_bank.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-25_v5_control_bank.md)
- [docs/reports/2026-04-25_v5_suffix_ablation_probe.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-25_v5_suffix_ablation_probe.md)
- [docs/reports/2026-04-25_v5_seed_replication_probe.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-25_v5_seed_replication_probe.md)
- [docs/reports/2026-04-12_v4_final_state.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-12_v4_final_state.md)
- [docs/reports/2026-04-16_v5_fortran_candidate_scaling.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-16_v5_fortran_candidate_scaling.md)
- [docs/reports/2026-04-17_fortran_hotloop_delta_feed.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-17_fortran_hotloop_delta_feed.md)
- [docs/reports/2026-04-17_fortran_structural_batch_pass.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-17_fortran_structural_batch_pass.md)

## What the System Is

### Search setup

The active paper configuration is a search over synthetic corpora derived from
modern French. The engine mutates the current corpus, scores each candidate
against a Latin target, and accepts or rejects based on the composite objective.

### Main axes used in the run

- `latin_structural_score`
- `latin_form_score`
- `family_alignment_score`

### Broad division of labor

- Python owns search logic, mutation operators, acceptance/rejection,
  coherence gating, run orchestration, and artifact writing.
- Fortran owns selected hot-loop batch math, primarily in the batched form and
  candidate evaluation path.

### Relevant code modules

- [src/retrodiction/engine_reinforced_v2.py](/C:/Code/Project%20RBT/project_rbt/src/retrodiction/engine_reinforced_v2.py)
- [src/retrodiction/engine_reinforced_v4.py](/C:/Code/Project%20RBT/project_rbt/src/retrodiction/engine_reinforced_v4.py)
- [src/retrodiction/engine_reinforced_v5.py](/C:/Code/Project%20RBT/project_rbt/src/retrodiction/engine_reinforced_v5.py)
- [src/retrodiction/long_run_v5.py](/C:/Code/Project%20RBT/project_rbt/src/retrodiction/long_run_v5.py)
- [src/accelerate/incremental_scoring_state.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/incremental_scoring_state.py)
- [src/accelerate/fortran_cosine.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/fortran_cosine.py)
- [src/accelerate/fortran_batch.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/fortran_batch.py)
- [src/accelerate/bridge_distance.f90](/C:/Code/Project%20RBT/project_rbt/src/accelerate/bridge_distance.f90)
- [src/validation/validator_bank_compare.py](/C:/Code/Project%20RBT/project_rbt/src/validation/validator_bank_compare.py)
- [src/validation/control_bank_compare.py](/C:/Code/Project%20RBT/project_rbt/src/validation/control_bank_compare.py)
- [src/validation/suffix_ablation_probe.py](/C:/Code/Project%20RBT/project_rbt/src/validation/suffix_ablation_probe.py)

## Engineering Path to the Paper Run

This section is for methods context, not for overlong narrative in the paper.
It explains how the production `v5` configuration emerged.

### 1. V4 baseline and incremental scoring

The long `v4` line demonstrated that French -> Latin search could drive the
structural axis to a near-zero endpoint, but runtime was expensive.

The key acceleration change before `v5` was incremental scoring via
[docs/decisions/025_incremental_scoring_state.md](/C:/Code/Project%20RBT/project_rbt/docs/decisions/025_incremental_scoring_state.md).

That pass added:

- cached `score_token()` lookups
- precomputed token and bigram counts
- incremental `evaluate()` over changed sequences only

Measured benchmark:

- baseline full proposal: `5263.577 ms`
- incremental full proposal: `245.65 ms`
- measured proposal speedup: `21.4x`
- estimated proposals/hour:
  - baseline: `683`
  - incremental: `14654`

Artifact:

- [data/validation/incremental_scoring_benchmark.json](/C:/Code/Project%20RBT/project_rbt/data/validation/incremental_scoring_benchmark.json)

Important paper point:

- this change preserved the search methodology
- it changed the runtime profile, not the optimization objective

### 2. Early Fortran work and integration lessons

Earlier `v4`-era Fortran integration proved the compile boundary and batch
tensor landscape extraction, but whole-loop speedup was not immediate.

Relevant design notes:

- [docs/decisions/022_fortran_v4_guidance_integration.md](/C:/Code/Project%20RBT/project_rbt/docs/decisions/022_fortran_v4_guidance_integration.md)
- [docs/decisions/024_phase4_v4_incremental_guidance_integration.md](/C:/Code/Project%20RBT/project_rbt/docs/decisions/024_phase4_v4_incremental_guidance_integration.md)

Key lesson:

- raw Fortran kernels could be fast in isolation
- whole-loop runtime only improved when the compiled path owned enough of the
  actual hot loop

This matters for the paper because the final `v5` build is not "Fortran
bolted onto Python for style." It reflects several rounds of narrowing the
Python/Fortran boundary to a region where it actually helps.

### 3. V5 prelaunch Fortran tuning

Three short reports establish the immediate prelaunch computational context for
the paper run.

#### Candidate scaling

- [docs/reports/2026-04-16_v5_fortran_candidate_scaling.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-16_v5_fortran_candidate_scaling.md)

This compared `plain_python` vs `fortran_batch` under:

- source corpus: processed French
- target: Latin
- proposals per probe: `50`
- seed: `77`
- incremental scoring: on
- semantic transparency: off
- culture bombs: off

Selected results:

- `32` candidates: Fortran `2561.5 p/h` vs Python `2280.2 p/h` (`1.123x`)
- `64` candidates: Fortran `798.5 p/h` vs Python `968.2 p/h` (`0.825x`)
- `100` candidates: Fortran `487.3 p/h` vs Python `450.4 p/h` (`1.082x`)

Interpretation at that stage:

- the Fortran path was real, not hypothetical
- widening candidate pools did not automatically guarantee better throughput
- `32` candidates looked like the best throughput-oriented production choice

#### Hot-loop delta feed

- [docs/reports/2026-04-17_fortran_hotloop_delta_feed.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-17_fortran_hotloop_delta_feed.md)

This moved candidate setup closer to the compiled path by sending a committed
baseline plus sparse candidate deltas instead of rebuilding full char-counter
copies in Python.

Selected benchmark results:

- `8` candidates: `8035.6 p/h` vs `6869.6 p/h` (`1.170x`)
- `16` candidates: `4193.4 p/h` vs `3583.1 p/h` (`1.170x`)

Matched head-to-head (`100` proposals, seed `77`, `16` candidates):

- `v4 baseline`: `3710 p/h`
- `v5 plain Python`: `3726 p/h`
- `v5 Fortran batch`: `3983 p/h`

Throughput gain over `v4` baseline at that stage:

- about `7.4%`

#### Structural batch pass

- [docs/reports/2026-04-17_fortran_structural_batch_pass.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-17_fortran_structural_batch_pass.md)

This reduced Python selection-time overhead further by:

- computing top-k coverage directly from counters
- skipping transient word-profile dict construction
- vectorizing structural and coherence calculations across the candidate batch

Selected benchmark results:

- `8` candidates: `7222.2 p/h` vs `5810.9 p/h` (`1.243x`)
- `16` candidates: `3693.1 p/h` vs `3009.8 p/h` (`1.227x`)
- `32` candidates: `1972.9 p/h` vs `1462.2 p/h` (`1.349x`)

Matched head-to-head:

- `v4 baseline`: `3184 p/h`
- `v5 plain Python`: `3100 p/h`
- `v5 Fortran batch`: `3839 p/h`

The main engineering conclusion after this pass was:

- the Fortran line was now measurably ahead of both matched baselines in the
  16-candidate configuration

## Frozen V4 Baseline

The main `v4` comparison line for the paper is:

- [docs/reports/2026-04-12_v4_final_state.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-12_v4_final_state.md)

Relevant endpoint:

- run: `v4_until_plateau_from_30k`
- final block: `block_0253`
- final structural: `-0.0000372424255784854`
- final form: `0.755795`
- final alignment: `0.547835`
- final cumulative proposals: `283000`
- halt mode: manual stop at clean block boundary after near-zero structural
  convergence

This `v4` endpoint is a real baseline, but it is not a native plateau stop.
The `v5` paper run is stronger on that point because it halted natively by
plateau.

## Completed V5 Paper Run

### Run identity

- run id: `v5_fortran_c16_seed45_paper_run`
- run root:
  [data/retrodiction/french/v5_fortran_c16_seed45_paper_run](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run)
- manifest:
  [manifest.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/manifest.json)
- start corpus:
  [data/processed/romance/french_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/processed/romance/french_tokens.json)

### Launch configuration

- `language = french`
- `start_corpus = data/processed/romance/french_tokens.json`
- `output_dir = data/retrodiction/french/v5_fortran_c16_seed45_paper_run`
- `total_target_proposals = 0` (indefinite until native halt)
- `block_proposals = 1000`
- `starting_proposals = 0`
- `num_sequences = 800`
- `n_candidates = 16`
- `max_accepted_stages = 512`
- requested `seed = 45`
- audited engine `seed = 42`
- `min_improvement = 0.0001`
- `struct_target = 0.0`
- `form_target = 1.0`
- `family_target = 1.0`
- `use_fortran_cosine = true`
- `use_fortran_batch = true`
- `use_incremental_scoring = true`
- `use_semantic_transparency = false`
- `transparency_weight = 0.0`
- `enable_culture_bombs = false`
- `validator_set = []`
- `validator_snapshot_every_blocks = 0`
- `live_event_mode = all`
- `live_event_buffer_size = 64`
- `plateau_window_blocks = 10`
- `plateau_struct_epsilon = 0.001`
- `plateau_form_epsilon = 0.001`
- `plateau_family_epsilon = 0.001`

### Runtime

- wall-clock start: `2026-04-18T17:23:49.299227+00:00`
- wall-clock stop: `2026-04-24T08:49:33.599011+00:00`
- elapsed runtime: `5.15:25:44.2997840`
- cumulative proposals: `614000`
- completed blocks: `614`
- observed proposals/hour: `4533.74`

### Native halt state

- manifest status: `plateau_hit`
- `plateau_hit = true`
- `joint_hit = false`
- `latin_hit = false`

Important implementation nuance:

- the outer run halted natively by plateau
- the final block still records `halt_reason = max_proposals` because each
  block itself finishes at its ordinary proposal boundary

### Final endpoint

- final block: `block_0614`
- final corpus:
  [FR_v5_001_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/blocks/block_0614/corpora/FR_v5_001_tokens.json)
- final preview:
  [FR_v5_001_preview.txt](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/blocks/block_0614/previews/FR_v5_001_preview.txt)
- final block summary:
  [run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/blocks/block_0614/run_summary.json)

Final scores:

- `latin_structural_score = -0.00010808926278147055`
- `latin_form_score = 0.7611594796180725`
- `family_alignment_score = 0.513866759690557`
- `accepted_mutation_stages = 1`

### Preserved secondary checkpoints

#### Best form checkpoint

- block: `block_0338`
- corpus:
  [FR_v5_006_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/blocks/block_0338/corpora/FR_v5_006_tokens.json)
- preview:
  [FR_v5_006_preview.txt](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/blocks/block_0338/previews/FR_v5_006_preview.txt)
- `latin_form_score = 0.8632466197013855`

#### Best structural checkpoint

- block: `block_0613`
- corpus:
  [FR_v5_002_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/blocks/block_0613/corpora/FR_v5_002_tokens.json)
- preview:
  [FR_v5_002_preview.txt](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/blocks/block_0613/previews/FR_v5_002_preview.txt)
- `latin_structural_score = -0.00010808926278147055`

### Main scientific observations from the paper run

- The run reached a near-zero Latin structural endpoint under the active
  structural fingerprint metric.
- Form peaked much earlier than the final structural endpoint.
- Family alignment improved above the French baseline but did not approach 1.0.
- The run halted natively by plateau rather than by manual stop.

These facts support a paper centered on:

- target-conditioned retrodictive convergence
- multi-axis tradeoff
- held-out post hoc comparison against attested validators and controls

## Seed Audit

The seed discrepancy is resolved and should be reported transparently.

### Problem

Historical artifacts disagreed:

- manifest launch field recorded `seed = 45`
- per-block summaries serialized `seed = 42`

### Audit method

`block_0001` was replayed exactly under the current codebase with two seeds:

- `42`
- `45`

The replay was compared against the historical `block_0001` artifact on:

- final structural score
- final form score
- final alignment score
- accepted-stage count
- accepted proposal indices
- operator sequence
- mutation details

### Audit result

The historical `block_0001` matches the `seed = 42` replay exactly and does not
match the `seed = 45` replay.

Historical `block_0001` endpoint:

- `final_latin_structural_score = -1.3435869472793973`
- `final_latin_form_score = 0.7181036472320557`
- `final_family_alignment_score = 0.4392205108144682`
- `accepted_mutation_stages = 5`

Replay `seed = 42` endpoint:

- `final_latin_structural_score = -1.3435869472793973`
- `final_latin_form_score = 0.7181036472320557`
- `final_family_alignment_score = 0.4392205108144682`
- `accepted_mutation_stages = 5`

Replay `seed = 45` endpoint:

- `final_latin_structural_score = -1.3062274172668467`
- `final_latin_form_score = 0.6928220391273499`
- `final_family_alignment_score = 0.4252034121913133`
- `accepted_mutation_stages = 6`

### Paper-safe wording

- requested launch seed: `45`
- audited engine seed: `42`

Relevant artifacts:

- [docs/reports/2026-04-24_v5_paper_run_seed_audit.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-24_v5_paper_run_seed_audit.md)
- [data/retrodiction/french/v5_fortran_c16_seed45_paper_run/seed_audit.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/seed_audit.json)
- [data/retrodiction/_seed_audit_replay/seed_42/run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/_seed_audit_replay/seed_42/run_summary.json)
- [data/retrodiction/_seed_audit_replay/seed_45/run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/_seed_audit_replay/seed_45/run_summary.json)

Important manuscript recommendation:

- do not pretend this was a clean single-seed-45 run
- do not hide the discrepancy
- state that the discrepancy was resolved by replay audit

## Attested Validator Bank

### Active validator set

The post hoc attested validator bank currently contains six active corpora:

- `old_french`
- `middle_french`
- `anglo_norman`
- `langue_d_oil`
- `old_spanish`
- `old_occitan`

Manifest:

- [data/raw/historical/validator_bank_manifest.csv](/C:/Code/Project%20RBT/project_rbt/data/raw/historical/validator_bank_manifest.csv)

These validators are attested-only and are treated as a separate validation
layer rather than as part of the optimization objective.

### Output artifacts

- CSV:
  [french_v5_fortran_c16_seed45_paper_run_validator_bank_vs_validator_bank.csv](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_fortran_c16_seed45_paper_run_validator_bank_vs_validator_bank.csv)
- JSON:
  [french_v5_fortran_c16_seed45_paper_run_validator_bank_vs_validator_bank.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_fortran_c16_seed45_paper_run_validator_bank_vs_validator_bank.json)
- chronology:
  [french_v5_fortran_c16_seed45_paper_run_validator_bank_vs_validator_bank_chronology.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_fortran_c16_seed45_paper_run_validator_bank_vs_validator_bank_chronology.json)

### Summary

Chronology summary records:

- validator count: `6`
- block count: `614`
- structural path:
  - `middle_french -> anglo_norman -> old_occitan`
- form path:
  - `middle_french -> langue_d_oil -> middle_french -> langue_d_oil -> middle_french -> langue_d_oil -> middle_french`

Final nearest validators:

- final nearest structural validator: `old_occitan`
- final nearest form validator: `middle_french`

Selected best structural hits by validator:

- `old_french`: `1.940476565238847` at `block_0465`
- `middle_french`: `2.173683680162571` at `block_0296`
- `anglo_norman`: `2.307027567554334` at `block_0353`
- `langue_d_oil`: `2.0475554407198437` at `block_0405`
- `old_spanish`: `4.629460911126837` at `block_0080`
- `old_occitan`: `1.762215854664522` at `block_0424`

### Safe interpretation

Paper-safe reading:

- the synthetic trajectory moved through a western Romance neighborhood rather
  than collapsing into arbitrary nonsense
- the observed path is not a simple reverse-chronological replay of French
  history
- the validator evidence is post hoc, descriptive, and held out from the run

Avoid saying:

- "the engine rediscovered the exact chronology of French"

## Null / Control Bank

### Controls used

The post hoc control bank currently compares the paper run against:

- Markov noise floor
- Sumerian structured non-Indo-European control
- Portuguese withheld positive control

Code path:

- [src/validation/control_bank_compare.py](/C:/Code/Project%20RBT/project_rbt/src/validation/control_bank_compare.py)

Control corpus/material paths:

- [data/processed/nulls/markov/markov_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/processed/nulls/markov/markov_tokens.json)
- [data/processed/nulls/sumerian/sumerian_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/processed/nulls/sumerian/sumerian_tokens.json)
- Portuguese is loaded as a sequestered positive control via
  [src/sequester/guard.py](/C:/Code/Project%20RBT/project_rbt/src/sequester/guard.py)

### Output artifacts

- CSV:
  [french_v5_fortran_c16_seed45_paper_run_vs_control_bank.csv](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_fortran_c16_seed45_paper_run_vs_control_bank.csv)
- JSON:
  [french_v5_fortran_c16_seed45_paper_run_vs_control_bank.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_fortran_c16_seed45_paper_run_vs_control_bank.json)
- chronology:
  [french_v5_fortran_c16_seed45_paper_run_vs_control_bank_chronology.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_fortran_c16_seed45_paper_run_vs_control_bank_chronology.json)
- report:
  [docs/reports/2026-04-25_v5_control_bank.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-25_v5_control_bank.md)

### Summary

Observed nearest-control paths:

- structural control path:
  - `vs_sumerian -> vs_portuguese_control`
- form control path:
  - `vs_portuguese_control`

First observed wins:

- first structural Sumerian win: `block_0001`
- first structural Portuguese win: `block_0300`
- first form Portuguese win: `block_0001`
- Markov wins: none on either axis

Final block (`block_0614`) structural ranking:

- Portuguese: `1.129272`
- Sumerian: `2.590237`
- Markov noise: `3.524765`

Final block form ranking:

- Portuguese: `0.355281`
- Sumerian: `0.104529`
- Markov noise: `0.0`

Best observed block by control:

- best Portuguese structural distance anywhere: `0.972993` at `block_0474`
- best Sumerian structural distance anywhere: `2.31845` at `block_0408`
- best Markov structural distance anywhere: `3.524765` at `block_0613`
- best Portuguese form score anywhere: `0.553427` at `block_0002`
- best Sumerian form score anywhere: `0.159201` at `block_0028`
- Markov form score remained `0.0`

### Safe interpretation

These controls support the narrower claim that:

- the synthetic trajectory did not collapse into random noise
- the run remained more language-like than Markov null output
- the run moved into a withheld Romance positive-control neighborhood

These controls do **not** by themselves prove chronology or blind discovery.

## Suffix Ablation Probe

This is a useful mechanism note, but it is not yet a full second production run.

### Purpose

Bounded ablation comparing:

- baseline Latin form scoring
- suffix-off Latin form scoring

### Configuration

- source language: `french`
- target language: `latin`
- start corpus:
  [data/processed/romance/french_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/processed/romance/french_tokens.json)
- seed: `42`
- max proposals per condition: `500`
- candidates per proposal: `16`
- sequences: `800`
- `use_incremental_scoring = true`
- `use_fortran_cosine = true`
- `use_fortran_batch = false`
- `use_semantic_transparency = false`
- `enable_culture_bombs = false`
- `live_event_mode = off`

### Condition weights

Baseline:

- char bigram weight: `0.40`
- char trigram weight: `0.40`
- suffix weight: `0.20`

Suffix-off:

- char bigram weight: `0.50`
- char trigram weight: `0.50`
- suffix weight: `0.00`

### Output artifacts

- report:
  [docs/reports/2026-04-25_v5_suffix_ablation_probe.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-25_v5_suffix_ablation_probe.md)
- probe summary:
  [summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_suffix_ablation_probe_seed42_p0500/summary.json)
- baseline:
  [baseline/run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_suffix_ablation_probe_seed42_p0500/baseline/run_summary.json)
- suffix-off:
  [suffix_off/run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_suffix_ablation_probe_seed42_p0500/suffix_off/run_summary.json)

### Result

Baseline final endpoint:

- `final_total_score = -0.10613149872945121`
- `final_latin_structural_score = -0.9884937520551179`
- `final_latin_form_score = 0.7127510786056519`
- `final_family_alignment_score = 0.4419887306417807`
- `accepted_mutation_stages = 5`

Suffix-off final endpoint:

- `final_total_score = -0.09606797474840026`
- `final_latin_structural_score = -0.9799469220233539`
- `final_latin_form_score = 0.7508169412612915`
- `final_family_alignment_score = 0.42924286060223005`
- `accepted_mutation_stages = 5`

Observed delta (`suffix_off - baseline`):

- total: `+0.010063523981050954`
- structural: `+0.008546830031764041`
- form: `+0.038065862655639604`
- alignment: `-0.012745870039550633`
- accepted stages: `0`

### Interpretation boundary

What this supports:

- in this bounded probe, removing suffix contribution did **not** reduce
  accepted-stage count
- the suffix-off condition ended with better total, structural, and form scores
  but worse family alignment

What this does **not** support yet:

- a strong manuscript claim that suffix profiles are globally harmful
- replacement of the production configuration

### Important caveat

Under `use_fortran_cosine = true`, some stage-level diagnostics in the probe
artifacts are `NaN` for component-level form internals because the fast path
stores the aggregate weighted form score rather than the full component trace.
The endpoint totals and main axis scores are still valid, but the probe should
be presented as a bounded ablation result, not as a complete mechanistic
decomposition.

## Light Multi-Seed Replication

This is the current bounded answer to the "single-run story" criticism.

### Report and artifacts

- report:
  [docs/reports/2026-04-25_v5_seed_replication_probe.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-25_v5_seed_replication_probe.md)
- summary:
  [summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_seed_replication_probe_seed42_43_45_p1000/summary.json)

### Condition

Three bounded runs from the same clean French start corpus:

- seeds: `42`, `43`, `45`
- proposals per seed: `1000`
- candidates: `16`
- sequences: `800`
- `use_incremental_scoring = true`
- `use_fortran_cosine = true`
- `use_fortran_batch = true`
- `use_semantic_transparency = false`
- `enable_culture_bombs = false`

### Result

All three seeds:

- completed normally
- remained coherent
- improved structural score relative to their own seed-stage baseline
- improved form score relative to their own seed-stage baseline
- improved family alignment relative to their own seed-stage baseline

Observed final ranges:

- structural:
  `-1.3613948534656453` to `-1.3062274172668467`
- form:
  `0.6928220391273499` to `0.7181036472320557`
- alignment:
  `0.4252034121913133` to `0.4392205108144682`
- accepted stages:
  `5` to `6`

Additional useful detail:

- the bounded `seed = 42` replication matches the historical paper-run
  `block_0001` exactly on the tracked endpoint fields

### Claim boundary

This supports a narrow claim of **bounded multi-seed early-window stability**.

It does **not** yet justify a stronger claim that the full long-horizon plateau
trajectory has been replicated across multiple seeds.

## Tests and Verification Already Completed

Relevant completed verification from the work summarized above:

- `tests/test_incremental_scoring_state.py`:
  - `11 passed`
- `tests/test_cosine_acceleration.py` after weighted-form update:
  - `23 passed`
- `tests/test_cosine_acceleration.py tests/test_incremental_scoring_state.py -q`
  during hot-loop tuning:
  - `33 passed`
- `tests/test_long_run_v4.py` after seed-audit hardening:
  - `6 passed`

Current paper-facing point:

- the production path and the post hoc analyses are backed by targeted test
  coverage, not only by one long run

## Suggested Paper Structure Inputs

If another LLM is adding to the paper, these are the main insertions it should
be able to make from this document.

### Methods

- describe the run as target-conditioned, non-ML retrodictive search
- describe the three scored axes
- describe the incremental scoring path
- describe the compiled batch path conservatively
- describe the plateau halt rule exactly
- describe validator and control banks as post hoc and held out
- describe the seed audit transparently

### Results

- final `v5` endpoint metrics
- best-form and best-structure checkpoints
- form/structure tradeoff across the run
- validator-bank path summary
- control-bank path summary

### Limitations / caveats

- target-conditioned, not blind
- one full paper-scale French -> Latin run
- seed discrepancy existed and was resolved by replay audit
- suffix ablation is bounded, not yet a full replication
- chronology signal is suggestive, not a clean reverse historical replay

## What the Next Manuscript Pass Should Probably Do

### Add

- a compact table for:
  - final plateau endpoint
  - best-form checkpoint
  - best-structure checkpoint
- a compact table for:
  - validator-bank nearest path
  - control-bank nearest path
- a short reproducibility paragraph about requested seed `45` vs audited engine
  seed `42`
- a brief bounded-ablation paragraph or appendix note

### Do not add

- speculative theoretical explanation of the phase transition unless directly
  supported by new evidence
- blind-reconstruction framing
- claims that the run "proved" chronology
- claims that the suffix ablation has settled mechanism questions globally

## Short Abstract-Safe Summary

If an LLM needs a compact factual summary, use something close to this:

Project RBT ran a completed French -> Latin `v5` retrodictive search under a
frozen target-conditioned, non-ML configuration with incremental scoring and
Fortran-backed batch acceleration. The production run started from processed
modern French, ran for `614000` proposals across `614` blocks, and halted
natively by plateau. The final endpoint reached
`latin_structural_score = -0.00010808926278147055`,
`latin_form_score = 0.7611594796180725`, and
`family_alignment_score = 0.513866759690557`. The best-form checkpoint occurred
earlier at `block_0338` with `latin_form_score = 0.8632466197013855`, while the
best-structure checkpoint was preserved at `block_0613`. Post hoc comparison
against a six-corpus attested validator bank yielded a structural path of
`middle_french -> anglo_norman -> old_occitan` and a final nearest form
validator of `middle_french`. Post hoc comparison against a control bank
comprising Markov noise, Sumerian, and a withheld Portuguese positive control
showed that Markov never became the nearest control on either axis and that the
run moved from Sumerian to Portuguese as the nearest structural control by
`block_0300`. A replay audit resolved a seed discrepancy by establishing that
the requested launch seed was `45` but the actual engine seed used by the run
was `42`. A bounded three-seed early-window replication (`42`, `43`, `45`)
showed coherent same-direction progress across the first `1000` proposals under
the same `v5` condition family, with all three seeds improving structure, form,
alignment, and total score relative to their own seed-stage baselines.

## Bottom Line

At this point the manuscript has a complete paper-run narrative with:

- frozen production configuration
- final run artifacts
- resolved seed audit
- post hoc attested validator-bank comparison
- post hoc control-bank comparison
- bounded ablation note
- clear claim boundaries

This is enough for another LLM to expand the paper without guessing the project
history.
