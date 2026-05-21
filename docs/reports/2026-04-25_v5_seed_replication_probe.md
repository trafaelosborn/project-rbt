# V5 Seed Replication Probe

Date: 2026-04-25  
Status: complete

## Purpose

Run a lightweight multi-seed replication under the same early-window French ->
Latin `v5` condition family as the paper run, without committing to another
plateau-scale production run.

This probe is meant to answer a narrower question:

- does the first `1000` proposals of the current `v5` configuration produce
  coherent, same-direction progress under multiple seeds?

It is **not** a second full paper-scale run and does **not** establish whether
the entire long-horizon phase structure is identical across seeds.

## Inputs

- Start corpus:
  [french_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/processed/romance/french_tokens.json)
- Probe runner:
  [v5_seed_replication_probe.py](/C:/Code/Project%20RBT/project_rbt/src/validation/v5_seed_replication_probe.py)
- Historical paper-run block for direct comparison:
  [block_0001/run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/blocks/block_0001/run_summary.json)
- Output root:
  [french_v5_seed_replication_probe_seed42_43_45_p1000](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_seed_replication_probe_seed42_43_45_p1000)

## Probe Configuration

- source language: `french`
- target language: `latin`
- seeds: `42, 43, 45`
- total proposals per seed: `1000`
- block proposals: `1000`
- candidates per proposal: `16`
- sequences: `800`
- `use_incremental_scoring = true`
- `use_fortran_cosine = true`
- `use_fortran_batch = true`
- `use_semantic_transparency = false`
- `enable_culture_bombs = false`
- `live_event_mode = all`
- `live_event_buffer_size = 64`

## Output Artifacts

- probe summary:
  [summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_seed_replication_probe_seed42_43_45_p1000/summary.json)

Per-seed artifacts:

- seed `42` manifest:
  [manifest.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_seed_replication_probe_seed42_43_45_p1000/seed_42/manifest.json)
- seed `43` manifest:
  [manifest.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_seed_replication_probe_seed42_43_45_p1000/seed_43/manifest.json)
- seed `45` manifest:
  [manifest.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_seed_replication_probe_seed42_43_45_p1000/seed_45/manifest.json)

Per-seed block summaries:

- seed `42`:
  [run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_seed_replication_probe_seed42_43_45_p1000/seed_42/blocks/block_0001/run_summary.json)
- seed `43`:
  [run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_seed_replication_probe_seed42_43_45_p1000/seed_43/blocks/block_0001/run_summary.json)
- seed `45`:
  [run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_seed_replication_probe_seed42_43_45_p1000/seed_45/blocks/block_0001/run_summary.json)

## Final Endpoint Comparison

All three seed runs:

- completed normally with `status = complete`
- used exactly `1000` proposals
- ended `coherent`

### Seed 42

- `final_total_score = -0.49454441348341216`
- `final_latin_structural_score = -1.3435869472793973`
- `final_latin_form_score = 0.7181036472320557`
- `final_family_alignment_score = 0.4392205108144682`
- `accepted_mutation_stages = 5`

Within-run gain from seed stage:

- structural: `+0.035352052721`
- form: `+0.151946647232`
- alignment: `+0.041521876598`
- total: `+0.320143586517`

Accepted operator mix:

- `macro_bundle_rewrite = 5`

Historical consistency check:

- this bounded `seed = 42` replication matches the historical paper-run
  `block_0001` exactly on the tracked endpoint fields

### Seed 43

- `final_total_score = -0.5149508485352493`
- `final_latin_structural_score = -1.3613948534656453`
- `final_latin_form_score = 0.7108287215232849`
- `final_family_alignment_score = 0.4263131778499346`
- `accepted_mutation_stages = 5`

Within-run gain from seed stage:

- structural: `+0.031148146534`
- form: `+0.144519721523`
- alignment: `+0.028074337738`
- total: `+0.315237151465`

Accepted operator mix:

- `macro_bundle_rewrite = 4`
- `function_word_burst = 1`

### Seed 45

- `final_total_score = -0.4499532712738164`
- `final_latin_structural_score = -1.3062274172668467`
- `final_latin_form_score = 0.6928220391273499`
- `final_family_alignment_score = 0.4252034121913133`
- `accepted_mutation_stages = 6`

Within-run gain from seed stage:

- structural: `+0.031142582733`
- form: `+0.124724039127`
- alignment: `+0.034807449619`
- total: `+0.321555728726`

Accepted operator mix:

- `macro_bundle_rewrite = 3`
- `paradigm_family_rewrite = 2`
- `suffix_family_rewrite = 1`

## Cross-Seed Ranges

Across the three seeds, the final ranges were:

- structural: `-1.3613948534656453` to `-1.3062274172668467`
- form: `0.6928220391273499` to `0.7181036472320557`
- alignment: `0.4252034121913133` to `0.4392205108144682`
- accepted stages: `5` to `6`

## Notes

- The runs did not diverge into incoherent or obviously failed outcomes within
  the first `1000` proposals.
- All three seeds improved structural score, form score, family alignment, and
  total score relative to their own seed-stage baselines.
- The dominant accepted operator family remained macro-bundle-led across the
  probe, but the exact accepted mix varied by seed.
- Seed `42` reproduced the historical first block of the paper run exactly,
  which is consistent with the earlier seed-audit result.

## Interpretation Boundary

This probe supports the narrower paper-facing claim that:

- the early-window behavior of the current `v5` French -> Latin configuration
  is not unique to a single lucky seed
- multiple seeds produce coherent, same-direction progress over the first
  `1000` proposals

This probe does **not** yet support stronger claims such as:

- full cross-seed replication of the long-horizon phase pattern
- cross-seed replication of the final plateau endpoint
- cross-seed replication of the validator-bank chronology

So this report should be used as a bounded replication note, not as a
replacement for the full production run.
