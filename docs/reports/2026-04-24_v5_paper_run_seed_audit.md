# V5 Paper Run Seed Audit

Date: 2026-04-24  
Status: resolved

## Purpose

Resolve the seed mismatch in the completed French -> Latin `v5` paper run.

The historical artifacts disagreed:

- [manifest.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/manifest.json) records `config.seed = 45`
- per-block [run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/blocks/block_0001/run_summary.json) files serialize `config.seed = 42`

This audit determines which seed actually drove the run.

## Method

The audit used exact replay of `block_0001` from the clean French start corpus
under the current codebase with two candidate seeds:

- replay seed `42`
- replay seed `45`

For each replay, the audit compared the resulting `block_0001`
accepted-stage history against the historical paper-run `block_0001`
artifact:

- final structural score
- final form score
- final alignment score
- accepted mutation stage count
- full accepted-stage sequence
- proposal indices
- mutation operators
- mutation details

Replay artifacts:

- [seed_42/run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/_seed_audit_replay/seed_42/run_summary.json)
- [seed_45/run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/_seed_audit_replay/seed_45/run_summary.json)

Structured audit artifact:

- [seed_audit.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/seed_audit.json)

## Result

The historical `block_0001` artifact matches the `seed = 42` replay exactly and
does not match the `seed = 45` replay.

Exact-match result:

- historical block matches replay `42`: `true`
- historical block matches replay `45`: `false`

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

The accepted-stage sequence also matches exactly under `seed = 42`, including:

- accepted proposal indices: `1, 11, 73, 139, 959`
- operator sequence: five `macro_bundle_rewrite` accepts
- stage mutation details for every accepted stage

## Conclusion

For the completed paper run:

- requested / intended seed: `45`
- actual engine seed used by the search: `42`

So the mismatch is resolved in favor of the block summaries and replay
evidence, not the manifest launch field.

## Interpretation for Reproducibility

The most likely historical explanation is:

- the launch-facing configuration path preserved the requested seed `45`
- the engine-facing path for the actual paper run executed with seed `42`

The current codebase no longer leaves this ambiguous. The long-run driver now
records a `seed_audit` trail with:

- `requested_seed`
- `engine_seed`
- `seeds_match`

for future runs.

## Paper-Facing Recommendation

For manuscript and report purposes, cite the paper run as:

- run id: `v5_fortran_c16_seed45_paper_run`
- requested launch seed: `45`
- audited engine seed: `42`

Do not describe the run as a clean single-seed-45 condition without this
qualification.
