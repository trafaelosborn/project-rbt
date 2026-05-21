# V5 Suffix Ablation Probe

Date: 2026-04-25  
Status: complete

## Purpose

Run a bounded French -> Latin `v5` ablation probe comparing:

- baseline Latin form scoring
- suffix-off Latin form scoring

This is a lightweight mechanism probe, not a second paper-scale production run.

## Inputs

- Start corpus:
  [french_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/processed/romance/french_tokens.json)
- Probe runner:
  [suffix_ablation_probe.py](/C:/Code/Project%20RBT/project_rbt/src/validation/suffix_ablation_probe.py)
- Output root:
  [french_v5_suffix_ablation_probe_seed42_p0500](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_suffix_ablation_probe_seed42_p0500)

## Probe Configuration

- source language: `french`
- target language: `latin`
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

## Conditions

### Baseline

- char bigram weight: `0.40`
- char trigram weight: `0.40`
- suffix weight: `0.20`

### Suffix-Off

- char bigram weight: `0.50`
- char trigram weight: `0.50`
- suffix weight: `0.00`

## Output Artifacts

- probe summary:
  [summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_suffix_ablation_probe_seed42_p0500/summary.json)
- baseline summary:
  [run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_suffix_ablation_probe_seed42_p0500/baseline/run_summary.json)
- suffix-off summary:
  [run_summary.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_suffix_ablation_probe_seed42_p0500/suffix_off/run_summary.json)

## Final Endpoint Comparison

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

- `final_total_score = +0.010063523981050954`
- `final_latin_structural_score = +0.008546830031764041`
- `final_latin_form_score = +0.038065862655639604`
- `final_family_alignment_score = -0.012745870039550633`
- `accepted_mutation_stages = 0`

## Notes

- In this bounded probe, removing suffix contribution from the Latin form score
  did not reduce accepted-stage count.
- Under this probe condition, the suffix-off variant ended with a better total
  score, a better structural score, and a better form score than the baseline.
- Family alignment was lower in the suffix-off condition at the final endpoint.
- This probe is intentionally small. It should be treated as a bounded ablation
  note rather than as a replacement for a larger replication.
