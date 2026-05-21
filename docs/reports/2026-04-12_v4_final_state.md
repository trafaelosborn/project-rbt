# V4 Final State

Date: 2026-04-12  
Status: complete

## Purpose

Freeze the long-running French -> Latin `v4` continuation after reaching
near-zero structural convergence, preserve the final baseline artifacts, and
close the run before beginning `v5` engine work.

## Run Frozen

- Run manifest:
  [manifest.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v4_until_plateau_from_30k/manifest.json)
- Final corpus:
  [FR_v4_001_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v4_until_plateau_from_30k/blocks/block_0253/corpora/FR_v4_001_tokens.json)
- Final preview:
  [FR_v4_001_preview.txt](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v4_until_plateau_from_30k/blocks/block_0253/previews/FR_v4_001_preview.txt)

The run was stopped manually on the next clean block boundary after the
structural axis entered a near-zero convergence band.

## Final Scores

Final block: `block_0253`

- `latin_structural_score = -0.000037`
- `latin_form_score = 0.755795`
- `family_alignment_score = 0.547835`
- `accepted_mutation_stages = 1`

## Total Search

- Starting proposal count: `30000`
- Completed continuation proposals: `253000`
- Final cumulative proposals: `283000`
- Final block count: `253`
- Block size: `1000`

## Structural Convergence

No exact structural zero crossing occurred before the manual stop.

Closest structural approach:

- block: `block_0253`
- cumulative proposals: `283000`
- structural score: `-0.0000372424255784854`

For ordinary interpretation this should be treated as near-zero structural
convergence, not as an exact mathematical zero hit.

## Halt Reason

Run halt mode: manual stop at block boundary.

Recorded manifest halt reason:

- `Near-zero structural convergence reached; run stopped manually to begin V5 refinement.`

This should be described as an operator decision, not as a native engine halt.
The engine itself had not triggered either `joint_hit` or `plateau_hit`.

## Final Validator Snapshot

Frozen validator-bank outputs:

- [french_v4_until_plateau_from_30k_final_vs_validator_bank.csv](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v4_until_plateau_from_30k_final_vs_validator_bank.csv)
- [french_v4_until_plateau_from_30k_final_vs_validator_bank.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v4_until_plateau_from_30k_final_vs_validator_bank.json)
- [french_v4_until_plateau_from_30k_final_vs_validator_bank_chronology.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v4_until_plateau_from_30k_final_vs_validator_bank_chronology.json)

Final validator-bank summary:

- validator count: `6`
- structural nearest-neighbor path: `old_occitan`
- form nearest-neighbor path: `middle_french`

So the frozen v4 endpoint remains best described as:

- structurally closest to `old_occitan`
- formally closest to `middle_french`

## Interpretation

This run is a successful `v4` baseline.

The important outcome is not that the engine hit exact zero on the structural
axis, but that it drove the structural fingerprint to an effectively resolved
near-zero target while preserving coherent movement on the other axes over a
very long search horizon.

That makes this run an appropriate stopping point for:

- baseline reporting
- validator-bank comparison
- `v5` engine design and implementation
