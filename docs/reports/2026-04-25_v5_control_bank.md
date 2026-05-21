# V5 Control-Bank Comparison

Date: 2026-04-25  
Status: complete

## Purpose

Record the null/control-bank comparison for the completed French -> Latin `v5`
paper run.

This pass compares each completed block endpoint from the paper run against the
project's control bank:

- Markov noise floor
- Sumerian structured non-IE control
- Portuguese withheld positive control

The comparison mirrors the post hoc validator-bank pass but keeps nulls and
controls as a separate analysis layer.

## Inputs

- Run manifest:
  [manifest.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_seed45_paper_run/manifest.json)
- Completed run report:
  [2026-04-18_v5_paper_run.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-18_v5_paper_run.md)
- Comparison tool:
  [control_bank_compare.py](/C:/Code/Project%20RBT/project_rbt/src/validation/control_bank_compare.py)

## Output Artifacts

- CSV:
  [french_v5_fortran_c16_seed45_paper_run_vs_control_bank.csv](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_fortran_c16_seed45_paper_run_vs_control_bank.csv)
- JSON:
  [french_v5_fortran_c16_seed45_paper_run_vs_control_bank.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_fortran_c16_seed45_paper_run_vs_control_bank.json)
- Chronology summary:
  [french_v5_fortran_c16_seed45_paper_run_vs_control_bank_chronology.json](/C:/Code/Project%20RBT/project_rbt/data/validation/french_v5_fortran_c16_seed45_paper_run_vs_control_bank_chronology.json)

## Scope

- controls compared: `3`
- blocks compared: `614`
- run id: `v5_fortran_c16_seed45_paper_run`

## Summary

Observed nearest-control paths:

- structural path: `vs_sumerian -> vs_portuguese_control`
- form path: `vs_portuguese_control`

First observed wins:

- first structural Sumerian win: `block_0001`
- first structural Portuguese win: `block_0300`
- first form Portuguese win: `block_0001`
- Markov wins: none on either axis

## Final Block Rankings

Final block: `block_0614`

Structural ranking at the final block:

- Portuguese: `1.129272`
- Sumerian: `2.590237`
- Markov noise: `3.524765`

Form ranking at the final block:

- Portuguese: `0.355281`
- Sumerian: `0.104529`
- Markov noise: `0.0`

## Preserved Checkpoints

### Best Form Checkpoint

Checkpoint: `block_0338`

- Portuguese structural distance: `2.726864`
- Sumerian structural distance: `2.859595`
- Markov structural distance: `6.0545`
- Portuguese form score: `0.45512`
- Sumerian form score: `0.144118`
- Markov form score: `0.0`

### Best Structural Checkpoint

Checkpoint: `block_0613`

- Portuguese structural distance: `1.129272`
- Sumerian structural distance: `2.590237`
- Markov structural distance: `3.524765`
- Portuguese form score: `0.354972`
- Sumerian form score: `0.104536`
- Markov form score: `0.0`

### Final Plateau Endpoint

Checkpoint: `block_0614`

- Portuguese structural distance: `1.129272`
- Sumerian structural distance: `2.590237`
- Markov structural distance: `3.524765`
- Portuguese form score: `0.355281`
- Sumerian form score: `0.104529`
- Markov form score: `0.0`

## Best Observed Block By Control

Best structural distance observed anywhere in the run:

- Portuguese: `0.972993` at `block_0474`
- Sumerian: `2.31845` at `block_0408`
- Markov noise: `3.524765` at `block_0613`

Best form score observed anywhere in the run:

- Portuguese: `0.553427` at `block_0002`
- Sumerian: `0.159201` at `block_0028`
- Markov noise: `0.0` at `block_0001`

## Notes

- The Portuguese comparison is post hoc and uses the sequestered positive
  control only in validation.
- The control-bank pass does not alter the completed run artifacts.
- This report is descriptive only. It records the observed control-bank
  rankings and path transitions without additional interpretation.
