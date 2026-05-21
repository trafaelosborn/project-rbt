# Paper Run Checklist

Date: 2026-04-18  
Status: active

## Purpose

Freeze the exact conditions of the current French -> Latin `v5` paper run while
it is live, so the final report can be written from preserved artifacts rather
than reconstructed from memory.

## Active Run

- Run directory:
  [v5_fortran_c16_clean_restart](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_clean_restart)
- Manifest:
  [manifest.json](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_clean_restart/manifest.json)

## Freeze Conditions

- Treat this run as the paper run unless explicitly retired.
- Do not change methodology mid-run.
- Do not reuse the output directory for any other experiment.
- Do not overwrite the starting corpus.
- Do not change the engine code, toggles, or candidate count and keep the same
  run name.

If architecture changes become necessary, start a new run in a new output
directory and document the break clearly.

## Launch State To Preserve

- Source language: `french`
- Target language: `latin`
- Start corpus:
  [french_tokens.json](/C:/Code/Project%20RBT/project_rbt/data/processed/romance/french_tokens.json)
- Output directory:
  [v5_fortran_c16_clean_restart](/C:/Code/Project%20RBT/project_rbt/data/retrodiction/french/v5_fortran_c16_clean_restart)
- Seed: `45`
- Block proposals: `1000`
- Candidate count: `16`
- `use_incremental_scoring = true`
- `use_fortran_batch = true`
- `use_fortran_cosine = true`
- `use_semantic_transparency = false`
- `enable_culture_bombs = false`
- `total_target_proposals = 0` (indefinite / halt by joint rule or operator stop)

## Paper-Run Artifact Checklist

- Keep the live manifest intact.
- Preserve the final manifest at stop time.
- Preserve the final corpus JSON.
- Preserve the final preview text.
- Preserve the block summary containing the stop boundary.
- Preserve benchmark context against the frozen `v4` baseline.
- Preserve any validator-bank outputs generated from the final corpus.
- Preserve the exact code snapshot used for the run.
- Preserve environment notes for Python, compiler, and Fortran bridge setup.

## During-Run Logging

Capture these in the final report while the run is active:

- wall-clock start time
- wall-clock stop time
- total elapsed runtime
- cumulative proposals
- completed blocks
- proposals/hour over full run
- best observed structural score
- best observed form score
- best observed family alignment score
- first block where structural entered a near-zero band, if reached
- halt reason

## Stop Protocol

When the run ends or is manually stopped:

1. Stop only at a clean block boundary.
2. Record the stop reason in the report.
3. Link the final manifest, final corpus, and final preview.
4. Record final values for:
   - `latin_structural_score`
   - `latin_form_score`
   - `family_alignment_score`
5. Record whether halt was due to:
   - joint hit
   - plateau hit
   - manual operator stop
   - error

## Follow-On Reporting

After the run stops, complete:

- [2026-04-18_v5_paper_run.md](/C:/Code/Project%20RBT/project_rbt/docs/reports/2026-04-18_v5_paper_run.md)

If validator-bank comparison is run afterward, add links to the generated CSV
and JSON outputs there rather than in this checklist.
