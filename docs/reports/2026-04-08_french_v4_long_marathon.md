# French v4 Long Marathon

Date: 2026-04-08
Status: complete

## Purpose

Run a long smooth `v4` continuation from the best partial late-stage French bridge and let it keep climbing toward the Latin-conditioned endpoint under a large proposal budget.

This is a local compute marathon, not an API-backed inference job.

## Run Configuration

- Engine: `src.retrodiction.long_run_v4`
- Language: `french`
- Starting corpus: `data/retrodiction/french/v4_1500_proposal_continuation/corpora/FR_v4_011_tokens.json`
- Starting proposal count: `201`
- Target total proposals: `1500`
- Block size: `200`
- Search mode: smooth `v4` continuation
- Stop conditions:
  - Latin hit: `latin_structural_score >= 0.0` and `latin_form_score >= 1.0`
  - Otherwise continue until the full proposal budget is exhausted

## Starting Baseline

Seed stage: `FR_v4_011`

- `latin_structural_score = -1.260193`
- `latin_form_score = 0.816971`
- `family_alignment_score = 0.536172`
- `coherence_label = coherent`

This seed already improved beyond the earlier smooth endpoint and remained historically legible on the Old French structural validator.

## Output Locations

- Marathon root: `data/retrodiction/french/v4_long_1500_local/`
- Block summaries: `data/retrodiction/french/v4_long_1500_local/blocks/block_*/run_summary.json`
- Manifest: `data/retrodiction/french/v4_long_1500_local/manifest.json`
- Stdout log: `data/retrodiction/french/v4_long_1500_local/stdout.log`
- Stderr log: `data/retrodiction/french/v4_long_1500_local/stderr.log`

## Notes

- The first attempt to detach this run from the sandbox failed because sandboxed background children were reaped after the shell returned.
- The resumable runner was then verified and prepared for an unrestricted local background launch.
- Progress should be recorded block by block through the manifest and block summaries rather than inferred from a single final summary.

## Live Checkpoint

As of the first post-launch inspection, the detached process was actively writing artifacts under `blocks/block_0001/`.

Observed early accepted stages:

- `FR_v4_001`: `total_score = -0.4861`, `latin_structural_score = -1.2602`, `latin_form_score = 0.8166`, `family_alignment_score = 0.5346`
- `FR_v4_002`: `total_score = -0.4713`, `latin_structural_score = -1.2597`, `latin_form_score = 0.8168`, `family_alignment_score = 0.5346`
- `FR_v4_003`: `total_score = -0.4459`, `latin_structural_score = -1.2593`, `latin_form_score = 0.8181`, `family_alignment_score = 0.5457`

That means the marathon was improving immediately within the first block rather than stalling at launch.

## Final Outcome

The marathon completed on 2026-04-09 without hitting the exact Latin stop target.

Run completion summary:

- cumulative proposals: `1500`
- new proposals beyond the seed lineage: `1299`
- total completed blocks: `7`
- final block halt: `max_proposals`
- final coherence label: `coherent`

Final endpoint:

- corpus: `data/retrodiction/french/v4_long_1500_local/blocks/block_0007/corpora/FR_v4_002_tokens.json`
- summary: `data/retrodiction/french/v4_long_1500_local/blocks/block_0007/run_summary.json`
- `latin_structural_score = -1.193785`
- `latin_form_score = 0.827672`
- `family_alignment_score = 0.563688`

Relative to the marathon seed `FR_v4_011`:

- structural improved from `-1.260193` to `-1.193785`
- form improved from `0.816971` to `0.827672`
- family alignment improved from `0.536172` to `0.563688`

Best observed within the marathon:

- strongest structural endpoint: final `FR_v4_002` in block 7
- strongest family alignment: final `FR_v4_002` in block 7
- strongest form score: `FR_v4_005` in block 5 at `0.829709`

Interpretation:

The long continuation was worth running. It produced substantial additional Latin-directed movement while staying coherent, but it still converged on a stronger synthetic late-stage bridge rather than literal Latin.
