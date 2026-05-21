# French v4 15000-Proposal Extension

Date: 2026-04-09
Status: running

## Purpose

Continue the completed `1500`-proposal French `v4` marathon from its final
best endpoint and extend the total search budget to `15000` proposals.

This is a direct follow-up to:

- `docs/reports/2026-04-08_french_v4_long_marathon.md`

## Launch Configuration

- Engine: `src.retrodiction.long_run_v4`
- Language: `french`
- Starting corpus:
  `data/retrodiction/french/v4_long_1500_local/blocks/block_0007/corpora/FR_v4_002_tokens.json`
- Starting proposal count: `1500`
- Target total proposals: `15000`
- Block size: `200`
- Search mode: smooth `v4` continuation

## Seed State

Seed stage: `FR_v4_002` from the completed marathon.

- `latin_structural_score = -1.193785`
- `latin_form_score = 0.827672`
- `family_alignment_score = 0.563688`
- `coherence_label = coherent`

## Planned Output Location

- `data/retrodiction/french/v4_long_15000_local/`

## Notes

- This run should be interpreted as an extension of the same smooth lineage, not
  as a fresh independent search.
- The `1500`-proposal marathon already ended on budget rather than on obvious
  exhaustion, which is why the larger continuation is justified.
- The detached local process is actively writing `block_0001` artifacts under
  `data/retrodiction/french/v4_long_15000_local/`.
