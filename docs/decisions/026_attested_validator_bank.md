# 026 - Attested Validator Bank

**Date:** 2026-04-10  
**Status:** Implemented

## Decision

Historical validation corpora are organized as a bank of separate attested
layers rather than a single mixed historical corpus.

Initial scaffolded validator families:

1. `old_french`
2. `middle_french`
3. `anglo_norman`
4. `langue_d_oil`
5. `old_spanish`
6. `old_occitan`

## Rationale

The project wants to know whether a synthetic bridge converges incidentally on
specific attested intermediates, not whether it resembles a hand-blended
"historical soup."

Keeping validators separate:

- preserves chronology and branch identity
- reduces circularity
- allows post hoc reconstruction of geography / chronology
- makes cross-ladder jumps observable instead of hidden by aggregation

## Constraints

- attested corpora only
- one validator corpus per directory / ingest target
- chronology and geography stay out of scoring inputs
- interpretation happens after checkpoint-vs-validator scoring

## Operational Notes

The raw drop zone is documented in:

- `data/raw/historical/README.md`
- `data/raw/historical/validator_bank_manifest.csv`

Each validator family gets its own raw directory and independent ingest via
`src.ingest.historical`.
