# Historical Validator Drop Zone

Place attested historical text files here when preparing validator corpora.

Recommended layout:

- `data/raw/historical/old_french/*.txt`
- `data/raw/historical/middle_french/*.txt`
- `data/raw/historical/anglo_norman/*.txt`
- `data/raw/historical/langue_d_oil/*.txt`
- `data/raw/historical/old_spanish/*.txt`
- `data/raw/historical/old_occitan/*.txt`

Then ingest with:

```powershell
python -m src.ingest.historical --name old_french --language french --period "Old French"
```

This writes:

- `data/processed/historical/old_french_tokens.json`
- `data/processed/historical/old_french_manifest.json`
- `data/matrices/old_french_*`

Then compare against a bridge run with:

```powershell
python -m src.validation.checkpoint_compare `
  --run-summary data/retrodiction/french/v2_convergence/run_summary.json `
  --validator data/processed/historical/old_french_tokens.json
```

Design rules for the validator bank:

- attested corpora only
- one corpus family per directory
- no mixed historical buckets
- chronology and geography are interpretation layers added after scoring
- keep source / date / region notes in each corpus README and in `validator_bank_manifest.csv`

Suggested workflow for the new bank:

1. Drop raw attested texts into one of the directories listed above.
2. Update `validator_bank_manifest.csv` with provenance and labeling metadata.
3. Ingest each corpus independently with `src.ingest.historical`.
4. Compare the same bridge checkpoints against every processed validator separately.
