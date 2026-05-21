# Attested Text Candidates

Date: 2026-04-08

## Purpose

This note records attested French-side historical text sources that look usable for
the new validator-ingestion workflow.

The goal is not to settle the perfect corpus yet. The goal is to identify sources
that are:

1. attested
2. public or openly accessible
3. usable as local text input for `src/ingest/historical.py`

## Recommended first target

### Oxford Text Archive: Old French corpus

Best quick-start source for immediate ingestion.

- Host: Oxford Text Archive
- Record: `ota:0176`
- Coverage: Old French (`842-ca. 1400`)
- Format: plain text (`ofrcorpus176.txt`)
- Size: about `82.63 KB`
- URL: `https://llds.phon.ox.ac.uk/llds/xmlui/handle/20.500.14106/0176`

Why this is the best immediate fit:

- already bundled as plain text
- explicitly described as an Old French corpus
- contains multiple early texts rather than a single witness
- low-friction path into `data/raw/historical/old_french/`

The preview shows it includes items such as:

- `EULALIA`
- `STRASBOURG OATHS`
- `AUBE BILINGUE`
- `Boeci`
- Alberic's `Alexandre`

## Best scholarly corpus

### Base de Francais Medieval (BFM)

Best source if we want a serious long-run validator layer.

- Host: ENS de Lyon / IHRIM
- Corpus size: hundreds of Old and Middle French texts
- Chronology: from the `9th century` to the end of the `15th century`
- Access model: searchable online; PDF downloads available; TEI files available on request
- URLs:
  - `https://www.ens-lyon.fr/actualite/recherche/publication-du-corpus-bfm2022-de-la-base-de-francais-medieval`
  - `https://ihrim.ens-lyon.fr/productions-scientifiques/ressources-numeriques/article/base-de-francais-medieval-bfm`
  - `https://tei-c.org/activities/projects/base-de-francais-medieval-old-french-corpus/`

Why it matters:

- this is the most obviously research-grade source in the set
- it includes the exact kind of attested historical material we care about
- it can support later scaling beyond a one-off validation pass

Why it is not the fastest first step:

- not as frictionless as a ready plain-text corpus
- TEI access appears to require a request path

## Strong single-text candidates

These are good if we want a smaller, manually curated validator packet.

### Serments de Strasbourg

- Date: `842`
- Value: extremely early Romance witness; historically central
- Good use: anchor text, not sufficient alone as a corpus
- URL: `https://fr.wikisource.org/wiki/Serments_de_Strasbourg`

### Sequence de sainte Eulalie

- Date: about `880-881`
- Value: one of the earliest literary texts in Old French / early Romance
- Good use: early-stage validator supplement
- URL: `https://fr.wikisource.org/wiki/S%C3%A9quence_de_sainte_Eulalie`

### La Vie de saint Alexis

- Date: `11th century` witness on Wikisource
- Value: substantially larger than the very earliest fragments
- Good use: first serious single-text validator if we want to start manually
- URL: `https://fr.wikisource.org/wiki/La_Vie_de_saint_Alexis`

### La Chanson de Roland

- Date: `11th / early 12th century` manuscript tradition
- Value: larger text, famous, openly accessible transcription paths
- Good use: later validator or genre-diversity addition
- URL: `https://fr.wikisource.org/wiki/Livre%3ALa_Chanson_de_Roland_-_MS_Oxford.djvu`

## Recommendation

If the goal is to move fastest:

1. start with the Oxford Text Archive Old French corpus
2. ingest it as `old_french`
3. run checkpoint comparison against the French `v2_convergence` ladder

If the goal is to build the strongest long-term validator layer:

1. use BFM as the primary target corpus program
2. use individual texts like `Strasbourg`, `Eulalie`, and `Saint Alexis` as sanity-check validators

## Next action

The shortest path to a real historical comparison is:

1. obtain the Oxford plain-text corpus locally
2. place it under `data/raw/historical/old_french/`
3. run:

```powershell
python -m src.ingest.historical --name old_french --language french --period "Old French"
python -m src.validation.checkpoint_compare `
  --run-summary project_rbt/data/retrodiction/french/v2_convergence/run_summary.json `
  --validator project_rbt/data/processed/historical/old_french_tokens.json
```
