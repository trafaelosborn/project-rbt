# Decision: Local historical validator ingestion and checkpoint comparison layer

Date: 2026-04-08

## Context

Project RBT now produces readable reinforced bridge corpora and stable convergence
paths, but the critical historical validation step remained blocked because no
attested intermediate-corpus workflow existed inside the repo.

This created an awkward gap:

- bridge generation was implemented,
- Latin-conditioned endpoint scoring was implemented,
- but attested-stage comparison still required ad hoc external handling.

The next phase needs a minimal, repeatable in-repo validator workflow.

## Decision

Add a local-file historical validator layer with two parts:

1. `src/ingest/historical.py`
   Ingest attested historical text already present on disk from
   `data/raw/historical/{name}/`, tokenize it with the same tokenizer used for the
   modern corpora, and fingerprint it into the standard matrix / n-gram files.

2. `src/validation/checkpoint_compare.py`
   Compare selected bridge checkpoints against a historical validator corpus using:
   - structural cosine
   - scaled structural distance
   - character / suffix form similarity

The default comparison ladder is:

1. start
2. quarter point
3. midpoint
4. three-quarter point
5. late-tail checkpoint
6. final endpoint

For the current French `v2_convergence` run this yields:

1. `FR_v2_000`
2. `FR_v2_015`
3. `FR_v2_030`
4. `FR_v2_045`
5. `FR_v2_058`
6. `FR_v2_061`

## Rationale

This keeps the validator layer:

- local and reproducible,
- independent of network access,
- consistent with the repo's existing tokenizer and fingerprint stack,
- usable for both attested matches and coherent unattested "ghost path" outcomes.

It also avoids overcommitting to a single historical source before the workflow is
proven. Any suitable attested text collection can be dropped into the raw historical
directory and ingested.

## Consequences

Positive:

- Historical validation is now a first-class repo workflow rather than a future note.
- French bridge checkpoints can be compared as soon as a corpus is dropped in.
- The workflow can generalize to Old Spanish, Old Italian, or other Romance validators.

Limitations:

- No attested validator corpus is bundled yet.
- The new layer supports local ingestion, not network harvesting.
- Endpoint-vs-Latin remains in-loop scoring; this layer supplements that rather than replacing it.
