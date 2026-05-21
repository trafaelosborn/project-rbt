# French Reinforced V2 Run - 2026-04-08

## Scope

First full run of the relational reinforced bridge engine:

1. source language: French
2. target/reinforcer: Latin
3. engine: `src/retrodiction/engine_reinforced_v2.py`

Command path:

- `python pipeline.py --step reinforced_v2 --force`

Output directory:

- `data/retrodiction/french/v2/`

## What changed relative to v1

The original reinforced engines mutate a fixed French-vocabulary transition model.
The v2 engine instead mutates the sampled corpus directly and lets accepted stages
become the new baseline.

Active mutation operators in this run:

1. `token_char_edit`
2. `suffix_family_rewrite`
3. `swap_bigram_order`
4. `split_token`
5. `merge_bigram`

Scoring combines:

1. Latin structural reward
2. Latin form reward from character n-grams and suffix profiles
3. coherence margin against the Markov floor
4. small mutation-cost penalty

## Configuration

Default `ReinforcedV2Config`:

- `num_sequences = 800`
- `max_proposals = 80`
- `max_accepted_stages = 18`
- `patience = 8`
- `seed = 42`
- `n_candidates = 6`
- `min_improvement = 0.001`
- `token_edit_attempts = 6`
- `suffix_candidate_samples = 8`
- `form_weight = 0.75`
- `coherence_weight = 0.05`
- `mutation_cost_weight = 0.005`

## Results

Run summary:

- total stages: `18`
- accepted mutation stages: `17`
- proposals attempted: `26`
- halt reason: `max_accepted_stages`
- final / best stage: `FR_v2_017`

Score movement:

- total score: `-0.814688 -> -0.699032`
- Latin structural score: `-1.378939 -> -1.371562`
- Latin form score: `0.566157 -> 0.715272`
- coherence label: `coherent -> coherent`

Accepted operator counts:

- `suffix_family_rewrite = 9`
- `token_char_edit = 7`
- `swap_bigram_order = 1`

No `split_token` or `merge_bigram` stages survived acceptance in this first French
run.

## Interpretation

The important result is that the v2 engine no longer stalls at the seed stage. It
constructs an actual bridge path with repeated accepted mutations while preserving
language-likeness under the current coherence diagnostic.

Movement in this run is driven mostly by form change rather than by large structural
reorganization:

1. the Latin form score rises substantially
2. the structural score improves modestly
3. the coherence label stays `coherent` throughout

This is a better fit for the current research question than the fixed-vocabulary
engines. The resulting corpus is not Latin, but it is also no longer simply French
with reweighted transitions. It begins to produce synthetic, path-like forms such as
`janvirum`, `descriptibus`, `paisaium`, `lieium`, and `spectactum`.

## Readable outputs

Best stage summary:

- `data/retrodiction/french/v2/run_summary.json`

Best stage preview:

- `data/retrodiction/french/v2/previews/FR_v2_017_preview.txt`

Best stage full corpus:

- `data/retrodiction/french/v2/corpora/FR_v2_017_tokens.json`

## Next questions

1. Compare v2 checkpoints against attested intermediate corpora once ingested.
2. Add source-overlap and vocabulary-churn diagnostics so bridge drift is easier to
   quantify.
3. Extend the operator set to model richer relational effects, including latent or
   family-linked changes that may not pay off immediately.
