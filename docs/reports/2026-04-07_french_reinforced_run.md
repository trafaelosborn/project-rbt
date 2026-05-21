# French Reinforced Run - 2026-04-07

Superseded by `2026-04-08_french_reinforced_rerun.md`.

This report is retained as an archive of the pre-fix run, when the Latin reward still
included `log_mean_seq_len`. That reward design was later revised because the
generator preserves source sentence-length distributions and cannot optimize that
dimension directly.

## Scope

Primary Phase 3R case study: French -> Latin under the reinforced single-blind
protocol.

Algorithms executed:

1. `stochastic`
2. `gradient`

Command path:

- `python pipeline.py --step reinforced`

Output directories:

- `data/retrodiction/french/stochastic/`
- `data/retrodiction/french/gradient/`

## Configuration

Both runs used the default `ReinforcedConfig`:

- `num_sequences = 2000`
- `max_iterations = 200`
- `stability_threshold = 0.002`
- `seed = 42`
- `n_candidates = 20`
- `noise_scale = 0.3`
- `alpha = 0.05`

Latin reference loaded during the run:

- `TTR = 0.1290`
- `bg_cov = 0.1620`
- `tg_cov = 0.1244`
- `log_mean_seq_len = 2.2258`

## Results

| Algorithm | Stages | First score | Final score | Delta | Best stage |
|---|---:|---:|---:|---:|---|
| stochastic | 2 | -0.374036 | -0.373255 | +0.000781 | `FR_stoch_001` |
| gradient | 9 | -0.369502 | -0.411549 | -0.042047 | `FR_grad_000` |

Structural changes:

- `stochastic`
  - `bigram_coverage`: `0.179592 -> 0.175017`
  - `trigram_coverage`: `0.085174 -> 0.082994`
  - `TTR`: `0.165713 -> 0.166277`
- `gradient`
  - `bigram_coverage`: `0.180965 -> 0.194743`
  - `trigram_coverage`: `0.083293 -> 0.107979`
  - `TTR`: `0.166066 -> 0.157695`

## Interpretation

The reinforced French run did not produce a simple story of monotonic convergence.

The stochastic search found a small improvement toward the Latin structural vector and
then stabilized almost immediately. The directed gradient path, despite having direct
access to the Latin bigram transition target, moved *away* from the Latin structural
reference under the current Euclidean structural score.

That matters methodologically:

1. The bridge is not trivial under the current representation.
2. "Move the transition matrix toward Latin" is not equivalent to "move the generated
   corpus toward Latin" in structural feature space.
3. The path geometry differs sharply by algorithm, which is exactly what this phase is
   meant to expose.

## What this does and does not show

What it shows:

- The reinforced protocol runs end-to-end.
- The two algorithms produce measurably different bridge behavior.
- The current structural score is sensitive enough to separate those behaviors.

What it does not yet show:

- Whether either bridge aligns with attested intermediates.
- Whether either bridge is readable as a coherent historical register by expert humans.
- Whether the same pattern generalizes beyond French.

Attested-stage comparison remains pending because no historical intermediate corpora are
yet ingested in the repository.
