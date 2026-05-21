# French Reinforced Rerun - 2026-04-08

## Scope

Reward-fixed rerun of the primary Phase 3R case study: French -> Latin under the
reinforced single-blind protocol.

Algorithms executed:

1. `stochastic`
2. `gradient`

Command path:

- `python pipeline.py --force --step reinforced`

Output directories:

- `data/retrodiction/french/stochastic/`
- `data/retrodiction/french/gradient/`

## Why the rerun was necessary

The original 2026-04-07 run scored candidates against all four dimensions of the
structural vector:

1. `type_token_ratio`
2. `bigram_coverage`
3. `trigram_coverage`
4. `log_mean_seq_len`

That fourth feature turned out to be untrainable in the reinforced generators. The
sampling engine preserves the source corpus sentence-length distribution, so
`log_mean_seq_len` can drift only through sampling noise, not directed optimization.

The Latin reward was therefore revised to use only the trainable subspace:

- `type_token_ratio`
- `bigram_coverage`
- `trigram_coverage`

Reward scale for the rerun:

- `latin_reward_score_scale = 5.0`

## Configuration

Both reruns used the default `ReinforcedConfig`:

- `num_sequences = 2000`
- `max_iterations = 200`
- `stability_threshold = 0.002`
- `seed = 42`
- `n_candidates = 20`
- `noise_scale = 0.3`
- `alpha = 0.05`

Latin reward features loaded during the run:

- `TTR = 0.1290`
- `bg_cov = 0.1620`
- `tg_cov = 0.1244`

Additional gradient diagnostic:

- `latin_vocab_overlap_ratio = 0.0302`

Only `3.02%` of capped French source-vocabulary tokens overlap with Latin tokens, so
the directed gradient engine should be interpreted as heuristic guidance rather than a
literal token-aligned path.

Null/coherence diagnostics added in this rerun pass:

- `vs_markov_noise`
- `vs_sumerian`
- `language_likeness_margin`
- `coherence_label`

## Results

| Algorithm | Stages | First score | Final score | Delta | Best stage |
|---|---:|---:|---:|---:|---|
| stochastic | 10 | -0.239143 | -0.144710 | +0.094433 | `FR_stoch_007` |
| gradient | 6 | -0.292680 | -0.242540 | +0.050140 | `FR_grad_004` |

Coherence / null diagnostics:

| Algorithm | Final coherence | Margin range | `vs_markov_noise` range | `vs_sumerian` range |
|---|---|---:|---:|---:|
| stochastic | `coherent` | `2.117820 -> 2.606176` | `0.996195 -> 0.996947` | `0.998871 -> 0.999229` |
| gradient | `coherent` | `1.966161 -> 2.283905` | `0.996356 -> 0.996798` | `0.998987 -> 0.999192` |

The raw cosine null scores remain highly compressed near `1.0`, so they should be read
as continuity diagnostics rather than as the primary "junk detector." The useful
separator in this pass is the positive language-likeness margin against the Markov
floor.

Structural changes:

- `stochastic`
  - `bigram_coverage`: `0.180513 -> 0.175266`
  - `trigram_coverage`: `0.092641 -> 0.118262`
  - `TTR`: `0.159557 -> 0.153960`
- `gradient`
  - `bigram_coverage`: `0.180965 -> 0.188361`
  - `trigram_coverage`: `0.083293 -> 0.095174`
  - `TTR`: `0.166066 -> 0.157317`

## Interpretation

The rerun changes the story substantially.

Under the reward-fixed objective, both algorithms now move French closer to the Latin
reference under the active score. The original apparent gradient failure was caused by
an objective mismatch, not by a clean failure of the reinforced protocol itself.

The two algorithms still trace meaningfully different paths:

1. `stochastic` finds the stronger best stage and reaches much higher trigram coverage.
2. `gradient` improves more conservatively and appears constrained by very low token
   overlap with Latin.
3. The divergence between those paths remains useful evidence that bridge geometry is
   nontrivial under the current representation.
4. Both paths remain structurally `coherent` throughout the run rather than collapsing
   toward the Markov floor.

## What this now shows

What it shows:

- The reinforced protocol runs end-to-end under a trainable Latin reward.
- Both algorithms can improve toward Latin in the reward subspace.
- The stochastic and gradient paths remain meaningfully different rather than
  collapsing to the same bridge.
- The low-overlap warning is now explicit, which makes the limitations of the
  deterministic gradient route much easier to interpret.
- The current French bridges are not noise-like under the active coherence diagnostic.

What it still does not show:

- Whether either bridge aligns with attested intermediates.
- Whether either bridge is readable as a coherent historical register by expert humans.
- Whether these path shapes generalize beyond French.

## Next questions

The next methodological layer is unchanged:

1. Compare bridge checkpoints against attested intermediate corpora once those corpora
   are ingested.
2. Validate the new coherence diagnostic against expert judgment and future attested
   intermediate corpora.
3. Repeat the reward-fixed reinforced run for additional Romance inputs and compare
   path convergence across languages.
