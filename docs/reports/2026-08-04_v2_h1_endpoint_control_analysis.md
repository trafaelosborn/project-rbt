# V2 H1 endpoint-to-control analysis

## Purpose

This read-only post-run analysis extends the first paper's held-out control-bank
method to all 15 completed production endpoints. It does not optimize controls.

## Method

- Compared each engine-sampled start and final corpus for five Romance sources
  and three seeds.
- Used a common 9,000-token budget, the largest round budget supported by all
  30 generated states (minimum available: Romanian seed 1729 start, 9,084).
- Used the frozen v2 corpus manifest and feature definitions.
- Estimated feature means and standard deviations from the 22 frozen attested
  reference rows only; generated states did not affect scaling.
- Reported standardized cosine and Euclidean distances in the six retained H1
  features and all 21 registered features.
- Scored character bigram, trigram, and suffix form similarity separately.
- Held-out references: Latin, Markov noise, Finnish, Turkish, Vietnamese,
  Sumerian, and optional Klingon.

## Primary results

1. All 15 final corpora are nearest to Klingon among the five non-Latin controls
   in both 6D and 21D, under both cosine and Euclidean distance.
2. All 15 move closer to Klingon in both 6D and 21D.
3. Only 6/15 move closer to Latin in 6D; only 2/15 move closer in 21D.
4. All 15 improve in Latin form similarity, while none improve in form
   similarity to Finnish, Turkish, Vietnamese, Sumerian, or Klingon.
5. The mean reference-centroid radius grows from 1.123 to 12.079 in 6D and from
   1.821 to 13.548 in 21D. The endpoints leave the attested reference manifold.
6. The largest standardized shifts are collapsing suffix entropy (-11.386 SD)
   and conditional character entropy (-4.435 SD). These dominate the displacement.

## Mean Latin movement by source

Negative distance delta means movement toward Latin.

| Source | 6D start | 6D final | 6D delta | 21D start | 21D final | 21D delta | Form delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| French | 1.3478 | 1.1132 | -0.2346 | 1.3022 | 1.4072 | +0.1050 | +0.2117 |
| Italian | 0.3964 | 1.0733 | +0.6769 | 0.3968 | 1.3347 | +0.9379 | +0.2143 |
| Portuguese | 0.4714 | 1.0956 | +0.6242 | 0.4227 | 1.3736 | +0.9509 | +0.1611 |
| Romanian | 0.6217 | 1.1129 | +0.4912 | 0.6946 | 1.3919 | +0.6973 | +0.2485 |
| Spanish | 1.5196 | 1.0987 | -0.4209 | 1.3654 | 1.3512 | -0.0142 | +0.2315 |

## Production-path audit

Despite every manifest carrying the v2 H1 feature-freeze commit and hash, the
production engine evaluates candidates through
`src.retrodiction.engine_reinforced_v2._evaluate_sequences`, which calls the
legacy four-component `structural_vector`. The Latin structural reward uses
only type-token ratio, top-100 bigram coverage, and top-100 trigram coverage;
mean sequence length is recorded but excluded from reward. The six retained H1
features are not used by the acceptance objective.

Therefore this was not an active high-dimensional optimization run. It was a
legacy 3D structural-reward run with v2 freeze metadata attached. The endpoint
control result explains the apparent contradiction: the old reward approaches
its ceiling while the corpora move far outside the frozen 6D/21D attested
geometry.

## Interpretation

The strongest defensible conclusion is a negative instrument-validation result.
The optimizer reliably increases Latin surface-form similarity and its legacy
three-feature reward, but does not reproduce Latin proximity in the registered
high-dimensional feature spaces. Instead, it generates an extreme low-entropy
region whose nearest available attested stress control is Klingon. This is not
evidence of a Klingon relationship; it is evidence of optimizer-induced
distributional collapse.

The preregistered H1 convergence claim cannot be tested from this matrix as a
high-dimensional active-optimization experiment. A corrected run must wire the
six retained features into candidate scoring and acceptance, then repeat the
Romance trajectories before the registered control/AUC analysis is meaningful.

## Artifacts

- `data/validation/v2_h1_endpoint_controls_20260804/features.csv`
- `data/validation/v2_h1_endpoint_controls_20260804/distances.csv`
- `data/validation/v2_h1_endpoint_controls_20260804/movement.csv`
- `data/validation/v2_h1_endpoint_controls_20260804/nearest_controls.csv`
- `data/validation/v2_h1_endpoint_controls_20260804/metadata.json`
- `src/validation/h1_endpoint_control_analysis.py`
