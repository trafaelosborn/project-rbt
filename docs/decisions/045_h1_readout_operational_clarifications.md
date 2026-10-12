# Decision 045: H1 E1 readout operational clarifications

Date: 2026-10-01

Status: disclosed implementation choices before corrected production runs.

Follow-up: Decision 046 withdraws the pooled-pair Mann-Whitney YES/NO rule from
the current readout. The mechanics below are retained as historical disclosure;
the new code reports a fixed-panel effect and a whole-language conditional
reference. See that decision for scope, assumptions and pending launch choice.

The frozen E1 specifies progress deciles, baseline-normalized pair distances,
AUC, a one-sided Mann-Whitney test at 0.05, and replication of direction in each
seed. It leaves several mechanical choices unspecified. These choices are
recorded here, not silently added to the frozen document. Cite this disclosure
with any corrected E1 result; it is not a claim that the original freeze
specified every detail below.

- Use the five Romance sources and four required typological controls (Finnish,
  Turkish, Vietnamese, Sumerian), each at seeds 42, 1729, 8675309. That is 27
  primary trajectories. Klingon, Markov and random-init robustness remain
  separate from the primary pair pool.
- Use the frozen six-feature structural score as S in the progress formula.
  The engine's form, coherence, mutation-cost and transition-bonus terms still
  influence acceptance, but do not redefine geometric progress.
- Work in the same standardized six-dimensional Euclidean geometry as the
  objective. At internal deciles use linear interpolation at the first
  chronological upward crossing. Use actual initial and final vectors for
  p=0 and p=1. Nonpositive net structural improvement makes progress undefined;
  report an analysis error, never omit that run or manufacture a gate verdict.
- Integrate the 11 normalized distances with the trapezoid rule. Reject pairs
  with zero starting distance.
- Within each seed, the 10 Romance-Romance pairs form the Romance group. The
  remaining 26 pairs (at least one required typological control) form the
  comparison group. Pool these AUCs across seeds for the one-sided test.
- Compute an exact label-permutation rank-sum probability, using average ranks
  for ties. Direction replication means lower mean Romance-pair AUC in each
  of the three seeds. Gate YES requires p < 0.05 and all three directions.
- Pair observations share trajectories and are statistically dependent. The
  requested Mann-Whitney calculation is implemented, but its permutation
  probability must not be presented as a calibrated independent-sample error
  guarantee. A change to a trajectory-level inferential design is a separate
  registration decision, not part of the scorer repair.
- Refuse a verdict for legacy runs, mixed objective identities, missing seeds
  or controls, edited saved corpora, or finite-budget pilots. Full runs must
  reach joint_hit or plateau_hit in unbounded mode.

This implements the primary E1 readout. It does not complete E2-E4 or the
registered random-init/Markov robustness requirements. Those remain separate
scientific work; primary E1 output must explicitly retain that limitation.
