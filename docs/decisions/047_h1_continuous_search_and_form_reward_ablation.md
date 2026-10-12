# Decision 047: continuous H1 search and form-reward ablation

Date: 2026-10-01. Authorized by the author's instruction to implement the
parameter review. This is a new fixed-budget experimental extension, not a
retroactive revision of Decision 046 or admission of earlier pilots as data.

## Question and scope

For the selected source corpora, how does disabling explicit Latin-form rewards
change convergence, diversity, and Latin structural distance after an equal
mutation-search budget? Does the Romance-versus-other pair contrast change?
This is a descriptive mechanism comparison within this corpus panel. It neither
establishes ancestry nor repairs population inference through extra seeds.

Use the nine Decision 046 languages in the main comparison. Retain Markov and
Klingon as separately reported stress cases. Cross each source with the full and
disabled reward arms and search seeds 42, 1729, 8675309: 66 runs, of which 54 are
main-panel runs. Both arms and every search seed use the same corpus seed 42.
This holds each language's sampled text fixed; it does not estimate corpus
sampling uncertainty. Sampling and search seeds are separate recorded settings.

## Execution amendment

Use the frozen six-feature calibration and 20,000-token measurement policy.
Retain the v5 Python mutation/scoring primitives, 16 candidates per proposal,
and the existing weights in the full arm. Use one continuous PCG64 search stream
per run. Save all accepted candidate fields at full precision, its effective
score including reward amplification, operator schedule, RNG state, counters,
and proposal position. The controller does not reset scores or RNG at block
boundaries. This differs from the historical block controller in both respects.

Complete exactly 1,000 proposals. Record at 0, 100, 250, 500, 1,000 proposals;
retain every proposal record and every accepted corpus for audit. A stalled run
continues to its budget and is an admissible outcome. There is no form, alignment,
plateau, or accepted-stage early stopping rule in either arm. This is a bounded
search question, not the natural-termination/normalized-score-progress question
of Decision 046. Its readouts must never be pooled with that study's AUCs.

The 1,000-proposal budget is an engineering choice matching the previously
planned block size. Short full-candidate benchmarks suggest roughly two hours
per run, with large uncertainty and no claim that this budget reaches equilibrium
or has calibrated statistical power. State the budget in all scientific claims.

Commit each proposal atomically through a checkpoint pointer to immutable state
and proposal files. Save a complete proposal before advancing that pointer.
On interruption before commit, rerun the uncommitted proposal deterministically;
identical orphan artifacts are reusable, conflicting artifacts are an error.
Use an OS-held exclusive writer lock, JSON state (no pickle), hashes, frozen
inputs, and pinned source/runtime identities. Resume rejects contract changes.

Session time and proposal limits pause at the next proposal boundary. Time limits
are soft: an in-flight proposal may finish after the deadline. A PAUSE file also
requests a boundary pause. A crash may lose work within one proposal; it must not
lose previously committed scientific evidence. A paused run is not a completed
negative result. Changes to session limits do not change scientific identity.

## Ablation definition

The disabled arm sets these six configuration fields to zero: `form_weight`,
`reward_form_gain_weight`, `reward_suffix_gain_weight`,
`reward_trigram_gain_weight`, `reward_joint_bonus`, and `reward_penalty_relief`.
Keep structural-gain reward, coherence, mutation cost, and mutation policy fixed.
Audit that the six switches affect runtime scoring and that all direct form,
suffix, trigram, joint, and form-conditioned cost-relief contributions are zero.

Latin-informed proposals, local form-guided edits, and alignment-based operator
scheduling remain in both arms. The proposal policy is the same function of
state; actual proposals can diverge after states diverge. Shared seeds do not
guarantee paired identical proposals throughout the run. This is a form-reward
ablation, not a target-free or structure-only search.

## Readout

For each arm/search seed/proposal checkpoint, measure all pairwise Euclidean
distances in frozen z-six-feature space divided by their own initial pair
distance. For the nine-language panel report the mean of the 26 other pairs
minus the mean of the ten Romance-Romance pairs. Positive values favor greater
Romance convergence. Report each seed, their equal-weight mean, and the paired
full-minus-disabled difference. Also report the trapezoid integral of each pair's
normalized distance over proposal fraction using the five specified checkpoints,
then the same group contrast and arm difference. This is proposal-budget AUC,
not structural-progress AUC. Report checkpoint Latin distance, all raw features,
diversity, score contributions, and candidate outcome counts for every source.

No unconditional significance test, ancestry verdict, or threshold-based success
label. Zero initial pair distance makes normalization not evaluable; report the
failure instead of dropping the pair. All 54 main runs are required to reach
their budget before the complete main comparison is evaluable. Missing stress
runs are listed separately and do not change the main-panel denominator. Never
exclude entropy collapse or a Klingon nearest neighbor because of its direction.

Candidate counters partition returned candidate slots into accepted, rejected,
no-payload/no-op, and token-budget-invalid. A returned unchanged corpus is flagged
separately but remains eligible for the existing scoring rule. The mutation
primitive can internally retry up to ten operators; the counters describe its
returned slots, not those hidden internal attempts. Report evaluated-candidate
counts separately. Entropy uses bits on the measured 20,000-token prefix.

## Provenance and limits

Register all jobs, protocol/decision/source/runtime hashes, effective engine
configuration, source data, and retained reference inputs before any production
run. Keep pilot registrations explicitly marked and reject them from scientific
readout. Archive this implementation and its validation separately from the
earlier repair. No production result exists at amendment time. Prior legacy
outcomes, repaired pilots, benchmark trajectories, and synthetic statistical
checks were seen before this amendment; this is disclosed post-freeze development.

Independent corpus replication, shuffled-source controls, target-guidance
ablation, known-history corpus recovery, other token budgets, and broader
language sampling remain future designs. They are not implemented or claimed
complete by adding this execution version.
