# Decision 046: E1 dependence correction and claim scope

Date: 2026-10-01

Status: implemented amendment candidate under the author's instruction to
proceed with the design repair. Production scope is pending the author's choice
between this fixed-panel study and an expanded inferential design. This is a
disclosed post-freeze amendment, not the unchanged June preregistration. The
original frozen files remain intact.

## Evidence already seen

The first French run, the 15 legacy-objective production trajectories, their
endpoint/control analysis, and bounded French/Finnish repair pilots were seen
before this amendment. A bounded production-setting benchmark was started
during this design repair. Its initial French log was inspected for activity;
it had three accepted stages. No corrected full naturally terminated matrix
exists. Benchmarks are excluded from scientific analysis and cannot be promoted
to production. This amendment is prospective relative to corrected production,
not relative to all project evidence.

## Question, estimand and scope

For the nine named source corpora, under identical Latin-directed optimization,
do the five Romance sources show greater relative convergence toward one another
across normalized structural-score progress than pairs involving the four
unrelated controls, consistently over the three specified seeds?

Preserve the frozen six features, calibration, 20,000-token window, source
corpora, seeds (42, 1729, 8675309), optimization settings, natural stopping rule,
first-crossing interpolation and baseline-normalized pair AUC defined in
Decisions 044/045. Each seed produces ten Romance-Romance AUCs and 26 other
AUCs. Define D_s = mean(other AUCs) - mean(Romance AUCs). The primary effect T
is the arithmetic mean of the three D_s values. Positive values favor greater
Romance relative convergence. Report group means, every D_s, T, every pair and
every decile; do not turn 108 pair records into 108 independent replicates.

The primary conclusion is a **fixed-panel descriptive result**. The predicted
direction is observed consistently only if all three D_s exceed 1e-12 (a
numerical zero tolerance, not a minimum scientifically meaningful effect).
Otherwise report mixed direction or no consistent predicted direction, retaining
all effect sizes. No p-value turns this result into a population-level H1 YES/NO.
A null result is not equivalence. Report effect magnitude even when the direction
criterion holds; arbitrarily small differences are not substantive evidence.

This scope is deliberate: the nine languages are selected, their family labels
are not randomized, and there is one corpus per language. A defensible analysis
cannot manufacture exchangeability or independent language replication by
permuting pairs or repeating seeds. The question is comparative and answerable
for the panel; causal genealogy, ancestor reconstruction, generalization to all
languages, and separation of register/initial-geometry effects are outside it.

## Whole-language label reference

Replace the pooled-pair Mann-Whitney gate with exhaustive relabeling of five of
the nine languages as the focal set. For each of C(9,5)=126 allocations, keep
each language's entire three-seed bundle together and recompute T using the ten
within-focal pairs and the other 26 pairs. The observed allocation is included.
Do not permute seeds separately or permute pair rows.

Report the upper-tail reference fraction:

    count(T_allocation >= T_observed - tolerance) / 126
    tolerance = 1e-12 * max(1, abs(T_observed))

Include the allocation statistics, numerator and denominator. Ties count as
extreme; the minimum fraction is 1/126 and identical data yield 1. Duplicating
all seeds must not make the result more significant.

This is a valid permutation p-value **under the additional null assumption**
that the joint trajectory distribution is invariant to whole-language label
permutations (equivalently the focal five are uniform over these allocations
conditional on the unlabeled data). That assumption is not established for this
selected corpus bank. Therefore output identifies the number as a label-reference
fraction, records the conditional null, and expressly prohibits using its 0.05
threshold as an unconditional scientific verdict. Enumerating labels fixes the
dependence calculation; it does not cure confounding or selection. The old
Mann-Whitney helper is retained only for reproducing the superseded calculation.

Hemerik and Goeman's [permutation-test treatment](https://pmc.ncbi.nlm.nih.gov/articles/PMC6405018/)
establishes validity under distributional invariance, not automatically for
arbitrary observational group labels. Our use of 126 allocations represents all
9! label permutations with equal 5!4! multiplicities. Our finite-panel scope and
choice of statistic are project decisions, not claims made by that paper.

## Prespecified sensitivity and diagnostics

- Recompute T and seed direction after leaving out each of the nine languages,
  preserving the remaining group membership. Report every deletion and the
  range; these are sensitivity analyses, not confidence intervals or extra
  hypothesis tests.
- Report primary initial/final six-dimensional vectors, all available raw
  feature changes (including suffix and conditional character entropy), and
  fixed-bank endpoint distances. A Klingon nearest neighbor or entropy decline
  remains observable; neither is filtered or used to retune the objective.
- Keep Markov and Klingon three-seed trajectories as separate stress diagnostics
  in the 33-job plan; they do not enter the nine-language pair test. Missing or
  unfinished stress jobs are explicitly reported as incomplete robustness.
- Random-init trajectories, other budgets, E2-E4 and genealogical claims are
  deferred. This narrows the present study to E1; it does not complete or quietly
  discharge the broader H1 registration. A later random-init experiment needs
  its own generator, preservation/destruction properties and analysis frozen
  before its production outcomes.
- If a required primary run has no positive net structural improvement, an
  invalid baseline distance, a missing checkpoint, or fails replay, report E1 as
  not evaluable with the failed run identified. Never drop it, replace its seed,
  invent progress or report a negative scientific answer from an analysis error.

## Validation and provenance

Before production, test known effects, tied cases, language-order invariance,
seed-duplication invariance, and invalid/missing pairs. Enumerate all allocations
under synthetic exchangeable nulls to check finite-sample tail calibration and
run seeded synthetic alternatives. Include an intentionally nonexchangeable
confounding example to demonstrate the limits. Synthetic findings validate the
calculation under constructed conditions, not exchangeability of real languages.

Store a machine-readable protocol and its pinned SHA-256, plus the analysis
source hashes and source-corpus/objective identities in a registration sidecar
before any production launch. The readout must require that sidecar and reject
changed protocol/code, foreign runs or mixed launch conditions. Preserve every
prior trial and documentation revision. Record benchmark runtime, resource
settings and launch decision in a separate evidence report.

## Supersession

This supersedes Decision 045's pooled Mann-Whitney YES/NO rule, while retaining
its trajectory measurement mechanics. Neither the primary observed-direction
label nor the conditional reference fraction constitutes full registered H1
confirmation. The paper must state this scope in its question and conclusions.
