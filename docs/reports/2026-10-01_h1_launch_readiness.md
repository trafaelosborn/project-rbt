# Corrected H1 launch readiness and research question

Date: 2026-10-01

Status: software correctness pilots passed; full confirmatory launch is not
ready. No corrected production matrix was launched during this review.

Follow-up: the [design repair audit](2026-10-01_h1_design_repair_audit.md)
records the implemented dependence correction, validated conditional reference,
registration guards and production-setting benchmarks. The remaining scope
choice is explicit: a fixed-panel comparison or an expanded inferential design.

This assessment follows the user asking whether we are ready to run and whether
the study addresses PLOS One's reported criticism of the first manuscript.
The rejection letter itself was not inspected; the editorial criticism is
known from the user's account. The current manuscript and frozen protocol were
inspected directly.

## What the repair establishes

The [repair audit](2026-10-01_h1_repair_audit.md) records 155 distinct passing
tests and successful French/Finnish production-launcher pilots. Those results
support execution of the intended six-feature objective, exact measurement
budget, checkpoint continuity and replay. They do not establish H1 or validate
the inferential design. The pilot used four proposals and two candidates per
proposal; production is planned with 16 candidates and natural termination.
Production-scale throughput has not been demonstrated with the reference
Python evaluator.

## The question and the answer the new study could provide

Operational E1 question: under the same Latin-directed optimization, do the
five Romance-source corpora converge toward one another more strongly across
matched normalized structural-score progress than pairs involving the four
specified unrelated-language controls?

The registered measurement is the area under each pair's distance curve,
normalized by its starting distance. Lower area means greater relative
convergence over the trajectory. Here, "earlier" refers to normalized score
progress, not chronological language history, elapsed runtime or proposal
count. The exact first-crossing interpolation and pair definitions are
disclosed in Decision 045.

The scientific contrast concerns a Romance-associated convergence pattern
against unrelated-source behavior under the same optimization conditions.
An improved Latin objective score alone does not answer it: Latin is already
part of the optimization target. A positive result would support the specified
geometric contrast for these corpora and conditions. It would not by itself
establish ancestry recovery or exclude corpus, register, initial geometry or
typological explanations. A negative or uncertain result must be reportable
without retuning the objective to obtain a preferred answer. Failure to reject
a null is not proof that the two groups are equivalent.

## Prelaunch issues

1. **Inference must respect shared trajectories.** Decision 045 pools 30
   Romance-pair AUCs and 78 comparison-pair AUCs across three seeds in a
   Mann-Whitney calculation. French-Spanish and French-Italian both contain the
   French trajectory; the pair observations are dependent. Seed repeats also
   do not create additional independent languages. The current numerical
   calculation does not provide an established error guarantee for this
   design. A caveat alongside a YES/NO gate is insufficient for a confirmatory
   claim. The caveat in Decision 045 should have been elevated to this readiness
   blocker in the earlier handoff.
2. **Specify and disclose the inferential revision before new production
   outcomes.** A possible approach is to preserve each language's full set of
   seeded trajectories when permuting group labels, recomputing all pair
   summaries together. This is a proposal, not an adopted replacement test:
   the exchangeability assumption for these deliberately selected languages
   needs justification, and dependence-preserving permutations alone do not
   solve confounding. Fix the estimand, null, assumptions, statistic, uncertainty
   and decision rule in a new registration/amendment; validate the procedure
   with synthetic null and alternative cases. Retain the original freeze and
   disclose that the legacy runs and bounded repair pilots have already been
   seen. Do not call a revised test the untouched original preregistration.
3. **Resolve the claimed scope and execution feasibility.** Define the missing
   random-init robustness condition or explicitly narrow the study's claims;
   the 33-job plan alone does not complete all registered robustness work or
   E2-E4. Benchmark the production candidate count and representative block
   before committing to an unbounded matrix. No runtime estimate from the
   two-candidate startup pilots is sufficient for that decision.

These issues do not undo the objective repair. They distinguish an executable
instrument from a complete confirmatory study design. This review does not
change the frozen feature files, analysis implementation or manuscript.

## The first manuscript and the journal concern

The current first manuscript opens its abstract by saying it documents one
French-to-Latin run and explicitly describes itself as descriptive. It does
contain an exploratory question about the intermediate states produced by
target pressure. It therefore is not literally devoid of a question, but the
framing and single-run evidence are weaker than an explicit empirical contrast
with specified competing outcomes.

The corrected study has an explicit comparative question. We have not yet
obtained its answer or rewritten the manuscript around completed evidence. New
controls cannot be retroactively described as part of the first run's design.
A revision or new paper should make its question, comparison, result and limits
central, and label the earlier runs as prior exploratory/method-development
evidence. Resolving this issue does not guarantee journal acceptance.

Official PLOS One [publication criteria](https://journals.plos.org/plosone/s/criteria-for-publication)
allow methods/software research and negative or null results, while requiring
rigorous analyses, appropriate controls and conclusions supported by data.
Thus a positive H1 finding is not a prerequisite for a defensible paper. A
clearly posed question and a sound, appropriately limited answer are the goal.

## Review record

- Read the repair audit, Decisions 044/045, frozen E1/robustness sections,
  current convergence implementation and tests, launch planner, and current
  first manuscript's abstract/introduction.
- Consulted official PLOS One publication criteria and submission guidelines.
- Preserved existing source, manuscript and historical evidence; no full run
  launched, no new statistical method silently adopted.
- Documentation-only follow-up; existing passing tests are historical repair
  evidence, not tests newly executed during this readiness review.
