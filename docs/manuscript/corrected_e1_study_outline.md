# Corrected E1 study: question and reporting plan

Date: 2026-10-01. Prospective outline; no production results are available.
This is separate from the first-run manuscript and does not rewrite its history.
The fixed-panel scope below is pending the author's scope choice.

## Working title

Do Romance-source corpora converge more strongly than unrelated controls under
Latin-directed corpus transformation? A fixed-panel computational comparison.

## Research question and rationale

For the five specified Romance corpora and four specified unrelated-source
controls, does the same Latin-directed transformation procedure produce greater
relative convergence among Romance sources, consistently across three seeded
runs? Improving the Latin target score is insufficient evidence because the
optimization directly rewards it. The comparison asks whether the relative
between-source trajectories differ under shared target pressure.

## Primary evidence to report

- Identify all nine corpora, source/register differences, preprocessing, budget,
  the fixed feature calibration and all seed/optimization/stopping settings.
- Show baseline-normalized distance curves and the complete per-pair AUC table.
- Report other-minus-Romance AUC for each seed, their mean, and the direction
  criterion. Explain that pairs share trajectories and seeds do not multiply
  the number of languages. Give effects, not just threshold labels.
- Report the whole-language label-reference distribution as conditional on
  exchangeability, which is not established for this selected panel. It is not
  an unconditional significance test or a test of genealogy.
- Show all leave-one-language-out effects and raw-feature/entropy changes,
  together with fixed-bank endpoint comparisons and separate Markov/Klingon
  stress diagnostics. Include inconvenient results.
- State missing/incomplete controls and any non-evaluable trajectories. Report
  deviations from this plan, including the earlier implementation failure.

## Results wording depends on the evidence

If every seed has a positive contrast, the supported sentence is: the specified
corpus panel showed greater relative convergence among Romance-source pairs
under the tested optimization and feature geometry. Report the effect magnitude
and its sensitivity to panel membership alongside that sentence.

If directions differ or the effect is not positive, report that the predicted
pattern was not consistently observed. Do not call this proof of equivalence,
absence of all family signal, or falsification of historical relatedness.

If trajectories cannot be measured under the prespecified rule, the study is
not evaluable on that endpoint. Explain what failed and retain all data; do not
convert execution/analysis failure into a scientific negative.

## Discussion boundaries

This selected-panel comparison does not establish that family membership causes
convergence. Shared target pressure, initial geometry, orthography, corpus size
and register, or other typological properties can explain a pattern. It does
not recover ancestry or historical chronology. Explain what additional matched
corpora, controls or randomized interventions would distinguish those accounts.

The first and second runs are prior methodological/exploratory evidence. State
which objective they executed and why the second matrix cannot answer the
corrected E1 question. Present Decision 046 as a disclosed post-freeze amendment;
do not describe the whole project as an untouched prospective registration.

Random-init robustness, other budgets and E2-E4 are deferred in this narrower
study and must not be claimed as completed. A broader inference paper requires
an expanded design before production. The original first-run paper remains
separate and needs its own editorial decision/revision.
