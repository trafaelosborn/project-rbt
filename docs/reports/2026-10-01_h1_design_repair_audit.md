# E1 analysis repair and throughput audit

Date: 2026-10-01

Status: dependence-aware fixed-panel analysis implemented; synthetic validation
and regressions passed. Bounded benchmarks finished or reached their recorded
cap. Production scope is pending the author's choice. No production matrix is
launched, and no benchmark remains running in its tool session.

## Authorization and scientific decision

The author requested proceeding with the remaining statistical repair,
validation, documentation and prelaunch work. An unconditional family-level
significance claim is not justified by the existing selected nine-corpus panel.
Decision 046 and its machine-readable companion implement an explicitly limited
panel comparison. The author has been asked to choose between collecting that
study and expanding the design before production. No unanswered scope question
is treated as permission to launch.

The old analysis treated 108 overlapping pair AUCs as independent observations
in its significance calculation. The repair retains pair data but aggregates
them into one contrast per seed, keeps each language's three trajectories
together through all 126 five-of-nine label allocations, and removes the
unconditional YES/NO significance gate. Labels are not randomized in this
corpus panel; the reference fraction is conditional, not a calibrated population
claim. This narrows claim scope rather than pretending that a new permutation
routine establishes exchangeability.

## Implementation

- `src/validation/h1_convergence.py`: effect/seed summaries, exhaustive
  whole-language reference, all nine leave-one-language-out summaries, raw
  feature and endpoint-bank diagnostics, separate stress completeness, and JSON
  plus readable Markdown reports. Malformed trajectories/pair panels and
  non-evaluable studies fail explicitly. The historical rank-sum helper remains
  only for reproducing the superseded calculation.
- `src/validation/h1_protocol.py`: pins the JSON protocol and Decision 046;
  writes a registration sidecar to a new directory before production; validates
  current analysis, objective, source hashes, commands, full job settings and
  manifest creation times. Existing runs cannot be adopted retroactively.
- `src/control/h1_matrix.py`: versioned analysis identity in plans and a
  `register` mode, separate from execution. There are still 33 planned jobs:
  27 primary and six separate Markov/Klingon stress trajectories.
- Tests and reproducible scripts cover statistics, registration, simulation and
  bounded production-setting benchmarks. The engine/scoring implementation and
  all original freeze artifacts have not changed during this analysis repair.
- Decision 045 is marked superseded for inference; README and a separate
  corrected-study outline explain the question, possible results and limits.
  The earlier manuscript and all pre-existing user changes are preserved.

## Validation

1. Initial targeted run: 54 tests passed in 8.20 seconds.
2. Expanded regression run: 78 tests passed in 54.29 seconds, including H1
   objective/execution, continuation and existing supervisor tests. The suites
   overlap; do not add their counts as independent tests.
3. Known positive, null/tied and reversed effects have the expected reference
   fractions. Duplicating seeds does not change the reference fraction; language
   order, pair orientation and row order do not change it. Mixed seed direction
   remains visible even with a positive pooled effect.
4. Exact allocation enumeration under ten constructed dependent-edge null
   panels never rejects more than 6/126 label allocations at a 0.05 cutoff.
5. One thousand additional synthetic panels per scenario, seed 20261001:
   exchangeable-null tail fraction below 0.05 in 57/1000 panels; designed
   within-Romance alternative in 1000/1000; deliberately label-correlated
   nuisance in 996/1000. Exact conditional null allocation rejection was at
   most 6/126 = 0.047619. The 57/1000 Monte Carlo result is a finite-simulation
   estimate, not the exact conditional rate. The nuisance case deliberately
   violates exchangeability and demonstrates why a small reference fraction
   cannot establish genealogy.
6. A real CLI readout against the earlier incomplete pilot matrix emitted
   `NOT_EVALUABLE` artifacts and a nonzero process result. It did not produce a
   negative scientific verdict. Planning created no production run directory.
7. Final statistics/registration suite: 27 passed in 10.97 seconds after adding
   full stress-report coverage and retained-reference input protection. There
   are **79 distinct passing tests** across this phase's overlapping selections.
   Registration additionally hashes 18 retained form/coherence/family reference
   inputs, including the Latin corpus and n-gram metadata; an edited reference
   is refused. The final command and JUnit output are listed below.
8. Independent replay of the four completed benchmark/profile jobs verified
   all 17 saved stages, exactly 20,000 measured tokens per stage, checkpoint
   hashes, actual proposal counts, and exclusion from scientific readout.
   Maximum rounded-score replay error was 4.765e-7. Verification changed no
   benchmark manifests.

Synthetic effect strengths are constructed examples, not estimates of empirical
power. Validation does not establish exchangeability of the real languages.

## Commands and retained evidence

All commands run from the repository root. Fresh pytest base paths preserve
earlier evidence. The exact test selections are retained in JUnit XML.

```powershell
python -m pytest tests/test_h1_convergence.py tests/test_h1_panel_statistics.py tests/test_h1_execution.py -q --tb=short -p no:cacheprovider --basetemp=data/tmp/h1_design_pytest_20261001_01 --junitxml=artifacts/h1_design_20261001/design_tests_01.xml
python -m pytest tests/test_h1_convergence.py tests/test_h1_panel_statistics.py tests/test_h1_execution.py tests/test_multi_run_supervisor.py tests/test_long_run_v4.py -q --tb=short -p no:cacheprovider --basetemp=data/tmp/h1_design_pytest_20261001_02 --junitxml=artifacts/h1_design_20261001/design_tests_02.xml
python scripts/validate_h1_panel_statistics.py --output artifacts/h1_design_20261001/synthetic_validation_01.json --repetitions 1000 --seed 20261001
python scripts/benchmark_h1_production.py --output-root artifacts/h1_design_20261001/benchmark_01 --proposals 1000 --timeout-seconds 1800
python scripts/benchmark_h1_production.py --output-root artifacts/h1_design_20261001/profile_01 --proposals 8 --timeout-seconds 600 --profile
python scripts/benchmark_h1_production.py --output-root artifacts/h1_design_20261001/benchmark_02 --proposals 32 --timeout-seconds 600
python -m src.validation.h1_convergence --run-root artifacts/h1_repair_20261001/pilot_01 --output artifacts/h1_design_20261001/pilot_exclusion.json
python -m src.control.h1_matrix plan --output-root data/v2_e1_fixed_panel_20261001 --plan-file artifacts/h1_design_20261001/production_plan.json
python -m pytest tests/test_h1_convergence.py tests/test_h1_panel_statistics.py -q --tb=short -p no:cacheprovider --basetemp=data/tmp/h1_design_pytest_20261001_04 --junitxml=artifacts/h1_design_20261001/design_tests_final_02.xml
python scripts/verify_h1_benchmarks.py artifacts/h1_design_20261001/benchmark_02
python scripts/verify_h1_benchmarks.py artifacts/h1_design_20261001/profile_01
python -m src.control.h1_matrix plan --output-root data/v2_e1_fixed_panel_20261001 --plan-file artifacts/h1_design_20261001/production_plan_final.json
python scripts/archive_h1_design_evidence.py
```

The first synthetic artifact predates final report/registration guard edits and
the clarification that the scope choice is pending. Its recorded code/document
hashes identify that intermediate revision. Retain it and produce a fresh final
validation artifact rather than overwriting it.

`synthetic_validation_final.json` reran all 3000 constructed panels with the
same numerical results after the statistical/readout changes. The subsequent
change to `h1_protocol.py` only added retained-reference data hashes; the final
statistical implementation is unchanged from that simulation and is covered
by the final tests. Earlier plans and intermediate validations are preserved.
`production_plan_final.json` captures the current analysis identity. The source
snapshot and `completion_inventory.json` record exact source hashes, runtime,
HEAD, test count and the pending production scope. No git commit/tag was made.

## Operational observations

Eight-proposal profiled French and Finnish trials with 16 candidates completed
in 98.37 and 115.46 seconds respectively, including startup/audit and profiling
overhead. Profiling attributed 47.15/59.63 seconds to feature computation,
including 23.93/34.75 seconds in conditional character entropy. Full candidate
evaluation dominated overall runtime. These profiled numbers are not production
throughput estimates. Unprofiled measurements are recorded separately.

All benchmark children use one numerical-library thread and below-normal
Windows priority, with hidden windows. Each benchmark is finite and excluded
from scientific readout. Benchmark 01 runs a full 1000-proposal block with a
30-minute per-job cap; benchmark 02 measures 32 proposals per language with a
10-minute cap. Separate concurrent trials and tests can affect wall time.

Unprofiled 32-proposal results at production candidate count:

| Source | Actual proposals | Wall seconds including startup/audit | Seconds/proposal |
|---|---:|---:|---:|
| French, seed 42 | 32 | 196.1138 | 6.1286 |
| Finnish, seed 42 | 32 | 240.2706 | 7.5085 |

At those initial rates, a 1000-proposal block would take approximately 102 and
125 minutes respectively. These are short-batch linear extrapolations, not
measured full-block times or forecasts of natural trajectory termination.
Neither startup, acceptance/audit workload nor later-state costs are constant.

The French 1000-proposal benchmark reached its predeclared cap after 1800.0553
seconds. The harness terminated and waited for that child, retained the partial
artifacts, wrote `status: timeout` in `benchmark_results.json`, and exited
nonzero. The corresponding full Finnish benchmark was not attempted after the
timeout. The partial French manifest can still say `running` because its block
did not finish; it is not evidence of a currently running scientific job. Do not
resume or promote it as production. The timeout is an operational result, not
evidence against the linguistic hypothesis.

## Operational issues

A multi-file patch reported a context error after some changes had already
landed, including a duplicated Decision 045 follow-up paragraph. The actual files
were inspected and the duplicate removed; remaining edits were applied
separately. No historical evidence or frozen input file was replaced.

The benchmark harness was extended to support profiling while benchmark 01
was already running. Its Python traceback therefore displayed current source
lines against the earlier in-memory code. The explicit error was the intended
timeout. `benchmark_01/benchmark_script_at_start.py` preserves the exact launch
version, reconstructed and verified against its recorded SHA-256
`83283dde6c25ed653d176b00f57ff7bfe7fa5f16d893e302fbf7ab18d713c0db`.
The engine/scorer source did not change. The earlier source snapshot under
`artifacts/h1_repair_20261001` retains the pre-analysis-repair dependencies.

A read-only Windows CIM process inventory returned Access denied. No escalation
or permission change was attempted. The execution tool reported completion of
each benchmark harness; the timeout handler explicitly kills/waits for its child.
No claim is made about unrelated system processes. `git diff --check` passed;
CRLF conversion notices were informational.

## Reproduction limits and remaining work

The fixed-panel implementation is ready for review but does not complete the
original broad H1 registration. A general inferential claim requires a different
design with defensible sampling/null assumptions; it cannot be obtained by
renaming a label-reference fraction. Random-init and E2-E4 remain explicitly
deferred in the fixed-panel option. Full natural-termination runtime is unknown.
No manuscript results, ancestry claim or full-production completion is implied.
