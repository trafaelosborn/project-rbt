# H1 repair audit — 2026-10-01

Status: repairs implemented; tests and bounded pilots passed. No corrected full
production experiment has been launched. H1 remains scientifically untested.

Readiness follow-up: the [launch/design review](2026-10-01_h1_launch_readiness.md)
elevates the dependent-pair inference limitation below to a blocker for a
confirmatory launch. Passing implementation checks is not approval of the
statistical decision rule or evidence of production-scale throughput.

## Scope and baseline

Implement Decision 044 and validate through a bounded production-path trial.
The earlier 15-run matrix remains a legacy-objective experiment. Its entropy
collapse and nearest-control results do not test the registered H1 hypothesis.

Repository HEAD at start: `3bf602fca67de5f5c0bea37ee97dae0415d65421`.
Frozen H1 commit: `232393a2d943f878b71d9c75aa40dce8d4602e4f`.

Pre-existing tracked modifications were present in the manuscript TEX/PDF,
`figures/Fig1.pdf`, `src/control/multi_run_supervisor.py`, and its test file.
These changes are not part of this repair and must be preserved.

The feature document, static matrix, feature tiers and corpus manifest were
compared against `git show v2-prereg-freeze:<path>`. Their content matches after
normalizing CRLF to LF; three files have Windows checkout line endings. Runtime
verification therefore hashes normalized text, not platform-specific newlines.

## Work log

- Traced launcher → continuation driver → v5/v4 engine → ordinary and
  incremental/batch scorers. Confirmed the protocol fields were metadata only.
- Verified the frozen six-feature calculation functions are already available.
- Selected one reference evaluator for H1 to eliminate alternate legacy
  scoring paths while correctness is established.
- Documented the sampling boundary policy and legacy composite reward terms
  explicitly in Decision 044.

## Validation

### Test results

- Affected regression suite: **153 passed in 49.63 seconds**. Covers engine
  v2/v3/v4/v5, continuation, run config/controller, existing supervisor changes,
  static features, incremental scoring, batch guidance, and new H1 tests.
- Final focused H1 suite after adding stop/resume and full synthetic matrix
  tests: **38 passed in 2.32 seconds**. The selections overlap: 155 distinct
  passing tests across the selections, not 191.
- Eight acceleration-flag combinations accepted an improvement whose legacy
  three-feature reward vector was exactly tied with its predecessor. Calling
  the legacy reward in these tests raises an assertion.
- Every retained coordinate independently changes the score in a controlled
  fixture. Independent arithmetic from the real frozen table matches the H1
  objective. Sparse and full candidate payloads are exercised.
- Checked exact token measurement, explicit undersized-candidate rejection,
  saved-score tampering, invalid freeze identifiers/budgets, incompatible resume,
  no downgrade to legacy scoring, edited checkpoints, and actual-engine manual
  stop/resume with preserved corpus order.
- Readout tests cover known interpolation/AUC geometry, exact rank probabilities
  including ties, a complete synthetic 27-trajectory matrix, mixed-condition
  refusal, and exclusion of finite pilots, partial matrices and legacy runs.
- `git diff --check` reported no whitespace errors. Git's CRLF conversion notices
  are not content errors.

Evidence:

- `artifacts/h1_repair_20261001/tests.log`
- `artifacts/h1_repair_20261001/tests.xml`
- `artifacts/h1_repair_20261001/final_h1_tests.log`
- `artifacts/h1_repair_20261001/final_h1_tests.xml`

### Real production-launcher trial

Commands, from the `project_rbt` repository directory:

```powershell
python -m src.control.h1_matrix pilot --output-root artifacts/h1_repair_20261001/pilot_01
python scripts/verify_h1_pilot.py artifacts/h1_repair_20261001/pilot_01
```

The pilot invokes `SupervisorJob.command()` in separate processes, using the
same `src.retrodiction.long_run_v5` entry point as production. The saved launch
plan contains the exact argument arrays. Conditions: seed 42, budget 20,000,
four proposals in two blocks of two, two candidates per proposal, all live
events, validator snapshot every block, incremental/batch flags requested and
effective backend explicitly `h1_python_reference`. Processes used one thread
per numerical library and below-normal Windows priority. Short proposal and
candidate limits are pilot-only departures; these jobs cannot decide H1.

| Source | Wall seconds including startup | Proposals | Accepted stages | Saved stages replayed | Initial H1 score | Final H1 score |
|---|---:|---:|---:|---:|---:|---:|
| French | 14.01 | 4 | 2 | 4 | -12.9037620384 | -12.8878177586 |
| Finnish | 15.17 | 4 | 3 | 5 | -13.0434542663 | -12.9719423573 |

These scores establish that the machinery runs. Their direction and magnitude
are not a convergence result. Saved-stage counts include repeated block
baselines and are not independent observations.

Checks against saved artifacts:

- Every measurement window contained exactly 20,000 tokens.
- All nine saved stages reproduced their structural score; maximum error
  against six-decimal serialization was 3.573018841e-7.
- Each next-block seed corpus exactly equaled the previous checkpoint,
  including sequence order.
- Endpoint Latin distance equaled `-structural_score / 5`. All reference
  comparisons used the frozen calibration, never a fit to generated states.
- Reinvoking the completed production commands changed no JSON run artifacts.
- The H1 readout explicitly rejected both finite-budget pilots.
- Read-only admission of the historical French seed-42 manifest was rejected;
  its file hash remained unchanged.
- French rejected zero candidates for insufficient tokens; Finnish rejected one
  in each block. This exercises the disclosed budget-floor rule.

Verified objective identity:

```text
a155a8d78d20c53b156a14107d845c035d4a916ee404a3df5399fedca0d5baa2
```

Evidence root: `artifacts/h1_repair_20261001/pilot_01/`. It contains the launch
plan, process logs, `pilot_results.json`, `verification.json`, manifests, block
summaries, saved corpora, stage records, live events, fixed-bank distances,
validator snapshots, and completed-resume logs. The replay script is retained
at `scripts/verify_h1_pilot.py`.

### Endpoint command and production planning

The existing validation command now dispatches H1 manifests to the frozen H1
comparison, rather than silently using the legacy validator metric:

```powershell
python -m src.control.cli validate --output-dir artifacts/h1_repair_20261001/pilot_01/french/seed_42 --output artifacts/h1_repair_20261001/pilot_01/french_validation
python -m src.control.h1_matrix plan --output-root data/v2_h1_corrected_production_20261001 --plan-file artifacts/h1_repair_20261001/production_plan.json
```

The plan lists **33 jobs**: 27 primary trajectories (five Romance languages plus
four required typological controls, three seeds each), three Markov stress
jobs, and three optional Klingon jobs. Planning created no production processes
or production output directories. The existing supervisor and its pre-existing
user changes were not edited. A plan is not a completed experiment.

Primary E1 readout entry point:

```powershell
python -m src.validation.h1_convergence --run-root data/v2_h1_corrected_production_20261001 --output artifacts/h1_repair_20261001/e1_result.json
```

This command requires all primary trajectories to exist and reach natural
terminal conditions. Decision 045 discloses interpolation, pair membership,
pooling, ties, and direction criteria.

### Test reproduction

Use a new workspace-local `--basetemp` directory on each invocation. Pytest may
clear an existing base directory; never point it at retained evidence. The
actual commands used were:

```powershell
python -m pytest tests/test_h1_execution.py tests/test_h1_convergence.py tests/test_engine_reinforced_v2.py tests/test_engine_reinforced_v3.py tests/test_engine_reinforced_v4.py tests/test_engine_reinforced_v5.py tests/test_long_run_v4.py tests/test_run_config.py tests/test_run_controller.py tests/test_multi_run_supervisor.py tests/test_static_feature_matrix.py tests/test_incremental_scoring_state.py tests/test_v4_batch_guidance.py -q --tb=short -p no:cacheprovider --basetemp=data/tmp/h1_repair_pytest_20261001_03 --junitxml=artifacts/h1_repair_20261001/tests.xml
python -m pytest tests/test_h1_execution.py tests/test_h1_convergence.py -q --tb=short -p no:cacheprovider --basetemp=data/tmp/h1_repair_pytest_20261001_05 --junitxml=artifacts/h1_repair_20261001/final_h1_tests.xml
```

Observed environment: Windows, Python 3.14, NumPy 2.4.3, pytest 9.0.2. Exact
versions are retained in `environment.json` in the evidence directory.

## Failures encountered and resolved

1. Initial affected-suite result: 34 passes and one failure. An old test supplied
   dummy freeze identifiers and expected serialization only; corrected code
   properly rejects them. The generic legacy test now supplies no registration.
   Dedicated H1 tests cover strict validation and actual scoring propagation.
2. The default pytest temp directory and existing cache were inaccessible. The
   initial focused invocation ran seven checks before three fixture setup
   errors. Subsequent runs used fresh workspace-local temp directories and no
   cache provider. These were environment failures, not scoring failures.
3. Creating an evidence directory under `data/validation` returned a Windows
   access error. Evidence was placed in the new repository-local `artifacts`
   directory instead. No ACLs were changed.
4. Multi-file patch attempts reported context mismatches after earlier file
   edits had already landed. Files were inspected and remaining changes applied
   separately; tests and final inspection verify the actual resulting files.

## Changed implementation surfaces

- `src/retrodiction/h1_objective.py`: frozen artifacts, objective/scaling,
  deterministic measurement, reference distances, implementation identity.
- `src/retrodiction/h1_execution.py`: launch/resume contracts, source/checkpoint
  hashes, independent block replay, fixed-geometry endpoint exports.
- Engine v2: active configuration, exact initial token sampling, checkpoint
  preservation, corrected structural reward and score-component records.
- Engine v4: one H1 scorer for all requested backends, sparse materialization,
  undersized-candidate handling and live measurement events.
- Drivers v4/v5: pass configuration into the engine, check contracts before
  writes, audit completed blocks, and preserve consistent structural
  stop/progress fields and validator snapshots.
- Validator-bank entry point/controller CLI: H1 dispatch and diagnostic labels.
- `src/control/h1_matrix.py`: source/control planning and bounded pilot.
- `src/validation/h1_convergence.py`: guarded E1 readout under Decision 045.
- New H1 tests and the legacy manifest test. Frozen registration files remain
  unchanged. Pre-existing manuscript, figure, and supervisor edits remain.

## Limits and remaining scientific work

- H1 candidates use full Python reference evaluation even when acceleration is
  requested. Native H1 acceleration and production-scale throughput have not
  been established.
- The budget-boundary policy affects candidate eligibility and is disclosed
  as a post-freeze operational clarification. It must accompany corrected
  results. No entropy penalty or control-specific reward was added to steer
  away from the earlier Klingon result.
- Existing form/coherence/cost terms and reward amplification remain in the
  acceptance composite. The legacy four-element vector remains explicitly
  labeled coherence/reporting diagnostics, not the H1 reward.
- The existing per-block PRNG restart policy remains. Preserving corpus order
  does not imply one continuous RNG stream across blocks.
- E1 pairwise AUCs share trajectories and are dependent. Decision 045 records
  that the requested Mann-Whitney probability is not an independent-sample
  error guarantee. Stronger trajectory-level inference requires a separately
  disclosed registration choice.
- Random-init robustness, completed Markov/control trajectories, optional
  Klingon analysis and E2-E4 are not completed by this repair.
- No evidence yet establishes that the corrected full experiment will converge,
  avoid entropy collapse, recover ancestry, or support H1. Those are outcomes
  to measure with the repaired instrument.
