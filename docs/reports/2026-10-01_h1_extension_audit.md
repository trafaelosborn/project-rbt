# H1 execution extension audit, 2026-10-01

## Scope and implementation

The author approved implementing the parameter review. Decision 047 implements
separate sampling/search seeds, continuous resume, proposal-boundary checkpoints,
candidate/trajectory diagnostics, resource pauses, and a form-reward ablation.
This extension has its own registration and readout. Historical engine files,
Decision 046, and its pinned protocol have not been edited for this extension.

New components:

- `src/retrodiction/h1_checkpointed.py`: configuration, reward accounting,
  diversity diagnostics, continuous search, immutable evidence, atomic pointer,
  replay audit, and pause/resume.
- `src/retrodiction/h1_run_lock.py`: process-owned exclusive lock on Windows/POSIX.
- `src/control/h1_ablation.py`: pre-launch registration, frozen input/runtime
  identity, exact sampled-source hashes, full/disabled settings, and CLI.
- `src/validation/h1_ablation.py`: equal-proposal-budget contrasts, paired arm
  differences, descriptive reports, and pilot/incomplete-run exclusion.
- Two test modules and `scripts/verify_h1_extension.py` validate behavior.

The scientific question is: **For this source-corpus panel, how does disabling
explicit Latin-form rewards change convergence and diversity at the same search
budget?** Proposal guidance still uses Latin in both arms. This is neither a
target-free comparison nor a claim to have recovered historical ancestry.

The prospective extension matrix contains 66 runs: nine main languages plus
Markov/Klingon stress sources, three search seeds, and two arms. All use corpus
seed 42 so text is fixed within each language. Each production job has 1,000
proposals and 16 candidate slots; checkpoints for the readout are 0, 100, 250,
500, and 1,000. The budget is an explicit feasibility choice, not a claim of
convergence or statistical power. The existing 20,000-token feature calibration
and mutation/reward weights in the full arm are retained.

Unlike the old block loop, the new controller preserves the accepted effective
score and RNG stream across sessions. Both arms use this same controller. It
disables form/alignment/plateau early stopping so the ablation is compared at an
equal proposal budget. Consequently its results must not be pooled with the
Decision 046 natural-termination, structural-progress AUC.

## Validation record

All evidence is under `artifacts/h1_extension_20261001`.

1. `tests_01.xml`: 19 initial controller/ablation tests passed in 2.17 seconds.
2. `tests_02.xml`: collection failed before executing the broad suite because
   a multi-file patch duplicated an import, a runtime metadata entry, and a
   `with threadpool_limits` line in the launcher. The duplicate `with` caused
   an IndentationError. The duplicates were removed with a single-file patch;
   this failed attempt is retained.
3. `tests_03.xml`: 129 tests passed in 52.05 seconds, covering both new modules,
   existing H1 execution/readout/registration, v2-v5 engines, long-run driver,
   and supervisor. No failures or skips. This is the broad-suite count, not
   a sum of overlapping earlier executions.

Behavioral coverage includes exact uninterrupted/resumed equality in both arms,
simulated interruption after writing a proposal but before publishing the
checkpoint pointer, deterministic reuse of the orphan evidence, writer-lock
exclusion/release, soft session limits and PAUSE files, separate corpus/search
seeds, all six reward switches, outcome-count partitioning, changed-contract and
corrupted-state rejection, source-sampling invariance across search seeds,
known-direction/zero-effect readouts, missing main runs, and separate stress
denominators. The full matrix test verifies 66 unique jobs and matching sampled
corpora across each language's arms and search seeds.

The replay audit recomputes every committed accepted state's features, scores,
reward terms, diversity, and operator schedule. It checks every proposal's
candidate contributions, score differences, winner, transition, and counters.
It does not recompute every rejected candidate from a saved corpus; only its
hash and score ledger are retained. Exact proposal reproduction is tested by
uninterrupted-versus-resumed runs. The code archives and data identities provide
the inputs needed for a separate full replay if required.

## Real-corpus validation

`pilot_01/verification_results.json` records four completed, paused/resumed
French/Finnish trajectories at 4 proposals and 2 candidate slots each:

| Source and arm | Saved states independently replayed | Evaluated candidates | Validation wall seconds |
|---|---:|---:|---:|
| French, full | 3 | 8 | 23.5381 |
| French, disabled | 3 | 8 | 23.8424 |
| Finnish, full | 3 | 7 | 25.2368 |
| Finnish, disabled | 3 | 7 | 25.3413 |

Wall time includes multiple engine/reference initializations and independent
audit; it is not production throughput. Each Finnish run also had one returned
candidate rejected for the token budget. There are 12 audited saved states in
these four runs, with the exact 20,000-token measurement policy. Two additional
uninterrupted French references matched the paused/resumed runs in both arms:
every candidate proposal ledger, corpus/state byte sequence, RNG state, schedule,
and counter matched. Registration-specific contract hashes differ intentionally
between output roots and were not misrepresented as equal. The pilot readout CLI
returned exit code 2 and wrote `not_evaluable` as required.

`process_resume_01/verification_results.json` repeats French in both arms using
six actual child-process launches: pause after one proposal with Python hash seed
101, resume in a new process with hash seed 202, and run an uninterrupted reference
with hash seed 303. All six commands exited zero (60.89 seconds total child wall
time). Both arms matched exactly for proposals, saved state bytes, RNG state,
and counters. Each child's complete stdout/stderr log and command are retained.
This extends the same-process checks to actual interpreter exits and different
hash iteration seeds for the tested paths; it does not claim cross-platform or
cross-version bitwise portability.

The final production plan is at `pilot_01/production_plan.json`. Its prospective
root `data/h1_ablation_production_20261001` does not exist: no production
registration or production trajectory was started. The pilot registrations are
separate and cannot be promoted. The code remained unchanged throughout both
real-corpus validations.

`git diff --check` passed; line-ending conversion notices were informational.
An AST check found no duplicate literal dictionary keys in the four new execution
and readout modules after correcting the patch issue. `source_snapshot.zip` and
`completion_inventory.json` preserve the final code/docs, source and evidence
hashes, protocol/runtime identity, 129-test count, and production-not-launched
status. Raw source corpora are hashed but not duplicated in that code archive.

## Reproduction commands

Run from the repository root. Use a fresh output path for every registration or
pilot; existing evidence is never adopted into a new registration.

```powershell
python -m pytest tests/test_h1_checkpointed.py tests/test_h1_ablation_registration.py tests/test_h1_execution.py tests/test_h1_convergence.py tests/test_h1_panel_statistics.py tests/test_engine_reinforced_v2.py tests/test_engine_reinforced_v3.py tests/test_engine_reinforced_v4.py tests/test_engine_reinforced_v5.py tests/test_long_run_v4.py tests/test_multi_run_supervisor.py -q -p no:cacheprovider --basetemp=data/tmp/h1_extension_03 --junitxml=artifacts/h1_extension_20261001/tests_03.xml
python scripts/verify_h1_extension.py --output-root artifacts/h1_extension_20261001/pilot_01
python scripts/verify_h1_process_resume.py --output-root artifacts/h1_extension_20261001/process_resume_01
python scripts/archive_h1_extension.py
```

The first command reproduces the test selection; choose a new `--basetemp` and
XML path on another execution. The pilot harness uses 4 proposals and 2 candidates
per run, with French/Finnish in both arms. It pauses each after one proposal,
resumes, audits all saved states, and separately compares French in both arms
with uninterrupted runs. It also invokes the real readout CLI and requires exit
code 2 with `not_evaluable` for the pilot registration.

Prospective usage (not evidence these production commands have been executed):

```powershell
python -m src.control.h1_ablation plan --output-root data/h1_ablation_production_20261001 --plan-file artifacts/h1_ablation_plan_new.json
python -m src.control.h1_ablation register --output-root data/h1_ablation_production_20261001
python -m src.control.h1_ablation run --output-root data/h1_ablation_production_20261001 --job-id french__corpus_42__search_42__full --max-session-seconds 1800
python -m src.control.h1_ablation status --output-root data/h1_ablation_production_20261001
python -m src.validation.h1_ablation --root data/h1_ablation_production_20261001 --output artifacts/h1_ablation_results_new.json
```

Repeat the same `run` command to continue a paused job. The time limit is a soft
boundary: an in-flight proposal may finish after it. The checkpoint protects
committed work even if the process is killed mid-proposal. A file named `PAUSE`
inside a job directory requests a pause; remove it to allow continuation. Paused
and not-yet-started runs are not results. The readout requires all 54 main runs
to complete and audits them; it reports unavailable stress runs separately.

Configuration/protocol/source/runtime changes after registration fail closed.
Use a new protocol/version and fresh outputs for deliberate scientific changes.
Session time/proposal limits can change without changing the scientific contract.
The time budget covers the search phase, including initial scoring; registration
validation, reference loading, and the resume audit precede that timer and add
overhead. It is not an end-to-end process deadline. Numerical libraries run under
a one-thread limit; their versions are recorded.

## Remaining limits

This implementation does not add shuffled-source corpora, independent document
replicates, target-guidance ablation, an end-to-end known-history positive
control, or a broader language sample. Those were conditional future designs
in the parameter review. Additional optimizer seeds cannot substitute for them.
There is no new entropy penalty, feature reweighting, or Klingon exclusion.

The initial full-candidate runtime observations came from the previous repair
and remain approximate. The four-proposal pilots cannot establish production
throughput, natural convergence, or a scientific effect. Raw source/reference
corpora remain at their registered paths, with hashes in the registration.
Preserved source snapshots include pre-existing repository work; no Git commit
or tag is created by this task.
