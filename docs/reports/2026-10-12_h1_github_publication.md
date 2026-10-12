# H1 code and evidence publication, 2026-10-12

The author authorized updating GitHub after a remote check showed that `main`
still pointed to `67cbe26d8582cbc6777051e1353c238bf01734ef`. The local branch had
three unpublished commits, ending at `3bf602fca67de5f5c0bea37ee97dae0415d65421`,
plus the validated H1 repairs and extension as uncommitted files. A successful
authenticated fetch confirmed that there was no competing remote branch update.

This release adds the corrected objective and execution path, the amended E1
readout, the continuous checkpointed form-reward ablation, the production queue,
their regression tests and verification scripts, Decisions 043–047, both analysis
protocols, the corrected study outline, and repair/launch/restart audit reports.
The three earlier commits preserve the original feature freeze and scheduling
history. Existing local edits to the first manuscript and its figure, submission
files, and host-specific legacy monitoring/startup helpers are outside this
publication and remain local.

## Experiment status and question

The active study asks: **How does disabling explicit Latin-form rewards change
fixed-budget convergence and diversity in the selected corpus panel?** It is a
descriptive mechanism comparison for this panel. Both arms retain Latin-informed
proposal guidance. Neither a historical-ancestry verdict nor an unconditional
significance claim is part of this design.

Production launched October 1 and resumed October 12 after a computer restart.
At the publication preflight (23:34:34 UTC), the runner reported `running` with
10,578 committed proposals across 66 jobs. No job had yet completed its registered
1,000-proposal budget. This is operational progress, not a scientific result.
The current state is recorded locally in `matrix_status.json`; these counts are
a dated observation and will become stale.

The production registration is published at
`data/h1_ablation_production_20261001/ablation_registration.json`, with sealed
registration identity:

`012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0`.

Publication does not alter scientific source, protocols, seeds, rewards, inputs,
checkpoint contents, or stopping rules. The README now points to the actual launch
and restart records. Earlier audits retain their historical pre-launch wording.

## Dependency setup and portability

`requirements.txt` now declares SciPy and threadpoolctl, which the registered
launcher already requires. `requirements-h1.txt` records the production numerical
package versions: NumPy 2.4.3, SciPy 1.17.1, and threadpoolctl 3.6.0. The registered
interpreter is CPython 3.14.3 on Windows AMD64. These packaging changes do not
install or upgrade anything in the active environment.

For a separate environment:

```powershell
python -m pip install -r requirements-h1.txt
```

The production registration binds exact runtime, code, input hashes and local
paths. A checkout and these package versions alone do not make another host an
authorized resume of this run. Reconstruct the separately stored corpora and
reference matrices, check their registered hashes, and create a fresh registration
in a new output root for an independent replication. Do not edit the published
sidecar to bypass its validation. See the extension and launch runbooks for
registration, queue execution, pause/resume and readout commands.

## Validation and evidence

`artifacts/h1_publication_20261012/preflight.json` records successful validation
of the existing 66-job registration, its runtime, and the live runner. All 130
source/test/script files listed in the October 1 extension archive still match
their archived LF-normalized hashes. The later queue wrapper still matches its
launch hash, `2c9d72cb497595b4faf87fecfc0f4d764fb1fa048b5a2547719c22442c4d8ed8`.

The publication regression selection includes the previous 129-test suite and
the six queue tests: **135 passed in 71.74 seconds, with no failures or skips**.
Its machine-readable result is
`artifacts/h1_publication_20261012/tests.xml`. The command is:

```powershell
python -m pytest tests/test_h1_checkpointed.py tests/test_h1_ablation_registration.py tests/test_h1_execution.py tests/test_h1_convergence.py tests/test_h1_panel_statistics.py tests/test_h1_matrix_runner.py tests/test_engine_reinforced_v2.py tests/test_engine_reinforced_v3.py tests/test_engine_reinforced_v4.py tests/test_engine_reinforced_v5.py tests/test_long_run_v4.py tests/test_multi_run_supervisor.py -q -p no:cacheprovider --basetemp=data/tmp/h1_publication_20261012_01 --junitxml=artifacts/h1_publication_20261012/tests.xml
```

Use fresh temporary/output paths for a repeat. The earlier real-corpus and
fresh-process resume validations were not rerun as production experiments.

Selected immutable evidence is included from the repair, design, extension,
launch and restart artifact directories: inventories, test XML (including failed
attempts documented in the audits), plans, synthetic-statistics results, pilot
verification receipts, launch checks, and restart checks. The August endpoint
analysis tables are also included with their report and analysis code. These
historical/pilot records remain excluded from the production readout.

The exact publication file list and LF-normalized text hashes are in
`artifacts/h1_publication_20261012/publication_manifest.json`. Source archive ZIPs,
bulk pilot states, raw corpus dumps and the growing production checkpoints are
retained locally. Inventory entries for those files identify evidence not bundled
in Git; they are not claims that every listed artifact is downloadable here.
Git ignores new bulk artifacts while retaining the explicitly selected receipts.

The original local archives remain unchanged. Their README and dependency files
describe the historical snapshots; this publication updates only packaging and
current-status documentation on top of the validated scientific implementation.
