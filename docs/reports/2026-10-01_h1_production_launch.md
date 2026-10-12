# H1 form-reward ablation production launch

The author authorized launch with “LFG!” after reviewing readiness. Production
started on 2026-10-01 at 20:39:00 UTC (14:39 America/Denver).

## Registered experiment

Production root: `data/h1_ablation_production_20261001`.
Registration: `ablation_registration.json` in that directory.
Registration SHA-256:
`012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0`.

The 66 jobs implement Decision 047: nine main source languages and two stress
sources, three search seeds, full/disabled form-reward arms, 1,000 proposals per
job, 16 candidate slots, and 20,000 measured tokens. Corpus seed is 42 throughout.
This is 66,000 proposals and at most 1,056,000 evaluated candidate slots; no-payload
and invalid-budget slots do not become completed candidate evaluations.

The question, estimands, calibration, feature set, and engine settings are
unchanged from the validated extension. The production plan was compared exactly
with the archived plan before registration; all 145 archived source files still
matched their recorded hashes. No pilot was promoted into production.

## Operational runner

`scripts/run_h1_ablation_matrix.py` is a new operational wrapper, outside the
frozen scientific implementation. It launches only registered job IDs through
the existing validated CLI. Two workers run with below-normal priority, hidden
windows, and one numerical-library thread each. The host reports 32 logical CPUs;
about 354 GiB disk space was free at the initial check. An optional psutil memory
query was unavailable because psutil is not installed; no dependency was added.

Each child runs at most 100 proposals or 1,800 search-phase seconds per session,
then unfinished jobs rejoin the queue. These are operational pauses, not changes
to the 1,000-proposal scientific budget. The engine saves every committed proposal
and replays saved state on resume. Preflight/loading/audit overhead and an
in-flight proposal can extend a session beyond the soft time limit.

The runner stops new dispatch on any failed child, a global/job PAUSE file, or
disk space below 10 GiB. It requests boundary pauses for remaining active jobs
and records the reason and any PAUSE files it created. It does not discard,
replace, or reinterpret failed scientific trajectories. Once all 66 jobs finish,
it invokes the registered readout CLI, which independently audits the evidence.
A readout failure is a failure state, not successful completion.

The background runner was launched as PID 8176. Its initial worker PIDs were
39224 (French/full) and 2888 (French/disabled). PIDs are historical identifiers;
use current `matrix_status.json` and process state when checking later.

At 20:39:36 UTC, both initial jobs had saved proposals (5 full, 6 disabled), the
runner was running, and the runner stderr log was empty. This establishes that
the production launch executed and progressed, not merely that a process was
created. It is not a scientific result or an assessment of the arm contrast.

## Queue validation and evidence

- Existing extension validation: 129 passing scientific/controller regression
  tests and the previously archived real-corpus and fresh-process resume checks.
- New queue validation: six tests passed in 0.73 seconds, covering fair rotation,
  failure drain, pause-before-launch, low-disk pause, completed-job skipping, and
  automatic analysis dispatch after the whole production matrix completes.
- `artifacts/h1_launch_20261001/queue_pilot` ran four actual French/Finnish pilot
  jobs with two workers and two-proposal sessions. All reached their four-proposal
  pilot budgets after rotation/resume, and the runner reported complete, 4/4.
  This pilot is excluded from scientific readout.
- `artifacts/h1_launch_20261001/preflight.json` records the unchanged validated
  plan/source checks. `process_launch.json` records the exact background launch.
- The runner copies its source into each `runner_sessions/<id>` directory before
  dispatch and stores its source hash, commands, child PIDs, logs, exit codes,
  proposal positions, and queue transitions. A launch supplement archives the
  wrapper, tests, launch report, preflight receipt, and registration separately
  from the earlier source snapshot.

## Observing, pausing, and resuming

`matrix_status.json` at the production root is updated about every two seconds.
It includes active job IDs/PIDs/logs, committed proposal counts for all jobs,
completed-job count, pending queue, and failure/pause reasons. Per-job
`checkpoint.json`, `proposals/`, `states/`, and `sessions/` preserve scientific
evidence. Worker logs and eventual readout files live in `runner_sessions/`.

To request a pause, create a file named `PAUSE` at the production root. The runner
drains active jobs at proposal boundaries. Before resuming, resolve the reason
for stopping and remove that request plus only the runner-created PAUSE files
listed in its status; preserve any independently requested job pause. Relaunch:

```powershell
python scripts/run_h1_ablation_matrix.py --output-root data/h1_ablation_production_20261001 --workers 2 --session-proposals 100 --session-seconds 1800
```

Use a hidden background process for unattended operation. The exclusive matrix
lock rejects a second runner for the same root; each scientific job also has its
own lock. Do not edit registered scientific code/data while production is active.
The run depends on the local computer remaining awake and running. Checkpoints
survive interruptions, but no automatic reboot/startup service or Codex reminder
was installed. Runtime will be measured from production progress; short earlier
benchmarks do not provide a reliable completion time for all sources.

Related documentation: [extension audit](2026-10-01_h1_extension_audit.md),
[Decision 047](../decisions/047_h1_continuous_search_and_form_reward_ablation.md).
