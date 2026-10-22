# H1 production recovery and resume, 2026-10-19

The author requested that the existing experiment resume. This continues the
October 1 registered matrix, including the October 12 continuation; it does not
start a new scientific run or replace any committed trajectory.

## Why the previous session stopped

The October 12 runner stopped at 2026-10-13 02:14:16 UTC (October 12, 20:14:16
America/Denver). Its final status was `failed`, with reason
`child_failed:sumerian__corpus_42__search_1729__full:exit_1`.

The failed worker's traceback records `PermissionError: [WinError 5] Access is
denied` from `os.replace` in `atomic_json`, while replacing that job's
`checkpoint.json`. The checkpoint still committed proposal 156. The immutable
proposal-157 record had already been written, but the checkpoint did not yet
commit it. The failing call was file publication, not feature scoring.

The runner then created a PAUSE file for the other active worker,
`sumerian__corpus_42__search_1729__disabled`. That worker paused normally at
proposal 152. The matrix retained **13,605 committed proposals out of 66,000**.
All 66 jobs had started; none had completed its 1,000-proposal budget.

The responsible Windows process or permission interaction was not identified.
The checkpoint did not have the read-only attribute, and the same registered
writer succeeded on recovery. This establishes successful recovery from the
observed write failure; it does not establish that such a failure cannot recur.

## Preflight and preservation

Evidence is under
`artifacts/h1_restart_20261019/20261019T152515Z`.

Before resuming, the old matrix status, both relevant worker logs, the failed
job's checkpoint and uncommitted proposal, and the runner-created PAUSE contents
were copied there. No scientific history was removed or rewritten manually.

The existing registration validated against the current implementation, runtime,
source corpora, reference inputs, protocol and engine settings. The sealed
registration identity remains:

`012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0`.

The unchanged queue wrapper matches its launch hash:

`2c9d72cb497595b4faf87fecfc0f4d764fb1fa048b5a2547719c22442c4d8ed8`.

The matrix lock and all 66 job locks were available. Every committed proposal
chain, counter ledger, checkpoint identity and current state hash passed the
existing integrity checks. Every current state recorded the registered objective
identity and 20,000-token measurement. Full feature/reward replay is performed
by the existing runner when each individual job resumes. Approximately 363 GiB
of disk space was available.

## Recovery of the failed job

The matrix lock was held while the failed Sumerian full-arm job was resumed
alone with a two-proposal session limit. The unchanged registered CLI performed
its normal independent score replay before continuing. Between 15:27:54 and
15:28:15 UTC on October 19, it advanced from proposal 156 to 158 and exited zero.

The saved proposal-157 file remained byte-for-byte identical. The existing
immutable-evidence check therefore accepted the deterministic replay of the
previously uncommitted proposal. Both new checkpoint replacements succeeded.
These two proposals are production work within the original budget, not a pilot
or an additional scientific replicate.

Afterward, the sole PAUSE file was checked against its archived hash, content
and the old runner's explicit list, then removed under the job lock. No other
pause requests existed. The queue restarted with 13,607 committed proposals.

## Queue relaunch and verified progress

An initial hidden-process launch at 15:29:00 UTC returned PID 33716 but did not
persist, create a runner session, or update matrix status. Its stdout/stderr
were empty. That launch was not counted as a successful restart; its receipt
and logs are retained with the `attempt_01` suffix.

At **2026-10-19 15:30:24 UTC (09:30:24 America/Denver)**, the queue was launched
with its hidden parent process waiting for the runner, keeping the execution
session open. Runner PID: **24108**. Runner session:
`20261019T153027Z_6bcf0729`.

```powershell
python -u scripts/run_h1_ablation_matrix.py --output-root data/h1_ablation_production_20261001 --workers 2 --session-proposals 100 --session-seconds 1800
```

The runner uses two hidden, below-normal-priority workers with one numerical
library thread each. These are the same operational limits as before. The
1,000-proposal scientific budget per job and all registered settings are intact.

At 15:31:19 UTC, the runner held the matrix lock, reported `running`, and both
French seed-42 workers had advanced from proposal 300 to 310. Their checkpoint
hashes, objective identities and 20,000-token measurements verified. The matrix
status recorded **13,627 / 66,000 proposals (20.6%)**, with no completed jobs;
runner stderr was empty. Registration validation also passed after relaunch.
This confirms new committed progress, not merely process creation.

The scientific code, parameters, seeds, input data and stopping rules were not
changed. The failure logs and uncommitted evidence were retained. No conclusion
about either arm's scientific effect is drawn from this operational recovery.

## Evidence index

- `preflight.json`: registration, all 66 checkpoint/chain checks, state identities,
  retained progress, available locks, disk space and pause-file provenance.
- `matrix_status_before.json`, the two copied worker logs,
  `failed_job_checkpoint_before.json`, `failed_job_uncommitted_000000157.json`,
  and the copied PAUSE text: evidence of the original failure and retained state.
- `failed_job_resume_command.json`, `failed_job_resume.log` and
  `failed_job_resume_result.json`: exact bounded production recovery and successful
  replay of the orphan proposal.
- `queue_resume_preparation.json`: verified pause removal and pre-queue positions.
- `process_launch_attempt_01.json` and associated attempt logs: unsuccessful first
  background launch, preserved separately.
- `process_launch.json`, `runner_stdout.log`, `runner_stderr.log`: actual relaunch
  settings and ongoing top-level logs.
- `resume_check.json`: independently verified advancement in both new workers,
  post-resume registration identity and matrix-lock ownership.

Current status remains at
`data/h1_ablation_production_20261001/matrix_status.json`. Per-worker logs and
the copied runner source are in the new `runner_sessions` directory. The existing
queue will invoke the registered independent readout when all 66 jobs complete.

Related: [October 12 resume](2026-10-12_h1_production_resume.md),
[original launch](2026-10-01_h1_production_launch.md), and
[extension audit](2026-10-01_h1_extension_audit.md).
