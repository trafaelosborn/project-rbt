# H1 production resumed after computer restart

Date: 2026-10-12 (America/Denver). The author reported restarting the computer
and requested that the running experiment be restarted. This action resumed
the existing registered matrix; it did not create a new scientific run.

## Recovery checks

The previous runner and its recorded worker PIDs were absent. The old status
file still said `running`, with its last update at 2026-10-02 01:51:28 UTC
(October 1, 19:51:28 Denver). That stale status was not treated as a live process.
No PAUSE files were present, and the matrix lock and all 66 job locks were
available. About 372 GiB of disk space was available.

The registered implementation, runtime, source corpora, reference inputs,
protocol, and engine settings passed the existing registration validation.
The queue runner source matched the originally launched source hash:
`2c9d72cb497595b4faf87fecfc0f4d764fb1fa048b5a2547719c22442c4d8ed8`.

All 66 committed checkpoint/proposal histories and their current state hashes
passed integrity checks. Each current state still recorded the approved
20,000-token measurement and objective identity. There were **8,497 committed
proposals out of 66,000 (12.87%)**, all 66 jobs had started, and none had completed
its 1,000-proposal budget. These committed results were retained. An uncommitted
proposal interrupted by shutdown is replayed by the existing recovery policy;
this check does not claim that uncommitted computation survived.

Before restarting, the old matrix status and top-level stdout/stderr logs were
copied into `artifacts/h1_restart_20261012/20261012T215738Z`. The prior runner's
session directory and scientific artifacts remain in place.

## Relaunch and observed progress

At 2026-10-12 21:59:06 UTC (15:59:06 Denver), the unchanged queue runner was
relaunched as a hidden, below-normal-priority background process:

```powershell
python -u scripts/run_h1_ablation_matrix.py --output-root data/h1_ablation_production_20261001 --workers 2 --session-proposals 100 --session-seconds 1800
```

Runner PID: 42312. Runner session: `20261012T215910Z_2642d88e`.
The French seed-42 full and disabled workers started from their existing
proposal-200 checkpoints, with PIDs 43352 and 43368 respectively. The dispatcher
rebuilds its queue from registered jobs and their saved positions; the interrupted
Portuguese jobs retain their own partial checkpoints for their next turns.

At 22:00:02 UTC, both active jobs had independently advanced to proposal 209,
after the engine's resume audit. The matrix had reached 8,515 committed proposals,
the two current states still measured 20,000 tokens, and the runner reported
`running`. Thus the restart was verified by new committed progress, not merely
process creation. No scientific effect or arm comparison was inferred.

The registration remains:
`012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0`.
No scientific code, parameters, seeds, data, or stopping rules were changed.
No reboot/startup service or scheduled reminder was installed.

## Evidence

Recovery evidence is under `artifacts/h1_restart_20261012/20261012T215738Z`:

- `matrix_status_before.json` and copied original runner logs.
- `preflight.json`: registration validation, 66 checkpoint identities, retained
  proposal counts, and queue source identity.
- `process_launch.json`: exact launch settings, timestamp, and runner PID.
- `runner_stdout.log` / `runner_stderr.log`: new background process logs.
- `resume_check_220002.json`: independently checked post-restart checkpoints and
  verified advancement in both workers.

Current operational status remains at
`data/h1_ablation_production_20261001/matrix_status.json`. New worker logs and the
copied queue source are in its new `runner_sessions` directory. Full state/score
replay occurs when each job resumes and again in the final registered readout.

Related: [original launch](2026-10-01_h1_production_launch.md) and
[extension audit/runbook](2026-10-01_h1_extension_audit.md).
