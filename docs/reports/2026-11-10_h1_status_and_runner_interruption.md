# H1 status and unexplained runner interruption, 2026-11-10

The author requested a progress check. The experiment is **inactive**, despite
the old queue status still saying `running`. This audit records the interruption
and retained progress. No restart or scientific change was made during the check.

## Observed activity and interruption

The October 31 launch began at 23:36:02 UTC (17:36:02 America/Denver), as runner
PID 41084, session `20261031T233606Z_1134eb6d`. The last heartbeat was
**October 31 at 23:38:51 UTC (17:38:51 Denver)**, approximately 2 minutes 49 seconds
after launch. There are only two worker-start events and no recorded worker exit
or clean queue shutdown. The launcher's expected `process_exit.json` is absent.
Both worker logs end without an exception traceback; runner stdout and stderr
are empty. The saved status records zero I/O retries, zero worker recoveries, and
no shutdown errors. It does not record another checkpoint-access failure.

Host inspection at November 11 06:00:37 UTC (November 10 23:00:37 Denver) found
no matching experiment processes. Windows reported its latest boot as October
31 00:07:51 Denver time, before the experiment launch. A subsequent computer
reboot therefore does not explain this interruption. The exact process
termination mechanism is not established. Parent-process or execution-session
teardown is a hypothesis, not a diagnosed cause.

The October 31 live check correctly established initial progress. It did not
establish sustained execution: the last heartbeat is about a minute after the
last published receipt. Reliable background-process lifetime remains unresolved;
the previous successful launch must not be described as continuous operation
through November 10.

## Retained progress and integrity

All 67 OS locks (one matrix plus 66 jobs) were available and held together during
checkpoint inspection. The full registration comparison passed, including the
scientific implementation, runtime, inputs, protocol, and job settings. Its
sealed identity is unchanged:
`012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0`.
Both operational source hashes exactly match the tested October 22 repair.

The checkpoints retain **23,480 / 66,000 proposals (35.58%)**, an increase of
**88** since the October 31 restart. None of the 66 jobs has completed its
1,000-proposal budget. The two changed jobs are:

| Job | October 31 preflight | Current committed checkpoint |
|---|---:|---:|
| French, corpus/search seed 42, full | 562 | 606 |
| French, corpus/search seed 42, disabled | 562 | 606 |

Their complete committed proposal chains, contracts, counter ledgers, and current
state hashes passed validation today. The other 64 checkpoints have exactly the
same verified hashes as the October 31 full audit; their entire proposal chains
were not redundantly reread today. All 66 checkpoint contracts, budget bounds,
and current-state hashes were checked. No uncommitted proposal records or PAUSE
files were found. Approximately 185.6 GiB of disk space remained.

The stale `matrix_status.json` total of 23,392 excludes the 88 proposals committed
during the interrupted sessions. That is consistent with its documented policy:
counts are published after child sessions exit. Its historical contents were
preserved rather than manually rewritten to manufacture a shutdown event.

## Evidence and scope

Evidence is under `artifacts/h1_status_20261110`: `audit.py` and `audit.json`
record the checks and all job hashes; `host_inspection.json` records the process
and boot-time observation. The stale matrix status, both worker logs, empty
top-level logs, and the two changed checkpoints were copied there. The publication
manifest identifies the exact report/evidence files and LF-normalized hashes.
Bulk production state remains local.

This was a status and integrity audit, not a new code test, automated monitor,
or restart. The scientific results are incomplete; no arm comparison or new
scientific conclusion is claimed. Prior records:
[October 31 restart](2026-10-31_h1_production_resume.md) and
[October 22 checkpoint-access repair](2026-10-22_h1_checkpoint_io_repair.md).
