# H1 durable Windows host repair and restart

Authorized on 2026-11-10, America/Denver; execution receipts use UTC on November
11. The author requested a diagnosis of recurring stops, a durable fix, restart,
and complete documentation. This report supplements
[Decision 049](../decisions/049_durable_windows_experiment_host.md).

## What was wrong, and what is established

The old launch method left the experiment inside the chat execution environment's
Windows process job. Native inspection found flags **0x2800**, including
kill-on-job-close. An ordinary hidden `Start-Process` child inherited the same
flags. The previous parent-wait workaround did not remove that ownership.
Closing or terminating the owning job can therefore end the parent, queue, and
workers without Python cleanup or traceback.

The most recent historical run's final heartbeat was October 31 at 23:38:51 UTC,
less than three minutes after launch. No clean exit receipt was written, and the
computer's reported boot predates the launch. A targeted check of 13 Application
and 65 System events around the stop found no messages identifying Python,
PowerShell, Codex, or OpenAI; no matching Task Scheduler events were available.
A narrow local execution-log query likewise found no matching shutdown rows.
The exact historical actor remains unproven. The repair addresses the measured
ownership defect without assigning unsupported blame to updates or antivirus.

In the first lifetime probe, both the old-method child and the independent
scheduler child completed 90 seconds of heartbeats. The former retained 0x2800;
the latter had a different Windows parent and job flags zero. Ending the launch
command alone did not reproduce the historical stop. The stronger recovery
tests below deliberately terminate a real pilot host.

## Repair

New operational files are `scripts/run_h1_scheduled.py`,
`src/control/h1_windows_host.py`, and `scripts/manage_h1_task.ps1`. The existing
matrix queue and its October 22 I/O helper are unchanged. The frozen scientific
engine, checkpoint writer, inputs, seeds, scores, candidate budgets, runtime,
and registration are unchanged.

Windows Task Scheduler owns the new host independently of this chat. The task
uses the signed-in user's limited token, has no execution-time cutoff, ignores
duplicate starts, checks every five minutes, and triggers at user sign-in. It
requires neither idle state nor network access and does not stop on battery
transition. It can continue while the workstation is locked; it cannot run
while the user is signed out. No password or global policy change was introduced.

The host pins all five operational source identities and the original scientific
registration. Before work or recovery it verifies the committed histories and
current states with every writer excluded. It persists a maximum of three
automatic interruption recoveries across host instances. Science/integrity
errors, PAUSE requests, completion, and exhausted recovery limits are latched;
timer triggers cannot erase them. The existing checkpoint-access recovery policy
is unchanged. All attempts and source snapshots are retained.

A private Windows job ends the host's descendants if the host dies, preventing
orphan writers. A scoped execution-state request prevents automatic idle sleep
during work while allowing display sleep and explicit user sleep/shutdown. It
does not change the power plan or prevent a deliberate shutdown.

## Failed attempts and corrections

Evidence is under `artifacts/h1_durable_runner_20261110`.

- Windows denied registration of a password-free S4U background-account probe.
  No task was created. A limited interactive-token task succeeded instead;
  signed-out/reboot-before-sign-in execution is not claimed.
- The first native integration pilot (`pilot_01`) registered only fresh excluded
  pilot roots, then stopped before installing its task because Windows PowerShell
  defaults to Restricted script execution. Its failed invocation is preserved.
  The harness now uses process-scoped RemoteSigned for the locally authored
  installer, without changing machine/user or enterprise policy.
- The first task export had UTF-8 bytes but an XML declaration specifying UTF-16.
  This did not affect the installed task or successful `pilot_02`, but direct
  parsing caught the artifact defect before production launch. The installer now
  exports UTF-16. The native pilot was repeated with that final installer.
- A settings inspection initially expected an explicit RunLevel XML element;
  Windows omits default-valued elements. Actual CIM inspection confirmed Limited
  privilege, Interactive logon, priority 7, `PT0S`, IgnoreNew, StartWhenAvailable,
  and RunOnlyIfIdle=false. The default StopOnIdleEnd field does not impose an idle
  condition when RunOnlyIfIdle is false.

## Automated and native validation

`tests_01.xml`: **27 focused tests passed** in 3.62 seconds.
`tests_02.xml`: **177 distinct tests passed** in 68.96 seconds, with no failures
or skips. This is the previous 163-test selection plus 14 host tests, not a sum
of overlapping runs. After the export-only correction, `tests_03.xml` repeated
the 27 focused tests successfully in 2.54 seconds.

The broad command was:

```powershell
python -m pytest tests/test_h1_scheduled.py tests/test_h1_io.py tests/test_h1_io_recovery.py tests/test_h1_matrix_runner.py tests/test_h1_checkpointed.py tests/test_h1_ablation_registration.py tests/test_h1_execution.py tests/test_h1_convergence.py tests/test_h1_panel_statistics.py tests/test_engine_reinforced_v2.py tests/test_engine_reinforced_v3.py tests/test_engine_reinforced_v4.py tests/test_engine_reinforced_v5.py tests/test_long_run_v4.py tests/test_multi_run_supervisor.py -q -p no:cacheprovider --basetemp=data/tmp/h1_scheduled_02 --junitxml=artifacts/h1_durable_runner_20261110/tests_02.xml
```

The reproducible integration harness is `scripts/verify_h1_scheduled.py`. Each
invocation requires a fresh root and registers excluded four-job French/Finnish
pilots with four proposals per job and two candidates per proposal. It installs
a temporary Windows task, starts it once, waits for actual proposal records,
then terminates only that pilot's host with exit code 73. Native process handles
verify descendant exit. Recovery must occur from the scheduled timer, without
a manual start command. The harness then compares both French reward arms with
fresh uninterrupted references, checks that completion cannot relaunch work,
and removes its temporary task after completion.

`pilot_02` passed: one automatic recovery, four jobs completed, identical French
proposal candidate/choice/state/counter trajectories, saved-state bytes, final
RNG state, accepted-state count and stagnation count. Different output roots
intentionally produce different contract hashes. Pilot data never enter
production analysis. Each pilot's `verification_results.json` preserves all
task commands, captured outputs, the controlled termination, final states, and
comparison results. The initial standalone lifetime-probe task was also removed
after its normal completion.

The final-installer `pilot_03` also passed the same complete test. Its host was
terminated at 06:56:15 UTC; the scheduler started its recovery at 06:56:54 UTC
without a manual start, and all four jobs finished at 06:58:11 UTC. Both French
trajectory comparisons and the completion latch passed again. Its temporary
task was removed. The final exported XML parses directly, and
`final_pilot_verification.json` confirms that production's pinned operational
sources exactly match this final pilot. `final_pilot_task_settings.json` records
the actual CIM settings while that task was active.

## Production configuration and operator runbook

The original registration identity is
`012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0`.
The retained starting checkpoint total is 23,480 / 66,000 proposals. The new
host retains two workers, one numerical-library thread per worker, and the
100-proposal / 1,800-second child-session limits.

Production task name: `ProjectRBT-H1-Production-20261001`.
Config: `data/h1_ablation_production_20261001/task_host/config.json`.

From the repository root, inspect live status with:

```powershell
& .\scripts\manage_h1_task.ps1 -Action Status -Config .\data\h1_ablation_production_20261001\task_host\config.json -TaskName ProjectRBT-H1-Production-20261001
```

The status command reports actual scheduler state and queue heartbeat age as well
as host state/reason and the persisted recovery count. A stale `running` string
alone is not evidence of live work. Queue totals still update at child-session
boundaries; do not poll active job checkpoint files.

For a clean stop, create `data/h1_ablation_production_20261001/PAUSE` and wait for
workers to finish their current proposal. `-Action Disable` prevents future task
triggers without terminating the current run. `-Action Remove` refuses a running
task. A latched pause/failure or exhausted recovery allowance requires explicit
review and preservation of the old host state before any rearm; scheduled
triggers cannot silently resume it.

At **2026-11-11 07:00:54 UTC (00:00:54 America/Denver)**, production task
`ProjectRBT-H1-Production-20261001` launched host PID **37612**. The launching
command returned normally. At 07:01:41 UTC, a host process inspection identified
its parent as **Windows svchost.exe PID 2640**, and confirmed that this production
task was the only remaining `ProjectRBT-H1-*` task. The Python host had no visible
main window and ran below normal priority. Its full checkpoint preflight runs
before workers are dispatched; during that stage an absent new host state and
the old stale matrix heartbeat do not indicate a failed start. The actual task
and process state distinguish validation from active search.

The full audit finished at **07:04:35 UTC**: all 66 committed histories and
current-state hashes passed, retaining **23,480** proposals. The new matrix
session is `20261111T070438Z_a7d9798b`, running in host PID 37612. French
corpus/search seed 42 workers began at 606, full PID 61872 and disabled PID 24448.
Host session: `20261111T070055Z_bd8531e6`.

At **07:14:15 and 07:14:17 UTC**, respectively, those workers exited normally
at **706**, both with exit code zero. The queue retained **23,680 / 66,000
proposals (35.88%)**, adding 200 committed proposals, and started the next pair
(French search seed 1729, full PID 30672 and disabled PID 49400) from their saved
500-proposal checkpoints. At 07:14:42 UTC the same Windows-owned host had a
fresh heartbeat, zero host interruption recoveries, zero worker recoveries,
zero I/O retries, and no shutdown errors. It had remained alive almost fourteen
minutes after its launching command returned. This verification includes a
successful worker-session handoff, not merely initial process creation.

At 07:16:28 UTC, the scheduler still reported Running and the queue heartbeat
was under two seconds old. Its last-trigger result was `0x800710E0`
([ERROR_REQUEST_REFUSED](https://learn.microsoft.com/en-us/windows/win32/debug/system-error-codes--4000-5999-)),
while the original host PID and session remained active. This is consistent with
IgnoreNew rejecting overlapping timer starts; it is not the live host's exit
code. Assess the actual scheduler state, matching process identity, and fresh
queue heartbeat together, rather than interpreting LastRunTime/LastTaskResult
as the lifetime of the existing instance.

The `production` evidence directory contains copied configuration, valid task
XML, install receipt, complete preflight, launch-state snapshot, and initial
matrix status. Timestamped `production_observation_*.json` files record safe
observations without opening active checkpoints. Actual task/process ownership
and final scheduler status are retained separately. Active proposal filenames
demonstrate additional work but are not counted as independently verified
checkpoint totals. No completed scientific arm comparison is claimed.

`source_inventory.json` records the final source hashes and archive
`validated_operational_source.zip`, SHA-256
`46f33b12d033af27a6d135840cb45a528adc2ce257dde0a19ec52a683164cf62`.
The exact publication list and per-file hashes are in `publication_manifest.json`.
It includes the failed attempts, tests, pilot receipts, source archive, production
receipts, Decision 049, and this report. Ongoing logs and bulk scientific state
remain local. The prior unrelated manuscript edits are outside this repair.
