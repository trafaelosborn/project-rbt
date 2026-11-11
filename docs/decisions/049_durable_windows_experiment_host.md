# Decision 049: Durable Windows ownership of the H1 experiment

Date: 2026-11-10 (America/Denver). The author requested investigation of repeated
stops, a fix that keeps the experiment running, resumption, and complete records.
This adds an operational host and a Windows scheduled task. The registered
scientific implementation, inputs, runtime, budget, and registration are unchanged.

## Diagnosis and attribution limits

The October 22/31 unexplained interruptions differ from the earlier documented
checkpoint-access exceptions. The October 31 session stopped updating after less
than three minutes, with no recorded exception, child exit, parent exit receipt,
or subsequent computer reboot. Initial successful work did not establish durable
execution; the previous launch checks were insufficient for that claim.

A read-only native API probe of an escalated execution command found Windows job
limit flags **0x2800**, including **JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (0x2000)**.
The old hidden `Start-Process` launch inherited that job. Keeping its PowerShell
parent waiting did not move it outside the owning job. A separate probe confirmed
that an ordinary hidden child inherited 0x2800, while a scheduler-launched child
had a different parent and job flags zero. Both probes completed normally while
the execution environment remained available: ending a command alone did not
reproduce the historical interruption.

Windows documents that ordinary child processes inherit job membership and that
closing the final handle of a kill-on-close job terminates its members. These
measurements establish a concrete lifetime dependency in the old launcher. They
do not establish the exact historical actor that closed or terminated the old
execution session. A narrow local execution-log search returned no matching
historical shutdown rows. No unproven antivirus, Windows Update, or scoring
failure is asserted. See Microsoft's [job-object documentation](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)
and [limit definitions](https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information).
OpenAI documents background terminals as thread-associated resources with
[explicit cleanup/termination operations](https://learn.chatgpt.com/docs/app-server#clean-background-terminals),
not as a guarantee of persistent service execution.

## Durable ownership and operational limits

1. Windows Task Scheduler launches `scripts/run_h1_scheduled.py` directly under
   the current user's existing interactive token, at limited privilege. Its
   parent is the Windows task infrastructure, independent of this chat's shell.
   No password is stored. The task runs while the user is signed in, including
   when the workstation is locked, and triggers again at the user's next sign-in
   after reboot. Signed-out execution is not claimed: Windows denied the initial
   S4U probe, and that attempt is preserved rather than bypassed.
2. A five-minute repeating trigger checks for interrupted work. Task Scheduler's
   `IgnoreNew` policy and a process-owned host lock exclude duplicate hosts.
   Existing matrix/job locks remain in force. The task has no execution-time
   ceiling (`PT0S`), avoiding the scheduler's default 72-hour limit. It does not
   require idle state or network availability, and battery transitions do not
   terminate it. The task and scientific workers use below-normal priority.
   See [execution-time limits](https://learn.microsoft.com/en-us/windows/win32/taskschd/tasksettings-executiontimelimit).
3. Before launching work, the host pins its operational sources and original
   scientific registration and verifies every committed proposal chain,
   contract, counter ledger, and current state under all 67 writer locks. An
   existing writer defers the attempt without reading checkpoints or consuming
   the recovery allowance. The unchanged scientific worker still independently
   replays accepted features and rewards on each job resume.
4. A persisted `running` host state left by abrupt process termination permits
   **at most three recovery launches** across host processes. Progress and
   successful child sessions do not reset that allowance. Exhaustion is latched
   for review. New host sessions retain the prior session ID, preflight records,
   exact source snapshots, logs, and interruption count.
5. Scientific exceptions, integrity errors, changed identities, existing failed
   matrix status, normal PAUSE requests, and completion are terminal states.
   Repeated scheduler triggers do not clear those states or PAUSE files. The
   October 22 narrowly classified checkpoint-access retry policy is unchanged.
   This host does not turn arbitrary failures into retryable events.
6. Before workers start, the host attaches itself to its own non-inheritable,
   kill-on-close Windows job. Its descendants inherit membership. Unexpected
   host death ends those workers too, preventing orphan checkpoint writers.
   The job handle remains open until OS process teardown; no unrelated process
   is included or terminated. Native Windows pilot testing exercises this path.
7. While running, a scoped `SetThreadExecutionState` request prevents automatic
   idle sleep without keeping the display awake or changing the power plan.
   User-requested sleep, sign-out, shutdown, or reboot remain possible. The
   request is released on normal exit or process death. CPU activity alone is
   not sufficient to prevent idle sleep; see [Microsoft's API documentation](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadexecutionstate).

## State and operator controls

The host stores sealed configuration and state in `<production-root>/task_host`,
with immutable session evidence below `task_host/sessions`. It invokes the
existing matrix runner in-process; it does not alter the engine or scientific
checkpoint writer. A caught failure is recorded as a terminal host state, even
when the task process exits normally. Task exit code alone is therefore not a
scientific success indication.

Use `scripts/manage_h1_task.ps1 -Action Status` with the exact config/task name.
It reports the actual scheduler state, host state/reason, interruption count,
and queue heartbeat age. The raw historical `running` string by itself is not
evidence of a live process. Proposal counts in `matrix_status.json` remain a
lower bound during active child sessions. Never poll active job checkpoints.

To stop cleanly, create the production root's `PAUSE` file, then wait for the
queue and workers to exit. Disabling the task prevents future triggers but does
not itself stop the active run. The management script refuses to remove a running
task and refuses to alter a same-named task with different action/configuration.
Resuming a latched pause/failure or exhausted allowance requires an explicit,
documented review; scheduled triggers cannot erase it.

The installer is a local PowerShell script. The integration harness uses
process-scoped `RemoteSigned` to execute it where the legacy shell defaults to
Restricted; it does not change machine/user execution policy, ACLs, antivirus,
or enterprise policy. The failed initial invocation is retained.

## Validation and remaining limits

Unit tests cover persistent recovery limits, completion/pause/error latches,
duplicate exclusion, external-writer deferral, changed identities, corrupt
history/state, and immutable configuration. A fresh, excluded native Windows
pilot deliberately terminates only its scheduled host, verifies descendant
exit, waits for an automatic timer-triggered recovery, and compares French
full/disabled trajectories with uninterrupted references. Detailed outcomes,
including failed attempts, are recorded in the repair report.

This removes the measured chat-process lifetime dependency and tests recovery
under controlled abrupt termination. It does not promise survival of every OS,
disk, or hardware failure, nor execution while the computer is powered off or
the user is signed out. Scientific failures remain visible and stopped.
