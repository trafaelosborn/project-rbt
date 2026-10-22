# Decision 048: Windows checkpoint access and operational recovery

Date: 2026-10-22. The author authorized repairing the repeated checkpoint-access
failures and continuing the existing experiment. This changes the operational
queue, not the registered scientific implementation or Decision 047 protocol.

## Problem and evidence

On October 12, a worker failed while atomically replacing its checkpoint. On
October 19, the queue failed while reading a checkpoint during a worker session.
Both failures were Windows access errors. Saved work survived, but the queue
stopped. Its exception handler also omitted the last four proposals committed
while workers drained, leaving its final status below the saved total.

A real Windows test reproduced a denied checkpoint replacement while an ordinary
reader held the destination open. An initial attempt using FILE_SHARE_DELETE
still failed the replacement test on this host and was abandoned. The underlying
sharing options are described in Microsoft's
[CreateFileW documentation](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew).
That documentation is not evidence that the attempted sharing scheme worked;
the retained failed test shows that it did not. The original interfering process
was not identified, so this repair addresses the observed failure class without
claiming exclusive attribution to the old polling loop.

## Operational policy

1. The queue reads a job checkpoint only before launch or after that child exits,
   while holding the same per-job OS lock used by the registered writer. It does
   not poll active workers' checkpoint files. An existing live/orphan writer
   prevents a queue read and causes a visible failure rather than concurrent access.
2. Windows access/sharing/lock errors (WinError 5, 32, 33, or EACCES without a
   WinError code) receive at most ten operation attempts, with 5.55 seconds total
   backoff. This applies to operational reads, status writes, pause creation and
   recovery verification. Other errors, including invalid JSON, disk-full errors
   and evidence-integrity failures, propagate immediately.
3. A failed child is eligible for recovery only when its final traceback line
   identifies a Windows access error at that exact job's `checkpoint.json`.
   Its committed history, contract, counters and current state must validate
   under the job lock before another session is queued. The unchanged child
   performs its full scientific replay again on resume. Existing orphan evidence
   is reused only if deterministic replay matches it exactly.
4. At most three such recoveries are allowed per job per matrix invocation.
   The allowance is not reset by progress. A persistent failure stops dispatch;
   unrelated failures and corrupted evidence are never classified as recoverable.
5. Every failed attempt, recovery count, verified checkpoint identity and I/O
   retry is recorded. No failed log or scientific trajectory is deleted. Both
   reward arms have the same recovery policy.
6. On a queue exception, workers are drained and their final committed positions
   refreshed. A secondary status-write failure must not skip draining. If pause
   creation itself persistently fails, only this runner's own live children are
   terminated; their committed checkpoints remain the recovery boundary.

## Scientific identity and observability

The original registration remains sealed and unchanged. The engine, checkpoint
writer, RNG behavior, data, seeds, scoring, accepted-state accounting, budgets,
readout and stopping rules are unchanged. No runtime monkeypatch or edited
registration is used to bypass source validation. The changed queue and new
`src/control/h1_io.py` are outside the frozen source set. Each runner session
archives both operational sources and records their hashes and policy version.

The status heartbeat still updates roughly every two seconds, but proposal counts
now reflect the last exited worker session. They are a lower bound while workers
are active. A session may run up to 100 proposals or 1,800 search-phase seconds,
plus initialization/audit and an in-flight proposal. This is not a stalled job.

For live observation, read **`matrix_status.json`** and worker/session logs. Do
not repeatedly open active workers' `checkpoint.json` files. Use direct checkpoint
inspection or the old scientific `status` CLI while the relevant workers are
stopped. Registered readout already waits until execution completes.

## Validation and limits

The repair audit records real Windows handle-contention tests, bounded/permanent
error cases, integrity rejection, exception-drain counts, and a real four-job
pilot with an intentionally blocked checkpoint replacement. The queue recovered
the affected worker and finished the pilot. French full and disabled trajectories
matched uninterrupted references for all candidate records, saved-state bytes,
RNG state and counters. Independent output roots intentionally have different
contract hashes. Pilots remain excluded from production analysis.

The policy does not promise survival of every OS/disk fault. It does not install
a startup service, bypass ACLs, change antivirus settings, or restart arbitrary
failed scientific jobs. Persistent or unexplained failures still stop the queue.
See [repair audit](../reports/2026-10-22_h1_checkpoint_io_repair.md) for commands,
source identities, failed attempts, validation and production continuation.
