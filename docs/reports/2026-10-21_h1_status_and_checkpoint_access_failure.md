# H1 status and recurring checkpoint-access failure, 2026-10-21

The author requested a status update. At 2026-10-21 23:25 UTC (17:25
America/Denver), the experiment was stopped. No jobs were running and no
scientific readout had been produced.

## Retained progress

The 66 saved job checkpoints contain **17,548 committed proposals out of 66,000
(26.59%)**. This is 3,871 more proposals than the last verified October 19 status
of 13,677. Individual jobs are between 100 and 400 proposals; none has reached
its registered 1,000-proposal budget.

The final queue status reports 17,544 proposals. Direct checkpoint inspection
finds four additional proposals committed while the workers were draining after
the supervisor failed. The final Turkish seed-1729 positions are 277/full and
264/disabled. Both worker logs report a normal `paused` result with reason
`pause_file`, and their recorded checkpoint hashes match the saved checkpoints.
The direct checkpoint count is therefore used for this status report.

## Failure evidence

Runner session `20261019T153027Z_6bcf0729` ended at 2026-10-19 18:06:36 UTC
(October 19, 12:06:36 America/Denver), about 2 hours 36 minutes after relaunch.
The recorded runner exception is a `PermissionError` reading
`turkish__corpus_42__search_1729__full/checkpoint.json`.

The top-level stderr traceback locates the failure in the operational queue's
`progress()` call, through `read_json()` and `Path.read_text()`. The queue then
created PAUSE files for both active Turkish seed-1729 workers and waited for
them to stop. Those two runner-created requests remain in place.

This is the second observed Windows checkpoint-access failure. The earlier
October 12 failure was a worker's `os.replace()` call publishing a checkpoint;
this failure is the supervisor's attempt to open a checkpoint for reading.
The evidence identifies the failing operations but does not identify the
process or exact Windows sharing/permission interaction that denied access.

## Checks performed

The matrix lock and all 66 job locks were available. For every job, this status
inspection verified the checkpoint seal, registered job-contract identity,
proposal-budget range, current-state hash, recorded frozen objective identity,
20,000-token measurement, and checkpoint link to the latest committed proposal
record. All passed. Full history-chain replay and independent feature/reward
recomputation were not repeated during this status check.

No tracked changes were present in scientific source, queue scripts, decision
documents, analysis protocols or the production registration relative to the
published commit. About 366 GiB of disk space was free.

The registration identity remains:
`012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0`.

The machine-readable status evidence is
`artifacts/h1_status_20261021/20261021T232546Z/status_check.json`.
The previous matrix status, runner traceback and two affected worker logs are
copied into the same directory. Original evidence remains in its run directories.

## Implication and next repair

No scientific arm comparison can yet be reported. The observed stop is an
operational file-access failure and provides no result about the study question.

Repeatedly relaunching the unchanged queue has not provided reliable unattended
execution. The next repair should address checkpoint access under concurrent
Windows reads and atomic replacement, with bounded retries for transient access
errors and explicit failure for persistent access or integrity errors. The
queue's progress snapshot should also refresh after drained workers stop so it
reports their last committed positions.

The operational queue is outside the frozen scientific implementation. Any
proposal to alter the shared registered checkpoint code requires an explicit
provenance/compatibility treatment; the registration must not simply be edited
to accept changed source. No such change, pause-file removal, restart, or
scientific reinterpretation was performed during this status check.

Related: [October 19 recovery](2026-10-19_h1_production_resume.md),
[October 12 recovery](2026-10-12_h1_production_resume.md), and
[original launch](2026-10-01_h1_production_launch.md).
