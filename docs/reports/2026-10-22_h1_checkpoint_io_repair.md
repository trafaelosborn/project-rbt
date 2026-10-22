# H1 Windows checkpoint I/O repair and continuation, 2026-10-22

The author authorized fixing the repeated checkpoint-access failures. The repair
changes the operational queue and adds a small I/O helper. The registered
scientific source, inputs, runtime, configuration and registration are unchanged.
The existing 17,548 committed production proposals are retained.

## Failure and repair

The previous sessions stopped on a denied worker checkpoint replacement
(October 12) and a denied supervisor checkpoint read (October 19). The latter
left the final status four proposals behind the drained workers' checkpoints.
See the retained [October 19 recovery](2026-10-19_h1_production_resume.md) and
[October 21 status audit](2026-10-21_h1_status_and_checkpoint_access_failure.md).

The repaired queue inspects a checkpoint only before launch or after that child
exits, under the writer's OS lock. It never polls an active worker's pointer.
Operational access errors have a bounded backoff: ten attempts, with at most
5.55 seconds of sleep. Persistent failures remain failures.

A worker that exits on a recognized access error at its exact checkpoint path
can be resumed at most three times per job per matrix invocation. The queue
first checks the contract, complete committed proposal chain, counters and
current-state hash. The unchanged child independently replays scientific state
on resume. Bad integrity, malformed JSON, unrelated exceptions and exhausted
retry budgets stop execution. All failed logs and retry events are retained.

Exception cleanup refreshes progress after workers drain, and a secondary
status-write error does not skip cleanup. Decision 048 documents the policy,
including the fallback if pause-file creation persistently fails.

## Validation, including the failed attempt

All repair evidence is under `artifacts/h1_io_repair_20261022`.

- `tests_01.xml`: 26 passed, one failed. The initial design used a Windows
  FILE_SHARE_DELETE reader; holding that reader still prevented the registered
  writer's `os.replace()` on this host. Its source/test snapshots are retained
  as `*_shared_handle_attempt.py` / `test_shared_handle_attempt.py`. That design
  was removed. The implementation now prevents overlapping access instead.
- `tests_02.xml`: all 34 focused tests passed. These cover real Windows denied
  reads/replacements, bounded retries, fail-fast unrelated/corrupt data, exclusion
  of active writers, worker retry classification/caps, failed recovery audit,
  scientific replay equivalence and final counts after exception draining.
- `tests_03.xml`: **163 passed in 64.34 seconds**, with no failures or skips.
  This is the 135-test prior selection plus 28 new tests, not a sum of overlapping
  test executions.

The broad command was:

```powershell
python -m pytest tests/test_h1_io.py tests/test_h1_io_recovery.py tests/test_h1_matrix_runner.py tests/test_h1_checkpointed.py tests/test_h1_ablation_registration.py tests/test_h1_execution.py tests/test_h1_convergence.py tests/test_h1_panel_statistics.py tests/test_engine_reinforced_v2.py tests/test_engine_reinforced_v3.py tests/test_engine_reinforced_v4.py tests/test_engine_reinforced_v5.py tests/test_long_run_v4.py tests/test_multi_run_supervisor.py -q -p no:cacheprovider --basetemp=data/tmp/h1_io_repair_03 --junitxml=artifacts/h1_io_repair_20261022/tests_03.xml
```

Use fresh evidence/temp paths for a repeat.

## Real-process Windows contention pilot

The new reproducible harness ran:

```powershell
python -u scripts/verify_h1_io_recovery.py --output-root artifacts/h1_io_repair_20261022/real_windows_pilot_01
```

It registered fresh, explicitly excluded pilot roots and ran actual child
processes. French/full was first stopped normally after proposal 1. An external
reader deliberately held its checkpoint open during the next worker session,
causing the real Windows replacement failure. After that failure appeared in the
child's log, the reader released its handle. The queue validated the committed
history, performed one automatic recovery, and completed all four French/Finnish
pilot jobs at their four-proposal budgets, with two candidate slots per proposal.

Fresh uninterrupted French references were run for both arms. Their proposal
records (candidate attempts, choices, state transitions and counters), saved-state
bytes, final RNG state, accepted-state count and stagnation count matched the
queue-managed trajectories exactly. Output-root-specific contract hashes differ
intentionally and are not represented as identical. The test includes the failed
attempt; it neither deletes it nor promotes the pilot into production evidence.

The complete result and queue events are in
`real_windows_pilot_01/verification_results.json`, with individual worker logs and
pilot states retained locally. The failure was reproduced by a concrete conflicting
handle; the responsible process in the historical incidents remains unidentified.

## Production continuity and launch

`production_preflight.json` records the unchanged registration/identity, all 66
committed histories and current-state checks, the availability of all 67 locks,
17,548 retained proposals, and the two runner-created pause-file hashes. About
368.5 GiB of disk space was free. Only those two verified pause requests were
removed, under the job locks, after tests and the real contention pilot passed.

Registration identity:
`012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0`.

Operational source identities (LF-normalized SHA-256):

| File | SHA-256 |
|---|---|
| `scripts/run_h1_ablation_matrix.py` | `9565366e561ffdb6e944b231b08aa01b3a6a4bfb93d69be4d6ff6d2ec0e08f95` |
| `src/control/h1_io.py` | `f3d9e91f3e7ba85ef30afff79a0b80842af8d8dc1f9363f6b78a0409d0c6fa47` |

At **2026-10-22 15:51:02 UTC (09:51:02 America/Denver)**, the repaired queue was
launched as hidden, below-normal-priority runner PID **51696**, with its parent
execution session waiting to preserve the background process. Session:
`20261022T155107Z_33db3698`. The first workers were French seed 42, full PID 17632
and disabled PID 13936, both beginning from proposal 400.

```powershell
python -u scripts/run_h1_ablation_matrix.py --output-root data/h1_ablation_production_20261001 --workers 2 --session-proposals 100 --session-seconds 1800
```

The original two-worker, one-thread-per-worker settings, 100-proposal/1,800-second
session limits, and 1,000-proposal scientific budgets are retained. The new queue
archives both operational source files in its session directory. The launch
receipt and top-level logs are retained in the repair evidence directory.

At 15:56:55 UTC, a deliberate operational PAUSE request checked clean boundary
shutdown. Both workers paused normally at proposal **462**, adding 124 committed
proposals. At 15:57:01 UTC, all 66 checkpoint totals and the queue's final status
agreed at **17,672 / 66,000 (26.78%)**. Checkpoint/state hashes passed, both workers
exited zero, and the repaired production session recorded no I/O retries or
worker recoveries. This was a validation pause, not another access failure.
Its status and exact pause-file provenance are preserved in
`verified_boundary_pause_status.json` and `verified_boundary_pause.json`.

Only that deliberate request and its two identified runner-created pause files
were removed. The queue continued at **15:57:21 UTC (09:57:21 Denver)** as hidden,
below-normal-priority runner PID **13428**, with the same limits and source hashes.
`process_launch_continuation.json` and separate continuation logs preserve this
second launch. Session boundaries do not reset the registered RNG stream or
change the fixed scientific budget.

`source_inventory.json` identifies the validated operational source archive
`validated_operational_source.zip`, SHA-256
`a1a2679e3da19d3c418c95aecb2144e4c270a28f787533b69d52de07d1e007bd`.
The source archive includes the queue, helper, verification harness, tests and
Decision 048. The final report and publication file list are recorded separately.

At 15:58:37 UTC, continuation session `20261022T155725Z_5d97298b` held the matrix
lock, reported `running`, and both workers had written immutable proposal records
through 471 from their 462 checkpoints. This observation did not open their active
checkpoint pointers or count an uncommitted proposal as committed. The last
independently verified total remains 17,672. The registration still validated
exactly. `continuation_check.json` records this observation.

The Git publication includes the repair, tests, harness, Decision 048, this audit,
the prior October 19/21 reports, and selected immutable evidence. Its exact file
list and hashes are in `publication_manifest.json`. Bulk pilot/production states,
unapplied experimental source copies, and ongoing continuation logs remain local.

## Reading live progress and remaining limits

Read `data/h1_ablation_production_20261001/matrix_status.json` and session/worker
logs. The heartbeat is live; proposal counts update after worker sessions exit
and are a lower bound during active sessions. Do not poll individual active
checkpoints. The frozen scientific `status` CLI should be used with the workers
stopped, since its checkpoint reads were deliberately not changed.

The finite pilot verifies this repair's recovery behavior; it does not guarantee
unlimited uptime or completion under persistent storage faults. No startup service,
permission/antivirus change, unbounded retry loop, scientific exclusion, new score,
or registration override was introduced. No completed scientific arm comparison
is claimed. Related: [Decision 048](../decisions/048_windows_checkpoint_access_recovery.md).
