# H1 production verification and resume, 2026-10-31

The author requested a restart if necessary. The existing October 1 registered
experiment was inactive and was resumed from its verified checkpoints. No
scientific code, input, seed, score, budget, or stopping rule changed.

## State found

The previous queue still said `running`, but its last heartbeat was
**2026-10-22 19:15:42 UTC (13:15:42 America/Denver)**. Its session was
`20261022T155725Z_5d97298b`, runner PID 13428. The saved status reported 23,272
proposals and two active Markov-null search-seed-1729 workers, each starting at
100 proposals. It recorded no I/O retries, worker recoveries, or shutdown errors.
The retained worker logs had no exception traceback; the top-level continuation
stdout and stderr were empty. There was no recorded clean queue shutdown.

Host inspection on October 31 found no experiment processes. Windows reported
its latest boot as October 31 at 00:07:45 Denver time. This establishes that the
old processes are absent; it does not identify the exact time or cause of the
earlier interruption. In particular, the stale status is not evidence of another
checkpoint-access failure. No existing process was terminated during this work.

## Preflight and preservation

Evidence is retained under `artifacts/h1_restart_20261031`. The previous status,
the two interrupted worker logs, previous top-level logs, and interrupted
checkpoints were copied there before launch.

The preflight acquired the matrix lock and all 66 job locks together. With every
writer excluded, it checked every committed proposal chain, contract identity,
candidate counter ledger, checkpoint hash, and current-state hash using the
existing validation functions. All 66 jobs passed. No PAUSE files existed and
none were removed. No proposal records extended beyond their committed
checkpoints; no history was deleted or manually rewritten. About 221.3 GiB of
disk space remained available.

The checkpoints retained **23,392 / 66,000 proposals (35.44%)**. None of the 66
jobs had finished its 1,000-proposal budget. The 120-proposal difference from the
stale status was fully accounted for by the interrupted workers:

| Job | Last exited-session status | Verified checkpoint |
|---|---:|---:|
| Markov null, corpus 42, search 1729, full | 100 | 170 |
| Markov null, corpus 42, search 1729, disabled | 100 | 150 |

The full registration comparison passed, including scientific implementation,
runtime, protocol, source corpora, reference inputs, and job settings. Its sealed
identity remains
`012560628d8d5f7c54f74560a711804cf40d962192cf6f79c08b2d987f60eeb0`.
The queue and I/O helper exactly match the October 22 tested repair:

| File | LF-normalized SHA-256 |
|---|---|
| `scripts/run_h1_ablation_matrix.py` | `9565366e561ffdb6e944b231b08aa01b3a6a4bfb93d69be4d6ff6d2ec0e08f95` |
| `src/control/h1_io.py` | `f3d9e91f3e7ba85ef30afff79a0b80842af8d8dc1f9363f6b78a0409d0c6fa47` |

This is an operational resume with data-integrity validation, not another code
repair. The previous 163-test suite and real Windows recovery pilot are recorded
in the [October 22 audit](2026-10-22_h1_checkpoint_io_repair.md); they were not
rerun or represented as new October 31 tests. Full feature/reward replay is
performed by the unchanged scientific worker when each job resumes.

## Restart and live verification

At **2026-10-31 23:36:02 UTC (17:36:02 America/Denver)**, the existing queue was
launched as hidden, below-normal-priority runner PID **41084**. Its parent
execution session waits for the runner. The command remains:

```powershell
python -u scripts/run_h1_ablation_matrix.py --output-root data/h1_ablation_production_20261001 --workers 2 --session-proposals 100 --session-seconds 1800
```

The queue uses two workers, one numerical-library thread per worker, and the
unchanged 100-proposal / 1,800-second session limits. Its new session is
`20261031T233606Z_1134eb6d`. The first workers were French corpus/search seed 42,
full PID 4540 and disabled PID 37316, both resuming from proposal 562.

At **23:36:26 UTC**, live verification found a fresh heartbeat, the matrix lock
held, two active workers, no I/O retries or recoveries, and empty runner stderr.
Both workers had written immutable proposal records through 564. This confirms
new work after restart. It does not count an active proposal record as an
independently verified committed checkpoint. Active checkpoint files were not
opened; the last fully audited total at launch remains 23,392.

`preflight.py` and `preflight.json` preserve the audit procedure and all 66 job
identities. `host_inspection.json`, `launch.ps1`, `process_launch.json`,
`verify_live.py`, and `live_verification.json` preserve the host check, launch,
and live observation. Selected immutable evidence and this report are published
with an exact file/hash list in `publication_manifest.json`. Ongoing runner logs
and the bulk production state remain local.

Live status comes from `matrix_status.json`; its proposal totals update after
worker sessions exit and are a lower bound during active sessions. This restart
does not install a boot service or change automatic-start behavior. No completed
scientific result or arm comparison is claimed.
