# 043 — Scheduler starvation fix (incumbent monopoly)

**Date:** 2026-06-18
**Status:** Accepted
**Scope:** Orchestration only (scheduler). Does NOT touch frozen feature code
(engine / operators / scoring / targets / seeds / corpus). Per the same
reasoning as decisions 041/042, changing *which job runs when* does not change
*what any job computes*: each job runs the identical frozen engine with its
fixed seed and corpus, so trajectory content is unaffected. The features freeze
(commit 232393a) is intact.

## Problem

In the `v2_h1_production_20260613` matrix, `italian_seed_1729` and
`italian_seed_8675309` never launched — 4 days in, the only two jobs of fifteen
that had never started. Not a crash: empty dirs, no stderr, never selected.

## Root cause

`select_next_pending_index` chose the pending job whose language was least
represented among **currently-running** jobs. `italian_seed_42` was a permanent
incumbent: it filled one of the 3 daytime slots, was never "over-limit", so it
was never stopped, so Italian's running-count was always >= 1. Every fill cycle
it therefore lost to any language sitting at 0, and on ties lost to earlier
list-order seeds. Its two siblings were structurally last in line forever. The
queue recirculates (unbounded jobs only leave the pool by plateauing), so
"they'll start once others finish" never happens.

## Fix

Key selection on **work done**, not on who is running now. Lowest wins:
1. blocks completed (never-started jobs = 0 -> scheduled first),
2. jobs ever started in that language (spreads coverage across languages),
3. manifest order (stable tiebreak).

Running incumbents are not evicted (selection only fills free slots), so the
deep converging seed_42 runs continue undisturbed; the starved/behind jobs get
priority for the next free (night-window) slot. Implemented in
`select_next_pending_index` + `pop_next_pending_job` (now reads job progress and
`started_language_counts`). Regression test:
`test_incumbent_does_not_starve_its_own_unstarted_seeds`.

## Rollout

The code fix takes effect on the next supervisor start (including every
reboot/auto-resume). Because per-job `manifest.json` writes are non-atomic
(`long_run_v4._save_manifest`), the live supervisor was NOT hard-restarted (a
kill mid-write could corrupt a deep runner's manifest and lose hundreds of
blocks). Instead the two starved Italian seeds were launched directly with their
exact frozen `SupervisorJob.command()` via `scripts/start_unstarted_jobs.py`
(below_normal priority, single-thread env), producing the identical trajectory
they would have under the scheduler. The fair scheduler subsumes everything at
the next clean restart.
