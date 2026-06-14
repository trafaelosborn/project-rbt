# Decision 041: v2 H1 Launch Seeds and Clocked Supervision

Date: 2026-06-13

Status: active launch-control specification for H1.

## Context

Decision 040 froze the v2 H1 feature, corpus, and analysis preregistration.
The frozen document requires multiple full seeds per source and records that
the exact production seed list should be written before launch.

The first frozen supervisor shakedown used seed `42` across the five H1 Romance
sources and completed successfully. Before production launch, the H1 seed list
and workstation resource policy are standardized here.

This decision does not alter:

- the registered feature pool,
- the retained in-loop tier,
- the corpus bank,
- null or control policy,
- thresholds,
- registered statistics,
- source languages.

## Production H1 Seed List

The registered H1 production seed list is:

```text
42
1729
8675309
```

These seeds are used for every required H1 source language:

```text
french
spanish
italian
portuguese
romanian
```

The production matrix therefore contains:

```text
5 source languages x 3 seeds = 15 jobs
```

## Run Mode

Production H1 runs use unbounded continuation mode:

```text
--unbounded
```

Each job continues until one of the registered terminal conditions is reached:

```text
joint_hit
plateau_hit
failed
manual stop
```

Manual stops are resumable and do not consume or rewrite completed block
outputs.

## Clocked Supervision Policy

The default production resource policy is the adaptive supervisor profile:

```text
profile: adaptive
night window: 23:00 through 07:59 local wall-clock time
night max_parallel: 6
day max_parallel: 3
priority: below_normal
thread_count: 1 per child process
live_event_mode: accepted_only
```

At night, the supervisor may launch up to six concurrent child jobs. During the
day, it caps active work at three concurrent child jobs.

If the clock changes from night to day while more than three child jobs are
running, the supervisor requests clean stop sentinels for the over-limit jobs.
Those jobs stop at their next block boundary and are requeued for later resume.
No in-progress block is killed by the scheduler.

## Canonical Launch Shape

```text
python -m src.control.multi_run_supervisor start ^
  --output-root data/v2_h1_production_20260613 ^
  --profile adaptive ^
  --languages french spanish italian portuguese romanian ^
  --seeds 42 1729 8675309 ^
  --unbounded ^
  --block-proposals 1000 ^
  --candidate-count 16 ^
  --num-sequences 800 ^
  --features-freeze-commit 232393a2d943f878b71d9c75aa40dce8d4602e4f ^
  --features-freeze-sha256 551a94d99b5426bf15968b9edf1769b76e609e6fe2927a1c8e2fdf30b9990a87 ^
  --corpus-budget 20000
```

The CLI default seed list is also set to the registered H1 production seeds so
that omitting `--seeds` does not accidentally launch a single-seed H1 matrix.

## Reporting Requirement

Every H1 production report must cite this decision alongside Decision 040 when
describing:

- the production seed list,
- the number of language-seed jobs,
- the adaptive wall-clock resource policy,
- any day-window scheduled stop/resume events.
