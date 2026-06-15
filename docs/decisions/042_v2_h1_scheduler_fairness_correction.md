# Decision 042: v2 H1 Scheduler Fairness Correction

Date: 2026-06-14

Status: active operational correction for the H1 production supervisor.

## Context

Decision 041 introduced the adaptive H1 production supervisor profile:

```text
night cap: 6 concurrent child jobs
day cap: 3 concurrent child jobs
```

The initial H1 production launch used the correct frozen feature/corpus
configuration and the registered seed list, but the supervisor expanded and
launched jobs in language-major order. That meant the first daytime cap was
occupied by all three French seeds, while later languages waited.

This did not alter scoring, corpora, feature definitions, seeds, or outputs.
It was a resource-scheduling problem: the launch order was inefficient and made
the status table look misleadingly bad.

## Correction

The supervisor now selects the next pending job from the language with the
fewest currently active jobs.

Effect:

```text
daytime cap 3: prefer three different source languages
night cap 6: prefer all five source languages before starting a second seed
```

This is a scheduling fairness rule only. It does not change:

- source languages,
- seed list,
- target,
- scoring,
- feature basis,
- corpora,
- plateau or joint-hit thresholds,
- already written block outputs.

## Production Handling

The active production run was paused cleanly at block boundaries on 2026-06-14.
The existing block outputs remain valid and resumable. After this correction is
committed, the same output root is resumed under the fairer launch scheduler.

The production output root remains:

```text
data/v2_h1_production_20260613
```

Reports should mention this operational correction if they describe wall-clock
run order, scheduler behavior, or the initial `stopped_manual` statuses caused
by adaptive downshift.
