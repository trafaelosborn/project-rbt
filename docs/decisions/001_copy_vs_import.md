# Decision: Copy shared infrastructure from Minos rather than importing

Date: 2026-04-07
Phase: P2 — Project setup

## What was decided

The shared statistical fingerprinting infrastructure (co-occurrence matrix builder,
positional frequency distribution builder) was copied from `c:/Code/Project Minos/`
into this repository rather than imported via a shared package or path dependency.

The copied modules are:
- `src/fingerprint/cooccurrence.py` — adapted from Minos `src/fingerprint/cooccurrence.py`
- `src/fingerprint/positional.py` — adapted from Minos `src/fingerprint/positional.py`

The pure algorithm functions (`build_vocab`, `count_cooccurrences`, `PositionalAccumulator`,
etc.) are identical to the Minos versions. The `run()` entry points and path conventions
were updated to match RBT's data layout and corpus format.

## Why

RBT and Minos operate on fundamentally different data (natural language Unicode text vs.
transliterated ancient script tokens) and will likely diverge in their fingerprinting
requirements as both projects develop. Coupling them via a shared import would mean that
a change in Minos's fingerprinting layer — driven by its specific needs for ancient script
analysis — could silently break RBT's pipeline, and vice versa.

The copy-on-diverge approach costs a small amount of duplication now in exchange for full
independence later. If an algorithmic improvement is made to one project's fingerprinting
layer, it can be ported to the other deliberately and with full review, rather than
automatically and potentially incorrectly.

A shared package approach (e.g. `pip install -e ../shared-fingerprint`) was considered and
rejected because:
1. It requires maintaining a third repository or directory as a formal dependency.
2. It couples the two projects' development velocity.
3. The algorithm functions are small enough that divergence is manageable.

## Impact

- Both projects maintain their own copies of the fingerprint algorithm code.
- When a bug fix or improvement is made in one project, manually check if it applies
  to the other and port it if so. This is a deliberate human-in-the-loop decision point.
- Tests in both projects cover the same algorithm; passing tests in one project do not
  guarantee correctness in the other after a diverging change.

## Revision history

- 2026-04-07: Initial decision. Minos Phase 1 was already built; RBT Phase 2 copies and
  adapts the fingerprint modules. Tokenizer is NOT copied — RBT's Unicode text tokenizer
  is entirely new (see `src/ingest/tokenize.py`).
