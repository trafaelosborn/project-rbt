# Decision 044: execute the frozen H1 objective

Date: 2026-10-01

Status: implemented and verified by targeted tests and bounded real-corpus trials.

## Authorization and scope

The author requested implementation of the diagnosed repairs and comprehensive
documentation. This change implements the six-feature objective in the frozen
H1 protocol. It does not change the feature tier or the scientific hypothesis.
The June production matrix and the August endpoint analysis remain historical
artifacts; they cannot be relabeled as corrected H1 runs.

## Confirmed defects

1. Freeze identifiers reached the manifest but not engine scoring. Ordinary,
   incremental, and batch candidate evaluation used the legacy three-feature
   Latin reward. The six registered feature calculations already existed in
   `src/validation/static_feature_matrix.py`.
2. `corpus_budget` was serialized but never used by the engine. Initial sampling
   selected 800 sequences rather than a common number of tokens.
3. Resumption did not validate objective identity before returning a completed
   manifest or continuing a run.
4. The endpoint-control analysis was a post-hoc 9,000-token analysis, with
   reference scaling recomputed at that budget. It is not the registered H1
   trajectory/convergence test.

## Implementation contract

- Both freeze fields select H1 execution. Incomplete or mismatching freeze
  identifiers fail before writing run artifacts. Runs without either field
  retain the legacy behavior for reproduction of the first paper.
- Frozen commit: `232393a2d943f878b71d9c75aa40dce8d4602e4f`.
- Feature document SHA-256:
  `551a94d99b5426bf15968b9edf1769b76e609e6fe2927a1c8e2fdf30b9990a87`.
- The objective is negative scaled Euclidean distance to the Latin row, using
  F02, F09, F15, F17, F18, F19 and population standard deviations from all 22
  rows of the frozen 20,000-token matrix. Generated states never fit scaling.
- The existing structural score scale (5.0), form/coherence/cost terms,
  reward amplification, and proposal operators remain as configured. H1
  replaces the structural term; it does not silently become a structural-only
  search. Component values and the active objective must be auditable.
- All H1 scoring routes use one materialized Python reference evaluator.
  Legacy incremental/batch scorers cannot receive H1 candidates. Acceleration
  requested versus actually used is recorded. Native acceleration can be added
  later only with equivalence tests against this evaluator.

## Disclosed sampling and execution clarifications

The frozen protocol requires equal token budgets but does not specify behavior
for length-changing mutations at the boundary. This implementation discloses
the following policy before the corrected production experiment:

- Initially shuffle source sequences deterministically using the run seed and
  source identity, then take exactly 20,000 tokens, truncating the last sequence
  as in the static calibration sampler.
- Preserve the evolving corpus and its order across blocks/resumption.
- Measure the first 20,000 tokens of each candidate in that fixed order. Extra
  tokens remain in the evolving corpus; candidates with fewer than 20,000
  tokens are invalid for this budget and are rejected explicitly. In
  particular, a merge can be invalid at the budget floor. Do not pad or repeat
  tokens to make an undersized corpus appear adequate.
- Only the frozen 20,000-token calibration is enabled. Another budget requires
  a separately documented calibration/amendment rather than reusing scales
  fitted at a different size.
- This policy is an explicit post-freeze operational clarification. It affects
  candidate eligibility and must be cited with corrected H1 results.
- Corrected runs begin with source corpora in fresh output directories. Old
  endpoints are not valid checkpoints for the corrected objective.
- The existing block driver restarts its PRNG from the configured seed in each
  block. That block-based search policy is retained. Checkpoint corpus order is
  now preserved; this does not claim equivalence to a single uninterrupted
  PRNG stream across the whole experiment.

## Verification requirements

Verify frozen input identities, direct formula agreement, feature sensitivity,
candidate ranking/acceptance, all acceleration-flag combinations, token budget
handling, persisted-score replay, and incompatible-resume rejection. Run a
bounded real-corpus trial through the production launcher. A pilot's scientific
direction is not a pass/fail criterion. Record commands, results, limitations,
and artifacts in `docs/reports/2026-10-01_h1_repair_audit.md`.

## Result

The 153-test affected regression suite passed. After adding stop/resume and
complete-matrix readout checks, all 38 focused H1 tests passed (155 distinct
tests across the two selections). French and Finnish pilots each completed
four proposals in two blocks with a 20,000-token measurement window. Nine
saved stages, including block baselines, replayed within 3.58e-7 of their
rounded serialized structural scores. No full production experiment was
launched and no H1 scientific verdict was generated.
