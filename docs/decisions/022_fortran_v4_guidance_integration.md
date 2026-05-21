# Decision 022: Fortran-Guided V4 Compatibility Layer

Date: 2026-04-09
Status: Accepted

## Context

Phase 2 proved that the Fortran side can extract a large top-K tensor adjustment
batch across aligned co-occurrence and positional slices far faster than the
Python reference.

The next safe step is not to rewrite the reinforced search around tensor-space
acceptance all at once. It is to let the existing `v4` engine consume that batch
as guidance while preserving:

- the same mutation operator families
- the same Latin structural and form scoring
- the same coherence gate
- the same family-alignment diagnostic
- the same Python-only path as the reference implementation

## Decision

Phase 3 adds an optional compatibility layer:

1. [v4_batch_guidance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/v4_batch_guidance.py)
   builds an in-memory current bridge tensor, aligns the Latin tensor into the
   current anchor vocabulary, asks either the Fortran or NumPy batch kernel for
   a top-K candidate landscape, and then runs a Hungarian frontier over that
   batch to produce a diverse set of hotspot hints.

2. [engine_reinforced_v4.py](/C:/Code/Project%20RBT/project_rbt/src/retrodiction/engine_reinforced_v4.py)
   now accepts an opt-in `acceleration_mode`:
   - `python_only`
   - `numpy_batch`
   - `fortran_batch`
   - `auto_batch`

3. The guidance hints steer existing operator families toward hotspot tokens,
   families, and spans, but they do not replace scoring or acceptance.

## Guidance Primitive

The Fortran batch is still tensor-space, not corpus-space. Phase 3 therefore
adds a narrow translation layer rather than pretending tensor cells are direct
mutations.

The translation rule is:

- build a current anchor vocabulary from the live bridge corpus
- align Latin co-occurrence and positional tensors into that anchor space
- extract the top-K signed adjustment cells
- use a Hungarian frontier to choose a non-overlapping subset
- convert that subset into:
  - hotspot tokens
  - hotspot token pairs
  - positional token targets

Those hotspots then bias the existing mutation operators.

## Consequences

### Good

- `v4` can now run in a real accelerated guidance mode without breaking the
  Python reference path.
- The guidance path remains interpretable because Python still owns the actual
  mutation semantics.
- The batch is filtered through a Hungarian frontier, which makes the hints more
  global and less redundant than a naive top-K list.

### Bad

- This is not yet the user's full target architecture.
- Current guidance still rebuilds live co-occurrence and positional slices in
  Python each proposal.
- End-to-end speedup is therefore not guaranteed yet, even though the raw
  Fortran batch kernel is fast.

## Why This Step Is Still Worth Keeping

Phase 3 de-risks the integration boundary that matters:

- live bridge tensor -> Fortran batch landscape -> Python operator guidance

That gives the project a working compatibility path for later deeper fusion
without forcing a wholesale rewrite of the search logic.
