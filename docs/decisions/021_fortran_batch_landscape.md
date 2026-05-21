# Decision 021: Phase 2 Fortran Batch Landscape

Date: 2026-04-09
Status: Accepted

## Context

Phase 1 proved that Project RBT can compile and call a local Fortran kernel from Python without changing methodology.

The next useful step is not more micro-kernels. It is to make the Fortran side produce a real batch view of the aligned tensor landscape while Python continues to own:

- scoring
- Hungarian acceptance logic
- coherence gating
- corpus recompilation

## Decision

Phase 2 adds three pieces:

1. [aligned_tensor.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/aligned_tensor.py)
   Source-vocabulary alignment for co-occurrence and positional components.

2. [bridge_distance.f90](/C:/Code/Project%20RBT/project_rbt/src/accelerate/bridge_distance.f90)
   Extended with `top_adjustments_batch`, a Fortran subroutine that scans the full aligned co-occurrence and positional landscape and returns the top-K signed adjustment candidates.

3. [fortran_batch.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/fortran_batch.py)
   Python wrappers, Python reference implementation, and benchmark helpers for the Phase 2 batch kernel.

## Alignment Rule

We inherit the same source-vocabulary alignment rule already used by the reinforced gradient path:

- the current/source vocabulary is the anchor space
- reference tensors are projected into that space
- missing reference tokens become zero rows or zero row/column blocks

This keeps the acceleration layer consistent with the project's earlier target-conditioned alignment assumption instead of introducing a new ontology.

## Candidate Definition

A Phase 2 adjustment candidate is still low-level and methodology-neutral:

- component id
- row index
- column index
- signed delta = `reference - current`
- absolute score = `abs(reference - current)`

This is a batch landscape primitive, not an acceptance rule.

## Consequences

### Good

- Fortran now returns a real scored candidate batch instead of only a distance matrix.
- Co-occurrence and positional slices are handled together under one aligned tensor view.
- Python can later layer Hungarian selection on top of the returned batch without changing the kernel.

### Bad

- The returned adjustment vectors are still tensor-space primitives, not direct corpus mutations.
- The current batch kernel does not yet include n-gram slices.
