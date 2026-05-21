# Decision 020: Fortran Acceleration Scaffold

Date: 2026-04-09
Status: Accepted

## Context

Project RBT's current reinforced search evaluates candidate tensor movement one accepted mutation at a time in Python. The immediate goal for the acceleration layer is not to change methodology, scoring, or operator families, but to move the most expensive tensor math out of the sequential Python loop.

Session 1 only targets the narrowest safe slice of that plan:

1. one Fortran kernel
2. one Python `f2py` wrapper
3. one exactness test against the NumPy reference
4. one benchmark harness on the real French/Latin co-occurrence matrices
5. one contiguous tensor-layout design for the later multi-component phase

## Decision

We scaffold the acceleration layer under [src/accelerate](/C:/Code/Project%20RBT/project_rbt/src/accelerate) with four pieces:

1. [bridge_distance.f90](/C:/Code/Project%20RBT/project_rbt/src/accelerate/bridge_distance.f90)
   A single Fortran subroutine for elementwise absolute distance between two 2D matrices.

2. [fortran_distance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/fortran_distance.py)
   The `f2py` build/load wrapper, NumPy reference implementation, and benchmark helper.

3. [tensor_layout.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/tensor_layout.py)
   A deterministic single-buffer layout for co-occurrence, positional, and n-gram components.

4. [benchmark_distance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/benchmark_distance.py)
   CLI harness for NumPy vs Fortran timing once a compiler is available.

## Why This Shape

### Elementwise absolute distance first

The first kernel is intentionally small and boring. It preserves exact semantics with the Python reference:

`np.abs(current - reference)`

That gives us a trustworthy compile/test/benchmark wedge before we batch candidate-vector generation.

### `f2py` over `ctypes`

`f2py` is preferred because:

- NumPy is already a dependency
- matrix-shaped arguments map naturally
- Fortran-order arrays can be passed directly
- it keeps the interface transparent and inspectable

### No-space Windows build root

NumPy `f2py` on Python 3.14 uses the Meson backend. On this machine, Meson staging inside the repo path
`C:\Code\Project RBT\...`
produced repeatable Windows permission failures during temp-directory source copying.

The accepted workaround is:

- keep sources in the repo
- compile in a no-space build directory
- default that build root to `C:\Code\RBT_FORTRAN_BUILD` on Windows
- allow overrides through `RBT_FORTRAN_BUILD_DIR`

This changes where the extension is compiled, not what it computes.

### Single contiguous tensor buffer

Later tensor phases need co-occurrence, positional, and n-gram components in one contiguous block. The design choice here is:

- one 1D `float64` buffer
- deterministic offsets
- reshaped views per component

This avoids opaque structs and makes it straightforward for both Python and Fortran to agree on memory layout.

## Consequences

### Good

- The sequential Python engine remains untouched and stays the reference implementation.
- The acceleration layer is testable in isolation.
- Future kernels can extend the same build/load path.

### Bad

- Windows builds currently depend on a separate no-space build directory.
- The first kernel compiles correctly, but this trivial elementwise workload is not yet faster than NumPy.

## Notes

The `rbt_brief.docx` file was not found in the local workspace during this session, so this scaffold follows the in-chat brief verbatim.
