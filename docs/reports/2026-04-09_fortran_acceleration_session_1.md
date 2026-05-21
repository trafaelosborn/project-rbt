# 2026-04-09 Fortran Acceleration Session 1

## Goal

Start the RBT Fortran acceleration layer with the smallest compileable kernel:

- accept two co-occurrence matrices
- return the elementwise distance matrix
- verify Python `f2py` integration
- benchmark against the NumPy reference
- define the full contiguous tensor layout for later phases

## What Was Built

- Fortran kernel scaffold: [bridge_distance.f90](/C:/Code/Project%20RBT/project_rbt/src/accelerate/bridge_distance.f90)
- Python wrapper and benchmark helper: [fortran_distance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/fortran_distance.py)
- Contiguous tensor layout design: [tensor_layout.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/tensor_layout.py)
- CLI benchmark harness: [benchmark_distance.py](/C:/Code/Project%20RBT/project_rbt/src/accelerate/benchmark_distance.py)
- Tests: [test_fortran_distance.py](/C:/Code/Project%20RBT/project_rbt/tests/test_fortran_distance.py)

## Local Environment Findings

- Python: `3.14.3`
- NumPy: `2.4.3`
- `numpy.f2py`: available
- Portable Fortran compiler: [gfortran.exe](C:/Code/Project%20RBT/toolchains/mingw64/mingw64/bin/gfortran.exe)
- Windows Meson backend tools: user-installed `meson` and `ninja`

No system-wide `gfortran`, `ifx`, `ifort`, or `flang` executable was detected on PATH at the start of the session. A portable MinGW toolchain was then downloaded locally and used for verification.

## Real Matrix Size

The current French co-occurrence fingerprint at [french_cooccurrence.npy](/C:/Code/Project%20RBT/project_rbt/data/matrices/french_cooccurrence.npy) is:

- shape: `5000 x 5000`
- dtype: `float32`

That is the matrix size the benchmark harness now targets by default.

## What Is Verified Now

- The Python reference distance implementation is correct and tested.
- The tensor-layout packer is correct and tested.
- The Fortran build path compiles successfully through `f2py`.
- The Python-to-Fortran bridge loads successfully.
- The exactness test passes against the NumPy reference.
- The NumPy baseline benchmark now writes to [fortran_distance_benchmark_session1.json](/C:/Code/Project%20RBT/project_rbt/data/validation/fortran_distance_benchmark_session1.json).

## Baseline Benchmark

Current baseline on the real `5000 x 5000` French vs Latin co-occurrence matrices:

- NumPy reference distance: `0.270025` seconds
- Fortran kernel: `0.302920` seconds
- Speedup vs NumPy: `0.891406x`

Interpretation:

- The bridge works.
- The current kernel is slightly slower than NumPy.
- That is acceptable for Session 1 because the main goal was compileability and interface correctness first.
- This kernel is too small and too memory-bound to guarantee a speed win by itself.

## What Is Not Yet Verified

- Batched candidate-vector generation in Fortran
- Multi-component tensor execution across co-occurrence, positional, and n-gram slices
- End-to-end integration with the `v4` search loop
- A speedup on a kernel large enough to amortize `f2py` and allocation overhead

## Commands

Compile check from Python:

```powershell
python -c "from src.accelerate.fortran_distance import build_extension; print(build_extension(force=True))"
```

Run exactness test:

```powershell
$env:RBT_RUN_FORTRAN_BUILD_TESTS='1'
python -m pytest project_rbt/tests/test_fortran_distance.py -q -p no:tmpdir -p no:cacheprovider
```

Run benchmark:

```powershell
python -m src.accelerate.benchmark_distance --output project_rbt/data/validation/fortran_distance_benchmark.json
```

## Next Move

Now that the bridge is working:

1. move the kernel from naive elementwise distance to batched adjustment-vector generation
2. pack co-occurrence plus positional slices into the contiguous tensor layout
3. keep the same scoring and Hungarian logic in Python
4. re-benchmark at the batch level, where Fortran has a real chance to win
