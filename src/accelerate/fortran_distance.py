"""
Fortran distance kernel wrapper for Project RBT.

Session 1 scope:
    - compile a single f2py-backed kernel for elementwise matrix distance
    - keep Python reference behavior explicit and testable
    - expose a benchmark helper against the existing French/Latin matrices
"""

from __future__ import annotations

import importlib.util
import json
import os
import site
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import ctypes

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_DEFAULT_BUILD_ROOT = PROJECT_ROOT.parent.parent / "RBT_FORTRAN_BUILD"
DEFAULT_BUILD_DIR = Path(
    os.environ.get(
        "RBT_FORTRAN_BUILD_DIR",
        str(WINDOWS_DEFAULT_BUILD_ROOT if os.name == "nt" else PROJECT_ROOT / "build" / "fortran"),
    )
)
DEFAULT_SOURCE_PATH = Path(__file__).with_name("bridge_distance.f90")
DEFAULT_MODULE_NAME = "rbt_distance_kernels"
DEFAULT_CURRENT_MATRIX = PROJECT_ROOT / "data" / "matrices" / "french_cooccurrence.npy"
DEFAULT_REFERENCE_MATRIX = PROJECT_ROOT / "data" / "matrices" / "latin_cooccurrence.npy"
KNOWN_FORTRAN_COMPILERS = ("gfortran", "ifx", "ifort", "flang")
PORTABLE_FORTRAN_CANDIDATES = (
    PROJECT_ROOT.parent / "toolchains" / "mingw64" / "mingw64" / "bin" / "gfortran.exe",
    PROJECT_ROOT / "toolchains" / "mingw64" / "mingw64" / "bin" / "gfortran.exe",
)
USER_PYTHON_SCRIPTS_DIR = Path(site.getusersitepackages()).resolve().parent / "Scripts"
_LOADED_MODULES: dict[tuple[str, str], object] = {}


class FortranUnavailableError(RuntimeError):
    """Raised when no usable Fortran toolchain is available."""


class FortranBuildError(RuntimeError):
    """Raised when f2py compilation fails."""


@dataclass(frozen=True)
class BenchmarkResult:
    current_shape: tuple[int, int]
    python_seconds: float
    fortran_seconds: float | None
    speedup_vs_numpy: float | None
    compiler: str | None
    status: str
    notes: str | None = None

    def to_dict(self) -> dict:
        return {
            "current_shape": list(self.current_shape),
            "python_seconds": round(self.python_seconds, 6),
            "fortran_seconds": None if self.fortran_seconds is None else round(self.fortran_seconds, 6),
            "speedup_vs_numpy": None
            if self.speedup_vs_numpy is None
            else round(self.speedup_vs_numpy, 6),
            "compiler": self.compiler,
            "status": self.status,
            "notes": self.notes,
        }


def detect_fortran_compiler() -> str | None:
    for candidate in PORTABLE_FORTRAN_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    for compiler in KNOWN_FORTRAN_COMPILERS:
        resolved = shutil.which(compiler)
        if resolved:
            return resolved
    return None


def compiler_available() -> bool:
    return detect_fortran_compiler() is not None


def _extension_glob(module_name: str) -> tuple[str, ...]:
    return (f"{module_name}*.pyd", f"{module_name}*.so", f"{module_name}*.dll", f"{module_name}*.dylib")


def _windows_short_path(path: Path) -> str:
    if os.name != "nt":
        return str(path)
    buffer = ctypes.create_unicode_buffer(32768)
    result = ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer))
    return buffer.value if result else str(path)


def compiled_extension_path(
    build_dir: Path | None = None,
    module_name: str = DEFAULT_MODULE_NAME,
) -> Path | None:
    build_root = Path(build_dir or DEFAULT_BUILD_DIR)
    matches: list[Path] = []
    for pattern in _extension_glob(module_name):
        matches.extend(sorted(build_root.glob(pattern)))
    return matches[-1] if matches else None


def _normalize_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"{label} must be a 2D matrix, got shape {arr.shape!r}")
    return np.asfortranarray(arr, dtype=np.float64)


def numpy_elementwise_distance(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    current_arr = _normalize_matrix(current, "current")
    reference_arr = _normalize_matrix(reference, "reference")
    if current_arr.shape != reference_arr.shape:
        raise ValueError(
            "current and reference matrices must have the same shape: "
            f"{current_arr.shape!r} != {reference_arr.shape!r}"
        )
    return np.abs(current_arr - reference_arr)


def build_extension(
    *,
    force: bool = False,
    build_dir: Path | None = None,
    module_name: str = DEFAULT_MODULE_NAME,
    source_path: Path | None = None,
    verbose: bool = False,
) -> Path:
    build_root = Path(build_dir or DEFAULT_BUILD_DIR)
    source = Path(source_path or DEFAULT_SOURCE_PATH)
    build_root.mkdir(parents=True, exist_ok=True)

    if force:
        for key in [key for key in _LOADED_MODULES if key[1] == module_name]:
            _LOADED_MODULES.pop(key, None)

    existing = None if force else compiled_extension_path(build_root, module_name)
    if existing and existing.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return existing

    compiler = detect_fortran_compiler()
    if compiler is None:
        raise FortranUnavailableError(
            "No local Fortran compiler found on PATH. Install gfortran, ifx, ifort, or flang "
            "to enable f2py compilation."
        )

    local_source = build_root / source.name
    if source.resolve() != local_source.resolve():
        shutil.copy2(source, local_source)

    temp_root = build_root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    build_root_short = _windows_short_path(build_root)
    temp_root_short = _windows_short_path(temp_root)
    compiler_bin_short = _windows_short_path(Path(compiler).parent)
    user_scripts_short = _windows_short_path(USER_PYTHON_SCRIPTS_DIR)

    command = [
        sys.executable,
        "-m",
        "numpy.f2py",
        "-c",
        "-m",
        module_name,
        local_source.name,
    ]

    result = subprocess.run(
        command,
        cwd=build_root_short,
        env={
            **os.environ,
            "PATH": (
                f"{compiler_bin_short}{os.pathsep}"
                f"{user_scripts_short}{os.pathsep}"
                f"{os.environ.get('PATH', '')}"
            ),
            "TMP": temp_root_short,
            "TEMP": temp_root_short,
            "TMPDIR": temp_root_short,
        },
        capture_output=not verbose,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        raise FortranBuildError(
            "f2py compilation failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Compiler: {compiler}\n"
            f"STDOUT:\n{stdout}\n"
            f"STDERR:\n{stderr}"
        )

    built = compiled_extension_path(build_root, module_name)
    if built is None:
        raise FortranBuildError("f2py reported success, but no extension module was produced.")
    return built


def _load_extension_from_path(path: Path, module_name: str = DEFAULT_MODULE_NAME):
    cache_key = (str(path), module_name)
    cached = _LOADED_MODULES.get(cache_key)
    if cached is not None:
        return cached

    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load compiled extension from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _LOADED_MODULES[cache_key] = module
    return module


def fortran_elementwise_distance(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    build_dir: Path | None = None,
    module_name: str = DEFAULT_MODULE_NAME,
    force_rebuild: bool = False,
) -> np.ndarray:
    current_arr = _normalize_matrix(current, "current")
    reference_arr = _normalize_matrix(reference, "reference")
    if current_arr.shape != reference_arr.shape:
        raise ValueError(
            "current and reference matrices must have the same shape: "
            f"{current_arr.shape!r} != {reference_arr.shape!r}"
        )

    extension_path = build_extension(
        force=force_rebuild,
        build_dir=build_dir,
        module_name=module_name,
    )
    module = _load_extension_from_path(extension_path, module_name=module_name)
    out = module.elementwise_abs_distance(current_arr, reference_arr)
    return np.asfortranarray(np.asarray(out, dtype=np.float64))


def load_default_benchmark_inputs() -> tuple[np.ndarray, np.ndarray]:
    current = np.load(DEFAULT_CURRENT_MATRIX)
    reference = np.load(DEFAULT_REFERENCE_MATRIX)
    return current, reference


def benchmark_distance(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    repeats: int = 3,
    build_dir: Path | None = None,
    force_rebuild: bool = False,
) -> BenchmarkResult:
    python_times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        numpy_elementwise_distance(current, reference)
        python_times.append(time.perf_counter() - started)
    python_seconds = min(python_times)

    compiler = detect_fortran_compiler()
    if compiler is None:
        return BenchmarkResult(
            current_shape=np.asarray(current).shape,
            python_seconds=python_seconds,
            fortran_seconds=None,
            speedup_vs_numpy=None,
            compiler=None,
            status="compiler_unavailable",
            notes="NumPy baseline measured; install a local Fortran compiler to benchmark f2py.",
        )

    fortran_times: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fortran_elementwise_distance(
            current,
            reference,
            build_dir=build_dir,
            force_rebuild=force_rebuild,
        )
        fortran_times.append(time.perf_counter() - started)
        force_rebuild = False
    fortran_seconds = min(fortran_times)
    return BenchmarkResult(
        current_shape=np.asarray(current).shape,
        python_seconds=python_seconds,
        fortran_seconds=fortran_seconds,
        speedup_vs_numpy=python_seconds / fortran_seconds if fortran_seconds else None,
        compiler=compiler,
        status="ok",
        notes=None,
    )


def benchmark_to_json(
    output_path: Path,
    *,
    current_path: Path = DEFAULT_CURRENT_MATRIX,
    reference_path: Path = DEFAULT_REFERENCE_MATRIX,
    repeats: int = 3,
    build_dir: Path | None = None,
    force_rebuild: bool = False,
) -> dict:
    current = np.load(current_path)
    reference = np.load(reference_path)
    result = benchmark_distance(
        current,
        reference,
        repeats=repeats,
        build_dir=build_dir,
        force_rebuild=force_rebuild,
    )
    payload = {
        "current_matrix": str(current_path),
        "reference_matrix": str(reference_path),
        **result.to_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


__all__ = [
    "BenchmarkResult",
    "FortranBuildError",
    "FortranUnavailableError",
    "benchmark_distance",
    "benchmark_to_json",
    "build_extension",
    "compiled_extension_path",
    "compiler_available",
    "detect_fortran_compiler",
    "fortran_elementwise_distance",
    "load_default_benchmark_inputs",
    "numpy_elementwise_distance",
]
