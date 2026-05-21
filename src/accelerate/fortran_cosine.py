"""
Fortran Cosine Scoring Layer
=============================
Purpose:
    Accelerate the inner loop of IncrementalScoringState._score_from_counters
    by replacing Python dict-intersection cosines with compiled Fortran
    vector operations.

Architecture
------------
The incremental scorer maintains sparse counter dicts for char bigrams,
trigrams, suffixes, and word n-grams. For each of the N candidates per
proposal it:
  1. Builds a sparse profile from the counter (dict of top-K entries)
  2. Computes cosine(candidate_profile, latin_reference_profile)

This module:
  1. Maintains dense float32 arrays indexed against a fixed vocabulary
     (the Latin reference profile keys, in a stable sorted order)
  2. Converts sparse counter deltas → dense delta arrays (O(changed_entries))
  3. Calls Fortran batch_form_scores_f32 for all N candidates at once
  4. Falls back to Python if Fortran is unavailable

The vocabulary index is built once per engine instantiation from the Latin
reference profiles. It is fixed for the lifetime of the run — no rehashing.

Integration
-----------
    scorer = FortranCosineScorer.build(latin_form_ref)
    # per proposal:
    batch_scores = scorer.score_form_batch(candidate_counter_tuples)
    # where candidate_counter_tuples is [(bg_counter, tg_counter, sfx_counter), ...]

Fallback
--------
If Fortran is unavailable or the build fails, FortranCosineScorer.score_form_batch
falls back to the existing _sparse_profile_cosine Python path transparently.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.retrodiction.engine_reinforced_v2 import LatinFormReference

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = Path(__file__).with_name("sparse_cosine.f90")
MODULE_NAME = "rbt_sparse_cosine"
DEFAULT_BUILD_DIR = Path(
    os.environ.get(
        "RBT_FORTRAN_BUILD_DIR",
        str(PROJECT_ROOT.parent.parent / "RBT_FORTRAN_BUILD"),
    )
)


# ---------------------------------------------------------------------------
# Build / load
# ---------------------------------------------------------------------------

def _build_cosine_extension(build_dir: Path, verbose: bool = False):
    """Compile sparse_cosine.f90 via f2py. Returns loaded module or None."""
    from src.accelerate.fortran_distance import (
        FortranBuildError,
        FortranUnavailableError,
        _load_extension_from_path,
        _windows_short_path,
        compiled_extension_path,
        detect_fortran_compiler,
        USER_PYTHON_SCRIPTS_DIR,
    )

    build_dir.mkdir(parents=True, exist_ok=True)
    source = SOURCE_PATH

    existing = compiled_extension_path(build_dir, MODULE_NAME)
    if existing and existing.exists() and existing.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        log.debug("Reusing existing Fortran cosine extension at %s", existing)
        return _load_extension_from_path(existing, MODULE_NAME)

    compiler = detect_fortran_compiler()
    if compiler is None:
        raise FortranUnavailableError("No Fortran compiler found.")

    local_source = build_dir / source.name
    if source.resolve() != local_source.resolve():
        shutil.copy2(source, local_source)

    temp_root = build_dir / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    build_dir_short = _windows_short_path(build_dir)
    temp_root_short = _windows_short_path(temp_root)
    compiler_bin_short = _windows_short_path(Path(compiler).parent)
    user_scripts_short = _windows_short_path(USER_PYTHON_SCRIPTS_DIR)

    command = [sys.executable, "-m", "numpy.f2py", "-c", "-m", MODULE_NAME, local_source.name]
    result = subprocess.run(
        command,
        cwd=build_dir_short,
        env={
            **os.environ,
            "PATH": f"{compiler_bin_short}{os.pathsep}{user_scripts_short}{os.pathsep}{os.environ.get('PATH', '')}",
            "TMP": temp_root_short,
            "TEMP": temp_root_short,
            "TMPDIR": temp_root_short,
        },
        capture_output=not verbose,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise FortranBuildError(
            f"sparse_cosine f2py build failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    built = compiled_extension_path(build_dir, MODULE_NAME)
    if built is None:
        raise FortranBuildError("f2py reported success but no .pyd/.so was produced.")

    return _load_extension_from_path(built, MODULE_NAME)


# ---------------------------------------------------------------------------
# Vocabulary index
# ---------------------------------------------------------------------------

class ProfileVocabIndex:
    """
    Maps sparse profile keys to dense array positions.

    Built once from the Latin reference profile keys. All subsequent
    sparse-to-dense conversions use this fixed index.
    """

    def __init__(self, keys: list[str]) -> None:
        self._keys = keys
        self._index: dict[str, int] = {k: i for i, k in enumerate(keys)}
        self.size = len(keys)
        self._ref_array: np.ndarray | None = None  # set by caller

    def sparse_to_dense(self, counter: Counter, top_n: int | None = None) -> np.ndarray:
        """Convert a Counter to a dense float32 array aligned to this vocab."""
        arr = np.zeros(self.size, dtype=np.float32)
        if top_n is not None:
            items = counter.most_common(top_n)
        else:
            items = counter.items()
        for key, val in items:
            idx = self._index.get(key)
            if idx is not None:
                arr[idx] = float(val)
        # Normalize
        total = arr.sum()
        if total > 0.0:
            arr /= total
        return arr

    def counter_to_dense_counts(self, counter: Counter) -> np.ndarray:
        """Convert raw counts to a dense float32 array without normalization."""
        arr = np.zeros(self.size, dtype=np.float32)
        self.add_counter_to_dense(arr, counter)
        return arr

    def add_counter_to_dense(self, arr: np.ndarray, counter: Counter) -> None:
        """Accumulate raw Counter values into an existing dense array."""
        for key, val in counter.items():
            if not val:
                continue
            idx = self._index.get(key)
            if idx is not None:
                arr[idx] += float(val)

    def profile_to_dense(self, profile: dict[str, float]) -> np.ndarray:
        """Convert a normalized sparse profile dict to a dense float32 array."""
        arr = np.zeros(self.size, dtype=np.float32)
        for key, val in profile.items():
            idx = self._index.get(key)
            if idx is not None:
                arr[idx] = float(val)
        return arr


# ---------------------------------------------------------------------------
# Main scorer class
# ---------------------------------------------------------------------------

class FortranCosineScorer:
    """
    Fortran-accelerated batch form scorer.

    Call score_form_batch() with a list of (bg_counter, tg_counter, sfx_counter)
    tuples for N candidates. Returns N float32 form scores.

    Falls back to Python _sparse_profile_cosine if Fortran is unavailable.
    """

    def __init__(
        self,
        bg_index: ProfileVocabIndex,
        tg_index: ProfileVocabIndex,
        sfx_index: ProfileVocabIndex,
        fortran_module,  # compiled f2py module or None
        *,
        bg_weight: float = 0.40,
        tg_weight: float = 0.40,
        sfx_weight: float = 0.20,
    ) -> None:
        self._bg_index = bg_index
        self._tg_index = tg_index
        self._sfx_index = sfx_index
        self._mod = fortran_module
        self._using_fortran = fortran_module is not None
        self._bg_weight = float(bg_weight)
        self._tg_weight = float(tg_weight)
        self._sfx_weight = float(sfx_weight)

    @classmethod
    def build(
        cls,
        latin_form_ref: "LatinFormReference",
        build_dir: Path | None = None,
        force_rebuild: bool = False,
        verbose: bool = False,
    ) -> "FortranCosineScorer":
        """
        Build vocab indexes from the Latin reference profiles and compile
        the Fortran extension. Falls back to Python if build fails.
        """
        build_root = build_dir or DEFAULT_BUILD_DIR

        bg_index = ProfileVocabIndex(sorted(latin_form_ref.char_bigram_profile.keys()))
        tg_index = ProfileVocabIndex(sorted(latin_form_ref.char_trigram_profile.keys()))
        sfx_index = ProfileVocabIndex(sorted(latin_form_ref.suffix_profile.keys()))

        # Store dense reference arrays on the index objects
        bg_index._ref_array = bg_index.profile_to_dense(latin_form_ref.char_bigram_profile)
        tg_index._ref_array = tg_index.profile_to_dense(latin_form_ref.char_trigram_profile)
        sfx_index._ref_array = sfx_index.profile_to_dense(latin_form_ref.suffix_profile)

        fortran_module = None
        try:
            if force_rebuild:
                # Delete existing .pyd to force rebuild
                from src.accelerate.fortran_distance import compiled_extension_path
                existing = compiled_extension_path(Path(build_root), MODULE_NAME)
                if existing and existing.exists():
                    existing.unlink()
            fortran_module = _build_cosine_extension(Path(build_root), verbose=verbose)
            log.info("Fortran cosine scoring extension loaded from %s", build_root)
        except Exception as exc:
            log.warning("Fortran cosine build failed (%s); falling back to Python cosine.", exc)

        return cls(
            bg_index,
            tg_index,
            sfx_index,
            fortran_module,
            bg_weight=latin_form_ref.char_bigram_weight,
            tg_weight=latin_form_ref.char_trigram_weight,
            sfx_weight=latin_form_ref.suffix_weight,
        )

    @property
    def using_fortran(self) -> bool:
        return self._using_fortran

    def score_single_form(
        self,
        bg_counter: Counter,
        tg_counter: Counter,
        sfx_counter: Counter,
        bg_top_n: int | None = None,
        tg_top_n: int | None = None,
        sfx_top_n: int | None = None,
    ) -> float:
        """
        Score one candidate (bg_counter, tg_counter, sfx_counter).

        Returns a single float form score in [0, 1].

        This is the per-candidate fast path used by IncrementalScoringState.
        It avoids matrix allocation overhead and calls BLAS-backed numpy or
        Fortran cosine_f32 directly on 1D float32 arrays.
        """
        bg_vec  = self._bg_index.sparse_to_dense(bg_counter,  bg_top_n)
        tg_vec  = self._tg_index.sparse_to_dense(tg_counter,  tg_top_n)
        sfx_vec = self._sfx_index.sparse_to_dense(sfx_counter, sfx_top_n)

        if self._using_fortran:
            return float(self._cosine_single_fortran(bg_vec, tg_vec, sfx_vec))
        else:
            return float(self._cosine_single_numpy(bg_vec, tg_vec, sfx_vec))

    def _cosine_single_fortran(
        self,
        bg_vec: np.ndarray,
        tg_vec: np.ndarray,
        sfx_vec: np.ndarray,
    ) -> float:
        """Fortran cosine_f32 for three 1D vectors against stored refs.

        f2py calling convention: intent(out) args become return values.
            result = mod.sparse_cosine_kernels.cosine_f32(a, b)
        """
        kern = self._mod.sparse_cosine_kernels
        bg_cos  = kern.cosine_f32(bg_vec,  self._bg_index._ref_array)
        tg_cos  = kern.cosine_f32(tg_vec,  self._tg_index._ref_array)
        sfx_cos = kern.cosine_f32(sfx_vec, self._sfx_index._ref_array)
        return (
            self._bg_weight * float(bg_cos)
            + self._tg_weight * float(tg_cos)
            + self._sfx_weight * float(sfx_cos)
        )

    def _cosine_single_numpy(
        self,
        bg_vec: np.ndarray,
        tg_vec: np.ndarray,
        sfx_vec: np.ndarray,
    ) -> float:
        """NumPy BLAS cosine for three 1D float32 vectors against stored refs."""
        def _cos1d(v: np.ndarray, ref: np.ndarray) -> float:
            dot   = float(np.dot(v, ref))
            norm_v = float(np.linalg.norm(v))
            norm_r = float(np.linalg.norm(ref))
            denom  = norm_v * norm_r
            return dot / denom if denom > 0.0 else 0.0

        return (
            self._bg_weight * _cos1d(bg_vec, self._bg_index._ref_array)
            + self._tg_weight * _cos1d(tg_vec, self._tg_index._ref_array)
            + self._sfx_weight * _cos1d(sfx_vec, self._sfx_index._ref_array)
        )

    def score_form_batch(
        self,
        candidates: list[tuple[Counter, Counter, Counter]],
    ) -> np.ndarray:
        """
        Score N candidate (bg_counter, tg_counter, sfx_counter) tuples.

        Returns float32 array of shape (N,) with form scores in [0, 1].
        """
        m = len(candidates)
        if m == 0:
            return np.zeros(0, dtype=np.float32)

        n_bg  = self._bg_index.size
        n_tg  = self._tg_index.size
        n_sfx = self._sfx_index.size

        # Build dense candidate matrices: shape (m, n_*). Use Fortran-order
        # buffers so the compiled path can consume them without an implicit
        # layout conversion.
        bg_mat  = np.zeros((m, n_bg),  dtype=np.float32, order="F")
        tg_mat  = np.zeros((m, n_tg),  dtype=np.float32, order="F")
        sfx_mat = np.zeros((m, n_sfx), dtype=np.float32, order="F")

        for i, (bg_c, tg_c, sfx_c) in enumerate(candidates):
            self._bg_index.add_counter_to_dense(bg_mat[i], bg_c)
            self._tg_index.add_counter_to_dense(tg_mat[i], tg_c)
            self._sfx_index.add_counter_to_dense(sfx_mat[i], sfx_c)

        self._normalize_rows(bg_mat)
        self._normalize_rows(tg_mat)
        self._normalize_rows(sfx_mat)

        if self._using_fortran and self._uses_default_weights():
            return self._score_fortran(bg_mat, tg_mat, sfx_mat)
        else:
            return self._score_python(bg_mat, tg_mat, sfx_mat)

    def score_form_batch_from_deltas(
        self,
        base_bg_counter: Counter,
        base_tg_counter: Counter,
        base_sfx_counter: Counter,
        candidate_deltas: list[tuple[Counter, Counter, Counter]],
    ) -> np.ndarray:
        """
        Score N candidates as sparse deltas off one committed baseline.

        This avoids rebuilding full per-candidate char Counter copies in Python.
        Each row starts from the committed baseline dense vector, then applies
        only the changed-sequence deltas before the compiled cosine step.
        """
        m = len(candidate_deltas)
        if m == 0:
            return np.zeros(0, dtype=np.float32)

        bg_mat = self._build_delta_matrix(self._bg_index, base_bg_counter, [d[0] for d in candidate_deltas])
        tg_mat = self._build_delta_matrix(self._tg_index, base_tg_counter, [d[1] for d in candidate_deltas])
        sfx_mat = self._build_delta_matrix(self._sfx_index, base_sfx_counter, [d[2] for d in candidate_deltas])

        if self._using_fortran and self._uses_default_weights():
            return self._score_fortran(bg_mat, tg_mat, sfx_mat)
        else:
            return self._score_python(bg_mat, tg_mat, sfx_mat)

    def _uses_default_weights(self) -> bool:
        return (
            abs(self._bg_weight - 0.40) < 1e-9
            and abs(self._tg_weight - 0.40) < 1e-9
            and abs(self._sfx_weight - 0.20) < 1e-9
        )

    def _build_delta_matrix(
        self,
        index: ProfileVocabIndex,
        base_counter: Counter,
        deltas: list[Counter],
    ) -> np.ndarray:
        """Materialize a normalized candidate matrix from one base counter."""
        base_vec = index.counter_to_dense_counts(base_counter)
        matrix = np.empty((len(deltas), index.size), dtype=np.float32, order="F")
        matrix[:] = base_vec
        for row_idx, delta in enumerate(deltas):
            index.add_counter_to_dense(matrix[row_idx], delta)
        np.clip(matrix, 0.0, None, out=matrix)
        self._normalize_rows(matrix)
        return matrix

    @staticmethod
    def _normalize_rows(matrix: np.ndarray) -> None:
        """Normalize each row in-place to sum to 1.0 when non-empty."""
        row_sums = matrix.sum(axis=1, dtype=np.float32)
        nonzero = row_sums > 0.0
        if not np.any(nonzero):
            matrix[:] = 0.0
            return
        matrix[~nonzero] = 0.0
        matrix[nonzero] /= row_sums[nonzero, None]

    def _score_fortran(
        self,
        bg_mat: np.ndarray,
        tg_mat: np.ndarray,
        sfx_mat: np.ndarray,
    ) -> np.ndarray:
        """Call Fortran batch_form_scores_f32.

        f2py calling convention: intent(out) form_scores is returned as a
        Python return value; dimension scalars (m, n_bg, …) are inferred from
        the array shapes and need not be passed explicitly.
        """
        kern = self._mod.sparse_cosine_kernels
        return kern.batch_form_scores_f32(
            bg_mat,  self._bg_index._ref_array,
            tg_mat,  self._tg_index._ref_array,
            sfx_mat, self._sfx_index._ref_array,
        )

    def _score_python(
        self,
        bg_mat: np.ndarray,
        tg_mat: np.ndarray,
        sfx_mat: np.ndarray,
    ) -> np.ndarray:
        """Python fallback: vectorized numpy cosines."""
        bg_ref  = self._bg_index._ref_array
        tg_ref  = self._tg_index._ref_array
        sfx_ref = self._sfx_index._ref_array

        def _batch_cos(mat: np.ndarray, ref: np.ndarray) -> np.ndarray:
            dots   = mat @ ref
            norms  = np.linalg.norm(mat, axis=1)
            ref_n  = float(np.linalg.norm(ref))
            denom  = norms * ref_n
            safe   = np.where(denom > 0.0, denom, 1.0)
            return np.where(denom > 0.0, dots / safe, 0.0).astype(np.float32)

        return (
            self._bg_weight * _batch_cos(bg_mat, bg_ref)
            + self._tg_weight * _batch_cos(tg_mat, tg_ref)
            + self._sfx_weight * _batch_cos(sfx_mat, sfx_ref)
        )
