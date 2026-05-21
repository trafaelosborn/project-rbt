"""Tests for the Phase 2 batch candidate kernel."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.accelerate.fortran_batch import fortran_top_adjustments, numpy_top_adjustments
from src.accelerate.fortran_distance import build_extension, compiler_available, detect_fortran_compiler


def test_numpy_top_adjustments_returns_descending_scores():
    cooc_current = np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64)
    cooc_reference = np.array([[5.0, 2.5], [1.0, -4.0]], dtype=np.float64)
    pos_current = np.array([[0.0, 1.0], [0.5, 1.5]], dtype=np.float64)
    pos_reference = np.array([[0.0, 3.0], [0.25, 1.5]], dtype=np.float64)

    batch = numpy_top_adjustments(
        cooc_current,
        cooc_reference,
        pos_current,
        pos_reference,
        top_k=4,
    )

    assert batch.size == 4
    assert np.all(batch.abs_scores[:-1] >= batch.abs_scores[1:])
    assert batch.abs_scores[0] == pytest.approx(5.0)
    assert batch.component_ids[0] == 1


@pytest.mark.skipif(
    (not compiler_available()) or os.environ.get("RBT_RUN_FORTRAN_BUILD_TESTS") != "1",
    reason="Set RBT_RUN_FORTRAN_BUILD_TESTS=1 with a local toolchain to run compile-backed verification.",
)
def test_fortran_top_adjustments_matches_python_reference():
    compiler = detect_fortran_compiler()
    assert compiler is not None

    cooc_current = np.asfortranarray(np.array([[0.0, 2.0], [1.0, 0.0]], dtype=np.float64))
    cooc_reference = np.asfortranarray(np.array([[5.0, 2.5], [1.0, -4.0]], dtype=np.float64))
    pos_current = np.asfortranarray(np.array([[0.0, 1.0], [0.5, 1.5]], dtype=np.float64))
    pos_reference = np.asfortranarray(np.array([[0.0, 3.0], [0.25, 1.5]], dtype=np.float64))

    build_dir = Path(os.environ.get("RBT_FORTRAN_TEST_BUILD_DIR", r"C:\Code\RBT_FORTRAN_TEST_BUILD")) / "batch"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_extension(build_dir=build_dir, force=True)

    try:
        python_batch = numpy_top_adjustments(
            cooc_current,
            cooc_reference,
            pos_current,
            pos_reference,
            top_k=5,
        )
        fortran_batch = fortran_top_adjustments(
            cooc_current,
            cooc_reference,
            pos_current,
            pos_reference,
            top_k=5,
            build_dir=build_dir,
        )
        assert np.array_equal(fortran_batch.component_ids, python_batch.component_ids)
        assert np.array_equal(fortran_batch.row_indices, python_batch.row_indices)
        assert np.array_equal(fortran_batch.col_indices, python_batch.col_indices)
        assert np.allclose(fortran_batch.signed_deltas, python_batch.signed_deltas)
        assert np.allclose(fortran_batch.abs_scores, python_batch.abs_scores)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
