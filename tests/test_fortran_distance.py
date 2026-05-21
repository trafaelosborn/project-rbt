"""Tests for the Session 1 Fortran distance scaffold."""

from __future__ import annotations

import shutil
import os
from pathlib import Path

import numpy as np
import pytest

from src.accelerate.fortran_distance import (
    FortranUnavailableError,
    build_extension,
    compiler_available,
    detect_fortran_compiler,
    fortran_elementwise_distance,
    numpy_elementwise_distance,
)
from src.accelerate.tensor_layout import FingerprintTensorLayout


def test_numpy_reference_distance_matches_abs_difference():
    current = np.array([[1.0, -2.0], [0.5, 4.0]])
    reference = np.array([[0.0, 3.0], [0.0, 4.5]])
    actual = numpy_elementwise_distance(current, reference)
    expected = np.array([[1.0, 5.0], [0.5, 0.5]])
    assert np.array_equal(actual, expected)


def test_build_extension_raises_clean_error_without_compiler(monkeypatch):
    monkeypatch.setattr("src.accelerate.fortran_distance.detect_fortran_compiler", lambda: None)
    build_dir = Path("data/retrodiction/_test_fortran_build")
    shutil.rmtree(build_dir, ignore_errors=True)
    try:
        with pytest.raises(FortranUnavailableError):
            build_extension(build_dir=build_dir)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)


def test_tensor_layout_packs_without_copying_component_shapes():
    layout = FingerprintTensorLayout(
        vocab_size=3,
        positional_width=2,
        bigram_profile_size=4,
        trigram_profile_size=5,
    )
    cooccurrence = np.arange(9, dtype=np.float64).reshape(3, 3)
    positional = np.arange(6, dtype=np.float64).reshape(3, 2)
    bigram_profile = np.linspace(0.1, 0.4, 4)
    trigram_profile = np.linspace(0.1, 0.5, 5)

    packed = layout.pack(
        cooccurrence=cooccurrence,
        positional=positional,
        bigram_profile=bigram_profile,
        trigram_profile=trigram_profile,
    )

    assert packed.ndim == 1
    assert packed.dtype == np.float64
    assert np.array_equal(layout.view(packed, "cooccurrence"), cooccurrence)
    assert np.array_equal(layout.view(packed, "positional"), positional)
    assert np.array_equal(layout.view(packed, "bigram_profile"), bigram_profile)
    assert np.array_equal(layout.view(packed, "trigram_profile"), trigram_profile)


@pytest.mark.skipif(
    (not compiler_available()) or os.environ.get("RBT_RUN_FORTRAN_BUILD_TESTS") != "1",
    reason="Set RBT_RUN_FORTRAN_BUILD_TESTS=1 with a local toolchain to run compile-backed verification.",
)
def test_fortran_distance_matches_python_reference():
    compiler = detect_fortran_compiler()
    assert compiler is not None

    current = np.asfortranarray(np.array([[1.0, -2.0], [0.5, 4.0]], dtype=np.float64))
    reference = np.asfortranarray(np.array([[0.0, 3.0], [0.0, 4.5]], dtype=np.float64))
    build_dir = Path(os.environ.get("RBT_FORTRAN_TEST_BUILD_DIR", r"C:\Code\RBT_FORTRAN_TEST_BUILD")) / "distance"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_extension(build_dir=build_dir, force=True)

    try:
        actual = fortran_elementwise_distance(current, reference, build_dir=build_dir)
        expected = numpy_elementwise_distance(current, reference)
        assert np.array_equal(actual, expected)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
