"""Tests for source-vocab tensor alignment."""

from __future__ import annotations

import numpy as np

from src.accelerate.aligned_tensor import (
    align_feature_matrix_to_anchor_tokens,
    align_square_matrix_to_anchor_tokens,
)


def test_align_square_matrix_projects_reference_into_anchor_vocab():
    anchor = ["b", "c", "x"]
    reference_tokens = {"a": 0, "b": 1, "c": 2}
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    aligned = align_square_matrix_to_anchor_tokens(
        anchor_tokens=anchor,
        matrix=matrix,
        token2idx=reference_tokens,
    )
    expected = np.array(
        [
            [5.0, 6.0, 0.0],
            [8.0, 9.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(aligned, expected)


def test_align_feature_matrix_projects_reference_rows_into_anchor_vocab():
    anchor = ["b", "c", "x"]
    reference_tokens = {"a": 0, "b": 1, "c": 2}
    matrix = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=np.float64,
    )
    aligned = align_feature_matrix_to_anchor_tokens(
        anchor_tokens=anchor,
        matrix=matrix,
        token2idx=reference_tokens,
    )
    expected = np.array(
        [
            [3.0, 4.0],
            [5.0, 6.0],
            [0.0, 0.0],
        ],
        dtype=np.float64,
    )
    assert np.array_equal(aligned, expected)
