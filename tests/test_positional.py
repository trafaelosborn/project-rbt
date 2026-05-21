"""
Unit tests for src.fingerprint.positional
"""

import numpy as np
import pytest
from src.fingerprint.positional import (
    PositionalAccumulator,
    validate_matrix,
    F_INITIAL,
    F_FINAL,
    F_MEDIAL,
    F_MEAN_POS,
    F_STD_POS,
    F_FREQ_NORM,
    N_FEATURES,
)


class TestPositionalAccumulator:
    def test_initial_rate_for_always_initial_token(self):
        acc = PositionalAccumulator({"start": 0, "mid": 1, "end": 2})
        acc.ingest_sequence(["start", "mid", "end"])
        acc.ingest_sequence(["start", "mid", "end"])
        matrix = acc.to_matrix()
        assert np.isclose(matrix[0, F_INITIAL], 1.0)

    def test_final_rate_for_always_final_token(self):
        acc = PositionalAccumulator({"start": 0, "mid": 1, "end": 2})
        acc.ingest_sequence(["start", "mid", "end"])
        acc.ingest_sequence(["start", "mid", "end"])
        matrix = acc.to_matrix()
        assert np.isclose(matrix[2, F_FINAL], 1.0)

    def test_medial_rate_for_always_medial_token(self):
        acc = PositionalAccumulator({"a": 0, "b": 1, "c": 2})
        acc.ingest_sequence(["a", "b", "c"])
        acc.ingest_sequence(["a", "b", "c"])
        matrix = acc.to_matrix()
        assert np.isclose(matrix[1, F_MEDIAL], 1.0)
        assert np.isclose(matrix[1, F_INITIAL], 0.0)
        assert np.isclose(matrix[1, F_FINAL], 0.0)

    def test_rates_sum_to_one(self):
        acc = PositionalAccumulator({"a": 0, "b": 1, "c": 2})
        acc.ingest_sequence(["a", "b", "c", "a", "b"])
        matrix = acc.to_matrix()
        for i in range(3):
            if acc.counts[i] > 0:
                rate_sum = matrix[i, F_INITIAL] + matrix[i, F_FINAL] + matrix[i, F_MEDIAL]
                assert np.isclose(rate_sum, 1.0, atol=1e-4), \
                    f"Token {i}: rates sum to {rate_sum}"

    def test_single_token_sequence_is_initial_only(self):
        # Single-token sequences count as initial only so rates sum to 1.0.
        acc = PositionalAccumulator({"x": 0})
        acc.ingest_sequence(["x"])
        matrix = acc.to_matrix()
        assert np.isclose(matrix[0, F_INITIAL], 1.0)
        assert np.isclose(matrix[0, F_FINAL], 0.0)
        assert np.isclose(matrix[0, F_MEDIAL], 0.0)
        assert np.isclose(matrix[0, F_INITIAL] + matrix[0, F_FINAL] + matrix[0, F_MEDIAL], 1.0)

    def test_mean_pos_initial_token(self):
        acc = PositionalAccumulator({"a": 0, "b": 1, "c": 2})
        acc.ingest_sequence(["a", "b", "c"])
        acc.ingest_sequence(["a", "b", "c"])
        matrix = acc.to_matrix()
        assert np.isclose(matrix[0, F_MEAN_POS], 0.0)

    def test_mean_pos_final_token(self):
        acc = PositionalAccumulator({"a": 0, "b": 1, "c": 2})
        acc.ingest_sequence(["a", "b", "c"])
        acc.ingest_sequence(["a", "b", "c"])
        matrix = acc.to_matrix()
        assert np.isclose(matrix[2, F_MEAN_POS], 1.0)

    def test_freq_norm_highest_for_most_frequent(self):
        acc = PositionalAccumulator({"a": 0, "b": 1})
        for _ in range(5):
            acc.ingest_sequence(["a"])
        acc.ingest_sequence(["b"])
        matrix = acc.to_matrix()
        assert matrix[0, F_FREQ_NORM] > matrix[1, F_FREQ_NORM]

    def test_freq_norm_max_token_is_one(self):
        acc = PositionalAccumulator({"a": 0, "b": 1})
        for _ in range(10):
            acc.ingest_sequence(["a"])
        acc.ingest_sequence(["b"])
        matrix = acc.to_matrix()
        assert np.isclose(matrix[0, F_FREQ_NORM], 1.0)

    def test_unseen_token_all_zero(self):
        acc = PositionalAccumulator({"a": 0, "b": 1})
        acc.ingest_sequence(["a"])
        matrix = acc.to_matrix()
        assert np.all(matrix[1] == 0.0)

    def test_empty_sequence_no_crash(self):
        acc = PositionalAccumulator({"a": 0})
        acc.ingest_sequence([])
        acc.ingest_sequence(["a"])
        matrix = acc.to_matrix()
        assert matrix[0, F_INITIAL] == 1.0

    def test_unknown_token_ignored(self):
        acc = PositionalAccumulator({"a": 0})
        acc.ingest_sequence(["a", "UNKNOWN_TOKEN"])
        matrix = acc.to_matrix()
        assert np.isclose(matrix[0, F_INITIAL], 1.0)

    def test_output_dtype_float32(self):
        acc = PositionalAccumulator({"a": 0})
        acc.ingest_sequence(["a"])
        matrix = acc.to_matrix()
        assert matrix.dtype == np.float32

    def test_std_pos_zero_for_fixed_position(self):
        acc = PositionalAccumulator({"a": 0, "b": 1, "c": 2})
        for _ in range(3):
            acc.ingest_sequence(["a", "b", "c"])
        matrix = acc.to_matrix()
        assert np.isclose(matrix[0, F_STD_POS], 0.0)
        assert np.isclose(matrix[2, F_STD_POS], 0.0)

    def test_std_pos_nonzero_for_variable_position(self):
        acc = PositionalAccumulator({"x": 0, "a": 1, "b": 2})
        acc.ingest_sequence(["x", "a"])
        acc.ingest_sequence(["a", "x"])
        matrix = acc.to_matrix()
        assert matrix[0, F_STD_POS] > 0.0

    def test_shape(self):
        acc = PositionalAccumulator({"a": 0, "b": 1, "c": 2})
        acc.ingest_sequence(["a", "b", "c"])
        matrix = acc.to_matrix()
        assert matrix.shape == (3, N_FEATURES)


class TestValidateMatrix:
    def _make_valid_matrix(self, n):
        m = np.zeros((n, N_FEATURES), dtype=np.float32)
        for i in range(n):
            m[i, F_INITIAL] = 0.3
            m[i, F_FINAL] = 0.3
            m[i, F_MEDIAL] = 0.4
            m[i, F_MEAN_POS] = 0.5
            m[i, F_STD_POS] = 0.1
            m[i, F_FREQ_NORM] = 0.8
        return m

    def test_valid_passes(self):
        m = self._make_valid_matrix(5)
        validate_matrix(m, ["a", "b", "c", "d", "e"])

    def test_wrong_shape_raises(self):
        m = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            validate_matrix(m, ["a", "b", "c"])

    def test_freq_norm_above_one_raises(self):
        m = self._make_valid_matrix(2)
        m[0, F_FREQ_NORM] = 1.5
        with pytest.raises(ValueError, match="freq_norm"):
            validate_matrix(m, ["a", "b"])

    def test_freq_norm_below_zero_raises(self):
        m = self._make_valid_matrix(2)
        m[1, F_FREQ_NORM] = -0.1
        with pytest.raises(ValueError, match="freq_norm"):
            validate_matrix(m, ["a", "b"])
