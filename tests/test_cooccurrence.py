"""
Unit tests for src.fingerprint.cooccurrence
"""

import numpy as np
import pytest
from src.fingerprint.cooccurrence import (
    build_vocab,
    count_cooccurrences,
    apply_ppmi,
    l2_normalize_rows,
    validate_matrix,
)


class TestBuildVocab:
    def test_basic(self):
        seqs = [["a", "b", "a"], ["b", "c"]]
        t2i, i2t = build_vocab(seqs)
        assert set(i2t) == {"a", "b", "c"}
        assert len(t2i) == 3
        assert t2i["a"] == i2t.index("a")

    def test_empty_sequences(self):
        t2i, i2t = build_vocab([[], []])
        assert len(t2i) == 0
        assert len(i2t) == 0

    def test_single_token(self):
        t2i, i2t = build_vocab([["x"]])
        assert "x" in t2i
        assert i2t[0] == "x"


class TestCountCooccurrences:
    def _vocab(self, tokens):
        t2i, _ = build_vocab([tokens])
        return t2i

    def test_symmetric(self):
        seq = ["a", "b", "c", "d"]
        t2i = self._vocab(seq)
        counts = count_cooccurrences([seq], t2i, window=1)
        assert np.array_equal(counts, counts.T)

    def test_zero_diagonal(self):
        seq = ["a", "b", "c"]
        t2i = self._vocab(seq)
        counts = count_cooccurrences([seq], t2i, window=2)
        assert np.all(np.diag(counts) == 0)

    def test_window_1_neighbors_only(self):
        seq = ["a", "b", "c"]
        t2i = self._vocab(seq)
        counts = count_cooccurrences([seq], t2i, window=1)
        ia, ib, ic = t2i["a"], t2i["b"], t2i["c"]
        assert counts[ia, ic] == 0
        assert counts[ic, ia] == 0
        assert counts[ia, ib] > 0
        assert counts[ib, ic] > 0

    def test_window_2_skips_one(self):
        seq = ["a", "b", "c", "d"]
        t2i = self._vocab(seq)
        counts = count_cooccurrences([seq], t2i, window=2)
        ia, ib, ic = t2i["a"], t2i["b"], t2i["c"]
        assert counts[ia, ic] > 0

    def test_sequence_boundary_not_crossed(self):
        seq1 = ["a", "b"]
        seq2 = ["c", "d"]
        t2i, _ = build_vocab([seq1, seq2])
        counts = count_cooccurrences([seq1, seq2], t2i, window=2)
        ib, ic = t2i["b"], t2i["c"]
        assert counts[ib, ic] == 0

    def test_repeated_token_counts_all_occurrences(self):
        seq = ["a", "b", "a"]
        t2i = self._vocab(seq)
        counts = count_cooccurrences([seq], t2i, window=1)
        ia, ib = t2i["a"], t2i["b"]
        assert counts[ia, ib] == 2

    def test_shape(self):
        seqs = [["a", "b", "c"]]
        t2i, _ = build_vocab(seqs)
        counts = count_cooccurrences(seqs, t2i, window=2)
        assert counts.shape == (3, 3)

    def test_empty_sequence(self):
        seqs = [["a", "b"], []]
        t2i, _ = build_vocab(seqs)
        counts_with_empty = count_cooccurrences(seqs, t2i, window=1)
        counts_without_empty = count_cooccurrences([["a", "b"]], t2i, window=1)
        assert np.array_equal(counts_with_empty, counts_without_empty)


class TestApplyPPMI:
    def test_output_nonnegative(self):
        counts = np.array([[0, 3, 1], [3, 0, 2], [1, 2, 0]], dtype=np.int64)
        ppmi = apply_ppmi(counts)
        assert np.all(ppmi >= 0)

    def test_all_zero_raises(self):
        counts = np.zeros((3, 3), dtype=np.int64)
        with pytest.raises(ValueError):
            apply_ppmi(counts)

    def test_output_dtype_float32(self):
        counts = np.array([[0, 5], [5, 0]], dtype=np.int64)
        ppmi = apply_ppmi(counts)
        assert ppmi.dtype == np.float32

    def test_symmetric_input_gives_symmetric_output(self):
        counts = np.array([[0, 4, 2], [4, 0, 3], [2, 3, 0]], dtype=np.int64)
        ppmi = apply_ppmi(counts)
        assert np.allclose(ppmi, ppmi.T, atol=1e-5)


class TestL2NormalizeRows:
    def test_unit_norm_rows(self):
        matrix = np.array([[3.0, 4.0], [0.0, 5.0], [1.0, 0.0]], dtype=np.float32)
        normed = l2_normalize_rows(matrix)
        norms = np.linalg.norm(normed, axis=1)
        assert np.isclose(norms[0], 1.0)
        assert np.isclose(norms[1], 1.0)
        assert np.isclose(norms[2], 1.0)

    def test_all_zero_row_remains_zero(self):
        matrix = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        normed = l2_normalize_rows(matrix)
        assert np.all(normed[1] == 0.0)

    def test_output_dtype_float32(self):
        matrix = np.array([[1.0, 2.0]], dtype=np.float64)
        normed = l2_normalize_rows(matrix)
        assert normed.dtype == np.float32


class TestValidateMatrix:
    def _make_symmetric_zero_diag(self, n):
        m = np.random.rand(n, n).astype(np.float32)
        m = (m + m.T) / 2
        np.fill_diagonal(m, 0.0)
        return m

    def test_valid_matrix_passes(self):
        counts = self._make_symmetric_zero_diag(4).astype(np.int64)
        m = l2_normalize_rows(counts.astype(np.float32))
        np.fill_diagonal(m, 0.0)
        validate_matrix(m, ["a", "b", "c", "d"], counts=counts)

    def test_wrong_shape_raises(self):
        m = np.zeros((3, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            validate_matrix(m, ["a", "b", "c"])

    def test_non_symmetric_counts_raises(self):
        counts = np.array([[0, 1], [3, 0]], dtype=np.int64)
        m = l2_normalize_rows(counts.astype(np.float32))
        with pytest.raises(ValueError, match="symmetric"):
            validate_matrix(m, ["a", "b"], counts=counts)

    def test_nonzero_diagonal_raises(self):
        counts = np.array([[0, 5], [5, 0]], dtype=np.int64)
        m = l2_normalize_rows(counts.astype(np.float32))
        m[0, 0] = 0.5
        with pytest.raises(ValueError, match="diagonal"):
            validate_matrix(m, ["a", "b"], counts=counts)
