"""Tests for src.retrodiction.similarity"""

import math
import numpy as np
import pytest
from src.retrodiction.similarity import (
    coherence_diagnostics,
    top_k_coverage,
    profile_entropy,
    scaled_euclidean_distance,
    structural_vector,
    cosine_similarity,
)


def _make_sequences(n=100, length=10, vocab_size=50, seed=0):
    rng = np.random.default_rng(seed)
    tokens = [str(i) for i in range(vocab_size)]
    return [list(rng.choice(tokens, size=length)) for _ in range(n)]


def _make_profile(n_grams=20, seed=0):
    rng = np.random.default_rng(seed)
    vals = rng.dirichlet(np.ones(n_grams))
    return {f"tok_{i}": float(v) for i, v in enumerate(vals)}


class TestTopKCoverage:
    def test_uniform_coverage_equals_k_over_n(self):
        n = 200
        profile = {str(i): 1.0 / n for i in range(n)}
        cov = top_k_coverage(profile, k=10)
        assert abs(cov - 10 / 200) < 1e-9

    def test_concentrated_profile_high_coverage(self):
        # Top item has all the mass
        profile = {"a": 1.0, **{str(i): 0.0 for i in range(99)}}
        assert abs(top_k_coverage(profile, k=1) - 1.0) < 1e-9

    def test_coverage_in_zero_to_one(self):
        profile = _make_profile(100)
        cov = top_k_coverage(profile, k=10)
        assert 0.0 <= cov <= 1.0

    def test_empty_profile_zero(self):
        assert top_k_coverage({}, k=10) == 0.0

    def test_coverage_increases_with_k(self):
        profile = _make_profile(200)
        assert top_k_coverage(profile, k=10) <= top_k_coverage(profile, k=50)

    def test_concentrated_higher_than_uniform(self):
        n = 100
        uniform = {str(i): 1.0 / n for i in range(n)}
        # Concentrated: top 10 items have 90% of mass
        concentrated = {str(i): 0.09 for i in range(10)}
        concentrated.update({str(i + 10): 0.001 for i in range(90)})
        assert top_k_coverage(concentrated, k=10) > top_k_coverage(uniform, k=10)


class TestProfileEntropy:
    def test_uniform_profile_has_higher_entropy_than_concentrated(self):
        uniform = {str(i): 0.25 for i in range(4)}
        concentrated = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0}
        assert profile_entropy(uniform) > profile_entropy(concentrated)

    def test_empty_profile_zero(self):
        assert profile_entropy({}) == 0.0


class TestStructuralVector:
    def test_shape(self):
        seqs = _make_sequences()
        bg = _make_profile()
        tg = _make_profile(seed=1)
        vec = structural_vector(seqs, bg, tg)
        assert vec.shape == (4,)

    def test_all_nonnegative(self):
        seqs = _make_sequences()
        bg = _make_profile()
        tg = _make_profile(seed=1)
        vec = structural_vector(seqs, bg, tg)
        assert (vec >= 0).all()

    def test_ttr_first_component(self):
        seqs = [["a", "b", "c"]] * 10  # all same sequence, low TTR
        bg = _make_profile()
        tg = _make_profile(seed=1)
        vec = structural_vector(seqs, bg, tg)
        assert vec[0] < 0.5  # TTR should be low

    def test_concentrated_profile_gives_higher_coverage_component(self):
        seqs = _make_sequences()
        n = 200
        uniform_profile = {str(i): 1.0 / n for i in range(n)}
        # Concentrated: top few items dominate
        concentrated_profile = {str(i): (0.5 if i == 0 else 0.5 / 199) for i in range(200)}
        vec_uniform = structural_vector(seqs, uniform_profile, uniform_profile)
        vec_concentrated = structural_vector(seqs, concentrated_profile, concentrated_profile)
        assert vec_concentrated[1] > vec_uniform[1]  # bigram coverage


class TestCosineSimilarity:
    def test_identical_vectors_give_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_give_zero(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert abs(cosine_similarity(v1, v2)) < 1e-9

    def test_zero_vector_gives_zero(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 2.0])
        assert cosine_similarity(v1, v2) == 0.0

    def test_range_zero_to_one_for_nonneg_vectors(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v1 = rng.random(4)
            v2 = rng.random(4)
            sim = cosine_similarity(v1, v2)
            assert 0.0 <= sim <= 1.0 + 1e-9


class TestScaledEuclideanDistance:
    def test_zero_for_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        scale = np.array([1.0, 1.0, 1.0])
        assert scaled_euclidean_distance(v, v, scale) == pytest.approx(0.0, abs=1e-9)

    def test_scaling_changes_distance(self):
        v1 = np.array([0.0, 0.0])
        v2 = np.array([2.0, 2.0])
        assert scaled_euclidean_distance(v1, v2, np.array([2.0, 2.0])) < scaled_euclidean_distance(
            v1, v2, np.array([1.0, 1.0])
        )


class TestCoherenceDiagnostics:
    def test_real_centroid_is_coherent(self):
        diag = coherence_diagnostics(
            vec=np.array([1.0, 1.0]),
            real_language_centroid=np.array([1.0, 1.0]),
            markov_vec=np.array([0.0, 0.0]),
            feature_scale=np.array([1.0, 1.0]),
        )
        assert diag["coherence_label"] == "coherent"
        assert diag["language_likeness_margin"] > 0.0

    def test_midpoint_can_be_borderline(self):
        diag = coherence_diagnostics(
            vec=np.array([0.75, 0.75]),
            real_language_centroid=np.array([1.0, 1.0]),
            markov_vec=np.array([0.0, 0.0]),
            feature_scale=np.array([1.0, 1.0]),
        )
        assert diag["coherence_label"] == "borderline"

    def test_markov_point_is_noise_like(self):
        diag = coherence_diagnostics(
            vec=np.array([0.0, 0.0]),
            real_language_centroid=np.array([1.0, 1.0]),
            markov_vec=np.array([0.0, 0.0]),
            feature_scale=np.array([1.0, 1.0]),
        )
        assert diag["coherence_label"] == "noise_like"
        assert diag["language_likeness_margin"] < 0.0
